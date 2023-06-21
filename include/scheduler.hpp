#ifndef GRAPH_PROTOTYPE_SCHEDULER_HPP
#define GRAPH_PROTOTYPE_SCHEDULER_HPP
#include <graph.hpp>
#include <set>
#include <queue>
#include <thread_pool.hpp>
#include <latch>
#include <utility>

namespace fair::graph::scheduler {

enum execution_policy { single_threaded, multi_threaded };
enum SchedulerState { IDLE, INITIALISED, RUNNING, REQUESTED_STOP, REQUESTED_PAUSE, STOPPED, PAUSE, SHUTTING_DOWN };

template<typename scheduler_type, execution_policy = single_threaded>
class scheduler_base : public node<scheduler_type> {
protected:
    using node_t                             = node_model *;
    using thread_pool_type                   = thread_pool::BasicThreadPool;
    SchedulerState                    _state = IDLE;
    fair::graph::graph                _graph;
    std::shared_ptr<thread_pool_type> _pool;
    std::vector<std::vector<node_t>>  _job_lists{};

public:
    explicit scheduler_base(fair::graph::graph              &&graph,
                            std::shared_ptr<thread_pool_type> thread_pool = std::make_shared<fair::thread_pool::BasicThreadPool>("simple-scheduler-pool", thread_pool::CPU_BOUND))
        : _graph(std::move(graph)), _pool(std::move(thread_pool)){};

    void
    init(fair::graph::graph &graph) {
        auto result = init_proof(std::all_of(graph.connection_definitions().begin(), graph.connection_definitions().end(),
                                             [](auto &connection_definition) { return connection_definition() == connection_result_t::SUCCESS; }));
        graph.clear_connection_definitions();
        return result;
    }

    template<typename node_t>
    work_return_t
    traverse_nodes(std::span<node_t> nodes) {
        bool something_happened = false;
        for (auto &currentNode : nodes) {
            auto result = currentNode->work();
            if (result == work_return_t::ERROR) {
                return work_return_t::ERROR;
            } else if (result == work_return_t::INSUFFICIENT_INPUT_ITEMS || result == work_return_t::DONE) {
                // nothing
            } else if (result == work_return_t::OK || result == work_return_t::INSUFFICIENT_OUTPUT_ITEMS) {
                something_happened = true;
            }
        }
        return something_happened ? work_return_t::OK : work_return_t::DONE;
    }

    void
    run_on_pool(std::span<node_model *> job, std::size_t n_batches, std::atomic_uint64_t &progress, std::latch &running_threads, std::atomic_bool &stop_requested) {
        uint32_t done           = 0;
        uint32_t progress_count = 0;
        while (done < n_batches && !stop_requested) {
            bool     something_happened = traverse_nodes(job) == work_return_t::OK;
            uint64_t progress_local, progress_new;
            if (something_happened) { // something happened in this thread => increase progress and reset done count
                do {
                    progress_local = progress.load();
                    progress_count = static_cast<std::uint32_t>((progress_local >> 32) & ((1ULL << 32) - 1));
                    done           = static_cast<std::uint32_t>(progress_local & ((1ULL << 32) - 1));
                    progress_new   = (progress_count + 1ULL) << 32;
                } while (!progress.compare_exchange_strong(progress_local, progress_new));
                progress.notify_all();
            } else { // nothing happened on this thread
                uint32_t progress_count_old = progress_count;
                do {
                    progress_local = progress.load();
                    progress_count = static_cast<std::uint32_t>((progress_local >> 32) & ((1ULL << 32) - 1));
                    done           = static_cast<std::uint32_t>(progress_local & ((1ULL << 32) - 1));
                    if (progress_count == progress_count_old) { // nothing happened => increase done count
                        progress_new = ((progress_count + 0ULL) << 32) + done + 1;
                    } else {                                    // something happened in another thread => keep progress and done count and rerun this task without waiting
                        progress_new = ((progress_count + 0ULL) << 32) + done;
                    }
                } while (!progress.compare_exchange_strong(progress_local, progress_new));
                progress.notify_all();
                if (progress_count == progress_count_old && done < n_batches) {
                    progress.wait(progress_new);
                }
            }
        } // while (done < n_batches)
        running_threads.count_down();
    }

    [[nodiscard]] const std::vector<std::vector<node_t>> &getJobLists() const {
        return _job_lists;
    }
};

/**
 * Trivial loop based scheduler, which iterates over all nodes in definition order in the graph until no node did any processing
 */
template<execution_policy executionPolicy = single_threaded>
class simple : public scheduler_base<simple<executionPolicy>>{
    using S = scheduler_base<simple<executionPolicy>>;
    using node_t = S::node_t; //node_model*;
    using thread_pool_type = S::thread_pool_type; //thread_pool::BasicThreadPool;
public:
    explicit simple(fair::graph::graph &&graph, std::shared_ptr<thread_pool_type> thread_pool = std::make_shared<thread_pool_type>("simple-scheduler-pool", thread_pool::CPU_BOUND))
            : S(std::move(graph), thread_pool) {
        // generate job list
        if constexpr (executionPolicy == multi_threaded) {
            const auto n_batches = std::min(static_cast<std::size_t>(this->_pool->maxThreads()), this->_graph.blocks().size());
            this->_job_lists.reserve(n_batches);
            for (std::size_t i = 0; i < n_batches; i++) {
                // create job-set for thread
                auto &job = this->_job_lists.emplace_back();
                job.reserve(this->_graph.blocks().size() / n_batches + 1);
                for (std::size_t j = i; j < this->_graph.blocks().size(); j += n_batches) {
                    job.push_back(this->_graph.blocks()[j].get());
                }
            }
        }
    }

    work_return_t
    work() {
        // if (!_init) {
        //     return work_return_t::ERROR;
        // }
        if constexpr (executionPolicy == single_threaded) {
            bool run = true;
            while (run) {
                if (auto result = this->traverse_nodes(std::span{this->_graph.blocks()}); result == work_return_t::ERROR) {
                    return result;
                } else {
                    run = result == work_return_t::OK;
                }
            }
        } else if (executionPolicy == multi_threaded) {
            std::atomic_bool stop_requested(false);
            std::atomic_uint64_t progress{0}; // upper uint32t: progress counter, lower uint32t: number of workers that finished all their work
            std::latch running_threads{static_cast<std::ptrdiff_t>(this->_job_lists.size())}; // latch to wait for completion of the flowgraph
            for (auto &job: this->_job_lists) {
                this->_pool->execute([this, &job, &progress, &running_threads, &stop_requested]() {
                    this->run_on_pool(std::span{job}, this->_job_lists.size(), progress, running_threads, stop_requested);
                });
            }
            running_threads.wait();
            return work_return_t::DONE;
        } else {
            throw std::invalid_argument("Unknown execution Policy");
        }
        return work_return_t::DONE;
    }
};

/**
 * Breadth first traversal scheduler which traverses the graph starting from the source nodes in a breath first fashion
 * detecting cycles and nodes which can be reached from several source nodes.
 */
template<execution_policy executionPolicy = single_threaded>
class breadth_first : public scheduler_base<breadth_first<executionPolicy>> {
    using S = scheduler_base<breadth_first<executionPolicy>>;
    using node_t = node_model*;
    using thread_pool_type = thread_pool::BasicThreadPool;
    std::vector<node_t> _nodelist;
public:
    explicit breadth_first(fair::graph::graph &&graph, std::shared_ptr<thread_pool_type> thread_pool = std::make_shared<thread_pool_type>("breadth-first-pool", thread_pool::CPU_BOUND))
                : S(std::move(graph), thread_pool) {
        std::map<node_t, std::vector<node_t>> _adjacency_list{};
        std::vector<node_t>                   _source_nodes{};
        // compute the adjacency list
        std::set<node_t> node_reached;
        for (auto &e : this->_graph.edges()) {
            _adjacency_list[e._src_node].push_back(e._dst_node);
            _source_nodes.push_back(e._src_node);
            node_reached.insert(e._dst_node);
        }
        _source_nodes.erase(std::remove_if(_source_nodes.begin(), _source_nodes.end(), [&node_reached](auto current_node) { return node_reached.contains(current_node); }), _source_nodes.end());
        // traverse graph
        std::queue<node_t> queue{};
        std::set<node_t>   reached;
        // add all source nodes to queue
        for (node_t source_node : _source_nodes) {
            if (!reached.contains(source_node)) {
                queue.push(source_node);
            }
            reached.insert(source_node);
        }
        // process all nodes, adding all unvisited child nodes to the queue
        while (!queue.empty()) {
            node_t current_node = queue.front();
            queue.pop();
            _nodelist.push_back(current_node);
            if (_adjacency_list.contains(current_node)) { // node has outgoing edges
                for (auto &dst : _adjacency_list.at(current_node)) {
                    if (!reached.contains(dst)) { // detect cycles. this could be removed if we guarantee cycle free graphs earlier
                        queue.push(dst);
                        reached.insert(dst);
                    }
                }
            }
        }
        // generate job list
        const auto n_batches = std::min(static_cast<std::size_t>(this->_pool->maxThreads()), _nodelist.size());
        this->_job_lists.reserve(n_batches);
        for (std::size_t i = 0; i < n_batches; i++) {
            // create job-set for thread
            auto &job = this->_job_lists.emplace_back();
            job.reserve(_nodelist.size() / n_batches + 1);
            for (std::size_t j = i; j < _nodelist.size(); j += n_batches) {
                job.push_back(_nodelist[j]);
            }
        }
    }

    work_return_t
    work() {
        // if (!_init) {
        //     return work_return_t::ERROR;
        // }
        if constexpr (executionPolicy == single_threaded) {
            bool run = true;
            while (run) {
                if (auto result = this->traverse_nodes(std::span{_nodelist}); result == work_return_t::ERROR) {
                    return work_return_t::ERROR;
                } else {
                    run = (result == work_return_t::OK);
                }
            }
        } else if (executionPolicy == multi_threaded) {
            std::atomic_bool stop_requested;
            std::atomic_uint64_t progress{0}; // upper uint32t: progress counter, lower uint32t: number of workers that finished all their work
            std::latch running_threads{static_cast<std::ptrdiff_t>(this->_job_lists.size())}; // latch to wait for completion of the flowgraph
            for (auto &job: this->_job_lists) {
                this->_pool->execute([this, &job, &progress, &running_threads, &stop_requested]() {
                    this->run_on_pool(std::span{job}, this->_job_lists.size(), progress, running_threads, stop_requested);
                });
            }
            running_threads.wait();
            return work_return_t::DONE;
        } else {
            throw std::invalid_argument("Unknown execution Policy");
        }
        return work_return_t::DONE;
    }

};
} // namespace fair::graph::scheduler

#endif // GRAPH_PROTOTYPE_SCHEDULER_HPP
