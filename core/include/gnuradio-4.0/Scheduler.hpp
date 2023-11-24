#ifndef GNURADIO_SCHEDULER_HPP
#define GNURADIO_SCHEDULER_HPP

#include <barrier>
#include <set>
#include <utility>
#include <queue>

#include "Graph.hpp"
#include "Profiler.hpp"
#include "thread/thread_pool.hpp"

namespace gr::scheduler {
using gr::thread_pool::BasicThreadPool;

enum ExecutionPolicy { singleThreaded, multiThreaded };

enum SchedulerState { IDLE, INITIALISED, RUNNING, REQUESTED_STOP, REQUESTED_PAUSE, STOPPED, PAUSED, SHUTTING_DOWN, ERROR };

template<profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class SchedulerBase {
protected:
    SchedulerState                      _state = IDLE;
    gr::Graph                           _graph;
    TProfiler                           _profiler;
    decltype(_profiler.forThisThread()) _profiler_handler;
    std::shared_ptr<BasicThreadPool>    _pool;
    std::atomic_uint64_t                _progress;
    std::atomic_size_t                  _running_threads;
    std::atomic_bool                    _stop_requested;

public:
    explicit SchedulerBase(gr::Graph &&graph, std::shared_ptr<BasicThreadPool> thread_pool = std::make_shared<BasicThreadPool>("simple-scheduler-pool", thread_pool::CPU_BOUND),
                           const profiling::Options &profiling_options = {})
        : _graph(std::move(graph)), _profiler{ profiling_options }, _profiler_handler{ _profiler.forThisThread() }, _pool(std::move(thread_pool)) {}

    ~SchedulerBase() {
        stop();
        _state = SHUTTING_DOWN;
    }

    void
    stop() {
        if (_state == STOPPED || _state == ERROR) {
            return;
        }
        if (_state == RUNNING) {
            requestStop();
        }
        waitDone();
        _state = STOPPED;
    }

    void
    pause() {
        if (_state == PAUSED || _state == ERROR) {
            return;
        }
        if (_state == RUNNING) {
            requestPause();
        }
        waitDone();
        _state = PAUSED;
    }

    void
    waitDone() {
        [[maybe_unused]] const auto pe = _profiler_handler.startCompleteEvent("scheduler_base.waitDone");
        for (auto running = _running_threads.load(); running > 0ul; running = _running_threads.load()) {
            _running_threads.wait(running);
        }
        if (_state == REQUESTED_PAUSE) {
            _state = PAUSED;
        } else {
            _state = STOPPED;
        }
    }

    void
    requestStop() {
        _stop_requested = true;
        _state          = REQUESTED_STOP;
    }

    void
    requestPause() {
        _stop_requested = true;
        _state          = REQUESTED_PAUSE;
    }

    void
    init() {
        [[maybe_unused]] const auto pe = _profiler_handler.startCompleteEvent("scheduler_base.init");
        if (_state != IDLE) {
            return;
        }
        const auto result = _graph.performConnections();
        if (result) {
            _state = INITIALISED;
        } else {
            _state = ERROR;
        }
    }

    void
    reset() {
        // since it is not possible to set up the graph connections a second time, this method leaves the graph in the initialized state with clear buffers.
        switch (_state) {
        case IDLE: init(); break;
        case RUNNING:
        case REQUESTED_STOP:
        case REQUESTED_PAUSE:
            pause();
            // intentional fallthrough
            FMT_FALLTHROUGH;
        case STOPPED:
            // clear buffers
            // std::for_each(_graph.edges().begin(), _graph.edges().end(), [](auto &edge) {
            //
            // });
            FMT_FALLTHROUGH;
        case PAUSED: _state = INITIALISED; break;
        case SHUTTING_DOWN:
        case INITIALISED:
        case ERROR: break;
        }
    }

    void
    runOnPool(const std::vector<std::vector<BlockModel *>> &jobs, const std::function<work::Result(const std::span<BlockModel *const> &)> work_function) {
        [[maybe_unused]] const auto pe = _profiler_handler.startCompleteEvent("scheduler_base.runOnPool");
        _progress                      = 0;
        _running_threads               = jobs.size();
        for (auto &jobset : jobs) {
            _pool->execute([this, &jobset, work_function, &jobs]() { poolWorker([&work_function, &jobset]() { return work_function(jobset); }, jobs.size()); });
        }
    }

    void
    poolWorker(const std::function<work::Result()> &work, std::size_t n_batches) {
        auto &profiler_handler = _profiler.forThisThread();

        uint32_t done           = 0;
        uint32_t progress_count = 0;
        while (done < n_batches && !_stop_requested) {
            auto pe                 = profiler_handler.startCompleteEvent("scheduler_base.work");
            bool something_happened = work().status == work::Status::OK;
            pe.finish();
            uint64_t progress_local = 0ULL;
            uint64_t progress_new   = 0ULL;
            if (something_happened) { // something happened in this thread => increase progress and reset done count
                do {
                    progress_local = _progress.load();
                    progress_count = static_cast<std::uint32_t>((progress_local >> 32) & ((1ULL << 32) - 1));
                    done           = static_cast<std::uint32_t>(progress_local & ((1ULL << 32) - 1));
                    progress_new   = (progress_count + 1ULL) << 32;
                } while (!_progress.compare_exchange_strong(progress_local, progress_new));
                _progress.notify_all();
            } else { // nothing happened on this thread
                uint32_t progress_count_old = progress_count;
                do {
                    progress_local = _progress.load();
                    progress_count = static_cast<std::uint32_t>((progress_local >> 32) & ((1ULL << 32) - 1));
                    done           = static_cast<std::uint32_t>(progress_local & ((1ULL << 32) - 1));
                    if (progress_count == progress_count_old) { // nothing happened => increase done count
                        progress_new = ((progress_count + 0ULL) << 32) + done + 1;
                    } else { // something happened in another thread => keep progress and done count and rerun this task without waiting
                        progress_new = ((progress_count + 0ULL) << 32) + done;
                    }
                } while (!_progress.compare_exchange_strong(progress_local, progress_new));
                _progress.notify_all();
                if (progress_count == progress_count_old && done + 1 < n_batches) {
                    _progress.wait(progress_new);
                }
            }
        } // while (done < n_batches)

        _running_threads.fetch_sub(1);
        _running_threads.notify_all();
    }
};

/**
 * Trivial loop based scheduler, which iterates over all blocks in definition order in the graph until no node did any processing
 */
template<ExecutionPolicy executionPolicy = singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class Simple : public SchedulerBase<TProfiler> {
    std::vector<std::vector<BlockModel *>> _job_lists{};

public:
    explicit Simple(gr::Graph &&graph, std::shared_ptr<BasicThreadPool> thread_pool = std::make_shared<BasicThreadPool>("simple-scheduler-pool", thread_pool::CPU_BOUND),
                    const profiling::Options &profiling_options = {})
        : SchedulerBase<TProfiler>(std::move(graph), thread_pool, profiling_options) {}

    void
    init() {
        SchedulerBase<TProfiler>::init();
        [[maybe_unused]] const auto pe = this->_profiler_handler.startCompleteEvent("scheduler_simple.init");
        // generate job list
        if constexpr (executionPolicy == multiThreaded) {
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

    template<typename block_type>
    work::Result
    workOnce(const std::span<block_type> &blocks) {
        constexpr std::size_t requested_work     = std::numeric_limits<std::size_t>::max();
        bool                  something_happened = false;
        std::size_t           performed_work     = 0_UZ;
        for (auto &currentBlock : blocks) {
            auto result = currentBlock->work(requested_work);
            performed_work += result.performed_work;
            if (result.status == work::Status::ERROR) {
                return { requested_work, performed_work, work::Status::ERROR };
            } else if (result.status == work::Status::INSUFFICIENT_INPUT_ITEMS || result.status == work::Status::DONE) {
                // nothing
            } else if (result.status == work::Status::OK || result.status == work::Status::INSUFFICIENT_OUTPUT_ITEMS) {
                something_happened = true;
            }
            if (currentBlock->isBlocking()) { // work-around for `DONE` issue when running with multithreaded BlockingIO blocks -> TODO: needs a better solution on a global scope
                std::vector<std::size_t> available_input_samples(20);
                std::ignore = currentBlock->availableInputSamples(available_input_samples);
                something_happened |= std::accumulate(available_input_samples.begin(), available_input_samples.end(), 0_UZ) > 0_UZ;
            }
        }
        return { requested_work, performed_work, something_happened ? work::Status::OK : work::Status::DONE };
    }

    // todo: could be moved to base class, but would make `start()` virtual or require CRTP
    // todo: iterate api for continuous flowgraphs vs ones that become "DONE" at some point
    void
    runAndWait() {
        [[maybe_unused]] const auto pe = this->_profiler_handler.startCompleteEvent("scheduler_simple.runAndWait");
        start();
        this->waitDone();
    }

    void
    start() {
        switch (this->_state) {
        case IDLE: this->init(); break;
        case STOPPED: this->reset(); break;
        case PAUSED: this->_state = INITIALISED; break;
        case INITIALISED:
        case RUNNING:
        case REQUESTED_PAUSE:
        case REQUESTED_STOP:
        case SHUTTING_DOWN:
        case ERROR: break;
        }
        if (this->_state != INITIALISED) {
            throw std::runtime_error("simple scheduler work(): graph not initialised");
        }
        if constexpr (executionPolicy == singleThreaded) {
            this->_state = RUNNING;
            work::Result result;
            auto         blocklist = std::span{ this->_graph.blocks() };
            do {
                result = workOnce(blocklist);
            } while (result.status == work::Status::OK);
            if (result.status == work::Status::ERROR) {
                this->_state = ERROR;
            } else {
                this->_state = STOPPED;
            }
        } else if (executionPolicy == multiThreaded) {
            this->runOnPool(this->_job_lists, [this](auto &job) { return this->workOnce(job); });
        } else {
            throw std::invalid_argument("Unknown execution Policy");
        }
    }
};

/**
 * Breadth first traversal scheduler which traverses the graph starting from the source blocks in a breath first fashion
 * detecting cycles and blocks which can be reached from several source blocks.
 */
template<ExecutionPolicy executionPolicy = singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class BreadthFirst : public SchedulerBase<TProfiler> {
    std::vector<BlockModel *>              _blocklist;
    std::vector<std::vector<BlockModel *>> _job_lists{};

public:
    explicit BreadthFirst(gr::Graph &&graph, std::shared_ptr<BasicThreadPool> thread_pool = std::make_shared<BasicThreadPool>("breadth-first-pool", thread_pool::CPU_BOUND),
                          const profiling::Options &profiling_options = {})
        : SchedulerBase<TProfiler>(std::move(graph), thread_pool, profiling_options) {}

    void
    init() {
        [[maybe_unused]] const auto pe = this->_profiler_handler.startCompleteEvent("breadth_first.init");
        using block_t                  = BlockModel *;
        SchedulerBase<TProfiler>::init();
        // calculate adjacency list
        std::map<block_t, std::vector<block_t>> _adjacency_list{};
        std::vector<block_t>                    _source_blocks{};
        // compute the adjacency list
        std::set<block_t> block_reached;
        for (auto &e : this->_graph.edges()) {
            _adjacency_list[e._sourceBlock].push_back(e._destinationBlock);
            _source_blocks.push_back(e._sourceBlock);
            block_reached.insert(e._destinationBlock);
        }
        _source_blocks.erase(std::remove_if(_source_blocks.begin(), _source_blocks.end(), [&block_reached](auto currentBlock) { return block_reached.contains(currentBlock); }), _source_blocks.end());
        // traverse graph
        std::queue<block_t> queue{};
        std::set<block_t>   reached;
        // add all source blocks to queue
        for (block_t sourceBlock : _source_blocks) {
            if (!reached.contains(sourceBlock)) {
                queue.push(sourceBlock);
            }
            reached.insert(sourceBlock);
        }
        // process all blocks, adding all unvisited child blocks to the queue
        while (!queue.empty()) {
            block_t currentBlock = queue.front();
            queue.pop();
            _blocklist.push_back(currentBlock);
            if (_adjacency_list.contains(currentBlock)) { // node has outgoing edges
                for (auto &dst : _adjacency_list.at(currentBlock)) {
                    if (!reached.contains(dst)) { // detect cycles. this could be removed if we guarantee cycle free graphs earlier
                        queue.push(dst);
                        reached.insert(dst);
                    }
                }
            }
        }
        // generate job list
        if constexpr (executionPolicy == multiThreaded) {
            const auto n_batches = std::min(static_cast<std::size_t>(this->_pool->maxThreads()), _blocklist.size());
            this->_job_lists.reserve(n_batches);
            for (std::size_t i = 0; i < n_batches; i++) {
                // create job-set for thread
                auto &job = this->_job_lists.emplace_back();
                job.reserve(_blocklist.size() / n_batches + 1);
                for (std::size_t j = i; j < _blocklist.size(); j += n_batches) {
                    job.push_back(_blocklist[j]);
                }
            }
        }
    }

    template<typename block_type>
    work::Result
    workOnce(const std::span<block_type> &blocks) {
        constexpr std::size_t requested_work     = std::numeric_limits<std::size_t>::max();
        bool                  something_happened = false;
        std::size_t           performed_work     = 0_UZ;

        for (auto &currentBlock : blocks) {
            auto result = currentBlock->work(requested_work);
            performed_work += result.performed_work;
            if (result.status == work::Status::ERROR) {
                return { requested_work, performed_work, work::Status::ERROR };
            } else if (result.status == work::Status::INSUFFICIENT_INPUT_ITEMS || result.status == work::Status::DONE) {
                // nothing
            } else if (result.status == work::Status::OK || result.status == work::Status::INSUFFICIENT_OUTPUT_ITEMS) {
                something_happened = true;
            }

            if (currentBlock->isBlocking()) { // work-around for `DONE` issue when running with multithreaded BlockingIO blocks -> TODO: needs a better solution on a global scope
                std::vector<std::size_t> available_input_samples(20);
                std::ignore = currentBlock->availableInputSamples(available_input_samples);
                something_happened |= std::accumulate(available_input_samples.begin(), available_input_samples.end(), 0_UZ) > 0_UZ;
            }
        }

        return { requested_work, performed_work, something_happened ? work::Status::OK : work::Status::DONE };
    }

    void
    runAndWait() {
        start();
        this->waitDone();
    }

    void
    start() {
        switch (this->_state) {
        case IDLE: this->init(); break;
        case STOPPED: this->reset(); break;
        case PAUSED: this->_state = INITIALISED; break;
        case INITIALISED:
        case RUNNING:
        case REQUESTED_PAUSE:
        case REQUESTED_STOP:
        case SHUTTING_DOWN:
        case ERROR: break;
        }
        if (this->_state != INITIALISED) {
            throw std::runtime_error("simple scheduler work(): graph not initialised");
        }
        if constexpr (executionPolicy == singleThreaded) {
            this->_state = RUNNING;
            work::Result result;
            auto         blocklist = std::span{ this->_blocklist };
            while ((result = workOnce(blocklist)).status == work::Status::OK) {
                if (result.status == work::Status::ERROR) {
                    this->_state = ERROR;
                    return;
                }
            }
            this->_state = STOPPED;
        } else if (executionPolicy == multiThreaded) {
            this->runOnPool(this->_job_lists, [this](auto &job) { return this->workOnce(job); });
        } else {
            throw std::invalid_argument("Unknown execution Policy");
        }
    }

    [[nodiscard]] const std::vector<std::vector<BlockModel *>> &
    jobs() const {
        return _job_lists;
    }
};
} // namespace gr::scheduler

#endif // GNURADIO_SCHEDULER_HPP
