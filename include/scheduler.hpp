#ifndef GRAPH_PROTOTYPE_SCHEDULER_HPP
#define GRAPH_PROTOTYPE_SCHEDULER_HPP
#include <graph.hpp>
#include <set>
#include <queue>

namespace fair::graph::scheduler {

struct init_proof {
    init_proof(bool _success) : success(_success) {}
    init_proof(init_proof && init): init_proof(init.success) {}
    bool success = true;

    init_proof& operator=(init_proof &&init)  noexcept {this->success = init; return *this;}
    operator bool() const { return success; }
};

init_proof init(fair::graph::graph &graph) {
    auto result = init_proof(
            std::all_of(graph._connection_definitions.begin(), graph._connection_definitions.end(), [] (auto& connection_definition) {
                return connection_definition() == connection_result_t::SUCCESS;
            }));
    graph._connection_definitions.clear();
    return result;
}

/**
 * Trivial loop based scheduler, which iterates over all nodes in definition order in the graph until no node did any processing
 */
class simple : public node<simple>{
    init_proof _init;
    fair::graph::graph _graph;
public:
    explicit simple(fair::graph::graph &&graph)  : _init{fair::graph::scheduler::init(graph)}, _graph(std::move(graph)) { }

    work_return_t work() {
        if (!_init) {
            return work_return_t::ERROR;
        }
        bool run = true;
        while (run) {
            bool something_happened = false;
            for (auto &node : _graph._nodes) {
                auto result = node->work();
                if (result == work_return_t::ERROR) {
                    return work_return_t::ERROR;
                } else if (result == work_return_t::INSUFFICIENT_INPUT_ITEMS) {
                    // nothing
                } else if (result == work_return_t::DONE) {
                    // nothing
                } else if (result == work_return_t::OK) {
                    something_happened = true;
                } else if (result == work_return_t::INSUFFICIENT_OUTPUT_ITEMS) {
                    something_happened = true;
                }
            }
            run = something_happened;
        }

        return work_return_t::DONE;
    }
};

/**
 * Breadth first traversal scheduler which traverses the graph starting from the source nodes in a breath first fashion
 * detecting cycles and nodes which can be reached from several source nodes.
 */
class breadth_first : public node<breadth_first> {
    init_proof _init;
    fair::graph::graph _graph;
    using node_t = fair::graph::graph::node_model*;
    std::map<node_t, std::set<node_t>> _adjacency_list{};
    std::set<node_t> _source_nodes{};
public:
    explicit breadth_first(fair::graph::graph &&graph) : _init{fair::graph::scheduler::init(graph)}, _graph(std::move(graph)) {
        // compute the adjacency list
        std::set<fair::graph::graph::node_model *> node_reached;
        for (auto &e : _graph.get_edges()) {
            _adjacency_list[e._src_node].insert(e._dst_node);
            _source_nodes.insert(e._src_node);
            node_reached.insert(e._dst_node);
        }
        for (auto &dst : node_reached) {
            _source_nodes.erase(dst);
        }
    }

    work_return_t work() {
        if (!_init) {
            return work_return_t::ERROR;
        }
        while (true) {
            bool anything_happened = false;
            std::queue<fair::graph::graph::node_model *> queue{};
            std::set<fair::graph::graph::node_model *>   reached;
            // add all source nodes to queue
            for (fair::graph::graph::node_model *source_node : _source_nodes) {
                queue.push(source_node);
                reached.insert(source_node);
            }
            // process all nodes, adding all unvisited child nodes to the queue
            while (!queue.empty()) {
                fair::graph::graph::node_model *node = queue.front();
                queue.pop();
                auto res = node->work();
                anything_happened |= (res == work_return_t::OK || res == work_return_t::INSUFFICIENT_OUTPUT_ITEMS);
                if (_adjacency_list.contains(node)) { // node has outgoing edges
                    for (auto &dst : _adjacency_list.at(node)) {
                        if (!reached.contains(dst)) { // detect cycles. this could be removed if we guarantee cycle free graphs earlier
                            queue.push(dst);
                            reached.insert(dst);
                        }
                    }
                }
            }
            if (!anything_happened) {
                return work_return_t::DONE;
            }
        }
    }
};
}

#endif // GRAPH_PROTOTYPE_SCHEDULER_HPP
