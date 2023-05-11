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
public:
    explicit simple(fair::graph::graph &graph)  : init{fair::graph::scheduler::init(graph)}, graph(graph) { }

    work_return_t work() {
        if (!init) {
            return work_return_t::ERROR;
        }
        bool run = true;
        while (run) {
            bool something_happened = false;
            for (auto &node : graph._nodes) {
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
private:
    init_proof init;
    fair::graph::graph &graph;
};

/**
 * Breadth first traversal scheduler which traverses the graph starting from the source nodes in a breath first fashion
 * detecting cycles and nodes which can be reached from several source nodes.
 */
class breadth_first : public node<breadth_first> {
public:
    explicit breadth_first(fair::graph::graph &graph) : init{fair::graph::scheduler::init(graph)} {
        // compute the adjacency list
        std::set<fair::graph::graph::node_model *> node_reached;
        for (auto &e : graph.get_edges()) {
            adjacency_list[e._src_node].insert(e._dst_node);
            source_nodes.insert(e._src_node);
            node_reached.insert(e._dst_node);
        }
        for (auto &dst : node_reached) {
            source_nodes.erase(dst);
        }
    }

    work_return_t work() {
        if (!init) {
            return work_return_t::ERROR;
        }
        while (true) {
            bool anything_happened = false;
            std::queue<fair::graph::graph::node_model *> queue{};
            std::set<fair::graph::graph::node_model *>   reached;
            // add all source nodes to queue
            for (fair::graph::graph::node_model *source_node : source_nodes) {
                queue.push(source_node);
                reached.insert(source_node);
            }
            // process all nodes, adding all unvisited child nodes to the queue
            while (!queue.empty()) {
                fair::graph::graph::node_model *node = queue.front();
                queue.pop();
                auto res = node->work();
                anything_happened |= (res == work_return_t::OK || res == work_return_t::INSUFFICIENT_OUTPUT_ITEMS);
                if (adjacency_list.contains(node)) { // node has outgoing edges
                    for (auto &dst : adjacency_list.at(node)) {
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

private:
    init_proof init;
    using node_t = fair::graph::graph::node_model*;
    std::map<node_t, std::set<node_t>> adjacency_list{};
    std::set<node_t> source_nodes{};
};
}
#endif // GRAPH_PROTOTYPE_SCHEDULER_HPP