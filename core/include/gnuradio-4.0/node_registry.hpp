#ifndef GRAPH_PROTOTYPE_NODE_REGISTRY_H
#define GRAPH_PROTOTYPE_NODE_REGISTRY_H

#include <memory>
#include <string>
#include <string_view>

#include "graph.hpp"
#include "utils.hpp"

namespace fair::graph {

using namespace std::string_literals;
using namespace std::string_view_literals;

class node_registry {
private:
    using node_type_handler = std::function<void(std::unique_ptr<fair::graph::node_model> &, const property_map &)>;
    std::vector<std::string>                                                            _node_types;
    std::unordered_map<std::string, std::unordered_map<std::string, node_type_handler>> _node_type_handlers;

    template<template<typename...> typename NodeTemplate, typename... NodeParameters>
    static auto
    create_handler() {
        return [](std::unique_ptr<fair::graph::node_model> &result, const property_map &params) {
            using NodeType = NodeTemplate<NodeParameters...>;

            if constexpr (std::is_constructible_v<NodeType, const property_map &>) {
                result = std::make_unique<fair::graph::node_wrapper<NodeType>>(params); // gp_plugin_node::wrap(std::make_shared<NodeType>(params));
            } else if constexpr (std::is_constructible_v<NodeType>) {
                result = std::make_unique<fair::graph::node_wrapper<NodeType>>();
            } else {
                fair::meta::print_types<fair::meta::message_type<"Can not default-construct the node instance, nor to construct it from const property_map&">, NodeType>{};
            }
            return true;
        };
    }

    auto &
    find_node_type_handlers_map(const std::string &node_type) {
        if (auto it = _node_type_handlers.find(node_type); it != _node_type_handlers.end()) {
            return it->second;
        } else {
            _node_types.emplace_back(node_type);
            return _node_type_handlers[node_type];
        }
    }

    template<typename... Types>
    static std::string
    encoded_list_of_types() {
        struct accumulator {
            std::string value;

            accumulator &
            operator%(const std::string &type) {
                if (value.empty()) value = type;
                else
                    value += ";"s + type;
                return *this;
            }
        };

        return (accumulator{} % ... % meta::type_name<Types>()).value;
    }

public:
    template<template<typename...> typename NodeTemplate, typename... NodeParameters>
    void
    add_node_type(std::string node_type) {
        auto &node_handlers                                       = find_node_type_handlers_map(node_type);
        node_handlers[encoded_list_of_types<NodeParameters...>()] = create_handler<NodeTemplate, NodeParameters...>();
        fmt::print("Registered {} {}\n", node_type, encoded_list_of_types<NodeParameters...>());
    }

    std::span<const std::string>
    provided_nodes() const {
        return _node_types;
    }

    std::unique_ptr<fair::graph::node_model>
    create_node(std::string_view name, std::string_view type, const property_map &params) {
        std::unique_ptr<fair::graph::node_model> result;
        auto                                     node_it = _node_type_handlers.find(std::string(name));
        if (node_it == _node_type_handlers.end()) return nullptr;

        auto &node_handlers = node_it->second;
        auto  handler_it    = node_handlers.find(std::string(type));
        if (handler_it == node_handlers.end()) return nullptr;

        handler_it->second(result, params);
        return result;
    }

    auto
    known_nodes() const {
        return _node_types;
    }
};

} // namespace fair::graph

#endif // include guard
