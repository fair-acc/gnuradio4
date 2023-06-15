#ifndef GRAPH_PROTOTYPE_NODE_REGISTRY_H
#define GRAPH_PROTOTYPE_NODE_REGISTRY_H

#include <memory>
#include <string>
#include <string_view>

#include <graph.hpp>
#include <utils.hpp>

namespace fair::graph {

using namespace std::string_literals;
using namespace std::string_view_literals;

class node_registry {
private:
    using node_type_handler = std::function<void(std::unique_ptr<fair::graph::node_model> &, const property_map &)>;
    std::vector<std::string>                                                            _node_types;
    std::unordered_map<std::string, std::unordered_map<std::string, node_type_handler>> _node_type_handlers;

    template<template<typename> typename NodeTemplate, typename ValueType>
    static auto
    create_handler() {
        return [](std::unique_ptr<fair::graph::node_model> &result, const property_map &params) {
            using NodeType = NodeTemplate<ValueType>;

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

public:
    template<template<typename> typename NodeTemplate, typename... AllowedTypes>
    void
    add_node_type(std::string node_type) {
        _node_types.push_back(node_type);
        auto &node_handlers = _node_type_handlers[node_type];

        ((node_handlers[std::string(meta::type_name<AllowedTypes>())] = create_handler<NodeTemplate, AllowedTypes>()), ...);
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
