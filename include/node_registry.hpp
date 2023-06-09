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

struct node_construction_param {
    std::string_view key;
    std::string_view value;
};

struct node_construction_params {
    std::span<node_construction_param> params;

    template<typename Collection>
        requires(!std::is_same_v<std::remove_cvref_t<Collection>, node_construction_params>)
    node_construction_params(Collection &collection) : params(collection) {}

    node_construction_params(const node_construction_params &other) : params(other.params) {}

    node_construction_params &
    operator=(node_construction_params &)
            = delete;

    node_construction_params() {}

    std::string_view
    value(std::string_view key) const {
        auto it = std::find_if(params.begin(), params.end(), [&](const auto &param) { return key == param.key; });
        return it != params.end() ? it->value : std::string_view{};
    }
};

class node_registry {
private:
    using node_type_handler = std::function<void(std::unique_ptr<fair::graph::node_model> &, node_construction_params)>;
    std::vector<std::string>                                                            _node_types;
    std::unordered_map<std::string, std::unordered_map<std::string, node_type_handler>> _node_type_handlers;

    template<template<typename> typename NodeTemplate, typename ValueType>
    static auto
    create_handler() {
        return [](std::unique_ptr<fair::graph::node_model> &result, fair::graph::node_construction_params params) {
            using NodeType = NodeTemplate<ValueType>;

            if constexpr (std::is_constructible_v<NodeType, fair::graph::node_construction_params>) {
                result = std::make_unique<fair::graph::node_wrapper<NodeType>>(params); // gp_plugin_node::wrap(std::make_shared<NodeType>(params));
            } else if constexpr (std::is_constructible_v<NodeType>) {
                result = std::make_unique<fair::graph::node_wrapper<NodeType>>();
            } else {
                fair::meta::print_types<fair::meta::message_type<"Can not default-construct the node instance, nor to construct it from node_construction_params">, NodeType>{};
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
    create_node(std::string_view name, std::string_view type, node_construction_params params) {
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
