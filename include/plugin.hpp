#ifndef GRAPH_PROTOTYPE_PLUGIN_H
#define GRAPH_PROTOTYPE_PLUGIN_H

#include <span>
#include <string>
#include <string_view>

#include <dlfcn.h>

#include <graph.hpp>
#include <node_registry.hpp>

#include <graph-prototype-plugin_export.h>

using namespace std::string_literals;
using namespace std::string_view_literals;

#define GP_PLUGIN_CURRENT_ABI_VERSION 1

struct GRAPH_PROTOTYPE_PLUGIN_EXPORT gp_plugin_metadata {
    std::string_view plugin_name;
    std::string_view plugin_author;
    std::string_view plugin_license;
    std::string_view plugin_version;
};

class GRAPH_PROTOTYPE_PLUGIN_EXPORT gp_plugin_base {
public:
    gp_plugin_metadata *metadata = nullptr;

    virtual ~gp_plugin_base();

    virtual std::uint8_t
    abi_version() const
            = 0;

    virtual std::span<const std::string>
    provided_nodes() const = 0;
    virtual std::unique_ptr<fair::graph::node_model>
    create_node(std::string_view name, std::string_view type, const fair::graph::property_map &params) = 0;
};

namespace fair::graph {
template<std::uint8_t ABI_VERSION = GP_PLUGIN_CURRENT_ABI_VERSION>
class plugin : public gp_plugin_base {
private:
    fair::graph::node_registry registry;

public:
    plugin() {}

    std::uint8_t
    abi_version() const override {
        return ABI_VERSION;
    }

    std::span<const std::string>
    provided_nodes() const override {
        return registry.provided_nodes();
    }

    std::unique_ptr<fair::graph::node_model>
    create_node(std::string_view name, std::string_view type, const property_map &params) override {
        return registry.create_node(name, type, params);
    }

    template<template<typename...> typename NodeTemplate, typename... Args>
    void
    add_node_type(std::string node_type) {
        registry.add_node_type<NodeTemplate, Args...>(std::move(node_type));
    }
};

} // namespace fair::graph

#define GP_PLUGIN(Name, Author, License, Version) \
    inline namespace GP_PLUGIN_DEFINITION_NAMESPACE { \
    fair::graph::plugin<> * \
    gp_plugin_instance() { \
        static fair::graph::plugin<> *instance = [] { \
            auto                     *result = new fair::graph::plugin<>(); \
            static gp_plugin_metadata plugin_metadata{ Name, Author, License, Version }; \
            result->metadata = &plugin_metadata; \
            return result; \
        }(); \
        return instance; \
    } \
    } \
    extern "C" { \
    void GRAPH_PROTOTYPE_PLUGIN_EXPORT \
    gp_plugin_make(gp_plugin_base **plugin) { \
        *plugin = gp_plugin_instance(); \
    } \
    void GRAPH_PROTOTYPE_PLUGIN_EXPORT \
    gp_plugin_free(gp_plugin_base *plugin) { \
        if (plugin != gp_plugin_instance()) { \
            assert(false && "Requested to delete something that is not us"); \
            return; \
        } \
        delete plugin; \
    } \
    }

#define GP_PLUGIN_REGISTER_NODE(...) GP_REGISTER_NODE(gp_plugin_instance(), __VA_ARGS__);
#define GP_PLUGIN_REGISTER_NODE_MULTI_PARAMETER(...) GP_REGISTER_NODE_MULTI_PARAMETER(gp_plugin_instance(), __VA_ARGS__);
#define GP_PLUGIN_REGISTER_NODE_EXPERIMENTAL(...) GP_REGISTER_NODE_EXPERIMENTAL(gp_plugin_instance(), __VA_ARGS__);

#endif // include guard
