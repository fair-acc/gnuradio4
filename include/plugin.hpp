#ifndef GRAPH_PROTOTYPE_PLUGIN_H
#define GRAPH_PROTOTYPE_PLUGIN_H

#include <span>
#include <string>
#include <string_view>

#include <dlfcn.h>

#include <block_registry.hpp>
#include <graph.hpp>

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
    provided_blocks() const = 0;
    virtual std::unique_ptr<fair::graph::block_model>
    create_block(std::string_view name, std::string_view type, const fair::graph::property_map &params) = 0;
};

namespace fair::graph {
template<std::uint8_t ABI_VERSION = GP_PLUGIN_CURRENT_ABI_VERSION>
class plugin : public gp_plugin_base {
private:
    fair::graph::block_registry registry;

public:
    plugin() {}

    std::uint8_t
    abi_version() const override {
        return ABI_VERSION;
    }

    std::span<const std::string>
    provided_blocks() const override {
        return registry.provided_blocks();
    }

    std::unique_ptr<fair::graph::block_model>
    create_block(std::string_view name, std::string_view type, const property_map &params) override {
        return registry.create_block(name, type, params);
    }

    template<template<typename...> typename BlockTemplate, typename... Args>
    void
    add_block_type(std::string block_type) {
        registry.add_block_type<BlockTemplate, Args...>(std::move(block_type));
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

#define GP_PLUGIN_REGISTER_BLOCK(...) GP_REGISTER_BLOCK(gp_plugin_instance(), __VA_ARGS__);

#endif // include guard
