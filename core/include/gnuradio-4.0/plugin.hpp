#ifndef GNURADIO_PLUGIN_H
#define GNURADIO_PLUGIN_H

#include <span>
#include <string>
#include <string_view>

#include <dlfcn.h>

#include "BlockRegistry.hpp"
#include "Graph.hpp"

#include <gnuradio-plugin_export.h>

using namespace std::string_literals;
using namespace std::string_view_literals;

#define GR_PLUGIN_CURRENT_ABI_VERSION 1

struct GNURADIO_PLUGIN_EXPORT gp_plugin_metadata {
    std::string_view plugin_name;
    std::string_view plugin_author;
    std::string_view plugin_license;
    std::string_view plugin_version;
};

class GNURADIO_PLUGIN_EXPORT gp_plugin_base {
public:
    gp_plugin_metadata *metadata = nullptr;

    virtual ~gp_plugin_base();

    virtual std::uint8_t
    abi_version() const
            = 0;

    virtual std::span<const std::string>
    providedBlocks() const = 0;
    virtual std::unique_ptr<gr::BlockModel>
    createBlock(std::string_view name, std::string_view type, const gr::property_map &params) = 0;
};

namespace gr {
template<std::uint8_t ABI_VERSION = GR_PLUGIN_CURRENT_ABI_VERSION>
class plugin : public gp_plugin_base {
private:
    gr::BlockRegistry registry;

public:
    plugin() {}

    std::uint8_t
    abi_version() const override {
        return ABI_VERSION;
    }

    std::span<const std::string>
    providedBlocks() const override {
        return registry.providedBlocks();
    }

    std::unique_ptr<gr::BlockModel>
    createBlock(std::string_view name, std::string_view type, const property_map &params) override {
        return registry.createBlock(name, type, params);
    }

    template<template<typename...> typename TBlock, typename... Args>
    void
    addBlockType(std::string block_type) {
        std::cout << "New block type: " << block_type << std::endl;
        registry.addBlockType<TBlock, Args...>(std::move(block_type));
    }
};

} // namespace gr

/*
 * Defines a plugin - creates the plugin meta-data and creates
 * a block registry for the plugin.
 *
 * Arguments:
 *  - plugin name
 *  - author
 *  - license of the plugin
 *  - plugin version
 *
 * Example usage:
 *     GR_PLUGIN("Good Base Plugin", "Unknown", "LGPL3", "v1")
 */
#define GR_PLUGIN(Name, Author, License, Version) \
    inline namespace GR_PLUGIN_DEFINITION_NAMESPACE { \
    gr::plugin<> * \
    gp_plugin_instance() { \
        static gr::plugin<> *instance = [] { \
            auto                     *result = new gr::plugin<>(); \
            static gp_plugin_metadata plugin_metadata{ Name, Author, License, Version }; \
            result->metadata = &plugin_metadata; \
            return result; \
        }(); \
        return instance; \
    } \
    } \
    extern "C" { \
    void GNURADIO_PLUGIN_EXPORT \
    gp_plugin_make(gp_plugin_base **plugin) { \
        *plugin = gp_plugin_instance(); \
    } \
    void GNURADIO_PLUGIN_EXPORT \
    gp_plugin_free(gp_plugin_base *plugin) { \
        if (plugin != gp_plugin_instance()) { \
            assert(false && "Requested to delete something that is not us"); \
            return; \
        } \
        delete plugin; \
    } \
    }

/**
 * This macro can be used to register a block defined in a plugin
 * (a library that contains block definitions that will be dynamically
 * loaded by the PluginLoader)
 *
 * Note that you first need to call GR_PLUGIN to create the plugin meta-data
 * and to create the block registry for the plugin.
 *
 * The arguments are:
 *  - the block template class
 *  - list of valid template parameters for this block type
 *
 * To register adder<T> block to be instantiatiatable with float and double:
 *     GR_PLUGIN_REGISTER_BLOCK(adder, float, double)
 *
 * To register converter<From, To> block to be instantiatiatable
 * with <float, double> and <double, float>:
 *     GR_PLUGIN_REGISTER_BLOCK(converter, BlockParameters<double, float>, BlockParameters<float, double>)
 */
#define GR_PLUGIN_REGISTER_BLOCK(...) GR_REGISTER_BLOCK(*gp_plugin_instance(), __VA_ARGS__);

#endif // include guard
