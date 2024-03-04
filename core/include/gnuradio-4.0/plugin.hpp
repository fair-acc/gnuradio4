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

struct GNURADIO_PLUGIN_EXPORT gr_plugin_metadata {
    std::string_view plugin_name;
    std::string_view plugin_author;
    std::string_view plugin_license;
    std::string_view plugin_version;
};

class GNURADIO_PLUGIN_EXPORT gr_plugin_base {
public:
    gr_plugin_metadata *metadata = nullptr;

    virtual ~gr_plugin_base();

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
class plugin : public gr_plugin_base {
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

    template<typename TBlock>
    void
    addBlockType(std::string blockType = {}, std::string blockParams = {}) {
        registry.addBlockType<TBlock>(std::move(blockType), std::move(blockParams));
    }
};

} // namespace gr

/*
 * Defines a plugin - creates the plugin meta-data and creates
 * a block registry (grPluginInstance()) for the plugin.
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
    gr::plugin<> & \
    grPluginInstance() { \
        static gr::plugin<> *instance = [] { \
            auto                     *result = new gr::plugin<>(); \
            static gr_plugin_metadata plugin_metadata{ Name, Author, License, Version }; \
            result->metadata = &plugin_metadata; \
            return result; \
        }(); \
        return *instance; \
    } \
    } \
    extern "C" { \
    void GNURADIO_PLUGIN_EXPORT \
    gr_plugin_make(gr_plugin_base **plugin) { \
        *plugin = &grPluginInstance(); \
    } \
    void GNURADIO_PLUGIN_EXPORT \
    gr_plugin_free(gr_plugin_base *plugin) { \
        if (plugin != &grPluginInstance()) { \
            assert(false && "Requested to delete something that is not us"); \
            return; \
        } \
        delete plugin; \
    } \
    }

#endif // include guard
