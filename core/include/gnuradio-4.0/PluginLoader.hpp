#ifndef GNURADIO_PLUGIN_LOADER_H
#define GNURADIO_PLUGIN_LOADER_H

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "BlockRegistry.hpp"
#include "Graph.hpp"

#ifndef __EMSCRIPTEN__
#include <dlfcn.h>

#include "plugin.hpp"
#endif

namespace gr {

using namespace std::string_literals;
using namespace std::string_view_literals;

#ifndef __EMSCRIPTEN__
// Plugins are not supported on WASM

using plugin_create_function_t  = void (*)(gr_plugin_base **);
using plugin_destroy_function_t = void (*)(gr_plugin_base *);

class PluginHandler {
private:
    void                     *_dl_handle  = nullptr;
    plugin_create_function_t  _create_fn  = nullptr;
    plugin_destroy_function_t _destroy_fn = nullptr;
    gr_plugin_base           *_instance   = nullptr;

    std::string _status;

    void
    release() {
        if (_instance) {
            _destroy_fn(_instance);
            _instance = nullptr;
        }

        if (_dl_handle) {
            dlclose(_dl_handle);
            _dl_handle = nullptr;
        }
    }

public:
    PluginHandler() = default;

    explicit PluginHandler(const std::string &plugin_file) {
        _dl_handle = dlopen(plugin_file.c_str(), RTLD_LAZY);
        if (!_dl_handle) {
            _status = "Failed to load the plugin file";
            return;
        }

        _create_fn = reinterpret_cast<plugin_create_function_t>(dlsym(_dl_handle, "gr_plugin_make"));
        if (!_create_fn) {
            _status = "Failed to load symbol gr_plugin_make";
            release();
            return;
        }

        _destroy_fn = reinterpret_cast<plugin_destroy_function_t>(dlsym(_dl_handle, "gr_plugin_free"));
        if (!_destroy_fn) {
            _status = "Failed to load symbol gr_plugin_free";
            release();
            return;
        }

        _create_fn(&_instance);
        if (!_instance) {
            _status = "Failed to create an instance of the plugin";
            release();
            return;
        }

        if (_instance->abi_version() != GR_PLUGIN_CURRENT_ABI_VERSION) {
            _status = "Wrong ABI version";
            release();
            return;
        }
    }

    PluginHandler(const PluginHandler &other) = delete;
    PluginHandler &
    operator=(const PluginHandler &other)
            = delete;

    PluginHandler(PluginHandler &&other) noexcept
        : _dl_handle(std::exchange(other._dl_handle, nullptr))
        , _create_fn(std::exchange(other._create_fn, nullptr))
        , _destroy_fn(std::exchange(other._destroy_fn, nullptr))
        , _instance(std::exchange(other._instance, nullptr)) {}

    PluginHandler &
    operator=(PluginHandler &&other) noexcept {
        auto tmp = std::move(other);
        std::swap(_dl_handle, tmp._dl_handle);
        std::swap(_create_fn, tmp._create_fn);
        std::swap(_destroy_fn, tmp._destroy_fn);
        std::swap(_instance, tmp._instance);
        return *this;
    }

    ~PluginHandler() { release(); }

    explicit
    operator bool() const {
        return _instance;
    }

    [[nodiscard]] const std::string &
    status() const {
        return _status;
    }

    auto *
    operator->() const {
        return _instance;
    }
};

class PluginLoader {
private:
    std::vector<PluginHandler>                        _handlers;
    std::unordered_map<std::string, gr_plugin_base *> _handlerForName;
    std::unordered_map<std::string, std::string>      _failedPlugins;
    std::unordered_set<std::string>                   _loadedPluginFiles;

    BlockRegistry           *_registry;
    std::vector<std::string> _knownBlocks;

public:
    PluginLoader(BlockRegistry &registry, std::span<const std::filesystem::path> plugin_directories) : _registry(&registry) {
        for (const auto &directory : plugin_directories) {
            std::cerr << std::filesystem::current_path() << std::endl;

            if (!std::filesystem::is_directory(directory)) continue;

            for (const auto &file : std::filesystem::directory_iterator{ directory }) {
                if (file.is_regular_file() && file.path().extension() == ".so") {
                    auto fileString = file.path().string();
                    if (_loadedPluginFiles.contains(fileString)) continue;
                    _loadedPluginFiles.insert(fileString);

                    if (PluginHandler handler(file.path().string()); handler) {
                        for (const auto &block_name : handler->providedBlocks()) {
                            _handlerForName.emplace(std::string(block_name), handler.operator->());
                            _knownBlocks.emplace_back(block_name);
                        }

                        _handlers.push_back(std::move(handler));

                    } else {
                        _failedPlugins[file.path()] = handler.status();
                    }
                }
            }
        }
    }

    const auto &
    plugins() const {
        return _handlers;
    }

    const auto &
    failed_plugins() const {
        return _failedPlugins;
    }

    auto
    knownBlocks() const {
        auto        result  = _knownBlocks;
        const auto &builtin = _registry->knownBlocks();
        result.insert(result.end(), builtin.begin(), builtin.end());
        return result;
    }

    std::unique_ptr<gr::BlockModel>
    instantiate(std::string_view name, std::string_view type, const property_map &params = {}) {
        // Try to create a node from the global registry
        if (auto result = _registry->createBlock(name, type, params)) {
            return result;
        }
        auto it = _handlerForName.find(std::string(name)); // TODO avoid std::string here
        if (it == _handlerForName.end()) return {};

        auto &handler = it->second;

        return handler->createBlock(name, type, params);
    }

    template<typename Graph, typename... InstantiateArgs>
    gr::BlockModel &
    instantiateInGraph(Graph &graph, InstantiateArgs &&...args) {
        auto block_load = instantiate(std::forward<InstantiateArgs>(args)...);
        if (!block_load) {
            throw fmt::format("Unable to create node");
        }
        return graph.addBlock(std::move(block_load));
    }
};
#else
// PluginLoader on WASM is just a wrapper on BlockRegistry to provide the
// same API as proper PluginLoader
class PluginLoader {
private:
    BlockRegistry *_registry;

public:
    PluginLoader(BlockRegistry &registry, std::span<const std::filesystem::path> /*plugin_directories*/) : _registry(&registry) {}

    auto
    knownBlocks() const {
        return _registry->knownBlocks();
    }

    std::unique_ptr<gr::BlockModel>
    instantiate(std::string_view name, std::string_view type, const property_map &params = {}) {
        return _registry->createBlock(name, type, params);
    }

    template<typename Graph, typename... InstantiateArgs>
    gr::BlockModel &
    instantiateInGraph(Graph &graph, InstantiateArgs &&...args) {
        auto block_load = instantiate(std::forward<InstantiateArgs>(args)...);
        if (!block_load) {
            throw fmt::format("Unable to create node");
        }
        return graph.addBlock(std::move(block_load));
    }
};
#endif

} // namespace gr

#endif // include guard
