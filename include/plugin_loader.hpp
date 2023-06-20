#ifndef GRAPH_PROTOTYPE_PLUGIN_LOADER_H
#define GRAPH_PROTOTYPE_PLUGIN_LOADER_H

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dlfcn.h>
#include <graph.hpp>
#include <plugin.hpp>

using plugin_create_function_t  = void (*)(gp_plugin_base **);
using plugin_destroy_function_t = void (*)(gp_plugin_base *);

namespace fair::graph {

using namespace std::string_literals;
using namespace std::string_view_literals;

#ifndef __EMSCRIPTEN__
// Plugins are not supported on WASM

class plugin_handler {
private:
    void                     *_dl_handle  = nullptr;
    plugin_create_function_t  _create_fn  = nullptr;
    plugin_destroy_function_t _destroy_fn = nullptr;
    gp_plugin_base           *_instance   = nullptr;

    std::string               _status;

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
    plugin_handler() = default;

    explicit plugin_handler(const std::string& plugin_file) {
        _dl_handle = dlopen(plugin_file.c_str(), RTLD_LAZY);
        if (!_dl_handle) {
            _status = "Failed to load the plugin file";
            return;
        }

        _create_fn = reinterpret_cast<plugin_create_function_t>(dlsym(_dl_handle, "gp_plugin_make"));
        if (!_create_fn) {
            _status = "Failed to load symbol gp_plugin_make";
            release();
            return;
        }

        _destroy_fn = reinterpret_cast<plugin_destroy_function_t>(dlsym(_dl_handle, "gp_plugin_free"));
        if (!_destroy_fn) {
            _status = "Failed to load symbol gp_plugin_free";
            release();
            return;
        }

        _create_fn(&_instance);
        if (!_instance) {
            _status = "Failed to create an instance of the plugin";
            release();
            return;
        }

        if (_instance->abi_version() != GP_PLUGIN_CURRENT_ABI_VERSION) {
            _status = "Wrong ABI version";
            release();
            return;
        }
    }

    plugin_handler(const plugin_handler &other) = delete;
    plugin_handler &
    operator=(const plugin_handler &other)
            = delete;

    plugin_handler(plugin_handler &&other) noexcept
        : _dl_handle(std::exchange(other._dl_handle, nullptr))
        , _create_fn(std::exchange(other._create_fn, nullptr))
        , _destroy_fn(std::exchange(other._destroy_fn, nullptr))
        , _instance(std::exchange(other._instance, nullptr)) {}

    plugin_handler &
    operator=(plugin_handler &&other) noexcept {
        auto tmp = std::move(other);
        std::swap(_dl_handle, tmp._dl_handle);
        std::swap(_create_fn, tmp._create_fn);
        std::swap(_destroy_fn, tmp._destroy_fn);
        std::swap(_instance, tmp._instance);
        return *this;
    }

    ~plugin_handler() { release(); }

    explicit operator bool() const { return _instance; }

    [[nodiscard]] const std::string &
    status() const {
        return _status;
    }

    auto *
    operator->() const {
        return _instance;
    }
};

class plugin_loader {
private:
    std::vector<plugin_handler>                       _handlers;
    std::unordered_map<std::string, gp_plugin_base *> _handler_for_name;
    std::unordered_map<std::string, std::string>      _failed_plugins;

    node_registry                                    *_global_registry;
    std::vector<std::string>                          _known_nodes;

public:
    plugin_loader(node_registry *global_registry, std::span<const std::filesystem::path> plugin_directories) : _global_registry(global_registry) {
        for (const auto &directory : plugin_directories) {
            std::cerr << std::filesystem::current_path() << std::endl;

            if (!std::filesystem::is_directory(directory)) continue;

            for (const auto &file : std::filesystem::directory_iterator{ directory }) {
                if (file.is_regular_file() && file.path().extension() == ".so") {
                    if (plugin_handler handler(file.path().string()); handler) {
                        for (const auto &node_name : handler->provided_nodes()) {
                            _handler_for_name.emplace(std::string(node_name), handler.operator->());
                            _known_nodes.emplace_back(node_name);
                        }

                        _handlers.push_back(std::move(handler));

                    } else {
                        _failed_plugins[file.path()] = handler.status();
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
        return _failed_plugins;
    }

    auto
    known_nodes() const {
        auto        result  = _known_nodes;
        const auto &builtin = _global_registry->known_nodes();
        result.insert(result.end(), builtin.begin(), builtin.end());
        return result;
    }

    std::unique_ptr<fair::graph::node_model>
    instantiate(std::string name, std::string_view type, const property_map &params = {}) {
        // Try to create a node from the global registry
        if (auto result = _global_registry->create_node(name, type, params)) {
            return result;
        }

        auto it = _handler_for_name.find(name);
        if (it == _handler_for_name.end()) return {};

        auto &handler = it->second;

        return handler->create_node(std::move(name), type, params);
    }

    template<typename Graph, typename... InstantiateArgs>
    fair::graph::node_model &
    instantiate_in_graph(Graph &graph, InstantiateArgs &&...args) {
        auto node_load = instantiate(std::forward<InstantiateArgs>(args)...);
        if (!node_load) {
            throw fmt::format("Unable to create node");
        }
        return graph.add_node(std::move(node_load));
    }
};
#endif

} // namespace fair::graph

#endif // include guard
