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

#ifdef ENABLE_BLOCK_PLUGINS
#include <dlfcn.h>

#include "plugin.hpp"
#endif

namespace gr {

using namespace std::string_literals;
using namespace std::string_view_literals;

#ifdef ENABLE_BLOCK_PLUGINS
// Plugins are not supported on WASM

using plugin_create_function_t  = void (*)(gr_plugin_base**);
using plugin_destroy_function_t = void (*)(gr_plugin_base*);

class PluginHandler {
private:
    void*                     _dl_handle  = nullptr;
    plugin_create_function_t  _create_fn  = nullptr;
    plugin_destroy_function_t _destroy_fn = nullptr;
    gr_plugin_base*           _instance   = nullptr;

    std::string _status;

    void release() {
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

    explicit PluginHandler(const std::string& plugin_file) {
        // TODO: Document why RTLD_LOCAL and not RTLD_GLOBAL is used here. (RTLD_LOCAL breaks RTTI/dynamic_cast across
        // plugin boundaries. Note that RTTI can be very helpful in the debugger.)
        _dl_handle = dlopen(plugin_file.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (!_dl_handle) {
            _status = "Failed to load the plugin file";
            return;
        }

        // FIXME: Casting a void* to function-pointer is UB in C++. Yes "… 'dlsym' is not C++ and therefore we can do
        // whateever …". But we don't need to. Simply have a single 'extern "C"' symbol in the plugin which is an object
        // storing two function pointers. Then we need a single cast from the 'dlsym' result to an aggregate type and
        // can then extract the two function pointers from it. That's simpler and more likely to be conforming C++.
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

    PluginHandler(const PluginHandler& other)            = delete;
    PluginHandler& operator=(const PluginHandler& other) = delete;

    PluginHandler(PluginHandler&& other) noexcept : _dl_handle(std::exchange(other._dl_handle, nullptr)), _create_fn(std::exchange(other._create_fn, nullptr)), _destroy_fn(std::exchange(other._destroy_fn, nullptr)), _instance(std::exchange(other._instance, nullptr)) {}

    PluginHandler& operator=(PluginHandler&& other) noexcept {
        auto tmp = std::move(other);
        std::swap(_dl_handle, tmp._dl_handle);
        std::swap(_create_fn, tmp._create_fn);
        std::swap(_destroy_fn, tmp._destroy_fn);
        std::swap(_instance, tmp._instance);
        return *this;
    }

    ~PluginHandler() { release(); }

    explicit operator bool() const { return _instance; }

    [[nodiscard]] const std::string& status() const { return _status; }

    auto* operator->() const { return _instance; }
};

class PluginLoader {
private:
    std::vector<PluginHandler>                       _handlers;
    std::unordered_map<std::string, gr_plugin_base*> _handlerForName;
    std::unordered_map<std::string, std::string>     _failedPlugins;
    std::unordered_set<std::string>                  _loadedPluginFiles;

    BlockRegistry*           _registry;
    std::vector<std::string> _knownBlocks;

    gr_plugin_base* handlerForName(std::string_view name) const {
        if (auto it = _handlerForName.find(std::string(name)); it != _handlerForName.end()) {
            return it->second;
        } else {
            return nullptr;
        }
    }

public:
    PluginLoader(BlockRegistry& registry, std::span<const std::filesystem::path> plugin_directories) : _registry(&registry) {
        for (const auto& directory : plugin_directories) {
            std::cerr << std::filesystem::current_path() << std::endl;

            if (!std::filesystem::is_directory(directory)) {
                continue;
            }

            for (const auto& file : std::filesystem::directory_iterator{directory}) {
                if (file.is_regular_file() && file.path().extension() == ".so") {
                    auto fileString = file.path().string();
                    if (_loadedPluginFiles.contains(fileString)) {
                        continue;
                    }
                    _loadedPluginFiles.insert(fileString);

                    if (PluginHandler handler(file.path().string()); handler) {
                        for (std::string_view block_name : handler->providedBlocks()) {
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

    BlockRegistry& registry() { return *_registry; }

    const auto& plugins() const { return _handlers; }

    const auto& failed_plugins() const { return _failedPlugins; }

    auto knownBlocks() const {
        auto        result  = _knownBlocks;
        const auto& builtin = _registry->knownBlocks();
        result.insert(result.end(), builtin.begin(), builtin.end());
        return result;
    }

    std::unique_ptr<gr::BlockModel> instantiate(std::string_view name, const property_map& params = {}) {
        // Try to create a node from the global registry
        if (auto result = _registry->createBlock(name, params)) {
            return result;
        }

        auto* handler = handlerForName(name);
        if (handler == nullptr) {
            return {};
        }

        return handler->createBlock(name, params);
    }

    bool isBlockKnown(std::string_view block) const { return _registry->isBlockKnown(block) || handlerForName(block) != nullptr; }
};
#else
// PluginLoader on WASM is just a wrapper on BlockRegistry to provide the
// same API as proper PluginLoader
class PluginLoader {
private:
    BlockRegistry* _registry;

public:
    PluginLoader(BlockRegistry& registry, std::span<const std::filesystem::path> /*plugin_directories*/) : _registry(&registry) {}

    BlockRegistry& registry() { return *_registry; }

    auto knownBlocks() const { return _registry->knownBlocks(); }

    std::unique_ptr<gr::BlockModel> instantiate(std::string_view name, const property_map& params = {}) { return _registry->createBlock(name, params); }

    bool isBlockKnown(std::string_view block) const { return _registry->isBlockKnown(block); }
};
#endif

inline auto& globalPluginLoader() {
    auto pluginPaths = [] {
        std::vector<std::filesystem::path> result;

        auto* envpath = ::getenv("GNURADIO4_PLUGIN_DIRECTORIES");
        if (envpath == nullptr) {
            // TODO choose proper paths when we get the system GR installation done
            result.emplace_back("core/test/plugins");

        } else {
            std::string_view paths(envpath);

            auto i = paths.cbegin();

            // TODO If we want to support Windows, this should be ; there
            auto isSeparator = [](char c) { return c == ':'; };

            while (i != paths.cend()) {
                i      = std::find_if_not(i, paths.cend(), isSeparator);
                auto j = std::find_if(i, paths.cend(), isSeparator);

                if (i != paths.cend()) {
                    result.emplace_back(std::string_view(i, j));
                }
                i = j;
            }
        }

        return result;
    };

    static PluginLoader instance(gr::globalBlockRegistry(), {pluginPaths()});
    return instance;
}

} // namespace gr

#endif // include guard
