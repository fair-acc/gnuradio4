#ifndef GNURADIO_PLUGIN_LOADER_HPP
#define GNURADIO_PLUGIN_LOADER_HPP

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_LIBCPP_VERSION)
#include <regex>
#endif

#include "BlockRegistry.hpp"

#include <gnuradio-4.0/PluginMetadata.hpp>
#include <gnuradio-4.0/YamlPmt.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>

#ifdef INTERNAL_ENABLE_BLOCK_PLUGINS
#include <dlfcn.h>

#include "Plugin.hpp"
#endif

#include <gnuradio-4.0/Profiler.hpp>

namespace gr {

using namespace std::string_literals;
using namespace std::string_view_literals;

// Forward declaration needed for instantiateBlockFromYamlDefinition before PluginLoader is fully defined.
class PluginLoader;

namespace detail {

using gr::pmt::yaml::ParseError;

template<typename R>
R optionalMapAt(const auto& map, std::string_view key, auto defaultResult) {
    if (auto it = map.find(std::string(key)); it != map.cend()) {
        return it->second;
    } else {
        return defaultResult;
    }
}

inline std::string joinUri(const std::string& base, const std::string& file) {
    return base.empty()          ? file        //
           : base.ends_with('/') ? base + file //
                                 : base + '/' + file;
}

inline std::expected<std::string, ParseError> readUriToString(std::string_view uri) {
    auto readerExp = gr::algorithm::fileio::readAsync(uri);
    if (!readerExp) {
        return std::unexpected(ParseError{.message = "Failed to read URI"});
    }
    auto bytesExp = readerExp->get();
    if (!bytesExp) {
        return std::unexpected(ParseError{.message = "Failed to read URI"});
    }
    return std::string(bytesExp->begin(), bytesExp->end());
}

inline std::expected<std::chrono::sys_seconds, ParseError> parseTimestamp(const std::string& ts) {
    // clang/libc++ does not implement std::chrono::parse
#if not defined(_LIBCPP_VERSION)
    std::istringstream ss{ts};
    if (std::chrono::sys_seconds tp{}; ss >> std::chrono::parse("%Y-%m-%d-%H:%M:%S", tp)) {
        return tp;
    }
#else
    static const std::regex pattern(R"(^(\d{4})-(\d{2})-(\d{2})-(\d{2}):(\d{2}):(\d{2})$)");

    std::smatch match;
    if (std::regex_match(ts, match, pattern)) {
        int y  = std::stoi(match[1]);
        int m  = std::stoi(match[2]);
        int d  = std::stoi(match[3]);
        int hh = std::stoi(match[4]);
        int mm = std::stoi(match[5]);
        int ss = std::stoi(match[6]);

        std::chrono::year_month_day ymd{std::chrono::year{y}, std::chrono::month{static_cast<unsigned>(m)}, std::chrono::day{static_cast<unsigned>(d)}};

        auto days = std::chrono::sys_days{ymd};
        auto time = std::chrono::hours{hh} + std::chrono::minutes{mm} + std::chrono::seconds{ss};

        return days + time;
    }
#endif
    return std::unexpected(ParseError{.message = std::format("Invalid timestamp {}", ts)});
}

inline std::string uriToCacheFilename(std::string uri) {
    std::ranges::replace_if(uri, [](unsigned char c) { return !std::isalnum(c) && c != '.' && c != '-'; }, '_');
    return uri;
}

struct YamlDefinitionsLoader {
    struct Definition {
        gr::property_map   definition;
        gr_plugin_metadata metadata;
    };

    static std::string assetsCacheDir() {
        if (const char* env = ::getenv("GR_DATA_CACHE_DIR"); env != nullptr) {
            return std::string(env);
        } else {
            return std::string(GR_DATA_CACHE_DIR);
        }
    }

    std::unordered_map<std::string, Definition> _definitionForBlockName;

    explicit YamlDefinitionsLoader(std::span<const std::string> uris) { loadBlockDefinitions(uris); }

    void loadBlockDefinitions(std::span<const std::string> uris) {
#ifndef __EMSCRIPTEN__
        const auto cacheDir = std::filesystem::path(assetsCacheDir()) / "asset_cache";
        std::filesystem::create_directories(cacheDir);
        if (!std::filesystem::is_directory(cacheDir)) {
            std::println("FATAL ERROR: Directory {} does not exist, can not proceed", cacheDir.string());
            std::terminate();
        }
#endif

        auto getMapField = []<typename R>(const auto& map, const auto& key, const R& defaultValue) {
            auto it = map.find(key);
            if (it == map.cend()) {
                return defaultValue;

            } else {
                return it->second.value_or(defaultValue);
            }
        };

        for (const auto& uriBase : uris) {
            // Note: If all this was expected-based, this could have been a chain of and_then calls
            const auto indexContent = readUriToString(joinUri(uriBase, "index.yaml"));
            if (!indexContent) {
                continue;
            }
            const auto indexMap = gr::pmt::yaml::deserialize(*indexContent);
            if (!indexMap) {
                continue;
            }
            const auto assetsList = getMapField(*indexMap, "assets", gr::Tensor<gr::pmt::Value>{});
            for (const gr::pmt::Value& assetEntry : assetsList) {
                const auto* assetMap = assetEntry.get_if<pmt::Value::Map>();
                if (!assetMap) {
                    continue;
                }
                const auto file = getMapField(*assetMap, "file", std::string());
                if (file.empty()) {
                    continue;
                }

                const auto blockUri = joinUri(uriBase, file);

                std::expected<std::string, ParseError> blockContent;
#ifndef __EMSCRIPTEN__
                const auto modified     = getMapField(*assetMap, "modified", "undefined"s);
                const auto modifiedTime = parseTimestamp(modified);
                const auto cachePath    = cacheDir / uriToCacheFilename(blockUri);
                if (const bool cacheHit = modifiedTime && std::filesystem::exists(cachePath) && std::chrono::file_clock::to_sys(std::filesystem::last_write_time(cachePath)) >= *modifiedTime; cacheHit) {
                    blockContent = readUriToString(cachePath.string());
                } else {
                    blockContent = readUriToString(blockUri);
                    if (blockContent) {
                        if (std::ofstream f(cachePath); f) {
                            f << *blockContent;
                        }
                    }
                }
#else
                blockContent = readUriToString(blockUri);
#endif
                if (!blockContent) {
                    continue;
                }

                auto blockMap = gr::pmt::yaml::deserialize(*blockContent);
                if (!blockMap) {
                    continue;
                }

                const auto meta  = getMapField(*blockMap, "definition_metadata", gr::property_map{});
                auto       field = [&](const auto& key) {
                    const auto it = meta.find(std::string(key));
                    return it != meta.end() ? it->second.value_or(std::string{}) : std::string{};
                };
                gr_plugin_metadata metadata{
                    .plugin_name    = field("plugin_name"),    //
                    .plugin_author  = field("plugin_author"),  //
                    .plugin_license = field("plugin_license"), //
                    .plugin_version = field("plugin_version"),
                    .block_type     = field("block_type"), //
                };

                if (metadata.block_type.empty()) {
                    continue;
                }

                auto blockType = metadata.block_type;
                _definitionForBlockName.insert_or_assign(std::move(blockType), Definition{std::move(*blockMap), std::move(metadata)});
            }
        }
    }

    std::optional<Definition> definitionForBlockName(std::string_view name) const { //
        return detail::optionalMapAt<std::optional<Definition>>(_definitionForBlockName, name, std::nullopt);
    }
};

std::shared_ptr<gr::BlockModel> instantiateBlockFromYamlDefinition(gr::PluginLoader& loader, const YamlDefinitionsLoader::Definition& def);

} // namespace detail

#ifdef INTERNAL_ENABLE_BLOCK_PLUGINS
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
        if (_instance && _destroy_fn) {
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
        // RTLD_LOCAL keeps plugin symbols isolated but breaks RTTI/dynamic_cast across dylib boundaries.
        // On macOS (Mach-O two-level namespace), RTLD_LOCAL also risks duplicating singletons such as
        // globalBlockRegistry(); use RTLD_GLOBAL there to match Linux ELF flat-namespace behaviour.
#ifdef __APPLE__
        _dl_handle = dlopen(plugin_file.c_str(), RTLD_LAZY | RTLD_GLOBAL);
#else
        _dl_handle = dlopen(plugin_file.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
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

        if (_instance->abiVersion() != GR_PLUGIN_CURRENT_ABI_VERSION) {
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
    detail::YamlDefinitionsLoader                _yamlRegistry;
    std::vector<PluginHandler>                   _pluginHandlers;
    std::unordered_map<std::string, std::string> _failedPlugins;
    std::unordered_set<std::string>              _loadedPluginFiles;

    std::unordered_map<std::string, gr_plugin_base*> _pluginForBlockName;
    std::unordered_map<std::string, gr_plugin_base*> _pluginForSchedulerName;

    BlockRegistry*     _registry;
    SchedulerRegistry* _schedulerRegistry;

    gr_plugin_base* pluginForBlockName(std::string_view name) const { //
        return detail::optionalMapAt<gr_plugin_base*>(_pluginForBlockName, name, nullptr);
    }

    gr_plugin_base* pluginForSchedulerName(std::string_view name) const { //
        return detail::optionalMapAt<gr_plugin_base*>(_pluginForSchedulerName, name, nullptr);
    }

public:
    PluginLoader(BlockRegistry& registry, SchedulerRegistry& scheduler_registry, std::span<const std::string> paths) : _yamlRegistry(paths), _registry(&registry), _schedulerRegistry(&scheduler_registry) {
        for (const auto& pathStr : paths) {
            const std::filesystem::path directory(pathStr);
            if (!std::filesystem::is_directory(directory)) {
                continue;
            }

            for (const auto& file : std::filesystem::directory_iterator{directory}) {
#if defined(_WIN32)
                if (file.is_regular_file() && file.path().extension() == ".dll") {
#elif defined(__APPLE__)
                if (file.is_regular_file() && (file.path().extension() == ".so" || file.path().extension() == ".dylib")) {
#else
                if (file.is_regular_file() && file.path().extension() == ".so") {
#endif
                    auto fileString = file.path().string();
                    if (_loadedPluginFiles.contains(fileString)) {
                        continue;
                    }
                    _loadedPluginFiles.insert(fileString);

                    if (PluginHandler handler(file.path().string()); handler) {
                        for (std::string_view blockName : handler->availableBlocks()) {
                            _pluginForBlockName.emplace(std::string(blockName), handler.operator->());
                        }

                        for (std::string_view schedulerName : handler->availableSchedulers()) {
                            _pluginForSchedulerName.emplace(std::string(schedulerName), handler.operator->());
                        }

                        _pluginHandlers.push_back(std::move(handler));

                    } else {
                        _failedPlugins[file.path().string()] = handler.status();
                    }
                }
            }
        }
    }

    BlockRegistry&     registry() { return *_registry; }
    SchedulerRegistry& schedulerRegistry() { return *_schedulerRegistry; }

    const auto& plugins() const { return _pluginHandlers; }

    const auto& failedPlugins() const { return _failedPlugins; }

    std::vector<std::string> availableBlocks() const {
        auto                     keysView = _pluginForBlockName | std::views::keys;
        std::vector<std::string> result(keysView.begin(), keysView.end());

        const auto& builtin = _registry->keys();
        result.insert(result.end(), builtin.begin(), builtin.end());

        // remove duplicates
        std::ranges::sort(result);
        auto newEnd = std::ranges::unique(result).begin();
        result.erase(newEnd, result.end());
        return result;
    }

    std::shared_ptr<gr::BlockModel> instantiate(std::string_view name, const property_map& params = property_map{}) {
        // Try to create a node from the global registry
        if (auto result = _registry->create(name, params)) {
            return result;
        }

        if (auto* plugin = pluginForBlockName(name); plugin != nullptr) {
            return plugin->createBlock(name, params);
        }

        if (const auto def = _yamlRegistry.definitionForBlockName(name)) {
            return detail::instantiateBlockFromYamlDefinition(*this, *def);
        }

#ifndef NDEBUG
        std::print("Available blocks in the registry\n");
        for (const auto& block : _registry->keys()) {
            std::print("    {}\n", block);
        }
        std::print("]\n");

        std::print("Available blocks from plugins [\n");
        for (const auto& [blockName, _] : _pluginForBlockName) {
            std::print("    {}\n", blockName);
        }
        std::print("]\n");

        std::print("Available YAML definitions[\n");
        for (const auto& [blockName, _] : _yamlRegistry._definitionForBlockName) {
            std::print("    {}\n", blockName);
        }
        std::print("]\n");
#endif
        std::print("Error: Plugin not found for '{}', returning nullptr.\n", name);
        return {};
    }

    std::shared_ptr<gr::SchedulerModel> instantiateScheduler(std::string_view name, const property_map& params = property_map{}) {
        if (auto result = _schedulerRegistry->create(name, params)) {
            return std::shared_ptr<gr::SchedulerModel>(result.release());
        }

        auto* plugin = pluginForSchedulerName(name);

        if (plugin == nullptr) {
#ifndef NDEBUG
            std::println("Could not find scheduler {}. Available schedulers in the registry", name);
            for (const auto& scheduler : _schedulerRegistry->keys()) {
                std::print("    {}\n", scheduler);
            }
            std::print("]\n");

            std::print("Available schedulers from plugins [\n", name);
            for (const auto& [schedulerName, _] : _pluginForSchedulerName) {
                std::print("    {}\n", schedulerName);
            }
            std::print("]\n");
#endif
            std::print("Error: Scheduler plugin not found for '{}', returning nullptr.\n", name);
            return {};
        }

        auto result = plugin->createScheduler(name, params);
        return std::shared_ptr<gr::SchedulerModel>(result.release());
    }

    std::vector<std::string> availableSchedulers() const {
        auto                     keysView = _pluginForSchedulerName | std::views::keys;
        std::vector<std::string> result(keysView.begin(), keysView.end());

        const auto& builtin = _schedulerRegistry->keys();
        result.insert(result.end(), builtin.begin(), builtin.end());

        // remove duplicates
        std::ranges::sort(result);
        auto newEnd = std::ranges::unique(result).begin();
        result.erase(newEnd, result.end());
        return result;
    }

    bool isBlockAvailable(std::string_view block) const { return _registry->contains(block) || pluginForBlockName(block) != nullptr; }

    bool isSchedulerAvailable(std::string_view scheduler) const { return _schedulerRegistry->contains(scheduler) || pluginForSchedulerName(scheduler) != nullptr; }

    const auto& definitionForBlockName() const { return _yamlRegistry._definitionForBlockName; }
};
#else
// PluginLoader on WASM is just a wrapper on BlockRegistry to provide the
// same API as proper PluginLoader
class PluginLoader {
private:
    detail::YamlDefinitionsLoader _yamlRegistry;
    BlockRegistry*                _registry;
    SchedulerRegistry*            _schedulerRegistry;

public:
    PluginLoader(BlockRegistry& registry, SchedulerRegistry& scheduler_registry, std::span<const std::string> paths) : _yamlRegistry(paths), _registry(&registry), _schedulerRegistry(&scheduler_registry) {}

    BlockRegistry&     registry() { return *_registry; }
    SchedulerRegistry& schedulerRegistry() { return *_schedulerRegistry; }

    auto availableBlocks() const { return _registry->keys(); }
    auto availableSchedulers() const { return _schedulerRegistry->keys(); }

    std::shared_ptr<gr::BlockModel> instantiate(std::string_view name, const property_map& params = {}) {
        if (auto result = _registry->create(name, params)) {
            return result;
        }

        if (const auto def = _yamlRegistry.definitionForBlockName(name)) {
            return detail::instantiateBlockFromYamlDefinition(*this, *def);
        }

        return nullptr;
    }

    std::shared_ptr<gr::SchedulerModel> instantiateScheduler(std::string_view name, const property_map& params = {}) {
        auto result = _schedulerRegistry->create(name, params);
        return result ? std::shared_ptr<gr::SchedulerModel>((result.release())) : nullptr;
    }

    bool isBlockAvailable(std::string_view block) const { return _registry->contains(block); }
    bool isSchedulerAvailable(std::string_view scheduler) const { return _schedulerRegistry->contains(scheduler); }

    const auto& definitionForBlockName() const { return _yamlRegistry._definitionForBlockName; }
};
#endif

PluginLoader& globalPluginLoader();

} // namespace gr

#endif // GNURADIO_PLUGIN_LOADER_HPP
