#ifndef GNURADIO_PLUGIN_LOADER_HPP
#define GNURADIO_PLUGIN_LOADER_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <print>

#if defined(_LIBCPP_VERSION)
#include <regex>
#endif

#include "BlockRegistry.hpp"

#include <gnuradio-4.0/PluginMetadata.hpp>
#include <gnuradio-4.0/YamlPmt.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>

#ifdef INTERNAL_ENABLE_BLOCK_PLUGINS
#include "Plugin.hpp"
#include <gnuradio-4.0/SharedLibrary.hpp>
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
    gr::algorithm::fileio::ReaderConfig config;
    auto                                readerExp = gr::algorithm::fileio::readAsync(uri, config);
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

inline std::string uriToCacheFilename(std::string_view uri) {
    // FNV-1a 64-bit: deterministic across runs, collision-resistant, NAME_MAX-safe.
    constexpr std::uint64_t fnvOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t fnvPrime  = 0x100000001b3ULL;
    std::uint64_t           hash      = fnvOffset;
    for (char c : uri) {
        hash ^= static_cast<std::uint8_t>(c);
        hash *= fnvPrime;
    }
    return std::format("{:016x}", hash);
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

    YamlDefinitionsLoader(const YamlDefinitionsLoader&)            = delete;
    YamlDefinitionsLoader& operator=(const YamlDefinitionsLoader&) = delete;

    void loadBlockDefinitions(std::span<const std::string> uris) {
        const auto      cacheDir = std::filesystem::path(assetsCacheDir()) / "asset_cache";
        std::error_code createEc;
        std::filesystem::create_directories(cacheDir, createEc);
        const bool cacheAvailable = !createEc && std::filesystem::is_directory(cacheDir);
        if (!cacheAvailable) {
            std::println("warning: plugin cache directory {} is not available; caching disabled", cacheDir.string());
        }

        auto getMapField = []<typename R>(const auto& map, const auto& key, const R& defaultValue) -> R {
            auto it = map.find(key);
            if (it == map.cend()) {
                return defaultValue;
            }
            // Tensor<Value> decode requires Value::_resource for sub-Value allocations.
            const gr::Value entry = (*it).second;
            if constexpr (std::same_as<R, gr::property_map>) {
                if (auto opt = entry.get_if<gr::property_map>()) {
                    return opt->owned(map.resource());
                }
                return defaultValue;
            } else {
                return entry.value_or(defaultValue);
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
            const auto assetsList = getMapField(*indexMap, "assets", gr::Tensor<gr::Value>{});
            for (const gr::Value& assetEntry : assetsList) {
                const auto assetMap = assetEntry.get_if<gr::property_map>();
                if (!assetMap) {
                    continue;
                }
                const auto file = getMapField(*assetMap, "file", std::string());
                if (file.empty()) {
                    continue;
                }

                const auto blockUri = joinUri(uriBase, file);

                std::expected<std::string, ParseError> blockContent;
                if (cacheAvailable) {
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
                } else {
                    blockContent = readUriToString(blockUri);
                }
                if (!blockContent) {
                    continue;
                }

                auto blockMap = gr::pmt::yaml::deserialize(*blockContent);
                if (!blockMap) {
                    continue;
                }

                const auto meta  = getMapField(*blockMap, "definition_metadata", gr::property_map{});
                auto       field = [&](const auto& key) -> std::string {
                    const auto it = meta.find(std::string_view{key});
                    return it != meta.end() ? std::string{(*it).second.value_or(std::string_view{})} : std::string{};
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

[[nodiscard]] std::expected<void, gr::Error> checkEmbeddedVersionConsistency(const std::unordered_map<std::string, YamlDefinitionsLoader::Definition>& knownDefs, const YamlDefinitionsLoader::Definition& def);

std::expected<std::shared_ptr<gr::BlockModel>, gr::Error> instantiateBlockFromYamlDefinition(gr::PluginLoader& loader, const YamlDefinitionsLoader::Definition& def) noexcept;

} // namespace detail

#ifdef INTERNAL_ENABLE_BLOCK_PLUGINS

using plugin_create_function_t  = void (*)(gr_plugin_base**);
using plugin_destroy_function_t = void (*)(gr_plugin_base*);

class PluginHandler {
private:
    SharedLibrary             _library;
    plugin_create_function_t  _create_fn  = nullptr;
    plugin_destroy_function_t _destroy_fn = nullptr;
    gr_plugin_base*           _instance   = nullptr;

    std::string _status;

    void releaseInstance() {
        if (_instance && _destroy_fn) {
            _destroy_fn(_instance);
            _instance = nullptr;
        }
        _create_fn  = nullptr;
        _destroy_fn = nullptr;
    }

    [[nodiscard]] bool bindFactories() {
        // Prefer a single aggregate export in a future ABI revision; for now resolve two C symbols.
        auto create = _library.resolve<void(gr_plugin_base**)>("gr_plugin_make");
        if (!create) {
            _status = "Failed to load symbol gr_plugin_make";
            releaseInstance();
            std::ignore = _library.unload();
            return false;
        }
        _create_fn = *create;

        auto destroy = _library.resolve<void(gr_plugin_base*)>("gr_plugin_free");
        if (!destroy) {
            _status = "Failed to load symbol gr_plugin_free";
            releaseInstance();
            std::ignore = _library.unload();
            return false;
        }
        _destroy_fn = *destroy;

        _create_fn(&_instance);
        if (!_instance) {
            _status = "Failed to create an instance of the plugin";
            releaseInstance();
            std::ignore = _library.unload();
            return false;
        }

        if (_instance->abiVersion() != GR_PLUGIN_CURRENT_ABI_VERSION) {
            _status = "Wrong ABI version";
            releaseInstance();
            std::ignore = _library.unload();
            return false;
        }
        return true;
    }

public:
    PluginHandler() = default;

    /// Synchronous open (native). On Emscripten this fails — use loadAsync().
    explicit PluginHandler(const std::string& plugin_file) {
        auto open = _library.load(plugin_file);
        if (!open) {
            _status = open.error().message.empty() ? "Failed to load the plugin file" : open.error().message;
            return;
        }
        if (!bindFactories()) {
            return;
        }
    }

    PluginHandler(const PluginHandler&)            = delete;
    PluginHandler& operator=(const PluginHandler&) = delete;

    PluginHandler(PluginHandler&& other) noexcept : _library(std::move(other._library)), _create_fn(std::exchange(other._create_fn, nullptr)), _destroy_fn(std::exchange(other._destroy_fn, nullptr)), _instance(std::exchange(other._instance, nullptr)), _status(std::move(other._status)) {}

    PluginHandler& operator=(PluginHandler&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        releaseInstance();
        std::ignore   = _library.unload();
        _library      = std::move(other._library);
        _create_fn    = std::exchange(other._create_fn, nullptr);
        _destroy_fn   = std::exchange(other._destroy_fn, nullptr);
        _instance     = std::exchange(other._instance, nullptr);
        _status       = std::move(other._status);
        return *this;
    }

    ~PluginHandler() {
        releaseInstance();
        std::ignore = _library.unload();
    }

    /// Completes when the library is open and factories are bound.
    /// On native platforms done runs before loadAsync returns.
    void loadAsync(const std::string& plugin_file, std::function<void(std::expected<void, Error>)> done) {
        if (!done) {
            return;
        }
        releaseInstance();
        if (_library.isLoaded()) {
            std::ignore = _library.unload();
        }

        _library.loadAsync(plugin_file, [this, done = std::move(done)](std::expected<void, Error> open) mutable {
            if (!open) {
                _status = open.error().message.empty() ? "Failed to load the plugin file" : open.error().message;
                done(std::unexpected(Error{_status}));
                return;
            }
            if (!bindFactories()) {
                done(std::unexpected(Error{_status}));
                return;
            }
            done({});
        });
    }

    explicit operator bool() const { return _instance != nullptr; }

    [[nodiscard]] const std::string& status() const { return _status; }

    auto* operator->() const { return _instance; }
};

class PluginLoader {
private:
    detail::YamlDefinitionsLoader                _yamlRegistry;
    std::vector<PluginHandler>                   _pluginHandlers;
    std::unordered_map<std::string, std::string> _failedPlugins;
    std::unordered_set<std::string>              _loadedPluginFiles;
    std::vector<std::string>                     _pluginSearchPaths;

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

    void registerHandler(PluginHandler&& handler, const std::string& fileString) {
        if (!handler) {
            _failedPlugins[fileString] = handler.status();
            return;
        }
        for (std::string_view blockName : handler->availableBlocks()) {
            _pluginForBlockName.emplace(std::string(blockName), handler.operator->());
        }
        for (std::string_view schedulerName : handler->availableSchedulers()) {
            _pluginForSchedulerName.emplace(std::string(schedulerName), handler.operator->());
        }
        _pluginHandlers.push_back(std::move(handler));
    }

    void loadPluginFileSync(const std::filesystem::path& file) {
        const auto fileString = file.string();
        if (_loadedPluginFiles.contains(fileString)) {
            return;
        }
        _loadedPluginFiles.insert(fileString);
        registerHandler(PluginHandler(fileString), fileString);
    }

public:
    PluginLoader(BlockRegistry& registry, SchedulerRegistry& scheduler_registry, std::span<const std::string> paths) : _yamlRegistry(paths), _pluginSearchPaths(paths.begin(), paths.end()), _registry(&registry), _schedulerRegistry(&scheduler_registry) {
#if !defined(__EMSCRIPTEN__)
        // Native: directory scan + synchronous open (loadAsync completes inline).
        for (const auto& pathStr : _pluginSearchPaths) {
            const std::filesystem::path directory(pathStr);
            if (!std::filesystem::is_directory(directory)) {
                continue;
            }
            for (const auto& file : std::filesystem::directory_iterator{directory}) {
                if (file.is_regular_file() && detail::isPluginFileExtension(file.path())) {
                    loadPluginFileSync(file.path());
                }
            }
        }
#else
        // Emscripten: open is asynchronous; use loadPluginAsync / loadPluginsAsync before
        // expecting plugin block types. YAML definitions from paths still load in the constructor.
        (void)0;
#endif
    }

    BlockRegistry&     registry() { return *_registry; }
    SchedulerRegistry& schedulerRegistry() { return *_schedulerRegistry; }

    const auto& plugins() const { return _pluginHandlers; }

    const auto& failedPlugins() const { return _failedPlugins; }

    /// Load a single plugin file or URL (Emscripten side module path/URL or native .so/.dll).
    void loadPluginAsync(std::string_view pathOrUri, std::function<void(std::expected<void, Error>)> done) {
        const std::string fileString(pathOrUri);
        if (_loadedPluginFiles.contains(fileString)) {
            if (done) {
                done({});
            }
            return;
        }
        _loadedPluginFiles.insert(fileString);

        // Heap-allocate so the PluginHandler outlives an asynchronous emscripten_dlopen.
        struct Pending {
            PluginHandler handler;
        };
        auto pending = std::make_shared<Pending>();
        pending->handler.loadAsync(fileString, [this, fileString, done = std::move(done), pending](std::expected<void, Error> result) mutable {
            if (!result || !pending->handler) {
                const std::string status = pending->handler ? pending->handler.status() : (result ? "unknown plugin error" : result.error().message);
                _failedPlugins[fileString] = status;
                if (done) {
                    done(std::unexpected(Error{status}));
                }
                return;
            }
            registerHandler(std::move(pending->handler), fileString);
            if (done) {
                done({});
            }
        });
    }

    /// Scan recorded plugin directories and load every matching plugin file asynchronously.
    /// On native platforms each open completes before the next starts and done runs at the end
    /// of this call. On Emscripten done runs after the last side module has been linked.
    void loadPluginsAsync(std::function<void(std::expected<void, Error>)> done) {
        std::vector<std::filesystem::path> candidates;
        for (const auto& pathStr : _pluginSearchPaths) {
            const std::filesystem::path directory(pathStr);
            std::error_code             ec;
            if (!std::filesystem::is_directory(directory, ec)) {
                continue;
            }
            for (const auto& file : std::filesystem::directory_iterator{directory, ec}) {
                if (ec) {
                    break;
                }
                if (file.is_regular_file() && detail::isPluginFileExtension(file.path())) {
                    candidates.push_back(file.path());
                }
            }
        }

        if (candidates.empty()) {
            if (done) {
                done({});
            }
            return;
        }

        struct State {
            std::atomic<std::size_t>                        remaining{0};
            std::function<void(std::expected<void, Error>)> done;
            Error                                           firstError;
            std::atomic<bool>                               hasError{false};
        };
        auto state       = std::make_shared<State>();
        state->remaining = candidates.size();
        state->done      = std::move(done);

        for (const auto& path : candidates) {
            loadPluginAsync(path.string(), [state](std::expected<void, Error> r) {
                if (!r) {
                    bool expected = false;
                    if (state->hasError.compare_exchange_strong(expected, true)) {
                        state->firstError = r.error();
                    }
                }
                if (state->remaining.fetch_sub(1) == 1 && state->done) {
                    if (state->hasError.load()) {
                        state->done(std::unexpected(state->firstError));
                    } else {
                        state->done({});
                    }
                }
            });
        }
    }

    std::vector<std::string> availableBlocks() const {
        auto properBlocks     = _pluginForBlockName | std::views::keys;
        auto blockDefinitions = _yamlRegistry._definitionForBlockName | std::views::keys;

        std::vector<std::string> result;
        result.reserve(std::ranges::size(properBlocks) + std::ranges::size(blockDefinitions));
        result.insert(result.end(), properBlocks.begin(), properBlocks.end());
        result.insert(result.end(), blockDefinitions.begin(), blockDefinitions.end());

#ifndef NDEBUG
        std::println("availableBlocks in {} are {}", static_cast<const void*>(this), result);
#endif

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
            auto result = detail::instantiateBlockFromYamlDefinition(*this, *def);
            if (!result) {
                std::print("Error: YAML block instantiation failed for '{}': {} ({})\n", name, result.error().message, result.error().srcLoc());
                return {};
            }
            return *result;
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

            std::print("Available schedulers from plugins [\n");
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
// Plugin system disabled (GR_ENABLE_BLOCK_REGISTRY / INTERNAL_ENABLE_BLOCK_PLUGINS off)
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
            auto result = detail::instantiateBlockFromYamlDefinition(*this, *def);
            if (!result) {
                std::print("Error: YAML block instantiation failed for '{}': {} ({})\n", name, result.error().message, result.error().srcLoc());
                return nullptr;
            }
            return *result;
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
