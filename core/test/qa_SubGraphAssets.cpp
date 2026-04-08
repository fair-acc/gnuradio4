#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <boost/ut.hpp>

#include <build_configure.hpp>

#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

namespace ut = boost::ut;

namespace {

const std::string kAssetsDir  = std::string(TESTS_SOURCE_PATH) + "/assets";
const std::string kServerBase = "http://127.0.0.1:" + std::to_string(HTTP_SERVER_PORT);
const std::string kCacheDir   = gr::detail::YamlDefinitionsLoader::assetsCacheDir() + "/asset_cache";
#ifdef __EMSCRIPTEN__
// Local filesystem and XHR are not available in the Node.js WASM test environment.
const bool kSkipRemote = true;
#else
const bool kSkipRemote = std::getenv("GR_TEST_DISABLE_REMOTE") != nullptr;
#endif

gr::PluginLoader makeLoader(const std::vector<std::string>& paths) {
    static gr::BlockRegistry     registry;
    static gr::SchedulerRegistry schedulerRegistry;
    return gr::PluginLoader(registry, schedulerRegistry, paths);
}

[[maybe_unused]]
gr::PluginLoader makeLoaderWithPlugins(const std::vector<std::string>& assetPaths) {
    static gr::BlockRegistry     registry;
    static gr::SchedulerRegistry schedulerRegistry;

    std::vector<std::string> allPaths;
    const char*              pluginDir = std::getenv("GNURADIO4_PLUGIN_DIRECTORIES");
    allPaths.emplace_back(pluginDir != nullptr ? pluginDir : "plugins");
    allPaths.insert(allPaths.end(), assetPaths.begin(), assetPaths.end());
    return gr::PluginLoader(registry, schedulerRegistry, allPaths);
}

// Derive the cache file name the same way PluginLoader does.
std::string uriToFilename(std::string_view uri) {
    std::string name(uri);
    std::ranges::replace_if(name, [](unsigned char c) { return !std::isalnum(c) && c != '.' && c != '-'; }, '_');
    return name;
}

std::filesystem::path cachePathFor(std::string_view uri) { return std::filesystem::path(kCacheDir) / uriToFilename(uri); }

void clearCache() { std::filesystem::remove_all(kCacheDir); }

} // namespace

bool hasOneSubgraphBlock(const gr::property_map& definition) {
    try {
        const auto blocks = definition.at("blocks").value_or(gr::Tensor<gr::pmt::Value>{});
        if (blocks.size() != 1uz) {
            return false;
        }
        const auto block = blocks[0].value_or(gr::property_map{});
        return block.at("id") == "SUBGRAPH";
    } catch (...) {
        return false;
    }
}

// exportedInputPorts()/exportedOutputPorts() return a nested map:
//   { blockUniqueName -> { internalPortName -> { "exportedName" -> name } } }
// This helper collects all exported port names from that structure.
std::vector<std::string> collectExportedNames(const gr::property_map& portsMap) {
    std::vector<std::string> names;
    for (const auto& [_blockName, portInfoVal] : portsMap) {
        const auto* portMap = portInfoVal.get_if<gr::property_map>();
        if (!portMap) {
            continue;
        }
        for (const auto& [_internalName, exportInfoVal] : *portMap) {
            const auto* exportMap = exportInfoVal.get_if<gr::property_map>();
            if (!exportMap) {
                continue;
            }
            auto it = exportMap->find("exportedName");
            if (it != exportMap->end()) {
                names.emplace_back(it->second.value_or(std::string_view{}));
            }
        }
    }
    return names;
}

const boost::ut::suite AssetsLoadingTests = [] {
    using namespace ut;
    using namespace ut::literals;
    using namespace std::string_literals;

    // ── local tests ──────────────────────────────────────────────────────────

#ifndef __EMSCRIPTEN__
    // Local files are not supported in WASM
    "happy path: two blocks loaded from root_a"_test = [] {
        auto loader = makeLoader({kAssetsDir + "/root_a"});

        const auto AlphaBlock = "MyAlphaBlock";
        const auto BetaBlock  = "MyBetaBlock";

        const auto& defs = loader.definitionForBlockName();
        expect(eq(defs.size(), 2_ul));
        expect(defs.contains(AlphaBlock));
        expect(defs.contains(BetaBlock));
        expect(eq(defs.at(AlphaBlock).metadata.block_type, "MyAlphaBlock"s));
        expect(eq(defs.at(AlphaBlock).metadata.plugin_name, "AlphaPlugin"s));
        expect(eq(defs.at(AlphaBlock).metadata.plugin_author, "Test Author"s));
        expect(eq(defs.at(AlphaBlock).metadata.plugin_license, "LGPL-3.0"s));
        expect(eq(defs.at(AlphaBlock).metadata.plugin_version, "2024-01-15"s));
        expect(eq(defs.at(BetaBlock).metadata.block_type, "MyBetaBlock"s));
        expect(defs.at(BetaBlock).metadata.plugin_name.empty());

        expect(hasOneSubgraphBlock(defs.at(AlphaBlock).definition));
        expect(hasOneSubgraphBlock(defs.at(BetaBlock).definition));
    };

    "missing index.yaml: map stays empty, no crash"_test = [] {
        auto loader = makeLoader({kAssetsDir + "/nonexistent_root"});
        expect(loader.definitionForBlockName().empty());
    };

    "malformed index.yaml: silently skipped"_test = [] {
        auto loader = makeLoader({kAssetsDir + "/root_malformed"});
        expect(loader.definitionForBlockName().empty());
    };

    "index.yaml without assets key: silently skipped"_test = [] {
        auto loader = makeLoader({kAssetsDir + "/root_no_files_key"});
        expect(loader.definitionForBlockName().empty());
    };

    "multiple URI roots: each contributes independent entries"_test = [] {
        auto loader = makeLoader({kAssetsDir + "/root_a", kAssetsDir + "/root_b"});

        const auto AlphaBlock = "MyAlphaBlock";
        const auto BetaBlock  = "MyBetaBlock";
        const auto GammaBlock = "MyGammaBlock";

        const auto& defs = loader.definitionForBlockName();
        expect(eq(defs.size(), 3_ul));
        expect(defs.contains(AlphaBlock));
        expect(defs.contains(BetaBlock));
        expect(defs.contains(GammaBlock));

        expect(hasOneSubgraphBlock(defs.at(GammaBlock).definition));
    };

    // instantiate a YAML-defined composite block from an asset definition.
    // The definition embeds a SUBGRAPH with two chained multiply blocks whose
    // exported ports are named 'in' and 'out'.
    "instantiate: YAML asset creates a composite block with exported ports"_test = [] {
        auto loader = makeLoaderWithPlugins({kAssetsDir + "/root_a"});

        auto block = loader.instantiate("MyAlphaBlock");
        expect(block != nullptr) << "instantiate must return a non-null block";
        if (!block) {
            return;
        }

        const auto inputNames  = collectExportedNames(block->exportedInputPorts());
        const auto outputNames = collectExportedNames(block->exportedOutputPorts());
        expect(eq(inputNames.size(), 1uz)) << "expected one exported input port";
        expect(eq(outputNames.size(), 1uz)) << "expected one exported output port";
        expect(std::ranges::find(inputNames, "in") != inputNames.end()) << "exported input port must be named 'in'";
        expect(std::ranges::find(outputNames, "out") != outputNames.end()) << "exported output port must be named 'out'";
    };
#endif

    // ── remote tests (server started by CMake fixture) ────────────────────────

    "remote happy path: two blocks loaded via http from root_a"_test = [] {
        if (kSkipRemote) {
            return;
        }
        clearCache();
        auto loader = makeLoader({kServerBase + "/root_a"});

        const auto AlphaBlock = "MyAlphaBlock";
        const auto BetaBlock  = "MyBetaBlock";

        const auto& defs = loader.definitionForBlockName();
        expect(eq(defs.size(), 2_ul));
        expect(defs.contains(AlphaBlock));
        expect(defs.contains(BetaBlock));

        expect(hasOneSubgraphBlock(defs.at(AlphaBlock).definition));
        expect(hasOneSubgraphBlock(defs.at(BetaBlock).definition));
    };

    "remote missing index.yaml: map stays empty, no crash"_test = [] {
        if (kSkipRemote) {
            return;
        }
        clearCache();
        auto loader = makeLoader({kServerBase + "/nonexistent_root"});
        expect(loader.definitionForBlockName().empty());
    };

    "remote multiple URI roots: each contributes independent entries"_test = [] {
        if (kSkipRemote) {
            return;
        }
        clearCache();
        auto loader = makeLoader({kServerBase + "/root_a", kServerBase + "/root_b"});

        const auto AlphaBlock = "MyAlphaBlock";
        const auto BetaBlock  = "MyBetaBlock";
        const auto GammaBlock = "MyGammaBlock";

        const auto& defs = loader.definitionForBlockName();
        expect(eq(defs.size(), 3_ul));
        expect(defs.contains(AlphaBlock));
        expect(defs.contains(BetaBlock));
        expect(defs.contains(GammaBlock));
    };

    // ── cache tests ───────────────────────────────────────────────────────────

    "cache: loading remote asset creates a cache file"_test = [] {
        if (kSkipRemote) {
            return;
        }
        clearCache();
        const std::string blockUri = kServerBase + "/root_cache/block_delta.yaml";

        auto loader = makeLoader({kServerBase + "/root_cache"});

        const auto DeltaBlock = "MyDeltaBlock";

        expect(loader.definitionForBlockName().contains(DeltaBlock));
        expect(std::filesystem::exists(cachePathFor(blockUri)));
    };

    "cache: fresh cache is used instead of remote"_test = [] {
        if (kSkipRemote) {
            return;
        }
        clearCache();
        const std::string blockUri = kServerBase + "/root_cache/block_delta.yaml";

        // First load: populates cache.
        {
            auto loader = makeLoader({kServerBase + "/root_cache"});
            expect(std::filesystem::exists(cachePathFor(blockUri)));
        }

        // Overwrite the cache file with a distinguishable block type, then set
        // its mtime to "now" (well after the 2020-06-15 modified stamp in index.yaml)
        // so that the cache is considered fresh on the next load.
        const auto cachePath = cachePathFor(blockUri);
        {
            std::ofstream f(cachePath);
            f << "definition_metadata:\n  block_type: CachedDeltaBlock\n";
        }
        std::filesystem::last_write_time(cachePath, std::filesystem::file_time_type::clock::now());

        auto loader = makeLoader({kServerBase + "/root_cache"});

        const auto DeltaBlock       = "MyDeltaBlock";
        const auto CachedDeltaBlock = "CachedDeltaBlock";

        // Should have read from cache, not remote.
        expect(loader.definitionForBlockName().contains(CachedDeltaBlock));
        expect(!loader.definitionForBlockName().contains(DeltaBlock));
    };

    "cache: stale cache is refreshed from remote"_test = [] {
        if (kSkipRemote) {
            return;
        }
        clearCache();
        const std::string blockUri = kServerBase + "/root_cache/block_delta.yaml";

        // Pre-seed cache with stale content.
        std::filesystem::create_directories(kCacheDir);
        const auto cachePath = cachePathFor(blockUri);
        {
            std::ofstream f(cachePath);
            f << "ndefinition_metadata:\n  block_type: StaleBlock\n";
        }

        // Recommended portable version:
        std::chrono::sys_days sys_tp = std::chrono::year{2019} / std::chrono::January / std::chrono::day{1};

        // Convert sys_days → file_time_type without clock_cast
        auto staleTime = std::chrono::file_clock::from_sys(sys_tp);

        std::filesystem::last_write_time(cachePath, staleTime);

        auto loader = makeLoader({kServerBase + "/root_cache"});

        const auto DeltaBlock = "MyDeltaBlock";
        const auto StaleBlock = "StaleBlock";

        // Stale cache should have been ignored; remote content loaded.
        expect(loader.definitionForBlockName().contains(DeltaBlock));
        expect(!loader.definitionForBlockName().contains(StaleBlock));
        // Cache should now be refreshed (mtime updated).
        expect(std::filesystem::last_write_time(cachePath) > staleTime);
    };
};

int main() { return boost::ut::cfg<boost::ut::override>.run(); }
