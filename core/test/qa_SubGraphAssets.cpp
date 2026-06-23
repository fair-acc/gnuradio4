#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <boost/ut.hpp>

#ifndef __EMSCRIPTEN__
#include <httplib.h>
#endif

#include <build_configure.hpp>

#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

#include "message_utils.hpp"

namespace ut = boost::ut;

namespace {

const std::string kAssetsDir  = std::string(TESTS_SOURCE_PATH) + "/assets";
const std::string kServerBase = "http://127.0.0.1:" + std::to_string(HTTP_SERVER_PORT);
const std::string kCacheDir   = gr::detail::YamlDefinitionsLoader::assetsCacheDir() + "/asset_cache";
const bool        kSkipRemote = std::getenv("GR_TEST_DISABLE_REMOTE") != nullptr;

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

std::filesystem::path cachePathFor(std::string_view uri) { return std::filesystem::path(kCacheDir) / gr::detail::uriToCacheFilename(uri); }

void clearCache() { std::filesystem::remove_all(kCacheDir); }

} // namespace

bool hasOneSubgraphBlock(const gr::property_map& definition) {
    try {
        const auto blocks = definition.find_value("blocks").value().value_or(gr::Tensor<gr::Value>{});
        if (blocks.size() != 1uz) {
            return false;
        }
        // get_if<ValueMap>() returns std::optional<ValueMap> view-mode; materialise into owning
        // copy so it survives the temp Value lifetime.
        const auto blockOpt = blocks[0].get_if<gr::property_map>();
        if (!blockOpt) {
            return false;
        }
        const auto block = blockOpt->owned();
        return block.find_value("id").value() == "SUBGRAPH";
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
        const auto portMap = portInfoVal.get_if<gr::property_map>();
        if (!portMap) {
            continue;
        }
        for (const auto& [_internalName, exportInfoVal] : *portMap) {
            const auto exportMap = exportInfoVal.get_if<gr::property_map>();
            if (!exportMap) {
                continue;
            }
            auto it = exportMap->find("exportedName");
            if (it != exportMap->end()) {
                names.emplace_back((*it).second.value_or(std::string_view{}));
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
    "happy path: three blocks loaded from root_a"_test = [] {
        auto loader = makeLoader({kAssetsDir + "/root_a"});

        const auto AlphaBlock       = "MyAlphaBlock";
        const auto BetaBlock        = "MyBetaBlock";
        const auto NestedGammaBlock = "MyNestedGammaBlock";

        const auto& defs = loader.definitionForBlockName();
        expect(eq(defs.size(), 3_ul));
        expect(defs.contains(AlphaBlock));
        expect(defs.contains(BetaBlock));
        expect(defs.contains(NestedGammaBlock));
        expect(eq(defs.at(AlphaBlock).metadata.block_type, "MyAlphaBlock"s));
        expect(eq(defs.at(AlphaBlock).metadata.plugin_name, "AlphaPlugin"s));
        expect(eq(defs.at(AlphaBlock).metadata.plugin_author, "Test Author"s));
        expect(eq(defs.at(AlphaBlock).metadata.plugin_license, "LGPL-3.0"s));
        expect(eq(defs.at(AlphaBlock).metadata.plugin_version, "2024-01-15"s));
        expect(eq(defs.at(BetaBlock).metadata.block_type, "MyBetaBlock"s));
        expect(defs.at(BetaBlock).metadata.plugin_name.empty());
        expect(eq(defs.at(NestedGammaBlock).metadata.block_type, "MyNestedGammaBlock"s));
        expect(eq(defs.at(NestedGammaBlock).metadata.plugin_name, "GammaPlugin"s));

        expect(hasOneSubgraphBlock(defs.at(AlphaBlock).definition));
        expect(hasOneSubgraphBlock(defs.at(BetaBlock).definition));
        expect(hasOneSubgraphBlock(defs.at(NestedGammaBlock).definition));
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

        const auto AlphaBlock       = "MyAlphaBlock";
        const auto BetaBlock        = "MyBetaBlock";
        const auto GammaBlock       = "MyGammaBlock";
        const auto NestedGammaBlock = "MyNestedGammaBlock";

        const auto& defs = loader.definitionForBlockName();
        expect(eq(defs.size(), 4_ul));
        expect(defs.contains(AlphaBlock));
        expect(defs.contains(BetaBlock));
        expect(defs.contains(GammaBlock));
        expect(defs.contains(NestedGammaBlock));

        expect(hasOneSubgraphBlock(defs.at(GammaBlock).definition));
        expect(hasOneSubgraphBlock(defs.at(NestedGammaBlock).definition));
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

        const auto yamlMetaVal = block->uiConstraints().find_value("yaml_definition_information");
        expect(yamlMetaVal.has_value()) << "yaml_definition_information must be present";
        if (yamlMetaVal) {
            expect(yamlMetaVal->is_map()) << "yaml_definition_information must be a map";
            if (const auto mapOpt = yamlMetaVal->get_if<gr::property_map>()) {
                expect(mapOpt->find_value("PLUGIN_NAME").has_value() && mapOpt->find_value("PLUGIN_NAME")->is_string()) << "PLUGIN_NAME must be a string";
                expect(mapOpt->find_value("PLUGIN_VERSION").has_value() && mapOpt->find_value("PLUGIN_VERSION")->is_string()) << "PLUGIN_VERSION must be a string";
                expect(mapOpt->find_value("BLOCK_DEFINITION").has_value() && mapOpt->find_value("BLOCK_DEFINITION")->is_map()) << "BLOCK_DEFINITION must be a map";
            }
        }

        const auto inputNames  = collectExportedNames(block->exportedInputPorts());
        const auto outputNames = collectExportedNames(block->exportedOutputPorts());
        expect(eq(inputNames.size(), 1uz)) << "expected one exported input port";
        expect(eq(outputNames.size(), 1uz)) << "expected one exported output port";
        expect(std::ranges::find(inputNames, "in") != inputNames.end()) << "exported input port must be named 'in'";
        expect(std::ranges::find(outputNames, "out") != outputNames.end()) << "exported output port must be named 'out'";
    };

    "nested sub-graph: gamma wraps alpha, both sub-graphs are instantiated"_test = [] {
        auto loader = makeLoaderWithPlugins({kAssetsDir + "/root_a"});

        auto block = loader.instantiate("MyNestedGammaBlock");
        expect(block != nullptr) << "instantiate must return a non-null block for MyNestedGammaBlock";
        if (!block) {
            return;
        }

        // The outer composite (gamma) must expose the forwarded alpha ports.
        const auto inputNames  = collectExportedNames(block->exportedInputPorts());
        const auto outputNames = collectExportedNames(block->exportedOutputPorts());
        expect(eq(inputNames.size(), 1uz)) << "nested gamma must have one exported input port";
        expect(eq(outputNames.size(), 1uz)) << "nested gamma must have one exported output port";
        expect(std::ranges::find(inputNames, "in") != inputNames.end()) << "exported input must be named 'in'";
        expect(std::ranges::find(outputNames, "out") != outputNames.end()) << "exported output must be named 'out'";

        // The outer sub-graph's internal graph must contain one block (the alpha instance).
        auto* graph = block->graph();
        expect(graph != nullptr) << "nested gamma must expose an inner graph";
        if (graph) {
            expect(eq(graph->blocks().size(), 1uz)) << "inner graph must contain exactly one block (alpha_inner)";
            if (!graph->blocks().empty()) {
                // The inner block is itself a sub-graph (MyAlphaBlock), so it must also
                // expose an inner graph with two multiply blocks chained together.
                auto* innerGraph = graph->blocks().front()->graph();
                expect(innerGraph != nullptr) << "alpha_inner must itself be a sub-graph";
                if (innerGraph) {
                    expect(eq(innerGraph->blocks().size(), 2uz)) << "alpha inner graph must contain two multiply blocks";
                }
            }
        }
    };

    // A stale version in yaml_definition_information must not prevent instantiation,
    // but the mismatch is recorded in BLOCK_DEFINITION_UPDATED_INFO.
    "nested sub-graph: stale embedded version records mismatch in BLOCK_DEFINITION_UPDATED_INFO"_test = [] {
        using namespace std::string_literals;
        auto loader = makeLoaderWithPlugins({kAssetsDir + "/root_a"});

        // Build a gamma-like definition whose inner MyAlphaBlock carries a stale version.
        constexpr std::string_view staleGammaYaml = R"(
definition_metadata:
  block_type: MyStaleGammaBlock
blocks:
  - id: SUBGRAPH
    parameters:
      name: stale_gamma
    graph:
      blocks:
        - id: MyAlphaBlock
          parameters:
            name: alpha_inner
          yaml_definition_information:
            BLOCK_TYPE: MyAlphaBlock
            PLUGIN_VERSION: "1970-01-01"
      exported_ports:
        - [alpha_inner, INPUT, in, in]
        - [alpha_inner, OUTPUT, out, out]
)";
        const auto                 parsedYaml     = gr::pmt::yaml::deserialize(staleGammaYaml);
        expect(parsedYaml.has_value()) << "stale gamma YAML must parse cleanly";
        if (!parsedYaml) {
            return;
        }

        const gr::detail::YamlDefinitionsLoader::Definition staleDef{
            *parsedYaml,
            gr_plugin_metadata{.plugin_name = ""s, .plugin_author = ""s, .plugin_license = ""s, .plugin_version = "2024-01-15"s, .block_type = "MyStaleGammaBlock"s},
        };

        // instantiateBlockFromYamlDefinition must succeed and store the mismatch message.
        auto result = gr::detail::instantiateBlockFromYamlDefinition(loader, staleDef);
        expect(result.has_value()) << "instantiation must succeed even with a version mismatch";
        if (!result) {
            return;
        }

        const auto yamlMetaVal = (*result)->uiConstraints().find_value("yaml_definition_information");
        expect(yamlMetaVal.has_value() && yamlMetaVal->is_map()) << "yaml_definition_information must be a map";
        std::string_view updatedInfo;
        if (yamlMetaVal) {
            if (const auto mapOpt = yamlMetaVal->get_if<gr::property_map>()) {
                const auto infoVal = mapOpt->find_value("BLOCK_DEFINITION_UPDATED_INFO");
                if (infoVal) {
                    updatedInfo = infoVal->value_or(std::string_view{});
                }
            }
        }
        expect(!updatedInfo.empty()) << "BLOCK_DEFINITION_UPDATED_INFO must be set on version mismatch";
        expect(updatedInfo.find("MyAlphaBlock") != std::string_view::npos) << "message must name the mismatched block";
        expect(updatedInfo.find("2024-01-15") != std::string_view::npos) << "message must include the current version";
    };
#endif

    // ── remote tests (server started by CMake fixture) ────────────────────────

    "remote happy path: three blocks loaded via http from root_a"_test = [] {
        if (kSkipRemote) {
            return;
        }
        clearCache();
        auto loader = makeLoader({kServerBase + "/root_a"});

        const auto AlphaBlock       = "MyAlphaBlock";
        const auto BetaBlock        = "MyBetaBlock";
        const auto NestedGammaBlock = "MyNestedGammaBlock";

        const auto& defs = loader.definitionForBlockName();
        expect(eq(defs.size(), 3_ul));
        expect(defs.contains(AlphaBlock));
        expect(defs.contains(BetaBlock));
        expect(defs.contains(NestedGammaBlock));

        expect(hasOneSubgraphBlock(defs.at(AlphaBlock).definition));
        expect(hasOneSubgraphBlock(defs.at(BetaBlock).definition));
        expect(hasOneSubgraphBlock(defs.at(NestedGammaBlock).definition));
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

        const auto AlphaBlock       = "MyAlphaBlock";
        const auto BetaBlock        = "MyBetaBlock";
        const auto GammaBlock       = "MyGammaBlock";
        const auto NestedGammaBlock = "MyNestedGammaBlock";

        const auto& defs = loader.definitionForBlockName();
        expect(eq(defs.size(), 4_ul));
        expect(defs.contains(AlphaBlock));
        expect(defs.contains(BetaBlock));
        expect(defs.contains(GammaBlock));
        expect(defs.contains(NestedGammaBlock));
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

#if defined(GR_ENABLE_BLOCK_REGISTRY) && defined(INTERNAL_ENABLE_BLOCK_PLUGINS)
const boost::ut::suite EmplaceBlockFromYamlAssetTests = [] {
    using namespace ut;
    using namespace ut::literals;
    using namespace std::string_literals;
    using namespace gr;
    using namespace gr::testing;
    using enum gr::message::Command;

    "kEmplaceBlock with YAML-defined block type creates a composite block"_test = [] {
        auto loader = makeLoaderWithPlugins({kAssetsDir + "/root_a"});

        gr::Graph                                                                     graph(loader);
        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::singleThreadedBlocking> scheduler;
        if (auto ret = scheduler.exchange(std::move(graph)); !ret) {
            expect(fatal(false)) << std::format("failed to init scheduler: {}", ret.error());
            return;
        }

        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        expect(toScheduler.connect(scheduler.msgIn).has_value());
        expect(scheduler.msgOut.connect(fromScheduler).has_value());

        expect(scheduler.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());
        expect(scheduler.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << "externalStep start() must prime to RUNNING without spawning a worker";

        auto schedulerThread = gr::test::thread_pool::executeScheduler("qa_SubGraphAssets::emplace", scheduler);
        expect(awaitCondition(scheduler, [&] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler must reach RUNNING";

        const std::size_t blocksBefore = scheduler.graph().blocks().size();

        auto reply = sendAndWaitForReply<Set>(toScheduler, fromScheduler, scheduler.unique_name, scheduler::property::kEmplaceBlock, //
            property_map{{"type", "MyAlphaBlock"s}, {"properties", property_map{}}},                                                 //
            [](const Message& msg) { return msg.endpoint == scheduler::property::kBlockEmplaced; });

        expect(reply.has_value()) << "kEmplaceBlock must succeed";
        expect(eq(scheduler.graph().blocks().size(), blocksBefore + 1UZ)) << "exactly one new block must be added";

        if (reply && reply->data.has_value()) {
            const auto& data = reply->data.value();
            expect(data.contains("id")) << "reply must contain id field";
        }

        scheduler.requestStop();
        schedulerThread.get();
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
    };

    "kEmplaceBlock with YAML-defined block type has exported ports"_test = [] {
        auto loader = makeLoaderWithPlugins({kAssetsDir + "/root_a"});

        gr::Graph                                                             graph(loader);
        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::singleThreaded> scheduler;
        if (auto ret = scheduler.exchange(std::move(graph)); !ret) {
            expect(fatal(false)) << std::format("failed to init scheduler: {}", ret.error());
            return;
        }

        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        expect(toScheduler.connect(scheduler.msgIn).has_value());
        expect(scheduler.msgOut.connect(fromScheduler).has_value());

        expect(scheduler.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());
        expect(scheduler.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << "externalStep start() must prime to RUNNING without spawning a worker";

        auto schedulerThread = gr::test::thread_pool::executeScheduler("qa_SubGraphAssets::ports", scheduler);
        expect(awaitCondition(scheduler, [&] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler must reach RUNNING";

        auto reply = sendAndWaitForReply<Set>(toScheduler, fromScheduler, scheduler.unique_name, scheduler::property::kEmplaceBlock, //
            property_map{{"type", "MyAlphaBlock"s}, {"properties", property_map{}}},                                                 //
            [](const Message& msg) { return msg.endpoint == scheduler::property::kBlockEmplaced; });

        expect(reply.has_value()) << "kEmplaceBlock must succeed";
        if (reply && reply->data.has_value()) {
            const auto& data               = reply->data.value();
            const auto  newBlockUniqueName = gr::test::get_value_or_fail<std::string>(data.find_value("unique_name").value());
            const auto& blocks             = scheduler.graph().blocks();
            auto        it                 = std::ranges::find_if(blocks, [&](const auto& b) { return b->uniqueName() == newBlockUniqueName; });
            expect(it != blocks.end()) << "emplaced block must be in graph";
            if (it != blocks.end()) {
                const auto inputNames  = collectExportedNames((*it)->exportedInputPorts());
                const auto outputNames = collectExportedNames((*it)->exportedOutputPorts());
                expect(eq(inputNames.size(), 1UZ)) << "must have one exported input port";
                expect(eq(outputNames.size(), 1UZ)) << "must have one exported output port";
                expect(std::ranges::find(inputNames, "in") != inputNames.end()) << "exported input must be 'in'";
                expect(std::ranges::find(outputNames, "out") != outputNames.end()) << "exported output must be 'out'";
            }
        }

        scheduler.requestStop();
        schedulerThread.get();
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
    };
};
#endif

int main() {
#ifndef __EMSCRIPTEN__
    httplib::Server httpServer;
    httpServer.set_mount_point("/", kAssetsDir);
    auto serverThread = std::thread([&httpServer] { httpServer.listen("127.0.0.1", HTTP_SERVER_PORT); });
    httpServer.wait_until_ready();
#else
    // Poll the pre-js HTTP server until a known asset responds. This replaces a
    // blind sleep; Node's http.createServer().listen(...) has no synchronous
    // ready signal, so probing is the only deterministic option here.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto reader = gr::algorithm::fileio::readAsync(kServerBase + "/root_a/index.yaml"); reader) {
            if (auto bytes = reader->get(); bytes && !bytes->empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
#endif

    const int result = boost::ut::cfg<boost::ut::override>.run();

#ifndef __EMSCRIPTEN__
    httpServer.stop();
    serverThread.join();
#endif
    return result;
}
