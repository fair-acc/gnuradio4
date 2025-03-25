#ifndef GNURADIO_GRAPH_YAML_IMPORTER_H
#define GNURADIO_GRAPH_YAML_IMPORTER_H

#include <charconv>

#include <gnuradio-4.0/YamlPmt.hpp>

#include "Graph.hpp"
#include "PluginLoader.hpp"

namespace gr {

namespace detail {

inline void loadGraphFromMap(PluginLoader& loader, gr::Graph& resultGraph, gr::property_map yaml, std::source_location location = std::source_location::current()) {

    std::map<std::string, BlockModel*> createdBlocks;

    auto blks = std::get<std::vector<pmtv::pmt>>(yaml.at("blocks"));
    for (const auto& blk : blks) {
        auto grcBlock = std::get<property_map>(blk);

        const auto blockName = std::get<std::string>(grcBlock["name"]);
        const auto blockType = std::get<std::string>(grcBlock["id"]);

        if (blockType == "SUBGRAPH") {
            auto& subGraph           = resultGraph.addBlock(std::make_unique<GraphWrapper<gr::Graph>>());
            createdBlocks[blockName] = &subGraph;
            subGraph.setName(blockName);

            auto* subGraphDirect = static_cast<GraphWrapper<gr::Graph>*>(&subGraph);
            subGraphDirect->setName(blockName);

            const auto& graphData = std::get<property_map>(grcBlock["graph"]);
            loadGraphFromMap(loader, subGraphDirect->blockRef(), graphData);

            const auto& exportedPorts = std::get<std::vector<pmtv::pmt>>(graphData.at("exported_ports"));
            for (const auto& exportedPort_ : exportedPorts) {
                auto exportedPort = std::get<std::vector<pmtv::pmt>>(exportedPort_);
                if (exportedPort.size() != 3) {
                    throw fmt::format("Unable to parse exported port ({} instead of 4 elements)", exportedPort.size());
                }

                auto& block = subGraphDirect->findFirstBlockWithName(std::get<std::string>(exportedPort[0]));

                subGraphDirect->exportPort(true,
                    /* block's unique name */ std::string(block.uniqueName()),
                    /* port direction */ std::get<std::string>(exportedPort[1]) == "INPUT" ? PortDirection::INPUT : PortDirection::OUTPUT,
                    /* port name */ std::get<std::string>(exportedPort[2]));
            }
        } else {
            auto currentBlock = loader.instantiate(blockType);
            if (!currentBlock) {
                throw fmt::format("Unable to create block of type '{}'", blockType);
            }

            currentBlock->setName(blockName);

            const auto parametersPmt = grcBlock["parameters"];
            if (const auto parameters = std::get_if<property_map>(&parametersPmt)) {
                currentBlock->settings().loadParametersFromPropertyMap(*parameters);
            } else {
                currentBlock->settings().loadParametersFromPropertyMap({});
            }

            if (auto it = grcBlock.find("ctx_parameters"); it != grcBlock.end()) {
                auto parametersCtx = std::get<std::vector<pmtv::pmt>>(it->second);
                for (const auto& ctxPmt : parametersCtx) {
                    auto       ctxPar        = std::get<property_map>(ctxPmt);
                    const auto ctxName       = std::get<std::string>(ctxPar[gr::tag::CONTEXT.shortKey()]);
                    const auto ctxTime       = std::get<std::uint64_t>(ctxPar[gr::tag::CONTEXT_TIME.shortKey()]); // in ns
                    const auto ctxParameters = std::get<property_map>(ctxPar["parameters"]);

                    currentBlock->settings().loadParametersFromPropertyMap(ctxParameters, SettingsCtx{ctxTime, ctxName});
                }
            }
            if (const auto failed = currentBlock->settings().activateContext(); failed == std::nullopt) {
                throw gr::exception("Settings for context could not be activated");
            }
            createdBlocks[blockName] = &resultGraph.addBlock(std::move(currentBlock));
        }
    } // for blocks

    auto connections = std::get<std::vector<pmtv::pmt>>(yaml.at("connections"));
    for (const auto& conn : connections) {
        auto connection = std::get<std::vector<pmtv::pmt>>(conn);
        if (connection.size() < 4) {
            throw fmt::format("Unable to parse connection ({} instead of >=4 elements)", connection.size());
        }

        auto parseBlockPort = [&](const auto& blockField, const auto& portField) {
            const auto blockName = std::get<std::string>(blockField);
            auto       block     = createdBlocks.find(blockName);
            if (block == createdBlocks.end()) {
                throw fmt::format("Unknown block '{}'", blockName);
            }

            struct result {
                decltype(block) block_it;
                PortDefinition  port_definition;
            };

            if (const auto portFields = std::get_if<std::vector<pmtv::pmt>>(&portField)) {
                if (portFields->size() != 2) {
                    throw fmt::format("Port definition has invalid length ({} instead of 2)", portFields->size());
                }
                const auto index    = std::get<std::int64_t>(portFields->at(0));
                const auto subIndex = std::get<std::int64_t>(portFields->at(1));
                return result{block, {static_cast<std::size_t>(index), static_cast<std::size_t>(subIndex)}};

            } else {
                const auto index = std::get<std::int64_t>(portField);
                return result{block, {static_cast<std::size_t>(index)}};
            }
        };

        auto src = parseBlockPort(connection[0], connection[1]);
        auto dst = parseBlockPort(connection[2], connection[3]);

        if (connection.size() == 4) {
            resultGraph.connect(*src.block_it->second, src.port_definition, *dst.block_it->second, dst.port_definition, undefined_size, graph::defaultWeight, graph::defaultEdgeName, location);
        } else {
            auto minBufferSize = std::visit(
                []<typename TValue>(const TValue& value) {
                    if constexpr (std::is_same_v<TValue, std::size_t>) {
                        return value;
                    } else if constexpr (std::is_integral_v<TValue>) {
                        return static_cast<std::size_t>(value);
                    } else {
                        return std::numeric_limits<std::size_t>::max();
                    }
                },
                connection[4]);

            resultGraph.connect(*src.block_it->second, src.port_definition, *dst.block_it->second, dst.port_definition, minBufferSize, graph::defaultWeight, graph::defaultEdgeName, location);
        }
    } // for connections
}

inline gr::property_map saveGraphToMap(PluginLoader& loader, const gr::Graph& rootGraph) {
    pmtv::map_t result;

    {
        std::vector<pmtv::pmt> serializedBlocks;
        rootGraph.forEachBlock([&](const auto& block) {
            pmtv::map_t map;
            map["name"] = std::string(block.name());

            const auto& fullTypeName = loader.registry().blockTypeName(block);
            if (fullTypeName == "gr::Graph") {
                map.emplace("id", "SUBGRAPH");
                auto* subGraphDirect = dynamic_cast<const GraphWrapper<gr::Graph>*>(std::addressof(block));
                if (subGraphDirect == nullptr) {
                    throw gr::Error(fmt::format("Can not serialize gr::Graph-based subgraph {} which is not added to the parent graph {} via GraphWrapper", block.uniqueName(), rootGraph.unique_name));
                }
                property_map graphYaml = detail::saveGraphToMap(loader, subGraphDirect->blockRef());

                std::vector<pmtv::pmt> exportedPortsData;
                for (const auto& [blockName, portName] : subGraphDirect->exportedInputPortsForBlock()) {
                    exportedPortsData.push_back(std::vector<pmtv::pmt>{blockName, "INPUT"s, portName});
                }
                for (const auto& [blockName, portName] : subGraphDirect->exportedOutputPortsForBlock()) {
                    exportedPortsData.push_back(std::vector<pmtv::pmt>{blockName, "OUTPUT"s, portName});
                }

                graphYaml["exported_ports"] = std::move(exportedPortsData);
                map.emplace("graph", std::move(graphYaml));

            } else {
                map.emplace("id", fullTypeName);

                // Helper function to write parameters
                auto writeParameters = [&](const property_map& settingsMap, const property_map& metaInformation = {}) {
                    pmtv::map_t parameters;
                    auto        writeMap = [&](const auto& localMap) {
                        for (const auto& [settingsKey, settingsValue] : localMap) {
                            std::visit([&]<typename T>(const T& value) { parameters[settingsKey] = value; }, settingsValue);
                        }
                    };
                    writeMap(settingsMap);
                    if (!metaInformation.empty()) {
                        writeMap(metaInformation);
                    }
                    return parameters;
                };

                const auto& stored = block.settings().getStoredAll();
                if (stored.contains("")) {
                    const auto& ctxParameters = stored.at("");
                    const auto& settingsMap   = ctxParameters.back().second; // write only the last parameters
                    if (!block.metaInformation().empty() || !settingsMap.empty()) {
                        map["parameters"] = writeParameters(settingsMap, block.metaInformation());
                    }
                }

                std::vector<pmtv::pmt> ctxParamsSeq;
                for (const auto& [ctx, ctxParameters] : stored) {
                    if (ctx == "") {
                        continue;
                    }

                    for (const auto& [ctxTime, settingsMap] : ctxParameters) {
                        pmtv::map_t ctxParam;

                        // Convert ctxTime.context to a string, regardless of its actual type
                        std::string contextStr = std::visit(
                            [](const auto& arg) -> std::string {
                                using T = std::decay_t<decltype(arg)>;
                                if constexpr (std::is_same_v<T, std::string>) {
                                    return arg;
                                } else if constexpr (std::is_arithmetic_v<T>) {
                                    return std::to_string(arg);
                                }
                                return "";
                            },
                            ctxTime.context);

                        ctxParam[gr::tag::CONTEXT.shortKey()]      = contextStr;
                        ctxParam[gr::tag::CONTEXT_TIME.shortKey()] = ctxTime.time;
                        ctxParam["parameters"]                     = writeParameters(settingsMap);
                        ctxParamsSeq.emplace_back(std::move(ctxParam));
                    }
                }
                map["ctx_parameters"] = ctxParamsSeq;
            }

            serializedBlocks.emplace_back(std::move(map));
        });
        result["blocks"] = std::move(serializedBlocks);
    }

    {
        std::vector<pmtv::pmt> serializedConnections;
        rootGraph.forEachEdge([&](const auto& edge) {
            std::vector<pmtv::pmt> seq;

            auto writePortDefinition = [&](const auto& definition) { //
                std::visit(meta::overloaded(                         //
                               [&](const PortDefinition::IndexBased& _definition) {
                                   if (_definition.subIndex != meta::invalid_index) {
                                       std::vector<pmtv::pmt> seqPort;

                                       seqPort.push_back(std::int64_t(_definition.topLevel));
                                       seqPort.push_back(std::int64_t(_definition.subIndex));
                                       seq.push_back(seqPort);
                                   } else {
                                       seq.push_back(std::int64_t(_definition.topLevel));
                                   }
                               }, //
                               [&](const PortDefinition::StringBased& _definition) { seq.push_back(_definition.name); }),
                    definition.definition);
            };

            seq.push_back(edge.sourceBlock().name().data());
            writePortDefinition(edge.sourcePortDefinition());

            seq.push_back(edge.destinationBlock().name().data());
            writePortDefinition(edge.destinationPortDefinition());

            if (edge.minBufferSize() != std::numeric_limits<std::size_t>::max()) {
                seq.push_back(edge.minBufferSize());
            }

            serializedConnections.emplace_back(seq);
        });
        result["connections"] = std::move(serializedConnections);
    }

    return result;
}

} // namespace detail

inline gr::Graph loadGrc(PluginLoader& loader, std::string_view yamlSrc, std::source_location location = std::source_location::current()) {
    Graph      resultGraph;
    const auto yaml = pmtv::yaml::deserialize(yamlSrc);
    if (!yaml) {
        throw gr::exception(fmt::format("Could not parse yaml: {}:{}\n{}", yaml.error().message, yaml.error().line, yamlSrc));
    }

    detail::loadGraphFromMap(loader, resultGraph, *yaml, location);
    return resultGraph;
}

inline std::string saveGrc(PluginLoader& loader, const gr::Graph& rootGraph) { return pmtv::yaml::serialize(detail::saveGraphToMap(loader, rootGraph)); }

} // namespace gr

#endif // include guard
