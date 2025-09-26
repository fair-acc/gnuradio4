#ifndef GNURADIO_GRAPH_YAML_IMPORTER_H
#define GNURADIO_GRAPH_YAML_IMPORTER_H

#include <ranges>

#include <gnuradio-4.0/meta/indirect.hpp>

#include <gnuradio-4.0/YamlPmt.hpp>

#include "BlockModel.hpp"
#include "Graph.hpp"
#include "PluginLoader.hpp"

namespace gr {

namespace detail {

template<typename T>
inline std::expected<T, gr::Error> getProperty(const gr::property_map& map, std::string_view propertyName) {
    auto it = map.find(propertyName);
    if (it == map.cend()) {
        return std::unexpected(gr::Error(std::format("Missing field {} in YAML object", propertyName)));
    }

    auto* value = std::get_if<T>(&it->second);
    if (value == nullptr) {
        return std::unexpected(gr::Error(std::format("Field {} in YAML object has an incorrect type index={} instead of {}", propertyName, it->second.index(), gr::meta::type_name<T>())));
    }

    return {*value};
}

template<typename T>
inline std::expected<T, gr::Error> getProperty(const gr::property_map& map, std::string_view propertyName, const auto&... propertySubNames)
requires(sizeof...(propertySubNames) > 0)
{
    static_assert((std::is_convertible_v<decltype(propertySubNames), std::string_view> && ...));
    auto it = map.find(propertyName);
    if (it == map.cend()) {
        return std::unexpected(gr::Error(std::format("Missing field {} in YAML object", propertyName)));
    }

    auto* value = std::get_if<gr::property_map>(&it->second);
    if (value == nullptr) {
        return std::unexpected(gr::Error(std::format("Field {} in YAML object has an incorrect type index={} instead of gr::property_map", propertyName, it->second.index())));
    }

    return getProperty<T>(*value, propertySubNames...);
}

template<typename T>
T getOrThrow(std::expected<T, gr::Error>&& expectedValue, std::source_location location = std::source_location::current()) {
    if (!expectedValue) {
        throw gr::exception(std::format("Got an error {}, caller {}:{}", expectedValue.error().message, location.file_name(), location.line()));
    } else {
        return *expectedValue;
    }
}

inline void loadGraphFromMap(PluginLoader& loader, gr::Graph& resultGraph, gr::property_map yaml, std::source_location location = std::source_location::current()) {

    std::map<std::string, std::shared_ptr<BlockModel>> createdBlocks;

    std::vector<pmtv::pmt> blks;
    if (auto it = yaml.find("blocks"); it != yaml.end()) {
        if (const auto* blkRef = std::get_if<std::vector<pmtv::pmt>>(&it->second)) {
            blks = *blkRef;
        }
    }

    for (const auto& blk : blks) {
        auto grcBlock = std::get<property_map>(blk);

        const auto blockName = getOrThrow(getProperty<std::string>(grcBlock, "parameters"sv, "name"sv));
        const auto blockType = getOrThrow(getProperty<std::string>(grcBlock, "id"sv));

        if (blockType == "SUBGRAPH") {
            const std::shared_ptr<BlockModel>& subGraph = resultGraph.addBlock(std::make_shared<GraphWrapper<gr::Graph>>());
            createdBlocks[blockName]                    = subGraph;
            subGraph->setName(blockName);

            auto* subGraphDirect = static_cast<GraphWrapper<gr::Graph>*>(subGraph.get());
            subGraphDirect->setName(blockName);

            const auto& graphData = std::get<property_map>(grcBlock["graph"]);
            loadGraphFromMap(loader, subGraphDirect->blockRef(), graphData);

            const auto& exportedPorts = std::get<std::vector<pmtv::pmt>>(graphData.at("exported_ports"));
            for (const auto& exportedPort_ : exportedPorts) {
                auto exportedPort = std::get<std::vector<pmtv::pmt>>(exportedPort_);
                if (exportedPort.size() != 3) {
                    throw std::format("Unable to parse exported port ({} instead of 4 elements)", exportedPort.size());
                }

                std::string requiredBlockName = std::get<std::string>(exportedPort[0]);
                std::string blockUniqueName;
                for (auto& b : subGraphDirect->blockRef().blocks()) {
                    if (b->name() == requiredBlockName) {
                        blockUniqueName = b->uniqueName();
                    }
                }
                if (blockUniqueName.empty()) {
                    throw gr::exception(std::format("Required Block {} not found in:\n{}", requiredBlockName, gr::graph::format(subGraphDirect->blockRef())), location);
                }

                subGraphDirect->exportPort(true,
                    /* block's unique name */ blockUniqueName,
                    /* port direction */ std::get<std::string>(exportedPort[1]) == "INPUT" ? PortDirection::INPUT : PortDirection::OUTPUT,
                    /* port name */ std::get<std::string>(exportedPort[2]));
            }
        } else {
            auto currentBlock = loader.instantiate(blockType);
            if (!currentBlock) {
                throw std::format("Unable to create block of type '{}'", blockType);
            }

            // This sets the previously read "name" field for the block
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

            createdBlocks[blockName] = resultGraph.addBlock(std::move(currentBlock));
        }
    } // for blocks

    std::vector<pmtv::pmt> connections;
    if (auto it = yaml.find("connections"); it != yaml.end()) {
        if (const auto* connRef = std::get_if<std::vector<pmtv::pmt>>(&it->second)) {
            connections = *connRef;
        }
    }

    for (const auto& conn : connections) {
        auto connection = std::get<std::vector<pmtv::pmt>>(conn);
        if (connection.size() < 4) {
            throw std::format("Unable to parse connection ({} instead of >=4 elements)", connection.size());
        }

        auto parseBlockPort = [&](const auto& blockField, const auto& portField) {
            const auto blockName = std::get<std::string>(blockField);
            auto       block     = createdBlocks.find(blockName);
            if (block == createdBlocks.end()) {
                throw std::format("Unknown block '{}'", blockName);
            }

            struct result {
                decltype(block) block_it;
                PortDefinition  port_definition;
            };

            if (const auto portFields = std::get_if<std::vector<pmtv::pmt>>(&portField)) {
                if (portFields->size() != 2) {
                    throw std::format("Port definition has invalid length ({} instead of 2)", portFields->size());
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
            resultGraph.connect(src.block_it->second, src.port_definition, dst.block_it->second, dst.port_definition, undefined_size, graph::defaultWeight, graph::defaultEdgeName, location);
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

            resultGraph.connect(src.block_it->second, src.port_definition, dst.block_it->second, dst.port_definition, minBufferSize, graph::defaultWeight, graph::defaultEdgeName, location);
        }
    } // for connections
}

inline gr::property_map saveGraphToMap(PluginLoader& loader, const gr::Graph& rootGraph) {
    property_map result;

    {
        const std::size_t      nBlocks = gr::graph::countBlocks<gr::block::Category::NormalBlock>(rootGraph);
        std::vector<pmtv::pmt> serializedBlocks;
        serializedBlocks.reserve(nBlocks);
        gr::graph::forEachBlock<gr::block::Category::NormalBlock>(rootGraph, [&](const std::shared_ptr<BlockModel>& block) {
            property_map map;
            const auto&  fullTypeName = loader.registry().typeName(block);
            if (fullTypeName == "gr::Graph") {
                map.emplace("id"s, "SUBGRAPH"s);
                map["unique_name"s] = std::string(block->uniqueName());
                map["name"s]        = std::string(block->name());

                auto* subGraphDirect = dynamic_cast<const GraphWrapper<gr::Graph>*>(block.get());
                if (subGraphDirect == nullptr) {
                    throw gr::Error(std::format("Can not serialize gr::Graph-based subgraph {} which is not added to the parent graph {} via GraphWrapper", block->uniqueName(), rootGraph.unique_name));
                }
                property_map graphYaml = detail::saveGraphToMap(loader, subGraphDirect->blockRef());

                const std::size_t      nExportedPorts = subGraphDirect->exportedInputPortsForBlock().size() + subGraphDirect->exportedOutputPortsForBlock().size();
                std::vector<pmtv::pmt> exportedPortsData;
                exportedPortsData.reserve(nExportedPorts);
                for (const auto& [blockName, portName] : subGraphDirect->exportedInputPortsForBlock()) {
                    exportedPortsData.push_back(std::vector<pmtv::pmt>{blockName, "INPUT"s, portName});
                }
                for (const auto& [blockName, portName] : subGraphDirect->exportedOutputPortsForBlock()) {
                    exportedPortsData.push_back(std::vector<pmtv::pmt>{blockName, "OUTPUT"s, portName});
                }

                graphYaml["exported_ports"s] = std::move(exportedPortsData);
                map.emplace("graph"s, std::move(graphYaml));

            } else {
                map = serializeBlock(loader, block, BlockSerializationFlags::All & (~BlockSerializationFlags::Ports));
            }

            serializedBlocks.emplace_back(std::move(map));
        });
        result["blocks"s] = std::move(serializedBlocks);
    }

    {
        const std::size_t      nEdges = gr::graph::countEdges<block::Category::NormalBlock>(rootGraph);
        std::vector<pmtv::pmt> serializedConnections;
        serializedConnections.reserve(nEdges);
        graph::forEachEdge<block::Category::NormalBlock>(rootGraph, [&](const Edge& edge) { // NormalBlock -> perhaps can be modelled to 'ALL' for a cleaner sub-graph handling
            std::vector<pmtv::pmt> seq;
            seq.reserve(7);

            auto writePortDefinition = [&](const auto& definition) { //
                std::visit(meta::overloaded(                         //
                               [&](const PortDefinition::IndexBased& _definition) {
                                   if (_definition.subIndex != meta::invalid_index) {
                                       std::vector<pmtv::pmt> seqPort;
                                       seqPort.reserve(2);
                                       seqPort.push_back(std::int64_t(_definition.topLevel));
                                       seqPort.push_back(std::int64_t(_definition.subIndex));
                                       seq.push_back(std::move(seqPort));
                                   } else {
                                       seq.push_back(std::int64_t(_definition.topLevel));
                                   }
                               }, //
                               [&](const PortDefinition::StringBased& _definition) { seq.push_back(_definition.name); }),
                    definition.definition);
            };

            seq.push_back(std::string(edge.sourceBlock()->name()));
            writePortDefinition(edge.sourcePortDefinition());

            seq.push_back(std::string(edge.destinationBlock()->name()));
            writePortDefinition(edge.destinationPortDefinition());

            if (edge.minBufferSize() != std::numeric_limits<std::size_t>::max()) {
                seq.push_back(edge.minBufferSize());
            }

            serializedConnections.emplace_back(std::move(seq));
        });
        result["connections"] = std::move(serializedConnections);
    }

    return result;
}

} // namespace detail

inline gr::meta::indirect<gr::Graph> loadGrc(PluginLoader& loader, std::string_view yamlSrc, std::source_location location = std::source_location::current()) {
    gr::meta::indirect<gr::Graph> resultGraph;
    const auto                    yaml = pmtv::yaml::deserialize(yamlSrc);
    if (!yaml) {
        throw gr::exception(std::format("Could not parse yaml: {}:{}\n{}", yaml.error().message, yaml.error().line, yamlSrc));
    }

    detail::loadGraphFromMap(loader, *resultGraph, *yaml, location);
    return resultGraph;
}

inline std::string saveGrc(PluginLoader& loader, const gr::Graph& rootGraph) { return pmtv::yaml::serialize(detail::saveGraphToMap(loader, rootGraph)); }

} // namespace gr

#endif // include guard
