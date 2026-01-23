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

    if constexpr (std::is_same_v<T, std::string>) {
        auto value = it->second.value_or(std::string_view{});
        if (value.data() != nullptr) {
            return std::string(value);
        }
    } else {
        auto value = checked_access_ptr{it->second.get_if<T>()};
        if (value != nullptr) {
            return *value;
        }
    }

    return std::unexpected(gr::Error(std::format("Field {} in YAML object {} has an incorrect type {}:{} instead of {}", propertyName, map, it->second.value_type(), it->second.container_type(), gr::meta::type_name<T>())));
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

    auto value = checked_access_ptr{it->second.get_if<gr::property_map>()};
    if (value == nullptr) {
        return std::unexpected(gr::Error(std::format("Field {} in YAML object has an incorrect type {}:{} instead of gr::property_map", propertyName, it->second.value_type(), it->second.container_type())));
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

    Tensor<pmt::Value> blks;
    if (auto it = yaml.find("blocks"); it != yaml.end()) {
        if (const auto blkRef = checked_access_ptr<Tensor<pmt::Value>, false>{it->second.get_if<Tensor<pmt::Value>>()}; blkRef != nullptr) {
            blks = *blkRef;
        }
    }

    for (const auto& blk : blks) {
        const auto _grcBlock = checked_access_ptr{blk.get_if<property_map>()};
        if (_grcBlock == nullptr) {
            continue;
        }
        const auto& grcBlock = *_grcBlock;

        const auto blockName = getOrThrow(getProperty<std::string>(grcBlock, "parameters"sv, "name"sv));
        const auto blockType = getOrThrow(getProperty<std::string>(grcBlock, "id"sv));

        if (blockType == "SUBGRAPH") {
            auto loadGraph = [&grcBlock, &loader, &location](auto graphWrapper) {
                const auto _graphData = checked_access_ptr{grcBlock.at("graph").get_if<property_map>()};
                if (_graphData == nullptr) {
                    return;
                }
                const auto& graphData = *_graphData;
                gr::Graph&  graph     = *graphWrapper->graph();
                loadGraphFromMap(loader, graph, graphData);

                const auto& exportedPorts = graphData.at("exported_ports").value_or(Tensor<pmt::Value>());
                for (const auto& exportedPort_ : exportedPorts) {
                    auto exportedPort = checked_access_ptr{exportedPort_.get_if<Tensor<pmt::Value>>()};
                    if (exportedPort == nullptr || exportedPort->size() != 4) {
                        throw gr::exception(std::format("Unable to parse exported port ({} instead of 4 elements)", exportedPort != nullptr ? exportedPort->size() : -1UZ));
                    }

                    const auto requiredBlockName   = (*exportedPort)[0].value_or(std::string_view{});
                    const auto portDirectionString = (*exportedPort)[1].value_or(std::string_view{});
                    const auto internalPortName    = (*exportedPort)[2].value_or(std::string_view{});
                    const auto exportedPortName    = (*exportedPort)[3].value_or(std::string_view{});
                    if (requiredBlockName.data() == nullptr || portDirectionString.data() == nullptr || internalPortName.data() == nullptr || exportedPortName.data() == nullptr) {
                        throw gr::exception(std::format("Required fields for exported ports missing"));
                    }

                    std::string blockUniqueName;
                    for (auto& b : graph.blocks()) {
                        if (b->name() == requiredBlockName) {
                            blockUniqueName = b->uniqueName();
                        }
                    }
                    if (blockUniqueName.empty()) {
                        throw gr::exception(std::format("Required Block {} not found in:\n{}", requiredBlockName, gr::graph::format(graph)), location);
                    }

                    graphWrapper->exportPort(true,                                                     //
                        blockUniqueName,                                                               //
                        portDirectionString == "INPUT" ? PortDirection::INPUT : PortDirection::OUTPUT, //
                        internalPortName,                                                              //
                        exportedPortName);
                }
            };

            auto       schedulerIt = grcBlock.find("scheduler");
            const bool isManaged   = schedulerIt != grcBlock.end();

            if (isManaged) {
                auto schedulerPmt = checked_access_ptr{schedulerIt->second.get_if<property_map>()};
                if (schedulerPmt == nullptr) {
                    throw gr::exception(std::format("scheduler is not a property_map"));
                }
                auto schedulerId = getOrThrow(getProperty<std::string>(*schedulerPmt, "id"sv));

                property_map schedulerParams;
                if (auto paramsIt = schedulerPmt->find("parameters"); paramsIt != schedulerPmt->end()) {
                    if (const auto params = checked_access_ptr{paramsIt->second.get_if<property_map>()}; params != nullptr) {
                        schedulerParams = *params;
                    }
                }

                auto scheduler = loader.instantiateScheduler(schedulerId, schedulerParams);
                if (!scheduler) {
                    throw gr::exception(std::format("Unable to create scheduler of type '{}'", schedulerId));
                }

                auto schedulerBlock = SchedulerModel::asBlockModelPtr(scheduler);
                resultGraph.addBlock(schedulerBlock);
                createdBlocks[blockName] = schedulerBlock;
                schedulerBlock->setName(blockName);

                loadGraph(schedulerBlock);

            } else {
                const std::shared_ptr<BlockModel>& subGraph = resultGraph.addBlock(std::make_shared<GraphWrapper<gr::Graph>>());
                createdBlocks[blockName]                    = subGraph;
                subGraph->setName(blockName);

                loadGraph(static_cast<GraphWrapper<gr::Graph>*>(subGraph.get()));
            }
        } else {
            auto currentBlock = loader.instantiate(blockType);
            if (!currentBlock) {
                throw gr::exception(std::format("Unable to create block of type '{}'", blockType));
            }

            // This sets the previously read "name" field for the block
            currentBlock->setName(blockName);

            const auto parametersPmt = grcBlock.at("parameters");
            if (const auto parameters = checked_access_ptr{parametersPmt.get_if<property_map>()}; parameters != nullptr) {
                currentBlock->settings().loadParametersFromPropertyMap(*parameters);
            } else {
                currentBlock->settings().loadParametersFromPropertyMap({});
            }

            if (auto it = grcBlock.find("ctx_parameters"); it != grcBlock.end()) {
                const auto parametersCtx = checked_access_ptr{it->second.get_if<Tensor<pmt::Value>>()};
                if (parametersCtx == nullptr) {
                    throw gr::exception(std::format("ctx_parameters is not a vector<pmt::Value>"));
                }

                for (const auto& ctxPmt : *parametersCtx) {
                    const auto ctxPar = checked_access_ptr{ctxPmt.get_if<property_map>()};
                    if (ctxPar == nullptr) {
                        throw gr::exception(std::format("ctxPar is not a property_map"));
                    }

                    const auto ctxName       = ctxPar->at(gr::tag::CONTEXT.shortKey()).value_or(std::string_view{});
                    const auto ctxTime       = checked_access_ptr{ctxPar->at(gr::tag::CONTEXT_TIME.shortKey()).get_if<std::uint64_t>()};
                    const auto ctxParameters = checked_access_ptr{ctxPar->at("parameters").get_if<property_map>()};
                    if (ctxName.data() == nullptr || ctxTime == nullptr || ctxParameters == nullptr) {
                        throw gr::exception(std::format("Missing context values for loadParametersFromPropertyMap"));
                    }

                    currentBlock->settings().loadParametersFromPropertyMap(*ctxParameters, SettingsCtx{*ctxTime, pmt::Value(ctxName)});
                }
            }

            if (const auto failed = currentBlock->settings().activateContext(); failed == std::nullopt) {
                throw gr::exception("Settings for context could not be activated");
            }

            createdBlocks[blockName] = resultGraph.addBlock(std::move(currentBlock));
        }
    } // for blocks

    Tensor<pmt::Value> connections;
    if (auto it = yaml.find("connections"); it != yaml.end()) {
        if (const auto connRef = checked_access_ptr<Tensor<pmt::Value>, false>{it->second.get_if<Tensor<pmt::Value>>()}; connRef != nullptr) {
            connections = *connRef;
        }
    }

    for (const auto& conn : connections) {
        const auto _connection = checked_access_ptr{conn.get_if<Tensor<pmt::Value>>()};
        if (_connection == nullptr || _connection->size() < 4) {
            throw gr::exception(std::format("Unable to parse connection ({} instead of >=4 elements)", _connection == nullptr ? -1UZ : _connection->size()));
        }
        const auto& connection = *_connection;

        auto parseBlockPort = [&](const pmt::Value& blockField, const pmt::Value& portField) {
            const auto blockName = blockField.value_or(std::string_view{});
            if (blockName.empty()) {
                throw gr::exception(std::format("Invalid blockField"));
            }
            auto block = createdBlocks.find(std::string(blockName));
            if (block == createdBlocks.end()) {
                throw gr::exception(std::format("Unknown block '{}'", blockName));
            }

            struct result {
                decltype(block) block_it;
                PortDefinition  port_definition;
            };

            if (const auto portFields = checked_access_ptr<const Tensor<pmt::Value>, false>{portField.template get_if<Tensor<pmt::Value>>()}; portFields != nullptr) {
                if (portFields->size() != 2) {
                    throw gr::exception(std::format("Port definition has invalid length ({} instead of 2)", portFields->size()));
                }
                const auto index    = checked_access_ptr{portFields->at(0).template get_if<std::int64_t>()};
                const auto subIndex = checked_access_ptr{portFields->at(1).template get_if<std::int64_t>()};
                if (index == nullptr || subIndex == nullptr) {
                    throw gr::exception(std::format("Port definition missing values"));
                }

                return result{block, {static_cast<std::size_t>(*index), static_cast<std::size_t>(*subIndex)}};

            } else {
                const auto index = checked_access_ptr{portField.template get_if<std::int64_t>()};
                if (index == nullptr) {
                    throw gr::exception(std::format("Port definition missing values"));
                }
                return result{block, {static_cast<std::size_t>(*index)}};
            }
        };

        auto src = parseBlockPort(connection[0], connection[1]);
        auto dst = parseBlockPort(connection[2], connection[3]);

        if (connection.size() == 4) {
            resultGraph.connect(src.block_it->second, src.port_definition, dst.block_it->second, dst.port_definition, undefined_size, graph::defaultWeight, graph::defaultEdgeName, location);
        } else {
            std::size_t minBufferSize{};
            pmt::ValueVisitor([&minBufferSize]<typename TValue>(const TValue& value) {
                if constexpr (std::is_same_v<TValue, std::size_t>) {
                    minBufferSize = value;
                } else if constexpr (std::is_integral_v<TValue>) {
                    minBufferSize = static_cast<std::size_t>(value);
                } else {
                    minBufferSize = std::numeric_limits<std::size_t>::max();
                }
            }).visit(connection[4]);

            resultGraph.connect(src.block_it->second, src.port_definition, dst.block_it->second, dst.port_definition, minBufferSize, graph::defaultWeight, graph::defaultEdgeName, location);
        }
    } // for connections
}

inline gr::property_map saveGraphToMap(PluginLoader& loader, const gr::Graph& rootGraph) {
    property_map result;

    {
        const std::size_t  nBlocks = gr::graph::countBlocks<gr::block::Category::NormalBlock>(rootGraph);
        Tensor<pmt::Value> serializedBlocks;
        serializedBlocks.reserve(nBlocks);
        gr::graph::forEachBlock<gr::block::Category::NormalBlock>(rootGraph, [&](const std::shared_ptr<BlockModel>& block) {
            property_map map;

            if (gr::Graph* subgraph = block->graph()) {
                map.emplace("id", "SUBGRAPH");
                map["unique_name"] = std::string(block->uniqueName());
                map["name"]        = std::string(block->name());

                property_map graphYaml = detail::saveGraphToMap(loader, *subgraph);

                const std::size_t  nExportedPorts = block->exportedInputPorts().size() + block->exportedOutputPorts().size();
                Tensor<pmt::Value> exportedPortsData;
                exportedPortsData.reserve(nExportedPorts);
                for (const auto& [blockName, portName] : block->exportedInputPorts()) {
                    exportedPortsData.push_back(Tensor<pmt::Value>(data_from, {gr::pmt::Value(blockName), gr::pmt::Value("INPUT"s), gr::pmt::Value(portName)}));
                }
                for (const auto& [blockName, portName] : block->exportedOutputPorts()) {
                    exportedPortsData.push_back(Tensor<pmt::Value>(data_from, {gr::pmt::Value(blockName), gr::pmt::Value("OUTPUT"s), gr::pmt::Value(portName)}));
                }

                graphYaml["exported_ports"] = pmt::Value(std::move(exportedPortsData));
                map.emplace("graph"s, std::move(graphYaml));

                // TODO: a unit-test that this is working
                auto* schedulerModel = dynamic_cast<const SchedulerModel*>(block.get());

                if (schedulerModel != nullptr) {
                    property_map schedulerMap;
                    schedulerMap["id"] = loader.schedulerRegistry().typeName(block);
                    map["scheduler"]   = std::move(schedulerMap);
                }

            } else {
                map = serializeBlock(loader, block, BlockSerializationFlags::All & (~BlockSerializationFlags::Ports));
            }

            serializedBlocks.emplace_back(std::move(map));
        });
        result["blocks"] = std::move(serializedBlocks);
    }

    {
        const std::size_t  nEdges = gr::graph::countEdges<block::Category::NormalBlock>(rootGraph);
        Tensor<pmt::Value> serializedConnections;
        serializedConnections.reserve(nEdges);
        graph::forEachEdge<block::Category::NormalBlock>(rootGraph, [&](const Edge& edge) { // NormalBlock -> perhaps can be modelled to 'ALL' for a cleaner sub-graph handling
            Tensor<pmt::Value> seq;
            seq.reserve(7);

            auto writePortDefinition = [&](const auto& definition) { //
                std::visit(meta::overloaded(                         //
                               [&](const PortDefinition::IndexBased& _definition) {
                                   if (_definition.subIndex != meta::invalid_index) {
                                       Tensor<pmt::Value> seqPort;
                                       seqPort.reserve(2);
                                       seqPort.push_back(pmt::Value(std::int64_t(_definition.topLevel)));
                                       seqPort.push_back(pmt::Value(std::int64_t(_definition.subIndex)));
                                       seq.push_back(std::move(seqPort));
                                   } else {
                                       seq.push_back(pmt::Value(std::int64_t(_definition.topLevel)));
                                   }
                               },                                                    //
                               [&](const PortDefinition::StringBased& _definition) { //
                                   seq.push_back(pmt::Value(_definition.name));
                               }),
                    definition.definition);
            };

            seq.push_back(pmt::Value(edge.sourceBlock()->name()));
            writePortDefinition(edge.sourcePortDefinition());

            seq.push_back(pmt::Value(edge.destinationBlock()->name()));
            writePortDefinition(edge.destinationPortDefinition());

            if (edge.minBufferSize() != std::numeric_limits<std::size_t>::max()) {
                seq.push_back(pmt::Value(edge.minBufferSize()));
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
    const auto                    yaml = pmt::yaml::deserialize(yamlSrc);
    if (!yaml) {
        throw gr::exception(std::format("Could not parse yaml: {}:{}\n{}", yaml.error().message, yaml.error().line, yamlSrc));
    }

    detail::loadGraphFromMap(loader, *resultGraph, *yaml, location);
    return resultGraph;
}

inline std::string saveGrc(PluginLoader& loader, const gr::Graph& rootGraph) { return pmt::yaml::serialize(detail::saveGraphToMap(loader, rootGraph)); }

} // namespace gr

#endif // include guard
