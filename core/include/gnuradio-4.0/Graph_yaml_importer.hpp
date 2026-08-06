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
        auto value = (*it).second.value_or(std::string_view{});
        if (value.data() != nullptr) {
            return std::string(value);
        }
    } else {
        const Value entry = (*it).second; // bind to lvalue; ValueMap iter yields by value
        auto        raw   = entry.get_if<T>();
        if constexpr (requires {
                          raw.has_value();
                          raw.value();
                      }) {
            if (raw.has_value()) {
                if constexpr (std::same_as<T, gr::property_map>) {
                    return raw->owned(map.resource()); // get_if<ValueMap> aliases entry's bytes; materialise onto the source map's arena
                } else {
                    static_assert(!std::is_same_v<T, std::string_view> && !gr::TensorViewLike<T>, "getProperty<T>: T is a view type aliasing the source map's bytes; instantiate with an owning type (std::string, gr::property_map, gr::Tensor<X>) or materialise at the call site.");
                    return T(std::move(*raw));
                }
            }
        } else {
            if (raw != nullptr) {
                return *raw;
            }
        }
    }

    return std::unexpected(gr::Error(std::format("Field {} in YAML object {} has an incorrect type {}:{} instead of {}", propertyName, map, (*it).second.value_type(), (*it).second.container_type(), gr::meta::type_name<T>())));
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

    const Value entryValue = (*it).second; // bind to lvalue: ValueMap iter yields by value, the Map view below would alias a temporary
    auto        value      = entryValue.get_if<gr::property_map>();
    if (!value) {
        return std::unexpected(gr::Error(std::format("Field {} in YAML object has an incorrect type {}:{} instead of gr::property_map", propertyName, entryValue.value_type(), entryValue.container_type())));
    }

    return getProperty<T>(*value, propertySubNames...);
}

template<typename T>
inline std::expected<T, gr::Error> getProperty(const gr::pmt::Value& val, std::string_view propertyName, const auto&... propertySubNames) {
    auto mapOpt = val.get_if<gr::property_map>();
    if (!mapOpt) {
        return std::unexpected(gr::Error(std::format("pmt::Value is not a property_map (type: {}:{})", val.value_type(), val.container_type())));
    }
    return getProperty<T>(*mapOpt, propertyName, propertySubNames...);
}

inline std::expected<void, gr::Error> loadGraphFromMap(PluginLoader& loader, gr::Graph& resultGraph, gr::property_map yaml, std::source_location location = std::source_location::current()) {
    std::map<std::string, std::shared_ptr<BlockModel>> createdBlocks;

    Tensor<Value> blks;
    if (auto it = yaml.find("blocks"); it != yaml.end()) {
        const Value blkValue = (*it).second; // ValueMap iter yields by value; bind to lvalue so view aliases live storage
        if (auto blkView = blkValue.get_if<TensorView<Value>>()) {
            blks = blkView->owned();
        }
    }

    for (const auto& blk : blks) {
        const auto _grcBlock = blk.get_if<property_map>();
        if (!_grcBlock) {
            continue;
        }
        const auto& grcBlock = *_grcBlock;

        auto blockName = getProperty<std::string>(grcBlock, "parameters"sv, "name"sv);
        if (!blockName) {
            return std::unexpected(std::move(blockName).error());
        }

        auto blockType = getProperty<std::string>(grcBlock, "id"sv);
        if (!blockType) {
            return std::unexpected(std::move(blockType).error());
        }

        if (*blockType == "SUBGRAPH") {
            auto loadGraph = [&grcBlock, &loader, &location](auto graphWrapper) -> std::expected<void, gr::Error> {
                const auto graphIt = grcBlock.find("graph");
                if (graphIt == grcBlock.end()) {
                    return {};
                }
                const Value graphEntry = (*graphIt).second;
                const auto  _graphData = graphEntry.get_if<property_map>();
                if (!_graphData) {
                    return {};
                }
                const auto& graphData = *_graphData;
                gr::Graph&  graph     = *graphWrapper->graph();
                if (auto result = loadGraphFromMap(loader, graph, graphData); !result) {
                    return result;
                }

                const auto exportedPortsIt = graphData.find("exported_ports");
                // Tensor<Value> decode requires Value::_resource for sub-Value allocations.
                const Value exportedPortsValue = exportedPortsIt != graphData.end() ? Value{(*exportedPortsIt).second} : Value{};
                const auto  exportedPorts      = exportedPortsValue.value_or(Tensor<Value>{});
                for (const auto& exportedPort_ : exportedPorts) {
                    auto exportedPort = exportedPort_.get_if<TensorView<Value>>();
                    if (!exportedPort || exportedPort->size() != 4) {
                        return std::unexpected(gr::Error(std::format("Unable to parse exported port ({} instead of 4 elements)", exportedPort ? exportedPort->size() : -1UZ)));
                    }

                    // snapshot keeps Values alive so string_views from value_or<> don't dangle
                    const auto fields              = exportedPort->owned();
                    const auto requiredBlockName   = fields.data()[0].value_or(std::string_view{});
                    const auto portDirectionString = fields.data()[1].value_or(std::string_view{});
                    const auto internalPortName    = fields.data()[2].value_or(std::string_view{});
                    const auto exportedPortName    = fields.data()[3].value_or(std::string_view{});
                    if (requiredBlockName.data() == nullptr || portDirectionString.data() == nullptr || internalPortName.data() == nullptr || exportedPortName.data() == nullptr) {
                        return std::unexpected(gr::Error("Required fields for exported ports missing"));
                    }

                    std::string blockUniqueName;
                    for (auto& b : graph.blocks()) {
                        if (b->name() == requiredBlockName) {
                            blockUniqueName = b->uniqueName();
                        }
                    }
                    if (blockUniqueName.empty()) {
                        return std::unexpected(gr::Error(std::format("Required Block {} not found in:\n{}", requiredBlockName, gr::graph::format(graph)), location));
                    }

                    if (auto result = graphWrapper->exportPort(true,                                       //
                            blockUniqueName,                                                               //
                            portDirectionString == "INPUT" ? PortDirection::INPUT : PortDirection::OUTPUT, //
                            internalPortName,                                                              //
                            exportedPortName);
                        !result.has_value()) {
                        return std::unexpected(std::move(result).error());
                    }
                }
                return {};
            };

            auto       schedulerIt = grcBlock.find("scheduler");
            const bool isManaged   = schedulerIt != grcBlock.end();

            if (isManaged) {
                const Value schedulerValue = (*schedulerIt).second; // bind to lvalue so the Map* below aliases live storage
                auto        schedulerPmt   = schedulerValue.get_if<property_map>();
                if (!schedulerPmt) {
                    return std::unexpected(gr::Error("scheduler is not a property_map"));
                }
                auto schedulerIdResult = getProperty<std::string>(*schedulerPmt, "id"sv);
                if (!schedulerIdResult) {
                    return std::unexpected(std::move(schedulerIdResult).error());
                }
                auto schedulerId = std::move(*schedulerIdResult);

                property_map schedulerParams;
                if (auto paramsIt = schedulerPmt->find("parameters"); paramsIt != schedulerPmt->end()) {
                    const Value paramsValue = (*paramsIt).second; // bind to lvalue
                    if (auto params = paramsValue.get_if<property_map>()) {
                        schedulerParams = *params;
                    }
                }

                auto scheduler = loader.instantiateScheduler(schedulerId, schedulerParams);
                if (!scheduler) {
                    return std::unexpected(gr::Error(std::format("Unable to create scheduler of type '{}'", schedulerId)));
                }

                auto schedulerBlock = SchedulerModel::asBlockModelPtr(scheduler);
                resultGraph.addBlock(schedulerBlock);
                createdBlocks[*blockName] = schedulerBlock;
                schedulerBlock->setName(*blockName);

                if (auto result = loadGraph(schedulerBlock); !result) {
                    return result;
                }

            } else {
                const std::shared_ptr<BlockModel>& subGraph = resultGraph.addBlock(std::make_shared<GraphWrapper<gr::Graph>>());
                createdBlocks[*blockName]                   = subGraph;
                subGraph->setName(*blockName);

                if (auto result = loadGraph(static_cast<GraphWrapper<gr::Graph>*>(subGraph.get())); !result) {
                    return result;
                }
            }
        } else {
            auto currentBlock = loader.instantiate(*blockType);
            if (!currentBlock) {
                return std::unexpected(gr::Error(std::format("Unable to create block of type '{}'", *blockType)));
            }

            currentBlock->setName(*blockName);

            if (auto paramsIt = grcBlock.find("parameters"); paramsIt != grcBlock.end()) {
                const Value parametersPmt = (*paramsIt).second; // bind to lvalue so get_if<property_map>() aliases live storage
                if (auto parameters = parametersPmt.get_if<property_map>()) {
                    currentBlock->settings().loadParametersFromPropertyMap(*parameters);
                } else {
                    currentBlock->settings().loadParametersFromPropertyMap(property_map{});
                }
            } else {
                currentBlock->settings().loadParametersFromPropertyMap(property_map{});
            }

            if (auto it = grcBlock.find("ctx_parameters"); it != grcBlock.end()) {
                const Value ctxParamsValue = (*it).second; // bind to lvalue so the TensorView aliases live storage
                auto        parametersCtx  = ctxParamsValue.get_if<TensorView<Value>>();
                if (!parametersCtx) {
                    return std::unexpected(gr::Error("ctx_parameters is not a vector<Value>"));
                }

                for (const auto& ctxPmt : *parametersCtx) {
                    const auto ctxPar = ctxPmt.get_if<property_map>();
                    if (!ctxPar) {
                        return std::unexpected(gr::Error("ctxPar is not a property_map"));
                    }

                    // bind to lvalues — string_view / get_if<>() pointers alias the Value's storage
                    const auto findOrImpl = [&ctxPar](std::string_view key) -> Value {
                        auto entryIt = ctxPar->find(key);
                        return entryIt != ctxPar->end() ? (*entryIt).second : Value{};
                    };
                    const auto findOr = [&findOrImpl](const auto& key) -> Value {
                        auto result = findOrImpl(key.key());
                        if (result) {
                            return result;
                        }
                        return findOrImpl(key.shortKey());
                    };
                    const Value ctxNameVal       = findOr(gr::tag::CONTEXT);
                    const Value ctxTimeVal       = findOr(gr::tag::CONTEXT_TIME);
                    const Value ctxParametersVal = findOrImpl("parameters");
                    const auto  ctxName          = std::string(ctxNameVal.value_or(std::string_view{}));
                    const auto  ctxTime          = ctxTimeVal.get_if<std::uint64_t>();
                    const auto  ctxParameters    = ctxParametersVal.get_if<property_map>();

                    if (ctxName.empty() || !ctxTime || !ctxParameters) {
                        return std::unexpected(gr::Error("Missing context values for loadParametersFromPropertyMap"));
                    }

                    currentBlock->settings().loadParametersFromPropertyMap(*ctxParameters, SettingsCtx{*ctxTime, ctxName});
                }
            }

            if (const auto failed = currentBlock->settings().activateContext(); failed == std::nullopt) {
                return std::unexpected(gr::Error("Settings for context could not be activated"));
            }

            createdBlocks[*blockName] = resultGraph.addBlock(std::move(currentBlock));
        }
    } // for blocks

    Tensor<Value> connections;
    if (auto it = yaml.find("connections"); it != yaml.end()) {
        const Value connValue = (*it).second; // bind to lvalue so the TensorView aliases live storage
        if (auto connView = connValue.get_if<TensorView<Value>>()) {
            connections = connView->owned();
        }
    }

    for (const auto& conn : connections) {
        auto _connection = conn.get_if<TensorView<Value>>();
        if (!_connection || _connection->size() < 4) {
            return std::unexpected(gr::Error(std::format("Unable to parse connection ({} instead of >=4 elements)", _connection ? _connection->size() : -1UZ)));
        }
        const auto& connection = *_connection;

        struct ParsedBlockPort {
            decltype(createdBlocks)::iterator block_it;
            PortDefinition                    port_definition;
        };

        auto parseBlockPort = [&](const Value& blockField, const Value& portField) -> std::expected<ParsedBlockPort, gr::Error> {
            const auto blockName = blockField.value_or(std::string_view{});
            if (blockName.empty()) {
                return std::unexpected(gr::Error("Invalid blockField"));
            }
            auto block = createdBlocks.find(std::string(blockName));
            if (block == createdBlocks.end()) {
                return std::unexpected(gr::Error(std::format("Unknown block '{}'", blockName)));
            }

            if (auto portFields = portField.template get_if<TensorView<Value>>()) {
                if (portFields->size() != 2) {
                    return std::unexpected(gr::Error(std::format("Port definition has invalid length ({} instead of 2)", portFields->size())));
                }
                const auto fields   = portFields->owned();
                const auto index    = checked_access_ptr{fields.data()[0].template get_if<std::int64_t>()};
                const auto subIndex = checked_access_ptr{fields.data()[1].template get_if<std::int64_t>()};
                if (index == nullptr || subIndex == nullptr) {
                    return std::unexpected(gr::Error("Port definition missing values"));
                }

                return ParsedBlockPort{block, {static_cast<std::size_t>(*index), static_cast<std::size_t>(*subIndex)}};

            } else if (const auto portFieldString = portField.value_or(std::string_view{}); portFieldString.data()) {
                return ParsedBlockPort{block, {std::string(portFieldString)}};

            } else {
                const auto index = checked_access_ptr{portField.template get_if<std::int64_t>()};
                if (index == nullptr) {
                    return std::unexpected(gr::Error("Port definition missing values"));
                }
                return ParsedBlockPort{block, {static_cast<std::size_t>(*index)}};
            }
        };

        auto src = parseBlockPort(connection[0], connection[1]);
        if (!src) {
            return std::unexpected(std::move(src).error());
        }
        auto dst = parseBlockPort(connection[2], connection[3]);
        if (!dst) {
            return std::unexpected(std::move(dst).error());
        }

        if (connection.size() == 4) {
            if (auto r = resultGraph.connect(src->block_it->second, src->port_definition, dst->block_it->second, dst->port_definition, EdgeParameters{.minBufferSize = undefined_size, .weight = graph::defaultWeight, .name = graph::defaultEdgeName}, location); !r) {
                return std::unexpected(gr::Error(std::format("connection failed: {}", r.error().message)));
            }
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

            if (auto r = resultGraph.connect(src->block_it->second, src->port_definition, dst->block_it->second, dst->port_definition, EdgeParameters{.minBufferSize = minBufferSize, .weight = graph::defaultWeight, .name = graph::defaultEdgeName}, location); !r) {
                return std::unexpected(gr::Error(std::format("connection failed: {}", r.error().message)));
            }
        }
    } // for connections

    return {};
}

inline gr::property_map saveGraphToMap(PluginLoader& loader, const gr::Graph& rootGraph) {
    property_map result;

    {
        const std::size_t nBlocks = gr::graph::countBlocks<gr::block::Category::NormalBlock>(rootGraph);
        Tensor<Value>     serializedBlocks;
        serializedBlocks.reserve(nBlocks);
        gr::graph::forEachBlock<gr::block::Category::NormalBlock>(rootGraph, [&serializedBlocks, &loader](const std::shared_ptr<BlockModel>& block) { serializedBlocks.emplace_back(serializeBlock(loader, block, BlockSerializationFlags::All)); });
        result.insert_or_assign(std::string_view{"blocks"}, std::move(serializedBlocks));
    }

    {
        const std::size_t nEdges = gr::graph::countEdges<block::Category::NormalBlock>(rootGraph);
        Tensor<Value>     serializedConnections;
        serializedConnections.reserve(nEdges);
        graph::forEachEdge<block::Category::NormalBlock>(rootGraph, [&serializedConnections](const Edge& edge) { // NormalBlock -> perhaps can be modelled to 'ALL' for a cleaner sub-graph handling
            Tensor<Value> seq;
            seq.reserve(7);

            auto writePortDefinition = [&](const auto& definition) {
                if (auto* idx = std::get_if<PortDefinition::IndexBased>(&definition.definition)) {
                    if (idx->subIndex != meta::invalid_index) {
                        Tensor<Value> seqPort;
                        seqPort.reserve(2);
                        seqPort.push_back(std::int64_t(idx->topLevel));
                        seqPort.push_back(std::int64_t(idx->subIndex));
                        seq.push_back(std::move(seqPort));
                    } else {
                        seq.push_back(std::int64_t(idx->topLevel));
                    }
                } else {
                    auto& str = std::get<PortDefinition::StringBased>(definition.definition);
                    seq.push_back(str.name);
                }
            };

            seq.push_back(edge.sourceBlock()->name());
            writePortDefinition(edge.sourcePortDefinition());

            seq.push_back(edge.destinationBlock()->name());
            writePortDefinition(edge.destinationPortDefinition());

            if (edge.minBufferSize() != std::numeric_limits<std::size_t>::max()) {
                seq.push_back(static_cast<gr::Size_t>(edge.minBufferSize()));
            }

            serializedConnections.emplace_back(std::move(seq));
        });
        result.insert_or_assign(std::string_view{"connections"}, std::move(serializedConnections));
    }

    return result;
}

} // namespace detail

inline std::expected<gr::meta::indirect<gr::Graph>, gr::Error> loadGrc(PluginLoader& loader, std::string_view yamlSrc, std::source_location location = std::source_location::current()) {
    gr::meta::indirect<gr::Graph> resultGraph;
    const auto                    yaml = pmt::yaml::deserialize(yamlSrc);
    if (!yaml) {
        return std::unexpected(gr::Error(std::format("Could not parse yaml: {}:{}\n{}", yaml.error().message, yaml.error().line, yamlSrc)));
    }

    if (auto result = detail::loadGraphFromMap(loader, *resultGraph, *yaml, location); !result) {
        return std::unexpected(std::move(result).error());
    }
    return resultGraph;
}

inline std::string saveGrc(PluginLoader& loader, const gr::Graph& rootGraph) { return pmt::yaml::serialize(detail::saveGraphToMap(loader, rootGraph)); }

inline std::expected<std::shared_ptr<gr::BlockModel>, gr::Error> detail::instantiateBlockFromYamlDefinition(PluginLoader& loader, const detail::YamlDefinitionsLoader::Definition& def) noexcept {
    auto      consistencyResult = detail::checkEmbeddedVersionConsistency(loader.definitionForBlockName(), def);
    gr::Graph tempGraph;
    if (auto result = detail::loadGraphFromMap(loader, tempGraph, def.definition); !result) {
        return std::unexpected(std::move(result).error());
    }
    auto blocks = tempGraph.blocks();
    if (blocks.empty()) {
        return std::unexpected(gr::Error{"YAML definition produced no blocks"});
    }

    auto& result = blocks.front();

    result->uiConstraints().insert_or_assign(std::string_view{"yaml_definition_information"}, gr::property_map{
                                                                                                  //
                                                                                                  {"BLOCK_TYPE", def.metadata.block_type},                                                                             //
                                                                                                  {"PLUGIN_NAME", def.metadata.plugin_name},                                                                           //
                                                                                                  {"PLUGIN_VERSION", def.metadata.plugin_version},                                                                     //
                                                                                                  {"BLOCK_DEFINITION", def.definition},                                                                                //
                                                                                                  {"BLOCK_DEFINITION_UPDATED_INFO", consistencyResult.has_value() ? std::string() : consistencyResult.error().message} //
                                                                                              });

    return result;
}

} // namespace gr

#endif // include guard
