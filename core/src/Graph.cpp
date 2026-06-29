#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

namespace gr {

Graph::Graph(gr::PluginLoader& pluginLoader, property_map settings) : gr::Block<Graph>(std::move(settings)), _pluginLoader(std::addressof(pluginLoader)) {
    _blocks.reserve(100); // TODO: remove

    propertyCallbacks[graph::property::kInspectBlock]           = static_cast<BlockBase::PropertyCallback>(&Graph::propertyCallbackInspectBlock);
    propertyCallbacks[graph::property::kGraphInspect]           = static_cast<BlockBase::PropertyCallback>(&Graph::propertyCallbackGraphInspect);
    propertyCallbacks[graph::property::kRegistryBlockTypes]     = static_cast<BlockBase::PropertyCallback>(&Graph::propertyCallbackRegistryBlockTypes);
    propertyCallbacks[graph::property::kRegistrySchedulerTypes] = static_cast<BlockBase::PropertyCallback>(&Graph::propertyCallbackRegistrySchedulerTypes);
}

Graph::Graph(property_map settings) : Graph(gr::globalPluginLoader(), std::move(settings)) {}

gr::PluginLoader& Graph::pluginLoader() noexcept { return _pluginLoader != nullptr ? *_pluginLoader : gr::globalPluginLoader(); }

std::expected<std::shared_ptr<BlockModel>, Error> Graph::emplaceBlock(std::string_view type, property_map initialSettings) {
    if (type.starts_with("gr::Graph")) {
        auto subGraphModel = std::unique_ptr<BlockModel>(std::make_unique<GraphWrapper<Graph>>().release());
        return addBlock(std::move(subGraphModel));
    } else if (std::shared_ptr<BlockModel> block_load = _pluginLoader->instantiate(type, std::move(initialSettings)); block_load) {
        return addBlock(block_load);
    } else if (std::shared_ptr<SchedulerModel> scheduler_load = _pluginLoader->instantiateScheduler(type, std::move(initialSettings)); scheduler_load) {
        return addBlock(SchedulerModel::asBlockModelPtr(scheduler_load));
    }
    return std::unexpected(Error(std::format("Cannot create block '{}'", type)));
}

std::expected<std::pair<std::shared_ptr<BlockModel>, std::shared_ptr<BlockModel>>, Error> Graph::replaceBlock(std::string_view uniqueName, std::string_view type, const property_map& properties) {
    auto it = std::ranges::find_if(_blocks, [&uniqueName](const auto& block) { return block->uniqueName() == uniqueName; });
    if (it == _blocks.end()) {
        return std::unexpected(Error(std::format("Block {} was not found in {}", uniqueName, this->unique_name)));
    }

    auto newBlock = _pluginLoader->instantiate(type, properties);
    if (!newBlock) {
        return std::unexpected(Error(std::format("Cannot create block '{}'", type)));
    }

    addBlock(newBlock);

    for (auto& edge : _edges) {
        if (edge._sourceBlock == *it) {
            edge._sourceBlock = newBlock;
        }

        if (edge._destinationBlock == *it) {
            edge._destinationBlock = newBlock;
        }
    }

    std::shared_ptr<BlockModel> oldBlock = std::move(*it);
    _blocks.erase(it);

    return std::pair{std::move(oldBlock), newBlock};
}

namespace {
std::expected<std::string, Error> definedPortName(const std::shared_ptr<BlockModel>& block, PortDirection direction, const PortDefinition& definition, const std::source_location& location) {
    constexpr static auto stringBased = [](const PortDefinition::StringBased& portdef) -> std::expected<std::string, Error> { return portdef.name; };
    const auto            indexBased  = [direction, &block, &location](const PortDefinition::IndexBased& portdef) -> std::expected<std::string, Error> {
        const auto& ports = direction == PortDirection::INPUT ? block->dynamicInputPorts() : block->dynamicOutputPorts();
        if (portdef.topLevel >= ports.size()) {
            return std::unexpected(Error(std::format("port index {} out of range for block {}", portdef.topLevel, block->uniqueName()), location));
        }
        auto baseName = BlockModel::portName(ports[portdef.topLevel]);
        return portdef.subIndex == meta::invalid_index ? baseName : std::format("{}#{}", baseName, portdef.subIndex);
    };
    return std::visit(meta::overloaded{stringBased, indexBased}, definition.definition);
}
} // namespace

std::expected<std::shared_ptr<BlockModel>, Error> Graph::groupBlocks(std::span<const std::shared_ptr<BlockModel>> targetBlocks, std::string_view schedulerTypename, std::source_location location) {
    if (targetBlocks.empty()) {
        return std::unexpected(Error("cannot group an empty set of blocks", location));
    }

    const auto isGrouped = [&targetBlocks](const std::shared_ptr<BlockModel>& block) { return std::ranges::contains(targetBlocks, block); };

    struct PortExport {
        std::shared_ptr<BlockModel> block;
        PortDirection               direction;
        std::string                 internalName;
        std::string                 exportedName;

        auto doExport(gr::BlockModel& subgraph, std::source_location srcloc) const { return subgraph.exportPort(true, block->uniqueName(), direction, internalName, exportedName, srcloc); }
    };
    std::vector<PortExport> exportOperations;

    const auto findOrCreateExport = [&exportOperations](const std::shared_ptr<BlockModel>& block, PortDirection direction, std::string internalName) -> std::string {
        auto alreadyPlanned = std::ranges::find_if(exportOperations, [&](const PortExport& exportOperation) { return exportOperation.block == block && exportOperation.direction == direction && exportOperation.internalName == internalName; });
        if (alreadyPlanned != exportOperations.end()) {
            return alreadyPlanned->exportedName;
        }

        const auto  isTaken      = [&](std::string_view candidate) { return std::ranges::any_of(exportOperations, [&](const PortExport& exportOperation) { return exportOperation.direction == direction && exportOperation.exportedName == candidate; }); };
        std::string exportedName = std::format("{}.{}", block->name(), internalName);
        for (std::size_t suffix = 1; isTaken(exportedName); ++suffix) {
            exportedName = std::format("{}.{}_{}", block->name(), internalName, suffix);
        }
        exportOperations.emplace_back(block, direction, std::move(internalName), exportedName);
        return exportedName;
    };

    struct CrossingEdge {
        Edge        edge;
        std::string newPortName;
        bool        sourceBlockIsInSubgraph;
    };
    std::vector<Edge>         internalEdges;
    std::vector<CrossingEdge> crossingEdges;
    for (const Edge& edge : _edges) {
        const bool sourceGrouped      = isGrouped(edge.sourceBlock());
        const bool destinationGrouped = isGrouped(edge.destinationBlock());
        if (sourceGrouped && destinationGrouped) {
            internalEdges.push_back(edge);
        } else if (sourceGrouped || destinationGrouped) {
            const auto& [insideBlock, definition, direction] = sourceGrouped //
                                                                   ? std::tuple{edge.sourceBlock(), edge.sourcePortDefinition(), PortDirection::OUTPUT}
                                                                   : std::tuple{edge.destinationBlock(), edge.destinationPortDefinition(), PortDirection::INPUT};
            auto portName                                    = definedPortName(insideBlock, direction, definition, location);
            if (!portName.has_value()) {
                return std::unexpected(portName.error());
            }
            crossingEdges.emplace_back(edge, findOrCreateExport(insideBlock, direction, portName.value()), sourceGrouped);
        }
    }

    std::shared_ptr<BlockModel> subGraphModel;
    if (schedulerTypename.starts_with("gr::Graph")) {
        subGraphModel = std::make_shared<GraphWrapper<Graph>>();
    } else if (std::shared_ptr<SchedulerModel> scheduler = pluginLoader().instantiateScheduler(schedulerTypename, {}); scheduler) {
        subGraphModel = SchedulerModel::asBlockModelPtr(std::move(scheduler));
    } else {
        return std::unexpected(Error(std::format("cannot spawn sub-graph/scheduler of type '{}'", schedulerTypename), location));
    }
    gr::Graph* subGraph = subGraphModel->graph();

    // move all edges and blocks into the subgraphs
    std::erase_if(_edges, [&isGrouped](const Edge& edge) { return isGrouped(edge.sourceBlock()) || isGrouped(edge.destinationBlock()); });
    std::erase_if(_blocks, isGrouped);

    for (auto& block : targetBlocks) { // isGrouped() does not make sense to call after this
        subGraph->addBlock(block, false);
    }
    for (Edge& edge : internalEdges) {
        std::ignore = subGraph->addEdge(std::move(edge), location);
    }

    for (const PortExport& exportOperation : exportOperations) {
        if (auto result = exportOperation.doExport(*subGraphModel, location); !result.has_value()) {
            return std::unexpected(result.error());
        }
    }

    const std::shared_ptr<BlockModel>& added = addBlock(subGraphModel);
    for (const CrossingEdge& crossover : crossingEdges) {
        const Edge&    edge = crossover.edge;
        EdgeParameters parameters{.minBufferSize = edge.minBufferSize(), .weight = edge.weight(), .name = std::string(edge.name())};
        auto           connectResult = crossover.sourceBlockIsInSubgraph //
                                           ? connect(subGraphModel, PortDefinition(crossover.newPortName), edge.destinationBlock(), edge.destinationPortDefinition(), std::move(parameters), location)
                                           : connect(edge.sourceBlock(), edge.sourcePortDefinition(), subGraphModel, PortDefinition(crossover.newPortName), std::move(parameters), location);
        if (!connectResult.has_value()) {
            return std::unexpected(connectResult.error());
        }
    }

    return added;
}

std::expected<void, Error> Graph::ungroupBlocks(std::shared_ptr<BlockModel> subgraphBlock, std::source_location location) {
    assert(subgraphBlock->blockCategory() == block::Category::ScheduledBlockGroup || subgraphBlock->blockCategory() == block::Category::TransparentBlockGroup);
    gr::Graph* subGraph = subgraphBlock->graph();
    if (subGraph == nullptr) {
        return std::unexpected(Error(std::format("sub-graph '{}' does not expose its graph", subgraphBlock->uniqueName()), location));
    }

    struct InternalPort {
        std::shared_ptr<BlockModel> block;
        std::string                 portName;
    };
    using ExportedPortMap = std::unordered_map<std::string, InternalPort>;

    // mapping from the exported port name that the outer graph sees to the
    // inner block and its port that it represents
    const auto mapExternalNameToPort = [&subGraph, &location](property_map exportedPorts) -> std::expected<ExportedPortMap, Error> {
        ExportedPortMap map;
        for (const auto& [blockUniqueName, portEntries] : exportedPorts) {
            const auto ports = portEntries.get_if<property_map>();
            if (!ports.has_value()) {
                continue;
            }

            const auto innerBlock = graph::findBlock(*subGraph, std::string_view(blockUniqueName), location);
            if (!innerBlock.has_value()) {
                return std::unexpected(innerBlock.error());
            }

            for (const auto& [internalName, exportEntry] : *ports) {
                const auto exportInfo = exportEntry.get_if<property_map>();
                if (!exportInfo.has_value()) {
                    continue;
                }

                const auto exportedName = exportInfo->value_or<std::string_view>("exportedName", std::string_view{});
                if (!exportedName.empty()) {
                    map.emplace(std::string(exportedName), InternalPort{innerBlock.value(), std::string(internalName)});
                }
            }
        }
        return map;
    };

    const auto exportedInputs  = mapExternalNameToPort(subgraphBlock->exportedInputPorts());
    const auto exportedOutputs = mapExternalNameToPort(subgraphBlock->exportedOutputPorts());
    if (!exportedInputs.has_value()) {
        return std::unexpected(exportedInputs.error());
    }
    if (!exportedOutputs.has_value()) {
        return std::unexpected(exportedOutputs.error());
    }

    // search for all the connections to the subgraph first to avoid failing in the middle of the ungrouping operation due to an invalid name
    struct ConnectedEdge {
        std::size_t  edgeIndex;
        InternalPort correspondingInnerPort;
        bool         connectedToOutputs;
    };
    std::vector<ConnectedEdge> connectedEdges;
    for (std::size_t edgeIndex = 0UZ; edgeIndex < _edges.size(); ++edgeIndex) {
        const Edge& edge = _edges[edgeIndex];
        for (const bool connectedToOutputs : {true, false}) {
            if ((connectedToOutputs ? edge.sourceBlock() : edge.destinationBlock()) != subgraphBlock) {
                continue;
            }

            const PortDirection direction    = connectedToOutputs ? PortDirection::OUTPUT : PortDirection::INPUT;
            auto                exportedName = definedPortName(subgraphBlock, direction, connectedToOutputs ? edge.sourcePortDefinition() : edge.destinationPortDefinition(), location);
            if (!exportedName.has_value()) {
                return std::unexpected(exportedName.error());
            }

            const ExportedPortMap& exportedPorts = connectedToOutputs ? exportedOutputs.value() : exportedInputs.value();
            const auto             exportedIt    = exportedPorts.find(exportedName.value());
            if (exportedIt == exportedPorts.end()) {
                return std::unexpected(Error(std::format("edge references non-exported port '{}' of sub-graph '{}'", exportedName.value(), subgraphBlock->uniqueName()), location));
            }

            connectedEdges.emplace_back(edgeIndex, exportedIt->second, connectedToOutputs);
        }
    }

    // move the blocks and internal edges into this graph
    for (const std::shared_ptr<BlockModel>& block : subGraph->blocks()) {
        addBlock(block, false);
    }
    for (Edge& edge : subGraph->edges()) {
        std::ignore = addEdge(std::move(edge), location);
    }

    for (const ConnectedEdge& oldEdgeDescription : connectedEdges) {
        Edge& edge = _edges[oldEdgeDescription.edgeIndex];
        if (oldEdgeDescription.connectedToOutputs) {
            edge = Edge(oldEdgeDescription.correspondingInnerPort.block,            //
                PortDefinition(oldEdgeDescription.correspondingInnerPort.portName), //
                edge.destinationBlock(), edge.destinationPortDefinition(),          //
                edge.minBufferSize(), edge.weight(), std::string(edge.name()));
        } else {
            edge = Edge(edge.sourceBlock(), edge.sourcePortDefinition(),            //
                oldEdgeDescription.correspondingInnerPort.block,                    //
                PortDefinition(oldEdgeDescription.correspondingInnerPort.portName), //
                edge.minBufferSize(), edge.weight(), std::string(edge.name()));
        }
    }

    subGraph->clear();
    std::erase_if(_blocks, [&subgraphBlock](const std::shared_ptr<BlockModel>& block) { return block == subgraphBlock; });
    return {};
}

std::optional<Message> Graph::propertyCallbackRegistryBlockTypes([[maybe_unused]] std::string_view propertyName, Message message) {
    assert(propertyName == graph::property::kRegistryBlockTypes);
    const auto&   availableBlocks = _pluginLoader->availableBlocks();
    Tensor<Value> types(availableBlocks | std::views::transform([](const std::string& type) { return Value(type); }));
    message.data = property_map{{"types", types}};
    return message;
}

std::optional<Message> Graph::propertyCallbackRegistrySchedulerTypes([[maybe_unused]] std::string_view propertyName, Message message) {
    assert(propertyName == graph::property::kRegistryBlockTypes);
    const auto&   availableSchedulers = _pluginLoader->availableSchedulers();
    Tensor<Value> types(availableSchedulers | std::views::transform([](const std::string& type) { return Value(type); }));
    message.data = property_map{{"types", types}};
    return message;
}

std::optional<Message> Graph::propertyCallbackInspectBlock([[maybe_unused]] std::string_view propertyName, Message message) {
    assert(propertyName == graph::property::kInspectBlock);
    using namespace std::string_literals;

    gr::Message reply;
    reply.endpoint = graph::property::kBlockInspected;

    if (!message.data) {
        reply.data = std::unexpected(Error{"Invalid block specification"s});
        return reply;
    }
    const auto& data = *message.data;
    // copy bytes into owning std::string — value_or<string_view>() aliases temp Value's storage.
    const auto uniqueName = std::string(data.value_or<std::string_view>("uniqueName", std::string_view{}));
    if (uniqueName.empty()) {
        reply.data = std::unexpected(Error{"Invalid block specification"s});
        return reply;
    }

    auto it = std::ranges::find_if(_blocks, [&uniqueName](const auto& block) { return block->uniqueName() == uniqueName; });
    if (it == _blocks.end()) {
        reply.data = std::unexpected(Error{std::format("Block {} was not found in {}", uniqueName, this->unique_name)});
        return reply;
    }

    const bool yamlSerialize = [&] {
        if (const auto fmt = data.find("serialization_format"); fmt != data.cend()) {
            // value_or<string_view>: alloc-free; comparing through Value would build a temp Value(const char*).
            return (*fmt).second.value_or(std::string_view{}) == "yaml";
        }
        return false;
    }();

    if (yamlSerialize) {
        reply.data = property_map{{"yamlData", pmt::yaml::serialize(serializeBlock(*_pluginLoader, *it, BlockSerializationFlags::All))}};
    } else {
        reply.data = serializeBlock(*_pluginLoader, *it, BlockSerializationFlags::All);
    }
    return {reply};
}

std::optional<Message> Graph::propertyCallbackGraphInspect([[maybe_unused]] std::string_view propertyName, Message message) {
    assert(propertyName == graph::property::kGraphInspect);

    if (const bool yamlSerialize =
            [&] {
                if (!message.data) {
                    return false;
                }
                if (const auto it = message.data->find("serialization_format"); it != message.data->cend()) {
                    return (*it).second.value_or(std::string_view{}) == "yaml";
                }
                return false;
            }();
        !yamlSerialize) {
        message.data = [&] {
            property_map _result;
            auto&        result = _result;

            result[std::pmr::string(serialization_fields::BLOCK_NAME)]        = std::string(name);
            result[std::pmr::string(serialization_fields::BLOCK_UNIQUE_NAME)] = std::string(unique_name);
            result[std::pmr::string(serialization_fields::BLOCK_CATEGORY)]    = std::string(gr::meta::enumName(blockCategory).value_or(""));

            property_map serializedChildren;
            for (const auto& child : blocks()) {
                serializedChildren[std::pmr::string(child->uniqueName())] = serializeBlock(*_pluginLoader, child, BlockSerializationFlags::All);
            }
            result[std::pmr::string(serialization_fields::BLOCK_CHILDREN)] = std::move(serializedChildren);

            property_map serializedEdges;
            std::size_t  index = 0UZ;
            for (const auto& edge : edges()) {
                serializedEdges[convert_string_domain(std::to_string(index))] = serializeEdge(edge);
                index++;
            }
            result[std::pmr::string(serialization_fields::BLOCK_EDGES)] = std::move(serializedEdges);
            return result;
        }();
    } else {
        message.data = {{"yamlData", saveGrc(*_pluginLoader, *this)}};
    }

    message.endpoint = graph::property::kGraphInspected;
    return message;
}
} // namespace gr
