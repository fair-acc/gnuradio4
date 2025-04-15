#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockModel.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Sequence.hpp>
#include <gnuradio-4.0/YamlPmt.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/meta/typelist.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <tuple>
#include <variant>

#if !__has_include(<source_location> )
#define HAVE_SOURCE_LOCATION 0
#else

#include <source_location>

#if defined __cpp_lib_source_location && __cpp_lib_source_location >= 201907L
#define HAVE_SOURCE_LOCATION 1
#else
#define HAVE_SOURCE_LOCATION 0
#endif
#endif

namespace gr {

class Graph;
std::string  saveGrc(PluginLoader& loader, const gr::Graph& rootGraph);
inline Graph loadGrc(PluginLoader& loader, std::string_view yamlSrc, std::source_location location = std::source_location::current());

namespace graph::property {
inline static const char* kEmplaceBlock = "EmplaceBlock";
inline static const char* kRemoveBlock  = "RemoveBlock";
inline static const char* kReplaceBlock = "ReplaceBlock";
inline static const char* kInspectBlock = "InspectBlock";

inline static const char* kBlockEmplaced  = "BlockEmplaced";
inline static const char* kBlockRemoved   = "BlockRemoved";
inline static const char* kBlockReplaced  = "BlockReplaced";
inline static const char* kBlockInspected = "BlockInspected";

inline static const char* kEmplaceEdge = "EmplaceEdge";
inline static const char* kRemoveEdge  = "RemoveEdge";

inline static const char* kEdgeEmplaced = "EdgeEmplaced";
inline static const char* kEdgeRemoved  = "EdgeRemoved";

inline static const char* kGraphInspect   = "GraphInspect";
inline static const char* kGraphInspected = "GraphInspected";

inline static const char* kGraphGRC = "GraphGRC";

inline static const char* kRegistryBlockTypes = "RegistryBlockTypes";

inline static const char* kSubgraphExportPort   = "SubgraphExportPort";
inline static const char* kSubgraphExportedPort = "SubgraphExportedPort";
} // namespace graph::property

namespace graph {
inline static constexpr std::size_t  defaultMinBufferSize(bool isArithmeticLike) { return isArithmeticLike ? 65536UZ : 64UZ; }
inline static constexpr std::int32_t defaultWeight   = 0;
inline static const std::string      defaultEdgeName = "unnamed edge"; // Emscripten doesn't want constexpr strings
} // namespace graph

template<typename TSubGraph>
class GraphWrapper : public BlockWrapper<TSubGraph> {
private:
    std::unordered_multimap<std::string, std::string> _exportedInputPortsForBlock;
    std::unordered_multimap<std::string, std::string> _exportedOutputPortsForBlock;

public:
    GraphWrapper() {
        // We need to make sure nobody touches our dynamic ports
        // as this class will handle them
        this->_dynamicPortsLoader.instance = nullptr;

        this->_block.propertyCallbacks[graph::property::kSubgraphExportPort] = [this](auto& /*self*/, std::string_view /*property*/, Message message) -> std::optional<Message> {
            const auto&        data            = message.data.value();
            const std::string& uniqueBlockName = std::get<std::string>(data.at("uniqueBlockName"s));
            auto               portDirection   = std::get<std::string>(data.at("portDirection"s)) == "input" ? PortDirection::INPUT : PortDirection::OUTPUT;
            const std::string& portName        = std::get<std::string>(data.at("portName"s));
            const bool         exportFlag      = std::get<bool>(data.at("exportFlag"s));

            exportPort(exportFlag, uniqueBlockName, portDirection, portName);

            message.endpoint = graph::property::kSubgraphExportedPort;
            return message;
        };
    }

    void exportPort(bool exportFlag, const std::string& uniqueBlockName, PortDirection portDirection, const std::string& portName) {
        auto [infoIt, infoFound] = findExportedPortInfo(uniqueBlockName, portDirection, portName);
        if (infoFound == exportFlag) {
            throw Error(fmt::format("Port {} in block {} export status already as desired {}", portName, uniqueBlockName, exportFlag));
        }

        auto& port                  = findPortInBlock(uniqueBlockName, portDirection, portName);
        auto& bookkeepingCollection = portDirection == PortDirection::INPUT ? _exportedInputPortsForBlock : _exportedOutputPortsForBlock;
        auto& portCollection        = portDirection == PortDirection::INPUT ? this->_dynamicInputPorts : this->_dynamicOutputPorts;
        if (exportFlag) {
            bookkeepingCollection.emplace(uniqueBlockName, portName);
            portCollection.push_back(port.weakRef());
        } else {
            bookkeepingCollection.erase(infoIt);
            // TODO: Add support for exporting port collections
            auto portIt = std::ranges::find_if(portCollection, [needleName = port.name](const auto& portOrCollection) {
                return std::visit(meta::overloaded{
                                      //
                                      [&](DynamicPort& in) { return in.name == needleName; }, //
                                      [](auto&) { return false; }                             //
                                  },
                    portOrCollection);
            });
            if (portIt != portCollection.end()) {
                portCollection.erase(portIt);
            } else {
                throw Error("Port was not exported, while it is registered as such");
            }
        }

        updateMetaInformation();
    }

    auto& blockRef() { return BlockWrapper<TSubGraph>::blockRef(); }
    auto& blockRef() const { return BlockWrapper<TSubGraph>::blockRef(); }

    const std::unordered_multimap<std::string, std::string>& exportedInputPortsForBlock() const { return _exportedInputPortsForBlock; }
    const std::unordered_multimap<std::string, std::string>& exportedOutputPortsForBlock() const { return _exportedOutputPortsForBlock; }

    BlockModel& findBlockWithUniqueName(std::string uniqueBlockName) {
        for (const auto& block : this->blocks()) {
            if (std::string(block->uniqueName()) == uniqueBlockName) {
                return *block;
            }
        }
        throw Error(fmt::format("Block {} not found in {}", uniqueBlockName, this->uniqueName()));
    }

    BlockModel& findFirstBlockWithName(std::string blockName) {
        for (const auto& block : this->blocks()) {
            if (std::string(block->name()) == blockName) {
                return *block;
            }
        }
        throw Error(fmt::format("Block {} not found in {}", blockName, this->uniqueName()));
    }

private:
    DynamicPort& findPortInBlock(const std::string& uniqueBlockName, PortDirection portDirection, const std::string& portName) {
        auto& block = findBlockWithUniqueName(uniqueBlockName);

        if (portDirection == PortDirection::INPUT) {
            return block.dynamicInputPort(portName);
        } else {
            return block.dynamicOutputPort(portName);
        }
    }

    auto findExportedPortInfo(const std::string& uniqueBlockName, PortDirection portDirection, const std::string& portName) const {
        auto& bookkeepingCollection = portDirection == PortDirection::INPUT ? _exportedInputPortsForBlock : _exportedOutputPortsForBlock;
        const auto& [from, to]      = bookkeepingCollection.equal_range(std::string(uniqueBlockName));
        for (auto it = from; it != to; it++) {
            if (it->second == portName) {
                return std::make_pair(it, true);
            }
        }
        return std::make_pair(bookkeepingCollection.end(), false);
    }

    void updateMetaInformation() {
        auto& info = BlockWrapper<TSubGraph>::metaInformation();

        auto fillMetaInformation = [](property_map& dest, auto& bookkeepingCollection) {
            std::string              previousUniqueName;
            std::vector<std::string> collectedPorts;
            for (const auto& [blockUniqueName, portName] : bookkeepingCollection) {
                if (previousUniqueName != blockUniqueName && !collectedPorts.empty()) {
                    dest[previousUniqueName] = std::move(collectedPorts);
                    collectedPorts.clear();
                }
                collectedPorts.push_back(portName);
                previousUniqueName = blockUniqueName;
            }
            if (!collectedPorts.empty()) {
                dest[previousUniqueName] = std::move(collectedPorts);
                collectedPorts.clear();
            }
        };

        property_map exportedInputPorts, exportedOutputPorts;
        fillMetaInformation(exportedInputPorts, _exportedInputPortsForBlock);
        fillMetaInformation(exportedOutputPorts, _exportedOutputPortsForBlock);

        info["exportedInputPorts"]  = std::move(exportedInputPorts);
        info["exportedOutputPorts"] = std::move(exportedOutputPorts);
    }
};

class Graph : public gr::Block<Graph> {
private:
    std::shared_ptr<gr::Sequence>                     _progress     = std::make_shared<gr::Sequence>();
    std::shared_ptr<gr::thread_pool::BasicThreadPool> _ioThreadPool = std::make_shared<gr::thread_pool::BasicThreadPool>("graph_thread_pool", gr::thread_pool::TaskType::IO_BOUND, 2UZ, std::numeric_limits<uint32_t>::max());
    std::atomic_bool                                  _topologyChanged{false};
    std::vector<Edge>                                 _edges;
    std::vector<std::unique_ptr<BlockModel>>          _blocks;

    gr::PluginLoader* _pluginLoader = std::addressof(gr::globalPluginLoader());

    template<typename TBlock>
    std::unique_ptr<BlockModel>& findBlock(TBlock& what) {
        static_assert(!std::is_pointer_v<std::remove_cvref_t<TBlock>>);
        auto it = [&, this] {
            if constexpr (std::is_same_v<TBlock, BlockModel>) {
                return std::find_if(_blocks.begin(), _blocks.end(), [&](const auto& block) { return block.get() == &what; });
            } else {
                return std::find_if(_blocks.begin(), _blocks.end(), [&](const auto& block) { return block->raw() == &what; });
            }
        }();

        if (it == _blocks.end()) {
            throw std::runtime_error(fmt::format("No such block in this graph"));
        }
        return *it;
    }

    // Just a dummy class that stores the graph and the source block and port
    // to be able to split the connection into two separate calls
    // connect(source) and .to(destination)
    template<typename Source, typename SourcePort, std::size_t sourcePortIndex = 1UZ, std::size_t sourcePortSubIndex = meta::invalid_index>
    struct SourceConnector {
        Graph&      self;
        Source&     sourceBlockRaw;
        SourcePort& sourcePortOrCollectionRaw;

        std::size_t  minBufferSize = undefined_size;
        std::int32_t weight        = graph::defaultWeight;
        std::string  edgeName      = graph::defaultEdgeName;

        SourceConnector(Graph& _self, Source& _source, SourcePort& _port, std::size_t _minBufferSize, std::int32_t _weight, std::string _edgeName) //
            : self(_self), sourceBlockRaw(_source), sourcePortOrCollectionRaw(_port), minBufferSize(_minBufferSize), weight(_weight), edgeName(_edgeName) {}

        SourceConnector(const SourceConnector&)            = delete;
        SourceConnector(SourceConnector&&)                 = delete;
        SourceConnector& operator=(const SourceConnector&) = delete;
        SourceConnector& operator=(SourceConnector&&)      = delete;

        static_assert(std::is_same_v<SourcePort, gr::Message> || traits::port::is_port_v<SourcePort> || (sourcePortSubIndex != meta::invalid_index), "When we have a collection of ports, we need to have an index to access the desired port in the collection");

    private:
        template<typename Destination, typename DestinationPort, std::size_t destinationPortIndex = meta::invalid_index, std::size_t destinationPortSubIndex = meta::invalid_index>
        [[nodiscard]] constexpr ConnectionResult to(Destination& destinationBlockRaw, DestinationPort& destinationPortOrCollectionRaw) {
            auto* sourceBlock      = self.findBlock(sourceBlockRaw).get();
            auto* destinationBlock = self.findBlock(destinationBlockRaw).get();

            if (sourceBlock == nullptr || destinationBlock == nullptr) {
                fmt::print("Source {} and/or destination {} do not belong to this graph\n", sourceBlockRaw.name, destinationBlockRaw.name);
                return ConnectionResult::FAILED;
            }

            auto* sourcePort = [&] {
                if constexpr (traits::port::is_port_v<SourcePort>) {
                    return &sourcePortOrCollectionRaw;
                } else {
                    return &sourcePortOrCollectionRaw[sourcePortSubIndex];
                }
            }();

            auto* destinationPort = [&] {
                if constexpr (traits::port::is_port_v<DestinationPort>) {
                    return &destinationPortOrCollectionRaw;
                } else {
                    return &destinationPortOrCollectionRaw[destinationPortSubIndex];
                }
            }();

            if constexpr (!std::is_same_v<typename std::remove_pointer_t<decltype(destinationPort)>::value_type, typename std::remove_pointer_t<decltype(sourcePort)>::value_type>) {
                meta::print_types<meta::message_type<"The source port type needs to match the sink port type">, typename std::remove_pointer_t<decltype(destinationPort)>::value_type, typename std::remove_pointer_t<decltype(sourcePort)>::value_type>{};
            }

            const bool        isArithmeticLike       = sourcePort->kIsArithmeticLikeValueType;
            const std::size_t sanitizedMinBufferSize = minBufferSize == undefined_size ? graph::defaultMinBufferSize(isArithmeticLike) : minBufferSize;
            self._edges.emplace_back(sourceBlock, PortDefinition{sourcePortIndex, sourcePortSubIndex}, destinationBlock, PortDefinition{destinationPortIndex, destinationPortSubIndex}, sanitizedMinBufferSize, weight, std::move(edgeName));

            return ConnectionResult::SUCCESS;
        }

    public:
        // connect using the port index

        template<std::size_t destinationPortIndex, std::size_t destinationPortSubIndex, typename Destination>
        [[nodiscard]] auto to_internal(Destination& destination) {
            auto& destinationPort = inputPort<destinationPortIndex, PortType::ANY>(&destination);
            return to<Destination, std::remove_cvref_t<decltype(destinationPort)>, destinationPortIndex, destinationPortSubIndex>(destination, destinationPort);
        }

        template<std::size_t destinationPortIndex, std::size_t destinationPortSubIndex, typename Destination>
        [[nodiscard, deprecated("For internal use only, the one with the port name should be used")]] auto to(Destination& destination) {
            return to_internal<destinationPortIndex, destinationPortSubIndex, Destination>(destination);
        }

        template<std::size_t destinationPortIndex, typename Destination>
        [[nodiscard]] auto to(Destination& destination) {
            if constexpr (destinationPortIndex == gr::meta::default_message_port_index) {
                return to<Destination, decltype(destination.msgIn)>(destination, destination.msgIn);

            } else {
                return to<destinationPortIndex, meta::invalid_index, Destination>(destination);
            }
        }

        // connect using the port name

        template<fixed_string destinationPortName, std::size_t destinationPortSubIndex, typename Destination>
        [[nodiscard]] constexpr auto to(Destination& destination) {
            using destination_input_ports              = typename traits::block::all_input_ports<Destination>;
            constexpr std::size_t destinationPortIndex = meta::indexForName<destinationPortName, destination_input_ports>();
            if constexpr (destinationPortIndex == meta::invalid_index) {
                meta::print_types<meta::message_type<"There is no input port with the specified name in this destination block">, Destination, meta::message_type<destinationPortName>, meta::message_type<"These are the known names:">, traits::block::all_input_port_names<Destination>, meta::message_type<"Full ports info:">, destination_input_ports> port_not_found_error{};
            }
            return to_internal<destinationPortIndex, destinationPortSubIndex, Destination>(destination);
        }

        template<fixed_string destinationPortName, typename Destination>
        [[nodiscard]] constexpr auto to(Destination& destination) {
            return to<destinationPortName, meta::invalid_index, Destination>(destination);
        }
    };

public:
    GR_MAKE_REFLECTABLE(Graph);

    constexpr static block::Category blockCategory = block::Category::TransparentBlockGroup;

    Graph(property_map settings = {}) : gr::Block<Graph>(std::move(settings)) {
        _blocks.reserve(100); // TODO: remove
        propertyCallbacks[graph::property::kEmplaceBlock]       = std::mem_fn(&Graph::propertyCallbackEmplaceBlock);
        propertyCallbacks[graph::property::kRemoveBlock]        = std::mem_fn(&Graph::propertyCallbackRemoveBlock);
        propertyCallbacks[graph::property::kInspectBlock]       = std::mem_fn(&Graph::propertyCallbackInspectBlock);
        propertyCallbacks[graph::property::kReplaceBlock]       = std::mem_fn(&Graph::propertyCallbackReplaceBlock);
        propertyCallbacks[graph::property::kEmplaceEdge]        = std::mem_fn(&Graph::propertyCallbackEmplaceEdge);
        propertyCallbacks[graph::property::kRemoveEdge]         = std::mem_fn(&Graph::propertyCallbackRemoveEdge);
        propertyCallbacks[graph::property::kGraphInspect]       = std::mem_fn(&Graph::propertyCallbackGraphInspect);
        propertyCallbacks[graph::property::kGraphGRC]           = std::mem_fn(&Graph::propertyCallbackGraphGRC);
        propertyCallbacks[graph::property::kRegistryBlockTypes] = std::mem_fn(&Graph::propertyCallbackRegistryBlockTypes);
    }

    Graph(gr::PluginLoader& pluginLoader) : Graph(property_map{}) { _pluginLoader = std::addressof(pluginLoader); }

    Graph(Graph&)            = delete; // there can be only one owner of Graph
    Graph& operator=(Graph&) = delete; // there can be only one owner of Graph
    Graph(Graph&& other) noexcept : gr::Block<Graph>(std::move(other)) { *this = std::move(other); }
    Graph& operator=(Graph&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        _progress     = std::move(other._progress);
        _ioThreadPool = std::move(other._ioThreadPool);
        _topologyChanged.store(other._topologyChanged.load(std::memory_order_acquire), std::memory_order_release);
        _edges  = std::move(other._edges);
        _blocks = std::move(other._blocks);

        return *this;
    }

    void               setTopologyChanged() noexcept { _topologyChanged.store(true, std::memory_order_release); }
    [[nodiscard]] bool hasTopologyChanged() const noexcept { return _topologyChanged; }
    void               ackTopologyChange() noexcept { _topologyChanged.store(false, std::memory_order_release); }

    [[nodiscard]] std::span<std::unique_ptr<BlockModel>> blocks() noexcept { return {_blocks}; }
    [[nodiscard]] std::span<Edge>                        edges() noexcept { return {_edges}; }

    /**
     * @return atomic sequence counter that indicates if any block could process some data or messages
     */
    [[nodiscard]] const Sequence& progress() noexcept { return *_progress.get(); }

    BlockModel& addBlock(std::unique_ptr<BlockModel> block, bool doEmitMessage = true) {
        auto& newBlock = _blocks.emplace_back(std::move(block));
        newBlock->init(_progress, _ioThreadPool);
        // TODO: Should we connectChildMessagePorts for these blocks as well?
        setTopologyChanged();
        if (doEmitMessage) {
            this->emitMessage(graph::property::kBlockEmplaced, serializeBlock(newBlock.get()));
        }
        return *newBlock.get();
    }

    template<BlockLike TBlock>
    requires std::is_constructible_v<TBlock, property_map>
    auto& emplaceBlock(gr::property_map initialSettings = gr::property_map()) {
        static_assert(std::is_same_v<TBlock, std::remove_reference_t<TBlock>>);
        auto& newBlock    = _blocks.emplace_back(std::make_unique<BlockWrapper<TBlock>>(std::move(initialSettings)));
        auto* rawBlockRef = static_cast<TBlock*>(newBlock->raw());
        rawBlockRef->init(_progress, _ioThreadPool);
        setTopologyChanged();
        this->emitMessage(graph::property::kBlockEmplaced, serializeBlock(newBlock.get()));
        return *rawBlockRef;
    }

    [[maybe_unused]] auto& emplaceBlock(std::string_view type, property_map initialSettings) {
        if (auto block_load = _pluginLoader->instantiate(type, std::move(initialSettings)); block_load) {
            setTopologyChanged();
            auto& newBlock = addBlock(std::move(block_load), false); // false == do not emit message

            this->emitMessage(graph::property::kBlockEmplaced, serializeBlock(std::addressof(newBlock)));

            return newBlock;
        }
        throw gr::exception(fmt::format("Can not create block {}", type));
    }

    static property_map serializeEdge(const auto& edge) {
        property_map result;
        auto         serializePortDefinition = [&](const std::string& key, const PortDefinition& portDefinition) {
            std::visit(meta::overloaded( //
                           [&](const PortDefinition::IndexBased& definition) {
                               result[key + ".topLevel"] = definition.topLevel;
                               result[key + ".subIndex"] = definition.subIndex;
                           }, //
                           [&](const PortDefinition::StringBased& definition) { result[key] = definition.name; }),
                        portDefinition.definition);
        };

        result["sourceBlock"s] = std::string(edge.sourceBlock().uniqueName());
        serializePortDefinition("sourcePort"s, edge.sourcePortDefinition());
        result["destinationBlock"s] = std::string(edge.destinationBlock().uniqueName());
        serializePortDefinition("destinationPort"s, edge.destinationPortDefinition());

        result["weight"s]        = edge.weight();
        result["minBufferSize"s] = edge.minBufferSize();
        result["edgeName"s]      = std::string(edge.name());

        result["bufferSize"s] = edge.bufferSize();
        result["edgeState"s]  = std::string(magic_enum::enum_name(edge.state()));
        result["nReaders"s]   = edge.nReaders();
        result["nWriters"s]   = edge.nWriters();
        result["type"s]       = std::string(magic_enum::enum_name(edge.edgeType()));

        return result;
    };

    static property_map serializeBlock(BlockModel* block) {
        auto serializePortOrCollection = [](const auto& portOrCollection) {
            // clang-format off
            // TODO: Type names can be mangled. We need proper type names...
            return std::visit(meta::overloaded{
                    [](const gr::DynamicPort& port) {
                        return property_map{
                            {"name"s, std::string(port.name)},
                            {"type"s, port.typeName()}
                        };
                    },
                    [](const BlockModel::NamedPortCollection& namedCollection) {
                        return property_map{
                            {"name"s, std::string(namedCollection.name)},
                            {"size"s, namedCollection.ports.size()},
                            {"type"s, namedCollection.ports.empty() ? std::string() : std::string(namedCollection.ports[0].typeName()) }
                        };
                    }},
                portOrCollection);
            // clang-format on
        };

        property_map result;
        result["name"s]            = std::string(block->name());
        result["uniqueName"s]      = std::string(block->uniqueName());
        result["typeName"s]        = std::string(block->typeName());
        result["isBlocking"s]      = block->isBlocking();
        result["metaInformation"s] = block->metaInformation();
        result["blockCategory"s]   = std::string(magic_enum::enum_name(block->blockCategory()));
        result["uiCategory"s]      = std::string(magic_enum::enum_name(block->uiCategory()));
        result["settings"s]        = block->settings().getStored().value_or(property_map());

        property_map inputPorts;
        for (auto& portOrCollection : block->dynamicInputPorts()) {
            inputPorts[BlockModel::portName(portOrCollection)] = serializePortOrCollection(portOrCollection);
        }
        result["inputPorts"] = std::move(inputPorts);

        property_map outputPorts;
        for (auto& portOrCollection : block->dynamicOutputPorts()) {
            outputPorts[BlockModel::portName(portOrCollection)] = serializePortOrCollection(portOrCollection);
        }
        result["outputPorts"] = std::move(outputPorts);

        if (block->blockCategory() != block::Category::NormalBlock) {
            property_map serializedChildren;
            for (const auto& child : block->blocks()) {
                serializedChildren[std::string(child->uniqueName())] = serializeBlock(child.get());
            }
            result["children"] = std::move(serializedChildren);
        }

        property_map serializedEdges;
        std::size_t  index = 0UZ;
        for (const auto& edge : block->edges()) {
            serializedEdges[std::to_string(index)] = serializeEdge(edge);
            index++;
        }
        result["edges"] = std::move(serializedEdges);

        return result;
    }

    std::optional<Message> propertyCallbackEmplaceBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == graph::property::kEmplaceBlock);
        using namespace std::string_literals;
        const auto&         data       = message.data.value();
        const std::string&  type       = std::get<std::string>(data.at("type"s));
        const property_map& properties = [&] {
            if (auto it = data.find("properties"s); it != data.end()) {
                return std::get<property_map>(it->second);
            } else {
                return property_map{};
            }
        }();

        emplaceBlock(type, properties);

        // Message is sent as a reaction to emplaceBlock, no need for a separate one
        return {};
    }

    std::optional<Message> propertyCallbackInspectBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == graph::property::kInspectBlock);
        using namespace std::string_literals;
        const auto&        data       = message.data.value();
        const std::string& uniqueName = std::get<std::string>(data.at("uniqueName"s));
        using namespace std::string_literals;

        auto it = std::ranges::find_if(_blocks, [&uniqueName](const auto& block) { return block->uniqueName() == uniqueName; });
        if (it == _blocks.end()) {
            throw gr::exception(fmt::format("Block {} was not found in {}", uniqueName, this->unique_name));
        }

        gr::Message reply;
        reply.endpoint = graph::property::kBlockInspected;
        reply.data     = serializeBlock(it->get());
        return {reply};
    }

    std::optional<Message> propertyCallbackRemoveBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == graph::property::kRemoveBlock);
        using namespace std::string_literals;
        const auto&        data       = message.data.value();
        const std::string& uniqueName = std::get<std::string>(data.at("uniqueName"s));
        auto               it         = std::ranges::find_if(_blocks, [&uniqueName](const auto& block) { return block->uniqueName() == uniqueName; });

        if (it == _blocks.end()) {
            throw gr::exception(fmt::format("Block {} was not found in {}", uniqueName, this->unique_name));
        }

        std::erase_if(_edges, [&it](const Edge& edge) { //
            return std::addressof(edge.sourceBlock()) == it->get() || std::addressof(edge.destinationBlock()) == it->get();
        });
        _blocks.erase(it);
        message.endpoint = graph::property::kBlockRemoved;

        return {message};
    }

    std::optional<Message> propertyCallbackReplaceBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == graph::property::kReplaceBlock);
        using namespace std::string_literals;
        const auto&         data       = message.data.value();
        const std::string&  uniqueName = std::get<std::string>(data.at("uniqueName"s));
        const std::string&  type       = std::get<std::string>(data.at("type"s));
        const property_map& properties = [&] {
            if (auto it = data.find("properties"s); it != data.end()) {
                return std::get<property_map>(it->second);
            } else {
                return property_map{};
            }
        }();

        auto it = std::ranges::find_if(_blocks, [&uniqueName](const auto& block) { return block->uniqueName() == uniqueName; });
        if (it == _blocks.end()) {
            throw gr::exception(fmt::format("Block {} was not found in {}", uniqueName, this->unique_name));
        }

        auto newBlock    = gr::globalPluginLoader().instantiate(type, properties);
        auto newBlockRaw = newBlock.get();
        if (!newBlock) {
            throw gr::exception(fmt::format("Can not create block {}", type));
        }

        addBlock(std::move(newBlock), false); // false == do not emit message

        BlockModel* oldBlock = it->get();
        for (auto& edge : _edges) {
            if (edge._sourceBlock == oldBlock) {
                edge._sourceBlock = newBlockRaw;
            }

            if (edge._destinationBlock == oldBlock) {
                edge._destinationBlock = newBlockRaw;
            }
        }
        _blocks.erase(it);

        std::optional<Message> result = gr::Message{};
        result->endpoint              = graph::property::kBlockReplaced;
        result->data                  = serializeBlock(newBlockRaw);

        (*result->data)["replacedBlockUniqueName"s] = uniqueName;

        return result;
    }

    std::optional<Message> propertyCallbackEmplaceEdge([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == graph::property::kEmplaceEdge);
        using namespace std::string_literals;
        const auto&                         data             = message.data.value();
        const std::string&                  sourceBlock      = std::get<std::string>(data.at("sourceBlock"s));
        const std::string&                  sourcePort       = std::get<std::string>(data.at("sourcePort"s));
        const std::string&                  destinationBlock = std::get<std::string>(data.at("destinationBlock"s));
        const std::string&                  destinationPort  = std::get<std::string>(data.at("destinationPort"s));
        [[maybe_unused]] const std::size_t  minBufferSize    = std::get<gr::Size_t>(data.at("minBufferSize"s));
        [[maybe_unused]] const std::int32_t weight           = std::get<std::int32_t>(data.at("weight"s));
        const std::string                   edgeName         = std::get<std::string>(data.at("edgeName"s));

        auto sourceBlockIt = std::ranges::find_if(_blocks, [&sourceBlock](const auto& block) { return block->uniqueName() == sourceBlock; });
        if (sourceBlockIt == _blocks.end()) {
            throw gr::exception(fmt::format("Block {} was not found in {}", sourceBlock, this->unique_name));
        }

        auto destinationBlockIt = std::ranges::find_if(_blocks, [&destinationBlock](const auto& block) { return block->uniqueName() == destinationBlock; });
        if (destinationBlockIt == _blocks.end()) {
            throw gr::exception(fmt::format("Block {} was not found in {}", destinationBlock, this->unique_name));
        }

        auto& sourcePortRef      = (*sourceBlockIt)->dynamicOutputPort(sourcePort);
        auto& destinationPortRef = (*destinationBlockIt)->dynamicInputPort(destinationPort);

        if (sourcePortRef.typeName() != destinationPortRef.typeName()) {
            throw gr::exception(fmt::format("{}.{} can not be connected to {}.{} -- different types", sourceBlock, sourcePort, destinationBlock, destinationPort));
        }

        auto connectionResult = sourcePortRef.connect(destinationPortRef);

        if (connectionResult != ConnectionResult::SUCCESS) {
            throw gr::exception(fmt::format("{}.{} can not be connected to {}.{}", sourceBlock, sourcePort, destinationBlock, destinationPort));
        }

        const bool        isArithmeticLike       = sourcePortRef.portInfo().isValueTypeArithmeticLike;
        const std::size_t sanitizedMinBufferSize = minBufferSize == undefined_size ? graph::defaultMinBufferSize(isArithmeticLike) : minBufferSize;
        _edges.emplace_back(sourceBlockIt->get(), sourcePort, destinationBlockIt->get(), destinationPort, sanitizedMinBufferSize, weight, edgeName);

        message.endpoint = graph::property::kEdgeEmplaced;
        return message;
    }

    std::optional<Message> propertyCallbackRemoveEdge([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == graph::property::kRemoveEdge);
        using namespace std::string_literals;
        const auto&        data        = message.data.value();
        const std::string& sourceBlock = std::get<std::string>(data.at("sourceBlock"s));
        const std::string& sourcePort  = std::get<std::string>(data.at("sourcePort"s));

        auto sourceBlockIt = std::ranges::find_if(_blocks, [&sourceBlock](const auto& block) { return block->uniqueName() == sourceBlock; });
        if (sourceBlockIt == _blocks.end()) {
            throw gr::exception(fmt::format("Block {} was not found in {}", sourceBlock, this->unique_name));
        }

        auto& sourcePortRef = (*sourceBlockIt)->dynamicOutputPort(sourcePort);

        if (sourcePortRef.disconnect() == ConnectionResult::FAILED) {
            throw gr::exception(fmt::format("Block {} sourcePortRef could not be disconnected {}", sourceBlock, this->unique_name));
        }
        message.endpoint = graph::property::kEdgeRemoved;
        return message;
    }

    std::optional<Message> propertyCallbackGraphInspect([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == graph::property::kGraphInspect);
        message.data = [&] {
            property_map result;
            result["name"s]          = std::string(name);
            result["uniqueName"s]    = std::string(unique_name);
            result["blockCategory"s] = std::string(magic_enum::enum_name(blockCategory));

            property_map serializedChildren;
            for (const auto& child : blocks()) {
                serializedChildren[std::string(child->uniqueName())] = serializeBlock(child.get());
            }
            result["children"] = std::move(serializedChildren);

            property_map serializedEdges;
            std::size_t  index = 0UZ;
            for (const auto& edge : edges()) {
                serializedEdges[std::to_string(index)] = serializeEdge(edge);
                index++;
            }
            result["edges"] = std::move(serializedEdges);
            return result;
        }();

        message.endpoint = graph::property::kGraphInspected;
        return message;
    }

    std::optional<Message> propertyCallbackGraphGRC([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == graph::property::kGraphGRC);

        if (message.cmd == message::Command::Get) {
            message.data = property_map{{"value", gr::saveGrc(*_pluginLoader, *this)}};
        } else if (message.cmd == message::Command::Set) {
            const auto& data        = message.data.value();
            auto        yamlContent = std::get<std::string>(data.at("value"s));
            *this                   = gr::loadGrc(*_pluginLoader, yamlContent);

            setTopologyChanged();

        } else {
            throw gr::exception(fmt::format("Unexpected command type {}", message.cmd));
        }

        return message;
    }

    std::optional<Message> propertyCallbackRegistryBlockTypes([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == graph::property::kRegistryBlockTypes);
        message.data = property_map{{"types", _pluginLoader->knownBlocks()}};
        return message;
    }

    // connect using the port index
    template<std::size_t sourcePortIndex, std::size_t sourcePortSubIndex, typename Source>
    [[nodiscard]] auto connectInternal(Source& source, std::size_t minBufferSize = undefined_size, std::int32_t weight = graph::defaultWeight, std::string edgeName = graph::defaultEdgeName, [[maybe_unused]] std::source_location location = std::source_location::current()) {
        auto& port_or_collection = outputPort<sourcePortIndex, PortType::ANY>(&source);
        return SourceConnector<Source, std::remove_cvref_t<decltype(port_or_collection)>, sourcePortIndex, sourcePortSubIndex>(*this, source, port_or_collection, minBufferSize, weight, edgeName);
    }

    template<std::size_t sourcePortIndex, std::size_t sourcePortSubIndex, typename Source>
    [[nodiscard, deprecated("The connect with the port name should be used")]] auto connect(Source& source, std::size_t minBufferSize = undefined_size, std::int32_t weight = graph::defaultWeight, std::string edgeName = graph::defaultEdgeName, std::source_location location = std::source_location::current()) {
        return connectInternal<sourcePortIndex, sourcePortSubIndex, Source>(source, minBufferSize, weight, edgeName, location);
    }

    template<std::size_t sourcePortIndex, typename Source>
    [[nodiscard]] auto connect(Source& source, std::size_t minBufferSize = undefined_size, std::int32_t weight = graph::defaultWeight, std::string edgeName = graph::defaultEdgeName, [[maybe_unused]] std::source_location location = std::source_location::current()) {
        if constexpr (sourcePortIndex == meta::default_message_port_index) {
            return SourceConnector<Source, decltype(source.msgOut), meta::invalid_index, meta::invalid_index>(*this, source, source.msgOut, minBufferSize, weight, edgeName);
        } else {
            return connect<sourcePortIndex, meta::invalid_index, Source>(source, minBufferSize, weight, edgeName, location);
        }
    }

    // connect using the port name

    template<fixed_string sourcePortName, std::size_t sourcePortSubIndex, typename Source>
    [[nodiscard]] auto connect(Source& source, std::size_t minBufferSize = undefined_size, std::int32_t weight = graph::defaultWeight, std::string edgeName = graph::defaultEdgeName, std::source_location location = std::source_location::current()) {
        using source_output_ports             = typename traits::block::all_output_ports<Source>;
        constexpr std::size_t sourcePortIndex = meta::indexForName<sourcePortName, source_output_ports>();
        if constexpr (sourcePortIndex == meta::invalid_index) {
            meta::print_types<meta::message_type<"There is no output port with the specified name in this source block">, Source, meta::message_type<sourcePortName>, meta::message_type<"These are the known names:">, traits::block::all_output_port_names<Source>, meta::message_type<"Full ports info:">, source_output_ports> port_not_found_error{};
        }
        return connectInternal<sourcePortIndex, sourcePortSubIndex, Source>(source, minBufferSize, weight, edgeName, location);
    }

    template<fixed_string sourcePortName, typename Source>
    [[nodiscard]] auto connect(Source& source, std::size_t minBufferSize = undefined_size, std::int32_t weight = graph::defaultWeight, std::string edgeName = graph::defaultEdgeName, std::source_location location = std::source_location::current()) {
        return connect<sourcePortName, meta::invalid_index, Source>(source, minBufferSize, weight, edgeName, location);
    }

    // dynamic/runtime connections

    template<typename Source, typename Destination>
    requires(!std::is_pointer_v<std::remove_cvref_t<Source>> && !std::is_pointer_v<std::remove_cvref_t<Destination>>)
    ConnectionResult connect(Source& sourceBlockRaw, PortDefinition sourcePortDefinition, Destination& destinationBlockRaw, PortDefinition destinationPortDefinition, std::size_t minBufferSize = undefined_size, std::int32_t weight = graph::defaultWeight, std::string edgeName = graph::defaultEdgeName, std::source_location location = std::source_location::current()) {
        auto findBlockNoexcept = [this]<typename Block>(Block&& blockRaw) noexcept -> BlockModel* {
            try {
                return this->findBlock(std::forward<Block>(blockRaw)).get();
            } catch (...) {
                return nullptr;
            }
        };
        auto* sourceBlock      = findBlockNoexcept(sourceBlockRaw);
        auto* destinationBlock = findBlockNoexcept(destinationBlockRaw);

        if (sourceBlock == nullptr || destinationBlock == nullptr) {
            return ConnectionResult::FAILED;
        }

        const auto&       sourcePort             = sourceBlock->dynamicOutputPort(sourcePortDefinition, location);
        const bool        isArithmeticLike       = sourcePort.portInfo().isValueTypeArithmeticLike;
        const std::size_t sanitizedMinBufferSize = minBufferSize == undefined_size ? graph::defaultMinBufferSize(isArithmeticLike) : minBufferSize;
        _edges.emplace_back(sourceBlock, sourcePortDefinition, destinationBlock, destinationPortDefinition, sanitizedMinBufferSize, weight, std::move(edgeName));
        return ConnectionResult::SUCCESS;
    }

    using Block<Graph>::processMessages;

    template<typename Anything>
    void processMessages(MsgPortInFromChildren& /*port*/, std::span<const Anything> /*input*/) {
        static_assert(meta::always_false<Anything>, "This is not called, children are processed in processScheduledMessages");
    }

    Edge::EdgeState applyEdgeConnection(Edge& edge) {
        try {
            auto& sourcePort      = edge._sourceBlock->dynamicOutputPort(edge._sourcePortDefinition);
            auto& destinationPort = edge._destinationBlock->dynamicInputPort(edge._destinationPortDefinition);

            if (sourcePort.typeName() != destinationPort.typeName()) {
                edge._state = Edge::EdgeState::IncompatiblePorts;
            } else {
                const bool hasConnectedEdges = std::ranges::any_of(_edges, [&](const Edge& e) { return edge.hasSameSourcePort(e) && e._state == Edge::EdgeState::Connected; });
                bool       resizeResult      = true;
                if (!hasConnectedEdges) {
                    const std::size_t bufferSize = calculateStreamBufferSize(edge);
                    resizeResult                 = sourcePort.resizeBuffer(bufferSize) == ConnectionResult::SUCCESS;
                }

                const bool connectionResult = sourcePort.connect(destinationPort) == ConnectionResult::SUCCESS;
                edge._state                 = connectionResult && resizeResult ? Edge::EdgeState::Connected : Edge::EdgeState::ErrorConnecting;
                edge._actualBufferSize      = sourcePort.bufferSize();
                edge._edgeType              = sourcePort.type();
                edge._sourcePort            = std::addressof(sourcePort);
                edge._destinationPort       = std::addressof(destinationPort);
            }
        } catch (gr::exception& e) {
            fmt::println("applyEdgeConnection({}): {}", edge, e.what());
            edge._state = Edge::EdgeState::PortNotFound;
        } catch (...) {
            fmt::println("applyEdgeConnection({}): unknown exception", edge);
            edge._state = Edge::EdgeState::PortNotFound;
        }

        return edge._state;
    }

    std::size_t calculateStreamBufferSize(const Edge& refEdge) const {
        // if one of the edge with the same source port is already connected -> use already existing buffer size
        for (const Edge& e : _edges) {
            if (refEdge.hasSameSourcePort(e) && e._state == Edge::EdgeState::Connected) {
                return e.bufferSize();
            }
        }

        std::size_t maxSize = 0UZ;
        forEachEdge([&](const Edge& e) {
            if (refEdge.hasSameSourcePort(e)) {
                std::size_t minBufferSize = e.minBufferSize();
                if (minBufferSize != undefined_size) {
                    maxSize = std::max(maxSize, e.minBufferSize());
                }
            }
        });
        // assert(maxSize != 0UZ);
        assert(maxSize != undefined_size);
        return maxSize;
    }

    void disconnectAllEdges() {
        for (auto& block : _blocks) {
            block->initDynamicPorts();

            auto disconnectAll = [](auto& ports) {
                for (auto& port : ports) {
                    std::ignore = std::visit([](auto& portOrCollection) { return portOrCollection.disconnect(); }, port);
                }
            };

            disconnectAll(block->dynamicInputPorts());
            disconnectAll(block->dynamicOutputPorts());
        }

        for (auto& edge : _edges) {
            edge._state = Edge::EdgeState::WaitingToBeConnected;
        }
    }

    bool reconnectAllEdges() {
        disconnectAllEdges();
        return connectPendingEdges();
    }

    bool connectPendingEdges() {
        bool allConnected = true;
        for (auto& edge : _edges) {
            if (edge.state() == Edge::EdgeState::WaitingToBeConnected) {
                applyEdgeConnection(edge);
                const bool wasConnected = edge.state() == Edge::EdgeState::Connected;
                if (!wasConnected) {
                    fmt::print("Edge could not be connected {}\n", edge);
                }
                allConnected = allConnected && wasConnected;
            }
        }
        return allConnected;
    }

    template<std::invocable<BlockModel&> F>
    void forEachBlockMutable(F&& f) {
        std::ranges::for_each(_blocks, [f](auto& block_ptr) { std::invoke(f, *block_ptr.get()); });
    }

    template<std::invocable<Edge&> F>
    void forEachEdgeMutable(F&& f) {
        std::ranges::for_each(_edges, f);
    }

    template<std::invocable<const BlockModel&> F>
    void forEachBlock(F&& f) const {
        std::ranges::for_each(_blocks, [f](auto& block_ptr) { std::invoke(f, std::as_const(*block_ptr.get())); });
    }

    template<std::invocable<const Edge&> F>
    void forEachEdge(F&& f) const {
        std::ranges::for_each(_edges, f);
    }
};

static_assert(BlockLike<Graph>);

/*******************************************************************************************************/
/**************************** begin of SIMD-Merged Graph Implementation ********************************/
/*******************************************************************************************************/

template<typename TBlock>
concept SourceBlockLike = traits::block::can_processOne<TBlock> and traits::block::template stream_output_port_types<TBlock>::size > 0;

static_assert(not SourceBlockLike<int>);

template<typename TBlock>
concept SinkBlockLike = traits::block::can_processOne<TBlock> and traits::block::template stream_input_port_types<TBlock>::size > 0;

static_assert(not SinkBlockLike<int>);

template<typename TDesc>
struct to_left_descriptor : TDesc {
    template<typename TBlock>
    static constexpr decltype(auto) getPortObject(TBlock&& obj) {
        return TDesc::getPortObject(obj.left);
    }
};

template<typename TDesc>
struct to_right_descriptor : TDesc {
    template<typename TBlock>
    static constexpr decltype(auto) getPortObject(TBlock&& obj) {
        return TDesc::getPortObject(obj.right);
    }
};

/**
 * Concepts and class for Merging Blocks to Sub-Graph Functionality
 *
 * This code provides a way to merge blocks of processing units in a flow-graph for efficient computation.
 * The merging occurs at compile-time, enabling the execution performance to be much better than running
 * each constituent block individually.
 *
 * Concepts:
 *  - `SourceBlockLike`: Represents a source block with processing capability and at least one output port.
 *  - `SinkBlockLike`: Represents a sink block with processing capability and at least one input port.
 *
 * Key Features:
 *  - `MergedGraph` class: Combines a source and sink block into a new unit, connecting them via specified
 *    output and input port indices.
 *  - The merged block can be efficiently optimized at compile-time.
 *  - Each `MergedGraph` instance has a unique ID and name, aiding in debugging and identification.
 *  - The merging works seamlessly for blocks that have single or multiple output ports.
 *  - It allows for SIMD optimizations if the constituent blocks support it.
 *
 * Utility Functions:
 *  - `mergeByIndex()`: A utility function to merge two blocks based on specified port indices.
 *    It checks if the output port of the source block and the input port of the sink block have matching types.
 *
 * Examples:
 *  This enables you to create a flow-graph where you merge blocks to create optimized processing paths.
 *  Example usage can be found in the documentation of `mergeByIndex()`.
 *
 * Dependencies:
 *  - Relies on various traits and meta-programming utilities for type introspection and compile-time checks.
 *
 * Note:
 *  - The implementation of the actual processing logic (e.g., `processOne()`, `processOne_simd()`, etc.)
 *    and their SIMD variants is specific to the logic and capabilities of the blocks being merged.
 *
 * Limitations:
 *  - Currently, SIMD support for multiple output ports is not implemented.
 */

template<SourceBlockLike Left, SinkBlockLike Right, std::size_t OutId, std::size_t InId>
class MergedGraph<Left, Right, OutId, InId> : public Block<MergedGraph<Left, Right, OutId, InId>> {
    // FIXME: How do we refuse connection to a vector<Port>?
    static std::atomic_size_t _unique_id_counter;

    template<typename TDesc>
    friend struct to_right_descriptor;

    template<typename TDesc>
    friend struct to_left_descriptor;

public:
    using AllPorts = meta::concat<
        // Left:
        typename meta::concat<typename traits::block::all_port_descriptors<Left>::template filter<traits::port::is_message_port>, traits::block::stream_input_ports<Left>, meta::remove_at<OutId, traits::block::stream_output_ports<Left>>>::template transform<to_left_descriptor>,
        // Right:
        typename meta::concat<typename traits::block::all_port_descriptors<Right>::template filter<traits::port::is_message_port>, meta::remove_at<InId, traits::block::stream_input_ports<Right>>, traits::block::stream_output_ports<Right>>::template transform<to_right_descriptor>>;

    using InputPortTypes = typename AllPorts::template filter<traits::port::is_input_port, traits::port::is_stream_port>::template transform<traits::port::type>;

    using ReturnType = typename AllPorts::template filter<traits::port::is_output_port, traits::port::is_stream_port>::template transform<traits::port::type>::tuple_or_type;

    GR_MAKE_REFLECTABLE(MergedGraph);

    // TODO: Add a comment why a unique ID is necessary for merged blocks but not for all other blocks. (I.e. unique_id
    // already is a member of the Block base class, this is shadowing that member with a different value. No other block
    // does this.)
    const std::size_t unique_id   = _unique_id_counter++;
    const std::string unique_name = fmt::format("MergedGraph<{}:{},{}:{}>#{}", gr::meta::type_name<Left>(), OutId, gr::meta::type_name<Right>(), InId, unique_id);

private:
    // copy-paste from above, keep in sync
    using base = Block<MergedGraph<Left, Right, OutId, InId>>;

    Left  left;
    Right right;

    // merged_work_chunk_size, that's what friends are for
    friend base;

    template<typename, typename, std::size_t, std::size_t>
    friend class MergedGraph;

    // returns the minimum of all internal max_samples port template parameters
    static constexpr std::size_t merged_work_chunk_size() noexcept {
        constexpr std::size_t left_size = []() {
            if constexpr (requires {
                              { Left::merged_work_chunk_size() } -> std::same_as<std::size_t>;
                          }) {
                return Left::merged_work_chunk_size();
            } else {
                return std::dynamic_extent;
            }
        }();
        constexpr std::size_t right_size = []() {
            if constexpr (requires {
                              { Right::merged_work_chunk_size() } -> std::same_as<std::size_t>;
                          }) {
                return Right::merged_work_chunk_size();
            } else {
                return std::dynamic_extent;
            }
        }();
        return std::min({traits::block::stream_input_ports<Right>::template apply<traits::port::max_samples>::value, traits::block::stream_output_ports<Left>::template apply<traits::port::max_samples>::value, left_size, right_size});
    }

    template<std::size_t I>
    constexpr auto apply_left(auto&& input_tuple) noexcept {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return left.processOne(std::get<Is>(std::forward<decltype(input_tuple)>(input_tuple))...); }(std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    constexpr auto apply_right(auto&& input_tuple, auto&& tmp) noexcept {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset  = traits::block::stream_input_port_types<Left>::size;
            constexpr std::size_t second_offset = traits::block::stream_input_port_types<Left>::size + sizeof...(Is);
            static_assert(second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return right.processOne(std::get<first_offset + Is>(std::forward<decltype(input_tuple)>(input_tuple))..., std::forward<decltype(tmp)>(tmp), std::get<second_offset + Js>(input_tuple)...);
        }(std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    constexpr MergedGraph(Left l, Right r) : left(std::move(l)), right(std::move(r)) {}

    // if the left block (source) implements available_samples (a customization point), then pass the call through
    friend constexpr std::size_t available_samples(const MergedGraph& self) noexcept
    requires requires(const Left& l) {
        { available_samples(l) } -> std::same_as<std::size_t>;
    }
    {
        return available_samples(self.left);
    }

    template<meta::any_simd... Ts>
    requires traits::block::can_processOne_simd<Left> and traits::block::can_processOne_simd<Right>
    constexpr vir::simdize<ReturnType, (0, ..., Ts::size())> processOne(const Ts&... inputs) {
        static_assert(traits::block::stream_output_port_types<Left>::size == 1, "TODO: SIMD for multiple output ports not implemented yet");
        return apply_right<InId, traits::block::stream_input_port_types<Right>::size() - InId - 1>(std::tie(inputs...), apply_left<traits::block::stream_input_port_types<Left>::size()>(std::tie(inputs...)));
    }

    constexpr auto processOne_simd(auto N)
    requires traits::block::can_processOne_simd<Right>
    {
        if constexpr (requires(Left& l) {
                          { l.processOne_simd(N) };
                      }) {
            return right.processOne(left.processOne_simd(N));
        } else {
            using LeftResult = typename traits::block::stream_return_type<Left>;
            using V          = vir::simdize<LeftResult, N>;
            alignas(stdx::memory_alignment_v<V>) LeftResult tmp[V::size()];
            for (std::size_t i = 0UZ; i < V::size(); ++i) {
                tmp[i] = left.processOne();
            }
            return right.processOne(V(tmp, stdx::vector_aligned));
        }
    }

    template<typename... Ts>
    // Nicer error messages for the following would be good, but not at the expense of breaking can_processOne_simd.
    requires(InputPortTypes::template are_equal<std::remove_cvref_t<Ts>...>)
    constexpr ReturnType processOne(Ts&&... inputs) {
        // if (sizeof...(Ts) == 0) we could call `return processOne_simd(integral_constant<size_t, width>)`. But if
        // the caller expects to process *one* sample (no inputs for the caller to explicitly
        // request simd), and we process more, we risk inconsistencies.
        if constexpr (traits::block::stream_output_port_types<Left>::size == 1) {
            // only the result from the right block needs to be returned
            return apply_right<InId, traits::block::stream_input_port_types<Right>::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...), apply_left<traits::block::stream_input_port_types<Left>::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out  = apply_left<traits::block::stream_input_port_types<Left>::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, traits::block::stream_input_port_types<Right>::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...), std::move(std::get<OutId>(left_out)));

            if constexpr (traits::block::stream_output_port_types<Left>::size == 2 && traits::block::stream_output_port_types<Right>::size == 1) {
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)), std::move(right_out));

            } else if constexpr (traits::block::stream_output_port_types<Left>::size == 2) {
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))), std::move(right_out));

            } else if constexpr (traits::block::stream_output_port_types<Right>::size == 1) {
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) { return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(right_out)); }(std::make_index_sequence<OutId>(), std::make_index_sequence<traits::block::stream_output_port_types<Left>::size - OutId - 1>());

            } else {
                return [&]<std::size_t... Is, std::size_t... Js, std::size_t... Ks>(std::index_sequence<Is...>, std::index_sequence<Js...>, std::index_sequence<Ks...>) { return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(std::get<Ks>(right_out)...)); }(std::make_index_sequence<OutId>(), std::make_index_sequence<traits::block::stream_output_port_types<Left>::size - OutId - 1>(), std::make_index_sequence<Right::output_port_types::size>());
            }
        }
    } // end:: processOne

    // work::Result // TODO: ask Matthias if this is still needed or whether this can be simplified.
    // work(std::size_t requested_work) noexcept {
    //     return base::work(requested_work);
    // }
};

template<SourceBlockLike Left, SinkBlockLike Right, std::size_t OutId, std::size_t InId>
inline std::atomic_size_t MergedGraph<Left, Right, OutId, InId>::_unique_id_counter{0UZ};

/**
 * This methods can merge simple blocks that are defined via a single `auto processOne(..)` producing a
 * new `merged` node, bypassing the dynamic run-time buffers.
 * Since the merged node can be highly optimised during compile-time, it's execution performance is usually orders
 * of magnitude more efficient than executing a cascade of the same constituent blocks. See the benchmarks for details.
 * This function uses the connect-by-port-ID API.
 *
 * Example:
 * @code
 * // declare flow-graph: 2 x in -> adder -> scale-by-2 -> scale-by-minus1 -> output
 * auto merged = merge_by_index<0, 0>(scale<int, -1>(), merge_by_index<0, 0>(scale<int, 2>(), adder<int>()));
 *
 * // execute graph
 * std::array<int, 4> a = { 1, 2, 3, 4 };
 * std::array<int, 4> b = { 10, 10, 10, 10 };
 *
 * int                r = 0;
 * for (std::size_t i = 0; i < 4; ++i) {
 *     r += merged.processOne(a[i], b[i]);
 * }
 * @endcode
 */
template<std::size_t OutId, std::size_t InId, SourceBlockLike A, SinkBlockLike B>
constexpr auto mergeByIndex(A&& a, B&& b) -> MergedGraph<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    if constexpr (!std::is_same_v<typename traits::block::stream_output_port_types<std::remove_cvref_t<A>>::template at<OutId>, typename traits::block::stream_input_port_types<std::remove_cvref_t<B>>::template at<InId>>) {
        gr::meta::print_types<                                                                                                                                                                                          //
            gr::meta::message_type<"OUTPUT_PORTS_ARE:">,                                                                                                                                                                //
            typename traits::block::stream_output_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, OutId>, typename traits::block::stream_output_port_types<std::remove_cvref_t<A>>::template at<OutId>, //
            gr::meta::message_type<"INPUT_PORTS_ARE:">,                                                                                                                                                                 //
            typename traits::block::stream_input_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, InId>, typename traits::block::stream_input_port_types<std::remove_cvref_t<A>>::template at<InId>>{};
    }
    return {std::forward<A>(a), std::forward<B>(b)};
}

/**
 * This methods can merge simple blocks that are defined via a single `auto processOne(..)` producing a
 * new `merged` node, bypassing the dynamic run-time buffers.
 * Since the merged node can be highly optimised during compile-time, it's execution performance is usually orders
 * of magnitude more efficient than executing a cascade of the same constituent blocks. See the benchmarks for details.
 * This function uses the connect-by-port-name API.
 *
 * Example:
 * @code
 * // declare flow-graph: 2 x in -> adder -> scale-by-2 -> output
 * auto merged = merge<"scaled", "addend1">(scale<int, 2>(), adder<int>());
 *
 * // execute graph
 * std::array<int, 4> a = { 1, 2, 3, 4 };
 * std::array<int, 4> b = { 10, 10, 10, 10 };
 *
 * int                r = 0;
 * for (std::size_t i = 0; i < 4; ++i) {
 *     r += merged.processOne(a[i], b[i]);
 * }
 * @endcode
 */
template<meta::fixed_string OutName, meta::fixed_string InName, SourceBlockLike A, SinkBlockLike B>
constexpr auto merge(A&& a, B&& b) {
    constexpr int OutIdUnchecked = meta::indexForName<OutName, typename traits::block::stream_output_ports<A>>();
    constexpr int InIdUnchecked  = meta::indexForName<InName, typename traits::block::stream_input_ports<B>>();
    static_assert(OutIdUnchecked != -1);
    static_assert(InIdUnchecked != -1);
    constexpr auto OutId = static_cast<std::size_t>(OutIdUnchecked);
    constexpr auto InId  = static_cast<std::size_t>(InIdUnchecked);
    static_assert(std::same_as<typename traits::block::stream_output_port_types<std::remove_cvref_t<A>>::template at<OutId>, typename traits::block::stream_input_port_types<std::remove_cvref_t<B>>::template at<InId>>, "Port types do not match");
    return MergedGraph<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId>{std::forward<A>(a), std::forward<B>(b)};
}

/*******************************************************************************************************/
/**************************** end of SIMD-Merged Graph Implementation **********************************/
/*******************************************************************************************************/

// TODO: add nicer enum formatter
inline std::ostream& operator<<(std::ostream& os, const ConnectionResult& value) { return os << static_cast<int>(value); }

inline std::ostream& operator<<(std::ostream& os, const PortType& value) { return os << static_cast<int>(value); }

inline std::ostream& operator<<(std::ostream& os, const PortDirection& value) { return os << static_cast<int>(value); }

template<PortDomainLike T>
inline std::ostream& operator<<(std::ostream& os, const T& value) {
    return os << value.Name;
}

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

inline gr::Graph loadGrc(PluginLoader& loader, std::string_view yamlSrc, std::source_location location) {
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
