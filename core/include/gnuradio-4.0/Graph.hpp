#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockModel.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Sequence.hpp>
#include <gnuradio-4.0/meta/typelist.hpp>
#include <gnuradio-4.0/reflection.hpp>
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

namespace graph::property {
inline static const char* kEmplaceBlock = "EmplaceBlock";
inline static const char* kRemoveBlock  = "RemoveBlock";
inline static const char* kReplaceBlock = "ReplaceBlock";

inline static const char* kBlockEmplaced = "BlockEmplaced";
inline static const char* kBlockRemoved  = "BlockRemoved";
inline static const char* kBlockReplaced = "BlockReplaced";

inline static const char* kEmplaceEdge = "EmplaceEdge";
inline static const char* kRemoveEdge  = "RemoveEdge";

inline static const char* kEdgeEmplaced = "EdgeEmplaced";
inline static const char* kEdgeRemoved  = "EdgeRemoved";

inline static const char* kGraphInspect   = "GraphInspect";
inline static const char* kGraphInspected = "GraphInspected";
} // namespace graph::property

class Graph : public gr::Block<Graph> {
private:
    std::shared_ptr<gr::Sequence>                     _progress     = std::make_shared<gr::Sequence>();
    std::shared_ptr<gr::thread_pool::BasicThreadPool> _ioThreadPool = std::make_shared<gr::thread_pool::BasicThreadPool>("graph_thread_pool", gr::thread_pool::TaskType::IO_BOUND, 2UZ, std::numeric_limits<uint32_t>::max());
    std::atomic_bool                                  _topologyChanged{false};
    std::vector<Edge>                                 _edges;
    std::vector<std::unique_ptr<BlockModel>>          _blocks;

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

        SourceConnector(Graph& _self, Source& _source, SourcePort& _port) : self(_self), sourceBlockRaw(_source), sourcePortOrCollectionRaw(_port) {}

        SourceConnector(const SourceConnector&)            = delete;
        SourceConnector(SourceConnector&&)                 = delete;
        SourceConnector& operator=(const SourceConnector&) = delete;
        SourceConnector& operator=(SourceConnector&&)      = delete;

        static_assert(std::is_same_v<SourcePort, gr::Message> || traits::port::is_port_v<SourcePort> || (sourcePortSubIndex != meta::invalid_index), "When we have a collection of ports, we need to have an index to access the desired port in the collection");

    private:
        template<typename Destination, typename DestinationPort, std::size_t destinationPortIndex = meta::invalid_index, std::size_t destinationPortSubIndex = meta::invalid_index>
        [[nodiscard]] constexpr ConnectionResult to(Destination& destinationBlockRaw, DestinationPort& destinationPortOrCollectionRaw, std::size_t minBufferSize = 65536, std::int32_t weight = 0, std::string edgeName = "unnamed edge") {
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

            self._edges.emplace_back(sourceBlock, PortDefinition{sourcePortIndex, sourcePortSubIndex}, destinationBlock, PortDefinition{destinationPortIndex, destinationPortSubIndex}, minBufferSize, weight, std::move(edgeName));

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
    constexpr static block::Category blockCategory = block::Category::TransparentBlockGroup;

    Graph(property_map settings = {}) : gr::Block<Graph>(std::move(settings)) {
        _blocks.reserve(100); // TODO: remove
        propertyCallbacks[graph::property::kEmplaceBlock] = &Graph::propertyCallbackEmplaceBlock;
        propertyCallbacks[graph::property::kRemoveBlock]  = &Graph::propertyCallbackRemoveBlock;
        propertyCallbacks[graph::property::kReplaceBlock] = &Graph::propertyCallbackReplaceBlock;
        propertyCallbacks[graph::property::kEmplaceEdge]  = &Graph::propertyCallbackEmplaceEdge;
        propertyCallbacks[graph::property::kRemoveEdge]   = &Graph::propertyCallbackRemoveEdge;
        propertyCallbacks[graph::property::kGraphInspect] = &Graph::propertyCallbackGraphInspect;
    }
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

    BlockModel& addBlock(std::unique_ptr<BlockModel> block) {
        auto& new_block_ref = _blocks.emplace_back(std::move(block));
        new_block_ref->init(_progress, _ioThreadPool);
        // TODO: Should we connectChildMessagePorts for these blocks as well?
        setTopologyChanged();
        return *new_block_ref.get();
    }

    template<BlockLike TBlock>
    requires std::is_constructible_v<TBlock, property_map>
    auto& emplaceBlock(gr::property_map initialSettings = gr::property_map()) {
        static_assert(std::is_same_v<TBlock, std::remove_reference_t<TBlock>>);
        auto  new_block = std::make_unique<BlockWrapper<TBlock>>(std::move(initialSettings));
        auto* raw_ref   = static_cast<TBlock*>(new_block->raw());
        raw_ref->init(_progress, _ioThreadPool);
        _blocks.push_back(std::move(new_block));
        setTopologyChanged();
        return *raw_ref;
    }

    [[maybe_unused]] auto& emplaceBlock(std::string_view type, std::string_view parameters, property_map initialSettings, PluginLoader& loader = gr::globalPluginLoader()) {
        if (auto block_load = loader.instantiate(type, parameters, std::move(initialSettings)); block_load) {
            setTopologyChanged();
            return addBlock(std::move(block_load));
        }
        throw gr::exception(fmt::format("Can not create block {}<{}>", type, parameters));
    }

    std::optional<Message> propertyCallbackEmplaceBlock(std::string_view /*propertyName*/, Message message) {
        using namespace std::string_literals;
        const auto&         data       = message.data.value();
        const std::string&  type       = std::get<std::string>(data.at("type"s));
        const std::string&  parameters = std::get<std::string>(data.at("parameters"s));
        const property_map& properties = std::get<property_map>(data.at("properties"s));

        auto& newBlock = emplaceBlock(type, parameters, properties);

        std::optional<Message> result = gr::Message{};
        result->endpoint              = graph::property::kBlockEmplaced;
        result->data                  = property_map{{"uniqueName"s, std::string(newBlock.uniqueName())}};

        return result;
    }

    std::optional<Message> propertyCallbackRemoveBlock(std::string_view /*propertyName*/, Message message) {
        using namespace std::string_literals;
        const auto&        data       = message.data.value();
        const std::string& uniqueName = std::get<std::string>(data.at("uniqueName"s));
        auto               it         = std::ranges::find_if(_blocks, [&uniqueName](const auto& block) { return block->uniqueName() == uniqueName; });

        if (it == _blocks.end()) {
            throw gr::exception(fmt::format("Block {} was not found in {}", uniqueName, this->unique_name));
        }

        _blocks.erase(it);
        message.endpoint = graph::property::kBlockRemoved;

        return {message};
    }

    std::optional<Message> propertyCallbackReplaceBlock(std::string_view /*propertyName*/, Message message) {
        using namespace std::string_literals;
        const auto&         data       = message.data.value();
        const std::string&  uniqueName = std::get<std::string>(data.at("uniqueName"s));
        const std::string&  type       = std::get<std::string>(data.at("type"s));
        const std::string&  parameters = std::get<std::string>(data.at("parameters"s));
        const property_map& properties = std::get<property_map>(data.at("properties"s));

        auto it = std::ranges::find_if(_blocks, [&uniqueName](const auto& block) { return block->uniqueName() == uniqueName; });
        if (it == _blocks.end()) {
            throw gr::exception(fmt::format("Block {} was not found in {}", uniqueName, this->unique_name));
        }

        auto newBlock = gr::globalPluginLoader().instantiate(type, parameters, properties);
        if (!newBlock) {
            throw gr::exception(fmt::format("Can not create block {}<{}>", type, parameters));
        }

        const auto newName = newBlock->uniqueName();
        addBlock(std::move(newBlock));

        BlockModel* oldBlock = it->get();
        for (auto& edge : _edges) {
            if (edge._sourceBlock == oldBlock) {
                edge._sourceBlock = newBlock.get();
            }

            if (edge._destinationBlock == oldBlock) {
                edge._destinationBlock = newBlock.get();
            }
        }
        _blocks.erase(it);

        std::optional<Message> result = gr::Message{};
        result->endpoint              = graph::property::kBlockReplaced;
        result->data                  = property_map{{"uniqueName"s, std::string(newName)}};

        return result;
    }

    std::optional<Message> propertyCallbackEmplaceEdge(std::string_view /*propertyName*/, Message message) {
        using namespace std::string_literals;
        const auto&        data             = message.data.value();
        const std::string& sourceBlock      = std::get<std::string>(data.at("sourceBlock"s));
        const std::string& sourcePort       = std::get<std::string>(data.at("sourcePort"s));
        const std::string& destinationBlock = std::get<std::string>(data.at("destinationBlock"s));
        const std::string& destinationPort  = std::get<std::string>(data.at("destinationPort"s));
        const std::size_t  minBufferSize    = std::get<gr::Size_t>(data.at("minBufferSize"s));
        const std::int32_t weight           = std::get<std::int32_t>(data.at("weight"s));
        const std::string  edgeName         = std::get<std::string>(data.at("edgeName"s));

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

        if (sourcePortRef.defaultValue().type() != destinationPortRef.defaultValue().type()) {
            throw gr::exception(fmt::format("{}.{} can not be connected to {}.{} -- different types", sourceBlock, sourcePort, destinationBlock, destinationPort));
        }

        auto connectionResult = sourcePortRef.connect(destinationPortRef);

        if (connectionResult != ConnectionResult::SUCCESS) {
            throw gr::exception(fmt::format("{}.{} can not be connected to {}.{}", sourceBlock, sourcePort, destinationBlock, destinationPort));
        }

        _edges.emplace_back(sourceBlockIt->get(), sourcePort, destinationBlockIt->get(), destinationPort,
            // TODO:
            65536UZ, 0, "unnamed edge");

        message.endpoint = graph::property::kEdgeEmplaced;
        return message;
    }

    std::optional<Message> propertyCallbackRemoveEdge(std::string_view /*propertyName*/, Message message) {
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
        auto serializePortOrCollection = [](const auto& portOrCollection) {
            // clang-format off
            return std::visit(meta::overloaded{
                    [](const gr::DynamicPort& port) {
                        return property_map{
                            {"name", port.name},
                            {"type", port.defaultValue().type().name()}
                        };
                    },
                    [](const BlockModel::NamedPortCollection& namedCollection) {
                        return property_map{
                            {"name", namedCollection.name},
                            {"size", namedCollection.ports.size()},
                            {"type", namedCollection.ports.empty() ? std::string() : std::string(namedCollection.ports[0].defaultValue().type().name()) }
                        };
                    }},
                portOrCollection);
            // clang-format on
        };
        auto serializeEdge = [](const auto& edge) {
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

            return result;
        };
        auto serializeBlock = [&serializeEdge, &serializePortOrCollection](auto _serializeBlock, const auto& block) -> property_map {
            property_map result;
            result["name"s]            = std::string(block->name());
            result["uniqueName"s]      = std::string(block->uniqueName());
            result["typeName"s]        = std::string(block->typeName());
            result["isBlocking"s]      = block->isBlocking();
            result["metaInformation"s] = block->metaInformation();
            result["blockCategory"s]   = std::string(magic_enum::enum_name(block->blockCategory()));
            result["uiCategory"s]      = std::string(magic_enum::enum_name(block->uiCategory()));
            result["settings"s]        = block->settings().getStored();

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
                    serializedChildren[std::string(child->uniqueName())] = _serializeBlock(_serializeBlock, child);
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
        };

        message.data = [&] {
            property_map result;
            result["name"s]          = std::string(name);
            result["uniqueName"s]    = std::string(unique_name);
            result["blockCategory"s] = std::string(magic_enum::enum_name(blockCategory));

            property_map serializedChildren;
            for (const auto& child : blocks()) {
                serializedChildren[std::string(child->uniqueName())] = serializeBlock(serializeBlock, child);
            }
            result["children"] = std::move(serializedChildren);

            property_map serializedEdges;
            std::size_t  index = 0UZ;
            for (const auto& edge : edges()) {
                serializedEdges[std::to_string(index)] = serializeEdge(edge);
                index++;
            }
            return result;
        }();

        message.endpoint = graph::property::kGraphInspected;
        return message;
    }

    // connect using the port index
    template<std::size_t sourcePortIndex, std::size_t sourcePortSubIndex, typename Source>
    [[nodiscard]] auto connectInternal(Source& source) {
        auto& port_or_collection = outputPort<sourcePortIndex, PortType::ANY>(&source);
        return SourceConnector<Source, std::remove_cvref_t<decltype(port_or_collection)>, sourcePortIndex, sourcePortSubIndex>(*this, source, port_or_collection);
    }

    template<std::size_t sourcePortIndex, std::size_t sourcePortSubIndex, typename Source>
    [[nodiscard, deprecated("The connect with the port name should be used")]] auto connect(Source& source) {
        return connectInternal<sourcePortIndex, sourcePortSubIndex, Source>(source);
    }

    template<std::size_t sourcePortIndex, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        if constexpr (sourcePortIndex == meta::default_message_port_index) {
            return SourceConnector<Source, decltype(source.msgOut), meta::invalid_index, meta::invalid_index>(*this, source, source.msgOut);
        } else {
            return connect<sourcePortIndex, meta::invalid_index, Source>(source);
        }
    }

    // connect using the port name

    template<fixed_string sourcePortName, std::size_t sourcePortSubIndex, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        using source_output_ports             = typename traits::block::all_output_ports<Source>;
        constexpr std::size_t sourcePortIndex = meta::indexForName<sourcePortName, source_output_ports>();
        if constexpr (sourcePortIndex == meta::invalid_index) {
            meta::print_types<meta::message_type<"There is no output port with the specified name in this source block">, Source, meta::message_type<sourcePortName>, meta::message_type<"These are the known names:">, traits::block::all_output_port_names<Source>, meta::message_type<"Full ports info:">, source_output_ports> port_not_found_error{};
        }
        return connectInternal<sourcePortIndex, sourcePortSubIndex, Source>(source);
    }

    template<fixed_string sourcePortName, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        return connect<sourcePortName, meta::invalid_index, Source>(source);
    }

    // dynamic/runtime connections

    template<typename Source, typename Destination>
    requires(!std::is_pointer_v<std::remove_cvref_t<Source>> && !std::is_pointer_v<std::remove_cvref_t<Destination>>)
    ConnectionResult connect(Source& sourceBlockRaw, PortDefinition sourcePortDefinition, Destination& destinationBlockRaw, PortDefinition destinationPortDefinition, std::size_t minBufferSize = 65536, std::int32_t weight = 0, std::string edgeName = "unnamed edge") {
        auto* sourceBlock      = findBlock(sourceBlockRaw).get();
        auto* destinationBlock = findBlock(destinationBlockRaw).get();

        if (sourceBlock == nullptr || destinationBlock == nullptr) {
            return ConnectionResult::FAILED;
        }

        _edges.emplace_back(sourceBlock, sourcePortDefinition, destinationBlock, destinationPortDefinition, minBufferSize, weight, std::move(edgeName));
        return ConnectionResult::SUCCESS;
    }

    using Block<Graph>::processMessages;

    template<typename Anything>
    void processMessages(MsgPortInNamed<"__FromChildren">& /*port*/, std::span<const Anything> /*input*/) {
        static_assert(meta::always_false<Anything>, "This is not called, children are processed in processScheduledMessages");
    }

    Edge::EdgeState applyEdgeConnection(Edge& edge) {
        try {
            auto& sourcePort      = edge._sourceBlock->dynamicOutputPort(edge._sourcePortDefinition);
            auto& destinationPort = edge._destinationBlock->dynamicInputPort(edge._destinationPortDefinition);

            if (sourcePort.defaultValue().type().name() != destinationPort.defaultValue().type().name()) {
                edge._state = Edge::EdgeState::IncompatiblePorts;
            } else {
                auto connectionResult = sourcePort.connect(destinationPort) == ConnectionResult::SUCCESS;
                edge._state           = connectionResult ? Edge::EdgeState::Connected : Edge::EdgeState::ErrorConnecting;
            }
        } catch (...) {
            edge._state = Edge::EdgeState::PortNotFound;
        }

        return edge._state;
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
                allConnected = allConnected && (edge.state() == Edge::EdgeState::Connected);
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

template<typename TBlock>
concept SourceBlockLike = traits::block::can_processOne<TBlock> and traits::block::template stream_output_port_types<TBlock>::size > 0;

static_assert(not SourceBlockLike<int>);

template<typename TBlock>
concept SinkBlockLike = traits::block::can_processOne<TBlock> and traits::block::template stream_input_port_types<TBlock>::size > 0;

static_assert(not SinkBlockLike<int>);

template<SourceBlockLike Left, SinkBlockLike Right, std::size_t OutId, std::size_t InId>
class MergedGraph : public Block<MergedGraph<Left, Right, OutId, InId>, meta::concat<typename traits::block::stream_input_ports<Left>, meta::remove_at<InId, typename traits::block::stream_input_ports<Right>>>, meta::concat<meta::remove_at<OutId, typename traits::block::stream_output_ports<Left>>, typename traits::block::stream_output_ports<Right>>> {
    static std::atomic_size_t _unique_id_counter;

public:
    const std::size_t unique_id   = _unique_id_counter++;
    const std::string unique_name = fmt::format("MergedGraph<{}:{},{}:{}>#{}", gr::meta::type_name<Left>(), OutId, gr::meta::type_name<Right>(), InId, unique_id);

private:
    // copy-paste from above, keep in sync
    using base = Block<MergedGraph<Left, Right, OutId, InId>, meta::concat<typename traits::block::stream_input_ports<Left>, meta::remove_at<InId, typename traits::block::stream_input_ports<Right>>>, meta::concat<meta::remove_at<OutId, typename traits::block::stream_output_ports<Left>>, typename traits::block::stream_output_ports<Right>>>;

    Left  left;
    Right right;

    // merged_work_chunk_size, that's what friends are for
    friend base;

    template<SourceBlockLike, SinkBlockLike, std::size_t, std::size_t>
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
    constexpr auto apply_left(std::size_t offset, auto&& input_tuple) noexcept {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return invokeProcessOneWithOrWithoutOffset(left, offset, std::get<Is>(std::forward<decltype(input_tuple)>(input_tuple))...); }(std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    constexpr auto apply_right(std::size_t offset, auto&& input_tuple, auto&& tmp) noexcept {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset  = traits::block::stream_input_port_types<Left>::size;
            constexpr std::size_t second_offset = traits::block::stream_input_port_types<Left>::size + sizeof...(Is);
            static_assert(second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return invokeProcessOneWithOrWithoutOffset(right, offset, std::get<first_offset + Is>(std::forward<decltype(input_tuple)>(input_tuple))..., std::forward<decltype(tmp)>(tmp), std::get<second_offset + Js>(input_tuple)...);
        }(std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    using TInputPortTypes  = typename traits::block::stream_input_port_types<base>;
    using TOutputPortTypes = typename traits::block::stream_output_port_types<base>;
    using TReturnType      = typename traits::block::stream_return_type<base>;

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
    constexpr meta::simdize<TReturnType, meta::simdize_size_v<std::tuple<Ts...>>> processOne(std::size_t offset, const Ts&... inputs) {
        static_assert(traits::block::stream_output_port_types<Left>::size == 1, "TODO: SIMD for multiple output ports not implemented yet");
        return apply_right<InId, traits::block::stream_input_port_types<Right>::size() - InId - 1>(offset, std::tie(inputs...), apply_left<traits::block::stream_input_port_types<Left>::size()>(offset, std::tie(inputs...)));
    }

    constexpr auto processOne_simd(std::size_t offset, auto N)
    requires traits::block::can_processOne_simd<Right>
    {
        if constexpr (requires(Left& l) {
                          { l.processOne_simd(offset, N) };
                      }) {
            return invokeProcessOneWithOrWithoutOffset(right, offset, left.processOne_simd(offset, N));
        } else if constexpr (requires(Left& l) {
                                 { l.processOne_simd(N) };
                             }) {
            return invokeProcessOneWithOrWithoutOffset(right, offset, left.processOne_simd(N));
        } else {
            using LeftResult = typename traits::block::stream_return_type<Left>;
            using V          = meta::simdize<LeftResult, N>;
            alignas(stdx::memory_alignment_v<V>) LeftResult tmp[V::size()];
            for (std::size_t i = 0UZ; i < V::size(); ++i) {
                tmp[i] = invokeProcessOneWithOrWithoutOffset(left, offset + i);
            }
            return invokeProcessOneWithOrWithoutOffset(right, offset, V(tmp, stdx::vector_aligned));
        }
    }

    template<typename... Ts>
    // Nicer error messages for the following would be good, but not at the expense of breaking can_processOne_simd.
    requires(TInputPortTypes::template are_equal<std::remove_cvref_t<Ts>...>)
    constexpr TReturnType processOne(std::size_t offset, Ts&&... inputs) {
        // if (sizeof...(Ts) == 0) we could call `return processOne_simd(integral_constant<size_t, width>)`. But if
        // the caller expects to process *one* sample (no inputs for the caller to explicitly
        // request simd), and we process more, we risk inconsistencies.
        if constexpr (traits::block::stream_output_port_types<Left>::size == 1) {
            // only the result from the right block needs to be returned
            return apply_right<InId, traits::block::stream_input_port_types<Right>::size() - InId - 1>(offset, std::forward_as_tuple(std::forward<Ts>(inputs)...), apply_left<traits::block::stream_input_port_types<Left>::size()>(offset, std::forward_as_tuple(std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out  = apply_left<traits::block::stream_input_port_types<Left>::size()>(offset, std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, traits::block::stream_input_port_types<Right>::size() - InId - 1>(offset, std::forward_as_tuple(std::forward<Ts>(inputs)...), std::move(std::get<OutId>(left_out)));

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
        gr::meta::print_types<gr::meta::message_type<"OUTPUT_PORTS_ARE:">, typename traits::block::stream_output_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, OutId>, typename traits::block::stream_output_port_types<std::remove_cvref_t<A>>::template at<OutId>,

            gr::meta::message_type<"INPUT_PORTS_ARE:">, typename traits::block::stream_input_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, InId>, typename traits::block::stream_input_port_types<std::remove_cvref_t<A>>::template at<InId>>{};
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
template<fixed_string OutName, fixed_string InName, SourceBlockLike A, SinkBlockLike B>
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

} // namespace gr

// minimal reflection declaration
REFL_TYPE(gr::Graph)
REFL_END

template<>
struct fmt::formatter<gr::Edge> {
    char formatSpecifier = 's';

    constexpr auto parse(fmt::format_parse_context& ctx) {
        auto it = ctx.begin();
        if (it != ctx.end() && (*it == 's' || *it == 'l')) {
            formatSpecifier = *it++;
        } else if (it != ctx.end() && *it != '}') {
            throw fmt::format_error("invalid format specifier");
        }
        return it;
    }

    template<typename FormatContext>
    auto format(const gr::Edge& e, FormatContext& ctx) {
        using PortIndex  = gr::PortDefinition;
        const auto& name = [this](const gr::BlockModel* block) { return (formatSpecifier == 'l') ? block->uniqueName() : block->name(); };

        const auto portIndex = [](const gr::PortDefinition& port) {
            return std::visit(gr::meta::overloaded(
                                  [](const gr::PortDefinition::IndexBased& index) {
                                      if (index.subIndex == gr::meta::invalid_index) {
                                          return fmt::format("{}", index.topLevel);
                                      } else {
                                          return fmt::format("{}#{}", index.topLevel, index.subIndex);
                                      }
                                  },
                                  [](const gr::PortDefinition::StringBased& index) { return index.name; }),
                port.definition);
        };

        return fmt::format_to(ctx.out(), "{}/{} ⟶ (name: '{}', size: {:2}, weight: {:2}, state: {}) ⟶ {}/{}", //
            name(e._sourceBlock), portIndex(e._sourcePortDefinition),                                         //
            e._name, e._minBufferSize, e._weight, magic_enum::enum_name(e._state),                            //
            name(e._destinationBlock), portIndex(e._destinationPortDefinition));
    }
};

#endif // include guard
