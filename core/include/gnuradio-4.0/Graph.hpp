#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockModel.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Sequence.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/meta/typelist.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <algorithm>
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
template<typename T>
concept GraphLike = requires(T t, const T& tc) {
    { tc.blocks() } -> std::same_as<std::span<const std::shared_ptr<BlockModel>>>;
    { t.blocks() } -> std::same_as<std::span<std::shared_ptr<BlockModel>>>;
    { tc.edges() } -> std::same_as<std::span<const Edge>>;
    { t.edges() } -> std::same_as<std::span<Edge>>;
};

using namespace std::string_literals;

class PluginLoader;

namespace graph::property {
inline const char* kInspectBlock   = "InspectBlock";
inline const char* kBlockInspected = "BlockInspected";
inline const char* kGraphInspect   = "GraphInspect";
inline const char* kGraphInspected = "GraphInspected";

inline const char* kRegistryBlockTypes     = "RegistryBlockTypes";
inline const char* kRegistrySchedulerTypes = "RegistrySchedulerTypes";

inline const char* kSubgraphExportPort   = "SubgraphExportPort";
inline const char* kSubgraphExportedPort = "SubgraphExportedPort";
} // namespace graph::property

namespace detail {
struct PortNameIndexPair {
    std::string_view portName;
    std::size_t      portIndex = 0;
};

constexpr PortNameIndexPair parsePort(std::string_view portString) {
    std::size_t pos = portString.find("#");
    if (pos == std::string_view::npos) {
        return {portString, meta::invalid_index};
    }

    std::string_view name = portString.substr(0, pos);
    std::string_view tail = portString.substr(pos + 1);

    std::size_t num = 0UZ;

    for (char c : tail) {
        if (c >= '0' && c <= '9') {
            num = num * 10 + static_cast<std::size_t>(c - '0');
        } else {
            break;
        }
    }
    return {name, num};
}

template<fixed_string Name>
struct for_name {
    static constexpr std::string_view name   = Name;
    static constexpr std::size_t      endPos = name.find('#');
    static constexpr std::size_t      size   = endPos == std::string_view::npos ? name.size() : endPos;

    template<typename TPort>
    static consteval bool matcherImpl() {
        return typename TPort::NameT{} == std::string_view(Name.data(), size);
    }

    template<typename TPort>
    struct matches : std::bool_constant<matcherImpl<TPort>()> {};
};

} // namespace detail

namespace graph {
inline static constexpr std::size_t  defaultMinBufferSize(bool isArithmeticLike) { return isArithmeticLike ? 65536UZ : 64UZ; }
inline static constexpr std::int32_t defaultWeight   = 0;
inline static const std::string      defaultEdgeName = "unnamed edge"; // Emscripten doesn't want constexpr strings

inline std::string format(GraphLike auto const& graph) {
    std::ostringstream os;
    for (const auto& block : graph.blocks()) {
        os << std::format("   -block: {} ({})\n", block->name(), block->uniqueName());
    }
    for (const auto& edge : graph.edges()) {
        os << std::format("   -edge: {}\n", edge);
    }
    return os.str();
}

std::expected<std::shared_ptr<BlockModel>, Error> findBlock(GraphLike auto const& graph, BlockLike auto const& what, std::source_location location = std::source_location::current()) noexcept {
    if (auto it = std::ranges::find_if(graph.blocks(), [&](const auto& block) { return block->uniqueName() == what.unique_name; }); it != graph.blocks().end()) {
        return *it;
    }
    return std::unexpected(Error(std::format("Block '{}' ({}) not in this graph:\n{}", what.name, what.unique_name, format(graph)), location));
}

std::expected<std::shared_ptr<BlockModel>, Error> findBlock(const GraphLike auto& graph, const std::shared_ptr<BlockModel>& what, std::source_location location = std::source_location::current()) noexcept {
    if (auto it = std::ranges::find_if(graph.blocks(), [&](const auto& block) { return block.get() == std::addressof(*what); }); it != graph.blocks().end()) {
        return *it;
    }
    return std::unexpected(Error(std::format("Block '{}' ({}) not in this graph:\n{}", what->name(), what->uniqueName(), format(graph)), location));
}

std::expected<std::shared_ptr<BlockModel>, Error> findBlock(const GraphLike auto& graph, std::string_view uniqueBlockName, std::source_location location = std::source_location::current()) noexcept {
    for (const auto& block : graph.blocks()) {
        if (block->uniqueName() == uniqueBlockName) {
            return block;
        }
    }
    return std::unexpected(Error(std::format("Block '{}' not found in:\n{}", uniqueBlockName, format(graph)), location));
}

std::expected<gr::Edge, Error> findEdge(const GraphLike auto& graph, std::string_view edgeName, std::source_location location = std::source_location::current()) noexcept {
    for (const auto& edge : graph.edges()) {
        if (edge.name() == edgeName) {
            return edge;
        }
    }
    return std::unexpected(Error(std::format("Edge '{}' not found in:\n{}", edgeName, format(graph)), location));
}

std::expected<std::size_t, Error> blockIndex(const GraphLike auto& graph, std::string_view uniqueBlockName, std::source_location location = std::source_location::current()) noexcept {
    std::size_t index = 0UZ;
    for (const auto& block : graph.blocks()) {
        if (block->uniqueName() == uniqueBlockName) {
            return index;
        }
        index++;
    }
    return std::unexpected(Error(std::format("Block {} not found in:\n{}", uniqueBlockName, format(graph)), location));
}

std::expected<std::size_t, Error> blockIndex(const GraphLike auto& graph, const std::shared_ptr<BlockModel>& what, std::source_location location = std::source_location::current()) noexcept { return blockIndex(graph, what->uniqueName(), location); }

// forward declaration
template<block::Category traverseCategory, typename Fn>
void forEachEdge(const GraphLike auto& root, Fn&& function, Edge::EdgeState filterCallable = Edge::EdgeState::Unknown);

} // namespace graph

template<typename TSelf, typename TSubGraph = TSelf>
class GraphWrapper : public BlockWrapper<TSelf> {
protected:
    struct PortNameMapper {
        std::string internalName;
        std::string exportedName;
    };
    std::unordered_multimap<std::string, PortNameMapper> _exportedInputPortsForBlock;
    std::unordered_multimap<std::string, PortNameMapper> _exportedOutputPortsForBlock;

    static std::optional<Message> subgraphExportHandler(void* context, Message message) {
        auto*       wrapper             = static_cast<GraphWrapper*>(context);
        const auto& data                = message.data.value();
        const auto  uniqueBlockName     = data.at("uniqueBlockName").value_or(std::string_view{});
        const auto  portDirectionString = data.at("portDirection").value_or(std::string_view{});
        const auto  portName            = data.at("portName").value_or(std::string_view{});
        const auto  exportFlag          = checked_access_ptr{data.at("exportFlag").get_if<bool>()};
        if (uniqueBlockName.data() == nullptr || portDirectionString.data() == nullptr || portName.data() == nullptr || exportFlag == nullptr) {
            message.data = std::unexpected(Error{std::format("Invalid definition for the kSubgraphExportPort message {}", message)});
            return message;
        }

        const auto portDirection = portDirectionString == "input" ? PortDirection::INPUT : PortDirection::OUTPUT;

        if (*exportFlag) {
            const auto exportedName = data.at("exportedName").value_or(std::string_view{});
            if (exportedName.data() == nullptr) {
                message.data = std::unexpected(Error{std::format("Invalid definition for exportName in the kSubgraphExportPort message {}", message)});
                return message;
            }
            if (auto result = wrapper->exportPort(*exportFlag, uniqueBlockName, portDirection, portName, exportedName); !result.has_value()) {
                message.data = std::unexpected(result.error());
            }
        } else {
            if (auto result = wrapper->exportPort(*exportFlag, uniqueBlockName, portDirection, portName, {}); !result.has_value()) {
                message.data = std::unexpected(result.error());
            }
        }

        message.endpoint = graph::property::kSubgraphExportedPort;
        return message;
    }

    void initExportPorts() {
        // We need to make sure nobody touches our dynamic ports
        // as this class will handle them
        this->_dynamicPortsLoader.instance = nullptr;

        // Register the handler for subgraph export port messages on the inner block
        // (works for both Graph and Scheduler wrapped types via BlockBase fields)
        this->_block._subgraphExportHandler                                  = &GraphWrapper::subgraphExportHandler;
        this->_block._subgraphExportContext                                  = this;
        this->_block.propertyCallbacks[graph::property::kSubgraphExportPort] = &BlockBase::propertyCallbackSubgraphExport;
    }

public:
    GraphWrapper(gr::property_map params = gr::property_map{}) : BlockWrapper<TSelf>(std::move(params)) { initExportPorts(); }

    GraphWrapper(TSubGraph&& original) : BlockWrapper<TSelf>(std::move(original)) { initExportPorts(); }

    [[nodiscard]] std::expected<void, Error> exportPort(bool exportFlag, std::string_view uniqueBlockName, PortDirection portDirection, std::string_view portName, std::string_view exportedName, std::source_location location = std::source_location::current()) override {
        auto [infoIt, infoFound] = findExportedPortInfo(uniqueBlockName, portDirection, portName);
        if (infoFound == exportFlag) {
            return std::unexpected(Error(std::format("Port {} in block {} export status already as desired {}", portName, uniqueBlockName, exportFlag)));
        }

        auto port = findPortInBlock(uniqueBlockName, portDirection, portName, location);
        if (!port.has_value()) {
            return std::unexpected(port.error());
        }

        auto& bookkeepingCollection = portDirection == PortDirection::INPUT ? _exportedInputPortsForBlock : _exportedOutputPortsForBlock;
        auto& portCollection        = portDirection == PortDirection::INPUT ? this->_dynamicInputPorts : this->_dynamicOutputPorts;
        if (exportFlag) {
            bookkeepingCollection.emplace(uniqueBlockName, PortNameMapper{std::string(portName), std::string(exportedName)});
            auto& createdDynamicPort                           = portCollection.emplace_back(gr::DynamicPort(port->get().weakRef()));
            std::get<gr::DynamicPort>(createdDynamicPort).name = exportedName;
        } else {
            auto exportedPortName = infoIt->second.exportedName;
            bookkeepingCollection.erase(infoIt);
            auto portIt = std::ranges::find_if(portCollection, [&exportedPortName](const auto& portOrCollection) { return std::visit([&](auto& in) { return in.name == exportedPortName; }, portOrCollection); });
            if (portIt != portCollection.end()) {
                portCollection.erase(portIt);
            } else {
                return std::unexpected(Error("Port was not exported, while it is registered as such"));
            }
        }

        updateMetaInformation();
        return {};
    }

    [[nodiscard]] std::span<const std::shared_ptr<BlockModel>> blocks() const noexcept override { return this->blockRef().blocks(); }
    [[nodiscard]] std::span<std::shared_ptr<BlockModel>>       blocks() noexcept override { return this->blockRef().blocks(); }
    [[nodiscard]] std::span<const Edge>                        edges() const noexcept override { return this->blockRef().edges(); }
    [[nodiscard]] std::span<Edge>                              edges() noexcept override { return this->blockRef().edges(); }

    [[nodiscard]] gr::Graph* graph() final {
        if constexpr (requires { this->blockRef().graph(); }) {
            return &(this->blockRef().graph());
        } else {
            return &(this->blockRef());
        }
    };

    [[nodiscard]] gr::property_map exportedPortsFor(const auto& collection) {
        auto fillMetaInformation = [](property_map& dest, auto& bookkeepingCollection) {
            std::string      previousUniqueName;
            gr::property_map collectedPortNames;
            for (const auto& [blockUniqueName, portNameMap] : bookkeepingCollection) {
                if (previousUniqueName != blockUniqueName && !collectedPortNames.empty()) {
                    dest[convert_string_domain(previousUniqueName)] = std::move(collectedPortNames);
                    collectedPortNames.clear();
                }
                collectedPortNames[convert_string_domain(portNameMap.internalName)] = gr::property_map{
                    {"exportedName", portNameMap.exportedName} //
                };
                previousUniqueName = blockUniqueName;
            }
            if (!collectedPortNames.empty()) {
                dest[convert_string_domain(previousUniqueName)] = std::move(collectedPortNames);
                collectedPortNames.clear();
            }
        };

        property_map result;
        fillMetaInformation(result, collection);
        return result;
    }
    [[nodiscard]] gr::property_map exportedInputPorts() final { return exportedPortsFor(_exportedInputPortsForBlock); }
    [[nodiscard]] gr::property_map exportedOutputPorts() final { return exportedPortsFor(_exportedOutputPortsForBlock); }

private:
    std::expected<std::reference_wrapper<DynamicPort>, Error> findPortInBlock(std::string_view uniqueBlockName, PortDirection portDirection, std::string_view portName, std::source_location location = std::source_location::current()) {
        const auto& asGraph = [this] -> const auto& {
            if constexpr (requires { this->blockRef().graph(); }) {
                return this->blockRef().graph();
            } else {
                return this->blockRef();
            }
        }();
        std::expected<std::shared_ptr<BlockModel>, Error> block = graph::findBlock(asGraph, uniqueBlockName, location);
        if (!block.has_value()) {
            return std::unexpected(Error(block.error().message, location));
        }

        return std::ref(portDirection == PortDirection::INPUT ? block.value()->dynamicInputPort(portName) : block.value()->dynamicOutputPort(portName));
    }

    auto findExportedPortInfo(std::string_view uniqueBlockName, PortDirection portDirection, std::string_view portName) const {
        auto& bookkeepingCollection = portDirection == PortDirection::INPUT ? _exportedInputPortsForBlock : _exportedOutputPortsForBlock;
        const auto& [from, to]      = bookkeepingCollection.equal_range(std::string(uniqueBlockName));
        for (auto it = from; it != to; it++) {
            if (it->second.internalName == portName) {
                return std::make_pair(it, true);
            }
        }
        return std::make_pair(bookkeepingCollection.end(), false);
    }

    void updateMetaInformation() {
        auto& info                  = this->metaInformation();
        info["exportedInputPorts"]  = exportedPortsFor(_exportedInputPortsForBlock);
        info["exportedOutputPorts"] = exportedPortsFor(_exportedOutputPortsForBlock);
    }
};

struct Graph : Block<Graph> {
    std::vector<Edge>                        _edges;
    std::vector<std::shared_ptr<BlockModel>> _blocks;

    std::shared_ptr<gr::Sequence> _progress = std::make_shared<gr::Sequence>();

    gr::PluginLoader* _pluginLoader = nullptr;

    // _subgraphExportHandler and _subgraphExportContext are on BlockBase

public:
    GR_MAKE_REFLECTABLE(Graph);

    constexpr static block::Category blockCategory = block::Category::TransparentBlockGroup;

    Graph(property_map settings = property_map{});

    Graph(gr::PluginLoader& pluginLoader, property_map settings = property_map{}) : Graph(std::move(settings)) { _pluginLoader = std::addressof(pluginLoader); }

    Graph(Graph&& other)
        : gr::Block<gr::Graph>(std::move(other)),                             //
          _edges(std::move(other._edges)), _blocks(std::move(other._blocks)), //
          _progress(std::move(other._progress)),                              //
          _pluginLoader(std::exchange(other._pluginLoader, nullptr)) {}

    Graph(Graph&)                   = delete; // there can be only one owner of Graph
    Graph& operator=(Graph&)        = delete; // there can be only one owner of Graph
    Graph& operator=(Graph&& other) = delete;

    [[nodiscard]] std::span<const std::shared_ptr<BlockModel>> blocks() const noexcept { return _blocks; }
    [[nodiscard]] std::span<std::shared_ptr<BlockModel>>       blocks() noexcept { return _blocks; }
    [[nodiscard]] std::span<const Edge>                        edges() const noexcept { return _edges; }
    [[nodiscard]] std::span<Edge>                              edges() noexcept { return _edges; }

    void clear() {
        _blocks.clear();
        _edges.clear();
    }

    /**
     * @return atomic sequence counter that indicates if any block could process some data or messages
     */
    [[nodiscard]] const Sequence& progress() const noexcept { return *_progress.get(); }

    std::shared_ptr<BlockModel> const& addBlock(std::shared_ptr<BlockModel> block, bool initBlock = true) {
        const std::shared_ptr<BlockModel>& newBlock = _blocks.emplace_back(block);
        if (initBlock) {
            newBlock->init(_progress, this->compute_domain);
        }
        return newBlock;
    }

    template<BlockLike TBlock>
    requires std::is_constructible_v<TBlock, property_map>
    TBlock& emplaceBlock(gr::property_map initialSettings = gr::property_map()) {
        static_assert(std::is_same_v<TBlock, std::remove_reference_t<TBlock>>);
        const std::shared_ptr<BlockModel>& newBlock    = _blocks.emplace_back(std::make_shared<BlockWrapper<TBlock>>(std::move(initialSettings)));
        TBlock*                            rawBlockRef = static_cast<TBlock*>(newBlock->raw());
        rawBlockRef->init(_progress);
        return *rawBlockRef;
    }

    [[maybe_unused]] std::shared_ptr<BlockModel> const& emplaceBlock(std::string_view type, property_map initialSettings);

    bool containsEdge(const Edge& edge) const {
        return std::ranges::any_of(_edges, [&](const Edge& e) { return e == edge; });
    }

    template<typename T>
    requires std::same_as<std::remove_cvref_t<T>, Edge>
    [[nodiscard]] std::expected<std::reference_wrapper<Edge>, Error> addEdge(T&& edge, std::source_location location = std::source_location::current()) {
        if (containsEdge(edge)) {
            return std::unexpected(Error(std::format("Edge already exists in graph:\n{}", edge), location));
        }
        return std::ref(_edges.emplace_back(std::forward<T>(edge)));
    }

    [[maybe_unused]] bool removeEdge(const Edge& edge) {
        return std::erase_if(_edges, [&](const Edge& e) { return e == edge; });
    }

    std::optional<Message> propertyCallbackInspectBlock([[maybe_unused]] std::string_view propertyName, Message message);

    std::expected<std::shared_ptr<BlockModel>, Error> removeBlockByName(std::string_view uniqueName) {
        auto it = std::ranges::find_if(_blocks, [&uniqueName](const auto& block) { return block->uniqueName() == uniqueName; });

        if (it == _blocks.end()) {
            return std::unexpected(Error(std::format("Block {} was not found in {}", uniqueName, this->unique_name)));
        }

        std::erase_if(_edges, [&it](const Edge& edge) { //
            return edge.sourceBlock() == *it || edge.destinationBlock() == *it;
        });

        std::shared_ptr<BlockModel> removedBlock = *it;
        _blocks.erase(it);

        return removedBlock;
    }

    std::pair<std::shared_ptr<BlockModel>, std::shared_ptr<BlockModel>> replaceBlock(std::string_view uniqueName, std::string_view type, const property_map& properties);

    [[nodiscard]] std::expected<void, Error> emplaceEdge(std::string_view sourceBlock, std::string sourcePort, std::string_view destinationBlock, //
        std::string destinationPort, [[maybe_unused]] const std::size_t minBufferSize, [[maybe_unused]] const std::int32_t weight, std::string_view edgeName) {
        auto sourceBlockIt = std::ranges::find_if(_blocks, [&sourceBlock](const auto& block) { return block->uniqueName() == sourceBlock; });
        if (sourceBlockIt == _blocks.end()) {
            return std::unexpected(Error(std::format("Block {} was not found in {}", sourceBlock, this->unique_name)));
        }

        auto destinationBlockIt = std::ranges::find_if(_blocks, [&destinationBlock](const auto& block) { return block->uniqueName() == destinationBlock; });
        if (destinationBlockIt == _blocks.end()) {
            return std::unexpected(Error(std::format("Block {} was not found in {}", destinationBlock, this->unique_name)));
        }

        auto& sourcePortRef      = (*sourceBlockIt)->dynamicOutputPort(std::string_view(sourcePort));
        auto& destinationPortRef = (*destinationBlockIt)->dynamicInputPort(std::string_view(destinationPort));

        if (sourcePortRef.typeName() != destinationPortRef.typeName()) {
            return std::unexpected(Error(std::format("{}.{} can not be connected to {}.{} -- different types", sourceBlock, sourcePort, destinationBlock, destinationPort)));
        }

        auto connectionResult = sourcePortRef.connect(destinationPortRef);

        if (connectionResult != ConnectionResult::SUCCESS) {
            return std::unexpected(Error(std::format("{}.{} can not be connected to {}.{}", sourceBlock, sourcePort, destinationBlock, destinationPort)));
        }

        const bool        isArithmeticLike       = sourcePortRef.portInfo().isValueTypeArithmeticLike;
        const std::size_t sanitizedMinBufferSize = minBufferSize == undefined_size ? graph::defaultMinBufferSize(isArithmeticLike) : minBufferSize;
        _edges.emplace_back(*sourceBlockIt, sourcePort, *destinationBlockIt, destinationPort, sanitizedMinBufferSize, weight, std::string(edgeName));
        return {};
    }

    std::expected<void, Error> removeEdgeBySourcePort(std::string_view sourceBlock, std::string_view sourcePort) {
        auto sourceBlockIt = std::ranges::find_if(_blocks, [&sourceBlock](const auto& block) { return block->uniqueName() == sourceBlock; });
        if (sourceBlockIt == _blocks.end()) {
            return std::unexpected(Error(std::format("Block {} was not found in {}", sourceBlock, this->unique_name)));
        }

        auto& sourcePortRef = (*sourceBlockIt)->dynamicOutputPort(sourcePort);

        if (sourcePortRef.disconnect() == ConnectionResult::FAILED) {
            return std::unexpected(Error(std::format("Block {} sourcePortRef could not be disconnected {}", sourceBlock, this->unique_name)));
        }

        return {};
    }

    std::optional<Message> propertyCallbackGraphInspect([[maybe_unused]] std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackRegistryBlockTypes([[maybe_unused]] std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackRegistrySchedulerTypes([[maybe_unused]] std::string_view propertyName, Message message);

    ConnectionResult connect(std::shared_ptr<BlockModel> sourceBlock, PortDefinition sourcePort, //
        std::shared_ptr<BlockModel> destinationBlock, PortDefinition destinationPort,            //
        EdgeParameters                        parameters = {},                                   //
        [[maybe_unused]] std::source_location location   = std::source_location::current()) {

        const bool isArithmeticLike = sourceBlock->dynamicOutputPort(sourcePort).portInfo().isValueTypeArithmeticLike;
        parameters.minBufferSize    = parameters.minBufferSize == undefined_size ? graph::defaultMinBufferSize(isArithmeticLike) : parameters.minBufferSize;

        _edges.emplace_back(sourceBlock, std::move(sourcePort), //
            destinationBlock, std::move(destinationPort),       //
            std::move(parameters));

        return ConnectionResult::SUCCESS;
    }

    template<gr::BlockLike SourceBlock, gr::BlockLike DestinationBlock>
    ConnectionResult connect(SourceBlock& sourceBlock, PortDefinition sourcePort, //
        DestinationBlock& destinationBlock, PortDefinition destinationPort,       //
        EdgeParameters       parameters = {},                                     //
        std::source_location location   = std::source_location::current()) {

        std::expected<std::shared_ptr<BlockModel>, Error> sourceBlockModel      = graph::findBlock(*this, sourceBlock, location);
        std::expected<std::shared_ptr<BlockModel>, Error> destinationBlockModel = graph::findBlock(*this, destinationBlock, location);

        if (!sourceBlockModel.has_value() || !destinationBlockModel.has_value()) {
            std::print(stderr, "Source {} and/or destination {} do not belong to this graph - loc: {}\n", sourceBlock.name, destinationBlock.name, location);
            return ConnectionResult::FAILED;
        }

        return connect(*sourceBlockModel, std::move(sourcePort), *destinationBlockModel, std::move(destinationPort), std::move(parameters), location);
    }

    template<fixed_string SourcePort, fixed_string DestinationPort, gr::BlockLike SourceBlock, gr::BlockLike DestinationBlock>
    ConnectionResult connect(SourceBlock& sourceBlock,      //
        DestinationBlock&                 destinationBlock, //
        EdgeParameters                    parameters = {},  //
        std::source_location              location   = std::source_location::current()) {

        std::expected<std::shared_ptr<BlockModel>, Error> sourceBlockModel      = graph::findBlock(*this, sourceBlock, location);
        std::expected<std::shared_ptr<BlockModel>, Error> destinationBlockModel = graph::findBlock(*this, destinationBlock, location);

        if (!sourceBlockModel.has_value() || !destinationBlockModel.has_value()) {
            std::print(stderr, "Source {} and/or destination {} do not belong to this graph - loc: {}\n", sourceBlock.name, destinationBlock.name, location);
            return ConnectionResult::FAILED;
        }

        using SourcePortDescriptor      = typename traits::block::all_output_ports<SourceBlock>::template find_or_default<detail::for_name<SourcePort>::template matches, void>;
        using DestinationPortDescriptor = typename traits::block::all_input_ports<DestinationBlock>::template find_or_default<detail::for_name<DestinationPort>::template matches, void>;

        static_assert(!std::is_same_v<void, SourcePortDescriptor>);
        static_assert(!std::is_same_v<void, DestinationPortDescriptor>);
        static_assert(std::is_same_v<typename SourcePortDescriptor::inner_value_type, typename DestinationPortDescriptor::inner_value_type>);

        struct Result {
            PortDefinition definition;
            bool           isArithmeticLike;
        };

        auto getPortDefinition = [&](const auto& ports, std::string_view portString) -> std::expected<Result, Error> {
            auto info = detail::parsePort(portString);

            std::size_t                  currentIndex = 0UZ;
            std::expected<Result, Error> result       = std::unexpected(Error("Not found"));

            gr::meta::tuple_for_each(
                [&]<typename PortOrCollection>(const PortOrCollection& portOrCollection) {
                    if constexpr (traits::port::is_port_v<PortOrCollection>) {
                        if (portOrCollection.name == info.portName) {
                            if (info.portIndex != meta::invalid_index) {
                                result = std::unexpected(Error("This is a port, not a collection of ports"));

                            } else {
                                result = Result{
                                    PortDefinition(currentIndex, meta::invalid_index), //
                                    portOrCollection.kIsArithmeticLikeValueType,       //
                                };
                            }
                        }
                    } else if constexpr (traits::block::detail::array_traits<PortOrCollection>::is_array) {
                        constexpr std::size_t arraySize = traits::block::detail::array_traits<PortOrCollection>::size;
                        if constexpr (arraySize > 0) {
                            assert(!portOrCollection[0].name.empty());
                            if (portOrCollection[0].name == info.portName) {
                                if (info.portIndex < arraySize) {
                                    result = Result{
                                        PortDefinition(currentIndex, info.portIndex),   //
                                        portOrCollection[0].kIsArithmeticLikeValueType, //
                                    };
                                } else {
                                    result = std::unexpected(Error("Index out of range"));
                                }
                            }
                        }

                    } else if constexpr (meta::is_instantiation_of<PortOrCollection, std::vector>) {
                        meta::print_types<meta::message_type<"Vector size is not known at compile-time, you need to use the dynamic connect function variant">, PortOrCollection>{};

                    } else {
                        meta::print_types<meta::message_type<"This is not a port or a collection of ports">, PortOrCollection>{};
                    }
                    currentIndex++;
                },
                ports);

            return result;
        };

        const auto sourcePortDefinition      = getPortDefinition(outputPorts<PortType::ANY>(&sourceBlock), SourcePort);
        const auto destinationPortDefinition = getPortDefinition(inputPorts<PortType::ANY>(&destinationBlock), DestinationPort);

        if (!sourcePortDefinition) {
            return ConnectionResult::FAILED;
        }
        if (!destinationPortDefinition) {
            return ConnectionResult::FAILED;
        }

        const bool        isArithmeticLike       = sourcePortDefinition->isArithmeticLike;
        const std::size_t sanitizedMinBufferSize = parameters.minBufferSize == undefined_size ? graph::defaultMinBufferSize(isArithmeticLike) : parameters.minBufferSize;

        _edges.emplace_back(sourceBlockModel.value(), sourcePortDefinition->definition, //
            destinationBlockModel.value(), destinationPortDefinition->definition,       //
            EdgeParameters{sanitizedMinBufferSize, parameters.weight, std::move(parameters.name)});

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
            std::println("applyEdgeConnection({}): {}", edge, e.what());
            edge._state = Edge::EdgeState::PortNotFound;
        } catch (...) {
            std::println("applyEdgeConnection({}): unknown exception", edge);
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
        graph::forEachEdge<block::Category::All>(*this, [&](const Edge& e) {
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
                    std::print("Edge could not be connected {}\n", edge);
                }
                allConnected = allConnected && wasConnected;
            }
        }
        return allConnected;
    }
};

static_assert(BlockLike<Graph>);

/*******************************************************************************************************/
/**************************** Graph helper functions ***************************************************/
/*******************************************************************************************************/

namespace graph {
namespace detail {
template<block::Category traverseCategory, typename Fn>
void traverseSubgraphs(GraphLike auto const& root, Fn&& visitGraph) {
    using enum block::Category;

    auto recurse = [&visitGraph](const GraphLike auto& graph, auto& self) -> void {
        visitGraph(graph);

        for (const auto& block : graph.blocks()) {
            const auto cat = block->blockCategory();
            if constexpr (traverseCategory == All) {
                if (cat == TransparentBlockGroup || cat == ScheduledBlockGroup) { // block is a sub-graph
                    self(*block, self);
                }
            } else if (cat == traverseCategory) {
                self(*block, self);
            }
        }
    };
    recurse(root, recurse);
}

} // namespace detail

template<block::Category traverseCategory, typename Fn>
void forEachBlock(GraphLike auto const& root, Fn&& function, block::Category filter = block::Category::All) {
    using enum block::Category;

    detail::traverseSubgraphs<traverseCategory>(root, [&](const GraphLike auto& graph) {
        for (auto& block : graph.blocks()) {
            const block::Category cat = block->blockCategory();
            if (filter == All || cat == filter) {
                function(block);
            }
        }
    });
}

template<block::Category traverseCategory>
[[nodiscard]] std::size_t countBlocks(GraphLike auto const& root, block::Category filter = block::Category::All) {
    std::size_t n = 0;
    forEachBlock<traverseCategory>(root, [&](auto const&) { n++; }, filter);
    return n;
}

template<block::Category traverseCategory, typename Fn>
void forEachEdge(GraphLike auto const& root, Fn&& function, Edge::EdgeState filter) {
    using enum Edge::EdgeState;

    detail::traverseSubgraphs<traverseCategory>(root, [&](const GraphLike auto& graph) {
        for (auto& edge : graph.edges()) {
            if (filter == Unknown || edge._state == filter) {
                function(edge);
            }
        }
    });
}

template<block::Category traverseCategory>
[[nodiscard]] std::size_t countEdges(GraphLike auto const& root, Edge::EdgeState filter = Edge::EdgeState::Unknown) {
    std::size_t n = 0;
    forEachEdge<traverseCategory>(root, [&](auto const&) { n++; }, filter);
    return n;
}

template<gr::block::Category traverseCategory = gr::block::Category::TransparentBlockGroup>
gr::Graph flatten(GraphLike auto const& root, std::source_location location = std::source_location::current()) {
    using enum block::Category;

    gr::Graph flattenedGraph;
    gr::graph::forEachBlock<traverseCategory>(root, [&](const std::shared_ptr<BlockModel>& block) { flattenedGraph.addBlock(block, false); });
    std::ranges::for_each(root.edges(), [&](const Edge& edge) { std::ignore = flattenedGraph.addEdge(edge, location); }); // add edges from root graph

    // add edges related to blocks in flattened Graph
    gr::graph::forEachBlock<traverseCategory>(root, [&](const std::shared_ptr<BlockModel>& block) { std::ranges::for_each(block->edges(), [&](const Edge& edge) { std::ignore = flattenedGraph.addEdge(edge, location); }); });

    return flattenedGraph;
}

using AdjacencyList = std::unordered_map<std::shared_ptr<gr::BlockModel>, //
    std::unordered_map<gr::PortDefinition, std::vector<const gr::Edge*>>>;

AdjacencyList computeAdjacencyList(const GraphLike auto& root) {
    AdjacencyList result;
    for (const gr::Edge& edge : root.edges()) {
        std::vector<const gr::Edge*>& srcMapPort = result[edge.sourceBlock()][edge.sourcePortDefinition()];
        srcMapPort.push_back(std::addressof(edge));
    }
    return result;
}

std::vector<gr::Graph> weaklyConnectedComponents(const GraphLike auto& graph) {
    const auto        blocksSpan = graph.blocks();
    const std::size_t N          = blocksSpan.size();

    std::unordered_map<const gr::BlockModel*, std::size_t> indexOf;
    indexOf.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        indexOf.emplace(blocksSpan[i].get(), i);
    }

    std::vector<std::vector<std::size_t>> adjacencyList(N);
    adjacencyList.reserve(N);
    for (const gr::Edge& e : graph.edges()) {
        const auto* sb  = e.sourceBlock().get();
        const auto* db  = e.destinationBlock().get();
        auto        sit = indexOf.find(sb);
        auto        dit = indexOf.find(db);
        if (sit != indexOf.end() && dit != indexOf.end()) {
            const auto s = sit->second;
            const auto d = dit->second;
            adjacencyList[s].push_back(d);
            adjacencyList[d].push_back(s);
        }
    }

    // BFS component labelling and collecting node lists
    std::vector<int>                      compId(N, -1);
    std::vector<std::vector<std::size_t>> comps;
    comps.reserve(N);

    for (std::size_t s = 0; s < N; ++s) {
        if (compId[s] != -1) {
            continue;
        }

        std::vector<std::size_t> comp;
        comp.reserve(8);
        std::deque<std::size_t> q;
        q.push_back(s);
        compId[s] = static_cast<int>(comps.size());

        while (!q.empty()) {
            const auto v = q.front();
            q.pop_front();
            comp.push_back(v);

            for (auto u : adjacencyList[v]) {
                if (compId[u] == -1) {
                    compId[u] = compId[v];
                    q.push_back(u);
                }
            }
        }
        comps.push_back(std::move(comp));
    }

    // sort components according to size (i.e. number of blocks)
    std::vector<std::size_t> order(comps.size());
    std::iota(order.begin(), order.end(), 0UZ);
    std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return comps[a].size() > comps[b].size(); });

    std::vector<std::size_t> compRank(comps.size());
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
        compRank[order[rank]] = rank;
    }

    std::vector<gr::Graph> result(order.size());
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
        const auto cid = order[rank];
        auto&      g   = result[rank];
        g.clear();
        for (auto idx : comps[cid]) {
            g.addBlock(blocksSpan[idx], /*initBlock=*/false);
        }
    }

    // add only edges that are exclusively in the same component
    for (const gr::Edge& e : graph.edges()) {
        const auto itS = indexOf.find(e.sourceBlock().get());
        const auto itD = indexOf.find(e.destinationBlock().get());
        if (itS == indexOf.end() || itD == indexOf.end()) {
            continue;
        }

        const auto cidS = compId[itS->second];
        const auto cidD = compId[itD->second];
        if (cidS >= 0 && cidS == cidD) {
            const auto rank = compRank[static_cast<std::size_t>(cidS)];
            std::ignore     = result[rank].addEdge(e); // shallow edge copy: same blocks/ports
        }
    }

    return result;
}

inline std::span<const gr::Edge* const> outgoingEdges(const AdjacencyList& adj, const std::shared_ptr<gr::BlockModel>& block, const gr::PortDefinition& port) {
    if (auto it = adj.find(block); it != adj.end()) {
        if (auto pit = it->second.find(port); pit != it->second.end()) {
            return std::span(pit->second);
        }
    }
    return {};
}

inline std::vector<std::shared_ptr<BlockModel>> findSourceBlocks(const AdjacencyList& adj) {
    std::vector<std::shared_ptr<BlockModel>> sources;
    std::set<std::shared_ptr<BlockModel>>    destinations;

    for (const auto& [src, ports] : adj) {
        sources.push_back(src);
        for (const auto& [_, edges] : ports) {
            for (const auto* edge : edges) {
                destinations.insert(edge->destinationBlock());
            }
        }
    }

    sources.erase(std::remove_if(sources.begin(), sources.end(), [&](const auto& b) { return destinations.contains(b); }), sources.end());
    std::sort(sources.begin(), sources.end(), [](const auto& a, const auto& b) { return a->name() < b->name(); });
    return sources;
}

struct FeedbackLoop {
    std::vector<Edge> edges;
};

std::vector<FeedbackLoop> detectFeedbackLoops(const GraphLike auto& graph) {
    enum class VisitState { Unvisited, Gray, Black };

    std::unordered_map<std::shared_ptr<BlockModel>, VisitState> visited;
    std::vector<FeedbackLoop>                                   loops;
    std::vector<std::shared_ptr<BlockModel>>                    path;
    std::vector<Edge>                                           pathEdges;

    const AdjacencyList adjList = computeAdjacencyList(graph);

    forEachBlock<block::Category::All>(graph, [&](const auto& block) { visited[block] = VisitState::Unvisited; });

    auto dfs = [&](this auto&& self, auto& current) -> void {
        visited[current] = VisitState::Gray;
        path.push_back(current);

        if (auto it = adjList.find(current); it != adjList.end()) {
            for (const auto& [_, edges] : it->second) {
                for (const Edge* edge : edges) {
                    auto next = edge->destinationBlock();
                    pathEdges.push_back(*edge);

                    if (visited[next] == VisitState::Gray) { // back-edge found - extract cycle
                        auto cycleStart = std::ranges::find(path, next);
                        if (cycleStart != path.end()) {
                            FeedbackLoop loop;
                            auto         startIdx = std::distance(path.begin(), cycleStart);

                            // copy cycle edges in order
                            std::ranges::copy(pathEdges | std::views::drop(startIdx), std::back_inserter(loop.edges));
                            loops.push_back(std::move(loop));
                        }
                    } else if (visited[next] == VisitState::Unvisited) {
                        self(next);
                    }

                    pathEdges.pop_back();
                }
            }
        }

        path.pop_back();
        visited[current] = VisitState::Black;
    };

    forEachBlock<block::Category::All>(graph, [&](auto& block) {
        if (visited[block] == VisitState::Unvisited) {
            dfs(block);
        }
    });

    return loops;
}

[[nodiscard]] inline std::expected<std::size_t, Error> calculateLoopPrimingSize(const FeedbackLoop& loop, std::source_location location = std::source_location::current()) {
    if (loop.edges.empty()) {
        return 0UZ;
    }

    // step 1: check for feedback loop rate consistency
    std::int32_t cumulativeNumerator{1};
    std::int32_t cumulativeDenominator{1};
    for (const auto& edge : loop.edges) {
        auto ratio = edge.destinationBlock()->resamplingRatio();
        cumulativeNumerator *= ratio.numerator;
        cumulativeDenominator *= ratio.denominator;
    }
    if (cumulativeNumerator != cumulativeDenominator) { // stable feedback, net transformation must be 1:1
        return std::unexpected(Error(std::format("feedback loop has unstable rate transformation: {}:{} (net gain: {:.3f})", cumulativeNumerator, cumulativeDenominator, static_cast<double>(cumulativeDenominator) / cumulativeNumerator), location));
    }

    // step 2: calculate minimum samples needed
    std::size_t samplesNeeded = 1UZ;

    std::size_t edgeIdx = 0UZ;
    for (const auto& edge : loop.edges) {
        auto destBlock   = edge.destinationBlock();
        auto destPortDef = edge.destinationPortDefinition();

        std::size_t destPortIdx = std::visit(
            [&destBlock](const auto& portDef) -> std::size_t {
                using T = std::decay_t<decltype(portDef)>;
                if constexpr (std::is_same_v<T, PortDefinition::IndexBased>) {
                    return portDef.topLevel;
                } else {
                    return destBlock->dynamicInputPortIndex(portDef.name);
                }
            },
            destPortDef.definition);

        // known invariant at this stage: loop is rate-stable
        if (edgeIdx == 0UZ) {
            auto sourceRatio = edge.sourceBlock()->resamplingRatio();
            if (sourceRatio.denominator > 0) {
                std::size_t sourceInputNeeded = (samplesNeeded * static_cast<std::size_t>(sourceRatio.numerator)) / static_cast<std::size_t>(sourceRatio.denominator);
                samplesNeeded                 = std::max(samplesNeeded, sourceInputNeeded);
            }

            if (edge.sourceBlock()->stride() > 0) {
                samplesNeeded = std::max(samplesNeeded, static_cast<std::size_t>(edge.sourceBlock()->stride()));
            }
        }
        edgeIdx++;

        auto destRatio = destBlock->resamplingRatio();
        auto destMinIn = destBlock->minInputRequirements();

        if (destPortIdx < destMinIn.size() && destMinIn[destPortIdx] > 0) {
            samplesNeeded = std::max(samplesNeeded, destMinIn[destPortIdx]);
        }

        if (destRatio.numerator > 0) {
            samplesNeeded = std::max(samplesNeeded, static_cast<std::size_t>(destRatio.numerator));
        }

        if (destBlock->stride() > 0) {
            samplesNeeded = std::max(samplesNeeded, static_cast<std::size_t>(destBlock->stride()));
        }
    }

    return samplesNeeded;
}

[[nodiscard]] inline std::expected<std::size_t, Error> primeLoop(const FeedbackLoop& loop, std::size_t nSamples, std::source_location location = std::source_location::current()) {
    if (loop.edges.empty()) {
        return std::unexpected(Error("empty feedback loop cannot be primed", location));
    }
    // need to inject in the last edge where the feedback loop is closed
    const auto& closingEdge = loop.edges.back();
    try {
        std::size_t inputPortIdx = gr::absolutePortIndex<gr::PortDirection::INPUT>(closingEdge.destinationBlock(), closingEdge.destinationPortDefinition());
        return closingEdge.destinationBlock()->primeInputPort(inputPortIdx, nSamples, location);
    } catch (const gr::exception& e) {
        return std::unexpected(Error(std::format("failed to prime loop: {}", e.what()), location));
    }
}

inline void printFeedbackLoop(const FeedbackLoop& loop, std::size_t loopIdx = 0UZ, std::source_location location = std::source_location::current()) {
    std::println("Feedback Loop #{} ({} edges):", loopIdx, loop.edges.size());

    std::size_t edgeIdx = 0UZ;
    for (const auto& edge : loop.edges) {
        std::println("  [{}] {} -> {} (buffer: {}, weight: {})", edgeIdx++, edge.sourceBlock()->name(), edge.destinationBlock()->name(), edge.minBufferSize(), edge.weight());
    }

    std::println("  Recommended priming: {} samples\n", calculateLoopPrimingSize(loop, location));
}

} // namespace graph
} // namespace gr

template<>
struct std::formatter<gr::graph::AdjacencyList> {
    char formatSpecifier = 's'; // 's' = short name, 'l' = long (unique) name

    constexpr auto parse(std::format_parse_context& ctx) {
        auto it = ctx.begin();
        if (it != ctx.end() && (*it == 's' || *it == 'l')) {
            formatSpecifier = *it++;
        } else if (it != ctx.end() && *it != '}') {
            throw std::format_error("invalid format specifier for AdjacencyList: must be 's' or 'l'");
        }
        return it;
    }

    template<typename FormatContext>
    auto format(const gr::graph::AdjacencyList& adj, FormatContext& ctx) const {
        auto       out     = ctx.out();
        const auto getName = [this](const std::shared_ptr<gr::BlockModel>& block) { return (formatSpecifier == 'l') ? block->uniqueName() : block->name(); };

        for (const auto& [srcBlock, portMap] : adj) {
            for (const auto& [srcPort, edges] : portMap) {
                out = std::format_to(out, "{}:{}\n", getName(srcBlock), srcPort);
                for (const gr::Edge* edge : edges) {
                    if (formatSpecifier == 'l') {
                        out = std::format_to(out, "    → {:l}\n", *edge);
                    } else {
                        out = std::format_to(out, "    → {:s}\n", *edge);
                    }
                }
            }
        }
        return out;
    }
};

namespace gr {

/*******************************************************************************************************/
/**************************** begin of SIMD-Merged Graph Implementation ********************************/
/*******************************************************************************************************/

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
 *  - `Merge` class: Combines a source and sink block into a new unit, connecting them via specified
 *    output and input port names.
 *  - `MergeByIndex` class: Combines a source and sink block into a new unit, connecting them via specified
 *    output and input port indices.
 *  - The merged blocks can be efficiently optimized at compile-time.
 *  - Each `Merge` instance has a unique ID and name, aiding in debugging and identification.
 *  - The merging works seamlessly for blocks that have single or multiple output ports.
 *  - It allows for SIMD optimizations if the constituent blocks support it.
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

/**
 * This type constructor can merge simple blocks that are defined via
 * a single `auto processOne(..)` to produce a new `merged` node,
 * bypassing the dynamic run-time buffers.
 *
 * Since the merged node can be highly optimised during compile-time, it's
 * execution performance is usually orders of magnitude more efficient than
 * executing a cascade of the same constituent blocks. See the benchmarks for
 * details.
 *
 * Example:
 * @code
 * // declare flow-graph: 2 x in -> adder -> scale-by-2 -> scale-by-minus1 -> output
 * auto merged = MergeByIndex<scale<int, -1>, 0, MergeByIndex<scale<int, 2>, 0, adder<int>, 0>, 0>();
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

template<BlockLike Left, std::size_t OutId, //
    BlockLike Right, std::size_t InId>
class MergeByIndex : public Block<MergeByIndex<Left, OutId, Right, InId>> {
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

    GR_MAKE_REFLECTABLE(MergeByIndex);

    // TODO: Add a comment why a unique ID is necessary for merged blocks but not for all other blocks. (I.e. unique_id
    // already is a member of the Block base class, this is shadowing that member with a different value. No other block
    // does this.)
    const std::size_t unique_id   = _unique_id_counter++;
    const std::string unique_name = std::format("MergeByIndex<{}:{},{}:{}>#{}", gr::meta::type_name<Left>(), OutId, gr::meta::type_name<Right>(), InId, unique_id);

    MergeByIndex(const MergeByIndex& other)       = delete;
    MergeByIndex& operator=(MergeByIndex& other)  = delete;
    MergeByIndex& operator=(MergeByIndex&& other) = delete;

    MergeByIndex(MergeByIndex&& other) : left(std::move(other.left)), right(std::move(other.right)) {}

private:
    // copy-paste from above, keep in sync
    using base = Block<MergeByIndex<Left, OutId, Right, InId>>;

    Left  left;
    Right right;

    // merged_work_chunk_size, that's what friends are for
    friend base;

    template<BlockLike, std::size_t, BlockLike, std::size_t>
    friend class MergeByIndex;

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
    constexpr MergeByIndex(Left&& l, Right&& r) : left(std::move(l)), right(std::move(r)) {}
    explicit constexpr MergeByIndex(gr::property_map init = {}) : left(init), right(init) {}

    // if the left block (source) implements available_samples (a customization point), then pass the call through
    friend constexpr std::size_t available_samples(const MergeByIndex& self) noexcept
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

template<BlockLike Left, std::size_t OutId, BlockLike Right, std::size_t InId>
inline std::atomic_size_t MergeByIndex<Left, OutId, Right, InId>::_unique_id_counter{0UZ};

namespace detail {
template<meta::fixed_string PortName, typename PortsTypeList>
consteval std::size_t checkedIndexForName() {
    constexpr std::size_t Id = meta::indexForName<PortName, PortsTypeList>();
    static_assert(Id != -1UZ);
    return Id;
}

} // namespace detail

/**
 * This type constructor can merge simple blocks that are defined via a single `auto processOne(..)` producing a
 * new `merged` node, bypassing the dynamic run-time buffers.
 * Since the merged node can be highly optimised during compile-time, it's execution performance is usually orders
 * of magnitude more efficient than executing a cascade of the same constituent blocks. See the benchmarks for details.
 * This function uses the connect-by-port-name API.
 *
 * Example:
 * @code
 * // declare flow-graph: 2 x in -> adder -> scale-by-2 -> output
 * auto merged = merge<scale<int, 2>, "scaled", adder<int>, "addend">();
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
template<BlockLike Left, meta::fixed_string OutName, BlockLike Right, meta::fixed_string InName>
using Merge = MergeByIndex<Left, detail::checkedIndexForName<OutName, typename traits::block::stream_output_ports<Left>>(), //
    Right, detail::checkedIndexForName<OutName, typename traits::block::stream_output_ports<Left>>()>;

/*******************************************************************************************************/
/**************************** end of SIMD-Merged Graph Implementation **********************************/
/*******************************************************************************************************/

/*******************************************************************************************************/
/**************************** begin of FeedbackMerge Implementation ************************************/
/*******************************************************************************************************/

/**
 * Feedback merge for blocks that feed data previously generated
 * to one of the ports.
 *
 * FeedbackMerge<Adder, "out", Scale<0.2f>, "out", "in2">;
 *
 *           Forward
 *           adder          *------------------> out of FeedbackMerge
 *           +----+        /
 * ------in1-|    |       /     Feedback
 *           |    |-out--*      scale
 *     *-in2-|    |       \     +----+
 *    /      +----+        *-in-|    |-out--*
 *    |                         +----+       \
 *    \______________________________________/
 *
 */
template<BlockLike Forward, std::size_t ForwardOutputPortIndex, //
    BlockLike Feedback, std::size_t FeedbackOutputPortIndex,    //
    std::size_t ForwardFeedbackInputPortIndex>
class FeedbackMergeByIndex : public Block<FeedbackMergeByIndex<Forward, ForwardOutputPortIndex, Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex>> {
    static std::atomic_size_t _unique_id_counter;

public:
    GR_MAKE_REFLECTABLE(FeedbackMergeByIndex);

    static_assert(traits::block::stream_input_port_types<Feedback>::size == 1, "Feedback block needs to have only one input port");
    static_assert(traits::block::stream_input_port_types<Forward>::size >= 2, "Forward block must have at least 2 input ports");

    using MergeConnectionForwardOutputType = typename traits::block::stream_output_port_types<Forward>::template at<ForwardOutputPortIndex>;
    using MergeConnectionFeedbackInputType = typename traits::block::stream_input_port_types<Feedback>::template at<0>;
    static_assert(std::is_same_v<MergeConnectionForwardOutputType, MergeConnectionFeedbackInputType>, "The chosen output port of Forward block needs to have the same type as the Feedback input port type");

    using FeedbackConnectionFeedbackOutputType = typename traits::block::stream_output_port_types<Feedback>::template at<FeedbackOutputPortIndex>;
    using FeedbackConnectionForwardInputType   = typename traits::block::stream_input_port_types<Forward>::template at<ForwardFeedbackInputPortIndex>;
    static_assert(std::is_same_v<FeedbackConnectionFeedbackOutputType, FeedbackConnectionForwardInputType>, "The chosen output port of Feedback block needs to have the same type as the chosen Forward input port type");

    // The type of port connected inside of the feedback merge
    using MergeConnectionPortType    = MergeConnectionForwardOutputType;
    using FeedbackConnectionPortType = FeedbackConnectionFeedbackOutputType;

    using SelfInputPortDescriptors  = meta::remove_at<ForwardFeedbackInputPortIndex, typename traits::block::stream_input_ports<Forward>>;
    using SelfOutputPortDescriptors = typename traits::block::stream_output_ports<Forward>;
    using AllPorts                  = meta::concat<SelfInputPortDescriptors, SelfOutputPortDescriptors>;

    using SelfInputPortTypes  = meta::remove_at<ForwardFeedbackInputPortIndex, typename traits::block::stream_input_port_types<Forward>>;
    using SelfOutputPortTypes = typename traits::block::stream_output_port_types<Forward>;
    using ReturnType          = typename SelfOutputPortTypes::tuple_or_type;

    const std::size_t unique_id   = _unique_id_counter++;
    const std::string unique_name = std::format("FeedbackMergeByIndex<{}:{},{}:{},feedback_to:{}>#{}", gr::meta::type_name<Forward>(), ForwardOutputPortIndex, gr::meta::type_name<Feedback>(), FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex, unique_id);

    FeedbackMergeByIndex(const FeedbackMergeByIndex&)            = delete;
    FeedbackMergeByIndex& operator=(const FeedbackMergeByIndex&) = delete;
    FeedbackMergeByIndex& operator=(FeedbackMergeByIndex&&)      = delete;

    FeedbackMergeByIndex(FeedbackMergeByIndex&& other) : forward(std::move(other.forward)), feedback(std::move(other.feedback)), _state(std::move(other._state)) {}

    constexpr FeedbackMergeByIndex(Forward&& fwd, Feedback&& fbk) : forward(std::move(fwd)), feedback(std::move(fbk)) {}

    FeedbackMergeByIndex(gr::property_map init = {}) : forward(init), feedback(init) {}

    template<typename... Ts>
    requires(SelfInputPortTypes::template are_equal<std::remove_cvref_t<Ts>...>)
    constexpr ReturnType processOne(Ts&&... inputs) {
        auto forwardInputTuple = std::forward_as_tuple(std::forward<Ts>(inputs)...);

        auto output = [&]<std::size_t... BeforeIdx, std::size_t... AfterIdx>(std::index_sequence<BeforeIdx...>, std::index_sequence<AfterIdx...>) {
            constexpr std::size_t afterOffset = ForwardFeedbackInputPortIndex + 1;
            return forward.processOne(std::get<BeforeIdx>(forwardInputTuple)..., _state, std::get<afterOffset + AfterIdx - 1>(forwardInputTuple)...);
        }(std::make_index_sequence<ForwardFeedbackInputPortIndex>(), std::make_index_sequence<sizeof...(Ts) - ForwardFeedbackInputPortIndex>());

        _state = feedback.processOne(output);
        return output;
    }

private:
    Forward                    forward;
    Feedback                   feedback;
    FeedbackConnectionPortType _state{};
};

template<BlockLike Forward, std::size_t ForwardOutputPortIndex, //
    BlockLike Feedback, std::size_t FeedbackOutputPortIndex,    //
    std::size_t ForwardFeedbackInputPortIndex>
inline std::atomic_size_t FeedbackMergeByIndex<Forward, ForwardOutputPortIndex, Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex>::_unique_id_counter{0UZ};

template<BlockLike Forward, meta::fixed_string ForwardOutputPortName, //
    BlockLike Feedback, meta::fixed_string FeedbackOutputPortName,    //
    meta::fixed_string ForwardFeedbackInputPortName>
using FeedbackMerge = FeedbackMergeByIndex<                                                                                 //
    Forward, detail::checkedIndexForName<ForwardOutputPortName, typename traits::block::stream_output_ports<Forward>>(),    //
    Feedback, detail::checkedIndexForName<FeedbackOutputPortName, typename traits::block::stream_output_ports<Feedback>>(), //
    detail::checkedIndexForName<ForwardFeedbackInputPortName, typename traits::block::stream_input_ports<Forward>>()>;

/*******************************************************************************************************/
/**************************** end of FeedbackMerge Implementation **************************************/
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

#endif // GNURADIO_GRAPH_HPP
