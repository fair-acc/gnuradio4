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
inline const char* const kInspectBlock   = "InspectBlock";
inline const char* const kBlockInspected = "BlockInspected";
inline const char* const kGraphInspect   = "GraphInspect";
inline const char* const kGraphInspected = "GraphInspected";

inline const char* const kRegistryBlockTypes     = "RegistryBlockTypes";
inline const char* const kRegistrySchedulerTypes = "RegistrySchedulerTypes";

inline const char* const kSubgraphExportPort   = "SubgraphExportPort";
inline const char* const kSubgraphExportedPort = "SubgraphExportedPort";
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
        auto*       wrapper = static_cast<GraphWrapper*>(context);
        const auto& data    = message.data.value();
        // ValueMap::at() returns Value by value; bind to lvalues so string_view / get_if<bool>() pointers
        // remain valid past the end of each init expression.
        const pmt::Value uniqueBlockNameVal     = data.at("uniqueBlockName");
        const pmt::Value portDirectionStringVal = data.at("portDirection");
        const pmt::Value portNameVal            = data.at("portName");
        const pmt::Value exportFlagVal          = data.at("exportFlag");
        const auto       uniqueBlockName        = uniqueBlockNameVal.value_or(std::string_view{});
        const auto       portDirectionString    = portDirectionStringVal.value_or(std::string_view{});
        const auto       portName               = portNameVal.value_or(std::string_view{});
        const auto       exportFlag             = checked_access_ptr{exportFlagVal.get_if<bool>()};
        if (uniqueBlockName.data() == nullptr || portDirectionString.data() == nullptr || portName.data() == nullptr || exportFlag == nullptr) {
            message.data = std::unexpected(Error{std::format("Invalid definition for the kSubgraphExportPort message {}", message)});
            return message;
        }

        const auto portDirection = portDirectionString == "input" ? PortDirection::INPUT : PortDirection::OUTPUT;

        if (*exportFlag) {
            const pmt::Value exportedNameVal = data.at("exportedName");
            const auto       exportedName    = exportedNameVal.value_or(std::string_view{});
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
            auto& createdDynamicPort                                    = portCollection.emplace_back(gr::DynamicPort(port.value()->weakRef()));
            std::get<gr::DynamicPort>(createdDynamicPort).metaInfo.name = exportedName;
        } else {
            auto exportedPortName = infoIt->second.exportedName;
            bookkeepingCollection.erase(infoIt);
            auto portIt = std::ranges::find_if(portCollection, [&exportedPortName](const auto& portOrCollection) { return BlockModel::portName(portOrCollection) == exportedPortName; });
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
    std::expected<DynamicPort*, Error> findPortInBlock(std::string_view uniqueBlockName, PortDirection portDirection, std::string_view portName, std::source_location location = std::source_location::current()) {
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

        return portDirection == PortDirection::INPUT ? block.value()->dynamicInputPort(portName) : block.value()->dynamicOutputPort(portName);
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
        auto& info = this->metaInformation();
        info.insert_or_assign(std::string_view{"exportedInputPorts"}, exportedPortsFor(_exportedInputPortsForBlock));
        info.insert_or_assign(std::string_view{"exportedOutputPorts"}, exportedPortsFor(_exportedOutputPortsForBlock));
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

    Graph(Graph&& other) noexcept
        : gr::Block<gr::Graph>(std::move(other)),                             //
          _edges(std::move(other._edges)), _blocks(std::move(other._blocks)), //
          _progress(std::move(other._progress)),                              //
          _pluginLoader(std::exchange(other._pluginLoader, nullptr)) {}

    Graph(Graph&)                   = delete; // there can be only one owner of Graph
    Graph& operator=(Graph&)        = delete; // there can be only one owner of Graph
    Graph& operator=(Graph&& other) = delete;
    ~Graph()                        = default;

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
        BlockModel*                        raw         = new BlockWrapper<TBlock>(std::move(initialSettings));
        const std::shared_ptr<BlockModel>& newBlock    = _blocks.emplace_back(std::shared_ptr<BlockModel>{raw});
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
        return std::erase_if(_edges, [&edge](const Edge& e) { return e == edge; });
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

        auto srcPortResult = (*sourceBlockIt)->dynamicOutputPort(std::string_view(sourcePort));
        if (!srcPortResult) {
            return std::unexpected(srcPortResult.error());
        }
        auto dstPortResult = (*destinationBlockIt)->dynamicInputPort(std::string_view(destinationPort));
        if (!dstPortResult) {
            return std::unexpected(dstPortResult.error());
        }
        auto& sourcePortRef      = *srcPortResult.value();
        auto& destinationPortRef = *dstPortResult.value();

        if (sourcePortRef.typeName() != destinationPortRef.typeName()) {
            return std::unexpected(Error(std::format("{}.{} can not be connected to {}.{} -- different types", sourceBlock, sourcePort, destinationBlock, destinationPort)));
        }

        auto connectionResult = sourcePortRef.connect(destinationPortRef);

        if (!connectionResult) {
            return std::unexpected(Error(std::format("{}.{} can not be connected to {}.{}: {}", sourceBlock, sourcePort, destinationBlock, destinationPort, connectionResult.error().message)));
        }

        const bool        isArithmeticLike       = sourcePortRef.isArithmeticLikeValueType();
        const std::size_t sanitizedMinBufferSize = minBufferSize == undefined_size ? graph::defaultMinBufferSize(isArithmeticLike) : minBufferSize;
        _edges.emplace_back(*sourceBlockIt, sourcePort, *destinationBlockIt, destinationPort, sanitizedMinBufferSize, weight, std::string(edgeName));
        return {};
    }

    std::expected<void, Error> removeEdgeBySourcePort(std::string_view sourceBlock, std::string_view sourcePort) {
        auto sourceBlockIt = std::ranges::find_if(_blocks, [&sourceBlock](const auto& block) { return block->uniqueName() == sourceBlock; });
        if (sourceBlockIt == _blocks.end()) {
            return std::unexpected(Error(std::format("Block {} was not found in {}", sourceBlock, this->unique_name)));
        }

        auto srcPortResult = (*sourceBlockIt)->dynamicOutputPort(sourcePort);
        if (!srcPortResult) {
            return std::unexpected(srcPortResult.error());
        }
        auto& sourcePortRef = *srcPortResult.value();

        if (auto result = sourcePortRef.disconnect(); !result) {
            return std::unexpected(Error(std::format("Block {} sourcePortRef could not be disconnected {}: {}", sourceBlock, this->unique_name, result.error().message)));
        }

        return {};
    }

    std::optional<Message> propertyCallbackGraphInspect([[maybe_unused]] std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackRegistryBlockTypes([[maybe_unused]] std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackRegistrySchedulerTypes([[maybe_unused]] std::string_view propertyName, Message message);

    [[nodiscard]] std::expected<void, Error> connect(std::shared_ptr<BlockModel> sourceBlock, PortDefinition sourcePort, //
        std::shared_ptr<BlockModel> destinationBlock, PortDefinition destinationPort,                                    //
        EdgeParameters                        parameters = {},                                                           //
        [[maybe_unused]] std::source_location location   = std::source_location::current()) {

        auto       srcPortResult    = sourceBlock->dynamicOutputPort(sourcePort);
        const bool isArithmeticLike = srcPortResult ? srcPortResult.value()->isArithmeticLikeValueType() : true;
        parameters.minBufferSize    = parameters.minBufferSize == undefined_size ? graph::defaultMinBufferSize(isArithmeticLike) : parameters.minBufferSize;

        _edges.emplace_back(sourceBlock, std::move(sourcePort), //
            destinationBlock, std::move(destinationPort),       //
            std::move(parameters));

        return {};
    }

    template<gr::BlockLike SourceBlock, gr::BlockLike DestinationBlock>
    [[nodiscard]] std::expected<void, Error> connect(SourceBlock& sourceBlock, PortDefinition sourcePort, //
        DestinationBlock& destinationBlock, PortDefinition destinationPort,                               //
        EdgeParameters       parameters = {},                                                             //
        std::source_location location   = std::source_location::current()) {

        std::expected<std::shared_ptr<BlockModel>, Error> sourceBlockModel      = graph::findBlock(*this, sourceBlock, location);
        std::expected<std::shared_ptr<BlockModel>, Error> destinationBlockModel = graph::findBlock(*this, destinationBlock, location);

        if (!sourceBlockModel.has_value() || !destinationBlockModel.has_value()) {
            return std::unexpected(Error(std::format("Source {} and/or destination {} do not belong to this graph", sourceBlock.name, destinationBlock.name), location));
        }

        return connect(*sourceBlockModel, std::move(sourcePort), *destinationBlockModel, std::move(destinationPort), std::move(parameters), location);
    }

    template<fixed_string SourcePort, fixed_string DestinationPort, gr::BlockLike SourceBlock, gr::BlockLike DestinationBlock>
    [[nodiscard]] std::expected<void, Error> connect(SourceBlock& sourceBlock,      //
        DestinationBlock&                                         destinationBlock, //
        EdgeParameters                                            parameters = {},  //
        std::source_location                                      location   = std::source_location::current()) {

        std::expected<std::shared_ptr<BlockModel>, Error> sourceBlockModel      = graph::findBlock(*this, sourceBlock, location);
        std::expected<std::shared_ptr<BlockModel>, Error> destinationBlockModel = graph::findBlock(*this, destinationBlock, location);

        if (!sourceBlockModel.has_value() || !destinationBlockModel.has_value()) {
            return std::unexpected(Error(std::format("Source {} and/or destination {} do not belong to this graph", sourceBlock.name, destinationBlock.name), location));
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
                [&currentIndex, &result, &info]<typename PortOrCollection>(const PortOrCollection& portOrCollection) {
                    if constexpr (traits::port::is_port_v<PortOrCollection>) {
                        if (portOrCollection.metaInfo.name == info.portName) {
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
                            assert(!portOrCollection[0].metaInfo.name.value.empty());
                            if (portOrCollection[0].metaInfo.name == info.portName) {
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
            return std::unexpected(sourcePortDefinition.error());
        }
        if (!destinationPortDefinition) {
            return std::unexpected(destinationPortDefinition.error());
        }

        const bool        isArithmeticLike       = sourcePortDefinition->isArithmeticLike;
        const std::size_t sanitizedMinBufferSize = parameters.minBufferSize == undefined_size ? graph::defaultMinBufferSize(isArithmeticLike) : parameters.minBufferSize;

        parameters.minBufferSize = sanitizedMinBufferSize;
        _edges.emplace_back(sourceBlockModel.value(), sourcePortDefinition->definition, //
            destinationBlockModel.value(), destinationPortDefinition->definition,       //
            std::move(parameters));

        return {};
    }

    using Block<Graph>::processMessages;

    template<typename Anything>
    void processMessages(MsgPortInFromChildren& /*port*/, std::span<const Anything> /*input*/) {
        static_assert(meta::always_false<Anything>, "This is not called, children are processed in processScheduledMessages");
    }

    Edge::EdgeState applyEdgeConnection(Edge& edge) {
        auto srcPortResult = edge._sourceBlock->dynamicOutputPort(edge._sourcePortDefinition);
        auto dstPortResult = edge._destinationBlock->dynamicInputPort(edge._destinationPortDefinition);

        if (!srcPortResult || !dstPortResult) {
            const auto& err = !srcPortResult ? srcPortResult.error() : dstPortResult.error();
            std::println("applyEdgeConnection({}): {}", edge, err.message);
            edge._state = Edge::EdgeState::PortNotFound;
            return edge._state;
        }

        // auto-populate edge domain from block compute_domain if edge domain is still host (default)
        if (edge._domain.kind == "host" && edge._dataResource == std::pmr::get_default_resource()) {
            auto tryResolveFromBlock = [&edge](const BlockModel& block) -> bool {
                const auto& staged    = block.settings().stagedParameters();
                auto        domainStr = std::string();
                if (auto it = staged.find(std::string_view{"compute_domain"}); it != staged.end()) {
                    auto sv = (*it).second.value_or(std::string_view{});
                    if (sv.data()) {
                        domainStr = std::string(sv);
                    }
                }
                if (domainStr.empty()) {
                    if (auto active = block.settings().get("compute_domain")) {
                        auto sv = active->value_or(std::string_view{});
                        if (sv.data()) {
                            domainStr = std::string(sv);
                        }
                    }
                }
                if (!domainStr.empty() && domainStr != gr::thread_pool::kDefaultIoPoolId && domainStr != gr::thread_pool::kDefaultCpuPoolId && domainStr != "host") {
                    edge._domainStr = std::move(domainStr);
                    edge._domain    = ComputeDomain::parse(edge._domainStr);
                    return true;
                }
                return false;
            };
            if (!tryResolveFromBlock(*edge._sourceBlock)) {
                tryResolveFromBlock(*edge._destinationBlock);
            }
        }
        // resolve domain → PMR resources when dataResource is still the default
        if (edge._domain.kind != "host" && edge._dataResource == std::pmr::get_default_resource()) {
            if (auto* mr = ComputeRegistry::instance().tryResolve(edge._domain, edge._domain.user)) {
                edge._dataResource = mr;
                edge._tagResource  = mr;
            }
        }

        auto& sourcePort      = *srcPortResult.value();
        auto& destinationPort = *dstPortResult.value();

        if (sourcePort.typeName() != destinationPort.typeName()) {
            edge._state = Edge::EdgeState::IncompatiblePorts;
        } else {
            const bool hasConnectedEdges = std::ranges::any_of(_edges, [&](const Edge& e) { return edge.hasSameSourcePort(e) && e._state == Edge::EdgeState::Connected; });
            bool       resizeResult      = true;
            if (!hasConnectedEdges) {
                const std::size_t bufferSize = calculateStreamBufferSize(edge);
                resizeResult                 = sourcePort.resizeBuffer(bufferSize, edge._dataResource, edge._tagResource).has_value();
            }

            const bool connectionResult = sourcePort.connect(destinationPort).has_value();
            edge._state                 = connectionResult && resizeResult ? Edge::EdgeState::Connected : Edge::EdgeState::ErrorConnecting;
            edge._actualBufferSize      = sourcePort.bufferSize();
            edge._edgeType              = port::decodePortType(sourcePort.portMaskInfo());
            edge._sourcePort            = std::addressof(sourcePort);
            edge._destinationPort       = std::addressof(destinationPort);
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
        graph::forEachEdge<block::Category::All>(*this, [&refEdge, &maxSize](const Edge& e) {
            if (refEdge.hasSameSourcePort(e)) {
                std::size_t minBufferSize = e.minBufferSize();
                if (minBufferSize != undefined_size) {
                    maxSize = std::max(maxSize, e.minBufferSize());
                }
            }
        });
        assert(maxSize != undefined_size);
        return maxSize;
    }

    void disconnectAllEdges() {
        for (auto& block : _blocks) {
            block->initDynamicPorts();

            auto disconnectAll = [](auto& ports) {
                for (auto& port : ports) {
                    if (auto* p = std::get_if<gr::DynamicPort>(&port)) {
                        std::ignore = p->disconnect();
                    } else {
                        std::ignore = std::get<BlockModel::NamedPortCollection>(port).disconnect();
                    }
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

    detail::traverseSubgraphs<traverseCategory>(root, [&function, filter](const GraphLike auto& graph) {
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
    forEachBlock<traverseCategory>(root, [&n](auto const&) { n++; }, filter);
    return n;
}

template<block::Category traverseCategory, typename Fn>
void forEachEdge(GraphLike auto const& root, Fn&& function, Edge::EdgeState filter) {
    using enum Edge::EdgeState;

    detail::traverseSubgraphs<traverseCategory>(root, [&function, filter](const GraphLike auto& graph) {
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
    forEachEdge<traverseCategory>(root, [&n](auto const&) { n++; }, filter);
    return n;
}

template<gr::block::Category traverseCategory = gr::block::Category::TransparentBlockGroup>
gr::Graph flatten(GraphLike auto const& root, std::source_location location = std::source_location::current()) {
    using enum block::Category;

    gr::Graph flattenedGraph;
    gr::graph::forEachBlock<traverseCategory>(root, [&flattenedGraph](const std::shared_ptr<BlockModel>& block) { flattenedGraph.addBlock(block, false); });
    std::ranges::for_each(root.edges(), [&flattenedGraph, &location](const Edge& edge) { std::ignore = flattenedGraph.addEdge(edge, location); }); // add edges from root graph

    // add edges related to blocks in flattened Graph
    gr::graph::forEachBlock<traverseCategory>(root, [&flattenedGraph, &location](const std::shared_ptr<BlockModel>& block) { std::ranges::for_each(block->edges(), [&flattenedGraph, &location](const Edge& edge) { std::ignore = flattenedGraph.addEdge(edge, location); }); });

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

    forEachBlock<block::Category::All>(graph, [&visited](const auto& block) { visited[block] = VisitState::Unvisited; });

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

    forEachBlock<block::Category::All>(graph, [&visited, &dfs](auto& block) {
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

        std::size_t destPortIdx;
        if (auto* idx = std::get_if<PortDefinition::IndexBased>(&destPortDef.definition)) {
            destPortIdx = idx->topLevel;
        } else {
            auto& str    = std::get<PortDefinition::StringBased>(destPortDef.definition);
            auto  parsed = gr::detail::parsePort(str.name);
            destPortIdx  = destBlock->dynamicInputPortIndex(std::string(parsed.portName)).value_or(meta::invalid_index);
        }

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
    const auto& closingEdge   = loop.edges.back();
    auto        portIdxResult = gr::absolutePortIndex<gr::PortDirection::INPUT>(closingEdge.destinationBlock(), closingEdge.destinationPortDefinition());
    if (!portIdxResult) {
        return std::unexpected(Error(std::format("failed to prime loop: {}", portIdxResult.error().message), location));
    }
    return closingEdge.destinationBlock()->primeInputPort(portIdxResult.value(), nSamples, location);
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

inline std::ostream& operator<<(std::ostream& os, const PortType& value) { return os << static_cast<int>(value); }

inline std::ostream& operator<<(std::ostream& os, const PortDirection& value) { return os << static_cast<int>(value); }

template<PortDomainLike T>
inline std::ostream& operator<<(std::ostream& os, const T& value) {
    return os << value.Name;
}
} // namespace gr

#endif // GNURADIO_GRAPH_HPP
