#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP

#include <gnuradio-4.0/meta/typelist.hpp>

#include "Block.hpp"
#include "Buffer.hpp"
#include "CircularBuffer.hpp"
#include "Port.hpp"
#include "Sequence.hpp"
#include "thread/thread_pool.hpp"

#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <ranges>
#include <tuple>
#include <variant>

#if !__has_include(<source_location>)
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

using namespace gr::literals;

class BlockModel {
protected:
    using DynamicPorts                         = std::vector<gr::DynamicPort>;
    bool                  _DynamicPorts_loaded = false;
    std::function<void()> _DynamicPorts_loader;
    DynamicPorts          _dynamic_input_ports;
    DynamicPorts          _dynamic_output_ports;

    BlockModel() = default;

public:
    BlockModel(const BlockModel &) = delete;
    BlockModel &
    operator=(const BlockModel &)
            = delete;
    BlockModel(BlockModel &&other) = delete;
    BlockModel &
    operator=(BlockModel &&other)
            = delete;

    void
    initDynamicPorts() const {
        if (!_DynamicPorts_loaded) _DynamicPorts_loader();
    }

    gr::DynamicPort &
    dynamicInputPort(std::size_t index) { // TODO: do we need the 'dynamic' prefix here?
        initDynamicPorts();
        return _dynamic_input_ports.at(index);
    }

    gr::DynamicPort &
    dynamicOutputPort(std::size_t index) { // TODO: do we need the 'dynamic' prefix here?
        initDynamicPorts();
        return _dynamic_output_ports.at(index);
    }

    [[nodiscard]] auto
    dynamicInputPortsSize() const { // TODO: return collection and rely on 'size()' of underlying collection
        initDynamicPorts();
        return _dynamic_input_ports.size();
    }

    [[nodiscard]] auto
    dynamicOutputPortsSize() const { // TODO: return collection and rely on 'size()' of underlying collection
        initDynamicPorts();
        return _dynamic_output_ports.size();
    }

    virtual ~BlockModel() = default;

    /**
     * @brief to be called by scheduler->graph to initialise block
     */
    virtual void
    init(std::shared_ptr<gr::Sequence> progress, std::shared_ptr<gr::thread_pool::BasicThreadPool> ioThreadPool)
            = 0;

    /**
     * @brief returns scheduling hint that invoking the work(...) function may block on IO or system-calls
     */
    [[nodiscard]] virtual constexpr bool
    isBlocking() const noexcept
            = 0;

    /**
     * @brief number of available readable samples at the block's input ports
     */
    [[nodiscard]] virtual constexpr std::size_t
    availableInputSamples(std::vector<std::size_t> &) const noexcept
            = 0;

    /**
     * @brief number of available writable samples at the block's output ports
     */
    [[nodiscard]] virtual constexpr std::size_t
    availableOutputSamples(std::vector<std::size_t> &) const noexcept
            = 0;

    /**
     * @brief user defined name
     */
    [[nodiscard]] virtual std::string_view
    name() const
            = 0;

    /**
     * @brief the type of the node as a string
     */
    [[nodiscard]] virtual std::string_view
    typeName() const
            = 0;

    /**
     * @brief user-defined name
     * N.B. may not be unique -> ::uniqueName
     */
    virtual void
    setName(std::string name) noexcept
            = 0;

    /**
     * @brief used to store non-graph-processing information like UI block position etc.
     */
    [[nodiscard]] virtual property_map &
    metaInformation() noexcept
            = 0;

    [[nodiscard]] virtual const property_map &
    metaInformation() const
            = 0;

    /**
     * @brief process-wide unique name
     * N.B. can be used to disambiguate in case user provided the same 'name()' for several blocks.
     */
    [[nodiscard]] virtual std::string_view
    uniqueName() const
            = 0;

    [[nodiscard]] virtual SettingsBase &
    settings() const
            = 0;

    [[nodiscard]] virtual work::Result
    work(std::size_t requested_work)
            = 0;

    [[nodiscard]] virtual void *
    raw() = 0;
};

template<BlockLike T>
class BlockWrapper : public BlockModel {
private:
    static_assert(std::is_same_v<T, std::remove_reference_t<T>>);
    T           _block;
    std::string _type_name = gr::meta::type_name<T>();

    [[nodiscard]] constexpr const auto &
    blockRef() const noexcept {
        if constexpr (requires { *_block; }) {
            return *_block;
        } else {
            return _block;
        }
    }

    [[nodiscard]] constexpr auto &
    blockRef() noexcept {
        if constexpr (requires { *_block; }) {
            return *_block;
        } else {
            return _block;
        }
    }

    void
    createDynamicPortsLoader() {
        _DynamicPorts_loader = [this] {
            if (_DynamicPorts_loaded) return;

            using TBlock       = std::remove_cvref_t<decltype(blockRef())>;

            auto register_port = []<typename PortOrCollection>(PortOrCollection &port_or_collection, auto &where) {
                auto process_port = [&where]<typename Port>(Port &port) { where.push_back(gr::DynamicPort(port, DynamicPort::non_owned_reference_tag{})); };

                if constexpr (traits::port::is_port_v<PortOrCollection>) {
                    process_port(port_or_collection);

                } else {
                    for (auto &port : port_or_collection) {
                        process_port(port);
                    }
                }
            };

            constexpr std::size_t input_port_count = gr::traits::block::template input_port_types<TBlock>::size;
            [this, register_port]<std::size_t... Is>(std::index_sequence<Is...>) {
                (register_port(gr::inputPort<Is>(&blockRef()), this->_dynamic_input_ports), ...);
            }(std::make_index_sequence<input_port_count>());

            constexpr std::size_t output_port_count = gr::traits::block::template output_port_types<TBlock>::size;
            [this, register_port]<std::size_t... Is>(std::index_sequence<Is...>) {
                (register_port(gr::outputPort<Is>(&blockRef()), this->_dynamic_output_ports), ...);
            }(std::make_index_sequence<output_port_count>());

            static_assert(input_port_count + output_port_count > 0);
            _DynamicPorts_loaded = true;
        };
    }

public:
    BlockWrapper(const BlockWrapper &other) = delete;
    BlockWrapper(BlockWrapper &&other)      = delete;
    BlockWrapper &
    operator=(const BlockWrapper &other)
            = delete;
    BlockWrapper &
    operator=(BlockWrapper &&other)
            = delete;

    ~BlockWrapper() override = default;

    BlockWrapper() { createDynamicPortsLoader(); }

    template<typename Arg>
        requires(!std::is_same_v<std::remove_cvref_t<Arg>, T>)
    explicit BlockWrapper(Arg &&arg) : _block(std::forward<Arg>(arg)) {
        createDynamicPortsLoader();
    }

    template<typename... Args>
        requires(sizeof...(Args) > 1)
    explicit BlockWrapper(Args &&...args) : _block{ std::forward<Args>(args)... } {
        createDynamicPortsLoader();
    }

    explicit BlockWrapper(std::initializer_list<std::pair<const std::string, pmtv::pmt>> init_parameter) : _block{ std::move(init_parameter) } { createDynamicPortsLoader(); }

    void
    init(std::shared_ptr<gr::Sequence> progress, std::shared_ptr<gr::thread_pool::BasicThreadPool> ioThreadPool) override {
        return blockRef().init(progress, ioThreadPool);
    }

    [[nodiscard]] constexpr work::Result
    work(std::size_t requested_work = std::numeric_limits<std::size_t>::max()) override {
        return blockRef().work(requested_work);
    }

    [[nodiscard]] constexpr bool
    isBlocking() const noexcept override {
        return blockRef().isBlocking();
    }

    [[nodiscard]] constexpr std::size_t
    availableInputSamples(std::vector<std::size_t> &data) const noexcept override {
        return blockRef().availableInputSamples(data);
    }

    [[nodiscard]] constexpr std::size_t
    availableOutputSamples(std::vector<std::size_t> &data) const noexcept override {
        return blockRef().availableOutputSamples(data);
    }

    [[nodiscard]] std::string_view
    name() const override {
        return blockRef().name;
    }

    void
    setName(std::string name) noexcept override {
        blockRef().name = std::move(name);
    }

    [[nodiscard]] std::string_view
    typeName() const override {
        return _type_name;
    }

    [[nodiscard]] property_map &
    metaInformation() noexcept override {
        return blockRef().meta_information;
    }

    [[nodiscard]] const property_map &
    metaInformation() const override {
        return blockRef().meta_information;
    }

    [[nodiscard]] std::string_view
    uniqueName() const override {
        return blockRef().unique_name;
    }

    [[nodiscard]] SettingsBase &
    settings() const override {
        return blockRef().settings();
    }

    [[nodiscard]] void *
    raw() override {
        return std::addressof(blockRef());
    }
};

class edge {
public: // TODO: consider making this private and to use accessors (that can be safely used by users)
    using PortDirection::INPUT;
    using PortDirection::OUTPUT;
    BlockModel  *_src_block;
    BlockModel  *_dst_block;
    std::size_t  _src_port_index;
    std::size_t  _dst_port_index;
    std::size_t  _min_buffer_size;
    std::int32_t _weight;
    std::string  _name; // custom edge name
    bool         _connected;

public:
    edge()             = delete;

    edge(const edge &) = delete;

    edge &
    operator=(const edge &)
            = delete;

    edge(edge &&) noexcept = default;

    edge &
    operator=(edge &&) noexcept
            = default;

    edge(BlockModel *src_block, std::size_t src_port_index, BlockModel *dst_block, std::size_t dst_port_index, std::size_t min_buffer_size, std::int32_t weight, std::string_view name)
        : _src_block(src_block), _dst_block(dst_block), _src_port_index(src_port_index), _dst_port_index(dst_port_index), _min_buffer_size(min_buffer_size), _weight(weight), _name(name) {}

    [[nodiscard]] constexpr const BlockModel &
    src_block() const noexcept {
        return *_src_block;
    }

    [[nodiscard]] constexpr const BlockModel &
    dst_block() const noexcept {
        return *_dst_block;
    }

    [[nodiscard]] constexpr std::size_t
    src_port_index() const noexcept {
        return _src_port_index;
    }

    [[nodiscard]] constexpr std::size_t
    dst_port_index() const noexcept {
        return _dst_port_index;
    }

    [[nodiscard]] constexpr std::string_view
    name() const noexcept {
        return _name;
    }

    [[nodiscard]] constexpr std::size_t
    min_buffer_size() const noexcept {
        return _min_buffer_size;
    }

    [[nodiscard]] constexpr std::int32_t
    weight() const noexcept {
        return _weight;
    }

    [[nodiscard]] constexpr bool
    is_connected() const noexcept {
        return _connected;
    }
};

struct Graph {
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::Sequence> progress                         = std::make_shared<gr::Sequence>();
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::thread_pool::BasicThreadPool> ioThreadPool = std::make_shared<gr::thread_pool::BasicThreadPool>(
            "graph_thread_pool", gr::thread_pool::TaskType::IO_BOUND, 2_UZ, std::numeric_limits<uint32_t>::max());

private:
    std::vector<std::function<ConnectionResult(Graph &)>> _connection_definitions;
    std::vector<std::unique_ptr<BlockModel>>              _blocks;
    std::vector<edge>                                     _edges;

    template<typename TBlock>
    std::unique_ptr<BlockModel> &
    findBlock(TBlock &what) {
        static_assert(!std::is_pointer_v<std::remove_cvref_t<TBlock>>);
        auto it = [&, this] {
            if constexpr (std::is_same_v<TBlock, BlockModel>) {
                return std::find_if(_blocks.begin(), _blocks.end(), [&](const auto &node) { return node.get() == &what; });
            } else {
                return std::find_if(_blocks.begin(), _blocks.end(), [&](const auto &node) { return node->raw() == &what; });
            }
        }();

        if (it == _blocks.end()) throw std::runtime_error(fmt::format("No such node in this graph"));
        return *it;
    }

    template<typename TBlock>
    [[nodiscard]] DynamicPort &
    dynamicOutputPort(TBlock &node, std::size_t index) {
        return findBlock(node)->dynamicOutputPort(index);
    }

    template<typename TBlock>
    [[nodiscard]] DynamicPort &
    dynamicInputPort(TBlock &node, std::size_t index) {
        return findBlock(node)->dynamicInputPort(index);
    }

    template<std::size_t src_port_index, std::size_t dst_port_index, typename Source, typename SourcePort, typename Destination, typename DestinationPort>
    [[nodiscard]] ConnectionResult
    connectImpl(Source &src_block_raw, SourcePort &source_port, Destination &dst_block_raw, DestinationPort &destination_port, std::size_t min_buffer_size = 65536, std::int32_t weight = 0,
                std::string_view name = "unnamed edge") {
        static_assert(std::is_same_v<typename SourcePort::value_type, typename DestinationPort::value_type>, "The source port type needs to match the sink port type");

        if (!std::any_of(_blocks.begin(), _blocks.end(), [&](const auto &registered_block) { return registered_block->raw() == std::addressof(src_block_raw); })
            || !std::any_of(_blocks.begin(), _blocks.end(), [&](const auto &registered_block) { return registered_block->raw() == std::addressof(dst_block_raw); })) {
            throw std::runtime_error(fmt::format("Can not connect nodes that are not registered first:\n {}:{} -> {}:{}\n", src_block_raw.name, src_port_index, dst_block_raw.name, dst_port_index));
        }

        auto result = source_port.connect(destination_port);
        if (result == ConnectionResult::SUCCESS) {
            auto *src_block = findBlock(src_block_raw).get();
            auto *dst_block = findBlock(dst_block_raw).get();
            _edges.emplace_back(src_block, src_port_index, dst_block, src_port_index, min_buffer_size, weight, name);
        }

        return result;
    }

    // Just a dummy class that stores the graph and the source node and port
    // to be able to split the connection into two separate calls
    // connect(source) and .to(destination)
    template<typename Source, typename Port, std::size_t src_port_index = 1_UZ>
    struct SourceConnector {
        Graph  &self;
        Source &source;
        Port   &port;

        SourceConnector(Graph &_self, Source &_source, Port &_port) : self(_self), source(_source), port(_port) {}

    private:
        template<typename Destination, typename DestinationPort, std::size_t dst_port_index = meta::invalid_index>
        [[nodiscard]] constexpr auto
        to(Destination &destination, DestinationPort &destination_port) {
            // Not overly efficient as the node doesn't know the graph it belongs to,
            // but this is not a frequent operation and the check is important.
            auto is_block_known = [this](const auto &query_block) {
                return std::any_of(self._blocks.cbegin(), self._blocks.cend(), [&query_block](const auto &known_block) { return known_block->raw() == std::addressof(query_block); });
            };
            if (!is_block_known(source) || !is_block_known(destination)) {
                throw fmt::format("Source {} and/or destination {} do not belong to this graph\n", source.name, destination.name);
            }
            self._connection_definitions.push_back([source_ = &source, source_port = &port, destination = &destination, destination_port = &destination_port](Graph &graph) {
                return graph.connectImpl<src_port_index, dst_port_index>(*source_, *source_port, *destination, *destination_port);
            });
            return ConnectionResult::SUCCESS;
        }

    public:
        template<typename Destination, typename DestinationPort, std::size_t dst_port_index = meta::invalid_index>
        [[nodiscard]] constexpr auto
        to(Destination &destination, DestinationPort Destination::*member_ptr) {
            return to<Destination, DestinationPort, dst_port_index>(destination, std::invoke(member_ptr, destination));
        }

        template<std::size_t dst_port_index, typename Destination>
        [[nodiscard]] constexpr auto
        to(Destination &destination) {
            auto &destination_port = inputPort<dst_port_index>(&destination);
            return to<Destination, std::remove_cvref_t<decltype(destination_port)>, dst_port_index>(destination, destination_port);
        }

        template<fixed_string dst_port_name, typename Destination>
        [[nodiscard]] constexpr auto
        to(Destination &destination) {
            using destination_input_ports        = typename traits::block::input_ports<Destination>;
            constexpr std::size_t dst_port_index = meta::indexForName<dst_port_name, destination_input_ports>();
            if constexpr (dst_port_index == meta::invalid_index) {
                meta::print_types<meta::message_type<"There is no input port with the specified name in this destination node">, Destination, meta::message_type<dst_port_name>,
                                  meta::message_type<"These are the known names:">, traits::block::input_port_names<Destination>, meta::message_type<"Full ports info:">, destination_input_ports>
                        port_not_found_error{};
            }
            return to<dst_port_index, Destination>(destination);
        }

        SourceConnector(const SourceConnector &) = delete;
        SourceConnector(SourceConnector &&)      = delete;
        SourceConnector &
        operator=(const SourceConnector &)
                = delete;
        SourceConnector &
        operator=(SourceConnector &&)
                = delete;
    };

    template<std::size_t src_port_index, typename Source>
    friend auto
    connect(Source &source);

    template<fixed_string src_port_name, typename Source>
    friend auto
    connect(Source &source);

    template<typename Source, typename Port>
    friend auto
    connect(Source &source, Port Source::*member_ptr);

public:
    Graph(Graph &)  = delete;
    Graph(Graph &&) = default;
    Graph()         = default;
    Graph &
    operator=(Graph &)
            = delete;
    Graph &
    operator=(Graph &&)
            = default;

    /**
     * @return a list of all blocks contained in this graph
     * N.B. some 'blocks' may be (sub-)graphs themselves
     */
    [[nodiscard]] std::span<std::unique_ptr<BlockModel>>
    blocks() noexcept {
        return { _blocks };
    }

    /**
     * @return a list of all edges in this graph connecting blocks
     */
    [[nodiscard]] std::span<edge>
    edges() noexcept {
        return { _edges };
    }

    BlockModel &
    addBlock(std::unique_ptr<BlockModel> block) {
        auto &new_block_ref = _blocks.emplace_back(std::move(block));
        new_block_ref->init(progress, ioThreadPool);
        return *new_block_ref.get();
    }

    template<BlockLike TBlock, typename... Args>
    auto &
    emplaceBlock(Args &&...args) { // TODO for review: do we still need this factory method or allow only pmt-map-type constructors (see below)
        static_assert(std::is_same_v<TBlock, std::remove_reference_t<TBlock>>);
        auto &new_block_ref = _blocks.emplace_back(std::make_unique<BlockWrapper<TBlock>>(std::forward<Args>(args)...));
        auto  raw_ref       = static_cast<TBlock *>(new_block_ref->raw());
        raw_ref->init(progress, ioThreadPool);
        return *raw_ref;
    }

    template<BlockLike TBlock>
    auto &
    emplaceBlock(const property_map &initial_settings) {
        static_assert(std::is_same_v<TBlock, std::remove_reference_t<TBlock>>);
        auto &new_block_ref = _blocks.emplace_back(std::make_unique<BlockWrapper<TBlock>>());
        auto  raw_ref       = static_cast<TBlock *>(new_block_ref->raw());
        std::ignore         = raw_ref->settings().set(initial_settings);
        raw_ref->init(progress, ioThreadPool);
        return *raw_ref;
    }

    template<std::size_t src_port_index, typename Source>
    [[nodiscard]] auto
    connect(Source &source) {
        auto &port = outputPort<src_port_index>(&source);
        return Graph::SourceConnector<Source, std::remove_cvref_t<decltype(port)>, src_port_index>(*this, source, port);
    }

    template<fixed_string src_port_name, typename Source>
    [[nodiscard]] auto
    connect(Source &source) {
        using source_output_ports            = typename traits::block::output_ports<Source>;
        constexpr std::size_t src_port_index = meta::indexForName<src_port_name, source_output_ports>();
        if constexpr (src_port_index == meta::invalid_index) {
            meta::print_types<meta::message_type<"There is no output port with the specified name in this source node">, Source, meta::message_type<src_port_name>,
                              meta::message_type<"These are the known names:">, traits::block::output_port_names<Source>, meta::message_type<"Full ports info:">, source_output_ports>
                    port_not_found_error{};
        }
        return connect<src_port_index, Source>(source);
    }

    template<typename Source, typename Port>
    [[nodiscard]] auto
    connect(Source &source, Port Source::*member_ptr) {
        return Graph::SourceConnector<Source, Port>(*this, source, std::invoke(member_ptr, source));
    }

    template<typename Source, typename Destination>
        requires(!std::is_pointer_v<std::remove_cvref_t<Source>> && !std::is_pointer_v<std::remove_cvref_t<Destination>>)
    ConnectionResult
    dynamic_connect(Source &src_block_raw, std::size_t src_port_index, Destination &dst_block_raw, std::size_t dst_port_index, std::size_t min_buffer_size = 65536, std::int32_t weight = 0,
                    std::string_view name = "unnamed edge") {
        auto result = dynamicOutputPort(src_block_raw, src_port_index).connect(dynamicInputPort(dst_block_raw, dst_port_index));
        if (result == ConnectionResult::SUCCESS) {
            auto *src_block = findBlock(src_block_raw).get();
            auto *dst_block = findBlock(dst_block_raw).get();
            _edges.emplace_back(src_block, src_port_index, dst_block, src_port_index, min_buffer_size, weight, name);
        }
        return result;
    }

    const std::vector<std::function<ConnectionResult(Graph &)>> &
    connections() {
        return _connection_definitions;
    }

    void
    clearConnections() {
        _connection_definitions.clear();
    }

    template<typename F> // TODO: F must be constraint by a descriptive concept
    void
    forEachBlock(F &&f) const {
        std::for_each(_blocks.cbegin(), _blocks.cend(), [f](const auto &block_ptr) { f(*block_ptr.get()); });
    }

    template<typename F> // TODO: F must be constraint by a descriptive concept
    void
    forEachEdge(F &&f) const {
        std::for_each(_edges.cbegin(), _edges.cend(), [f](const auto &edge) { f(edge); });
    }
};

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
concept SourceBlockLike = traits::block::can_processOne<TBlock> and traits::block::template output_port_types<TBlock>::size > 0;

static_assert(not SourceBlockLike<int>);

template<typename TBlock>
concept SinkBlockLike = traits::block::can_processOne<TBlock> and traits::block::template input_port_types<TBlock>::size > 0;

static_assert(not SinkBlockLike<int>);

template<SourceBlockLike Left, SinkBlockLike Right, std::size_t OutId, std::size_t InId>
class MergedGraph : public Block<MergedGraph<Left, Right, OutId, InId>, meta::concat<typename traits::block::input_ports<Left>, meta::remove_at<InId, typename traits::block::input_ports<Right>>>,
                                 meta::concat<meta::remove_at<OutId, typename traits::block::output_ports<Left>>, typename traits::block::output_ports<Right>>> {
    static std::atomic_size_t _unique_id_counter;

public:
    const std::size_t unique_id   = _unique_id_counter++;
    const std::string unique_name = fmt::format("MergedGraph<{}:{},{}:{}>#{}", gr::meta::type_name<Left>(), OutId, gr::meta::type_name<Right>(), InId, unique_id);

private:
    // copy-paste from above, keep in sync
    using base = Block<MergedGraph<Left, Right, OutId, InId>, meta::concat<typename traits::block::input_ports<Left>, meta::remove_at<InId, typename traits::block::input_ports<Right>>>,
                       meta::concat<meta::remove_at<OutId, typename traits::block::output_ports<Left>>, typename traits::block::output_ports<Right>>>;

    Left  left;
    Right right;

    // merged_work_chunk_size, that's what friends are for
    friend base;

    template<SourceBlockLike, SinkBlockLike, std::size_t, std::size_t>
    friend class MergedGraph;

    // returns the minimum of all internal max_samples port template parameters
    static constexpr std::size_t
    merged_work_chunk_size() noexcept {
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
        return std::min({ traits::block::input_ports<Right>::template apply<traits::port::max_samples>::value, traits::block::output_ports<Left>::template apply<traits::port::max_samples>::value,
                          left_size, right_size });
    }

    template<std::size_t I>
    constexpr auto
    apply_left(std::size_t offset, auto &&input_tuple) noexcept {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return invokeProcessOneWithOrWithoutOffset(left, offset, std::get<Is>(std::forward<decltype(input_tuple)>(input_tuple))...);
        }(std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    constexpr auto
    apply_right(std::size_t offset, auto &&input_tuple, auto &&tmp) noexcept {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset  = traits::block::input_port_types<Left>::size;
            constexpr std::size_t second_offset = traits::block::input_port_types<Left>::size + sizeof...(Is);
            static_assert(second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return invokeProcessOneWithOrWithoutOffset(right, offset, std::get<first_offset + Is>(std::forward<decltype(input_tuple)>(input_tuple))..., std::forward<decltype(tmp)>(tmp),
                                                       std::get<second_offset + Js>(input_tuple)...);
        }(std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    using TInputPortTypes  = typename traits::block::input_port_types<base>;
    using TOutputPortTypes = typename traits::block::output_port_types<base>;
    using TReturnType      = typename traits::block::return_type<base>;

    constexpr MergedGraph(Left l, Right r) : left(std::move(l)), right(std::move(r)) {}

    // if the left node (source) implements available_samples (a customization point), then pass the call through
    friend constexpr std::size_t
    available_samples(const MergedGraph &self) noexcept
        requires requires(const Left &l) {
            { available_samples(l) } -> std::same_as<std::size_t>;
        }
    {
        return available_samples(self.left);
    }

    template<meta::any_simd... Ts>
        requires traits::block::can_processOne_simd<Left> and traits::block::can_processOne_simd<Right>
    constexpr meta::simdize<TReturnType, meta::simdize_size_v<std::tuple<Ts...>>>
    processOne(std::size_t offset, const Ts &...inputs) {
        static_assert(traits::block::output_port_types<Left>::size == 1, "TODO: SIMD for multiple output ports not implemented yet");
        return apply_right<InId, traits::block::input_port_types<Right>::size() - InId - 1>(offset, std::tie(inputs...),
                                                                                            apply_left<traits::block::input_port_types<Left>::size()>(offset, std::tie(inputs...)));
    }

    constexpr auto
    processOne_simd(std::size_t offset, auto N)
        requires traits::block::can_processOne_simd<Right>
    {
        if constexpr (requires(Left &l) {
                          { l.processOne_simd(offset, N) };
                      }) {
            return invokeProcessOneWithOrWithoutOffset(right, offset, left.processOne_simd(offset, N));
        } else if constexpr (requires(Left &l) {
                                 { l.processOne_simd(N) };
                             }) {
            return invokeProcessOneWithOrWithoutOffset(right, offset, left.processOne_simd(N));
        } else {
            using LeftResult = typename traits::block::return_type<Left>;
            using V          = meta::simdize<LeftResult, N>;
            alignas(stdx::memory_alignment_v<V>) LeftResult tmp[V::size()];
            for (std::size_t i = 0; i < V::size(); ++i) {
                tmp[i] = invokeProcessOneWithOrWithoutOffset(left, offset + i);
            }
            return invokeProcessOneWithOrWithoutOffset(right, offset, V(tmp, stdx::vector_aligned));
        }
    }

    template<typename... Ts>
    // Nicer error messages for the following would be good, but not at the expense of breaking can_processOne_simd.
        requires(TInputPortTypes::template are_equal<std::remove_cvref_t<Ts>...>)
    constexpr TReturnType
    processOne(std::size_t offset, Ts &&...inputs) {
        // if (sizeof...(Ts) == 0) we could call `return processOne_simd(integral_constant<size_t, width>)`. But if
        // the caller expects to process *one* sample (no inputs for the caller to explicitly
        // request simd), and we process more, we risk inconsistencies.
        if constexpr (traits::block::output_port_types<Left>::size == 1) {
            // only the result from the right node needs to be returned
            return apply_right<InId, traits::block::input_port_types<Right>::size() - InId - 1>(offset, std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                                                                                apply_left<traits::block::input_port_types<Left>::size()>(offset, std::forward_as_tuple(
                                                                                                                                                                          std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out  = apply_left<traits::block::input_port_types<Left>::size()>(offset, std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, traits::block::input_port_types<Right>::size() - InId - 1>(offset, std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                                                                                          std::move(std::get<OutId>(left_out)));

            if constexpr (traits::block::output_port_types<Left>::size == 2 && traits::block::output_port_types<Right>::size == 1) {
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)), std::move(right_out));

            } else if constexpr (traits::block::output_port_types<Left>::size == 2) {
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))), std::move(right_out));

            } else if constexpr (traits::block::output_port_types<Right>::size == 1) {
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(right_out));
                }(std::make_index_sequence<OutId>(), std::make_index_sequence<traits::block::output_port_types<Left>::size - OutId - 1>());

            } else {
                return [&]<std::size_t... Is, std::size_t... Js, std::size_t... Ks>(std::index_sequence<Is...>, std::index_sequence<Js...>, std::index_sequence<Ks...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(std::get<Ks>(right_out)...));
                }(std::make_index_sequence<OutId>(), std::make_index_sequence<traits::block::output_port_types<Left>::size - OutId - 1>(), std::make_index_sequence<Right::output_port_types::size>());
            }
        }
    } // end:: processOne

    work::Result
    work(std::size_t requested_work) noexcept {
        return base::work(requested_work);
    }
};

template<SourceBlockLike Left, SinkBlockLike Right, std::size_t OutId, std::size_t InId>
inline std::atomic_size_t MergedGraph<Left, Right, OutId, InId>::_unique_id_counter{ 0_UZ };

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
constexpr auto
mergeByIndex(A &&a, B &&b) -> MergedGraph<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    if constexpr (!std::is_same_v<typename traits::block::output_port_types<std::remove_cvref_t<A>>::template at<OutId>,
                                  typename traits::block::input_port_types<std::remove_cvref_t<B>>::template at<InId>>) {
        gr::meta::print_types<gr::meta::message_type<"OUTPUT_PORTS_ARE:">, typename traits::block::output_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, OutId>,
                              typename traits::block::output_port_types<std::remove_cvref_t<A>>::template at<OutId>,

                              gr::meta::message_type<"INPUT_PORTS_ARE:">, typename traits::block::input_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, InId>,
                              typename traits::block::input_port_types<std::remove_cvref_t<A>>::template at<InId>>{};
    }
    return { std::forward<A>(a), std::forward<B>(b) };
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
constexpr auto
merge(A &&a, B &&b) {
    constexpr int OutIdUnchecked = meta::indexForName<OutName, typename traits::block::output_ports<A>>();
    constexpr int InIdUnchecked  = meta::indexForName<InName, typename traits::block::input_ports<B>>();
    static_assert(OutIdUnchecked != -1);
    static_assert(InIdUnchecked != -1);
    constexpr auto OutId = static_cast<std::size_t>(OutIdUnchecked);
    constexpr auto InId  = static_cast<std::size_t>(InIdUnchecked);
    static_assert(std::same_as<typename traits::block::output_port_types<std::remove_cvref_t<A>>::template at<OutId>,
                               typename traits::block::input_port_types<std::remove_cvref_t<B>>::template at<InId>>,
                  "Port types do not match");
    return MergedGraph<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId>{ std::forward<A>(a), std::forward<B>(b) };
}

#if !DISABLE_SIMD
namespace test { // TODO: move to dedicated tests

struct copy : public Block<copy> {
    PortIn<float>  in;
    PortOut<float> out;

public:
    template<meta::t_or_simd<float> V>
    [[nodiscard]] constexpr V
    processOne(const V &a) const noexcept {
        return a;
    }
};
} // namespace test
#endif
} // namespace gr

#if !DISABLE_SIMD
ENABLE_REFLECTION(gr::test::copy, in, out);
#endif

namespace gr {

#if !DISABLE_SIMD
namespace test {
static_assert(traits::block::input_port_types<copy>::size() == 1);
static_assert(std::same_as<traits::block::return_type<copy>, float>);
static_assert(traits::block::can_processOne_scalar<copy>);
static_assert(traits::block::can_processOne_simd<copy>);
static_assert(traits::block::can_processOne_scalar_with_offset<decltype(mergeByIndex<0, 0>(copy(), copy()))>);
static_assert(traits::block::can_processOne_simd_with_offset<decltype(mergeByIndex<0, 0>(copy(), copy()))>);
static_assert(SourceBlockLike<copy>);
static_assert(SinkBlockLike<copy>);
static_assert(SourceBlockLike<decltype(mergeByIndex<0, 0>(copy(), copy()))>);
static_assert(SinkBlockLike<decltype(mergeByIndex<0, 0>(copy(), copy()))>);
} // namespace test
#endif

/*******************************************************************************************************/
/**************************** end of SIMD-Merged Graph Implementation **********************************/
/*******************************************************************************************************/

// TODO: add nicer enum formatter
inline std::ostream &
operator<<(std::ostream &os, const ConnectionResult &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const PortType &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const PortDirection &value) {
    return os << static_cast<int>(value);
}

template<PortDomainLike T>
inline std::ostream &
operator<<(std::ostream &os, const T &value) {
    return os << value.Name;
}

#if HAVE_SOURCE_LOCATION
inline auto
this_source_location(std::source_location l = std::source_location::current()) {
    return fmt::format("{}:{},{}", l.file_name(), l.line(), l.column());
}
#else
inline auto
this_source_location() {
    return "not yet implemented";
}
#endif // HAVE_SOURCE_LOCATION

} // namespace gr

#endif // include guard
