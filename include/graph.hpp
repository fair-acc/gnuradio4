#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP

#include "block.hpp"
#include "buffer.hpp"
#include "circular_buffer.hpp"
#include "port.hpp"
#include "sequence.hpp"
#include "typelist.hpp"
#include <thread_pool.hpp>

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

namespace fair::graph {

using namespace fair::literals;

class block_model {
protected:
    using dynamic_ports                         = std::vector<fair::graph::dynamic_port>;
    bool                  _dynamic_ports_loaded = false;
    std::function<void()> _dynamic_ports_loader;
    dynamic_ports         _dynamic_input_ports;
    dynamic_ports         _dynamic_output_ports;

    block_model() = default;

public:
    block_model(const block_model &) = delete;
    block_model &
    operator=(const block_model &)
            = delete;
    block_model(block_model &&other) = delete;
    block_model &
    operator=(block_model &&other)
            = delete;

    void
    init_dynamic_ports() const {
        if (!_dynamic_ports_loaded) _dynamic_ports_loader();
    }

    fair::graph::dynamic_port &
    dynamic_input_port(std::size_t index) {
        init_dynamic_ports();
        return _dynamic_input_ports.at(index);
    }

    fair::graph::dynamic_port &
    dynamic_output_port(std::size_t index) {
        init_dynamic_ports();
        return _dynamic_output_ports.at(index);
    }

    [[nodiscard]] auto
    dynamic_input_ports_size() const {
        init_dynamic_ports();
        return _dynamic_input_ports.size();
    }

    [[nodiscard]] auto
    dynamic_output_ports_size() const {
        init_dynamic_ports();
        return _dynamic_output_ports.size();
    }

    virtual ~block_model() = default;

    /**
     * @brief to be called by scheduler->graph to initialise block
     */
    virtual void
    init(std::shared_ptr<gr::Sequence> progress, std::shared_ptr<fair::thread_pool::BasicThreadPool> ioThreadPool)
            = 0;

    /**
     * @brief returns scheduling hint that invoking the work(...) function may block on IO or system-calls
     */
    [[nodiscard]] virtual constexpr bool
    is_blocking() const noexcept
            = 0;

    /**
     * @brief number of available readable samples at the block's input ports
     */
    [[nodiscard]] virtual constexpr std::size_t
    available_input_samples(std::vector<std::size_t> &) const noexcept
            = 0;

    /**
     * @brief number of available writable samples at the block's output ports
     */
    [[nodiscard]] virtual constexpr std::size_t
    available_output_samples(std::vector<std::size_t> &) const noexcept
            = 0;

    /**
     * @brief user defined name
     */
    [[nodiscard]] virtual std::string_view
    name() const
            = 0;

    /**
     * @brief the type of the block as a string
     */
    [[nodiscard]] virtual std::string_view
    type_name() const
            = 0;

    /**
     * @brief user-defined name
     * N.B. may not be unique -> ::unique_name
     */
    virtual void
    set_name(std::string name) noexcept
            = 0;

    /**
     * @brief used to store non-graph-processing information like UI block position etc.
     */
    [[nodiscard]] virtual property_map &
    meta_information() noexcept
            = 0;

    [[nodiscard]] virtual const property_map &
    meta_information() const
            = 0;

    /**
     * @brief process-wide unique name
     * N.B. can be used to disambiguate in case user provided the same 'name()' for several blocks.
     */
    [[nodiscard]] virtual std::string_view
    unique_name() const
            = 0;

    [[nodiscard]] virtual settings_base &
    settings() const
            = 0;

    [[nodiscard]] virtual work_return_t
    work(std::size_t requested_work)
            = 0;

    [[nodiscard]] virtual void *
    raw() = 0;
};

template<BlockType T>
class block_wrapper : public block_model {
private:
    static_assert(std::is_same_v<T, std::remove_reference_t<T>>);
    T           _block;
    std::string _type_name = fair::meta::type_name<T>();

    [[nodiscard]] constexpr const auto &
    block_ref() const noexcept {
        if constexpr (requires { *_block; }) {
            return *_block;
        } else {
            return _block;
        }
    }

    [[nodiscard]] constexpr auto &
    block_ref() noexcept {
        if constexpr (requires { *_block; }) {
            return *_block;
        } else {
            return _block;
        }
    }

    void
    create_dynamic_ports_loader() {
        _dynamic_ports_loader = [this] {
            if (_dynamic_ports_loaded) return;

            using Block        = std::remove_cvref_t<decltype(block_ref())>;

            auto register_port = []<typename PortOrCollection>(PortOrCollection &port_or_collection, auto &where) {
                auto process_port = [&where]<typename Port>(Port &port) { where.push_back(fair::graph::dynamic_port(port, dynamic_port::non_owned_reference_tag{})); };

                if constexpr (traits::port::is_port_v<PortOrCollection>) {
                    process_port(port_or_collection);

                } else {
                    for (auto &port : port_or_collection) {
                        process_port(port);
                    }
                }
            };

            constexpr std::size_t input_port_count = fair::graph::traits::block::template input_port_types<Block>::size;
            [this, register_port]<std::size_t... Is>(std::index_sequence<Is...>) {
                (register_port(fair::graph::input_port<Is>(&block_ref()), this->_dynamic_input_ports), ...);
            }(std::make_index_sequence<input_port_count>());

            constexpr std::size_t output_port_count = fair::graph::traits::block::template output_port_types<Block>::size;
            [this, register_port]<std::size_t... Is>(std::index_sequence<Is...>) {
                (register_port(fair::graph::output_port<Is>(&block_ref()), this->_dynamic_output_ports), ...);
            }(std::make_index_sequence<output_port_count>());

            static_assert(input_port_count + output_port_count > 0);
            _dynamic_ports_loaded = true;
        };
    }

public:
    block_wrapper(const block_wrapper &other) = delete;
    block_wrapper(block_wrapper &&other)      = delete;
    block_wrapper &
    operator=(const block_wrapper &other)
            = delete;
    block_wrapper &
    operator=(block_wrapper &&other)
            = delete;

    ~block_wrapper() override = default;

    block_wrapper() { create_dynamic_ports_loader(); }

    template<typename Arg>
        requires(!std::is_same_v<std::remove_cvref_t<Arg>, T>)
    explicit block_wrapper(Arg &&arg) : _block(std::forward<Arg>(arg)) {
        create_dynamic_ports_loader();
    }

    template<typename... Args>
        requires(sizeof...(Args) > 1)
    explicit block_wrapper(Args &&...args) : _block{ std::forward<Args>(args)... } {
        create_dynamic_ports_loader();
    }

    explicit block_wrapper(std::initializer_list<std::pair<const std::string, pmtv::pmt>> init_parameter) : _block{ std::move(init_parameter) } { create_dynamic_ports_loader(); }

    void
    init(std::shared_ptr<gr::Sequence> progress, std::shared_ptr<fair::thread_pool::BasicThreadPool> ioThreadPool) override {
        return block_ref().init(progress, ioThreadPool);
    }

    [[nodiscard]] constexpr work_return_t
    work(std::size_t requested_work = std::numeric_limits<std::size_t>::max()) override {
        return block_ref().work(requested_work);
    }

    [[nodiscard]] constexpr bool
    is_blocking() const noexcept override {
        return block_ref().is_blocking();
    }

    [[nodiscard]] constexpr std::size_t
    available_input_samples(std::vector<std::size_t> &data) const noexcept override {
        return block_ref().available_input_samples(data);
    }

    [[nodiscard]] constexpr std::size_t
    available_output_samples(std::vector<std::size_t> &data) const noexcept override {
        return block_ref().available_output_samples(data);
    }

    [[nodiscard]] std::string_view
    name() const override {
        return block_ref().name;
    }

    void
    set_name(std::string name) noexcept override {
        block_ref().name = std::move(name);
    }

    [[nodiscard]] std::string_view
    type_name() const override {
        return _type_name;
    }

    [[nodiscard]] property_map &
    meta_information() noexcept override {
        return block_ref().meta_information;
    }

    [[nodiscard]] const property_map &
    meta_information() const noexcept override {
        return block_ref().meta_information;
    }

    [[nodiscard]] std::string_view
    unique_name() const override {
        return block_ref().unique_name;
    }

    [[nodiscard]] settings_base &
    settings() const override {
        return block_ref().settings();
    }

    [[nodiscard]] void *
    raw() override {
        return std::addressof(block_ref());
    }
};

class edge {
public: // TODO: consider making this private and to use accessors (that can be safely used by users)
    using port_direction_t::INPUT;
    using port_direction_t::OUTPUT;
    block_model *_src_block;
    block_model *_dst_block;
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

    edge(block_model *src_block, std::size_t src_port_index, block_model *dst_block, std::size_t dst_port_index, std::size_t min_buffer_size, std::int32_t weight, std::string_view name)
        : _src_block(src_block), _dst_block(dst_block), _src_port_index(src_port_index), _dst_port_index(dst_port_index), _min_buffer_size(min_buffer_size), _weight(weight), _name(name) {}

    [[nodiscard]] constexpr const block_model &
    src_block() const noexcept {
        return *_src_block;
    }

    [[nodiscard]] constexpr const block_model &
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

struct graph {
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::Sequence> progress                           = std::make_shared<gr::Sequence>();
    alignas(hardware_destructive_interference_size) std::shared_ptr<fair::thread_pool::BasicThreadPool> ioThreadPool = std::make_shared<fair::thread_pool::BasicThreadPool>(
            "graph_thread_pool", fair::thread_pool::TaskType::IO_BOUND, 2_UZ, std::numeric_limits<uint32_t>::max());

private:
    std::vector<std::function<connection_result_t(graph &)>> _connection_definitions;
    std::vector<std::unique_ptr<block_model>>                _blocks;
    std::vector<edge>                                        _edges;

    template<typename Block>
    std::unique_ptr<block_model> &
    find_block(Block &what) {
        static_assert(!std::is_pointer_v<std::remove_cvref_t<Block>>);
        auto it = [&, this] {
            if constexpr (std::is_same_v<Block, block_model>) {
                return std::find_if(_blocks.begin(), _blocks.end(), [&](const auto &block) { return block.get() == &what; });
            } else {
                return std::find_if(_blocks.begin(), _blocks.end(), [&](const auto &block) { return block->raw() == &what; });
            }
        }();

        if (it == _blocks.end()) throw std::runtime_error(fmt::format("No such block in this graph"));
        return *it;
    }

    template<typename Block>
    [[nodiscard]] dynamic_port &
    dynamic_output_port(Block &block, std::size_t index) {
        return find_block(block)->dynamic_output_port(index);
    }

    template<typename Block>
    [[nodiscard]] dynamic_port &
    dynamic_input_port(Block &block, std::size_t index) {
        return find_block(block)->dynamic_input_port(index);
    }

    template<std::size_t src_port_index, std::size_t dst_port_index, typename Source, typename SourcePort, typename Destination, typename DestinationPort>
    [[nodiscard]] connection_result_t
    connect_impl(Source &src_block_raw, SourcePort &source_port, Destination &dst_block_raw, DestinationPort &destination_port, std::size_t min_buffer_size = 65536, std::int32_t weight = 0,
                 std::string_view name = "unnamed edge") {
        static_assert(std::is_same_v<typename SourcePort::value_type, typename DestinationPort::value_type>, "The source port type needs to match the sink port type");

        if (!std::any_of(_blocks.begin(), _blocks.end(), [&](const auto &registered_block) { return registered_block->raw() == std::addressof(src_block_raw); })
            || !std::any_of(_blocks.begin(), _blocks.end(), [&](const auto &registered_block) { return registered_block->raw() == std::addressof(dst_block_raw); })) {
            throw std::runtime_error(fmt::format("Can not connect blocks that are not registered first:\n {}:{} -> {}:{}\n", src_block_raw.name, src_port_index, dst_block_raw.name, dst_port_index));
        }

        auto result = source_port.connect(destination_port);
        if (result == connection_result_t::SUCCESS) {
            auto *src_block = find_block(src_block_raw).get();
            auto *dst_block = find_block(dst_block_raw).get();
            _edges.emplace_back(src_block, src_port_index, dst_block, src_port_index, min_buffer_size, weight, name);
        }

        return result;
    }

    // Just a dummy class that stores the graph and the source block and port
    // to be able to split the connection into two separate calls
    // connect(source) and .to(destination)
    template<typename Source, typename Port, std::size_t src_port_index = 1_UZ>
    struct source_connector {
        graph  &self;
        Source &source;
        Port   &port;

        source_connector(graph &_self, Source &_source, Port &_port) : self(_self), source(_source), port(_port) {}

    private:
        template<typename Destination, typename DestinationPort, std::size_t dst_port_index = meta::invalid_index>
        [[nodiscard]] constexpr auto
        to(Destination &destination, DestinationPort &destination_port) {
            // Not overly efficient as the block doesn't know the graph it belongs to,
            // but this is not a frequent operation and the check is important.
            auto is_block_known = [this](const auto &query_block) {
                return std::any_of(self._blocks.cbegin(), self._blocks.cend(), [&query_block](const auto &known_block) { return known_block->raw() == std::addressof(query_block); });
            };
            if (!is_block_known(source) || !is_block_known(destination)) {
                throw fmt::format("Source {} and/or destination {} do not belong to this graph\n", source.name, destination.name);
            }
            self._connection_definitions.push_back([source = &source, source_port = &port, destination = &destination, destination_port = &destination_port](graph &graph) {
                return graph.connect_impl<src_port_index, dst_port_index>(*source, *source_port, *destination, *destination_port);
            });
            return connection_result_t::SUCCESS;
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
            auto &destination_port = input_port<dst_port_index>(&destination);
            return to<Destination, std::remove_cvref_t<decltype(destination_port)>, dst_port_index>(destination, destination_port);
        }

        template<fixed_string dst_port_name, typename Destination>
        [[nodiscard]] constexpr auto
        to(Destination &destination) {
            using destination_input_ports        = typename traits::block::input_ports<Destination>;
            constexpr std::size_t dst_port_index = meta::indexForName<dst_port_name, destination_input_ports>();
            if constexpr (dst_port_index == meta::invalid_index) {
                meta::print_types<meta::message_type<"There is no input port with the specified name in this destination block">, Destination, meta::message_type<dst_port_name>,
                                  meta::message_type<"These are the known names:">, traits::block::input_port_names<Destination>, meta::message_type<"Full ports info:">, destination_input_ports>
                        port_not_found_error{};
            }
            return to<dst_port_index, Destination>(destination);
        }

        source_connector(const source_connector &) = delete;
        source_connector(source_connector &&)      = delete;
        source_connector &
        operator=(const source_connector &)
                = delete;
        source_connector &
        operator=(source_connector &&)
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
    graph(graph &)  = delete;
    graph(graph &&) = default;
    graph()         = default;
    graph &
    operator=(graph &)
            = delete;
    graph &
    operator=(graph &&)
            = default;

    /**
     * @return a list of all blocks contained in this graph
     * N.B. some 'blocks' may be (sub-)graphs themselves
     */
    [[nodiscard]] std::span<std::unique_ptr<block_model>>
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

    block_model &
    add_block(std::unique_ptr<block_model> block) {
        auto &new_block_ref = _blocks.emplace_back(std::move(block));
        new_block_ref->init(progress, ioThreadPool);
        return *new_block_ref.get();
    }

    template<BlockType Block, typename... Args>
    auto &
    make_block(Args &&...args) { // TODO for review: do we still need this factory method or allow only pmt-map-type constructors (see below)
        static_assert(std::is_same_v<Block, std::remove_reference_t<Block>>);
        auto &new_block_ref = _blocks.emplace_back(std::make_unique<block_wrapper<Block>>(std::forward<Args>(args)...));
        auto  raw_ref       = static_cast<Block *>(new_block_ref->raw());
        raw_ref->init(progress, ioThreadPool);
        return *raw_ref;
    }

    template<BlockType Block>
    auto &
    make_block(const property_map &initial_settings) {
        static_assert(std::is_same_v<Block, std::remove_reference_t<Block>>);
        auto &new_block_ref = _blocks.emplace_back(std::make_unique<block_wrapper<Block>>());
        auto  raw_ref       = static_cast<Block *>(new_block_ref->raw());
        std::ignore         = raw_ref->settings().set(initial_settings);
        raw_ref->init(progress, ioThreadPool);
        return *raw_ref;
    }

    template<std::size_t src_port_index, typename Source>
    [[nodiscard]] auto
    connect(Source &source) {
        auto &port = output_port<src_port_index>(&source);
        return graph::source_connector<Source, std::remove_cvref_t<decltype(port)>, src_port_index>(*this, source, port);
    }

    template<fixed_string src_port_name, typename Source>
    [[nodiscard]] auto
    connect(Source &source) {
        using source_output_ports            = typename traits::block::output_ports<Source>;
        constexpr std::size_t src_port_index = meta::indexForName<src_port_name, source_output_ports>();
        if constexpr (src_port_index == meta::invalid_index) {
            meta::print_types<meta::message_type<"There is no output port with the specified name in this source block">, Source, meta::message_type<src_port_name>,
                              meta::message_type<"These are the known names:">, traits::block::output_port_names<Source>, meta::message_type<"Full ports info:">, source_output_ports>
                    port_not_found_error{};
        }
        return connect<src_port_index, Source>(source);
    }

    template<typename Source, typename Port>
    [[nodiscard]] auto
    connect(Source &source, Port Source::*member_ptr) {
        return graph::source_connector<Source, Port>(*this, source, std::invoke(member_ptr, source));
    }

    template<typename Source, typename Destination>
        requires(!std::is_pointer_v<std::remove_cvref_t<Source>> && !std::is_pointer_v<std::remove_cvref_t<Destination>>)
    connection_result_t
    dynamic_connect(Source &src_block_raw, std::size_t src_port_index, Destination &dst_block_raw, std::size_t dst_port_index, std::size_t min_buffer_size = 65536, std::int32_t weight = 0,
                    std::string_view name = "unnamed edge") {
        auto result = dynamic_output_port(src_block_raw, src_port_index).connect(dynamic_input_port(dst_block_raw, dst_port_index));
        if (result == connection_result_t::SUCCESS) {
            auto *src_block = find_block(src_block_raw).get();
            auto *dst_block = find_block(dst_block_raw).get();
            _edges.emplace_back(src_block, src_port_index, dst_block, src_port_index, min_buffer_size, weight, name);
        }
        return result;
    }

    const std::vector<std::function<connection_result_t(graph &)>> &
    connection_definitions() {
        return _connection_definitions;
    }

    void
    clear_connection_definitions() {
        _connection_definitions.clear();
    }

    template<typename F>
    void
    for_each_block(F &&f) const {
        std::for_each(_blocks.cbegin(), _blocks.cend(), [f](const auto &block_ptr) { f(*block_ptr.get()); });
    }

    template<typename F>
    void
    for_each_edge(F &&f) const {
        std::for_each(_edges.cbegin(), _edges.cend(), [f](const auto &edge) { f(edge); });
    }
};

// TODO: add nicer enum formatter
inline std::ostream &
operator<<(std::ostream &os, const connection_result_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_type_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_direction_t &value) {
    return os << static_cast<int>(value);
}

template<PortDomainType T>
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

} // namespace fair::graph

#endif // include guard
