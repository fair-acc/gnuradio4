#ifndef GRAPH_PROTOTYPE_GRAPH_CONTRACTS_HPP
#define GRAPH_PROTOTYPE_GRAPH_CONTRACTS_HPP

#include <complex>
#include <concepts>
#include <cstdint>
#include <map>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "buffer.hpp"
#include "circular_buffer.hpp"
#include "refl.hpp"

namespace fair {
// clang-format off

namespace helper {
template <class... T>
constexpr bool always_false = false;
}

/**
 * little compile-time string class (N.B. ideally std::string should become constexpr (everything!! ;-)))
 */
template<typename CharT, std::size_t SIZE>
struct fixed_string {
    constexpr static std::size_t N = SIZE;
    CharT _data[N + 1] = {};

    constexpr explicit(false) fixed_string(const CharT (&str)[N + 1]) noexcept {
        if constexpr (N != 0) for (std::size_t i = 0; i < N; ++i) _data[i] = str[i];
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return N; }
    [[nodiscard]] constexpr bool empty() const noexcept { return N == 0; }
    [[nodiscard]] constexpr explicit operator std::string_view() const noexcept { return {_data, N}; }
    [[nodiscard]] constexpr explicit operator std::string() const noexcept { return {_data, N}; }
    [[nodiscard]] operator const char *() const noexcept { return _data; }
    [[nodiscard]] constexpr bool operator==(const fixed_string &other) const noexcept {
        return std::string_view{_data, N} == std::string_view(other);
    }

    template<std::size_t N2>
    [[nodiscard]] friend constexpr bool operator==(const fixed_string &, const fixed_string<CharT, N2> &) { return false; }
};
template<typename CharT, std::size_t N>
fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;

// #### default supported types -- TODO: to be replaced by pmt::pmtv declaration
using supported_type = std::variant<uint8_t, uint32_t, int8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> /*, ...*/>;

enum class connection_result_t { SUCCESS, FAILED };
enum class port_type_t { STREAM, MESSAGE };
enum class port_direction_t { INPUT, OUTPUT, ANY }; // 'ANY' only for query and not to be used for port declarations
enum class port_domain_t { CPU, GPU, NET, FPGA, DSP, MLU };

template<class T>
concept Port = requires(T t, const std::size_t n_items) { // dynamic definitions
    typename T::value_type;
    { t.pmt_type() }                -> std::same_as<supported_type>;
    { t.type() }                    -> std::same_as<port_type_t>;
    { t.direction() }               -> std::same_as<port_direction_t>;
    { t.name() }                    -> std::same_as<std::string_view>;
    { t.resize_buffer(n_items) }    -> std::same_as<connection_result_t>;
    { t.disconnect() }              -> std::same_as<connection_result_t>;
};


class edge;
class dyn_port;
class block;

template<class T>
concept Block = requires(T& t, const std::size_t src_port, T& dst_block, const std::size_t dst_port, port_direction_t direction, dyn_port& port) {
    { t.name() } -> std::same_as<std::string_view>;
    { t.work() } -> std::same_as<void>; // primary function to be called by scheduler
    //{ t.template add_port<port_direction_t::INPUT>(port) } -> std::same_as<dyn_port&>;
    { t.input_ports() } -> std::same_as<std::span<const dyn_port>>;
    { t.output_ports() } -> std::same_as<std::span<const dyn_port>>;
    //{ t.connect(src_port, dst_block, dst_port) } -> std::same_as<std::vector<connection_result_t>>;
};

template<typename T, fixed_string PortName, port_type_t PortType, port_direction_t PortDirection, // TODO: sort default arguments
    std::size_t N_HISTORY = std::dynamic_extent, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, bool OPTIONAL = false,
    gr::Buffer BufferType = gr::circular_buffer<T>>
class port {
    static_assert(PortDirection != port_direction_t::ANY, "ANY reserved for queries and not port direction declarations");
public:
    using value_type = T;
    static constexpr bool IS_INPUT = PortDirection == port_direction_t::INPUT;
private:
    using ReaderType = decltype(std::declval<BufferType>().new_reader());
    using WriterType = decltype(std::declval<BufferType>().new_writer());
    using IoType = std::conditional_t<IS_INPUT, ReaderType, WriterType>;

    std::string _port_name{PortName};
    std::int16_t _priority = 0; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    std::size_t _n_history = N_HISTORY;
    std::size_t _min_samples = MIN_SAMPLES;
    std::size_t _max_samples = MAX_SAMPLES;
    bool _connected = false;

    IoType _ioHandler = new_io_handler();

    constexpr auto new_io_handler() noexcept {
        if constexpr (IS_INPUT) { return BufferType(4096).new_reader();
        } else { return BufferType(4096).new_writer(); }
    }

    void* writer_handler_internal() noexcept {
        if constexpr (!IS_INPUT) {
            static_assert(!IS_INPUT, "only to be used with output ports");
        }
        return static_cast<void*>(std::addressof(_ioHandler));
    }
    bool update_reader_internal(void* buffer_writer_handler_other) noexcept {
        if constexpr (IS_INPUT) {
            static_assert(IS_INPUT, "only to be used with input ports");
        }
        try {
            auto typed_buffer_writer = static_cast<WriterType*>(buffer_writer_handler_other);
            if (typed_buffer_writer == nullptr) {
                return false;
            }
            setBuffer(typed_buffer_writer->buffer());
        } catch (...) {
            // needed for runtime polymorphism
            assert(false && "invalid static_cast from 'void* -> WriterType*' in update_reader_internal()");
            return false;
        }
        return true;
    }
    friend dyn_port;

    public:
        port() = default;
        port(const port&) = delete;
        auto operator=(const port&) = delete;
        constexpr port(std::string port_name, std::int16_t priority = 0, std::size_t n_history = 0,
                       std::size_t min_samples = 0U, std::size_t max_samples = SIZE_MAX) noexcept:
            _port_name(std::move(port_name)), _priority{priority}, _n_history(n_history),
            _min_samples(min_samples), _max_samples(max_samples) {
            static_assert(PortName.empty(), "port name must be exclusively declared via NTTP or constructor parameter");
        }
        constexpr port(port&& other) noexcept {
            *this = std::move(other);
        }
        constexpr port& operator=(port&& other) {
            if (this == &other) {
                return *this;
            }

            std::swap(_port_name, other._port_name);
            std::swap(_priority, other._priority);
            std::swap(_n_history, other._n_history);
            std::swap(_min_samples, other._min_samples);
            std::swap(_max_samples, other._max_samples);
            std::swap(_connected, other._connected);
            std::swap(_ioHandler, other._ioHandler);

            return *this;
        }

        [[nodiscard]] constexpr static port_type_t type() noexcept { return PortType; }
        [[nodiscard]] constexpr static port_direction_t direction() noexcept { return PortDirection; }

        template<bool enable = !PortName.empty()>
        [[nodiscard]] static constexpr std::enable_if_t<enable,decltype(PortName)> static_name() noexcept { return PortName; }

        [[nodiscard]] constexpr supported_type pmt_type() const noexcept { return T(); }
        [[nodiscard]] constexpr std::string_view name() const noexcept {
            if constexpr (!PortName.empty()) {
                return static_cast<std::string_view>(PortName);
            } else {
                return _port_name;
            }
        }

        [[nodiscard]] constexpr static bool optional() noexcept { return OPTIONAL; }
        [[nodiscard]] constexpr std::int16_t priority() const noexcept { return _priority; }

        [[nodiscard]] constexpr static std::size_t available() noexcept { return 0; } //  ↔ maps to Buffer::Buffer[Reader, Writer].available()
        [[nodiscard]] constexpr std::size_t n_history() const noexcept {
            if constexpr (N_HISTORY == std::dynamic_extent) {
                return _n_history;
            } else {
                return N_HISTORY;
            }
        }
        [[nodiscard]] constexpr std::size_t min_buffer_size() const noexcept {
            if constexpr (MIN_SAMPLES == std::dynamic_extent) {
                return _min_samples;
            } else {
                return MIN_SAMPLES;
            }
        }
        [[nodiscard]] constexpr std::size_t max_buffer_size() const noexcept {
            if constexpr (MAX_SAMPLES == std::dynamic_extent) {
                return _max_samples;
            } else {
                return MAX_SAMPLES;
            }
        }

        [[nodiscard]] constexpr connection_result_t resize_buffer(std::size_t min_size) noexcept {
            if constexpr (IS_INPUT) {
                static_assert(IS_INPUT, "can only resize output buffers");
            } else {
                try {
                    _ioHandler =  BufferType(min_size).new_writer();
                } catch (...) {
                    return connection_result_t::FAILED;
                }
            }
            return connection_result_t::SUCCESS;
        }
        [[nodiscard]] BufferType buffer() { return _ioHandler.buffer(); }

        void setBuffer(gr::Buffer auto buffer) {
            if constexpr (IS_INPUT) {
                _ioHandler = std::move(buffer.new_reader());
                _connected = true;
            } else { _ioHandler = std::move(buffer.new_writer()); }
        }

        [[nodiscard]] ReaderType& reader() noexcept {
            if constexpr (PortDirection == port_direction_t::OUTPUT) {
                static_assert(helper::always_false<T>, "reader() not applicable for outputs (yet)");
            }
            return _ioHandler;
        }

        [[nodiscard]] WriterType& writer() noexcept {
            if constexpr (PortDirection == port_direction_t::INPUT) {
                static_assert(helper::always_false<T>, "writer() not applicable for inputs (yet)");
            }
            return _ioHandler;
        }

        [[nodiscard]] connection_result_t disconnect() noexcept {
            if (_connected == false) {
                return connection_result_t::FAILED;
            }
            _ioHandler = new_io_handler();
            _connected = false;
            return connection_result_t::SUCCESS;
        }
};

template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using IN = port<T, PortName, port_type_t::STREAM, port_direction_t::INPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using OUT = port<T, PortName, port_type_t::STREAM, port_direction_t::OUTPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using IN_MSG = port<T, PortName, port_type_t::MESSAGE, port_direction_t::INPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using OUT_MSG = port<T, PortName, port_type_t::MESSAGE, port_direction_t::OUTPUT, MIN_SAMPLES, MAX_SAMPLES>;

static_assert(Port<IN<float, "in">>);
static_assert(Port<decltype(IN<float>("in"))>);
static_assert(Port<OUT<float, "out">>);
static_assert(Port<IN_MSG<float, "in_msg">>);
static_assert(Port<OUT_MSG<float, "out_msg">>);

static_assert(IN<float, "in">::static_name() == fixed_string("in"));
static_assert(requires { IN<float>("in").name(); });

/**
 *  Runtime capable wrapper to be used within a block. It's primary purpose is to allow the runtime
 *  initialisation/connections between blocks that are not in the same compilation unit.
 *  Ownership is defined by if the strongly-typed port P is either passed
 *  a) as an lvalue (i.e. P& -> keep reference), or
 *  b) as an rvalue (P&& -> being moved into dyn_port).
 *
 *  N.B. the intended use is within the node/block interface where there is -- once initialised --
 *  always a strong-reference between the strongly-typed port and it's dyn_port wrapper. I.e no ports
 *  are added or removed after the initialisation and the port life-time is coupled to that of it's
 *  parent block/node.
 */
class dyn_port {
    struct model { // intentionally class-private definition to limit interface exposure and enhance composition
        virtual ~model() = default;

        [[nodiscard]] virtual supported_type pmt_type() const noexcept = 0;
        [[nodiscard]] virtual port_type_t type() const noexcept = 0;
        [[nodiscard]] virtual port_direction_t direction() const noexcept = 0;
        [[nodiscard]] virtual std::string_view name() const noexcept = 0;
        [[nodiscard]] virtual connection_result_t resize_buffer(std::size_t min_size) noexcept = 0;
        [[nodiscard]] virtual connection_result_t disconnect() noexcept = 0;
        [[nodiscard]] virtual connection_result_t connect(dyn_port& dst_port) noexcept = 0;

        // internal runtime polymorphism access
        [[nodiscard]] virtual bool update_reader_internal(void* buffer_other) noexcept = 0;
    };

    std::unique_ptr<model> _accessor;

    template<Port T, bool owning>
    class wrapper final : public model {
        using PortType = std::decay_t<T>;
        std::conditional_t<owning, PortType, PortType&> _value;

        [[nodiscard]] void* writer_handler_internal() noexcept { return _value.writer_handler_internal(); };
        [[nodiscard]] bool update_reader_internal(void* buffer_other) noexcept override {
            return _value.update_reader_internal(buffer_other);
        }

    public:
        explicit constexpr wrapper(T &arg) noexcept : _value{ arg } {
            if constexpr (T::IS_INPUT) {
                static_assert(requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(requires { arg.update_reader_internal(std::declval<void*>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }
        explicit constexpr wrapper(T &&arg) noexcept : _value{ std::move(arg)} {
            if constexpr (T::IS_INPUT) {
                static_assert(requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(requires { arg.update_reader_internal(std::declval<void*>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }
        ~wrapper() override = default;
        [[nodiscard]] constexpr supported_type pmt_type() const noexcept override { return _value.pmt_type(); }
        [[nodiscard]] constexpr port_type_t type() const noexcept override { return _value.type(); }
        [[nodiscard]] constexpr port_direction_t direction() const noexcept override { return _value.direction(); }
        [[nodiscard]] constexpr std::string_view name() const noexcept override {  return _value.name(); }
        [[nodiscard]] connection_result_t resize_buffer(std::size_t min_size) noexcept override {  return _value.resize_buffer(min_size); }
        [[nodiscard]] connection_result_t disconnect() noexcept override {  return _value.disconnect(); }
        [[nodiscard]] connection_result_t connect(dyn_port& dst_port) noexcept override {
            auto src_buffer = _value.writer_handler_internal();
            return dst_port.update_reader_internal(src_buffer) ? connection_result_t::SUCCESS: connection_result_t::FAILED;
        }
    };

    bool update_reader_internal(void* buffer_other)  noexcept { return _accessor->update_reader_internal(buffer_other); }

public:
    using value_type = void; // a sterile port
    constexpr dyn_port() = delete;
    template<Port T>
    constexpr dyn_port(const T &arg) = delete;
    template<Port T>
    explicit constexpr dyn_port(T &arg) noexcept : _accessor{ std::make_unique<wrapper<T, false>>(arg) } {}
    template<Port T>
    explicit constexpr dyn_port(T &&arg) noexcept : _accessor{ std::make_unique<wrapper<T, true>>(std::forward<T>(arg)) } {}

    [[nodiscard]] supported_type pmt_type() const noexcept { return _accessor->pmt_type(); }
    [[nodiscard]] port_type_t type() const noexcept { return _accessor->type(); }
    [[nodiscard]] port_direction_t direction() const noexcept { return _accessor->direction(); }
    [[nodiscard]] std::string_view name() const noexcept { return _accessor->name(); }
    [[nodiscard]] connection_result_t resize_buffer(std::size_t min_size) {
        if (direction() == port_direction_t::OUTPUT) {
            return _accessor->resize_buffer(min_size);
        }
        return connection_result_t::FAILED;
    }
    [[nodiscard]] connection_result_t disconnect() noexcept {  return _accessor->disconnect(); }
    [[nodiscard]] connection_result_t connect(dyn_port& dst_port) noexcept { return _accessor->connect(dst_port); }
};
static_assert(Port<dyn_port>);

class block { // TODO: test block for development purposes to be merged/replaced with impl in graph.hpp
    using setting_map = std::map<std::string, int, std::less<>>;
    using port_direction_t::INPUT;
    using port_direction_t::OUTPUT;

    std::string _name;
    std::vector<dyn_port> _input_ports{};
    std::vector<dyn_port> _output_ports{};
    setting_map _exec_metrics{}; //  →  std::map<string, pmt> → fair scheduling, 'int' stand-in for pmtv

    friend edge;
public:
    explicit block(std::string_view name) : _name(name) {};

    template<typename T = dyn_port>
    void add_port(T&& port) {
        switch (port.direction()) {
            case port_direction_t::INPUT: {
                if (auto portID = port_id<port_direction_t::INPUT>(port.name()); portID.has_value()) {
                    throw std::invalid_argument(fmt::format("port already has a defined input port named '{}' at ID {}",
                                                            port.name(), portID.value()));
                }
                _input_ports.emplace_back(std::forward<T>(port));
            } break;
            case port_direction_t::OUTPUT:
                if (auto portID = port_id<port_direction_t::OUTPUT>(port.name()); portID.has_value()) {
                    throw std::invalid_argument(fmt::format("port already has a defined output port named '{}' at ID {}",
                                                            port.name(), portID.value()));
                }
                _output_ports.emplace_back(std::forward<T>(port));
                break;
            default:
                assert(false && "cannot add port with ANY designation");
        }
    }
    [[nodiscard]] std::string_view name() const noexcept { return _name; }
    template<port_direction_t direction>
    [[nodiscard]] std::optional<dyn_port*> port(std::size_t id) {
        if constexpr (direction == port_direction_t::INPUT) {
            return id < _input_ports.size() ? std::optional{&_input_ports[id] } : std::nullopt;
        } else if constexpr (direction == port_direction_t::OUTPUT) {
            return id < _output_ports.size() ? std::optional{&_output_ports[id] } : std::nullopt;
        }
        return std::nullopt;
    }
    template<port_direction_t direction>
    [[nodiscard]] std::optional<std::size_t> port_id(std::string_view name) const {
        const auto matcher = [&name](const dyn_port& port){ return port.name() == name; };
        if constexpr (direction == port_direction_t::INPUT) {
            const auto it = std::find_if(_input_ports.cbegin(), _input_ports.cend(), matcher);
            return  it != _input_ports.cend() ? std::optional{it - _input_ports.cbegin()} : std::nullopt;
        } else if constexpr (direction == port_direction_t::OUTPUT) {
            const auto it = std::find_if(_output_ports.cbegin(), _output_ports.cend(), matcher);
            return  it != _output_ports.cend() ? std::optional{it - _output_ports.cbegin()} : std::nullopt;
        }
        return std::nullopt;
    }
    template<port_direction_t direction>
    [[nodiscard]] std::optional<dyn_port*> port(std::string_view name) {
        if (const auto id = port_id<direction>(name); id.has_value()) {
            return direction == port_direction_t::INPUT ? &_input_ports[*id] : &_output_ports[*id];
        }
        return std::nullopt;
    }

    void work() noexcept { /* to be implemented in derived classes */ }
    [[nodiscard]] std::span<const dyn_port> input_ports() const noexcept { return _input_ports; }
    [[nodiscard]] std::span<const dyn_port> output_ports() const noexcept { return _output_ports; }


    [[nodiscard]] setting_map &exec_metrics() noexcept { return _exec_metrics; }
    [[nodiscard]] setting_map const &exec_metrics() const noexcept { return _exec_metrics; }

//    { t.connect(src_port, dst_block, dst_port) } -> std::same_as<std::vector<connection_result_t>>;
};
static_assert(Block<block>);


class edge {
    using port_direction_t::INPUT;
    using port_direction_t::OUTPUT;
    std::shared_ptr<block> _src_block;
    std::shared_ptr<block> _dst_block;
    std::size_t _src_port_id;
    std::size_t _dst_port_id;
    int32_t _weight;
    std::string _name; // custom edge name
    bool _connected;

public:
    edge() = delete;
    edge& operator=(const edge&) = delete;
    edge(std::shared_ptr<block> src_block, std::size_t src_port_id, std::shared_ptr<block> dst_block, std::size_t dst_port_id, int32_t weight, std::string_view name) :
            _src_block(src_block), _dst_block(dst_block), _src_port_id(src_port_id), _dst_port_id(dst_port_id), _weight(weight), _name(name) {
        if (!src_block->port<OUTPUT>(_src_port_id)) {
            throw fmt::format("source block '{}' has not output port id {}", _src_block->name(), _src_port_id);
        }
        if (!dst_block->port<INPUT>(_dst_port_id)) {
            throw fmt::format("destination block '{}' has not output port id {}", _dst_block->name(), _dst_port_id);
        }
        const dyn_port& src_port = *src_block->port<OUTPUT>(_src_port_id).value();
        const dyn_port& dst_port = *dst_block->port<INPUT>(_dst_port_id).value();
        if (src_port.pmt_type().index() != dst_port.pmt_type().index()) {
            throw fmt::format("edge({}::{}<{}> -> {}::{}<{}>, weight: {}, name:\"{}\") incompatible to type id='{}'",
                _src_block->name(), src_port.name(), src_port.pmt_type().index(),
                _dst_block->name(), dst_port.name(), dst_port.pmt_type().index(),
                _weight, _name, dst_port.pmt_type().index());
        }
    }
    edge(std::shared_ptr<block> src_block, std::string_view src_port_name, std::shared_ptr<block> dst_block, std::string_view dst_port_name, int32_t weight, std::string_view name) :
            _src_block(src_block), _dst_block(dst_block), _weight(weight), _name(name) {
        const auto src_id = _src_block->port_id<OUTPUT>(src_port_name);
        const auto dst_id = _dst_block->port_id<INPUT>(dst_port_name);
        if (!src_id) {
            throw std::invalid_argument(fmt::format("source block '{}' has not output port '{}'", _src_block->name(), src_port_name));
        }
        if (!dst_id) {
            throw fmt::format("destination block '{}' has not output port '{}'", _dst_block->name(), dst_port_name);
        }
        _src_port_id = src_id.value();
        _dst_port_id = dst_id.value();
        const dyn_port& src_port = *src_block->port<OUTPUT>(_src_port_id).value();
        const dyn_port& dst_port = *dst_block->port<INPUT>(_dst_port_id).value();
        if (src_port.pmt_type().index() != dst_port.pmt_type().index()) {
            throw fmt::format("edge({}::{}<{}> -> {}::{}<{}>, weight: {}, name:\"{}\") incompatible to type id='{}'",
                              _src_block->name(), src_port.name(), src_port.pmt_type().index(),
                              _dst_block->name(), dst_port.name(), dst_port.pmt_type().index(),
                              _weight, _name, dst_port.pmt_type().index());
        }
    }

    [[nodiscard]] constexpr int32_t weight() const noexcept { return _weight; }
    [[nodiscard]] constexpr std::string_view name() const noexcept { return _name; }
    [[nodiscard]] constexpr bool connected() const noexcept { return _connected; }
    [[nodiscard]] connection_result_t connect() const noexcept { return connection_result_t::FAILED; }
    [[nodiscard]] connection_result_t disconnect() const noexcept { return _dst_block->port<INPUT>(_dst_port_id).value()->disconnect(); }
};

[[nodiscard]] connection_result_t add_edge(std::vector<edge>& edges,
                                           std::shared_ptr<block> src_block, std::size_t src_port_id,
                                           std::shared_ptr<block> dst_block, std::size_t dst_port_id,
                                           int32_t weight = 0, std::string_view name = "unnamed edge") noexcept {
    try {
        edges.emplace_back(src_block, src_port_id, dst_block, dst_port_id, weight, name);
        return connection_result_t::SUCCESS;
    } catch (...) {
        // TODO: add logger or other communication of sorts (dynamic failure case)
        return connection_result_t::FAILED;
    }
}

// TODO: alt versions to be completed once ports are integrated into the node
[[nodiscard]] connection_result_t add_edge(std::vector<edge>& edges,
                                           std::shared_ptr<block> src_block, std::string_view src_port_name,
                                           std::shared_ptr<block> dst_block, std::string_view dst_port_name,
                                           int32_t weight = 0, std::string_view name = "unnamed edge") noexcept {
    try {
        edges.emplace_back(src_block, src_port_name, dst_block, dst_port_name, weight, name);
        return connection_result_t::SUCCESS;
    } catch (...) {
        // TODO: add logger or other communication of sorts (dynamic failure case)
        return connection_result_t::FAILED;
    }
    // impl detail:
    // check if port name exists -> get dyn_port
    // call add_edge(... dyn_port&) variant
    // else
    // fail - verbose:
    // a) duplicate names -> recommend to connect by ID
    // b) indicate block::port type mismatches
    return connection_result_t::FAILED;
}

//    collection|edge_base> auto remove_edge(edge|edge_base) → edge auto add_edge(edge<T>) → <tbd>

//    either:-- rstein: need to decide/agree on naming
//    bool connected()
//    auto connect(BufferFactory f = DefaultBufferFactory()) → std::shared> – initialises buffers
//    auto disconnect() → std::shared> – shuts-down buffer

// clang-format on

// TODO: add nicer enum formatter
std::ostream& operator<<(std::ostream& os, const connection_result_t& value) {
    return os << static_cast<int>(value);
}
std::ostream& operator<<(std::ostream& os, const port_type_t& value) {
    return os << static_cast<int>(value);
}
std::ostream& operator<<(std::ostream& os, const port_direction_t& value) {
    return os << static_cast<int>(value);
}
std::ostream& operator<<(std::ostream& os, const port_domain_t& value) {
    return os << static_cast<int>(value);
}
} // namespace fair


#endif // GRAPH_PROTOTYPE_GRAPH_CONTRACTS_HPP
