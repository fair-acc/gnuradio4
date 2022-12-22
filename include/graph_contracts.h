#ifndef GRAPH_PROTOTYPE_GRAPH_CONTRACTS_HPP
#define GRAPH_PROTOTYPE_GRAPH_CONTRACTS_HPP

#include <concepts>
#include <cstdint>
#include <map>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "buffer.hpp"
#include "circular_buffer.hpp"
#include "refl.hpp"

namespace fair {
// clang-format off
    //template<std::size_t N>
    // using fixed_string = refl::const_string<N>;

namespace helper {
template <class... T>
constexpr bool always_false = false;
}

/**
 * little compile-time string class (N.B. ideally std::string should become constexpr (everything!! ;-)))
 */
template<typename CharT, std::size_t N>
struct fixed_string {
    constexpr static std::size_t _size = N;
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

enum class port_type_t { STREAM, MESSAGE };
enum class port_direction_t { INPUT, OUTPUT };
enum class port_domain_t { CPU, GPU, NET, FPGA, DSP, MLU };

template<class T>
concept Port = requires(T t, const std::size_t n_items) {
    //{ T::value_type}   -> std::same_as<std::span<const gr::util::value_type_t<T>>>;
    { T::type() }      -> std::same_as<port_type_t>;
    { T::direction() } -> std::same_as<port_direction_t>;
    { t.name() };//    -> std::same_as<std::string_view>;
} or requires(T t, const std::size_t n_items) {
    //{ T::value_type}   -> std::same_as<std::span<const gr::util::value_type_t<T>>>;
    { t.type() }       -> std::same_as<port_type_t>;
    { t.direction() }  -> std::same_as<port_direction_t>;
    { t.name() }       -> std::same_as<std::string_view>;
};


template<port_type_t PortType, port_direction_t PortDirection, typename T, // TODO: sort default arguments
    fixed_string PortName, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent,
bool OPTIONAL = false>
class port {
    static constexpr bool is_input = PortDirection == port_direction_t::INPUT;
    using BufferType = gr::circular_buffer<T>;
    using ReaderType = decltype(std::declval<BufferType>().new_reader());
    using WriterType = decltype(std::declval<BufferType>().new_writer());
    using IoType = std::conditional_t<is_input, ReaderType, WriterType>;
    using setting_map = std::map<std::string, int, std::less<>>;

    const std::string _port_name = "";
    const std::int16_t _priority = 0; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    setting_map _exec_metrics{}; //  →  std::map<string, pmt> → fair scheduling, 'int' stand-in for pmtv


    BufferType _buffer = init_buffer(4096);
    IoType _ioHandler = get_handler();

    constexpr auto get_handler() noexcept {
        if constexpr (is_input) { return _buffer.new_reader();
        } else { return _buffer.new_writer(); }
    }
    constexpr BufferType init_buffer(std::size_t min_size) noexcept {
        auto buffer = BufferType(min_size);
        return buffer;
    }

    public:
        using value_type = T;
        constexpr port() noexcept : _port_name(""), _priority{0} { }
        constexpr port(std::string_view port_name, std::int16_t priority = 0) noexcept:
            _port_name(std::string(port_name)), _priority{priority} {
            static_assert(PortName.empty(), "port name must be exclusively declared via NTTP or constructor parameter");
        }

        [[nodiscard]] constexpr static port_type_t type() noexcept { return PortType; }
        [[nodiscard]] constexpr static port_direction_t direction() noexcept { return PortDirection; }

        template<bool enable = !PortName.empty()>
        [[nodiscard]] static constexpr std::enable_if_t<enable,decltype(PortName)> static_name() noexcept { return PortName; }

        [[nodiscard]] constexpr std::string_view name() const noexcept {
            if constexpr (PortName.empty()) {
                return static_cast<std::string_view>(PortName);
            } else {
                return _port_name;
            }
        }

        [[nodiscard]] constexpr static bool optional() noexcept { return OPTIONAL; }
        [[nodiscard]] constexpr std::int16_t priority() const noexcept { return _priority; }

        [[nodiscard]] constexpr static std::size_t available() noexcept { return 0; } //  ↔ maps to Buffer::Buffer[Reader, Writer].available()
        [[nodiscard]] constexpr static std::size_t min_buffer_size() noexcept { return MIN_SAMPLES; }
        [[nodiscard]] constexpr static std::size_t max_buffer_size() noexcept { return MAX_SAMPLES; }
        [[nodiscard]] setting_map &exec_metrics() noexcept { return _exec_metrics; }
        [[nodiscard]] setting_map const &exec_metrics() const noexcept { return _exec_metrics; }

        [[nodiscard]] BufferType& get_buffer() { return _buffer; }

        void setBuffer(gr::Buffer auto& buffer) { // to be refactored/initialised based on dependents (via edge<T>)
            if constexpr (is_input) { _ioHandler = std::move(buffer.new_reader());
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
                static_assert(helper::always_false<T>, "writer() not applicable for outputs (yet)");
            }
            return _ioHandler;
        }

};

template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using IN = port<port_type_t::STREAM, port_direction_t::INPUT, T, PortName, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using OUT = port<port_type_t::STREAM, port_direction_t::OUTPUT, T, PortName, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using IN_MSG = port<port_type_t::MESSAGE, port_direction_t::INPUT, T, PortName, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using OUT_MSG = port<port_type_t::MESSAGE, port_direction_t::OUTPUT, T, PortName, MIN_SAMPLES, MAX_SAMPLES>;

static_assert(Port<IN<float, "in">>);
static_assert(Port<decltype(IN<float>("in"))>);
static_assert(Port<OUT<float, "out">>);
static_assert(Port<IN_MSG<float, "in_msg">>);
static_assert(Port<OUT_MSG<float, "out_msg">>);

static_assert(IN<float, "in">::static_name() == fixed_string("in"));
static_assert(requires { IN<float>("in").name(); });


class dyn_port {
    struct model {
        virtual ~model() = default;

        virtual port_type_t type() const noexcept = 0;
        virtual port_direction_t direction() const noexcept = 0;
        virtual std::string_view name() const noexcept = 0;
    };

    std::unique_ptr<model> _accessor;

    template<Port T>
    class wrapper final : public model {
        using PortType = std::decay_t<T>;
        PortType _value;

    public:
        template<Port P>
        explicit constexpr wrapper(P &&arg) noexcept : _value{ std::forward<decltype(arg)>(arg) } {}
        ~wrapper() override = default;
        [[nodiscard]] constexpr port_type_t type() const noexcept override { return T::type(); }
        [[nodiscard]] constexpr port_direction_t direction() const noexcept override { return T::direction(); }
        [[nodiscard]] constexpr std::string_view name() const noexcept override {  return _value.name(); }
    };

public:
    template<Port T>
    explicit constexpr dyn_port(T &&arg) noexcept : _accessor{ std::make_unique<wrapper<T>>(std::forward<T>(arg)) } {}

    [[nodiscard]] port_type_t type() const noexcept { return _accessor->type(); }
    [[nodiscard]] port_direction_t direction() const noexcept { return _accessor->direction(); }
    [[nodiscard]] std::string_view name() const noexcept { return _accessor->name(); }
};

static_assert(Port<dyn_port>);

//    Type: STREAM or MESSAGE -- rstein: do we need to distinguish between these, i.e. they
//    essentially differ only by their 'synch' vs. 'async' behaviour template<name, T,
//    port_direction, port_domain_t, minSize, <sched_info>, ...> port() = default; auto edges() →
//    collection|edge_base> auto remove_edge(edge|edge_base) → edge auto add_edge(edge<T>) → <tbd>

//    either:-- rstein: need to decide/agree on naming
//    bool connected()
//    auto connect(BufferFactory f = DefaultBufferFactory()) → std::shared> – initialises buffers
//    auto disconnect() → std::shared> – shuts-down buffer
//    or: -- rstein: need to decide/agree on naming
//    bool active()
//    auto activate(BufferFactory f = DefaultBufferFactory()) → std::shared> – initialises buffers
//    auto deactivate() → std::shared> – shuts-down buffer
//    auto get_reader() → Buffer::BufferReader → PR#6348
//    auto get_writer() → Buffer::BufferWriter → PR#6348

// clang-format on
} // namespace fair

#endif // GRAPH_PROTOTYPE_GRAPH_CONTRACTS_HPP
