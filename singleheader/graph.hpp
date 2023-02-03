#ifndef GNURADIO_NODE_HPP
#define GNURADIO_NODE_HPP

#include <map>


#include <vir/simd.h>
#include <fmt/format.h>

namespace fair::graph {

namespace stdx = vir::stdx;
using fair::meta::fixed_string;

enum class work_return_t {
    ERROR = -100, /// error occurred in the work function
    INSUFFICIENT_OUTPUT_ITEMS =
        -3, /// work requires a larger output buffer to produce output
    INSUFFICIENT_INPUT_ITEMS =
        -2, /// work requires a larger input buffer to produce output
    DONE =
        -1, /// this block has completed its processing and the flowgraph should be done
    OK = 0, /// work call was successful and return values in i/o structs are valid
    CALLBACK_INITIATED =
        1, /// rather than blocking in the work function, the block will call back to the
           /// parent interface when it is ready to be called again
};

namespace work_strategies {
template<typename Self>
static auto
inputs_status(Self &self) noexcept {
    bool              at_least_one_input_has_data = false;
    const std::size_t available_values_count      = [&self, &at_least_one_input_has_data]() {
        if constexpr (Self::input_ports::size > 0) {
            const auto availableForPort = [&at_least_one_input_has_data]<typename Port>(Port &port) noexcept {
                const std::size_t available = port.reader().available();
                if (available > 0LU) at_least_one_input_has_data = true;
                if (available < port.min_buffer_size()) {
                    return 0LU;
                } else {
                    return std::min(available, port.max_buffer_size());
                }
            };

            const auto availableInAll = std::apply(
                    [&availableForPort] (auto&... input_port) {
                        return meta::safe_min(availableForPort(input_port)...);
                    },
                    input_ports(&self));

            if (availableInAll < self.min_samples()) {
                return 0LU;
            } else {
                return std::min(availableInAll, self.max_samples());
            }
        } else {
            (void) self;
            return std::size_t{ 1 };
        }
    }();

    struct result {
        bool at_least_one_input_has_data;
       std::size_t available_values_count;
    };

    return result {
        .at_least_one_input_has_data = at_least_one_input_has_data,
        .available_values_count = available_values_count
    };
}

template<typename Self>
static auto
write_to_outputs(Self& self, std::size_t available_values_count, auto& writers_tuple) {
    if constexpr (Self::output_ports::size > 0) {
        meta::tuple_for_each([available_values_count] (auto& output_port, auto& writer) {
                output_port.writer().publish(writer.second, available_values_count);
                },
                output_ports(&self), writers_tuple);
    }
}

template<typename Self>
static bool
consume_readers(Self& self, std::size_t available_values_count) {
    bool success = true;
    if constexpr (Self::input_ports::size > 0) {
        std::apply([available_values_count, &success] (auto&... input_port) {
                ((success = success && input_port.reader().consume(available_values_count)), ...);
            }, input_ports(&self));
    }
    return success;
}

struct read_many_and_publish_many {

    template<typename Self>
    static work_return_t
    work(Self &self) noexcept {
        // Capturing structured bindings does not work in Clang...
        auto inputs_status = work_strategies::inputs_status(self);

        if (inputs_status.available_values_count == 0) {
            return inputs_status.at_least_one_input_has_data ? work_return_t::INSUFFICIENT_INPUT_ITEMS : work_return_t::DONE;
        }

        bool all_writers_available = std::apply([inputs_status](auto&... output_port) {
                return ((output_port.writer().available() >= inputs_status.available_values_count) && ... && true);
            }, output_ports(&self));

        if (!all_writers_available) {
            return work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
        }

        auto input_spans = meta::tuple_transform([inputs_status](auto& input_port) {
                return input_port.reader().get(inputs_status.available_values_count);
            }, input_ports(&self));

        auto writers_tuple = meta::tuple_transform([inputs_status](auto& output_port) {
                return output_port.writer().get(inputs_status.available_values_count);
            }, output_ports(&self));

        // TODO: check here whether a process_one(...) or a bulk access process has been defined, cases:
        // case 1a: N-in->N-out -> process_one(...) -> auto-handling of streaming tags
        // case 1b: N-in->N-out -> process_bulk(<ins...>, <outs...>) -> auto-handling of streaming tags
        // case 2a: N-in->M-out -> process_bulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N)
        // case 2b: N-in->M-out -> process_bulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling
        // case 3:  N-in->M-out -> work() N,M arbitrary -> used need to handle the full logic (e.g. PLL algo)
        // case 4:  Python -> map to cases 1-3 and/or dedicated callback
        // special cases:
        // case sources: HW triggered vs. generating data per invocation (generators via Port::MIN)
        // case sinks: HW triggered vs. fixed-size consumer (may block/never finish for insufficient input data and fixed Port::MIN>0)
        for (std::size_t i = 0; i < inputs_status.available_values_count; ++i) {
            auto results = std::apply([&self, &input_spans, i](auto&... input_span) {
                    return meta::invoke_void_wrapped([&self]<typename... Args>(Args... args) { return self.process_one(std::forward<Args>(args)...); }, input_span[i]...);
                }, input_spans);

            if constexpr (std::is_same_v<decltype(results), meta::dummy_t>) {
                // process_one returned void

            } else if constexpr (requires { std::get<0>(results); }) {
                static_assert(std::tuple_size_v<decltype(results)> == Self::output_ports::size);

                meta::tuple_for_each(
                        [i] (auto& writer, auto& result) {
                            writer.first/*data*/[i] = std::move(result); },
                        writers_tuple, results);

            } else {
                static_assert(Self::output_ports::size == 1);
                std::get<0>(writers_tuple).first /*data*/[i] = std::move(results);
            }
        }

        write_to_outputs(self, inputs_status.available_values_count, writers_tuple);

        const bool success = consume_readers(self, inputs_status.available_values_count);

        if (!success) {
            fmt::print("Node {} failed to consume {} values from inputs\n", self.name(), inputs_status.available_values_count);
        }

        return success ? work_return_t::OK : work_return_t::ERROR;
    }
};

using default_strategy = read_many_and_publish_many;
} // namespace work_strategies

template<typename...>
struct fixed_node_ports_data_helper;

template<meta::is_typelist_v InputPorts, meta::is_typelist_v OutputPorts>
    requires InputPorts::template
all_of<has_fixed_port_info> &&OutputPorts::template all_of<has_fixed_port_info> struct fixed_node_ports_data_helper<InputPorts, OutputPorts> {
    using input_ports       = InputPorts;
    using output_ports      = OutputPorts;

    using input_port_types  = typename input_ports ::template transform<port_type>;
    using output_port_types = typename output_ports ::template transform<port_type>;

    using all_ports         = meta::concat<input_ports, output_ports>;
};

template<has_fixed_port_info_v... Ports>
struct fixed_node_ports_data_helper<Ports...> {
    using all_ports         = meta::concat<std::conditional_t<fair::meta::is_typelist_v<Ports>, Ports, meta::typelist<Ports>>...>;

    using input_ports       = typename all_ports ::template filter<is_in_port>;
    using output_ports      = typename all_ports ::template filter<is_out_port>;

    using input_port_types  = typename input_ports ::template transform<port_type>;
    using output_port_types = typename output_ports ::template transform<port_type>;
};

template<typename... Arguments>
using fixed_node_ports_data = typename meta::typelist<Arguments...>::template filter<has_fixed_port_info_or_is_typelist>::template apply<fixed_node_ports_data_helper>;

// Ports can either be a list of ports instances,
// or two typelists containing port instances -- one for input
// ports and one for output ports
template<typename Derived, typename... Arguments>
// class node: fixed_node_ports_data<Arguments...>::all_ports::tuple_type {
class node : protected std::tuple<Arguments...> {
public:
    using fixed_ports_data  = fixed_node_ports_data<Arguments...>;

    using all_ports         = typename fixed_ports_data::all_ports;
    using input_ports       = typename fixed_ports_data::input_ports;
    using output_ports      = typename fixed_ports_data::output_ports;
    using input_port_types  = typename fixed_ports_data::input_port_types;
    using output_port_types = typename fixed_ports_data::output_port_types;

    using return_type       = typename output_port_types::tuple_or_type;
    using work_strategy       = work_strategies::default_strategy;
    friend work_strategy;

    using min_max_limits = typename meta::typelist<Arguments...>::template filter<is_limits>;
    static_assert(min_max_limits::size <= 1);

private:
    using setting_map = std::map<std::string, int, std::less<>>;
    std::string _name{ std::string(fair::meta::type_name<Derived>()) };

    setting_map _exec_metrics{}; //  →  std::map<string, pmt> → fair scheduling, 'int' stand-in for pmtv

    friend class graph;

public:
    auto &
    self() {
        return *static_cast<Derived *>(this);
    }

    const auto &
    self() const {
        return *static_cast<const Derived *>(this);
    }

    [[nodiscard]] std::string_view
    name() const noexcept {
        return _name;
    }

    void
    set_name(std::string name) noexcept {
        _name = std::move(name);
    }

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    input_port(Self *self) noexcept;

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    output_port(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    input_port(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    output_port(Self *self) noexcept;

    template<std::size_t N>
    [[gnu::always_inline]] constexpr bool
    process_batch_simd_epilogue(std::size_t n, auto out_ptr, auto... in_ptr) {
        if constexpr (N == 0) return true;
        else if (N <= n) {
            using In0                    = meta::first_type<input_port_types>;
            using V                      = stdx::resize_simd_t<N, stdx::native_simd<In0>>;
            using Vs                     = meta::transform_types<meta::rebind_simd_helper<V>::template rebind, input_port_types>;
            const std::tuple input_simds = Vs::template construct<meta::simd_load_ctor>(std::tuple{ in_ptr... });
            const stdx::simd result      = std::apply([this](auto... args) { return self().process_one(args...); }, input_simds);
            result.copy_to(out_ptr, stdx::element_aligned);
            return process_batch_simd_epilogue<N / 2>(n, out_ptr + N, (in_ptr + N)...);
        } else
            return process_batch_simd_epilogue<N / 2>(n, out_ptr, in_ptr...);
    }

#ifdef NOT_YET_PORTED_AS_IT_IS_UNUSED
    template<std::ranges::forward_range... Ins>
        requires(std::ranges::sized_range<Ins> && ...) && input_port_types::template
    are_equal<std::ranges::range_value_t<Ins>...> constexpr bool process_batch(port_data<return_type, 1024> &out, Ins &&...inputs) {
        const auto       &in0 = std::get<0>(std::tie(inputs...));
        const std::size_t n   = std::ranges::size(in0);
        detail::precondition(((n == std::ranges::size(inputs)) && ...));
        auto &&out_range = out.request_write(n);
        // if SIMD makes sense (i.e. input and output ranges are contiguous and all types are
        // vectorizable)
        if constexpr ((std::ranges::contiguous_range<decltype(out_range)> && ... && std::ranges::contiguous_range<Ins>) &&detail::vectorizable<return_type> && detail::node_can_process_simd<Derived>
                      && input_port_types ::template transform<stdx::native_simd>::template all_of<std::is_constructible>) {
            using V       = detail::reduce_to_widest_simd<input_port_types>;
            using Vs      = detail::transform_by_rebind_simd<V, input_port_types>;
            std::size_t i = 0;
            for (i = 0; i + V::size() <= n; i += V::size()) {
                const std::tuple input_simds = Vs::template construct<detail::simd_load_ctor>(std::tuple{ (std::ranges::data(inputs) + i)... });
                const stdx::simd result      = std::apply([this](auto... args) { return self().process_one(args...); }, input_simds);
                result.copy_to(std::ranges::data(out_range) + i, stdx::element_aligned);
            }

            return process_batch_simd_epilogue<std::bit_ceil(V::size()) / 2>(n - i, std::ranges::data(out_range) + i, (std::ranges::data(inputs) + i)...);
        } else { // no explicit SIMD
            auto             out_it    = out_range.begin();
            std::tuple       it_tuple  = { std::ranges::begin(inputs)... };
            const std::tuple end_tuple = { std::ranges::end(inputs)... };
            while (std::get<0>(it_tuple) != std::get<0>(end_tuple)) {
                *out_it = std::apply([this](auto &...its) { return self().process_one((*its++)...); }, it_tuple);
                ++out_it;
            }
            return true;
        }
    }
#endif

    [[nodiscard]] setting_map &
    exec_metrics() noexcept {
        return _exec_metrics;
    }

    [[nodiscard]] setting_map const &
    exec_metrics() const noexcept {
        return _exec_metrics;
    }

    [[nodiscard]] constexpr std::size_t
    min_samples() const noexcept {
        if constexpr (min_max_limits::size == 1) {
            return min_max_limits::template at<0>::min;
        } else {
            return 0;
        }
    }

    [[nodiscard]] constexpr std::size_t
    max_samples() const noexcept {
        if constexpr (min_max_limits::size == 1) {
            return min_max_limits::template at<0>::max;
        } else {
            return std::numeric_limits<std::size_t>::max();
        }
    }

    work_return_t
    work() noexcept {
        return work_strategy::work(self());
    }
};

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    return std::get<typename Self::input_ports::template at<Index>>(*self);
}

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    return std::get<typename Self::output_ports::template at<Index>>(*self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, typename Self::input_ports>();
    return std::get<typename Self::input_ports::template at<Index>>(*self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, typename Self::output_ports>();
    return std::get<typename Self::output_ports::template at<Index>>(*self);
}

template<typename Self>
[[nodiscard]] constexpr auto
input_ports(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return std::tie(input_port<Idx>(self)...);
    }
    (std::make_index_sequence<Self::input_ports::size>());
}

template<typename Self>
[[nodiscard]] constexpr auto
output_ports(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return std::tie(output_port<Idx>(self)...);
    }
    (std::make_index_sequence<Self::output_ports::size>());
}

} // namespace fair::graph

#endif // include guard

#ifndef GNURADIO_PORT_HPP
#define GNURADIO_PORT_HPP

#include <variant>
#include <complex>
#include <span>

#include <utils.hpp>
#include <circular_buffer.hpp>

namespace fair::graph {

using fair::meta::fixed_string;

// #### default supported types -- TODO: to be replaced by pmt::pmtv declaration
using supported_type = std::variant<uint8_t, uint32_t, int8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> /*, ...*/>;

enum class port_direction_t { INPUT, OUTPUT, ANY }; // 'ANY' only for query and not to be used for port declarations
enum class connection_result_t { SUCCESS, FAILED };
enum class port_type_t { STREAM, MESSAGE }; // TODO: Think of a better name
enum class port_domain_t { CPU, GPU, NET, FPGA, DSP, MLU };

template<class T>
concept Port = requires(T t, const std::size_t n_items) { // dynamic definitions
                   typename T::value_type;
                   { t.pmt_type() } -> std::same_as<supported_type>;
                   { t.type() } -> std::same_as<port_type_t>;
                   { t.direction() } -> std::same_as<port_direction_t>;
                   { t.name() } -> std::same_as<std::string_view>;
                   { t.resize_buffer(n_items) } -> std::same_as<connection_result_t>;
                   { t.disconnect() } -> std::same_as<connection_result_t>;
               };

template<typename T>
concept has_fixed_port_info_v = requires {
                                    typename T::value_type;
                                    { T::static_name() };
                                    { T::direction() } -> std::same_as<port_direction_t>;
                                    { T::type() } -> std::same_as<port_type_t>;
                                };

template<typename T>
using has_fixed_port_info = std::integral_constant<bool, has_fixed_port_info_v<T>>;

template<typename T>
struct has_fixed_port_info_or_is_typelist : std::false_type {};

template<typename T>
    requires has_fixed_port_info_v<T>
struct has_fixed_port_info_or_is_typelist<T> : std::true_type {};

template<typename T>
    requires(meta::is_typelist_v<T> and T::template all_of<has_fixed_port_info>)
struct has_fixed_port_info_or_is_typelist<T> : std::true_type {};

template<std::size_t _min, std::size_t _max>
struct limits {
    using limits_tag                 = std::true_type;
    static constexpr std::size_t min = _min;
    static constexpr std::size_t max = _max;
};

template<typename T>
concept is_limits_v = requires { typename T::limits_tag; };

static_assert(!is_limits_v<int>);
static_assert(!is_limits_v<std::size_t>);
static_assert(is_limits_v<limits<0, 1024>>);

template<typename T>
using is_limits = std::integral_constant<bool, is_limits_v<T>>;

template<typename Port>
using port_type = typename Port::value_type;

template<typename Port>
using is_in_port = std::integral_constant<bool, Port::direction() == port_direction_t::INPUT>;

template<typename Port>
concept InPort = is_in_port<Port>::value;

template<typename Port>
using is_out_port = std::integral_constant<bool, Port::direction() == port_direction_t::OUTPUT>;

template<typename Port>
concept OutPort = is_out_port<Port>::value;


template<typename T, fixed_string PortName, port_type_t Porttype, port_direction_t PortDirection, // TODO: sort default arguments
         std::size_t N_HISTORY = std::dynamic_extent, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, bool OPTIONAL = false,
         gr::Buffer BufferType = gr::circular_buffer<T>>
class port {
public:
    static_assert(PortDirection != port_direction_t::ANY, "ANY reserved for queries and not port direction declarations");

    using value_type                = T;

    static constexpr bool IS_INPUT  = PortDirection == port_direction_t::INPUT;
    static constexpr bool IS_OUTPUT = PortDirection == port_direction_t::OUTPUT;

    using port_tag                  = std::true_type;

private:
    using ReaderType          = decltype(std::declval<BufferType>().new_reader());
    using WriterType          = decltype(std::declval<BufferType>().new_writer());
    using IoType              = std::conditional_t<IS_INPUT, ReaderType, WriterType>;

    std::string  _name        = static_cast<std::string>(PortName);
    std::int16_t _priority    = 0; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    std::size_t  _n_history   = N_HISTORY;
    std::size_t  _min_samples = (MIN_SAMPLES == std::dynamic_extent ? 1 : MIN_SAMPLES);
    std::size_t  _max_samples = MAX_SAMPLES;
    bool         _connected   = false;

    IoType       _ioHandler   = new_io_handler();

public:
    [[nodiscard]] constexpr auto
    new_io_handler() const noexcept {
        if constexpr (IS_INPUT) {
            return BufferType(65536).new_reader();
        } else {
            return BufferType(65536).new_writer();
        }
    }

    [[nodiscard]] void *
    writer_handler_internal() noexcept {
        static_assert(IS_OUTPUT, "only to be used with output ports");
        return static_cast<void *>(std::addressof(_ioHandler));
    }

    [[nodiscard]] bool
    update_reader_internal(void *buffer_writer_handler_other) noexcept {
        static_assert(IS_INPUT, "only to be used with input ports");

        if (buffer_writer_handler_other == nullptr) {
            return false;
        }

        // TODO: If we want to allow ports with different buffer types to be mixed
        //       this will fail. We need to add a check that two ports that
        //       connect to each other use the same buffer type
        //       (std::any could be a viable approach)
        auto typed_buffer_writer = static_cast<WriterType *>(buffer_writer_handler_other);
        setBuffer(typed_buffer_writer->buffer());
        return true;
    }

public:
    port()             = default;

    port(const port &) = delete;

    auto
    operator=(const port &)
            = delete;

    port(std::string port_name, std::int16_t priority = 0, std::size_t n_history = 0, std::size_t min_samples = 0U, std::size_t max_samples = SIZE_MAX) noexcept
        : _name(std::move(port_name))
        , _priority{ priority }
        , _n_history(n_history)
        , _min_samples(min_samples)
        , _max_samples(max_samples) {
        static_assert(PortName.empty(), "port name must be exclusively declared via NTTP or constructor parameter");
    }

    constexpr port(port &&other) noexcept
        : _name(std::move(other._name))
        , _priority{ other._priority }
        , _n_history(other._n_history)
        , _min_samples(other._min_samples)
        , _max_samples(other._max_samples) {}

    constexpr port &
    operator=(port &&other) {
        port tmp(std::move(other));
        std::swap(_name, tmp._name);
        std::swap(_priority, tmp._priority);
        std::swap(_n_history, tmp._n_history);
        std::swap(_min_samples, tmp._min_samples);
        std::swap(_max_samples, tmp._max_samples);
        std::swap(_connected, tmp._connected);
        std::swap(_ioHandler, tmp._ioHandler);
        return *this;
    }

    [[nodiscard]] constexpr static port_type_t
    type() noexcept {
        return Porttype;
    }

    [[nodiscard]] constexpr static port_direction_t
    direction() noexcept {
        return PortDirection;
    }

    [[nodiscard]] constexpr static decltype(PortName)
    static_name() noexcept
        requires(!PortName.empty())
    {
        return PortName;
    }

    [[nodiscard]] constexpr supported_type
    pmt_type() const noexcept {
        return T();
    }

    [[nodiscard]] constexpr std::string_view
    name() const noexcept {
        if constexpr (!PortName.empty()) {
            return static_cast<std::string_view>(PortName);
        } else {
            return _name;
        }
    }

    [[nodiscard]] constexpr static bool
    is_optional() noexcept {
        return OPTIONAL;
    }

    [[nodiscard]] constexpr std::int16_t
    priority() const noexcept {
        return _priority;
    }

    [[nodiscard]] constexpr static std::size_t
    available() noexcept {
        return 0;
    } //  ↔ maps to Buffer::Buffer[Reader, Writer].available()

    [[nodiscard]] constexpr std::size_t
    n_history() const noexcept {
        if constexpr (N_HISTORY == std::dynamic_extent) {
            return _n_history;
        } else {
            return N_HISTORY;
        }
    }

    [[nodiscard]] constexpr std::size_t
    min_buffer_size() const noexcept {
        if constexpr (MIN_SAMPLES == std::dynamic_extent) {
            return _min_samples;
        } else {
            return MIN_SAMPLES;
        }
    }

    [[nodiscard]] constexpr std::size_t
    max_buffer_size() const noexcept {
        if constexpr (MAX_SAMPLES == std::dynamic_extent) {
            return _max_samples;
        } else {
            return MAX_SAMPLES;
        }
    }

    [[nodiscard]] constexpr connection_result_t
    resize_buffer(std::size_t min_size) noexcept {
        if constexpr (IS_INPUT) {
            return connection_result_t::SUCCESS;
        } else {
            try {
                _ioHandler = BufferType(min_size).new_writer();
            } catch (...) {
                return connection_result_t::FAILED;
            }
        }
        return connection_result_t::SUCCESS;
    }

    [[nodiscard]] BufferType
    buffer() {
        return _ioHandler.buffer();
    }

    void
    setBuffer(gr::Buffer auto buffer) noexcept {
        if constexpr (IS_INPUT) {
            _ioHandler = std::move(buffer.new_reader());
            _connected = true;
        } else {
            _ioHandler = std::move(buffer.new_writer());
        }
    }

    [[nodiscard]] constexpr ReaderType &
    reader() const noexcept {
        static_assert(!IS_OUTPUT, "reader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr ReaderType &
    reader() noexcept {
        static_assert(!IS_OUTPUT, "reader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr WriterType &
    writer() const noexcept {
        static_assert(!IS_INPUT, "writer() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr WriterType &
    writer() noexcept {
        static_assert(!IS_INPUT, "writer() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] connection_result_t
    disconnect() noexcept {
        if (_connected == false) {
            return connection_result_t::FAILED;
        }
        _ioHandler = new_io_handler();
        _connected = false;
        return connection_result_t::SUCCESS;
    }

    template<typename Other>
    [[nodiscard]] connection_result_t
    connect(Other &&other) {
        static_assert(IS_OUTPUT && std::remove_cvref_t<Other>::IS_INPUT);
        auto src_buffer = writer_handler_internal();
        return std::forward<Other>(other).update_reader_internal(src_buffer) ? connection_result_t::SUCCESS
                                                                             : connection_result_t::FAILED;
    }

    friend class dynamic_port;
};


namespace detail {
template<typename T, auto>
using just_t = T;

template<typename T, std::size_t... Is>
consteval fair::meta::typelist<just_t<T, Is>...>
repeated_ports_impl(std::index_sequence<Is...>) {
    return {};
}
} // namespace detail

// TODO: Add port index to BaseName
template<std::size_t Count, typename T, fixed_string BaseName, port_type_t Porttype, port_direction_t PortDirection, std::size_t MIN_SAMPLES = std::dynamic_extent,
         std::size_t MAX_SAMPLES = std::dynamic_extent>
using repeated_ports = decltype(detail::repeated_ports_impl<port<T, BaseName, Porttype, PortDirection, MIN_SAMPLES, MAX_SAMPLES>>(std::make_index_sequence<Count>()));

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


}

#endif // include guard
/*
    Copyright © 2022 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH
                     Matthias Kretz <m.kretz@gsi.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VIR_SIMD_H_
#define VIR_SIMD_H_

#if __cplusplus < 201703L
#error "simd requires C++17 or later"
#endif

#if __has_include (<experimental/simd>) && !defined VIR_DISABLE_STDX_SIMD
#include <experimental/simd>
#endif

#if defined __cpp_lib_experimental_parallel_simd && __cpp_lib_experimental_parallel_simd >= 201803

namespace vir::stdx
{
  using namespace std::experimental::parallelism_v2;
}

#else

#include <cmath>
#include <cstring>
#ifdef _GLIBCXX_DEBUG_UB
#include <cstdio>
#endif
#include <functional>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#ifdef VIR_SIMD_TS_DROPIN
namespace std::experimental
{
  inline namespace parallelism_v2
#else
namespace vir::stdx
#endif
{
  using std::size_t;

  namespace detail
  {
    template <typename T>
      struct type_identity
      { using type = T; };

    template <typename T>
      using type_identity_t = typename type_identity<T>::type;

    constexpr size_t
    bit_ceil(size_t x)
    {
      size_t r = 1;
      while (r < x)
        r <<= 1;
      return r;
    }

    constexpr size_t
    bit_floor(size_t x)
    {
      size_t r = x;
      do {
        r = x;
        x &= x - 1;
      } while (x);
      return r;
    }

    template <typename T>
      typename T::value_type
      value_type_or_identity_impl(int);

    template <typename T>
      T
      value_type_or_identity_impl(float);

    template <typename T>
      using value_type_or_identity_t
        = decltype(value_type_or_identity_impl<T>(int()));

    class ExactBool
    {
      const bool data;

    public:
      constexpr ExactBool(bool b) : data(b) {}

      ExactBool(int) = delete;

      constexpr operator bool() const { return data; }
    };

    template <typename... Args>
      [[noreturn]] [[gnu::always_inline]] inline void
      invoke_ub([[maybe_unused]] const char* msg,
                [[maybe_unused]] const Args&... args)
      {
#ifdef _GLIBCXX_DEBUG_UB
        std::fprintf(stderr, msg, args...);
        __builtin_trap();
#else
        __builtin_unreachable();
#endif
      }

    template <typename T>
      using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

    template <typename T>
      using L = std::numeric_limits<T>;

    template <bool B>
      using BoolConstant = std::integral_constant<bool, B>;

    template <size_t X>
      using SizeConstant = std::integral_constant<size_t, X>;

    template <size_t I, typename T, typename... Ts>
      constexpr auto
      pack_simd_subscript(const T& x0, const Ts&... xs)
      {
        if constexpr (I >= T::size())
          return pack_simd_subscript<I - T::size()>(xs...);
        else
          return x0[I];
      }

    template <class T>
      struct is_vectorizable : std::is_arithmetic<T>
      {};

    template <>
      struct is_vectorizable<bool> : std::false_type
      {};

    template <class T>
      inline constexpr bool is_vectorizable_v = is_vectorizable<T>::value;

    // Deduces to a vectorizable type
    template <typename T, typename = std::enable_if_t<is_vectorizable_v<T>>>
      using Vectorizable = T;

    // Deduces to a floating-point type
    template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
      using FloatingPoint = T;

    // Deduces to a signed integer type
    template <typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>,
                                                                         std::is_signed<T>>>>
      using SignedIntegral = T;

    // is_higher_integer_rank<T, U> (T has higher or equal integer rank than U)
    template <typename T, typename U, bool = (sizeof(T) > sizeof(U)),
              bool = (sizeof(T) == sizeof(U))>
      struct is_higher_integer_rank;

    template <typename T>
      struct is_higher_integer_rank<T, T, false, true>
      : public std::true_type
      {};

    template <typename T, typename U>
      struct is_higher_integer_rank<T, U, true, false>
      : public std::true_type
      {};

    template <typename T, typename U>
      struct is_higher_integer_rank<T, U, false, false>
      : public std::false_type
      {};

    // this may fail for char -> short if sizeof(char) == sizeof(short)
    template <typename T, typename U>
      struct is_higher_integer_rank<T, U, false, true>
      : public std::is_same<decltype(std::declval<T>() + std::declval<U>()), T>
      {};

    // is_value_preserving<From, To>
    template <typename From, typename To, bool = std::is_arithmetic_v<From>,
              bool = std::is_arithmetic_v<To>>
      struct is_value_preserving;

    // ignore "signed/unsigned mismatch" in the following trait.
    // The implicit conversions will do the right thing here.
    template <typename From, typename To>
      struct is_value_preserving<From, To, true, true>
      : public BoolConstant<L<From>::digits <= L<To>::digits
                              && L<From>::max() <= L<To>::max()
                              && L<From>::lowest() >= L<To>::lowest()
                              && !(std::is_signed_v<From> && std::is_unsigned_v<To>)> {};

    template <typename T>
      struct is_value_preserving<T, bool, true, true>
      : public std::false_type {};

    template <>
      struct is_value_preserving<bool, bool, true, true>
      : public std::true_type {};

    template <typename T>
      struct is_value_preserving<T, T, true, true>
      : public std::true_type {};

    template <typename From, typename To>
      struct is_value_preserving<From, To, false, true>
      : public std::is_convertible<From, To> {};

    template <typename From, typename To,
              typename = std::enable_if_t<is_value_preserving<remove_cvref_t<From>, To>::value>>
      using ValuePreserving = From;

    template <typename From, typename To,
              typename DecayedFrom = remove_cvref_t<From>,
              typename = std::enable_if_t<std::conjunction<
                                            std::is_convertible<From, To>,
                                            std::disjunction<
                                              std::is_same<DecayedFrom, To>,
                                              std::is_same<DecayedFrom, int>,
                                              std::conjunction<std::is_same<DecayedFrom, unsigned>,
                                                               std::is_unsigned<To>>,
                                              is_value_preserving<DecayedFrom, To>>>::value>>
      using ValuePreservingOrInt = From;

    // LoadStorePtr / is_possible_loadstore_conversion
    template <typename Ptr, typename ValueType>
      struct is_possible_loadstore_conversion
      : std::conjunction<is_vectorizable<Ptr>, is_vectorizable<ValueType>>
      {};

    template <>
      struct is_possible_loadstore_conversion<bool, bool> : std::true_type {};

    // Deduces to a type allowed for load/store with the given value type.
    template <typename Ptr, typename ValueType,
              typename = std::enable_if_t<
                           is_possible_loadstore_conversion<Ptr, ValueType>::value>>
      using LoadStorePtr = Ptr;
  }

  namespace simd_abi
  {
    struct scalar
    {};

    template <typename>
      inline constexpr int max_fixed_size = 32;

    template <int N>
      struct fixed_size
      {};

    template <class T>
      using native =
        std::conditional_t<(sizeof(T) > 8),
                           scalar,
                           fixed_size<
#ifdef __AVX512F__
                             64
#elif defined __AVX2__
                             32
#elif defined __AVX__
                             std::is_floating_point_v<T> ? 32 : 16
#else
                             16
#endif
                               / sizeof(T)
                           >
                          >;

    template <class T>
      using compatible = std::conditional_t<(sizeof(T) > 8),
                                            scalar,
                                            fixed_size<16 / sizeof(T)>>;

    template <typename T, size_t N, typename...>
      struct deduce
      { using type = std::conditional_t<N == 1, scalar, fixed_size<int(N)>>; };

    template <typename T, size_t N, typename... Abis>
      using deduce_t = typename deduce<T, N, Abis...>::type;
  }

  // flags //
  struct element_aligned_tag
  {};

  struct vector_aligned_tag
  {};

  template <size_t>
    struct overaligned_tag
    {};

  inline constexpr element_aligned_tag element_aligned{};

  inline constexpr vector_aligned_tag vector_aligned{};

  template <size_t N>
    inline constexpr overaligned_tag<N> overaligned{};

  // fwd decls //
  template <class T, class A = simd_abi::compatible<T>>
    class simd;

  template <class T, class A = simd_abi::compatible<T>>
    class simd_mask;

  // aliases //
  template <class T>
    using native_simd = simd<T, simd_abi::native<T>>;

  template <class T>
    using native_simd_mask = simd_mask<T, simd_abi::native<T>>;

  template <class T, int N>
    using fixed_size_simd = simd<T, simd_abi::fixed_size<N>>;

  template <class T, int N>
    using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;

  // Traits //
  template <class T>
    struct is_abi_tag : std::false_type
    {};

  template <class T>
    inline constexpr bool is_abi_tag_v = is_abi_tag<T>::value;

  template <>
    struct is_abi_tag<simd_abi::scalar> : std::true_type
    {};

  template <int N>
    struct is_abi_tag<simd_abi::fixed_size<N>> : std::true_type
    {};

  template <class T>
    struct is_simd : std::false_type
    {};

  template <class T>
    inline constexpr bool is_simd_v = is_simd<T>::value;

  template <class T, class A>
    struct is_simd<simd<T, A>>
    : std::conjunction<detail::is_vectorizable<T>, is_abi_tag<A>>
    {};

  template <class T>
    struct is_simd_mask : std::false_type
    {};

  template <class T>
    inline constexpr bool is_simd_mask_v = is_simd_mask<T>::value;

  template <class T, class A>
    struct is_simd_mask<simd_mask<T, A>>
    : std::conjunction<detail::is_vectorizable<T>, is_abi_tag<A>>
    {};

  template <class T>
    struct is_simd_flag_type : std::false_type
    {};

  template <class T>
    inline constexpr bool is_simd_flag_type_v = is_simd_flag_type<T>::value;

  template <class T, class A = simd_abi::compatible<T>>
    struct simd_size;

  template <class T, class A = simd_abi::compatible<T>>
    inline constexpr size_t simd_size_v = simd_size<T, A>::value;

  template <class T>
    struct simd_size<detail::Vectorizable<T>, simd_abi::scalar>
    : std::integral_constant<size_t, 1>
    {};

  template <class T, int N>
    struct simd_size<detail::Vectorizable<T>, simd_abi::fixed_size<N>>
    : std::integral_constant<size_t, N>
    {};

  template <class T, class U = typename T::value_type>
    struct memory_alignment;

  template <class T, class U = typename T::value_type>
    inline constexpr size_t memory_alignment_v = memory_alignment<T, U>::value;

  template <class T, class A, class U>
    struct memory_alignment<simd<T, A>, detail::Vectorizable<U>>
    : std::integral_constant<size_t, alignof(U)>
    {};

  template <class T, class A>
    struct memory_alignment<simd_mask<T, A>, bool>
    : std::integral_constant<size_t, alignof(bool)>
    {};

  template <class T, class V,
            class = typename std::conjunction<detail::is_vectorizable<T>,
                                              std::disjunction<is_simd<V>, is_simd_mask<V>>>::type>
    struct rebind_simd;

  template <class T, class V>
    using rebind_simd_t = typename rebind_simd<T, V>::type;

  template <class T, class U, class A>
    struct rebind_simd<T, simd<U, A>, std::true_type>
    { using type = simd<T, A>; };

  template <class T, class U, class A>
    struct rebind_simd<T, simd_mask<U, A>, std::true_type>
    { using type = simd_mask<T, A>; };

  template <int N, class V,
            class = typename std::conjunction<
                               detail::BoolConstant<(N > 0)>,
                               std::disjunction<is_simd<V>, is_simd_mask<V>>
                             >::type>
    struct resize_simd;

  template <int N, class V>
    using resize_simd_t = typename resize_simd<N, V>::type;

  template <int N, class T, class A>
    struct resize_simd<N, simd<T, A>, std::true_type>
    {
      using type = simd<T, std::conditional_t<N == 1, simd_abi::scalar, simd_abi::fixed_size<N>>>;
    };

  template <int N, class T, class A>
    struct resize_simd<N, simd_mask<T, A>, std::true_type>
    {
      using type = simd_mask<T, std::conditional_t<
                                  N == 1, simd_abi::scalar, simd_abi::fixed_size<N>>>;
    };

  // simd_mask (scalar)
  template <class T>
    class simd_mask<detail::Vectorizable<T>, simd_abi::scalar>
    {
      bool data;

    public:
      using value_type = bool;
      using reference = bool&;
      using abi_type = simd_abi::scalar;
      using simd_type = simd<T, abi_type>;

      static constexpr size_t size() noexcept
      { return 1; }

      constexpr simd_mask() = default;
      constexpr simd_mask(const simd_mask&) = default;
      constexpr simd_mask(simd_mask&&) noexcept = default;
      constexpr simd_mask& operator=(const simd_mask&) = default;
      constexpr simd_mask& operator=(simd_mask&&) noexcept = default;

      // explicit broadcast constructor
      explicit constexpr
      simd_mask(bool x)
      : data(x) {}

      template <typename F>
        explicit constexpr
        simd_mask(F&& gen, std::enable_if_t<
                             std::is_same_v<decltype(std::declval<F>()(detail::SizeConstant<0>())),
                                            value_type>>* = nullptr)
        : data(gen(detail::SizeConstant<0>()))
        {}

      // load constructor
      template <typename Flags>
        simd_mask(const value_type* mem, Flags)
        : data(mem[0])
        {}

      template <typename Flags>
        simd_mask(const value_type* mem, simd_mask k, Flags)
        : data(k ? mem[0] : false)
        {}

      // loads [simd_mask.load]
      template <typename Flags>
        void
        copy_from(const value_type* mem, Flags)
        { data = mem[0]; }

      // stores [simd_mask.store]
      template <typename Flags>
        void
        copy_to(value_type* mem, Flags) const
        { mem[0] = data; }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      // negation
      constexpr simd_mask
      operator!() const
      { return simd_mask(not data); }

      // simd_mask binary operators [simd_mask.binary]
      friend constexpr simd_mask
      operator&&(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data && y.data); }

      friend constexpr simd_mask
      operator||(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data || y.data); }

      friend constexpr simd_mask
      operator&(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data & y.data); }

      friend constexpr simd_mask
      operator|(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data | y.data); }

      friend constexpr simd_mask
      operator^(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data ^ y.data); }

      friend constexpr simd_mask&
      operator&=(simd_mask& x, const simd_mask& y)
      {
        x.data &= y.data;
        return x;
      }

      friend constexpr simd_mask&
      operator|=(simd_mask& x, const simd_mask& y)
      {
        x.data |= y.data;
        return x;
      }

      friend constexpr simd_mask&
      operator^=(simd_mask& x, const simd_mask& y)
      {
        x.data ^= y.data;
        return x;
      }

      // simd_mask compares [simd_mask.comparison]
      friend constexpr simd_mask
      operator==(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data == y.data); }

      friend constexpr simd_mask
      operator!=(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data != y.data); }
    };

  // simd_mask (fixed_size)
  template <class T, int N>
    class simd_mask<detail::Vectorizable<T>, simd_abi::fixed_size<N>>
    {
    private:
      template <typename V, int M, size_t Parts>
        friend std::enable_if_t<M == Parts * V::size() && is_simd_mask_v<V>, std::array<V, Parts>>
        split(const simd_mask<typename V::simd_type::value_type, simd_abi::fixed_size<M>>&);

      bool data[N];

      template <typename F, size_t... Is>
        constexpr
        simd_mask(std::index_sequence<Is...>, F&& init)
        : data {init(detail::SizeConstant<Is>())...}
        {}

    public:
      using value_type = bool;
      using reference = bool&;
      using abi_type = simd_abi::fixed_size<N>;
      using simd_type = simd<T, abi_type>;

      static constexpr size_t size() noexcept
      { return N; }

      constexpr simd_mask() = default;
      constexpr simd_mask(const simd_mask&) = default;
      constexpr simd_mask(simd_mask&&) noexcept = default;
      constexpr simd_mask& operator=(const simd_mask&) = default;
      constexpr simd_mask& operator=(simd_mask&&) noexcept = default;

      // explicit broadcast constructor
      explicit constexpr
      simd_mask(bool x)
      : simd_mask([x](size_t) { return x; })
      {}

      template <typename F>
        explicit constexpr
        simd_mask(F&& gen, std::enable_if_t<
                             std::is_same_v<decltype(std::declval<F>()(detail::SizeConstant<0>())),
                                            value_type>>* = nullptr)
        : simd_mask(std::make_index_sequence<N>(), std::forward<F>(gen))
        {}

      // implicit conversions
      template <typename U>
        constexpr
        simd_mask(const simd_mask<U, abi_type>& x)
        : simd_mask([&x](auto i) { return x[i]; })
        {}

      // load constructor
      template <typename Flags>
        simd_mask(const value_type* mem, Flags)
        : simd_mask([mem](size_t i) { return mem[i]; })
        {}

      template <typename Flags>
        simd_mask(const value_type* mem, const simd_mask& k, Flags)
        : simd_mask([mem, &k](size_t i) { return k[i] ? mem[i] : false; })
        {}

      // loads [simd_mask.load]
      template <typename Flags>
        void
        copy_from(const value_type* mem, Flags)
        { std::memcpy(data, mem, N * sizeof(bool)); }

      // stores [simd_mask.store]
      template <typename Flags>
        void
        copy_to(value_type* mem, Flags) const
        { std::memcpy(mem, data, N * sizeof(bool)); }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      // negation
      constexpr simd_mask
      operator!() const
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = !data[i];
        return r;
      }

      // simd_mask binary operators [simd_mask.binary]
      friend constexpr simd_mask
      operator&&(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] & y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator||(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] | y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator&(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] & y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator|(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] | y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator^(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] ^ y.data[i];
        return r;
      }

      friend constexpr simd_mask&
      operator&=(simd_mask& x, const simd_mask& y)
      {
        for (int i = 0; i < N; ++i)
          x.data[i] &= y.data[i];
        return x;
      }

      friend constexpr simd_mask&
      operator|=(simd_mask& x, const simd_mask& y)
      {
        for (int i = 0; i < N; ++i)
          x.data[i] |= y.data[i];
        return x;
      }

      friend constexpr simd_mask&
      operator^=(simd_mask& x, const simd_mask& y)
      {
        for (int i = 0; i < N; ++i)
          x.data[i] ^= y.data[i];
        return x;
      }

      // simd_mask compares [simd_mask.comparison]
      friend constexpr simd_mask
      operator==(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] == y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator!=(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] != y.data[i];
        return r;
      }
    };

  // simd_mask reductions [simd_mask.reductions]
  template <typename T>
    constexpr bool
    all_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return k[0]; }

  template <typename T>
    constexpr bool
    any_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return k[0]; }

  template <typename T>
    constexpr bool
    none_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return not k[0]; }

  template <typename T>
    constexpr bool
    some_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return false; }

  template <typename T>
    constexpr int
    popcount(simd_mask<T, simd_abi::scalar> k) noexcept
    { return static_cast<int>(k[0]); }

  template <typename T>
    constexpr int
    find_first_set(simd_mask<T, simd_abi::scalar> k) noexcept
    {
      if (not k[0])
        detail::invoke_ub("find_first_set(empty mask) is UB");
      return 0;
    }

  template <typename T>
    constexpr int
    find_last_set(simd_mask<T, simd_abi::scalar> k) noexcept
    {
      if (not k[0])
        detail::invoke_ub("find_last_set(empty mask) is UB");
      return 0;
    }

  template <typename T, int N>
    constexpr bool
    all_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (not k[i])
            return false;
        }
      return true;
    }

  template <typename T, int N>
    constexpr bool
    any_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (k[i])
            return true;
        }
      return false;
    }

  template <typename T, int N>
    constexpr bool
    none_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (k[i])
            return false;
        }
      return true;
    }

  template <typename T, int N>
    constexpr bool
    some_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      bool last = k[0];
      for (int i = 1; i < N; ++i)
        {
          if (last != k[i])
            return true;
        }
      return false;
    }

  template <typename T, int N>
    constexpr int
    popcount(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      int cnt = k[0];
      for (int i = 1; i < N; ++i)
        cnt += k[i];
      return cnt;
    }

  template <typename T, int N>
    constexpr int
    find_first_set(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (k[i])
            return i;
        }
      detail::invoke_ub("find_first_set(empty mask) is UB");
    }

  template <typename T, int N>
    constexpr int
    find_last_set(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = N - 1; i >= 0; --i)
        {
          if (k[i])
            return i;
        }
      detail::invoke_ub("find_last_set(empty mask) is UB");
    }

  constexpr bool
  all_of(detail::ExactBool x) noexcept
  { return x; }

  constexpr bool
  any_of(detail::ExactBool x) noexcept
  { return x; }

  constexpr bool
  none_of(detail::ExactBool x) noexcept
  { return !x; }

  constexpr bool
  some_of(detail::ExactBool) noexcept
  { return false; }

  constexpr int
  popcount(detail::ExactBool x) noexcept
  { return x; }

  constexpr int
  find_first_set(detail::ExactBool)
  { return 0; }

  constexpr int
  find_last_set(detail::ExactBool)
  { return 0; }

  // scalar_simd_int_base
  template <class T, bool = std::is_integral_v<T>>
    class scalar_simd_int_base
    {};

  template <class T>
    class scalar_simd_int_base<T, true>
    {
      using Derived = simd<T, simd_abi::scalar>;

      constexpr T&
      d() noexcept
      { return static_cast<Derived*>(this)->data; }

      constexpr const T&
      d() const noexcept
      { return static_cast<const Derived*>(this)->data; }

    public:
      friend constexpr Derived&
      operator%=(Derived& lhs, Derived x)
      {
        lhs.d() %= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator&=(Derived& lhs, Derived x)
      {
        lhs.d() &= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator|=(Derived& lhs, Derived x)
      {
        lhs.d() |= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator^=(Derived& lhs, Derived x)
      {
        lhs.d() ^= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator<<=(Derived& lhs, Derived x)
      {
        lhs.d() <<= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator>>=(Derived& lhs, Derived x)
      {
        lhs.d() >>= x.d();
        return lhs;
      }

      friend constexpr Derived
      operator%(Derived x, Derived y)
      {
        x.d() %= y.d();
        return x;
      }

      friend constexpr Derived
      operator&(Derived x, Derived y)
      {
        x.d() &= y.d();
        return x;
      }

      friend constexpr Derived
      operator|(Derived x, Derived y)
      {
        x.d() |= y.d();
        return x;
      }

      friend constexpr Derived
      operator^(Derived x, Derived y)
      {
        x.d() ^= y.d();
        return x;
      }

      friend constexpr Derived
      operator<<(Derived x, Derived y)
      {
        x.d() <<= y.d();
        return x;
      }

      friend constexpr Derived
      operator>>(Derived x, Derived y)
      {
        x.d() >>= y.d();
        return x;
      }

      friend constexpr Derived
      operator<<(Derived x, int y)
      {
        x.d() <<= y;
        return x;
      }

      friend constexpr Derived
      operator>>(Derived x, int y)
      {
        x.d() >>= y;
        return x;
      }

      constexpr Derived
      operator~() const
      { return Derived(static_cast<T>(~d())); }
    };

  // simd (scalar)
  template <class T>
    class simd<T, simd_abi::scalar>
    : public scalar_simd_int_base<T>
    {
      friend class scalar_simd_int_base<T>;

      T data;

    public:
      using value_type = T;
      using reference = T&;
      using abi_type = simd_abi::scalar;
      using mask_type = simd_mask<T, abi_type>;

      static constexpr size_t size() noexcept
      { return 1; }

      constexpr simd() = default;
      constexpr simd(const simd&) = default;
      constexpr simd(simd&&) noexcept = default;
      constexpr simd& operator=(const simd&) = default;
      constexpr simd& operator=(simd&&) noexcept = default;

      // simd constructors
      template <typename U>
        constexpr
        simd(detail::ValuePreservingOrInt<U, value_type>&& value) noexcept
        : data(value)
        {}

      // generator constructor
      template <typename F>
        explicit constexpr
        simd(F&& gen, detail::ValuePreservingOrInt<
                        decltype(std::declval<F>()(std::declval<detail::SizeConstant<0>&>())),
                        value_type>* = nullptr)
        : data(gen(detail::SizeConstant<0>()))
        {}

      // load constructor
      template <typename U, typename Flags>
        simd(const U* mem, Flags)
        : data(mem[0])
        {}

      // loads [simd.load]
      template <typename U, typename Flags>
        void
        copy_from(const detail::Vectorizable<U>* mem, Flags)
        { data = mem[0]; }

      // stores [simd.store]
      template <typename U, typename Flags>
        void
        copy_to(detail::Vectorizable<U>* mem, Flags) const
        { mem[0] = data; }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      // increment and decrement:
      constexpr simd&
      operator++()
      {
        ++data;
        return *this;
      }

      constexpr simd
      operator++(int)
      {
        simd r = *this;
        ++data;
        return r;
      }

      constexpr simd&
      operator--()
      {
        --data;
        return *this;
      }

      constexpr simd
      operator--(int)
      {
        simd r = *this;
        --data;
        return r;
      }

      // unary operators
      constexpr mask_type
      operator!() const
      { return mask_type(not data); }

      constexpr simd
      operator+() const
      { return *this; }

      constexpr simd
      operator-() const
      { return -data; }

      // compound assignment [simd.cassign]
      constexpr friend simd&
      operator+=(simd& lhs, const simd& x)
      { return lhs = lhs + x; }

      constexpr friend simd&
      operator-=(simd& lhs, const simd& x)
      { return lhs = lhs - x; }

      constexpr friend simd&
      operator*=(simd& lhs, const simd& x)
      { return lhs = lhs * x; }

      constexpr friend simd&
        operator/=(simd& lhs, const simd& x)
      { return lhs = lhs / x; }

      // binary operators [simd.binary]
      constexpr friend simd
      operator+(const simd& x, const simd& y)
      { simd r = x; r.data += y.data; return r; }

      constexpr friend simd
      operator-(const simd& x, const simd& y)
      { simd r = x; r.data -= y.data; return r; }

      constexpr friend simd
      operator*(const simd& x, const simd& y)
      { simd r = x; r.data *= y.data; return r; }

      constexpr friend simd
      operator/(const simd& x, const simd& y)
      { simd r = x; r.data /= y.data; return r; }

      // compares [simd.comparison]
      constexpr friend mask_type
      operator==(const simd& x, const simd& y)
      { return mask_type(x.data == y.data); }

      constexpr friend mask_type
      operator!=(const simd& x, const simd& y)
      { return mask_type(x.data != y.data); }

      constexpr friend mask_type
      operator<(const simd& x, const simd& y)
      { return mask_type(x.data < y.data); }

      constexpr friend mask_type
      operator<=(const simd& x, const simd& y)
      { return mask_type(x.data <= y.data); }

      constexpr friend mask_type
      operator>(const simd& x, const simd& y)
      { return mask_type(x.data > y.data); }

      constexpr friend mask_type
      operator>=(const simd& x, const simd& y)
      { return mask_type(x.data >= y.data); }
    };

  // fixed_simd_int_base
  template <class T, int N, bool = std::is_integral_v<T>>
    class fixed_simd_int_base
    {};

  template <class T, int N>
    class fixed_simd_int_base<T, N, true>
    {
      using Derived = simd<T, simd_abi::fixed_size<N>>;

      constexpr T&
      d(int i) noexcept
      { return static_cast<Derived*>(this)->data[i]; }

      constexpr const T&
      d(int i) const noexcept
      { return static_cast<const Derived*>(this)->data[i]; }

    public:
      friend constexpr Derived&
      operator%=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) %= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator&=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) &= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator|=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) |= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator^=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) ^= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator<<=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) <<= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator>>=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) >>= x.d(i);
        return lhs;
      }

      friend constexpr Derived
      operator%(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] % y[i]; }); }

      friend constexpr Derived
      operator&(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] & y[i]; }); }

      friend constexpr Derived
      operator|(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] | y[i]; }); }

      friend constexpr Derived
      operator^(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] ^ y[i]; }); }

      friend constexpr Derived
      operator<<(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] << y[i]; }); }

      friend constexpr Derived
      operator>>(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] >> y[i]; }); }

      friend constexpr Derived
      operator<<(const Derived& x, int y)
      { return Derived([&](auto i) -> T { return x[i] << y; }); }

      friend constexpr Derived
      operator>>(const Derived& x, int y)
      { return Derived([&](auto i) -> T { return x[i] >> y; }); }

      constexpr Derived
      operator~() const
      { return Derived([&](auto i) -> T { return ~d(i); }); }
    };

  // simd (fixed_size)
  template <class T, int N>
    class simd<T, simd_abi::fixed_size<N>>
    : public fixed_simd_int_base<T, N>
    {
    private:
      friend class fixed_simd_int_base<T, N>;

      template <typename V, int M, size_t Parts>
        friend std::enable_if_t<M == Parts * V::size() && is_simd_v<V>, std::array<V, Parts>>
        split(const simd<typename V::value_type, simd_abi::fixed_size<M>>&);

      template <size_t... Sizes, typename U>
        friend std::tuple<simd<U, simd_abi::deduce_t<U, int(Sizes)>>...>
        split(const simd<U, simd_abi::fixed_size<int((Sizes + ...))>>&);

      T data[N];

      template <typename F, size_t... Is>
        constexpr
        simd(std::index_sequence<Is...>, F&& init)
        : data {static_cast<value_type>(init(detail::SizeConstant<Is>()))...}
        {}

    public:
      using value_type = T;
      using reference = T&;
      using abi_type = simd_abi::fixed_size<N>;
      using mask_type = simd_mask<T, abi_type>;

      static constexpr size_t size() noexcept
      { return N; }

      constexpr simd() = default;
      constexpr simd(const simd&) = default;
      constexpr simd(simd&&) noexcept = default;
      constexpr simd& operator=(const simd&) = default;
      constexpr simd& operator=(simd&&) noexcept = default;

      // simd constructors
      template <typename U>
        constexpr
        simd(detail::ValuePreservingOrInt<U, value_type>&& value) noexcept
        : simd([v = static_cast<value_type>(value)](size_t) { return v; })
        {}

      // conversion constructors
      template <typename U,
                typename = std::enable_if_t<
                             std::conjunction_v<detail::is_value_preserving<U, value_type>,
                                                detail::is_higher_integer_rank<value_type, U>>>>
        constexpr
        simd(const simd<U, abi_type>& x)
        : simd([&x](auto i) { return static_cast<value_type>(x[i]); })
        {}

      // generator constructor
      template <typename F>
        explicit constexpr
        simd(F&& gen, detail::ValuePreservingOrInt<
                        decltype(std::declval<F>()(std::declval<detail::SizeConstant<0>&>())),
                        value_type>* = nullptr)
        : simd(std::make_index_sequence<N>(), std::forward<F>(gen))
        {}

      // load constructor
      template <typename U, typename Flags>
        simd(const U* mem, Flags)
        : simd([mem](auto i) -> value_type { return mem[i]; })
        {}

      // loads [simd.load]
      template <typename U, typename Flags>
        void
        copy_from(const detail::Vectorizable<U>* mem, Flags)
        {
          for (int i = 0; i < N; ++i)
            data[i] = mem[i];
        }

      // stores [simd.store]
      template <typename U, typename Flags>
        void
        copy_to(detail::Vectorizable<U>* mem, Flags) const
        {
          for (int i = 0; i < N; ++i)
            mem[i] = data[i];
        }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      // increment and decrement:
      constexpr simd&
      operator++()
      {
        for (int i = 0; i < N; ++i)
          ++data[i];
        return *this;
      }

      constexpr simd
      operator++(int)
      {
        simd r = *this;
        for (int i = 0; i < N; ++i)
          ++data[i];
        return r;
      }

      constexpr simd&
      operator--()
      {
        for (int i = 0; i < N; ++i)
          --data[i];
        return *this;
      }

      constexpr simd
      operator--(int)
      {
        simd r = *this;
        for (int i = 0; i < N; ++i)
          --data[i];
        return r;
      }

      // unary operators
      constexpr mask_type
      operator!() const
      { return mask_type([&](auto i) { return !data[i]; }); }

      constexpr simd
      operator+() const
      { return *this; }

      constexpr simd
      operator-() const
      { return simd([&](auto i) -> value_type { return -data[i]; }); }

      // compound assignment [simd.cassign]
      constexpr friend simd&
      operator+=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] += x.data[i];
        return lhs;
      }

      constexpr friend simd&
      operator-=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] -= x.data[i];
        return lhs;
      }

      constexpr friend simd&
      operator*=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] *= x.data[i];
        return lhs;
      }

      constexpr friend simd&
      operator/=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] /= x.data[i];
        return lhs;
      }

      // binary operators [simd.binary]
      constexpr friend simd
      operator+(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] + y.data[i]; }); }

      constexpr friend simd
      operator-(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] - y.data[i]; }); }

      constexpr friend simd
      operator*(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] * y.data[i]; }); }

      constexpr friend simd
      operator/(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] / y.data[i]; }); }

      // compares [simd.comparison]
      constexpr friend mask_type
      operator==(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] == y.data[i]; }); }

      constexpr friend mask_type
      operator!=(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] != y.data[i]; }); }

      constexpr friend mask_type
      operator<(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] < y.data[i]; }); }

      constexpr friend mask_type
      operator<=(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] <= y.data[i]; }); }

      constexpr friend mask_type
      operator>(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] > y.data[i]; }); }

      constexpr friend mask_type
      operator>=(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] >= y.data[i]; }); }
    };

  // casts [simd.casts]
  // static_simd_cast
  template <typename T, typename U, typename A,
            typename = std::enable_if_t<detail::is_vectorizable_v<T>>>
    constexpr simd<T, A>
    static_simd_cast(const simd<U, A>& x)
    { return simd<T, A>([&x](auto i) { return static_cast<T>(x[i]); }); }

  template <typename V, typename U, typename A,
            typename = std::enable_if_t<is_simd_v<V>>>
    constexpr V
    static_simd_cast(const simd<U, A>& x)
    { return V([&x](auto i) { return static_cast<typename V::value_type>(x[i]); }); }

  template <typename T, typename U, typename A,
            typename = std::enable_if_t<detail::is_vectorizable_v<T>>>
    constexpr simd_mask<T, A>
    static_simd_cast(const simd_mask<U, A>& x)
    { return simd_mask<T, A>([&x](auto i) { return x[i]; }); }

  template <typename M, typename U, typename A,
            typename = std::enable_if_t<M::size() == simd_size_v<U, A>>>
    constexpr M
    static_simd_cast(const simd_mask<U, A>& x)
    { return M([&x](auto i) { return x[i]; }); }

  // simd_cast
  template <typename T, typename U, typename A,
            typename To = detail::value_type_or_identity_t<T>>
    constexpr auto
    simd_cast(const simd<detail::ValuePreserving<U, To>, A>& x)
    -> decltype(static_simd_cast<T>(x))
    { return static_simd_cast<T>(x); }

  // to_fixed_size
  template <typename T, int N>
    constexpr fixed_size_simd<T, N>
    to_fixed_size(const fixed_size_simd<T, N>& x)
    { return x; }

  template <typename T, int N>
    constexpr fixed_size_simd_mask<T, N>
    to_fixed_size(const fixed_size_simd_mask<T, N>& x)
    { return x; }

  template <typename T>
    constexpr fixed_size_simd<T, 1>
    to_fixed_size(const simd<T> x)
    { return x[0]; }

  template <typename T>
    constexpr fixed_size_simd_mask<T, 1>
    to_fixed_size(const simd_mask<T> x)
    { return fixed_size_simd_mask<T, 1>(x[0]); }

  // to_native
  template <typename T>
    constexpr simd<T>
    to_native(const fixed_size_simd<T, 1> x)
    { return x[0]; }

  template <typename T>
    constexpr simd_mask<T>
    to_native(const fixed_size_simd_mask<T, 1> x)
    { return simd_mask<T>(x[0]); }

  // to_compatible
  template <typename T>
    constexpr simd<T>
    to_compatible(const fixed_size_simd<T, 1> x)
    { return x[0]; }

  template <typename T>
    constexpr simd_mask<T>
    to_compatible(const fixed_size_simd_mask<T, 1> x)
    { return simd_mask<T>(x[0]); }

  // split(simd)
  template <typename V, int N, size_t Parts = N / V::size()>
    std::enable_if_t<N == Parts * V::size() && is_simd_v<V>, std::array<V, Parts>>
    split(const simd<typename V::value_type, simd_abi::fixed_size<N>>& x)
    {
      const auto* data = x.data;
      return [&]<size_t... Is>(std::index_sequence<Is...>)
               -> std::array<V, Parts> {
                 return {V(data + Is * V::size(), element_aligned)...};
               }(std::make_index_sequence<Parts>());
    }

  // split(simd_mask)
  template <typename V, int N, size_t Parts = N / V::size()>
    std::enable_if_t<N == Parts * V::size() && is_simd_mask_v<V>, std::array<V, Parts>>
    split(const simd_mask<typename V::simd_type::value_type, simd_abi::fixed_size<N>>& x)
    {
      const auto* data = x.data;
      return [&]<size_t... Is>(std::index_sequence<Is...>)
               -> std::array<V, Parts> {
                 return {V(data + Is * V::size(), element_aligned)...};
               }(std::make_index_sequence<Parts>());
    }

  // split<Sizes...>
  template <size_t... Sizes, typename T>
    std::tuple<simd<T, simd_abi::deduce_t<T, int(Sizes)>>...>
    split(const simd<T, simd_abi::fixed_size<int((Sizes + ...))>>& x)
    {
      using R = std::tuple<simd<T, simd_abi::deduce_t<T, int(Sizes)>>...>;
      const auto* data = x.data;
      return [&]<size_t... Is>(std::index_sequence<Is...>) -> R {
        constexpr size_t offsets[sizeof...(Sizes)] = {
          []<size_t... Js>(std::index_sequence<Js...>) {
            constexpr size_t sizes[sizeof...(Sizes)] = {Sizes...};
            return (sizes[Js] + ... + 0);
          }(std::make_index_sequence<Is>())...
        };
        return {simd<T, simd_abi::deduce_t<T, int(Sizes)>>(data + offsets[Is],
                                                           element_aligned)...};
      }(std::make_index_sequence<sizeof...(Sizes)>());
    }

  // split<V>(V)
  template <typename V>
    std::enable_if_t<std::disjunction_v<is_simd<V>, is_simd_mask<V>>, std::array<V, 1>>
    split(const V& x)
    { return {x}; }

  // concat(simd...)
  template <typename T, typename... As>
    inline constexpr
    simd<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>
    concat(const simd<T, As>&... xs)
    {
      using R = simd<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>;
      return R([&](auto i) {
               return detail::pack_simd_subscript<i>(xs...);
             });
    }

  // concat(simd_mask...)
  template <typename T, typename... As>
    inline constexpr
    simd_mask<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>
    concat(const simd_mask<T, As>&... xs)
    {
      using R = simd_mask<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>;
      return R([&](auto i) -> bool {
               return detail::pack_simd_subscript<i>(xs...);
             });
    }

  // concat(array<simd>)
  template <typename T, typename A, size_t N>
    inline constexpr
    simd<T, simd_abi::deduce_t<T, N * simd_size_v<T, A>>>
    concat(const std::array<simd<T, A>, N>& x)
    {
      constexpr int K = simd_size_v<T, A>;
      using R = simd<T, simd_abi::deduce_t<T, N * K>>;
      return R([&](auto i) {
               return x[i / K][i % K];
             });
    }

  // concat(array<simd_mask>)
  template <typename T, typename A, size_t N>
    inline constexpr
    simd_mask<T, simd_abi::deduce_t<T, N * simd_size_v<T, A>>>
    concat(const std::array<simd_mask<T, A>, N>& x)
    {
      constexpr int K = simd_size_v<T, A>;
      using R = simd_mask<T, simd_abi::deduce_t<T, N * K>>;
      return R([&](auto i) -> bool {
               return x[i / K][i % K];
             });
    }

  // const_where_expression<M, T>
  template <typename M, typename V>
    class const_where_expression
    {
      static_assert(std::is_same_v<V, detail::remove_cvref_t<V>>);

      struct Wrapper { using value_type = V; };

    protected:
      using value_type =
        typename std::conditional_t<std::is_arithmetic_v<V>, Wrapper, V>::value_type;

      friend const M&
      get_mask(const const_where_expression& x)
      { return x.m_k; }

      friend const V&
      get_lvalue(const const_where_expression& x)
      { return x.m_value; }

      const M& m_k;
      V& m_value;

    public:
      const_where_expression(const const_where_expression&) = delete;
      const_where_expression& operator=(const const_where_expression&) = delete;

      constexpr const_where_expression(const M& kk, const V& dd)
      : m_k(kk), m_value(const_cast<V&>(dd)) {}

      constexpr V
      operator-() const &&
      {
        return V([&](auto i) {
                 return m_k[i] ? static_cast<value_type>(-m_value[i]) : m_value[i];
               });
      }

      template <typename Up, typename Flags>
        [[nodiscard]] constexpr V
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        {
          return V([&](auto i) {
                   return m_k[i] ? static_cast<value_type>(mem[i]) : m_value[i];
                 });
        }

      template <typename Up, typename Flags>
        constexpr void
        copy_to(detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        {
          for (size_t i = 0; i < V::size(); ++i)
            {
              if (m_k[i])
                mem[i] = static_cast<Up>(m_value[i]);
            }
        }
    };

  // const_where_expression<bool, T>
  template <typename V>
    class const_where_expression<bool, V>
    {
      using M = bool;

      static_assert(std::is_same_v<V, detail::remove_cvref_t<V>>);

      struct Wrapper { using value_type = V; };

    protected:
      using value_type =
        typename std::conditional_t<std::is_arithmetic_v<V>, Wrapper, V>::value_type;

      friend const M&
      get_mask(const const_where_expression& x)
      { return x.m_k; }

      friend const V&
      get_lvalue(const const_where_expression& x)
      { return x.m_value; }

      const bool m_k;
      V& m_value;

    public:
      const_where_expression(const const_where_expression&) = delete;
      const_where_expression& operator=(const const_where_expression&) = delete;

      constexpr const_where_expression(const bool kk, const V& dd)
      : m_k(kk), m_value(const_cast<V&>(dd)) {}

      constexpr V
      operator-() const &&
      { return m_k ? -m_value : m_value; }

      template <typename Up, typename Flags>
        [[nodiscard]] constexpr V
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        { return m_k ? static_cast<V>(mem[0]) : m_value; }

      template <typename Up, typename Flags>
        constexpr void
        copy_to(detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        {
          if (m_k)
            mem[0] = m_value;
        }
    };

  // where_expression<M, T>
  template <typename M, typename V>
    class where_expression : public const_where_expression<M, V>
    {
      static_assert(not std::is_const_v<V>,
                    "where_expression may only be instantiated with a non-const V parameter");

      using typename const_where_expression<M, V>::value_type;
      using const_where_expression<M, V>::m_k;
      using const_where_expression<M, V>::m_value;

      static_assert(std::is_same_v<typename M::abi_type, typename V::abi_type>);
      static_assert(M::size() == V::size());

      friend V&
      get_lvalue(where_expression& x)
      { return x.m_value; }

      template <typename Up>
        constexpr auto
        as_simd(Up&& x)
        {
          using UU = detail::remove_cvref_t<Up>;
          if constexpr (std::is_same_v<V, UU>)
            return x;
          else if constexpr (std::is_convertible_v<Up&&, value_type>)
            return V(static_cast<value_type>(static_cast<Up&&>(x)));
          else if constexpr (std::is_convertible_v<Up&&, V>)
            return static_cast<V>(static_cast<Up&&>(x));
          else
            return static_simd_cast<V>(static_cast<Up&&>(x));
        }

    public:
      where_expression(const where_expression&) = delete;
      where_expression& operator=(const where_expression&) = delete;

      constexpr where_expression(const M& kk, V& dd)
      : const_where_expression<M, V>(kk, dd)
      {}

      template <typename Up>
        constexpr void
        operator=(Up&& x) &&
        {
          const V& rhs = as_simd(x);
          for (size_t i = 0; i < V::size(); ++i)
            {
              if (m_k[i])
                m_value[i] = rhs[i];
            }
        }

#define SIMD_OP_(op)                              \
      template <typename Up>                      \
        constexpr void                            \
        operator op##=(Up&& x) &&                 \
        {                                         \
          const V& rhs = as_simd(x);              \
          for (size_t i = 0; i < V::size(); ++i)  \
            {                                     \
              if (m_k[i])                         \
                m_value[i] op##= rhs[i];          \
            }                                     \
        }                                         \
      static_assert(true)
      SIMD_OP_(+);
      SIMD_OP_(-);
      SIMD_OP_(*);
      SIMD_OP_(/);
      SIMD_OP_(%);
      SIMD_OP_(&);
      SIMD_OP_(|);
      SIMD_OP_(^);
      SIMD_OP_(<<);
      SIMD_OP_(>>);
#undef SIMD_OP_

      constexpr void operator++() &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              ++m_value[i];
          }
      }

      constexpr void operator++(int) &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              ++m_value[i];
          }
      }

      constexpr void operator--() &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              --m_value[i];
          }
      }

      constexpr void operator--(int) &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              --m_value[i];
          }
      }

      // intentionally hides const_where_expression::copy_from
      template <typename Up, typename Flags>
        constexpr void
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) &&
        {
          for (size_t i = 0; i < V::size(); ++i)
            {
              if (m_k[i])
                m_value[i] = mem[i];
            }
        }
    };

  // where_expression<bool, T>
  template <typename V>
    class where_expression<bool, V> : public const_where_expression<bool, V>
    {
      using M = bool;
      using typename const_where_expression<M, V>::value_type;
      using const_where_expression<M, V>::m_k;
      using const_where_expression<M, V>::m_value;

    public:
      where_expression(const where_expression&) = delete;
      where_expression& operator=(const where_expression&) = delete;

      constexpr where_expression(const M& kk, V& dd)
      : const_where_expression<M, V>(kk, dd) {}

#define SIMD_OP_(op)                                \
      template <typename Up>                        \
        constexpr void operator op(Up&& x) &&       \
        { if (m_k) m_value op static_cast<Up&&>(x); }

      SIMD_OP_(=)
      SIMD_OP_(+=)
      SIMD_OP_(-=)
      SIMD_OP_(*=)
      SIMD_OP_(/=)
      SIMD_OP_(%=)
      SIMD_OP_(&=)
      SIMD_OP_(|=)
      SIMD_OP_(^=)
      SIMD_OP_(<<=)
      SIMD_OP_(>>=)
#undef SIMD_OP_

      constexpr void operator++() &&
      { if (m_k) ++m_value; }

      constexpr void operator++(int) &&
      { if (m_k) ++m_value; }

      constexpr void operator--() &&
      { if (m_k) --m_value; }

      constexpr void operator--(int) &&
      { if (m_k) --m_value; }

      // intentionally hides const_where_expression::copy_from
      template <typename Up, typename Flags>
        constexpr void
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) &&
        { if (m_k) m_value = mem[0]; }
    };

  // where
  template <typename Tp, typename Ap>
    constexpr where_expression<simd_mask<Tp, Ap>, simd<Tp, Ap>>
    where(const typename simd<Tp, Ap>::mask_type& k, simd<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr const_where_expression<simd_mask<Tp, Ap>, simd<Tp, Ap>>
    where(const typename simd<Tp, Ap>::mask_type& k,
          const simd<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr where_expression<simd_mask<Tp, Ap>, simd_mask<Tp, Ap>>
    where(const std::remove_const_t<simd_mask<Tp, Ap>>& k,
          simd_mask<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr const_where_expression<simd_mask<Tp, Ap>, simd_mask<Tp, Ap>>
    where(const std::remove_const_t<simd_mask<Tp, Ap>>& k,
          const simd_mask<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp>
    constexpr where_expression<bool, Tp>
    where(detail::ExactBool k, Tp& value)
    { return {k, value}; }

  template <typename Tp>
    constexpr const_where_expression<bool, Tp>
    where(detail::ExactBool k, const Tp& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr void
    where(bool k, simd<Tp, Ap>& value) = delete;

  template <typename Tp, typename Ap>
    constexpr void
    where(bool k, const simd<Tp, Ap>& value) = delete;

  // reductions [simd.reductions]
  template <typename T, typename A, typename BinaryOperation = std::plus<>>
    constexpr T
    reduce(const simd<T, A>& v,
           BinaryOperation binary_op = BinaryOperation())
    {
      constexpr int N = simd_size_v<T, A>;
      if constexpr (N > 3)
        {
          constexpr int N2 = detail::bit_floor(N / 2);
          constexpr int NRem = N - 2 * N2;
          if constexpr (NRem > 0)
            {
              const auto [l, r, rem] = split<N2, N2, N - 2 * N2>(v);
              return binary_op(reduce(binary_op(l, r), binary_op), reduce(rem, binary_op));
            }
          else
            {
              const auto [l, r] = split<N2, N2>(v);
              return reduce(binary_op(l, r), binary_op);
            }
        }
      else
        {
          T r = v[0];
          for (size_t i = 1; i < simd_size_v<T, A>; ++i)
            r = binary_op(r, v[i]);
          return r;
        }
    }

  template <typename M, typename V, typename BinaryOperation = std::plus<>>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x,
        typename V::value_type identity_element,
        BinaryOperation binary_op)
    {
      const M& k = get_mask(x);
      const V& v = get_lvalue(x);
      auto r = identity_element;
      if (any_of(k)) [[likely]]
        {
          for (size_t i = 0; i < V::size(); ++i)
            if (k[i])
              r = binary_op(r, v[i]);
        }
      return r;
    }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::plus<> binary_op = {})
    { return reduce(x, 0, binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::multiplies<> binary_op)
    { return reduce(x, 1, binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::bit_and<> binary_op)
    { return reduce(x, ~typename V::value_type(), binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::bit_or<> binary_op)
    { return reduce(x, 0, binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::bit_xor<> binary_op)
    { return reduce(x, 0, binary_op); }

  template <typename T, typename A>
    constexpr T
    hmin(const simd<T, A>& v) noexcept
    {
      return reduce(v, [](const auto& l, const auto& r) {
               using std::min;
               return min(l, r);
             });
    }

  template <typename T, typename A>
    constexpr T
    hmax(const simd<T, A>& v) noexcept
    {
      return reduce(v, [](const auto& l, const auto& r) {
               using std::max;
               return max(l, r);
             });
    }

  template <typename M, typename V>
    constexpr typename V::value_type
    hmin(const const_where_expression<M, V>& x) noexcept
    {
      using T = typename V::value_type;
      constexpr T id_elem =
#ifdef __FINITE_MATH_ONLY__
        std::numeric_limits<T>::max();
#else
        std::numeric_limits<T>::infinity();
#endif
      return reduce(x, id_elem, [](const auto& l, const auto& r) {
               using std::min;
               return min(l, r);
             });
    }

  template <typename M, typename V>
    constexpr
    typename V::value_type
    hmax(const const_where_expression<M, V>& x) noexcept
    {
      using T = typename V::value_type;
      constexpr T id_elem =
#ifdef __FINITE_MATH_ONLY__
        std::numeric_limits<T>::lowest();
#else
        -std::numeric_limits<T>::infinity();
#endif
      return reduce(x, id_elem, [](const auto& l, const auto& r) {
               using std::max;
               return max(l, r);
             });
    }

  // algorithms [simd.alg]
  template <typename T, typename A>
    constexpr simd<T, A>
    min(const simd<T, A>& a, const simd<T, A>& b)
    { return simd<T, A>([&](auto i) { return std::min(a[i], b[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    max(const simd<T, A>& a, const simd<T, A>& b)
    { return simd<T, A>([&](auto i) { return std::max(a[i], b[i]); }); }

  template <typename T, typename A>
    constexpr
    std::pair<simd<T, A>, simd<T, A>>
    minmax(const simd<T, A>& a, const simd<T, A>& b)
    { return {min(a, b), max(a, b)}; }

  template <typename T, typename A>
    constexpr simd<T, A>
    clamp(const simd<T, A>& v, const simd<T, A>& lo,
        const simd<T, A>& hi)
    { return simd<T, A>([&](auto i) { return std::clamp(v[i], lo[i], hi[i]); }); }

  // math
#define SIMD_MATH_1ARG(name, return_temp)                                                          \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x) noexcept                                      \
    { return return_temp<T, A>([&x](auto i) { return std::name(x[i]); }); }

#define SIMD_MATH_1ARG_FIXED(name, R)                                                              \
  template <typename T, typename A>                                                                \
    constexpr fixed_size_simd<R, simd_size_v<T, A>>                                                \
    name(const simd<detail::FloatingPoint<T>, A>& x) noexcept                                      \
    { return fixed_size_simd<R, simd_size_v<T, A>>([&x](auto i) { return std::name(x[i]); }); }

#define SIMD_MATH_2ARG(name, return_temp)                                                          \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x, const simd<T, A>& y) noexcept                 \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i]); }); }                   \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x,                                               \
         const detail::type_identity_t<simd<T, A>>& y) noexcept                                    \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i]); }); }                   \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const detail::type_identity_t<simd<T, A>>& x,                                             \
         const simd<detail::FloatingPoint<T>, A>& y) noexcept                                      \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i]); }); }

#define SIMD_MATH_3ARG(name, return_temp)                                                          \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x,                                               \
         const simd<T, A>& y, const simd<T, A> &z) noexcept                                        \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }             \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x,                                               \
         const detail::type_identity_t<simd<T, A>>& y,                                             \
         const detail::type_identity_t<simd<T, A>> &z) noexcept                                    \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }             \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const detail::type_identity_t<simd<T, A>>& x,                                             \
         const simd<detail::FloatingPoint<T>, A>& y,                                               \
         const detail::type_identity_t<simd<T, A>> &z) noexcept                                    \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }             \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const detail::type_identity_t<simd<T, A>>& x,                                             \
         const detail::type_identity_t<simd<T, A>>& y,                                             \
         const simd<detail::FloatingPoint<T>, A> &z) noexcept                                      \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }

  template <typename T, typename A, typename U = detail::SignedIntegral<T>>
    constexpr simd<T, A>
    abs(const simd<T, A>& x) noexcept
    { return simd<T, A>([&x](auto i) { return std::abs(x[i]); }); }

  SIMD_MATH_1ARG(abs, simd)
  SIMD_MATH_1ARG(isnan, simd_mask)
  SIMD_MATH_1ARG(isfinite, simd_mask)
  SIMD_MATH_1ARG(isinf, simd_mask)
  SIMD_MATH_1ARG(isnormal, simd_mask)
  SIMD_MATH_1ARG(signbit, simd_mask)
  SIMD_MATH_1ARG_FIXED(fpclassify, int)

  SIMD_MATH_2ARG(hypot, simd)
  SIMD_MATH_3ARG(hypot, simd)

  template <typename T, typename A>
    constexpr simd<T, A>
    remquo(const simd<T, A>& x, const simd<T, A>& y,
           fixed_size_simd<int, simd_size_v<T, A>>* quo) noexcept
    { return simd<T, A>([&x, &y, quo](auto i) { return std::remquo(x[i], y[i], &(*quo)[i]); }); }

  SIMD_MATH_1ARG(erf, simd)
  SIMD_MATH_1ARG(erfc, simd)
  SIMD_MATH_1ARG(tgamma, simd)
  SIMD_MATH_1ARG(lgamma, simd)

  SIMD_MATH_2ARG(pow, simd)
  SIMD_MATH_2ARG(fmod, simd)
  SIMD_MATH_2ARG(remainder, simd)
  SIMD_MATH_2ARG(nextafter, simd)
  SIMD_MATH_2ARG(copysign, simd)
  SIMD_MATH_2ARG(fdim, simd)
  SIMD_MATH_2ARG(fmax, simd)
  SIMD_MATH_2ARG(fmin, simd)
  SIMD_MATH_2ARG(isgreater, simd_mask)
  SIMD_MATH_2ARG(isgreaterequal, simd_mask)
  SIMD_MATH_2ARG(isless, simd_mask)
  SIMD_MATH_2ARG(islessequal, simd_mask)
  SIMD_MATH_2ARG(islessgreater, simd_mask)
  SIMD_MATH_2ARG(isunordered, simd_mask)

  template <typename T, typename A>
    constexpr simd<T, A>
    modf(const simd<detail::FloatingPoint<T>, A>& x, simd<T, A>* iptr) noexcept
    { return simd<T, A>([&x, iptr](auto i) { return std::modf(x[i], &(*iptr)[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    frexp(const simd<detail::FloatingPoint<T>, A>& x,
          fixed_size_simd<int, simd_size_v<T, A>>* exp) noexcept
    { return simd<T, A>([&x, exp](auto i) { return std::frexp(x[i], &(*exp)[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    scalbln(const simd<detail::FloatingPoint<T>, A>& x,
            const fixed_size_simd<long int, simd_size_v<T, A>>& exp) noexcept
    { return simd<T, A>([&x, &exp](auto i) { return std::scalbln(x[i], exp[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    scalbn(const simd<detail::FloatingPoint<T>, A>& x,
           const fixed_size_simd<int, simd_size_v<T, A>>& exp) noexcept
    { return simd<T, A>([&x, &exp](auto i) { return std::scalbn(x[i], exp[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    ldexp(const simd<detail::FloatingPoint<T>, A>& x,
          const fixed_size_simd<int, simd_size_v<T, A>>& exp) noexcept
    { return simd<T, A>([&x, &exp](auto i) { return std::ldexp(x[i], exp[i]); }); }

  SIMD_MATH_1ARG(sqrt, simd)

  SIMD_MATH_3ARG(fma, simd)

  SIMD_MATH_1ARG(trunc, simd)
  SIMD_MATH_1ARG(ceil, simd)
  SIMD_MATH_1ARG(floor, simd)
  SIMD_MATH_1ARG(round, simd)
  SIMD_MATH_1ARG_FIXED(lround, long)
  SIMD_MATH_1ARG_FIXED(llround, long long)
  SIMD_MATH_1ARG(nearbyint, simd)
  SIMD_MATH_1ARG(rint, simd)
  SIMD_MATH_1ARG_FIXED(lrint, long)
  SIMD_MATH_1ARG_FIXED(llrint, long long)
  SIMD_MATH_1ARG_FIXED(ilogb, int)

  // trig functions
  SIMD_MATH_1ARG(sin, simd)
  SIMD_MATH_1ARG(cos, simd)
  SIMD_MATH_1ARG(tan, simd)
  SIMD_MATH_1ARG(asin, simd)
  SIMD_MATH_1ARG(acos, simd)
  SIMD_MATH_1ARG(atan, simd)
  SIMD_MATH_2ARG(atan2, simd)
  SIMD_MATH_1ARG(sinh, simd)
  SIMD_MATH_1ARG(cosh, simd)
  SIMD_MATH_1ARG(tanh, simd)
  SIMD_MATH_1ARG(asinh, simd)
  SIMD_MATH_1ARG(acosh, simd)
  SIMD_MATH_1ARG(atanh, simd)

  // logarithms
  SIMD_MATH_1ARG(log, simd)
  SIMD_MATH_1ARG(log10, simd)
  SIMD_MATH_1ARG(log1p, simd)
  SIMD_MATH_1ARG(log2, simd)
  SIMD_MATH_1ARG(logb, simd)

#undef SIMD_MATH_1ARG
#undef SIMD_MATH_1ARG_FIXED
#undef SIMD_MATH_2ARG
#undef SIMD_MATH_3ARG
}
#ifdef VIR_SIMD_TS_DROPIN
}

namespace vir::stdx
{
  using namespace std::experimental::parallelism_v2;
}
#endif

#endif
#endif  // VIR_SIMD_H_
#ifndef GNURADIO_TYPELIST_HPP
#define GNURADIO_TYPELIST_HPP

#include <bit>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <string_view>
#include <string>

namespace fair::meta {

template<typename... Ts>
struct typelist;

// concat ///////////////
namespace detail {
template<typename...>
struct concat_impl;

template<>
struct concat_impl<> {
    using type = typelist<>;
};

template<typename A>
struct concat_impl<A> {
    using type = typelist<A>;
};

template<typename... As>
struct concat_impl<typelist<As...>> {
    using type = typelist<As...>;
};

template<typename A, typename B>
struct concat_impl<A, B> {
    using type = typelist<A, B>;
};

template<typename... As, typename B>
struct concat_impl<typelist<As...>, B> {
    using type = typelist<As..., B>;
};

template<typename A, typename... Bs>
struct concat_impl<A, typelist<Bs...>> {
    using type = typelist<A, Bs...>;
};

template<typename... As, typename... Bs>
struct concat_impl<typelist<As...>, typelist<Bs...>> {
    using type = typelist<As..., Bs...>;
};

template<typename A, typename B, typename C>
struct concat_impl<A, B, C> {
    using type = typename concat_impl<typename concat_impl<A, B>::type, C>::type;
};

template<typename A, typename B, typename C, typename D, typename... More>
struct concat_impl<A, B, C, D, More...> {
    using type =
            typename concat_impl<typename concat_impl<A, B>::type, typename concat_impl<C, D>::type,
                                 typename concat_impl<More...>::type>::type;
};
} // namespace detail

template<typename... Ts>
using concat = typename detail::concat_impl<Ts...>::type;

// split_at, left_of, right_of ////////////////
namespace detail {
template<unsigned N>
struct splitter;

template<>
struct splitter<0> {
    template<typename...>
    using first = typelist<>;
    template<typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<1> {
    template<typename T0, typename...>
    using first = typelist<T0>;
    template<typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<2> {
    template<typename T0, typename T1, typename...>
    using first = typelist<T0, T1>;
    template<typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<4> {
    template<typename T0, typename T1, typename T2, typename T3, typename...>
    using first = typelist<T0, T1, T2, T3>;
    template<typename, typename, typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<8> {
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7, typename...>
    using first = typelist<T0, T1, T2, T3, T4, T5, T6, T7>;

    template<typename, typename, typename, typename, typename, typename, typename, typename,
             typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<16> {
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14, typename T15, typename...>
    using first = typelist<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>;

    template<typename, typename, typename, typename, typename, typename, typename, typename, typename,
             typename, typename, typename, typename, typename, typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<unsigned N>
struct splitter {
    static constexpr unsigned FirstSplit = std::has_single_bit(N) ? N / 2 : std::bit_floor(N);
    using A                              = splitter<FirstSplit>;
    using B                              = splitter<N - FirstSplit>;

    template<typename... Ts>
    using first = concat<typename A::template first<Ts...>,
                         typename B::template first<typename A::template second<Ts...>>>;

    template<typename... Ts>
    using second = typename B::template second<typename A::template second<Ts...>>;
};
} // namespace detail

template<unsigned N, typename List>
struct split_at;

template<unsigned N, typename... Ts>
struct split_at<N, typelist<Ts...>> {
    using first  = typename detail::splitter<N>::template first<Ts...>;
    using second = typename detail::splitter<N>::template second<Ts...>;
};

template<std::size_t N, typename List>
using left_of = typename split_at<N, List>::first;

template<std::size_t N, typename List>
using right_of = typename split_at<N + 1, List>::second;

// remove_at /////////////
template<std::size_t Idx, typename List>
using remove_at = concat<left_of<Idx, List>, right_of<Idx, List>>;

// first_type ////////////
namespace detail {
template<typename List>
struct first_type_impl {};

template<typename T0, typename... Ts>
struct first_type_impl<typelist<T0, Ts...>> {
    using type = T0;
};
} // namespace detail

template<typename List>
using first_type = typename detail::first_type_impl<List>::type;

// transform_types ////////////
namespace detail {
template<template<typename> class Template, typename List>
struct transform_types_impl;

template<template<typename> class Template, typename... Ts>
struct transform_types_impl<Template, typelist<Ts...>> {
    using type = typelist<Template<Ts>...>;
};
} // namespace detail

template<template<typename> class Template, typename List>
using transform_types = typename detail::transform_types_impl<Template, List>::type;

// transform_value_type
template<typename T>
using transform_value_type = typename T::value_type;

// reduce ////////////////
namespace detail {
template<template<typename, typename> class Method, typename List>
struct reduce_impl;

template<template<typename, typename> class Method, typename T0>
struct reduce_impl<Method, typelist<T0>> {
    using type = T0;
};

template<template<typename, typename> class Method, typename T0, typename T1, typename... Ts>
struct reduce_impl<Method, typelist<T0, T1, Ts...>>
    : public reduce_impl<Method, typelist<typename Method<T0, T1>::type, Ts...>> {};

template<template<typename, typename> class Method, typename T0, typename T1, typename T2,
         typename T3, typename... Ts>
struct reduce_impl<Method, typelist<T0, T1, T2, T3, Ts...>>
    : public reduce_impl<
              Method, typelist<typename Method<T0, T1>::type, typename Method<T2, T3>::type, Ts...>> {
};
} // namespace detail

template<template<typename, typename> class Method, typename List>
using reduce = typename detail::reduce_impl<Method, List>::type;

// typelist /////////////////
template<typename T>
concept is_typelist_v = requires { typename T::typelist_tag; };

template<typename... Ts>
struct typelist {
    using this_t = typelist<Ts...>;
    using typelist_tag = std::true_type;

    static inline constexpr std::integral_constant<std::size_t, sizeof...(Ts)> size = {};

    template<template<typename...> class Other>
    using apply = Other<Ts...>;

    template<std::size_t I>
    using at = first_type<typename detail::splitter<I>::template second<Ts...>>;

    template<typename... Other>
    static constexpr inline bool are_equal = std::same_as<typelist, meta::typelist<Other...>>;

    template<typename... Other>
    static constexpr inline bool are_convertible_to = (std::convertible_to<Ts, Other> && ...);

    template<typename... Other>
    static constexpr inline bool are_convertible_from = (std::convertible_to<Other, Ts> && ...);

    template<typename F, typename Tup>
        requires(sizeof...(Ts) == std::tuple_size_v<std::remove_cvref_t<Tup>>)
    static constexpr auto
    construct(Tup &&args_tuple) {
        return std::apply(
                []<typename... Args>(Args &&...args) {
                    return std::make_tuple(F::template apply<Ts>(std::forward<Args>(args))...);
                },
                std::forward<Tup>(args_tuple));
    }

    template<template<typename> typename Trafo>
    using transform = meta::transform_types<Trafo, this_t>;

    template<template<typename...> typename Pred>
    constexpr static bool all_of = (Pred<Ts>::value && ...);

    template<template<typename...> typename Pred>
    constexpr static bool none_of = (!Pred<Ts>::value && ...);

    template<template<typename...> typename Predicate>
    using filter = concat<std::conditional_t<Predicate<Ts>::value, typelist<Ts>, typelist<>>...>;

    using tuple_type    = std::tuple<Ts...>;
    using tuple_or_type = std::remove_pointer_t<decltype(
            [] {
                if constexpr (sizeof...(Ts) == 0) {
                    return static_cast<void*>(nullptr);
                } else if constexpr (sizeof...(Ts) == 1) {
                    return static_cast<at<0>*>(nullptr);
                } else {
                    return static_cast<tuple_type*>(nullptr);
                }
            }())>;

};




} // namespace fair::meta

#endif // include guard
#ifndef GNURADIO_GRAPH_UTILS_HPP
#define GNURADIO_GRAPH_UTILS_HPP

#include <functional>
#include <iostream>
#include <string>
#include <string_view>


#ifndef __EMSCRIPTEN__
#include <cxxabi.h>
#include <iostream>
#include <typeinfo>
#endif

namespace fair::meta {

template<typename... Ts>
struct print_types;

template<typename CharT, std::size_t SIZE>
struct fixed_string {
    constexpr static std::size_t N            = SIZE;
    CharT                        _data[N + 1] = {};

    constexpr explicit(false) fixed_string(const CharT (&str)[N + 1]) noexcept {
        if constexpr (N != 0)
            for (std::size_t i = 0; i < N; ++i) _data[i] = str[i];
    }

    [[nodiscard]] constexpr std::size_t
    size() const noexcept {
        return N;
    }

    [[nodiscard]] constexpr bool
    empty() const noexcept {
        return N == 0;
    }

    [[nodiscard]] constexpr explicit operator std::string_view() const noexcept { return { _data, N }; }

    [[nodiscard]] explicit           operator std::string() const noexcept { return { _data, N }; }

    [[nodiscard]]                    operator const char *() const noexcept { return _data; }

    [[nodiscard]] constexpr bool
    operator==(const fixed_string &other) const noexcept {
        return std::string_view{ _data, N } == std::string_view(other);
    }

    template<std::size_t N2>
    [[nodiscard]] friend constexpr bool
    operator==(const fixed_string &, const fixed_string<CharT, N2> &) {
        return false;
    }
};

template<typename CharT, std::size_t N>
fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;

template<typename T>
[[nodiscard]] std::string
type_name() noexcept {
#ifndef __EMSCRIPTEN__
    std::string type_name = typeid(T).name();
    int         status;
    char       *demangled_name = abi::__cxa_demangle(type_name.c_str(), nullptr, nullptr, &status);
    if (status == 0) {
        return demangled_name;
    } else {
        return typeid(T).name();
    }
#else
    return typeid(T).name(); // TODO: to be replaced by refl-cpp
#endif
}

template<fixed_string val>
struct message_type {};

template<class... T>
constexpr bool always_false = false;

struct dummy_t {};

template<typename F, typename... Args>
auto
invoke_void_wrapped(F &&f, Args &&...args) {
    if constexpr (std::is_same_v<void, std::invoke_result_t<F, Args...>>) {
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
        return dummy_t{};
    } else {
        return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    }
}

static_assert(std::is_same_v<decltype(invoke_void_wrapped([] {})), dummy_t>);
static_assert(std::is_same_v<decltype(invoke_void_wrapped([] { return 42; })), int>);
static_assert(std::is_same_v<decltype(invoke_void_wrapped([](int) {}, 42)), dummy_t>);
static_assert(std::is_same_v<decltype(invoke_void_wrapped([](int i) { return i; }, 42)), int>);

#if HAVE_SOURCE_LOCATION
[[gnu::always_inline]] inline void
precondition(bool cond, const std::source_location loc = std::source_location::current()) {
    struct handle {
        [[noreturn]] static void
        failure(std::source_location const &loc) {
            std::clog << "failed precondition in " << loc.file_name() << ':' << loc.line() << ':' << loc.column() << ": `" << loc.function_name() << "`\n";
            __builtin_trap();
        }
    };

    if (not cond) [[unlikely]]
        handle::failure(loc);
}
#else
[[gnu::always_inline]] inline void
precondition(bool cond) {
    struct handle {
        [[noreturn]] static void
        failure() {
            std::clog << "failed precondition\n";
            __builtin_trap();
        }
    };

    if (not cond) [[unlikely]]
        handle::failure();
}
#endif

namespace stdx = vir::stdx;

template<typename V, typename T = void>
concept any_simd = stdx::is_simd_v<V> && (std::same_as<T, void> || std::same_as<T, typename V::value_type>);

template<typename V, typename T>
concept t_or_simd = std::same_as<V, T> || any_simd<V, T>;

template<typename T>
concept vectorizable = std::constructible_from<stdx::simd<T>>;

template<typename A, typename B>
struct wider_native_simd_size : std::conditional<(stdx::native_simd<A>::size() > stdx::native_simd<B>::size()), A, B> {};

template<typename A>
struct wider_native_simd_size<A, A> {
    using type = A;
};

template<typename V>
struct rebind_simd_helper {
    template<typename T>
    using rebind = stdx::rebind_simd_t<T, V>;
};

struct simd_load_ctor {
    template<any_simd W>
    static constexpr W
    apply(typename W::value_type const *ptr) {
        return W(ptr, stdx::element_aligned);
    }
};

template<typename List>
using reduce_to_widest_simd = stdx::native_simd<meta::reduce<wider_native_simd_size, List>>;

template<typename V, typename List>
using transform_by_rebind_simd = meta::transform_types<rebind_simd_helper<V>::template rebind, List>;

template<typename List>
using transform_to_widest_simd = transform_by_rebind_simd<reduce_to_widest_simd<List>, List>;

template<typename Node>
concept source_node = requires(Node &node, typename Node::input_port_types::tuple_type const &inputs) {
                          {
                              [](Node &n, auto &inputs) {
                                  if constexpr (Node::input_port_types::size > 0) {
                                      return []<std::size_t... Is>(Node & n_inside, auto const &tup, std::index_sequence<Is...>)->decltype(n_inside.process_one(std::get<Is>(tup)...)) { return {}; }
                                      (n, inputs, std::make_index_sequence<Node::input_port_types::size>());
                                  } else {
                                      return n.process_one();
                                  }
                              }(node, inputs)
                              } -> std::same_as<typename Node::return_type>;
                      };

template<typename Node>
concept sink_node = requires(Node &node, typename Node::input_port_types::tuple_type const &inputs) {
                        {
                            [](Node &n, auto &inputs) {
                                []<std::size_t... Is>(Node & n_inside, auto const &tup, std::index_sequence<Is...>) {
                                    if constexpr (Node::output_port_types::size > 0) {
                                        auto a [[maybe_unused]] = n_inside.process_one(std::get<Is>(tup)...);
                                    } else {
                                        n_inside.process_one(std::get<Is>(tup)...);
                                    }
                                }
                                (n, inputs, std::make_index_sequence<Node::input_port_types::size>());
                            }(node, inputs)
                        };
                    };

template<typename Node>
concept any_node = source_node<Node> || sink_node<Node>;

template<typename Node>
concept node_can_process_simd = any_node<Node> && requires(Node &n, typename transform_to_widest_simd<typename Node::input_port_types>::template apply<std::tuple> const &inputs) {
                                                      {
                                                          []<std::size_t... Is>(Node & n, auto const &tup, std::index_sequence<Is...>)->decltype(n.process_one(std::get<Is>(tup)...)) { return {}; }
                                                          (n, inputs, std::make_index_sequence<Node::input_port_types::size>())
                                                          } -> any_simd<typename Node::return_type>;
                                                  };

template<fixed_string Name, typename PortList>
consteval int
indexForName() {
    auto helper = []<std::size_t... Ids>(std::index_sequence<Ids...>) {
        int result = -1;
        ((PortList::template at<Ids>::static_name() == Name ? (result = Ids) : 0), ...);
        return result;
    };
    return helper(std::make_index_sequence<PortList::size>());
}

template<typename... Lambdas>
struct overloaded : Lambdas... {
    using Lambdas::operator()...;
};

template<typename... Lambdas>
overloaded(Lambdas...) -> overloaded<Lambdas...>;

namespace detail {
template<template<typename...> typename Mapper, template<typename...> typename Wrapper, typename... Args>
Wrapper<Mapper<Args>...> *
type_transform_impl(Wrapper<Args...> *);

template<template<typename...> typename Mapper, typename T>
Mapper<T> *
type_transform_impl(T *);

template<template<typename...> typename Mapper>
void *
type_transform_impl(void *);

template<template<typename...> typename Mapper>
fair::meta::dummy_t *
type_transform_impl(fair::meta::dummy_t *);
} // namespace detail

template<template<typename...> typename Mapper, typename T>
using type_transform = std::remove_pointer_t<decltype(detail::type_transform_impl<Mapper>(static_cast<T *>(nullptr)))>;

template<typename Arg, typename... Args>
auto safe_min(Arg&& arg, Args&&... args)
{
    if constexpr (sizeof...(Args) == 0) {
        return arg;
    } else {
        return std::min(std::forward<Arg>(arg), std::forward<Args>(args)...);
    }
}

template<typename Function, typename Tuple, typename... Tuples>
auto tuple_for_each(Function&& function, Tuple&& tuple, Tuples&&... tuples)
{
    static_assert(((std::tuple_size_v<std::remove_cvref_t<Tuple>> == std::tuple_size_v<std::remove_cvref_t<Tuples>>) && ...));
    return [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        auto callFunction = [&function, &tuple, &tuples...]<std::size_t I>() {
            function(std::get<I>(tuple), std::get<I>(tuples)...);
        };
        ((callFunction.template operator()<Idx>(), ...));
    }(std::make_index_sequence<std::tuple_size_v<Tuple>>());
}

template<typename Function, typename Tuple, typename... Tuples>
auto tuple_transform(Function&& function, Tuple&& tuple, Tuples&&... tuples)
{
    static_assert(((std::tuple_size_v<std::remove_cvref_t<Tuple>> == std::tuple_size_v<std::remove_cvref_t<Tuples>>) && ...));
    return [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        auto callFunction = [&function, &tuple, &tuples...]<std::size_t I>() {
            return function(std::get<I>(tuple), std::get<I>(tuples)...);
        };
        return std::make_tuple(callFunction.template operator()<Idx>()...);
    }(std::make_index_sequence<std::tuple_size_v<Tuple>>());
}


static_assert(std::is_same_v<std::vector<int>, type_transform<std::vector, int>>);
static_assert(std::is_same_v<std::tuple<std::vector<int>, std::vector<float>>, type_transform<std::vector, std::tuple<int, float>>>);
static_assert(std::is_same_v<void, type_transform<std::vector, void>>);
static_assert(std::is_same_v<dummy_t, type_transform<std::vector, dummy_t>>);

} // namespace fair::meta

#endif // include guard
#ifndef GNURADIO_BUFFER2_H
#define GNURADIO_BUFFER2_H

#include <bit>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <span>

namespace gr {
namespace util {
template <typename T, typename...>
struct first_template_arg_helper;

template <template <typename...> class TemplateType,
          typename ValueType,
          typename... OtherTypes>
struct first_template_arg_helper<TemplateType<ValueType, OtherTypes...>> {
    using value_type = ValueType;
};

template <typename T>
constexpr auto* value_type_helper()
{
    if constexpr (requires { typename T::value_type; }) {
        return static_cast<typename T::value_type*>(nullptr);
    }
    else {
        return static_cast<typename first_template_arg_helper<T>::value_type*>(nullptr);
    }
}

template <typename T>
using value_type_t = std::remove_pointer_t<decltype(value_type_helper<T>())>;

template <typename... A>
struct test_fallback {
};

template <typename, typename ValueType>
struct test_value_type {
    using value_type = ValueType;
};

static_assert(std::is_same_v<int, value_type_t<test_fallback<int, float, double>>>);
static_assert(std::is_same_v<float, value_type_t<test_value_type<int, float>>>);
static_assert(std::is_same_v<int, value_type_t<std::span<int>>>);
static_assert(std::is_same_v<double, value_type_t<std::array<double, 42>>>);

} // namespace util

// clang-format off
// disable formatting until clang-format (v16) supporting concepts
template<class T>
concept BufferReader = requires(T /*const*/ t, const std::size_t n_items) {
    { t.get(n_items) }     -> std::same_as<std::span<const util::value_type_t<T>>>;
    { t.consume(n_items) } -> std::same_as<bool>;
    { t.position() }       -> std::same_as<std::int64_t>;
    { t.available() }      -> std::same_as<std::size_t>;
    { t.buffer() };
};

template<class Fn, typename T, typename ...Args>
concept WriterCallback = std::is_invocable_v<Fn, std::span<T>&, std::int64_t, Args...> || std::is_invocable_v<Fn, std::span<T>&, Args...>;

template<class T, typename ...Args>
concept BufferWriter = requires(T t, const std::size_t n_items, std::pair<std::size_t, std::int64_t> token, Args ...args) {
    { t.publish([](std::span<util::value_type_t<T>> &/*writable_data*/, Args ...) { /* */ }, n_items, args...) }                                 -> std::same_as<void>;
    { t.publish([](std::span<util::value_type_t<T>> &/*writable_data*/, std::int64_t /* writePos */, Args ...) { /* */  }, n_items, args...) }   -> std::same_as<void>;
    { t.try_publish([](std::span<util::value_type_t<T>> &/*writable_data*/, Args ...) { /* */ }, n_items, args...) }                             -> std::same_as<bool>;
    { t.try_publish([](std::span<util::value_type_t<T>> &/*writable_data*/, std::int64_t /* writePos */, Args ...) { /* */  }, n_items, args...) }-> std::same_as<bool>;
    { t.get(n_items) } -> std::same_as<std::pair<std::span<util::value_type_t<T>>, std::pair<std::size_t, std::int64_t>>>;
    { t.publish(token, n_items) } -> std::same_as<void>;
    { t.available() }         -> std::same_as<std::size_t>;
    { t.buffer() };
};

template<class T, typename ...Args>
concept Buffer = requires(T t, const std::size_t min_size, Args ...args) {
    { T(min_size, args...) };
    { t.size() }       -> std::same_as<std::size_t>;
    { t.new_reader() } -> BufferReader;
    { t.new_writer() } -> BufferWriter;
};

// compile-time unit-tests
namespace test {
template <typename T>
struct non_compliant_class {
};
template <typename T, typename... Args>
using WithSequenceParameter = decltype([](std::span<T>&, std::int64_t, Args...) { /* */ });
template <typename T, typename... Args>
using NoSequenceParameter = decltype([](std::span<T>&, Args...) { /* */ });
} // namespace test

static_assert(!Buffer<test::non_compliant_class<int>>);
static_assert(!BufferReader<test::non_compliant_class<int>>);
static_assert(!BufferWriter<test::non_compliant_class<int>>);

static_assert(WriterCallback<test::WithSequenceParameter<int>, int>);
static_assert(!WriterCallback<test::WithSequenceParameter<int>, int, std::span<bool>>);
static_assert(WriterCallback<test::WithSequenceParameter<int, std::span<bool>>, int, std::span<bool>>);
static_assert(WriterCallback<test::NoSequenceParameter<int>, int>);
static_assert(!WriterCallback<test::NoSequenceParameter<int>, int, std::span<bool>>);
static_assert(WriterCallback<test::NoSequenceParameter<int, std::span<bool>>, int, std::span<bool>>);
// clang-format on
} // namespace gr

#endif // GNURADIO_BUFFER2_H
#ifndef GNURADIO_SEQUENCE_HPP
#define GNURADIO_SEQUENCE_HPP

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <vector>

namespace gr {

#ifndef forceinline
// use this for hot-spots only <-> may bloat code size, not fit into cache and
// consequently slow down execution
#define forceinline inline __attribute__((always_inline))
#endif

static constexpr const std::size_t kCacheLine
        = 64; // waiting for clang: std::hardware_destructive_interference_size
static constexpr const std::int64_t kInitialCursorValue = -1L;

/**
 * Concurrent sequence class used for tracking the progress of the ring buffer and event
 * processors. Support a number of concurrent operations including CAS and order writes.
 * Also attempts to be more efficient with regards to false sharing by adding padding
 * around the volatile field.
 */
// clang-format off
class Sequence
{
    alignas(kCacheLine) std::atomic<std::int64_t> _fieldsValue{};

public:
    Sequence(const Sequence&) = delete;
    Sequence(const Sequence&&) = delete;
    void operator=(const Sequence&) = delete;
    explicit Sequence(std::int64_t initialValue = kInitialCursorValue) noexcept
        : _fieldsValue(initialValue)
    {
    }

    [[nodiscard]] forceinline std::int64_t value() const noexcept
    {
        return std::atomic_load_explicit(&_fieldsValue, std::memory_order_acquire);
    }

    forceinline void setValue(const std::int64_t value) noexcept
    {
        std::atomic_store_explicit(&_fieldsValue, value, std::memory_order_release);
    }

    [[nodiscard]] forceinline bool compareAndSet(std::int64_t expectedSequence,
                                                 std::int64_t nextSequence) noexcept
    {
        // atomically set the value to the given updated value if the current value == the
        // expected value (true, otherwise folse).
        return std::atomic_compare_exchange_strong(
            &_fieldsValue, &expectedSequence, nextSequence);
    }

    [[nodiscard]] forceinline std::int64_t incrementAndGet() noexcept
    {
        return std::atomic_fetch_add(&_fieldsValue, 1L) + 1L;
    }

    [[nodiscard]] forceinline std::int64_t addAndGet(std::int64_t value) noexcept
    {
        return std::atomic_fetch_add(&_fieldsValue, value) + value;
    }
};

namespace detail {
/**
 * Get the minimum sequence from an array of Sequences.
 *
 * \param sequences sequences to compare.
 * \param minimum an initial default minimum.  If the array is empty this value will
 * returned. \returns the minimum sequence found or lon.MaxValue if the array is empty.
 */
inline std::int64_t getMinimumSequence(
    const std::vector<std::shared_ptr<Sequence>>& sequences,
    std::int64_t minimum = std::numeric_limits<std::int64_t>::max()) noexcept
{
    if (sequences.empty()) {
        return minimum;
    }
#if not defined(_LIBCPP_VERSION)
    return std::min(minimum, std::ranges::min(sequences, std::less{}, [](const auto &sequence) noexcept { return sequence->value(); })->value());
#else
    std::vector<int64_t> v(sequences.size());
    std::transform(sequences.cbegin(), sequences.cend(), v.begin(), [](auto val) { return val->value(); });
    auto min = std::min(v.begin(), v.end());
    return std::min(*min, minimum);
#endif
}

inline void addSequences(std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>& sequences,
             const Sequence& cursor,
             const std::vector<std::shared_ptr<Sequence>>& sequencesToAdd)
{
    std::int64_t cursorSequence;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> updatedSequences;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> currentSequences;

    do {
        currentSequences = std::atomic_load_explicit(&sequences, std::memory_order_acquire);
        updatedSequences = std::make_shared<std::vector<std::shared_ptr<Sequence>>>(currentSequences->size() + sequencesToAdd.size());

#if not defined(_LIBCPP_VERSION)
        std::ranges::copy(currentSequences->begin(), currentSequences->end(), updatedSequences->begin());
#else
        std::copy(currentSequences->begin(), currentSequences->end(), updatedSequences->begin());
#endif

        cursorSequence = cursor.value();

        auto index = currentSequences->size();
        for (auto&& sequence : sequencesToAdd) {
            sequence->setValue(cursorSequence);
            (*updatedSequences)[index] = sequence;
            index++;
        }
    } while (!std::atomic_compare_exchange_weak(&sequences, &currentSequences, updatedSequences)); // xTODO: explicit memory order

    cursorSequence = cursor.value();

    for (auto&& sequence : sequencesToAdd) {
        sequence->setValue(cursorSequence);
    }
}

inline bool removeSequence(std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>& sequences, const std::shared_ptr<Sequence>& sequence)
{
    std::uint32_t numToRemove;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> oldSequences;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> newSequences;

    do {
        oldSequences = std::atomic_load_explicit(&sequences, std::memory_order_acquire);
#if not defined(_LIBCPP_VERSION)
        numToRemove = static_cast<std::uint32_t>(std::ranges::count(*oldSequences, sequence)); // specifically uses identity
#else
        numToRemove = static_cast<std::uint32_t>(std::count((*oldSequences).begin(), (*oldSequences).end(), sequence)); // specifically uses identity
#endif
        if (numToRemove == 0) {
            break;
        }

        auto oldSize = static_cast<std::uint32_t>(oldSequences->size());
        newSequences = std::make_shared<std::vector<std::shared_ptr<Sequence>>>(
            oldSize - numToRemove);

        for (auto i = 0U, pos = 0U; i < oldSize; ++i) {
            const auto& testSequence = (*oldSequences)[i];
            if (sequence != testSequence) {
                (*newSequences)[pos] = testSequence;
                pos++;
            }
        }
    } while (!std::atomic_compare_exchange_weak(&sequences, &oldSequences, newSequences));

    return numToRemove != 0;
}

// clang-format on

} // namespace detail

} // namespace gr

#ifdef FMT_FORMAT_H_
#include <fmt/core.h>
#include <fmt/ostream.h>

template<>
struct fmt::formatter<gr::Sequence> {
    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto
    format(gr::Sequence const &value, FormatContext &ctx) {
        return fmt::format_to(ctx.out(), "{}", value.value());
    }
};

namespace gr {
inline std::ostream &
operator<<(std::ostream &os, const Sequence &v) {
    return os << fmt::format("{}", v);
}
} // namespace gr

#endif // FMT_FORMAT_H_

#endif // GNURADIO_SEQUENCE_HPP
#ifndef GNURADIO_WAIT_STRATEGY_HPP
#define GNURADIO_WAIT_STRATEGY_HPP

#include <condition_variable>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>


namespace gr {
// clang-format off
/**
 * Wait for the given sequence to be available.  It is possible for this method to return a value less than the sequence number supplied depending on the implementation of the WaitStrategy.
 * A common use for this is to signal a timeout.Any EventProcessor that is using a WaitStrategy to get notifications about message becoming available should remember to handle this case.
 * The BatchEventProcessor<T> explicitly handles this case and will signal a timeout if required.
 *
 * \param sequence sequence to be waited on.
 * \param cursor Ring buffer cursor on which to wait.
 * \param dependentSequence on which to wait.
 * \param barrier barrier the IEventProcessor is waiting on.
 * \returns the sequence that is available which may be greater than the requested sequence.
 */
template<typename T>
inline constexpr bool isWaitStrategy = requires(T /*const*/ t, const std::int64_t sequence, const Sequence &cursor, std::vector<std::shared_ptr<Sequence>> &dependentSequences) {
    { t.waitFor(sequence, cursor, dependentSequences) } -> std::same_as<std::int64_t>;
};
static_assert(!isWaitStrategy<int>);

/**
 * signal those waiting that the cursor has advanced.
 */
template<typename T>
inline constexpr bool hasSignalAllWhenBlocking = requires(T /*const*/ t) {
    { t.signalAllWhenBlocking() } -> std::same_as<void>;
};
static_assert(!hasSignalAllWhenBlocking<int>);

template<typename T>
concept WaitStrategy = isWaitStrategy<T>;



/**
 * Blocking strategy that uses a lock and condition variable for IEventProcessor's waiting on a barrier.
 * This strategy should be used when performance and low-latency are not as important as CPU resource.
 */
class BlockingWaitStrategy {
    std::recursive_mutex        _gate;
    std::condition_variable_any _conditionVariable;

public:
    std::int64_t waitFor(const std::int64_t sequence, const Sequence &cursor, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) {
        if (cursor.value() < sequence) {
            std::unique_lock uniqueLock(_gate);

            while (cursor.value() < sequence) {
                // optional: barrier check alert
                _conditionVariable.wait(uniqueLock);
            }
        }

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            // optional: barrier check alert
        }

        return availableSequence;
    }

    void signalAllWhenBlocking() {
        std::unique_lock uniqueLock(_gate);
        _conditionVariable.notify_all();
    }
};
static_assert(WaitStrategy<BlockingWaitStrategy>);

/**
 * Busy Spin strategy that uses a busy spin loop for IEventProcessor's waiting on a barrier.
 * This strategy will use CPU resource to avoid syscalls which can introduce latency jitter.
 * It is best used when threads can be bound to specific CPU cores.
 */
struct BusySpinWaitStrategy {
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) const {
        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            // optional: barrier check alert
        }
        return availableSequence;
    }
};
static_assert(WaitStrategy<BusySpinWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<BusySpinWaitStrategy>);

/**
 * Sleeping strategy that initially spins, then uses a std::this_thread::yield(), and eventually sleep. T
 * his strategy is a good compromise between performance and CPU resource.
 * Latency spikes can occur after quiet periods.
 */
class SleepingWaitStrategy {
    static const std::int32_t _defaultRetries = 200;
    std::int32_t              _retries        = 0;

public:
    explicit SleepingWaitStrategy(std::int32_t retries = _defaultRetries)
        : _retries(retries) {
    }

    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) const {
        auto       counter    = _retries;
        const auto waitMethod = [&counter]() {
            // optional: barrier check alert

            if (counter > 100) {
                --counter;
            } else if (counter > 0) {
                --counter;
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(0));
            }
        };

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            waitMethod();
        }

        return availableSequence;
    }
};
static_assert(WaitStrategy<SleepingWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<SleepingWaitStrategy>);

struct TimeoutException : public std::runtime_error {
    TimeoutException() : std::runtime_error("TimeoutException") {}
};

class TimeoutBlockingWaitStrategy {
    using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    Clock::duration             _timeout;
    std::recursive_mutex        _gate;
    std::condition_variable_any _conditionVariable;

public:
    explicit TimeoutBlockingWaitStrategy(Clock::duration timeout)
        : _timeout(timeout) {}

    std::int64_t waitFor(const std::int64_t sequence, const Sequence &cursor, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) {
        auto timeSpan = std::chrono::duration_cast<std::chrono::microseconds>(_timeout);

        if (cursor.value() < sequence) {
            std::unique_lock uniqueLock(_gate);

            while (cursor.value() < sequence) {
                // optional: barrier check alert

                if (_conditionVariable.wait_for(uniqueLock, timeSpan) == std::cv_status::timeout) {
                    throw TimeoutException();
                }
            }
        }

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            // optional: barrier check alert
        }

        return availableSequence;
    }

    void signalAllWhenBlocking() {
        std::unique_lock uniqueLock(_gate);
        _conditionVariable.notify_all();
    }
};
static_assert(WaitStrategy<TimeoutBlockingWaitStrategy>);
static_assert(hasSignalAllWhenBlocking<TimeoutBlockingWaitStrategy>);

/**
 * Yielding strategy that uses a Thread.Yield() for IEventProcessors waiting on a barrier after an initially spinning.
 * This strategy is a good compromise between performance and CPU resource without incurring significant latency spikes.
 */
class YieldingWaitStrategy {
    const std::size_t _spinTries = 100;

public:
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) const {
        auto       counter    = _spinTries;
        const auto waitMethod = [&counter]() {
            // optional: barrier check alert

            if (counter == 0) {
                std::this_thread::yield();
            } else {
                --counter;
            }
        };

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            waitMethod();
        }

        return availableSequence;
    }
};
static_assert(WaitStrategy<YieldingWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<YieldingWaitStrategy>);

struct NoWaitStrategy {
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> & /*dependentSequences*/) const {
        // wait for nothing
        return sequence;
    }
};
static_assert(WaitStrategy<NoWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<NoWaitStrategy>);


/**
 *
 * SpinWait is meant to be used as a tool for waiting in situations where the thread is not allowed to block.
 *
 * In order to get the maximum performance, the implementation first spins for `YIELD_THRESHOLD` times, and then
 * alternates between yielding, spinning and putting the thread to sleep, to allow other threads to be scheduled
 * by the kernel to avoid potential CPU contention.
 *
 * The number of spins, yielding, and sleeping for either '0 ms' or '1 ms' is controlled by the NTTP constants
 * @tparam YIELD_THRESHOLD
 * @tparam SLEEP_0_EVERY_HOW_MANY_TIMES
 * @tparam SLEEP_1_EVERY_HOW_MANY_TIMES
 */
template<std::int32_t YIELD_THRESHOLD = 10, std::int32_t SLEEP_0_EVERY_HOW_MANY_TIMES = 5, std::int32_t SLEEP_1_EVERY_HOW_MANY_TIMES = 20>
class SpinWait {
    using Clock         = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    std::int32_t _count = 0;
    static void  spinWaitInternal(std::int32_t iterationCount) noexcept {
        for (auto i = 0; i < iterationCount; i++) {
            yieldProcessor();
        }
    }
#ifndef __EMSCRIPTEN__
    static void yieldProcessor() noexcept { asm volatile("rep\nnop"); }
#else
    static void yieldProcessor() noexcept { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
#endif

public:
    SpinWait() = default;

    [[nodiscard]] std::int32_t count() const noexcept { return _count; }
    [[nodiscard]] bool         nextSpinWillYield() const noexcept { return _count > YIELD_THRESHOLD; }

    void                       spinOnce() {
        if (nextSpinWillYield()) {
            auto num = _count >= YIELD_THRESHOLD ? _count - 10 : _count;
            if (num % SLEEP_1_EVERY_HOW_MANY_TIMES == SLEEP_1_EVERY_HOW_MANY_TIMES - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                if (num % SLEEP_0_EVERY_HOW_MANY_TIMES == SLEEP_0_EVERY_HOW_MANY_TIMES - 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(0));
                } else {
                    std::this_thread::yield();
                }
            }
        } else {
            spinWaitInternal(4 << _count);
        }

        if (_count == std::numeric_limits<std::int32_t>::max()) {
            _count = YIELD_THRESHOLD;
        } else {
            ++_count;
        }
    }

    void reset() noexcept { _count = 0; }

    template<typename T>
    requires std::is_nothrow_invocable_r_v<bool, T>
    bool
    spinUntil(const T &condition) const { return spinUntil(condition, -1); }

    template<typename T>
    requires std::is_nothrow_invocable_r_v<bool, T>
    bool
    spinUntil(const T &condition, std::int64_t millisecondsTimeout) const {
        if (millisecondsTimeout < -1) {
            throw std::out_of_range("Timeout value is out of range");
        }

        std::int64_t num = 0;
        if (millisecondsTimeout != 0 && millisecondsTimeout != -1) {
            num = getTickCount();
        }

        SpinWait spinWait;
        while (!condition()) {
            if (millisecondsTimeout == 0) {
                return false;
            }

            spinWait.spinOnce();

            if (millisecondsTimeout != 1 && spinWait.nextSpinWillYield() && millisecondsTimeout <= (getTickCount() - num)) {
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] static std::int64_t getTickCount() { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); }
};

/**
 * Spin strategy that uses a SpinWait for IEventProcessors waiting on a barrier.
 * This strategy is a good compromise between performance and CPU resource.
 * Latency spikes can occur after quiet periods.
 */
struct SpinWaitWaitStrategy {
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequence) const {
        std::int64_t availableSequence;

        SpinWait     spinWait;
        while ((availableSequence = detail::getMinimumSequence(dependentSequence)) < sequence) {
            // optional: barrier check alert
            spinWait.spinOnce();
        }

        return availableSequence;
    }
};
static_assert(WaitStrategy<SpinWaitWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<SpinWaitWaitStrategy>);

struct NO_SPIN_WAIT {};

template<typename SPIN_WAIT = NO_SPIN_WAIT>
class AtomicMutex {
    std::atomic_flag _lock = ATOMIC_FLAG_INIT;
    SPIN_WAIT        _spin_wait;

public:
    AtomicMutex()                    = default;
    AtomicMutex(const AtomicMutex &) = delete;
    AtomicMutex &operator=(const AtomicMutex &) = delete;

    //
    void lock() {
        while (_lock.test_and_set(std::memory_order_acquire)) {
            if constexpr (requires { _spin_wait.spin_once(); }) {
                _spin_wait.spin_once();
            }
        }
        if constexpr (requires { _spin_wait.spin_once(); }) {
            _spin_wait.reset();
        }
    }
    void unlock() { _lock.clear(std::memory_order::release); }
};


// clang-format on
} // namespace gr


#endif // GNURADIO_WAIT_STRATEGY_HPP
#ifndef GNURADIO_CLAIM_STRATEGY_HPP
#define GNURADIO_CLAIM_STRATEGY_HPP

#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>


namespace gr {

struct NoCapacityException : public std::runtime_error {
    NoCapacityException() : std::runtime_error("NoCapacityException"){};
};

// clang-format off

template<typename T>
concept ClaimStrategy = requires(T /*const*/ t, const std::vector<std::shared_ptr<Sequence>> &dependents, const int requiredCapacity,
        const std::int64_t cursorValue, const std::int64_t sequence, const std::int64_t availableSequence, const std::int32_t n_slots_to_claim) {
    { t.hasAvailableCapacity(dependents, requiredCapacity, cursorValue) } -> std::same_as<bool>;
    { t.next(dependents, n_slots_to_claim) } -> std::same_as<std::int64_t>;
    { t.tryNext(dependents, n_slots_to_claim) } -> std::same_as<std::int64_t>;
    { t.getRemainingCapacity(dependents) } -> std::same_as<std::int64_t>;
    { t.publish(sequence) } -> std::same_as<void>;
    { t.isAvailable(sequence) } -> std::same_as<bool>;
    { t.getHighestPublishedSequence(sequence, availableSequence) } -> std::same_as<std::int64_t>;
};

namespace claim_strategy::util {
constexpr unsigned    floorlog2(std::size_t x) { return x == 1 ? 0 : 1 + floorlog2(x >> 1); }
constexpr unsigned    ceillog2(std::size_t x) { return x == 1 ? 0 : floorlog2(x - 1) + 1; }
}

template<std::size_t SIZE = std::dynamic_extent, WaitStrategy WAIT_STRATEGY = BusySpinWaitStrategy>
class alignas(kCacheLine) SingleThreadedStrategy {
    alignas(kCacheLine) const std::size_t _size;
    alignas(kCacheLine) Sequence &_cursor;
    alignas(kCacheLine) WAIT_STRATEGY &_waitStrategy;
    alignas(kCacheLine) std::int64_t _nextValue{ kInitialCursorValue }; // N.B. no need for atomics since this is called by a single publisher
    alignas(kCacheLine) mutable std::int64_t _cachedValue{ kInitialCursorValue };

public:
    SingleThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy, const std::size_t buffer_size = SIZE)
        : _size(buffer_size), _cursor(cursor), _waitStrategy(waitStrategy){};
    SingleThreadedStrategy(const SingleThreadedStrategy &)  = delete;
    SingleThreadedStrategy(const SingleThreadedStrategy &&) = delete;
    void operator=(const SingleThreadedStrategy &) = delete;

    bool hasAvailableCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents, const int requiredCapacity, const std::int64_t /*cursorValue*/) const noexcept {
        if (const std::int64_t wrapPoint = (_nextValue + requiredCapacity) - static_cast<std::int64_t>(_size); wrapPoint > _cachedValue || _cachedValue > _nextValue) {
            auto minSequence = detail::getMinimumSequence(dependents, _nextValue);
            _cachedValue     = minSequence;
            if (wrapPoint > minSequence) {
                return false;
            }
        }
        return true;
    }

    std::int64_t next(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::int32_t n_slots_to_claim = 1) noexcept {
        assert((n_slots_to_claim > 0 && n_slots_to_claim <= static_cast<std::int32_t>(_size)) && "n_slots_to_claim must be > 0 and <= bufferSize");

        auto nextSequence = _nextValue + n_slots_to_claim;
        auto wrapPoint    = nextSequence - static_cast<std::int64_t>(_size);

        if (const auto cachedGatingSequence = _cachedValue; wrapPoint > cachedGatingSequence || cachedGatingSequence > _nextValue) {
            _cursor.setValue(_nextValue);

            SpinWait     spinWait;
            std::int64_t minSequence;
            while (wrapPoint > (minSequence = detail::getMinimumSequence(dependents, _nextValue))) {
                if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
                    _waitStrategy.signalAllWhenBlocking();
                }
                spinWait.spinOnce();
            }
            _cachedValue = minSequence;
        }
        _nextValue = nextSequence;

        return nextSequence;
    }

    std::int64_t tryNext(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::size_t n_slots_to_claim) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        if (!hasAvailableCapacity(dependents, n_slots_to_claim, 0 /* unused cursor value */)) {
            throw NoCapacityException();
        }

        const auto nextSequence = _nextValue + n_slots_to_claim;
        _nextValue              = nextSequence;

        return nextSequence;
    }

    std::int64_t getRemainingCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents) const noexcept {
        const auto consumed = detail::getMinimumSequence(dependents, _nextValue);
        const auto produced = _nextValue;

        return static_cast<std::int64_t>(_size) - (produced - consumed);
    }

    void publish(std::int64_t sequence) {
        _cursor.setValue(sequence);
        if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

    [[nodiscard]] forceinline bool isAvailable(std::int64_t sequence) const noexcept { return sequence <= _cursor.value(); }
    [[nodiscard]] std::int64_t     getHighestPublishedSequence(std::int64_t /*nextSequence*/, std::int64_t availableSequence) const noexcept { return availableSequence; }
};
static_assert(ClaimStrategy<SingleThreadedStrategy<1024, NoWaitStrategy>>);

/**
 * Claim strategy for claiming sequences for access to a data structure while tracking dependent Sequences.
 * Suitable for use for sequencing across multiple publisher threads.
 * Note on cursor:  With this sequencer the cursor value is updated after the call to SequencerBase::next(),
 * to determine the highest available sequence that can be read, then getHighestPublishedSequence should be used.
 */
template<std::size_t SIZE = std::dynamic_extent, WaitStrategy WAIT_STRATEGY = BusySpinWaitStrategy>
class MultiThreadedStrategy {
    alignas(kCacheLine) const std::size_t _size;
    alignas(kCacheLine) Sequence &_cursor;
    alignas(kCacheLine) WAIT_STRATEGY &_waitStrategy;
    alignas(kCacheLine) std::vector<std::int32_t> _availableBuffer; // tracks the state of each ringbuffer slot
    alignas(kCacheLine) std::shared_ptr<Sequence> _gatingSequenceCache = std::make_shared<Sequence>();
    const std::int32_t _indexMask;
    const std::int32_t _indexShift;

public:
    MultiThreadedStrategy() = delete;
    explicit MultiThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy, const std::size_t buffer_size = SIZE)
        : _size(buffer_size), _cursor(cursor), _waitStrategy(waitStrategy), _availableBuffer(_size),
        _indexMask(_size - 1), _indexShift(claim_strategy::util::ceillog2(_size)) {
        for (std::size_t i = _size - 1; i != 0; i--) {
            setAvailableBufferValue(i, -1);
        }
        setAvailableBufferValue(0, -1);
    }
    MultiThreadedStrategy(const MultiThreadedStrategy &)  = delete;
    MultiThreadedStrategy(const MultiThreadedStrategy &&) = delete;
    void               operator=(const MultiThreadedStrategy &) = delete;

    [[nodiscard]] bool hasAvailableCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::int64_t requiredCapacity, const std::int64_t cursorValue) const noexcept {
        const auto wrapPoint = (cursorValue + requiredCapacity) - static_cast<std::int64_t>(_size);

        if (const auto cachedGatingSequence = _gatingSequenceCache->value(); wrapPoint > cachedGatingSequence || cachedGatingSequence > cursorValue) {
            const auto minSequence = detail::getMinimumSequence(dependents, cursorValue);
            _gatingSequenceCache->setValue(minSequence);

            if (wrapPoint > minSequence) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::int64_t next(const std::vector<std::shared_ptr<Sequence>> &dependents, std::size_t n_slots_to_claim = 1) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        std::int64_t current;
        std::int64_t next;

        SpinWait     spinWait;
        do {
            current                           = _cursor.value();
            next                              = current + n_slots_to_claim;

            std::int64_t wrapPoint            = next - static_cast<std::int64_t>(_size);
            std::int64_t cachedGatingSequence = _gatingSequenceCache->value();

            if (wrapPoint > cachedGatingSequence || cachedGatingSequence > current) {
                std::int64_t gatingSequence = detail::getMinimumSequence(dependents, current);

                if (wrapPoint > gatingSequence) {
                    if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
                        _waitStrategy.signalAllWhenBlocking();
                    }
                    spinWait.spinOnce();
                    continue;
                }

                _gatingSequenceCache->setValue(gatingSequence);
            } else if (_cursor.compareAndSet(current, next)) {
                break;
            }
        } while (true);

        return next;
    }

    [[nodiscard]] std::int64_t tryNext(const std::vector<std::shared_ptr<Sequence>> &dependents, std::size_t n_slots_to_claim = 1) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        std::int64_t current;
        std::int64_t next;

        do {
            current = _cursor.value();
            next    = current + n_slots_to_claim;

            if (!hasAvailableCapacity(dependents, n_slots_to_claim, current)) {
                throw NoCapacityException();
            }
        } while (!_cursor.compareAndSet(current, next));

        return next;
    }

    [[nodiscard]] std::int64_t getRemainingCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents) const noexcept {
        const auto produced = _cursor.value();
        const auto consumed = detail::getMinimumSequence(dependents, produced);

        return static_cast<std::int64_t>(_size) - (produced - consumed);
    }

    void publish(std::int64_t sequence) {
        setAvailable(sequence);
        if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

    [[nodiscard]] forceinline bool isAvailable(std::int64_t sequence) const noexcept {
        const auto index = calculateIndex(sequence);
        const auto flag  = calculateAvailabilityFlag(sequence);

        return _availableBuffer[static_cast<std::size_t>(index)] == flag;
    }

    [[nodiscard]] forceinline std::int64_t getHighestPublishedSequence(const std::int64_t lowerBound, const std::int64_t availableSequence) const noexcept {
        for (std::int64_t sequence = lowerBound; sequence <= availableSequence; sequence++) {
            if (!isAvailable(sequence)) {
                return sequence - 1;
            }
        }

        return availableSequence;
    }

private:
    void                      setAvailable(std::int64_t sequence) noexcept { setAvailableBufferValue(calculateIndex(sequence), calculateAvailabilityFlag(sequence)); }
    forceinline void          setAvailableBufferValue(std::size_t index, std::int32_t flag) noexcept { _availableBuffer[index] = flag; }
    [[nodiscard]] forceinline std::int32_t calculateAvailabilityFlag(const std::int64_t sequence) const noexcept { return static_cast<std::int32_t>(static_cast<std::uint64_t>(sequence) >> _indexShift); }
    [[nodiscard]] forceinline std::size_t calculateIndex(const std::int64_t sequence) const noexcept { return static_cast<std::size_t>(static_cast<std::int32_t>(sequence) & _indexMask); }
};
static_assert(ClaimStrategy<MultiThreadedStrategy<1024, NoWaitStrategy>>);
// clang-format on

enum class ProducerType {
    /**
     * creates a buffer assuming a single producer-thread and multiple consumer
     */
    Single,

    /**
     * creates a buffer assuming multiple producer-threads and multiple consumer
     */
    Multi
};

namespace detail {
template <std::size_t size, ProducerType producerType, WaitStrategy WAIT_STRATEGY>
struct producer_type;

template <std::size_t size, WaitStrategy WAIT_STRATEGY>
struct producer_type<size, ProducerType::Single, WAIT_STRATEGY> {
    using value_type = SingleThreadedStrategy<size, WAIT_STRATEGY>;
};
template <std::size_t size, WaitStrategy WAIT_STRATEGY>
struct producer_type<size, ProducerType::Multi, WAIT_STRATEGY> {
    using value_type = MultiThreadedStrategy<size, WAIT_STRATEGY>;
};

template <std::size_t size, ProducerType producerType, WaitStrategy WAIT_STRATEGY>
using producer_type_v = typename producer_type<size, producerType, WAIT_STRATEGY>::value_type;

} // namespace detail

} // namespace gr


#endif // GNURADIO_CLAIM_STRATEGY_HPP
#ifndef GNURADIO_CIRCULAR_BUFFER_HPP
#define GNURADIO_CIRCULAR_BUFFER_HPP

#if defined(_LIBCPP_VERSION) and _LIBCPP_VERSION < 16000
#include <experimental/memory_resource>

namespace std::pmr {
using memory_resource = std::experimental::pmr::memory_resource;
template<typename T>
using polymorphic_allocator = std::experimental::pmr::polymorphic_allocator<T>;
} // namespace std::pmr
#else
#include <memory_resource>
#endif
#include <algorithm>
#include <bit>
#include <cassert> // to assert if compiled for debugging
#include <functional>
#include <numeric>
#include <ranges>
#include <span>

#include <fmt/format.h>

// header for creating/opening or POSIX shared memory objects
#include <cerrno>
#include <fcntl.h>
#if defined __has_include && not __EMSCRIPTEN__
#if __has_include(<sys/mman.h>) && __has_include(<sys/stat.h>) && __has_include(<unistd.h>)
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gr {
static constexpr bool has_posix_mmap_interface = true;
}

#define HAS_POSIX_MAP_INTERFACE
#else
namespace gr {
static constexpr bool has_posix_mmap_interface = false;
}
#endif
#else // #if defined __has_include -- required for portability
namespace gr {
static constexpr bool has_posix_mmap_interface = false;
}
#endif


namespace gr {

namespace util {
constexpr std::size_t
round_up(std::size_t num_to_round, std::size_t multiple) noexcept {
    if (multiple == 0) {
        return num_to_round;
    }
    const auto remainder = num_to_round % multiple;
    if (remainder == 0) {
        return num_to_round;
    }
    return num_to_round + multiple - remainder;
}
} // namespace util

// clang-format off
class double_mapped_memory_resource : public std::pmr::memory_resource {
#ifdef HAS_POSIX_MAP_INTERFACE
    [[nodiscard]] void* do_allocate(const std::size_t required_size, std::size_t alignment) override {

        const std::size_t size = 2 * required_size;
        if (size % 2LU != 0LU || size % static_cast<std::size_t>(getpagesize()) != 0LU) {
            throw std::runtime_error(fmt::format("incompatible buffer-byte-size: {} -> {} alignment: {} vs. page size: {}", required_size, size, alignment, getpagesize()));
        }
        const std::size_t size_half = size/2;

        static std::size_t _counter;
        const auto buffer_name = fmt::format("/double_mapped_memory_resource-{}-{}-{}", getpid(), size, _counter++);
        const auto memfd_create = [name = buffer_name.c_str()](unsigned int flags) -> long {
            return syscall(__NR_memfd_create, name, flags);
        };
        int shm_fd = static_cast<int>(memfd_create(0));
        if (shm_fd < 0) {
            throw std::runtime_error(fmt::format("{} - memfd_create error {}: {}",  buffer_name, errno, strerror(errno)));
        }

        if (ftruncate(shm_fd, static_cast<off_t>(size)) == -1) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - ftruncate {}: {}",  buffer_name, errno, strerror(errno)));
        }

        void* first_copy = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t>(0));
        if (first_copy == MAP_FAILED) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed munmap for first half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // unmap the 2nd half
        if (munmap(static_cast<char*>(first_copy) + size_half, size_half) == -1) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed munmap for second half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // map the first half into the now available hole where the
        if (const void* second_copy = mmap(static_cast<char*> (first_copy) + size_half, size_half, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t> (0)); second_copy == MAP_FAILED) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed mmap for second copy {}: {}",  buffer_name, errno, strerror(errno)));
        }

        close(shm_fd); // file-descriptor is no longer needed. The mapping is retained.
        return first_copy;
}
#else
    [[nodiscard]] void* do_allocate(const std::size_t, std::size_t) override {
        throw std::runtime_error("OS does not provide POSIX interface for mmap(...) and munmao(...)");
        // static_assert(false, "OS does not provide POSIX interface for mmap(...) and munmao(...)");
    }
#endif


#ifdef HAS_POSIX_MAP_INTERFACE
    void  do_deallocate(void* p, std::size_t size, size_t alignment) override {

        if (munmap(p, size) == -1) {
            throw std::runtime_error(fmt::format("double_mapped_memory_resource::do_deallocate(void*, {}, {}) - munmap(..) failed", size, alignment));
        }
    }
#else
    void  do_deallocate(void*, std::size_t, size_t) override { }
#endif

    bool  do_is_equal(const memory_resource& other) const noexcept override { return this == &other; }

public:
    static inline double_mapped_memory_resource* defaultAllocator() {
        static auto instance = double_mapped_memory_resource();
        return &instance;
    }

    template<typename T>
    requires (std::has_single_bit(sizeof(T)))
    static inline std::pmr::polymorphic_allocator<T> allocator()
    {
        return std::pmr::polymorphic_allocator<T>(gr::double_mapped_memory_resource::defaultAllocator());
    }
};



/**
 * @brief circular buffer implementation using double-mapped memory allocations
 * where the first SIZE-ed buffer is mirrored directly its end to mimic wrap-around
 * free bulk memory access. The buffer keeps a list of indices (using 'Sequence')
 * to keep track of which parts can be tread-safely read and/or written
 *
 *                          wrap-around point
 *                                 |
 *                                 v
 *  | buffer segment #1 (original) | buffer segment #2 (copy of #1) |
 *  0                            SIZE                            2*SIZE
 *                      writeIndex
 *                          v
 *  wrap-free write access  |<-  N_1 < SIZE   ->|
 *
 *  readIndex < writeIndex-N_2
 *      v
 *      |<- N_2 < SIZE ->|
 *
 * N.B N_AVAILABLE := (SIZE + writeIndex - readIndex ) % SIZE
 *
 * citation: <find appropriate first and educational reference>
 *
 * This implementation provides single- as well as multi-producer/consumer buffer
 * combinations for thread-safe CPU-to-CPU data transfer (optionally) using either
 * a) the POSIX mmaped(..)/munmapped(..) MMU interface, if available, and/or
 * b) the guaranteed portable standard C/C++ (de-)allocators as a fall-back.
 *
 * for more details see
 */
template <typename T, std::size_t SIZE = std::dynamic_extent, ProducerType producer_type = ProducerType::Single, WaitStrategy WAIT_STRATEGY = SleepingWaitStrategy>
requires (std::has_single_bit(sizeof(T)))
class circular_buffer
{
    using Allocator         = std::pmr::polymorphic_allocator<T>;
    using BufferType        = circular_buffer<T, SIZE, producer_type, WAIT_STRATEGY>;
    using ClaimType         = detail::producer_type_v<SIZE, producer_type, WAIT_STRATEGY>;
    using DependendsType    = std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>;

    struct buffer_impl {
        alignas(kCacheLine) Allocator                   _allocator{};
        alignas(kCacheLine) const bool                  _is_mmap_allocated;
        alignas(kCacheLine) const std::size_t           _size;
        alignas(kCacheLine) std::vector<T, Allocator>   _data;
        alignas(kCacheLine) Sequence                    _cursor;
        alignas(kCacheLine) WAIT_STRATEGY               _wait_strategy = WAIT_STRATEGY();
        alignas(kCacheLine) ClaimType                   _claim_strategy;
        // list of dependent reader indices
        alignas(kCacheLine) DependendsType              _read_indices{ std::make_shared<std::vector<std::shared_ptr<Sequence>>>() };

        buffer_impl() = delete;
        buffer_impl(const std::size_t min_size, Allocator allocator) : _allocator(allocator), _is_mmap_allocated(dynamic_cast<double_mapped_memory_resource *>(_allocator.resource())),
            _size(align_with_page_size(min_size, _is_mmap_allocated)), _data(buffer_size(_size, _is_mmap_allocated), _allocator), _claim_strategy(ClaimType(_cursor, _wait_strategy, _size)) {
        }

#ifdef HAS_POSIX_MAP_INTERFACE
        static std::size_t align_with_page_size(const std::size_t min_size, bool _is_mmap_allocated) {
            return _is_mmap_allocated ? util::round_up(min_size * sizeof(T), static_cast<std::size_t>(getpagesize())) / sizeof(T) : min_size;
        }
#else
        static std::size_t align_with_page_size(const std::size_t min_size, bool) {
            return min_size; // mmap() & getpagesize() not supported for non-POSIX OS
        }
#endif

        static std::size_t buffer_size(const std::size_t size, bool _is_mmap_allocated) {
            // double-mmaped behaviour requires the different size/alloc strategy
            // i.e. the second buffer half may not default-constructed as it's identical to the first one
            // and would result in a double dealloc during the default destruction
            return _is_mmap_allocated ? size : 2 * size;
        }
    };

    template <typename U = T>
    class buffer_writer {
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        alignas(kCacheLine) BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        alignas(kCacheLine) bool                        _is_mmap_allocated;
        alignas(kCacheLine) std::size_t                 _size;
        alignas(kCacheLine) std::vector<U, Allocator>*  _data;
        alignas(kCacheLine) ClaimType*                  _claim_strategy;

    public:
        buffer_writer() = delete;
        buffer_writer(std::shared_ptr<buffer_impl> buffer) :
            _buffer(std::move(buffer)), _is_mmap_allocated(_buffer->_is_mmap_allocated),
            _size(_buffer->_size), _data(std::addressof(_buffer->_data)), _claim_strategy(std::addressof(_buffer->_claim_strategy)) { };
        buffer_writer(buffer_writer&& other)
            : _buffer(std::move(other._buffer))
            , _is_mmap_allocated(_buffer->_is_mmap_allocated)
            , _size(_buffer->_size)
            , _data(std::addressof(_buffer->_data))
            , _claim_strategy(std::addressof(_buffer->_claim_strategy)) { };
        buffer_writer& operator=(buffer_writer tmp) {
            std::swap(_buffer, tmp._buffer);
            _is_mmap_allocated = _buffer->_is_mmap_allocated;
            _size = _buffer->_size;
            _data = std::addressof(_buffer->_data);
            _claim_strategy = std::addressof(_buffer->_claim_strategy);

            return *this;
        }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return circular_buffer(_buffer); };

        [[nodiscard]] constexpr auto get(std::size_t n_slots_to_claim) noexcept -> std::pair<std::span<U>, std::pair<std::size_t, std::int64_t>> {
            try {
                const auto sequence = _claim_strategy->next(*_buffer->_read_indices, n_slots_to_claim); // alt: try_next
                const std::size_t index = (sequence + _size - n_slots_to_claim) % _size;
                return {{ &(*_data)[index], n_slots_to_claim }, {index, sequence - n_slots_to_claim } };
            } catch (const NoCapacityException &) {
                return { { /* empty span */ }, { 0, 0 }};
            }
        }

        constexpr void publish(std::pair<std::size_t, std::int64_t> token, std::size_t n_produced) {
            if (!_is_mmap_allocated) {
                // mirror samples below/above the buffer's wrap-around point
                const std::size_t index = token.first;
                const size_t nFirstHalf = std::min(_size - index, n_produced);
                const size_t nSecondHalf = n_produced  - nFirstHalf;

                auto& data = *_data;
                std::copy(&data[index], &data[index + nFirstHalf], &data[index+ _size]);
                std::copy(&data[_size],  &data[_size + nSecondHalf], &data[0]);
            }
            _claim_strategy->publish(token.second + n_produced); // points at first non-writable index
        }

        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void publish(Translator&& translator, std::size_t n_slots_to_claim = 1, Args&&... args) {
            if (n_slots_to_claim <= 0 || _buffer->_read_indices->empty()) {
                return;
            }
            const auto sequence = _claim_strategy->next(*_buffer->_read_indices, n_slots_to_claim);
            translate_and_publish(std::forward<Translator>(translator), n_slots_to_claim, sequence, std::forward<Args>(args)...);
        } // blocks until elements are available

        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr bool try_publish(Translator&& translator, std::size_t n_slots_to_claim = 1, Args&&... args) {
            if (n_slots_to_claim <= 0 || _buffer->_read_indices->empty()) {
                return true;
            }
            try {
                const auto sequence = _claim_strategy->tryNext(*_buffer->_read_indices, n_slots_to_claim);
                translate_and_publish(std::forward<Translator>(translator), n_slots_to_claim, sequence, std::forward<Args>(args)...);
                return true;
            } catch (const NoCapacityException &) {
                return false;
            }
        }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return _claim_strategy->getRemainingCapacity(*_buffer->_read_indices);
        }

        private:
        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void translate_and_publish(Translator&& translator, const std::size_t n_slots_to_claim, const std::int64_t publishSequence, const Args&... args) {
            try {
                auto& data = *_data;
                const std::size_t index = (publishSequence + _size - n_slots_to_claim) % _size;
                std::span<U> writable_data(&data[index], n_slots_to_claim);
                if constexpr (std::is_invocable<Translator, std::span<T>&, std::int64_t, Args...>::value) {
                    std::invoke(std::forward<Translator>(translator), std::forward<std::span<T>&>(writable_data), publishSequence - n_slots_to_claim, args...);
                } else {
                    std::invoke(std::forward<Translator>(translator), std::forward<std::span<T>&>(writable_data), args...);
                }

                if (!_is_mmap_allocated) {
                    // mirror samples below/above the buffer's wrap-around point
                    const size_t nFirstHalf = std::min(_size - index, n_slots_to_claim);
                    const size_t nSecondHalf = n_slots_to_claim  - nFirstHalf;

                    std::copy(&data[index], &data[index + nFirstHalf], &data[index+ _size]);
                    std::copy(&data[_size],  &data[_size + nSecondHalf], &data[0]);
                }
                _claim_strategy->publish(publishSequence); // points at first non-writable index
            } catch (const std::exception& e) {
                throw e;
            } catch (...) {
                throw std::runtime_error("circular_buffer::translate_and_publish() - unknown user exception thrown");
            }
        }
    };

    template<typename U = T>
    class buffer_reader
    {
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        alignas(kCacheLine) std::shared_ptr<Sequence>   _read_index = std::make_shared<Sequence>();
        alignas(kCacheLine) std::int64_t                _read_index_cached;
        alignas(kCacheLine) BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        alignas(kCacheLine) std::size_t                 _size;
        alignas(kCacheLine) std::vector<U, Allocator>*  _data;

    public:
        buffer_reader() = delete;
        buffer_reader(std::shared_ptr<buffer_impl> buffer) :
            _buffer(buffer), _size(buffer->_size), _data(std::addressof(buffer->_data)) {
            gr::detail::addSequences(_buffer->_read_indices, _buffer->_cursor, {_read_index});
            _read_index_cached = _read_index->value();
        }
        buffer_reader(buffer_reader&& other)
            : _read_index(std::move(other._read_index))
            , _read_index_cached(std::exchange(other._read_index_cached, _read_index->value()))
            , _buffer(other._buffer)
            , _size(_buffer->_size)
            , _data(std::addressof(_buffer->_data)) {
        }
        buffer_reader& operator=(buffer_reader tmp) noexcept {
            std::swap(_read_index, tmp._read_index);
            std::swap(_read_index_cached, tmp._read_index_cached);
            std::swap(_buffer, tmp._buffer);
            _size = _buffer->_size;
            _data = std::addressof(_buffer->_data);
            return *this;
        };
        ~buffer_reader() { gr::detail::removeSequence( _buffer->_read_indices, _read_index); }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return circular_buffer(_buffer); };

        template <bool strict_check = true>
        [[nodiscard]] constexpr std::span<const U> get(const std::size_t n_requested = 0) const noexcept {
            const auto& data = *_data;
            if constexpr (strict_check) {
                const std::size_t n = n_requested > 0 ? std::min(n_requested, available()) : available();
                return { &data[static_cast<std::uint64_t>(_read_index_cached) % _size], n };
            }
            const std::size_t n = n_requested > 0 ? n_requested : available();
            return { &data[static_cast<std::uint64_t>(_read_index_cached) % _size], n };
        }

        template <bool strict_check = true>
        [[nodiscard]] constexpr bool consume(const std::size_t n_elements = 1) noexcept {
            if constexpr (strict_check) {
                if (n_elements <= 0) {
                    return true;
                }
                if (n_elements > available()) {
                    return false;
                }
            }
            _read_index_cached = _read_index->addAndGet(static_cast<int64_t>(n_elements));
            return true;
        }

        [[nodiscard]] constexpr std::int64_t position() const noexcept { return _read_index_cached; }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return _buffer->_cursor.value() - _read_index_cached;
        }
    };

    [[nodiscard]] constexpr static Allocator DefaultAllocator() {
        if constexpr (has_posix_mmap_interface) {
            return double_mapped_memory_resource::allocator<T>();
        } else {
            return Allocator();
        }
    }

    std::shared_ptr<buffer_impl> _shared_buffer_ptr;
    circular_buffer(std::shared_ptr<buffer_impl> shared_buffer_ptr) : _shared_buffer_ptr(shared_buffer_ptr) {}

public:
    circular_buffer() = delete;
    circular_buffer(std::size_t min_size, Allocator allocator = DefaultAllocator())
        : _shared_buffer_ptr(std::make_shared<buffer_impl>(min_size, allocator)) { }
    ~circular_buffer() = default;

    [[nodiscard]] std::size_t       size() const noexcept { return _shared_buffer_ptr->_size; }
    [[nodiscard]] BufferWriter auto new_writer() { return buffer_writer<T>(_shared_buffer_ptr); }
    [[nodiscard]] BufferReader auto new_reader() { return buffer_reader<T>(_shared_buffer_ptr); }

    // implementation specific interface -- not part of public Buffer / production-code API
    [[nodiscard]] auto n_readers()       { return _shared_buffer_ptr->_read_indices->size(); }
    [[nodiscard]] auto claim_strategy()  { return _shared_buffer_ptr->_claim_strategy; }
    [[nodiscard]] auto wait_strategy()   { return _shared_buffer_ptr->_wait_strategy; }
    [[nodiscard]] auto cursor_sequence() { return _shared_buffer_ptr->_cursor; }

};
static_assert(Buffer<circular_buffer<int32_t>>);
// clang-format on

} // namespace gr

#endif // GNURADIO_CIRCULAR_BUFFER_HPP
#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP


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
class dynamic_port {
    struct model { // intentionally class-private definition to limit interface exposure and enhance composition
        virtual ~model() = default;

        [[nodiscard]] virtual supported_type
        pmt_type() const noexcept
                = 0;

        [[nodiscard]] virtual port_type_t
        type() const noexcept
                = 0;

        [[nodiscard]] virtual port_direction_t
        direction() const noexcept
                = 0;

        [[nodiscard]] virtual std::string_view
        name() const noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        resize_buffer(std::size_t min_size) noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        disconnect() noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        connect(dynamic_port &dst_port) = 0;

        // internal runtime polymorphism access
        [[nodiscard]] virtual bool
        update_reader_internal(void *buffer_other) noexcept
                = 0;
    };

    std::unique_ptr<model> _accessor;

    template<Port T, bool owning>
    class wrapper final : public model {
        using PortType = std::decay_t<T>;
        std::conditional_t<owning, PortType, PortType &> _value;

        [[nodiscard]] void *
        writer_handler_internal() noexcept {
            return _value.writer_handler_internal();
        };

        [[nodiscard]] bool
        update_reader_internal(void *buffer_other) noexcept override {
            if constexpr (T::IS_INPUT) {
                return _value.update_reader_internal(buffer_other);
            } else {
                assert(!"This works only on input ports");
                return false;
            }
        }

    public:
        wrapper()                = delete;

        wrapper(const wrapper &) = delete;

        auto &
        operator=(const wrapper &)
                = delete;

        auto &
        operator=(wrapper &&)
                = delete;

        explicit constexpr wrapper(T &arg) noexcept : _value{ arg } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<void *>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        explicit constexpr wrapper(T &&arg) noexcept : _value{ std::move(arg) } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<void *>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        ~wrapper() override = default;

        [[nodiscard]] constexpr supported_type
        pmt_type() const noexcept override {
            return _value.pmt_type();
        }

        [[nodiscard]] constexpr port_type_t
        type() const noexcept override {
            return _value.type();
        }

        [[nodiscard]] constexpr port_direction_t
        direction() const noexcept override {
            return _value.direction();
        }

        [[nodiscard]] constexpr std::string_view
        name() const noexcept override {
            return _value.name();
        }

        [[nodiscard]] connection_result_t
        resize_buffer(std::size_t min_size) noexcept override {
            return _value.resize_buffer(min_size);
        }

        [[nodiscard]] connection_result_t
        disconnect() noexcept override {
            return _value.disconnect();
        }

        [[nodiscard]] connection_result_t
        connect(dynamic_port &dst_port) override {
            if constexpr (T::IS_OUTPUT) {
                auto src_buffer = _value.writer_handler_internal();
                return dst_port.update_reader_internal(src_buffer) ? connection_result_t::SUCCESS
                                                                   : connection_result_t::FAILED;
            } else {
                assert(!"This works only on input ports");
                return connection_result_t::FAILED;
            }
        }
    };

    bool
    update_reader_internal(void *buffer_other) noexcept {
        return _accessor->update_reader_internal(buffer_other);
    }

public:
    using value_type         = void; // a sterile port

    constexpr dynamic_port() = delete;

    template<Port T>
    constexpr dynamic_port(const T &arg) = delete;

    template<Port T>
    explicit constexpr dynamic_port(T &arg) noexcept : _accessor{ std::make_unique<wrapper<T, false>>(arg) } {}

    template<Port T>
    explicit constexpr dynamic_port(T &&arg) noexcept : _accessor{ std::make_unique<wrapper<T, true>>(std::forward<T>(arg)) } {}

    [[nodiscard]] supported_type
    pmt_type() const noexcept {
        return _accessor->pmt_type();
    }

    [[nodiscard]] port_type_t
    type() const noexcept {
        return _accessor->type();
    }

    [[nodiscard]] port_direction_t
    direction() const noexcept {
        return _accessor->direction();
    }

    [[nodiscard]] std::string_view
    name() const noexcept {
        return _accessor->name();
    }

    [[nodiscard]] connection_result_t
    resize_buffer(std::size_t min_size) {
        if (direction() == port_direction_t::OUTPUT) {
            return _accessor->resize_buffer(min_size);
        }
        return connection_result_t::FAILED;
    }

    [[nodiscard]] connection_result_t
    disconnect() noexcept {
        return _accessor->disconnect();
    }

    [[nodiscard]] connection_result_t
    connect(dynamic_port &dst_port) {
        return _accessor->connect(dst_port);
    }
};

static_assert(Port<dynamic_port>);

#define ENABLE_PYTHON_INTEGRATION
#ifdef ENABLE_PYTHON_INTEGRATION

// TODO: Not yet implemented
class dynamic_node {
private:
    // TODO: replace the following with array<2, vector<dynamic_port>>
    using dynamic_ports = std::vector<dynamic_port>;
    dynamic_ports                                         _dynamic_input_ports;
    dynamic_ports                                         _dynamic_output_ports;

    std::function<void(dynamic_ports &, dynamic_ports &)> _process;

public:
    void
    work() {
        _process(_dynamic_input_ports, _dynamic_output_ports);
    }

    template<typename T>
    void
    add_port(T &&port) {
        switch (port.direction()) {
        case port_direction_t::INPUT:
            if (auto portID = port_index<port_direction_t::INPUT>(port.name()); portID.has_value()) {
                throw std::invalid_argument(fmt::format("port already has a defined input port named '{}' at ID {}", port.name(), portID.value()));
            }
            _dynamic_input_ports.emplace_back(std::forward<T>(port));
            break;

        case port_direction_t::OUTPUT:
            if (auto portID = port_index<port_direction_t::OUTPUT>(port.name()); portID.has_value()) {
                throw std::invalid_argument(fmt::format("port already has a defined output port named '{}' at ID {}", port.name(), portID.value()));
            }
            _dynamic_output_ports.emplace_back(std::forward<T>(port));
            break;

        default: assert(false && "cannot add port with ANY designation");
        }
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_input_port(std::size_t index) {
        return index < _dynamic_input_ports.size() ? std::optional{ &_dynamic_input_ports[index] } : std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    dynamic_input_port_index(std::string_view name) const {
        auto       portNameMatches = [name](const auto &port) { return port.name() == name; };
        const auto it              = std::find_if(_dynamic_input_ports.cbegin(), _dynamic_input_ports.cend(), portNameMatches);
        return it != _dynamic_input_ports.cend() ? std::optional{ std::distance(_dynamic_input_ports.cbegin(), it) } : std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_input_port(std::string_view name) {
        if (const auto index = dynamic_input_port_index(name); index.has_value()) {
            return &_dynamic_input_ports[*index];
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_output_port(std::size_t index) {
        return index < _dynamic_output_ports.size() ? std::optional{ &_dynamic_output_ports[index] } : std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    dynamic_output_port_index(std::string_view name) const {
        auto       portNameMatches = [name](const auto &port) { return port.name() == name; };
        const auto it              = std::find_if(_dynamic_output_ports.cbegin(), _dynamic_output_ports.cend(), portNameMatches);
        return it != _dynamic_output_ports.cend() ? std::optional{ std::distance(_dynamic_output_ports.cbegin(), it) } : std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_output_port(std::string_view name) {
        if (const auto index = dynamic_output_port_index(name); index.has_value()) {
            return &_dynamic_output_ports[*index];
        }
        return std::nullopt;
    }

    [[nodiscard]] std::span<const dynamic_port>
    dynamic_input_ports() const noexcept {
        return _dynamic_input_ports;
    }

    [[nodiscard]] std::span<const dynamic_port>
    dynamic_output_ports() const noexcept {
        return _dynamic_output_ports;
    }
};

#endif


class graph {
private:
    class node_model {
    public:
        virtual ~node_model() = default;

        virtual std::string_view
        name() const
                = 0;

        virtual work_return_t
        work() = 0;

        virtual void *
        raw() const
                = 0;
    };

    template<typename T>
    class reference_node_wrapper final : public node_model {
    private:
        T *_node;

        auto &
        data() {
            return *_node;
        }

        const auto &
        data() const {
            return *_node;
        }

    public:
        reference_node_wrapper(const reference_node_wrapper &other) = delete;

        reference_node_wrapper &
        operator=(const reference_node_wrapper &other)
                = delete;

        reference_node_wrapper(reference_node_wrapper &&other) : _node(std::exchange(other._node, nullptr)) {}

        reference_node_wrapper &
        operator=(reference_node_wrapper &&other) {
            auto tmp = std::move(other);
            std::swap(_node, tmp._node);
            return *this;
        }

        ~reference_node_wrapper() override = default;

        template<typename In>
        reference_node_wrapper(In &&node) : _node(std::forward<In>(node)) {}

        work_return_t
        work() override {
            return data().work();
        }

        std::string_view
        name() const override {
            return data().name();
        }

        void *
        raw() const override {
            return _node;
        }
    };

    class edge {
    public:
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;
        std::unique_ptr<node_model> _src_node;
        std::unique_ptr<node_model> _dst_node;
        std::size_t                 _src_port_index;
        std::size_t                 _dst_port_index;
        int32_t                     _weight;
        std::string                 _name; // custom edge name
        bool                        _connected;

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

        edge(std::unique_ptr<node_model> src_node, std::size_t src_port_index, std::unique_ptr<node_model> dst_node, std::size_t dst_port_index, int32_t weight, std::string_view name)
            : _src_node(std::move(src_node))
            , _dst_node(std::move(dst_node))
            , _src_port_index(src_port_index)
            , _dst_port_index(dst_port_index)
            , _weight(weight)
            , _name(name) {
            // if (!_src_node->port<OUTPUT>(_src_port_index)) {
            //     throw fmt::format("source node '{}' has not output port id {}", std::string() /* _src_node->name() */, _src_port_index);
            // }
            // if (!_dst_node->port<INPUT>(_dst_port_index)) {
            //     throw fmt::format("destination node '{}' has not output port id {}", std::string() /*_dst_node->name()*/, _dst_port_index);
            // }
            // const dynamic_port& src_port = *_src_node->port<OUTPUT>(_src_port_index).value();
            // const dynamic_port& dst_port = *_dst_node->port<INPUT>(_dst_port_index).value();
            // if (src_port.pmt_type().index() != dst_port.pmt_type().index()) {
            //     throw fmt::format("edge({}::{}<{}> -> {}::{}<{}>, weight: {}, name:\"{}\") incompatible to type id='{}'",
            //         std::string() /*_src_node->name()*/, std::string() /*src_port.name()*/, src_port.pmt_type().index(),
            //         std::string() /*_dst_node->name()*/, std::string() /*dst_port.name()*/, dst_port.pmt_type().index(),
            //         _weight, _name, dst_port.pmt_type().index());
            // }
        }

        // edge(std::shared_ptr<node_model> src_node, std::string_view src_port_name, std::shared_ptr<node_model> dst_node, std::string_view dst_port_name, int32_t weight, std::string_view name) :
        //         _src_node(src_node), _dst_node(dst_node), _weight(weight), _name(name) {
        //     const auto src_id = _src_node->port_index<OUTPUT>(src_port_name);
        //     const auto dst_id = _dst_node->port_index<INPUT>(dst_port_name);
        //     if (!src_id) {
        //         throw std::invalid_argument(fmt::format("source node '{}' has not output port '{}'", std::string() /*_src_node->name()*/, src_port_name));
        //     }
        //     if (!dst_id) {
        //         throw fmt::format("destination node '{}' has not output port '{}'", std::string() /*_dst_node->name()*/, dst_port_name);
        //     }
        //     _src_port_index = src_id.value();
        //     _dst_port_index = dst_id.value();
        //     const dynamic_port& src_port = *src_node->port<OUTPUT>(_src_port_index).value();
        //     const dynamic_port& dst_port = *dst_node->port<INPUT>(_dst_port_index).value();
        //     if (src_port.pmt_type().index() != dst_port.pmt_type().index()) {
        //         throw fmt::format("edge({}::{}<{}> -> {}::{}<{}>, weight: {}, name:\"{}\") incompatible to type id='{}'",
        //                           std::string() /*_src_node->name()*/, src_port.name(), src_port.pmt_type().index(),
        //                           std::string() /*_dst_node->name()*/, dst_port.name(), dst_port.pmt_type().index(),
        //                           _weight, _name, dst_port.pmt_type().index());
        //     }
        // }

        [[nodiscard]] constexpr int32_t
        weight() const noexcept {
            return _weight;
        }

        [[nodiscard]] constexpr std::string_view
        name() const noexcept {
            return _name;
        }

        [[nodiscard]] constexpr bool
        connected() const noexcept {
            return _connected;
        }

        [[nodiscard]] connection_result_t
        connect() noexcept {
            return connection_result_t::FAILED;
        }

        [[nodiscard]] connection_result_t
        disconnect() noexcept { /* return _dst_node->port<INPUT>(_dst_port_index).value()->disconnect(); */
            return connection_result_t::FAILED;
        }
    };

    std::vector<edge>                        _edges;
    std::vector<std::unique_ptr<node_model>> _nodes;

    template<std::size_t src_port_index, std::size_t dst_port_index, typename Source_, typename Destination_>
    [[nodiscard]] connection_result_t
    connect_impl(Source_ &src_node_raw, Destination_ &dst_node_raw, int32_t weight = 0,
            std::string_view name = "unnamed edge") {
        using Source = std::remove_cvref_t<Source_>;
        using Destination = std::remove_cvref_t<Destination_>;
        static_assert(
                std::is_same_v<typename Source::output_port_types::template at<src_port_index>, typename Destination::input_port_types::template at<dst_port_index>>,
                "The source port type needs to match the sink port type");

        OutPort auto &source_port = output_port<src_port_index>(&src_node_raw);
        InPort auto &destination_port = input_port<dst_port_index>(&dst_node_raw);

        if (!std::any_of(_nodes.begin(), _nodes.end(), [&](const auto &registered_node) {
            return registered_node->raw() == std::addressof(src_node_raw);
        })
            || !std::any_of(_nodes.begin(), _nodes.end(), [&](const auto &registered_node) {
            return registered_node->raw() == std::addressof(dst_node_raw);
        })) {
            throw std::runtime_error(fmt::format("Can not connect nodes that are not registered first:\n {}:{} -> {}:{}\n", src_node_raw.name(), src_port_index, dst_node_raw.name(), dst_port_index));
        }

        auto result = source_port.connect(destination_port);
        if (result == connection_result_t::SUCCESS) {
            std::unique_ptr<node_model> src_node = std::make_unique<reference_node_wrapper<Source>>(std::addressof(src_node_raw));
            std::unique_ptr<node_model> dst_node = std::make_unique<reference_node_wrapper<Destination>>(std::addressof(dst_node_raw));
            _edges.emplace_back(std::move(src_node), src_port_index, std::move(dst_node), src_port_index, weight, name);
        }

        return result;
    }

    std::vector<std::function<connection_result_t(graph&)>> _connection_definitions;

    // Just a dummy class that stores the graph and the source node and port
    // to be able to split the connection into two separate calls
    // connect(source) and .to(destination)
    template <std::size_t src_port_index, typename Source>
    struct source_connector {
        graph& self;
        Source& source;

        source_connector(graph& _self, Source& _source) : self(_self), source(_source) {}

        template <std::size_t dst_port_index, typename Destination>
        [[nodiscard]] auto to(Destination& destination) {
            self._connection_definitions.push_back([source = &source, &destination] (graph& _self) {
                return _self.connect_impl<src_port_index, dst_port_index>(*source, destination);
            });
            return connection_result_t::SUCCESS;
        }

        template <fixed_string dst_port_name, typename Destination>
        [[nodiscard]] auto to(Destination& destination) {
            return to<meta::indexForName<dst_port_name, typename Destination::input_ports>()>(destination);
        }

        source_connector(const source_connector&) = delete;
        source_connector(source_connector&&) = delete;
        source_connector& operator=(const source_connector&) = delete;
        source_connector& operator=(source_connector&&) = delete;
    };

    struct init_proof {
        init_proof(bool _success) : success(_success) {}
        bool success = true;

        operator bool() const { return success; }
    };

public:
    template<std::size_t src_port_index, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        return source_connector<src_port_index, Source>(*this, source);
    }

    template<fixed_string src_port_name, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        return connect<meta::indexForName<src_port_name, typename Source::output_ports>(), Source>(source);
    }

    auto
    edges_count() const {
        return _edges.size();
    }

    template<typename Node>
    void
    register_node(Node &node) {
        static_assert(std::is_same_v<Node, std::remove_reference_t<Node>>);
        _nodes.push_back(std::make_unique<reference_node_wrapper<Node>>(std::addressof(node)));
    }

    init_proof init() {
        return init_proof(
            std::all_of(_connection_definitions.begin(), _connection_definitions.end(), [this] (auto& connection_definition) {
                return connection_definition(*this) == connection_result_t::SUCCESS;
            }));
    }

    work_return_t
    work(init_proof& init) {
        if (!init) {
            return work_return_t::ERROR;
        }
        bool run = true;
        while (run) {
            bool something_happened = false;
            for (auto &node : _nodes) {
                auto result = node->work();
                if (result == work_return_t::ERROR) {
                    return work_return_t::ERROR;
                } else if (result == work_return_t::INSUFFICIENT_INPUT_ITEMS) {
                    // nothing
                } else if (result == work_return_t::DONE) {
                    // nothing
                } else if (result == work_return_t::OK) {
                    something_happened = true;
                } else if (result == work_return_t::INSUFFICIENT_OUTPUT_ITEMS) {
                    something_happened = true;
                }
            }

            run = something_happened;
        }

        return work_return_t::DONE;
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

inline std::ostream &
operator<<(std::ostream &os, const port_domain_t &value) {
    return os << static_cast<int>(value);
}

#ifndef __EMSCRIPTEN__
auto
this_source_location(std::source_location l = std::source_location::current()) {
    return fmt::format("{}:{},{}", l.file_name(), l.line(), l.column());
}
#else
auto
this_source_location() {
    return "not yet implemented";
}
#endif // __EMSCRIPTEN__

} // namespace fair::graph

#endif // include guard
