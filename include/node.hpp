#ifndef GNURADIO_NODE_HPP
#define GNURADIO_NODE_HPP

#include <map>

#include <typelist.hpp> // localinclude
#include <port.hpp> // localinclude
#include <utils.hpp> // localinclude

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

