#ifndef GNURADIO_NODE_HPP
#define GNURADIO_NODE_HPP

#include <map>

#include <typelist.hpp> // localinclude
#include <port.hpp> // localinclude
#include <utils.hpp> // localinclude
#include <node_traits.hpp> // localinclude

#include <fmt/format.h>
#include <refl.hpp>

namespace fair::graph {

class graph;
using namespace fair::literals;

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
        if constexpr (traits::node::input_ports<Self>::size > 0) {
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
            return 1_UZ;
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
    if constexpr (traits::node::output_ports<Self>::size > 0) {
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
    if constexpr (traits::node::input_ports<Self>::size > 0) {
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
                static_assert(std::tuple_size_v<decltype(results)> == traits::node::output_ports<Self>::size);

                meta::tuple_for_each(
                        [i] (auto& writer, auto& result) {
                            writer.first/*data*/[i] = std::move(result); },
                        writers_tuple, results);

            } else {
                static_assert(traits::node::output_ports<Self>::size == 1);
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

template<typename Node>
concept node_can_process_simd = requires(Node &n, typename meta::transform_to_widest_simd<typename Node::input_port_types>::template apply<std::tuple> const &inputs) {
    {
        []<std::size_t... Is>(Node & n, auto const &tup, std::index_sequence<Is...>)->decltype(n.process_one(std::get<Is>(tup)...)) { return {}; }
        (n, inputs, std::make_index_sequence<Node::input_port_types::size>())
    } -> meta::any_simd<typename Node::return_type>;
};

// Ports can either be a list of ports instances,
// or two typelists containing port instances -- one for input
// ports and one for output ports
template<typename Derived, typename... Arguments>
class node : protected std::tuple<Arguments...> {
public:
    using derived_t = Derived;
    using node_template_parameters = meta::typelist<Arguments...>;

    using work_strategy     = work_strategies::default_strategy;
    friend work_strategy;

    using min_max_limits = typename meta::typelist<Arguments...>::template filter<traits::port::is_limits>;
    static_assert(min_max_limits::size <= 1);

private:
    using setting_map = std::map<std::string, int, std::less<>>;
    std::string _name{ std::string(fair::meta::type_name<Derived>()) };

    setting_map _exec_metrics{}; //  →  std::map<string, pmt> → fair scheduling, 'int' stand-in for pmtv

    friend class graph;
    graph* _owning_graph = nullptr;

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
            using In0                    = meta::first_type<traits::node::input_port_types<Derived>>;
            using V                      = stdx::resize_simd_t<N, stdx::native_simd<In0>>;
            using Vs                     = meta::transform_types<meta::rebind_simd_helper<V>::template rebind, traits::node::input_port_types<Derived>>;
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
    using requested_port_type = typename traits::node::input_ports<Self>::template at<Index>;
    if constexpr (traits::node::node_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::node::get_port_member_descriptor<Self, requested_port_type>;
        return member_descriptor()(*self);
    } else {
        return std::get<requested_port_type>(*self);
    }
}

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    using requested_port_type = typename traits::node::output_ports<Self>::template at<Index>;
    if constexpr (traits::node::node_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::node::get_port_member_descriptor<Self, requested_port_type>;
        return member_descriptor()(*self);
    } else {
        return std::get<requested_port_type>(*self);
    }
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::node::input_ports<Self>>();
    return input_port<Index, Self>(self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::node::output_ports<Self>>();
    return output_port<Index, Self>(self);
}

template<typename Self>
[[nodiscard]] constexpr auto
input_ports(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return std::tie(input_port<Idx>(self)...);
    }
    (std::make_index_sequence<traits::node::input_ports<Self>::size>());
}

template<typename Self>
[[nodiscard]] constexpr auto
output_ports(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return std::tie(output_port<Idx>(self)...);
    }
    (std::make_index_sequence<traits::node::output_ports<Self>::size>());
}

template<typename Node>
concept source_node = requires(Node &node, typename traits::node::input_port_types<Node>::tuple_type const &inputs) {
                          {
                              [](Node &n, auto &inputs) {
                                  constexpr std::size_t port_count = traits::node::input_port_types<Node>::size;
                                  if constexpr (port_count > 0) {
                                      return []<std::size_t... Is>(Node & n_inside, auto const &tup, std::index_sequence<Is...>)->decltype(n_inside.process_one(std::get<Is>(tup)...)) { return {}; }
                                      (n, inputs, std::make_index_sequence<port_count>());
                                  } else {
                                      return n.process_one();
                                  }
                              }(node, inputs)
                              } -> std::same_as<typename traits::node::return_type<Node>>;
                      };

template<typename Node>
concept sink_node = requires(Node &node, typename traits::node::input_port_types<Node>::tuple_type const &inputs) {
                        {
                            [](Node &n, auto &inputs) {
                                constexpr std::size_t port_count = traits::node::output_port_types<Node>::size;
                                []<std::size_t... Is>(Node & n_inside, auto const &tup, std::index_sequence<Is...>) {
                                    if constexpr (port_count > 0) {
                                        auto a [[maybe_unused]] = n_inside.process_one(std::get<Is>(tup)...);
                                    } else {
                                        n_inside.process_one(std::get<Is>(tup)...);
                                    }
                                }
                                (n, inputs, std::make_index_sequence<traits::node::input_port_types<Node>::size>());
                            }(node, inputs)
                        };
                    };

template<source_node Left, sink_node Right, std::size_t OutId, std::size_t InId>
class merged_node : public node<merged_node<Left, Right, OutId, InId>, meta::concat<typename traits::node::input_ports<Left>, meta::remove_at<InId, typename traits::node::input_ports<Right>>>,
                                meta::concat<meta::remove_at<OutId, typename traits::node::output_ports<Left>>, typename traits::node::output_ports<Right>>> {
private:
    // copy-paste from above, keep in sync
    using base = node<merged_node<Left, Right, OutId, InId>, meta::concat<typename traits::node::input_ports<Left>, meta::remove_at<InId, typename traits::node::input_ports<Right>>>,
                      meta::concat<meta::remove_at<OutId, typename traits::node::output_ports<Left>>, typename traits::node::output_ports<Right>>>;

    Left  left;
    Right right;

    template<std::size_t I>
    [[gnu::always_inline]] constexpr auto
    apply_left(auto &&input_tuple) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return left.process_one(std::get<Is>(input_tuple)...); }
        (std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    [[gnu::always_inline]] constexpr auto
    apply_right(auto &&input_tuple, auto &&tmp) {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset  = traits::node::input_port_types<Left>::size;
            constexpr std::size_t second_offset = traits::node::input_port_types<Left>::size + sizeof...(Is);
            static_assert(second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return right.process_one(std::get<first_offset + Is>(input_tuple)..., std::move(tmp), std::get<second_offset + Js>(input_tuple)...);
        }
        (std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    using input_port_types  = typename traits::node::input_port_types<base>;
    using output_port_types = typename traits::node::output_port_types<base>;
    using return_type       = typename traits::node::return_type<base>;

    [[gnu::always_inline]] constexpr merged_node(Left l, Right r) : left(std::move(l)), right(std::move(r)) {}

    template<meta::any_simd... Ts>
        requires meta::vectorizable<return_type> && input_port_types::template
    are_equal<typename std::remove_cvref_t<Ts>::value_type...> && node_can_process_simd<Left>
            && node_can_process_simd<Right> constexpr stdx::rebind_simd_t<return_type, meta::first_type<meta::typelist<std::remove_cvref_t<Ts>...>>>
              process_one(Ts... inputs) {
        return apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(std::tie(inputs...), apply_left<traits::node::input_port_types<Left>::size()>(std::tie(inputs...)));
    }

    template<typename... Ts>
    // In order to have nicer error messages, this is checked in the function body
    // requires input_port_types::template are_equal<std::remove_cvref_t<Ts>...>
    constexpr return_type
    process_one(Ts &&...inputs) {
        if constexpr (!input_port_types::template are_equal<std::remove_cvref_t<Ts>...>) {
            meta::print_types<decltype(this), input_port_types, std::remove_cvref_t<Ts>...> error{};
        }

        if constexpr (traits::node::output_port_types<Left>::size == 1) { // only the result from the right node needs to be returned
            return apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                                                                 apply_left<traits::node::input_port_types<Left>::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out  = apply_left<traits::node::input_port_types<Left>::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...), std::move(std::get<OutId>(left_out)));

            if constexpr (traits::node::output_port_types<Left>::size == 2 && traits::node::output_port_types<Right>::size == 1) {
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)), std::move(right_out));

            } else if constexpr (traits::node::output_port_types<Left>::size == 2) {
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))), std::move(right_out));

            } else if constexpr (traits::node::output_port_types<Right>::size == 1) {
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(right_out));
                }
                (std::make_index_sequence<OutId>(), std::make_index_sequence<traits::node::output_port_types<Left>::size - OutId - 1>());

            } else {
                return [&]<std::size_t... Is, std::size_t... Js, std::size_t... Ks>(std::index_sequence<Is...>, std::index_sequence<Js...>, std::index_sequence<Ks...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(std::get<Ks>(right_out)...));
                }
                (std::make_index_sequence<OutId>(), std::make_index_sequence<traits::node::output_port_types<Left>::size - OutId - 1>(), std::make_index_sequence<Right::output_port_types::size>());
            }
        }
    }
};

template<std::size_t OutId, std::size_t InId, source_node A, sink_node B>
[[gnu::always_inline]] constexpr auto
merge_by_index(A &&a, B &&b) -> merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    if constexpr (!std::is_same_v<typename traits::node::output_port_types<std::remove_cvref_t<A>>::template at<OutId>, typename traits::node::input_port_types<std::remove_cvref_t<B>>::template at<InId>>) {
        fair::meta::print_types<fair::meta::message_type<"OUTPUT_PORTS_ARE:">, typename traits::node::output_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, OutId>,
                                typename traits::node::output_port_types<std::remove_cvref_t<A>>::template at<OutId>,

                                fair::meta::message_type<"INPUT_PORTS_ARE:">, typename traits::node::input_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, InId>,
                                typename traits::node::input_port_types<std::remove_cvref_t<A>>::template at<InId>>{};
    }
    return { std::forward<A>(a), std::forward<B>(b) };
}

template<fixed_string OutName, fixed_string InName, source_node A, sink_node B>
[[gnu::always_inline]] constexpr auto
merge(A &&a, B &&b) {
    constexpr std::size_t OutId = meta::indexForName<OutName, typename traits::node::output_ports<A>>();
    constexpr std::size_t InId  = meta::indexForName<InName, typename traits::node::input_ports<B>>();
    static_assert(OutId != -1);
    static_assert(InId != -1);
    static_assert(std::same_as<typename traits::node::output_port_types<std::remove_cvref_t<A>>::template at<OutId>, typename traits::node::input_port_types<std::remove_cvref_t<B>>::template at<InId>>,
                  "Port types do not match");
    return merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId>{ std::forward<A>(a), std::forward<B>(b) };
}


#define ENABLE_REFLECTION(TypeName, ...) \
    REFL_TYPE(TypeName __VA_OPT__(, )) \
    REFL_DETAIL_FOR_EACH(REFL_DETAIL_EX_1_field __VA_OPT__(, ) __VA_ARGS__) \
    REFL_END

#define ENABLE_REFLECTION_FOR_TEMPLATE_FULL(TemplateDef, TypeName, ...) \
    REFL_TEMPLATE(TemplateDef, TypeName __VA_OPT__(, )) \
    REFL_DETAIL_FOR_EACH(REFL_DETAIL_EX_1_field __VA_OPT__(, ) __VA_ARGS__) \
    REFL_END

#define ENABLE_REFLECTION_FOR_TEMPLATE(Type, ...) \
    ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (Type<T>), __VA_ARGS__)

} // namespace fair::graph

#endif // include guard
