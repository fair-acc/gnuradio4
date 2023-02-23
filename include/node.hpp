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

/**
 * @brief The 'node<Derived>' is a base class for blocks that perform specific signal processing operations. It stores
 * references to its input and output 'ports' that can be zero, one, or many, depending on the use case.
 * As the base class for all user-defined nodes, it implements common convenience functions and a default public API
 * through the Curiously-Recurring-Template-Pattern (CRTP). For example:
 * @code
 * struct user_defined_block : node<user_defined_block> {
 *   IN<float> in;
 *   OUT<float> out;
 *   // implement one of the possible work or abstracted functions
 * };
 * ENABLE_REFLECTION(user_defined_block, in, out);
 * @endcode
 * The macro `ENABLE_REFLECTION` since it relies on a template specialisation needs to be declared on the global scope.
 *
 * As an alternative definition that does not require the 'ENABLE_REFLECTION' macro and that also supports arbitrary
 * types for input 'T' and for the return 'R':
 * @code
 * template<typename T, typename R>
 * struct user_defined_block : node<user_defined_block, IN<T, 0, N_MAX, "in">, OUT<R, 0, N_MAX, "out">> {
 *   // implement one of the possible work or abstracted functions
 * };
 * @endcode
 * This implementation provides efficient compile-time static polymorphism (i.e. access to the ports, settings, etc. does
 * not require virtual functions or inheritance, which can have performance penalties in high-performance computing contexts).
 * Note: The template parameter '<Derived>' can be dropped once C++23's 'deducing this' is widely supported by compilers.
 *
 * The 'node<Derived>' implementation provides simple defaults for users who want to focus on generic signal-processing
 * algorithms and don't need full flexibility (and complexity) of using the generic `work_return_t work() {...}`.
 * The following defaults are defined for one of the two 'user_defined_block' block definitions (WIP):
 * <ul>
 * <li> <b>case 1a</b> - non-decimating N-in->N-out mechanic and automatic handling of streaming tags and settings changes:
 * @code
 *  fg::IN<T> in;
 *  fg::OUT<R> out;
 *  T _factor = T{1.0};
 *
 *  [[nodiscard]] constexpr auto process_one(T a) const noexcept {
 *      return static_cast<R>(a * _factor);
 *  }
 * @endcode
 * The number, type, and ordering of input and arguments of `process_one(..)` are defined by the port definitions.
 * <li> <b>case 1b</b> - non-decimating N-in->N-out mechanic providing bulk access to the input/output data and automatic
 * handling of streaming tags and settings changes:
 * @code
 *  [[nodiscard]] constexpr auto process_bulk(std::span<const T> input, std::span<R> output) const noexcept {
 *      std::ranges::copy(input, output | std::views::transform([a = this->_factor](T x) { return static_cast<R>(x * a); }));
 *  }
 * @endcode
 * <li> <b>case 2a</b>: N-in->M-out -> process_bulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N) (to-be-done)
 * <li> <b>case 2b</b>: N-in->M-out -> process_bulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling (to-be-done)
 * <li> <b>case 3</b> -- generic `work()` providing full access/logic capable of handling any N-in->M-out tag-handling case:
 * @code
 * [[nodiscard]] constexpr work_return_t work() const noexcept {
 *     auto &out_port = output_port<"out">(this);
 *     auto &in_port = input_port<"in">(this);
 *
 *     auto &reader = in_port.reader();
 *     auto &writer = out_port.writer();
 *     const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
 *     const auto n_writable = std::min(writer.available(), out_port.max_buffer_size());
 *     if (n_readable == 0) {
 *         return fair::graph::work_return_t::INSUFFICIENT_INPUT_ITEMS;
 *     } else if (n_writable == 0) {
 *         return fair::graph::work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
 *     }
 *     const std::size_t n_to_publish = std::min(n_readable, n_writable); // N.B. here enforcing N_input == N_output
 *
 *     writer.publish([&reader, n_to_publish, this](std::span<T> output) {
 *         const auto input = reader.get(n_to_publish);
 *         for (; i < n_to_publish; i++) {
 *             output[i] = input[i] * value;
 *         }
 *     }, n_to_publish);
 *
 *     if (!reader.consume(n_to_publish)) {
 *         return fair::graph::work_return_t::ERROR;
 *     }
 *     return fair::graph::work_return_t::OK;
 * }
 * @endcode
 * <li> <b>case 4</b>:  Python -> map to cases 1-3 and/or dedicated callback (to-be-implemented)
 * <li> <b>special cases<b>: (to-be-implemented)
 *     * case sources: HW triggered vs. generating data per invocation (generators via Port::MIN)
 *     * case sinks: HW triggered vs. fixed-size consumer (may block/never finish for insufficient input data and fixed Port::MIN>0)
 * <ul>
 * @tparam Derived the user-defined block CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 * @tparam Arguments NTTP list containing the compile-time defined port instances, setting structs, or other constraints.
 */
template<typename Derived, typename... Arguments>
class node : protected std::tuple<Arguments...> {
public:
    using derived_t = Derived;
    using node_template_parameters = meta::typelist<Arguments...>;

private:
    using setting_map = std::map<std::string, int, std::less<>>;
    std::string _name{std::string(fair::meta::type_name<Derived>())};

    [[nodiscard]] constexpr auto &
    self() noexcept {
        return *static_cast<Derived *>(this);
    }

    [[nodiscard]] constexpr const auto &
    self() const noexcept {
        return *static_cast<const Derived *>(this);
    }

public:
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

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    template<typename Self>
    [[nodiscard]] constexpr auto static
    inputs_status(Self &self) noexcept {
        bool at_least_one_input_has_data = false;
        const std::size_t available_values_count = [&self, &at_least_one_input_has_data]() {
            if constexpr (traits::node::input_ports<Derived>::size > 0) {
                const auto availableForPort = [&at_least_one_input_has_data]<typename Port>(Port &port) noexcept {
                    const std::size_t available = port.reader().available();
                    if (available > 0LU) at_least_one_input_has_data = true;
                    if (available < port.min_buffer_size()) {
                        return 0LU;
                    } else {
                        return std::min(available, port.max_buffer_size());
                    }
                };

                return std::apply(
                        [&availableForPort] (auto&... input_port) {
                            return meta::safe_min(availableForPort(input_port)...);
                        },
                        input_ports(&self));
            } else {
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

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    template<typename Self>
    auto
    write_to_outputs(Self &self, std::size_t available_values_count, auto &writers_tuple) noexcept {
        if constexpr (traits::node::output_ports<Derived>::size > 0) {
            meta::tuple_for_each([available_values_count](auto &output_port, auto &writer) {
                                     output_port.writer().publish(writer.second, available_values_count);
                                 },
                                 output_ports(&self), writers_tuple);
        }
    }

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    template<typename Self>
    bool
    consume_readers(Self& self, std::size_t available_values_count) {
        bool success = true;
        if constexpr (traits::node::input_ports<Derived>::size > 0) {
            std::apply([available_values_count, &success] (auto&... input_port) {
                    ((success = success && input_port.reader().consume(available_values_count)), ...);
                }, input_ports(&self));
        }
        return success;
    }

    template <typename... Ts>
    constexpr auto
    invoke_process_one(Ts&&... inputs)
    {
        if constexpr (traits::node::output_ports<Derived>::size == 0) {
            self().process_one(std::forward<Ts>(inputs)...);
            return std::tuple{};
        } else if constexpr (traits::node::output_ports<Derived>::size == 1) {
            return std::tuple{self().process_one(std::forward<Ts>(inputs)...)};
        } else {
            return self().process_one(std::forward<Ts>(inputs)...);
        }
    }

    work_return_t
    work() noexcept {
        // Capturing structured bindings does not work in Clang...
        const auto inputs_status = self().inputs_status(self());

        if (inputs_status.available_values_count == 0) {
            return inputs_status.at_least_one_input_has_data ? work_return_t::INSUFFICIENT_INPUT_ITEMS : work_return_t::DONE;
        }

        const bool all_writers_available = std::apply([inputs_status](auto &... output_port) noexcept {
            return ((output_port.writer().available() >= inputs_status.available_values_count) && ... && true);
        }, output_ports(&self()));

        if (!all_writers_available) {
            return work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
        }

        const auto input_spans = meta::tuple_transform([inputs_status](auto &input_port) noexcept {
            return input_port.reader().get(inputs_status.available_values_count);
        }, input_ports(&self()));

        const auto writers_tuple = meta::tuple_transform([inputs_status](auto &output_port) noexcept {
            return output_port.writer().get(inputs_status.available_values_count);
        }, output_ports(&self()));

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

        std::size_t i = 0_UZ;
        using input_types = traits::node::input_port_types<Derived>;
        using output_types = traits::node::output_port_types<Derived>;

        if constexpr (requires { &Derived::process_bulk;  }) {
            const work_return_t ret = std::apply([this](auto... args) { return static_cast<Derived *>(this)->process_bulk(args...); },
                                        std::tuple_cat(input_spans, meta::tuple_transform([](const auto &span) { return span.first; }, writers_tuple)));

            write_to_outputs(self(), inputs_status.available_values_count, writers_tuple);
            const bool success = consume_readers(self(), inputs_status.available_values_count);
            return success ? ret : work_return_t::ERROR;
        }

        // Loop for SIMD-enabled processing
        if constexpr (output_types::template all_of<meta::vectorizable>
                      && traits::node::can_process_simd<Derived>) {

            using Vec = meta::reduce_to_widest_simd<input_types>;

            constexpr auto simd_size = Vec::size();
            for (; i + simd_size <= inputs_status.available_values_count; i += simd_size) {
                const auto results = std::apply(
                        [&]<typename... Ts>(Ts const &...inputs) {
                            return invoke_process_one(Vec(inputs.data() + i, stdx::element_aligned)...);
                        },
                        input_spans);
                meta::tuple_for_each(
                        [i](auto &writer, auto &result) {
                            result.copy_to(writer.first /*data*/.data() + i, stdx::element_aligned);
                        },
                        writers_tuple, results);
            }
        }

        // Continues from the last index processed by SIMD loop
        for (; i < inputs_status.available_values_count; ++i) {
            const auto results = std::apply([this, i](auto &...inputs) { return invoke_process_one(inputs[i]...); },
                                            input_spans);
            meta::tuple_for_each([i](auto &writer, auto &result) { writer.first /*data*/[i] = std::move(result); },
                                 writers_tuple, results);
        }

        write_to_outputs(self(), inputs_status.available_values_count, writers_tuple);

        const bool success = consume_readers(self(), inputs_status.available_values_count);

#ifdef _DEBUG
        if (!success) {
            fmt::print("Node {} failed to consume {} values from inputs\n", self().name(), inputs_status.available_values_count);
        }
#endif

        return success ? work_return_t::OK : work_return_t::ERROR;
    } // end: work_return_t work() noexcept { ..}
};

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
    apply_left(auto &&input_tuple) noexcept {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return left.process_one(std::get<Is>(std::forward<decltype(input_tuple)>(input_tuple))...);
        }
                (std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    [[gnu::always_inline]] constexpr auto
    apply_right(auto &&input_tuple, auto &&tmp) noexcept {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset = traits::node::input_port_types<Left>::size;
            constexpr std::size_t second_offset = traits::node::input_port_types<Left>::size + sizeof...(Is);
            static_assert(
                    second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return right.process_one(std::get<first_offset + Is>(std::forward<decltype(input_tuple)>(input_tuple))...,
                                     std::forward<decltype(tmp)>(tmp), std::get<second_offset + Js>(input_tuple)...);
        }
                (std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    using input_port_types  = typename traits::node::input_port_types<base>;
    using output_port_types = typename traits::node::output_port_types<base>;
    using return_type       = typename traits::node::return_type<base>;

    [[gnu::always_inline]] constexpr merged_node(Left l, Right r) : left(std::move(l)), right(std::move(r)) {}

    template<meta::any_simd... Ts>
        requires meta::vectorizable_v<return_type> && input_port_types::template
    are_equal<typename std::remove_cvref_t<Ts>::value_type...> && traits::node::can_process_simd<Left>
            && traits::node::can_process_simd<Right> constexpr stdx::rebind_simd_t<return_type, meta::first_type<meta::typelist<std::remove_cvref_t<Ts>...>>>
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
                        (std::make_index_sequence<OutId>(),
                         std::make_index_sequence<traits::node::output_port_types<Left>::size - OutId - 1>());

            } else {
                return [&]<std::size_t... Is, std::size_t... Js, std::size_t... Ks>(std::index_sequence<Is...>,
                                                                                    std::index_sequence<Js...>,
                                                                                    std::index_sequence<Ks...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))...,
                                           std::move(std::get<OutId + 1 + Js>(left_out))...,
                                           std::move(std::get<Ks>(right_out)...));
                }
                        (std::make_index_sequence<OutId>(),
                         std::make_index_sequence<traits::node::output_port_types<Left>::size - OutId - 1>(),
                         std::make_index_sequence<Right::output_port_types::size>());
            }
        }
    } // end:: process_one

    [[gnu::always_inline]] work_return_t
    work() noexcept {
        return base::work();
    }
};

/**
 * This methods can merge simple blocks that are defined via a single `auto process_one(..)` producing a
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
 *     r += merged.process_one(a[i], b[i]);
 * }
 * @endcode
 */
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

/**
 * This methods can merge simple blocks that are defined via a single `auto process_one(..)` producing a
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
 *     r += merged.process_one(a[i], b[i]);
 * }
 * @endcode
 */
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

/**
 *
 *                                |￣￣￣￣￣￣￣￣￣￣￣￣￣￣|
 *                                |     !!!Horray!!!      |
 *                                |      you made it!     |
 *                                |                       |
 *                                |        Warning!       |
 *                                |  Beneath are dragons! |
 *                                |＿＿＿＿＿＿＿＿＿＿＿＿＿＿|
 *                                    (\__/)  ||
 *                                    (•ㅅ•)  ||
 *                                    /  　  づ
 * ******************************************************************************************************************
 *
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠊⠶⣂⣠⢠⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡑⡑⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣼⡾⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠂⠑⡳⠀⣀⠠⠤⠠⡀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣷⣤⡀⡄⠚⢸⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠁⠁⠀⠀⠀⠙⡕⠀⠀⠀⠀⢻⣿⣿⣿⣿⢿⢧⡭⣶⣿⣿⣿⣷⣶⣶⣤⡀⣀⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠪⠤⡠⣙⣿⡿⡫⠉⣿⣭⡞⣿⣿⣿⣿⣿⣿⣿⣷⠂⠨⣌⠚⡴⠂⠤⣀⢀⡂⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠁⠀⠀⠈⢩⣿⢿⣿⣿⡋⣿⣿⣧⣀⠛⠛⠟⢿⢹⣿⣷⣶⣶⣤⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣦⣾⠿⣷⣯⣿⣿⣿⢤⣿⣿⣿⣶⣶⣤⣭⣒⣕⡁⠛⠛⠂⠝⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣞⣿⣾⣿⣿⠿⠿⠿⣅⡐⣶⣿⣿⣿⣿⣿⣿⣿⡏⣿⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⣀⠅⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣷⣿⣶⣿⣿⣿⣿⣿⡿⣿⣿⣷⢿⣿⡄⠀⠛⠂⠀⠀⠀⠀⠀⠀⠀⠀⢀⢤⠀⠀⠀⢀⠠⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠜⢠⢁⡁⠒⡈⠐⠒⠒⠛⠊
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣕⢂⣬⣶⠒⠠⠀⣿⣿⠉⣿⡇⣿⣿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣷⠙⠛⡄⠀⠀⠀⠀⠀⠄⣀⢤⣴⣶⣶⡶⣿⣾⣿⣶⣶⣿⣶⣶⡂⡔⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⣸⢃⣼⠫⢓⠬⠤⠄⠶⠒⠀
 *    ⠀⠀⠀⠀⠀⠀⠴⠂⣼⣿⡄⣿⠁⣴⣷⠾⠁⠉⠛⠏⣿⠛⠿⣷⡍⣿⣿⢿⣿⣿⣿⣈⠛⡄⠀⠀⠀⠀⠀⣀⡼⣶⣿⣷⣿⣿⣯⣿⢷⣟⣿⡻⣿⣯⣿⣿⣿⣿⣾⣖⠒⠤⠀⠀⠀⠀⠀⠀⣤⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⠀⡀⣿⠁⠁⣒⢒⡠⡀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⠛⣀⡤⢱⠀⠀⠀⠉⠃⠀⠈⣳⣿⡿⣿⠿⣿⠟⠀⠀⣀⣤⠀⠀⠁⣬⣿⣿⣿⣿⣿⣟⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣿⡿⠮⡂⠀⣀⠀⢀⣼⠏⡖⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠢⠂⣞⡿⠰⠠⢀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⡿⠛⠛⠛⠃⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⡫⡱⠀⣾⠁⣀⠀⢴⣾⣿⣿⣷⣿⣿⣿⣿⣿⠟⠛⠉⠉⠁⢀⣉⡙⠛⣿⣿⣿⣿⣿⣯⣿⣿⣶⣧⣶⣟⣁⣾⠁⠀⠀⠀⠀⠀⣰⠃⣶⣷⣶⣶⣶⣶⣈⡄⢀⠒⣤⣶⣿⠁⡉⢒⠁⠑⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⠋⠙⣿⣿⣶⣄⠀⠀⠀⠀⠀⣴⣿⣿⢿⣿⣿⣿⠟⠀⢀⣾⣁⠀⠄⣴⣿⣿⣷⣿⣿⣿⣿⠟⠁⠀⠀⠀⢀⠀⣿⠉⢿⣿⢙⣐⡈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣭⢩⠂⠀⠀⠀⣀⠀⡠⣵⣯⣷⣗⣿⣿⣽⣽⣿⣯⣿⣿⠯⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠃⣒⢿⡏⠀⠀⠈⠛⣽⡽⠿⣿⣿⣿⣿⣿⣿⣿⣯⣿⣿⣤⣷⠿⣯⠟⢠⣢⣿⢿⣿⣿⣿⣽⣿⠟⠀⠀⠀⠀⠀⠀⠋⣦⣍⣻⣮⣿⣿⠋⠀⠀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣶⣥⣬⣿⣿⣿⣿⣿⠿⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣏⣶⣤⣿⢿⣿⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⢺⠋⠋⠉⠛⣿⣿⣯⢿⢿⣿⣯⣿⡿⢻⣿⣿⣽⢷⢿⣿⠿⣯⣿⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣿⣿⣍⠙⣿⣿⣿⣿⣿⣿⢿⣿⣽⣿⣿⢿⣿⣿⣿⣷⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣆⡀⠁⠙⣿⣿⣧⡀⠛⣿⣿⣿⢿⣿⣿⣿⣿⣿⡟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢻⣄⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣌⠝⠋⠀⠀⠉⠉⠉⠛⠁⠀⠀⠙⢿⣟⣟⣿⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢿⣿⣿⣿⣿⣿⣿⠿⣟⣿⣭⠤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⣤⡘⣿⣯⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⣤⠴⣒⣀⣄⠀⠀⣿⡿⣿⡻⣿⣧⣿⣿⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣶⣥⣈⢿⣦⣿⣿⠃⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣟⠀⠛⣿⣦⣸⣿⠁⢀⣤⣿⣿⢿⣿⢿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢧⣁⣉⣭⣿⣿⠿⠛⠛⠛⠛⣹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞⡻⣿⣷⣶⣿⣿⣿⣾⣿⣿⣽⡿⠟⠋⠁⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡈⢋⣴⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⢶⢷⣿⡿⠛⠛⠛⢿⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠈⠊⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢄⠀⠀⠀⠀⠀⠀⠉⠛⡿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *
 * The following macros are helpers to wrap around the existing refl-cpp macros: https://github.com/veselink1/refl-cpp
 * The are basically needed to do a struct member-field introspections, to support
 *   a) compile-time serialiser generation between std::map<std::string, pmt::pmtv> <-> user-defined settings structs
 *   b) allow for block ports being defined a member fields rather than as NTTPs of the node<T, ...> template

 * Their use is limited to the namespace scope where the block is defined (i.e. not across .so boundaries) and will be
 * supplanted once the compile-time reflection language feature is merged with the C++ standard, e.g.
 * Matúš Chochlík, Axel Naumann, David Sankel: Static reflection, P0194R3, ISO/IEC JTC1 SC22 WG21
 *    https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0194r3.html
 *
 *  These macros need to be defined in a global scope due to relying on template specialisation that cannot be done in
 *  any other namespace than the one they were declared -- for illustration see, for example:
 *  https://github.com/veselink1/refl-cpp/issues/59
 *  https://compiler-explorer.com/z/MG7fxzK4j
 *
 *  For practical purposes, the macro can be defined either at the end of the struct declaring namespace or the specific
 *  namespace exited/re-enteres such as
 *  @code
 *  namespace private::library {
 *     struct my_struct {
 *         int field_a;
 *         std::string field_b;
 *     };
 *  }
 *  ENABLE_REFLECTION(private::library:my_struct, field_a, field_b)
 *  namespace private::library {
 *   // ...
 *  @endcode
 *
 *  And please, if you want to accelerator the compile-time reflection process, please give your support and shout-out
 *  to the above authors, and contact your C++ STD Committee representative that this feature should not be delayed.
 */


/**
 * This macro can be used for simple non-templated structs and classes, e.g.
 * @code
 * struct my_struct {
 *     int field_a;
 *      std::string field_b;
 * };
 * ENABLE_REFLECTION(private::library:my_struct, field_a, field_b)
 */
#define ENABLE_REFLECTION(TypeName, ...) \
    REFL_TYPE(TypeName __VA_OPT__(, )) \
    REFL_DETAIL_FOR_EACH(REFL_DETAIL_EX_1_field __VA_OPT__(, ) __VA_ARGS__) \
    REFL_END

/**
 * This macro can be used for arbitrary templated structs and classes, that depend
 * on mixed typename and NTTP parameters
 * @code
 * template<typename T, std::size_t size>
 * struct custom_struct {
 *     T field_a;
 *     T field_b;
 *
 *     [[nodiscard]] constexpr std::size_t size() const noexcept { return size; }
 * };
 * ENABLE_REFLECTION_FOR_TEMPLATE_FULL(typename T, std::size_t size), (custom_struct<T, size>), field_a, field_a);
 */
#define ENABLE_REFLECTION_FOR_TEMPLATE_FULL(TemplateDef, TypeName, ...) \
    REFL_TEMPLATE(TemplateDef, TypeName __VA_OPT__(, )) \
    REFL_DETAIL_FOR_EACH(REFL_DETAIL_EX_1_field __VA_OPT__(, ) __VA_ARGS__) \
    REFL_END

/**
 * This macro can be used for simple templated structs and classes, that depend
 * only on pure typename-template lists
 * @code
 * template<typename T>
 * struct my_templated_struct {
 *     T field_a;
 *     T field_b;
 * };
 * ENABLE_REFLECTION_FOR_TEMPLATE(my_templated_struct, field_a, field_b);
 */
#define ENABLE_REFLECTION_FOR_TEMPLATE(Type, ...) \
    ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename ...Ts), (Type<Ts...>), __VA_ARGS__)

} // namespace fair::graph

#endif // include guard
