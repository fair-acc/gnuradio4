#ifndef GNURADIO_NODE_NODE_TRAITS_HPP
#define GNURADIO_NODE_NODE_TRAITS_HPP

#include <reflection.hpp>

#include <port.hpp> // localinclude
#include <port_traits.hpp> // localinclude
#include <utils.hpp> // localinclude

#include <vir/simd.h>

namespace fair::graph {
enum class work_return_status_t;
}

namespace fair::graph::traits::node {

namespace detail {
    template <typename FieldDescriptor>
    using member_type = typename FieldDescriptor::value_type;

    template <typename Type>
    using is_port = std::integral_constant<bool, port::is_port_v<Type>>;

    template <typename Port>
    constexpr bool is_port_descriptor_v = port::is_port_v<member_type<Port>>;

    template <typename Port>
    using is_port_descriptor = std::integral_constant<bool, is_port_descriptor_v<Port>>;

    template <typename PortDescriptor>
    using member_to_named_port = typename PortDescriptor::value_type::template with_name<fixed_string(refl::descriptor::get_name(PortDescriptor()).data)>;

    template<typename Node>
    struct member_ports_detector {
        static constexpr bool value = false;
    };

    template<class T, typename ValueType = std::remove_cvref_t<T>>
    concept Reflectable = refl::is_reflectable<ValueType>();

    template<Reflectable Node>
    struct member_ports_detector<Node> {
        using member_ports =
                    typename meta::to_typelist<refl::descriptor::member_list<Node>>
                        ::template filter<is_port_descriptor>
                        ::template transform<member_to_named_port>;

        static constexpr bool value = member_ports::size != 0;
    };

    template<typename Node>
    using port_name = typename Node::static_name();

    template<typename RequestedType>
    struct member_descriptor_has_type {
        template <typename Descriptor>
        using matches = std::is_same<RequestedType, member_to_named_port<Descriptor>>;
    };



} // namespace detail

template<typename...>
struct fixed_node_ports_data_helper;

// This specialization defines node attributes when the node is created
// with two type lists - one list for input and one for output ports
template<typename Node, meta::is_typelist_v InputPorts, meta::is_typelist_v OutputPorts>
    requires InputPorts::template all_of<port::has_fixed_info> &&OutputPorts::template all_of<port::has_fixed_info>
struct fixed_node_ports_data_helper<Node, InputPorts, OutputPorts> {
    using member_ports_detector = std::false_type;

    // using member_ports_detector = detail::member_ports_detector<Node>;

    using input_ports       = InputPorts;
    using output_ports      = OutputPorts;

    using input_port_types  = typename input_ports ::template transform<port::type>;
    using output_port_types = typename output_ports ::template transform<port::type>;

    using all_ports         = meta::concat<input_ports, output_ports>;
};

// This specialization defines node attributes when the node is created
// with a list of ports as template arguments
template<typename Node, port::has_fixed_info_v... Ports>
struct fixed_node_ports_data_helper<Node, Ports...> {
    using member_ports_detector = detail::member_ports_detector<Node>;

    using all_ports = std::remove_pointer_t<
        decltype([] {
            if constexpr (member_ports_detector::value) {
                return static_cast<typename member_ports_detector::member_ports*>(nullptr);
            } else {
                return static_cast<typename meta::concat<std::conditional_t<fair::meta::is_typelist_v<Ports>, Ports, meta::typelist<Ports>>...>*>(nullptr);
            }
        }())>;

    using input_ports       = typename all_ports ::template filter<port::is_input>;
    using output_ports      = typename all_ports ::template filter<port::is_output>;

    using input_port_types  = typename input_ports ::template transform<port::type>;
    using output_port_types = typename output_ports ::template transform<port::type>;
};

// clang-format off
template<typename Node,
         typename Derived = typename Node::derived_t,
         typename ArgumentList = typename Node::node_template_parameters>
using fixed_node_ports_data =
    typename ArgumentList::template filter<port::has_fixed_info_or_is_typelist>
                         ::template prepend<Node>
                         ::template apply<fixed_node_ports_data_helper>;
// clang-format on

template<typename Node>
using all_ports = typename fixed_node_ports_data<Node>::all_ports;

template<typename Node>
using input_ports = typename fixed_node_ports_data<Node>::input_ports;

template<typename Node>
using output_ports = typename fixed_node_ports_data<Node>::output_ports;

template<typename Node>
using input_port_types = typename fixed_node_ports_data<Node>::input_port_types;

template<typename Node>
using output_port_types = typename fixed_node_ports_data<Node>::output_port_types;

template<typename Node>
using input_port_types_tuple = typename input_port_types<Node>::tuple_type;

template<typename Node>
using return_type = typename output_port_types<Node>::tuple_or_type;

template<typename Node>
using input_port_names = typename input_ports<Node>::template transform<detail::port_name>;

template<typename Node>
using output_port_names = typename output_ports<Node>::template transform<detail::port_name>;

template<typename Node>
constexpr bool node_defines_ports_as_member_variables = fixed_node_ports_data<Node>::member_ports_detector::value;

template<typename Node, typename PortType>
using get_port_member_descriptor =
    typename meta::to_typelist<refl::descriptor::member_list<Node>>
        ::template filter<detail::is_port_descriptor>
        ::template filter<detail::member_descriptor_has_type<PortType>::template matches>::template at<0>;

namespace detail {
template<std::size_t... Is>
auto
can_process_one_invoke_test(auto &node, const auto &input, std::index_sequence<Is...>)
        -> decltype(node.process_one(std::get<Is>(input)...));

template<typename T>
struct exact_argument_type {
    template<std::same_as<T> U>
    constexpr operator U() const noexcept;
};

template<std::size_t... Is>
auto
can_process_one_with_offset_invoke_test(auto &node, const auto &input, std::index_sequence<Is...>) -> decltype(node.process_one(exact_argument_type<std::size_t>(), std::get<Is>(input)...));

template<typename Node>
using simd_return_type_of_can_process_one = meta::simdize<return_type<Node>, meta::simdize_size_v<meta::simdize<input_port_types_tuple<Node>>>>;
}

/* A node "can process simd" if its `process_one` function takes at least one argument and all
 * arguments can be simdized types of the actual port data types.
 *
 * The node can be a sink (no output ports).
 * The requirement of at least one function argument disallows sources.
 *
 * There is another (unnamed) concept for source nodes: Source nodes can implement
 * `process_one_simd(integral_constant)`, which returns SIMD object(s) of width N.
 */
template<typename Node>
concept can_process_one_simd =
#if DISABLE_SIMD
        false;
#else
        traits::node::input_port_types<Node>::size() > 0 and requires(Node &node, const meta::simdize<input_port_types_tuple<Node>> &input_simds) {
            {
                detail::can_process_one_invoke_test(node, input_simds, std::make_index_sequence<traits::node::input_ports<Node>::size()>())
            } -> std::same_as<detail::simd_return_type_of_can_process_one<Node>>;
        };
#endif

template<typename Node>
concept can_process_one_simd_with_offset =
#if DISABLE_SIMD
        false;
#else
        traits::node::input_port_types<Node>::size() > 0 && requires(Node &node, const meta::simdize<input_port_types_tuple<Node>> &input_simds) {
            {
                detail::can_process_one_with_offset_invoke_test(node, input_simds, std::make_index_sequence<traits::node::input_ports<Node>::size()>())
            } -> std::same_as<detail::simd_return_type_of_can_process_one<Node>>;
        };
#endif

template<typename Node>
concept can_process_one_scalar = requires(Node &node, const input_port_types_tuple<Node> &inputs) {
    { detail::can_process_one_invoke_test(node, inputs, std::make_index_sequence<traits::node::input_ports<Node>::size()>()) } -> std::same_as<return_type<Node>>;
};

template<typename Node>
concept can_process_one_scalar_with_offset = requires(Node &node, const input_port_types_tuple<Node> &inputs) {
    { detail::can_process_one_with_offset_invoke_test(node, inputs, std::make_index_sequence<traits::node::input_ports<Node>::size()>()) } -> std::same_as<return_type<Node>>;
};

template<typename Node>
concept can_process_one = can_process_one_scalar<Node> or can_process_one_simd<Node> or can_process_one_scalar_with_offset<Node> or can_process_one_simd_with_offset<Node>;

template<typename Node>
concept can_process_one_with_offset = can_process_one_scalar_with_offset<Node> or can_process_one_simd_with_offset<Node>;

namespace detail {
template<typename T>
struct dummy_input_span : public std::span<const T> {    // NOSONAR
    dummy_input_span(const dummy_input_span &) = delete; // NOSONAR
    dummy_input_span(dummy_input_span &&) noexcept;      // NOSONAR
    constexpr void consume(std::size_t) noexcept;
};

template<typename T>
struct dummy_copyable_input_span : public std::span<const T> {
    constexpr void consume(std::size_t) noexcept;
};

template<typename T>
struct dummy_output_span : public std::span<T> {           // NOSONAR
    dummy_output_span(const dummy_output_span &) = delete; // NOSONAR
    dummy_output_span(dummy_output_span &&) noexcept;      // NOSONAR
    constexpr void publish(std::size_t) noexcept;
};

template<typename T>
struct dummy_copyable_output_span : public std::span<T> {
    constexpr void publish(std::size_t) noexcept;
};

template<typename>
struct nothing_you_ever_wanted {};

// This alias template is only necessary as a workaround for a bug in Clang. Instead of passing dynamic_span to transform_conditional below, C++ allows passing std::span directly.
template<typename T>
using dynamic_span = std::span<T>;

template<std::size_t... InIdx, std::size_t... OutIdx>
auto
can_process_bulk_invoke_test(auto &node, const auto &inputs, auto &outputs, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>)
        -> decltype(node.process_bulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(outputs)...));
} // namespace detail

template<typename Node>
concept can_process_bulk = requires(Node &n, typename meta::transform_types<detail::dummy_input_span, traits::node::input_port_types<Node>>::tuple_type inputs,
                                    typename meta::transform_types<detail::dummy_output_span, traits::node::output_port_types<Node>>::tuple_type outputs) {
    {
        detail::can_process_bulk_invoke_test(n, inputs, outputs, std::make_index_sequence<input_port_types<Node>::size>(), std::make_index_sequence<output_port_types<Node>::size>())
    } -> std::same_as<work_return_status_t>;
};

template<typename Node>
concept can_process_bulk_by_value = requires(Node &n, typename meta::transform_types<detail::dummy_copyable_input_span, traits::node::input_port_types<Node>>::tuple_type inputs,
                                             typename meta::transform_types<detail::dummy_copyable_output_span, traits::node::output_port_types<Node>>::tuple_type outputs) {
    {
        detail::can_process_bulk_invoke_test(n, inputs, outputs, std::make_index_sequence<input_port_types<Node>::size>(), std::make_index_sequence<output_port_types<Node>::size>())
    } -> std::same_as<work_return_status_t>;
};

/*
 * Satisfied if `Derived` has a member function `process_bulk` which can be invoked with a number of arguments matching the number of input and output ports. Input arguments must accept either a
 * std::span<const T> or any type satisfying ConsumableSpan<T>. Output arguments must accept either a std::span<T> or any type satisfying PublishableSpan<T>, except for the I-th output argument, which
 * must be std::span<T> and *not* a type satisfying PublishableSpan<T>.
 */
template<typename Derived, std::size_t I>
concept process_bulk_requires_ith_output_as_span = requires(Derived                                                                                                                      &d,
                                                            typename meta::transform_types<detail::dummy_input_span, traits::node::input_port_types<Derived>>::template apply<std::tuple> inputs,
                                                            typename meta::transform_conditional<decltype([](auto j) { return j == I; }), detail::dynamic_span, detail::dummy_output_span,
                                                                                                 traits::node::output_port_types<Derived>>::template apply<std::tuple>
                                                                    outputs,
                                                            typename meta::transform_conditional<decltype([](auto j) { return j == I; }), detail::nothing_you_ever_wanted, detail::dummy_output_span,
                                                                                                 traits::node::output_port_types<Derived>>::template apply<std::tuple>
                                                                    bad_outputs) {
    {
        []<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>,
                                                        std::index_sequence<OutIdx...>) -> decltype(d.process_bulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(outputs)...)) {
            return {};
        }(std::make_index_sequence<traits::node::input_port_types<Derived>::size>(), std::make_index_sequence<traits::node::output_port_types<Derived>::size>())
    } -> std::same_as<work_return_status_t>;
    not requires {
        []<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>,
                                                        std::index_sequence<OutIdx...>) -> decltype(d.process_bulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(bad_outputs)...)) {
            return {};
        }(std::make_index_sequence<traits::node::input_port_types<Derived>::size>(), std::make_index_sequence<traits::node::output_port_types<Derived>::size>());
    };
};

} // namespace fair::graph::traits::node

#endif // include guard
