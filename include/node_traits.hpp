#ifndef GNURADIO_NODE_NODE_TRAITS_HPP
#define GNURADIO_NODE_NODE_TRAITS_HPP

#include <refl.hpp>

#include <port.hpp> // localinclude
#include <port_traits.hpp> // localinclude
#include <utils.hpp> // localinclude

#include <vir/simd.h>

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
        ::template filter<detail::member_descriptor_has_type<PortType>::template matches>::template at<0>;

template<typename Node>
concept can_process_simd =
    traits::node::input_port_types<Node>::size() > 0 &&
    not vir::stdx::is_simd_v<typename traits::node::input_port_types<Node>::safe_head> &&
    traits::node::input_port_types<Node>::template all_same<> &&
    traits::node::output_ports<Node>::size() > 0 &&
    requires (Node& node,
              typename traits::node::input_port_types<Node>::template transform<vir::stdx::native_simd>::template apply<std::tuple>& input_simds) {
        {
            []<std::size_t... Is>(Node &node, auto const &input, std::index_sequence<Is...>) -> decltype(node.process_one(std::get<Is>(input)...)) { return {}; }
            (node, input_simds, std::make_index_sequence<traits::node::input_ports<Node>::size()>())
        };
    };

} // namespace node

#endif // include guard
