#ifndef GNURADIO_NODE_PORT_TRAITS_HPP
#define GNURADIO_NODE_PORT_TRAITS_HPP

#include "port.hpp"
#include <refl.hpp>
#include <utils.hpp> // localinclude

namespace fair::graph::traits::port {

template<typename T>
concept has_fixed_info_v = requires {
                                    typename T::value_type;
                                    { T::static_name() };
                                    { T::direction() } -> std::same_as<port_direction_t>;
                                    { T::type() } -> std::same_as<port_type_t>;
                                };

template<typename T>
using has_fixed_info = std::integral_constant<bool, has_fixed_info_v<T>>;

template<typename T>
struct has_fixed_info_or_is_typelist : std::false_type {};

template<typename T>
    requires has_fixed_info_v<T>
struct has_fixed_info_or_is_typelist<T> : std::true_type {};

template<typename T>
    requires(meta::is_typelist_v<T> and T::template all_of<has_fixed_info>)
struct has_fixed_info_or_is_typelist<T> : std::true_type {};

template<typename Port>
using type = typename Port::value_type;

template<typename Port>
using is_input = std::integral_constant<bool, Port::direction() == port_direction_t::INPUT>;

template<typename Port>
concept is_input_v = is_input<Port>::value;

template<typename Port>
using is_output = std::integral_constant<bool, Port::direction() == port_direction_t::OUTPUT>;

template<typename Port>
concept is_output_v = is_output<Port>::value;

template <typename Type>
concept is_port_v = is_output_v<Type> || is_input_v<Type>;

template<typename... Ports>
struct min_samples : std::integral_constant<std::size_t, std::max({ min_samples<Ports>::value... })> {};

template<typename T, fixed_string PortName, port_type_t PortType, port_direction_t PortDirection,
         std::size_t MIN_SAMPLES, std::size_t MAX_SAMPLES, gr::Buffer BufferType>
struct min_samples<fair::graph::port<T, PortName, PortType, PortDirection, MIN_SAMPLES, MAX_SAMPLES, BufferType>>
    : std::integral_constant<std::size_t, MIN_SAMPLES> {};

template<typename... Ports>
struct max_samples : std::integral_constant<std::size_t, std::min({ max_samples<Ports>::value... })> {};

template<typename T, fixed_string PortName, port_type_t PortType, port_direction_t PortDirection,
         std::size_t MIN_SAMPLES, std::size_t MAX_SAMPLES, gr::Buffer BufferType>
struct max_samples<fair::graph::port<T, PortName, PortType, PortDirection, MIN_SAMPLES, MAX_SAMPLES, BufferType>>
    : std::integral_constant<std::size_t, MAX_SAMPLES> {};

} // namespace port

#endif // include guard
