#ifndef GNURADIO_NODE_PORT_TRAITS_HPP
#define GNURADIO_NODE_PORT_TRAITS_HPP

#include <gnuradio-4.0/meta/utils.hpp>

#include "Port.hpp"

namespace gr::traits::port {

// -------------------------------
// traits for 'Port' types

template<typename Port>
using is_input = std::integral_constant<bool, Port::direction() == PortDirection::INPUT>;

template<typename Port>
concept is_input_v = is_input<Port>::value;

template<typename Port>
using is_output = std::integral_constant<bool, Port::direction() == PortDirection::OUTPUT>;

template<typename Type>
concept is_port_v = is_output<Type>::value || is_input_v<Type>;

template<typename Type>
using is_port = std::integral_constant<bool, is_port_v<Type>>;

// actually tuple-like (including std::array)
template<typename T>
struct is_port_tuple : std::false_type {};

template<typename... Ts>
struct is_port_tuple<std::tuple<Ts...>> : std::conjunction<is_port<Ts>...> {};

template<typename T, std::size_t N>
struct is_port_tuple<std::array<T, N>> : is_port<T> {};

template<typename Collection>
concept is_port_collection_v = is_port_v<typename Collection::value_type> and not is_port_tuple<Collection>::value;

template<typename T>
using is_port_collection = std::bool_constant<is_port_collection_v<T>>;

template<typename T>
concept AnyPort = is_port_v<T> or is_port_v<typename T::value_type> or is_port_tuple<T>::value;

// -------------------------------
// traits for 'PortDescriptor' types
// FIXME: better name "describes_" instead of "is_"?

template<gr::detail::PortDescription T>
using is_stream_port = std::bool_constant<PortType::STREAM == T::kPortType>;

template<gr::detail::PortDescription T>
using is_message_port = std::bool_constant<PortType::MESSAGE == T::kPortType>;

template<gr::detail::PortDescription T>
using is_input_port = std::bool_constant<T::kIsInput>;

template<gr::detail::PortDescription T>
using is_output_port = std::bool_constant<T::kIsOutput>;

template<gr::detail::PortDescription T>
using is_dynamic_port_collection = std::bool_constant<T::kIsDynamicCollection>;

template<PortType portType>
struct is_port_type {
    template<gr::detail::PortDescription T>
    using eval = std::bool_constant<portType == PortType::ANY or portType == T::kPortType>;
};

template<gr::detail::PortDescription PortOrCollection>
using type = typename PortOrCollection::value_type;

template<gr::detail::PortDescription... Ports>
struct min_samples : std::integral_constant<std::size_t, std::max({Ports::Required::kMinSamples...})> {};

template<gr::detail::PortDescription... Ports>
struct max_samples : std::integral_constant<std::size_t, std::max({Ports::Required::kMaxSamples...})> {};

} // namespace gr::traits::port

#endif // include guard
