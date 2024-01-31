#ifndef GNURADIO_NODE_PORT_TRAITS_HPP
#define GNURADIO_NODE_PORT_TRAITS_HPP

#include <gnuradio-4.0/meta/utils.hpp>

#include "Port.hpp"

namespace gr::traits::port {

template<typename T>
concept HasFixedInfo = requires {
    typename T::value_type;
    { T::static_name() };
    { T::direction() } -> std::same_as<PortDirection>;
    { T::type() } -> std::same_as<PortType>;
};

template<typename T>
using has_fixed_info = std::integral_constant<bool, HasFixedInfo<T>>;

template<typename T>
struct has_fixed_info_or_is_typelist : std::false_type {};

template<typename T>
    requires HasFixedInfo<T>
struct has_fixed_info_or_is_typelist<T> : std::true_type {};

template<typename T>
    requires(meta::is_typelist_v<T> and T::template all_of<has_fixed_info>)
struct has_fixed_info_or_is_typelist<T> : std::true_type {};

template<typename Port>
using is_input = std::integral_constant<bool, Port::direction() == PortDirection::INPUT>;

template<typename Port>
concept is_input_v = is_input<Port>::value;

template<typename Port>
using is_output = std::integral_constant<bool, Port::direction() == PortDirection::OUTPUT>;

template<typename Port>
concept is_output_v = is_output<Port>::value;

template<typename Type>
concept is_port_v = is_output_v<Type> || is_input_v<Type>;

template<typename Type>
using is_port = std::integral_constant<bool, is_port_v<Type>>;

template<typename Collection>
concept is_port_collection_v = is_port_v<typename Collection::value_type>;

template<typename T>
auto
unwrap_port_helper() {
    if constexpr (port::is_port_v<T>) {
        return static_cast<T *>(nullptr);
    } else if constexpr (port::is_port_collection_v<T>) {
        return static_cast<typename T::value_type *>(nullptr);
    } else {
        meta::print_types<meta::message_type<"Is not a port or a collection of ports">, T>{};
    }
}

template<typename T>
using unwrap_port = std::remove_pointer_t<decltype(unwrap_port_helper<T>())>;

struct kind {
    template<typename Port>
    static constexpr auto
    value_helper() {
        if constexpr (std::is_same_v<typename Port::value_type, gr::Message>) {
            return gr::PortType::MESSAGE;
        } else {
            return gr::PortType::STREAM;
        }
    }

    template<PortType portType>
    struct tester_for {
        template<typename Port>
        static constexpr bool matches_kind = portType == PortType::ANY || kind::value_helper<Port>() == portType;

        template<typename T>
        constexpr static bool
        is_port_or_collection_helper() {
            if constexpr (port::is_port_v<T> || port::is_port_collection_v<T>) {
                return matches_kind<unwrap_port<T>>;
            } else {
                return false;
            }
        }

        template<typename T>
        using is_port_or_collection = std::integral_constant<bool, is_port_or_collection_helper<T>()>;

        template<typename T>
        using is_input_port_or_collection = std::integral_constant<bool, is_port_or_collection<T>() && port::is_input_v<unwrap_port<T>>>;

        template<typename T>
        using is_output_port_or_collection = std::integral_constant<bool, is_port_or_collection<T>() && port::is_output_v<unwrap_port<T>>>;
    };
};

template<typename PortOrCollection>
auto
type_helper() {
    if constexpr (is_port_v<PortOrCollection>) {
        return static_cast<typename PortOrCollection::value_type *>(nullptr);
    } else {
        return static_cast<std::vector<typename PortOrCollection::value_type::value_type> *>(nullptr);
    }
}

template<typename PortOrCollection>
using type = std::remove_pointer_t<decltype(type_helper<PortOrCollection>())>;

template<typename... Ports>
struct min_samples : std::integral_constant<std::size_t, std::max({ Ports::RequiredSamples::MinSamples... })> {};

template<typename... Ports>
struct max_samples : std::integral_constant<std::size_t, std::max({ Ports::RequiredSamples::MaxSamples... })> {};

template<typename Type>
constexpr bool is_not_any_port_or_collection = !gr::traits::port::kind::tester_for<PortType::ANY>::is_port_or_collection<Type>();

} // namespace gr::traits::port

#endif // include guard
