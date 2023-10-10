#ifndef GNURADIO_NODE_NODE_TRAITS_HPP
#define GNURADIO_NODE_NODE_TRAITS_HPP

#include <gnuradio-4.0/meta/utils.hpp>

#include "port.hpp"
#include "port_traits.hpp"
#include "reflection.hpp"

#include <vir/simd.h>

namespace gr {
enum class WorkReturnStatus;
}

namespace gr::traits::block {

namespace detail {

template<typename FieldDescriptor>
using member_type = typename FieldDescriptor::value_type;

template<typename T>
auto
unwrap_port_helper() {
    if constexpr (port::is_port_v<T>) {
        return static_cast<T *>(nullptr);
    } else if constexpr (port::is_port_collection_v<T>) {
        return static_cast<typename T::value_type *>(nullptr);
    } else {
        static_assert(meta::always_false<T>, "Not a port or a collection of ports");
    }
}

template<typename T>
using unwrap_port = std::remove_pointer_t<decltype(unwrap_port_helper<T>())>;

template<typename T>
using is_port_or_collection = std::integral_constant<bool, port::is_port_v<T> || port::is_port_collection_v<T>>;

template<typename T>
using is_input_port_or_collection = std::integral_constant<bool, is_port_or_collection<T>() && port::is_input_v<unwrap_port<T>>>;

template<typename T>
using is_output_port_or_collection = std::integral_constant<bool, is_port_or_collection<T>() && port::is_output_v<unwrap_port<T>>>;

template<typename Port>
constexpr bool is_port_descriptor_v = port::is_port_v<member_type<Port>>;

template<typename Collection>
constexpr bool is_port_collection_descriptor_v = port::is_port_collection_v<member_type<Collection>>;

template<typename Descriptor>
using is_port_or_collection_descriptor = std::integral_constant<bool, is_port_descriptor_v<Descriptor> || is_port_collection_descriptor_v<Descriptor>>;

template<typename Descriptor>
constexpr auto
member_to_named_port_helper() {
    // Collections of ports don't get names inside the type as
    // the ports inside are dynamically created
    if constexpr (is_port_descriptor_v<Descriptor>) {
        return static_cast<typename Descriptor::value_type::template with_name<fixed_string(refl::descriptor::get_name(Descriptor()).data)> *>(nullptr);
    } else if constexpr (is_port_collection_descriptor_v<Descriptor>) {
        return static_cast<typename Descriptor::value_type *>(nullptr);
    } else {
        return static_cast<void *>(nullptr);
    }
}

template<typename Descriptor>
using member_to_named_port = std::remove_pointer_t<decltype(member_to_named_port_helper<Descriptor>())>;

template<typename TBlock>
struct member_ports_detector {
    static constexpr bool value = false;
};

template<class T, typename ValueType = std::remove_cvref_t<T>>
concept Reflectable = refl::is_reflectable<ValueType>();

template<Reflectable TBlock>
struct member_ports_detector<TBlock> {
    using member_ports          = typename meta::to_typelist<refl::descriptor::member_list<TBlock>>::template filter<is_port_or_collection_descriptor>::template transform<member_to_named_port>;

    static constexpr bool value = member_ports::size != 0;
};

template<typename TBlock>
using port_name = typename TBlock::static_name();

template<typename RequestedType>
struct member_descriptor_has_type {
    template<typename Descriptor>
    using matches = std::is_same<RequestedType, member_to_named_port<Descriptor>>;
};

} // namespace detail

template<typename...>
struct fixedBlock_ports_data_helper;

// This specialization defines node attributes when the node is created
// with two type lists - one list for input and one for output ports
template<typename T, meta::is_typelist_v InputPorts, meta::is_typelist_v OutputPorts>
    requires InputPorts::template
all_of<port::has_fixed_info> &&OutputPorts::template all_of<port::has_fixed_info> struct fixedBlock_ports_data_helper<T, InputPorts, OutputPorts> {
    using member_ports_detector = std::false_type;

    // using member_ports_detector = detail::member_ports_detector<TBlock>;

    using input_ports       = InputPorts;
    using output_ports      = OutputPorts;

    using input_port_types  = typename input_ports ::template transform<port::type>;
    using output_port_types = typename output_ports ::template transform<port::type>;

    using all_ports         = meta::concat<input_ports, output_ports>;
};

// This specialization defines node attributes when the node is created
// with a list of ports as template arguments
template<typename TBlock, port::has_fixed_info_v... Ports>
struct fixedBlock_ports_data_helper<TBlock, Ports...> {
    using member_ports_detector = detail::member_ports_detector<TBlock>;

    using all_ports             = std::remove_pointer_t<decltype([] {
        if constexpr (member_ports_detector::value) {
            return static_cast<typename member_ports_detector::member_ports *>(nullptr);
        } else {
            return static_cast<typename meta::concat<std::conditional_t<gr::meta::is_typelist_v<Ports>, Ports, meta::typelist<Ports>>...> *>(nullptr);
        }
    }())>;

    using input_ports           = typename all_ports ::template filter<detail::is_input_port_or_collection>;
    using output_ports          = typename all_ports ::template filter<detail::is_output_port_or_collection>;

    using input_port_types      = typename input_ports ::template transform<port::type>;
    using output_port_types     = typename output_ports ::template transform<port::type>;
};

// clang-format off
template<typename TBlock,
         typename Derived = typename TBlock::derived_t,
         typename ArgumentList = typename TBlock::block_template_parameters>
using fixedBlock_ports_data =
    typename ArgumentList::template filter<port::has_fixed_info_or_is_typelist>
                         ::template prepend<TBlock>
                         ::template apply<fixedBlock_ports_data_helper>;
// clang-format on

template<typename TBlock>
using all_ports = typename fixedBlock_ports_data<TBlock>::all_ports;

template<typename TBlock>
using input_ports = typename fixedBlock_ports_data<TBlock>::input_ports;

template<typename TBlock>
using output_ports = typename fixedBlock_ports_data<TBlock>::output_ports;

template<typename TBlock>
using input_port_types = typename fixedBlock_ports_data<TBlock>::input_port_types;

template<typename TBlock>
using output_port_types = typename fixedBlock_ports_data<TBlock>::output_port_types;

template<typename TBlock>
using input_port_types_tuple = typename input_port_types<TBlock>::tuple_type;

template<typename TBlock>
using return_type = typename output_port_types<TBlock>::tuple_or_type;

template<typename TBlock>
using input_port_names = typename input_ports<TBlock>::template transform<detail::port_name>;

template<typename TBlock>
using output_port_names = typename output_ports<TBlock>::template transform<detail::port_name>;

template<typename TBlock>
constexpr bool block_defines_ports_as_member_variables = fixedBlock_ports_data<TBlock>::member_ports_detector::value;

template<typename TBlock, typename PortType>
using get_port_member_descriptor = typename meta::to_typelist<refl::descriptor::member_list<TBlock>>::template filter<detail::is_port_or_collection_descriptor>::template filter<
        detail::member_descriptor_has_type<PortType>::template matches>::template at<0>;

// TODO: Why is this not done with requires?
namespace detail {
template<std::size_t... Is>
auto
can_processOne_invoke_test(auto &node, const auto &input, std::index_sequence<Is...>) -> decltype(node.processOne(std::get<Is>(input)...));

template<typename T>
struct exact_argument_type {
    template<std::same_as<T> U>
    constexpr
    operator U() const noexcept;
};

template<std::size_t... Is>
auto
can_processOne_with_offset_invoke_test(auto &node, const auto &input, std::index_sequence<Is...>) -> decltype(node.processOne(exact_argument_type<std::size_t>(), std::get<Is>(input)...));

template<typename TBlock>
using simd_return_type_of_can_processOne = meta::simdize<return_type<TBlock>, meta::simdize_size_v<meta::simdize<input_port_types_tuple<TBlock>>>>;
} // namespace detail

/* A node "can process simd" if its `processOne` function takes at least one argument and all
 * arguments can be simdized types of the actual port data types.
 *
 * The node can be a sink (no output ports).
 * The requirement of at least one function argument disallows sources.
 *
 * There is another (unnamed) concept for source nodes: Source nodes can implement
 * `processOne_simd(integral_constant)`, which returns SIMD object(s) of width N.
 */
template<typename TBlock>
concept can_processOne_simd =
#if DISABLE_SIMD
        false;
#else
        traits::block::input_ports<TBlock>::template all_of<port::is_port> and // checks we don't have port collections inside
        traits::block::input_port_types<TBlock>::size() > 0 and requires(TBlock &node, const meta::simdize<input_port_types_tuple<TBlock>> &input_simds) {
            {
                detail::can_processOne_invoke_test(node, input_simds, std::make_index_sequence<traits::block::input_ports<TBlock>::size()>())
            } -> std::same_as<detail::simd_return_type_of_can_processOne<TBlock>>;
        };
#endif

template<typename TBlock>
concept can_processOne_simd_with_offset =
#if DISABLE_SIMD
        false;
#else
        traits::block::input_ports<TBlock>::template all_of<port::is_port> and // checks we don't have port collections inside
        traits::block::input_port_types<TBlock>::size() > 0 && requires(TBlock &node, const meta::simdize<input_port_types_tuple<TBlock>> &input_simds) {
            {
                detail::can_processOne_with_offset_invoke_test(node, input_simds, std::make_index_sequence<traits::block::input_ports<TBlock>::size()>())
            } -> std::same_as<detail::simd_return_type_of_can_processOne<TBlock>>;
        };
#endif

template<typename TBlock>
concept can_processOne_scalar = requires(TBlock &node, const input_port_types_tuple<TBlock> &inputs) {
    { detail::can_processOne_invoke_test(node, inputs, std::make_index_sequence<traits::block::input_ports<TBlock>::size()>()) } -> std::same_as<return_type<TBlock>>;
};

template<typename TBlock>
concept can_processOne_scalar_with_offset = requires(TBlock &node, const input_port_types_tuple<TBlock> &inputs) {
    { detail::can_processOne_with_offset_invoke_test(node, inputs, std::make_index_sequence<traits::block::input_ports<TBlock>::size()>()) } -> std::same_as<return_type<TBlock>>;
};

template<typename TBlock>
concept can_processOne = can_processOne_scalar<TBlock> or can_processOne_simd<TBlock> or can_processOne_scalar_with_offset<TBlock> or can_processOne_simd_with_offset<TBlock>;

template<typename TBlock>
concept can_processOne_with_offset = can_processOne_scalar_with_offset<TBlock> or can_processOne_simd_with_offset<TBlock>;

namespace detail {
template<typename T>
struct dummy_input_span : std::span<const T> {           // NOSONAR
    dummy_input_span(const dummy_input_span &) = delete; // NOSONAR
    dummy_input_span(dummy_input_span &&) noexcept;      // NOSONAR
    constexpr void consume(std::size_t) noexcept;
};

template<typename T>
struct dummy_output_span : std::span<T> {                  // NOSONAR
    dummy_output_span(const dummy_output_span &) = delete; // NOSONAR
    dummy_output_span(dummy_output_span &&) noexcept;      // NOSONAR
    constexpr void publish(std::size_t) noexcept;
};

struct to_any_vector {
    template<typename Any>
    operator std::vector<Any>() const {
        return {};
    } // NOSONAR

    template<typename Any>
    operator std::vector<Any> &() const {
        return {};
    } // NOSONAR
};

struct to_any_pointer {
    template<typename Any>
    operator Any *() const {
        return {};
    } // NOSONAR
};

template<typename T>
struct dummy_reader {
    using type = to_any_pointer;
};

template<typename T>
struct dummy_writer {
    using type = to_any_pointer;
};

template<typename Port>
constexpr auto *
port_to_processBulk_argument_helper() {
    if constexpr (requires(Port p) {
                      typename Port::value_type;
                      p.cbegin() != p.cend();
                  }) {
        return static_cast<to_any_vector *>(nullptr);

    } else if constexpr (Port::synchronous) {
        if constexpr (Port::IS_INPUT) {
            return static_cast<dummy_input_span<typename Port::value_type> *>(nullptr);
        } else if constexpr (Port::IS_OUTPUT) {
            return static_cast<dummy_output_span<typename Port::value_type> *>(nullptr);
        }
    } else {
        if constexpr (Port::IS_INPUT) {
            return static_cast<to_any_pointer *>(nullptr);
        } else if constexpr (Port::IS_OUTPUT) {
            return static_cast<to_any_pointer *>(nullptr);
        }
    }
}

template<typename Port>
struct port_to_processBulk_argument {
    using type = std::remove_pointer_t<decltype(port_to_processBulk_argument_helper<Port>())>;
};

template<typename>
struct nothing_you_ever_wanted {};

// This alias template is only necessary as a workaround for a bug in Clang. Instead of passing dynamic_span to transform_conditional below, C++ allows passing std::span directly.
template<typename T>
using dynamic_span = std::span<T>;

template<std::size_t... InIdx, std::size_t... OutIdx>
auto
can_processBulk_invoke_test(auto &node, const auto &inputs, auto &outputs, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>)
        -> decltype(node.processBulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(outputs)...));
} // namespace detail

template<typename TBlock>
concept can_processBulk = requires(TBlock &n, typename meta::transform_types_nested<detail::port_to_processBulk_argument, traits::block::input_ports<TBlock>>::tuple_type inputs,
                                    typename meta::transform_types_nested<detail::port_to_processBulk_argument, traits::block::output_ports<TBlock>>::tuple_type outputs) {
    {
        detail::can_processBulk_invoke_test(n, inputs, outputs, std::make_index_sequence<input_port_types<TBlock>::size>(), std::make_index_sequence<output_port_types<TBlock>::size>())
    } -> std::same_as<WorkReturnStatus>;
};

/*
 * Satisfied if `Derived` has a member function `processBulk` which can be invoked with a number of arguments matching the number of input and output ports. Input arguments must accept either a
 * std::span<const T> or any type satisfying ConsumableSpan<T>. Output arguments must accept either a std::span<T> or any type satisfying PublishableSpan<T>, except for the I-th output argument, which
 * must be std::span<T> and *not* a type satisfying PublishableSpan<T>.
 */
template<typename Derived, std::size_t I>
concept processBulk_requires_ith_output_as_span = requires(Derived                                                                                                                      &d,
                                                            typename meta::transform_types<detail::dummy_input_span, traits::block::input_port_types<Derived>>::template apply<std::tuple> inputs,
                                                            typename meta::transform_conditional<decltype([](auto j) { return j == I; }), detail::dynamic_span, detail::dummy_output_span,
                                                                                                 traits::block::output_port_types<Derived>>::template apply<std::tuple>
                                                                    outputs,
                                                            typename meta::transform_conditional<decltype([](auto j) { return j == I; }), detail::nothing_you_ever_wanted, detail::dummy_output_span,
                                                                                                 traits::block::output_port_types<Derived>>::template apply<std::tuple>
                                                                    bad_outputs) {
    {
        []<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>,
                                                        std::index_sequence<OutIdx...>) -> decltype(d.processBulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(outputs)...)) {
            return {};
        }(std::make_index_sequence<traits::block::input_port_types<Derived>::size>(), std::make_index_sequence<traits::block::output_port_types<Derived>::size>())
    } -> std::same_as<WorkReturnStatus>;
    not requires {
        []<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>,
                                                        std::index_sequence<OutIdx...>) -> decltype(d.processBulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(bad_outputs)...)) {
            return {};
        }(std::make_index_sequence<traits::block::input_port_types<Derived>::size>(), std::make_index_sequence<traits::block::output_port_types<Derived>::size>());
    };
};

} // namespace gr::traits::block

#endif // include guard
