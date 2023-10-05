#ifndef GNURADIO_BLOCK_BLOCK_TRAITS_HPP
#define GNURADIO_BLOCK_BLOCK_TRAITS_HPP

#include <reflection.hpp>

#include <port.hpp>        // localinclude
#include <port_traits.hpp> // localinclude
#include <utils.hpp>       // localinclude

#include <vir/simd.h>

namespace fair::graph {
enum class work_return_status_t;
}

namespace fair::graph::traits::block {

namespace detail {
template<typename FieldDescriptor>
using member_type = typename FieldDescriptor::value_type;

template<typename Type>
auto
unwrap_port_helper() {
    if constexpr (port::is_port_v<Type>) {
        return static_cast<Type *>(nullptr);
    } else if constexpr (port::is_port_collection_v<Type>) {
        return static_cast<typename Type::value_type *>(nullptr);
    } else {
        static_assert(meta::always_false<Type>, "Not a port or a collection of ports");
    }
}

template<typename Type>
using unwrap_port = std::remove_pointer_t<decltype(unwrap_port_helper<Type>())>;

template<typename Type>
using is_port_or_collection = std::integral_constant<bool, port::is_port_v<Type> || port::is_port_collection_v<Type>>;

template<typename Type>
using is_input_port_or_collection = std::integral_constant<bool, is_port_or_collection<Type>() && port::is_input_v<unwrap_port<Type>>>;

template<typename Type>
using is_output_port_or_collection = std::integral_constant<bool, is_port_or_collection<Type>() && port::is_output_v<unwrap_port<Type>>>;

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

template<typename Block>
struct member_ports_detector {
    static constexpr bool value = false;
};

template<class T, typename ValueType = std::remove_cvref_t<T>>
concept Reflectable = refl::is_reflectable<ValueType>();

template<Reflectable Block>
struct member_ports_detector<Block> {
    using member_ports          = typename meta::to_typelist<refl::descriptor::member_list<Block>>::template filter<is_port_or_collection_descriptor>::template transform<member_to_named_port>;

    static constexpr bool value = member_ports::size != 0;
};

template<typename Block>
using port_name = typename Block::static_name();

template<typename RequestedType>
struct member_descriptor_has_type {
    template<typename Descriptor>
    using matches = std::is_same<RequestedType, member_to_named_port<Descriptor>>;
};

} // namespace detail

template<typename...>
struct fixed_block_ports_data_helper;

// This specialization defines block attributes when the block is created
// with two type lists - one list for input and one for output ports
template<typename Block, meta::is_typelist_v InputPorts, meta::is_typelist_v OutputPorts>
    requires InputPorts::template
all_of<port::has_fixed_info> &&OutputPorts::template all_of<port::has_fixed_info> struct fixed_block_ports_data_helper<Block, InputPorts, OutputPorts> {
    using member_ports_detector = std::false_type;

    // using member_ports_detector = detail::member_ports_detector<Block>;

    using input_ports       = InputPorts;
    using output_ports      = OutputPorts;

    using input_port_types  = typename input_ports ::template transform<port::type>;
    using output_port_types = typename output_ports ::template transform<port::type>;

    using all_ports         = meta::concat<input_ports, output_ports>;
};

// This specialization defines block attributes when the block is created
// with a list of ports as template arguments
template<typename Block, port::has_fixed_info_v... Ports>
struct fixed_block_ports_data_helper<Block, Ports...> {
    using member_ports_detector = detail::member_ports_detector<Block>;

    using all_ports             = std::remove_pointer_t<decltype([] {
        if constexpr (member_ports_detector::value) {
            return static_cast<typename member_ports_detector::member_ports *>(nullptr);
        } else {
            return static_cast<typename meta::concat<std::conditional_t<fair::meta::is_typelist_v<Ports>, Ports, meta::typelist<Ports>>...> *>(nullptr);
        }
    }())>;

    using input_ports           = typename all_ports ::template filter<detail::is_input_port_or_collection>;
    using output_ports          = typename all_ports ::template filter<detail::is_output_port_or_collection>;

    using input_port_types      = typename input_ports ::template transform<port::type>;
    using output_port_types     = typename output_ports ::template transform<port::type>;
};

// clang-format off
template<typename Block,
         typename Derived = typename Block::derived_t,
         typename ArgumentList = typename Block::block_template_parameters>
using fixed_block_ports_data =
    typename ArgumentList::template filter<port::has_fixed_info_or_is_typelist>
                         ::template prepend<Block>
                         ::template apply<fixed_block_ports_data_helper>;
// clang-format on

template<typename Block>
using all_ports = typename fixed_block_ports_data<Block>::all_ports;

template<typename Block>
using input_ports = typename fixed_block_ports_data<Block>::input_ports;

template<typename Block>
using output_ports = typename fixed_block_ports_data<Block>::output_ports;

template<typename Block>
using input_port_types = typename fixed_block_ports_data<Block>::input_port_types;

template<typename Block>
using output_port_types = typename fixed_block_ports_data<Block>::output_port_types;

template<typename Block>
using input_port_types_tuple = typename input_port_types<Block>::tuple_type;

template<typename Block>
using return_type = typename output_port_types<Block>::tuple_or_type;

template<typename Block>
using input_port_names = typename input_ports<Block>::template transform<detail::port_name>;

template<typename Block>
using output_port_names = typename output_ports<Block>::template transform<detail::port_name>;

template<typename Block>
constexpr bool block_defines_ports_as_member_variables = fixed_block_ports_data<Block>::member_ports_detector::value;

template<typename Block, typename PortType>
using get_port_member_descriptor = typename meta::to_typelist<refl::descriptor::member_list<Block>>::template filter<detail::is_port_or_collection_descriptor>::template filter<
        detail::member_descriptor_has_type<PortType>::template matches>::template at<0>;

// TODO: Why is this not done with requires?
namespace detail {
template<std::size_t... Is>
auto
can_process_one_invoke_test(auto &block, const auto &input, std::index_sequence<Is...>) -> decltype(block.process_one(std::get<Is>(input)...));

template<typename T>
struct exact_argument_type {
    template<std::same_as<T> U>
    constexpr operator U() const noexcept;
};

template<std::size_t... Is>
auto
can_process_one_with_offset_invoke_test(auto &block, const auto &input, std::index_sequence<Is...>) -> decltype(block.process_one(exact_argument_type<std::size_t>(), std::get<Is>(input)...));

template<typename Block>
using simd_return_type_of_can_process_one = meta::simdize<return_type<Block>, meta::simdize_size_v<meta::simdize<input_port_types_tuple<Block>>>>;
} // namespace detail

/* A block "can process simd" if its `process_one` function takes at least one argument and all
 * arguments can be simdized types of the actual port data types.
 *
 * The block can be a sink (no output ports).
 * The requirement of at least one function argument disallows sources.
 *
 * There is another (unnamed) concept for source blocks: Source blocks can implement
 * `process_one_simd(integral_constant)`, which returns SIMD object(s) of width N.
 */
template<typename Block>
concept can_process_one_simd =
#if DISABLE_SIMD
        false;
#else
        traits::block::input_ports<Block>::template all_of<port::is_port> and // checks we don't have port collections inside
        traits::block::input_port_types<Block>::size() > 0 and requires(Block &block, const meta::simdize<input_port_types_tuple<Block>> &input_simds) {
            {
                detail::can_process_one_invoke_test(block, input_simds, std::make_index_sequence<traits::block::input_ports<Block>::size()>())
            } -> std::same_as<detail::simd_return_type_of_can_process_one<Block>>;
        };
#endif

template<typename Block>
concept can_process_one_simd_with_offset =
#if DISABLE_SIMD
        false;
#else
        traits::block::input_ports<Block>::template all_of<port::is_port> and // checks we don't have port collections inside
        traits::block::input_port_types<Block>::size() > 0 && requires(Block &block, const meta::simdize<input_port_types_tuple<Block>> &input_simds) {
            {
                detail::can_process_one_with_offset_invoke_test(block, input_simds, std::make_index_sequence<traits::block::input_ports<Block>::size()>())
            } -> std::same_as<detail::simd_return_type_of_can_process_one<Block>>;
        };
#endif

template<typename Block>
concept can_process_one_scalar = requires(Block &block, const input_port_types_tuple<Block> &inputs) {
    { detail::can_process_one_invoke_test(block, inputs, std::make_index_sequence<traits::block::input_ports<Block>::size()>()) } -> std::same_as<return_type<Block>>;
};

template<typename Block>
concept can_process_one_scalar_with_offset = requires(Block &block, const input_port_types_tuple<Block> &inputs) {
    { detail::can_process_one_with_offset_invoke_test(block, inputs, std::make_index_sequence<traits::block::input_ports<Block>::size()>()) } -> std::same_as<return_type<Block>>;
};

template<typename Block>
concept can_process_one = can_process_one_scalar<Block> or can_process_one_simd<Block> or can_process_one_scalar_with_offset<Block> or can_process_one_simd_with_offset<Block>;

template<typename Block>
concept can_process_one_with_offset = can_process_one_scalar_with_offset<Block> or can_process_one_simd_with_offset<Block>;

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
port_to_process_bulk_argument_helper() {
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
struct port_to_process_bulk_argument {
    using type = std::remove_pointer_t<decltype(port_to_process_bulk_argument_helper<Port>())>;
};

template<typename>
struct nothing_you_ever_wanted {};

// This alias template is only necessary as a workaround for a bug in Clang. Instead of passing dynamic_span to transform_conditional below, C++ allows passing std::span directly.
template<typename T>
using dynamic_span = std::span<T>;

template<std::size_t... InIdx, std::size_t... OutIdx>
auto
can_process_bulk_invoke_test(auto &block, const auto &inputs, auto &outputs, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>)
        -> decltype(block.process_bulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(outputs)...));
} // namespace detail

template<typename Block>
concept can_process_bulk = requires(Block &n, typename meta::transform_types_nested<detail::port_to_process_bulk_argument, traits::block::input_ports<Block>>::tuple_type inputs,
                                    typename meta::transform_types_nested<detail::port_to_process_bulk_argument, traits::block::output_ports<Block>>::tuple_type outputs) {
    {
        detail::can_process_bulk_invoke_test(n, inputs, outputs, std::make_index_sequence<input_port_types<Block>::size>(), std::make_index_sequence<output_port_types<Block>::size>())
    } -> std::same_as<work_return_status_t>;
};

/*
 * Satisfied if `Derived` has a member function `process_bulk` which can be invoked with a number of arguments matching the number of input and output ports. Input arguments must accept either a
 * std::span<const T> or any type satisfying ConsumableSpan<T>. Output arguments must accept either a std::span<T> or any type satisfying PublishableSpan<T>, except for the I-th output argument, which
 * must be std::span<T> and *not* a type satisfying PublishableSpan<T>.
 */
template<typename Derived, std::size_t I>
concept process_bulk_requires_ith_output_as_span = requires(Derived                                                                                                                       &d,
                                                            typename meta::transform_types<detail::dummy_input_span, traits::block::input_port_types<Derived>>::template apply<std::tuple> inputs,
                                                            typename meta::transform_conditional<decltype([](auto j) { return j == I; }), detail::dynamic_span, detail::dummy_output_span,
                                                                                                 traits::block::output_port_types<Derived>>::template apply<std::tuple>
                                                                    outputs,
                                                            typename meta::transform_conditional<decltype([](auto j) { return j == I; }), detail::nothing_you_ever_wanted, detail::dummy_output_span,
                                                                                                 traits::block::output_port_types<Derived>>::template apply<std::tuple>
                                                                    bad_outputs) {
    {
        []<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>,
                                                        std::index_sequence<OutIdx...>) -> decltype(d.process_bulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(outputs)...)) {
            return {};
        }(std::make_index_sequence<traits::block::input_port_types<Derived>::size>(), std::make_index_sequence<traits::block::output_port_types<Derived>::size>())
    } -> std::same_as<work_return_status_t>;
    not requires {
        []<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>,
                                                        std::index_sequence<OutIdx...>) -> decltype(d.process_bulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(bad_outputs)...)) {
            return {};
        }(std::make_index_sequence<traits::block::input_port_types<Derived>::size>(), std::make_index_sequence<traits::block::output_port_types<Derived>::size>());
    };
};

} // namespace fair::graph::traits::block

#endif // include guard
