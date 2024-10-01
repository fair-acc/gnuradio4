#ifndef GNURADIO_NODE_NODE_TRAITS_HPP
#define GNURADIO_NODE_NODE_TRAITS_HPP

#include <gnuradio-4.0/meta/utils.hpp>

#include "Port.hpp"
#include "PortTraits.hpp"
#include "reflection.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <vir/simd.h>
#include <vir/simdize.h>
#pragma GCC diagnostic pop

namespace gr::work {
enum class Status;
}

namespace gr::traits::block {

namespace detail {

template<typename FieldDescriptor>
using member_type = typename FieldDescriptor::value_type;

template<typename Port>
constexpr bool is_port_descriptor_v = port::is_port_v<member_type<Port>>;

template<typename Collection>
constexpr bool is_port_collection_descriptor_v = port::is_port_collection_v<member_type<Collection>>;

template<typename Descriptor>
using is_port_or_collection_descriptor = std::integral_constant<bool, is_port_descriptor_v<Descriptor> || is_port_collection_descriptor_v<Descriptor>>;

template<typename Descriptor>
constexpr auto member_to_named_port_helper() {
    // Collections of ports don't get names inside the type as
    // the ports inside are dynamically created
    if constexpr (is_port_descriptor_v<Descriptor>) {
        return static_cast<typename Descriptor::value_type::template with_name_and_descriptor<fixed_string(refl::descriptor::get_name(Descriptor()).data), Descriptor>*>(nullptr);
    } else if constexpr (is_port_collection_descriptor_v<Descriptor>) {
        if constexpr (gr::meta::is_std_array_type<typename Descriptor::value_type>()) {
            auto value_type_updater = []<template<typename, auto> typename Template, typename Arg, auto Size>(Template<Arg, Size>*) {
                return static_cast< //
                    Template<typename Arg::template with_name_and_descriptor<fixed_string(refl::descriptor::get_name(Descriptor()).data), Descriptor>, Size>*>(nullptr);
            };
            return value_type_updater(static_cast<typename Descriptor::value_type*>(nullptr));
        } else {
            auto value_type_updater = []<template<typename...> typename Template, typename Arg, typename... Args>(Template<Arg, Args...>*) {
                // This type is not going to be used for a variable, it is just meant to be
                // a compile-time hint of what the port collection looks like.
                // We're ignoring the Args... because they might depend on the
                // main type (for example, allocator in a vector)
                return static_cast<Template<typename Arg::template with_name_and_descriptor<fixed_string(refl::descriptor::get_name(Descriptor()).data), Descriptor>>*>(nullptr);
            };
            return value_type_updater(static_cast<typename Descriptor::value_type*>(nullptr));
        }
    } else {
        return static_cast<void*>(nullptr);
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
    using member_ports = typename meta::to_typelist<refl::descriptor::member_list<TBlock>>::template filter<is_port_or_collection_descriptor>::template transform<member_to_named_port>;

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

// This specialization defines block attributes when the block is created
// with two type lists - one list for input and one for output ports
template<typename TBlock, meta::is_typelist_v InputPorts, meta::is_typelist_v OutputPorts>
requires(InputPorts::template all_of<port::has_fixed_info> && OutputPorts::template all_of<port::has_fixed_info>)
struct fixedBlock_ports_data_helper<TBlock, InputPorts, OutputPorts> {
    using member_ports_detector = std::false_type;

    using defined_input_ports  = InputPorts;
    using defined_output_ports = OutputPorts;

    template<gr::PortType portType>
    struct for_type {
        using input_ports  = typename defined_input_ports ::template filter<traits::port::kind::tester_for<portType>::template is_input_port_or_collection>;
        using output_ports = typename defined_output_ports ::template filter<traits::port::kind::tester_for<portType>::template is_output_port_or_collection>;
        using all_ports    = meta::concat<input_ports, output_ports>;

        using input_port_types  = typename input_ports ::template transform<port::type>;
        using output_port_types = typename output_ports ::template transform<port::type>;
    };

    using all     = for_type<PortType::ANY>;
    using stream  = for_type<PortType::STREAM>;
    using message = for_type<PortType::MESSAGE>;
};

// This specialization defines block attributes when the block is created
// with a list of ports as template arguments
template<typename TBlock, port::HasFixedInfo... Ports>
struct fixedBlock_ports_data_helper<TBlock, Ports...> {
    using member_ports_detector = detail::member_ports_detector<TBlock>;

    using all_ports = std::remove_pointer_t<decltype([] {
        if constexpr (member_ports_detector::value) {
            return static_cast<typename member_ports_detector::member_ports*>(nullptr);
        } else {
            return static_cast<typename meta::concat<std::conditional_t<gr::meta::is_typelist_v<Ports>, Ports, meta::typelist<Ports>>...>*>(nullptr);
        }
    }())>;

    template<PortType portType>
    struct for_type {
        using input_ports  = typename all_ports ::template filter<traits::port::kind::tester_for<portType>::template is_input_port_or_collection>;
        using output_ports = typename all_ports ::template filter<traits::port::kind::tester_for<portType>::template is_output_port_or_collection>;

        using input_port_types  = typename input_ports ::template transform<port::type>;
        using output_port_types = typename output_ports ::template transform<port::type>;
    };

    using all     = for_type<PortType::ANY>;
    using stream  = for_type<PortType::STREAM>;
    using message = for_type<PortType::MESSAGE>;
};

// clang-format off
template<typename TBlock,
         typename TDerived = typename TBlock::derived_t,
         typename ArgumentList = typename TBlock::block_template_parameters>
using fixedBlock_ports_data =
    typename ArgumentList::template filter<port::has_fixed_info_or_is_typelist>
                         ::template prepend<TBlock>
                         ::template apply<fixedBlock_ports_data_helper>;
// clang-format on

template<typename TBlock>
using ports_data = fixedBlock_ports_data<TBlock>;

template<typename TBlock>
using all_input_ports = typename fixedBlock_ports_data<TBlock>::all::input_ports;

template<typename TBlock>
using all_output_ports = typename fixedBlock_ports_data<TBlock>::all::output_ports;

template<typename TBlock>
using all_input_port_types = typename fixedBlock_ports_data<TBlock>::all::input_port_types;

template<typename TBlock>
using all_output_port_types = typename fixedBlock_ports_data<TBlock>::all::output_port_types;

template<typename TBlock>
using all_input_port_types_tuple = typename all_input_port_types<TBlock>::tuple_type;

template<typename TBlock>
using stream_input_ports = typename fixedBlock_ports_data<TBlock>::stream::input_ports;

template<typename TBlock>
using stream_output_ports = typename fixedBlock_ports_data<TBlock>::stream::output_ports;

template<typename TBlock>
using stream_input_port_types = typename fixedBlock_ports_data<TBlock>::stream::input_port_types;

template<typename TBlock>
using stream_output_port_types = typename fixedBlock_ports_data<TBlock>::stream::output_port_types;

template<typename TBlock>
using stream_input_port_types_tuple = typename stream_input_port_types<TBlock>::tuple_type;

template<typename TBlock>
using stream_return_type = typename fixedBlock_ports_data<TBlock>::stream::output_port_types::tuple_or_type;

template<typename TBlock>
using all_input_port_names = typename all_input_ports<TBlock>::template transform<detail::port_name>;

template<typename TBlock>
using all_output_port_names = typename all_output_ports<TBlock>::template transform<detail::port_name>;

template<typename TBlock>
constexpr bool block_defines_ports_as_member_variables = fixedBlock_ports_data<TBlock>::member_ports_detector::value;

template<typename TBlock, typename TPortType>
using get_port_member_descriptor = typename meta::to_typelist<refl::descriptor::member_list<TBlock>>::template filter<detail::is_port_or_collection_descriptor>::template filter<detail::member_descriptor_has_type<TPortType>::template matches>::template at<0>;

// TODO: Why is this not done with requires?
// mkretz: I don't understand the question. "this" in the question is unclear.
/* Helper to determine the return type of `block.processOne` for the given inputs.
 *
 * This helper is necessary because we need a pack of indices to expand the input tuples. In princple we should be able
 * to use std::apply to the same effect. Except that `block` would need to be the first element of the tuple. This here
 * is simpler and cheaper.
 */
namespace detail {
template<std::size_t... Is>
auto can_processOne_invoke_test(auto& block, const auto& input, std::index_sequence<Is...>) -> decltype(block.processOne(std::get<Is>(input)...));

template<typename TBlock>
using simd_return_type_of_can_processOne = vir::simdize<stream_return_type<TBlock>, vir::simdize<stream_input_port_types_tuple<TBlock>>::size()>;
} // namespace detail

/* A block "can process simd" if its `processOne` function takes at least one argument and all
 * arguments can be simdized types of the actual port data types.
 *
 * The block can be a sink (no output ports).
 * The requirement of at least one function argument disallows sources.
 *
 * There is another (unnamed) concept for source nodes: Source nodes can implement
 * `processOne_simd(integral_constant N)`, which returns SIMD object(s) of width N.
 */
template<typename TBlock>
concept can_processOne_simd =                                                     //
    traits::block::stream_input_ports<TBlock>::template all_of<port::is_port> and // checks we don't have port collections inside
    traits::block::stream_input_port_types<TBlock>::size() > 0 and requires(TBlock& block, const vir::simdize<stream_input_port_types_tuple<TBlock>>& input_simds) {
        { detail::can_processOne_invoke_test(block, input_simds, std::make_index_sequence<traits::block::stream_input_ports<TBlock>::size()>()) } -> std::same_as<detail::simd_return_type_of_can_processOne<TBlock>>;
    };

template<typename TBlock>
concept can_processOne_simd_const =                                               //
    traits::block::stream_input_ports<TBlock>::template all_of<port::is_port> and // checks we don't have port collections inside
    traits::block::stream_input_port_types<TBlock>::size() > 0 and requires(const TBlock& block, const vir::simdize<stream_input_port_types_tuple<TBlock>>& input_simds) {
        { detail::can_processOne_invoke_test(block, input_simds, std::make_index_sequence<traits::block::stream_input_ports<TBlock>::size()>()) } -> std::same_as<detail::simd_return_type_of_can_processOne<TBlock>>;
    };

template<typename TBlock>
concept can_processOne_scalar = requires(TBlock& block, const stream_input_port_types_tuple<TBlock>& inputs) {
    { detail::can_processOne_invoke_test(block, inputs, std::make_index_sequence<traits::block::stream_input_ports<TBlock>::size()>()) } -> std::same_as<stream_return_type<TBlock>>;
};

template<typename TBlock>
concept can_processOne_scalar_const = requires(const TBlock& block, const stream_input_port_types_tuple<TBlock>& inputs) {
    { detail::can_processOne_invoke_test(block, inputs, std::make_index_sequence<traits::block::stream_input_ports<TBlock>::size()>()) } -> std::same_as<stream_return_type<TBlock>>;
};

template<typename TBlock>
concept can_processOne = can_processOne_scalar<TBlock> or can_processOne_simd<TBlock>;

template<typename TBlock>
concept can_processOne_const = can_processOne_scalar_const<TBlock> or can_processOne_simd_const<TBlock>;

template<typename TBlock, typename TPort>
concept can_processMessagesForPortConsumableSpan = requires(TBlock& block, TPort& inPort) { block.processMessages(inPort, inPort.streamReader().get(1UZ)); };

template<typename TBlock, typename TPort>
concept can_processMessagesForPortStdSpan = requires(TBlock& block, TPort& inPort, std::span<const Message> msgSpan) { block.processMessages(inPort, msgSpan); };

// clang-format off
namespace detail {

template<typename T>
struct DummyConsumableSpan {
    using value_type = typename std::remove_cv_t<T>;
    using iterator = typename std::span<const T>::iterator;

private:
    std::span<const T> internalSpan; // Internal span, used for fake implementation

public:
    DummyConsumableSpan() = default;
    DummyConsumableSpan(const DummyConsumableSpan& other) = default;
    DummyConsumableSpan& operator=(const DummyConsumableSpan& other) = default;
    DummyConsumableSpan(DummyConsumableSpan&& other) noexcept = default;
    DummyConsumableSpan& operator=(DummyConsumableSpan&& other) noexcept = default;
    ~DummyConsumableSpan() = default;

    [[nodiscard]] constexpr iterator begin() const noexcept { return internalSpan.begin(); }
    [[nodiscard]] constexpr iterator end() const noexcept { return internalSpan.end(); }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return internalSpan.size(); }
    operator const std::span<const T>&() const noexcept { return internalSpan; }
    operator std::span<const T>&() noexcept  { return internalSpan; }
    operator std::span<const T>&&() = delete;

    [[nodiscard]] bool consume(std::size_t /* nSamples */) const noexcept { return true; }
};
static_assert(ConsumableSpan<DummyConsumableSpan<int>>);

template<typename T>
struct DummyInputSpan: public DummyConsumableSpan<T> {
    DummyConsumableSpan<gr::Tag> rawTags{};
    auto tags() { return std::views::empty<std::pair<Tag::signed_index_type, const property_map&>>; }
    [[nodiscard]] inline Tag getMergedTag(gr::Tag::signed_index_type /*untilLocalIndex*/) const { return {}; }
    void consumeTags(gr::Tag::signed_index_type /*untilLocalIndex*/) { }
};
static_assert(ConsumableSpan<DummyInputSpan<int>>);
static_assert(InputSpan<DummyInputSpan<int>>);

template<typename T>
struct DummyPublishableSpan {
    using value_type = typename std::remove_cv_t<T>;
    using iterator = typename std::span<T>::iterator;

private:
    std::span<T> internalSpan; // Internal span, used for fake implementation

public:
    DummyPublishableSpan() = default;
    DummyPublishableSpan(const DummyPublishableSpan& other) = delete;
    DummyPublishableSpan& operator=(const DummyPublishableSpan& other) = delete;
    DummyPublishableSpan(DummyPublishableSpan&& other) noexcept = default;
    DummyPublishableSpan& operator=(DummyPublishableSpan&& other) noexcept = default;
    ~DummyPublishableSpan() = default;

    [[nodiscard]] constexpr iterator begin() const noexcept { return internalSpan.begin(); }
    [[nodiscard]] constexpr iterator end() const noexcept { return internalSpan.end(); }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return internalSpan.size(); }
    operator const std::span<T>&() const noexcept { return internalSpan; }
    operator std::span<T>&() noexcept  { return internalSpan; }

    constexpr void publish(std::size_t) noexcept {}
};
static_assert(PublishableSpan<DummyPublishableSpan<int>>);

template<typename T>
struct DummyPublishablePortSpan: public DummyPublishableSpan<T> {
    DummyPublishableSpan<gr::Tag> tags{};

    void publishTag(property_map&, gr::Tag::signed_index_type) {}
};
static_assert(PublishablePortSpan<DummyPublishablePortSpan<int>>);

// clang-format on

struct to_any_vector {
    template<typename Any>
    operator std::vector<Any>() const {
        return {};
    } // NOSONAR

    template<typename Any>
    operator std::vector<Any>&() const {
        return {};
    } // NOSONAR
};

struct to_any_pointer {
    template<typename Any>
    operator Any*() const {
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

template<typename Port, bool isVectorOfSpansReturned>
constexpr auto* port_to_processBulk_argument_helper() {
    if constexpr (requires(Port p) { // array of ports
                      typename Port::value_type;
                      p.cbegin() != p.cend();
                  }) {
        if constexpr (Port::value_type::kIsInput) {
            if constexpr (isVectorOfSpansReturned) {
                return static_cast<std::span<std::span<const typename Port::value_type::value_type>>*>(nullptr);
            } else {
                return static_cast<std::span<DummyInputSpan<const typename Port::value_type::value_type>>*>(nullptr);
            }
        } else if constexpr (Port::value_type::kIsOutput) {
            if constexpr (isVectorOfSpansReturned) {
                return static_cast<std::span<std::span<typename Port::value_type::value_type>>*>(nullptr);
            } else {
                return static_cast<std::span<DummyPublishablePortSpan<typename Port::value_type::value_type>>*>(nullptr);
            }
        }

    } else { // single port
        if constexpr (Port::kIsInput) {
            return static_cast<DummyInputSpan<const typename Port::value_type>*>(nullptr);
        } else if constexpr (Port::kIsOutput) {
            return static_cast<DummyPublishablePortSpan<typename Port::value_type>*>(nullptr);
        }
    }
}

template<typename Port>
struct port_to_processBulk_argument_consumable_publishable {
    using type = std::remove_pointer_t<decltype(port_to_processBulk_argument_helper<Port, false>())>;
};

template<typename Port>
struct port_to_processBulk_argument_std_span {
    using type = std::remove_pointer_t<decltype(port_to_processBulk_argument_helper<Port, true>())>;
};

template<typename>
struct nothing_you_ever_wanted {};

// This alias template is only necessary as a workaround for a bug in Clang. Instead of passing dynamic_span to transform_conditional below, C++ allows passing std::span directly.
template<typename T>
using dynamic_span = std::span<T>;

template<std::size_t... InIdx, std::size_t... OutIdx>
auto can_processBulk_invoke_test(auto& block, const auto& inputs, auto& outputs, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) -> decltype(block.processBulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(outputs)...));
} // namespace detail

template<typename TBlock, template<typename> typename TArguments>
concept can_processBulk_helper = requires(TBlock& n, typename meta::transform_types_nested<TArguments, traits::block::stream_input_ports<TBlock>>::tuple_type inputs, typename meta::transform_types_nested<TArguments, traits::block::stream_output_ports<TBlock>>::tuple_type outputs) {
    { detail::can_processBulk_invoke_test(n, inputs, outputs, std::make_index_sequence<stream_input_port_types<TBlock>::size>(), std::make_index_sequence<stream_output_port_types<TBlock>::size>()) } -> std::same_as<work::Status>;
};

template<typename TBlock>
concept can_processBulk = can_processBulk_helper<TBlock, detail::port_to_processBulk_argument_consumable_publishable> || can_processBulk_helper<TBlock, detail::port_to_processBulk_argument_std_span>;

/**
 * Satisfied if `TDerived` has a member function `processBulk` which can be invoked with a number of arguments matching the number of input and output ports. Input arguments must accept either a
 * std::span<const T> or any type satisfying ConsumableSpan<T>. Output arguments must accept either a std::span<T> or any type satisfying PublishableSpan<T>, except for the I-th output argument, which
 * must be std::span<T> and *not* a type satisfying PublishableSpan<T>.
 */
template<typename TDerived, std::size_t I>
concept processBulk_requires_ith_output_as_span = can_processBulk<TDerived> && (I < traits::block::stream_output_port_types<TDerived>::size) && (I >= 0) && requires(TDerived& d, typename meta::transform_types<detail::DummyInputSpan, traits::block::stream_input_port_types < TDerived>>::template apply<std::tuple> inputs, typename meta::transform_conditional<decltype([](auto j) { return j == I; }), detail::dynamic_span, detail::DummyPublishablePortSpan, traits::block::stream_output_port_types < TDerived>>::template apply<std::tuple> outputs, typename meta::transform_conditional<decltype([](auto j) { return j == I; }), detail::nothing_you_ever_wanted, detail::DummyPublishablePortSpan, traits::block::stream_output_port_types < TDerived>>::template apply<std::tuple> bad_outputs) {
    { detail::can_processBulk_invoke_test(d, inputs, outputs, std::make_index_sequence<stream_input_port_types<TDerived>::size>(), std::make_index_sequence<stream_output_port_types<TDerived>::size>()) } -> std::same_as<work::Status>;
    // TODO: Is this check redundant?
    not requires { []<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) -> decltype(d.processBulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(bad_outputs)...)) { return {}; }(std::make_index_sequence<traits::block::stream_input_port_types<TDerived>::size>(), std::make_index_sequence<traits::block::stream_output_port_types<TDerived>::size>()); };
};

} // namespace gr::traits::block

#endif // include guard
