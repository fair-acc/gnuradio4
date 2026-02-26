#ifndef GNURADIO_NODE_NODE_TRAITS_HPP
#define GNURADIO_NODE_NODE_TRAITS_HPP

#include <array>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include "Port.hpp"
#include "PortTraits.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <vir/simd.h>
#include <vir/simdize.h>
#pragma GCC diagnostic pop

namespace gr::work {
enum class Status;
}

namespace gr {
template<typename T>
concept PortReflectable = refl::reflectable<T> and std::same_as<std::remove_const_t<T>, typename T::derived_t>;

namespace work {
enum class Status;
struct Result;
} // namespace work
struct SettingsBase;

template<typename T>
concept HasWork = requires(T t, std::size_t requested_work) {
    { t.work(requested_work) } -> std::same_as<work::Result>;
};

template<typename T>
concept BlockLike = requires(T t, std::size_t requested_work) {
    { t.unique_name } -> std::convertible_to<const std::string&>;
    { unwrap_if_wrapped_t<decltype(t.name)>{} } -> std::same_as<std::string>;
    { unwrap_if_wrapped_t<decltype(t.meta_information)>{} } -> std::same_as<property_map>;
    { t.description } noexcept -> std::same_as<const std::string_view&>;

    { t.isBlocking() } noexcept -> std::same_as<bool>;

    { t.settings() } -> std::same_as<SettingsBase&>;

    // N.B. TODO discuss these requirements
    requires !std::is_copy_constructible_v<T>;
    requires !std::is_copy_assignable_v<T>;
} && HasWork<T>;

template<BlockLike Left, std::size_t OutId, BlockLike Right, std::size_t InId>
class MergeByIndex;

template<BlockLike Forward, std::size_t ForwardOutputPortIndex, //
    BlockLike Feedback, std::size_t FeedbackOutputPortIndex,    //
    std::size_t ForwardFeedbackInputPortIndex>
class FeedbackMergeByIndex;

} // namespace gr

namespace gr::traits::block {

namespace detail {

template<typename TPortDescr>
using port_name = typename TPortDescr::NameT;

template<typename T>
struct array_traits {
    static constexpr bool is_array = false;
};

template<typename T, std::size_t Size>
struct array_traits<std::array<T, Size>> {
    static constexpr bool        is_array = true;
    static constexpr std::size_t size     = Size;
};

// see also MergeByIndex partial specialization below
template<PortReflectable TBlock>
struct all_port_descriptors_impl {
    using type = refl::make_typelist_from_index_sequence<std::make_index_sequence<refl::data_member_count<TBlock>>, //
        [](auto MemberIdx) consteval {
            using T                           = refl::data_member_type<TBlock, MemberIdx>;
            constexpr meta::fixed_string name = refl::data_member_name<TBlock, MemberIdx>;
            if constexpr (port::is_port<T>::value) {
                // Port -> PortDescriptor
                return meta::typelist<typename T::template make_port_descriptor<name, gr::detail::SinglePort, /*unused*/ 0, MemberIdx>>{};
            } else if constexpr (port::is_port_tuple<T>::value) {
                // tuple<Ports...> -> typelist<PortDescriptor...>
                return [=]<size_t... Is>(std::index_sequence<Is...>) { return meta::typelist<typename std::tuple_element_t<Is, T>::template make_port_descriptor<name + meta::fixed_string_from_number<Is>, gr::detail::TupleOfPorts, /* index in tuple */ Is, MemberIdx>...>{}; }(std::make_index_sequence<std::tuple_size_v<T>>());
            } else if constexpr (port::is_port_collection<T>::value) {
                using traits = array_traits<T>;
                if constexpr (traits::is_array) {
                    // vector<Port> -> PortDescriptor|DynamicPortCollection
                    // array<Port, N> -> typelist<PortDescriptor, PortDescriptor, ...>
                    return meta::typelist<typename T::value_type::template make_port_descriptor<name, gr::detail::StaticPortCollection, /* size for array */ traits::size, MemberIdx>>{};
                } else {
                    return meta::typelist<typename T::value_type::template make_port_descriptor<name, gr::detail::DynamicPortCollection, /* not used */ 0, MemberIdx>>{};
                }
            } else {
                // not a Port, nothing to add to the resulting typelist
                return meta::typelist<>{};
            }
        }>;
};

// This partial specialization could be generalized into a customization point. But we probably want to think of a
// better name than 'AllPorts' for triggering that customization.
template<refl::reflectable Left, refl::reflectable Right, size_t OutId, size_t InId>
struct all_port_descriptors_impl<gr::MergeByIndex<Left, OutId, Right, InId>> {
    using type = gr::MergeByIndex<Left, OutId, Right, InId>::AllPorts;
};

// This partial specialization could be generalized into a customization point. But we probably want to think of a
// better name than 'AllPorts' for triggering that customization.
template<refl::reflectable Forward, std::size_t ForwardOutputPortIndex, //
    refl::reflectable Feedback, std::size_t FeedbackOutputPortIndex,    //
    std::size_t ForwardFeedbackInputPortIndex>
struct all_port_descriptors_impl<gr::FeedbackMergeByIndex<Forward, ForwardOutputPortIndex, Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex>> {
    using type = gr::FeedbackMergeByIndex<Forward, ForwardOutputPortIndex, Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex>::AllPorts;
};
} // namespace detail

template<PortReflectable TBlock>
using all_port_descriptors = typename detail::all_port_descriptors_impl<TBlock>::type;

template<PortReflectable TBlock, PortType portType>
using input_port_descriptors = typename all_port_descriptors<TBlock>::template filter<port::is_input_port, port::is_port_type<portType>::template eval>;

template<PortReflectable TBlock, PortType portType>
using output_port_descriptors = typename all_port_descriptors<TBlock>::template filter<port::is_output_port, port::is_port_type<portType>::template eval>;

template<PortReflectable TBlock>
using all_input_ports = typename all_port_descriptors<TBlock>::template filter<port::is_input_port>;

template<PortReflectable TBlock>
using all_output_ports = typename all_port_descriptors<TBlock>::template filter<port::is_output_port>;

template<PortReflectable TBlock>
using all_input_port_types = typename all_input_ports<TBlock>::template transform<port::type>;

template<PortReflectable TBlock>
using all_output_port_types = typename all_output_ports<TBlock>::template transform<port::type>;

template<PortReflectable TBlock>
using all_input_port_types_tuple = typename all_input_port_types<TBlock>::tuple_type;

template<PortReflectable TBlock>
using stream_input_ports = input_port_descriptors<TBlock, PortType::STREAM>;

template<PortReflectable TBlock>
using stream_output_ports = output_port_descriptors<TBlock, PortType::STREAM>;

template<PortReflectable TBlock>
using stream_input_port_types = typename stream_input_ports<TBlock>::template transform<port::type>;

template<PortReflectable TBlock>
using stream_output_port_types = typename stream_output_ports<TBlock>::template transform<port::type>;

template<PortReflectable TBlock>
using stream_input_port_types_tuple = typename stream_input_port_types<TBlock>::tuple_type;

template<PortReflectable TBlock>
using stream_return_type = typename stream_output_port_types<TBlock>::tuple_or_type;

template<PortReflectable TBlock>
using all_input_port_names = typename all_input_ports<TBlock>::template transform<detail::port_name>;

template<PortReflectable TBlock>
using all_output_port_names = typename all_output_ports<TBlock>::template transform<detail::port_name>;

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

template<PortReflectable TBlock>
using simd_return_type_of_can_processOne = vir::simdize<stream_return_type<TBlock>, vir::simdize<stream_input_port_types_tuple<TBlock>>::size()>;
} // namespace detail

/* A block "can process simd" if its `processOne` function takes at least one argument and all
 * arguments can be simdized types of the actual port data types.
 *
 * The block can be a sink (no output ports).
 * The requirement of at least one function argument disallows source blocks.
 *
 * There is another (unnamed) concept for source blocks: Source blocks can implement
 * `processOne_simd(integral_constant N)`, which returns SIMD object(s) of width N.
 */
template<typename TBlock>
concept can_processOne_simd = //
    PortReflectable<TBlock> and traits::block::stream_input_ports<TBlock>::template none_of<port::is_dynamic_port_collection> and traits::block::stream_input_port_types<TBlock>::size() > 0 and requires(TBlock& block, const vir::simdize<stream_input_port_types_tuple<TBlock>>& input_simds) {
        { detail::can_processOne_invoke_test(block, input_simds, stream_input_ports<TBlock>::index_sequence) } -> std::same_as<detail::simd_return_type_of_can_processOne<TBlock>>;
    };

template<typename TBlock>
concept can_processOne_simd_const = //
    PortReflectable<TBlock> and traits::block::stream_input_ports<TBlock>::template none_of<port::is_dynamic_port_collection> and traits::block::stream_input_port_types<TBlock>::size() > 0 and requires(const TBlock& block, const vir::simdize<stream_input_port_types_tuple<TBlock>>& input_simds) {
        { detail::can_processOne_invoke_test(block, input_simds, stream_input_ports<TBlock>::index_sequence) } -> std::same_as<detail::simd_return_type_of_can_processOne<TBlock>>;
    };

template<typename TBlock>
concept can_processOne_scalar = PortReflectable<TBlock> and requires(TBlock& block, const stream_input_port_types_tuple<TBlock>& inputs) {
    { detail::can_processOne_invoke_test(block, inputs, stream_input_ports<TBlock>::index_sequence) } -> std::same_as<stream_return_type<TBlock>>;
};

template<typename TBlock>
concept can_processOne_scalar_const = PortReflectable<TBlock> and requires(const TBlock& block, const stream_input_port_types_tuple<TBlock>& inputs) {
    { detail::can_processOne_invoke_test(block, inputs, stream_input_ports<TBlock>::index_sequence) } -> std::same_as<stream_return_type<TBlock>>;
};

template<typename TBlock>
concept can_processOne = can_processOne_scalar<TBlock> or can_processOne_simd<TBlock>;

template<typename TBlock>
concept can_processOne_const = can_processOne_scalar_const<TBlock> or can_processOne_simd_const<TBlock>;

template<typename TBlock, typename TPort>
concept can_processMessagesForPortReaderSpan = requires(TBlock& block, TPort& inPort) { block.processMessages(inPort, inPort.streamReader().get(1UZ)); };

template<typename TBlock, typename TPort>
concept can_processMessagesForPortStdSpan = requires(TBlock& block, TPort& inPort, std::span<const Message> msgSpan) { block.processMessages(inPort, msgSpan); };

namespace detail {

template<typename T>
struct DummyReaderSpan {
    using value_type = typename std::remove_cv_t<T>;
    using iterator   = typename std::span<const T>::iterator;

private:
    std::span<const T> internalSpan; // Internal span, used for fake implementation

public:
    DummyReaderSpan()                                            = default;
    DummyReaderSpan(const DummyReaderSpan& other)                = default;
    DummyReaderSpan& operator=(const DummyReaderSpan& other)     = default;
    DummyReaderSpan(DummyReaderSpan&& other) noexcept            = default;
    DummyReaderSpan& operator=(DummyReaderSpan&& other) noexcept = default;
    ~DummyReaderSpan()                                           = default;

    [[nodiscard]] constexpr iterator    begin() const noexcept { return internalSpan.begin(); }
    [[nodiscard]] constexpr iterator    end() const noexcept { return internalSpan.end(); }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return internalSpan.size(); }

    operator const std::span<const T>&() const noexcept { return internalSpan; }
    operator std::span<const T>&() noexcept { return internalSpan; }
    // operator std::span<const T>&&() = delete;

    [[nodiscard]] bool consume(std::size_t /* nSamples */) noexcept { return true; }
};
static_assert(ReaderSpanLike<DummyReaderSpan<int>>);

template<typename T>
struct DummyInputSpan : public DummyReaderSpan<T> {
    DummyReaderSpan<gr::Tag> rawTags{};
    bool                     isConnected = true;
    bool                     isSync      = true;
    auto                     tags() { return std::views::empty<std::pair<std::size_t, const property_map&>>; }
    [[nodiscard]] auto       tags(std::size_t /*untilLocalIndex*/) { return std::views::empty<std::pair<std::size_t, const property_map&>>; }
    void                     consumeTags(std::size_t /*untilLocalIndex*/) {}
};
static_assert(ReaderSpanLike<DummyInputSpan<int>>);
static_assert(InputSpanLike<DummyInputSpan<int>>);

template<typename T>
struct DummyWriterSpan {
    using value_type = typename std::remove_cv_t<T>;
    using iterator   = typename std::span<T>::iterator;

private:
    std::span<T> internalSpan; // Internal span, used for fake implementation

public:
    DummyWriterSpan()                                            = default;
    DummyWriterSpan(const DummyWriterSpan& other)                = default;
    DummyWriterSpan& operator=(const DummyWriterSpan& other)     = default;
    DummyWriterSpan(DummyWriterSpan&& other) noexcept            = default;
    DummyWriterSpan& operator=(DummyWriterSpan&& other) noexcept = default;
    ~DummyWriterSpan()                                           = default;

    [[nodiscard]] constexpr iterator    begin() const noexcept { return internalSpan.begin(); }
    [[nodiscard]] constexpr iterator    end() const noexcept { return internalSpan.end(); }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return internalSpan.size(); }

    operator const std::span<T>&() const noexcept { return internalSpan; }
    operator std::span<T>&() noexcept { return internalSpan; }

    constexpr void publish(std::size_t) noexcept {}
};
static_assert(WriterSpanLike<DummyWriterSpan<int>>);

template<typename T>
struct DummyOutputSpan : public DummyWriterSpan<T> {
    DummyWriterSpan<gr::Tag> tags{};
    bool                     isConnected = true;
    bool                     isSync      = true;

    void publishTag(property_map&, std::size_t) {}
};
static_assert(OutputSpanLike<DummyOutputSpan<int>>);

template<gr::detail::PortDescription Port, bool isInputStdSpan, bool isOutputStdSpan>
constexpr auto* port_to_processBulk_argument_helper() {
    if constexpr (Port::kIsDynamicCollection || Port::kIsStaticCollection) { // vector or array of ports
        if constexpr (Port::kIsInput) {
            if constexpr (isInputStdSpan) {
                return static_cast<std::span<std::span<const typename Port::value_type::value_type>>*>(nullptr);
            } else {
                return static_cast<std::span<DummyInputSpan<const typename Port::value_type::value_type>>*>(nullptr);
            }
        } else if constexpr (Port::kIsOutput) {
            if constexpr (isOutputStdSpan) {
                return static_cast<std::span<std::span<typename Port::value_type::value_type>>*>(nullptr);
            } else {
                return static_cast<std::span<DummyOutputSpan<typename Port::value_type::value_type>>*>(nullptr);
            }
        }

    } else { // single port
        if constexpr (Port::kIsInput) {
            if constexpr (isInputStdSpan) {
                return static_cast<std::span<const typename Port::value_type>*>(nullptr);
            } else {
                return static_cast<DummyInputSpan<const typename Port::value_type>*>(nullptr);
            }
        } else if constexpr (Port::kIsOutput) {
            if constexpr (isOutputStdSpan) {
                return static_cast<std::span<typename Port::value_type>*>(nullptr);
            } else {
                return static_cast<DummyOutputSpan<typename Port::value_type>*>(nullptr);
            }
        }
    }
}

template<gr::detail::PortDescription Port>
using port_to_processBulk_argument_std_std = std::remove_pointer_t<decltype(port_to_processBulk_argument_helper<Port, true, true>())>;

template<gr::detail::PortDescription Port>
using port_to_processBulk_argument_std_notstd = std::remove_pointer_t<decltype(port_to_processBulk_argument_helper<Port, true, false>())>;

template<gr::detail::PortDescription Port>
using port_to_processBulk_argument_notstd_std = std::remove_pointer_t<decltype(port_to_processBulk_argument_helper<Port, false, true>())>;

template<gr::detail::PortDescription Port>
using port_to_processBulk_argument_notstd_notstd = std::remove_pointer_t<decltype(port_to_processBulk_argument_helper<Port, false, false>())>;

template<typename>
struct nothing_you_ever_wanted {};

// This alias template is only necessary as a workaround for a bug in Clang. Instead of passing dynamic_span to transform_conditional below, C++ allows passing std::span directly.
template<typename T>
using dynamic_span = std::span<T>;

template<std::size_t... InIdx, std::size_t... OutIdx>
auto can_processBulk_invoke_test(auto& block, auto& inputs, auto& outputs, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) -> decltype(block.processBulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(outputs)...));
} // namespace detail

template<typename TBlock, template<typename> typename TArguments>
concept can_processBulk_helper = requires(TBlock& n, typename traits::block::stream_input_ports<TBlock>::template transform<TArguments>::tuple_type inputs, typename traits::block::stream_output_ports<TBlock>::template transform<TArguments>::tuple_type outputs) {
    { detail::can_processBulk_invoke_test(n, inputs, outputs, stream_input_ports<TBlock>::index_sequence, stream_output_ports<TBlock>::index_sequence) } -> std::same_as<work::Status>;
};

template<typename TBlock>
concept can_processBulk = PortReflectable<TBlock> &&                                                                                                                                          //
                          (can_processBulk_helper<TBlock, detail::port_to_processBulk_argument_std_std> || can_processBulk_helper<TBlock, detail::port_to_processBulk_argument_std_notstd> || //
                              can_processBulk_helper<TBlock, detail::port_to_processBulk_argument_notstd_std> || can_processBulk_helper<TBlock, detail::port_to_processBulk_argument_notstd_notstd>);

/**
 * Satisfied if `TDerived` has a member function `processBulk` which can be invoked with a number of arguments matching the number of input and output ports. Input arguments must accept either a
 * std::span<const T> or any type satisfying InputSpanLike<T>. Output arguments must accept either a std::span<T> or any type satisfying OutputSpanLike<T>, except for the I-th output argument, which
 * must be std::span<T> and *not* a type satisfying OutputSpanLike<T>.
 */
template<typename TDerived, std::size_t I>
concept processBulk_requires_ith_output_as_span = can_processBulk<TDerived> && (I < traits::block::stream_output_port_types<TDerived>::size) && (I >= 0) && requires(TDerived& d, typename meta::transform_types<detail::DummyInputSpan, traits::block::stream_input_port_types<TDerived>>::template apply<std::tuple> inputs, typename meta::transform_conditional<decltype([](auto j) { return j == I; }), detail::dynamic_span, detail::DummyOutputSpan, traits::block::stream_output_port_types<TDerived>>::template apply<std::tuple> outputs, typename meta::transform_conditional<decltype([](auto j) { return j == I; }), detail::nothing_you_ever_wanted, detail::DummyOutputSpan, traits::block::stream_output_port_types<TDerived>>::template apply<std::tuple> bad_outputs) {
    { detail::can_processBulk_invoke_test(d, inputs, outputs, std::make_index_sequence<stream_input_port_types<TDerived>::size>(), std::make_index_sequence<stream_output_port_types<TDerived>::size>()) } -> std::same_as<work::Status>;
    // TODO: Is this check redundant?
    not requires { []<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) -> decltype(d.processBulk(std::get<InIdx>(inputs)..., std::get<OutIdx>(bad_outputs)...)) { return {}; }(std::make_index_sequence<traits::block::stream_input_port_types<TDerived>::size>(), std::make_index_sequence<traits::block::stream_output_port_types<TDerived>::size>()); };
};

} // namespace gr::traits::block

namespace gr {
template<typename Derived>
concept HasProcessOneFunction = traits::block::can_processOne<Derived>;

template<typename Derived>
concept HasConstProcessOneFunction = traits::block::can_processOne_const<Derived>;

template<typename Derived>
concept HasNoexceptProcessOneFunction = HasProcessOneFunction<Derived> && gr::meta::IsNoexceptMemberFunction<decltype(&Derived::processOne)>;

template<typename Derived>
concept HasProcessBulkFunction = traits::block::can_processBulk<Derived>;

template<typename Derived>
concept HasNoexceptProcessBulkFunction = HasProcessBulkFunction<Derived> && gr::meta::IsNoexceptMemberFunction<decltype(&Derived::processBulk)>;

template<typename Derived>
concept HasRequiredProcessFunction = (HasProcessBulkFunction<Derived> or HasProcessOneFunction<Derived>) and (HasProcessOneFunction<Derived> + HasProcessBulkFunction<Derived>) == 1;

} // namespace gr

#endif // include guard
