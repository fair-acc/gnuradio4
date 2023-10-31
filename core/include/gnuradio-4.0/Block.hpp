#ifndef GNURADIO_BLOCK_HPP
#define GNURADIO_BLOCK_HPP

#include <map>
#include <limits>

#include <fmt/format.h>

#include <gnuradio-4.0/meta/typelist.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include "BlockTraits.hpp"
#include "Port.hpp"
#include "Sequence.hpp"
#include "Tag.hpp"
#include "thread/thread_pool.hpp"

#include "annotated.hpp" // This needs to be included after fmt/format.h, as it defines formatters only if FMT_FORMAT_H_ is defined
#include "reflection.hpp"
#include "Settings.hpp"

namespace gr {

using namespace gr::literals;

namespace stdx = vir::stdx;
using gr::meta::fixed_string;

template<typename F>
constexpr void
simd_epilogue(auto kWidth, F &&fun) {
    using namespace vir::literals;
    static_assert(std::has_single_bit(unsigned(kWidth)));
    auto kHalfWidth = kWidth / 2_cw;
    if constexpr (kHalfWidth > 0) {
        fun(kHalfWidth);
        simd_epilogue(kHalfWidth, std::forward<F>(fun));
    }
}

template<std::ranges::contiguous_range... Ts, typename Flag = stdx::element_aligned_tag>
constexpr auto
simdize_tuple_load_and_apply(auto width, const std::tuple<Ts...> &rngs, auto offset, auto &&fun, Flag f = {}) {
    using Tup = vir::simdize<std::tuple<std::ranges::range_value_t<Ts>...>, width>;
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return fun(std::tuple_element_t<Is, Tup>(std::ranges::data(std::get<Is>(rngs)) + offset, f)...);
    }(std::make_index_sequence<sizeof...(Ts)>());
}

template<typename T, typename... Us>
auto
invokeProcessOneWithOrWithoutOffset(T &node, std::size_t offset, const Us &...inputs) {
    if constexpr (traits::block::can_processOne_with_offset<T>) return node.processOne(offset, inputs...);
    else
        return node.processOne(inputs...);
}

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
inputPort(Self *self) noexcept {
    using TRequestedPortType = typename traits::block::input_ports<Self>::template at<Index>;
    if constexpr (traits::block::block_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::block::get_port_member_descriptor<Self, TRequestedPortType>;
        return member_descriptor()(*self);
    } else {
        return std::get<TRequestedPortType>(*self);
    }
}

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
outputPort(Self *self) noexcept {
    using requested_port_type = typename traits::block::output_ports<Self>::template at<Index>;
    if constexpr (traits::block::block_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::block::get_port_member_descriptor<Self, requested_port_type>;
        return member_descriptor()(*self);
    } else {
        return std::get<requested_port_type>(*self);
    }
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
inputPort(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::block::input_ports<Self>>();
    return inputPort<Index, Self>(self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
outputPort(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::block::output_ports<Self>>();
    return outputPort<Index, Self>(self);
}

template<typename Self>
[[nodiscard]] constexpr auto
inputPorts(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::tie(inputPort<Idx>(self)...); }(std::make_index_sequence<traits::block::input_ports<Self>::size>());
}

template<typename Self>
[[nodiscard]] constexpr auto
outputPorts(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::tie(outputPort<Idx>(self)...); }(std::make_index_sequence<traits::block::output_ports<Self>::size>());
}

namespace work {

class Counter {
    std::atomic_uint64_t encodedCounter{ static_cast<uint64_t>(std::numeric_limits<std::uint32_t>::max()) << 32 };

public:
    void
    increment(std::size_t workRequestedInc, std::size_t workDoneInc) {
        uint64_t oldCounter;
        uint64_t newCounter;
        do {
            oldCounter         = encodedCounter;
            auto workRequested = static_cast<std::uint32_t>(oldCounter >> 32);
            auto workDone      = static_cast<std::uint32_t>(oldCounter & 0xFFFFFFFF);
            if (workRequested != std::numeric_limits<std::uint32_t>::max()) {
                workRequested = static_cast<uint32_t>(std::min(static_cast<std::uint64_t>(workRequested) + workRequestedInc, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())));
            }
            workDone += static_cast<std::uint32_t>(workDoneInc);
            newCounter = (static_cast<uint64_t>(workRequested) << 32) | workDone;
        } while (!encodedCounter.compare_exchange_weak(oldCounter, newCounter));
    }

    std::pair<std::size_t, std::size_t>
    getAndReset() {
        uint64_t oldCounter    = encodedCounter.exchange(0);
        auto     workRequested = static_cast<std::uint32_t>(oldCounter >> 32);
        auto     workDone      = static_cast<std::uint32_t>(oldCounter & 0xFFFFFFFF);
        if (workRequested == std::numeric_limits<std::uint32_t>::max()) {
            return { std::numeric_limits<std::size_t>::max(), static_cast<std::size_t>(workDone) };
        }
        return { static_cast<std::size_t>(workRequested), static_cast<std::size_t>(workDone) };
    }

    std::pair<std::size_t, std::size_t>
    get() {
        uint64_t oldCounter    = encodedCounter.load();
        auto     workRequested = static_cast<std::uint32_t>(oldCounter >> 32);
        auto     workDone      = static_cast<std::uint32_t>(oldCounter & 0xFFFFFFFF);
        if (workRequested == std::numeric_limits<std::uint32_t>::max()) {
            return { std::numeric_limits<std::size_t>::max(), static_cast<std::size_t>(workDone) };
        }
        return { static_cast<std::size_t>(workRequested), static_cast<std::size_t>(workDone) };
    }
};

enum class Status {
    ERROR                     = -100, /// error occurred in the work function
    INSUFFICIENT_OUTPUT_ITEMS = -3,   /// work requires a larger output buffer to produce output
    INSUFFICIENT_INPUT_ITEMS  = -2,   /// work requires a larger input buffer to produce output
    DONE                      = -1,   /// this block has completed its processing and the flowgraph should be done
    OK                        = 0,    /// work call was successful and return values in i/o structs are valid
};

struct Result {
    std::size_t requested_work = std::numeric_limits<std::size_t>::max();
    std::size_t performed_work = 0;
    Status      status         = Status::OK;
};
} // namespace work

template<typename T>
concept BlockLike = requires(T t, std::size_t requested_work) {
    { t.unique_name } -> std::same_as<const std::string &>;
    { unwrap_if_wrapped_t<decltype(t.name)>{} } -> std::same_as<std::string>;
    { unwrap_if_wrapped_t<decltype(t.meta_information)>{} } -> std::same_as<property_map>;
    { t.description } noexcept -> std::same_as<const std::string_view &>;

    { t.isBlocking() } noexcept -> std::same_as<bool>;

    { t.settings() } -> std::same_as<SettingsBase &>;
    { t.work(requested_work) } -> std::same_as<work::Result>;

    // N.B. TODO discuss these requirements
    requires !std::is_copy_constructible_v<T>;
    requires !std::is_copy_assignable_v<T>;
    // requires !std::is_move_constructible_v<T>;
    // requires !std::is_move_assignable_v<T>;
};

template<typename Derived>
concept HasProcessOneFunction = traits::block::can_processOne<Derived>;

template<typename Derived>
concept HasProcessBulkFunction = traits::block::can_processBulk<Derived>;

template<typename Derived>
concept HasRequiredProcessFunction = (HasProcessBulkFunction<Derived> or HasProcessOneFunction<Derived>) and(HasProcessOneFunction<Derived> + HasProcessBulkFunction<Derived>) == 1;

template<typename T>
concept ConsumableSpan = std::ranges::contiguous_range<T> and std::convertible_to<T, std::span<const std::remove_cvref_t<typename T::value_type>>> and requires(T &s) { s.consume(0); };

static_assert(ConsumableSpan<traits::block::detail::dummy_input_span<float>>);

template<typename T>
concept PublishableSpan = std::ranges::contiguous_range<T> and std::ranges::output_range<T, std::remove_cvref_t<typename T::value_type>>
                      and std::convertible_to<T, std::span<std::remove_cvref_t<typename T::value_type>>> and requires(T &s) { s.publish(0_UZ); };

static_assert(PublishableSpan<traits::block::detail::dummy_output_span<float>>);

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
 *  gr::IN<T> in;
 *  gr::OUT<R> out;
 *  T _factor = T{1.0};
 *
 *  [[nodiscard]] constexpr auto processOne(T a) const noexcept {
 *      return static_cast<R>(a * _factor);
 *  }
 * @endcode
 * The number, type, and ordering of input and arguments of `processOne(..)` are defined by the port definitions.
 * <li> <b>case 1b</b> - non-decimating N-in->N-out mechanic providing bulk access to the input/output data and automatic
 * handling of streaming tags and settings changes:
 * @code
 *  [[nodiscard]] constexpr auto processBulk(std::span<const T> input, std::span<R> output) const noexcept {
 *      std::ranges::copy(input, output | std::views::transform([a = this->_factor](T x) { return static_cast<R>(x * a); }));
 *  }
 * @endcode
 * <li> <b>case 2a</b>: N-in->M-out -> processBulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N) (to-be-done)
 * <li> <b>case 2b</b>: N-in->M-out -> processBulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling (to-be-done)
 * <li> <b>case 3</b> -- generic `work()` providing full access/logic capable of handling any N-in->M-out tag-handling case:
 * @code
 * [[nodiscard]] constexpr work_return_t work() const noexcept {
 *     auto &out_port = outputPort<"out">(this);
 *     auto &in_port = inputPort<"in">(this);
 *
 *     auto &reader = in_port.streamReader();
 *     auto &writer = out_port.streamWriter();
 *     const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
 *     const auto n_writable = std::min(writer.available(), out_port.max_buffer_size());
 *     if (n_readable == 0) {
 *         return { 0, gr::work::Status::INSUFFICIENT_INPUT_ITEMS };
 *     } else if (n_writable == 0) {
 *         return { 0, gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS };
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
 *         return { n_to_publish, gr::work::Status::ERROR };
 *     }
 *     return { n_to_publish, gr::work::Status::OK };
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
struct Block : protected std::tuple<Arguments...> {
    static std::atomic_size_t _unique_id_counter;
    template<typename T, gr::meta::fixed_string description = "", typename... Args>
    using A                          = Annotated<T, description, Args...>;

    using base_t                     = Block<Derived, Arguments...>;
    using derived_t                  = Derived;
    using ArgumentsTypeList          = typename gr::meta::typelist<Arguments...>;
    using block_template_parameters  = meta::typelist<Arguments...>;
    using Description                = typename block_template_parameters::template find_or_default<is_doc, EmptyDoc>;
    using Resampling                 = ArgumentsTypeList::template find_or_default<is_resampling_ratio, ResamplingRatio<1_UZ, 1_UZ, true>>;
    using StrideControl              = ArgumentsTypeList::template find_or_default<is_stride, Stride<0_UZ, true>>;
    constexpr static bool blockingIO = std::disjunction_v<std::is_same<BlockingIO<true>, Arguments>...> || std::disjunction_v<std::is_same<BlockingIO<false>, Arguments>...>;

    alignas(hardware_destructive_interference_size) std::atomic_uint32_t ioThreadRunning{ 0_UZ };
    alignas(hardware_destructive_interference_size) std::atomic_bool ioThreadShallRun{ false };
    alignas(hardware_destructive_interference_size) std::atomic<std::size_t> ioRequestedWork{ std::numeric_limits<std::size_t>::max() };
    alignas(hardware_destructive_interference_size) work::Counter ioWorkDone{};
    alignas(hardware_destructive_interference_size) std::atomic<work::Status> ioLastWorkStatus{ work::Status::OK };
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::Sequence> progress                         = std::make_shared<gr::Sequence>();
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::thread_pool::BasicThreadPool> ioThreadPool = std::make_shared<gr::thread_pool::BasicThreadPool>(
            "block_thread_pool", gr::thread_pool::TaskType::IO_BOUND, 2_UZ, std::numeric_limits<uint32_t>::max());

    constexpr static TagPropagationPolicy tag_policy = TagPropagationPolicy::TPP_ALL_TO_ALL;
    //
    using RatioValue = std::conditional_t<Resampling::kIsConst, const std::size_t, std::size_t>;
    A<RatioValue, "numerator", Doc<"Top of resampling ratio (<1: Decimate, >1: Interpolate, =1: No change)">, Limits<1_UZ, std::size_t(-1)>>      numerator   = Resampling::kNumerator;
    A<RatioValue, "denominator", Doc<"Bottom of resampling ratio (<1: Decimate, >1: Interpolate, =1: No change)">, Limits<1_UZ, std::size_t(-1)>> denominator = Resampling::kDenominator;
    using StrideValue = std::conditional_t<StrideControl::kIsConst, const std::size_t, std::size_t>;
    A<StrideValue, "stride", Doc<"samples between data processing. <N for overlap, >N for skip, =0 for back-to-back.">> stride         = StrideControl::kStride;
    std::size_t                                                                                                         stride_counter = 0_UZ;
    const std::size_t                                                                                                   unique_id      = _unique_id_counter++;
    const std::string                                                                                                   unique_name = fmt::format("{}#{}", gr::meta::type_name<Derived>(), unique_id);
    A<std::string, "user-defined name", Doc<"N.B. may not be unique -> ::unique_name">>                                 name        = gr::meta::type_name<Derived>();
    A<property_map, "meta-information", Doc<"store non-graph-processing information like UI block position etc.">>      meta_information;
    constexpr static std::string_view                                                                                   description = static_cast<std::string_view>(Description::value);

    struct PortsStatus {
        std::size_t in_min_samples{ std::numeric_limits<std::size_t>::min() };         // max of `port.min_buffer_size()` of all input ports
        std::size_t in_max_samples{ std::numeric_limits<std::size_t>::max() };         // min of `port.max_buffer_size()` of all input ports
        std::size_t in_available{ std::numeric_limits<std::size_t>::max() };           // min of `port.streamReader().available()` of all input ports
        std::size_t in_samples_to_next_tag{ std::numeric_limits<std::size_t>::max() }; // min of `port.samples_to_next_tag` of all input ports

        std::size_t out_min_samples{ std::numeric_limits<std::size_t>::min() }; // max of `port.min_buffer_size()` of all output ports
        std::size_t out_max_samples{ std::numeric_limits<std::size_t>::max() }; // min of `port.max_buffer_size()` of all output ports
        std::size_t out_available{ std::numeric_limits<std::size_t>::max() };   // min of `port.streamWriter().available()` of all input ports

        std::size_t in_samples{ 0 };  // number of input samples to process
        std::size_t out_samples{ 0 }; // number of output samples, calculated based on `numerator` and `denominator`

        bool        in_at_least_one_port_has_data{ false }; // at least one port has data
        bool        in_at_least_one_tag_available{ false }; // at least one port has a tag

        bool        has_sync_input_ports{ false };  // if all ports are async, status is not important
        bool        has_sync_output_ports{ false }; // if all ports are async, status is not important

        constexpr bool
        enoughSamplesForOutputPorts(std::size_t n) {
            return !has_sync_output_ports || n >= out_min_samples;
        }

        constexpr bool
        spaceAvailableOnOutputPorts(std::size_t n) {
            return !has_sync_output_ports || n <= out_available;
        }
    };

    PortsStatus ports_status{};

protected:
    bool             _input_tags_present  = false;
    bool             _output_tags_changed = false;
    std::vector<Tag> _tags_at_input;
    std::vector<Tag> _tags_at_output;

    // intermediate non-real-time<->real-time setting states
    std::unique_ptr<SettingsBase> _settings = std::make_unique<BasicSettings<Derived>>(self());

    [[nodiscard]] constexpr auto &
    self() noexcept {
        return *static_cast<Derived *>(this);
    }

    [[nodiscard]] constexpr const auto &
    self() const noexcept {
        return *static_cast<const Derived *>(this);
    }

    void
    updatePortsStatus() {
        ports_status               = PortsStatus();

        auto adjust_for_input_port = [&ps = ports_status]<PortLike Port>(Port &port) {
            if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                ps.has_sync_input_ports          = true;
                ps.in_min_samples                = std::max(ps.in_min_samples, port.min_buffer_size());
                ps.in_max_samples                = std::min(ps.in_max_samples, port.max_buffer_size());
                ps.in_available                  = std::min(ps.in_available, port.streamReader().available());
                ps.in_samples_to_next_tag        = std::min(ps.in_samples_to_next_tag, samples_to_next_tag(port));
                ps.in_at_least_one_port_has_data = ps.in_at_least_one_port_has_data | (port.streamReader().available() > 0);
                ps.in_at_least_one_tag_available = ps.in_at_least_one_port_has_data | (port.tagReader().available() > 0);
            }
        };
        meta::tuple_for_each(
                [&adjust_for_input_port]<typename Port>(Port &port_or_collection) {
                    if constexpr (traits::port::is_port_v<Port>) {
                        adjust_for_input_port(port_or_collection);
                    } else {
                        for (auto &port : port_or_collection) {
                            adjust_for_input_port(port);
                        }
                    }
                },
                inputPorts(&self()));

        auto adjust_for_output_port = [&ps = ports_status]<PortLike Port>(Port &port) {
            if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                ps.has_sync_output_ports = true;
                ps.out_min_samples       = std::max(ps.out_min_samples, port.min_buffer_size());
                ps.out_max_samples       = std::min(ps.out_max_samples, port.max_buffer_size());
                ps.out_available         = std::min(ps.out_available, port.streamWriter().available());
            }
        };
        meta::tuple_for_each(
                [&adjust_for_output_port]<typename Port>(Port &port_or_collection) {
                    if constexpr (traits::port::is_port_v<Port>) {
                        adjust_for_output_port(port_or_collection);
                    } else {
                        for (auto &port : port_or_collection) {
                            adjust_for_output_port(port);
                        }
                    }
                },
                outputPorts(&self()));

        ports_status.in_samples = ports_status.in_available;
        if (ports_status.in_samples < ports_status.in_min_samples) ports_status.in_samples = 0;
        if (ports_status.in_samples > ports_status.in_max_samples) ports_status.in_samples = ports_status.in_max_samples;

        // By default N-in == N-out
        // TODO: adjust `samples_to_proceed` to output limits?
        ports_status.out_samples = ports_status.in_samples;

        if (ports_status.has_sync_input_ports && ports_status.in_min_samples > ports_status.in_max_samples)
            throw std::runtime_error(fmt::format("Min samples for input ports ({}) is larger then max samples for input ports ({})", ports_status.in_min_samples, ports_status.in_max_samples));
        if (ports_status.has_sync_output_ports && ports_status.out_min_samples > ports_status.out_max_samples)
            throw std::runtime_error(fmt::format("Min samples for output ports ({}) is larger then max samples for output ports ({})", ports_status.out_min_samples, ports_status.out_max_samples));

        if (!ports_status.has_sync_input_ports) {
            ports_status.in_samples   = 0;
            ports_status.in_available = 0;
        }
        if (!ports_status.has_sync_output_ports) {
            ports_status.out_samples   = 0;
            ports_status.out_available = 0;
        }
    }

public:
    Block() noexcept : Block({}) {}

    Block(std::initializer_list<std::pair<const std::string, pmtv::pmt>> init_parameter)
        : _tags_at_input(traits::block::input_port_types<Derived>::size())
        , _tags_at_output(traits::block::output_port_types<Derived>::size())
        , _settings(std::make_unique<BasicSettings<Derived>>(*static_cast<Derived *>(this))) { // N.B. safe delegated use of this (i.e. not used during construction)
        if (init_parameter.size() != 0) {
            const auto failed = settings().set(init_parameter);
            if (!failed.empty()) {
                throw std::invalid_argument("Settings not applied successfully");
            }
        }
    }

    Block(Block &&other) noexcept
        : std::tuple<Arguments...>(std::move(other)), _tags_at_input(std::move(other._tags_at_input)), _tags_at_output(std::move(other._tags_at_output)), _settings(std::move(other._settings)) {}

    ~Block() { // NOSONAR -- need to request the (potentially) running ioThread to stop
        stop();
    }

    void
    init(std::shared_ptr<gr::Sequence> progress_, std::shared_ptr<gr::thread_pool::BasicThreadPool> ioThreadPool_) {
        progress     = std::move(progress_);
        ioThreadPool = std::move(ioThreadPool_);

        // Set names of port member variables
        // TODO: Refactor the library not to assign names to ports. The
        // block and the graph are the only things that need the port name
        auto setPortName = [&]([[maybe_unused]] std::size_t index, auto &&t) {
            using CurrentPortType = std::remove_cvref_t<decltype(t)>;
            if constexpr (traits::port::is_port_v<CurrentPortType>) {
                using PortDescriptor = typename CurrentPortType::ReflDescriptor;
                if constexpr (refl::trait::is_descriptor_v<PortDescriptor>) {
                    auto &port = (self().*(PortDescriptor::pointer));
                    port.name  = CurrentPortType::Name;
                }
            } else {
                using PortCollectionDescriptor = typename CurrentPortType::value_type::ReflDescriptor;
                if constexpr (refl::trait::is_descriptor_v<PortCollectionDescriptor>) {
                    auto       &collection     = (self().*(PortCollectionDescriptor::pointer));
                    std::string collectionName = refl::descriptor::get_name(PortCollectionDescriptor()).data;
                    for (auto &port : collection) {
                        port.name = collectionName;
                    }
                }
            }
        };
        traits::block::input_ports<Derived>::template apply_func(setPortName);
        traits::block::output_ports<Derived>::template apply_func(setPortName);

        // Handle settings
        if (const auto forward_parameters = settings().applyStagedParameters(); !forward_parameters.empty()) {
            std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [&forward_parameters](Tag &tag) {
                for (const auto &[key, value] : forward_parameters) {
                    tag.map.insert_or_assign(key, value);
                }
            });
            _output_tags_changed = true;
        }

        // store default settings -> can be recovered with 'resetDefaults()'
        settings().storeDefaults();
    }

    void
    stop() {
        std::atomic_store_explicit(&ioThreadShallRun, false, std::memory_order_release);
        ioThreadShallRun.notify_all();
        // wait for done
        for (auto running = ioThreadRunning.load(); running > 0; running = ioThreadRunning.load()) {
            ioThreadRunning.wait(running);
        }
    }

    template<gr::meta::array_or_vector_type Container>
    [[nodiscard]] constexpr std::size_t
    availableInputSamples(Container &data) const noexcept {
        if constexpr (gr::meta::vector_type<Container>) {
            data.resize(traits::block::input_port_types<Derived>::size);
        } else if constexpr (gr::meta::array_type<Container>) {
            static_assert(std::tuple_size<Container>::value >= traits::block::input_port_types<Derived>::size);
        } else {
            static_assert(gr::meta::always_false<Container>, "type not supported");
        }
        meta::tuple_for_each_enumerate(
                [&data]<typename Port>(auto index, Port &input_port) {
                    if constexpr (traits::port::is_port_v<Port>) {
                        data[index] = input_port.streamReader().available();
                    } else {
                        data[index] = 0;
                        for (auto &port : input_port) {
                            data[index] += port.streamReader().available();
                        }
                    }
                },
                inputPorts(&self()));
        return traits::block::input_port_types<Derived>::size;
    }

    template<gr::meta::array_or_vector_type Container>
    [[nodiscard]] constexpr std::size_t
    availableOutputSamples(Container &data) const noexcept {
        if constexpr (gr::meta::vector_type<Container>) {
            data.resize(traits::block::output_port_types<Derived>::size);
        } else if constexpr (gr::meta::array_type<Container>) {
            static_assert(std::tuple_size<Container>::value >= traits::block::output_port_types<Derived>::size);
        } else {
            static_assert(gr::meta::always_false<Container>, "type not supported");
        }
        meta::tuple_for_each_enumerate(
                [&data]<typename Port>(auto index, Port &output_port) {
                    if constexpr (traits::port::is_port_v<Port>) {
                        data[index] = output_port.streamWriter().available();
                    } else {
                        data[index] = 0;
                        for (auto &port : output_port) {
                            data[index] += port.streamWriter().available();
                        }
                    }
                },
                outputPorts(&self()));
        return traits::block::output_port_types<Derived>::size;
    }

    [[nodiscard]] constexpr bool
    isBlocking() const noexcept {
        return blockingIO;
    }

    [[nodiscard]] constexpr bool
    input_tags_present() const noexcept {
        return _input_tags_present;
    };

    constexpr bool
    acknowledge_input_tags() noexcept {
        if (_input_tags_present) {
            _input_tags_present = false;
            return true;
        }
        return false;
    };

    [[nodiscard]] constexpr std::span<const Tag>
    input_tags() const noexcept {
        return { _tags_at_input.data(), _tags_at_input.size() };
    }

    [[nodiscard]] constexpr std::span<const Tag>
    output_tags() const noexcept {
        return { _tags_at_output.data(), _tags_at_output.size() };
    }

    [[nodiscard]] constexpr std::span<Tag>
    output_tags() noexcept {
        _output_tags_changed = true;
        return { _tags_at_output.data(), _tags_at_output.size() };
    }

    [[nodiscard]] constexpr SettingsBase &
    settings() const noexcept {
        return *_settings;
    }

    [[nodiscard]] constexpr SettingsBase &
    settings() noexcept {
        return *_settings;
    }

    template<typename T>
    void
    setSettings(std::unique_ptr<T> &settings) {
        _settings = std::move(settings);
    }

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    inputPort(Self *self) noexcept;

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    outputPort(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    inputPort(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    outputPort(Self *self) noexcept;

    void
    write_to_outputs(std::size_t available_values_count, auto &writers_tuple) noexcept {
        if constexpr (traits::block::output_ports<Derived>::size > 0) {
            meta::tuple_for_each_enumerate(
                    [available_values_count]<typename OutputRange>(auto i, OutputRange &output_range) {
                        if constexpr (traits::block::can_processOne<Derived> or traits::block::processBulk_requires_ith_output_as_span<Derived, i>) {
                            auto process_out = [available_values_count]<typename Out>(Out &out) {
                                // This will be a pointer if the port was async
                                // TODO: Make this check more specific
                                if constexpr (not std::is_pointer_v<std::remove_cvref_t<Out>>) {
                                    out.publish(available_values_count);
                                }
                            };
                            if (available_values_count) {
                                if constexpr (refl::trait::is_instance_of_v<std::vector, std::remove_cvref_t<OutputRange>>) {
                                    for (auto &out : output_range) {
                                        process_out(out);
                                    }
                                } else {
                                    process_out(output_range);
                                }
                            }
                        } else {
                            if constexpr (requires { output_range.is_published(); }) {
                                if (not output_range.is_published()) {
                                    fmt::print(stderr, "processBulk failed to publish one of its outputs. Use a std::span argument if you do not want to publish manually.\n");
                                    std::abort();
                                }
                            }
                        }
                    },
                    writers_tuple);
        }
    }

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    template<typename Self>
    bool
    consumeReaders(Self &self, std::size_t available_values_count) {
        bool success = true;
        if constexpr (traits::block::input_ports<Derived>::size > 0) {
            std::apply(
                    [available_values_count, &success](auto &...input_port) {
                        auto consume_port = [&]<typename Port>(Port &port_or_collection) {
                            if constexpr (traits::port::is_port_v<Port>) {
                                success = success && port_or_collection.streamReader().consume(available_values_count);
                            } else {
                                for (auto &port : port_or_collection) {
                                    success = success && port.streamReader().consume(available_values_count);
                                }
                            }
                        };
                        (consume_port(input_port), ...);
                    },
                    inputPorts(&self));
        }
        return success;
    }

    template<typename... Ts>
    constexpr auto
    invoke_processOne(std::size_t offset, Ts &&...inputs) {
        if constexpr (traits::block::output_ports<Derived>::size == 0) {
            invokeProcessOneWithOrWithoutOffset(self(), offset, std::forward<Ts>(inputs)...);
            return std::tuple{};
        } else if constexpr (traits::block::output_ports<Derived>::size == 1) {
            return std::tuple{ invokeProcessOneWithOrWithoutOffset(self(), offset, std::forward<Ts>(inputs)...) };
        } else {
            return invokeProcessOneWithOrWithoutOffset(self(), offset, std::forward<Ts>(inputs)...);
        }
    }

    template<typename... Ts>
    constexpr auto
    invoke_processOne_simd(std::size_t offset, auto width, Ts &&...input_simds) {
        if constexpr (sizeof...(Ts) == 0) {
            if constexpr (traits::block::output_ports<Derived>::size == 0) {
                self().processOne_simd(offset, width);
                return std::tuple{};
            } else if constexpr (traits::block::output_ports<Derived>::size == 1) {
                return std::tuple{ self().processOne_simd(offset, width) };
            } else {
                return self().processOne_simd(offset, width);
            }
        } else {
            return invoke_processOne(offset, std::forward<Ts>(input_simds)...);
        }
    }

    constexpr void
    forward_tags() noexcept {
        if (!(_output_tags_changed || _input_tags_present)) {
            return;
        }
        std::size_t port_id = 0; // TODO absorb this as optional tuple_for_each argument
        // TODO: following function does not call the lvalue but erroneously the lvalue version of publish_tag(...) ?!?!
        // meta::tuple_for_each([&port_id, this](auto &output_port) noexcept { publish_tag2(output_port, _tags_at_output[port_id++]); }, output_ports(&self()));
        meta::tuple_for_each(
                [&port_id, this]<typename Port>(Port &output_port) noexcept {
                    if constexpr (!traits::port::is_port_v<Port>) {
                        // TODO Add tag support to port collections?
                        return;
                    } else {
                        if (_tags_at_output[port_id].map.empty()) {
                            port_id++;
                            return;
                        }
                        auto data                 = output_port.tagWriter().reserve_output_range(1);
                        auto stream_writer_offset = std::max(static_cast<decltype(output_port.streamWriter().position())>(0), output_port.streamWriter().position() + 1);
                        data[0].index             = stream_writer_offset + _tags_at_output[port_id].index;
                        data[0].map               = _tags_at_output[port_id].map;
                        data.publish(1);
                        port_id++;
                    }
                },
                outputPorts(&self()));
        // clear input/output tags after processing,  N.B. ranges omitted because of missing Clang/Emscripten support
        _input_tags_present  = false;
        _output_tags_changed = false;
        std::for_each(_tags_at_input.begin(), _tags_at_input.end(), [](Tag &tag) { tag.reset(); });
        std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [](Tag &tag) { tag.reset(); });
    }

protected:
    /**
     * @brief
     * @return struct { std::size_t produced_work, work_return_t}
     */
    work::Result
    workInternal(std::size_t requested_work) {
        using gr::work::Status;
        using TInputTypes             = traits::block::input_port_types<Derived>;
        using TOutputTypes            = traits::block::output_port_types<Derived>;

        constexpr bool kIsSourceBlock = TInputTypes::size == 0;
        constexpr bool kIsSinkBlock   = TOutputTypes::size == 0;

        // TODO: these checks can be moved to setting changed
        if constexpr (Resampling::kEnabled) {
            static_assert(!kIsSinkBlock, "Decimation/interpolation is not available for sink blocks. Remove 'ResamplingRatio<>' from the block definition.");
            static_assert(!kIsSourceBlock, "Decimation/interpolation is not available for source blocks. Remove 'ResamplingRatio<>' from the block definition.");
            static_assert(HasProcessBulkFunction<Derived>, "Blocks which allow decimation/interpolation must implement processBulk(...) method. Remove 'ResamplingRatio<>' from the block definition.");
        } else {
            if (numerator != 1_UZ || denominator != 1_UZ) {
                throw std::runtime_error(fmt::format("Block is not defined as `ResamplingRatio<>`, but numerator = {}, denominator = {}, they both must equal to 1.", numerator, denominator));
            }
        }

        if constexpr (StrideControl::kEnabled) {
            static_assert(!kIsSourceBlock, "Stride is not available for source blocks. Remove 'Stride<>' from the block definition.");
        } else {
            if (stride != 0_UZ) {
                throw std::runtime_error(fmt::format("Block is not defined as `Stride<>`, but stride = {}, it must equal to 0.", stride));
            }
        }

        updatePortsStatus();

        if constexpr (kIsSourceBlock) {
            ports_status.in_samples_to_next_tag = std::numeric_limits<std::size_t>::max(); // no tags to processed for source node
            if constexpr (requires(const Derived &d) {
                              { self().available_samples(d) } -> std::same_as<std::make_signed_t<std::size_t>>;
                          }) {
                // the (source) node wants to determine the number of samples to process
                std::size_t                           max_buffer        = ports_status.out_available;
                const std::make_signed_t<std::size_t> available_samples = self().available_samples(self());
                if (available_samples < 0 && max_buffer > 0) {
                    return { requested_work, 0_UZ, work::Status::DONE };
                }
                if (available_samples == 0) {
                    return { requested_work, 0_UZ, work::Status::OK };
                }
                std::size_t samples_to_process = std::max(0UL, std::min(static_cast<std::size_t>(available_samples), max_buffer));
                if (not ports_status.enoughSamplesForOutputPorts(samples_to_process)) {
                    return { requested_work, 0_UZ, work::Status::INSUFFICIENT_INPUT_ITEMS };
                }
                if (samples_to_process == 0) {
                    return { requested_work, 0_UZ, work::Status::INSUFFICIENT_OUTPUT_ITEMS };
                }
                ports_status.in_samples  = std::min(samples_to_process, requested_work);
                ports_status.out_samples = ports_status.in_samples;

            } else if constexpr (requires(const Derived &d) {
                                     { available_samples(d) } -> std::same_as<std::size_t>;
                                 }) {
                // the (source) node wants to determine the number of samples to process
                std::size_t samples_to_process = available_samples(self());
                if (samples_to_process == 0) {
                    return { requested_work, 0_UZ, work::Status::OK };
                }
                if (not ports_status.enoughSamplesForOutputPorts(samples_to_process)) {
                    return { requested_work, 0_UZ, work::Status::INSUFFICIENT_INPUT_ITEMS };
                }
                if (not ports_status.spaceAvailableOnOutputPorts(samples_to_process)) {
                    return { requested_work, 0_UZ, work::Status::INSUFFICIENT_OUTPUT_ITEMS };
                }
                ports_status.in_samples  = std::min(samples_to_process, requested_work);
                ports_status.out_samples = ports_status.in_samples;

            } else if constexpr (kIsSinkBlock) {
                // no input or output buffers, derive from internal "buffer sizes" (i.e. what the
                // buffer size would be if the node were not merged)
                constexpr std::size_t chunk_size = Derived::merged_work_chunk_size();
                static_assert(chunk_size != std::dynamic_extent && chunk_size > 0, "At least one internal port must define a maximum number of samples or the non-member/hidden "
                                                                                   "friend function `available_samples(const BlockType&)` must be defined.");
                ports_status.in_samples  = std::min(chunk_size, requested_work);
                ports_status.out_samples = ports_status.in_samples;

            } else {
                // derive value from output buffer size
                std::size_t samples_to_process = std::min(ports_status.out_available, ports_status.out_max_samples);
                if (not ports_status.enoughSamplesForOutputPorts(samples_to_process)) {
                    return { requested_work, 0_UZ, work::Status::INSUFFICIENT_OUTPUT_ITEMS };
                }
                ports_status.in_samples  = std::min(samples_to_process, requested_work);
                ports_status.out_samples = ports_status.in_samples;
                // space_available_on_output_ports is true by construction of samples_to_process
            }

        } else {
            ports_status.in_samples  = std::min(ports_status.in_samples, requested_work);
            ports_status.out_samples = ports_status.in_samples;

            if (ports_status.has_sync_input_ports && ports_status.in_available == 0) {
                return { requested_work, 0_UZ, ports_status.in_at_least_one_port_has_data ? work::Status::INSUFFICIENT_INPUT_ITEMS : work::Status::DONE };
            }

            if constexpr (Resampling::kEnabled) {
                if (numerator != 1_UZ || denominator != 1_UZ) {
                    // TODO: this ill-defined checks can be done only once after parameters were changed
                    const double ratio  = static_cast<double>(numerator) / static_cast<double>(denominator);
                    bool is_ill_defined = (denominator > ports_status.in_max_samples) || (static_cast<double>(ports_status.in_min_samples) * ratio > static_cast<double>(ports_status.out_max_samples))
                                       || (static_cast<double>(ports_status.in_max_samples) * ratio < static_cast<double>(ports_status.out_min_samples));
                    assert(!is_ill_defined);
                    if (is_ill_defined) {
                        return { requested_work, 0_UZ, work::Status::ERROR };
                    }

                    ports_status.in_samples          = static_cast<std::size_t>(ports_status.in_samples / denominator) * denominator; // remove reminder

                    const std::size_t out_min_limit  = ports_status.out_min_samples;
                    const std::size_t out_max_limit  = std::min(ports_status.out_available, ports_status.out_max_samples);

                    std::size_t       in_min_samples = static_cast<std::size_t>(static_cast<double>(out_min_limit) / ratio);
                    if (in_min_samples % denominator != 0) in_min_samples += denominator;
                    std::size_t       in_min_wo_reminder = (in_min_samples / denominator) * denominator;

                    const std::size_t in_max_samples     = static_cast<std::size_t>(static_cast<double>(out_max_limit) / ratio);
                    std::size_t       in_max_wo_reminder = (in_max_samples / denominator) * denominator;

                    if (ports_status.in_samples < in_min_wo_reminder) return { requested_work, 0_UZ, work::Status::INSUFFICIENT_INPUT_ITEMS };
                    ports_status.in_samples  = std::clamp(ports_status.in_samples, in_min_wo_reminder, in_max_wo_reminder);
                    ports_status.out_samples = numerator * (ports_status.in_samples / denominator);
                }
            }

            // TODO: special case for ports_status.in_samples == 0 ?

            if (not ports_status.enoughSamplesForOutputPorts(ports_status.out_samples)) {
                return { requested_work, 0_UZ, work::Status::INSUFFICIENT_INPUT_ITEMS };
            }
            if (not ports_status.spaceAvailableOnOutputPorts(ports_status.out_samples)) {
                return { requested_work, 0_UZ, work::Status::INSUFFICIENT_OUTPUT_ITEMS };
            }
        }

        if (ports_status.in_samples_to_next_tag == 0) {
            if constexpr (HasProcessOneFunction<Derived>) {
                ports_status.in_samples  = 1; // N.B. limit to one so that only one process_on(...) invocation receives the tag
                ports_status.out_samples = 1;
            }
            property_map merged_tag_map;
            _input_tags_present    = true;
            std::size_t port_index = 0; // TODO absorb this as optional tuple_for_each argument
            meta::tuple_for_each(
                    [&merged_tag_map, &port_index, this]<typename Port>(Port &input_port) noexcept {
                        // TODO: Do we want to support tags for non-compile-time ports? [ivan][port_group][move_to_policy?]
                        if constexpr (traits::port::is_port_v<Port>) {
                            auto &tag_at_present_input = _tags_at_input[port_index++];
                            tag_at_present_input.reset();
                            if (!input_port.tagReader().available()) {
                                return;
                            }
                            const auto tags           = input_port.tagReader().get(1_UZ);
                            const auto readPos        = input_port.streamReader().position();
                            const auto tag_stream_pos = tags[0].index - 1 - readPos;
                            if ((readPos == -1 && tags[0].index <= 0) // first tag on initialised stream
                                || tag_stream_pos <= 0) {
                                for (const auto &[index, map] : tags) {
                                    for (const auto &[key, value] : map) {
                                        tag_at_present_input.map.insert_or_assign(key, value);
                                        merged_tag_map.insert_or_assign(key, value);
                                    }
                                }
                                std::ignore = input_port.tagReader().consume(1_UZ);
                            }
                        }
                    },
                    inputPorts(&self()));

            if (_input_tags_present && !merged_tag_map.empty()) { // apply tags as new settings if matching
                settings().autoUpdate(merged_tag_map);
            }

            if constexpr (Derived::tag_policy == TagPropagationPolicy::TPP_ALL_TO_ALL) {
                // N.B. ranges omitted because of missing Clang/Emscripten support
                std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [&merged_tag_map](Tag &tag) { tag.map = merged_tag_map; });
                _output_tags_changed = true;
            }
        }

        if (settings().changed() || _input_tags_present || _output_tags_changed) {
            if (const auto forward_parameters = settings().applyStagedParameters(); !forward_parameters.empty()) {
                std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [&forward_parameters](Tag &tag) {
                    for (const auto &[key, value] : forward_parameters) {
                        tag.map.insert_or_assign(key, value);
                    }
                });
                _output_tags_changed = true;
            }
            settings()._changed.store(false);
        }

        // TODO: check here whether a processOne(...) or a bulk access process has been defined, cases:
        // case 1a: N-in->N-out -> processOne(...) -> auto-handling of streaming tags
        // case 1b: N-in->N-out -> processBulk(<ins...>, <outs...>) -> auto-handling of streaming tags
        // case 2a: N-in->M-out -> processBulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N)
        // case 2b: N-in->M-out -> processBulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling
        // case 3:  N-in->M-out -> work() N,M arbitrary -> used need to handle the full logic (e.g. PLL algo)
        // case 4:  Python -> map to cases 1-3 and/or dedicated callback
        // special cases:
        // case sources: HW triggered vs. generating data per invocation (generators via Port::MIN)
        // case sinks: HW triggered vs. fixed-size consumer (may block/never finish for insufficient input data and fixed Port::MIN>0)

        std::size_t n_samples_to_consume = ports_status.in_samples; // default stride == 0
        if constexpr (StrideControl::kEnabled) {
            if (stride != 0_UZ) {
                const bool first_time_stride = stride_counter == 0;
                if (first_time_stride) {
                    // sample processing are done as usual, ports_status.in_samples samples will be processed
                    if (stride.value > stride_counter + ports_status.in_available) { // stride can not be consumed at once -> start stride_counter
                        stride_counter += ports_status.in_available;
                        n_samples_to_consume = ports_status.in_available;
                    } else { // if the stride can be consumed at once -> no stride_counter is needed
                        stride_counter       = 0;
                        n_samples_to_consume = stride.value;
                    }
                } else {
                    // |====================|...|====================|==============----| -> ====== is the stride
                    //   ^first                    ^we are here (1)  or ^here (2)
                    // if it is not the "first time" stride -> just consume (1) all samples or (2) missing rest of the samples
                    // forward tags but no additional sample processing are done ->return
                    if (stride.value > stride_counter + ports_status.in_available) {
                        stride_counter += ports_status.in_available;
                        n_samples_to_consume = ports_status.in_available;
                    } else { // stride is at the end -> reset stride_counter
                        n_samples_to_consume = stride.value - stride_counter;
                        stride_counter       = 0;
                    }
                    const bool success = consumeReaders(self(), n_samples_to_consume);
                    forward_tags();
                    return { requested_work, n_samples_to_consume, success ? work::Status::OK : work::Status::ERROR };
                }
            }
        }

        const auto input_spans = meta::tuple_transform(
                [&self = self(), sync_in_samples = self().ports_status.in_samples]<typename PortOrCollection>(PortOrCollection &input_port_or_collection) noexcept {
                    auto in_samples          = sync_in_samples;

                    auto process_single_port = [&in_samples]<typename Port>(Port &&port) {
                        if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                            return std::forward<Port>(port).streamReader().get(in_samples);
                        } else {
                            return std::addressof(std::forward<Port>(port).streamReader());
                        }
                    };
                    if constexpr (traits::port::is_port_v<PortOrCollection>) {
                        return process_single_port(input_port_or_collection);
                    } else {
                        using value_span = decltype(process_single_port(std::declval<typename PortOrCollection::value_type>()));
                        std::vector<value_span> result;
                        std::transform(input_port_or_collection.begin(), input_port_or_collection.end(), std::back_inserter(result), process_single_port);
                        return result;
                    }
                },
                inputPorts(&self()));
        auto writers_tuple = meta::tuple_transform(
                [&self = self(), sync_out_samples = ports_status.out_samples]<typename PortOrCollection>(PortOrCollection &output_port_or_collection) noexcept {
                    auto out_samples         = sync_out_samples;

                    auto process_single_port = [&out_samples]<typename Port>(Port &&port) {
                        if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                            return std::forward<Port>(port).streamWriter().reserve_output_range(out_samples);
                        } else {
                            return std::addressof(std::forward<Port>(port).streamWriter());
                        }
                    };
                    if constexpr (traits::port::is_port_v<PortOrCollection>) {
                        return process_single_port(output_port_or_collection);
                    } else {
                        using value_span = decltype(process_single_port(std::declval<typename PortOrCollection::value_type>()));
                        std::vector<value_span> result;
                        std::transform(output_port_or_collection.begin(), output_port_or_collection.end(), std::back_inserter(result), process_single_port);
                        return result;
                    }
                },
                outputPorts(&self()));

        if constexpr (HasProcessBulkFunction<Derived>) {
            // cannot use std::apply because it requires tuple_cat(input_spans, writers_tuple). The latter doesn't work because writers_tuple isn't copyable.
            const work::Status ret = [&]<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) {
                return self().processBulk(std::get<InIdx>(input_spans)..., std::get<OutIdx>(writers_tuple)...);
            }(std::make_index_sequence<traits::block::input_ports<Derived>::size>(), std::make_index_sequence<traits::block::output_ports<Derived>::size>());

            write_to_outputs(ports_status.out_samples, writers_tuple);
            const bool success = consumeReaders(self(), n_samples_to_consume);
            forward_tags();
            return { requested_work, ports_status.in_samples, success ? ret : work::Status::ERROR };

        } else if constexpr (HasProcessOneFunction<Derived>) {
            if (ports_status.in_samples != ports_status.out_samples)
                throw std::runtime_error(fmt::format("N input samples ({}) does not equal to N output samples ({}) for processOne() method.", ports_status.in_samples, ports_status.out_samples));
            // handle processOne(...)
            static constexpr std::size_t kMaxWidth = stdx::simd_abi::max_fixed_size<double>;
            // A block determines it's simd::size() via its input types. However, a source block doesn't have any input
            // types and therefore wouldn't be able to produce simd output on processOne calls. To overcome this
            // limitation, a source block can implement `processOne_simd(vir::constexpr_value auto width)` instead of
            // `processOne()` and then return simd objects with simd::size() == width.
            constexpr bool kIsSimdSourceBlock
                    = kIsSourceBlock and requires(Derived &d) { d.processOne_simd(vir::cw<kMaxWidth>); };
            if constexpr (kIsSimdSourceBlock or traits::block::can_processOne_simd<Derived>) {
                // SIMD loop
                const auto kWidth = [] {
                    if constexpr (kIsSourceBlock) {
                        return vir::cw<kMaxWidth>;
                    } else {
                        using input_simd_types = vir::simdize<typename TInputTypes::template apply<std::tuple>>;
                        return vir::cw<std::min(kMaxWidth, input_simd_types::size() * 4_UZ)>;
                    }
                }();

                std::size_t i = 0;
                for (; i + kWidth <= ports_status.in_samples; i += kWidth) {
                    const auto &results = simdize_tuple_load_and_apply(
                            kWidth, input_spans, i, [&](const auto &...input_simds) {
                                return invoke_processOne_simd(i, kWidth, input_simds...);
                            });
                    meta::tuple_for_each(
                            [i](auto &output_range, const auto &result) {
                                result.copy_to(output_range.data() + i, stdx::element_aligned);
                            },
                            writers_tuple, results);
                }
                simd_epilogue(kWidth, [&](auto w) {
                    if (i + w <= ports_status.in_samples) {
                        const auto results = simdize_tuple_load_and_apply(
                                w, input_spans, i, [&](auto &&...input_simds) {
                                    return invoke_processOne_simd(i, w, input_simds...);
                                });
                        meta::tuple_for_each(
                                [i](auto &output_range, auto &result) {
                                    result.copy_to(output_range.data() + i, stdx::element_aligned);
                                },
                                writers_tuple, results);
                        i += w;
                    }
                });
            } else {
                // Non-SIMD loop
                for (std::size_t i = 0; i < ports_status.in_samples; ++i) {
                    const auto results = std::apply([this, i](auto &...inputs) { return this->invoke_processOne(i, inputs[i]...); }, input_spans);
                    meta::tuple_for_each([i](auto &output_range, auto &result) { output_range[i] = std::move(result); }, writers_tuple, results);
                }
            }

            write_to_outputs(ports_status.out_samples, writers_tuple);

            const bool success = consumeReaders(self(), n_samples_to_consume);

#ifdef _DEBUG
            if (!success) {
                fmt::print("Block {} failed to consume {} values from inputs\n", self().name(), samples_to_process);
            }
#endif
            forward_tags();
            return { requested_work, ports_status.in_samples, success ? work::Status::OK : work::Status::ERROR };
        } // processOne(...) handling
        //        else {
        //            static_assert(gr::meta::always_false<Derived>, "neither processBulk(...) nor processOne(...) implemented");
        //        }
        return { requested_work, 0_UZ, work::Status::ERROR };
    } // end: work_return_t work_internal() noexcept { ..}

public:
    work::Status
    invokeWork()
        requires(blockingIO)
    {
        auto [work_requested, work_done, last_status] = workInternal(ioRequestedWork.load(std::memory_order_relaxed));
        ioWorkDone.increment(work_requested, work_done);
        if (auto [incWorkRequested, incWorkDone] = ioWorkDone.get(); last_status == work::Status::DONE && incWorkDone > 0) {
            // finished local iteration but need to report more work to be done until
            // external scheduler loop acknowledged all samples being processed
            // via the 'ioWorkDone.getAndReset()' call
            last_status = work::Status::OK;
        }
        ioLastWorkStatus.exchange(last_status, std::memory_order_relaxed);

        std::ignore = progress->incrementAndGet();
        progress->notify_all();
        return last_status;
    }

    /**
     * @brief Process as many samples as available and compatible with the internal boundary requirements or limited by 'requested_work`
     *
     * @param requested_work: usually the processed number of input samples, but could be any other metric as long as
     * requested_work limit as an affine relation with the returned performed_work.
     * @return { requested_work, performed_work, status}
     */
    work::Result
    work(std::size_t requested_work = std::numeric_limits<std::size_t>::max()) noexcept {
        if constexpr (blockingIO) {
            constexpr bool useIoThread = std::disjunction_v<std::is_same<BlockingIO<true>, Arguments>...>;
            std::atomic_store_explicit(&ioRequestedWork, requested_work, std::memory_order_release);
            if (bool expectedThreadState = false; ioThreadShallRun.compare_exchange_strong(expectedThreadState, true, std::memory_order_acq_rel)) {
                if constexpr (useIoThread) { // use graph-provided ioThreadPool
                    ioThreadPool->execute([this]() {
#ifdef _DEBUG
                        fmt::print("starting thread for {} count {}\n", name, ioThreadRunning.fetch_add(1));
#else
                        ioThreadRunning.fetch_add(1);
#endif
                        for (int retryCount = 2; ioThreadShallRun && retryCount > 0; retryCount--) {
                            while (ioThreadShallRun && retryCount) {
                                if (invokeWork() == work::Status::DONE) {
                                    break;
                                } else {
                                    // processed data before shutting down wait (see below) and retry (here: once)
                                    retryCount = 2;
                                }
                            }
                            // delayed shut-down in case there are more tasks to be processed
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        }
                        std::atomic_store_explicit(&ioThreadShallRun, false, std::memory_order_release);
                        ioThreadShallRun.notify_all();
#ifdef _DEBUG
                        fmt::print("shutting down thread for {} count {}\n", name, ioThreadRunning.fetch_sub(1));
#else
                        ioThreadRunning.fetch_sub(1);
#endif
                        ioThreadRunning.notify_all();
                    });
                } else {
                    // let user call '' explicitly
                }
            }
            const work::Status lastStatus                         = ioLastWorkStatus.exchange(work::Status::OK, std::memory_order_relaxed);
            const auto &[accumulatedRequestedWork, performedWork] = ioWorkDone.getAndReset();
            // TODO: this is just "working" solution for deadlock with emscripten, need to be investigated further
#if defined(__EMSCRIPTEN__)
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
#endif
            return { accumulatedRequestedWork, performedWork, performedWork > 0 ? work::Status::OK : lastStatus };
        } else {
            return workInternal(requested_work);
        }
    }
};

template<typename Derived, typename... Arguments>
inline std::atomic_size_t Block<Derived, Arguments...>::_unique_id_counter{ 0_UZ };
} // namespace gr

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, typename... Arguments), (gr::Block<T, Arguments...>), numerator, denominator, stride, unique_name, name, meta_information);

namespace gr {

/**
 * @brief a short human-readable/markdown description of the node -- content is not contractual and subject to change
 */
template<BlockLike TBlock>
[[nodiscard]] /*constexpr*/ std::string
blockDescription() noexcept {
    using DerivedBlock         = typename TBlock::derived_t;
    using ArgumentList         = typename TBlock::block_template_parameters;
    using Description          = typename ArgumentList::template find_or_default<is_doc, EmptyDoc>;
    using SupportedTypes       = typename ArgumentList::template find_or_default<is_supported_types, DefaultSupportedTypes>;
    constexpr bool kIsBlocking = ArgumentList::template contains<BlockingIO<true>> || ArgumentList::template contains<BlockingIO<false>>;

    // re-enable once string and constexpr static is supported by all compilers
    /*constexpr*/ std::string ret = fmt::format("# {}\n{}\n{}\n**supported data types:**", //
                                                gr::meta::type_name<DerivedBlock>(), Description::value._data,
                                                kIsBlocking ? "**BlockingIO**\n_i.e. potentially non-deterministic/non-real-time behaviour_\n" : "");
    gr::meta::typelist<SupportedTypes>::template apply_func([&](std::size_t index, auto &&t) {
        std::string type_name = gr::meta::type_name<decltype(t)>();
        ret += fmt::format("{}:{} ", index, type_name);
    });
    ret += fmt::format("\n**Parameters:**\n");
    if constexpr (refl::is_reflectable<DerivedBlock>()) {
        for_each(refl::reflect<DerivedBlock>().members, [&](auto member) {
            using RawType = std::remove_cvref_t<typename decltype(member)::value_type>;
            using Type    = unwrap_if_wrapped_t<RawType>;

            if constexpr (is_readable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string>) ) {
                if constexpr (is_annotated<RawType>()) {
                    const std::string type_name   = refl::detail::get_type_name<Type>().str();
                    const std::string member_name = get_display_name_const(member).str();
                    ret += fmt::format("{}{:10} {:<20} - annotated info: {} unit: [{}] documentation: {}{}\n",
                                       RawType::visible() ? "" : "_", //
                                       type_name,
                                       member_name, //
                                       RawType::description(), RawType::unit(),
                                       RawType::documentation(), //
                                       RawType::visible() ? "" : "_");
                } else {
                    const std::string type_name   = refl::detail::get_type_name<Type>().str();
                    const std::string member_name = get_display_name_const(member).str();
                    ret += fmt::format("_{:10} {}_\n", type_name, member_name);
                }
            }
        });
    }
    ret += fmt::format("\n~~Ports:~~\ntbd.");
    return ret;
}

namespace detail {
template<typename... Types>
struct BlockParameters {
    template<template<typename...> typename TBlock, typename RegisterInstance>
    void
    registerOn(RegisterInstance *plugin_instance, std::string block_type) const {
        plugin_instance->template add_block_type<TBlock, Types...>(block_type);
    }
};

template<template<typename...> typename TBlock, typename... TBlockParameters>
struct RegisterBlock {
    template<typename RegisterInstance>
    RegisterBlock(RegisterInstance *plugin_instance, std::string block_type) {
        auto add_block_type = [&]<typename Type> {
            if constexpr (meta::is_instantiation_of<Type, BlockParameters>) {
                Type().template registerOn<TBlock>(plugin_instance, block_type);
            } else {
                plugin_instance->template add_block_type<TBlock, Type>(block_type);
            }
        };
        ((add_block_type.template operator()<TBlockParameters>()), ...);
    }
};
} // namespace detail

} // namespace gr

template<>
struct fmt::formatter<gr::work::Status> {
    static constexpr auto
    parse(const format_parse_context &ctx) {
        const auto it = ctx.begin();
        if (it != ctx.end() && *it != '}') throw format_error("invalid format");
        return it;
    }

    template<typename FormatContext>
    auto
    format(const gr::work::Status &status, FormatContext &ctx) {
        using enum gr::work::Status;
        switch (status) {
        case ERROR: return fmt::format_to(ctx.out(), "ERROR");
        case INSUFFICIENT_OUTPUT_ITEMS: return fmt::format_to(ctx.out(), "INSUFFICIENT_OUTPUT_ITEMS");
        case INSUFFICIENT_INPUT_ITEMS: return fmt::format_to(ctx.out(), "INSUFFICIENT_INPUT_ITEMS");
        case DONE: return fmt::format_to(ctx.out(), "DONE");
        case OK: return fmt::format_to(ctx.out(), "OK");
        default: return fmt::format_to(ctx.out(), "UNKNOWN");
        }
        return fmt::format_to(ctx.out(), "UNKNOWN");
    }
};

template<>
struct fmt::formatter<gr::work::Result> {
    static constexpr auto
    parse(const format_parse_context &ctx) {
        const auto it = ctx.begin();
        if (it != ctx.end() && *it != '}') throw format_error("invalid format");
        return it;
    }

    template<typename FormatContext>
    auto
    format(const gr::work::Result &work_return, FormatContext &ctx) {
        return fmt::format_to(ctx.out(), "requested_work: {}, performed_work: {}, status: {}", work_return.requested_work, work_return.performed_work, fmt::format("{}", work_return.status));
    }
};

#endif // include guard
