#ifndef GNURADIO_BLOCK_HPP
#define GNURADIO_BLOCK_HPP

#include <limits>
#include <map>
#include <source_location>

#include <pmtv/pmt.hpp>

#include <fmt/format.h>
#ifdef __GNUC__
#pragma GCC diagnostic push // ignore warning of external libraries that from this lib-context we do not have any control over
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

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

namespace stdx = vir::stdx;
using gr::meta::fixed_string;

namespace lifecycle {
/**
 * @enum lifecycle::State enumerates the possible states of a `Scheduler` lifecycle.
 *
 * Transition between the following states is triggered by specific actions or events:
 * - `IDLE`: The initial state before the scheduler has been initialized.
 * - `INITIALISED`: The scheduler has been initialized and is ready to start running.
 * - `RUNNING`: The scheduler is actively running.
 * - `REQUESTED_PAUSE`: A pause has been requested, and the scheduler is in the process of pausing.
 * - `PAUSED`: The scheduler is paused and can be resumed or stopped.
 * - `REQUESTED_STOP`: A stop has been requested, and the scheduler is in the process of stopping.
 * - `STOPPED`: The scheduler has been stopped and can be reset or re-initialized.
 * - `ERROR`: An error state that can be reached from any state at any time, requiring a reset.
 *
 * @note All `Block<T>`-derived classes can optionally implement any subset of the lifecycle methods
 * (`start()`, `stop()`, `reset()`, `pause()`, `resume()`) to handle state changes of the `Scheduler`.
 *
 * State diagram:
 *
 *               Block<T>()              can be reached from
 *                  │                   anywhere and anytime.
 *            ┌─────┴────┐                   ┌────┴────┐
 *            │   IDLE   │                   │  ERROR  │
 *            └────┬─────┘                   └────┬────┘
 *                 │ init()                       │ reset()
 *                 v                              │
 *         ┌───────┴───────┐                      │
 *         │  INITIALISED  ├<─────────────────────┤
 *         └───────┬───────┘                      │
 *                 │ start()                      │
 *                 v                              │
 *   stop() ┌──────┴──────┐                       │  ╓
 *   ┌──────┤   RUNNING   ├<──────────┐           │  ║
 *   │      └─────┬───────┘           │           │  ║
 *   │            │ pause()           │           │  ║  isActive(lifecycle::State) ─> true
 *   │            v                   │ resume()  │  ║
 *   │  ┌─────────┴─────────┐   ┌─────┴─────┐     │  ║
 *   │  │  REQUESTED_PAUSE  ├──>┤  PAUSED   │     │  ║
 *   │  └──────────┬────────┘   └─────┬─────┘     │  ╙
 *   │             │ stop()           │ stop()    │
 *   │             v                  │           │
 *   │   ┌─────────┴────────┐         │           │  ╓
 *   └──>┤  REQUESTED_STOP  ├<────────┘           │  ║
 *       └────────┬─────────┘                     │  ║
 *                │                               │  ║  isShuttingDown(lifecycle::State) ─> true
 *                v                               │  ║
 *          ┌─────┴─────┐ reset()                 │  ║
 *          │  STOPPED  ├─────────────────────────┘  ║
 *          └─────┬─────┘                            ╙
 *                │
 *                v
 *            ~Block<T>()
 */
enum class State : char { IDLE, INITIALISED, RUNNING, REQUESTED_PAUSE, PAUSED, REQUESTED_STOP, STOPPED, ERROR };
using enum State;

inline constexpr bool
isActive(lifecycle::State state) noexcept {
    return state == RUNNING || state == REQUESTED_PAUSE || state == PAUSED;
}

inline constexpr bool
isShuttingDown(lifecycle::State state) noexcept {
    return state == REQUESTED_STOP || state == STOPPED;
}

} // namespace lifecycle

template<typename F>
constexpr void
simd_epilogue(auto width, F &&fun) {
    static_assert(std::has_single_bit(+width));
    auto w2 = std::integral_constant<std::size_t, width / 2>{};
    if constexpr (w2 > 0) {
        fun(w2);
        simd_epilogue(w2, std::forward<F>(fun));
    }
}

template<std::ranges::contiguous_range... Ts, typename Flag = stdx::element_aligned_tag>
constexpr auto
simdize_tuple_load_and_apply(auto width, const std::tuple<Ts...> &rngs, auto offset, auto &&fun, Flag f = {}) {
    using Tup = meta::simdize<std::tuple<std::ranges::range_value_t<Ts>...>, width>;
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

template<std::size_t Index, PortType portType, typename Self>
[[nodiscard]] constexpr auto &
inputPort(Self *self) noexcept {
    using TRequestedPortType = typename traits::block::ports_data<Self>::template for_type<portType>::input_ports::template at<Index>;
    if constexpr (traits::block::block_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::block::get_port_member_descriptor<Self, TRequestedPortType>;
        return member_descriptor()(*self);
    } else {
        return self->template getArgument<TRequestedPortType>();
    }
}

template<std::size_t Index, PortType portType, typename Self>
[[nodiscard]] constexpr auto &
outputPort(Self *self) noexcept {
    using TRequestedPortType = typename traits::block::ports_data<Self>::template for_type<portType>::output_ports::template at<Index>;
    if constexpr (traits::block::block_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::block::get_port_member_descriptor<Self, TRequestedPortType>;
        return member_descriptor()(*self);
    } else {
        return self->template getArgument<TRequestedPortType>();
    }
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
inputPort(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::block::all_input_ports<Self>>();
    if constexpr (Index == meta::default_message_port_index) {
        return self->msgIn;
    }
    return inputPort<Index, PortType::ANY, Self>(self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
outputPort(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::block::all_output_ports<Self>>();
    if constexpr (Index == meta::default_message_port_index) {
        return self->msgOut;
    }
    return outputPort<Index, PortType::ANY, Self>(self);
}

template<PortType portType, typename Self>
[[nodiscard]] constexpr auto
inputPorts(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return std::tie(inputPort<Idx, portType>(self)...);
    }(std::make_index_sequence<traits::block::ports_data<Self>::template for_type<portType>::input_ports::size()>());
}

template<PortType portType, typename Self>
[[nodiscard]] constexpr auto
outputPorts(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return std::tie(outputPort<Idx, portType>(self)...);
    }(std::make_index_sequence<traits::block::ports_data<Self>::template for_type<portType>::output_ports::size>());
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
        uint64_t oldCounter    = std::atomic_load_explicit(&encodedCounter, std::memory_order_acquire);
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
concept HasConstProcessOneFunction = traits::block::can_processOne<Derived> && gr::meta::is_const_member_function(&Derived::processOne);

template<typename Derived>
concept HasProcessBulkFunction = traits::block::can_processBulk<Derived>;

template<typename Derived>
concept HasRequiredProcessFunction = (HasProcessBulkFunction<Derived> or HasProcessOneFunction<Derived>) and(HasProcessOneFunction<Derived> + HasProcessBulkFunction<Derived>) == 1;

template<typename T>
concept ConsumableSpan = std::ranges::contiguous_range<T> and std::convertible_to<T, std::span<const std::remove_cvref_t<typename T::value_type>>> and requires(T &s) { s.consume(0); };

static_assert(ConsumableSpan<traits::block::detail::dummy_input_span<float>>);

template<typename T>
concept PublishableSpan = std::ranges::contiguous_range<T> and std::ranges::output_range<T, std::remove_cvref_t<typename T::value_type>>
                      and std::convertible_to<T, std::span<std::remove_cvref_t<typename T::value_type>>> and requires(T &s) { s.publish(0UZ); };

static_assert(PublishableSpan<traits::block::detail::dummy_output_span<float>>);

/**
 * @brief The 'Block<Derived>' is a base class for blocks that perform specific signal processing operations. It stores
 * references to its input and output 'ports' that can be zero, one, or many, depending on the use case.
 * As the base class for all user-defined blocks, it implements common convenience functions and a default public API
 * through the Curiously-Recurring-Template-Pattern (CRTP). For example:
 * @code
 * struct user_defined_block : Block<user_defined_block> {
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
 * struct user_defined_block : Block<user_defined_block, IN<T, 0, N_MAX, "in">, OUT<R, 0, N_MAX, "out">> {
 *   // implement one of the possible work or abstracted functions
 * };
 * @endcode
 * This implementation provides efficient compile-time static polymorphism (i.e. access to the ports, settings, etc. does
 * not require virtual functions or inheritance, which can have performance penalties in high-performance computing contexts).
 * Note: The template parameter '<Derived>' can be dropped once C++23's 'deducing this' is widely supported by compilers.
 *
 * The 'Block<Derived>' implementation provides simple defaults for users who want to focus on generic signal-processing
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
 *
 * In addition derived classes can optionally implement any subset of the lifecycle methods ( `start()`, `stop()`, `reset()`, `pause()`, `resume()`).
 * The Scheduler invokes these methods on each Block instance, if they are implemented, just before invoking its corresponding method of the same name.
 * @code
 * struct userBlock : public Block<userBlock> {
 * void start() {...} // Implement any startup logic required for the block within this method.
 * void stop() {...} // Use this method for handling any clean-up procedures.
 * void pause() {...} // Implement logic to temporarily halt the block's operation, maintaining its current state.
 * void resume() {...} // This method should contain logic to restart operations after a pause, continuing from the same state as when it was paused.
 * void reset() {...} // Reset the block's state to defaults in this method.
 * };
 * @endcode
 *
 * @tparam Derived the user-defined block CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 * @tparam Arguments NTTP list containing the compile-time defined port instances, setting structs, or other constraints.
 */
template<typename Derived, typename... Arguments>
class Block : protected std::tuple<Arguments...> {
    static std::atomic_size_t _unique_id_counter;
    template<typename T, gr::meta::fixed_string description = "", typename... Args>
    using A = Annotated<T, description, Args...>;

public:
    using base_t                     = Block<Derived, Arguments...>;
    using derived_t                  = Derived;
    using ArgumentsTypeList          = typename gr::meta::typelist<Arguments...>;
    using block_template_parameters  = meta::typelist<Arguments...>;
    using Description                = typename block_template_parameters::template find_or_default<is_doc, EmptyDoc>;
    using Resampling                 = ArgumentsTypeList::template find_or_default<is_resampling_ratio, ResamplingRatio<1UZ, 1UZ, true>>;
    using StrideControl              = ArgumentsTypeList::template find_or_default<is_stride, Stride<0UZ, true>>;
    constexpr static bool blockingIO = std::disjunction_v<std::is_same<BlockingIO<true>, Arguments>...> || std::disjunction_v<std::is_same<BlockingIO<false>, Arguments>...>;

    template<typename T>
    auto &
    getArgument() {
        return std::get<T>(*this);
    }

    template<typename T>
    const auto &
    getArgument() const {
        return std::get<T>(*this);
    }

    // TODO: These are not involved in move operations, might be a problem later
    alignas(hardware_destructive_interference_size) std::atomic<std::size_t> ioRequestedWork{ std::numeric_limits<std::size_t>::max() };
    alignas(hardware_destructive_interference_size) work::Counter ioWorkDone{};
    alignas(hardware_destructive_interference_size) std::atomic<work::Status> ioLastWorkStatus{ work::Status::OK };
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::Sequence> progress                         = std::make_shared<gr::Sequence>();
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::thread_pool::BasicThreadPool> ioThreadPool = std::make_shared<gr::thread_pool::BasicThreadPool>(
            "block_thread_pool", gr::thread_pool::TaskType::IO_BOUND, 2UZ, std::numeric_limits<uint32_t>::max());

    constexpr static TagPropagationPolicy tag_policy = TagPropagationPolicy::TPP_ALL_TO_ALL;

    //
    using RatioValue = std::conditional_t<Resampling::kIsConst, const std::size_t, std::size_t>;
    A<RatioValue, "numerator", Doc<"Top of resampling ratio (<1: Decimate, >1: Interpolate, =1: No change)">, Limits<1UZ, std::size_t(-1)>>      numerator   = Resampling::kNumerator;
    A<RatioValue, "denominator", Doc<"Bottom of resampling ratio (<1: Decimate, >1: Interpolate, =1: No change)">, Limits<1UZ, std::size_t(-1)>> denominator = Resampling::kDenominator;
    using StrideValue = std::conditional_t<StrideControl::kIsConst, const std::size_t, std::size_t>;
    A<StrideValue, "stride", Doc<"samples between data processing. <N for overlap, >N for skip, =0 for back-to-back.">> stride = StrideControl::kStride;

    //
    std::size_t                   stride_counter = 0UZ;
    std::atomic<lifecycle::State> state          = lifecycle::State::IDLE;

    // TODO: These are not involved in move operations, might be a problem later
    const std::size_t unique_id   = _unique_id_counter++;
    const std::string unique_name = fmt::format("{}#{}", gr::meta::type_name<Derived>(), unique_id);

    //
    A<std::string, "user-defined name", Doc<"N.B. may not be unique -> ::unique_name">>                            name = gr::meta::type_name<Derived>();
    A<property_map, "meta-information", Doc<"store non-graph-processing information like UI block position etc.">> meta_information;

    //
    constexpr static std::string_view description = static_cast<std::string_view>(Description::value);
    static_assert(std::atomic<lifecycle::State>::is_always_lock_free, "std::atomic<lifecycle::State> is not lock-free");

    // TODO: C++26 make sure these are not reflected
    // We support ports that are template parameters or reflected member variables,
    // so these are handled in a special way
    MsgPortInNamed<"__Builtin">  msgIn;
    MsgPortOutNamed<"__Builtin"> msgOut;

    struct PortsStatus {
        std::size_t in_min_samples{ 1UZ };                                             // max of `port.min_samples()` of all input ports
        std::size_t in_max_samples{ std::numeric_limits<std::size_t>::max() };         // min of `port.max_samples()` of all input ports
        std::size_t in_available{ std::numeric_limits<std::size_t>::max() };           // min of `port.streamReader().available()` of all input ports
        std::size_t nSamplesToNextTag{ std::numeric_limits<std::size_t>::max() };      // min distance to next Tag
        std::size_t nSamplesToNextTagAfter{ std::numeric_limits<std::size_t>::max() }; // min distance after the next Tag
        std::size_t nSamplesToEosTag{ std::numeric_limits<std::size_t>::max() };       // min of `port.samples_to_eos_tag` of all input ports

        std::size_t out_min_samples{ 1UZ };                                     // max of `port.min_samples()` of all output ports
        std::size_t out_max_samples{ std::numeric_limits<std::size_t>::max() }; // min of `port.max_samples()` of all output ports
        std::size_t out_available{ std::numeric_limits<std::size_t>::max() };   // min of `port.streamWriter().available()` of all input ports

        std::size_t in_samples{ 0UZ };  // number of input samples to process
        std::size_t out_samples{ 0UZ }; // number of output samples, calculated based on `numerator` and `denominator`

        bool in_at_least_one_port_has_data{ false }; // at least one port has data
        bool in_at_least_one_tag_available{ false }; // at least one port has a tag

        bool has_sync_input_ports{ false };  // if all ports are async, status is not important
        bool has_sync_output_ports{ false }; // if all ports are async, status is not important

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
    bool _output_tags_changed = false;
    Tag  _mergedInputTag{};

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
        ports_status = PortsStatus();
        // TODO: recheck definition of denominator vs. numerator w.r.t. up-/down-sampling
        //  ports_status.in_min_samples  = denominator;
        //  ports_status.out_min_samples = numerator;

        auto adjust_for_input_port = [&ps = ports_status]<PortLike Port>(Port &port) {
            if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                if (!port.isConnected()) {
                    return;
                }
                ps.has_sync_input_ports          = true;
                ps.in_min_samples                = std::max(ps.in_min_samples, port.min_samples);
                ps.in_max_samples                = std::min(ps.in_max_samples, port.max_samples);
                ps.in_available                  = std::min(ps.in_available, port.streamReader().available());
                ps.nSamplesToNextTag             = std::min(ps.nSamplesToNextTag, nSamplesUntilNextTag(port).value_or(std::numeric_limits<std::size_t>::max()));
                ps.nSamplesToNextTagAfter        = std::min(ps.nSamplesToNextTagAfter, nSamplesUntilNextTag(port, 1).value_or(std::numeric_limits<std::size_t>::max())); /* 1: in case nextTag == 0 */
                ps.nSamplesToEosTag              = std::min(ps.nSamplesToEosTag, samples_to_eos_tag(port).value_or(std::numeric_limits<std::size_t>::max()));
                ps.in_at_least_one_port_has_data = ps.in_at_least_one_port_has_data | (port.streamReader().available() > 0);
                ps.in_at_least_one_tag_available = ps.in_at_least_one_port_has_data | (port.tagReader().available() > 0);
            }
        };
        for_each_port([&adjust_for_input_port](PortLike auto &port) { adjust_for_input_port(port); }, inputPorts<PortType::STREAM>(&self()));

        auto adjust_for_output_port = [&ps = ports_status]<PortLike Port>(Port &port) {
            if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                ps.has_sync_output_ports = true;
                ps.out_min_samples       = std::max(ps.out_min_samples, port.min_samples);
                ps.out_max_samples       = std::min(ps.out_max_samples, port.max_samples);
                ps.out_available         = std::min(ps.out_available, port.streamWriter().available());
            }
        };
        for_each_port([&adjust_for_output_port](PortLike auto &port) { adjust_for_output_port(port); }, outputPorts<PortType::STREAM>(&self()));

        ports_status.in_samples = ports_status.in_available;
        if (ports_status.in_samples < ports_status.in_min_samples) ports_status.in_samples = 0;
        if (ports_status.in_samples > ports_status.in_max_samples) ports_status.in_samples = ports_status.in_max_samples;

        // By default N-in == N-out
        // TODO: adjust `samples_to_proceed` to output limits?
        ports_status.out_samples = ports_status.in_samples;

        if (ports_status.has_sync_input_ports && ports_status.in_min_samples > ports_status.in_max_samples)
            throw std::invalid_argument(fmt::format("Min samples for input ports ({}) is larger then max samples for input ports ({})", ports_status.in_min_samples, ports_status.in_max_samples));
        if (ports_status.has_sync_output_ports && ports_status.out_min_samples > ports_status.out_max_samples)
            throw std::invalid_argument(fmt::format("Min samples for output ports ({}) is larger then max samples for output ports ({})", ports_status.out_min_samples, ports_status.out_max_samples));

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
        : _settings(std::make_unique<BasicSettings<Derived>>(*static_cast<Derived *>(this))) { // N.B. safe delegated use of this (i.e. not used during construction)
        if (init_parameter.size() != 0) {
            const auto failed = settings().set(init_parameter);
            if (!failed.empty()) {
                throw std::invalid_argument("Settings not applied successfully");
            }
        }
    }

    Block(Block &&other) noexcept
        : std::tuple<Arguments...>(std::move(other))
        , numerator(std::move(other.numerator))
        , denominator(std::move(other.denominator))
        , stride(std::move(other.stride))
        , stride_counter(std::move(other.stride_counter))
        , state(other.state.load()) // atomic, not moving
        , msgIn(std::move(other.msgIn))
        , msgOut(std::move(other.msgOut))
        , ports_status(std::move(other.ports_status))
        , _output_tags_changed(std::move(other._output_tags_changed))
        , _mergedInputTag(std::move(other._mergedInputTag))
        , _settings(std::move(other._settings)) {}

    // There are a few const or conditionally const member variables,
    // we can not have a move-assignment that is equivalent to
    // the move constructor
    Block &
    operator=(Block &&other) = delete;

    ~Block() { // NOSONAR -- need to request the (potentially) running ioThread to stop
        if (lifecycle::isActive(std::atomic_load_explicit(&state, std::memory_order_acquire))) {
            std::atomic_store_explicit(&state, lifecycle::State::REQUESTED_STOP, std::memory_order_release);
            state.notify_all();
        }
        if (isBlocking()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // wait for done
        for (auto actualState = std::atomic_load_explicit(&state, std::memory_order_acquire); lifecycle::isActive(actualState);
             actualState      = std::atomic_load_explicit(&state, std::memory_order_acquire)) {
            state.wait(actualState);
        }
        std::atomic_store_explicit(&state, lifecycle::State::STOPPED, std::memory_order_release);
        state.notify_all();
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
        traits::block::all_input_ports<Derived>::for_each(setPortName);
        traits::block::all_output_ports<Derived>::for_each(setPortName);

        // Handle settings
        // important: these tags need to be queued because at this stage the block is not yet connected to other downstream blocks
        if (const auto forward_parameters = settings().applyStagedParameters(); !forward_parameters.empty()) {
            if constexpr (Derived::tag_policy == TagPropagationPolicy::TPP_ALL_TO_ALL) {
                publishTag(forward_parameters);
            }
        }

        // store default settings -> can be recovered with 'resetDefaults()'
        settings().storeDefaults();
        state = lifecycle::State::INITIALISED;
    }

    template<gr::meta::array_or_vector_type Container>
    [[nodiscard]] constexpr std::size_t
    availableInputSamples(Container &data) const noexcept {
        if constexpr (gr::meta::vector_type<Container>) {
            data.resize(traits::block::stream_input_port_types<Derived>::size);
        } else if constexpr (gr::meta::array_type<Container>) {
            static_assert(std::tuple_size<Container>::value >= traits::block::stream_input_port_types<Derived>::size);
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
                inputPorts<PortType::STREAM>(&self()));
        return traits::block::stream_input_port_types<Derived>::size;
    }

    template<gr::meta::array_or_vector_type Container>
    [[nodiscard]] constexpr std::size_t
    availableOutputSamples(Container &data) const noexcept {
        if constexpr (gr::meta::vector_type<Container>) {
            data.resize(traits::block::stream_output_port_types<Derived>::size);
        } else if constexpr (gr::meta::array_type<Container>) {
            static_assert(std::tuple_size<Container>::value >= traits::block::stream_output_port_types<Derived>::size);
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
                outputPorts<PortType::STREAM>(&self()));
        return traits::block::stream_output_port_types<Derived>::size;
    }

    [[nodiscard]] constexpr bool
    isBlocking() const noexcept {
        return blockingIO;
    }

    [[nodiscard]] constexpr bool
    input_tags_present() const noexcept {
        return !_mergedInputTag.map.empty();
    };

    [[nodiscard]] Tag
    mergedInputTag() const noexcept {
        return _mergedInputTag;
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

    constexpr void
    checkParametersAndThrowIfNeeded() {
        constexpr bool kIsSourceBlock = traits::block::stream_input_port_types<Derived>::size == 0;
        constexpr bool kIsSinkBlock   = traits::block::stream_output_port_types<Derived>::size == 0;

        if constexpr (Resampling::kEnabled) {
            static_assert(!kIsSinkBlock, "Decimation/interpolation is not available for sink blocks. Remove 'ResamplingRatio<>' from the block definition.");
            static_assert(!kIsSourceBlock, "Decimation/interpolation is not available for source blocks. Remove 'ResamplingRatio<>' from the block definition.");
            static_assert(HasProcessBulkFunction<Derived>, "Blocks which allow decimation/interpolation must implement processBulk(...) method. Remove 'ResamplingRatio<>' from the block definition.");
        } else {
            if (numerator != 1UZ || denominator != 1UZ) {
                throw std::runtime_error(fmt::format("Block is not defined as `ResamplingRatio<>`, but numerator = {}, denominator = {}, they both must equal to 1.", numerator, denominator));
            }
        }

        if constexpr (StrideControl::kEnabled) {
            static_assert(!kIsSourceBlock, "Stride is not available for source blocks. Remove 'Stride<>' from the block definition.");
        } else {
            if (stride != 0UZ) {
                throw std::runtime_error(fmt::format("Block is not defined as `Stride<>`, but stride = {}, it must equal to 0.", stride));
            }
        }
    }

    void
    write_to_outputs(std::size_t available_values_count, auto &writers_tuple) noexcept {
        if constexpr (traits::block::stream_output_ports<Derived>::size > 0) {
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
        if constexpr (traits::block::stream_input_ports<Derived>::size > 0) {
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
                    inputPorts<PortType::STREAM>(&self));
        }
        return success;
    }

    template<typename... Ts>
    constexpr auto
    invoke_processOne(std::size_t offset, Ts &&...inputs) {
        if constexpr (traits::block::stream_output_ports<Derived>::size == 0) {
            invokeProcessOneWithOrWithoutOffset(self(), offset, std::forward<Ts>(inputs)...);
            return std::tuple{};
        } else if constexpr (traits::block::stream_output_ports<Derived>::size == 1) {
            return std::tuple{ invokeProcessOneWithOrWithoutOffset(self(), offset, std::forward<Ts>(inputs)...) };
        } else {
            return invokeProcessOneWithOrWithoutOffset(self(), offset, std::forward<Ts>(inputs)...);
        }
    }

    template<typename... Ts>
    constexpr auto
    invoke_processOne_simd(std::size_t offset, auto width, Ts &&...input_simds) {
        if constexpr (sizeof...(Ts) == 0) {
            if constexpr (traits::block::stream_output_ports<Derived>::size == 0) {
                self().processOne_simd(offset, width);
                return std::tuple{};
            } else if constexpr (traits::block::stream_output_ports<Derived>::size == 1) {
                return std::tuple{ self().processOne_simd(offset, width) };
            } else {
                return self().processOne_simd(offset, width);
            }
        } else {
            return invoke_processOne(offset, std::forward<Ts>(input_simds)...);
        }
    }

    constexpr void
    forwardTags() noexcept {
        if (input_tags_present()) {
            // clear temporary cached input tags after processing - won't be needed after this
            _mergedInputTag.map.clear();
        }

        for_each_port([](PortLike auto &outPort) noexcept { outPort.publishPendingTags(); }, outputPorts<PortType::STREAM>(&self()));
        _output_tags_changed = false;
    }

    constexpr void
    updateInputAndOutputTags(Tag::signed_index_type untilOffset = 0) noexcept {
        if constexpr (HasProcessOneFunction<Derived>) {
            ports_status.in_samples  = 1; // N.B. limit to one so that only one process_on(...) invocation receives the tag
            ports_status.out_samples = 1;
        }
        for_each_port(
                [untilOffset, this]<typename Port>(Port &input_port) noexcept {
                    auto mergeSrcMapInto = [](const property_map &sourceMap, property_map &destinationMap) {
                        assert(&sourceMap != &destinationMap);
                        for (const auto &[key, value] : sourceMap) {
                            destinationMap.insert_or_assign(key, value);
                        }
                    };

                    const Tag mergedPortTags = input_port.getTag(untilOffset);
                    mergeSrcMapInto(mergedPortTags.map, _mergedInputTag.map);
                },
                inputPorts<PortType::STREAM>(&self()));

        if (!mergedInputTag().map.empty()) {
            settings().autoUpdate(mergedInputTag().map); // apply tags as new settings if matching
            if constexpr (Derived::tag_policy == TagPropagationPolicy::TPP_ALL_TO_ALL) {
                for_each_port([this](PortLike auto &outPort) noexcept { outPort.publishTag(mergedInputTag().map, 0); }, outputPorts<PortType::STREAM>(&self()));
            }
            if (mergedInputTag().map.contains(gr::tag::END_OF_STREAM)) {
                requestStop();
            }
        }
    }

    constexpr void
    updateOutputTagsWithSettingParametersIfNeeded() {
        if (settings().changed()) {
            if (const auto forward_parameters = settings().applyStagedParameters(); !forward_parameters.empty()) {
                for_each_port([&forward_parameters](PortLike auto &outPort) { outPort.publishTag(forward_parameters, 0); }, outputPorts<PortType::STREAM>(&self()));
            }
            settings()._changed.store(false);
        }
    }

    constexpr work::Status
    doResampling() {
        if (numerator != 1UZ || denominator != 1UZ) {
            // TODO: this ill-defined checks can be done only once after parameters were changed
            const double ratio          = static_cast<double>(numerator) / static_cast<double>(denominator);
            bool         is_ill_defined = (denominator > ports_status.in_max_samples)                                                            //
                               || (static_cast<double>(ports_status.in_min_samples) * ratio > static_cast<double>(ports_status.out_max_samples)) //
                               || (static_cast<double>(ports_status.in_max_samples) * ratio < static_cast<double>(ports_status.out_min_samples));

            if (denominator > ports_status.in_max_samples) { // TODO: convert to proper error message send to msgOut
                fmt::println(stderr, "configuration error for block {}: denominator {} > ports_status.in_max_samples {}", name, denominator, ports_status.in_max_samples);
                assert(false && "denominator needs to be <= max InPort sample constraints");
                return work::Status::ERROR;
            }
            if (static_cast<double>(ports_status.in_min_samples) * ratio > static_cast<double>(ports_status.out_max_samples)) { // TODO: convert to proper error message send to msgOut
                fmt::println(stderr, "configuration error for block {}:  ports_status.in_min_samples * ratio {} > ports_status.out_max_samples {}, ratio (num {}/den {} = {})", name,
                             static_cast<double>(ports_status.in_min_samples) * ratio, static_cast<double>(ports_status.out_max_samples), numerator, denominator, ratio);
                assert(false && "reduced min-required input sample needs to be <= max OutPort sample constraints");
                return work::Status::ERROR;
            }
            if (static_cast<double>(ports_status.in_max_samples) * ratio < static_cast<double>(ports_status.out_min_samples)) { // TODO: convert to proper error message send to msgOut
                fmt::println(stderr, "configuration error for block {}:  ports_status.in_max_samples * ratio {} > ports_status.out_min_samples {}, ratio (num {}/den {} = {})", name,
                             static_cast<double>(ports_status.in_max_samples) * ratio, static_cast<double>(ports_status.out_min_samples), numerator, denominator, ratio);
                assert(false && "reduced max-required input sample needs to be <= min OutPort sample constraints");
                return work::Status::ERROR;
            }

            if (is_ill_defined) {
                assert(!is_ill_defined && "ill-defined");
                return work::Status::ERROR;
            }

            ports_status.in_samples = static_cast<std::size_t>(ports_status.in_samples / denominator) * denominator; // remove reminder

            const std::size_t out_min_limit = ports_status.out_min_samples;
            const std::size_t out_max_limit = std::min(ports_status.out_available, ports_status.out_max_samples);

            std::size_t in_min_samples = static_cast<std::size_t>(static_cast<double>(out_min_limit) / ratio);
            if (in_min_samples % denominator != 0) in_min_samples += denominator;
            const std::size_t in_min_wo_remainder = (in_min_samples / denominator) * denominator;

            const std::size_t in_max_samples      = static_cast<std::size_t>(static_cast<double>(out_max_limit) / ratio);
            const std::size_t in_max_wo_remainder = (in_max_samples / denominator) * denominator;

            if (ports_status.in_samples < in_min_wo_remainder) {
                return work::Status::INSUFFICIENT_INPUT_ITEMS;
            }

            if (in_min_wo_remainder < in_max_wo_remainder) {
                ports_status.in_samples = std::clamp(ports_status.in_samples, in_min_wo_remainder, in_max_wo_remainder);
            } else {
                ports_status.in_samples = std::min(ports_status.in_samples, std::min(in_min_wo_remainder, in_max_wo_remainder));
            }
            ports_status.out_samples = numerator * (ports_status.in_samples / denominator);
        }
        return work::Status::OK;
    }

    constexpr auto
    prepareInputStreams() {
        return meta::tuple_transform(
                [&self = self(), sync_in_samples = self().ports_status.in_samples]<typename PortOrCollection>(PortOrCollection &input_port_or_collection) noexcept {
                    auto in_samples = sync_in_samples;

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
                        std::vector<value_span> result{};
                        std::transform(input_port_or_collection.begin(), input_port_or_collection.end(), std::back_inserter(result), process_single_port);
                        return result;
                    }
                },
                inputPorts<PortType::STREAM>(&self()));
    }

    constexpr auto
    prepareOutputStreams() {
        return meta::tuple_transform(
                [&self = self(), sync_out_samples = ports_status.out_samples]<typename PortOrCollection>(PortOrCollection &output_port_or_collection) noexcept {
                    auto out_samples = sync_out_samples;

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
                        std::vector<value_span> result{};
                        std::transform(output_port_or_collection.begin(), output_port_or_collection.end(), std::back_inserter(result), process_single_port);
                        return result;
                    }
                },
                outputPorts<PortType::STREAM>(&self()));
    }

    inline constexpr void
    publishTag(property_map &&tag_data, Tag::signed_index_type tagOffset = -1) noexcept {
        for_each_port([tag_data = std::move(tag_data), tagOffset](PortLike auto &outPort) { outPort.publishTag(tag_data, tagOffset); }, outputPorts<PortType::STREAM>(&self()));
    }

    inline constexpr void
    publishTag(const property_map &tag_data, Tag::signed_index_type tagOffset = -1) noexcept {
        for_each_port([&tag_data, tagOffset](PortLike auto &outPort) { outPort.publishTag(tag_data, tagOffset); }, outputPorts<PortType::STREAM>(&self()));
    }

    constexpr void
    requestStop() noexcept {
        std::atomic_store_explicit(&this->state, lifecycle::State::REQUESTED_STOP, std::memory_order_release);
        this->state.notify_all();
    }

    constexpr void
    processScheduledMessages() {
        auto processPort = [this]<PortLike TPort>(TPort &inPort) {
            const auto available = inPort.streamReader().available();
            if constexpr (traits::block::can_processMessagesForPort<Derived, TPort>) {
                if (available > 0) {
                    self().processMessages(inPort, inPort.streamReader().get(available));
                }
            }
            if (available > 0) {
                if (auto consumed = inPort.streamReader().consume(available); !consumed) {
                    throw fmt::format("Could not consume the messages from the message port");
                }
            }
        };
        processPort(msgIn);
        for_each_port(processPort, inputPorts<PortType::MESSAGE>(&self()));
    }

    void
    emitMessage(auto &port, Message message) {
        message[gr::message::key::Sender] = unique_name;
        port.streamWriter().publish([&](auto &out) { out[0] = std::move(message); }, 1);
    }

protected:
    /**
     * @brief
     * @return struct { std::size_t produced_work, work_return_t}
     */
    work::Result
    workInternal(std::size_t requested_work) {
        using gr::work::Status;
        using TInputTypes  = traits::block::stream_input_port_types<Derived>;
        using TOutputTypes = traits::block::stream_output_port_types<Derived>;

        constexpr bool kIsSourceBlock = TInputTypes::size == 0;
        constexpr bool kIsSinkBlock   = TOutputTypes::size == 0;

        if (std::atomic_load_explicit(&state, std::memory_order_acquire) == lifecycle::State::STOPPED) {
            return { requested_work, 0UZ, work::Status::DONE };
        }

        // TODO: these checks can be moved to setting changed
        checkParametersAndThrowIfNeeded();

        updatePortsStatus();

        if constexpr (kIsSourceBlock) {
            // TODO: available_samples() methods are no longer needed for Source blocks, the are presently still/only needed for merge graph/block.
            // TODO: Review if they can be removed.
            ports_status.nSamplesToNextTag = std::numeric_limits<std::size_t>::max(); // no tags to processed for source node
            if constexpr (requires(const Derived &d) {
                              { available_samples(d) } -> std::same_as<std::size_t>;
                          }) {
                // the (source) node wants to determine the number of samples to process
                std::size_t samples_to_process = available_samples(self());
                if (samples_to_process == 0) {
                    return { requested_work, 0UZ, work::Status::OK };
                }
                if (!ports_status.enoughSamplesForOutputPorts(samples_to_process)) {
                    return { requested_work, 0UZ, work::Status::INSUFFICIENT_INPUT_ITEMS };
                }
                if (!ports_status.spaceAvailableOnOutputPorts(samples_to_process)) {
                    return { requested_work, 0UZ, work::Status::INSUFFICIENT_OUTPUT_ITEMS };
                }
                ports_status.in_samples  = std::min(samples_to_process, requested_work);
                ports_status.out_samples = ports_status.in_samples;

            } else if constexpr (kIsSinkBlock) {
                // no input or output buffers, derive from internal "buffer sizes" (i.e. what the
                // buffer size would be if the node were not merged)
                constexpr std::size_t chunkSize = Derived::merged_work_chunk_size();
                static_assert(chunkSize != std::dynamic_extent && chunkSize > 0, "At least one internal port must define a maximum number of samples or the non-member/hidden "
                                                                                 "friend function `available_samples(const BlockType&)` must be defined.");
                ports_status.in_samples  = std::min(chunkSize, requested_work);
                ports_status.out_samples = ports_status.in_samples;

            } else {
                // derive value from output buffer size
                std::size_t samplesToProcess = std::min(ports_status.out_available, ports_status.out_max_samples);
                if (!ports_status.enoughSamplesForOutputPorts(samplesToProcess)) {
                    return { requested_work, 0UZ, work::Status::INSUFFICIENT_OUTPUT_ITEMS };
                }
                ports_status.in_samples  = std::min(samplesToProcess, requested_work);
                ports_status.out_samples = ports_status.in_samples;
                // space_available_on_output_ports is true by construction of samplesToProcess
            }

        } else {                                                                         // end of kIsSourceBlock
            ports_status.in_samples = std::min(ports_status.in_samples, requested_work); // clamp input to scheduler work constraint
#ifdef _DEBUG
            fmt::println("block {} - mark1 - in {} -> out {}", name, ports_status.in_samples, ports_status.out_samples);
#endif
            if constexpr (!Resampling::kEnabled) { // clamp input until (excluding) next tag unless the read-position is on the tag
                // N.B. minimum size '1' because we could have two tags on adjacent samples.
                ports_status.in_samples  = ports_status.nSamplesToNextTag == 0 ? std::min(ports_status.in_samples, std::max(ports_status.nSamplesToNextTagAfter, 1UZ))
                                                                               : std::min(ports_status.in_samples, ports_status.nSamplesToNextTag);
                ports_status.out_samples = ports_status.in_samples;
            }
#ifdef _DEBUG
            fmt::println("block {} - mark2 - in {} -> out {}", name, ports_status.in_samples, ports_status.out_samples);
#endif
            ports_status.in_samples = std::min(ports_status.in_samples, ports_status.nSamplesToEosTag); // clamp input until EOS tag
            // TODO: recheck definitions of numerator and denominator
            //   Provided we we define N := <numerator> and M := <denominator>, do we expect (assuming a simple in->out block) that
            //  a) for N input samples M output samples are produced, or
            //  b) for a given number of input samples X -> X * N/M output samples are produced.
            //  Some of the logic conditions vary depending on whether we assume 'a)' or 'b)'.
            const bool isEOSTagPresent = ports_status.nSamplesToEosTag == 0                                                             //
                                      || ports_status.nSamplesToEosTag < std::max(ports_status.in_samples, ports_status.in_min_samples) //
                                      || ports_status.nSamplesToEosTag < numerator;
#ifdef _DEBUG
            if (isEOSTagPresent) {
                fmt::println("##block {} received EOS tag at {} < in_min_samples {}", name, ports_status.nSamplesToEosTag, ports_status.in_min_samples);
            }
#endif

            if (isEOSTagPresent || lifecycle::isShuttingDown(state)) {
                state = lifecycle::State::STOPPED;
                state.notify_all();
#ifdef _DEBUG
                fmt::println("##block {} received EOS tag at {} in_samples {} -> lifecycle::State::STOPPED", name, ports_status.nSamplesToEosTag, ports_status.in_samples);
#endif
                updateInputAndOutputTags(static_cast<Tag::signed_index_type>(ports_status.in_min_samples));
                updateOutputTagsWithSettingParametersIfNeeded();
                return { requested_work, 0UZ, work::Status::DONE };
            }

            if constexpr (Resampling::kEnabled) {
                const auto resamplingStatus = doResampling();
                if (resamplingStatus != work::Status::OK) {
                    if (resamplingStatus == work::Status::INSUFFICIENT_INPUT_ITEMS || isEOSTagPresent) {
                        std::atomic_store_explicit(&this->state, lifecycle::State::STOPPED, std::memory_order_release);
                        state.notify_all();
                        updateInputAndOutputTags(static_cast<Tag::signed_index_type>(ports_status.in_min_samples));
                        updateOutputTagsWithSettingParametersIfNeeded();
                        forwardTags();
                        //  EOS is not at 0 position and thus not read by updateInputAndOutputTags(), we need to publish new EOS
                        publishTag({ { gr::tag::END_OF_STREAM, true } }, 0);
                        return { requested_work, 0UZ, work::Status::DONE };
                    }
                    return { requested_work, 0UZ, resamplingStatus };
                }
            }
#ifdef _DEBUG
            fmt::println("block {} - mark3 - in {} -> out {}", name, ports_status.in_samples, ports_status.out_samples);
#endif
            if (ports_status.has_sync_input_ports && ports_status.in_available == 0) {
                return { requested_work, 0UZ, work::Status::INSUFFICIENT_INPUT_ITEMS };
            }

            // TODO: special case for portsStatus.in_samples == 0 ?
            if (!ports_status.enoughSamplesForOutputPorts(ports_status.out_samples)) { // !(out_samples > out_min_samples)
#ifdef _DEBUG
                fmt::println("mark1 - state {} - ports_status.nSamplesToEosTag = {} in/out: {}/{} available {} {} numerator {} denominator {}, in_min_samples {} out_min_samples {} synchOut: {}  "
                             "isEOSTagPresent: {}",                                              //
                             magic_enum::enum_name(state.load()), ports_status.nSamplesToEosTag, //
                             ports_status.in_samples, ports_status.out_samples,                  //
                             ports_status.in_available, ports_status.out_available,              //
                             this->numerator, this->denominator,                                 //
                             ports_status.in_min_samples, ports_status.out_min_samples,          //
                             ports_status.has_sync_output_ports, isEOSTagPresent);
#endif
                return { requested_work, 0UZ, work::Status::INSUFFICIENT_INPUT_ITEMS };
            }
            if (!ports_status.spaceAvailableOnOutputPorts(ports_status.out_samples)) {
                return { requested_work, 0UZ, work::Status::INSUFFICIENT_OUTPUT_ITEMS };
            }
        }

        if (ports_status.nSamplesToNextTag == 0) {
            updateInputAndOutputTags(0);
        }

        updateOutputTagsWithSettingParametersIfNeeded();

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

        std::size_t nSamplesToConsume = ports_status.in_samples; // default stride == 0
        if constexpr (StrideControl::kEnabled) {
            if (stride != 0UZ) {
                const bool firstTimeStride = stride_counter == 0;
                if (firstTimeStride) {
                    // sample processing are done as usual, portsStatus.in_samples samples will be processed
                    if (stride.value > stride_counter + ports_status.in_available) { // stride can not be consumed at once -> start stride_counter
                        stride_counter += ports_status.in_available;
                        nSamplesToConsume = ports_status.in_available;
                    } else { // if the stride can be consumed at once -> no stride_counter is needed
                        stride_counter    = 0;
                        nSamplesToConsume = stride.value;
                    }
                } else {
                    // |====================|...|====================|==============----| -> ====== is the stride
                    //   ^first                    ^we are here (1)  or ^here (2)
                    // if it is not the "first time" stride -> just consume (1) all samples or (2) missing rest of the samples
                    // forward tags but no additional sample processing are done ->return
                    if (stride.value > stride_counter + ports_status.in_available) {
                        stride_counter += ports_status.in_available;
                        nSamplesToConsume = ports_status.in_available;
                    } else { // stride is at the end -> reset stride_counter
                        nSamplesToConsume = stride.value - stride_counter;
                        stride_counter    = 0;
                    }
                    const bool success = consumeReaders(self(), nSamplesToConsume);
                    return { requested_work, nSamplesToConsume, success ? work::Status::OK : work::Status::ERROR };
                }
            }
        }

        const auto inputSpans   = prepareInputStreams();
        auto       writersTuple = prepareOutputStreams();

        if constexpr (HasProcessBulkFunction<Derived>) {
            // cannot use std::apply because it requires tuple_cat(inputSpans, writersTuple). The latter doesn't work because writersTuple isn't copyable.
            const work::Status ret = [&]<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) {
                return self().processBulk(std::get<InIdx>(inputSpans)..., std::get<OutIdx>(writersTuple)...);
            }(std::make_index_sequence<traits::block::stream_input_ports<Derived>::size>(), std::make_index_sequence<traits::block::stream_output_ports<Derived>::size>());

            forwardTags();
            if constexpr (kIsSourceBlock) {
                if (ret == work::Status::DONE) {
                    std::atomic_store_explicit(&this->state, lifecycle::State::STOPPED, std::memory_order_release);
                    state.notify_all();
                    publishTag({ { gr::tag::END_OF_STREAM, true } }, 0);
                    return { requested_work, ports_status.in_samples, work::Status::DONE };
                }
            }
            write_to_outputs(ports_status.out_samples, writersTuple);
            const bool success = consumeReaders(self(), nSamplesToConsume);
            return { requested_work, ports_status.in_samples, success ? ret : work::Status::ERROR };

        } else if constexpr (HasProcessOneFunction<Derived>) {
            std::size_t nOutSamplesBeforeRequestedStop = 0; // for the source block it consume only till END_OF_STREAM tag
            if (ports_status.in_samples != ports_status.out_samples) {
                throw std::logic_error(fmt::format("N input samples ({}) does not equal to N output samples ({}) for processOne() method.", ports_status.in_samples, ports_status.out_samples));
            }
            // handle processOne(...)
            using input_simd_types  = meta::simdize<typename TInputTypes::template apply<std::tuple>>;
            using output_simd_types = meta::simdize<typename TOutputTypes::template apply<std::tuple>>;

            std::integral_constant<std::size_t, (meta::simdize_size_v<input_simd_types> == 0 ? std::size_t(stdx::simd_abi::max_fixed_size<double>)
                                                                                             : std::min(std::size_t(stdx::simd_abi::max_fixed_size<double>), meta::simdize_size_v<input_simd_types> * 4))>
                    width{};

            if constexpr ((kIsSinkBlock or meta::simdize_size_v<output_simd_types> != 0) and ((kIsSourceBlock and requires(Derived &d) {
                                                                                                  { d.processOne_simd(width) };
                                                                                              }) or (meta::simdize_size_v<input_simd_types> != 0 and traits::block::can_processOne_simd<Derived>))) {
                // SIMD loop
                std::size_t i = 0;
                for (; i + width <= ports_status.in_samples; i += width) {
                    const auto &results = simdize_tuple_load_and_apply(width, inputSpans, i, [&](const auto &...input_simds) { return invoke_processOne_simd(i, width, input_simds...); });
                    meta::tuple_for_each([i](auto &output_range, const auto &result) { result.copy_to(output_range.data() + i, stdx::element_aligned); }, writersTuple, results);
                }
                simd_epilogue(width, [&](auto w) {
                    if (i + w <= ports_status.in_samples) {
                        const auto results = simdize_tuple_load_and_apply(w, inputSpans, i, [&](auto &&...input_simds) { return invoke_processOne_simd(i, w, input_simds...); });
                        meta::tuple_for_each([i](auto &output_range, auto &result) { result.copy_to(output_range.data() + i, stdx::element_aligned); }, writersTuple, results);
                        i += w;
                    }
                });
            } else {
                // Non-SIMD loop
                if constexpr (HasConstProcessOneFunction<Derived>) {
                    // processOne is const -> can process whole batch similar to SIMD-ised call
                    for (std::size_t i = 0; i < ports_status.in_samples; ++i) {
                        const auto results = std::apply([this, i](auto &...inputs) { return this->invoke_processOne(i, inputs[i]...); }, inputSpans);
                        meta::tuple_for_each([i](auto &output_range, auto &result) { output_range[i] = std::move(result); }, writersTuple, results);
                    }
                } else {
                    // processOne isn't const i.e. not a pure function w/o side effects -> need to evaluate state after each sample
                    for (std::size_t i = 0; i < ports_status.in_samples; ++i) {
                        const auto results = std::apply([this, i](auto &...inputs) { return this->invoke_processOne(i, inputs[i]...); }, inputSpans);
                        meta::tuple_for_each([i](auto &output_range, auto &result) { output_range[i] = std::move(result); }, writersTuple, results);
                        nOutSamplesBeforeRequestedStop++;
                        if (_output_tags_changed || lifecycle::isShuttingDown(std::atomic_load_explicit(&state, std::memory_order_acquire))) [[unlikely]] {
                            // emitted tag and/or requested to stop
                            break;
                        }
                    }
                }
            }

            //            if constexpr (kIsSourceBlock) {
            if (nOutSamplesBeforeRequestedStop > 0) {
                ports_status.out_samples = nOutSamplesBeforeRequestedStop;
                nSamplesToConsume        = nOutSamplesBeforeRequestedStop;
            }
            //            }

            forwardTags();
            write_to_outputs(ports_status.out_samples, writersTuple);
            const bool success = consumeReaders(self(), nSamplesToConsume);
            if (lifecycle::isShuttingDown(std::atomic_load_explicit(&state, std::memory_order_acquire))) [[unlikely]] {
                std::atomic_store_explicit(&this->state, lifecycle::State::STOPPED, std::memory_order_release);
                this->state.notify_all();
                publishTag({ { gr::tag::END_OF_STREAM, true } }, 0);
                return { requested_work, ports_status.in_samples, success ? work::Status::DONE : work::Status::ERROR };
            }

            // return { requested_work, ports_status.in_samples, success ? (lifecycle::isShuttingDown(std::atomic_load_explicit(&state, std::memory_order_acquire)) ? work::Status::DONE :
            // work::Status::OK) : work::Status::ERROR };
            return { requested_work, ports_status.in_samples, success ? work::Status::OK : work::Status::ERROR };
        } // processOne(...) handling
        //        else {
        //            static_assert(gr::meta::always_false<Derived>, "neither processBulk(...) nor processOne(...) implemented");
        //        }
        return { requested_work, 0UZ, work::Status::ERROR };
    } // end: work_return_t work_internal() noexcept { ..}

public:
    work::Status
    invokeWork()
        requires(blockingIO)
    {
        auto [work_requested, work_done, last_status] = workInternal(std::atomic_load_explicit(&ioRequestedWork, std::memory_order_acquire));
        ioWorkDone.increment(work_requested, work_done);
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
        processScheduledMessages();

        if constexpr (blockingIO) {
            constexpr bool useIoThread = std::disjunction_v<std::is_same<BlockingIO<true>, Arguments>...>;
            std::atomic_store_explicit(&ioRequestedWork, requested_work, std::memory_order_release);
            if (lifecycle::State expectedThreadState = lifecycle::State::INITIALISED; state.compare_exchange_strong(expectedThreadState, lifecycle::State::RUNNING, std::memory_order_acq_rel)) {
                if constexpr (useIoThread) { // use graph-provided ioThreadPool
                    ioThreadPool->execute([this]() {
                        assert(lifecycle::isActive(std::atomic_load_explicit(&state, std::memory_order_acquire)));
                        lifecycle::State actualThreadState = std::atomic_load_explicit(&state, std::memory_order_acquire);
                        while (lifecycle::isActive(actualThreadState)) {
                            // execute ten times before testing actual state -- minimises overhead atomic load to work execution if the latter is a noop or very fast to execute
                            for (std::size_t testState = 0UZ; testState < 10UZ; ++testState) {
                                if (invokeWork() == work::Status::DONE) {
                                    actualThreadState = lifecycle::State::REQUESTED_STOP;
                                    std::atomic_store_explicit(&this->state, lifecycle::State::REQUESTED_STOP, std::memory_order_release);
                                    state.notify_all();
                                    break;
                                }
                            }
                            actualThreadState = std::atomic_load_explicit(&state, std::memory_order_acquire);
                        }
                        std::atomic_store_explicit(&this->state, lifecycle::State::STOPPED, std::memory_order_release);
                        state.notify_all();
                    });
                } else { // use user-provided ioThreadPool
                    // let user call 'work' explicitly and set both 'ioWorkDone' and 'ioLastWorkStatus'
                }
            }
            if constexpr (!useIoThread) {
                const bool blockIsActive = lifecycle::isActive(state.load());
                if (!blockIsActive) {
                    publishTag({ { gr::tag::END_OF_STREAM, true } }, 0);
                    ioLastWorkStatus.exchange(work::Status::DONE, std::memory_order_relaxed);
                }
            }

            const auto &[accumulatedRequestedWork, performedWork] = ioWorkDone.getAndReset();
            // TODO: this is just "working" solution for deadlock with emscripten, need to be investigated further
#if defined(__EMSCRIPTEN__)
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
#endif
            return { accumulatedRequestedWork, performedWork, ioLastWorkStatus.load() };
        } else {
            return workInternal(requested_work);
        }
    }

    void
    processMessages(MsgPortInNamed<"__Builtin"> &port, std::span<const Message> messages) {
        if (std::addressof(port) != std::addressof(msgIn)) {
            fmt::print("{} got a message on a wrong port\n", self().unique_name);
            return;
        }

        for (const auto &message : messages) {
            const auto kind   = messageField<std::string>(message, gr::message::key::Kind).value_or(std::string{});
            const auto target = messageField<std::string>(message, gr::message::key::Target);

            if (target && !target->empty() && *target != self().unique_name) {
                continue;
            }

            if (kind == gr::message::kind::UpdateSettings) {
                const auto data   = messageField<property_map>(message, gr::message::key::Data).value();
                auto       notSet = settings().set(data);

                std::string keysNotSet;
                for (const auto &[k, v] : notSet) {
                    keysNotSet += " " + k;
                }

                Message settingsUpdated;
                settingsUpdated[gr::message::key::Kind] = gr::message::kind::SettingsChanged;
                settingsUpdated[gr::message::key::Data] = settings().get();

                if (!notSet.empty()) {
                    Message errorMessage;
                    errorMessage[gr::message::key::Kind]         = gr::message::kind::Error;
                    errorMessage[gr::message::key::Data]         = notSet;
                    settingsUpdated[gr::message::key::ErrorInfo] = std::move(errorMessage);
                }
                emitMessage(msgOut, std::move(settingsUpdated));
            }
        }
    }
};

template<typename Derived, typename... Arguments>
inline std::atomic_size_t Block<Derived, Arguments...>::_unique_id_counter{ 0UZ };
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
    gr::meta::typelist<SupportedTypes>::for_each([&](std::size_t index, auto &&t) {
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
        plugin_instance->template addBlockType<TBlock, Types...>(block_type);
    }
};

template<template<typename...> typename TBlock, typename... TBlockParameters>
struct RegisterBlock {
    template<typename RegisterInstance>
    RegisterBlock(RegisterInstance *plugin_instance, std::string block_type) {
        std::cout << "registerBlock " << block_type << std::endl;
        auto addBlockType = [&]<typename Type> {
            if constexpr (meta::is_instantiation_of<Type, BlockParameters>) {
                Type().template registerOn<TBlock>(plugin_instance, block_type);
            } else {
                plugin_instance->template addBlockType<TBlock, Type>(block_type);
            }
        };
        ((addBlockType.template operator()<TBlockParameters>()), ...);
    }
};
} // namespace detail

template<typename Function, typename Tuple, typename... Tuples>
auto
for_each_port(Function &&function, Tuple &&tuple, Tuples &&...tuples) {
    return gr::meta::tuple_for_each(
            [&function](auto &&...args) {
                (..., ([&function](auto &&arg) {
                     using ArgType = std::decay_t<decltype(arg)>;
                     if constexpr (traits::port::is_port_v<ArgType>) {
                         function(arg); // arg is a port, apply function directly
                     } else if constexpr (traits::port::is_port_collection_v<ArgType>) {
                         for (auto &port : arg) { // arg is a collection of ports, apply function to each port
                             function(port);
                         }
                     } else {
                         static_assert(gr::meta::always_false<Tuple>, "not a port or collection of ports");
                     }
                 }(args)));
            },
            std::forward<Tuple>(tuple), std::forward<Tuples>(tuples)...);
}

} // namespace gr

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
        return fmt::format_to(ctx.out(), "requested_work: {}, performed_work: {}, status: {}", work_return.requested_work, work_return.performed_work, magic_enum::enum_name(work_return.status));
    }
};

#endif // include guard
