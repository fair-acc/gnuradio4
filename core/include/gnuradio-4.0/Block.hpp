#ifndef GNURADIO_BLOCK_HPP
#define GNURADIO_BLOCK_HPP

#include <limits>
#include <map>
#include <source_location>

#include <pmtv/pmt.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/meta/typelist.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Sequence.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <gnuradio-4.0/Settings.hpp>
#include <gnuradio-4.0/annotated.hpp> // This needs to be included after fmt/format.h, as it defines formatters only if FMT_FORMAT_H_ is defined
#include <gnuradio-4.0/reflection.hpp>

#include <gnuradio-4.0/LifeCycle.hpp>

namespace gr {

namespace stdx = vir::stdx;
using gr::meta::fixed_string;

template<typename F>
constexpr void simd_epilogue(auto width, F&& fun) {
    static_assert(std::has_single_bit(+width));
    auto w2 = std::integral_constant<std::size_t, width / 2>{};
    if constexpr (w2 > 0) {
        fun(w2);
        simd_epilogue(w2, std::forward<F>(fun));
    }
}

template<std::ranges::contiguous_range... Ts, typename Flag = stdx::element_aligned_tag>
constexpr auto simdize_tuple_load_and_apply(auto width, const std::tuple<Ts...>& rngs, auto offset, auto&& fun, Flag f = {}) {
    using Tup = meta::simdize<std::tuple<std::ranges::range_value_t<Ts>...>, width>;
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return fun(std::tuple_element_t<Is, Tup>(std::ranges::data(std::get<Is>(rngs)) + offset, f)...); }(std::make_index_sequence<sizeof...(Ts)>());
}

template<typename T, typename... Us>
auto invokeProcessOneWithOrWithoutOffset(T& node, std::size_t offset, const Us&... inputs) {
    if constexpr (traits::block::can_processOne_with_offset<T>) {
        return node.processOne(offset, inputs...);
    } else {
        return node.processOne(inputs...);
    }
}

template<std::size_t Index, PortType portType, typename Self>
[[nodiscard]] constexpr auto& inputPort(Self* self) noexcept {
    using TRequestedPortType = typename traits::block::ports_data<Self>::template for_type<portType>::input_ports::template at<Index>;
    if constexpr (traits::block::block_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::block::get_port_member_descriptor<Self, TRequestedPortType>;
        return member_descriptor()(*self);
    } else {
        return self->template getArgument<TRequestedPortType>();
    }
}

template<std::size_t Index, PortType portType, typename Self>
[[nodiscard]] constexpr auto& outputPort(Self* self) noexcept {
    using TRequestedPortType = typename traits::block::ports_data<Self>::template for_type<portType>::output_ports::template at<Index>;
    if constexpr (traits::block::block_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::block::get_port_member_descriptor<Self, TRequestedPortType>;
        return member_descriptor()(*self);
    } else {
        return self->template getArgument<TRequestedPortType>();
    }
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto& inputPort(Self* self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::block::all_input_ports<Self>>();
    if constexpr (Index == meta::default_message_port_index) {
        return self->msgIn;
    }
    return inputPort<Index, PortType::ANY, Self>(self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto& outputPort(Self* self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::block::all_output_ports<Self>>();
    if constexpr (Index == meta::default_message_port_index) {
        return self->msgOut;
    }
    return outputPort<Index, PortType::ANY, Self>(self);
}

template<PortType portType, typename Self>
[[nodiscard]] constexpr auto inputPorts(Self* self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::tie(inputPort<Idx, portType>(self)...); }(std::make_index_sequence<traits::block::ports_data<Self>::template for_type<portType>::input_ports::size()>());
}

template<PortType portType, typename Self>
[[nodiscard]] constexpr auto outputPorts(Self* self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::tie(outputPort<Idx, portType>(self)...); }(std::make_index_sequence<traits::block::ports_data<Self>::template for_type<portType>::output_ports::size>());
}

namespace work {

class Counter {
    std::atomic_uint64_t encodedCounter{static_cast<uint64_t>(std::numeric_limits<gr::Size_t>::max()) << 32};

public:
    void increment(std::size_t workRequestedInc, std::size_t workDoneInc) {
        uint64_t oldCounter;
        uint64_t newCounter;
        do {
            oldCounter         = encodedCounter;
            auto workRequested = static_cast<gr::Size_t>(oldCounter >> 32);
            auto workDone      = static_cast<gr::Size_t>(oldCounter & 0xFFFFFFFF);
            if (workRequested != std::numeric_limits<gr::Size_t>::max()) {
                workRequested = static_cast<uint32_t>(std::min(static_cast<std::uint64_t>(workRequested) + workRequestedInc, static_cast<std::uint64_t>(std::numeric_limits<gr::Size_t>::max())));
            }
            workDone += static_cast<gr::Size_t>(workDoneInc);
            newCounter = (static_cast<uint64_t>(workRequested) << 32) | workDone;
        } while (!encodedCounter.compare_exchange_weak(oldCounter, newCounter));
    }

    std::pair<std::size_t, std::size_t> getAndReset() {
        uint64_t oldCounter    = encodedCounter.exchange(0);
        auto     workRequested = static_cast<gr::Size_t>(oldCounter >> 32);
        auto     workDone      = static_cast<gr::Size_t>(oldCounter & 0xFFFFFFFF);
        if (workRequested == std::numeric_limits<gr::Size_t>::max()) {
            return {std::numeric_limits<std::size_t>::max(), static_cast<std::size_t>(workDone)};
        }
        return {static_cast<std::size_t>(workRequested), static_cast<std::size_t>(workDone)};
    }

    std::pair<std::size_t, std::size_t> get() {
        uint64_t oldCounter    = std::atomic_load_explicit(&encodedCounter, std::memory_order_acquire);
        auto     workRequested = static_cast<gr::Size_t>(oldCounter >> 32);
        auto     workDone      = static_cast<gr::Size_t>(oldCounter & 0xFFFFFFFF);
        if (workRequested == std::numeric_limits<std::uint32_t>::max()) {
            return {std::numeric_limits<std::size_t>::max(), static_cast<std::size_t>(workDone)};
        }
        return {static_cast<std::size_t>(workRequested), static_cast<std::size_t>(workDone)};
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
concept HasWork = requires(T t, std::size_t requested_work) {
    { t.work(requested_work) } -> std::same_as<work::Result>;
};

template<typename T>
concept BlockLike = requires(T t, std::size_t requested_work) {
    { t.unique_name } -> std::same_as<const std::string&>;
    { unwrap_if_wrapped_t<decltype(t.name)>{} } -> std::same_as<std::string>;
    { unwrap_if_wrapped_t<decltype(t.meta_information)>{} } -> std::same_as<property_map>;
    { t.description } noexcept -> std::same_as<const std::string_view&>;

    { t.isBlocking() } noexcept -> std::same_as<bool>;

    { t.settings() } -> std::same_as<SettingsBase&>;

    // N.B. TODO discuss these requirements
    requires !std::is_copy_constructible_v<T>;
    requires !std::is_copy_assignable_v<T>;
} && HasWork<T>;

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

template<typename TBlock, typename TDecayedBlock = std::remove_cvref_t<TBlock>>
inline void checkBlockContracts();

template<typename T>
struct isBlockDependent {
    static constexpr bool value = PortLike<T> || BlockLike<T>;
};

namespace block::property {
inline static const char* kHeartbeat      = "Heartbeat";      ///< heartbeat property - the canary in the coal mine (supports block-specific subscribe/unsubscribe)
inline static const char* kEcho           = "Echo";           ///< basic property that receives any matching message and sends a mirror with it's serviceName/unique_name
inline static const char* kLifeCycleState = "LifecycleState"; ///< basic property that sets the block's @see lifecycle::StateMachine
inline static const char* kSetting        = "Settings";       ///< asynchronous message-based setting handling,
                                                              // N.B. 'Set' Settings are first staged before being applied within the work(...) function (real-time/non-real-time decoupling)
inline static const char* kStagedSetting = "StagedSettings";  ///< asynchronous message-based staging of settings

inline static const char* kStoreDefaults = "StoreDefaults"; ///< store present settings as default, for counterpart @see kResetDefaults
inline static const char* kResetDefaults = "ResetDefaults"; ///< retrieve and reset to default setting, for counterpart @see kStoreDefaults
} // namespace block::property

namespace block {
enum class Category {
    NormalBlock,           ///< Block that does not contain children blocks
    TransparentBlockGroup, ///< Block with children blocks which do not have a dedicated scheduler
    ScheduledBlockGroup    ///< Block with children that have a dedicated scheduler
};
}

/**
 * @brief The 'Block<Derived>' is a base class for blocks that perform specific signal processing operations. It stores
 * references to its input and output 'ports' that can be zero, one, or many, depending on the use case.
 * As the base class for all user-defined blocks, it implements common convenience functions and a default public API
 * through the Curiously-Recurring-Template-Pattern (CRTP). For example:
 * @code
 * struct UserDefinedBlock : Block<UserDefinedBlock> {
 *   PortIn<float> in;
 *   PortOut<float> out;
 *   // implement one of the possible processOne or processBulk functions
 * };
 * ENABLE_REFLECTION_FOR_TEMPLATE(UserDefinedBlock, in, out);
 * @endcode
 * The macro `ENABLE_REFLECTION_FOR_TEMPLATE` since it relies on a template specialisation needs to be declared on the global scope.
 *
 * As an alternative definition that does not require the 'ENABLE_REFLECTION_FOR_TEMPLATE' macro and that also supports arbitrary
 * types for input 'T' and for the return 'R':
 * @code
 * template<typename T, typename R>
 * struct UserDefinedBlock : Block<UserDefinedBlock, PortInNamed<T, "in">, PortInNamed<R, "out">> {
 *   // implement one of the possible processOne or processBulk functions
 * };
 * @endcode
 * This implementation provides efficient compile-time static polymorphism (i.e. access to the ports, settings, etc. does
 * not require virtual functions or inheritance, which can have performance penalties in high-performance computing contexts).
 * Note: The template parameter '<Derived>' can be dropped once C++23's 'deducing this' is widely supported by compilers.
 *
 * The 'Block<Derived>' implementation provides simple defaults for users who want to focus on generic signal-processing
 * algorithms and don't need full flexibility (and complexity) of using the generic `work_return_t work() {...}`.
 * The following defaults are defined for one of the two 'UserDefinedBlock' block definitions (WIP):
 * <ul>
 * <li> <b>case 1a</b> - non-decimating N-in->N-out mechanic and automatic handling of streaming tags and settings changes:
 * @code
 *  gr::PortIn<T> in;
 *  gr::PortOut<R> out;
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
 * <li> <b>case 2a</b>: N-in->M-out -> processBulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N)
 * Two fields define the resampling ratio are `input_chunk_size` and `output_chunk_size`.
 * For each `input_chunk_size` input samples, `output_chunk_size` output samples are published.
 * Thus the total number of input/output samples can be calculated as `n_input = k * input_chunk_size` and `n_output = k * output_chunk_size`.
 * They also act as constraints for the minimum number of input samples (`input_chunk_size`)  and output samples (`output_chunk_size`).
 *
 * │input_chunk_size│...│input_chunk_size│ ───► │output_chunk_size│...│output_chunk_size│
 * └────────────────┘   └────────────────┘      └─────────────────┘   └─────────────────┘
 *    n_inputs = k * input_chunk_size              n_outputs = k * output_chunk_size
 *
 * <li> <b>case 2b</b>: N-in->M-out -> processBulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling (to-be-done)
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
 * Properties System:
 * Properties offer a standardized way to manage runtime configuration and state. This system is built upon a message-passing model, allowing blocks
 * to dynamically adjust their behavior, respond to queries, and notify about state changes. Defined under the `block::property` namespace,
 * these properties leverage the Majordomo Protocol (MDP) pattern for structured and efficient communication.
 *
 * Predefined properties include:
 * - `kHeartbeat`: Monitors and reports the block's operational state.
 * - `kEcho`: Responds to messages by echoing them back, aiding in communication testing.
 * - `kLifeCycleState`: Manages and reports the block's lifecycle state.
 * - `kSetting` & `kStagedSetting`: Handle real-time and non-real-time configuration adjustments.
 * - `kStoreDefaults` & `kResetDefaults`: Facilitate storing and reverting to default settings.
 *
 * These properties can be interacted with through messages, supporting operations like setting values, querying states, and subscribing to updates.
 * This model provides a flexible interface for blocks to adapt their processing based on runtime conditions and external inputs.
 *
 * Implementing a Property:
 * Blocks can implement custom properties by registering them in the `propertyCallbacks` map within the `start()` method.
 * This allows the block to handle `SET`, `GET`, `SUBSCRIBE`, and `UNSUBSCRIBE` commands targeted at the property, enabling dynamic interaction with the block's functionality and configuration.
 *
 * @code
 * struct MyBlock : public Block<MyBlock> {
 *     static inline const char* kMyCustomProperty = "MyCustomProperty";
 *     std::optional<Message> propertyCallbackMyCustom(std::string_view propertyName, Message message) {
 *         using enum gr::message::Command;
 *         assert(kMyCustomProperty  == propertyName); // internal check that the property-name to callback is correct
 *
 *         switch (message.cmd) {
 *           case Set: // handle property setting
 *             break;
 *           case Get: // handle property querying
 *             return Message{ populate reply message };
 *           case Subscribe: // handle subscription
 *             break;
 *           case Unsubscribe: // handle unsubscription
 *             break;
 *           default: throw gr::exception(fmt::format("unsupported command {} for property {}", message.cmd, propertyName));
 *         }
 *       return std::nullopt; // no reply needed for Set, Subscribe, Unsubscribe
 *     }
 *
 *     void start() override {
 *         propertyCallbacks.emplace(kMyCustomProperty, &MyBlock::propertyCallbackMyCustom);
 *     }
 * };
 * @endcode
 *
 * @tparam Derived the user-defined block CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 * @tparam Arguments NTTP list containing the compile-time defined port instances, setting structs, or other constraints.
 */
template<typename Derived, typename... Arguments>
class Block : public lifecycle::StateMachine<Derived>, public std::tuple<Arguments...> {
    static std::atomic_size_t _uniqueIdCounter;
    template<typename T, gr::meta::fixed_string description = "", typename... Args>
    using A = Annotated<T, description, Args...>;

public:
    using base_t                     = Block<Derived, Arguments...>;
    using derived_t                  = Derived;
    using ArgumentsTypeList          = typename gr::meta::typelist<Arguments...>;
    using block_template_parameters  = meta::typelist<Arguments...>;
    using ResamplingControl          = ArgumentsTypeList::template find_or_default<is_resampling, Resampling<1UL, 1UL, true>>;
    using StrideControl              = ArgumentsTypeList::template find_or_default<is_stride, Stride<0UL, true>>;
    using AllowIncompleteFinalUpdate = ArgumentsTypeList::template find_or_default<is_incompleteFinalUpdatePolicy, IncompleteFinalUpdatePolicy<IncompleteFinalUpdateEnum::DROP>>;
    using DrawableControl            = ArgumentsTypeList::template find_or_default<is_drawable, Drawable<UICategory::None, "">>;

    constexpr static bool            blockingIO    = std::disjunction_v<std::is_same<BlockingIO<true>, Arguments>...> || std::disjunction_v<std::is_same<BlockingIO<false>, Arguments>...>;
    constexpr static block::Category blockCategory = block::Category::NormalBlock;

    template<typename T>
    auto& getArgument() {
        return std::get<T>(*this);
    }

    template<typename T>
    const auto& getArgument() const {
        return std::get<T>(*this);
    }

    // TODO: These are not involved in move operations, might be a problem later
    alignas(hardware_destructive_interference_size) std::atomic<std::size_t> ioRequestedWork{std::numeric_limits<std::size_t>::max()};
    alignas(hardware_destructive_interference_size) work::Counter ioWorkDone{};
    alignas(hardware_destructive_interference_size) std::atomic<work::Status> ioLastWorkStatus{work::Status::OK};
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::Sequence> progress = std::make_shared<gr::Sequence>();
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::thread_pool::BasicThreadPool> ioThreadPool;
    alignas(hardware_destructive_interference_size) std::atomic<bool> ioThreadRunning{false};

    constexpr static TagPropagationPolicy tag_policy = TagPropagationPolicy::TPP_ALL_TO_ALL;

    using RasamplingValue = std::conditional_t<ResamplingControl::kIsConst, const gr::Size_t, gr::Size_t>;
    using ResamplingLimit = Limits<1UL, std::numeric_limits<RasamplingValue>::max()>;
    using ReasmplingDoc   = Doc<"For each `input_chunk_size` input samples, `output_chunk_size` output samples are published (in>out: Decimate, in<out: Interpolate, in==out: No change)">;

    A<RasamplingValue, "input_chunk_size", ReasmplingDoc, ResamplingLimit>  input_chunk_size                                                     = ResamplingControl::kInputChunkSize;
    A<RasamplingValue, "output_chunk_size", ReasmplingDoc, ResamplingLimit> output_chunk_size                                                    = ResamplingControl::kOutputChunkSize;
    using StrideValue                                                                                                                            = std::conditional_t<StrideControl::kIsConst, const gr::Size_t, gr::Size_t>;
    A<StrideValue, "stride", Doc<"samples between data processing. <N for overlap, >N for skip, =0 for back-to-back.">>       stride             = StrideControl::kStride;
    A<bool, "disconnect on done", Doc<"If no downstream blocks, declare itself 'DONE' and disconnect from upstream blocks.">> disconnect_on_done = true;

    gr::Size_t strideCounter = 0UL; // leftover stride from previous calls

    // TODO: These are not involved in move operations, might be a problem later
    const std::size_t unique_id   = _uniqueIdCounter++;
    const std::string unique_name = fmt::format("{}#{}", gr::meta::type_name<Derived>(), unique_id);

    //
    A<std::string, "user-defined name", Doc<"N.B. may not be unique -> ::unique_name">> name = gr::meta::type_name<Derived>();
    //
    constexpr static std::string_view description = [] {
        if constexpr (requires { typename Derived::Description; }) {
            return static_cast<std::string_view>(Derived::Description::value);
        } else {
            return "please add a public 'using Description = Doc<\"...\">' documentation annotation to your block definition";
        }
    }();
#ifndef __EMSCRIPTEN__
    static_assert(std::atomic<lifecycle::State>::is_always_lock_free, "std::atomic<lifecycle::State> is not lock-free");
#endif

    //
    static property_map initMetaInfo() {
        using namespace std::string_literals;
        property_map ret;
        if constexpr (!std::is_same_v<NotDrawable, DrawableControl>) {
            property_map info;
            info.insert_or_assign("Category"s, std::string(magic_enum::enum_name(DrawableControl::kCategory)));
            info.insert_or_assign("Toolkit"s, std::string(DrawableControl::kToolkit));

            ret.insert_or_assign("Drawable"s, info);
        }
        return ret;
    }

    A<property_map, "meta-information", Doc<"store non-graph-processing information like UI block position etc.">> meta_information = initMetaInfo();

    // TODO: C++26 make sure these are not reflected
    // We support ports that are template parameters or reflected member variables,
    // so these are handled in a special way
    MsgPortInNamed<"__Builtin">  msgIn;
    MsgPortOutNamed<"__Builtin"> msgOut;

    using PropertyCallback = std::optional<Message> (Derived::*)(std::string_view, Message);
    std::map<std::string, PropertyCallback> propertyCallbacks{
        {block::property::kHeartbeat, &Block::propertyCallbackHeartbeat},           //
        {block::property::kEcho, &Block::propertyCallbackEcho},                     //
        {block::property::kLifeCycleState, &Block::propertyCallbackLifecycleState}, //
        {block::property::kSetting, &Block::propertyCallbackSettings},              //
        {block::property::kStagedSetting, &Block::propertyCallbackStagedSettings},  //
        {block::property::kStoreDefaults, &Block::propertyCallbackStoreDefaults},   //
        {block::property::kResetDefaults, &Block::propertyCallbackResetDefaults},   //
    };
    std::map<std::string, std::set<std::string>> propertySubscriptions;

protected:
    bool _outputTagsChanged = false;
    Tag  _mergedInputTag{};

    // intermediate non-real-time<->real-time setting states
    CtxSettings<Derived> _settings;

    [[nodiscard]] constexpr auto& self() noexcept { return *static_cast<Derived*>(this); }

    [[nodiscard]] constexpr const auto& self() const noexcept { return *static_cast<const Derived*>(this); }

    template<typename TFunction, typename... Args>
    [[maybe_unused]] constexpr inline auto invokeUserProvidedFunction(std::string_view callingSite, TFunction&& func, Args&&... args, const std::source_location& location = std::source_location::current()) noexcept {
        if constexpr (noexcept(func(std::forward<Args>(args)...))) { // function declared as 'noexcept' skip exception handling
            return std::forward<TFunction>(func)(std::forward<Args>(args)...);
        } else { // function not declared with 'noexcept' -> may throw
            try {
                return std::forward<TFunction>(func)(std::forward<Args>(args)...);
            } catch (const gr::exception& e) {
                emitErrorMessageIfAny(callingSite, std::unexpected(gr::Error(std::move(e))));
            } catch (const std::exception& e) {
                emitErrorMessageIfAny(callingSite, std::unexpected(gr::Error(e, location)));
            } catch (...) {
                emitErrorMessageIfAny(callingSite, std::unexpected(gr::Error("unknown error", location)));
            }
        }
    }

public:
    Block() : Block(gr::property_map()) {}
    Block(std::initializer_list<std::pair<const std::string, pmtv::pmt>> initParameter) noexcept(false) : Block(property_map(initParameter)) {}
    Block(property_map initParameters) noexcept(false)                                                                                    // N.B. throws in case of on contract violations
        : lifecycle::StateMachine<Derived>(), std::tuple<Arguments...>(), _settings(CtxSettings<Derived>(*static_cast<Derived*>(this))) { // N.B. safe delegated use of this (i.e. not used during construction)

        // check Block<T> contracts
        checkBlockContracts<decltype(*static_cast<Derived*>(this))>();

        if constexpr (refl::is_reflectable<Derived>()) {
            settings().setInitBlockParameters(initParameters);
        }
    }

    Block(Block&& other) noexcept : lifecycle::StateMachine<Derived>(std::move(other)), std::tuple<Arguments...>(std::move(other)), input_chunk_size(std::move(other.input_chunk_size)), output_chunk_size(std::move(other.output_chunk_size)), stride(std::move(other.stride)), strideCounter(std::move(other.strideCounter)), msgIn(std::move(other.msgIn)), msgOut(std::move(other.msgOut)), propertyCallbacks(std::move(other.propertyCallbacks)), _outputTagsChanged(std::move(other._outputTagsChanged)), _mergedInputTag(std::move(other._mergedInputTag)), _settings(std::move(other._settings)) {}

    // There are a few const or conditionally const member variables,
    // we can not have a move-assignment that is equivalent to
    // the move constructor
    Block& operator=(Block&& other) = delete;

    ~Block() { // NOSONAR -- need to request the (potentially) running ioThread to stop
        if (lifecycle::isActive(this->state())) {
            emitErrorMessageIfAny("~Block()", this->changeStateTo(lifecycle::State::REQUESTED_STOP));
        }
        if constexpr (blockingIO) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // wait for done
        for (auto actualState = this->state(); lifecycle::isActive(actualState); actualState = this->state()) {
            this->waitOnState(actualState);
        }

        emitErrorMessageIfAny("~Block()", this->changeStateTo(lifecycle::State::STOPPED));
    }

    void init(std::shared_ptr<gr::Sequence> progress_, std::shared_ptr<gr::thread_pool::BasicThreadPool> ioThreadPool_) {
        progress     = std::move(progress_);
        ioThreadPool = std::move(ioThreadPool_);

        // Set names of port member variables
        // TODO: Refactor the library not to assign names to ports. The
        // block and the graph are the only things that need the port name
        auto setPortName = [&]([[maybe_unused]] std::size_t index, auto* t) {
            using CurrentPortType = std::remove_pointer_t<decltype(t)>;
            if constexpr (traits::port::is_port_v<CurrentPortType>) {
                using PortDescriptor = typename CurrentPortType::ReflDescriptor;
                if constexpr (refl::trait::is_descriptor_v<PortDescriptor>) {
                    auto& port = (self().*(PortDescriptor::pointer));
                    port.name  = CurrentPortType::Name;
                }
            } else if constexpr (traits::port::is_port_collection_v<CurrentPortType>) {
                using PortCollectionDescriptor = typename CurrentPortType::value_type::ReflDescriptor;
                if constexpr (refl::trait::is_descriptor_v<PortCollectionDescriptor>) {
                    auto&       collection     = (self().*(PortCollectionDescriptor::pointer));
                    std::string collectionName = refl::descriptor::get_name(PortCollectionDescriptor()).data;
                    for (auto& port : collection) {
                        port.name = collectionName;
                    }
                }
            } else {
                meta::print_types<meta::message_type<"Not a port, not a collection of ports">, CurrentPortType>{};
            }
        };
        traits::block::all_input_ports<Derived>::for_each(setPortName);
        traits::block::all_output_ports<Derived>::for_each(setPortName);

        settings().init();

        // important: these tags need to be queued because at this stage the block is not yet connected to other downstream blocks
        invokeUserProvidedFunction("init() - applyStagedParameters", [this] noexcept(false) {
            if (const auto applyResult = settings().applyStagedParameters(); !applyResult.forwardParameters.empty()) {
                if constexpr (Derived::tag_policy == TagPropagationPolicy::TPP_ALL_TO_ALL) {
                    publishTag(applyResult.forwardParameters, 0);
                }
                notifyListeners(block::property::kSetting, settings().get());
            }
        });
        checkBlockParameterConsistency();
        // store default settings -> can be recovered with 'resetDefaults()'
        settings().storeDefaults();
        emitErrorMessageIfAny("init(..) -> INITIALISED", this->changeStateTo(lifecycle::State::INITIALISED));
    }

    template<gr::meta::array_or_vector_type Container>
    [[nodiscard]] constexpr std::size_t availableInputSamples(Container& data) const noexcept {
        if constexpr (gr::meta::vector_type<Container>) {
            data.resize(traits::block::stream_input_port_types<Derived>::size);
        } else if constexpr (gr::meta::array_type<Container>) {
            static_assert(std::tuple_size<Container>::value >= traits::block::stream_input_port_types<Derived>::size);
        } else {
            static_assert(gr::meta::always_false<Container>, "type not supported");
        }
        meta::tuple_for_each_enumerate(
            [&data]<typename Port>(auto index, Port& input_port) {
                if constexpr (traits::port::is_port_v<Port>) {
                    data[index] = input_port.streamReader().available();
                } else {
                    data[index] = 0;
                    for (auto& port : input_port) {
                        data[index] += port.streamReader().available();
                    }
                }
            },
            inputPorts<PortType::STREAM>(&self()));
        return traits::block::stream_input_port_types<Derived>::size;
    }

    template<gr::meta::array_or_vector_type Container>
    [[nodiscard]] constexpr std::size_t availableOutputSamples(Container& data) const noexcept {
        if constexpr (gr::meta::vector_type<Container>) {
            data.resize(traits::block::stream_output_port_types<Derived>::size);
        } else if constexpr (gr::meta::array_type<Container>) {
            static_assert(std::tuple_size<Container>::value >= traits::block::stream_output_port_types<Derived>::size);
        } else {
            static_assert(gr::meta::always_false<Container>, "type not supported");
        }
        meta::tuple_for_each_enumerate(
            [&data]<typename Port>(auto index, Port& output_port) {
                if constexpr (traits::port::is_port_v<Port>) {
                    data[index] = output_port.streamWriter().available();
                } else {
                    data[index] = 0;
                    for (auto& port : output_port) {
                        data[index] += port.streamWriter().available();
                    }
                }
            },
            outputPorts<PortType::STREAM>(&self()));
        return traits::block::stream_output_port_types<Derived>::size;
    }

    [[nodiscard]] constexpr bool isBlocking() const noexcept { return blockingIO; }

    [[nodiscard]] constexpr bool input_tags_present() const noexcept { return !_mergedInputTag.map.empty(); };

    [[nodiscard]] Tag mergedInputTag() const noexcept { return _mergedInputTag; }

    [[nodiscard]] constexpr const SettingsBase& settings() const noexcept { return _settings; }

    [[nodiscard]] constexpr SettingsBase& settings() noexcept { return _settings; }

    void setSettings(CtxSettings<Derived>& settings) { _settings = std::move(settings); }

    template<std::size_t Index, typename Self>
    friend constexpr auto& inputPort(Self* self) noexcept;

    template<std::size_t Index, typename Self>
    friend constexpr auto& outputPort(Self* self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto& inputPort(Self* self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto& outputPort(Self* self) noexcept;

    constexpr void checkBlockParameterConsistency() {
        constexpr bool kIsSourceBlock = traits::block::stream_input_port_types<Derived>::size == 0;
        constexpr bool kIsSinkBlock   = traits::block::stream_output_port_types<Derived>::size == 0;

        if constexpr (ResamplingControl::kEnabled) {
            static_assert(!kIsSinkBlock, "input_chunk_size and output_chunk_size are not available for sink blocks. Remove 'Resampling<>' from the block definition.");
            static_assert(!kIsSourceBlock, "input_chunk_size and output_chunk_size are not available for source blocks. Remove 'Resampling<>' from the block definition.");
            static_assert(HasProcessBulkFunction<Derived>, "Blocks which allow input_chunk_size and output_chunk_size must implement processBulk(...) method. Remove 'Resampling<>' from the block definition.");
        } else {
            if (input_chunk_size != 1ULL || output_chunk_size != 1ULL) {
                emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", fmt::format("Block is not defined as `Resampling<>`, but input_chunk_size = {}, output_chunk_size = {}, they both must equal to 1.", input_chunk_size, output_chunk_size));
                requestStop();
                return;
            }
        }
        if constexpr (StrideControl::kEnabled) {
            static_assert(!kIsSourceBlock, "Stride is not available for source blocks. Remove 'Stride<>' from the block definition.");
        } else {
            if (stride != 0ULL) {
                emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", fmt::format("Block is not defined as `Stride<>`, but stride = {}, it must equal to 0.", stride));
                requestStop();
                return;
            }
        }
        const auto [minSyncIn, maxSyncIn, _, _1]    = getPortLimits(inputPorts<PortType::STREAM>(&self()));
        const auto [minSyncOut, maxSyncOut, _2, _3] = getPortLimits(outputPorts<PortType::STREAM>(&self()));
        if (minSyncIn > maxSyncIn) {
            emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", fmt::format("Min samples for input ports ({}) is larger then max samples for input ports ({})", minSyncIn, maxSyncIn));
            requestStop();
            return;
        }
        if (minSyncOut > maxSyncOut) {
            emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", fmt::format("Min samples for output ports ({}) is larger then max samples for output ports ({})", minSyncOut, maxSyncOut));
            requestStop();
            return;
        }
        if (input_chunk_size > maxSyncIn) {
            emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", fmt::format("resampling input_chunk_size ({}) is larger then max samples for input ports ({})", input_chunk_size, maxSyncIn));
            requestStop();
            return;
        }
        if (output_chunk_size > maxSyncOut) {
            emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", fmt::format("resampling output_chunk_size ({}) is larger then max samples for output ports ({})", output_chunk_size, maxSyncOut));
            requestStop();
            return;
        }
    }

    void publishSamples(std::size_t nSamples, auto& publishableSpanTuple) noexcept {
        if constexpr (traits::block::stream_output_ports<Derived>::size > 0) {
            meta::tuple_for_each_enumerate(
                [nSamples]<typename OutputRange>(auto, OutputRange& outputRange) {
                    auto processOneRange = [nSamples]<typename Out>(Out& out) {
                        if constexpr (Out::isMultiProducerStrategy()) {
                            if (!out.isFullyPublished()) {
                                std::abort();
                            }
                        }
                        if (!out.isPublished()) {
                            using enum gr::SpanReleasePolicy;
                            if constexpr (Out::spanReleasePolicy() == Terminate) {
                                std::abort();
                            } else if constexpr (Out::spanReleasePolicy() == ProcessAll) {
                                out.publish(nSamples);
                            } else if constexpr (Out::spanReleasePolicy() == ProcessNone) {
                                out.publish(0U);
                            }
                        }
                    };
                    if constexpr (refl::trait::is_instance_of_v<std::vector, std::remove_cvref_t<OutputRange>>) {
                        for (auto& out : outputRange) {
                            processOneRange(out);
                        }
                    } else {
                        processOneRange(outputRange);
                    }
                },
                publishableSpanTuple);
        }
    }

    bool consumeReaders(std::size_t nSamples, auto& consumableSpanTuple) {
        bool success = true;
        if constexpr (traits::block::stream_input_ports<Derived>::size > 0) {
            meta::tuple_for_each_enumerate(
                [nSamples, &success]<typename InputRange>(auto, InputRange& inputRange) {
                    auto processOneRange = [nSamples, &success]<typename In>(In& in) {
                        if (!in.isConsumeRequested()) {
                            using enum gr::SpanReleasePolicy;
                            if constexpr (In::spanReleasePolicy() == Terminate) {
                                std::abort();
                            } else if constexpr (In::spanReleasePolicy() == ProcessAll) {
                                success = success && in.consume(nSamples);
                            } else if constexpr (In::spanReleasePolicy() == ProcessNone) {
                                success = success && in.consume(0U);
                            }
                        }
                    };
                    if constexpr (refl::trait::is_instance_of_v<std::vector, std::remove_cvref_t<InputRange>>) {
                        for (auto& in : inputRange) {
                            processOneRange(in);
                        }
                    } else {
                        processOneRange(inputRange);
                    }
                },
                consumableSpanTuple);
        }
        return success;
    }

    template<typename... Ts>
    constexpr auto invoke_processOne(std::size_t offset, Ts&&... inputs) {
        if constexpr (traits::block::stream_output_ports<Derived>::size == 0) {
            invokeProcessOneWithOrWithoutOffset(self(), offset, std::forward<Ts>(inputs)...);
            return std::tuple{};
        } else if constexpr (traits::block::stream_output_ports<Derived>::size == 1) {
            return std::tuple{invokeProcessOneWithOrWithoutOffset(self(), offset, std::forward<Ts>(inputs)...)};
        } else {
            return invokeProcessOneWithOrWithoutOffset(self(), offset, std::forward<Ts>(inputs)...);
        }
    }

    template<typename... Ts>
    constexpr auto invoke_processOne_simd(std::size_t offset, auto width, Ts&&... input_simds) {
        if constexpr (sizeof...(Ts) == 0) {
            if constexpr (traits::block::stream_output_ports<Derived>::size == 0) {
                self().processOne_simd(offset, width);
                return std::tuple{};
            } else if constexpr (traits::block::stream_output_ports<Derived>::size == 1) {
                return std::tuple{self().processOne_simd(offset, width)};
            } else {
                return self().processOne_simd(offset, width);
            }
        } else {
            return invoke_processOne(offset, std::forward<Ts>(input_simds)...);
        }
    }

    constexpr void forwardTags() noexcept {
        if (input_tags_present()) {
            if constexpr (Derived::tag_policy == TagPropagationPolicy::TPP_ALL_TO_ALL) {
                for_each_port([this](PortLike auto& outPort) noexcept { outPort.publishTag(mergedInputTag().map, 0); }, outputPorts<PortType::STREAM>(&self()));
            }
            if (mergedInputTag().map.contains(gr::tag::END_OF_STREAM)) {
                requestStop();
            }
            // clear temporary cached input tags after processing - won't be needed after this
            _mergedInputTag.map.clear();
        }

        for_each_port([](PortLike auto& outPort) noexcept { outPort.publishPendingTags(); }, outputPorts<PortType::STREAM>(&self()));
        _outputTagsChanged = false;
    }

    /**
     * Collects tags from each input port, merges them into a single map, applies settings and if requested propagates
     * them to the output ports.
     * @param untilOffset defaults to 0, if bigger merges all tags from samples 0...untilOffset for each port before merging
     *                    them
     */
    constexpr void updateInputAndOutputTags(std::size_t /*untilOffset*/ = 0UZ) noexcept {
        for_each_port(
            [this]<PortLike TPort>(TPort& input_port) noexcept {
                auto mergeSrcMapInto = [](const property_map& sourceMap, property_map& destinationMap) {
                    assert(&sourceMap != &destinationMap);
                    for (const auto& [key, value] : sourceMap) {
                        destinationMap.insert_or_assign(key, value);
                    }
                };

                mergeSrcMapInto(input_port.getMergedTag().map, _mergedInputTag.map);
            },
            inputPorts<PortType::STREAM>(&self()));

        if (!mergedInputTag().map.empty()) {
            settings().autoUpdate(mergedInputTag()); // apply tags as new settings if matching
            if (mergedInputTag().map.contains(gr::tag::END_OF_STREAM)) {
                requestStop();
            }
        }
    }

    void applyChangedSettings() {
        if (!settings().changed()) {
            return;
        }
        invokeUserProvidedFunction("applyChangedSettings()", [this] noexcept(false) {
            auto applyResult = settings().applyStagedParameters();
            checkBlockParameterConsistency();

            if (!applyResult.forwardParameters.empty()) {
                for (auto& [key, value] : applyResult.forwardParameters) {
                    _mergedInputTag.insert_or_assign(key, value);
                }
            }

            settings().setChanged(false);

            if (!applyResult.appliedParameters.empty()) {
                notifyListeners(block::property::kStagedSetting, applyResult.appliedParameters);
            }
            notifyListeners(block::property::kSetting, settings().get());
        });
    }

    constexpr static auto prepareStreams(auto ports, std::size_t sync_samples) {
        return meta::tuple_transform(
            [sync_samples]<typename PortOrCollection>(PortOrCollection& output_port_or_collection) noexcept {
                auto process_single_port = [&sync_samples]<typename Port>(Port&& port) {
                    using enum gr::SpanReleasePolicy;
                    if constexpr (std::remove_cvref_t<Port>::kIsInput) {
                        if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                            return std::forward<Port>(port).template get<ProcessAll>(sync_samples);
                        } else {
                            // return std::forward<Port>(port).template get<ProcessNone>(std::max(port.min_samples, std::min(port.streamReader().available(), port.max_samples)));
                            return std::forward<Port>(port).template get<ProcessNone>(port.streamReader().available());
                        }
                    } else if constexpr (std::remove_cvref_t<Port>::kIsOutput) {
                        if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                            return std::forward<Port>(port).template tryReserve<ProcessAll>(sync_samples);
                        } else {
                            return std::forward<Port>(port).template tryReserve<ProcessNone>(port.streamWriter().available());
                        }
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
            ports);
    }

    template<typename TOutTuple>
    constexpr static bool containsEmptyOutputSpans(TOutTuple& outputTuple) noexcept {
        bool result = false;
        meta::tuple_for_each(
            [&result]<typename TOut>(TOut& out) {
                if constexpr (PublishableSpan<TOut>) {
                    if (out.empty()) {
                        result = true;
                    }
                } else if constexpr (PublishableSpan<typename TOut::value_type>) {
                    for (auto& span : out) {
                        if (span.empty()) {
                            result = true;
                        }
                    }
                }
            },
            outputTuple);
        return result;
    }

    inline constexpr void publishTag(property_map&& tag_data, Tag::signed_index_type tagOffset = -1) noexcept {
        for_each_port([tag_data = std::move(tag_data), tagOffset](PortLike auto& outPort) { outPort.publishTag(tag_data, tagOffset); }, outputPorts<PortType::STREAM>(&self()));
    }

    inline constexpr void publishTag(const property_map& tag_data, Tag::signed_index_type tagOffset = -1) noexcept {
        for_each_port([&tag_data, tagOffset](PortLike auto& outPort) { outPort.publishTag(tag_data, tagOffset); }, outputPorts<PortType::STREAM>(&self()));
    }

    inline constexpr void publishEoS() noexcept {
        const property_map& tag_data{{gr::tag::END_OF_STREAM, true}};
        for_each_port([&tag_data](PortLike auto& outPort) { outPort.publishTag(tag_data, static_cast<Tag::signed_index_type>(outPort.streamWriter().nSamplesPublished())); }, outputPorts<PortType::STREAM>(&self()));
    }

    constexpr void requestStop() noexcept { emitErrorMessageIfAny("requestStop()", this->changeStateTo(lifecycle::State::REQUESTED_STOP)); }

    constexpr void processScheduledMessages() {
        using namespace std::chrono;
        const std::uint64_t nanoseconds_count = static_cast<uint64_t>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count());
        notifyListeners(block::property::kHeartbeat, {{"heartbeat", nanoseconds_count}});

        auto processPort = [this]<PortLike TPort>(TPort& inPort) {
            const auto available = inPort.streamReader().available();
            if (available == 0UZ) {
                return;
            }
            ConsumableSpan auto inSpan = inPort.streamReader().get(available);
            if constexpr (traits::block::can_processMessagesForPortConsumableSpan<Derived, TPort>) {
                self().processMessages(inPort, inSpan);
                // User could have consumed the span in the custom processMessages handler
                std::ignore = inSpan.tryConsume(inSpan.size());
            } else if constexpr (traits::block::can_processMessagesForPortStdSpan<Derived, TPort>) {
                self().processMessages(inPort, static_cast<std::span<const Message>>(inSpan));
                if (auto consumed = inSpan.tryConsume(inSpan.size()); !consumed) {
                    throw gr::exception(fmt::format("Block {}::processScheduledMessages() could not consume the messages from the message port", unique_name));
                }
            } else {
                return;
            }
            // notify scheduler and others that block did some work -> progress
            progress->incrementAndGet();
            progress->notify_all();
        };
        processPort(msgIn);
        for_each_port(processPort, inputPorts<PortType::MESSAGE>(&self()));
    }

protected:
    std::optional<Message> propertyCallbackHeartbeat(std::string_view propertyName, Message message) {
        using enum gr::message::Command;
        assert(propertyName == block::property::kHeartbeat);

        if (message.cmd == Set || message.cmd == Get) {
            std::uint64_t nanoseconds_count = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
            message.data                    = {{"heartbeat", nanoseconds_count}};
            return message;
        } else if (message.cmd == Subscribe) {
            if (!message.clientRequestID.empty()) {
                propertySubscriptions[std::string(propertyName)].insert(message.clientRequestID);
            }
            return std::nullopt;
        } else if (message.cmd == Unsubscribe) {
            propertySubscriptions[std::string(propertyName)].erase(message.clientRequestID);
            return std::nullopt;
        }

        throw gr::exception(fmt::format("block {} property {} does not implement command {}, msg: {}", unique_name, propertyName, message.cmd, message));
    }

    std::optional<Message> propertyCallbackEcho(std::string_view propertyName, Message message) {
        using enum gr::message::Command;
        assert(propertyName == block::property::kEcho);

        if (message.cmd == Set) {
            return message; // mirror message as is
        }

        throw gr::exception(fmt::format("block {} property {} does not implement command {}, msg: {}", unique_name, propertyName, message.cmd, message));
    }

    std::optional<Message> propertyCallbackLifecycleState(std::string_view propertyName, Message message) {
        using enum gr::message::Command;
        assert(propertyName == block::property::kLifeCycleState);

        if (message.cmd == Set) {
            if (!message.data.has_value() || !message.data.value().contains("state")) { // Changed '&&' to '||'
                throw gr::exception(fmt::format("propertyCallbackLifecycleState - cannot set block state w/o 'state' data msg: {}", message));
            }

            const auto& dataMap = message.data.value(); // Introduced const auto& dataMap
            auto        it      = dataMap.find("state");
            if (it == dataMap.end()) {
                throw gr::exception(fmt::format("propertyCallbackLifecycleState - state not found, msg: {}", message));
            }

            const std::string* stateStr = std::get_if<std::string>(&it->second); // Used std::get_if instead of std::get and try-catch block
            if (!stateStr) {
                throw gr::exception(fmt::format("propertyCallbackLifecycleState - state is not a string, msg: {}", message));
            }

            auto state = magic_enum::enum_cast<lifecycle::State>(*stateStr); // Changed to dereference stateStr
            if (!state.has_value()) {
                throw gr::exception(fmt::format("propertyCallbackLifecycleState - invalid lifecycle::State conversion from {}, msg: {}", *stateStr, message));
            }

            if (auto e = this->changeStateTo(state.value()); !e) {
                throw gr::exception(fmt::format("propertyCallbackLifecycleState - error in state transition - what: {}", e.error().message, e.error().sourceLocation, e.error().errorTime));
            }

            return std::nullopt;
        }

        if (message.cmd == Get) { // Merged 'else if' with 'if'
            message.data = {{"state", std::string(magic_enum::enum_name(this->state()))}};
            return message;
        }

        if (message.cmd == Subscribe) { // Merged 'else if' with 'if'
            if (!message.clientRequestID.empty()) {
                propertySubscriptions[std::string(propertyName)].insert(message.clientRequestID);
            }
            return std::nullopt;
        }

        if (message.cmd == Unsubscribe) { // Merged 'else if' with 'if'
            propertySubscriptions[std::string(propertyName)].erase(message.clientRequestID);
            return std::nullopt;
        }

        throw gr::exception(fmt::format("propertyCallbackLifecycleState - does not implement command {}, msg: {}", message.cmd, message));
    }

    std::optional<Message> propertyCallbackSettings(std::string_view propertyName, Message message) {
        using enum gr::message::Command;
        assert(propertyName == block::property::kSetting);

        if (message.cmd == Set) {
            if (!message.data.has_value()) {
                throw gr::exception(fmt::format("block {} (aka. {}) cannot set {} w/o data msg: {}", unique_name, name, propertyName, message));
            }
            // delegate to 'propertyCallbackStagedSettings' since we cannot set but only stage new settings due to mandatory real-time/non-real-time decoupling
            // settings are applied during the next work(...) invocation.
            propertyCallbackStagedSettings(block::property::kStagedSetting, message);
            return std::nullopt;
        } else if (message.cmd == Get) {
            message.data = self().settings().get();
            return message;
        } else if (message.cmd == Subscribe) {
            if (!message.clientRequestID.empty()) {
                propertySubscriptions[std::string(propertyName)].insert(message.clientRequestID);
            }
            return std::nullopt;
        } else if (message.cmd == Unsubscribe) {
            propertySubscriptions[std::string(propertyName)].erase(message.clientRequestID);
            return std::nullopt;
        }

        throw gr::exception(fmt::format("block {} property {} does not implement command {}, msg: {}", unique_name, propertyName, message.cmd, message));
    }

    std::optional<Message> propertyCallbackStagedSettings(std::string_view propertyName, Message message) {
        using enum gr::message::Command;
        assert(propertyName == block::property::kStagedSetting);
        const auto keys = [](const property_map& map) noexcept {
            std::string result;
            for (const auto& pair : map) {
                if (!result.empty()) {
                    result += ", ";
                }
                result += pair.first;
            }
            return result;
        };

        if (message.cmd == Set) {
            if (!message.data.has_value()) {
                throw gr::exception(fmt::format("block {} (aka. {}) cannot set {} w/o data msg: {}", unique_name, name, propertyName, message));
            }

            property_map notSet          = self().settings().setStaged(*message.data);
            property_map stagedParameter = self().settings().stagedParameters();

            if (notSet.empty()) {
                if (!message.clientRequestID.empty()) {
                    message.cmd  = Final;
                    message.data = std::move(stagedParameter);
                    return message;
                }
                return std::nullopt;
            }

            throw gr::exception(fmt::format("propertyCallbackStagedSettings - could not set fields: {}\nvs. available: {}", keys(std::move(notSet)), keys(settings().get())));
        } else if (message.cmd == Get) {
            message.data = self().settings().stagedParameters();
            return message;
        } else if (message.cmd == Subscribe) {
            if (!message.clientRequestID.empty()) {
                propertySubscriptions[std::string(propertyName)].insert(message.clientRequestID);
            }
            return std::nullopt;
        } else if (message.cmd == Unsubscribe) {
            propertySubscriptions[std::string(propertyName)].erase(message.clientRequestID);
            return std::nullopt;
        }

        throw gr::exception(fmt::format("block {} property {} does not implement command {}, msg: {}", unique_name, propertyName, message.cmd, message));
    }

    std::optional<Message> propertyCallbackStoreDefaults(std::string_view propertyName, Message message) {
        using enum gr::message::Command;
        assert(propertyName == block::property::kStoreDefaults);

        if (message.cmd == Set) {
            settings().storeDefaults();
            return std::nullopt;
        }

        throw gr::exception(fmt::format("block {} property {} does not implement command {}, msg: {}", unique_name, propertyName, message.cmd, message));
    }

    std::optional<Message> propertyCallbackResetDefaults(std::string_view propertyName, Message message) {
        using enum gr::message::Command;
        assert(propertyName == block::property::kResetDefaults);

        if (message.cmd == Set) {
            settings().resetDefaults();
            return std::nullopt;
        }

        throw gr::exception(fmt::format("block {} property {} does not implement command {}, msg: {}", unique_name, propertyName, message.cmd, message));
    }

protected:
    /***
     * Aggregate the amount of samples that can be consumed/produced from a range of ports.
     * @param ports a typelist of input or output ports
     * @return an anonymous struct representing the amount of available data on the ports
     */
    template<typename P>
    auto getPortLimits(P&& ports) {
        struct {
            std::size_t minSync      = 0UL;                                     // the minimum amount of samples that the block needs for processing on the sync ports
            std::size_t maxSync      = std::numeric_limits<std::size_t>::max(); // the maximum amount of that can be consumed on all sync ports
            std::size_t maxAvailable = std::numeric_limits<std::size_t>::max(); // the maximum amount of that are available on all sync ports
            bool        hasAsync     = false;                                   // true if there is at least one async input/output that has available samples/remaining capacity
        } result;
        auto adjustForInputPort = [&result]<PortLike Port>(Port& port) {
            const std::size_t available = [&port]() {
                if constexpr (gr::traits::port::is_input_v<Port>) {
                    return port.streamReader().available();
                } else {
                    return port.streamWriter().available();
                }
            }();
            if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                result.minSync      = std::max(result.minSync, port.min_samples);
                result.maxSync      = std::min(result.maxSync, port.max_samples);
                result.maxAvailable = std::min(result.maxAvailable, available);
            } else {                                 // async port
                if (available >= port.min_samples) { // ensure that process function is called if at least one async port has data available
                    result.hasAsync = true;
                }
            }
        };
        for_each_port([&adjustForInputPort](PortLike auto& port) { adjustForInputPort(port); }, std::forward<P>(ports));
        return result;
    }

    /***
     * Check the input ports for available samples
     */
    auto getNextTagAndEosPosition() {
        struct {
            bool        hasTag     = false;
            std::size_t nextTag    = std::numeric_limits<std::size_t>::max();
            std::size_t nextEosTag = std::numeric_limits<std::size_t>::max();
            bool        asyncEoS   = false;
        } result;

        auto adjustForInputPort = [&result]<PortLike Port>(Port& port) {
            if (port.isConnected()) {
                if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                    // get the tag after the one at position 0 that will be evaluated for this chunk.
                    // nextTag limits the size of the chunk except if this would violate port constraints
                    result.nextTag                        = std::min(result.nextTag, nSamplesUntilNextTag(port, 1).value_or(std::numeric_limits<std::size_t>::max()));
                    result.nextEosTag                     = std::min(result.nextEosTag, samples_to_eos_tag(port).value_or(std::numeric_limits<std::size_t>::max()));
                    const gr::ConsumableSpan auto tagData = port.tagReader().get();
                    result.hasTag                         = result.hasTag || (!tagData.empty() && tagData[0].index == port.streamReader().position() && !tagData[0].map.empty());
                } else { // async port
                    if (samples_to_eos_tag(port).transform([&port](auto n) { return n <= port.min_samples; }).value_or(false)) {
                        result.asyncEoS = true;
                    }
                }
            }
        };
        for_each_port([&adjustForInputPort](PortLike auto& port) { adjustForInputPort(port); }, inputPorts<PortType::STREAM>(&self()));
        return result;
    }

    /***
     * skip leftover stride
     * @param availableSamples number of samples that can be consumed from each sync port
     * @return inputSamples to skip before the chunk
     */
    std::size_t inputSamplesToSkipBeforeNextChunk(std::size_t availableSamples) {
        if constexpr (StrideControl::kEnabled) { // check if stride was removed at compile time
            const bool  isStrideActiveAndNotDefault = stride.value != 0 && stride.value != input_chunk_size;
            std::size_t toSkip                      = 0;
            if (isStrideActiveAndNotDefault && strideCounter > 0) {
                toSkip = std::min(static_cast<std::size_t>(strideCounter), availableSamples);
                strideCounter -= static_cast<gr::Size_t>(toSkip);
            }
            return toSkip;
        }
        return 0Z;
    }

    /***
     * calculate how many samples to consume taking into account stride
     * @return number of samples to consume or 0 if stride is disabled
     */
    std::size_t inputSamplesToConsumeAdjustedWithStride(std::size_t remainingSamples) {
        if constexpr (StrideControl::kEnabled) {
            const bool  isStrideActiveAndNotDefault = stride.value != 0 && stride.value != input_chunk_size;
            std::size_t toSkip                      = 0;
            if (isStrideActiveAndNotDefault && strideCounter == 0 && remainingSamples > 0) {
                toSkip        = std::min(static_cast<std::size_t>(stride.value), remainingSamples);
                strideCounter = stride.value - static_cast<gr::Size_t>(toSkip);
            }
            return toSkip;
        }
        return 0UZ;
    }

    auto computeResampling(std::size_t minSyncIn, std::size_t maxSyncIn, std::size_t minSyncOut, std::size_t maxSyncOut) {
        struct ResamplingResult {
            std::size_t  resampledIn;
            std::size_t  resampledOut;
            work::Status status = work::Status::OK;
        };

        if constexpr (!ResamplingControl::kEnabled) { // no resampling
            const std::size_t n = std::min(maxSyncIn, maxSyncOut);
            if (n < minSyncIn) {
                return ResamplingResult{.resampledIn = 0UZ, .resampledOut = 0UZ, .status = work::Status::INSUFFICIENT_INPUT_ITEMS};
            }
            if (n < minSyncOut) {
                return ResamplingResult{.resampledIn = 0UZ, .resampledOut = 0UZ, .status = work::Status::INSUFFICIENT_OUTPUT_ITEMS};
            }
            return ResamplingResult{.resampledIn = n, .resampledOut = n};
        }
        if (input_chunk_size == 1UL && output_chunk_size == 1UL) { // no resampling
            const std::size_t n = std::min(maxSyncIn, maxSyncOut);
            if (n < minSyncIn) {
                return ResamplingResult{.resampledIn = 0UZ, .resampledOut = 0UZ, .status = work::Status::INSUFFICIENT_INPUT_ITEMS};
            }
            if (n < minSyncOut) {
                return ResamplingResult{.resampledIn = 0UZ, .resampledOut = 0UZ, .status = work::Status::INSUFFICIENT_OUTPUT_ITEMS};
            }
            return ResamplingResult{.resampledIn = n, .resampledOut = n};
        }
        std::size_t nResamplingChunks;
        if constexpr (StrideControl::kEnabled) { // with stride, we cannot process more than one chunk
            if (stride.value != 0 && stride.value != input_chunk_size) {
                nResamplingChunks = input_chunk_size <= maxSyncIn && output_chunk_size <= maxSyncOut ? 1 : 0;
            } else {
                nResamplingChunks = std::min(maxSyncIn / input_chunk_size, maxSyncOut / output_chunk_size);
            }
        } else {
            nResamplingChunks = std::min(maxSyncIn / input_chunk_size, maxSyncOut / output_chunk_size);
        }

        if (nResamplingChunks * input_chunk_size < minSyncIn) {
            return ResamplingResult{.resampledIn = 0UZ, .resampledOut = 0UZ, .status = work::Status::INSUFFICIENT_INPUT_ITEMS};
        } else if (nResamplingChunks * output_chunk_size < minSyncOut) {
            return ResamplingResult{.resampledIn = 0UZ, .resampledOut = 0UZ, .status = work::Status::INSUFFICIENT_OUTPUT_ITEMS};
        } else {
            return ResamplingResult{.resampledIn = static_cast<std::size_t>(nResamplingChunks * input_chunk_size), .resampledOut = static_cast<std::size_t>(nResamplingChunks * output_chunk_size)};
        }
    }

    std::size_t getMergedBlockLimit() {
        if constexpr (requires(const Derived& d) {
                          { available_samples(d) } -> std::same_as<std::size_t>;
                      }) {
            return available_samples(self());
        } else if constexpr (traits::block::stream_input_port_types<Derived>::size == 0 && traits::block::stream_output_port_types<Derived>::size == 0) { // allow blocks that have neither input nor output ports (by merging source to sink block) -> use internal buffer size
            constexpr gr::Size_t chunkSize = Derived::merged_work_chunk_size();
            static_assert(chunkSize != std::dynamic_extent && chunkSize > 0, "At least one internal port must define a maximum number of samples or the non-member/hidden "
                                                                             "friend function `available_samples(const BlockType&)` must be defined.");
            return chunkSize;
        }
        return std::numeric_limits<std::size_t>::max();
    }

    template<typename TIn, typename TOut>
    gr::work::Status invokeProcessBulk(TIn& inputReaderTuple, TOut& outputReaderTuple) {
        auto tempInputSpanStorage = std::apply(
            []<typename... PortReader>(PortReader&... args) {
                return std::tuple{([](auto& a) {
                    if constexpr (gr::meta::array_or_vector_type<PortReader>) {
                        return std::span{a.data(), a.size()};
                    } else {
                        return a;
                    }
                }(args))...};
            },
            inputReaderTuple);

        auto tempOutputSpanStorage = std::apply([]<typename... PortReader>(PortReader&... args) { return std::tuple{(gr::meta::array_or_vector_type<PortReader> ? std::span{args.data(), args.size()} : args)...}; }, outputReaderTuple);

        auto refToSpan = []<typename T, typename U>(T&& original, U&& temporary) -> decltype(auto) {
            if constexpr (gr::meta::array_or_vector_type<std::decay_t<T>>) {
                return std::forward<U>(temporary);
            } else {
                return std::forward<T>(original);
            }
        };

        return [&]<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) { return self().processBulk(refToSpan(std::get<InIdx>(inputReaderTuple), std::get<InIdx>(tempInputSpanStorage))..., refToSpan(std::get<OutIdx>(outputReaderTuple), std::get<OutIdx>(tempOutputSpanStorage))...); }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(inputReaderTuple)>>>(), std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(outputReaderTuple)>>>());
    }

    work::Status invokeProcessOneSimd(auto& inputSpans, auto& outputSpans, auto width, std::size_t nSamplesToProcess) {
        std::size_t i = 0;
        for (; i + width <= nSamplesToProcess; i += width) {
            const auto& results = simdize_tuple_load_and_apply(width, inputSpans, i, [&](const auto&... input_simds) { return invoke_processOne_simd(i, width, input_simds...); });
            meta::tuple_for_each([i](auto& output_range, const auto& result) { result.copy_to(output_range.data() + i, stdx::element_aligned); }, outputSpans, results);
        }
        simd_epilogue(width, [&](auto w) {
            if (i + w <= nSamplesToProcess) {
                const auto results = simdize_tuple_load_and_apply(w, inputSpans, i, [&](auto&&... input_simds) { return invoke_processOne_simd(i, w, input_simds...); });
                meta::tuple_for_each([i](auto& output_range, auto& result) { result.copy_to(output_range.data() + i, stdx::element_aligned); }, outputSpans, results);
                i += w;
            }
        });
        return work::Status::OK;
    }

    work::Status invokeProcessOnePure(auto& inputSpans, auto& outputSpans, std::size_t nSamplesToProcess) {
        for (std::size_t i = 0; i < nSamplesToProcess; ++i) {
            auto results = std::apply([this, i](auto&... inputs) { return this->invoke_processOne(i, inputs[i]...); }, inputSpans);
            meta::tuple_for_each([i]<typename R>(auto& output_range, R&& result) { output_range[i] = std::forward<R>(result); }, outputSpans, results);
        }
        return work::Status::OK;
    }

    auto invokeProcessOneNonConst(auto& inputSpans, auto& outputSpans, std::size_t nSamplesToProcess) {
        using enum work::Status;

        struct ProcessOneResult {
            work::Status status;
            std::size_t  processedIn;
            std::size_t  processedOut;
        };

        std::size_t nOutSamplesBeforeRequestedStop = 0;
        for (std::size_t i = 0; i < nSamplesToProcess; ++i) {
            auto results = std::apply([this, i](auto&... inputs) { return this->invoke_processOne(i, inputs[i]...); }, inputSpans);
            meta::tuple_for_each(
                [i]<typename R>(auto& output_range, R&& result) {
                    if constexpr (meta::array_or_vector_type<std::remove_cvref<decltype(result)>>) {
                        for (int j = 0; j < result.size(); j++) {
                            output_range[i][j] = std::move(result[j]);
                        }
                    } else {
                        output_range[i] = std::forward<R>(result);
                    }
                },
                outputSpans, results);
            nOutSamplesBeforeRequestedStop++;
            // the block implementer can set `_outputTagsChanged` to true in `processOne` to prematurely leave the loop and apply his changes
            if (_outputTagsChanged || lifecycle::isShuttingDown(this->state())) [[unlikely]] { // emitted tag and/or requested to stop
                break;
            }
        }
        return ProcessOneResult{lifecycle::isShuttingDown(this->state()) ? DONE : OK, nSamplesToProcess, std::min(nSamplesToProcess, nOutSamplesBeforeRequestedStop)};
    }

    [[nodiscard]] bool hasNoDownStreamConnectedChildren() const noexcept {
        std::size_t nMandatoryChildren          = 0UZ;
        std::size_t nMandatoryConnectedChildren = 0UZ;
        for_each_port(
            [&nMandatoryChildren, &nMandatoryConnectedChildren]<PortLike Port>(const Port& outputPort) {
                if constexpr (!Port::isOptional()) {
                    nMandatoryChildren++;
                    if (outputPort.isConnected()) {
                        nMandatoryConnectedChildren++;
                    }
                }
            },
            outputPorts<PortType::STREAM>(&self()));
        return nMandatoryChildren > 0UZ && nMandatoryConnectedChildren == 0UZ;
    }

    constexpr void disconnectFromUpStreamParents() noexcept {
        using TInputTypes = traits::block::stream_input_port_types<Derived>;
        if constexpr (TInputTypes::size.value > 0UZ) {
            if (!disconnect_on_done) {
                return;
            }
            for_each_port(
                []<PortLike Port>(Port& inputPort) {
                    if (inputPort.isConnected()) {
                        std::ignore = inputPort.disconnect();
                    }
                },
                inputPorts<PortType::STREAM>(&self()));
        }
    }

    void emitMessage(std::string_view endpoint, property_map message, std::string_view clientRequestID = "") noexcept { sendMessage<message::Command::Notify>(msgOut, unique_name /* serviceName */, endpoint, std::move(message), clientRequestID); }

    void notifyListeners(std::string_view endpoint, property_map message) noexcept {
        const auto it = propertySubscriptions.find(std::string(endpoint));
        if (it != propertySubscriptions.end()) {
            for (const auto& clientID : it->second) {
                emitMessage(endpoint, message, clientID);
            }
        }
    }

    void emitErrorMessage(std::string_view endpoint, std::string_view errorMsg, std::string_view clientRequestID = "", std::source_location location = std::source_location::current()) noexcept { emitErrorMessageIfAny(endpoint, std::unexpected(Error(errorMsg, location)), clientRequestID); }

    void emitErrorMessage(std::string_view endpoint, Error e, std::string_view clientRequestID = "") noexcept { emitErrorMessageIfAny(endpoint, std::unexpected(e), clientRequestID); }

    inline void emitErrorMessageIfAny(std::string_view endpoint, std::expected<void, Error> e, std::string_view clientRequestID = "") noexcept {
        if (!e.has_value()) [[unlikely]] {
            sendMessage<message::Command::Notify>(msgOut, unique_name /* serviceName */, endpoint, std::move(e.error()), clientRequestID);
        }
    }

    /**
     * Central function managing the dispatch of work to the block implementation provided work implementation
     * @brief
     * This function performs a series of steps to handle common block mechanics and determine the amount of work to be
     * dispatched to the block-provided work implementation. It can be sub-structured into the following steps:
     * - input validation and processing
     *   - apply settings
     *   - stream tags
     *     - settings
     *     - chunk by tags or realign tags to chunks
     *       - DEFAULT: chunk s.th. that tags are always on the first sample of a chunk
     *       - MOVE_FW/MOVE_BW: move the tags to the first sample of the current/next chunk
     *       - special case EOS tag: send incomplete chunk even if it violates work/block constraints -> implementations choose to drop/pad/...
     *     - propagate tags:
     *       - in the generic case the only tag in the current chunk is on the first sample
     *       - different strategies, see TagPropagation
     *   - settings
     *      - apply and reset cached/merged tag
     *   - get available samples count
     *     - syncIn: min/max/available samples to consume on SYNC ports
     *     - syncOut: min/max/available samples to produce on SYNC ports
     *     - check whether there are available samples for any ASYNC port
     *     - limit to requestedWork
     *     - correctly consider Resampling and Stride
     *     - deprecated: available_samples limits the amount of work to produce for source blocks
     * - perform work: processBulk/One/SIMD
     * - publishing
     *   - publish tags (done first so tags are guaranteed to be fully published for all available samples)
     *   - publish out samples
     *   - consume in samples (has to be last to correctly propagate back-pressure)
     * @return struct { std::size_t produced_work, work_return_t}
     */
    work::Result workInternal(std::size_t requested_work) {
        using enum gr::work::Status;
        using TInputTypes  = traits::block::stream_input_port_types<Derived>;
        using TOutputTypes = traits::block::stream_output_port_types<Derived>;

        applyChangedSettings(); // apply settings even if the block is already stopped

        if constexpr (!blockingIO) { // N.B. no other thread/constraint to consider before shutting down
            if (this->state() == lifecycle::State::REQUESTED_STOP) {
                emitErrorMessageIfAny("workInternal(): REQUESTED_STOP -> STOPPED", this->changeStateTo(lifecycle::State::STOPPED));
            }
        }

        using TOutputTypes = traits::block::stream_output_port_types<Derived>;
        if constexpr (TOutputTypes::size.value > 0UZ) {
            if (disconnect_on_done && hasNoDownStreamConnectedChildren()) {
                this->requestStop(); // no dependent non-optional children, should stop processing
            }
        }

        if (this->state() == lifecycle::State::STOPPED) {
            disconnectFromUpStreamParents();
            return {requested_work, 0UZ, DONE};
        }

        // evaluate number of available and processable samples
        const auto [minSyncIn, maxSyncIn, maxSyncAvailableIn, hasAsyncIn]     = getPortLimits(inputPorts<PortType::STREAM>(&self()));
        const auto [minSyncOut, maxSyncOut, maxSyncAvailableOut, hasAsyncOut] = getPortLimits(outputPorts<PortType::STREAM>(&self()));
        auto [hasTag, nextTag, nextEosTag, asyncEoS]                          = getNextTagAndEosPosition();
        std::size_t maxChunk                                                  = getMergedBlockLimit(); // handle special cases for merged blocks. TODO: evaluate if/how we can get rid of these
        const auto  inputSkipBefore                                           = inputSamplesToSkipBeforeNextChunk(std::min({maxSyncAvailableIn, nextTag, nextEosTag}));
        const auto  nextTagLimit                                              = (nextTag - inputSkipBefore) >= minSyncIn ? (nextTag - inputSkipBefore) : std::numeric_limits<std::size_t>::max();
        const auto  ensureMinimalDecimation                                   = nextTagLimit >= input_chunk_size ? nextTagLimit : static_cast<long unsigned int>(input_chunk_size); // ensure to process at least one input_chunk_size (may shift tags)
        const auto  availableToProcess                                        = std::min({maxSyncIn, maxChunk, (maxSyncAvailableIn - inputSkipBefore), ensureMinimalDecimation, (nextEosTag - inputSkipBefore)});
        const auto  availableToPublish                                        = std::min({maxSyncOut, maxSyncAvailableOut});
        const auto [resampledIn, resampledOut, resampledStatus]               = computeResampling(std::min(minSyncIn, nextEosTag), availableToProcess, minSyncOut, availableToPublish);
        const auto nextEosTagSkipBefore                                       = nextEosTag - inputSkipBefore;
        const bool isEosTagPresent                                            = nextEosTag <= 0 || nextEosTagSkipBefore < minSyncIn || nextEosTagSkipBefore < input_chunk_size || output_chunk_size * (nextEosTagSkipBefore / input_chunk_size) < minSyncOut;

        if (inputSkipBefore > 0) {                                                                          // consume samples on sync ports that need to be consumed due to the stride
            updateInputAndOutputTags(inputSkipBefore);                                                      // apply all tags in the skipped data range
            const auto inputSpans = prepareStreams(inputPorts<PortType::STREAM>(&self()), inputSkipBefore); // only way to consume is via the ConsumableSpan now
            consumeReaders(inputSkipBefore, inputSpans);
        }
        // return if there is no work to be performed // todo: add eos policy
        if (isEosTagPresent || lifecycle::isShuttingDown(this->state()) || asyncEoS) {
            emitErrorMessageIfAny("workInternal(): EOS tag arrived -> REQUESTED_STOP", this->changeStateTo(lifecycle::State::REQUESTED_STOP));
            publishEoS();
            this->setAndNotifyState(lifecycle::State::STOPPED);
            return {requested_work, 0UZ, DONE};
        }
        if (asyncEoS || (resampledIn == 0 && resampledOut == 0 && !hasAsyncIn && !hasAsyncOut)) {
            return {requested_work, 0UZ, resampledStatus};
        }

        // for non-bulk processing, the processed span has to be limited to the first sample if it contains a tag s.t. the tag is not applied to every sample
        const bool limitByFirstTag = (!HasProcessBulkFunction<Derived> && HasProcessOneFunction<Derived>) && hasTag;

        // call the block implementation's work function
        work::Status userReturnStatus = ERROR; // default if nothing has been set
        std::size_t  processedIn      = limitByFirstTag ? 1UZ : resampledIn;
        std::size_t  processedOut     = limitByFirstTag ? 1UZ : resampledOut;
        const auto   inputSpans       = prepareStreams(inputPorts<PortType::STREAM>(&self()), processedIn);
        auto         outputSpans      = prepareStreams(outputPorts<PortType::STREAM>(&self()), processedOut);

        if (containsEmptyOutputSpans(outputSpans)) {
            return {requested_work, 0UZ, INSUFFICIENT_OUTPUT_ITEMS};
        }

        updateInputAndOutputTags();
        applyChangedSettings();

        if constexpr (HasProcessBulkFunction<Derived>) {
            invokeUserProvidedFunction("invokeProcessBulk", [&userReturnStatus, &inputSpans, &outputSpans, this] noexcept(HasNoexceptProcessBulkFunction<Derived>) { userReturnStatus = invokeProcessBulk(inputSpans, outputSpans); });
            meta::tuple_for_each(
                [&processedIn]<typename TIn>(TIn& in) {
                    if constexpr (ConsumableSpan<TIn>) {
                        if (in.isConsumeRequested()) {
                            processedIn = std::min(processedIn, in.getConsumeRequested());
                        }
                    } else if constexpr (ConsumableSpan<typename TIn::value_type>) {
                        for (auto& span : in) {
                            if (span.isConsumeRequested()) {
                                processedIn = std::min(processedIn, span.getConsumeRequested());
                            }
                        }
                    }
                },
                inputSpans);
            meta::tuple_for_each(
                [&processedOut]<typename TOut>(TOut& out) {
                    if constexpr (PublishableSpan<TOut>) {
                        if (out.isPublished()) {
                            processedOut = std::min(processedOut, out.samplesToPublish());
                        }
                    } else if constexpr (PublishableSpan<typename TOut::value_type>) {
                        for (auto& span : out) {
                            if (span.isPublished()) {
                                processedOut = std::min(processedOut, span.samplesToPublish());
                            }
                        }
                    }
                },
                outputSpans);
        } else if constexpr (HasProcessOneFunction<Derived>) {
            if (processedIn != processedOut) {
                emitErrorMessage("Block::workInternal:", fmt::format("N input samples ({}) does not equal to N output samples ({}) for processOne() method.", resampledIn, resampledOut));
                requestStop();
                processedIn  = 0;
                processedOut = 0;
            } else {
                using input_simd_types  = meta::simdize<typename TInputTypes::template apply<std::tuple>>;
                using output_simd_types = meta::simdize<typename TOutputTypes::template apply<std::tuple>>;

                constexpr auto                                 input_types_simd_size = meta::simdize_size_v<input_simd_types>;
                constexpr std::size_t                          max_simd_double_size  = stdx::simd_abi::max_fixed_size<double>;
                constexpr std::size_t                          simd_size             = input_types_simd_size == 0 ? max_simd_double_size : std::min(max_simd_double_size, input_types_simd_size * 4);
                std::integral_constant<std::size_t, simd_size> width{};

                if constexpr ((meta::simdize_size_v<output_simd_types> != 0) and ((requires(Derived& d) {
                                  { d.processOne_simd(simd_size) };
                              }) or (meta::simdize_size_v<input_simd_types> != 0 and traits::block::can_processOne_simd<Derived>))) { // SIMD loop
                    invokeUserProvidedFunction("invokeProcessOneSimd", [&userReturnStatus, &inputSpans, &outputSpans, &width, &processedIn, this] noexcept(HasNoexceptProcessOneFunction<Derived>) { userReturnStatus = invokeProcessOneSimd(inputSpans, outputSpans, width, processedIn); });
                } else {                                                 // Non-SIMD loop
                    if constexpr (HasConstProcessOneFunction<Derived>) { // processOne is const -> can process whole batch similar to SIMD-ised call
                        invokeUserProvidedFunction("invokeProcessOnePure", [&userReturnStatus, &inputSpans, &outputSpans, &processedIn, this] noexcept(HasNoexceptProcessOneFunction<Derived>) { userReturnStatus = invokeProcessOnePure(inputSpans, outputSpans, processedIn); });
                    } else { // processOne isn't const i.e. not a pure function w/o side effects -> need to evaluate state after each sample
                        const auto result = invokeProcessOneNonConst(inputSpans, outputSpans, processedIn);
                        userReturnStatus  = result.status;
                        processedIn       = result.processedIn;
                        processedOut      = result.processedOut;
                    }
                }
            }
        } else { // block does not define any valid processing function
            static_assert(meta::always_false<traits::block::stream_input_port_types_tuple<Derived>>, "neither processBulk(...) nor processOne(...) implemented");
        }

        // sanitise input/output samples based on explicit user-defined processBulk(...) return status
        if (userReturnStatus == INSUFFICIENT_OUTPUT_ITEMS || userReturnStatus == INSUFFICIENT_INPUT_ITEMS || userReturnStatus == ERROR) {
            processedIn  = 0UZ;
            processedOut = 0UZ;
        }

        if (processedIn > 0 && processedOut > 0) {
            forwardTags();
        }
        if (lifecycle::isShuttingDown(this->state())) {
            emitErrorMessageIfAny("isShuttingDown -> STOPPED", this->changeStateTo(lifecycle::State::REQUESTED_STOP));
            applyChangedSettings();
            userReturnStatus = DONE;
            processedIn      = 0UZ;
        }

        // publish/consume
        publishSamples(processedOut, outputSpans);
        if (processedIn == 0UZ) {
            consumeReaders(0UZ, inputSpans);
        } else {
            const auto inputSamplesToConsume = inputSamplesToConsumeAdjustedWithStride(resampledIn);
            if (inputSamplesToConsume > 0) {
                if (!consumeReaders(inputSamplesToConsume, inputSpans)) {
                    userReturnStatus = ERROR;
                }
            } else {
                if (!consumeReaders(processedIn, inputSpans)) {
                    userReturnStatus = ERROR;
                }
            }
        }

        // if the block state changed to DONE, publish EOS tag on the next sample
        if (userReturnStatus == DONE) {
            this->setAndNotifyState(lifecycle::State::STOPPED);
            publishEoS();
        }
        for_each_port([](PortLike auto& outPort) { outPort.publishPendingTags(); }, outputPorts<PortType::STREAM>(&self()));

        // check/sanitise return values (N.B. these are used by the scheduler as indicators
        // whether and how much 'work' has been done to -- for example -- prioritise one block over another
        std::size_t performedWork = 0UZ;
        if (userReturnStatus == OK) {
            constexpr bool kIsSourceBlock = traits::block::stream_input_port_types<Derived>::size == 0;
            constexpr bool kIsSinkBlock   = traits::block::stream_output_port_types<Derived>::size == 0;
            if constexpr (kIsSourceBlock && kIsSinkBlock) {
                performedWork = processedIn;
            } else if constexpr (kIsSinkBlock) {
                performedWork = processedIn;
            } else if constexpr (kIsSourceBlock) {
                performedWork = processedOut;
            } else {
                performedWork = 1UZ;
            }

            progress->incrementAndGet();
            if constexpr (blockingIO) {
                progress->notify_all();
            }
        }
        return {requested_work, performedWork, userReturnStatus};
    } // end: work::Result workInternal(std::size_t requested_work) { ... }

public:
    work::Status invokeWork()
    requires(blockingIO)
    {
        auto [work_requested, work_done, last_status] = workInternal(std::atomic_load_explicit(&ioRequestedWork, std::memory_order_acquire));
        ioWorkDone.increment(work_requested, work_done);
        ioLastWorkStatus.exchange(last_status, std::memory_order_relaxed);
        return last_status;
    }

    /**
     * @brief Process as many samples as available and compatible with the internal boundary requirements or limited by 'requested_work`
     *
     * @param requested_work: usually the processed number of input samples, but could be any other metric as long as
     * requested_work limit as an affine relation with the returned performed_work.
     * @return { requested_work, performed_work, status}
     */
    template<typename = void>
    work::Result work(std::size_t requested_work = std::numeric_limits<std::size_t>::max()) noexcept
    requires(!blockingIO) // regular non-blocking call
    {
        return workInternal(requested_work);
    }

    /**
     * @brief Process as many samples as available and compatible with the internal boundary requirements or limited by 'requested_work`
     *
     * @param requested_work: usually the processed number of input samples, but could be any other metric as long as
     * requested_work limit as an affine relation with the returned performed_work.
     * @return { requested_work, performed_work, status}
     */
    template<typename = void>
    work::Result work(std::size_t requested_work = std::numeric_limits<std::size_t>::max()) noexcept
    requires(blockingIO) // regular blocking call (e.g. wating on HW, timer, blocking for any other reasons) -> this should be an exceptional use
    {
        constexpr bool useIoThread = std::disjunction_v<std::is_same<BlockingIO<true>, Arguments>...>;
        std::atomic_store_explicit(&ioRequestedWork, requested_work, std::memory_order_release);

        bool expectedThreadState = false;
        if (lifecycle::isActive(this->state()) && this->ioThreadRunning.compare_exchange_strong(expectedThreadState, true, std::memory_order_acq_rel)) {
            if constexpr (useIoThread) { // use graph-provided ioThreadPool
                if (!ioThreadPool) {
                    emitErrorMessage("work(..)", "blockingIO with useIoThread - no ioThreadPool being set");
                    return {requested_work, 0UZ, work::Status::ERROR};
                }
                ioThreadPool->execute([this]() {
                    assert(lifecycle::isActive(this->state()));

                    lifecycle::State actualThreadState = this->state();
                    while (lifecycle::isActive(actualThreadState)) {
                        // execute ten times before testing actual state -- minimises overhead atomic load to work execution if the latter is a noop or very fast to execute
                        for (std::size_t testState = 0UZ; testState < 10UZ; ++testState) {
                            if (invokeWork() == work::Status::DONE) {
                                actualThreadState = lifecycle::State::REQUESTED_STOP;
                                emitErrorMessageIfAny("REQUESTED_STOP -> REQUESTED_STOP", this->changeStateTo(lifecycle::State::REQUESTED_STOP));
                                break;
                            }
                        }
                        actualThreadState = this->state();
                    }
                    emitErrorMessageIfAny("-> STOPPED", this->changeStateTo(lifecycle::State::STOPPED));
                    ioThreadRunning.store(false);
                });
            } else { // use user-provided ioThreadPool
                // let user call 'work' explicitly and set both 'ioWorkDone' and 'ioLastWorkStatus'
            }
        }
        if constexpr (!useIoThread) {
            const bool blockIsActive = lifecycle::isActive(this->state());
            if (!blockIsActive) {
                publishEoS();
                ioLastWorkStatus.exchange(work::Status::DONE, std::memory_order_relaxed);
            }
        }

        const auto& [accumulatedRequestedWork, performedWork] = ioWorkDone.getAndReset();
        // TODO: this is just "working" solution for deadlock with emscripten, need to be investigated further
#if defined(__EMSCRIPTEN__)
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
#endif
        return {accumulatedRequestedWork, performedWork, ioLastWorkStatus.load()};
    }

    void processMessages([[maybe_unused]] const MsgPortInNamed<"__Builtin">& port, std::span<const Message> messages) {
        using enum gr::message::Command;
        assert(std::addressof(port) == std::addressof(msgIn) && "got a message on wrong port");

        for (const auto& message : messages) {
            if (!message.serviceName.empty() && message.serviceName != unique_name && message.serviceName != name) {
                // Skip if target does not match the block's (unique) name and is not empty.
                continue;
            }

            PropertyCallback callback = nullptr;
            // Attempt to find a matching property callback or use the unmatchedPropertyHandler.
            if (auto it = propertyCallbacks.find(message.endpoint); it != propertyCallbacks.end()) {
                callback = it->second;
            } else {
                if constexpr (requires(std::string_view sv, Message m) {
                                  { self().unmatchedPropertyHandler(sv, m) } -> std::same_as<std::optional<Message>>;
                              }) {
                    callback = &Derived::unmatchedPropertyHandler;
                }
            }

            if (callback == nullptr) {
                continue; // did not find matching property callback
            }

            std::optional<Message> retMessage;
            try {
                retMessage = (self().*callback)(message.endpoint, message); // N.B. life-time: message is copied
            } catch (const gr::exception& e) {
                retMessage       = Message{message};
                retMessage->data = std::unexpected(Error(e));
            } catch (const std::exception& e) {
                retMessage       = Message{message};
                retMessage->data = std::unexpected(Error(e));
            } catch (...) {
                retMessage       = Message{message};
                retMessage->data = std::unexpected(Error(fmt::format("unknown exception in Block {} property '{}'\n request message: {} ", unique_name, message.endpoint, message)));
            }

            if (!retMessage.has_value()) {
                continue; // function does not produce any return message
            }

            retMessage->cmd              = Final; // N.B. could enable/allow for partial if we return multiple messages (e.g. using coroutines?)
            retMessage->serviceName      = unique_name;
            PublishableSpan auto msgSpan = msgOut.streamWriter().tryReserve<SpanReleasePolicy::ProcessAll>(1UZ);
            if (msgSpan.empty()) {
                throw gr::exception(fmt::format("{}::processMessages() can not reserve span for message\n", name));
            } else {
                msgSpan[0] = *retMessage;
            }
        } // - end - for (const auto &message : messages) { ..
    }

}; // template<typename Derived, typename... Arguments> class Block : ...

namespace detail {
template<typename List, std::size_t Index = 0, typename StringFunction>
inline constexpr auto for_each_type_to_string(StringFunction func) -> std::string {
    if constexpr (Index < List::size) {
        using T = typename List::template at<Index>;
        return std::string(Index > 0 ? ", " : "") + func(Index, T()) + for_each_type_to_string<List, Index + 1>(func);
    } else {
        return "";
    }
}

template<typename T>
inline constexpr std::string container_type_name() {
    if constexpr (requires { typename T::allocator_type; }) {
        return fmt::format("std::vector<{}>", gr::meta::type_name<typename T::value_type>());
    } else if constexpr (requires { std::tuple_size<T>::value; }) {
        return fmt::format("std::array<{}, {}>", gr::meta::type_name<typename T::value_type>(), std::tuple_size<T>::value);
    } else if constexpr (requires(T a) {
                             { std::real(a) } -> std::convertible_to<typename T::value_type>;
                             { std::imag(a) } -> std::convertible_to<typename T::value_type>;
                         }) {
        return fmt::format("std::complex<{}>", gr::meta::type_name<typename T::value_type>());
    } else { // fallback
        return gr::meta::type_name<T>();
    }
}
} // namespace detail

template<typename TBlock, typename TDecayedBlock>
inline void checkBlockContracts() {
    // N.B. some checks could be evaluated during compile time but the expressed intent is to do this during runtime to allow
    // for more verbose feedback on method signatures etc.
    constexpr static auto processMembers = []<typename Func>(Func func) {
        if constexpr (detail::HasBaseType<TDecayedBlock>) {
            using BaseType = typename TDecayedBlock::base_t;
            if constexpr (refl::is_reflectable<BaseType>()) {
                refl::util::for_each(refl::reflect<BaseType>().members, func);
            }
        }
        if constexpr (refl::is_reflectable<TDecayedBlock>()) {
            refl::util::for_each(refl::reflect<TDecayedBlock>().members, func);
        }
    };

    constexpr static auto shortTypeName = []<typename T>() {
        if constexpr (std::is_same_v<T, gr::property_map>) {
            return "gr::property_map";
        } else if constexpr (std::is_same_v<T, std::string>) {
            return "std::string";
        } else if constexpr (requires { typename T::value_type; }) {
            return detail::container_type_name<T>();
        } else {
            return gr::meta::type_name<T>();
        }
    };

    constexpr static auto checkSettingsTypes = [](auto member) {
        using MemberType           = decltype(member)::value_type;
        using RawType              = std::remove_cvref_t<MemberType>;
        using Type                 = std::remove_cvref_t<unwrap_if_wrapped_t<RawType>>;
        constexpr bool isAnnotated = !std::is_same_v<RawType, Type>;
        // N.B. this function is compile-time ready but static_assert does not allow for configurable error messages
        if constexpr (!gr::settings::isSupportedType<Type>() && !(traits::port::is_port_v<Type> || traits::port::is_port_collection_v<Type>)) {
            throw std::invalid_argument(fmt::format("block {} {}member '{}' has unsupported setting type '{}'", //
                gr::meta::type_name<TDecayedBlock>(), isAnnotated ? "" : "annotated ", get_display_name(member), shortTypeName.template operator()<Type>()));
        }
    };
    processMembers(checkSettingsTypes);

    using TDerived = typename TDecayedBlock::derived_t;
    if constexpr (requires { &TDerived::work; }) {
        // N.B. implementing this is still allowed for workaround but should be discouraged as default API since this often leads to
        // important variants not being implemented such as lifecycle::State handling, Tag forwarding, etc.
        return;
    }

    using TInputTypes  = traits::block::stream_input_port_types<TDerived>;
    using TOutputTypes = traits::block::stream_output_port_types<TDerived>;

    if constexpr (((TInputTypes::size.value + TOutputTypes::size.value) > 0UZ) && !gr::HasRequiredProcessFunction<TDecayedBlock>) {
        const auto b1 = (TOutputTypes::size.value == 1UZ) ? "" : "{ "; // optional opening brackets
        const auto b2 = (TOutputTypes::size.value == 1UZ) ? "" : " }"; // optional closing brackets
                                                                       // clang-format off
        std::string signatureProcessOne = fmt::format("* Option Ia (pure function):\n\n{}\n\n* Option Ib (allows modifications: settings, Tags, state, errors,...):\n\n{}\n\n* Option Ic (explicit return types):\n\n{}\n\n", //
fmt::format(R"(auto processOne({}) const noexcept {{
    /* add code here */
    return {}{}{};
}})",
    detail::for_each_type_to_string<TInputTypes>([]<typename T>(auto index, T) { return fmt::format("{} in{}", shortTypeName.template operator()<T>(), index); }),
    b1, detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto, T) { return fmt::format("{}()", shortTypeName.template operator()<T>()); }), b2),
fmt::format(R"(auto processOne({}) {{
    /* add code here */
    return {}{}{};
}})",
    detail::for_each_type_to_string<TInputTypes>([]<typename T>(auto index, T) { return fmt::format("{} in{}", shortTypeName.template operator()<T>(), index); }),
    b1, detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto, T) { return fmt::format("{}()", shortTypeName.template operator()<T>()); }), b2),
fmt::format(R"(std::tuple<{}> processOne({}) {{
    /* add code here */
    return {}{}{};
}})",
   detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto, T) { return fmt::format("{}", shortTypeName.template operator()<T>()); }), //
   detail::for_each_type_to_string<TInputTypes>([]<typename T>(auto index, T) { return fmt::format("{} in{}", shortTypeName.template operator()<T>(), index); }), //
   b1, detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto, T) { return fmt::format("{}()", shortTypeName.template operator()<T>()); }), b2)
);

std::string signaturesProcessBulk = fmt::format("* Option II:\n\n{}\n\nadvanced:* Option III:\n\n{}\n\n\n",
fmt::format(R"(gr::work::Status processBulk({}{}{}) {{
    /* add code here */
    return gr::work::Status::OK;
}})", //
    detail::for_each_type_to_string<TInputTypes>([]<typename T>(auto index, T) { return fmt::format("std::span<const {}> in{}", shortTypeName.template operator()<T>(), index); }), //
    (TInputTypes::size == 0UZ || TOutputTypes::size == 0UZ ? "" : ", "),                                                                             //
    detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto index, T) { return fmt::format("std::span<{}> out{}", shortTypeName.template operator()<T>(), index); })),
fmt::format(R"(gr::work::Status processBulk({}{}{}) {{
    /* add code here */
    return gr::work::Status::OK;
}})", //
    detail::for_each_type_to_string<TInputTypes>([]<typename T>(auto index, T) { return fmt::format("std::span<const {}> in{}", shortTypeName.template operator()<T>(), index); }), //
    (TInputTypes::size == 0UZ || TOutputTypes::size == 0UZ ? "" : ", "),                                                                             //
    detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto index, T) { return fmt::format("PublishableSpan auto out{}", shortTypeName.template operator()<T>(), index); })));
        // clang-format on

        bool has_port_collection = false;
        TInputTypes::for_each([&has_port_collection]<typename T>(auto, T) { has_port_collection |= requires { typename T::value_type; }; });
        TOutputTypes::for_each([&has_port_collection]<typename T>(auto, T) { has_port_collection |= requires { typename T::value_type; }; });
        const std::string signatures = (has_port_collection ? "" : signatureProcessOne) + signaturesProcessBulk;
        throw std::invalid_argument(fmt::format("block {} has neither a valid processOne(...) nor valid processBulk(...) method\nPossible valid signatures (copy-paste):\n\n{}", shortTypeName.template operator()<TDecayedBlock>(), signatures));
    }

    // test for optional Drawable interface
    if constexpr (!std::is_same_v<NotDrawable, typename TDecayedBlock::DrawableControl> && !requires(TDecayedBlock t) {
                      { t.draw() } -> std::same_as<work::Status>;
                  }) {
        static_assert(gr::meta::always_false<TDecayedBlock>, "annotated Block<Derived, Drawable<...>, ...> must implement 'work::Status draw() {}'");
    }
}

template<typename Derived, typename... Arguments>
inline std::atomic_size_t Block<Derived, Arguments...>::_uniqueIdCounter{0UZ};
} // namespace gr

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, typename... Arguments), (gr::Block<T, Arguments...>), input_chunk_size, output_chunk_size, stride, disconnect_on_done, unique_name, name, meta_information);

namespace gr {

/**
 * @brief a short human-readable/markdown description of the node -- content is not contractual and subject to change
 */
template<BlockLike TBlock>
[[nodiscard]] /*constexpr*/ std::string blockDescription() noexcept {
    using DerivedBlock         = typename TBlock::derived_t;
    using ArgumentList         = typename TBlock::block_template_parameters;
    using SupportedTypes       = typename ArgumentList::template find_or_default<is_supported_types, DefaultSupportedTypes>;
    constexpr bool kIsBlocking = ArgumentList::template contains<BlockingIO<true>> || ArgumentList::template contains<BlockingIO<false>>;

    // re-enable once string and constexpr static is supported by all compilers
    /*constexpr*/ std::string ret = fmt::format("# {}\n{}\n{}\n**supported data types:**", //
        gr::meta::type_name<DerivedBlock>(), TBlock::description, kIsBlocking ? "**BlockingIO**\n_i.e. potentially non-deterministic/non-real-time behaviour_\n" : "");
    gr::meta::typelist<SupportedTypes>::for_each([&](std::size_t index, auto&& t) {
        std::string type_name = gr::meta::type_name<decltype(t)>();
        ret += fmt::format("{}:{} ", index, type_name);
    });
    ret += fmt::format("\n**Parameters:**\n");
    if constexpr (refl::is_reflectable<DerivedBlock>()) {
        for_each(refl::reflect<DerivedBlock>().members, [&]<typename TFieldMeta>(TFieldMeta member) {
            using RawType = std::remove_cvref_t<typename TFieldMeta::value_type>;
            using Type    = unwrap_if_wrapped_t<RawType>;

            if constexpr (is_readable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string>)) {
                if constexpr (is_annotated<RawType>()) {
                    ret += fmt::format("{}{:10} {:<20} - annotated info: {} unit: [{}] documentation: {}{}\n",
                        RawType::visible() ? "" : "_", //
                        refl::detail::get_type_name<Type>().c_str(),
                        get_display_name_const(member).c_str(), //
                        RawType::description(), RawType::unit(),
                        RawType::documentation(), //
                        RawType::visible() ? "" : "_");
                } else {
                    ret += fmt::format("_{:10} {}_\n", refl::detail::get_type_name<Type>().c_str(), get_display_name_const(member).c_str());
                }
            }
        });
    }
    ret += fmt::format("\n~~Ports:~~\ntbd.");
    return ret;
}

namespace detail {

template<typename Type>
std::string reflFirstTypeName() {
    //
    // Using refl cpp for getting names of types does not work
    // with class templates. It returns "Template<T...>" as the name
    // instead of replacing "T..." with the names of types.
    //
    // Until we get proper reflection support in C++, we need to
    // cover the special cases manually.
    //
    if constexpr (DataSetLike<Type>) {
        return fmt::format("gr::DataSet<{}>", reflFirstTypeName<typename Type::value_type>());
    } else if constexpr (UncertainValueLike<Type>) {
        return fmt::format("gr::UncertainValue<{}>", reflFirstTypeName<typename Type::value_type>());
    } else if constexpr (refl::is_reflectable<Type>()) {
        return refl::reflect<Type>().name.str();

    } else {
        return meta::type_name<Type>;
    }
}

template<typename... Types>
std::string encodeListOfTypes() {
    using namespace std::string_literals;
    struct accumulator {
        std::string value;

        accumulator& operator%(const std::string& type) {
            if (value.empty()) {
                value = type;
            } else {
                value += ","s + type;
            }

            return *this;
        }
    };

    return (accumulator{} % ... % reflFirstTypeName<Types>()).value;
}

template<typename TBlock>
std::string blockBaseName() {
    auto blockName = reflFirstTypeName<TBlock>();
    auto it        = std::ranges::find(blockName, '<');
    return std::string(blockName.begin(), it);
}

template<auto Value>
std::string nttpToString() {
    if constexpr (magic_enum::is_scoped_enum_v<decltype(Value)> || magic_enum::is_unscoped_enum_v<decltype(Value)>) {
        return std::string(magic_enum::enum_name(Value));
    } else {
        return std::to_string(Value);
    }
}
} // namespace detail

template<typename... Types>
struct BlockParameters : meta::typelist<Types...> {
    static std::string toString() { return detail::encodeListOfTypes<Types...>(); }
};

/**
 * This function (and overloads) can be used to register a block with
 * the block registry to be used for runtime instantiation of blocks
 * based on their stringified types.
 *
 * The arguments are:
 *  - registerInstance -- a reference to the registry (common to use gr::globalBlockRegistry)
 *  - TBlock -- the block class template
 *  - Value0 and Value1 -- if the block has non-template-type parameters,
 *    set these to the values of NTTPs you want to register
 *  - TBlockParameters -- types that the block can be instantiated with
 */
template<template<typename> typename TBlock, typename... TBlockParameters, typename TRegisterInstance>
inline constexpr int registerBlock(TRegisterInstance& registerInstance) {
    auto addBlockType = [&]<typename Type> {
        using ThisBlock = TBlock<Type>;
        static_assert(!meta::is_instantiation_of<Type, BlockParameters>);
        registerInstance.template addBlockType<ThisBlock>(detail::blockBaseName<TBlock<Type>>(), detail::reflFirstTypeName<Type>());
    };
    ((addBlockType.template operator()<TBlockParameters>()), ...);
    return {};
}

/**
 * This function can be used to register a block with two templated types with the block registry
 * to be used for runtime instantiation of blocks based on their stringified types.
 *
 * The arguments are:
 *  - registerInstance -- a reference to the registry (common to use gr::globalBlockRegistry)
 *  - TBlock -- the block class template with two template parameters
 *  - Tuple1 -- a std::tuple containing the types for the first template parameter of TBlock
 *  - Tuple2 -- a std::tuple containing the types for the second template parameter of TBlock
 *
 * This function iterates over all combinations of the types in Tuple1 and Tuple2,
 * instantiates TBlock with each combination, and registers the block with the registry.
 */
template<template<typename, typename> typename TBlock, typename Tuple1, typename Tuple2, typename TRegisterInstance>
inline constexpr int registerBlockTT(TRegisterInstance& registerInstance) {
    auto addBlockType = [&]<typename Type1, typename Type2> {
        using ThisBlock = TBlock<Type1, Type2>;
        registerInstance.template addBlockType<ThisBlock>( //
            detail::blockBaseName<ThisBlock>(), detail::reflFirstTypeName<Type1>() + "," + detail::reflFirstTypeName<Type2>());
    };

    std::apply(
        [&]<typename... T1>(T1...) { // iterate over first type
            std::apply(
                [&]<typename... T2>(T2...) { // iterate over second type
                    (([&]<typename Type1>() { ((addBlockType.template operator()<Type1, T2>()), ...); }.template operator()<T1>()), ...);
                },
                Tuple2{});
        },
        Tuple1{});

    return {};
}

template<template<typename, typename> typename TBlock, typename... TBlockParameters, typename TRegisterInstance>
inline constexpr int registerBlock(TRegisterInstance& registerInstance) {
    auto addBlockType = [&]<typename Type> {
        using ThisBlock = TBlock<typename Type::template at<0>, typename Type::template at<1>>;
        static_assert(meta::is_instantiation_of<Type, BlockParameters>);
        static_assert(Type::size == 2);
        registerInstance.template addBlockType<ThisBlock>(detail::blockBaseName<ThisBlock>(), Type::toString());
    };
    ((addBlockType.template operator()<TBlockParameters>()), ...);
    return {};
}

template<template<typename, auto> typename TBlock, auto Value0, typename... TBlockParameters, typename TRegisterInstance>
inline constexpr int registerBlock(TRegisterInstance& registerInstance) {
    auto addBlockType = [&]<typename Type> {
        static_assert(!meta::is_instantiation_of<Type, BlockParameters>);
        using ThisBlock = TBlock<Type, Value0>;
        registerInstance.template addBlockType<ThisBlock>(detail::blockBaseName<ThisBlock>(), //
            detail::reflFirstTypeName<Type>() + "," + detail::nttpToString<Value0>());
    };
    ((addBlockType.template operator()<TBlockParameters>()), ...);
    return {};
}

template<template<typename, typename, auto> typename TBlock, auto Value0, typename... TBlockParameters, typename TRegisterInstance>
inline constexpr int registerBlock(TRegisterInstance& registerInstance) {
    auto addBlockType = [&]<typename Type> {
        static_assert(meta::is_instantiation_of<Type, BlockParameters>);
        static_assert(Type::size == 2);
        using ThisBlock = TBlock<typename Type::template at<0>, typename Type::template at<1>, Value0>;
        registerInstance.template addBlockType<ThisBlock>(detail::blockBaseName<ThisBlock>(), //
            Type::toString() + "," + detail::nttpToString<Value0>());
    };
    ((addBlockType.template operator()<TBlockParameters>()), ...);
    return {};
}

template<template<typename, auto, auto> typename TBlock, auto Value0, auto Value1, typename... TBlockParameters, typename TRegisterInstance>
inline constexpr int registerBlock(TRegisterInstance& registerInstance) {
    auto addBlockType = [&]<typename Type> {
        static_assert(!meta::is_instantiation_of<Type, BlockParameters>);
        using ThisBlock = TBlock<Type, Value0, Value1>;
        registerInstance.template addBlockType<ThisBlock>(detail::blockBaseName<ThisBlock>(), //
            detail::reflFirstTypeName<Type>() + "," + detail::nttpToString<Value0>() + "," + detail::nttpToString<Value1>());
    };
    ((addBlockType.template operator()<TBlockParameters>()), ...);
    return {};
}

template<template<typename, typename, auto, auto> typename TBlock, auto Value0, auto Value1, typename... TBlockParameters, typename TRegisterInstance>
inline constexpr int registerBlock(TRegisterInstance& registerInstance) {
    auto addBlockType = [&]<typename Type> {
        static_assert(meta::is_instantiation_of<Type, BlockParameters>);
        static_assert(Type::size == 2);
        using ThisBlock = TBlock<typename Type::template at<0>, typename Type::template at<1>, Value0, Value1>;
        registerInstance.template addBlockType<ThisBlock>(detail::blockBaseName<ThisBlock>(), //
            Type::toString() + "," + detail::nttpToString<Value0>() + "," + detail::nttpToString<Value1>());
    };
    ((addBlockType.template operator()<TBlockParameters>()), ...);
    return {};
}

template<typename Function, typename Tuple, typename... Tuples>
inline constexpr auto for_each_port(Function&& function, Tuple&& tuple, Tuples&&... tuples) {
    return gr::meta::tuple_for_each(
        [&function](auto&&... args) {
            (..., ([&function](auto&& arg) {
                using ArgType = std::decay_t<decltype(arg)>;
                if constexpr (traits::port::is_port_v<ArgType>) {
                    function(arg); // arg is a port, apply function directly
                } else if constexpr (traits::port::is_port_collection_v<ArgType>) {
                    for (auto& port : arg) { // arg is a collection of ports, apply function to each port
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
    static constexpr auto parse(const format_parse_context& ctx) {
        const auto it = ctx.begin();
        if (it != ctx.end() && *it != '}') {
            throw format_error("invalid format");
        }
        return it;
    }

    template<typename FormatContext>
    auto format(const gr::work::Result& work_return, FormatContext& ctx) {
        return fmt::format_to(ctx.out(), "requested_work: {}, performed_work: {}, status: {}", work_return.requested_work, work_return.performed_work, magic_enum::enum_name(work_return.status));
    }
};

#endif // include guard
