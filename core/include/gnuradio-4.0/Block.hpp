#ifndef GNURADIO_BLOCK_HPP
#define GNURADIO_BLOCK_HPP

#include <limits>
#include <map>
#include <source_location>

#include <format>

#include <gnuradio-4.0/meta/RangesHelper.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/immutable.hpp>
#include <gnuradio-4.0/meta/typelist.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Sequence.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <gnuradio-4.0/Settings.hpp>
#include <gnuradio-4.0/annotated.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <gnuradio-4.0/LifeCycle.hpp>

namespace gr {

namespace stdx = vir::stdx;
using gr::meta::fixed_string;

template<typename F>
constexpr void simd_epilogue(auto kWidth, F&& fun) {
    using namespace vir::literals;
    static_assert(std::has_single_bit(unsigned(kWidth)));
    auto kHalfWidth = kWidth / 2_cw;
    if constexpr (kHalfWidth > 0) {
        fun(kHalfWidth);
        simd_epilogue(kHalfWidth, std::forward<F>(fun));
    }
}

template<std::ranges::contiguous_range... Ts, typename Flag = stdx::element_aligned_tag>
constexpr auto simdize_tuple_load_and_apply(auto width, const std::tuple<Ts...>& rngs, auto offset, auto&& fun, Flag f = {}) {
    using Tup = vir::simdize<std::tuple<std::ranges::range_value_t<Ts>...>, width>;
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return fun(std::tuple_element_t<Is, Tup>(std::ranges::data(std::get<Is>(rngs)) + offset, f)...); }(std::make_index_sequence<sizeof...(Ts)>());
}

template<std::size_t Index, PortType portType, PortReflectable Self>
[[nodiscard]] constexpr auto& inputPort(Self* self) noexcept {
    using TRequestedPortType = typename traits::block::input_port_descriptors<Self, portType>::template at<Index>;
    return TRequestedPortType::getPortObject(*self);
}

template<std::size_t Index, PortType portType, PortReflectable Self>
[[nodiscard]] constexpr auto& outputPort(Self* self) noexcept {
    using TRequestedPortType = typename traits::block::output_port_descriptors<Self, portType>::template at<Index>;
    return TRequestedPortType::getPortObject(*self);
}

template<fixed_string Name, PortReflectable Self>
[[nodiscard]] constexpr auto& inputPort(Self* self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::block::all_input_ports<Self>>();
    if constexpr (Index == meta::default_message_port_index) {
        return self->msgIn;
    }
    return inputPort<Index, PortType::ANY, Self>(self);
}

template<fixed_string Name, PortReflectable Self>
[[nodiscard]] constexpr auto& outputPort(Self* self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::block::all_output_ports<Self>>();
    if constexpr (Index == meta::default_message_port_index) {
        return self->msgOut;
    }
    return outputPort<Index, PortType::ANY, Self>(self);
}

template<PortType portType, PortReflectable Self>
[[nodiscard]] constexpr auto inputPorts(Self* self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::tie(inputPort<Idx, portType>(self)...); }(traits::block::input_port_descriptors<Self, portType>::index_sequence);
}

template<PortType portType, PortReflectable Self>
[[nodiscard]] constexpr auto outputPorts(Self* self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::tie(outputPort<Idx, portType>(self)...); }(traits::block::output_port_descriptors<Self, portType>::index_sequence);
}

namespace detail {
template<std::ranges::range Range, typename T = std::ranges::range_value_t<Range>>
[[nodiscard]] constexpr std::optional<T> min_element(Range&& range) {
    if (auto it = std::ranges::min_element(range); it != std::ranges::end(range)) {
        return *it;
    }
    return std::nullopt;
}

template<auto... MatchPortEnums, std::ranges::range Range, typename T = std::ranges::range_value_t<Range>>
[[nodiscard]] constexpr std::optional<T> min_element_masked(Range&& range, const std::span<const port::BitMask>& portMaskVec) {
    auto filtered = std::ranges::views::zip(range, portMaskVec) | std::views::filter([](const auto& pair) {
        const auto& mask = std::get<1>(pair);
        return port::pattern<MatchPortEnums...>().matches(mask); // actual & pattern.mask == pattern.value
    }) | std::views::transform([](const auto& pair) { return std::get<0>(pair); });

    return min_element(filtered);
}

template<std::ranges::range Range, typename T = std::ranges::range_value_t<Range>>
std::optional<T> max_element(Range&& range) {
    if (auto it = std::ranges::max_element(range); it != std::ranges::end(range)) {
        return *it;
    }
    return std::nullopt;
}

template<auto... MatchPortEnums, std::ranges::range Range, typename T = std::ranges::range_value_t<Range>>
[[nodiscard]] constexpr std::optional<T> max_element_masked(Range&& range, const std::span<const port::BitMask>& portMaskVec) {
    auto zipped   = std::ranges::views::zip(range, portMaskVec);
    auto filtered = zipped | std::views::filter([](const auto& pair) {
        const auto& mask = std::get<1>(pair);
        return port::pattern<MatchPortEnums...>().matches(mask); // actual & pattern.mask == pattern.value
    }) | std::views::transform([](const auto& pair) { return std::get<0>(pair); });

    return max_element(filtered);
}

template<typename T, std::size_t simd_width = stdx::simd_abi::max_fixed_size<T>>
[[nodiscard]] constexpr bool compareSpans(const std::span<T>& a, const std::span<T>& b) noexcept {
    const std::size_t size = std::min(a.size(), b.size());
    if (size == 0UZ) {
        return false;
    }

    // SIMD loop
    std::size_t i = 0UZ;
    for (; i + simd_width <= size; i += simd_width) {
        using TSimd = stdx::fixed_size_simd<T, simd_width>;
        TSimd a_chunk(&a[i], stdx::vector_aligned);
        TSimd b_chunk(&b[i], stdx::vector_aligned);

        if (stdx::any_of(a_chunk >= b_chunk)) { // true: any element of a[i] >= b[i]
            return true;
        }
    }

    // SIMD epilogue
    for (; i < size; ++i) {
        if (a[i] >= b[i]) {
            return true;
        }
    }

    return false;
}

template<auto... MatchPortEnums, std::ranges::input_range RangeA, std::ranges::input_range RangeB, std::ranges::input_range MaskRange>
[[nodiscard]] constexpr bool compareRangesMasked(RangeA&& a, RangeB&& b, MaskRange&& mask) {
    auto zipped = std::ranges::views::zip(a, b, mask) | std::views::filter([](const auto& tup) { return port::pattern<MatchPortEnums...>().matches(std::get<2>(tup)); });
    return std::ranges::any_of(zipped, [](const auto& tup) { return std::get<0UZ>(tup) >= std::get<1UZ>(tup); });
}
} // namespace detail

template<typename Derived, PortDirection portDirection, PortType portType>
class PortCache {
    using AllocatorSize    = gr::allocator::Aligned<std::size_t, gr::meta::kCacheLine>;
    using AllocatorBitMask = gr::allocator::Aligned<port::BitMask, gr::meta::kCacheLine>;

    // reference to derived class containing the ports
    Derived& _self;

    // updated on settings/port-config change
    bool                                         _dirtyConfig = true;
    std::vector<port::BitMask, AllocatorBitMask> _types;
    std::vector<std::size_t, AllocatorSize>      _minSamples;
    std::vector<std::size_t, AllocatorSize>      _maxSamples;

    // updated every work(...) function invokation
    bool                                    _dirtyAvailable = true;
    std::vector<std::size_t, AllocatorSize> _available;

    // aggregated values
    std::size_t _minSyncRequirement = 0UZ;
    std::size_t _maxSyncRequirement = gr::undefined_size;
    std::size_t _maxSyncAvailable   = gr::undefined_size;
    bool        _hasASyncAvailable  = false;

protected:
    template<std::ranges::range Range, typename T = std::ranges::range_value_t<Range>>
    requires(std::is_same_v<T, port::BitMask>)
    constexpr void getPortTypes(const Derived& self, Range& result) const noexcept {
        result.clear();
        auto func = [&result]<gr::PortLike Port>(const Port& port) noexcept {
            port::BitMask mask = port::encodeMask(Port::kDirection, Port::kPortType, Port::kIsSynch, Port::kIsOptional, port.isConnected());
            result.push_back(mask);
        };
        if constexpr (portDirection == PortDirection::INPUT) {
            for_each_port(std::move(func), inputPorts<portType>(&self));
        } else {
            for_each_port(std::move(func), outputPorts<portType>(&self));
        }
    }

    template<std::ranges::range Range, typename Fn>
    constexpr void getPortConstraints(const Derived& self, Range& storage, Fn&& function) {
        if (std::ranges::size(storage) == 0UZ) {
            return;
        }
#if __has_cpp_attribute(assume) && !defined(__clang__)
        [[assume(std::ranges::size(storage) > 0UZ)]]; // non-empty storage guarantee (has been allocated externally)
#endif
        auto&& fn = std::forward<Fn>(function);
        auto   it = storage.begin();
        if constexpr (portDirection == PortDirection::INPUT) {
            for_each_port([&it, &fn](auto& port) { *it++ = std::invoke(fn, port); }, inputPorts<portType>(&self));
        } else {
            for_each_port([&it, &fn](auto& port) { *it++ = std::invoke(fn, port); }, outputPorts<portType>(&self));
        }
        assert(std::distance(std::begin(storage), it) >= 0);
        assert(storage.size() == static_cast<std::size_t>(std::distance(std::begin(storage), it)));
    }

    void updateConfig() {
        _dirtyAvailable = true;
        getPortTypes(_self, _types);
        _available.resize(_types.size(), 0UZ);
        _minSamples.resize(_types.size(), 0UZ);
        _maxSamples.resize(_types.size(), gr::undefined_size);
        getPortConstraints(_self, _minSamples, [](auto& port) {
            if constexpr (std::remove_cvref_t<decltype(port)>::isOptional()) {
                return port.isConnected() ? port.min_samples : 0UZ;
            } else {
                return port.min_samples;
            }
        });
        getPortConstraints(_self, _maxSamples, [](auto& port) { return port.max_samples; });
        _minSyncRequirement = detail::max_element_masked<PortSync::SYNCHRONOUS>(_minSamples, _types).value_or(0UZ);
        _maxSyncRequirement = detail::min_element_masked<PortSync::SYNCHRONOUS>(_maxSamples, _types).value_or(gr::undefined_size);
        _dirtyConfig        = false;
    }

    void updateAvailable() {
        if (_dirtyConfig) {
            updateConfig();
        }
        assert(!_dirtyConfig);
        if (!_dirtyAvailable) {
            return; // already updated
        }
        getPortConstraints(_self, _available, [](auto& port) {
            if constexpr (std::remove_cvref_t<decltype(port)>::isOptional()) {
                return port.isConnected() ? port.available() : gr::undefined_size;
            } else {
                return port.available();
            }
        });
        _maxSyncAvailable  = detail::min_element_masked<PortSync::SYNCHRONOUS>(_available, _types).value_or(gr::undefined_size);
        _hasASyncAvailable = detail::compareRangesMasked<PortSync::ASYNCHRONOUS>(_available, _minSamples, _types);
        _dirtyAvailable    = false;
    }

public:
    PortCache(Derived& self) : _self(self) {}

    void invalidateConfig() noexcept { _dirtyConfig = true; }
    void invalidateStatistic() noexcept { _dirtyAvailable = true; }

    std::span<const port::BitMask> types() {
        if (_dirtyConfig) {
            updateConfig();
        }
        assert(!_dirtyConfig);
        return std::span<const port::BitMask>{_types.data(), _types.size()};
    }

    std::span<const std::size_t> minSamples() {
        if (_dirtyConfig) {
            updateConfig();
        }
        assert(!_dirtyConfig);
        return std::span<const std::size_t>{_minSamples.data(), _minSamples.size()};
    }

    std::span<const std::size_t> maxSamples() {
        if (_dirtyConfig) {
            updateConfig();
        }
        assert(!_dirtyConfig);
        return std::span<const std::size_t>{_maxSamples.data(), _maxSamples.size()};
    }

    std::span<const std::size_t> availableSamples(bool reset = false) {
        if (_dirtyAvailable || reset) {
            updateAvailable();
        }
        assert(!_dirtyAvailable);
        return std::span<const std::size_t>{_available.data(), _available.size()};
    }

    std::size_t minSyncRequirement() {
        if (_dirtyConfig) {
            updateConfig();
        }
        return _minSyncRequirement;
    }
    std::size_t maxSyncRequirement() {
        if (_dirtyConfig) {
            updateConfig();
        }
        return _maxSyncRequirement;
    }
    std::size_t maxSyncAvailable() {
        if (_dirtyAvailable) {
            updateAvailable();
        }
        return _maxSyncAvailable;
    }
    bool hasASyncAvailable() {
        if (_dirtyAvailable) {
            updateAvailable();
        }
        return _hasASyncAvailable;
    }

    std::expected<std::size_t, gr::Error> primePort(std::size_t portIdx, std::size_t nSamples, std::source_location loc = std::source_location::current()) noexcept {
        if constexpr (requires { _self.state(); }) {
            if (_self.state() == lifecycle::State::RUNNING) {
                return std::unexpected(Error(std::format("primePort({}, {}) - block must not be in RUNNING state", portIdx, nSamples), loc));
            }
        }
        if (nSamples == 0UZ) {
            return nSamples; // explicit NOOP
        }

        if (_dirtyConfig) {
            updateConfig();
        }
        if (portIdx >= _types.size()) {
            return std::unexpected(Error(std::format("primePort({}, {}) failed: portIdx out of range [0, {}]", portIdx, nSamples, _types.size()), loc));
        }

        std::expected<std::size_t, gr::Error> result = std::unexpected(Error(std::format("primePort({}, {}) - unexpected failure", portIdx, nSamples), loc));

        std::size_t idx        = 0UZ;
        auto        primerFunc = [&result, &loc, &idx, &portIdx, &nSamples](PortLike auto& port) {
            if (idx++ != portIdx) {
                return;
            }

            if (!port.isConnected()) {
                result = std::unexpected(gr::Error(std::format("primePort({}, {}) - port {} ({}) is not connected", portIdx, nSamples, portIdx, port.name), loc));
                return;
            }

            // can safely assume that port is either INPUT XOR OUTPUT
            auto getAvailable = [&port]() -> std::size_t { return port.available(); };

            std::size_t availableBefore = getAvailable();
            // prime port
            auto publishSamples = [&result, &loc, &portIdx](WriterSpanLike auto& publishSpan, std::size_t nRequested) {
                if (publishSpan.size() < nRequested) {
                    result = std::unexpected(gr::Error(std::format("primePort({}, {}) - failed requested {} and got {} samples", portIdx, nRequested, nRequested, publishSpan.size()), loc));
                    return;
                }
                using T = typename std::remove_reference_t<decltype(port)>::value_type;
                for (std::size_t i = 0UZ; i < nRequested; ++i) {
                    publishSpan[i] = T{};
                }
                publishSpan.publish(nRequested);
            };

            if constexpr (portDirection == PortDirection::INPUT) {
                auto writer = port.buffer().streamBuffer.new_writer();

                WriterSpanLike auto publishSpan = writer.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(nSamples);
                publishSamples(publishSpan, nSamples);
            } else {
                WriterSpanLike auto publishSpan = port.streamWriter().template tryReserve<gr::SpanReleasePolicy::ProcessAll>(nSamples);
                publishSamples(publishSpan, nSamples);
            }
            // N.B. actual publish is done when this context finished/publishSpan is destructure

            // check if samples have been actually published
            std::size_t availableAfter = getAvailable();

            if constexpr (portDirection == PortDirection::INPUT) {
                if (availableAfter - availableBefore != nSamples) {
                    result = std::unexpected(gr::Error(std::format("primePort({}, {}) - failed requested {} and got {} samples", portIdx, nSamples, nSamples, availableAfter - availableBefore), loc));
                    return;
                }
            } else { // N.B. available decreases on output ports
                if (availableBefore - availableAfter != nSamples) {
                    result = std::unexpected(gr::Error(std::format("primePort({}, {}) - failed requested {} and got {} samples", portIdx, nSamples, nSamples, availableBefore - availableAfter), loc));
                    return;
                }
            }

            result = nSamples;
        };

        if constexpr (portDirection == PortDirection::INPUT) {
            for_each_port(primerFunc, inputPorts<portType>(&_self));
        } else {
            for_each_port(primerFunc, outputPorts<portType>(&_self));
        }

        _dirtyAvailable = true;
        return result;
    }

    std::vector<gr::PortMetaInfo> metaInfos(bool reset = true) {
        if (_dirtyConfig || reset) {
            updateConfig();
        }
        std::vector<gr::PortMetaInfo> metaInfo;
        metaInfo.reserve(_types.size());

        if constexpr (portDirection == PortDirection::INPUT) {
            for_each_port([&metaInfo](auto& port) { metaInfo.push_back(port.metaInfo); }, inputPorts<portType>(&_self));
        } else {
            for_each_port([&metaInfo](auto& port) { metaInfo.push_back(port.metaInfo); }, outputPorts<portType>(&_self));
        }

        return metaInfo;
    }
};

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

    std::pair<std::size_t, std::size_t> get() const {
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

inline static const char* kMetaInformation = "MetaInformation"; ///< asynchronous message-based retrieval of the static meta-information (i.e. Annotated<> interfaces, constraints, etc...)
inline static const char* kUiConstraints   = "UiConstraints";   ///< asynchronous message-based retrieval of user-defined UI constraints

inline static const char* kStoreDefaults    = "StoreDefaults";    ///< store present settings as default, for counterpart @see kResetDefaults
inline static const char* kResetDefaults    = "ResetDefaults";    ///< retrieve and reset to default setting, for counterpart @see kStoreDefaults
inline static const char* kActiveContext    = "ActiveContext";    ///< retrieve and set active context
inline static const char* kSettingsCtx      = "SettingsCtx";      ///< retrieve/creates/remove a new stored context
inline static const char* kSettingsContexts = "SettingsContexts"; ///< retrieve/creates/remove a new stored context

} // namespace block::property

/**
 * @brief Non-templated base providing accessor function pointers for the 12 standard propertyCallback implementations.
 *
 * By compiling these callbacks once (in BlockBase.cpp) instead of per Block<T> instantiation,
 * this eliminates ~147 KiB of duplicated .text across 14 block types. The callbacks operate
 * exclusively through stored function pointers to SettingsBase& and non-templated data.
 *
 * Uses function pointers (not virtual functions) to avoid introducing a vtable, which would
 * break aggregate initialization of derived block types.
 */
struct BlockBase {
    using PropertyCallback = std::optional<Message> (BlockBase::*)(std::string_view, Message);

    // Pointer to the actual Block<Derived> object. Required because Block<Derived> uses multiple
    // inheritance (StateMachine + BlockBase), so BlockBase's `this` differs from the Block* address.
    void* _blockSelf = nullptr;

    // Non-virtual accessor function pointers, set by Block<Derived> constructor (cold-path only)
    SettingsBase& (*_cbSettings)(void*)                                     = nullptr;
    lifecycle::State (*_cbState)(const void*)                               = nullptr;
    std::expected<void, Error> (*_cbChangeStateTo)(void*, lifecycle::State) = nullptr;
    std::string_view (*_cbUniqueName)(const void*)                          = nullptr;
    std::string_view (*_cbName)(const void*)                                = nullptr;
    property_map& (*_cbMetaInformation)(void*)                              = nullptr;
    property_map& (*_cbUiConstraints)(void*)                                = nullptr;

    // Hook for GraphWrapper to handle subgraph export port messages on any block type
    using SubgraphExportHandler                  = std::optional<Message> (*)(void* context, Message);
    SubgraphExportHandler _subgraphExportHandler = nullptr;
    void*                 _subgraphExportContext = nullptr;

    std::map<std::string, PropertyCallback>      propertyCallbacks;
    std::map<std::string, std::set<std::string>> propertySubscriptions;

    // accessor helpers (delegate to function pointers, using _blockSelf for correct Block* address)
    SettingsBase&              cbSettings() { return _cbSettings(_blockSelf); }
    lifecycle::State           cbState() const { return _cbState(_blockSelf); }
    std::expected<void, Error> cbChangeStateTo(lifecycle::State s) { return _cbChangeStateTo(_blockSelf, s); }
    std::string_view           cbUniqueName() const { return _cbUniqueName(_blockSelf); }
    std::string_view           cbName() const { return _cbName(_blockSelf); }
    property_map&              cbMetaInformation() { return _cbMetaInformation(_blockSelf); }
    property_map&              cbUiConstraints() { return _cbUiConstraints(_blockSelf); }

    // 12 callback implementations (compiled once, not per block type)
    std::optional<Message> propertyCallbackHeartbeat(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackEcho(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackLifecycleState(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackSettings(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackStagedSettings(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackStoreDefaults(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackResetDefaults(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackActiveContext(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackSettingsCtx(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackSettingsContexts(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackMetaInformation(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackUiConstraints(std::string_view propertyName, Message message);
    std::optional<Message> propertyCallbackSubgraphExport(std::string_view propertyName, Message message);
};

namespace block {
enum class Category {
    All,                   ///< all Blocks
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
 *   GR_MAKE_REFLECTABLE(UserDefinedBlock, in, out);
 *   // implement one of the possible processOne or processBulk functions
 * };
 * @endcode
 *
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
 * - `kActiveContext`: Returns current active context and allows to set a new one
 * - `kSettingsCtx`: Manages Settings Contexts Add/Remove/Get
 * - `kSettingsContexts`: Returns all Contextxs
 * - `kMetaInformation`: returns static meta-information (i.e. Annotated<> interfaces, constraints, etc...) (N.B. does not affect Block operation)
 * - `kUiConstraints`: returns user-defined UI constraints (N.B. does not affect Block operation)
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
 *           default: throw gr::exception(std::format("unsupported command {} for property {}", message.cmd, propertyName));
 *         }
 *       return std::nullopt; // no reply needed for Set, Subscribe, Unsubscribe
 *     }
 *
 *     void start() override {
 *         propertyCallbacks.emplace(kMyCustomProperty, std::mem_fn(&MyBlock::propertyCallbackMyCustom));
 *     }
 * };
 * @endcode
 *
 * @tparam Derived the user-defined block CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 * @tparam Arguments NTTP list containing the compile-time defined port instances, setting structs, or other constraints.
 */
template<typename Derived, typename... Arguments>
class Block : public lifecycle::StateMachine<Derived>, public BlockBase {
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

    constexpr static bool blockingIO             = std::disjunction_v<std::is_same<BlockingIO<true>, Arguments>..., std::is_same<BlockingIO<false>, Arguments>...>;
    constexpr static bool noDefaultTagForwarding = std::disjunction_v<std::is_same<NoDefaultTagForwarding, Arguments>...>;
    constexpr static bool backwardTagForwarding  = std::disjunction_v<std::is_same<BackwardTagForwarding, Arguments>...>;

    constexpr static block::Category blockCategory = block::Category::NormalBlock;

    template<typename T>
    auto& getArgument() {
        return std::get<T>(*this);
    }

    template<typename T>
    const auto& getArgument() const {
        return std::get<T>(*this);
    }

    alignas(hardware_destructive_interference_size) std::atomic<std::size_t> ioRequestedWork{std::numeric_limits<std::size_t>::max()};
    alignas(hardware_destructive_interference_size) work::Counter ioWorkDone{};
    alignas(hardware_destructive_interference_size) std::atomic<work::Status> ioLastWorkStatus{work::Status::OK};
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::Sequence> progress = std::make_shared<gr::Sequence>();
    alignas(hardware_destructive_interference_size) std::atomic<bool> ioThreadRunning{false};

    using ResamplingValue = std::conditional_t<ResamplingControl::kIsConst, const gr::Size_t, gr::Size_t>;
    using ResamplingLimit = Limits<1UL, std::numeric_limits<ResamplingValue>::max()>;
    using ResamplingDoc   = Doc<"For each `input_chunk_size` input samples, `output_chunk_size` output samples are published (in>out: Decimate, in<out: Interpolate, in==out: No change)">;

    A<ResamplingValue, "input_chunk_size", ResamplingDoc, ResamplingLimit>  input_chunk_size                                                     = ResamplingControl::kInputChunkSize;
    A<ResamplingValue, "output_chunk_size", ResamplingDoc, ResamplingLimit> output_chunk_size                                                    = ResamplingControl::kOutputChunkSize;
    using StrideValue                                                                                                                            = std::conditional_t<StrideControl::kIsConst, const gr::Size_t, gr::Size_t>;
    A<StrideValue, "stride", Doc<"samples between data processing. <N for overlap, >N for skip, =0 for back-to-back.">>       stride             = StrideControl::kStride;
    A<bool, "disconnect on done", Doc<"If no downstream blocks, declare itself 'DONE' and disconnect from upstream blocks.">> disconnect_on_done = true;
    A<std::string, "compute domain", Doc<"compute domain/IO thread pool name">>                                               compute_domain     = gr::thread_pool::kDefaultIoPoolId;

    gr::Size_t strideCounter = 0UL; // leftover stride from previous calls

    gr::meta::immutable<std::size_t> unique_id   = _uniqueIdCounter++;
    gr::meta::immutable<std::string> unique_name = std::format("{}#{}", gr::meta::type_name<Derived>(), unique_id);

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
            info.insert_or_assign("Category", std::string(magic_enum::enum_name(DrawableControl::kCategory)));
            info.insert_or_assign("Toolkit", std::string(DrawableControl::kToolkit));

            ret.insert_or_assign("Drawable", info);
        }
        return ret;
    }

    A<property_map, "ui-constraints", Doc<"store non-graph-processing information like UI block position etc.">>         ui_constraints;
    A<property_map, "meta-information", Doc<"store static non-graph-processing information like Annotated<> info etc.">> meta_information = initMetaInfo();

    GR_MAKE_REFLECTABLE(Block, input_chunk_size, output_chunk_size, stride, disconnect_on_done, compute_domain, unique_name, name, ui_constraints);

    // TODO: C++26 make sure these are not reflected
    // We support ports that are template parameters or reflected member variables,
    // so these are handled in a special way
    MsgPortInBuiltin  msgIn;
    MsgPortOutBuiltin msgOut;

    // PropertyCallback, propertyCallbacks and propertySubscriptions are inherited from BlockBase

    PortCache<Derived, PortDirection::INPUT, PortType::STREAM>  inputStreamCache;
    PortCache<Derived, PortDirection::OUTPUT, PortType::STREAM> outputStreamCache;

protected:
    Tag _mergedInputTag{};

    bool             _outputTagsChanged = false; // It is used to indicate that processOne published a Tag and want prematurely break a loop. Should be set to "true" in block implementation processOne().
    std::vector<Tag> _outputTags{};              // This std::vector is used to cache published Tags when block implements processOne method. The tags are then copied to output spans. Note: that for he processOne each tag is published for all output ports

    // intermediate non-real-time<->real-time setting states
    CtxSettings<Derived> _settings;

    [[nodiscard]] constexpr auto&       self() noexcept { return *static_cast<Derived*>(this); }
    [[nodiscard]] constexpr const auto& self() const noexcept { return *static_cast<const Derived*>(this); }

    // BlockBase accessor function pointer initializers (cold-path only, used by propertyCallbacks)
    static SettingsBase&              cbSettingsImpl(void* self) { return static_cast<Block*>(self)->_settings; }
    static lifecycle::State           cbStateImpl(const void* self) { return static_cast<const Block*>(self)->state(); }
    static std::expected<void, Error> cbChangeStateToImpl(void* self, lifecycle::State s) { return static_cast<Block*>(self)->changeStateTo(s); }
    static std::string_view           cbUniqueNameImpl(const void* self) { return static_cast<const Block*>(self)->unique_name; }
    static std::string_view           cbNameImpl(const void* self) { return static_cast<const Block*>(self)->name; }
    static property_map&              cbMetaInformationImpl(void* self) { return static_cast<Block*>(self)->meta_information.value; }
    static property_map&              cbUiConstraintsImpl(void* self) { return static_cast<Block*>(self)->ui_constraints.value; }

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
    Block(std::initializer_list<std::pair<const std::pmr::string, pmt::Value>> initParameter) noexcept(false) : Block(property_map(initParameter)) {}
    Block(property_map initParameters) noexcept(false)                                                     // N.B. throws in case of on contract violations
        : lifecycle::StateMachine<Derived>(),                                                              //
          inputStreamCache(static_cast<Derived&>(*this)), outputStreamCache(static_cast<Derived&>(*this)), //
          _settings(CtxSettings<Derived>(*static_cast<Derived*>(this))) {                                  // N.B. safe delegated use of this (i.e. not used during construction)

        // store the actual Block* address (differs from BlockBase's this due to multiple inheritance)
        _blockSelf = static_cast<void*>(this);

        // initialize inherited BlockBase accessor function pointers
        _cbSettings        = &Block::cbSettingsImpl;
        _cbState           = &Block::cbStateImpl;
        _cbChangeStateTo   = &Block::cbChangeStateToImpl;
        _cbUniqueName      = &Block::cbUniqueNameImpl;
        _cbName            = &Block::cbNameImpl;
        _cbMetaInformation = &Block::cbMetaInformationImpl;
        _cbUiConstraints   = &Block::cbUiConstraintsImpl;

        // initialize inherited BlockBase::propertyCallbacks
        propertyCallbacks = {
            {block::property::kHeartbeat, &BlockBase::propertyCallbackHeartbeat},
            {block::property::kEcho, &BlockBase::propertyCallbackEcho},
            {block::property::kLifeCycleState, &BlockBase::propertyCallbackLifecycleState},
            {block::property::kSetting, &BlockBase::propertyCallbackSettings},
            {block::property::kStagedSetting, &BlockBase::propertyCallbackStagedSettings},
            {block::property::kStoreDefaults, &BlockBase::propertyCallbackStoreDefaults},
            {block::property::kResetDefaults, &BlockBase::propertyCallbackResetDefaults},
            {block::property::kActiveContext, &BlockBase::propertyCallbackActiveContext},
            {block::property::kSettingsCtx, &BlockBase::propertyCallbackSettingsCtx},
            {block::property::kSettingsContexts, &BlockBase::propertyCallbackSettingsContexts},
            {block::property::kMetaInformation, &BlockBase::propertyCallbackMetaInformation},
            {block::property::kUiConstraints, &BlockBase::propertyCallbackUiConstraints},
        };

        // check Block<T> contracts
        checkBlockContracts<decltype(*static_cast<Derived*>(this))>();

        if constexpr (refl::reflectable<Derived>) {
            settings().setInitBlockParameters(initParameters);
        }
    }

    Block(Block&& other) noexcept
        : lifecycle::StateMachine<Derived>(std::move(other)),                                                                                                    //
          BlockBase(std::move(other)),                                                                                                                           //
          input_chunk_size(std::move(other.input_chunk_size)), output_chunk_size(std::move(other.output_chunk_size)),                                            //
          stride(std::move(other.stride)),                                                                                                                       //
          disconnect_on_done(other.disconnect_on_done),                                                                                                          //
          compute_domain(std::move(other.compute_domain)),                                                                                                       //
          strideCounter(other.strideCounter),                                                                                                                    //
          unique_id(std::move(other.unique_id)), unique_name(std::move(other.unique_name)), name(std::move(other.name)),                                         //
          ui_constraints(std::move(other.ui_constraints)), meta_information(std::move(other.meta_information)),                                                  //
          msgIn(std::move(other.msgIn)), msgOut(std::move(other.msgOut)),                                                                                        //
          inputStreamCache(static_cast<Derived&>(*this)), outputStreamCache(static_cast<Derived&>(*this)),                                                       //
          _mergedInputTag(std::move(other._mergedInputTag)), _outputTagsChanged(std::move(other._outputTagsChanged)), _outputTags(std::move(other._outputTags)), //
          _settings(CtxSettings<Derived>(*static_cast<Derived*>(this), std::move(other._settings)))                                                              //
    {
        _blockSelf       = static_cast<void*>(this);
        other._blockSelf = nullptr;
    }

    Block& operator=(Block&& other) noexcept = delete;

    ~Block() { // NOSONAR -- need to request the (potentially) running ioThread to stop
        if (lifecycle::isActive(this->state())) {
            // Only happens in artificial cases likes qa_Block test. In practice blocks stay in zombie list if active
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

    void init(std::shared_ptr<gr::Sequence> progress_, std::string_view computeDomain = gr::thread_pool::kDefaultIoPoolId) {
        progress       = std::move(progress_);
        compute_domain = computeDomain;

        // Set names of port member variables
        // TODO: Refactor the library not to assign names to ports. The
        // block and the graph are the only things that need the port name
        auto setPortName = [&](std::size_t, auto* t) {
            using Description = std::remove_pointer_t<decltype(t)>;
            auto& port        = Description::getPortObject(self());
            if constexpr (Description::kIsDynamicCollection || Description::kIsStaticCollection) {
                for (auto& actualPort : port) {
                    actualPort.name = Description::Name;
                }
            } else {
                port.name = Description::Name;
            }
        };
        traits::block::all_input_ports<Derived>::for_each(setPortName);
        traits::block::all_output_ports<Derived>::for_each(setPortName);

        settings().init();

        // important: these tags need to be queued because at this stage the block is not yet connected to other downstream blocks
        invokeUserProvidedFunction("init() - applyStagedParameters", [this] noexcept(false) {
            if (const auto applyResult = settings().applyStagedParameters(); !applyResult.forwardParameters.empty()) {
                if constexpr (!noDefaultTagForwarding) {
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

    [[nodiscard]] constexpr bool isBlocking() const noexcept { return blockingIO; }

    [[nodiscard]] constexpr bool inputTagsPresent() const noexcept { return !_mergedInputTag.map.empty(); };

    [[nodiscard]] constexpr const Tag& mergedInputTag() const noexcept { return _mergedInputTag; }

    [[nodiscard]] constexpr const SettingsBase& settings() const noexcept { return _settings; }

    [[nodiscard]] constexpr SettingsBase& settings() noexcept { return _settings; }

    void setSettings(const CtxSettings<Derived>& settings) { _settings.assignFrom(settings); }
    void setSettings(CtxSettings<Derived>&& settings) { _settings.assignFrom(std::move(settings)); }

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
                emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", std::format("Block is not defined as `Resampling<>`, but input_chunk_size = {}, output_chunk_size = {}, they both must equal to 1.", input_chunk_size, output_chunk_size));
                requestStop();
                return;
            }
        }
        if constexpr (StrideControl::kEnabled) {
            static_assert(!kIsSourceBlock, "Stride is not available for source blocks. Remove 'Stride<>' from the block definition.");
        } else {
            if (stride != 0ULL) {
                emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", std::format("Block is not defined as `Stride<>`, but stride = {}, it must equal to 0.", stride));
                requestStop();
                return;
            }
        }
        // TODO: remove these obsolete lines
        // const auto [minSyncIn, maxSyncIn, _, _1]    = getPortLimits(inputPorts<PortType::STREAM>(&self()));
        // const auto [minSyncOut, maxSyncOut, _2, _3] = getPortLimits(outputPorts<PortType::STREAM>(&self()));
        inputStreamCache.invalidateConfig();
        outputStreamCache.invalidateConfig();
        const std::size_t minSyncIn  = inputStreamCache.minSyncRequirement();
        const std::size_t maxSyncIn  = inputStreamCache.maxSyncRequirement();
        const std::size_t minSyncOut = outputStreamCache.minSyncRequirement();
        const std::size_t maxSyncOut = outputStreamCache.maxSyncRequirement();
        inputStreamCache.invalidateConfig();
        outputStreamCache.invalidateConfig();
        if (minSyncIn > maxSyncIn) {
            emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", std::format("Min samples for input ports ({}) is larger then max samples for input ports ({})", minSyncIn, maxSyncIn));
            requestStop();
            return;
        }
        if (minSyncOut > maxSyncOut) {
            emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", std::format("Min samples for output ports ({}) is larger then max samples for output ports ({})", minSyncOut, maxSyncOut));
            requestStop();
            return;
        }
        if (input_chunk_size > maxSyncIn) {
            emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", std::format("resampling input_chunk_size ({}) is larger then max samples for input ports ({})", input_chunk_size, maxSyncIn));
            requestStop();
            return;
        }
        if (output_chunk_size > maxSyncOut) {
            emitErrorMessage("Block::checkParametersAndThrowIfNeeded:", std::format("resampling output_chunk_size ({}) is larger then max samples for output ports ({})", output_chunk_size, maxSyncOut));
            requestStop();
            return;
        }
    }

    void publishSamples(std::size_t nSamples, auto& outputSpanTuple) noexcept {
        if constexpr (traits::block::stream_output_ports<Derived>::size > 0) {
            for_each_writer_span(
                [nSamples]<typename Out>(Out& out) {
                    if constexpr (Out::isMultiProducerStrategy()) {
                        if (!out.isFullyPublished()) {
                            std::abort();
                        }
                    }
                    if (!out.isPublishRequested()) {
                        using enum gr::SpanReleasePolicy;
                        if constexpr (Out::spanReleasePolicy() == Terminate) {
                            std::abort();
                        } else if constexpr (Out::spanReleasePolicy() == ProcessAll) {
                            out.publish(nSamples);
                        } else if constexpr (Out::spanReleasePolicy() == ProcessNone) {
                            out.publish(0U);
                        }
                    }
                },
                outputSpanTuple);
        }
    }

    bool consumeReaders(std::size_t nSamples, auto& consumableSpanTuple) {
        bool success = true;
        if constexpr (traits::block::stream_input_ports<Derived>::size > 0) {
            for_each_reader_span(
                [nSamples, &success]<typename In>(In& in) {
                    if (!in.isConsumeRequested()) {
                        using enum gr::SpanReleasePolicy;
                        if constexpr (In::spanReleasePolicy() == Terminate) {
                            std::abort();
                        } else if constexpr (In::spanReleasePolicy() == ProcessAll) {
                            // for unconnected Optional ports, consume 0 instead of nSamples
                            const auto toConsume = in.isConnected ? nSamples : 0UZ;
                            success              = success && in.consume(toConsume);
                        } else if constexpr (In::spanReleasePolicy() == ProcessNone) {
                            success = success && in.consume(0U);
                        }
                    }
                },
                consumableSpanTuple);
        }
        return success;
    }

    template<typename... Ts>
    constexpr auto invoke_processOne(Ts&&... inputs) {
        if constexpr (traits::block::stream_output_ports<Derived>::size == 0) {
            self().processOne(std::forward<Ts>(inputs)...);
            return std::tuple{};
        } else if constexpr (traits::block::stream_output_ports<Derived>::size == 1) {
            return std::tuple{self().processOne(std::forward<Ts>(inputs)...)};
        } else {
            return self().processOne(std::forward<Ts>(inputs)...);
        }
    }

    template<typename... Ts>
    constexpr auto invoke_processOne_simd(auto width, Ts&&... input_simds) {
        if constexpr (sizeof...(Ts) == 0) {
            if constexpr (traits::block::stream_output_ports<Derived>::size == 0) {
                self().processOne_simd(width);
                return std::tuple{};
            } else if constexpr (traits::block::stream_output_ports<Derived>::size == 1) {
                return std::tuple{self().processOne_simd(width)};
            } else {
                return self().processOne_simd(width);
            }
        } else {
            return invoke_processOne(std::forward<Ts>(input_simds)...);
        }
    }

    constexpr void publishMergedInputTag(auto& outputSpanTuple) noexcept {
        if constexpr (!noDefaultTagForwarding) {
            if (inputTagsPresent()) {
                const auto&  autoForwardKeys = settings().autoForwardParameters();
                property_map onlyAutoForwardMap;
                std::ranges::copy_if(_mergedInputTag.map, std::inserter(onlyAutoForwardMap, onlyAutoForwardMap.end()), [&autoForwardKeys](const auto& kv) { return autoForwardKeys.contains(convert_string_domain(kv.first)); });
                for_each_writer_span([&onlyAutoForwardMap](auto& outSpan) { outSpan.publishTag(onlyAutoForwardMap, 0); }, outputSpanTuple);
            }
        }
    }

    constexpr void publishCachedOutputTags(auto& outputSpanTuple) noexcept {
        if (_outputTags.empty()) {
            return;
        }
        for (const auto& tag : _outputTags) {
            for_each_writer_span([&tag](auto& outSpan) { outSpan.publishTag(tag.map, tag.index); }, outputSpanTuple);
        }
        _outputTags.clear();
    }

    /**
     * Merge tags from all sync ports into one merged tag, apply auto-update parameters
     */
    void updateMergedInputTagAndApplySettings(auto& inputSpans, std::size_t untilLocalIndex = 1UZ) noexcept {
        std::size_t untilLocalIndexAdjusted = untilLocalIndex;
        if constexpr (!backwardTagForwarding) {
            untilLocalIndexAdjusted = 1UZ;
        }
        const auto isIndexEqual       = [](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; };
        const auto isIndexAndMapEqual = [](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first && lhs.second.get() == rhs.second.get(); };

        // TODO: we still fill _mergedInputTag, but this will be removed in the one of the next PR
        for_each_reader_span(
            [this, untilLocalIndexAdjusted, isIndexEqual, isIndexAndMapEqual](auto& in) {
                if (in.isSync && in.isConnected) {
                    auto inTags = in.tags(untilLocalIndexAdjusted) | PairDeduplicateView(isIndexEqual, isIndexAndMapEqual);
                    for (const auto& [_, tagMap] : inTags) {
                        for (const auto& [key, value] : tagMap.get()) {
                            _mergedInputTag.map.insert_or_assign(key, value);
                        }
                    }
                }
            },
            inputSpans);

        // non-duplicated, ordered by index, the last Tag (wih max index) wins
        using InputSpanT = typename gr::PortIn<float>::InputSpan<SpanReleasePolicy::ProcessNone>;
        using ViewT      = decltype(std::declval<InputSpanT>().tags(0UZ));
        std::vector<ViewT> allPairViews;
        allPairViews.reserve(8);
        for_each_reader_span(
            [&allPairViews, untilLocalIndexAdjusted](auto& in) {
                if (in.isSync && in.isConnected) {
                    auto inTags = in.tags(untilLocalIndexAdjusted);
                    static_assert(std::ranges::input_range<decltype(inTags)>);
                    static_assert(std::ranges::forward_range<decltype(inTags)>);
                    allPairViews.push_back(std::move(inTags));
                }
            },
            inputSpans);

        auto mergedPairsLazy        = allPairViews | Merge{[](const PairRelIndexMapRef& lhs, const PairRelIndexMapRef& rhs) { return lhs.first < rhs.first; }};
        auto nonDuplicatedInputTags = mergedPairsLazy | PairDeduplicateView(isIndexEqual, isIndexAndMapEqual);

        if (inputTagsPresent()) {
            for (const auto& tag : nonDuplicatedInputTags) {
                // TODO: autoUpdate does not really need Tag, it should be changed to accept property_map
                settings().autoUpdate(Tag{tag.first < 0 ? 0UZ : static_cast<std::size_t>(tag.first), tag.second.get()});
            }
        }

        // update PortMetaInfo
        for_each_port_and_reader_span(
            [this, &untilLocalIndexAdjusted, isIndexEqual, isIndexAndMapEqual]<PortLike TPort, ReaderSpanLike TReaderSpan>(TPort& port, TReaderSpan& span) { //
                auto inTags = span.tags(untilLocalIndexAdjusted) | PairDeduplicateView(isIndexEqual, isIndexAndMapEqual);
                for (const auto& [_, tagMap] : inTags) {
                    emitErrorMessageIfAny("Block::updateMergedInputTagAndApplySettings", port.metaInfo.update(tagMap.get()));
                }
            },
            inputPorts<PortType::STREAM>(&self()), inputSpans);
    }

    void applyChangedSettings() {
        if (!settings().changed()) {
            return;
        }
        invokeUserProvidedFunction("applyChangedSettings()", [this] noexcept(false) {
            auto applyResult = settings().applyStagedParameters();
            checkBlockParameterConsistency();

            auto& forwardParametersMap = applyResult.forwardParameters;
            if (!forwardParametersMap.empty()) {
                for (auto& [key, value] : forwardParametersMap) {
                    _mergedInputTag.insert_or_assign(convert_string_domain(key), value);
                }
            }

            settings().setChanged(false);

            auto& appliedParametersMap = applyResult.appliedParameters;
            if (!appliedParametersMap.empty()) {
                notifyListeners(block::property::kStagedSetting, appliedParametersMap);
            }
            notifyListeners(block::property::kSetting, settings().get());
        });

        // update input/output port caches
        inputStreamCache.invalidateConfig();
        outputStreamCache.invalidateConfig();
    }

    constexpr static auto prepareStreams(auto ports, std::size_t nSyncSamples) {
        return meta::tuple_transform(
            [nSyncSamples]<typename PortOrCollection>(PortOrCollection& outputPortOrCollection) noexcept {
                auto processSinglePort = [&nSyncSamples]<typename Port>(Port&& port) {
                    using enum gr::SpanReleasePolicy;
                    if constexpr (std::remove_cvref_t<Port>::kIsInput) {
                        if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                            if constexpr (std::remove_cvref_t<Port>::isOptional()) { // handle unconnected Optional ports: request 0 samples (like async)
                                return std::forward<Port>(port).template get<ProcessAll, !backwardTagForwarding>(port.isConnected() ? nSyncSamples : 0UZ);
                            } else {
                                return std::forward<Port>(port).template get<ProcessAll, !backwardTagForwarding>(nSyncSamples);
                            }
                        } else {
                            return std::forward<Port>(port).template get<ProcessNone, !backwardTagForwarding>(port.streamReader().available());
                        }
                    } else if constexpr (std::remove_cvref_t<Port>::kIsOutput) {
                        if constexpr (std::remove_cvref_t<Port>::kIsSynch) {
                            return std::forward<Port>(port).template tryReserve<ProcessAll>(nSyncSamples);
                        } else {
                            return std::forward<Port>(port).template tryReserve<ProcessNone>(port.streamWriter().available());
                        }
                    }
                };
                if constexpr (traits::port::is_port_v<PortOrCollection>) {
                    return processSinglePort(outputPortOrCollection);
                } else {
                    using value_span = decltype(processSinglePort(std::declval<typename PortOrCollection::value_type>()));
                    std::vector<value_span> result{};
                    std::transform(outputPortOrCollection.begin(), outputPortOrCollection.end(), std::back_inserter(result), processSinglePort);
                    return result;
                }
            },
            ports);
    }

    inline constexpr void publishTag(property_map&& tag_data, std::size_t tagOffset = 0UZ) noexcept { processPublishTag(std::move(tag_data), tagOffset); }

    inline constexpr void publishTag(const property_map& tag_data, std::size_t tagOffset = 0UZ) noexcept { processPublishTag(tag_data, tagOffset); }

    template<PropertyMapType PropertyMap>
    inline constexpr void processPublishTag(PropertyMap&& tagData, std::size_t tagOffset) noexcept {
        if (_outputTags.empty()) {
            _outputTags.emplace_back(Tag(tagOffset, std::forward<PropertyMap>(tagData)));
        } else {
            auto& lastTag = _outputTags.back();
#ifndef NDEBUG
            if (lastTag.index > tagOffset) { // check the order of published Tags.index
                std::println(stderr, "{}::processPublishTag() - Tag indices are not in the correct order, lastTag.index:{}, index:{}", this->name, lastTag.index, tagOffset);
                // std::abort();
            }
#endif
            if (lastTag.index == tagOffset) { // -> merge tags with the same index
                auto& lastTagMap = lastTag.map;
                for (auto&& [key, value] : tagData) {
                    lastTagMap.insert_or_assign(std::forward<decltype(key)>(key), std::forward<decltype(value)>(value));
                }
            } else {
                _outputTags.emplace_back(Tag(tagOffset, std::forward<PropertyMap>(tagData)));
            }
        }
    }

    inline constexpr void publishEoS() noexcept {
        const property_map tag_data{{static_cast<std::pmr::string>(gr::tag::END_OF_STREAM), true}};
        for_each_port([&tag_data](PortLike auto& outPort) { outPort.publishTag(tag_data, static_cast<std::size_t>(outPort.streamWriter().nRequestedSamplesToPublish())); }, outputPorts<PortType::STREAM>(&self()));
    }

    inline constexpr void publishEoS(auto& outputSpanTuple) noexcept {
        const property_map& tagData{{gr::tag::END_OF_STREAM, true}};
        for_each_writer_span([&tagData](auto& outSpan) { outSpan.publishTag(tagData, static_cast<std::size_t>(outSpan.nRequestedSamplesToPublish())); }, outputSpanTuple);
    }

    constexpr void requestStop() noexcept { emitErrorMessageIfAny("requestStop()", this->changeStateTo(lifecycle::State::REQUESTED_STOP)); }

    constexpr void processScheduledMessages() {
        using namespace std::chrono;
        const std::uint64_t nanoseconds_count = static_cast<uint64_t>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count());
        notifyListeners(block::property::kHeartbeat, pmt::Value::Map{{"heartbeat", nanoseconds_count}});

        auto processPort = [this]<PortLike TPort>(TPort& inPort) {
            const auto available = inPort.streamReader().available();
            if (available == 0UZ) {
                return;
            }
            ReaderSpanLike auto inSpan = inPort.streamReader().get(available);
            if constexpr (traits::block::can_processMessagesForPortReaderSpan<Derived, TPort>) {
                self().processMessages(inPort, inSpan);
                // User could have consumed the span in the custom processMessages handler
                std::ignore = inSpan.tryConsume(inSpan.size());
            } else if constexpr (traits::block::can_processMessagesForPortStdSpan<Derived, TPort>) {
                self().processMessages(inPort, static_cast<std::span<const Message>>(inSpan));
                if (auto consumed = inSpan.tryConsume(inSpan.size()); !consumed) {
                    throw gr::exception(std::format("Block {}::processScheduledMessages() could not consume the messages from the message port", unique_name));
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
    /***
     * Aggregate the amount of samples that can be consumed/produced from a range of ports.
     * @param ports a typelist of input or output ports
     * @return an anonymous struct representing the amount of available data on the ports
     */
    template<typename TPorts>
    auto getPortLimits(TPorts&& ports) {
        struct {
            std::size_t minSync      = 0UL;                                     // the minimum amount of samples that the block needs for processing on the sync ports
            std::size_t maxSync      = std::numeric_limits<std::size_t>::max(); // the maximum amount of that can be consumed on all sync ports
            std::size_t maxAvailable = std::numeric_limits<std::size_t>::max(); // the maximum amount of that are available on all sync ports
            bool        hasAsync     = false;                                   // true if there is at least one async input/output that has available samples/remaining capacity
        } result;
        auto adjustForInputPort = [&result]<PortLike Port>(Port& port) {
            const std::size_t available = port.available();
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
        for_each_port([&adjustForInputPort](PortLike auto& port) { adjustForInputPort(port); }, std::forward<TPorts>(ports));
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
                    result.nextTag                    = std::min(result.nextTag, nSamplesUntilNextTag(port, 1).value_or(std::numeric_limits<std::size_t>::max()));
                    result.nextEosTag                 = std::min(result.nextEosTag, samples_to_eos_tag(port).value_or(std::numeric_limits<std::size_t>::max()));
                    const ReaderSpanLike auto tagData = port.tagReader().get();
                    result.hasTag                     = result.hasTag || (!tagData.empty() && tagData[0].index == port.streamReader().position() && !tagData[0].map.empty());
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

    auto computeResampling(std::size_t minSyncIn, std::size_t maxSyncIn, std::size_t minSyncOut, std::size_t maxSyncOut, std::size_t requestedWork = std::numeric_limits<std::size_t>::max()) {
        if (requestedWork == 0UZ) {
            requestedWork = std::numeric_limits<std::size_t>::max();
        }
        struct ResamplingResult {
            std::size_t  resampledIn;
            std::size_t  resampledOut;
            work::Status status = work::Status::OK;
        };

        if constexpr (!ResamplingControl::kEnabled) { // no resampling
            const std::size_t maxSync = std::min(maxSyncIn, maxSyncOut);
            if (maxSync < minSyncIn) {
                return ResamplingResult{.resampledIn = 0UZ, .resampledOut = 0UZ, .status = work::Status::INSUFFICIENT_INPUT_ITEMS};
            }
            if (maxSync < minSyncOut) {
                return ResamplingResult{.resampledIn = 0UZ, .resampledOut = 0UZ, .status = work::Status::INSUFFICIENT_OUTPUT_ITEMS};
            }
            const auto minSync   = std::max(minSyncIn, minSyncOut);
            const auto resampled = std::clamp(requestedWork, minSync, maxSync);
            return ResamplingResult{.resampledIn = resampled, .resampledOut = resampled};
        }
        if (input_chunk_size == 1UL && output_chunk_size == 1UL) { // no resampling
            const std::size_t maxSync = std::min(maxSyncIn, maxSyncOut);
            if (maxSync < minSyncIn) {
                return ResamplingResult{.resampledIn = 0UZ, .resampledOut = 0UZ, .status = work::Status::INSUFFICIENT_INPUT_ITEMS};
            }
            if (maxSync < minSyncOut) {
                return ResamplingResult{.resampledIn = 0UZ, .resampledOut = 0UZ, .status = work::Status::INSUFFICIENT_OUTPUT_ITEMS};
            }
            const auto minSync   = std::max(minSyncIn, minSyncOut);
            const auto resampled = std::clamp(requestedWork, minSync, maxSync);
            return ResamplingResult{.resampledIn = resampled, .resampledOut = resampled};
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
            if (requestedWork < nResamplingChunks * input_chunk_size) { // if we still can apply requestedWork soft cut
                const auto minSync                   = std::max(minSyncIn, static_cast<std::size_t>(input_chunk_size));
                requestedWork                        = std::clamp(requestedWork, minSync, maxSyncIn);
                const std::size_t nResamplingChunks2 = std::min(requestedWork / input_chunk_size, maxSyncOut / output_chunk_size);
                if (static_cast<std::size_t>(nResamplingChunks2 * input_chunk_size) >= minSyncIn && static_cast<std::size_t>(nResamplingChunks2 * output_chunk_size) >= minSyncOut) {
                    return ResamplingResult{.resampledIn = static_cast<std::size_t>(nResamplingChunks2 * input_chunk_size), .resampledOut = static_cast<std::size_t>(nResamplingChunks2 * output_chunk_size)};
                }
            }
            return ResamplingResult{.resampledIn = static_cast<std::size_t>(nResamplingChunks * input_chunk_size), .resampledOut = static_cast<std::size_t>(nResamplingChunks * output_chunk_size)};
        }
    }

    std::size_t getMergedBlockLimit() {
        if constexpr (Derived::blockCategory != block::Category::NormalBlock) {
            return 0UZ;
        } else if constexpr (requires(const Derived& d) {
                                 { available_samples(d) } -> std::same_as<std::size_t>;
                             }) {
            return available_samples(self());
        } else if constexpr (traits::block::stream_input_port_types<Derived>::size == 0UZ     // allow blocks that have neither input nor output ports
                             && traits::block::stream_output_port_types<Derived>::size == 0UZ // (by merging source to sink block) -> use internal buffer size
                             && requires { Derived::merged_work_chunk_size(); }) {            //
            constexpr gr::Size_t chunkSize = Derived::merged_work_chunk_size();
            static_assert(chunkSize != std::dynamic_extent && chunkSize > 0, "At least one internal port must define a maximum number of samples or the non-member/hidden "
                                                                             "friend function `available_samples(const BlockType&)` must be defined.");
            return chunkSize;
        } else {
            return std::numeric_limits<std::size_t>::max();
        }
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
        std::size_t i = 0UZ;
        for (; i + width <= nSamplesToProcess; i += width) {
            const auto& results = simdize_tuple_load_and_apply(width, inputSpans, i, [&](const auto&... input_simds) { return invoke_processOne_simd(width, input_simds...); });
            meta::tuple_for_each([i](auto& output_range, const auto& result) { result.copy_to(output_range.data() + i, stdx::element_aligned); }, outputSpans, results);
        }
        simd_epilogue(width, [&](auto w) {
            if (i + w <= nSamplesToProcess) {
                const auto results = simdize_tuple_load_and_apply(w, inputSpans, i, [&](auto&&... input_simds) { return invoke_processOne_simd(w, input_simds...); });
                meta::tuple_for_each([i](auto& output_range, auto& result) { result.copy_to(output_range.data() + i, stdx::element_aligned); }, outputSpans, results);
                i += w;
            }
        });
        return work::Status::OK;
    }

    work::Status invokeProcessOnePure(auto& inputSpans, auto& outputSpans, std::size_t nSamplesToProcess) {
        for (std::size_t i = 0UZ; i < nSamplesToProcess; ++i) {
            auto results = std::apply([this, i](auto&... inputs) { return this->invoke_processOne(inputs[i]...); }, inputSpans);
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

        std::size_t nOutSamplesBeforeRequestedStop = 0UZ;
        for (std::size_t i = 0UZ; i < nSamplesToProcess; ++i) {
            auto results = std::apply([this, i](auto&... inputs) { return this->invoke_processOne(inputs[i]...); }, inputSpans);
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
        _outputTagsChanged = false;
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

    work::Result workInternal(std::size_t requestedWork)
    requires(Derived::blockCategory == block::Category::NormalBlock)
    {
        using enum gr::work::Status;
        using TInputTypes  = traits::block::stream_input_port_types<Derived>;
        using TOutputTypes = traits::block::stream_output_port_types<Derived>;

        applyChangedSettings(); // apply settings even if the block is already stopped

        if constexpr (!blockingIO) { // N.B. no other thread/constraint to consider before shutting down
            if (this->state() == lifecycle::State::REQUESTED_STOP) {
                emitErrorMessageIfAny("workInternal(): REQUESTED_STOP -> STOPPED", this->changeStateTo(lifecycle::State::STOPPED));
            }
        }

        if constexpr (TOutputTypes::size.value > 0UZ) {
            if (disconnect_on_done && hasNoDownStreamConnectedChildren()) {
                this->requestStop(); // no dependent non-optional children, should stop processing
            }
        }

        if (this->state() == lifecycle::State::STOPPED) {
            disconnectFromUpStreamParents();
            return {requestedWork, 0UZ, DONE};
        }

        // TODO: finally remove me
        // const auto [minSyncIn, maxSyncIn, maxSyncAvailableIn, hasAsyncIn] = getPortLimits(inputPorts<PortType::STREAM>(&self()));
        // const auto [minSyncOut, maxSyncOut, maxSyncAvailableOut, hasAsyncOut] = getPortLimits(outputPorts<PortType::STREAM>(&self()));

        on_scope_exit _cacheGuard = [&] {
            inputStreamCache.invalidateStatistic();
            outputStreamCache.invalidateStatistic();
        };
        std::size_t minSyncIn           = inputStreamCache.minSyncRequirement();
        std::size_t maxSyncIn           = inputStreamCache.maxSyncRequirement();
        std::size_t maxSyncAvailableIn  = inputStreamCache.maxSyncAvailable();
        bool        hasAsyncIn          = inputStreamCache.hasASyncAvailable();
        std::size_t minSyncOut          = outputStreamCache.minSyncRequirement();
        std::size_t maxSyncOut          = outputStreamCache.maxSyncRequirement();
        std::size_t maxSyncAvailableOut = outputStreamCache.maxSyncAvailable();
        bool        hasAsyncOut         = outputStreamCache.hasASyncAvailable();

        auto [hasTag, nextTag, nextEosTag, asyncEoS]      = getNextTagAndEosPosition();
        std::size_t maxChunk                              = getMergedBlockLimit(); // handle special cases for merged blocks. TODO: evaluate if/how we can get rid of these
        const auto  inputSkipBefore                       = inputSamplesToSkipBeforeNextChunk(std::min({maxSyncAvailableIn, nextTag, nextEosTag}));
        const auto  nextTagLimit                          = (nextTag - inputSkipBefore) >= minSyncIn ? (nextTag - inputSkipBefore) : std::numeric_limits<std::size_t>::max();
        const auto  ensureMinimalDecimation               = nextTagLimit >= input_chunk_size ? nextTagLimit : static_cast<long unsigned int>(input_chunk_size); // ensure to process at least one input_chunk_size (may shift tags)
        const auto  availableToProcess                    = std::min({maxSyncIn, maxChunk, (maxSyncAvailableIn - inputSkipBefore), ensureMinimalDecimation, (nextEosTag - inputSkipBefore)});
        const auto  availableToPublish                    = std::min({maxSyncOut, maxSyncAvailableOut});
        auto [resampledIn, resampledOut, resampledStatus] = computeResampling(std::min(minSyncIn, nextEosTag), availableToProcess, minSyncOut, availableToPublish, requestedWork);
        const auto nextEosTagSkipBefore                   = nextEosTag - inputSkipBefore;
        const bool isEosTagPresent                        = nextEosTag <= 0 || nextEosTagSkipBefore < minSyncIn || nextEosTagSkipBefore < input_chunk_size || output_chunk_size * (nextEosTagSkipBefore / input_chunk_size) < minSyncOut;

        if (inputSkipBefore > 0) {                                                                    // consume samples on sync ports that need to be consumed due to the stride
            auto inputSpans = prepareStreams(inputPorts<PortType::STREAM>(&self()), inputSkipBefore); // only way to consume is via the ReaderSpanLike now
            updateMergedInputTagAndApplySettings(inputSpans, inputSkipBefore);                        // apply all tags in the skipped data range
            consumeReaders(inputSkipBefore, inputSpans);
        }
        // return if there is no work to be performed // todo: add eos policy
        if (isEosTagPresent || lifecycle::isShuttingDown(this->state()) || asyncEoS) {
            emitErrorMessageIfAny("workInternal(): EOS tag arrived -> REQUESTED_STOP", this->changeStateTo(lifecycle::State::REQUESTED_STOP));
            publishEoS();
            this->setAndNotifyState(lifecycle::State::STOPPED);
            return {requestedWork, 0UZ, DONE};
        }

        if (resampledIn == 0 && resampledOut == 0 && !hasAsyncIn && !hasAsyncOut) {
            return {requestedWork, 0UZ, resampledStatus};
        }

        // for non-bulk processing, the processed span has to be limited to the first sample if it contains a tag s.t. the tag is not applied to every sample
        const bool limitByFirstTag = (!HasProcessBulkFunction<Derived> && HasProcessOneFunction<Derived>) && hasTag;

        // call the block implementation's work function
        work::Status userReturnStatus = ERROR; // default if nothing has been set
        std::size_t  processedIn      = limitByFirstTag ? 1UZ : resampledIn;
        std::size_t  processedOut     = limitByFirstTag ? 1UZ : resampledOut;

        auto inputSpans  = prepareStreams(inputPorts<PortType::STREAM>(&self()), processedIn);
        auto outputSpans = prepareStreams(outputPorts<PortType::STREAM>(&self()), processedOut);

        updateMergedInputTagAndApplySettings(inputSpans, processedIn);

        applyChangedSettings();

        // Actual publishing occurs when outputSpans go out of scope. If processedOut == 0, the Tags will not be published.
        publishCachedOutputTags(outputSpans);
        publishMergedInputTag(outputSpans);

        if constexpr (HasProcessBulkFunction<Derived>) {
            invokeUserProvidedFunction("invokeProcessBulk", [&userReturnStatus, &inputSpans, &outputSpans, this] noexcept(HasNoexceptProcessBulkFunction<Derived>) { userReturnStatus = invokeProcessBulk(inputSpans, outputSpans); });

            for_each_reader_span(
                [&processedIn](auto& in) {
                    if (in.isConsumeRequested() && in.isConnected && in.isSync) {
                        processedIn = std::min(processedIn, in.nRequestedSamplesToConsume());
                    }
                },
                inputSpans);

            for_each_writer_span(
                [&processedOut](auto& out) {
                    if (out.isPublishRequested() && out.isConnected && out.isSync) {
                        processedOut = std::min(processedOut, out.nRequestedSamplesToPublish());
                    }
                },
                outputSpans);

        } else if constexpr (HasProcessOneFunction<Derived>) {
            if (processedIn != processedOut) {
                emitErrorMessage("Block::workInternal:", std::format("N input samples ({}) does not equal to N output samples ({}) for processOne() method.", resampledIn, resampledOut));
                requestStop();
                processedIn  = 0;
                processedOut = 0;
            } else {
                constexpr bool        kIsSourceBlock = TInputTypes::size() == 0;
                constexpr std::size_t kMaxWidth      = stdx::simd_abi::max_fixed_size<double>;
                // A block determines it's simd::size() via its input types. However, a source block doesn't have any
                // input types and therefore wouldn't be able to produce simd output on processOne calls. To overcome
                // this limitation, a source block can implement `processOne_simd(vir::constexpr_value auto width)`
                // instead of `processOne()` and then return simd objects with simd::size() == width.
                constexpr bool kIsSimdSourceBlock = kIsSourceBlock and requires(Derived& d) { d.processOne_simd(vir::cw<kMaxWidth>); };
                if constexpr (HasConstProcessOneFunction<Derived>) { // processOne is const -> can process whole batch similar to SIMD-ised call
                    if constexpr (kIsSimdSourceBlock or traits::block::can_processOne_simd<Derived>) {
                        // SIMD loop
                        constexpr auto kWidth = [&] {
                            if constexpr (kIsSourceBlock) {
                                return vir::cw<kMaxWidth>;
                            } else {
                                return vir::cw<std::min(kMaxWidth, vir::simdize<typename TInputTypes::template apply<std::tuple>>::size() * std::size_t(4))>;
                            }
                        }();
                        invokeUserProvidedFunction("invokeProcessOneSimd", [&userReturnStatus, &inputSpans, &outputSpans, &kWidth, &processedIn, this] noexcept(HasNoexceptProcessOneFunction<Derived>) { userReturnStatus = invokeProcessOneSimd(inputSpans, outputSpans, kWidth, processedIn); });
                    } else { // Non-SIMD loop
                        invokeUserProvidedFunction("invokeProcessOnePure", [&userReturnStatus, &inputSpans, &outputSpans, &processedIn, this] noexcept(HasNoexceptProcessOneFunction<Derived>) { userReturnStatus = invokeProcessOnePure(inputSpans, outputSpans, processedIn); });
                    }
                } else { // processOne isn't const i.e. not a pure function w/o side effects -> need to evaluate state
                         // after each sample
                    static_assert(not kIsSimdSourceBlock and not traits::block::can_processOne_simd<Derived>, "A non-const processOne function implies sample-by-sample processing, which is not compatible with SIMD arguments. Consider marking the function 'const' or using non-SIMD argument types.");
                    const auto result = invokeProcessOneNonConst(inputSpans, outputSpans, processedIn);
                    userReturnStatus  = result.status;
                    processedIn       = result.processedIn;
                    processedOut      = result.processedOut;
                }
            }
        } else { // block does not define any valid processing function
            meta::print_types<meta::message_type<"neither processBulk(...) nor processOne(...) implemented for:">, Derived>{};
        }

        // sanitise input/output samples based on explicit user-defined processBulk(...) return status
        if (userReturnStatus == INSUFFICIENT_OUTPUT_ITEMS || userReturnStatus == INSUFFICIENT_INPUT_ITEMS || userReturnStatus == ERROR) {
            processedIn  = 0UZ;
            processedOut = 0UZ;
        }

        if (processedOut > 0) {
            publishCachedOutputTags(outputSpans);
            _mergedInputTag.map.clear(); // clear temporary cached input tags after processing - won't be needed after this
        } else {
            // if no data is published or consumed => do not publish any tags
            for_each_writer_span([](auto& outSpan) { outSpan.tagsPublished = 0; }, outputSpans);
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
            publishEoS(outputSpans);
        }

        // check/sanitise return values (N.B. these are used by the scheduler as indicators
        // whether and how much 'work' has been done to -- for example -- prioritise one block over another
        std::size_t performedWork = 0UZ;
        if (userReturnStatus == OK) {
            constexpr bool kIsSourceBlock = traits::block::stream_input_port_types<Derived>::size == 0;
            constexpr bool kIsSinkBlock   = traits::block::stream_output_port_types<Derived>::size == 0;
            if constexpr (!kIsSourceBlock && !kIsSinkBlock) { // normal block with input(s) and output(s)
                performedWork = processedIn;
            } else if constexpr (kIsSinkBlock) {
                performedWork = processedIn;
            } else if constexpr (kIsSourceBlock) {
                performedWork = processedOut;
            } else {
                performedWork = 1UZ;
            }

            if (performedWork > 0UZ) {
                progress->incrementAndGet();
            }
            if constexpr (blockingIO) {
                progress->notify_all();
            }
        }
        return {requestedWork, performedWork, userReturnStatus};
    } // end: work::Result workInternal(std::size_t requestedWork) { ... }

public:
    /**
     * @brief Process as many samples as available and compatible with the internal boundary requirements or limited by 'requested_work`
     *
     * @param requested_work: usually the processed number of input samples, but could be any other metric as long as
     * requested_work limit as an affine relation with the returned performed_work.
     * @return { requested_work, performed_work, status}
     */
    template<typename = void>
    work::Result work(std::size_t requestedWork = std::numeric_limits<std::size_t>::max()) noexcept
    requires(!blockingIO) // regular non-blocking call
    {
        if constexpr (Derived::blockCategory != block::Category::NormalBlock) {
            return {requestedWork, 0UZ, gr::work::Status::OK};
        } else {
            return workInternal(requestedWork);
        }
    }

    work::Status invokeWork()
    requires(blockingIO && Derived::blockCategory == block::Category::NormalBlock)
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
    requires(blockingIO && Derived::blockCategory == block::Category::NormalBlock) // regular blocking call (e.g. wating on HW, timer, blocking for any other reasons) -> this should be an exceptional use
    {
        constexpr bool useIoThread = std::disjunction_v<std::is_same<BlockingIO<true>, Arguments>...>;
        std::atomic_store_explicit(&ioRequestedWork, requested_work, std::memory_order_release);

        bool expectedThreadState = false;
        if (lifecycle::isActive(this->state()) && this->ioThreadRunning.compare_exchange_strong(expectedThreadState, true, std::memory_order_acq_rel)) {
            if constexpr (useIoThread) { // use graph-provided ioThreadPool
                std::shared_ptr<thread_pool::TaskExecutor> executor = gr::thread_pool::Manager::instance().get(compute_domain);
                if (!executor) {
                    emitErrorMessage("work(..)", std::format("blockingIO with useIoThread - no ioThreadPool being set or '{}' is unknown", compute_domain));
                    return {requested_work, 0UZ, work::Status::ERROR};
                }

                executor->execute([this]() {
                    assert(lifecycle::isActive(this->state()));
                    gr::thread_pool::thread::setThreadName(gr::meta::shorten_type_name(this->unique_name));

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

    void processMessages([[maybe_unused]] const MsgPortInBuiltin& port, std::span<const Message> messages) {
        using enum gr::message::Command;
        assert(std::addressof(port) == std::addressof(msgIn) && "got a message on wrong port");

        for (const auto& message : messages) {
            if (!message.serviceName.empty() && message.serviceName != unique_name && message.serviceName != name) {
                // Skip if target does not match the block's (unique) name and is not empty.
                continue;
            }

            auto it = propertyCallbacks.find(message.endpoint);
            if (it == propertyCallbacks.end()) {
                continue; // did not find matching property callback
            }
            BlockBase::PropertyCallback callback = it->second;

            std::optional<Message> retMessage;
            try {
                retMessage = (this->*callback)(message.endpoint, message); // N.B. life-time: message is copied
            } catch (const gr::exception& e) {
                retMessage       = Message{message};
                retMessage->data = std::unexpected(Error(e));
            } catch (const std::exception& e) {
                retMessage       = Message{message};
                retMessage->data = std::unexpected(Error(e));
            } catch (...) {
                retMessage       = Message{message};
                retMessage->data = std::unexpected(Error(std::format("unknown exception in Block {} property '{}'\n request message: {} ", unique_name, message.endpoint, message)));
            }

            if (!retMessage.has_value()) {
                continue; // function does not produce any return message
            }

            retMessage->cmd             = Final; // N.B. could enable/allow for partial if we return multiple messages (e.g. using coroutines?)
            retMessage->serviceName     = unique_name;
            WriterSpanLike auto msgSpan = msgOut.streamWriter().tryReserve<SpanReleasePolicy::ProcessAll>(1UZ);
            if (msgSpan.empty()) {
                throw gr::exception(std::format("{}::processMessages() can not reserve span for message\n", name));
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
constexpr std::string_view shortTypeName() {
    if constexpr (std::is_same_v<T, gr::property_map>) {
        return "gr::property_map";
    } else {
        return refl::type_name<T>;
    }
};

} // namespace detail

template<typename TBlock, typename TDecayedBlock>
void checkBlockContracts() {
    // N.B. some checks could be evaluated during compile time but the expressed intent is to do this during runtime to allow
    // for more verbose feedback on method signatures etc.
    if constexpr (refl::reflectable<TDecayedBlock>) {
        []<std::size_t... Idxs>(std::index_sequence<Idxs...>) {
            (
                [] {
                    using MemberType           = refl::data_member_type<TDecayedBlock, Idxs>;
                    using RawType              = std::remove_cvref_t<MemberType>;
                    using Type                 = std::remove_cvref_t<unwrap_if_wrapped_t<RawType>>;
                    constexpr bool isAnnotated = !std::is_same_v<RawType, Type>;
                    // N.B. this function is compile-time ready but static_assert does not allow for configurable error
                    // messages
                    if constexpr (!gr::settings::isReadableMember<Type>() && !traits::port::AnyPort<Type>) {
                        throw std::invalid_argument(std::format("block {} {}member '{}' has unsupported setting type '{}'", //
                            gr::meta::type_name<TDecayedBlock>(), isAnnotated ? "" : "annotated ", refl::data_member_name<TDecayedBlock, Idxs>.view(), detail::shortTypeName<Type>()));
                    }
                }(),
                ...);
        }(std::make_index_sequence<refl::data_member_count<TDecayedBlock>>());
    }

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
        std::string signatureProcessOne = std::format("* Option Ia (pure function):\n\n{}\n\n* Option Ib (allows modifications: settings, Tags, state, errors,...):\n\n{}\n\n* Option Ic (explicit return types):\n\n{}\n\n", //
std::format(R"(auto processOne({}) const noexcept {{
    /* add code here */
    return {}{}{};
}})",
    detail::for_each_type_to_string<TInputTypes>([]<typename T>(auto index, T) { return std::format("{} in{}", detail::shortTypeName<T>(), index); }),
    b1, detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto, T) { return std::format("{}()", detail::shortTypeName<T>()); }), b2),
std::format(R"(auto processOne({}) {{
    /* add code here */
    return {}{}{};
}})",
    detail::for_each_type_to_string<TInputTypes>([]<typename T>(auto index, T) { return std::format("{} in{}", detail::shortTypeName<T>(), index); }),
    b1, detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto, T) { return std::format("{}()", detail::shortTypeName<T>()); }), b2),
std::format(R"(std::tuple<{}> processOne({}) {{
    /* add code here */
    return {}{}{};
}})",
   detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto, T) { return std::format("{}", detail::shortTypeName<T>()); }), //
   detail::for_each_type_to_string<TInputTypes>([]<typename T>(auto index, T) { return std::format("{} in{}", detail::shortTypeName<T>(), index); }), //
   b1, detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto, T) { return std::format("{}()", detail::shortTypeName<T>()); }), b2)
);

std::string signaturesProcessBulk = std::format("* Option II:\n\n{}\n\nadvanced:* Option III:\n\n{}\n\n\n",
std::format(R"(gr::work::Status processBulk({}{}{}) {{
    /* add code here */
    return gr::work::Status::OK;
}})", //
    detail::for_each_type_to_string<TInputTypes>([]<typename T>(auto index, T) { return std::format("std::span<const {}> in{}", detail::shortTypeName<T>(), index); }), //
    (TInputTypes::size == 0UZ || TOutputTypes::size == 0UZ ? "" : ", "),                                                                             //
    detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto index, T) { return std::format("std::span<{}> out{}", detail::shortTypeName<T>(), index); })),
std::format(R"(gr::work::Status processBulk({}{}{}) {{
    /* add code here */
    return gr::work::Status::OK;
}})", //
    detail::for_each_type_to_string<TInputTypes>([]<typename T>(auto index, T) { return std::format("std::span<const {}> in{}", detail::shortTypeName<T>(), index); }), //
    (TInputTypes::size == 0UZ || TOutputTypes::size == 0UZ ? "" : ", "),                                                                             //
    detail::for_each_type_to_string<TOutputTypes>([]<typename T>(auto index, T) { return std::format("OutputSpanLike auto out{}", detail::shortTypeName<T>(), index); })));
        // clang-format on

        bool has_port_collection = false;
        TInputTypes::for_each([&has_port_collection]<typename T>(auto, T) { has_port_collection |= requires { typename T::value_type; }; });
        TOutputTypes::for_each([&has_port_collection]<typename T>(auto, T) { has_port_collection |= requires { typename T::value_type; }; });
        const std::string signatures = (has_port_collection ? "" : signatureProcessOne) + signaturesProcessBulk;
        throw std::invalid_argument(std::format("block {} has neither a valid processOne(...) nor valid processBulk(...) method\nPossible valid signatures (copy-paste):\n\n{}", detail::shortTypeName<TDecayedBlock>(), signatures));
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
    /*constexpr*/ std::string ret = std::format("# {}\n{}\n{}\n**supported data types:**", //
        gr::meta::type_name<DerivedBlock>(), TBlock::description, kIsBlocking ? "**BlockingIO**\n_i.e. potentially non-deterministic/non-real-time behaviour_\n" : "");
    gr::meta::typelist<SupportedTypes>::for_each([&](std::size_t index, auto&& t) {
        std::string type_name = gr::meta::type_name<decltype(t)>();
        ret += std::format("{}:{} ", index, type_name);
    });
    ret += std::format("\n**Parameters:**\n");
    if constexpr (refl::reflectable<DerivedBlock>) {
        refl::for_each_data_member_index<DerivedBlock>([&](auto kIdx) {
            using RawType = std::remove_cvref_t<refl::data_member_type<DerivedBlock, kIdx>>;
            using Type    = unwrap_if_wrapped_t<RawType>;
            if constexpr ((std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string>)) {
                if constexpr (is_annotated<RawType>()) {
                    ret += std::format("{}{:10} {:<20} - annotated info: {} unit: [{}] documentation: {}{}\n",
                        RawType::visible() ? "" : "_",                                                   //
                        refl::type_name<Type>.view(), refl::data_member_name<DerivedBlock, kIdx>.view(), //
                        RawType::description(), RawType::unit(),
                        RawType::documentation(), //
                        RawType::visible() ? "" : "_");
                } else {
                    ret += std::format("_{:10} {}_\n", refl::type_name<Type>.view(), refl::data_member_name<DerivedBlock, kIdx>.view());
                }
            }
        });
    }
    ret += std::format("\n~~Ports:~~\ntbd.");
    return ret;
}

namespace detail {

template<meta::fixed_string Acc>
struct fixed_string_concat_helper {
    static constexpr auto value = Acc;

    template<meta::fixed_string Append>
    constexpr auto operator%(meta::constexpr_string<Append>) const {
        if constexpr (Acc.empty()) {
            return fixed_string_concat_helper<Append>{};
        } else {
            return fixed_string_concat_helper<Acc + "," + Append>{};
        }
    }
};

template<typename... Types>
constexpr auto encodeListOfTypes() {
    return meta::constexpr_string<(fixed_string_concat_helper<"">{} % ... % refl::type_name<Types>).value>();
}
} // namespace detail

template<typename... Types>
struct BlockParameters : meta::typelist<Types...> {
    static constexpr /*meta::constexpr_string*/ auto toString() { return detail::encodeListOfTypes<Types...>(); }
};

template<typename TBlock, fixed_string OverrideName = "">
int registerBlock(auto& registerInstance) {
    using namespace vir::literals;
    constexpr auto name     = refl::class_name<TBlock>;
    constexpr auto longname = refl::type_name<TBlock>;
    if constexpr (OverrideName != "") {
        registerInstance.template insert<TBlock>(OverrideName, {});

    } else if constexpr (name != longname) {
        constexpr auto tmpl = longname.substring(name.size + 1_cw, longname.size - 2_cw - name.size);
        registerInstance.template insert<TBlock>(name, tmpl);
    } else {
        registerInstance.template insert<TBlock>(name, {});
    }
    return 0;
}

template<typename TBlock0, typename TBlock1, typename... More>
int registerBlock(auto& registerInstance) {
    registerBlock<TBlock0>(registerInstance);
    return registerBlock<TBlock1, More...>(registerInstance);
}

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
template<fixed_string Alias, template<typename...> typename TBlock, typename TBlockParameter0, typename... TBlockParameters>
int registerBlock(auto& registerInstance) {
    using List0     = std::conditional_t<meta::is_instantiation_of<TBlockParameter0, BlockParameters>, TBlockParameter0, BlockParameters<TBlockParameter0>>;
    using ThisBlock = typename List0::template apply<TBlock>;
    registerInstance.template insert<ThisBlock>(Alias, List0::toString());
    if constexpr (sizeof...(TBlockParameters) != 0) {
        return registerBlock<Alias, TBlock, TBlockParameters...>(registerInstance);
    } else {
        return {};
    }
}

template<template<typename...> typename TBlock, typename TBlockParameter0, typename... TBlockParameters>
int registerBlock(auto& registerInstance) {
    return registerBlock<"", TBlock, TBlockParameter0, TBlockParameters...>(registerInstance);
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
 *  - optionally more Tuples
 *
 * This function iterates over all combinations of the types in Tuple1 and Tuple2,
 * instantiates TBlock with each combination, and registers the block with the registry.
 */
template<template<typename...> typename TBlock, typename... Tuples>
inline constexpr int registerBlockTT(auto& registerInstance) {
    meta::outer_product<meta::to_typelist<Tuples>...>::for_each([&]<typename Types>(std::size_t, Types*) { registerBlock<TBlock, typename Types::template apply<BlockParameters>>(registerInstance); });
    return {};
}

// FIXME: the following are inconsistent in how they specialize the template. Multiple types can be given, resulting in
// multiple specializations for each type. If the template requires more than one type then the types must be passed as
// a typelist (BlockParameters) instead. NTTP arguments, however, can only be given exactly as many as there are NTTPs
// in the template. Also the order is "messed up".
// Suggestion:
// template <auto X> struct Nttp { static constexpr value = X; };
// then use e.g.
// - BlockParameters<float, Nttp<Value>>
// - BlockParameters<float, Nttp<Value0>, double, Nttp<Value1>, Nttp<Value2>>
// Sadly, we can't remove any of the following overloads because we can't generalize the template template parameter
// (yet). And in principle, there's always another overload missing.
template<template<typename, auto> typename TBlock, auto Value0, typename... TBlockParameters, typename TRegisterInstance>
inline constexpr int registerBlock(TRegisterInstance& registerInstance) {
    auto addBlockType = [&]<typename Type> {
        static_assert(!meta::is_instantiation_of<Type, BlockParameters>);
        using ThisBlock = TBlock<Type, Value0>;
        registerInstance.template addBlockType<ThisBlock>();
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
        registerInstance.template addBlockType<ThisBlock>();
    };
    ((addBlockType.template operator()<TBlockParameters>()), ...);
    return {};
}

template<template<typename, auto, auto> typename TBlock, auto Value0, auto Value1, typename... TBlockParameters, typename TRegisterInstance>
inline constexpr int registerBlock(TRegisterInstance& registerInstance) {
    auto addBlockType = [&]<typename Type> {
        static_assert(!meta::is_instantiation_of<Type, BlockParameters>);
        using ThisBlock = TBlock<Type, Value0, Value1>;
        registerInstance.template addBlockType<ThisBlock>();
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
        registerInstance.template addBlockType<ThisBlock>();
    };
    ((addBlockType.template operator()<TBlockParameters>()), ...);
    return {};
}

template<typename Function, typename Tuple>
inline constexpr void for_each_port(Function&& function, Tuple&& tuple) {
    gr::meta::tuple_for_each(
        [&function](auto&& arg) {
            using ArgType = std::decay_t<decltype(arg)>;
            if constexpr (traits::port::is_port_v<ArgType>) {
                function(arg); // arg is a port, apply function directly
            } else if constexpr (traits::port::is_port_collection_v<ArgType>) {
                for (auto& port : arg) { // arg is a collection of ports, apply function to each port
                    function(port);
                }
            } else {
                static_assert(gr::meta::always_false<ArgType>, "not a port or collection of ports");
            }
        },
        std::forward<Tuple>(tuple));
}

template<typename Function, typename Tuple>
inline constexpr void for_each_reader_span(Function&& function, Tuple&& tuple) {
    gr::meta::tuple_for_each(
        [&function](auto&& arg) {
            using ArgType = std::decay_t<decltype(arg)>;
            if constexpr (ReaderSpanLike<typename ArgType::value_type>) {
                for (auto& param : arg) {
                    function(param);
                }
            } else if constexpr (ReaderSpanLike<ArgType>) {
                function(arg);
            }
        },
        std::forward<Tuple>(tuple));
}

template<typename Function, typename Tuple>
inline constexpr void for_each_writer_span(Function&& function, Tuple&& tuple) {
    gr::meta::tuple_for_each(
        [&function](auto&& arg) {
            using ArgType = std::decay_t<decltype(arg)>;
            if constexpr (WriterSpanLike<typename ArgType::value_type>) {
                for (auto& param : arg) {
                    function(param);
                }
            } else if constexpr (WriterSpanLike<ArgType>) {
                function(arg);
            }
        },
        std::forward<Tuple>(tuple));
}

template<typename TFunction, typename TPortsTuple, typename TSpansTuple>
inline constexpr void for_each_port_and_reader_span(TFunction&& function, TPortsTuple&& ports, TSpansTuple&& spans) {
    static_assert(std::tuple_size_v<std::remove_cvref_t<TPortsTuple>> == std::tuple_size_v<std::remove_cvref_t<TSpansTuple>>, "ports and spans must have the same tuple size");

    gr::meta::tuple_for_each(
        [&function](auto&& portOrCollection, auto&& spanOrCollection) {
            using PortArgType = std::decay_t<decltype(portOrCollection)>;
            using SpanArgType = std::decay_t<decltype(spanOrCollection)>;

            static_assert(traits::port::is_port_v<PortArgType> == ReaderSpanLike<SpanArgType>);
            static_assert(traits::port::is_port_collection_v<PortArgType> == ReaderSpanLike<typename SpanArgType::value_type>);

            if constexpr (traits::port::is_port_v<PortArgType>) {
                std::invoke(function, portOrCollection, spanOrCollection);
            } else if constexpr (traits::port::is_port_collection_v<PortArgType>) {
                static_assert(traits::port::is_port_v<PortArgType> == traits::port::is_port_v<SpanArgType>);

                assert(std::distance(std::begin(portOrCollection), std::end(portOrCollection)) == std::distance(std::begin(spanOrCollection), std::end(spanOrCollection)));

                std::ranges::for_each(std::views::zip(portOrCollection, spanOrCollection), [&](auto&& portAndSpan) {
                    auto& [p, s] = portAndSpan;
                    std::invoke(function, p, s);
                });
            } else {
                static_assert(gr::meta::always_false<PortArgType>, "Not a port or collection of ports");
            }
        },
        std::forward<TPortsTuple>(ports), std::forward<TSpansTuple>(spans));
}
} // namespace gr

template<>
struct std::formatter<gr::work::Result, char> {
    constexpr auto parse(std::format_parse_context& ctx) {
        auto it = ctx.begin();
        if (it != ctx.end() && *it != '}') {
            throw std::format_error("invalid format");
        }
        return it;
    }

    template<typename FormatContext>
    auto format(const gr::work::Result& result, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "requested_work: {}, performed_work: {}, status: {}", result.requested_work, result.performed_work, magic_enum::enum_name(result.status));
    }
};

#endif // include guard
