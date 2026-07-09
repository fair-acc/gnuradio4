#ifndef GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP
#define GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP

#include <atomic>
#include <concepts>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/WorkStatus.hpp>

#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <gnuradio-4.0/WindowGeometry.hpp>
#include <gnuradio-4.0/device/DeviceBlockShadow.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/DeviceRelocatable.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>

namespace gr::device {

namespace detail {
template<std::size_t... InIdx, std::size_t... OutIdx>
auto canProcessBulkWithContextInvokeTest(DeviceContext& ctx, auto& block, auto& inputSpans, auto& outputSpans, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) -> decltype(block.processBulk(ctx, std::get<InIdx>(inputSpans)..., std::get<OutIdx>(outputSpans)...));

template<std::size_t... InIdx, std::size_t... OutIdx>
[[nodiscard]] gr::work::Status invokeProcessBulkDevice(DeviceContext& ctx, auto& block, auto& inputSpans, auto& outputSpans, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) {
    return block.processBulk(ctx, std::get<InIdx>(inputSpans)..., std::get<OutIdx>(outputSpans)...);
}

// trait-test blocks own no warn-once flag and always warn
template<std::size_t... InIdx, std::size_t... OutIdx>
[[nodiscard]] auto invokeBulkOverSpans(auto& block, auto& inputSpans, auto& outputSpans, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) //
    -> decltype(block.processBulk(std::get<InIdx>(inputSpans)..., std::get<OutIdx>(outputSpans)...)) {
    return block.processBulk(std::get<InIdx>(inputSpans)..., std::get<OutIdx>(outputSpans)...);
}

template<std::size_t... InIdx, std::size_t... OutIdx>
auto invokeProcessOneOverSpans(auto& block, [[maybe_unused]] auto& inputSpans, [[maybe_unused]] auto& outputSpans, std::size_t i, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) //
    -> decltype(block.processOne(std::get<InIdx>(inputSpans)[i]...), void()) {
    if constexpr (sizeof...(OutIdx) == 0UZ) {
        block.processOne(std::get<InIdx>(inputSpans)[i]...); // a sink returns nothing to place
    } else {
        auto results = block.processOne(std::get<InIdx>(inputSpans)[i]...); // an empty input pack is a source
        if constexpr (sizeof...(OutIdx) == 1UZ) {
            ((std::get<OutIdx>(outputSpans)[i] = results), ...);
        } else {
            gr::meta::tuple_for_each([i]<typename R>(auto& output, R&& result) { output[i] = std::forward<R>(result); }, outputSpans, results);
        }
    }
}

template<typename TBlock>
[[nodiscard]] bool firstSerialBulkWarning(TBlock& block) noexcept {
    if constexpr (requires {
                      { block.markDeviceBulkSerialWarned() } -> std::same_as<bool>;
                  }) {
        return block.markDeviceBulkSerialWarned();
    } else {
        return true;
    }
}
} // namespace detail

template<typename TBlock, typename InputSpans, typename OutputSpans>
concept HasDeviceBulkHatch = requires(DeviceContext& ctx, TBlock& block, InputSpans& inputSpans, OutputSpans& outputSpans) {
    { detail::canProcessBulkWithContextInvokeTest(ctx, block, inputSpans, outputSpans, std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<InputSpans>>>(), std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<OutputSpans>>>()) } -> std::same_as<gr::work::Status>;
};

/// @brief Does this block's `processBulk` take plain views, so the framework may run windows of it at once?
///
/// The author-facing pair. `HasDeviceProcessBulk` and `HasDeviceProcessBulkSpans` state, in the sample types a block
/// author actually writes, which of the two `processBulk` shapes a block offers; the dispatcher asks the
/// `...ForSpans` forms below instead, which take the runtime span tuples it has in hand. Both exist on purpose: a
/// `static_assert` in a block's own test should read in the author's terms, not the dispatcher's.
///
/// const because every work item shares one device mirror.
template<typename TBlock, typename InT, typename OutT>
concept HasDeviceProcessBulk = requires(const TBlock& block, std::span<const InT>& in, std::span<OutT>& out) {
    { block.processBulk(in, out) } -> std::same_as<gr::work::Status>;
};

/// fixed because a kernel cannot allocate: the host reserves the slots and replays them after the launch
inline constexpr std::size_t kDeviceTagSlots     = 64UZ;
inline constexpr std::size_t kDeviceTagSlotBytes = 1024UZ; // multiple of gr::pmt::kBlobAlignment; holds a payload with a nested map

/// the classical span signature as ONE work item: it may consume/publish at its own rate and keep state.
template<typename TBlock, typename InT, typename OutT>
concept HasDeviceProcessBulkSpans = requires(const TBlock& block, DeviceInputSpan<InT>& in, DeviceOutputSpan<OutT>& out) {
    { block.processBulk(in, out) } -> std::same_as<gr::work::Status>;
};

namespace detail {
template<typename Spans, std::size_t kIdx>
using PortValue = std::ranges::range_value_t<std::remove_cvref_t<std::tuple_element_t<kIdx, std::remove_cvref_t<Spans>>>>;

template<typename TBlock, typename InputSpans, typename OutputSpans, std::size_t... InIdx, std::size_t... OutIdx>
auto canProcessBulkViewsInvokeTest(std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) -> decltype(std::declval<const TBlock&>().processBulk(std::declval<std::span<const PortValue<InputSpans, InIdx>>&>()..., std::declval<std::span<PortValue<OutputSpans, OutIdx>>&>()...));

template<typename TBlock, typename InputSpans, typename OutputSpans, std::size_t... InIdx, std::size_t... OutIdx>
auto canProcessBulkDeviceSpansInvokeTest(std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) -> decltype(std::declval<const TBlock&>().processBulk(std::declval<DeviceInputSpan<PortValue<InputSpans, InIdx>>&>()..., std::declval<DeviceOutputSpan<PortValue<OutputSpans, OutIdx>>&>()...));

template<typename Spans>
inline constexpr std::size_t kPortCount = std::tuple_size_v<std::remove_cvref_t<Spans>>;

/// a plain functor has no port descriptors, so naming them has to be deferred rather than merely guarded
template<typename TBlock, std::size_t kIdx, bool = gr::PortReflectable<TBlock>>
struct OutputPortValue {
    using type = void;
};

template<typename TBlock, std::size_t kIdx>
struct OutputPortValue<TBlock, kIdx, true> {
    using type = typename traits::block::stream_output_port_types<TBlock>::template at<kIdx>;
};

template<typename TBlock, std::size_t kIdx, bool = gr::PortReflectable<TBlock>>
struct InputPortValue {
    using type = void;
};

template<typename TBlock, std::size_t kIdx>
struct InputPortValue<TBlock, kIdx, true> {
    using type = typename traits::block::stream_input_port_types<TBlock>::template at<kIdx>;
};
} // namespace detail

template<typename TBlock, typename InputSpans, typename OutputSpans>
concept HasDeviceProcessBulkSpansForSpans = detail::kPortCount<InputSpans> > 0UZ && detail::kPortCount<OutputSpans> > 0UZ //
                                            && requires {
                                                   { detail::canProcessBulkDeviceSpansInvokeTest<TBlock, InputSpans, OutputSpans>(std::make_index_sequence<detail::kPortCount<InputSpans>>(), std::make_index_sequence<detail::kPortCount<OutputSpans>>()) } -> std::same_as<gr::work::Status>;
                                               };

template<typename TBlock, typename InputSpans, typename OutputSpans>
concept HasDeviceProcessBulkForSpans = detail::kPortCount<InputSpans> > 0UZ && detail::kPortCount<OutputSpans> > 0UZ //
                                       && requires {
                                              { detail::canProcessBulkViewsInvokeTest<TBlock, InputSpans, OutputSpans>(std::make_index_sequence<detail::kPortCount<InputSpans>>(), std::make_index_sequence<detail::kPortCount<OutputSpans>>()) } -> std::same_as<gr::work::Status>;
                                          };

/**
 * @brief Composed device dispatch helper for Block<T>::workInternal(); see docs/USER_API_GPU_Blocks.md.
 *
 * Three body shapes reach a device: a `const noexcept processOne` the framework runs one work item per sample; a
 * `const processBulk`, run one work item per declared window over plain views or as a single work item over port
 * spans that owns its own `consume`/`publish` and tags; and `processBulk(ctx, spans...)`, the expert hatch,
 * which runs on the host thread with the context in hand. The first two require `DeviceRelocatable` -- the block
 * *is* the functor. Nothing here falls back to the CPU: a shape the device cannot take is refused, and the block's
 * `compute_domain` decides whether that stops the graph or drops it to the host.
 */
template<typename TBlock>
struct ExecutionStrategy {
    struct DispatchOutcome {
        gr::work::Status status         = gr::work::Status::OK;
        bool             blockManagedIO = false;
        /// the tier consumed and produced at the block's declared ratio, so the two counts must not be collapsed
        bool honoursDeclaredRatio = false;

        constexpr DispatchOutcome(gr::work::Status s) noexcept : status(s) {} // framework-managed, 1:1
        constexpr DispatchOutcome(gr::work::Status s, bool managed) noexcept : status(s), blockManagedIO(managed) {}
        constexpr DispatchOutcome(gr::work::Status s, bool managed, bool declaredRatio) noexcept : status(s), blockManagedIO(managed), honoursDeclaredRatio(declaredRatio) {}
    };
    using DispatchResult = std::expected<DispatchOutcome, gr::Error>;

    template<typename InputSpans, typename OutputSpans>
    static consteval bool canDispatch() {
        constexpr auto nInputs  = std::tuple_size_v<std::remove_cvref_t<InputSpans>>;
        constexpr auto nOutputs = std::tuple_size_v<std::remove_cvref_t<OutputSpans>>;
        return HasDeviceBulkHatch<TBlock, InputSpans, OutputSpans>                                                  //
               || (DeviceRelocatable<TBlock> && HasDeviceProcessBulkSpansForSpans<TBlock, InputSpans, OutputSpans>) //
               || (DeviceRelocatable<TBlock> && HasDeviceProcessBulkForSpans<TBlock, InputSpans, OutputSpans>)      //
               || (AutoParallelisable<TBlock> && DeviceRelocatable<TBlock> && nInputs > 0UZ && nOutputs > 0UZ);
    }

    template<typename InputSpans, typename OutputSpans>
    static DispatchResult dispatch(TBlock& block, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t nIn, std::size_t nOut, std::string_view computeDomain, DeviceContext*& contextCache) {
        // every tier but the span one is 1:1 by construction, so they see the smaller of the two
        const std::size_t count = std::min(nIn, nOut);
        // `Block::dispatchProcessing` only calls this with a context already latched, but this is a public entry
        // point and `qa_DeviceSyclBulkTraits` pins the refusal for an unserved domain, so it resolves for itself
        DeviceContext* resolved = contextCache != nullptr ? contextCache : DeviceContextRegistry::instance().tryResolve(computeDomain);
        if (resolved == nullptr) {
            return refuseDeviceDispatch(std::format("compute_domain '{}' selects a device but no backend is wired", computeDomain));
        }
        contextCache       = resolved;
        DeviceContext& ctx = *resolved;
        // waitless at entry: every path that submits work polls at its end
        if (auto deviceErr = ctx.peekDeviceError()) {
            return fail(std::format("device context poisoned by a prior error: {}", *deviceErr));
        }

        if constexpr (HasDeviceBulkHatch<TBlock, InputSpans, OutputSpans>) {
            return dispatchDeviceHatch(block, ctx, inputSpans, outputSpans);
        }

        if constexpr (DeviceRelocatable<TBlock> && (HasDeviceProcessBulkSpansForSpans<TBlock, InputSpans, OutputSpans> || HasDeviceProcessBulkForSpans<TBlock, InputSpans, OutputSpans>)) {
            return dispatchDeviceProcessBulk(block, ctx, inputSpans, outputSpans, nIn, nOut);
        } else if constexpr (AutoParallelisable<TBlock> && DeviceRelocatable<TBlock> //
                             && std::tuple_size_v<std::remove_cvref_t<InputSpans>> > 0UZ && std::tuple_size_v<std::remove_cvref_t<OutputSpans>> > 0UZ) {
            return dispatchAutoParallel(block, ctx, inputSpans, outputSpans, count);
        } else if constexpr (!DeviceRelocatable<TBlock>) {
            // the only place the offending member can be named
            constexpr std::string_view offender = firstNonRelocatableMember<TBlock>();
            return refuseDeviceDispatch(std::format("member '{}' cannot be relocated to device memory (use a fundamental, trivially copyable, or pmr type)", offender));
        } else {
            return refuseDeviceDispatch("the resolved backend serves no device path for this block");
        }
    }

private:
    [[nodiscard]] static DispatchResult fail(std::string message) {
        gr::log::error("device dispatch: {}", message);
        return std::unexpected(gr::Error{message});
    }

    /// the hatch names no backend, so neither does this: the body is handed the context and asks it for what it needs
    template<typename InputSpans, typename OutputSpans>
    static DispatchResult dispatchDeviceHatch(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans)
    requires HasDeviceBulkHatch<TBlock, InputSpans, OutputSpans>
    {
        constexpr auto         nInputs  = std::tuple_size_v<std::remove_cvref_t<InputSpans>>;
        constexpr auto         nOutputs = std::tuple_size_v<std::remove_cvref_t<OutputSpans>>;
        const gr::work::Status status   = detail::invokeProcessBulkDevice(ctx, block, inputSpans, outputSpans, std::make_index_sequence<nInputs>(), std::make_index_sequence<nOutputs>());
        if (auto deviceErr = ctx.pollDeviceError()) {
            return fail(std::format("device fault during processBulk(ctx, ...): {}", *deviceErr));
        }
        return DispatchOutcome{status, true};
    }

    /// functors outside the Block hierarchy own no shadow, so their mirror is per-call and must be freed again
    static constexpr bool kOwnsDeviceShadow = requires(TBlock& b) { b.deviceShadow(); };

    /// hands out `count` objects of `T` from a control area, advancing `offset`. Every region starts on
    /// `max_align_t`, so no region needs alignment arithmetic of its own and the layout is the call order.
    template<typename T>
    [[nodiscard]] static T* carve(std::byte* area, std::size_t& offset, std::size_t count) noexcept {
        if (area == nullptr) {
            return nullptr;
        }
        T* const              slot  = std::launder(reinterpret_cast<T*>(area + offset));
        constexpr std::size_t kStep = alignof(std::max_align_t);
        offset += ((count * sizeof(T) + kStep - 1UZ) / kStep) * kStep;
        return slot;
    }

    /// what `carve` will consume for `count` objects of `T`, so a caller can size the area before carving it
    template<typename T>
    [[nodiscard]] static constexpr std::size_t carvedBytes(std::size_t count) noexcept {
        constexpr std::size_t kStep = alignof(std::max_align_t);
        return ((count * sizeof(T) + kStep - 1UZ) / kStep) * kStep;
    }

    static DeviceBuffer deviceMirror(TBlock& block, DeviceContext& ctx) {
        if constexpr (kOwnsDeviceShadow) {
            DeviceBlockShadow& shadow = block.deviceShadow();
            DeviceBuffer       mirror = shadow.acquire(ctx, sizeof(TBlock), alignof(TBlock));
            if (TBlock* p = mirror.devicePointer<TBlock>(); p != nullptr && shadow.epoch != block.settingsEpoch()) {
                // the settings changed, so the mirror is about to be rewritten from the host -- but a previous
                // dispatch may have returned while its kernel was still reading it. Nothing else orders a host store
                // against work already on the queue, so this is where a deferred launch is collected.
                shadow.awaitWorkInFlight();
                if (shadow.epoch == DeviceBlockShadow::kNeverRefreshed) {
                    relocateBlockToDevice(p, block);
                } else {
                    refreshDeviceSettings(p, block);
                }
                shadow.epoch = block.settingsEpoch();
            }
            return mirror;
        } else {
            DeviceBuffer mirror = ctx.allocateShared<TBlock>(1);
            if (TBlock* p = mirror.devicePointer<TBlock>(); p != nullptr) {
                relocateBlockToDevice(p, block);
            }
            return mirror;
        }
    }

    /// the probe costs one extra invocation, so it runs exactly when the mirror is about to be (re)built
    [[nodiscard]] static bool isFirstUseOfTheseSettings(TBlock& block) { // deviceShadow() is non-const
        if constexpr (kOwnsDeviceShadow) {
            return block.deviceShadow().epoch != block.settingsEpoch();
        }
        return true; // no shadow means the mirror is rebuilt on every call, so every call is a first use
    }

    /// A body reaching the owning `publishTag` overload does not merely misbehave on the device -- the SSCP
    /// JIT cannot build a kernel containing it at all, and the failure poisons the context for every later block. So
    /// the question is asked on the host, on a bit-copy, BEFORE anything is enqueued.
    ///
    /// It is a partial guard, deliberately: it answers only for the chunk it is given (a tag branch quiet at start-up
    /// slips through) and the caller may only ask at a host boundary, because on a device-resident span the pointers
    /// are memory this thread must not read. `qa_DeviceDspChain` shows half its span dispatches taking no check at
    /// all for that reason. The shape cannot be refused at compile time either -- `gr::OutputSpanLike` requires the
    /// owning overload -- so the rest is a documented contract; see docs/USER_API_GPU_Blocks.md.
    template<typename TInPtrs, typename TOutPtrs>
    [[nodiscard]] static bool bulkPublishesTags(const TBlock& block, TInPtrs inPtrs, TOutPtrs outPtrs, std::size_t count) {
        if constexpr (DeviceProbeSafe<TBlock>) {
            std::array<DeviceSpanAccounting, detail::kPortCount<TInPtrs>>  probeInAcct{};
            std::array<DeviceSpanAccounting, detail::kPortCount<TOutPtrs>> probeOutAcct{};
            std::ignore = mutatesItsOwnState(block, [&](TBlock& copy) {                                                                                                         // blocks are move-only: probe a bit-copy
                std::ignore = invokeBulkDeviceSpans(copy, inPtrs, outPtrs, count, count, DeviceSpanPortResources{.inAcct = probeInAcct.data(), .outAcct = probeOutAcct.data()}, //
                    std::make_index_sequence<detail::kPortCount<TInPtrs>>{}, std::make_index_sequence<detail::kPortCount<TOutPtrs>>{});
            });
            return std::ranges::any_of(probeOutAcct, [](const DeviceSpanAccounting& a) { return a.tagPublishAttempted; });
        } else {
            return false; // cannot probe safely; the post-kernel flag reports it instead
        }
    }

    template<typename TInPtrs, typename TOutPtrs>
    [[nodiscard]] static gr::work::Status invokeBulkViews(const TBlock& block, TInPtrs inPtrs, TOutPtrs outPtrs, std::size_t count) {
        return invokeBulkViewsAt(block, inPtrs, outPtrs, 0UZ, count, 0UZ, count);
    }

    /// the same body over one window: input at `inOffset`, output at `outOffset`, which is what lets N of them run at once.
    /// the spans are named locals because a body taking `InputViewLike auto&` cannot bind a temporary.
    template<typename TInPtrs, typename TOutPtrs>
    [[nodiscard]] static gr::work::Status invokeBulkViewsAt(const TBlock& block, TInPtrs inPtrs, TOutPtrs outPtrs, std::size_t inOffset, std::size_t inCount, std::size_t outOffset, std::size_t outCount) {
        auto inSpans  = std::apply([=](auto*... ins) { return std::tuple{std::span<const std::remove_pointer_t<decltype(ins)>>{ins + inOffset, inCount}...}; }, inPtrs);
        auto outSpans = std::apply([=](auto*... outs) { return std::tuple{std::span<std::remove_pointer_t<decltype(outs)>>{outs + outOffset, outCount}...}; }, outPtrs);
        return std::apply([&](auto&... in) { return std::apply([&](auto&... out) { return block.processBulk(in..., out...); }, outSpans); }, inSpans);
    }

    /// all members are device pointers into per-port slices, keeping this trivially copyable into a kernel
    struct DeviceSpanPortResources {
        DeviceSpanAccounting* inAcct        = nullptr; // one record per input port
        DeviceSpanAccounting* outAcct       = nullptr; // one record per output port
        gr::Tag*              inTags        = nullptr; // kDeviceTagSlots per input port
        std::size_t*          inTagCounts   = nullptr;
        std::size_t*          inStreamIndex = nullptr;
        std::byte*            tagSlots      = nullptr; // kDeviceTagSlots * kDeviceTagSlotBytes per output port
        std::size_t*          tagOffsets    = nullptr;
    };

    /// the port index comes from the pack: function arguments have no evaluation order
    template<typename TInPtrs, typename TOutPtrs, std::size_t... InIdx, std::size_t... OutIdx>
    [[nodiscard]] static gr::work::Status invokeBulkDeviceSpans(TBlock& block, TInPtrs inPtrs, TOutPtrs outPtrs, std::size_t nIn, std::size_t nOut, DeviceSpanPortResources res, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) {
        const auto inSpanFor = [&]<std::size_t kIdx>() {
            using T = std::remove_pointer_t<std::tuple_element_t<kIdx, TInPtrs>>;
            return DeviceInputSpan<T>{._data = std::get<kIdx>(inPtrs),
                ._size                       = nIn, //
                ._tags                       = res.inTags == nullptr ? nullptr : res.inTags + kIdx * kDeviceTagSlots,
                ._tagCount                   = res.inTagCounts == nullptr ? 0UZ : res.inTagCounts[kIdx], //
                ._acct                       = res.inAcct + kIdx,
                .streamIndex                 = res.inStreamIndex == nullptr ? 0UZ : res.inStreamIndex[kIdx]};
        };
        const auto outSpanFor = [&]<std::size_t kIdx>() {
            using T = std::remove_pointer_t<std::tuple_element_t<kIdx, TOutPtrs>>;
            return DeviceOutputSpan<T>{._data = std::get<kIdx>(outPtrs),
                ._size                        = nOut,
                ._acct                        = res.outAcct + kIdx,
                .tags                         = {},                                                                                              //
                ._tagSlots                    = res.tagSlots == nullptr ? nullptr : res.tagSlots + kIdx * kDeviceTagSlots * kDeviceTagSlotBytes, //
                ._tagOffsets                  = res.tagOffsets == nullptr ? nullptr : res.tagOffsets + kIdx * kDeviceTagSlots,
                ._tagSlotCount                = kDeviceTagSlots,
                ._tagSlotBytes                = kDeviceTagSlotBytes};
        };
        auto inSpans  = std::tuple { inSpanFor.template operator()<InIdx>()... };
        auto outSpans = std::tuple { outSpanFor.template operator()<OutIdx>()... };
        return block.processBulk(std::get<InIdx>(inSpans)..., std::get<OutIdx>(outSpans)...);
    }

    /// Debug-only: a pmr member reassigned outside the settings system leaves the mirror pointing at freed storage.
    /// Skipped while a deferred kernel is still reading that mirror -- a diagnostic that had to synchronise to run
    /// would cost exactly the barrier the dispatch just avoided, and would only ever fire on the next call anyway.
    [[nodiscard]] static std::optional<std::string> staleMirrorDiagnostic([[maybe_unused]] TBlock& block, [[maybe_unused]] const TBlock* mirror) {
        if constexpr (gr::meta::kDebugBuild) {
            if constexpr (kOwnsDeviceShadow) {
                if (block.deviceShadow().workInFlight) {
                    return std::nullopt;
                }
            }
            if (mirror != nullptr) {
                if (const std::string_view stale = firstStaleMirrorMember(block, *mirror); !stale.empty()) {
                    return std::format("member '{}' was reassigned without going through the settings system, so the device mirror still points at its previous storage", stale);
                }
            }
        }
        return std::nullopt;
    }

    /// decided per port: one block may be fed by a device edge and a host edge at once
    template<std::size_t kIdx, typename TSpans, typename TScratch>
    [[nodiscard]] static auto stageInputPort(DeviceContext& ctx, TSpans& spans, TScratch& scratch, std::size_t count, bool& staged) {
        auto& span = std::get<kIdx>(spans);
        if constexpr (kInputIsCollection<kIdx>) { // the same channel-major shape the output side stages into
            constexpr std::size_t nChannels = std::tuple_size_v<InputPortValue<kIdx>>;
            using T                         = typename InputPortValue<kIdx>::value_type;
            std::array<const T*, nChannels> channels{};

            std::size_t nResident = 0UZ; // one USM query per channel, asked once and remembered
            for (std::size_t channel = 0UZ; channel < nChannels; ++channel) {
                const T* const edge     = channel < span.size() ? span[channel].data() : nullptr;
                const bool     onDevice = edge != nullptr && ctx.isDeviceAccessible(edge);
                channels[channel]       = onDevice ? edge : nullptr;
                nResident += onDevice ? 1UZ : 0UZ;
            }
            if (nResident == nChannels) { // an interior device chain reads its channels where they already live
                return channels;
            }

            scratch[kIdx] = ctx.allocateShared<T>(nChannels * count);
            T* base       = scratch[kIdx].template devicePointer<T>();
            if (base == nullptr) {
                staged = false;
                channels.fill(nullptr);
                return channels;
            }
            for (std::size_t channel = 0UZ; channel < nChannels; ++channel) {
                if (channels[channel] != nullptr) {
                    continue; // already on the device, so there is nothing to bring over
                }
                T* const slot = base + channel * count;
                if (channel < span.size()) {
                    std::copy_n(span[channel].data(), count, slot);
                }
                channels[channel] = slot;
            }
            return channels;
        } else {
            using T = std::ranges::range_value_t<std::remove_cvref_t<decltype(span)>>;
            if (ctx.isDeviceAccessible(span.data())) {
                return const_cast<T*>(span.data());
            }
            scratch[kIdx] = ctx.allocateShared<T>(count);
            T* device     = scratch[kIdx].template devicePointer<T>();
            if (device == nullptr) {
                staged = false;
                return static_cast<T*>(nullptr);
            }
            ctx.copyHostToDevice(span.data(), scratch[kIdx], count);
            return device;
        }
    }

    template<typename TSpans, typename TScratch>
    [[nodiscard]] static auto stageInputPorts(DeviceContext& ctx, TSpans& spans, TScratch& scratch, std::size_t count, bool& staged) {
        return [&]<std::size_t... kIdx>(std::index_sequence<kIdx...>) { return std::tuple{stageInputPort<kIdx>(ctx, spans, scratch, count, staged)...}; }(std::make_index_sequence<detail::kPortCount<TSpans>>{});
    }

    /// a port collection carries its channel count in the block's port types, which the runtime span tuple has lost
    template<std::size_t kIdx>
    using OutputPortValue = typename detail::OutputPortValue<TBlock, kIdx>::type;

    /// plain functors carry no port descriptors, so the question does not arise for them
    template<std::size_t kIdx>
    static constexpr bool kOutputIsCollection = [] { // a complex sample also has a tuple size, so the port value must be a range over channels too
        return std::ranges::range<OutputPortValue<kIdx>> && requires { std::tuple_size<OutputPortValue<kIdx>>::value; };
    }();

    template<std::size_t kIdx>
    using InputPortValue = typename detail::InputPortValue<TBlock, kIdx>::type;

    template<std::size_t kIdx>
    static constexpr bool kInputIsCollection = [] { return std::ranges::range<InputPortValue<kIdx>> && requires { std::tuple_size<InputPortValue<kIdx>>::value; }; }();

    template<std::size_t kIdx, typename TSpans, typename TScratch, typename TCopyBack>
    [[nodiscard]] static auto stageOutputPort(DeviceContext& ctx, TSpans& spans, TScratch& scratch, TCopyBack& needsCopyBack, std::size_t count, bool& staged) {
        auto& span = std::get<kIdx>(spans);
        if constexpr (kOutputIsCollection<kIdx>) { // the channels are separate edges, so they get one shared scratch laid out channel-major
            constexpr std::size_t nChannels = std::tuple_size_v<OutputPortValue<kIdx>>;
            using T                         = typename OutputPortValue<kIdx>::value_type;
            std::array<T*, nChannels> channels{};

            // one USM query per channel, asked once and remembered. a short span leaves its tail unresident, so
            // the all-resident test below cannot be satisfied without every channel genuinely being on the device
            std::size_t nResident = 0UZ;
            for (std::size_t channel = 0UZ; channel < nChannels; ++channel) {
                T* const   edge     = channel < span.size() ? span[channel].data() : nullptr;
                const bool onDevice = edge != nullptr && ctx.isDeviceAccessible(edge);
                channels[channel]   = onDevice ? edge : nullptr;
                nResident += onDevice ? 1UZ : 0UZ;
            }
            if (nResident == nChannels) { // an interior device chain writes its channels where they already live
                needsCopyBack[kIdx] = false;
                return channels;
            }

            scratch[kIdx] = ctx.allocateShared<T>(nChannels * count);
            T* base       = scratch[kIdx].template devicePointer<T>();
            if (base == nullptr) {
                staged = false;
                channels.fill(nullptr);
                return channels;
            }
            for (std::size_t channel = 0UZ; channel < nChannels; ++channel) { // a resident channel keeps its own edge, so copy-back tells the two apart by pointer
                if (channels[channel] == nullptr) {
                    channels[channel] = base + channel * count;
                }
            }
            needsCopyBack[kIdx] = true;
            return channels;
        } else {
            using T = std::ranges::range_value_t<std::remove_cvref_t<decltype(span)>>;
            if (ctx.isDeviceAccessible(span.data())) {
                return span.data();
            }
            scratch[kIdx] = ctx.allocateShared<T>(count);
            T* device     = scratch[kIdx].template devicePointer<T>();
            if (device == nullptr) {
                staged = false;
            }
            needsCopyBack[kIdx] = true;
            return device;
        }
    }

    template<typename TSpans, typename TScratch, typename TCopyBack>
    [[nodiscard]] static auto stageOutputPorts(DeviceContext& ctx, TSpans& spans, TScratch& scratch, TCopyBack& needsCopyBack, std::size_t count, bool& staged) {
        return [&]<std::size_t... kIdx>(std::index_sequence<kIdx...>) { return std::tuple{stageOutputPort<kIdx>(ctx, spans, scratch, needsCopyBack, count, staged)...}; }(std::make_index_sequence<detail::kPortCount<TSpans>>{});
    }

    template<typename TSpans, typename TScratch, typename TCopyBack, typename TOutPtrs>
    static void copyBackOutputPorts(DeviceContext& ctx, TSpans& spans, TScratch& scratch, const TCopyBack& needsCopyBack, const TOutPtrs& outPtrs, std::size_t count) {
        const auto copyBackOne = [&]<std::size_t kIdx>() {
            if (!needsCopyBack[kIdx]) {
                return;
            }
            if constexpr (kOutputIsCollection<kIdx>) { // channel-major scratch, one edge per channel to land in
                constexpr std::size_t nChannels = std::tuple_size_v<OutputPortValue<kIdx>>;
                using T                         = typename OutputPortValue<kIdx>::value_type;
                const T* base                   = scratch[kIdx].template devicePointer<T>();
                for (std::size_t channel = 0UZ; channel < nChannels && channel < std::get<kIdx>(spans).size(); ++channel) {
                    if (std::get<kIdx>(outPtrs)[channel] != base + channel * count) {
                        continue; // the kernel wrote this channel's edge directly, so there is nothing to bring back
                    }
                    std::copy_n(base + channel * count, count, std::get<kIdx>(spans)[channel].data());
                }
            } else {
                ctx.copyDeviceToHost(scratch[kIdx], std::get<kIdx>(spans).data(), count);
            }
        };
        [&]<std::size_t... kIdx>(std::index_sequence<kIdx...>) { (copyBackOne.template operator()<kIdx>(), ...); }(std::make_index_sequence<detail::kPortCount<TSpans>>{});
    }

    /// ONE work item, for residency: a sequential body stays device-resident between its neighbours
    /**
     * @brief One kernel-body dispatcher for both `processBulk` forms.
     *
     * They differ in exactly two things: what the kernel is handed, and who owns the accounting. A body constrained
     * to the *view* concepts is run once per declared window, N windows at once, and the framework consumes and
     * publishes for it. A body constrained to the *span* concepts is run as a single work item and does its own
     * `consume`/`publish` and tags, which the host replays afterwards. Everything around that -- the mirror, the
     * staging, the control area, the release, the error poll -- is common, and used to be written out twice.
     *
     * A view-form body satisfies both concepts (`ReaderSpanLike` refines `InputViewLike`), so the window count
     * decides: declare one and it is parallelised, declare none and it runs as one work item like a span body.
     */
    template<typename InputSpans, typename OutputSpans>
    static DispatchResult dispatchDeviceProcessBulk(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t nIn, std::size_t nOut) {
        constexpr auto nInputs      = detail::kPortCount<InputSpans>;
        constexpr auto nOutputs     = detail::kPortCount<OutputSpans>;
        constexpr bool kHasSpanBody = HasDeviceProcessBulkSpansForSpans<TBlock, InputSpans, OutputSpans>;
        constexpr bool kHasViewBody = HasDeviceProcessBulkForSpans<TBlock, InputSpans, OutputSpans>;

        WindowGeometry geometry{}; // a ternary would instantiate windowGeometry() for a body that declares no chunk
        if constexpr (kHasViewBody) {
            geometry = windowGeometry(block, nIn, nOut);
        }
        const bool        runsAsWindows = kHasViewBody && geometry.nWindows >= 2UZ;
        const bool        bodyOwnsIO    = kHasSpanBody && !runsAsWindows;
        const std::size_t count         = std::min(nIn, nOut);

        if (!bodyOwnsIO && !runsAsWindows && detail::firstSerialBulkWarning(block)) {
            gr::log::warning("device dispatch: processBulk runs as one work item over the whole span, consuming and publishing all {} samples; declare a window with Resampling<>/Stride<> to have them run in parallel, or use processBulk(ctx, ...) to own the accounting", count);
        }

        const bool firstUse = isFirstUseOfTheseSettings(block); // ask BEFORE deviceMirror(): it refreshes that epoch

        DeviceBuffer dBlockBuf = deviceMirror(block, ctx);
        TBlock*      dBlock    = dBlockBuf.devicePointer<TBlock>();
        if (dBlock == nullptr) {
            // structural (the backend has no shared residency) or transient — either way, refuse rather than crash
            return refuseDeviceDispatch("the device context cannot provide shared (host-writable) device memory for a framework-managed kernel body");
        }
        if (auto stale = staleMirrorDiagnostic(block, dBlock)) {
            if constexpr (!kOwnsDeviceShadow) {
                ctx.deallocate(dBlockBuf);
            }
            return fail(*stale);
        }

        std::array<DeviceBuffer, nInputs>  inScratch{};
        std::array<DeviceBuffer, nOutputs> outScratch{};
        std::array<bool, nOutputs>         outNeedsCopyBack{};
        bool                               staged = true;

        auto inPtrs  = stageInputPorts(ctx, inputSpans, inScratch, nIn, staged);
        auto outPtrs = stageOutputPorts(ctx, outputSpans, outScratch, outNeedsCopyBack, nOut, staged);

        // one persistent area, carved in a fixed order, instead of an allocation and a free per work() call: every
        // size is a compile-time constant of the block's shape, and a `sycl::free` is an implicit device
        // synchronisation on CUDA. A body that owns no accounting needs only the status word.
        constexpr std::size_t kAccountingBytes = kHasSpanBody ? carvedBytes<DeviceSpanAccounting>(nInputs) + carvedBytes<DeviceSpanAccounting>(nOutputs)                                              //
                                                                    + carvedBytes<std::byte>(nOutputs * kDeviceTagSlots * kDeviceTagSlotBytes) + carvedBytes<std::size_t>(nOutputs * kDeviceTagSlots) //
                                                                    + carvedBytes<gr::Tag>(nInputs * kDeviceTagSlots) + carvedBytes<std::byte>(nInputs * kDeviceTagSlots * kDeviceTagSlotBytes)       //
                                                                    + carvedBytes<std::size_t>(nInputs) + carvedBytes<std::size_t>(nInputs)                                                           //
                                                              : 0UZ;
        constexpr std::size_t kControlBytes    = carvedBytes<std::uint32_t>(1UZ) + kAccountingBytes;

        DeviceBuffer ownedControl{}; // only a functor without a shadow owns one per call
        std::byte*   control = nullptr;
        if constexpr (kOwnsDeviceShadow) {
            control = block.deviceShadow().controlArea(ctx, kControlBytes);
        } else {
            ownedControl = ctx.allocate(kControlBytes, alignof(std::max_align_t), Residency::shared);
            control      = ownedControl.devicePointer<std::byte>();
        }

        std::size_t             carved     = 0UZ;
        std::uint32_t* const    statusPtr  = carve<std::uint32_t>(control, carved, 1UZ);
        std::byte*              inTagBlobs = nullptr;
        DeviceSpanPortResources res{};
        if constexpr (kHasSpanBody) {
            res.inAcct        = carve<DeviceSpanAccounting>(control, carved, nInputs);
            res.outAcct       = carve<DeviceSpanAccounting>(control, carved, nOutputs);
            res.tagSlots      = carve<std::byte>(control, carved, nOutputs * kDeviceTagSlots * kDeviceTagSlotBytes);
            res.tagOffsets    = carve<std::size_t>(control, carved, nOutputs * kDeviceTagSlots);
            res.inTags        = carve<gr::Tag>(control, carved, nInputs * kDeviceTagSlots);
            inTagBlobs        = carve<std::byte>(control, carved, nInputs * kDeviceTagSlots * kDeviceTagSlotBytes);
            res.inTagCounts   = carve<std::size_t>(control, carved, nInputs);
            res.inStreamIndex = carve<std::size_t>(control, carved, nInputs);
        }

        const auto release = [&] {
            if constexpr (!kOwnsDeviceShadow) {
                ctx.deallocate(dBlockBuf);
            }
            for (DeviceBuffer& buffer : inScratch) {
                ctx.deallocate(buffer);
            }
            for (DeviceBuffer& buffer : outScratch) {
                ctx.deallocate(buffer);
            }
            if (ownedControl) {
                ctx.deallocate(ownedControl);
            }
        };

        const bool controlIsComplete = statusPtr != nullptr && (!kHasSpanBody || (res.inAcct != nullptr && res.outAcct != nullptr && res.inTagCounts != nullptr && res.inStreamIndex != nullptr //
                                                                                     && res.inTags != nullptr && inTagBlobs != nullptr && res.tagSlots != nullptr && res.tagOffsets != nullptr));
        if (!staged || !controlIsComplete) { // else tags vanish without a word
            release();
            return fail(std::format("shared allocation failed for the device processBulk path ({} in / {} out samples over {} input and {} output ports)", nIn, nOut, nInputs, nOutputs));
        }
        *statusPtr = static_cast<std::uint32_t>(gr::work::Status::OK);

        if constexpr (kHasSpanBody) {
            std::ranges::fill(std::span<DeviceSpanAccounting>{res.inAcct, nInputs}, DeviceSpanAccounting{});
            std::ranges::fill(std::span<DeviceSpanAccounting>{res.outAcct, nOutputs}, DeviceSpanAccounting{});
        }

        if constexpr (kHasSpanBody) {
            if (bodyOwnsIO) {
                if (auto refusal = runSpanBody(block, ctx, inputSpans, outputSpans, inPtrs, outPtrs, res, inTagBlobs, statusPtr, dBlock, nIn, nOut, firstUse); refusal) {
                    release();
                    return *refusal;
                }
            }
        }
        // the span form can never defer: it replays consume/publish and tag slots out of accounting the kernel is
        // still writing. The view form can, on the same terms as `processOne` -- nothing this thread reads was touched
        const bool deferCompletion = !bodyOwnsIO && kOwnsDeviceShadow && allSpansDeviceOnly(ctx, inputSpans, outputSpans);
        if constexpr (kHasViewBody) { // a span-only body never takes this branch, and has no view form to instantiate
            if (!bodyOwnsIO) {
                // no status word when deferring: it lives in shared USM, the host resets it at the top of the NEXT
                // dispatch, and a still-running kernel writing the same word is the concurrent managed-memory access
                // the deferral contract says never happens. Its value is unreported anyway -- see below.
                runDeviceBulkCore(ctx, dBlock, inPtrs, outPtrs, count, geometry, deferCompletion ? nullptr : statusPtr, false);
            }
        }
        if constexpr (kOwnsDeviceShadow) {
            block.deviceShadow().workInFlight = deferCompletion;
        }

        // ONE barrier, not two: the launch above is enqueued without waiting, and `pollDeviceError` -- itself a queue
        // wait -- is what collects it. On an in-order queue the poll always subsumed the launch's own event wait, so
        // the pair cost two barriers to learn one thing. It must precede the copy-back and the release, which are
        // what may not run while the kernel does.
        const std::optional<std::string> deviceErr = deferCompletion ? ctx.peekDeviceError() : ctx.pollDeviceError();

        if constexpr (kHasViewBody) {
            if (!bodyOwnsIO) {
                copyBackOutputPorts(ctx, outputSpans, outScratch, outNeedsCopyBack, outPtrs, nOut);
            }
        }

        // A deferred kernel has not written the status word yet, and reading it would be the very wait just avoided,
        // so the dispatch reports OK and that kernel's status is not reported at all. Sound because on this tier the
        // FRAMEWORK owns consume/publish -- the body's status is advisory, not flow control -- and a genuine fault
        // still arrives, one work() call late, through the error poll. A body whose status must be heard belongs on
        // the span form or the hatch, neither of which defers.
        const gr::work::Status kernelStatus = deferCompletion ? gr::work::Status::OK : static_cast<gr::work::Status>(static_cast<std::int32_t>(*statusPtr));

        if constexpr (kHasSpanBody) {
            if (bodyOwnsIO) {
                replaySpanAccounting(ctx, inputSpans, outputSpans, outScratch, outNeedsCopyBack, res, nIn, nOut);
            }
        }

        release();
        if (deviceErr) {
            return fail(std::format("device fault during processBulk dispatch: {}", *deviceErr));
        }
        // a body that owns its accounting requested nothing of the framework; one that does not honours its window
        return bodyOwnsIO ? DispatchOutcome{kernelStatus, true} : DispatchOutcome{kernelStatus, false, runsAsWindows};
    }

    /// the span form's kernel: stage this chunk's input tags where a kernel can read them, run the body as ONE work
    /// item, and refuse if it reached for a tag payload only the host can build. Returns a refusal, or nothing.
    template<typename InputSpans, typename OutputSpans, typename TInPtrs, typename TOutPtrs>
    [[nodiscard]] static std::optional<DispatchResult> runSpanBody(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, TInPtrs inPtrs, TOutPtrs outPtrs, //
        const DeviceSpanPortResources& res, std::byte* inTagBlobs, std::uint32_t* statusPtr, TBlock* dBlock, std::size_t nIn, std::size_t nOut, bool firstUse) {
        constexpr auto nInputs  = detail::kPortCount<TInPtrs>;
        constexpr auto nOutputs = detail::kPortCount<TOutPtrs>;

        // EVERY span, input and output: the probe runs the body on THIS thread, so one device-resident port among
        // them is enough to make that a wild write. Dropping the outputs from this test segfaults an upsampler whose
        // input is staged but whose output ring is device memory.
        const bool anyResident = [&]<std::size_t... kIn, std::size_t... kOut>(std::index_sequence<kIn...>, std::index_sequence<kOut...>) { return (ctx.isDeviceAccessible(std::get<kIn>(inputSpans).data()) || ...) || (ctx.isDeviceAccessible(std::get<kOut>(outputSpans).data()) || ...); }(std::make_index_sequence<nInputs>{}, std::make_index_sequence<nOutputs>{});
        if (firstUse && !anyResident && bulkPublishesTags(block, inPtrs, outPtrs, std::min(nIn, nOut))) {
            return refuseDeviceDispatch("processBulk publishes tags, which a device kernel cannot build");
        }

        // staged, not pointed at: `rawTags()` is a lazy projection, and a host tag-ring payload is neither
        // device-reachable nor `kBlobAlignment`-aligned
        [&]<std::size_t... kIdx>(std::index_sequence<kIdx...>) {
            const auto stageTagsOf = [&]<std::size_t kPort>() {
                auto&       span         = std::get<kPort>(inputSpans);
                gr::Tag*    portTags     = res.inTags + kPort * kDeviceTagSlots;
                std::byte*  portBlobs    = inTagBlobs + kPort * kDeviceTagSlots * kDeviceTagSlotBytes;
                std::size_t nStaged      = 0UZ;
                res.inStreamIndex[kPort] = span.streamIndex; // the kernel sees the same absolute positions the host does
                for (const auto& tag : span.rawTags()) {
                    if (nStaged >= kDeviceTagSlots) {
                        res.inAcct[kPort].inputTagsTruncated = true;
                        break;
                    }
                    const std::span<const std::byte> blob = tag.map.blob();
                    if (blob.size() > kDeviceTagSlotBytes) {
                        res.inAcct[kPort].inputTagsTruncated = true;
                        continue;
                    }
                    std::byte* slot = portBlobs + nStaged * kDeviceTagSlotBytes;
                    std::memcpy(slot, blob.data(), blob.size());
                    portTags[nStaged] = gr::Tag{tag.index, gr::pmt::ValueMap::makeView(std::span<const std::byte>(slot, blob.size()))};
                    ++nStaged;
                }
                res.inTagCounts[kPort] = nStaged;
            };
            (stageTagsOf.template operator()<kIdx>(), ...);
        }(std::make_index_sequence<nInputs>{});

        parallelFor(ctx, 1UZ, [inPtrs, outPtrs, dBlock, nIn, nOut, res, statusPtr](std::size_t) { //
            *statusPtr = static_cast<std::uint32_t>(invokeBulkDeviceSpans(*dBlock, inPtrs, outPtrs, nIn, nOut, res, std::make_index_sequence<nInputs>{}, std::make_index_sequence<nOutputs>{}));
        });

        if (std::ranges::any_of(std::span<const DeviceSpanAccounting>{res.outAcct, nOutputs}, [](const DeviceSpanAccounting& a) { return a.tagPublishAttempted; })) {
            return refuseDeviceDispatch("processBulk published a tag from a device kernel; a tag payload is not device-constructible (run this block on the host, or use processBulk(ctx, ...))");
        }
        return std::nullopt;
    }

    /// replay each port's own accounting onto its real span; `blockManagedIO` takes it from there
    template<typename InputSpans, typename OutputSpans, typename TOutScratch, typename TCopyBack>
    static void replaySpanAccounting(DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, TOutScratch& outScratch, const TCopyBack& outNeedsCopyBack, //
        const DeviceSpanPortResources& res, std::size_t nIn, std::size_t nOut) {
        constexpr auto nInputs  = detail::kPortCount<InputSpans>;
        constexpr auto nOutputs = detail::kPortCount<OutputSpans>;

        [&]<std::size_t... kIdx>(std::index_sequence<kIdx...>) {
            const auto replayInput = [&]<std::size_t kPort>() {
                if (res.inAcct[kPort].inputTagsTruncated) { // recorded while staging, and until now never said out loud
                    gr::log::warning("device dispatch: input port {} had more tags than the {} staging slots (or a payload above {} B); the kernel did not see them all", kPort, kDeviceTagSlots, kDeviceTagSlotBytes);
                }
                if (res.inAcct[kPort].consumeRequested) {
                    std::ignore = std::get<kPort>(inputSpans).consume(std::min(res.inAcct[kPort].consumed, nIn));
                }
            };
            (replayInput.template operator()<kIdx>(), ...);
        }(std::make_index_sequence<nInputs>{});

        [&]<std::size_t... kIdx>(std::index_sequence<kIdx...>) {
            const auto replayOutput = [&]<std::size_t kPort>() {
                const DeviceSpanAccounting& acct       = res.outAcct[kPort];
                const std::size_t           nPublished = acct.publishRequested ? std::min(acct.published, nOut) : nOut;
                if (outNeedsCopyBack[kPort]) {
                    ctx.copyDeviceToHost(outScratch[kPort], std::get<kPort>(outputSpans).data(), nPublished);
                }
                // a single work item wrote the slots in order, so publishTag's index ordering holds
                for (std::size_t slot = 0UZ; slot < acct.tagsPublished; ++slot) {
                    const std::span<const std::byte> blob{res.tagSlots + (kPort * kDeviceTagSlots + slot) * kDeviceTagSlotBytes, kDeviceTagSlotBytes};
                    std::get<kPort>(outputSpans).publishTag(gr::pmt::ValueMap::makeView(blob), res.tagOffsets[kPort * kDeviceTagSlots + slot]);
                }
                if (acct.tagSlotsExhausted) {
                    gr::log::warning("device dispatch: more tags than the {} pre-reserved slots (or a payload above {} B); the excess was dropped", kDeviceTagSlots, kDeviceTagSlotBytes);
                }
                if (acct.publishRequested) {
                    std::get<kPort>(outputSpans).publish(nPublished);
                }
            };
            (replayOutput.template operator()<kIdx>(), ...);
        }(std::make_index_sequence<nOutputs>{});
    }

    /// kernel body only, over already-resident device pointers — one work item per declared window, else one for the span
    template<typename TInPtrs, typename TOutPtrs>
    static void runDeviceBulkCore(DeviceContext& ctx, TBlock* dBlock, TInPtrs inPtrs, TOutPtrs outPtrs, std::size_t count, WindowGeometry geometry, std::uint32_t* dStatus, bool await = true) {
        // OK never overwrites a complaint: N windows share one status word, and a plain assignment made the LAST
        // window to finish the only one heard from -- so a body that ran out of input in window 0 reported success
        const auto recordStatus = [](std::uint32_t* slot, gr::work::Status status) {
            if (slot != nullptr && status != gr::work::Status::OK) {
                *slot = static_cast<std::uint32_t>(status);
            }
        };
        if (geometry.nWindows < 2UZ) {
            parallelFor(ctx, 1UZ, [inPtrs, outPtrs, dBlock, count, dStatus, recordStatus](std::size_t) { recordStatus(dStatus, invokeBulkViews(*dBlock, inPtrs, outPtrs, count)); }, await);
            return;
        }
        const std::size_t hop = geometry.hop, inChunk = geometry.inChunk, outChunk = geometry.outChunk;
        parallelFor(
            ctx, geometry.nWindows,
            [inPtrs, outPtrs, dBlock, hop, inChunk, outChunk, dStatus, recordStatus](std::size_t w) { //
                recordStatus(dStatus, invokeBulkViewsAt(*dBlock, inPtrs, outPtrs, w * hop, inChunk, w * outChunk, outChunk));
            },
            await);
    }

    /// every span this dispatch touches is memory the host may not read, so nothing on this thread can observe the
    /// kernel's half-finished output. A port collection answers false: its span holds spans, not samples.
    template<typename InputSpans, typename OutputSpans>
    [[nodiscard]] static bool allSpansDeviceOnly(const DeviceContext& ctx, const InputSpans& inputSpans, const OutputSpans& outputSpans) {
        constexpr auto nIn  = std::tuple_size_v<std::remove_cvref_t<InputSpans>>;
        constexpr auto nOut = std::tuple_size_v<std::remove_cvref_t<OutputSpans>>;
        return [&]<std::size_t... kIn, std::size_t... kOut>(std::index_sequence<kIn...>, std::index_sequence<kOut...>) {
            const auto deviceOnly = [&](const auto& span) { return requires { ctx.isDeviceOnly(span.data()); } ? ctx.isDeviceOnly(span.data()) : false; };
            return (deviceOnly(std::get<kIn>(inputSpans)) && ...) && (deviceOnly(std::get<kOut>(outputSpans)) && ...);
        }(std::make_index_sequence<nIn>{}, std::make_index_sequence<nOut>{});
    }

    /// kernel body only, over already-resident device pointers — one work item per sample, N inputs to M outputs
    template<typename TInPtrs, typename TOutPtrs>
    static void runAutoParallelCore(DeviceContext& ctx, TBlock* dBlock, TInPtrs inPtrs, TOutPtrs outPtrs, std::size_t count, bool await = true) {
        parallelFor(
            ctx, count,
            [inPtrs, outPtrs, dBlock](std::size_t i) {
                const auto readOne = [i]<typename TIn>(const TIn& in) {
                    if constexpr (std::is_pointer_v<TIn>) {
                        return in[i];
                    } else { // a collection is one pointer per channel, and the body takes one value per channel
                        std::array<std::remove_const_t<std::remove_pointer_t<typename TIn::value_type>>, std::tuple_size_v<TIn>> perChannel{};
                        for (std::size_t channel = 0UZ; channel < perChannel.size(); ++channel) {
                            perChannel[channel] = in[channel][i];
                        }
                        return perChannel;
                    }
                };
                auto       results  = std::apply([&readOne, dBlock](const auto&... ins) { return dBlock->processOne(readOne(ins)...); }, inPtrs);
                const auto writeOne = [i]<typename TOut, typename R>(TOut& out, R&& result) {
                    if constexpr (requires { out[i] = std::forward<R>(result); }) {
                        out[i] = std::forward<R>(result);
                    } else { // a collection is one pointer per channel, and the body returned one value per channel
                        for (std::size_t channel = 0UZ; channel < out.size(); ++channel) {
                            out[channel][i] = result[channel];
                        }
                    }
                };
                if constexpr (std::tuple_size_v<TOutPtrs> == 1UZ) {
                    writeOne(std::get<0>(outPtrs), results); // a single output returns the value itself, not a one-tuple
                } else {
                    gr::meta::tuple_for_each([&writeOne]<typename R>(auto& out, R&& result) { writeOne(out, std::forward<R>(result)); }, outPtrs, results);
                }
            },
            await);
    }

    template<typename InputSpans, typename OutputSpans>
    static DispatchResult dispatchAutoParallel(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count) {
        constexpr auto nInputs  = std::tuple_size_v<std::remove_cvref_t<InputSpans>>;
        constexpr auto nOutputs = std::tuple_size_v<std::remove_cvref_t<OutputSpans>>;

        if constexpr (nInputs == 0UZ || nOutputs == 0UZ) {
            std::ignore = ctx;
            return refuseDeviceDispatch("auto-parallel needs at least one input and one output; a source or sink has no per-sample shape to parallelise");
        } else {
            DeviceBuffer dBlockBuf = deviceMirror(block, ctx);
            TBlock*      dBlock    = dBlockBuf.devicePointer<TBlock>();
            if (dBlock == nullptr) {
                return refuseDeviceDispatch("the device context cannot provide shared (host-writable) device memory for a framework-managed kernel body");
            }
            if (auto stale = staleMirrorDiagnostic(block, dBlock)) {
                if constexpr (!kOwnsDeviceShadow) {
                    ctx.deallocate(dBlockBuf);
                }
                return fail(*stale);
            }

            // residency is per port, not per block: a mixed graph must get both edges right
            std::array<DeviceBuffer, nInputs>  inScratch{};
            std::array<DeviceBuffer, nOutputs> outScratch{};
            std::array<bool, nOutputs>         outNeedsCopyBack{};
            bool                               staged = true;

            auto inPtrs  = stageInputPorts(ctx, inputSpans, inScratch, count, staged);
            auto outPtrs = stageOutputPorts(ctx, outputSpans, outScratch, outNeedsCopyBack, count, staged);

            const auto release = [&] {
                if constexpr (!kOwnsDeviceShadow) {
                    ctx.deallocate(dBlockBuf);
                }
                for (DeviceBuffer& buffer : inScratch) {
                    ctx.deallocate(buffer);
                }
                for (DeviceBuffer& buffer : outScratch) {
                    ctx.deallocate(buffer);
                }
            };
            if (!staged) {
                release();
                return fail(std::format("shared allocation failed for the auto-parallel path ({} samples over {} input and {} output ports)", count, nInputs, nOutputs));
            }

            // Deferred completion: return once the kernel is enqueued rather than once it has run. Sound only when
            // nothing this thread can read was touched -- every span device-only -- and when `release()` frees
            // nothing the kernel still owns, which is why it also needs the persistent shadow. On an in-order queue
            // the next dispatch is ordered behind this one, so a device chain pipelines with no barrier between
            // hops; the host boundary drains it. A fault then surfaces one `work()` later, hence `peek` not `poll`.
            const bool deferCompletion = kOwnsDeviceShadow && allSpansDeviceOnly(ctx, inputSpans, outputSpans);

            runAutoParallelCore(ctx, dBlock, inPtrs, outPtrs, count, false);
            if constexpr (kOwnsDeviceShadow) {
                block.deviceShadow().workInFlight = deferCompletion;
            }

            // as in the processBulk path: the poll is the queue wait that collects the launch, so it comes before
            // the copy-back reads the result and before the release frees what the kernel was reading
            const std::optional<std::string> deviceErr = deferCompletion ? ctx.peekDeviceError() : ctx.pollDeviceError();

            copyBackOutputPorts(ctx, outputSpans, outScratch, outNeedsCopyBack, outPtrs, count);

            release();
            if (deviceErr) {
                return fail(std::format("device fault during auto-parallel dispatch: {}", *deviceErr));
            }
            return gr::work::Status::OK;
        }
    }

    [[nodiscard]] static DispatchResult refuseDeviceDispatch(std::string_view reason) {
        const std::string message = std::format("device dispatch refused: {}", reason);
        gr::log::error("{}", message);
        return std::unexpected(gr::Error{message});
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP
