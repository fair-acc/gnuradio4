#ifndef GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP
#define GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP

#include <concepts>
#include <expected>
#include <format>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/WorkStatus.hpp>
#include <gnuradio-4.0/device/BackendCompat.hpp>
#include <gnuradio-4.0/device/BackendDetect.hpp>

#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextGLSL.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/device/SchedulerRegistry.hpp>
#include <gnuradio-4.0/device/ShaderFragment.hpp>

namespace gr::device {

namespace detail {
template<std::size_t... InIdx, std::size_t... OutIdx>
auto canProcessBulkSyclInvokeTest(SyclQueue& queue, auto& block, auto& inputSpans, auto& outputSpans, std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) -> decltype(block.processBulk_sycl(queue, std::get<InIdx>(inputSpans)..., std::get<OutIdx>(outputSpans)...));

// blocks outside the Block<T> hierarchy (trait tests) have no warn-once flag; they always warn
template<typename TBlock>
[[nodiscard]] bool firstFallbackWarning(TBlock& block) noexcept {
    if constexpr (requires {
                      { block.markDeviceFallbackWarned() } -> std::same_as<bool>;
                  }) {
        return block.markDeviceFallbackWarned();
    } else {
        return true;
    }
}
} // namespace detail

template<typename TBlock, typename InputSpans, typename OutputSpans>
concept HasSyclBulkForSpans = requires(SyclQueue& queue, TBlock& block, InputSpans& inputSpans, OutputSpans& outputSpans) {
    { detail::canProcessBulkSyclInvokeTest(queue, block, inputSpans, outputSpans, std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<InputSpans>>>(), std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<OutputSpans>>>()) } -> std::same_as<gr::work::Status>;
};

/**
 * @brief Composed device dispatch helper for Block<T>::workInternal().
 *
 * Routes to the appropriate execution path based on backend and block traits:
 * 1. HasSyclBulk + SYCL backend: call processBulk_sycl with SyclQueue.
 * 2. HasShaderFragment + GLSL backend: compile shader, dispatch via DeviceContextGLSL.
 * 3. AutoParallelisable + SYCL: mirror state, parallelFor(processOne).
 * 4. Fallback: CPU sequential loop.
 *
 * A dispatch either performs the work or returns the reason it could not. Recoverable situations —
 * a compute domain with no registered backend, a stream type a backend cannot carry — fall back to
 * the CPU and warn once per block. Everything else is an error carrying its cause.
 */
template<typename TBlock>
struct ExecutionStrategy {
    using DispatchResult = std::expected<gr::work::Status, gr::Error>;

    template<typename InputSpans, typename OutputSpans>
    static consteval bool canDispatch() {
        constexpr auto nInputs  = std::tuple_size_v<std::remove_cvref_t<InputSpans>>;
        constexpr auto nOutputs = std::tuple_size_v<std::remove_cvref_t<OutputSpans>>;
        return HasSyclBulkForSpans<TBlock, InputSpans, OutputSpans> || HasShaderFragment<TBlock> || (AutoParallelisable<TBlock> && std::is_trivially_copyable_v<TBlock> && nInputs == 1UZ && nOutputs == 1UZ);
    }

    template<typename InputSpans, typename OutputSpans>
    static DispatchResult dispatch(TBlock& block, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count, std::string_view computeDomain) {
        execution::DeviceScheduler* scheduler = SchedulerRegistry::instance().tryResolve(computeDomain);
        if (scheduler == nullptr) { // no silent CPU substitution: say so, then fall back if the block can
            return dispatchCpuFallback(block, inputSpans, outputSpans, count, std::format("compute_domain '{}' selects a device but no backend is wired", computeDomain));
        }
        DeviceContext& ctx = scheduler->context();

        if constexpr (HasSyclBulkForSpans<TBlock, InputSpans, OutputSpans>) {
            return dispatchSyclBulk(block, ctx, inputSpans, outputSpans, count);
        } else if constexpr (requires(const TBlock& b) { b.shaderFragment(); }) {
            return dispatchGlsl(block, ctx, inputSpans, outputSpans, count);
        } else {
            return dispatchAutoParallel(block, ctx, inputSpans, outputSpans, count);
        }
    }

private:
    [[nodiscard]] static DispatchResult fail(std::string message) {
        gr::log::error("device dispatch: {}", message);
        return std::unexpected(gr::Error{message});
    }

    template<typename T>
    static void deallocateIfAllocated(DeviceContext& ctx, T* ptr) {
        if (ptr != nullptr) {
            ctx.deallocate(ptr);
        }
    }

    template<typename InputSpans, typename OutputSpans>
    static DispatchResult dispatchSyclBulk(TBlock& block, [[maybe_unused]] DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count) {
#if GR_DEVICE_HAS_SYCL_IMPL
        if (ctx.backend() == DeviceBackend::SYCL) {
            auto& syclCtx = static_cast<DeviceContextSycl&>(ctx); // backend() pre-checked; no RTTI
            auto& inSpan  = std::get<0>(inputSpans);
            auto& outSpan = std::get<0>(outputSpans);
            if constexpr (requires { block.processBulk_sycl(*syclCtx.queue, inSpan, outSpan); }) {
                return block.processBulk_sycl(*syclCtx.queue, inSpan, outSpan);
            } else {
                return fail("processBulk_sycl is not callable with the runtime spans");
            }
        }
#endif
        return dispatchCpuFallback(block, inputSpans, outputSpans, count, "no SYCL backend for the bulk path");
    }

    template<typename InputSpans, typename OutputSpans>
    static DispatchResult dispatchGlsl(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count) {
        auto& inSpan  = std::get<0>(inputSpans);
        auto& outSpan = std::get<0>(outputSpans);
        using InT     = std::ranges::range_value_t<std::remove_cvref_t<decltype(inSpan)>>;
        using OutT    = std::ranges::range_value_t<std::remove_cvref_t<decltype(outSpan)>>;

        if constexpr (!std::same_as<std::remove_cv_t<InT>, float> || !std::same_as<std::remove_cv_t<OutT>, float>) {
            return dispatchCpuFallback(block, inputSpans, outputSpans, count, "GLSL shader fragments currently support float streams only");
        } else {
            if (ctx.backend() != DeviceBackend::GLSL) {
                return dispatchCpuFallback(block, inputSpans, outputSpans, count, "no GLSL backend");
            }
            auto& glCtx = static_cast<DeviceContextGLSL&>(ctx); // backend() pre-checked; no RTTI

            const ShaderFragment fragment = block.shaderFragment();
            const auto           program  = glCtx.compileOrGetCached(generateElementWiseShader(fragment, count));
            if (!program) {
                return fail(std::format("GLSL shader compilation failed: {}", program.error()));
            }

            auto* dIn  = ctx.allocateDevice<float>(count);
            auto* dOut = ctx.allocateDevice<float>(count);
            if (dIn == nullptr || dOut == nullptr) {
                deallocateIfAllocated(ctx, dIn);
                deallocateIfAllocated(ctx, dOut);
                return fail(std::format("GLSL device allocation failed for {} samples", count));
            }

            ctx.copyHostToDevice(inSpan.data(), dIn, count);
            glCtx.dispatch(*program, dIn, dOut, count, fragment.workgroupSize);
            ctx.copyDeviceToHost(dOut, outSpan.data(), count);
            ctx.deallocate(dIn);
            ctx.deallocate(dOut);
            return gr::work::Status::OK;
        }
    }

    template<typename InputSpans, typename OutputSpans>
    static DispatchResult dispatchAutoParallel(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count) {
        constexpr auto nInputs  = std::tuple_size_v<std::remove_cvref_t<InputSpans>>;
        constexpr auto nOutputs = std::tuple_size_v<std::remove_cvref_t<OutputSpans>>;

        if constexpr (!std::is_trivially_copyable_v<TBlock> || nInputs != 1UZ || nOutputs != 1UZ) {
            std::ignore = ctx;
            return dispatchCpuFallback(block, inputSpans, outputSpans, count, "auto-parallel path unsupported for this processOne shape");
        } else {
            auto& inSpan  = std::get<0>(inputSpans);
            auto& outSpan = std::get<0>(outputSpans);
            using InT     = std::ranges::range_value_t<std::remove_cvref_t<decltype(inSpan)>>;
            using OutT    = std::ranges::range_value_t<std::remove_cvref_t<decltype(outSpan)>>;

            auto* dIn    = ctx.allocateShared<InT>(count);
            auto* dOut   = ctx.allocateShared<OutT>(count);
            auto* dBlock = ctx.allocateShared<TBlock>(1);
            if (dIn == nullptr || dOut == nullptr || dBlock == nullptr) {
                deallocateIfAllocated(ctx, dIn);
                deallocateIfAllocated(ctx, dOut);
                deallocateIfAllocated(ctx, dBlock);
                return fail(std::format("shared allocation failed for the auto-parallel path ({} samples)", count));
            }

            ctx.copyHostToDevice(inSpan.data(), dIn, count);
            std::memcpy(dBlock, &block, sizeof(TBlock)); // the device copy is read-only; nothing is copied back
            parallelFor(ctx, count, [dIn, dOut, dBlock](std::size_t i) { dOut[i] = dBlock->processOne(dIn[i]); });
            ctx.deallocate(dBlock);

            ctx.copyDeviceToHost(dOut, outSpan.data(), count);
            ctx.deallocate(dIn);
            ctx.deallocate(dOut);
            return gr::work::Status::OK;
        }
    }

    template<typename InputSpans, typename OutputSpans>
    static DispatchResult dispatchCpuFallback(TBlock& block, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count, std::string_view reason) {
        auto warnOnce = [&block, reason](std::string_view path) {
            if (detail::firstFallbackWarning(block)) {
                gr::log::warning("device dispatch: {}; running {} on the CPU", reason, path);
            }
        };

        constexpr auto nInputs  = std::tuple_size_v<std::remove_cvref_t<InputSpans>>;
        constexpr auto nOutputs = std::tuple_size_v<std::remove_cvref_t<OutputSpans>>;

        if constexpr (nInputs == 1UZ && nOutputs == 1UZ) {
            auto& inSpan  = std::get<0>(inputSpans);
            auto& outSpan = std::get<0>(outputSpans);
            if constexpr (requires { block.processBulk(inSpan, outSpan); }) {
                warnOnce("processBulk");
                return block.processBulk(inSpan, outSpan);
            } else if constexpr (requires(std::size_t i) { outSpan[i] = block.processOne(inSpan[i]); }) {
                warnOnce("processOne");
                for (std::size_t i = 0UZ; i < count; ++i) {
                    outSpan[i] = block.processOne(inSpan[i]);
                }
                return gr::work::Status::OK;
            }
        } else if constexpr (nInputs == 0UZ && nOutputs == 1UZ) {
            auto& outSpan = std::get<0>(outputSpans);
            if constexpr (requires(std::size_t i) { outSpan[i] = block.processOne(); }) {
                warnOnce("processOne");
                for (std::size_t i = 0UZ; i < count; ++i) {
                    outSpan[i] = block.processOne();
                }
                return gr::work::Status::OK;
            }
        } else if constexpr (nInputs == 1UZ && nOutputs == 0UZ) {
            auto& inSpan = std::get<0>(inputSpans);
            if constexpr (requires(std::size_t i) { block.processOne(inSpan[i]); }) {
                warnOnce("processOne");
                for (std::size_t i = 0UZ; i < count; ++i) {
                    block.processOne(inSpan[i]);
                }
                return gr::work::Status::OK;
            }
        }

        return fail(std::format("{} and no CPU fallback for this span shape", reason));
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP
