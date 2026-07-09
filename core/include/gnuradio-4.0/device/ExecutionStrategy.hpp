#ifndef GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP
#define GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP

#include <concepts>
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
}

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
 */
template<typename TBlock>
struct ExecutionStrategy {
    template<typename InputSpans, typename OutputSpans>
    static consteval bool canDispatch() {
        constexpr auto nInputs  = std::tuple_size_v<std::remove_cvref_t<InputSpans>>;
        constexpr auto nOutputs = std::tuple_size_v<std::remove_cvref_t<OutputSpans>>;
        return HasSyclBulkForSpans<TBlock, InputSpans, OutputSpans> || HasShaderFragment<TBlock> || (AutoParallelisable<TBlock> && std::is_trivially_copyable_v<TBlock> && nInputs == 1UZ && nOutputs == 1UZ);
    }

    template<typename InputSpans, typename OutputSpans>
    static gr::work::Status dispatch(TBlock& block, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count, std::string_view computeDomain) {
        auto& sched = SchedulerRegistry::instance().resolve(computeDomain);
        auto& ctx   = sched.context();

        if constexpr (HasSyclBulkForSpans<TBlock, InputSpans, OutputSpans>) {
            return dispatchSyclBulk(block, ctx, inputSpans, outputSpans, count);
        } else if constexpr (requires(const TBlock& b) { b.shaderFragment(); }) {
            return dispatchGlsl(block, ctx, inputSpans, outputSpans, count);
        } else {
            return dispatchAutoParallel(block, ctx, inputSpans, outputSpans, count);
        }
    }

private:
    template<typename InputSpans, typename OutputSpans>
    static gr::work::Status dispatchSyclBulk(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t /*count*/) {
#if !GR_DEVICE_HAS_SYCL_IMPL
        std::ignore = ctx;
#endif
        auto& inSpan  = std::get<0>(inputSpans);
        auto& outSpan = std::get<0>(outputSpans);

#if GR_DEVICE_HAS_SYCL_IMPL
        if (ctx.backend() == DeviceBackend::SYCL) {
            auto& syclCtx = static_cast<DeviceContextSycl&>(ctx); // backend() pre-checked; no RTTI
            if constexpr (requires { block.processBulk_sycl(*syclCtx.queue, inSpan, outSpan); }) {
                return block.processBulk_sycl(*syclCtx.queue, inSpan, outSpan);
            } else {
                gr::log::error("device dispatch: processBulk_sycl is not callable with the runtime spans");
                return gr::work::Status::ERROR;
            }
        }
#endif
        // Keep fallback visible; device dispatch must not report OK while silently changing backend.
        if constexpr (requires { block.processBulk(inSpan, outSpan); }) {
            gr::log::warning("device dispatch: no SYCL backend for the bulk path; running processBulk on the CPU");
            return block.processBulk(inSpan, outSpan);
        }
        gr::log::error("device dispatch: no SYCL backend and no CPU processBulk fallback");
        return gr::work::Status::ERROR;
    }

    template<typename InputSpans, typename OutputSpans>
    static gr::work::Status dispatchGlsl(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count) {
        auto& inSpan  = std::get<0>(inputSpans);
        auto& outSpan = std::get<0>(outputSpans);
        using InT     = std::ranges::range_value_t<std::remove_cvref_t<decltype(inSpan)>>;
        using OutT    = std::ranges::range_value_t<std::remove_cvref_t<decltype(outSpan)>>;

        if constexpr (!std::same_as<std::remove_cv_t<InT>, float> || !std::same_as<std::remove_cv_t<OutT>, float>) {
            if constexpr (requires { block.processOne(inSpan[0]); }) {
                gr::log::warning("device dispatch: GLSL shader fragments currently support float streams only; running processOne on the CPU");
                for (std::size_t i = 0; i < count; ++i) {
                    outSpan[i] = block.processOne(inSpan[i]);
                }
                return gr::work::Status::OK;
            } else {
                gr::log::error("device dispatch: GLSL shader fragments currently support float streams only");
                return gr::work::Status::ERROR;
            }
        }

        if (ctx.backend() != DeviceBackend::GLSL) {
            if constexpr (requires { block.processOne(inSpan[0]); }) {
                gr::log::warning("device dispatch: no GLSL backend; running processOne on the CPU");
                for (std::size_t i = 0; i < count; ++i) {
                    outSpan[i] = block.processOne(inSpan[i]);
                }
                return gr::work::Status::OK;
            } else {
                gr::log::error("device dispatch: no GLSL backend and no CPU processOne fallback");
                return gr::work::Status::ERROR;
            }
        }
        auto& glCtx = static_cast<DeviceContextGLSL&>(ctx); // backend() pre-checked; no RTTI

        auto frag = block.shaderFragment();
        auto glsl = generateElementWiseShader(frag, count);
        auto prog = glCtx.compileOrGetCached(glsl);
        if (!prog) {
            gr::log::error("device dispatch: GLSL shader compilation failed");
            return gr::work::Status::ERROR;
        }

        auto* dIn  = ctx.allocateDevice<float>(count);
        auto* dOut = ctx.allocateDevice<float>(count);
        if (dIn == nullptr || dOut == nullptr) {
            if (dIn != nullptr) {
                ctx.deallocate(dIn);
            }
            if (dOut != nullptr) {
                ctx.deallocate(dOut);
            }
            gr::log::error("device dispatch: GLSL device allocation failed");
            return gr::work::Status::ERROR;
        }
        ctx.copyHostToDevice(inSpan.data(), dIn, count);
        glCtx.dispatch(*prog, dIn, dOut, count, frag.workgroupSize);
        ctx.copyDeviceToHost(dOut, outSpan.data(), count);
        ctx.deallocate(dIn);
        ctx.deallocate(dOut);
        return gr::work::Status::OK;
    }

    template<typename InputSpans, typename OutputSpans>
    static gr::work::Status dispatchAutoParallel(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count) {
        constexpr auto nInputs  = std::tuple_size_v<std::remove_cvref_t<InputSpans>>;
        constexpr auto nOutputs = std::tuple_size_v<std::remove_cvref_t<OutputSpans>>;
        if constexpr (!std::is_trivially_copyable_v<TBlock> || nInputs != 1UZ || nOutputs != 1UZ) {
            std::ignore = ctx;
            gr::log::warning("device dispatch: auto-parallel path unsupported for this processOne shape; running processOne on the CPU");
            return dispatchProcessOneCpu(block, inputSpans, outputSpans, count);
        } else {
            auto& inSpan  = std::get<0>(inputSpans);
            auto& outSpan = std::get<0>(outputSpans);
            using InT     = std::ranges::range_value_t<std::remove_cvref_t<decltype(inSpan)>>;
            using OutT    = std::ranges::range_value_t<std::remove_cvref_t<decltype(outSpan)>>;

            static_assert(std::is_trivially_copyable_v<TBlock>, "auto-parallelisation requires trivially copyable blocks; use processBulk_sycl or shaderFragment instead");

            auto* dIn    = ctx.allocateShared<InT>(count);
            auto* dOut   = ctx.allocateShared<OutT>(count);
            auto* dBlock = ctx.allocateShared<TBlock>(1);
            if (dIn == nullptr || dOut == nullptr || dBlock == nullptr) {
                if (dIn != nullptr) {
                    ctx.deallocate(dIn);
                }
                if (dOut != nullptr) {
                    ctx.deallocate(dOut);
                }
                if (dBlock != nullptr) {
                    ctx.deallocate(dBlock);
                }
                gr::log::error("device dispatch: shared allocation failed for the auto-parallel path");
                return gr::work::Status::ERROR;
            }

            ctx.copyHostToDevice(inSpan.data(), dIn, count);
            std::memcpy(dBlock, &block, sizeof(TBlock));
            parallelFor(ctx, count, [dIn, dOut, dBlock](std::size_t i) { dOut[i] = dBlock->processOne(dIn[i]); });
            ctx.deallocate(dBlock);

            ctx.copyDeviceToHost(dOut, outSpan.data(), count);
            ctx.deallocate(dIn);
            ctx.deallocate(dOut);
            return gr::work::Status::OK;
        }
    }

    template<typename InputSpans, typename OutputSpans>
    static gr::work::Status dispatchProcessOneCpu(TBlock& block, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count) {
        constexpr auto nInputs  = std::tuple_size_v<std::remove_cvref_t<InputSpans>>;
        constexpr auto nOutputs = std::tuple_size_v<std::remove_cvref_t<OutputSpans>>;
        std::ignore             = inputSpans;
        std::ignore             = outputSpans;

        if constexpr (nInputs == 1UZ && nOutputs == 1UZ) {
            auto& inSpan  = std::get<0>(inputSpans);
            auto& outSpan = std::get<0>(outputSpans);
            if constexpr (requires(std::size_t i) { outSpan[i] = block.processOne(inSpan[i]); }) {
                for (std::size_t i = 0UZ; i < count; ++i) {
                    outSpan[i] = block.processOne(inSpan[i]);
                }
                return gr::work::Status::OK;
            }
        } else if constexpr (nInputs == 0UZ && nOutputs == 1UZ) {
            auto& outSpan = std::get<0>(outputSpans);
            if constexpr (requires(std::size_t i) { outSpan[i] = block.processOne(); }) {
                for (std::size_t i = 0UZ; i < count; ++i) {
                    outSpan[i] = block.processOne();
                }
                return gr::work::Status::OK;
            }
        } else if constexpr (nInputs == 1UZ && nOutputs == 0UZ) {
            auto& inSpan = std::get<0>(inputSpans);
            if constexpr (requires(std::size_t i) { block.processOne(inSpan[i]); }) {
                for (std::size_t i = 0UZ; i < count; ++i) {
                    block.processOne(inSpan[i]);
                }
                return gr::work::Status::OK;
            }
        }

        gr::log::error("device dispatch: no supported processOne CPU fallback for this span shape");
        return gr::work::Status::ERROR;
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP
