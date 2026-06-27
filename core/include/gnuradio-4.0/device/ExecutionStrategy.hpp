#ifndef GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP
#define GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP

#include <string_view>

#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextGLSL.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/SchedulerRegistry.hpp>
#include <gnuradio-4.0/device/ShaderFragment.hpp>

namespace gr::device {

/**
 * @brief Composed device dispatch helper for Block<T>::workInternal().
 *
 * Routes to the appropriate execution path based on backend and block traits:
 * 1. HasSyclBulk + SYCL backend → call processBulk_sycl with native sycl::queue
 * 2. HasShaderFragment + GLSL backend → compile shader, dispatch via DeviceContextGLSL
 * 3. AutoParallelisable + SYCL → mirror state, parallelFor(processOne)
 * 4. Fallback → CPU sequential loop
 */
template<typename TBlock>
struct ExecutionStrategy {
    template<typename InputSpans, typename OutputSpans>
    static gr::work::Status dispatch(TBlock& block, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count, std::string_view computeDomain) {
        auto& sched = SchedulerRegistry::instance().resolve(computeDomain);
        auto& ctx   = sched.context();

        if constexpr (requires { &TBlock::processBulk_sycl; }) {
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
        auto& inSpan  = std::get<0>(inputSpans);
        auto& outSpan = std::get<0>(outputSpans);

#if GR_DEVICE_HAS_SYCL_IMPL
        if (auto* syclCtx = dynamic_cast<DeviceContextSycl*>(&ctx)) {
            // pass the original spans through — block handles consume/publish
            // if edges are USM-backed, .data() is device-accessible
            return block.processBulk_sycl(*syclCtx->queue, inSpan, outSpan);
        }
#endif
        // CPU fallback: call processBulk if available, else error
        if constexpr (requires { block.processBulk(inSpan, outSpan); }) {
            return block.processBulk(inSpan, outSpan);
        }
        return gr::work::Status::ERROR;
    }

    template<typename InputSpans, typename OutputSpans>
    static gr::work::Status dispatchGlsl(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count) {
        auto& inSpan  = std::get<0>(inputSpans);
        auto& outSpan = std::get<0>(outputSpans);

        auto* glCtx = dynamic_cast<DeviceContextGLSL*>(&ctx);
        if (!glCtx) {
            // fallback: CPU
            for (std::size_t i = 0; i < count; ++i) {
                outSpan[i] = block.processOne(inSpan[i]);
            }
            return gr::work::Status::OK;
        }

        auto frag = block.shaderFragment();
        auto glsl = generateElementWiseShader(frag, count);
        auto prog = glCtx->compileOrGetCached(glsl);
        if (!prog) {
            return gr::work::Status::ERROR;
        }

        auto* dIn  = ctx.allocateDevice<float>(count);
        auto* dOut = ctx.allocateDevice<float>(count);
        ctx.copyHostToDevice(inSpan.data(), dIn, count);
        glCtx->dispatch(*prog, dIn, dOut, count, frag.workgroupSize);
        ctx.copyDeviceToHost(dOut, outSpan.data(), count);
        ctx.deallocate(dIn);
        ctx.deallocate(dOut);
        return gr::work::Status::OK;
    }

    template<typename InputSpans, typename OutputSpans>
    static gr::work::Status dispatchAutoParallel(TBlock& block, DeviceContext& ctx, InputSpans& inputSpans, OutputSpans& outputSpans, std::size_t count) {
        auto& inSpan  = std::get<0>(inputSpans);
        auto& outSpan = std::get<0>(outputSpans);
        using InT     = std::ranges::range_value_t<std::remove_cvref_t<decltype(inSpan)>>;
        using OutT    = std::ranges::range_value_t<std::remove_cvref_t<decltype(outSpan)>>;

        auto* dIn  = ctx.allocateShared<InT>(count);
        auto* dOut = ctx.allocateShared<OutT>(count);
        ctx.copyHostToDevice(inSpan.data(), dIn, count);

        static_assert(std::is_trivially_copyable_v<TBlock>, "auto-parallelisation requires trivially copyable blocks; use processBulk_sycl or shaderFragment instead");
        auto* dBlock = ctx.allocateShared<TBlock>(1);
        std::memcpy(dBlock, &block, sizeof(TBlock));
#if GR_DEVICE_HAS_SYCL_IMPL
        if (auto* sycl = dynamic_cast<DeviceContextSycl*>(&ctx)) {
            sycl->parallelFor(count, [dIn, dOut, dBlock](std::size_t i) { dOut[i] = dBlock->processOne(dIn[i]); });
        } else
#endif
        {
            for (std::size_t i = 0; i < count; ++i) {
                dOut[i] = dBlock->processOne(dIn[i]);
            }
        }
        ctx.deallocate(dBlock);

        ctx.copyDeviceToHost(dOut, outSpan.data(), count);
        ctx.deallocate(dIn);
        ctx.deallocate(dOut);
        return gr::work::Status::OK;
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_EXECUTION_STRATEGY_HPP
