#ifndef GNURADIO_BASIC_GENERATE_DEVICE_SAMPLES_HPP
#define GNURADIO_BASIC_GENERATE_DEVICE_SAMPLES_HPP

#include <algorithm>
#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::basic::detail {

enum class GenerationError { ALLOCATION, EXECUTION };

template<typename TState, typename T, typename TGenerate>
[[nodiscard]] std::expected<void, GenerationError> generateDeviceSamples(gr::device::DeviceContext& ctx, TState& state, std::span<T> output, TGenerate generate) {
    gr::device::DeviceBuffer stateStorage  = ctx.allocateShared<TState>(1UZ);
    const bool               stageOutput   = !ctx.isDeviceAccessible(output.data());
    gr::device::DeviceBuffer sampleStorage = stageOutput ? ctx.allocateShared<T>(output.size()) : gr::device::DeviceBuffer{};
    const gr::on_scope_exit  release{[&] {
        if (ctx.peekDeviceError()) {
            return; // a poisoned context is never touched again (AsyncErrorState)
        }
        ctx.deallocate(sampleStorage);
        ctx.deallocate(stateStorage);
    }};
    TState*                  deviceState = stateStorage.template devicePointer<TState>();
    T*                       samples     = stageOutput ? sampleStorage.template devicePointer<T>() : output.data();
    if (deviceState == nullptr || samples == nullptr) [[unlikely]] {
        return std::unexpected(ctx.peekDeviceError() ? GenerationError::EXECUTION : GenerationError::ALLOCATION);
    }

    std::construct_at(deviceState, state);
    const gr::on_scope_exit destroyState{[deviceState] { std::destroy_at(deviceState); }};
    const std::size_t       count = output.size();
    // one work item: every generator state is a sequential recurrence; the device path keeps the samples resident
    gr::device::parallelFor(ctx, 1UZ, [deviceState, samples, count, generate] GR_DEVICE_LAMBDA(std::size_t) {
        for (std::size_t i = 0UZ; i < count; ++i) {
            samples[i] = generate(*deviceState);
        }
    });
    if (ctx.pollDeviceError()) [[unlikely]] {
        return std::unexpected(GenerationError::EXECUTION);
    }
    if (stageOutput) {
        ctx.copy(output.data(), samples, output.size_bytes());
        if (ctx.pollDeviceError()) [[unlikely]] {
            return std::unexpected(GenerationError::EXECUTION);
        }
    }
    state = *deviceState;
    return {};
}

template<typename TState, typename T, typename TGenerate>
[[nodiscard]] [[gnu::noinline]] bool generateHostSamples(gr::device::DeviceContext& ctx, TState& state, std::span<T> output, TGenerate generate) {
    TState         stagedState(state);
    std::vector<T> samples(output.size());
    std::ranges::generate(samples, [&] { return generate(stagedState); });
    ctx.copy(output.data(), samples.data(), output.size_bytes());
    if (ctx.pollDeviceError()) [[unlikely]] {
        return false;
    }
    state = stagedState;
    return true;
}

// `state` must be device-reachable after copy construction and advances only once all of `output` is delivered;
// false: nothing was delivered (device fault, or a required device domain refused the host fallback) and the block returns ERROR
template<typename TBlock, typename TState, typename T, typename TGenerate>
[[nodiscard]] bool generateSamples(TBlock& block, gr::device::DeviceContext& ctx, TState& state, std::span<T> output, TGenerate generate) {
    if (output.empty()) {
        return true;
    }
    const auto onDevice = generateDeviceSamples(ctx, state, output, generate);
    if (onDevice) {
        return true;
    }
    return onDevice.error() == GenerationError::ALLOCATION && block.deviceFallbackIsAllowed("could not allocate generator state or samples") && generateHostSamples(ctx, state, output, generate);
}

} // namespace gr::basic::detail

#endif // GNURADIO_BASIC_GENERATE_DEVICE_SAMPLES_HPP
