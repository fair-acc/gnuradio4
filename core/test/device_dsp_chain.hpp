#ifndef GNURADIO_DEVICE_DSP_CHAIN_HPP
#define GNURADIO_DEVICE_DSP_CHAIN_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory_resource>
#include <string_view>
#include <vector>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/ExecutionStrategy.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "device_test_helpers.hpp"

/*
 * One DSP chain -- source -> FIR -> magnitude -> sink -- written once and run on the host and on every
 * device this build serves. Nothing in the blocks below is device-specific: between the runs the only
 * thing that changes is the value of `compute_domain`. See docs/USER_API_GPU_Blocks.md.
 *
 * Shared so that `qa_DeviceDspChain` can assert what the chain computes while `bm_DeviceDspChain`
 * measures how fast it computes it, without either owning the other's concern.
 */

namespace gr::dsp::demo {

/// direct-form FIR that keeps no delay line: the overlap `stride < input_chunk_size` leaves in the edge is the history
template<typename T>
struct DirectFir : Block<DirectFir<T>, Resampling<>, Stride<>> {
    PortIn<T>  in;
    PortOut<T> out;

    std::pmr::vector<T> taps{};

    GR_MAKE_REFLECTABLE(DirectFir, in, out, taps);

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) const noexcept {
        const std::size_t nTaps = taps.size();
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            T accumulator{};
            for (std::size_t k = 0UZ; k < nTaps; ++k) {
                accumulator += taps[k] * input[n + nTaps - 1UZ - k];
            }
            output[n] = accumulator;
        }
        return gr::work::Status::OK;
    }
};

/// the same filter, submitting its own kernel: one work item per output sample instead of one for the whole span
template<typename T>
struct DirectFirSycl : Block<DirectFirSycl<T>, Resampling<>, Stride<>> {
    PortIn<T>  in;
    PortOut<T> out;

    std::pmr::vector<T> taps{};

    GR_MAKE_REFLECTABLE(DirectFirSycl, in, out, taps);

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) const noexcept {
        const std::size_t nTaps = taps.size();
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            T accumulator{};
            for (std::size_t k = 0UZ; k < nTaps; ++k) {
                accumulator += taps[k] * input[n + nTaps - 1UZ - k];
            }
            output[n] = accumulator;
        }
        return gr::work::Status::OK;
    }

    [[nodiscard]] gr::work::Status processBulkDevice(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) const {
        const T*          tapData = taps.data();
        const std::size_t nTaps   = taps.size();
        const T*          samples = input.data();
        T*                results = output.data();
        gr::device::parallelFor(ctx, output.size(), [tapData, nTaps, samples, results](std::size_t n) {
            T accumulator{};
            for (std::size_t k = 0UZ; k < nTaps; ++k) {
                accumulator += tapData[k] * samples[n + nTaps - 1UZ - k];
            }
            results[n] = accumulator;
        });
        return gr::work::Status::OK;
    }
};

/// one figure per frame -- the segmented-reduction shape (RMS, peak, AGC gain) declared as a window
template<typename T>
struct FrameRms : Block<FrameRms<T>, Resampling<>, Stride<>> {
    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(FrameRms, in, out);

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) const noexcept {
        const std::size_t frame = static_cast<std::size_t>(this->input_chunk_size);
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            T sumOfSquares{};
            for (std::size_t i = 0UZ; i < frame; ++i) {
                const T sample = input[n * frame + i];
                sumOfSquares += sample * sample;
            }
            output[n] = static_cast<T>(std::sqrt(static_cast<double>(sumOfSquares) / static_cast<double>(frame)));
        }
        return gr::work::Status::OK;
    }
};

/// the same run-time channel count, but owning a hatch: the gate must let this one through
template<typename T>
struct DynamicCombinerWithHatch : Block<DynamicCombinerWithHatch<T>> {
    std::vector<PortIn<T>> in;
    PortOut<T>             out;

    Annotated<gr::Size_t, "n_inputs", Doc<"number of input channels">> n_inputs = 2U;

    GR_MAKE_REFLECTABLE(DynamicCombinerWithHatch, in, out, n_inputs);

    void settingsChanged(const gr::property_map& /*old*/, const gr::property_map& newSettings) {
        if (newSettings.contains("n_inputs")) {
            in.resize(n_inputs);
        }
    }

    template<gr::InputSpanLike TInSpan>
    [[nodiscard]] gr::work::Status processBulk(const std::span<TInSpan>& ins, OutputSpanLike auto& output) const {
        for (std::size_t i = 0UZ; i < output.size(); ++i) {
            T sum{};
            for (const TInSpan& channel : ins) {
                sum += channel[i];
            }
            output[i] = sum;
        }
        return gr::work::Status::OK;
    }

    template<gr::InputSpanLike TInSpan>
    [[nodiscard]] gr::work::Status processBulkDevice(gr::device::DeviceContext&, const std::vector<TInSpan>& ins, OutputSpanLike auto& output) const {
        for (std::size_t i = 0UZ; i < output.size(); ++i) {
            T sum{};
            for (const TInSpan& channel : ins) {
                sum += channel[i];
            }
            output[i] = sum;
        }
        return gr::work::Status::OK;
    }
};

/// a channel count only known at run time: the device gate must refuse this by name
template<typename T>
struct DynamicCombiner : Block<DynamicCombiner<T>> {
    std::vector<PortIn<T>> in;
    PortOut<T>             out;

    Annotated<gr::Size_t, "n_inputs", Doc<"number of input channels">> n_inputs = 2U;

    GR_MAKE_REFLECTABLE(DynamicCombiner, in, out, n_inputs);

    void settingsChanged(const gr::property_map& /*old*/, const gr::property_map& newSettings) {
        if (newSettings.contains("n_inputs")) {
            in.resize(n_inputs);
        }
    }

    [[nodiscard]] constexpr T processOne(std::vector<T> perChannel) const noexcept {
        T sum{};
        for (const T& value : perChannel) {
            sum += value;
        }
        return sum;
    }
};

static_assert(!gr::traits::block::stream_input_ports<DynamicCombiner<float>>::template none_of<gr::traits::port::is_dynamic_port_collection>, "PROBE: DynamicCombiner in-ports are not seen as a dynamic collection");

/// gathers a fixed set of input channels into one stream -- the port-collection shape on the way in
template<typename T, std::size_t nChannels>
struct Combiner : Block<Combiner<T, nChannels>> {
    std::array<PortIn<T>, nChannels> in;
    PortOut<T>                       out;

    std::pmr::vector<T> weights{};

    GR_MAKE_REFLECTABLE(Combiner, in, out, weights);

    [[nodiscard]] constexpr T processOne(std::array<T, nChannels> perChannel) const noexcept {
        T sum{};
        for (std::size_t channel = 0UZ; channel < nChannels; ++channel) {
            sum += channel < weights.size() ? perChannel[channel] * weights[channel] : perChannel[channel];
        }
        return sum;
    }
};

/// splits one stream into a fixed number of channels -- the port-collection shape, one work item per sample
template<typename T, std::size_t nChannels>
struct Channeliser : Block<Channeliser<T, nChannels>> {
    PortIn<T>                         in;
    std::array<PortOut<T>, nChannels> out;

    std::pmr::vector<T> gains{};

    GR_MAKE_REFLECTABLE(Channeliser, in, out, gains);

    [[nodiscard]] constexpr std::array<T, nChannels> processOne(T x) const noexcept {
        std::array<T, nChannels> perChannel{};
        for (std::size_t channel = 0UZ; channel < nChannels; ++channel) {
            perChannel[channel] = channel < gains.size() ? x * gains[channel] : x;
        }
        return perChannel;
    }
};

/// cross-correlation against a stored reference: one lag per output sample, which is one work item per lag
template<typename T>
struct Correlator : Block<Correlator<T>, Resampling<>, Stride<>> {
    PortIn<T>  in;
    PortOut<T> out;

    std::pmr::vector<T> reference{};

    GR_MAKE_REFLECTABLE(Correlator, in, out, reference);

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) const noexcept {
        const std::size_t length = reference.size();
        for (std::size_t lag = 0UZ; lag < output.size(); ++lag) {
            T sum{};
            for (std::size_t k = 0UZ; k < length; ++k) {
                sum += reference[k] * input[lag + k];
            }
            output[lag] = sum;
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
struct Magnitude : Block<Magnitude<T>> {
    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(Magnitude, in, out);

    [[nodiscard]] constexpr T processOne(T x) const noexcept { return x < T{} ? -x : x; }
};

// which tier each block takes follows from its signature, so it is pinned here rather than inferred from a timing
static_assert(gr::device::HasDeviceProcessBulk<gr::dsp::demo::DirectFir<float>, float, float>, "the FIR body must compile against plain spans, which is what lets the framework run its windows at once");
static_assert(!gr::AutoParallelisable<gr::dsp::demo::DirectFir<float>>, "and not per-sample, which would lose the overlap it depends on");
static_assert(gr::AutoParallelisable<gr::dsp::demo::Magnitude<float>>, "the magnitude stage is per-sample, so it gets one work item per sample");

inline const std::vector<float> kTaps = {1.f, -2.f, 0.5f}; // on a ramp x[n] = n this is y[n] = 1 - n/2, so the magnitude stage has work to do

/// a filter long enough for the arithmetic to matter rather than the memory traffic
[[nodiscard]] inline std::vector<float> rampTaps(std::size_t nTaps) {
    std::vector<float> taps(nTaps);
    for (std::size_t k = 0UZ; k < nTaps; ++k) {
        taps[k] = (k % 2UZ == 0UZ ? 1.f : -1.f) / static_cast<float>(k + 1UZ);
    }
    return taps;
}

constexpr gr::Size_t kWarmUpSamples = 1U << 16;

struct ChainRun {
    std::vector<float>        samples;
    std::chrono::microseconds elapsed{0};
};

template<typename TFir = DirectFir<float>>
[[nodiscard]] ChainRun runChainOn(std::string_view domain, gr::Size_t frame, gr::Size_t nSamples, const std::vector<float>& taps = kTaps) {
    using namespace gr::testing;
    const auto nTaps = static_cast<gr::Size_t>(taps.size());

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});

    auto& source    = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto& fir       = flow.emplaceBlock<TFir>({{"gr:compute_domain", std::string(domain)}, //
              {"input_chunk_size", frame + nTaps - 1U}, {"output_chunk_size", frame}, {"stride", frame}});
    auto& magnitude = flow.emplaceBlock<Magnitude<float>>({{"gr:compute_domain", std::string(domain)}});
    auto& sink      = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    fir.taps.assign(taps.begin(), taps.end());

    boost::ut::expect(flow.connect<"out", "in">(source, fir).has_value());
    boost::ut::expect(flow.connect<"out", "in">(fir, magnitude).has_value());
    boost::ut::expect(flow.connect<"out", "in">(magnitude, sink).has_value());

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());

    const auto started = std::chrono::steady_clock::now();
    gr::test::runAbsorbingRefusal(sched);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started);

    ChainRun result{.samples = std::vector<float>(sink._samples.size()), .elapsed = elapsed};
    for (std::size_t i = 0UZ; i < result.samples.size(); ++i) {
        result.samples[i] = sink._samples[i];
    }
    return result;
}

/// Three device stages, so the MIDDLE one has device memory on both sides. Every other chain in this file is two
/// stages, which means both of its device blocks sit on the host boundary and neither can ever defer -- the
/// view-form body's deferred path had no coverage at all until this chain existed.
[[nodiscard]] inline ChainRun runTwoFirChainOn(std::string_view domain, gr::Size_t frame, gr::Size_t nSamples, const std::vector<float>& taps = kTaps) {
    using namespace gr::testing;
    const auto nTaps = static_cast<gr::Size_t>(taps.size());

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});

    const gr::property_map firSettings{{"gr:compute_domain", std::string(domain)}, //
        {"input_chunk_size", frame + nTaps - 1U}, {"output_chunk_size", frame}, {"stride", frame}};

    auto& source    = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto& firFirst  = flow.emplaceBlock<DirectFir<float>>(firSettings);
    auto& firSecond = flow.emplaceBlock<DirectFir<float>>(firSettings); // interior: device memory in AND out
    auto& magnitude = flow.emplaceBlock<Magnitude<float>>({{"gr:compute_domain", std::string(domain)}});
    auto& sink      = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    firFirst.taps.assign(taps.begin(), taps.end());
    firSecond.taps.assign(taps.begin(), taps.end());

    boost::ut::expect(flow.connect<"out", "in">(source, firFirst).has_value());
    boost::ut::expect(flow.connect<"out", "in">(firFirst, firSecond).has_value());
    boost::ut::expect(flow.connect<"out", "in">(firSecond, magnitude).has_value());
    boost::ut::expect(flow.connect<"out", "in">(magnitude, sink).has_value());

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());

    const auto started = std::chrono::steady_clock::now();
    gr::test::runAbsorbingRefusal(sched);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started);

    ChainRun result{.samples = std::vector<float>(sink._samples.size()), .elapsed = elapsed};
    for (std::size_t i = 0UZ; i < result.samples.size(); ++i) {
        result.samples[i] = sink._samples[i];
    }
    return result;
}

/// AdaptiveCpp JITs on first use and the host device shares the cores this process is pinned to, so a single
/// timing is not a throughput; the fastest of a few is

[[nodiscard]] inline double bestMegaSamplesPerSecond(auto runChain, gr::Size_t nSamples, int attempts = 3) {
    std::ignore = runChain(kWarmUpSamples);
    double best = 0.0;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        const ChainRun run = runChain(nSamples);
        if (run.elapsed.count() > 0) {
            best = std::max(best, static_cast<double>(run.samples.size()) / static_cast<double>(run.elapsed.count()));
        }
    }
    return best;
}

} // namespace gr::dsp::demo

static_assert(gr::device::HasDeviceProcessBulk<gr::dsp::demo::DirectFir<float>, float, float>, "the FIR body must compile against plain spans, which is what lets the framework run its windows at once");
static_assert(!gr::AutoParallelisable<gr::dsp::demo::DirectFir<float>>, "and not per-sample, which would lose the overlap it depends on");
static_assert(gr::AutoParallelisable<gr::dsp::demo::Magnitude<float>>, "the magnitude stage is per-sample, so it gets one work item per sample");

#endif // GNURADIO_DEVICE_DSP_CHAIN_HPP
