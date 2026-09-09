#include <boost/ut.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <memory_resource>
#include <print>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/ExecutionStrategy.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "device_test_helpers.hpp"

/*
 * One DSP chain -- source -> FIR -> magnitude -> sink -- written once and run on the host and on every
 * device this build serves. Nothing in the two blocks below is device-specific: between the runs the
 * only thing that changes is the value of `compute_domain`. See docs/USER_API_GPU_Blocks.md.
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

    [[nodiscard]] gr::work::Status processBulk_sycl(gr::device::SyclQueue& queue, InputSpanLike auto& input, OutputSpanLike auto& output) const {
        const T*          tapData = taps.data();
        const std::size_t nTaps   = taps.size();
        const T*          samples = input.data();
        T*                results = output.data();
        gr::device::parallelFor(gr::device::syclContextFor(queue), output.size(), [tapData, nTaps, samples, results](std::size_t n) {
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
    [[nodiscard]] gr::work::Status processBulk_sycl(gr::device::SyclQueue&, const std::vector<TInSpan>& ins, OutputSpanLike auto& output) const {
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

} // namespace gr::dsp::demo

// which tier each block takes follows from its signature, so it is pinned here rather than inferred from a timing
static_assert(gr::device::HasDeviceProcessBulk<gr::dsp::demo::DirectFir<float>, float, float>, "the FIR body must compile against plain spans, which is what lets the framework run its windows at once");
static_assert(!gr::AutoParallelisable<gr::dsp::demo::DirectFir<float>>, "and not per-sample, which would lose the overlap it depends on");
static_assert(gr::AutoParallelisable<gr::dsp::demo::Magnitude<float>>, "the magnitude stage is per-sample, so it gets one work item per sample");

namespace {

using namespace gr::dsp::demo;

const std::vector<float> kTaps = {1.f, -2.f, 0.5f}; // on a ramp x[n] = n this is y[n] = 1 - n/2, so the magnitude stage has work to do

/// a filter long enough for the arithmetic to matter rather than the memory traffic
[[nodiscard]] std::vector<float> rampTaps(std::size_t nTaps) {
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

/// AdaptiveCpp JITs on first use and the host device shares the cores this process is pinned to, so a single
/// timing is not a throughput; the fastest of a few is
using gr::test::servedDomains;

[[nodiscard]] double bestMegaSamplesPerSecond(auto runChain, gr::Size_t nSamples, int attempts = 3) {
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

} // namespace

int main() {
    using namespace boost::ut;

    std::ignore = gr::device::registerSyclRuntime();

    "the chain computes the filter it is supposed to compute"_test = [] {
        const ChainRun host = runChainOn("host", 256U, 4096U);
        expect(gt(host.samples.size(), 0UZ)) << "the host arm has to produce something before it can be an oracle";

        const std::size_t transient = kTaps.size() - 1UZ; // the outputs the overlap discards, so output m is y[m + K - 1]
        bool              matches   = true;
        for (std::size_t m = 0UZ; m < host.samples.size(); ++m) {
            matches = matches && std::abs(host.samples[m] - std::abs(1.f - 0.5f * static_cast<float>(m + transient))) < 1e-3f;
        }
        expect(matches) << "a ramp through {1, -2, 0.5} is 1 - n/2 in closed form, and the magnitude stage takes it positive";
    };

    "the same source runs unchanged on every served device"_test = [] {
        const ChainRun host = runChainOn("host", 256U, 4096U);

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            ChainRun          onDevice;
            const std::size_t refusals = gr::test::deviceRefusalsDuring([&] { onDevice = runChainOn(domain, 256U, 4096U); });

            expect(eq(refusals, 0UZ)) << std::format("'{}' must reach the kernel, or the comparison below proves nothing", domain);
            expect(eq(onDevice.samples.size(), host.samples.size())) << std::format("'{}' produced a different number of samples", domain);
            expect(std::ranges::equal(onDevice.samples, host.samples)) << std::format("'{}' must return exactly what the host returns from the same source", domain);
        }
    };

    "a filter that submits its own kernel returns the same samples"_test = [] {
        const ChainRun host = runChainOn<DirectFirSycl<float>>("host", 256U, 4096U);

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            ChainRun          onDevice;
            const std::size_t refusals = gr::test::deviceRefusalsDuring([&] { onDevice = runChainOn<DirectFirSycl<float>>(domain, 256U, 4096U); });

            expect(eq(refusals, 0UZ)) << std::format("'{}' must reach the hatch", domain);
            expect(std::ranges::equal(onDevice.samples, host.samples)) << std::format("'{}' must return through its own kernel exactly what the host body returns", domain);
        }
    };

    "a per-frame reduction gives the same figures on every served device"_test = [] {
        static constexpr gr::Size_t kFrame    = 64U;
        static constexpr gr::Size_t kNSamples = 4096U;

        const auto runRms = [](std::string_view domain) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            auto&     rms    = flow.emplaceBlock<FrameRms<float>>({{"gr:compute_domain", std::string(domain)}, {"input_chunk_size", kFrame}, {"output_chunk_size", gr::Size_t(1)}, {"stride", kFrame}});
            auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
            expect(flow.connect<"out", "in">(source, rms).has_value());
            expect(flow.connect<"out", "in">(rms, sink).has_value());
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            gr::test::runAbsorbingRefusal(sched);
            std::vector<float> figures(sink._samples.size());
            for (std::size_t i = 0UZ; i < figures.size(); ++i) {
                figures[i] = sink._samples[i];
            }
            return figures;
        };

        const std::vector<float> host = runRms("host");
        expect(eq(host.size(), static_cast<std::size_t>(kNSamples / kFrame))) << "one figure per frame, and no frame left behind";

        // the source ramps, so frame f covers samples [f*N, (f+1)*N) and its RMS is known in closed form
        bool matchesClosedForm = true;
        for (std::size_t f = 0UZ; f < host.size(); ++f) {
            double sumOfSquares = 0.0;
            for (std::size_t i = 0UZ; i < kFrame; ++i) {
                const double sample = static_cast<double>(f * kFrame + i);
                sumOfSquares += sample * sample;
            }
            matchesClosedForm = matchesClosedForm && std::abs(static_cast<double>(host[f]) - std::sqrt(sumOfSquares / kFrame)) < 1e-2;
        }
        expect(matchesClosedForm) << "the host arm has to be right before it can be the oracle";

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            const std::vector<float> onDevice = runRms(domain);
            expect(eq(onDevice.size(), host.size())) << std::format("'{}' produced a different number of figures", domain);
            expect(std::ranges::equal(onDevice, host)) << std::format("'{}' must reduce each frame to what the host reduces it to", domain);
        }
    };

    "a fixed number of channels reaches a device as one work item per sample"_test = [] {
        static constexpr std::size_t kChannels = 4UZ;
        static constexpr gr::Size_t  kNSamples = 1024U;
        const std::vector<float>     kGains    = {1.f, 2.f, 3.f, 4.f};

        const auto runChanneliser = [&](std::string_view domain) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            auto&     dut    = flow.emplaceBlock<Channeliser<float, kChannels>>({{"gr:compute_domain", std::string(domain)}});
            dut.gains.assign(kGains.begin(), kGains.end());
            std::vector<TagSink<float, ProcessFunction::USE_PROCESS_ONE>*> sinks;
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                sinks.push_back(&flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}}));
            }
            expect(flow.connect<"out", "in">(source, dut).has_value());
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                expect(flow.connect(dut, gr::PortDefinition{std::format("out#{}", channel)}, *sinks[channel], gr::PortDefinition{"in"}).has_value());
            }
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            gr::test::runAbsorbingRefusal(sched);

            std::vector<std::vector<float>> perChannel(kChannels);
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                perChannel[channel].resize(sinks[channel]->_samples.size());
                for (std::size_t i = 0UZ; i < perChannel[channel].size(); ++i) {
                    perChannel[channel][i] = sinks[channel]->_samples[i];
                }
            }
            return perChannel;
        };

        const auto host        = runChanneliser("host");
        bool       hostIsRight = true;
        for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
            hostIsRight = hostIsRight && host[channel].size() == static_cast<std::size_t>(kNSamples);
            for (std::size_t i = 0UZ; hostIsRight && i < host[channel].size(); ++i) {
                hostIsRight = std::abs(host[channel][i] - static_cast<float>(i) * kGains[channel]) < 1e-3f;
            }
        }
        expect(hostIsRight) << "each channel carries the source scaled by its own gain";

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            const auto onDevice = runChanneliser(domain);
            bool       matches  = true;
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                matches = matches && std::ranges::equal(onDevice[channel], host[channel]);
            }
            expect(matches) << std::format("'{}' must fill every channel with what the host fills it with", domain);
        }
    };

    "a device-resident Channeliser output elides its per-channel scratch"_test = [] {
        const auto gpuDomain = gr::test::firstServedDomain({"gpu:sycl"});
        if (!gpuDomain) {
            return; // the elision only differs from the copy-back path once memory can genuinely be device-resident
        }
        static constexpr std::size_t kChannels = 4UZ;
        static constexpr gr::Size_t  kNSamples = 1024U;
        const std::vector<float>     kGains    = {1.f, 2.f, 3.f, 4.f};

        // identical to `runChanneliser` above except for one thing: the domain given to the four out#c -> in
        // edges. Left default, they cross to the host on a boundary edge (HostOnly access, not device-accessible).
        // Set to a shared GPU domain, ExecutionStrategy::stageOutputPort finds every channel already
        // device-accessible and returns the edges' own pointers instead of allocating a channel-major scratch block.
        const auto runChanneliser = [&](gr::EdgeParameters outputEdge) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            auto&     dut    = flow.emplaceBlock<Channeliser<float, kChannels>>({{"gr:compute_domain", std::string(*gpuDomain)}});
            dut.gains.assign(kGains.begin(), kGains.end());
            std::vector<TagSink<float, ProcessFunction::USE_PROCESS_ONE>*> sinks;
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                sinks.push_back(&flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}}));
            }
            expect(flow.connect<"out", "in">(source, dut).has_value());
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                expect(flow.connect(dut, gr::PortDefinition{std::format("out#{}", channel)}, *sinks[channel], gr::PortDefinition{"in"}, outputEdge).has_value());
            }
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            gr::test::runAbsorbingRefusal(sched);

            std::vector<std::vector<float>> perChannel(kChannels);
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                perChannel[channel].assign(sinks[channel]->_samples.begin(), sinks[channel]->_samples.end());
            }
            return perChannel;
        };

        const auto resident = runChanneliser(gr::EdgeParameters{.domain = gr::ComputeDomain::gpu_shared()});
        const auto hostSide = runChanneliser(gr::EdgeParameters{});

        // the resident arm takes the elided branch, so a channel pointed at the wrong place shows up here as
        // wrong numbers. that the allocation is genuinely skipped is a throughput claim, and bm_DeviceDispatch
        // is where it is measured.
        expect(!resident.empty() && !resident[0].empty()) << "the resident arm must produce samples for the comparison to mean anything";
        bool matches = true;
        for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
            matches = matches && std::ranges::equal(resident[channel], hostSide[channel]);
        }
        expect(matches) << "the elided and copy-back paths must fill every channel with the same values";
    };

    "a combiner gathers its input channels the same way on every served device"_test = [] {
        static constexpr std::size_t kChannels = 4UZ;
        static constexpr gr::Size_t  kNSamples = 4096U;
        const std::vector<float>     kWeights  = {1.f, 2.f, 3.f, 4.f};

        const auto runCombiner = [&](std::string_view domain) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     split = flow.emplaceBlock<Channeliser<float, kChannels>>();
            split.gains.assign(kWeights.begin(), kWeights.end());
            auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            auto& dut    = flow.emplaceBlock<Combiner<float, kChannels>>({{"gr:compute_domain", std::string(domain)}});
            dut.weights.assign(kWeights.begin(), kWeights.end());
            auto& sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

            expect(flow.connect<"out", "in">(source, split).has_value());
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                expect(flow.connect(split, gr::PortDefinition{std::format("out#{}", channel)}, dut, gr::PortDefinition{std::format("in#{}", channel)}).has_value());
            }
            expect(flow.connect<"out", "in">(dut, sink).has_value());

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            gr::test::runAbsorbingRefusal(sched);
            return std::vector<float>(sink._samples.begin(), sink._samples.end());
        };

        const std::vector<float> host = runCombiner("host");
        expect(!host.empty()) << "the host arm must produce samples for the device arms to be compared against";

        for (std::string_view domain : gr::test::servedDomains()) {
            const std::vector<float> onDevice = runCombiner(domain);
            expect(eq(onDevice.size(), host.size())) << std::format("domain '{}' produced a different number of samples", domain);
            bool matches = onDevice.size() == host.size();
            for (std::size_t i = 0UZ; matches && i < host.size(); ++i) {
                matches = std::abs(onDevice[i] - host[i]) <= 1e-3f * std::max(1.f, std::abs(host[i]));
            }
            expect(matches) << std::format("domain '{}' must gather the same channels into the same sums as the host", domain);
        }
    };

    "a run-time channel count is refused by name, not silently run on the host"_test = [] {
        const auto deviceDomain = gr::test::firstServedDomain({"gpu:sycl", "host:sycl"});
        if (!deviceDomain) {
            return; // nothing serves a device here, so there is no gate to exercise
        }
        using namespace gr::testing;
        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     dut  = flow.emplaceBlock<DynamicCombiner<float>>({{"gr:compute_domain", std::string(*deviceDomain) + "!"}, {"n_inputs", gr::Size_t(2)}});
        auto&     sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});
        for (std::size_t channel = 0UZ; channel < 2UZ; ++channel) {
            auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(64)}, {"mark_tag", false}});
            expect(flow.connect(source, gr::PortDefinition{"out"}, dut, gr::PortDefinition{std::format("in#{}", channel)}).has_value());
        }
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        gr::test::runAbsorbingRefusal(sched);

        expect(sched.state() == gr::lifecycle::State::ERROR) << "a run-time channel count must stop the run, not be quietly demoted to the host";
        expect(sink._samples.empty()) << "a refused block must not have produced anything";
    };

    "without the marker the same block warns once and runs on the host"_test = [] {
        const auto deviceDomain = gr::test::firstServedDomain({"gpu:sycl", "host:sycl"});
        if (!deviceDomain) {
            return; // nothing serves a device here, so there is no gate to exercise
        }
        using namespace gr::testing;
        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     dut  = flow.emplaceBlock<DynamicCombiner<float>>({{"gr:compute_domain", std::string(*deviceDomain)}, {"n_inputs", gr::Size_t(2)}});
        auto&     sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
        for (std::size_t channel = 0UZ; channel < 2UZ; ++channel) {
            auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(64)}, {"mark_tag", false}});
            expect(flow.connect(source, gr::PortDefinition{"out"}, dut, gr::PortDefinition{std::format("in#{}", channel)}).has_value());
        }
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        gr::test::runAbsorbingRefusal(sched);

        expect(sched.state() != gr::lifecycle::State::ERROR) << "without '!' an unreachable domain is a warning, not a stop";
        expect(!sink._samples.empty()) << "the block must have run on the host";
    };

    "a run-time channel count with a hatch is not caught by that refusal"_test = [] {
        const auto deviceDomain = gr::test::firstServedDomain({"gpu:sycl", "host:sycl"});
        if (!deviceDomain) {
            return; // nothing serves a device here, so there is no gate to exercise
        }
        using namespace gr::testing;
        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     dut  = flow.emplaceBlock<DynamicCombinerWithHatch<float>>({{"gr:compute_domain", std::string(*deviceDomain) + "!"}, {"n_inputs", gr::Size_t(2)}});
        auto&     sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
        for (std::size_t channel = 0UZ; channel < 2UZ; ++channel) {
            auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(64)}, {"mark_tag", false}});
            expect(flow.connect(source, gr::PortDefinition{"out"}, dut, gr::PortDefinition{std::format("in#{}", channel)}).has_value());
        }
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        gr::test::runAbsorbingRefusal(sched);

        // the refusal names the hatch as the way out, so a block that already owns one must not be caught by it
        expect(sched.state() != gr::lifecycle::State::ERROR) << "a block owning a processBulk_sycl hatch must survive the run-time-channel-count gate";
        expect(!sink._samples.empty()) << "the hatch must have run and produced samples";
    };

    "a correlator gives every lag the same value on every served device"_test = [] {
        static constexpr gr::Size_t kLength   = 32U; // reference length
        static constexpr gr::Size_t kLags     = 64U; // lags computed per window
        static constexpr gr::Size_t kNSamples = 4096U;

        const std::vector<float> reference = [] {
            std::vector<float> pattern(kLength);
            for (std::size_t k = 0UZ; k < kLength; ++k) {
                pattern[k] = std::sin(0.4f * static_cast<float>(k));
            }
            return pattern;
        }();

        const auto runCorrelator = [&](std::string_view domain) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            auto&     dut    = flow.emplaceBlock<Correlator<float>>({{"gr:compute_domain", std::string(domain)}, //
                       {"input_chunk_size", kLags + kLength - 1U}, {"output_chunk_size", kLags}, {"stride", kLags}});
            auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
            dut.reference.assign(reference.begin(), reference.end());
            expect(flow.connect<"out", "in">(source, dut).has_value());
            expect(flow.connect<"out", "in">(dut, sink).has_value());
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            gr::test::runAbsorbingRefusal(sched);
            std::vector<float> lags(sink._samples.size());
            for (std::size_t i = 0UZ; i < lags.size(); ++i) {
                lags[i] = sink._samples[i];
            }
            return lags;
        };

        const std::vector<float> host = runCorrelator("host");
        expect(gt(host.size(), 0UZ));

        // the source ramps, so lag l of the first window correlates the reference against samples [l, l+K)
        bool matchesDefinition = true;
        for (std::size_t lag = 0UZ; lag < std::min(host.size(), std::size_t{16}); ++lag) {
            double expected = 0.0;
            for (std::size_t k = 0UZ; k < kLength; ++k) {
                expected += static_cast<double>(reference[k]) * static_cast<double>(lag + k);
            }
            matchesDefinition = matchesDefinition && std::abs(static_cast<double>(host[lag]) - expected) < 1e-2;
        }
        expect(matchesDefinition) << "the host arm must compute the correlation before it can be the oracle";

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            const std::vector<float> onDevice = runCorrelator(domain);
            expect(eq(onDevice.size(), host.size())) << std::format("'{}' produced a different number of lags", domain);
            // a correlation sums thirty-two products; the device may reassociate them, so the tolerance is the
            // arithmetic's, not the algorithm's -- unlike the pointwise chains above, which really are bit-equal
            // a device may contract a*b+c into one rounding, so it is not bit-equal to the host and is in fact
            // closer to the exact value; both are therefore measured against a double-precision reference
            double worst = 0.0;
            for (std::size_t lag = 0UZ; lag < std::min<std::size_t>(256UZ, host.size()); ++lag) {
                double exact = 0.0;
                for (std::size_t k = 0UZ; k < kLength; ++k) {
                    exact += static_cast<double>(reference[k]) * static_cast<double>(lag + k);
                }
                worst = std::max(worst, std::abs(static_cast<double>(onDevice[lag]) - exact) / std::max(1.0, std::abs(exact)));
            }
            expect(lt(worst, 1e-5)) << std::format("'{}' departs from the exact correlation by {} relative", domain, worst);
        }
    };

    "cascade throughput against the frame the chain is dispatched in"_test = [] {
        constexpr gr::Size_t kNSamples = 1U << 20;

        const auto row = [](std::string_view label, auto runFrame) {
            std::vector<double> throughput;
            for (gr::Size_t frame : {256U, 4096U, 65536U}) {
                throughput.push_back(bestMegaSamplesPerSecond([&](gr::Size_t samples) { return runFrame(frame, samples); }, kNSamples));
                expect(gt(throughput.back(), 0.0)) << std::format("'{}' at frame {} produced nothing", label, frame);
            }
            std::println("  {:<24} {:>10.2f} {:>10.2f} {:>10.2f}", label, throughput[0], throughput[1], throughput[2]);
        };

        std::println("\n  cascade: source -> FIR(3 taps) -> Magnitude -> sink, {} samples", kNSamples);
        std::println("  {:<24} {:>10} {:>10} {:>10}", "domain / filter", "frame 256", "frame 4k", "frame 64k");
        for (std::string_view domain : servedDomains()) {
            row(std::format("{}, whole span", domain), [domain](gr::Size_t frame, gr::Size_t samples) { return runChainOn(domain, frame, samples); });
        }
        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            row(std::format("{}, own kernel", domain), [domain](gr::Size_t frame, gr::Size_t samples) { return runChainOn<DirectFirSycl<float>>(domain, frame, samples); });
        }
        std::println("  (MSample/s; a declared window is run per work item, so parallelism is nOut/output_chunk_size --");
        std::println("   the larger the frame, the fewer windows a span holds and the less there is to spread)\n");
    };

    "throughput of each spike block, on every served domain"_test = [] {
        static constexpr gr::Size_t kNSamples = 1U << 20;
        static constexpr gr::Size_t kFrame    = 1024U;

        // the rate quoted is the INPUT rate: a reduction produces one output per frame, so counting its outputs
        // would make the fastest block look like the slowest
        const auto timeGraph = [](auto&& build) {
            const auto started  = std::chrono::steady_clock::now();
            const auto produced = build();
            const auto elapsed  = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started);
            return (elapsed.count() == 0 || produced == 0UZ) ? 0.0 : static_cast<double>(kNSamples) / static_cast<double>(elapsed.count());
        };

        const auto runOne = [](std::string_view domain, auto&& emplaceDut, gr::property_map dutSettings, auto&& configureDut) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            dutSettings.insert_or_assign("gr:compute_domain", std::string(domain));
            auto& dut  = emplaceDut(flow, std::move(dutSettings));
            auto& sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});
            configureDut(dut);
            std::ignore = flow.connect<"out", "in">(source, dut);
            std::ignore = flow.connect<"out", "in">(dut, sink);
            gr::scheduler::Simple<> sched;
            std::ignore = sched.exchange(std::move(flow));
            gr::test::runAbsorbingRefusal(sched);
            return static_cast<std::size_t>(sink._nSamplesProduced);
        };

        const std::vector<float> taps      = {1.f, -2.f, 0.5f};
        const std::vector<float> reference = std::vector<float>(32UZ, 0.25f);

        std::println("\n  spike blocks, {} samples, best of three after a warm-up", kNSamples);
        std::println("  {:<26} {:>12} {:>12} {:>12}", "block", "host", "host:sycl", "gpu:sycl");

        const auto row = [&](std::string_view label, auto&& runFor) {
            std::vector<double> throughput;
            for (std::string_view domain : {"host", "host:sycl", "gpu:sycl"}) {
                if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr && domain != "host") {
                    throughput.push_back(0.0);
                    continue;
                }
                std::ignore = runFor(domain); // warm the JIT
                double best = 0.0;
                for (int attempt = 0; attempt < 3; ++attempt) {
                    best = std::max(best, timeGraph([&] { return runFor(domain); }));
                }
                throughput.push_back(best);
            }
            std::println("  {:<26} {:>12.2f} {:>12.2f} {:>12.2f}", label, throughput[0], throughput[1], throughput[2]);
        };

        row("DirectFir (3 taps)", [&](std::string_view domain) { return runOne(domain, [](gr::Graph& f, gr::property_map m) -> auto& { return f.emplaceBlock<DirectFir<float>>(std::move(m)); }, {{"input_chunk_size", kFrame + 2U}, {"output_chunk_size", kFrame}, {"stride", kFrame}}, [&](auto& dut) { dut.taps.assign(taps.begin(), taps.end()); }); });
        row("FrameRms (frame 1024)", [&](std::string_view domain) { return runOne(domain, [](gr::Graph& f, gr::property_map m) -> auto& { return f.emplaceBlock<FrameRms<float>>(std::move(m)); }, {{"input_chunk_size", kFrame}, {"output_chunk_size", gr::Size_t(1)}, {"stride", kFrame}}, [](auto&) {}); });
        row("Correlator (32 lags)", [&](std::string_view domain) { return runOne(domain, [](gr::Graph& f, gr::property_map m) -> auto& { return f.emplaceBlock<Correlator<float>>(std::move(m)); }, {{"input_chunk_size", kFrame + 31U}, {"output_chunk_size", kFrame}, {"stride", kFrame}}, [&](auto& dut) { dut.reference.assign(reference.begin(), reference.end()); }); });
        row("Magnitude (per sample)", [&](std::string_view domain) { return runOne(domain, [](gr::Graph& f, gr::property_map m) -> auto& { return f.emplaceBlock<Magnitude<float>>(std::move(m)); }, {}, [](auto&) {}); });
        std::println("  (input MSample/s; 0.00 means the domain is not served, or the block produced nothing)\n");
    };

    "cascade throughput against filter length, where the arithmetic starts to matter"_test = [] {
        std::println("  same chain against filter length, each length given the window it deserves");
        std::print("  {:<10}", "taps");
        for (std::string_view domain : servedDomains()) {
            std::print(" {:>12}", domain);
        }
        std::println("");

        for (std::size_t nTaps : gr::test::kFilterLengths) {
            const std::vector<float> taps    = rampTaps(nTaps);
            const auto               window  = static_cast<gr::Size_t>(gr::test::windowForFilterLength(nTaps));
            const auto               samples = static_cast<gr::Size_t>(gr::test::samplesForDirectFilter(nTaps));
            std::print("  {:<10}", nTaps);
            for (std::string_view domain : servedDomains()) {
                const auto   runChain   = [&](gr::Size_t n) { return domain == "host" ? runChainOn(domain, window, n, taps) : runChainOn<DirectFirSycl<float>>(domain, window, n, taps); };
                const double throughput = bestMegaSamplesPerSecond(runChain, samples, gr::test::timingAttemptsForFilterLength(nTaps));
                expect(gt(throughput, 0.0)) << std::format("'{}' at {} taps produced nothing", domain, nTaps);
                std::print(" {:>12.2f}", throughput);
            }
            std::println("");
        }
        std::println("  (MSample/s; the host arm runs the whole span, the device arms own their kernel)\n");
    };
}
