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

    gr::Graph flow;
    flow.autoSizeEdgesToChunks = true; // the largest frame exceeds the default edge, and one frame of ring would let no stage overlap

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
[[nodiscard]] double bestMegaSamplesPerSecond(auto runChain, gr::Size_t nSamples) {
    std::ignore = runChain(kWarmUpSamples);
    double best = 0.0;
    for (int attempt = 0; attempt < 3; ++attempt) {
        const ChainRun run = runChain(nSamples);
        if (run.elapsed.count() > 0) {
            best = std::max(best, static_cast<double>(run.samples.size()) / static_cast<double>(run.elapsed.count()));
        }
    }
    return best;
}

[[nodiscard]] std::vector<std::string_view> servedDomains() {
    std::vector<std::string_view> domains{"host"};
    for (std::string_view candidate : {"host:sycl", "gpu:sycl"}) {
        if (gr::device::DeviceContextRegistry::instance().tryResolve(candidate) != nullptr) {
            domains.push_back(candidate);
        }
    }
    return domains;
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

    "cascade throughput against filter length, where the arithmetic starts to matter"_test = [] {
        constexpr gr::Size_t kNSamples = 1U << 20;
        constexpr gr::Size_t kFrame    = 65536U;

        std::println("  same chain at frame {}, kernel-owning filter, against filter length", kFrame);
        std::println("  {:<24} {:>10} {:>10} {:>10}", "domain / filter", "3 taps", "32 taps", "128 taps");
        for (std::string_view domain : servedDomains()) {
            std::vector<double> throughput;
            for (std::size_t nTaps : {3UZ, 32UZ, 128UZ}) {
                const std::vector<float> taps     = nTaps == 3UZ ? kTaps : rampTaps(nTaps);
                const auto               runChain = [&](gr::Size_t samples) { return domain == "host" ? runChainOn(domain, kFrame, samples, taps) : runChainOn<DirectFirSycl<float>>(domain, kFrame, samples, taps); };
                throughput.push_back(bestMegaSamplesPerSecond(runChain, kNSamples));
                expect(gt(throughput.back(), 0.0)) << std::format("'{}' at {} taps produced nothing", domain, nTaps);
            }
            std::println("  {:<24} {:>10.2f} {:>10.2f} {:>10.2f}", domain == "host" ? std::format("{}, whole span", domain) : std::format("{}, own kernel", domain), throughput[0], throughput[1], throughput[2]);
        }
        std::println("  (MSample/s, best of three; indicative rather than a benchmark)\n");
    };
}
