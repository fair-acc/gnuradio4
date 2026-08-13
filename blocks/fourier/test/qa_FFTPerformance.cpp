#include <boost/ut.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <format>
#include <memory>
#include <memory_resource>
#include <print>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/fourier/fft.hpp>
#include <gnuradio-4.0/testing/PerformanceMonitor.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

/*
 * Throughput, compute rate and resident-memory evolution of the FFT block across compute backends, in the style of
 * qa_PerformanceMonitor: a correctness gate per device backend against the host reference, then a fixed-wall-clock
 * source -> FFT -> PerformanceMonitor run per streaming-capable backend, stopped by a watchdog.
 */

namespace {

using namespace std::chrono_literals;
using C = std::complex<float>;

template<typename Scheduler>
auto createWatchdog(Scheduler& sched, std::chrono::seconds timeOut, std::chrono::milliseconds pollingPeriod = 40ms) {
    auto        stopFlag = std::make_shared<std::atomic_bool>(false);
    std::thread watchdog([&sched, stopFlag, timeOut, pollingPeriod]() {
        const auto deadline = std::chrono::steady_clock::now() + timeOut;
        while (std::chrono::steady_clock::now() < deadline) {
            if (sched.state() == gr::lifecycle::State::STOPPED) {
                return;
            }
            std::this_thread::sleep_for(pollingPeriod);
        }
        stopFlag->store(true, std::memory_order_relaxed);
        sched.requestStop();
    });
    return std::make_pair(std::move(watchdog), stopFlag);
}

struct BoundedResult {
    std::vector<C> spectrum;
    double         seconds = 0.0;
};

// one bounded forward transform; returns the spectrum (for the correctness gate) and the wall-clock time it took
BoundedResult transformOnce(std::string_view domain, std::size_t fftSize, std::size_t batches) {
    using namespace gr::testing;
    const gr::Size_t total = static_cast<gr::Size_t>(fftSize * batches);

    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<C, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", total}, {"mark_tag", false}});
    auto&     fft    = flow.emplaceBlock<gr::blocks::fft::FFT<float>>({{"gr:compute_domain", std::string(domain)}, {"fft_size", static_cast<gr::Size_t>(fftSize)}});
    auto&     sink   = flow.emplaceBlock<TagSink<C, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", total}, {"log_samples", true}});
    std::ignore      = flow.connect<"out", "in">(source, fft);
    std::ignore      = flow.connect<"out", "in">(fft, sink);

    gr::scheduler::Simple<> scheduler;
    std::ignore      = scheduler.exchange(std::move(flow));
    const auto start = std::chrono::steady_clock::now();
    std::ignore      = scheduler.runAndWait();
    const auto stop  = std::chrono::steady_clock::now();

    return {std::vector<C>(sink._samples.begin(), sink._samples.end()), std::chrono::duration<double>(stop - start).count()};
}

// relative L1 error of a spectrum against the host reference; 1e9 when the sizes disagree
double relativeError(const std::vector<C>& reference, const std::vector<C>& other) {
    if (reference.size() != other.size()) {
        return 1e9;
    }
    double numerator = 0.0;
    double denom     = 0.0;
    for (std::size_t i = 0UZ; i < reference.size(); ++i) {
        numerator += static_cast<double>(std::abs(reference[i] - other[i]));
        denom += static_cast<double>(std::abs(reference[i]));
    }
    return denom > 0.0 ? numerator / denom : numerator;
}

struct StreamSummary {
    double hotRate         = 0.0; // median sample rate over the second half of the run [S/s]
    double peakResidentMiB = 0.0;
    // only set by the two-FFT chain runs; without checking these, an A/B could silently measure the same
    // configuration twice and report a meaningless ratio
    bool middleEdgeInterior = false;
    bool allEdgesConnected    = true;
};

// runs source -> FFT -> PerformanceMonitor for `runTimeSeconds`, printing the monitor's continuous metrics, and
// distils a hot-state throughput and peak resident memory from the streamed rate/memory samples
StreamSummary streamFor(std::string_view domain, std::size_t fftSize, int runTimeSeconds) {
    using namespace gr;
    using namespace gr::testing;

    std::pmr::unsynchronized_pool_resource poolData;
    std::pmr::unsynchronized_pool_resource poolTag;
    std::pmr::unsynchronized_pool_resource poolMechanics;
    std::pmr::memory_resource* const       previousDefault = std::pmr::set_default_resource(&poolMechanics);

    Graph flow(gr::ResourceProfile{.data = &poolData, .tag = &poolTag, .mechanics = &poolMechanics});
    auto& source   = flow.emplaceBlock<TagSource<C, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(0)}, {"mark_tag", false}, {"name", "Source"}});
    auto& fft      = flow.emplaceBlock<blocks::fft::FFT<float>>({{"gr:compute_domain", std::string(domain)}, {"fft_size", static_cast<gr::Size_t>(fftSize)}});
    auto& monitor  = flow.emplaceBlock<PerformanceMonitor<C>>({{"name", "PerformanceMonitor"}, {"evaluate_perf_rate", static_cast<gr::Size_t>(1'000'000)}, {"publish_rate", 1.f}});
    auto& sinkRes  = flow.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>({{"name", "SinkRes"}, {"log_samples", true}, {"log_tags", false}});
    auto& sinkRate = flow.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>({{"name", "SinkRate"}, {"log_samples", true}, {"log_tags", false}});

    std::ignore = flow.connect<"out", "in">(source, fft);
    std::ignore = flow.connect<"out", "in">(fft, monitor);
    std::ignore = flow.connect<"outRes", "in">(monitor, sinkRes);
    std::ignore = flow.connect<"outRate", "in">(monitor, sinkRate);

    gr::scheduler::Simple<> scheduler;
    std::ignore                     = scheduler.exchange(std::move(flow));
    auto [watchdogThread, stopFlag] = createWatchdog(scheduler, std::chrono::seconds(runTimeSeconds));
    std::ignore                     = scheduler.runAndWait();
    if (watchdogThread.joinable()) {
        watchdogThread.join();
    }
    std::pmr::set_default_resource(previousDefault);

    StreamSummary             summary;
    const std::vector<double> rates(sinkRate._samples.begin(), sinkRate._samples.end());
    if (!rates.empty()) {
        std::vector<double> hot(rates.begin() + static_cast<std::ptrdiff_t>(rates.size() / 2), rates.end());
        std::ranges::sort(hot);
        summary.hotRate = hot[hot.size() / 2];
    }
    for (const double resident : sinkRes._samples) {
        summary.peakResidentMiB = std::max(summary.peakResidentMiB, resident / 1048576.0);
    }
    return summary;
}

/*
 * source -> FFT forward -> FFT inverse -> PerformanceMonitor, both transforms on the GPU, so the middle edge has a
 * device block at each end, so it never has to cross to the host. The A/B varies only that edge: `interiorEdge ==
 * false` spells the second domain `gpu:sycl:0` instead of `gpu:sycl`, which the graph reads as a boundary crossing
 * while still running on the same device. The residency obtained is asserted, so an A/B cannot silently measure one
 * configuration twice.
 */
StreamSummary streamChainFor(bool interiorEdge, std::size_t fftSize, int runTimeSeconds) {
    using namespace gr;
    using namespace gr::testing;

    std::pmr::unsynchronized_pool_resource poolData;
    std::pmr::unsynchronized_pool_resource poolTag;
    std::pmr::unsynchronized_pool_resource poolMechanics;
    std::pmr::memory_resource* const       previousDefault = std::pmr::set_default_resource(&poolMechanics);

    const std::string secondDomain = interiorEdge ? "gpu:sycl" : "gpu:sycl:0";

    Graph flow(gr::ResourceProfile{.data = &poolData, .tag = &poolTag, .mechanics = &poolMechanics});
    auto& source   = flow.emplaceBlock<TagSource<C, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(0)}, {"mark_tag", false}, {"name", "Source"}});
    auto& forward  = flow.emplaceBlock<blocks::fft::FFT<float>>({{"name", "FftForward"}, {"gr:compute_domain", std::string("gpu:sycl")}, {"fft_size", static_cast<gr::Size_t>(fftSize)}});
    auto& inverse  = flow.emplaceBlock<blocks::fft::FFT<float>>({{"name", "FftInverse"}, {"gr:compute_domain", secondDomain}, {"fft_size", static_cast<gr::Size_t>(fftSize)}, {"inverse", true}});
    auto& monitor  = flow.emplaceBlock<PerformanceMonitor<C>>({{"name", "PerformanceMonitor"}, {"evaluate_perf_rate", static_cast<gr::Size_t>(1'000'000)}, {"publish_rate", 1.f}});
    auto& sinkRes  = flow.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>({{"name", "SinkRes"}, {"log_samples", true}, {"log_tags", false}});
    auto& sinkRate = flow.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>({{"name", "SinkRate"}, {"log_samples", true}, {"log_tags", false}});

    std::ignore = flow.connect<"out", "in">(source, forward);
    std::ignore = flow.connect<"out", "in">(forward, inverse); // the edge under test
    std::ignore = flow.connect<"out", "in">(inverse, monitor);
    std::ignore = flow.connect<"outRes", "in">(monitor, sinkRes);
    std::ignore = flow.connect<"outRate", "in">(monitor, sinkRate);

    gr::scheduler::Simple<> scheduler;
    std::ignore                     = scheduler.exchange(std::move(flow));
    auto [watchdogThread, stopFlag] = createWatchdog(scheduler, std::chrono::seconds(runTimeSeconds));
    std::ignore                     = scheduler.runAndWait();
    if (watchdogThread.joinable()) {
        watchdogThread.join();
    }
    std::pmr::set_default_resource(previousDefault);

    StreamSummary summary;
    for (const gr::Edge& edge : scheduler.graph().edges()) {
        if (edge._state != gr::Edge::EdgeState::Connected) {
            summary.allEdgesConnected = false;
        }
        if (edge.sourceBlock() && edge.destinationBlock() && edge.sourceBlock()->name() == "FftForward" && edge.destinationBlock()->name() == "FftInverse") {
            summary.middleEdgeInterior = edge._domain.access == gr::Access::DeviceOnly;
        }
    }

    const std::vector<double> rates(sinkRate._samples.begin(), sinkRate._samples.end());
    if (!rates.empty()) {
        std::vector<double> hot(rates.begin() + static_cast<std::ptrdiff_t>(rates.size() / 2), rates.end());
        std::ranges::sort(hot);
        summary.hotRate = hot[hot.size() / 2];
    }
    for (const double resident : sinkRes._samples) {
        summary.peakResidentMiB = std::max(summary.peakResidentMiB, resident / 1048576.0);
    }
    return summary;
}

struct ChainScaling {
    double      hotRate       = 0.0;
    std::size_t interiorEdges = 0UZ;
    std::size_t totalEdges    = 0UZ;
    bool        allConnected  = true;
};

/*
 * Interior-fraction experiment: `nPairs` forward/inverse GPU FFT pairs on one domain keep the host boundary edges
 * fixed at two while interior edges grow as 2*nPairs-1. The marginal cost of a pair therefore contains only
 * interior edges and compute.
 */
ChainScaling streamFftPairsFor(std::size_t nPairs, std::size_t fftSize, int runTimeSeconds) {
    using namespace gr;
    using namespace gr::testing;

    std::pmr::unsynchronized_pool_resource poolData;
    std::pmr::unsynchronized_pool_resource poolTag;
    std::pmr::unsynchronized_pool_resource poolMechanics;
    std::pmr::memory_resource* const       previousDefault = std::pmr::set_default_resource(&poolMechanics);

    Graph flow(gr::ResourceProfile{.data = &poolData, .tag = &poolTag, .mechanics = &poolMechanics});
    auto& source = flow.emplaceBlock<TagSource<C, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(0)}, {"mark_tag", false}, {"name", "Source"}});

    std::vector<blocks::fft::FFT<float>*> transforms;
    for (std::size_t pair = 0UZ; pair < nPairs; ++pair) {
        transforms.push_back(&flow.emplaceBlock<blocks::fft::FFT<float>>({{"name", std::format("Fwd{}", pair)}, {"gr:compute_domain", std::string("gpu:sycl")}, {"fft_size", static_cast<gr::Size_t>(fftSize)}}));
        transforms.push_back(&flow.emplaceBlock<blocks::fft::FFT<float>>({{"name", std::format("Inv{}", pair)}, {"gr:compute_domain", std::string("gpu:sycl")}, {"fft_size", static_cast<gr::Size_t>(fftSize)}, {"inverse", true}}));
    }

    auto& monitor  = flow.emplaceBlock<PerformanceMonitor<C>>({{"name", "PerformanceMonitor"}, {"evaluate_perf_rate", static_cast<gr::Size_t>(1'000'000)}, {"publish_rate", 1.f}});
    auto& sinkRes  = flow.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>({{"name", "SinkRes"}, {"log_samples", true}, {"log_tags", false}});
    auto& sinkRate = flow.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>({{"name", "SinkRate"}, {"log_samples", true}, {"log_tags", false}});

    std::ignore = flow.connect<"out", "in">(source, *transforms.front());
    for (std::size_t i = 1UZ; i < transforms.size(); ++i) {
        std::ignore = flow.connect<"out", "in">(*transforms[i - 1UZ], *transforms[i]); // interior edges
    }
    std::ignore = flow.connect<"out", "in">(*transforms.back(), monitor);
    std::ignore = flow.connect<"outRes", "in">(monitor, sinkRes);
    std::ignore = flow.connect<"outRate", "in">(monitor, sinkRate);

    gr::scheduler::Simple<> scheduler;
    std::ignore                     = scheduler.exchange(std::move(flow));
    auto [watchdogThread, stopFlag] = createWatchdog(scheduler, std::chrono::seconds(runTimeSeconds));
    std::ignore                     = scheduler.runAndWait();
    if (watchdogThread.joinable()) {
        watchdogThread.join();
    }
    std::pmr::set_default_resource(previousDefault);

    ChainScaling scaling;
    for (const gr::Edge& edge : scheduler.graph().edges()) {
        ++scaling.totalEdges;
        if (edge._state != gr::Edge::EdgeState::Connected) {
            scaling.allConnected = false;
        }
        if (edge._domain.access == gr::Access::DeviceOnly) { // both endpoints on one device: no host crossing
            ++scaling.interiorEdges;
        }
    }
    const std::vector<double> rates(sinkRate._samples.begin(), sinkRate._samples.end());
    if (!rates.empty()) {
        std::vector<double> hot(rates.begin() + static_cast<std::ptrdiff_t>(rates.size() / 2), rates.end());
        std::ranges::sort(hot);
        scaling.hotRate = hot[hot.size() / 2];
    }
    return scaling;
}

} // namespace

// launches kernels, so the tests are registered and run from main() while the AdaptiveCpp runtime is alive (gotcha G10)
int main(int argc, char* argv[]) {
    using namespace boost::ut;
    using namespace std::string_view_literals;

    int runTime = 10; // seconds of continuous streaming per backend
    if (argc >= 2) {
        runTime = std::atoi(argv[1]);
    }
    std::println("qa_FFTPerformance <stream_time>[s] (default 10) — correctness gate, then each streaming backend runs {} s", runTime);

    const bool syclAvailable = gr::device::registerSyclRuntime();

    constexpr std::size_t kFftSize = 4096UZ;
    constexpr std::size_t kBatches = 128UZ;
    const double          log2N    = static_cast<double>(std::bit_width(kFftSize) - 1);
    const double          samples  = static_cast<double>(kFftSize * kBatches);

    const auto          resolves  = [](std::string_view domain) { return gr::device::DeviceContextRegistry::instance().tryResolve(domain) != nullptr; };
    const BoundedResult reference = transformOnce("host"sv, kFftSize, kBatches);

    // 1) correctness gate: one bounded transform per device backend, compared against the host reference
    std::println("\nFFT<float> forward transform, N={} — correctness vs host\n", kFftSize);
    for (const auto& [domain, available] : {std::pair{"host:sycl"sv, syclAvailable}, std::pair{"gpu:sycl"sv, syclAvailable}}) {
        const std::string name{domain};
        if (!available || !resolves(domain)) {
            std::println("  {:<10} bounded 1x: (backend not present in this build)", name);
            continue;
        }
        const BoundedResult run = transformOnce(domain, kFftSize, kBatches);
        const double        err = relativeError(reference.spectrum, run.spectrum);
        std::println("  {:<10} bounded 1x: {:>7.1f} MS/s   rel err {:.1e}{}", name, (samples / run.seconds) / 1e6, err, err > 1e-3 ? "   INCORRECT" : "");
        test(name + ": reproduces the host spectrum") = [err] { expect(err < 1e-3) << "the device FFT must compute what the CPU FFT computes"; };
    }

    // 2) continuous throughput + resident-memory evolution: host and the SYCL backends each stream for `runTime` seconds
    for (const auto& [domain, available] : {std::pair{"host"sv, true}, std::pair{"host:sycl"sv, syclAvailable}, std::pair{"gpu:sycl"sv, syclAvailable}}) {
        if (!available || (domain != "host"sv && !resolves(domain))) {
            continue;
        }
        const std::string name{domain};
        std::println("── {:<10}: {} s continuous ───────────────", name, runTime);
        const StreamSummary summary = streamFor(domain, kFftSize, runTime);
        const double        gflops  = summary.hotRate * 5.0 * log2N / 1e9;
        std::println("\n   {:<10} steady-state (median over second half): {:.1f} MS/s, {:.2f} GFLOP/s, peak resident {:.1f} MiB\n", name, summary.hotRate / 1e6, gflops, summary.peakResidentMiB);
        test(name + " streaming reached a steady state") = [rate = summary.hotRate] { expect(gt(rate, 0.0)) << "continuous streaming must sustain a positive sample rate"; };
    }

    // 3) what keeping the edge between two device blocks off the host is worth; compare A against B, not against section 2
    if (syclAvailable && resolves("gpu:sycl"sv)) {
        std::println("── two chained GPU FFTs, N={}: what the middle edge's residency is worth ───────────────", kFftSize);
        const StreamSummary crossing = streamChainFor(false, kFftSize, runTime);
        const StreamSummary interior = streamChainFor(true, kFftSize, runTime);

        std::println("\n   middle edge crosses to the host:  {:>7.1f} MS/s, peak resident {:.1f} MiB   [interior: {}]", crossing.hotRate / 1e6, crossing.peakResidentMiB, crossing.middleEdgeInterior);
        std::println("   middle edge stays on the device: {:>7.1f} MS/s, peak resident {:.1f} MiB   [interior: {}]", interior.hotRate / 1e6, interior.peakResidentMiB, interior.middleEdgeInterior);
        if (crossing.hotRate > 0.0) {
            std::println("   speed-up from keeping the edge on the device: {:.2f}x\n", interior.hotRate / crossing.hotRate);
        }

        test("the A/B really varied the middle edge's residency") = [crossing, interior] {
            expect(!crossing.middleEdgeInterior) << "the crossing arm's middle edge must be a host boundary";
            expect(interior.middleEdgeInterior) << "the interior arm's middle edge must stay on the device";
        };
        test("both chain arms connected every edge") = [crossing, interior] {
            expect(crossing.allEdgesConnected) << "an unconnected edge falls back to a default host buffer and invalidates the comparison";
            expect(interior.allEdgesConnected);
        };
        test("an interior middle edge sustains streaming") = [interior] { expect(gt(interior.hotRate, 0.0)); };
    } else {
        std::println("── two chained GPU FFTs: skipped (needs a served gpu:sycl domain) ──\n");
    }

    // 4) interior-fraction experiment: does adding transforms that bring only interior edges cost much? The two host
    //    boundary edges are the same in every row, so the marginal cost per added pair excludes them by construction.
    if (syclAvailable && resolves("gpu:sycl"sv)) {
        std::println("── interior-fraction: chains of N forward/inverse GPU FFT pairs, N={} ───────────────", kFftSize);
        std::vector<std::pair<std::size_t, ChainScaling>> rows;
        for (const std::size_t nPairs : {1UZ, 2UZ, 3UZ}) {
            rows.emplace_back(nPairs, streamFftPairsFor(nPairs, kFftSize, runTime));
        }
        std::println("\n   pairs  transforms  interior/total edges   rate [MS/s]   per-sample [ns]   marginal per pair [ns]");
        double previousPerSample = 0.0;
        for (const auto& [nPairs, row] : rows) {
            const double perSampleNs = row.hotRate > 0.0 ? 1e9 / row.hotRate : 0.0;
            const double marginal    = previousPerSample > 0.0 ? perSampleNs - previousPerSample : perSampleNs;
            std::println("   {:>5}  {:>10}  {:>10}/{:<9}  {:>10.1f}   {:>15.1f}   {:>21.1f}", nPairs, 2UZ * nPairs, row.interiorEdges, row.totalEdges, row.hotRate / 1e6, perSampleNs, marginal);
            previousPerSample = perSampleNs;
        }
        std::println("\n   Reading it: a marginal cost per added pair that is small against the first row's per-sample cost");
        std::println("   means the two fixed host-boundary edges dominate; one close to the first row means per-transform");
        std::println("   compute dominates instead.\n");

        test("every chain length kept its interior edges device-resident") = [rows] {
            for (const auto& [nPairs, row] : rows) {
                expect(row.allConnected) << "chain of " << nPairs << " pairs left an edge unconnected";
                expect(eq(row.interiorEdges, 2UZ * nPairs - 1UZ)) << "a chain of " << nPairs << " pairs has 2N-1 interior edges";
                expect(gt(row.hotRate, 0.0));
            }
        };
    }

    "the host reference produced a full spectrum"_test = [&] { expect(eq(reference.spectrum.size(), static_cast<std::size_t>(samples))); };

    return 0;
}
