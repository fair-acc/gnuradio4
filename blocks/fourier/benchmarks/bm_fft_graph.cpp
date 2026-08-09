#include <chrono>
#include <complex>
#include <print>
#include <string>
#include <thread>
#include <vector>

#include <gnuradio-4.0/AtomicRef.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/SchedulerModel.hpp>

#include <gnuradio-4.0/fourier/fft.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

// What the FFT block sustains once a scheduler drives it, in both modes it has and in both vehicles it can sit in,
// against a control that measures the framework alone. The gap to bm_FFT_backends.cpp's bare SimdFFT row for the
// same N is therefore framework cost, and the `%ctrl` column says how much of the attainable rate is left.
//
//   STREAM     complex<float> -> complex<float>, one sample out per sample in
//   SPECTRUM   float -> DataSet<float>, one DataSet per fft_size samples
//   sub-Graph  the SPECTRUM block inside a managed sub-Graph whose ports are exported to the parent
//
// Runs are sized in samples rather than repetitions: below ~1 Mi the per-run wall time approaches the 200 us poll
// granularity below and rows reorder. A count of zero means no data crossed the graph -- for the sub-Graph row, the
// exported-port boundary.

using namespace gr;

using Scheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
using SubGraph  = gr::SchedulerWrapper<gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>>;

using Sample = float;

struct RunResult {
    double      seconds  = 0.0;
    std::size_t samples  = 0UZ;
    bool        reached  = false;
    bool        timedOut = false;

    [[nodiscard]] double megaSamplesPerSecond() const noexcept { return seconds <= 0.0 ? 0.0 : static_cast<double>(samples) / seconds / 1e6; }
    [[nodiscard]] double nsPerSample() const noexcept { return samples == 0UZ ? 0.0 : seconds * 1e9 / static_cast<double>(samples); }
};

/// the sink's count is read here while the scheduler's workers write it; a torn read costs one poll iteration of
/// accuracy, far below the measurement's resolution. `perCount` converts the sink's unit into samples, so a
/// DataSet-producing mode is reported on the same axis as a sample-producing one.
template<typename TScheduler, typename TSink>
RunResult driveUntil(TScheduler& scheduler, const TSink& sink, gr::Size_t targetCount, std::size_t perCount) {
    constexpr auto kDeadline = std::chrono::seconds(20);

    const auto  start = std::chrono::steady_clock::now();
    std::thread runner([&scheduler] { std::ignore = scheduler.runAndWait(); });

    bool timedOut = true;
    auto reached  = start;
    while (std::chrono::steady_clock::now() - start < kDeadline) {
        if (gr::atomic_ref(const_cast<gr::Size_t&>(sink.count.value)).load_relaxed() >= targetCount) {
            reached  = std::chrono::steady_clock::now();
            timedOut = false;
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    if (timedOut) {
        reached = std::chrono::steady_clock::now();
    }

    const gr::Size_t observed = gr::atomic_ref(const_cast<gr::Size_t&>(sink.count.value)).load_relaxed();
    std::ignore               = scheduler.changeStateTo(gr::lifecycle::State::REQUESTED_STOP);
    runner.join();

    return RunResult{.seconds = std::chrono::duration<double>(reached - start).count(), .samples = static_cast<std::size_t>(observed) * perCount, .reached = observed >= targetCount, .timedOut = timedOut};
}

/// source -> sink with nothing between them but a copy: no filter row can exceed this, and it is the framework's
/// own per-sample cost made visible
RunResult runControl(gr::Size_t targetSamples) {
    gr::Graph graph;
    auto&     src  = graph.emplaceBlock<gr::testing::CountingSource<std::complex<Sample>>>();
    auto&     pass = graph.emplaceBlock<gr::testing::Copy<std::complex<Sample>>>();
    auto&     sink = graph.emplaceBlock<gr::testing::CountingSink<std::complex<Sample>>>();

    std::ignore = graph.connect(src, "out", pass, "in");
    std::ignore = graph.connect(pass, "out", sink, "in");

    Scheduler scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    return driveUntil(scheduler, sink, targetSamples, 1UZ);
}

RunResult runStream(gr::Size_t targetSamples, gr::Size_t fftSize) {
    gr::Graph graph;
    auto&     src  = graph.emplaceBlock<gr::testing::CountingSource<std::complex<Sample>>>();
    auto&     fft  = graph.emplaceBlock<gr::blocks::fft::FFT<Sample>>({{"fft_size", fftSize}}); // U defaults to complex -> STREAM
    auto&     sink = graph.emplaceBlock<gr::testing::CountingSink<std::complex<Sample>>>();

    std::ignore = graph.connect(src, "out", fft, "in");
    std::ignore = graph.connect(fft, "out", sink, "in");

    Scheduler scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    return driveUntil(scheduler, sink, targetSamples, 1UZ);
}

RunResult runSpectrum(gr::Size_t targetSamples, gr::Size_t fftSize) {
    using FftBlock = gr::blocks::fft::FFT<Sample, gr::DataSet<Sample>>;

    gr::Graph graph;
    auto&     src  = graph.emplaceBlock<gr::testing::CountingSource<Sample>>();
    auto&     fft  = graph.emplaceBlock<FftBlock>({{"fft_size", fftSize}});
    auto&     sink = graph.emplaceBlock<gr::testing::CountingSink<gr::DataSet<Sample>>>();

    std::ignore = graph.connect(src, "out", fft, "in");
    std::ignore = graph.connect(fft, "out", sink, "in");

    Scheduler scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    return driveUntil(scheduler, sink, targetSamples / fftSize, static_cast<std::size_t>(fftSize));
}

/// the same SPECTRUM block, reached through a managed sub-Graph's exported ports: the difference is the vehicle
RunResult runSubGraph(gr::Size_t targetSamples, gr::Size_t fftSize) {
    using FftBlock = gr::blocks::fft::FFT<Sample, gr::DataSet<Sample>>;

    gr::Graph innerGraph;
    auto&     fft = innerGraph.emplaceBlock<FftBlock>({{"fft_size", fftSize}});
    // both peers are attached from the parent, so the member must not stop for want of a neighbour
    fft.disconnect_on_done    = false;
    const std::string fftName = std::string(fft.unique_name);

    auto  wrapperOwned = std::static_pointer_cast<gr::BlockModel>(std::make_shared<SubGraph>());
    auto* wrapper      = static_cast<SubGraph*>(wrapperOwned.get());
    wrapper->setGraph(std::move(innerGraph));
    std::ignore = wrapper->exportPort(true, fftName, gr::PortDirection::INPUT, "in", "inExp");
    std::ignore = wrapper->exportPort(true, fftName, gr::PortDirection::OUTPUT, "out", "outExp");

    gr::Graph         graph;
    auto&             src        = graph.emplaceBlock<gr::testing::CountingSource<Sample>>();
    auto&             sink       = graph.emplaceBlock<gr::testing::CountingSink<gr::DataSet<Sample>>>();
    const auto&       wrapperRef = graph.addBlock(std::move(wrapperOwned));
    const std::string wrapperName(wrapperRef->uniqueName());

    std::ignore = graph.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(wrapperName), "inExp", gr::undefined_size, 0, "src->sub");
    std::ignore = graph.emplaceEdge(std::string_view(wrapperName), "outExp", std::string_view(sink.unique_name), "in", gr::undefined_size, 0, "sub->sink");

    Scheduler scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    return driveUntil(scheduler, sink, targetSamples / fftSize, static_cast<std::size_t>(fftSize));
}

template<typename Fn>
RunResult bestOf(int repeats, Fn&& run) {
    RunResult best{.seconds = 1e30};
    for (int i = 0; i < repeats; ++i) {
        const RunResult r = run();
        if (r.reached && r.seconds < best.seconds) {
            best = r;
        }
    }
    return best.seconds == 1e30 ? run() : best; // nothing reached the target: report one run rather than nothing
}

int main() {
    constexpr gr::Size_t          kTargetSamples = 8U * 1024U * 1024U; // ~8 Mi samples per run -- see the file comment
    constexpr int                 kRepeats       = 3;
    const std::vector<gr::Size_t> kFftSizes{1024U, 4096U, 16384U}; // matches bm_FFT_backends' N columns

    std::println("FFT block driven by a scheduler -- MSample/s in, best of {}, {} samples per run", kRepeats, kTargetSamples);
    std::println("`%ctrl` is the share of the control ceiling, which is the framework's own rate with no transform in the graph\n");

    const RunResult control = bestOf(kRepeats, [] { return runControl(kTargetSamples); });
    const double    ceiling = control.megaSamplesPerSecond();
    std::println("  {:<34} {:>10.2f} MS/s {:>8.2f} ns/sample   <- control: source -> copy -> sink, no FFT", "control ceiling", ceiling, control.nsPerSample());
    std::println("");

    const auto row = [ceiling](std::string_view label, const RunResult& r) {
        std::println("  {:<34} {:>10.2f} MS/s {:>8.2f} ns/sample {:>7.1f}%{}", label, r.megaSamplesPerSecond(), r.nsPerSample(), //
            ceiling <= 0.0 ? 0.0 : 100.0 * r.megaSamplesPerSecond() / ceiling,                                                   //
            r.samples == 0UZ ? "   <-- NO DATA CROSSED THE GRAPH" : (r.timedOut ? "   <-- TIMED OUT" : ""));
    };

    bool allReached = control.reached;
    for (const gr::Size_t fftSize : kFftSizes) {
        const RunResult stream   = bestOf(kRepeats, [&] { return runStream(kTargetSamples, fftSize); });
        const RunResult spectrum = bestOf(kRepeats, [&] { return runSpectrum(kTargetSamples, fftSize); });
        const RunResult sub      = bestOf(kRepeats, [&] { return runSubGraph(kTargetSamples, fftSize); });

        row(std::format("N={:<6} STREAM", fftSize), stream);
        row(std::format("N={:<6} SPECTRUM (flat)", fftSize), spectrum);
        row(std::format("N={:<6} SPECTRUM (sub-Graph)", fftSize), sub);
        if (spectrum.reached && sub.reached) {
            std::println("  {:<34} {:>+9.1f} %  ({:+.2f} ns/sample)", "  -> sub-Graph vehicle costs", 100.0 * (sub.seconds - spectrum.seconds) / spectrum.seconds, sub.nsPerSample() - spectrum.nsPerSample());
        }
        std::println("");

        allReached = allReached && stream.reached && spectrum.reached && sub.reached;
    }

    if (!allReached) {
        std::println("\n  NOTE: at least one row did not reach its sample target inside the deadline; the figures above are floors");
    }
    return 0;
}
