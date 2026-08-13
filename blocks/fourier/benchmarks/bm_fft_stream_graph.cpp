#include <array>
#include <chrono>
#include <complex>
#include <concepts>
#include <cstdio>
#include <print>
#include <thread>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/fourier/fft.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

// FFT throughput driven by a scheduler in STREAM mode, so the gap to bm_FFT_backends.cpp's bare SimdFFT is
// framework cost alone; bm_fft_subgraph.cpp covers SPECTRUM mode. A CountingSource -> Copy -> CountingSink control
// bounds what source/sink/scheduler alone sustain, though FFT chunks to fft_size and Copy does not. Runs are ~8 Mi
// samples: at 1 Mi, per-run wall time approaches driveUntil's 200 us poll granularity and rows reorder.

using namespace gr;

using Scheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;

using SampleType   = float;
using FftBlock     = gr::blocks::fft::FFT<SampleType>; // U defaults to complex<SampleType> -> STREAM mode, not SPECTRUM
using ControlBlock = gr::testing::Copy<std::complex<SampleType>>;
using Source       = gr::testing::CountingSource<std::complex<SampleType>>;
using Sink         = gr::testing::CountingSink<std::complex<SampleType>>;

struct RunResult {
    double     seconds  = 0.0;
    gr::Size_t samples  = 0U;
    bool       timedOut = false;

    [[nodiscard]] double nsPerSample() const noexcept { return samples == 0U ? 0.0 : seconds * 1e9 / static_cast<double>(samples); }
    [[nodiscard]] double megaSamplesPerSecond() const noexcept { return seconds <= 0.0 ? 0.0 : static_cast<double>(samples) / seconds / 1e6; }
};

// the sink's count is read here while the scheduler's workers write it; a torn read costs one poll iteration of
// accuracy, far below the measurement's resolution
RunResult driveUntil(Scheduler& scheduler, const Sink& sink, gr::Size_t targetSamples) {
    constexpr auto kDeadline = std::chrono::seconds(20);

    const auto  start = std::chrono::steady_clock::now();
    std::thread runner([&scheduler] { std::ignore = scheduler.runAndWait(); });

    bool timedOut = true;
    auto reached  = start;
    while (std::chrono::steady_clock::now() - start < kDeadline) {
        if (sink.count.value >= targetSamples) {
            reached  = std::chrono::steady_clock::now();
            timedOut = false;
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    if (timedOut) {
        reached = std::chrono::steady_clock::now();
    }

    const gr::Size_t observed = sink.count.value;
    std::ignore               = scheduler.changeStateTo(gr::lifecycle::State::REQUESTED_STOP);
    runner.join();

    return RunResult{std::chrono::duration<double>(reached - start).count(), observed, timedOut};
}

RunResult runStream(gr::Size_t targetSamples, gr::Size_t fftSize) {
    gr::Graph graph;
    auto&     src  = graph.emplaceBlock<Source>();
    auto&     fft  = graph.emplaceBlock<FftBlock>({{"fft_size", fftSize}});
    auto&     sink = graph.emplaceBlock<Sink>();

    std::ignore = graph.connect(src, "out", fft, "in");
    std::ignore = graph.connect(fft, "out", sink, "in");

    Scheduler scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    return driveUntil(scheduler, sink, targetSamples);
}

RunResult runControl(gr::Size_t targetSamples) {
    gr::Graph graph;
    auto&     src  = graph.emplaceBlock<Source>();
    auto&     pass = graph.emplaceBlock<ControlBlock>();
    auto&     sink = graph.emplaceBlock<Sink>();

    std::ignore = graph.connect(src, "out", pass, "in");
    std::ignore = graph.connect(pass, "out", sink, "in");

    Scheduler scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    return driveUntil(scheduler, sink, targetSamples);
}

void report(std::string_view label, const RunResult& r) {
    std::println("{:<30} {:>9.1f} ms {:>9.2f} MS/s {:>8.2f} ns/sample   samples={}{}", //
        label, r.seconds * 1e3, r.megaSamplesPerSecond(), r.nsPerSample(), r.samples,  //
        r.samples == 0U ? "   <-- NO DATA CROSSED THE GRAPH" : (r.timedOut ? "   <-- TIMED OUT" : ""));
}

template<typename Fn>
requires std::invocable<Fn>
RunResult bestOf(int repeats, gr::Size_t targetSamples, Fn&& run) {
    RunResult best{.seconds = 1e30};
    for (int i = 0; i < repeats; ++i) {
        std::print("  run {}/{}: ...", i + 1, repeats);
        std::fflush(stdout);
        const RunResult r = run();
        std::println(" {:.1f} ms (samples={})", r.seconds * 1e3, r.samples);
        std::fflush(stdout);

        if (r.samples >= targetSamples && r.seconds < best.seconds) {
            best = r;
        }
    }
    return best;
}

int main() {
    constexpr std::array<gr::Size_t, 3> kFftSizes      = {1024U, 4096U, 16384U}; // matches bm_FFT_backends' N columns
    constexpr gr::Size_t                kTargetSamples = 8U * 1024U * 1024U;     // ~8 Mi samples/run -- see file comment
    constexpr int                       kRepeats       = 3;

    std::println("FFT STREAM-mode throughput in a scheduler-driven gr::Graph (complex<float> -> complex<float>, {} samples/run, best of {})", kTargetSamples, kRepeats);
    std::println("compare the MS/s below directly against the algorithm-layer complex<float> SimdFFT row in bm_FFT_backends for the same N");

    std::println("");
    std::println("control: CountingSource -> Copy passthrough -> CountingSink (no FFT)");
    const RunResult bestControl = bestOf(kRepeats, kTargetSamples, [] { return runControl(kTargetSamples); });
    std::println("");
    report("Control (passthrough, no FFT)", bestControl);
    std::println("=> control ceiling: {:.2f} MS/s (source/sink/scheduler, unchunked passthrough, no FFT compute)", bestControl.megaSamplesPerSecond());

    bool allReachedTarget = bestControl.samples >= kTargetSamples;

    for (gr::Size_t fftSize : kFftSizes) {
        std::println("");
        std::println("fft_size={}", fftSize);

        const RunResult best = bestOf(kRepeats, kTargetSamples, [fftSize] { return runStream(kTargetSamples, fftSize); });

        std::println("");
        report("FFT stream (flat graph)", best);

        allReachedTarget = allReachedTarget && best.samples >= kTargetSamples;
        std::println("=> fft_size={:>6}: {:.2f} MS/s", fftSize, best.megaSamplesPerSecond());
    }

    return allReachedTarget ? 0 : 1;
}
