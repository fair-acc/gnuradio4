#include <atomic>
#include <chrono>
#include <cstdio>
#include <print>
#include <thread>

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/SchedulerModel.hpp>

#include <gnuradio-4.0/fourier/fft.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

// FFT throughput driven by a scheduler, measured with the block directly in the top-level graph and with the same
// block inside a managed sub-Graph whose ports are exported to the parent; the difference is the vehicle's cost.
// Both are timed to a fixed number of produced DataSets and then stopped, which keeps EOS/termination differences
// out of the number. A DataSet count of 0 means no data ever crossed the sub-Graph boundary.

using namespace gr;

using OuterScheduler  = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
using InnerScheduler  = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
using SubGraphWrapper = gr::SchedulerWrapper<InnerScheduler>;

using SampleType = float;
using FftBlock   = gr::blocks::fft::FFT<SampleType, gr::DataSet<SampleType>>;
using Source     = gr::testing::CountingSource<SampleType>;
using Sink       = gr::testing::CountingSink<gr::DataSet<SampleType>>;

struct RunResult {
    double      seconds  = 0.0;
    gr::Size_t  dataSets = 0U;
    std::size_t fftSize  = 0UZ;
    bool        timedOut = false;

    [[nodiscard]] std::size_t samples() const noexcept { return static_cast<std::size_t>(dataSets) * fftSize; }
    [[nodiscard]] double      nsPerSample() const noexcept { return samples() == 0UZ ? 0.0 : seconds * 1e9 / static_cast<double>(samples()); }
    [[nodiscard]] double      megaSamplesPerSecond() const noexcept { return seconds <= 0.0 ? 0.0 : static_cast<double>(samples()) / seconds / 1e6; }
};

// the sink's count is read here while the scheduler's workers write it; a torn read costs one poll iteration of
// accuracy, far below the measurement's resolution
template<typename TScheduler>
RunResult driveUntil(TScheduler& scheduler, const Sink& sink, gr::Size_t targetDataSets, gr::Size_t fftSize) {
    constexpr auto kDeadline = std::chrono::seconds(20);

    const auto  start = std::chrono::steady_clock::now();
    std::thread runner([&scheduler] { std::ignore = scheduler.runAndWait(); });

    bool timedOut = true;
    auto reached  = start;
    while (std::chrono::steady_clock::now() - start < kDeadline) {
        if (sink.count.value >= targetDataSets) {
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

    return RunResult{std::chrono::duration<double>(reached - start).count(), observed, static_cast<std::size_t>(fftSize), timedOut};
}

RunResult runFlat(gr::Size_t targetDataSets, gr::Size_t fftSize) {
    gr::Graph graph;
    auto&     src  = graph.emplaceBlock<Source>();
    auto&     fft  = graph.emplaceBlock<FftBlock>({{"fft_size", fftSize}});
    auto&     sink = graph.emplaceBlock<Sink>();

    std::ignore = graph.connect(src, "out", fft, "in");
    std::ignore = graph.connect(fft, "out", sink, "in");

    OuterScheduler scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    return driveUntil(scheduler, sink, targetDataSets, fftSize);
}

RunResult runSubGraph(gr::Size_t targetDataSets, gr::Size_t fftSize) {
    gr::Graph innerGraph;
    auto&     fft = innerGraph.emplaceBlock<FftBlock>({{"fft_size", fftSize}});
    // both peers are attached from the parent, so the member must not stop for want of a neighbour
    fft.disconnect_on_done    = false;
    const std::string fftName = std::string(fft.unique_name);

    auto  wrapperOwned = std::static_pointer_cast<gr::BlockModel>(std::make_shared<SubGraphWrapper>());
    auto* wrapper      = static_cast<SubGraphWrapper*>(wrapperOwned.get());
    wrapper->setGraph(std::move(innerGraph));

    std::ignore = wrapper->exportPort(true, fftName, gr::PortDirection::INPUT, "in", "inExp");
    std::ignore = wrapper->exportPort(true, fftName, gr::PortDirection::OUTPUT, "out", "outExp");

    gr::Graph graph;
    auto&     src              = graph.emplaceBlock<Source>();
    auto&     sink             = graph.emplaceBlock<Sink>();
    const auto&       wrapperRef = graph.addBlock(std::move(wrapperOwned));
    const std::string wrapperName(wrapperRef->uniqueName());

    std::ignore = graph.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(wrapperName), "inExp", gr::undefined_size, 0, "src->sub");
    std::ignore = graph.emplaceEdge(std::string_view(wrapperName), "outExp", std::string_view(sink.unique_name), "in", gr::undefined_size, 0, "sub->sink");

    OuterScheduler scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    return driveUntil(scheduler, sink, targetDataSets, fftSize);
}

void report(std::string_view label, const RunResult& r) {
    std::println("{:<30} {:>9.1f} ms {:>9.2f} MS/s {:>8.2f} ns/sample   dataSets={}{}", //
        label, r.seconds * 1e3, r.megaSamplesPerSecond(), r.nsPerSample(), r.dataSets,  //
        r.dataSets == 0U ? "   <-- NO DATA CROSSED THE BOUNDARY" : (r.timedOut ? "   <-- TIMED OUT" : ""));
}

int main() {
    constexpr gr::Size_t kFftSize = 1024U;
    constexpr gr::Size_t kTarget  = 1024U; // DataSets => 1 Mi samples through the transform
    constexpr int        kRepeats = 3;

    std::println("FFT throughput: flat graph vs managed sub-Graph (fft_size={}, {} DataSets, best of {})", kFftSize, kTarget, kRepeats);

    RunResult bestFlat{.seconds = 1e30};
    RunResult bestSub{.seconds = 1e30};

    for (int i = 0; i < kRepeats; ++i) {
        std::print("  run {}/{}: flat ...", i + 1, kRepeats);
        std::fflush(stdout);
        const RunResult flat = runFlat(kTarget, kFftSize);
        std::print(" {:.1f} ms, sub ...", flat.seconds * 1e3);
        std::fflush(stdout);
        const RunResult sub = runSubGraph(kTarget, kFftSize);
        std::println(" {:.1f} ms (dataSets={})", sub.seconds * 1e3, sub.dataSets);
        std::fflush(stdout);

        if (flat.dataSets >= kTarget && flat.seconds < bestFlat.seconds) {
            bestFlat = flat;
        }
        if (sub.dataSets >= kTarget && sub.seconds < bestSub.seconds) {
            bestSub = sub;
        }
    }

    std::println("");
    report("FFT only (flat graph)", bestFlat);
    report("FFT in managed sub-Graph", bestSub);

    if (bestFlat.dataSets >= kTarget && bestSub.dataSets >= kTarget) {
        std::println("\nsub-Graph overhead: {:+.1f} %  ({:+.2f} ns/sample)", //
            100.0 * (bestSub.seconds - bestFlat.seconds) / bestFlat.seconds, bestSub.nsPerSample() - bestFlat.nsPerSample());
    }
    return bestSub.dataSets >= kTarget ? 0 : 1;
}
