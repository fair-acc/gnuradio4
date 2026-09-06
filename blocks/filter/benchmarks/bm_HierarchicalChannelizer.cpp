
#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdlib>
#include <format>
#include <print>
#include <string>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/DriftResampler.hpp>
#include <gnuradio-4.0/filter/FrequencyXlatingFilter.hpp>
#include <gnuradio-4.0/filter/HierarchicalPolyphaseChannelizer.hpp>
#include <gnuradio-4.0/filter/PolyphaseChannelizer.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

namespace {
using C = std::complex<float>;

/// INPUT samples per second, so a channelizer is not flattered by counting the channels it fans out to
///
/// `NullSource` and `CountingSink` are the cheapest pair the testing library offers: the source emits a
/// default-constructed sample with no counter and no tag machinery, and the run is bounded at the SINK, which
/// is also what stops the graph. Each port therefore has to be given the count IT will see rather than the
/// input count -- `inputsPerOutput` is the per-port decimation (the channel count for a bank, `decimation`
/// for the xlating filter, 1 for a 1:1 block), and getting it wrong measures a different number of samples
/// than the rate is divided by.
///
/// `nChannels == 0` means the block has a single `out` port rather than a collection
template<typename TBlock>
[[nodiscard]] double megaSamplesPerSecond(std::string_view domain, gr::property_map settings, std::size_t nChannels, std::size_t inputsPerOutput, gr::Size_t nSamples, int attempts) {
    using namespace gr::testing;

    double best = 0.0;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     source = flow.emplaceBlock<NullSource<C>>({});

        gr::property_map dutSettings     = settings;
        dutSettings["gr:compute_domain"] = std::string(domain);
        auto& dut                        = flow.emplaceBlock<TBlock>(std::move(dutSettings));

        if (!flow.connect(source, "out", dut, "in").has_value()) {
            return 0.0;
        }
        const gr::Size_t perPort = static_cast<gr::Size_t>(static_cast<std::size_t>(nSamples) / std::max(std::size_t{1}, inputsPerOutput));
        if (nChannels == 0UZ) {
            auto& sink = flow.emplaceBlock<CountingSink<C>>({{"n_samples_max", perPort}});
            if (!flow.connect(dut, "out", sink, "in").has_value()) {
                return 0.0;
            }
        }
        for (std::size_t k = 0UZ; k < nChannels; ++k) {
            auto& sink = flow.emplaceBlock<CountingSink<C>>({{"n_samples_max", perPort}});
            if (!flow.connect(dut, "out#" + std::to_string(k), sink, "in").has_value()) {
                return 0.0;
            }
        }

        gr::scheduler::Simple<> sched;
        if (!sched.exchange(std::move(flow)).has_value()) {
            return 0.0;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!sched.runAndWait().has_value()) {
            return 0.0;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
        if (elapsed.count() > 0) {
            best = std::max(best, static_cast<double>(nSamples) / static_cast<double>(elapsed.count()));
        }
    }
    return best; // samples/us == MSample/s
}
} // namespace

int main() {
    // GR4's host path is single-threaded by design, and 'host:sycl' is a HOST device -- it must be held to the
    // same rule or its column is a thread count rather than a throughput. AdaptiveCpp's OpenMP backend
    // otherwise spreads `parallel_for` over every core it can see, which read as a 2.6x lead over plain `host`
    // on the banked rows and reversed entirely when both were pinned to one core.
    // Set before the runtime is registered, because the backend reads this when it builds its queue. An
    // explicit setting in the environment still wins, so the threaded figure remains measurable on purpose.
    setenv("OMP_NUM_THREADS", "1", 0);

    // without this no SYCL context exists and every device domain resolves to the host, which reads as a
    // device number but is not one
    std::ignore = gr::device::registerSyclRuntime();

    // 2^24, not 2^20: `runAndWait()` calls `start()`, which connects the edges (Scheduler.hpp:start), and a
    // pinned device edge allocates there -- inside the timed region. That setup is per RUN and per EDGE, so at
    // 2^20 it distorted the many-port rows far more than the single-edge ceiling. Amortising it is what makes
    // the columns comparable; it does not remove it.
    constexpr gr::Size_t kSamples  = 1U << 24;
    constexpr int        kAttempts = 3;

    std::println("Hierarchical channelizer: FIR polyphase bank against an allpass IIR half-band tree.");
    std::println("Input MSample/s, best of {}, {} samples per run. A channelizer's cost is per INPUT sample.", kAttempts, kSamples);
    std::println("Driven by NullSource into CountingSink: the cheapest pair available, so what is measured is");
    std::println("the block under test rather than the harness around it.");
    std::println("SYCL host backend held to OMP_NUM_THREADS={}: 'host:sycl' is a host device and GR4's host path", std::getenv("OMP_NUM_THREADS"));
    std::println("is single-threaded by design, so a threaded figure there would not be comparable.");
    std::println("");
    std::println("| channels | structure          | domain   | MSample/s |");
    std::println("| -------- | ------------------ | -------- | --------- |");

    for (const auto& domain : {"host", "host:sycl", "gpu:sycl"}) {
        // what the graph itself sustains with nothing but a copy in the middle: every row below is bounded by
        // this, so a filter figure is only meaningful as a fraction of it
        const double ceiling = megaSamplesPerSecond<gr::testing::Copy<C>>(domain, gr::property_map{}, 0UZ, 1UZ, kSamples, kAttempts);
        std::println("| {:>8} | chain ceiling      | {:<8} | {:>9.1f} |", "-", domain, ceiling);

        // one band translated and filtered, for scale against a bank that produces every band at once
        // the tap count is reported because `filter_order` sets the TRANSITION WIDTH, not the length: order 32
        // asks for 0.1/32 and the Kaiser estimate answers with 715 taps, which is the whole story of this row
        // 'Auto' is the shipped default and picks by tap count, so the FFT row is what a graph gets for free at
        // 715 taps -- and on a device it is the tap form again, the transform being a host evaluation
        for (const auto& [name, type, order, evaluation] : {std::tuple{"Xlating FIR", "FIR", gr::Size_t(32), "Time"}, std::tuple{"Xlating FFT", "FIR", gr::Size_t(32), "Auto"}, std::tuple{"Xlating FIR short", "FIR", gr::Size_t(4), "Time"}, std::tuple{"Xlating IIR", "IIR", gr::Size_t(4), "Time"}}) {
            const gr::property_map settings{{"frequency", 250'000.0}, {"sample_rate", 1'000'000.f}, {"cutoff", 60'000.f}, {"filter_type", std::string(type)}, {"filter_order", order}, {"filter_domain", std::string(evaluation)}, {"decimation", gr::Size_t(8)}, {"samples_per_frame", gr::Size_t(512)}};

            gr::filter::FrequencyXlatingFilter<C> probe(settings); // what the design actually produced
            probe.settings().init();
            std::ignore = probe.settings().applyStagedParameters();

            const double xlat = megaSamplesPerSecond<gr::filter::FrequencyXlatingFilter<C>>(domain, settings, 0UZ, 8UZ, kSamples, kAttempts);
            std::println("| {:>8} | {:<18} | {:<8} | {:>9.1f} |  {} taps", 1, name, domain, xlat, probe._nCoefficients);
        }

        // the arbitrary-ratio resampler, both interpolation kernels, at the same 2:1 rate so the two are
        // compared on their interpolation cost rather than on how much they produce
        for (const auto* kernel : {"Hermite", "Polyphase"}) {
            const double drift = megaSamplesPerSecond<gr::filter::DriftResampler<C>>(domain, //
                {{"ratio", 0.5f}, {"interpolation", std::string(kernel)}, {"n_phases", gr::Size_t(32)}, {"n_taps", gr::Size_t(1024)}}, 0UZ, 2UZ, kSamples, kAttempts);
            std::println("| {:>8} | {:<18} | {:<8} | {:>9.1f} |", 1, std::string("Drift ") + kernel, domain, drift);
        }

        for (const gr::Size_t channels : {8U, 16U}) {
            const gr::Size_t stage1 = channels == 8U ? 2U : 4U;
            const gr::Size_t stage2 = 4U; // 8 = 2x4, 16 = 4x4

            const double fir = megaSamplesPerSecond<gr::filter::HierarchicalPolyphaseChannelizer<C>>(domain, //
                {{"stage1_channels", stage1}, {"stage2_channels", stage2}, {"n_taps", gr::Size_t(32)}, {"outputs_per_frame", gr::Size_t(32)}}, channels, channels, kSamples, kAttempts);
            std::println("| {:>8} | FIR polyphase {}x{}  | {:<8} | {:>9.1f} |", channels, stage1, stage2, domain, fir);

            const double iir = megaSamplesPerSecond<gr::filter::HierarchicalIirChannelizer<C>>(domain, //
                {{"n_channels", channels}, {"n_sections", gr::Size_t(3)}, {"outputs_per_frame", gr::Size_t(64)}}, channels, channels, kSamples, kAttempts);
            std::println("| {:>8} | IIR allpass tree   | {:<8} | {:>9.1f} |", channels, domain, iir);

            // the flat bank without the hierarchy above it: one stage of M arms rather than two of sqrt(M)
            const double flat = megaSamplesPerSecond<gr::filter::PolyphaseChannelizer<C>>(domain, //
                {{"n_channels", channels}, {"n_taps", gr::Size_t(32)}, {"outputs_per_frame", gr::Size_t(64)}}, channels, channels, kSamples, kAttempts);
            std::println("| {:>8} | FIR flat bank      | {:<8} | {:>9.1f} |", channels, domain, flat);
        }
    }
    std::println("");
    std::println("NOTE: 'Xlating FFT' is the same design as 'Xlating FIR', evaluated by overlap-save instead of a");
    std::println("tap per sample -- so the two rows differ only in cost. The transform keeps its tap spectrum");
    std::println("on the host, so on a device domain that row falls back to the tap form by design.");
    std::println("Every other row is a real device figure -- all three channelizers, FrequencyXlatingFilter");
    std::println("and DriftResampler implement processBulk(ctx, ...), and none of them falls back.");
    std::println("The two IIR structures are on a device for RESIDENCY, not speed: a recursion has no");
    std::println("parallelism to offer, so it runs in one work item and reads WORSE than the host.");
    std::println("'chain ceiling' is source -> copy -> sink at 1:1, so it does NOT bound a decimating row:");
    std::println("the xlating rows pass an eighth of their input downstream and the Drift rows a half, which");
    std::println("is why those figures may read above the ceiling.");
    std::println("The bracketed tap count is what `filter_order` produced: it sets the transition width");
    std::println("(0.1/order), not the length, so order 32 -> 715 taps and order 4 -> 91.");
}
