#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <memory_resource>
#include <print>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

/*
 * What a device dispatch costs, and where the cost is. The fixed per-dispatch cost is measured on its own before
 * any filter runs, so that a slow device IIR is not blamed on the kernel.
 */

namespace bm_device {

using namespace gr;

struct IirSection : Block<IirSection> {
    static constexpr std::size_t kMaxOrder = 2UZ;

    PortIn<float>  in;
    PortOut<float> out;

    std::pmr::vector<float> b{0.2f};
    std::pmr::vector<float> a{1.f, -0.8f};

    GR_MAKE_REFLECTABLE(IirSection, in, out, b, a);

    mutable std::array<float, kMaxOrder> _x{};
    mutable std::array<float, kMaxOrder> _y{};

    [[nodiscard]] work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) const {
        const std::size_t count    = std::min(input.size(), output.size());
        const std::size_t nForward = std::min(b.size(), kMaxOrder + 1UZ);
        const std::size_t nBack    = std::min(a.size(), kMaxOrder + 1UZ);
        for (std::size_t n = 0UZ; n < count; ++n) {
            const float sample = input[n];
            float       acc    = b[0] * sample;
            for (std::size_t k = 1UZ; k < nForward; ++k) {
                acc += b[k] * _x[k - 1UZ];
            }
            for (std::size_t k = 1UZ; k < nBack; ++k) {
                acc -= a[k] * _y[k - 1UZ];
            }
            for (std::size_t k = kMaxOrder; k-- > 1UZ;) {
                _x[k] = _x[k - 1UZ];
                _y[k] = _y[k - 1UZ];
            }
            _x[0]     = sample;
            _y[0]     = acc;
            output[n] = acc;
        }
        std::ignore = input.consume(count);
        output.publish(count);
        return work::Status::OK;
    }
};

struct Gain : Block<Gain> {
    PortIn<float>  in;
    PortOut<float> out;

    float gain = 1.000001f;

    GR_MAKE_REFLECTABLE(Gain, in, out, gain);

    [[nodiscard]] constexpr float processOne(float sample) const noexcept { return gain * sample; }
};

struct Polynomial : Block<Polynomial> {
    PortIn<float>  in;
    PortOut<float> out;

    float gain = 1.000001f;

    GR_MAKE_REFLECTABLE(Polynomial, in, out, gain);

    [[nodiscard]] constexpr float processOne(float sample) const noexcept {
        float acc = sample;
        for (std::size_t k = 0UZ; k < 64UZ; ++k) { // ~128 dependent flops per sample
            acc = acc * gain + sample;
        }
        return acc;
    }
};

template<typename TBlock>
[[nodiscard]] double runParallelChain(std::string_view domain, gr::Size_t nSamples, std::size_t chunk) {
    using namespace gr::testing;
    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<TBlock>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}});
    std::ignore      = flow.connect<"out", "in">(source, dut, {.minBufferSize = chunk});
    std::ignore      = flow.connect<"out", "in">(dut, sink, {.minBufferSize = chunk});

    gr::scheduler::Simple<> sched;
    std::ignore      = sched.exchange(std::move(flow));
    const auto start = std::chrono::steady_clock::now(); // graph construction and scheduler setup are not the thing under test
    std::ignore      = sched.runAndWait();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// N device stages in a row, so that N-2 of them are INTERIOR: device memory on both sides, hence deferred. This is
/// the cascade claim in the unit it is made in -- if an interior hop costs no synchronisation, the time is a fixed
/// boundary cost plus N kernels, and the slope is kernel time alone rather than kernel time plus a barrier.
template<typename TBlock>
[[nodiscard]] double runChainOfLength(std::string_view domain, std::size_t nStages, gr::Size_t nSamples, std::size_t chunk) {
    using namespace gr::testing;
    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}});

    std::vector<TBlock*> stages;
    stages.reserve(nStages);
    for (std::size_t stage = 0UZ; stage < nStages; ++stage) {
        stages.push_back(std::addressof(flow.emplaceBlock<TBlock>({{"gr:compute_domain", std::string(domain)}})));
    }

    std::ignore = flow.connect<"out", "in">(source, *stages.front(), {.minBufferSize = chunk});
    for (std::size_t stage = 1UZ; stage < nStages; ++stage) {
        std::ignore = flow.connect<"out", "in">(*stages[stage - 1UZ], *stages[stage], {.minBufferSize = chunk});
    }
    std::ignore = flow.connect<"out", "in">(*stages.back(), sink, {.minBufferSize = chunk});

    gr::scheduler::Simple<> sched;
    std::ignore      = sched.exchange(std::move(flow));
    const auto start = std::chrono::steady_clock::now();
    std::ignore      = sched.runAndWait();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// port-collection shape, matching gr::dsp::demo::Channeliser in qa_DeviceDspChain.cpp: one work item per sample,
/// N separate output edges instead of one -- what ExecutionStrategy::stageOutputPort elides scratch for once
/// every edge is already device-resident
struct Splitter : Block<Splitter> {
    static constexpr std::size_t kChannels = 4UZ;

    PortIn<float>                         in;
    std::array<PortOut<float>, kChannels> out;

    GR_MAKE_REFLECTABLE(Splitter, in, out);

    [[nodiscard]] constexpr std::array<float, kChannels> processOne(float x) const noexcept {
        std::array<float, kChannels> perChannel{};
        for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
            perChannel[channel] = x * static_cast<float>(channel + 1UZ);
        }
        return perChannel;
    }
};

/// `outputEdge` decides whether the N out#c -> in edges stay host-resident (default) or are forced onto shared
/// GPU USM (gr::ComputeDomain::gpu_shared()), which is what makes every channel already device-accessible
[[nodiscard]] double runSplitterChain(std::string_view domain, gr::Size_t nSamples, std::size_t chunk, gr::EdgeParameters outputEdge) {
    using namespace gr::testing;
    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<Splitter>({{"gr:compute_domain", std::string(domain)}});

    std::vector<TagSink<float, ProcessFunction::USE_PROCESS_ONE>*> sinks;
    for (std::size_t channel = 0UZ; channel < Splitter::kChannels; ++channel) {
        sinks.push_back(&flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}}));
    }
    std::ignore = flow.connect<"out", "in">(source, dut, {.minBufferSize = chunk});
    for (std::size_t channel = 0UZ; channel < Splitter::kChannels; ++channel) {
        std::ignore = flow.connect(dut, PortDefinition{std::format("out#{}", channel)}, *sinks[channel], PortDefinition{"in"}, outputEdge);
    }

    gr::scheduler::Simple<> sched;
    std::ignore      = sched.exchange(std::move(flow));
    const auto start = std::chrono::steady_clock::now(); // graph construction and scheduler setup are not the thing under test
    std::ignore      = sched.runAndWait();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

[[nodiscard]] double runChain(std::string_view domain, gr::Size_t nSamples, std::size_t chunk) {
    using namespace gr::testing;
    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<IirSection>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}});
    std::ignore      = flow.connect<"out", "in">(source, dut, {.minBufferSize = chunk});
    std::ignore      = flow.connect<"out", "in">(dut, sink, {.minBufferSize = chunk});

    gr::scheduler::Simple<> sched;
    std::ignore      = sched.exchange(std::move(flow));
    const auto start = std::chrono::steady_clock::now(); // graph construction and scheduler setup are not the thing under test
    std::ignore      = sched.runAndWait();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace bm_device

inline constexpr std::size_t kRepetitions = 5UZ;

/// `run` returns the seconds it wants counted, so each case decides what its own timed region is -- here, the
/// scheduler run alone. Timing the whole call would fold graph construction into every row, which at a small chunk
/// is most of the wall clock and is why these numbers used to swing run to run.
template<typename TRun>
[[nodiscard]] double bestOfSeconds(TRun&& run) {
    double best = std::numeric_limits<double>::max();
    for (std::size_t repetition = 0UZ; repetition < kRepetitions; ++repetition) {
        best = std::min(best, run());
    }
    return best;
}

int main() {
    using namespace bm_device;

    if (!gr::device::registerSyclRuntime()) {
        std::println("no SYCL backend: nothing to measure");
        return 0;
    }

    std::vector<std::string> domains{"host"};
    for (std::string_view candidate : {"host:sycl", "gpu:sycl"}) {
        if (gr::device::DeviceContextRegistry::instance().tryResolve(candidate) != nullptr) {
            domains.emplace_back(candidate);
        }
    }

    constexpr std::size_t kSlotBytes  = gr::device::kDeviceTagSlots * gr::device::kDeviceTagSlotBytes;
    constexpr std::size_t kScratchPer = 2UZ * kSlotBytes;

    std::println("\n== fixed cost of one dispatch's CONTROL area ({} KiB of tag arenas), which the shadow now keeps ==", kScratchPer / 1024UZ);
    for (const std::string& domain : domains) {
        gr::device::DeviceContext* ctx = gr::device::DeviceContextRegistry::instance().tryResolve(domain);
        if (ctx == nullptr) {
            continue; // the plain host domain has no context, and pays none of this
        }
        constexpr std::size_t kLoops  = 200UZ;
        const double          seconds = bestOfSeconds([ctx] {
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0UZ; i < kLoops; ++i) {
                std::array<gr::device::DeviceBuffer, 9> scratch{ctx->allocateShared<std::byte>(64UZ), ctx->allocateShared<std::byte>(64UZ), ctx->allocateShared<std::uint32_t>(1UZ), //
                    ctx->allocateShared<std::byte>(kSlotBytes), ctx->allocateShared<std::size_t>(gr::device::kDeviceTagSlots),                                                       //
                    ctx->allocateShared<gr::Tag>(gr::device::kDeviceTagSlots), ctx->allocateShared<std::byte>(kSlotBytes),                                                           //
                    ctx->allocateShared<std::size_t>(1UZ), ctx->allocateShared<std::size_t>(1UZ)};
                for (gr::device::DeviceBuffer& buffer : scratch) {
                    ctx->deallocate(buffer);
                }
            }
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        });
        std::println("  {:<12} {:>10.1f} us per dispatch", domain, 1e6 * seconds / static_cast<double>(kLoops));
    }

    constexpr gr::Size_t kSamples = 1U << 16U;
    std::println("\n== the same recursive IIR section, {} samples, by chunk size ==", static_cast<std::size_t>(kSamples));
    std::println("  {:<12} {:>10} {:>10} {:>10} {:>10}", "domain", "chunk 64", "chunk 512", "chunk 4k", "chunk 64k");
    for (const std::string& domain : domains) {
        std::print("  {:<12}", domain);
        for (const std::size_t chunk : {64UZ, 512UZ, 4096UZ, 65536UZ}) {
            const double seconds = bestOfSeconds([&domain, chunk] { return runChain(domain, kSamples, chunk); });
            std::print(" {:>8.2f} MS/s", 1e-6 * static_cast<double>(kSamples) / seconds);
        }
        std::println("");
    }

    std::println("\n== the auto-parallel tier: N independent work items, {} samples, chunk 64k ==", static_cast<std::size_t>(kSamples));
    std::println("  {:<12} {:>18} {:>18}", "domain", "gain (1 flop)", "polynomial (128)");
    for (const std::string& domain : domains) {
        const double light = bestOfSeconds([&domain] { return runParallelChain<Gain>(domain, kSamples, 65536UZ); });
        const double heavy = bestOfSeconds([&domain] { return runParallelChain<Polynomial>(domain, kSamples, 65536UZ); });
        std::println("  {:<12} {:>12.2f} MS/s {:>12.2f} MS/s", domain, 1e-6 * static_cast<double>(kSamples) / light, 1e-6 * static_cast<double>(kSamples) / heavy);
    }

    std::println("\n== port-collection dispatch: {} channels, {} samples, chunk 64k ==", Splitter::kChannels, static_cast<std::size_t>(kSamples));
    std::println("  {:<12} {:>16} {:>16}", "domain", "host-resident", "device-resident");
    for (const std::string& domain : domains) {
        const gr::EdgeParameters hostResident{};
        const gr::EdgeParameters deviceResident{.domain = gr::ComputeDomain::gpu_shared()};
        const double             boundary = bestOfSeconds([&domain, &hostResident] { return runSplitterChain(domain, kSamples, 65536UZ, hostResident); });
        const double             resident = bestOfSeconds([&domain, &deviceResident] { return runSplitterChain(domain, kSamples, 65536UZ, deviceResident); });
        std::println("  {:<12} {:>12.2f} MS/s {:>12.2f} MS/s", domain, 1e-6 * static_cast<double>(kSamples) / boundary, 1e-6 * static_cast<double>(kSamples) / resident);
    }
    std::println("  (device-resident puts the output edges on shared GPU USM, where stageOutputPort finds every");
    std::println("   channel already device-accessible and skips its per-channel scratch. Both columns are");
    std::println("   measured for every domain; the host ones coincide because the edge kind changes nothing there)\n");

    std::println("\n== auto-parallel scaling, 128 flops/sample, one dispatch per run ==");
    std::println("  {:<12} {:>12} {:>12} {:>12} {:>12}", "domain", "64k", "256k", "1M", "4M");
    for (const std::string& domain : domains) {
        std::print("  {:<12}", domain);
        for (const gr::Size_t n : {1U << 16U, 1U << 18U, 1U << 20U, 1U << 22U}) {
            const double seconds = bestOfSeconds([&domain, n] { return runParallelChain<Polynomial>(domain, n, static_cast<std::size_t>(n)); });
            std::print(" {:>7.1f} MS/s", 1e-6 * static_cast<double>(n) / seconds);
        }
        std::println("");
    }

    // Where a device starts to pay. The same canonical sizes the FFT benchmarks sweep (bm_FFT_backends.cpp), used
    // here as the CHUNK a dispatch is handed, at a fixed total sample count so every row does the same work. Two
    // blocks bracket the answer by arithmetic per sample: a multiply does one flop, a polynomial 128. The
    // cross-over is where a device row first beats the plain-host row. Read the rows against each other rather than
    // against a bus: `host:sycl` crosses no bus and tracks `gpu:sycl` closely, so neither row is transfer-bound.
    // log-spaced from the frame sizes the filters actually run at up to the canonical FFT sizes
    // (`bm_FFT_backends.cpp`): a FIR frame is commonly tens of samples and an FFT thousands, and the interesting
    // part of the curve is where those two regimes meet
    constexpr std::array kCanonicalSizes = {16UZ, 64UZ, 256UZ, 1024UZ, 4096UZ, 16384UZ, 65536UZ};
    constexpr gr::Size_t kSweepSamples   = 1U << 22U;
    const auto           crossOverSweep  = [&](std::string_view label, auto runOne) {
        std::println("\n== {}: MSample/s against the chunk a dispatch is handed, {} samples total ==", label, static_cast<std::size_t>(kSweepSamples));
        std::print("  {:<12}", "domain");
        for (const std::size_t chunk : kCanonicalSizes) {
            std::print(" {:>9}", chunk);
        }
        std::println("");

        std::vector<std::vector<double>> rows;
        for (const std::string& domain : domains) {
            std::print("  {:<12}", domain);
            std::vector<double> row;
            for (const std::size_t chunk : kCanonicalSizes) {
                const double seconds = bestOfSeconds([&domain, chunk, &runOne] { return runOne(domain, kSweepSamples, chunk); });
                row.push_back(1e-6 * static_cast<double>(kSweepSamples) / seconds);
                std::print(" {:>9.1f}", row.back());
            }
            rows.push_back(std::move(row));
            std::println("");
        }

        // the first chunk at which each device row overtakes the plain-host row on the same chunk
        for (std::size_t d = 1UZ; d < domains.size() && d < rows.size(); ++d) {
            std::size_t crossOver = 0UZ;
            for (std::size_t i = 0UZ; i < kCanonicalSizes.size(); ++i) {
                if (rows[d][i] > rows[0][i]) {
                    crossOver = kCanonicalSizes[i];
                    break;
                }
            }
            if (crossOver == 0UZ) {
                std::println("  {} never overtakes '{}' in this range", domains[d], domains[0]);
            } else {
                std::println("  {} overtakes '{}' from a chunk of {} samples", domains[d], domains[0], crossOver);
            }
        }
    };
    crossOverSweep("multiply, 1 flop/sample (memory-bound)", [](std::string_view domain, gr::Size_t n, std::size_t chunk) { return runParallelChain<Gain>(domain, n, chunk); });
    crossOverSweep("polynomial, 128 flops/sample (arithmetic-bound)", [](std::string_view domain, gr::Size_t n, std::size_t chunk) { return runParallelChain<Polynomial>(domain, n, chunk); });
    // the sequential shape: one work item, its own consume/publish, state carried between calls. It cannot be
    // parallelised over samples at all, so a device can only ever win it back through residency
    crossOverSweep("IIR biquad, sequential (one work item)", [](std::string_view domain, gr::Size_t n, std::size_t chunk) { return runParallelChain<IirSection>(domain, n, chunk); });

    // Is a CHAIN bound by the same boundary transfer as a single block? Sweep the chunk for a fixed 8-stage chain
    // and compare against the single-block row above: if the two curves coincide, the interior stages are free and
    // the ceiling is the boundary; if the chain falls below, something inside it is the bound instead.
    std::println("\n== 8-stage chain vs 1 stage, MSample/s against the chunk, {} samples ==", static_cast<std::size_t>(kSweepSamples));
    std::println("  {:<22}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}", "domain / stages", 16, 64, 256, 1024, 4096, 16384, 65536);
    for (const std::string& domain : domains) {
        for (const std::size_t nStages : {1UZ, 8UZ}) {
            std::print("  {:<22}", std::format("{} / {}", domain, nStages));
            for (const std::size_t chunk : kCanonicalSizes) {
                const double seconds = bestOfSeconds([&domain, nStages, chunk] { return runChainOfLength<Polynomial>(domain, nStages, kSweepSamples, chunk); });
                std::print("{:>9.1f}", 1e-6 * static_cast<double>(kSweepSamples) / seconds);
            }
            std::println("");
        }
    }

    // The cascade claim, in wall time rather than in barrier counts. Stages 2..N-1 are interior -- device memory on
    // both sides -- so their kernels are enqueued and never awaited. If that holds, the per-stage cost is kernel
    // time alone and the row grows linearly with a shallow slope; a barrier per hop would show as a steeper one.
    constexpr gr::Size_t  kChainSamples = 1U << 20U;
    constexpr std::size_t kChainChunk   = 1UZ << 16U;
    std::println("\n== cost per added stage in a chain of N device blocks, {} samples, {} samples per chunk ==", static_cast<std::size_t>(kChainSamples), kChainChunk);
    std::println("  {:<12} {:>10} {:>10} {:>10} {:>10} {:>14}", "domain", "N=1", "N=2", "N=4", "N=8", "ms per stage");
    for (const std::string& domain : domains) {
        std::print("  {:<12}", domain);
        std::array<double, 4UZ> milliseconds{};
        std::size_t             column = 0UZ;
        for (const std::size_t nStages : {1UZ, 2UZ, 4UZ, 8UZ}) {
            const double seconds   = bestOfSeconds([&domain, nStages] { return runChainOfLength<Polynomial>(domain, nStages, kChainSamples, kChainChunk); });
            milliseconds[column++] = 1e3 * seconds;
            std::print(" {:>7.2f} ms", 1e3 * seconds);
        }
        std::println(" {:>13.3f}", (milliseconds[3] - milliseconds[0]) / 7.0);
    }
}
