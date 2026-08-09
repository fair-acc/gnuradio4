#include <algorithm>
#include <array>
#include <chrono>
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
void runParallelChain(std::string_view domain, gr::Size_t nSamples, std::size_t chunk) {
    using namespace gr::testing;
    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<TBlock>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}});
    std::ignore      = flow.connect<"out", "in">(source, dut, {.minBufferSize = chunk});
    std::ignore      = flow.connect<"out", "in">(dut, sink, {.minBufferSize = chunk});

    gr::scheduler::Simple<> sched;
    std::ignore = sched.exchange(std::move(flow));
    std::ignore = sched.runAndWait();
}

void runChain(std::string_view domain, gr::Size_t nSamples, std::size_t chunk) {
    using namespace gr::testing;
    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<IirSection>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}});
    std::ignore      = flow.connect<"out", "in">(source, dut, {.minBufferSize = chunk});
    std::ignore      = flow.connect<"out", "in">(dut, sink, {.minBufferSize = chunk});

    gr::scheduler::Simple<> sched;
    std::ignore = sched.exchange(std::move(flow));
    std::ignore = sched.runAndWait();
}

} // namespace bm_device

inline constexpr std::size_t kRepetitions = 5UZ;

template<typename TRun>
[[nodiscard]] double bestOfSeconds(TRun&& run) {
    double best = std::numeric_limits<double>::max();
    for (std::size_t repetition = 0UZ; repetition < kRepetitions; ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        run();
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        best                                        = std::min(best, elapsed.count());
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

    std::println("\n== fixed cost of ONE dispatch's scratch: 9 shared-USM regions, {} KiB of which is the two tag arenas ==", kScratchPer / 1024UZ);
    for (const std::string& domain : domains) {
        gr::device::DeviceContext* ctx = gr::device::DeviceContextRegistry::instance().tryResolve(domain);
        if (ctx == nullptr) {
            continue; // the plain host domain has no context, and pays none of this
        }
        constexpr std::size_t kLoops  = 200UZ;
        const double          seconds = bestOfSeconds([ctx] {
            for (std::size_t i = 0UZ; i < kLoops; ++i) {
                std::array<gr::device::DeviceBuffer, 9> scratch{ctx->allocateShared<std::byte>(64UZ), ctx->allocateShared<std::byte>(64UZ), ctx->allocateShared<std::uint32_t>(1UZ), //
                    ctx->allocateShared<std::byte>(kSlotBytes), ctx->allocateShared<std::size_t>(gr::device::kDeviceTagSlots),                                                       //
                    ctx->allocateShared<gr::Tag>(gr::device::kDeviceTagSlots), ctx->allocateShared<std::byte>(kSlotBytes),                                                           //
                    ctx->allocateShared<std::size_t>(1UZ), ctx->allocateShared<std::size_t>(1UZ)};
                for (gr::device::DeviceBuffer& buffer : scratch) {
                    ctx->deallocate(buffer);
                }
            }
        });
        std::println("  {:<12} {:>10.1f} us per dispatch", domain, 1e6 * seconds / static_cast<double>(kLoops));
    }

    constexpr gr::Size_t kSamples = 1U << 16U;
    std::println("\n== the same recursive IIR section, {} samples, by chunk size ==", static_cast<std::size_t>(kSamples));
    std::println("  {:<12} {:>10} {:>10} {:>10} {:>10}", "domain", "chunk 64", "chunk 512", "chunk 4k", "chunk 64k");
    for (const std::string& domain : domains) {
        std::print("  {:<12}", domain);
        for (const std::size_t chunk : {64UZ, 512UZ, 4096UZ, 65536UZ}) {
            const double seconds = bestOfSeconds([&domain, chunk] { runChain(domain, kSamples, chunk); });
            std::print(" {:>8.2f} MS/s", 1e-6 * static_cast<double>(kSamples) / seconds);
        }
        std::println("");
    }

    std::println("\n== the auto-parallel tier: N independent work items, {} samples, chunk 64k ==", static_cast<std::size_t>(kSamples));
    std::println("  {:<12} {:>18} {:>18}", "domain", "gain (1 flop)", "polynomial (128)");
    for (const std::string& domain : domains) {
        const double light = bestOfSeconds([&domain] { runParallelChain<Gain>(domain, kSamples, 65536UZ); });
        const double heavy = bestOfSeconds([&domain] { runParallelChain<Polynomial>(domain, kSamples, 65536UZ); });
        std::println("  {:<12} {:>12.2f} MS/s {:>12.2f} MS/s", domain, 1e-6 * static_cast<double>(kSamples) / light, 1e-6 * static_cast<double>(kSamples) / heavy);
    }

    std::println("\n== auto-parallel scaling, 128 flops/sample, one dispatch per run ==");
    std::println("  {:<12} {:>12} {:>12} {:>12} {:>12}", "domain", "64k", "256k", "1M", "4M");
    for (const std::string& domain : domains) {
        std::print("  {:<12}", domain);
        for (const gr::Size_t n : {1U << 16U, 1U << 18U, 1U << 20U, 1U << 22U}) {
            const double seconds = bestOfSeconds([&domain, n] { runParallelChain<Polynomial>(domain, n, static_cast<std::size_t>(n)); });
            std::print(" {:>7.1f} MS/s", 1e-6 * static_cast<double>(n) / seconds);
        }
        std::println("");
    }
}
