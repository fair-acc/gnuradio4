#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <boost/ut.hpp>

#if defined(__EMSCRIPTEN__)
#include <emscripten/em_asm.h>
#include <emscripten/threading.h>
#endif

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContextWebGpu.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace gr::webgpu_test {

struct Scale : Block<Scale> {
    PortIn<float>  in;
    PortOut<float> out;

    Annotated<float, "gain"> gain = 2.f;
    GR_MAKE_REFLECTABLE(Scale, in, out, gain);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x * gain; }
};

/// a sink that never brings its samples to the host: a kernel folds their bit patterns into one shared word
struct DeviceChecksumSink : Block<DeviceChecksumSink> {
    PortIn<float> in;
    GR_MAKE_REFLECTABLE(DeviceChecksumSink, in);

    std::uint32_t              checksum = 0U; // the shared word, as the last work() left it
    gr::device::DeviceBuffer   _sum{};
    gr::device::DeviceContext* _context = nullptr;

    ~DeviceChecksumSink() {
        if (_sum) {
            _context->deallocate(_sum);
        }
    }

    gr::work::Status processBulk(gr::device::DeviceContext& context, InputSpanLike auto& samples) {
        if (!_sum) {
            _context                             = &context;
            _sum                                 = context.allocateShared<std::uint32_t>(1UZ);
            *_sum.devicePointer<std::uint32_t>() = 0U;
        }
        std::uint32_t* const sum  = _sum.devicePointer<std::uint32_t>();
        const float* const   data = std::ranges::data(samples);
        gr::device::parallelFor(context, samples.size(), [data, sum](std::size_t i) { std::atomic_ref<std::uint32_t>(*sum).fetch_add(std::bit_cast<std::uint32_t>(data[i]), std::memory_order_relaxed); });
        checksum    = *sum; // four bytes of shared memory, back from the launch; no sample is read here
        std::ignore = samples.consume(samples.size());
        return gr::work::Status::OK;
    }

    /// the host domain runs the same body on the CPU context
    gr::work::Status processBulk(InputSpanLike auto& samples) { return processBulk(gr::device::hostBackend(), samples); }
};

/// the sum of the bit patterns of 2·i for i in [0, n), wrapping at 2^32: what the sink must find, worked out here
[[nodiscard]] std::uint32_t expectedChecksum(std::size_t n, bool scaledInDouble = false) {
    std::uint32_t sum = 0U;
    for (std::size_t i = 0UZ; i < n; ++i) {
        const float twice = 2.f * static_cast<float>(i);
        sum += std::bit_cast<std::uint32_t>(scaledInDouble ? static_cast<float>(static_cast<double>(twice) * 0.1) : twice);
    }
    return sum;
}

struct Run {
    std::uint32_t checksum          = 0U;
    std::uint64_t gpuDispatches     = 0U;
    std::uint64_t deviceToHost      = 0U;
    std::int64_t  deviceOnlyBytes   = 0; // device-only memory the graph holds after its run: where the scale-to-sink edge lives
    bool          completedNormally = false;
};

template<bool refusedStage = false>
[[nodiscard]] Run checksumOn(std::string_view domain, gr::Size_t nSamples) {
    gr::device::DeviceContext* const       context            = domain == "host" ? nullptr : gr::device::DeviceContextRegistry::instance().tryResolve(domain);
    gr::device::DeviceContextWebGpu* const webGpu             = context == nullptr ? nullptr : context->as<gr::device::DeviceContextWebGpu>();
    const std::uint64_t                    dispatchesBefore   = webGpu == nullptr ? 0U : webGpu->gpuDispatches.load();
    const std::uint64_t                    deviceToHostBefore = webGpu == nullptr ? 0U : webGpu->deviceToHostBytes.load();
    const auto                             liveDeviceOnly     = [] {
        const std::vector<std::uint32_t> ranges = gr::device::webGpuDeviceMemory().liveRanges();
        std::int64_t                     bytes  = 0;
        for (std::size_t r = 0UZ; r < ranges.size(); r += 2UZ) {
            bytes += ranges[r + 1UZ] - ranges[r];
        }
        return bytes;
    };
    const std::int64_t deviceOnlyBefore = liveDeviceOnly();

    gr::Graph flow;
    auto&     source = flow.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     scale  = flow.emplaceBlock<Scale>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink   = flow.emplaceBlock<DeviceChecksumSink>({{"gr:compute_domain", std::string(domain)}});
    std::ignore      = flow.connect<"out", "in">(source, scale);
    if constexpr (refusedStage) { // between two device-only edges: its CPU fallback must see and write the GPU's samples
        auto& refused = flow.emplaceBlock<gr::testing::GainInDouble>({{"gr:compute_domain", std::string(domain)}});
        std::ignore   = flow.connect<"out", "in">(scale, refused);
        std::ignore   = flow.connect<"out", "in">(refused, sink);
    } else {
        std::ignore = flow.connect<"out", "in">(scale, sink);
    }

    gr::scheduler::Simple<> scheduler;
    std::ignore    = scheduler.exchange(std::move(flow));
    const auto ran = scheduler.runAndWait();
    return Run{.checksum = sink.checksum, .gpuDispatches = webGpu == nullptr ? 0U : webGpu->gpuDispatches.load() - dispatchesBefore, .deviceToHost = webGpu == nullptr ? 0U : webGpu->deviceToHostBytes.load() - deviceToHostBefore, .deviceOnlyBytes = liveDeviceOnly() - deviceOnlyBefore, .completedNormally = ran.has_value()};
}

} // namespace gr::webgpu_test

int main() {
    using namespace boost::ut;
    using namespace gr::webgpu_test;
    gr::testing::registerDeviceRuntimes();

    "the checksum sink on the host finds the worked-out sum"_test = [] {
        constexpr gr::Size_t kN  = 4096U;
        const Run            run = checksumOn("host", kN);
        expect(run.completedNormally);
        expect(eq(run.checksum, expectedChecksum(kN))) << "host: the sum of the bit patterns of 2i";
    };

    "a sink on the GPU checksums samples that stay in GPU memory"_test = [] {
        constexpr gr::Size_t kN = 4096U;
        for (const std::string_view domain : gr::testing::kWebGpuDomainNames) {
            if (!gr::testing::isServedForTest(domain)) {
                continue;
            }
            const Run run = checksumOn(domain, kN);
            expect(run.completedNormally) << domain;
            expect(eq(run.checksum, expectedChecksum(kN))) << domain << ": the kernel saw every sample the scale wrote";
            expect(ge(run.gpuDispatches, 2UZ)) << domain << ": the scale and the checksum ran on the adapter";
            expect(ge(run.deviceOnlyBytes, static_cast<std::int64_t>(kN * sizeof(float)))) << domain << ": the scale-to-sink edge is device-only memory";
            expect(eq(run.deviceToHost, 0UZ)) << domain << ": no sample of the device-only edge came back to the host";
        }
    };

    "a refused kernel between two device-only edges computes on the GPU's samples"_test = [] {
        constexpr gr::Size_t kN = 4096U;
        for (const std::string_view domain : gr::testing::kWebGpuDomainNames) {
            if (!gr::testing::isServedForTest(domain)) {
                continue;
            }
            const Run run = checksumOn<true>(domain, kN);
            expect(run.completedNormally) << domain;
            expect(eq(run.checksum, expectedChecksum(kN, true))) << domain << ": the fallback read the scale's output and the sink read the fallback's";
        }
    };

    "uploads to device-only memory, back to back and of growing size, read back as written"_test = [] {
        // back to back, the first two uploads take the two staging buffers and the third, before the GPU copied either
        // out, waits for one; after the read-backs a larger fourth grows a staging buffer
        constexpr std::array<std::size_t, 4UZ> kWords{1024UZ, 1024UZ, 1024UZ, 300'000UZ};
        const auto                             pattern = [](std::size_t upload, std::size_t words) { return std::views::iota(0UZ, words) | std::views::transform([upload](std::size_t w) { return static_cast<std::uint32_t>(w * 2654435761UZ + upload); }) | std::ranges::to<std::vector>(); };
        for (const std::string_view domain : gr::testing::kWebGpuDomainNames) {
            if (!gr::testing::isServedForTest(domain)) {
                continue;
            }
            gr::device::DeviceContext& ctx       = *gr::device::DeviceContextRegistry::instance().tryResolve(domain);
            const auto                 roundTrip = [&](std::span<const std::size_t> uploads, std::size_t first) {
                std::vector<gr::device::DeviceBuffer> buffers;
                for (std::size_t i = 0UZ; i < uploads.size(); ++i) {
                    const std::vector<std::uint32_t> written = pattern(first + i, uploads[i]);
                    buffers.push_back(ctx.allocate(uploads[i] * sizeof(std::uint32_t), alignof(std::uint32_t), gr::device::Residency::devicePtr));
                    ctx.upload(written.data(), buffers.back(), uploads[i] * sizeof(std::uint32_t));
                }
                for (std::size_t i = 0UZ; i < uploads.size(); ++i) {
                    std::vector<std::uint32_t> read(uploads[i]);
                    ctx.download(buffers[i], read.data(), uploads[i] * sizeof(std::uint32_t));
                    expect(read == pattern(first + i, uploads[i])) << std::format("{}: upload {} of {} words", domain, first + i, uploads[i]);
                    ctx.deallocate(buffers[i]);
                }
            };
            roundTrip(std::span(kWords).first(3UZ), 0UZ);
            roundTrip(std::span(kWords).last(1UZ), 3UZ);
            expect(!ctx.peekDeviceError()) << domain;
        }
    };

    "freeing WebGPU memory first runs the launches still queued, which may write it"_test = [] {
        constexpr std::size_t kN = 1024UZ;
        for (const std::string_view domain : gr::testing::kWebGpuDomainNames) {
            if (!gr::testing::isServedForTest(domain)) {
                continue;
            }
            gr::device::DeviceContext& ctx     = *gr::device::DeviceContextRegistry::instance().tryResolve(domain);
            gr::device::DeviceBuffer   written = ctx.allocate(kN * sizeof(std::uint32_t), alignof(std::uint32_t), gr::device::Residency::shared);
            gr::device::DeviceBuffer   freed   = ctx.allocate(64UZ, alignof(std::uint32_t), gr::device::Residency::devicePtr);
            std::uint32_t* const       out     = written.devicePointer<std::uint32_t>();
            const auto                 fill    = [out](std::size_t i) { out[i] = static_cast<std::uint32_t>(i) + 1U; };
            gr::device::parallelFor(ctx, kN, fill, true); // a first launch builds its pipeline and is never queued
            std::ranges::fill(std::span(out, kN), 0U);
            gr::device::parallelFor(ctx, kN, fill, false);
            expect(eq(out[kN - 1UZ], 0U)) << domain << ": queued, not yet run";
            ctx.deallocate(freed);
            expect(eq(out[kN - 1UZ], static_cast<std::uint32_t>(kN))) << domain << ": the free ran it, so its results are in wasm memory";
            ctx.deallocate(written);
            expect(!ctx.peekDeviceError()) << domain;
        }
    };

#if defined(__EMSCRIPTEN__)
    "a free on the browser main thread, which cannot wait, is held until the next flush runs the queued launches"_test = [] {
        constexpr std::size_t kN = 1024UZ;
        for (const std::string_view domain : gr::testing::kWebGpuDomainNames) {
            if (!gr::testing::isServedForTest(domain)) {
                continue;
            }
            gr::device::DeviceContext& ctx     = *gr::device::DeviceContextRegistry::instance().tryResolve(domain);
            gr::device::DeviceBuffer   written = ctx.allocate(kN * sizeof(std::uint32_t), alignof(std::uint32_t), gr::device::Residency::shared);
            gr::device::DeviceBuffer   freed   = ctx.allocate(64UZ, alignof(std::uint32_t), gr::device::Residency::devicePtr);
            std::uint32_t* const       out     = written.devicePointer<std::uint32_t>();
            const void* const          address = freed.devicePointer<std::byte>();
            const auto                 fill    = [out](std::size_t i) { out[i] = static_cast<std::uint32_t>(i) + 1U; };
            gr::device::parallelFor(ctx, kN, fill, true); // a first launch builds its pipeline and is never queued
            std::ranges::fill(std::span(out, kN), 0U);
            gr::device::parallelFor(ctx, kN, fill, false);

            struct Release {
                gr::device::DeviceContext* ctx;
                gr::device::DeviceBuffer   buffer;
            } release{.ctx = &ctx, .buffer = freed};
            emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VI, +[](void* arg) { static_cast<Release*>(arg)->ctx->deallocate(static_cast<Release*>(arg)->buffer); }, &release);
            expect(gr::device::webGpuDeviceMemory().contains(address)) << domain << ": held, as a queued launch may still write it";
            expect(eq(out[kN - 1UZ], 0U)) << domain << ": the free on the main thread ran nothing";
            ctx.wait();
            expect(!gr::device::webGpuDeviceMemory().contains(address)) << domain << ": released by the flush";
            expect(eq(out[kN - 1UZ], static_cast<std::uint32_t>(kN))) << domain << ": the queued launch ran";
            ctx.deallocate(written);
            expect(!ctx.peekDeviceError()) << domain;
        }
    };

    "a JavaScript consumer on the browser main thread reads device-only samples in the GPU buffer by their address"_test = [] {
        constexpr std::size_t kN            = 4096UZ;
        constexpr float       kExpectedSum  = 12285.f; // sum of i mod 7 over i < 4096: 585 whole cycles of 0..6, then a 0
        constexpr auto        kReadDeadline = std::chrono::seconds(30);
        for (const std::string_view domain : gr::testing::kWebGpuDomainNames) {
            if (!gr::testing::isServedForTest(domain)) {
                continue;
            }
            gr::device::DeviceContext&             ctx     = *gr::device::DeviceContextRegistry::instance().tryResolve(domain);
            const gr::device::DeviceContextWebGpu* webGpu  = ctx.as<gr::device::DeviceContextWebGpu>();
            gr::device::DeviceBuffer               samples = ctx.allocateDevice<float>(kN);
            float* const                           data    = samples.devicePointer<float>();
            gr::device::parallelFor(ctx, kN, [data](std::size_t i) { data[i] = static_cast<float>(i % 7U); }, true);
            const std::uint64_t deviceToHostBefore = webGpu->deviceToHostBytes.load();

            std::atomic<int> read{0}; // 1: read; -1: no accessor, buffer or generation; -2: the map failed; -3: start-up answers another device
            float            sum = -1.f;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension" // EM_ASM names its arguments $0, $1, ...
            // clang-format off: the macro's argument is JavaScript
            MAIN_THREAD_ASYNC_EM_ASM({
                const api = Module["gr4WebGpuDevice"]?.();
                if (!api?.heap || !(api.generation >= 1)) {
                    Atomics.store(HEAP32, $2 >> 2, -1);
                    return;
                }
                const copy = api.device.createBuffer({ size: $1, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
                const encoder = api.device.createCommandEncoder();
                encoder.copyBufferToBuffer(api.heap, $0, copy, 0, $1);
                api.device.queue.submit([encoder.finish()]);
                Module["gr4WebGpuStart"]().then((started) => {
                    if (started !== api.device || api.error !== "" || !(api.heap.usage & GPUBufferUsage.VERTEX)) {
                        Atomics.store(HEAP32, $2 >> 2, -3);
                        return;
                    }
                    return copy.mapAsync(GPUMapMode.READ).then(() => {
                        HEAPF32[$3 >> 2] = new Float32Array(copy.getMappedRange()).reduce((a, b) => a + b, 0);
                        copy.destroy();
                        Atomics.store(HEAP32, $2 >> 2, 1);
                    }, () => Atomics.store(HEAP32, $2 >> 2, -2));
                });
            }, data, kN * sizeof(float), &read, &sum);
            // clang-format on
#pragma clang diagnostic pop
            const auto deadline = std::chrono::steady_clock::now() + kReadDeadline;
            while (read.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
            expect(eq(read.load(), 1)) << domain << ": Module.gr4WebGpuDevice() serves the started device, error-free, and the drawable buffer, within the deadline";
            expect(eq(sum, kExpectedSum)) << domain << ": the samples, read on the GPU at their wasm address";
            expect(eq(webGpu->deviceToHostBytes.load(), deviceToHostBefore)) << domain << ": no download through the context";
            ctx.deallocate(samples);
        }
    };
#endif

    "bridgedRanges closes gaps up to the limit, except one holding device-only memory"_test = [] {
        const std::vector<std::uint32_t> ranges{0U, 16U, 32U, 48U, 100U, 116U, 8192U, 8208U};
        const std::vector<std::uint32_t> keepOut{64U, 80U};
        expect(gr::device::bridgedRanges(ranges, keepOut, 4096U) == std::vector<std::uint32_t>{0U, 48U, 100U, 116U, 8192U, 8208U}) << "16 bytes bridged; [48, 100) holds [64, 80); [116, 8192) exceeds 4096 bytes";
        expect(gr::device::bridgedRanges(ranges, {}, 8076U) == std::vector<std::uint32_t>{0U, 8208U}) << "every gap is at most 8,076 bytes";
        expect(gr::device::bridgedRanges(ranges, std::vector<std::uint32_t>{16U, 32U, 116U, 8192U}, 8192U) == std::vector<std::uint32_t>{0U, 16U, 32U, 116U, 8192U, 8208U}) << "a keep-out filling a gap exactly";
        expect(gr::device::bridgedRanges(ranges, std::vector<std::uint32_t>{0U, 16U, 48U, 64U}, 4096U) == std::vector<std::uint32_t>{0U, 48U, 100U, 116U, 8192U, 8208U}) << "a keep-out ending where a gap begins lies outside it, one beginning there inside";
        expect(gr::device::bridgedRanges({}, keepOut, 4096U).empty());
    };

    "WebGpuMemory merges live allocations into word-aligned ranges and forgets freed ones"_test = [] {
        gr::device::WebGpuMemory memory;
        void* const              a   = memory.allocate(10UZ, 16UZ);
        void* const              b   = memory.allocate(100UZ, 16UZ);
        const auto               end = [](const void* p, std::size_t bytes) { return reinterpret_cast<std::uintptr_t>(p) + bytes; };
        expect(memory.contains(a) && memory.contains(static_cast<const std::byte*>(b) + 99));
        expect(!memory.contains(static_cast<const std::byte*>(b) + 100)) << "one past the end is not live";
        const std::vector<std::uint32_t> ranges = memory.liveRanges();
        expect(eq(ranges.size() % 2UZ, 0UZ) && !ranges.empty());
        expect(std::ranges::all_of(ranges, [](std::uint32_t edge) { return edge % 4U == 0U; })) << "every edge is word-aligned";
        expect(eq(memory.highestEnd(), static_cast<std::uint32_t>(std::max(end(a, 10UZ), end(b, 100UZ))))) << "the highest allocation's end, narrowed to 32 bits after comparing: on a 64-bit host the low words need not keep the order";
        memory.deallocate(a, 10UZ, 16UZ);
        const std::vector<std::uint32_t> onlyB{static_cast<std::uint32_t>(end(b, 0UZ)), static_cast<std::uint32_t>(end(b, 100UZ))}; // 16-aligned and 100 bytes: both edges are words
        expect(memory.liveRanges() == onlyB) << "a freed allocation is no longer moved by a launch";
        memory.deallocate(b, 100UZ, 16UZ);
        expect(memory.liveRanges().empty());
        expect(eq(memory.highestEnd(), 0U));
    };

    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
