#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <boost/ut.hpp>

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

/// a double setting: a kernel the translator refuses, which the WebGPU context runs on the CPU
struct ScaleInDouble : Block<ScaleInDouble> {
    PortIn<float>  in;
    PortOut<float> out;

    Annotated<double, "gain"> gain = 0.1;
    GR_MAKE_REFLECTABLE(ScaleInDouble, in, out, gain);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return static_cast<float>(static_cast<double>(x) * gain.value); }
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
    std::uint64_t deviceOnlyBytes   = 0U; // allocated as device-only memory during the run: where the scale-to-sink edge lives
    bool          completedNormally = false;
};

template<bool kRefusedStage = false>
[[nodiscard]] Run checksumOn(std::string_view domain, gr::Size_t nSamples) {
    gr::device::DeviceContext* const       context            = domain == "host" ? nullptr : gr::device::DeviceContextRegistry::instance().tryResolve(domain);
    gr::device::DeviceContextWebGpu* const webGpu             = context == nullptr ? nullptr : context->as<gr::device::DeviceContextWebGpu>();
    const std::uint64_t                    dispatchesBefore   = webGpu == nullptr ? 0U : webGpu->gpuDispatches.load();
    const std::uint64_t                    deviceToHostBefore = webGpu == nullptr ? 0U : webGpu->deviceToHostBytes.load();
    const std::uint64_t                    deviceOnlyBefore   = gr::device::webGpuDeviceMemory().allocatedBytes.load();

    gr::Graph flow;
    auto&     source = flow.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     scale  = flow.emplaceBlock<Scale>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink   = flow.emplaceBlock<DeviceChecksumSink>({{"gr:compute_domain", std::string(domain)}});
    std::ignore      = flow.connect<"out", "in">(source, scale);
    if constexpr (kRefusedStage) { // between two device-only edges: its CPU fallback must see and write the GPU's samples
        auto& refused = flow.emplaceBlock<ScaleInDouble>({{"gr:compute_domain", std::string(domain)}});
        std::ignore   = flow.connect<"out", "in">(scale, refused);
        std::ignore   = flow.connect<"out", "in">(refused, sink);
    } else {
        std::ignore = flow.connect<"out", "in">(scale, sink);
    }

    gr::scheduler::Simple<> scheduler;
    std::ignore    = scheduler.exchange(std::move(flow));
    const auto ran = scheduler.runAndWait();
    return Run{.checksum = sink.checksum, .gpuDispatches = webGpu == nullptr ? 0U : webGpu->gpuDispatches.load() - dispatchesBefore, .deviceToHost = webGpu == nullptr ? 0U : webGpu->deviceToHostBytes.load() - deviceToHostBefore, .deviceOnlyBytes = gr::device::webGpuDeviceMemory().allocatedBytes.load() - deviceOnlyBefore, .completedNormally = ran.has_value()};
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
            if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
                expect(!gr::testing::deviceDomainRequired(domain)) << std::format("GR4_REQUIRE_DEVICE names '{}', so it must be served", domain);
                continue;
            }
            const Run run = checksumOn(domain, kN);
            expect(run.completedNormally) << domain;
            expect(eq(run.checksum, expectedChecksum(kN))) << domain << ": the kernel saw every sample the scale wrote";
            expect(ge(run.gpuDispatches, 2UZ)) << domain << ": the scale and the checksum ran on the adapter";
            expect(ge(run.deviceOnlyBytes, std::uint64_t{kN * sizeof(float)})) << domain << ": the scale-to-sink edge was device-only memory";
            expect(eq(run.deviceToHost, 0UZ)) << domain << ": no sample of the device-only edge came back to the host";
        }
    };

    "a refused kernel between two device-only edges computes on the GPU's samples"_test = [] {
        constexpr gr::Size_t kN = 4096U;
        for (const std::string_view domain : gr::testing::kWebGpuDomainNames) {
            if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
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
            if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
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
            if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
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
        expect(!memory.contains(a)) << "a freed allocation is no longer moved by a launch";
        memory.deallocate(b, 100UZ, 16UZ);
        expect(memory.liveRanges().empty());
        expect(eq(memory.highestEnd(), 0U));
    };

    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
