#include <boost/ut.hpp>

#include <algorithm>
#include <format>
#include <memory_resource>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "device_test_helpers.hpp"

/*
 * The three ways to write a block that runs on an accelerator; see docs/USER_API_GPU_Blocks.md.
 * Each style is exercised through a real graph, on whichever backends this build actually has.
 */

namespace gr::styles {

/// style 1 — the framework wraps `processOne` in a kernel. Nothing here is device-specific.
struct Gain : Block<Gain> {
    PortIn<float>  in;
    PortOut<float> out;

    Annotated<float, "gain"> gain = 3.f;
    GR_MAKE_REFLECTABLE(Gain, in, out, gain);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x * gain; }
};

/// style 1b — array settings live in a pmr container, which the framework re-seats onto device memory at init()
struct Biquad : Block<Biquad> {
    PortIn<float>  in;
    PortOut<float> out;

    std::pmr::vector<float> taps;
    GR_MAKE_REFLECTABLE(Biquad, in, out, taps);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return taps.size() < 2UZ ? x : x * taps[0] + taps[1]; }
};

/// style 2 — not a kernel body: it runs on the host, owns the queue, and submits its own work
struct SyclGain : Block<SyclGain> {
    PortIn<float>  in;
    PortOut<float> out;

    Annotated<float, "gain"> gain = 3.f;
    GR_MAKE_REFLECTABLE(SyclGain, in, out, gain);

    /// the CPU path every block needs: `processBulk_sycl` only runs when a SYCL backend serves the domain
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t count = std::min(input.size(), output.size());
        for (std::size_t i = 0UZ; i < count; ++i) {
            output[i] = input[i] * gain;
        }
        std::ignore = input.consume(count);
        output.publish(count);
        return gr::work::Status::OK;
    }

    [[nodiscard]] gr::work::Status processBulk_sycl(gr::device::SyclQueue& queue, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t count = std::min(input.size(), output.size());
        queue.memcpy(output.data(), input.data(), count * sizeof(float)).wait();
        for (std::size_t i = 0UZ; i < count; ++i) { // a real block would submit a kernel here
            output[i] *= gain;
        }
        std::ignore = input.consume(count);
        output.publish(count);
        return gr::work::Status::OK;
    }
};

/// the settings demo: one plain-C++ block reading a scalar, an integral and a pmr collection from the SAME kernel.
/// Gain and Biquad each cover one kind; a real DSP block has all three at once, which is what this pins down.
struct Mixer : Block<Mixer> {
    PortIn<float>  in;
    PortOut<float> out;

    Annotated<float, "gain">     gain = 2.f;
    Annotated<gr::Size_t, "tap"> tap  = 1U;
    std::pmr::vector<float>      taps{};

    GR_MAKE_REFLECTABLE(Mixer, in, out, gain, tap, taps);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return tap < taps.size() ? x * gain + taps[tap] : x * gain; }
};

/// the same three settings, read from the framework-managed `processBulk` tier instead of `processOne`
struct MixerBulk : Block<MixerBulk> {
    PortIn<float>  in;
    PortOut<float> out;

    Annotated<float, "gain">     gain = 2.f;
    Annotated<gr::Size_t, "tap"> tap  = 1U;
    std::pmr::vector<float>      taps{};

    GR_MAKE_REFLECTABLE(MixerBulk, in, out, gain, tap, taps);

    [[nodiscard]] gr::work::Status processBulk(gr::InputViewLike auto& input, gr::OutputViewLike auto& output) const noexcept {
        const float       bias  = tap < taps.size() ? taps[tap] : 0.f;
        const std::size_t count = std::min(input.size(), output.size());
        for (std::size_t i = 0UZ; i < count; ++i) {
            output[i] = input[i] * gain + bias;
        }
        return gr::work::Status::OK;
    }
};

/// style 1c — `processBulk` as a framework-managed kernel body: `const`, because it shares the device mirror.
/// the framework moves the data and hands the block the same span it would see on the CPU; constrained on the view
/// concepts, so one definition serves host and device, since a real host span satisfies them too.
struct SpanSum : Block<SpanSum> {
    PortIn<float>  in;
    PortOut<float> out;

    GR_MAKE_REFLECTABLE(SpanSum, in, out);

    [[nodiscard]] gr::work::Status processBulk(gr::InputViewLike auto& input, gr::OutputViewLike auto& output) const noexcept {
        float sum = 0.f;
        for (std::size_t i = 0UZ; i < input.size(); ++i) {
            sum += input[i];
        }
        for (std::size_t i = 0UZ; i < output.size(); ++i) {
            output[i] = sum; // every sample carries the span's sum
        }
        return gr::work::Status::OK;
    }
};

/// two inputs through the framework-managed view tier. `a - 2b` is asymmetric, so ports crossed on the way in
/// cannot produce the expected stream
struct WeightedDifferenceBulk : Block<WeightedDifferenceBulk> {
    PortIn<float>  in0;
    PortIn<float>  in1;
    PortOut<float> out;

    GR_MAKE_REFLECTABLE(WeightedDifferenceBulk, in0, in1, out);

    [[nodiscard]] gr::work::Status processBulk(gr::InputViewLike auto& a, gr::InputViewLike auto& b, gr::OutputViewLike auto& output) const noexcept {
        const std::size_t count = std::min({a.size(), b.size(), output.size()});
        for (std::size_t i = 0UZ; i < count; ++i) {
            output[i] = a[i] - 2.f * b[i];
        }
        return gr::work::Status::OK;
    }
};

/// two inputs and a setting that cannot travel: it must still run, on the CPU, rather than fail the graph
struct TwoInputNonRelocatable : Block<TwoInputNonRelocatable> {
    PortIn<float>  in0;
    PortIn<float>  in1;
    PortOut<float> out;

    Annotated<std::string, "label"> label = "held on the host"; // SSO storage lives inside the object

    GR_MAKE_REFLECTABLE(TwoInputNonRelocatable, in0, in1, out, label);

    [[nodiscard]] constexpr float processOne(float a, float b) const noexcept { return a - 2.f * b; }
};

} // namespace gr::styles

namespace {

struct ChainResult {
    std::vector<float>                  samples;
    std::vector<gr::testing::OwningTag> tags;
};

/// run `source -> dut -> sink` on `domain`, optionally publishing `sourceTags` from the source, and return what the sink saw
template<typename TBlock, typename TConfigure>
[[nodiscard]] ChainResult runChainFull(std::string_view domain, gr::Size_t nSamples, TConfigure configure, std::vector<gr::testing::OwningTag> sourceTags = {}) {
    using namespace gr::testing;

    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<TBlock>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}, {"log_samples", true}, {"log_tags", true}});
    configure(dut);
    source._tags = std::move(sourceTags);

    std::ignore = flow.connect<"out", "in">(source, dut);
    std::ignore = flow.connect<"out", "in">(dut, sink);

    gr::scheduler::Simple<> sched;
    std::ignore = sched.exchange(std::move(flow));
    std::ignore = sched.runAndWait();

    ChainResult result;
    result.samples.resize(sink._samples.size());
    for (std::size_t i = 0UZ; i < result.samples.size(); ++i) {
        result.samples[i] = sink._samples[i];
    }
    result.tags = sink._tags;
    return result;
}

/// run `source -> dut -> sink` on `domain` and return the samples the sink saw
template<typename TBlock, typename TConfigure>
[[nodiscard]] std::vector<float> runChain(std::string_view domain, gr::Size_t nSamples, TConfigure configure) {
    return runChainFull<TBlock>(domain, nSamples, std::move(configure)).samples;
}

[[nodiscard]] bool sameSamples(const std::vector<float>& lhs, const std::vector<float>& rhs) { return lhs.size() == rhs.size() && std::ranges::equal(lhs, rhs); }

[[nodiscard]] bool domainIsServed(std::string_view domain) { return gr::device::DeviceContextRegistry::instance().tryResolve(domain) != nullptr; }

} // namespace

// AdaptiveCpp aborts if a kernel is launched while Boost.UT is running suites from ~runner (static destruction),
// so the tests are registered and run from main() -- see gotcha G10.
int main() {
    using namespace boost::ut;

    const bool syclAvailable = gr::device::registerSyclRuntime();

    "a multi-port block that cannot be relocated still runs, on the CPU"_test = [syclAvailable] {
        if (!syclAvailable) {
            return;
        }
        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return;
        }
        using namespace gr::testing;
        constexpr gr::Size_t kN = 64U;

        std::vector<float> samples;
        const auto         fallbacks = gr::test::cpuFallbacksDuring([&] {
            gr::Graph flow;
            auto&     sourceA = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
            auto&     sourceB = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
            auto&     scaleB  = flow.emplaceBlock<gr::styles::Gain>({{"gain", 10.f}}); // the arms must differ, or a - 2b is the same for every wiring
            auto&     dut     = flow.emplaceBlock<gr::styles::TwoInputNonRelocatable>({{"gr:compute_domain", std::string(*servedDomain)}});
            auto&     sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});
            expect(flow.connect<"out", "in0">(sourceA, dut).has_value());
            expect(flow.connect<"out", "in">(sourceB, scaleB).has_value());
            expect(flow.connect<"out", "in1">(scaleB, dut).has_value());
            expect(flow.connect<"out", "in">(dut, sink).has_value());
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            expect(sched.runAndWait().has_value()) << "a block the device cannot take must not fail the graph";
            samples.assign(sink._samples.begin(), sink._samples.end());
        });

        expect(gt(fallbacks, 0UZ)) << "and it must say so rather than substitute the CPU silently";
        bool valuesOk = samples.size() == static_cast<std::size_t>(kN);
        for (std::size_t i = 0UZ; valuesOk && i < samples.size(); ++i) {
            valuesOk = samples[i] == static_cast<float>(i) - 2.f * (static_cast<float>(i) * 10.f);
        }
        expect(valuesOk) << "the CPU fallback must serve every declared port, not just the first";
    };

    "style 1c with two inputs: the view tier serves every declared port"_test = [syclAvailable] {
        if (!syclAvailable) {
            return;
        }
        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return;
        }
        using namespace gr::testing;
        const std::string    computeDomain(*servedDomain);
        constexpr gr::Size_t kN = 256U;

        const auto runIt = [&computeDomain, kN](std::vector<float>& samples) {
            gr::Graph flow;
            auto&     sourceA = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
            auto&     sourceB = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
            auto&     scaleB  = flow.emplaceBlock<gr::styles::Gain>({{"gr:compute_domain", computeDomain}, {"gain", 10.f}});
            auto&     combine = flow.emplaceBlock<gr::styles::WeightedDifferenceBulk>({{"gr:compute_domain", computeDomain}});
            auto&     sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});

            expect(flow.connect<"out", "in0">(sourceA, combine).has_value());
            expect(flow.connect<"out", "in">(sourceB, scaleB).has_value());
            expect(flow.connect<"out", "in1">(scaleB, combine).has_value());
            expect(flow.connect<"out", "in">(combine, sink).has_value());

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            expect(sched.runAndWait().has_value());
            samples.assign(sink._samples.begin(), sink._samples.end());
        };

        std::vector<float> samples;
        expect(eq(gr::test::cpuFallbacksDuring([&] { runIt(samples); }), 0UZ)) << "a two-input view body that fell back would return these same numbers";

        bool valuesOk = samples.size() == static_cast<std::size_t>(kN);
        for (std::size_t i = 0UZ; valuesOk && i < samples.size(); ++i) {
            valuesOk = samples[i] == static_cast<float>(i) - 2.f * (static_cast<float>(i) * 10.f);
        }
        expect(valuesOk) << "both input views reached the kernel, in the order the block declares them";
    };

    "style 1: an ordinary processOne runs unchanged on every backend"_test = [] {
        constexpr gr::Size_t kN     = 32U;
        const auto           onHost = runChain<gr::styles::Gain>("host", kN, [](auto&) {});
        expect(eq(onHost.size(), static_cast<std::size_t>(kN)));
        expect(eq(onHost[4], 12.f)) << "4 * gain(3)";

        if (const auto domain = gr::test::firstServedSyclDomain()) {
            const auto onDevice = runChain<gr::styles::Gain>(*domain, kN, [](auto&) {});
            expect(sameSamples(onDevice, onHost)) << "the same block, the same answer, on the device";
            expect(eq(gr::test::cpuFallbacksDuring([&] { std::ignore = runChain<gr::styles::Gain>(*domain, kN, [](auto&) {}); }), 0UZ)) << "matching answers alone would also hold if the block had quietly fallen back";
        }
    };

    "style 1b: pmr array settings are re-seated onto device memory and read by the kernel"_test = [syclAvailable] {
        constexpr gr::Size_t kN      = 32U;
        const auto           setTaps = [](gr::styles::Biquad& b) { b.taps = std::pmr::vector<float>{2.f, 1.f}; };

        const auto onHost = runChain<gr::styles::Biquad>("host", kN, setTaps);
        expect(eq(onHost[3], 7.f)) << "3 * taps[0](2) + taps[1](1)";

        if (syclAvailable && domainIsServed("gpu:sycl")) {
            expect(sameSamples(runChain<gr::styles::Biquad>("gpu:sycl", kN, setTaps), onHost)) << "the kernel indexes the migrated taps";
        }
        if (syclAvailable && domainIsServed("host:sycl")) {
            expect(sameSamples(runChain<gr::styles::Biquad>("host:sycl", kN, setTaps), onHost)) << "host memory, SYCL execution";
        }
    };

    "style 1c: a const processBulk is moved to the device and computes what the CPU computes"_test = [] {
        constexpr gr::Size_t kN = 32U;

        const auto onHost = runChain<gr::styles::SpanSum>("host", kN, [](auto&) {});
        expect(eq(onHost.size(), static_cast<std::size_t>(kN)));

        if (const auto domain = gr::test::firstServedSyclDomain()) {
            expect(sameSamples(runChain<gr::styles::SpanSum>(*domain, kN, [](auto&) {}), onHost)) << "the device path must not change what the block computes";
        }
    };

    "a plain-C++ block reads a scalar, an integral and a pmr collection from one kernel"_test = [syclAvailable] {
        // the crucial demo: nothing in Mixer mentions a device, and the same source computes the same answer
        // whether the framework runs it on the CPU or wraps it in a kernel.
        constexpr gr::Size_t kN        = 32U;
        const auto           configure = [](auto& block) {
            block.gain = 4.f;
            block.tap  = 2U;
            block.taps = std::pmr::vector<float>{0.f, 10.f, 100.f};
        };

        const auto onHost = runChain<gr::styles::Mixer>("host", kN, configure);
        expect(eq(onHost.size(), static_cast<std::size_t>(kN)));
        expect(eq(onHost[3], 112.f)) << "3 * gain(4) + taps[tap(2)](100) -- all three settings reached the body";

        for (std::string_view domain : {"gpu:sycl", "host:sycl"}) {
            if (!syclAvailable || !domainIsServed(domain)) {
                continue;
            }
            expect(sameSamples(runChain<gr::styles::Mixer>(domain, kN, configure), onHost)) //
                << std::format("scalar, integral and pmr settings all read identically on {}", domain);
        }
    };

    "the same mixed settings read from the framework-managed processBulk tier"_test = [syclAvailable] {
        constexpr gr::Size_t kN        = 32U;
        const auto           configure = [](auto& block) {
            block.gain = 4.f;
            block.tap  = 2U;
            block.taps = std::pmr::vector<float>{0.f, 10.f, 100.f};
        };

        const auto onHost = runChain<gr::styles::MixerBulk>("host", kN, configure);
        expect(eq(onHost.size(), static_cast<std::size_t>(kN)));
        expect(eq(onHost[3], 112.f)) << "the bulk tier reads the same three settings as processOne";

        for (std::string_view domain : {"gpu:sycl", "host:sycl"}) {
            if (!syclAvailable || !domainIsServed(domain)) {
                continue;
            }
            expect(sameSamples(runChain<gr::styles::MixerBulk>(domain, kN, configure), onHost)) //
                << std::format("a const processBulk reads its settings on {} exactly as it does on the host", domain);
        }
    };

    "style 3: processBulk_sycl owns the queue and runs on the host thread"_test = [] {
        constexpr gr::Size_t kN     = 32U;
        const auto           domain = gr::test::firstServedSyclDomain();
        if (!domain) {
            return;
        }
        const auto samples = runChain<gr::styles::SyclGain>(*domain, kN, [](auto&) {});
        expect(eq(samples.size(), static_cast<std::size_t>(kN)));
        expect(eq(samples[5], 15.f)) << "5 * gain(3), computed through the user's own submission";
    };

    "tags flow through a device block on every served tier"_test = [] {
        // tags are forwarded on the host before dispatch and never enter a kernel. N.B. forwardInputTags() only
        // forwards `gr:`-prefixed or autoForwardParameters()-registered keys, so a bare "kind" is dropped on every
        // tier, host included.
        constexpr gr::Size_t  kN      = 32U;
        constexpr std::size_t kTagIdx = 5UZ;
        gr::property_map      payload;
        gr::tag::put(payload, "gr:kind", gr::pmt::Value("device-tag"));
        const std::vector<gr::testing::OwningTag> tags{gr::testing::OwningTag{kTagIdx, payload}};

        const auto expectTagPreserved = [kTagIdx]<typename TBlock>(std::string_view domain, const std::vector<gr::testing::OwningTag>& sourceTags, std::string_view label) {
            const auto onHost   = runChainFull<TBlock>("host", kN, [](auto&) {}, sourceTags);
            const auto onDevice = runChainFull<TBlock>(domain, kN, [](auto&) {}, sourceTags);
            expect(eq(onHost.tags.size(), 1UZ)) << label;
            expect(eq(onDevice.tags.size(), 1UZ)) << label;
            if (onHost.tags.size() != 1UZ || onDevice.tags.size() != 1UZ) {
                return;
            }
            expect(eq(onHost.tags[0].index, kTagIdx)) << label;
            expect(eq(onDevice.tags[0].index, onHost.tags[0].index)) << label << ": tag must arrive at the same sample index as on the host";
            expect(onDevice.tags[0].map == onHost.tags[0].map) << label << ": tag payload must be unchanged";
        };

        if (const auto domain = gr::test::firstServedSyclDomain()) {
            expectTagPreserved.operator()<gr::styles::Gain>(*domain, tags, "style 1 (processOne, tier: auto-parallel) on SYCL");
            expectTagPreserved.operator()<gr::styles::SpanSum>(*domain, tags, "style 1c (processBulk, tier: framework device-bulk) on SYCL");
        }
    };

    "a settings change mid-stream reaches the device mirror"_test = [] {
        // the device mirror only refreshes when the block's settings epoch moves, so a setting
        // changed by a tag while the graph is running must still take effect from that sample on.
        constexpr gr::Size_t  kN           = 32U;
        constexpr std::size_t kChangeAt    = 16UZ;
        constexpr float       kInitialGain = 3.f; // gr::styles::Gain's default
        constexpr float       kUpdatedGain = 5.f;

        const std::vector<gr::testing::OwningTag> tags{gr::testing::OwningTag{kChangeAt, gr::property_map{{"gain", kUpdatedGain}}}};

        const auto onHost = runChainFull<gr::styles::Gain>("host", kN, [](auto&) {}, tags);
        expect(eq(onHost.samples.size(), static_cast<std::size_t>(kN)));
        for (std::size_t i = 0UZ; i < kChangeAt && i < onHost.samples.size(); ++i) {
            expect(eq(onHost.samples[i], static_cast<float>(i) * kInitialGain)) << std::format("host sample {} runs with the initial gain", i);
        }
        for (std::size_t i = kChangeAt; i < onHost.samples.size(); ++i) {
            expect(eq(onHost.samples[i], static_cast<float>(i) * kUpdatedGain)) << std::format("host sample {} runs with the updated gain", i);
        }

        if (const auto domain = gr::test::firstServedSyclDomain()) {
            const auto onDevice = runChainFull<gr::styles::Gain>(*domain, kN, [](auto&) {}, tags);
            expect(sameSamples(onDevice.samples, onHost.samples)) << "the device mirror must observe the mid-stream gain change identically to the host";
        }
    };

    return 0;
}
