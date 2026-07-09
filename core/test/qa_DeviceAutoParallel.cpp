#include <boost/ut.hpp>

#include <array>
#include <memory_resource>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceRelocatable.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/math/Math.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "device_test_helpers.hpp"

namespace gr::test {

/// an ordinary block: plain C++ settings, one `const noexcept processOne`, no device-specific entry point
struct ScaleByTaps : Block<ScaleByTaps> {
    PortIn<float>  in;
    PortOut<float> out;

    std::pmr::vector<float> taps; // re-seated onto device memory during init(), then read from the kernel

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(ScaleByTaps, in, out, taps);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return taps.size() < 2UZ ? x : x * taps[0] + taps[1]; }
};

/// a stateless gain, so two of them can be chained on the device with an internal device-to-device edge
struct Gain : Block<Gain> {
    PortIn<float>  in;
    PortOut<float> out;

    Annotated<float, "gain"> gain = 2.f;
    using DeviceStateIsReflected  = void;
    GR_MAKE_REFLECTABLE(Gain, in, out, gain);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x * gain; }
};

/// two inputs, and deliberately not symmetric: `a - 2b` tells a swapped argument order from a correct one, which
/// `a * b` or `a + b` could not
struct WeightedDifference : Block<WeightedDifference> {
    PortIn<float>  in0;
    PortIn<float>  in1;
    PortOut<float> out;

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(WeightedDifference, in0, in1, out);

    [[nodiscard]] constexpr float processOne(float a, float b) const noexcept { return a - 2.f * b; }
};

/// one input, two outputs: the body returns a tuple the tier must distribute across the ports. `2x` and `x - 1`
/// agree only at x = -1, which the ramp below never reaches, so swapped outputs cannot pass
struct SplitScaled : Block<SplitScaled> {
    PortIn<float>  in;
    PortOut<float> out0;
    PortOut<float> out1;

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(SplitScaled, in, out0, out1);

    [[nodiscard]] constexpr std::tuple<float, float> processOne(float x) const noexcept { return {x * 2.f, x - 1.f}; }
};

/// two in AND two out: the only shape where mixing up the input and output index packs is a silent miscount rather
/// than a compile error. Each output depends on both inputs, differently.
struct CrossMix : Block<CrossMix> {
    PortIn<float>  in0;
    PortIn<float>  in1;
    PortOut<float> out0;
    PortOut<float> out1;

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(CrossMix, in0, in1, out0, out1);

    [[nodiscard]] constexpr std::tuple<float, float> processOne(float a, float b) const noexcept { return {a - 2.f * b, 3.f * a + b}; }
};

/// the stimulus harness cannot emit `gr::complex` (its `values` setting has no supported type), so a complex
/// device test builds its samples from a float ramp instead
struct ToComplex : Block<ToComplex> {
    PortIn<float>               in;
    PortOut<gr::complex<float>> out;

    Annotated<float, "imaginary_scale"> imaginary_scale = 1.f;

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(ToComplex, in, out, imaginary_scale);

    [[nodiscard]] constexpr gr::complex<float> processOne(float x) const noexcept { return {x, imaginary_scale * x}; }
};

} // namespace gr::test

static_assert(gr::AutoParallelisable<gr::test::ScaleByTaps>);
static_assert(gr::device::DeviceRelocatable<gr::test::ScaleByTaps>, "a pmr array setting must not disqualify a block");
static_assert(!std::is_trivially_copyable_v<gr::test::ScaleByTaps>, "no real block ever was");

// Kernels must be launched while the AdaptiveCpp runtime is alive. Boost.UT runs registered suites from ~runner,
// i.e. during static destruction, where acpp's kernel cache is already gone and its mutex aborts the process
// (glibc: `pthread_mutex_lock ... e != ESRCH`). The tests are therefore registered and run from main().
int main() {
    using namespace boost::ut;
    using namespace gr::testing;

    // the docs promise "a later settings().set(...) reallocates through the same device resource". Nothing pinned
    // that: the test below asserts the seat only after the INITIAL seating. This drives a real settings change
    // through a tag while the graph runs -- the documented path -- and checks both halves of the promise.
    "a pmr setting changed mid-run keeps its device seat and the kernel reads the new values"_test = [] {
        expect(gr::device::registerSyclRuntime()) << "this test is only built for AdaptiveCpp";
        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return;
        }
        const std::string     computeDomain(*servedDomain);
        constexpr gr::Size_t  kN        = 32U;
        constexpr std::size_t kChangeAt = 16UZ;

        gr::Graph flow;
        auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     dut    = flow.emplaceBlock<gr::test::ScaleByTaps>({{"gr:compute_domain", computeDomain}});
        auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});

        dut.taps     = std::pmr::vector<float>{2.f, 1.f};
        source._tags = {gr::testing::OwningTag{kChangeAt, gr::property_map{{"taps", std::vector<float>{10.f, 5.f}}}}};

        expect(flow.connect<"out", "in">(source, dut).has_value());
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(eq(gr::test::cpuFallbacksDuring([&sched] { expect(sched.runAndWait().has_value()); }), 0UZ)) //
            << "the same numbers would come back if the dispatcher had quietly refused the kernel";

        const gr::ComputeDomain    domain   = gr::ComputeDomain::parse(computeDomain);
        std::pmr::memory_resource* deviceMr = gr::ComputeRegistry::instance().tryResolve(domain, domain.user);
        expect(deviceMr != nullptr);
        expect(dut.taps.get_allocator().resource() == deviceMr) << "a settings change must reallocate through the device resource, not the default one";
        expect(eq(dut.taps.size(), 2UZ));
        expect(eq(dut.taps[0], 10.f)) << "the new taps were applied";

        expect(eq(sink._samples.size(), static_cast<std::size_t>(kN)));
        bool beforeOk = true;
        for (std::size_t i = 0UZ; i < kChangeAt && i < sink._samples.size(); ++i) {
            beforeOk = beforeOk && sink._samples[i] == static_cast<float>(i) * 2.f + 1.f;
        }
        expect(beforeOk) << "samples before the change use the initial taps";
        bool afterOk = true;
        for (std::size_t i = kChangeAt; i < sink._samples.size(); ++i) {
            afterOk = afterOk && sink._samples[i] == static_cast<float>(i) * 10.f + 5.f;
        }
        expect(afterOk) << "the device mirror refreshed: samples after the change use the new taps";
    };

    "a block with pmr array settings runs as a kernel and reads them from device memory"_test = [] {
        expect(gr::device::registerSyclRuntime()) << "this test is only built for AdaptiveCpp";

        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return; // firstServedSyclDomain() already announced that no SYCL device is registered
        }
        const std::string computeDomain(*servedDomain);

        constexpr gr::Size_t kN = 32U;

        gr::Graph flow;
        auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     dut    = flow.emplaceBlock<gr::test::ScaleByTaps>({{"gr:compute_domain", computeDomain}});
        auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});

        dut.taps                            = std::pmr::vector<float>{2.f, 1.f}; // init() re-seats this onto the device's resource
        std::pmr::memory_resource* seededOn = dut.taps.get_allocator().resource();

        expect(flow.connect<"out", "in">(source, dut).has_value());
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(eq(gr::test::cpuFallbacksDuring([&sched] { expect(sched.runAndWait().has_value()); }), 0UZ)) //
            << "the same numbers would come back if the dispatcher had quietly refused the kernel";

        const gr::ComputeDomain    domain   = gr::ComputeDomain::parse(computeDomain);
        std::pmr::memory_resource* deviceMr = gr::ComputeRegistry::instance().tryResolve(domain, domain.user);
        expect(deviceMr != nullptr) << "the sycl provider must serve the selected compute domain";
        expect(seededOn == deviceMr) << "init() re-seats the block's pmr fields onto the device's own resource";
        expect(dut.taps.get_allocator().resource() == deviceMr) << "the seat survives the run";
        expect(eq(dut.taps.size(), 2UZ)) << "the values survive the migration";
        expect(eq(dut.taps[0], 2.f));

        expect(eq(sink._nSamplesProduced, kN)) << "every sample went through the device path";
        bool valuesOk = sink._samples.size() == static_cast<std::size_t>(kN);
        for (std::size_t i = 0UZ; valuesOk && i < sink._samples.size(); ++i) {
            valuesOk = sink._samples[i] == static_cast<float>(i) * 2.f + 1.f;
        }
        expect(valuesOk) << "the kernel read the migrated taps: out[i] == i*taps[0] + taps[1]";
    };

    "the device mirror is reused across work() calls rather than rebuilt"_test = [] {
        expect(gr::device::registerSyclRuntime());

        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return;
        }
        const std::string computeDomain(*servedDomain);

        constexpr gr::Size_t kN = 64U;

        gr::Graph flow;
        auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     dut    = flow.emplaceBlock<gr::test::ScaleByTaps>({{"gr:compute_domain", computeDomain}});
        auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}});

        dut.taps           = std::pmr::vector<float>{1.f, 0.f};
        dut.in.max_samples = 8UZ; // force several dispatches

        expect(flow.connect<"out", "in">(source, dut).has_value());
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(eq(gr::test::cpuFallbacksDuring([&sched] { expect(sched.runAndWait().has_value()); }), 0UZ)) //
            << "the same numbers would come back if the dispatcher had quietly refused the kernel";

        expect(static_cast<bool>(dut.deviceShadow().mirror)) << "the mirror outlives a single dispatch";
        expect(eq(dut.deviceShadow().epoch, dut.settingsEpoch())) << "refreshed exactly up to the current settings epoch";
        expect(eq(sink._nSamplesProduced, kN));
    };

    "the residency query, on which copy elision is gated, distinguishes device memory from host"_test = [] {
        // no CPU stand-in here: on a host backend get_pointer_type() reports `host` for malloc_shared,
        // malloc_device and a stack address alike, so there is no separation left to distinguish
        const auto servedDomain = gr::test::firstServedDomain({"gpu:sycl"});
        if (!servedDomain) {
            return;
        }
        auto* scheduler = gr::device::DeviceContextRegistry::instance().tryResolve(*servedDomain);
        expect(scheduler != nullptr);
        gr::device::DeviceContext& context = *scheduler;

        auto shared = context.allocateShared<float>(16UZ);
        expect(static_cast<bool>(shared));
        expect(context.isDeviceAccessible(shared.devicePointer<float>())) << "USM is device-accessible, so the kernel reads it in place";

        std::array<float, 16> host{};
        expect(!context.isDeviceAccessible(host.data())) << "plain host memory is not, so it must be copied";
        context.deallocate(shared);
    };

    "two device blocks chain across an internal device-to-device edge"_test = [] {
        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return;
        }
        const std::string    computeDomain(*servedDomain);
        constexpr gr::Size_t kN = 4096U;

        gr::Graph flow;
        auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     gainA  = flow.emplaceBlock<gr::test::Gain>({{"gr:compute_domain", computeDomain}, {"gain", 2.f}});
        auto&     gainB  = flow.emplaceBlock<gr::test::Gain>({{"gr:compute_domain", computeDomain}, {"gain", 3.f}});
        auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});

        expect(flow.connect<"out", "in">(source, gainA).has_value());
        expect(flow.connect<"out", "in">(gainA, gainB).has_value()); // both device: the edge buffer is USM, no host round-trip
        expect(flow.connect<"out", "in">(gainB, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(eq(gr::test::cpuFallbacksDuring([&sched] { expect(sched.runAndWait().has_value()); }), 0UZ)) //
            << "the same numbers would come back if the dispatcher had quietly refused the kernel";

        bool valuesOk = sink._samples.size() == static_cast<std::size_t>(kN);
        for (std::size_t i = 0UZ; valuesOk && i < sink._samples.size(); ++i) {
            valuesOk = sink._samples[i] == static_cast<float>(i) * 2.f * 3.f;
        }
        expect(valuesOk) << "the second kernel read the first kernel's output straight from the device edge";
    };

    "a two-input processOne runs as a kernel, with its arguments in the right order"_test = [] {
        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return;
        }
        const std::string    computeDomain(*servedDomain);
        constexpr gr::Size_t kN = 1024U;

        gr::Graph flow;
        auto&     sourceA = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     sourceB = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     scaleB  = flow.emplaceBlock<gr::test::Gain>({{"gr:compute_domain", computeDomain}, {"gain", 10.f}}); // so the two arms carry different values
        auto&     combine = flow.emplaceBlock<gr::test::WeightedDifference>({{"gr:compute_domain", computeDomain}});
        auto&     sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});

        expect(flow.connect<"out", "in0">(sourceA, combine).has_value());
        expect(flow.connect<"out", "in">(sourceB, scaleB).has_value());
        expect(flow.connect<"out", "in1">(scaleB, combine).has_value());
        expect(flow.connect<"out", "in">(combine, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(eq(gr::test::cpuFallbacksDuring([&sched] { expect(sched.runAndWait().has_value()); }), 0UZ)) //
            << "a two-input block that fell back to the CPU computes the very same numbers, so the count is the assertion";

        bool valuesOk = sink._samples.size() == static_cast<std::size_t>(kN);
        for (std::size_t i = 0UZ; valuesOk && i < sink._samples.size(); ++i) {
            const float expected = static_cast<float>(i) - 2.f * (static_cast<float>(i) * 10.f);
            valuesOk             = sink._samples[i] == expected;
        }
        expect(valuesOk) << "both input ports reached the kernel, in the order the block declares them";
    };

    "a two-output processOne runs as a kernel, with each result on its own port"_test = [] {
        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return;
        }
        const std::string    computeDomain(*servedDomain);
        constexpr gr::Size_t kN = 1024U;

        gr::Graph flow;
        auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     split  = flow.emplaceBlock<gr::test::SplitScaled>({{"gr:compute_domain", computeDomain}});
        auto&     sink0  = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});
        auto&     sink1  = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});

        expect(flow.connect<"out", "in">(source, split).has_value());
        expect(flow.connect<"out0", "in">(split, sink0).has_value());
        expect(flow.connect<"out1", "in">(split, sink1).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(eq(gr::test::cpuFallbacksDuring([&sched] { expect(sched.runAndWait().has_value()); }), 0UZ)) //
            << "a block the dispatcher refused would produce these same two streams on the CPU";

        bool valuesOk = sink0._samples.size() == static_cast<std::size_t>(kN) && sink1._samples.size() == static_cast<std::size_t>(kN);
        for (std::size_t i = 0UZ; valuesOk && i < sink0._samples.size(); ++i) {
            valuesOk = sink0._samples[i] == static_cast<float>(i) * 2.f && sink1._samples[i] == static_cast<float>(i) - 1.f;
        }
        expect(valuesOk) << "each element of the returned tuple reached the port that declares it";
    };

    "the shipped two-port multiply runs as a kernel"_test = [] {
        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return;
        }
        const std::string    computeDomain(*servedDomain);
        constexpr gr::Size_t kN = 512U;

        gr::Graph flow;
        auto&     sourceA = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     sourceB = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     product = flow.emplaceBlock<gr::blocks::math::MultiplyPair<float>>({{"gr:compute_domain", computeDomain}});
        auto&     sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});

        expect(flow.connect<"out", "in0">(sourceA, product).has_value());
        expect(flow.connect<"out", "in1">(sourceB, product).has_value());
        expect(flow.connect<"out", "in">(product, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(eq(gr::test::cpuFallbacksDuring([&sched] { expect(sched.runAndWait().has_value()); }), 0UZ)) //
            << "the multi-port Multiply cannot reach a device tier at all; this two-port form is why the chain can";

        bool valuesOk = sink._samples.size() == static_cast<std::size_t>(kN);
        for (std::size_t i = 0UZ; valuesOk && i < sink._samples.size(); ++i) {
            valuesOk = sink._samples[i] == static_cast<float>(i) * static_cast<float>(i);
        }
        expect(valuesOk) << "and it multiplies the two streams, sample by sample";
    };

    "the two-port multiply carries gr::complex through a kernel"_test = [] {
        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return;
        }
        const std::string    computeDomain(*servedDomain);
        constexpr gr::Size_t kN = 512U;

        gr::Graph flow;
        auto&     sourceA = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     sourceB = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     toA     = flow.emplaceBlock<gr::test::ToComplex>({{"imaginary_scale", 1.f}});
        auto&     toB     = flow.emplaceBlock<gr::test::ToComplex>({{"imaginary_scale", 2.f}});
        auto&     product = flow.emplaceBlock<gr::blocks::math::MultiplyPair<gr::complex<float>>>({{"gr:compute_domain", computeDomain}});
        auto&     sink    = flow.emplaceBlock<TagSink<gr::complex<float>, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});

        expect(flow.connect<"out", "in">(sourceA, toA).has_value());
        expect(flow.connect<"out", "in">(sourceB, toB).has_value());
        expect(flow.connect<"out", "in0">(toA, product).has_value());
        expect(flow.connect<"out", "in1">(toB, product).has_value());
        expect(flow.connect<"out", "in">(product, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(eq(gr::test::cpuFallbacksDuring([&sched] { expect(sched.runAndWait().has_value()); }), 0UZ)) //
            << "std::complex would not get this far: its operator* needs a libgcc helper the device has no copy of";

        bool valuesOk = sink._samples.size() == static_cast<std::size_t>(kN);
        for (std::size_t i = 0UZ; valuesOk && i < sink._samples.size(); ++i) {
            const float x = static_cast<float>(i);
            valuesOk      = sink._samples[i] == gr::complex<float>{-x * x, 3.f * x * x};
        }
        expect(valuesOk) << "and both the real and the imaginary cross terms survive the round trip";
    };

    "two inputs and two outputs at once: the index packs stay apart"_test = [] {
        const auto servedDomain = gr::test::firstServedSyclDomain();
        if (!servedDomain) {
            return;
        }
        const std::string    computeDomain(*servedDomain);
        constexpr gr::Size_t kN = 512U;

        gr::Graph flow;
        auto&     sourceA = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     sourceB = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&     scaleB  = flow.emplaceBlock<gr::test::Gain>({{"gr:compute_domain", computeDomain}, {"gain", 10.f}});
        auto&     mix     = flow.emplaceBlock<gr::test::CrossMix>({{"gr:compute_domain", computeDomain}});
        auto&     sink0   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});
        auto&     sink1   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}});

        expect(flow.connect<"out", "in0">(sourceA, mix).has_value());
        expect(flow.connect<"out", "in">(sourceB, scaleB).has_value());
        expect(flow.connect<"out", "in1">(scaleB, mix).has_value());
        expect(flow.connect<"out0", "in">(mix, sink0).has_value());
        expect(flow.connect<"out1", "in">(mix, sink1).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(eq(gr::test::cpuFallbacksDuring([&sched] { expect(sched.runAndWait().has_value()); }), 0UZ)) //
            << "a two-by-two block that fell back would return these same four streams";

        bool valuesOk = sink0._samples.size() == static_cast<std::size_t>(kN) && sink1._samples.size() == static_cast<std::size_t>(kN);
        for (std::size_t i = 0UZ; valuesOk && i < sink0._samples.size(); ++i) {
            const float a = static_cast<float>(i);
            const float b = static_cast<float>(i) * 10.f;
            valuesOk      = sink0._samples[i] == a - 2.f * b && sink1._samples[i] == 3.f * a + b;
        }
        expect(valuesOk) << "each input reached the parameter that names it, and each result the port that names it";
    };

    return 0;
}
