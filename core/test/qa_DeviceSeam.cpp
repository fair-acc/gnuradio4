#include <boost/ut.hpp>

#include <cstddef>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "device_test_helpers.hpp"

namespace gr::test {
struct Copy : Block<Copy> {
    PortIn<float>  in;
    PortOut<float> out;

    GR_MAKE_REFLECTABLE(Copy, in, out);

    template<meta::t_or_simd<float> V>
    [[nodiscard]] constexpr V processOne(const V& a) const noexcept {
        return a;
    }

    [[nodiscard]] gr::work::Status processBulk_sycl(gr::device::SyclQueue& queue, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const auto count = std::min(input.size(), output.size());
        queue.memcpy(output.data(), input.data(), count * sizeof(float)).wait();
        for (const auto& [relIndex, tagMapRef] : input.tags(count)) {
            output.publishTag(tagMapRef.get(), relIndex < 0 ? 0UZ : static_cast<std::size_t>(relIndex));
        }
        std::ignore = input.consume(count);
        output.publish(count);
        return gr::work::Status::OK;
    }
};
} // namespace gr::test

// AdaptiveCpp aborts if a kernel is launched while Boost.UT runs suites from ~runner, so tests run from main (gotcha G10)
int main() {
    using namespace boost::ut;
    using namespace gr::testing;

    "device-eligible block on a served SYCL domain runs through the registered runtime"_test = [] {
        constexpr gr::Size_t kN     = 16U;
        const auto           domain = gr::test::firstServedSyclDomain();
        if (!domain) {
            return; // firstServedSyclDomain() already announced that no SYCL device is registered
        }
        const std::string computeDomain(*domain);

        auto* deviceSched = gr::device::DeviceContextRegistry::instance().tryResolve(computeDomain);
        expect(deviceSched != nullptr);
        expect(deviceSched->backend() == gr::device::DeviceBackend::SYCL);

        gr::Graph        flow;
        auto&            source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kN}, {"mark_tag", false}});
        auto&            dut    = flow.emplaceBlock<gr::test::Copy>({{"gr:compute_domain", computeDomain}});
        auto&            sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kN}, {"log_samples", true}, {"log_tags", true}});
        gr::property_map payload;
        gr::tag::put(payload, "kind", gr::pmt::Value("p1-device"));
        source._tags.push_back(OwningTag{3UZ, payload});
        expect(flow.connect<"out", "in">(source, dut).has_value());
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        const std::size_t fallbacks = gr::test::cpuFallbacksDuring([&sched] { expect(sched.runAndWait().has_value()) << "device-eligible block on the selected compute domain must complete"; });

        expect(eq(fallbacks, 0UZ)) << "matching samples would also hold if the dispatcher had quietly refused the kernel";
        expect(eq(sink._nSamplesProduced, kN)) << "all samples flowed through the selected execution path";
        bool valuesOk = sink._samples.size() == static_cast<std::size_t>(kN);
        for (std::size_t i = 0; valuesOk && i < sink._samples.size(); ++i) {
            valuesOk = sink._samples[i] == static_cast<float>(i);
        }
        expect(valuesOk) << "the selected execution path produced the expected identity copy";

        expect(eq(sink._tags.size(), 1UZ));
        if (sink._tags.size() == 1UZ) {
            expect(eq(sink._tags[0].index, 3UZ));
            expect(sink._tags[0].map.contains("kind"));
            expect(eq(gr::test::get_value_or_fail<std::string>(sink._tags[0].map.find_value("kind").value()), std::string{"p1-device"}));
        }
    };

    return 0;
}
