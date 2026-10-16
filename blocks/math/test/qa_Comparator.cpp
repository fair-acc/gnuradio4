#include <boost/ut.hpp>

#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gnuradio-4.0/math/Comparator.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::blocks::math::Comparator;

static_assert(std::same_as<Comparator<std::complex<float>>, Comparator<std::complex<float>, float>>, "a complex input compares its magnitude and answers in the real type");

template<typename T, typename TOut = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] std::vector<TOut> compared(const std::vector<T>& samples, gr::property_map settings) {
    Comparator<T, TOut> block(std::move(settings));
    block.init(block.progress);
    block.start();
    std::vector<TOut> outputs;
    for (const T& sample : samples) {
        outputs.push_back(block.processOne(sample));
    }
    return outputs;
}

const auto definitions = [] {
    "the output rises above high, falls below low, and holds inside the band"_test = [] {
        const std::vector<float> samples{0.f, 1.5f, 3.f, 1.5f, 2.f, -1.f, 0.f, 1.5f, 2.5f};
        expect(compared(samples, {{"low", 1.f}, {"high", 2.f}}) == std::vector<float>{0.f, 0.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f}) << "a value at an edge is inside the band";
    };

    "initial is the output until the first sample leaves the band"_test = [] { expect(compared(std::vector{1.5f, 1.5f, 0.f}, {{"low", 1.f}, {"high", 2.f}, {"initial", true}}) == std::vector<float>{1.f, 1.f, 0.f}); };

    "low == high is a plain comparator"_test = [] { expect(compared(std::vector{-1., 0.5, 1.5, 0.5}, {{"low", 1.}, {"high", 1.}}) == std::vector<double>{0., 0., 1., 0.}); };

    "a reversed band is the same band"_test = [] { expect(compared(std::vector{0.f, 1.5f, 3.f, 1.5f}, {{"low", 2.f}, {"high", 1.f}}) == std::vector<float>{0.f, 0.f, 1.f, 1.f}); };

    "NaN keeps the previous output"_test = [] {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        expect(compared(std::vector{3.f, nan, -1.f, nan}, {{"low", 1.f}, {"high", 2.f}}) == std::vector<float>{1.f, 1.f, 0.f, 0.f});
    };

    "a complex sample is compared by its magnitude"_test = [] {
        const std::vector<std::complex<float>> samples{{0.f, 3.f}, {1.f, 1.f}, {0.f, -0.5f}};
        expect(compared(samples, {{"low", 1.f}, {"high", 2.f}}) == std::vector<float>{1.f, 1.f, 0.f}) << "|j3| = 3, |1 + j| = 1.41, |-0.5j| = 0.5";
    };

    "the uint8_t output is a 0/1 control stream"_test = [] { expect(compared<std::int16_t, std::uint8_t>(std::vector<std::int16_t>{0, 300, 150, -300}, {{"low", std::int16_t{-100}}, {"high", std::int16_t{100}}}) == std::vector<std::uint8_t>{0U, 1U, 1U, 0U}); };

    "a complex sample drives a uint8_t output by its magnitude"_test = [] { expect(compared<std::complex<double>, std::uint8_t>(std::vector<std::complex<double>>{{0., 3.}, {1., 1.}, {0., -0.5}}, {{"low", 1.}, {"high", 2.}}) == std::vector<std::uint8_t>{1U, 1U, 0U}); };

    "integer extremes compare without overflow"_test = [] {
        using Limits = std::numeric_limits<std::int64_t>;
        expect(compared(std::vector{Limits::lowest(), Limits::max(), std::int64_t{0}}, {{"low", std::int64_t{-1}}, {"high", std::int64_t{1}}}) == std::vector<std::int64_t>{0, 1, 1});
    };

    "reset returns to initial"_test = [] {
        Comparator<float> block({{"low", 1.f}, {"high", 2.f}});
        block.init(block.progress);
        block.start();
        expect(eq(block.processOne(3.f), 1.f));
        block.reset();
        expect(eq(block.processOne(1.5f), 0.f));
    };

    "a device placement is refused: the hysteresis state is a sequence"_test = [] {
        const auto device = gr::testing::firstServedDomain({"host:sycl", "gpu:sycl"});
        if (!device) {
            return;
        }
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(64)}, {"mark_tag", false}});
        auto&                       dut    = fixture.emplace<Comparator<float>>({{"gr:compute_domain", std::string(*device) + "!"}});
        auto&                       sink   = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"log_samples", true}});
        expect(fixture.connect<"out", "in">(source, dut).has_value());
        expect(fixture.connect<"out", "in">(dut, sink).has_value());
        std::ignore = fixture.run();
        expect(dut.state() == gr::lifecycle::State::ERROR) << *device;
    };
};

} // namespace

int main() {
    "Comparator"_test = definitions;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
