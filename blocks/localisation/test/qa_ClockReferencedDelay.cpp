#include <boost/ut.hpp>
#include <gnuradio-4.0/localisation/ClockReferencedDelay.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
struct Measurement {
    gr::testing::GraphFixture<>                                                   fixture;
    gr::testing::TagSource<std::uint8_t>&                                         clock   = fixture.emplace<gr::testing::TagSource<std::uint8_t>>({{"n_samples_max", gr::Size_t{16U}}});
    gr::testing::TagSource<float>&                                                arrival = fixture.emplace<gr::testing::TagSource<float>>({{"n_samples_max", gr::Size_t{16U}}});
    gr::localisation::ClockReferencedDelay<float>&                                measure = fixture.emplace<gr::localisation::ClockReferencedDelay<float>>({{"interface_latency", 0.001}});
    gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_BULK>& sink    = fixture.emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();

    [[nodiscard]] std::vector<double> run() {
        boost::ut::expect(fixture.connect<"out", "clk">(clock, measure).has_value());
        boost::ut::expect(fixture.connect<"out", "in">(arrival, measure).has_value());
        boost::ut::expect(fixture.connect<"out", "in">(measure, sink).has_value());
        boost::ut::expect(fixture.run().has_value());
        return {sink._samples.begin(), sink._samples.end()};
    }
};

[[nodiscard]] gr::property_map triggerTime(std::uint64_t ns) { return {{gr::tag::kTriggerTime, ns}}; }
} // namespace

const boost::ut::suite<"ClockReferencedDelay"> clockReferencedDelayTests = [] {
    using namespace boost::ut;
    "timestamp subtraction retains nanoseconds near a present day UTC epoch"_test = [] {
        constexpr std::uint64_t kEpoch = 1800000000000000000ULL;
        expect(approx(gr::localisation::clockReferencedDelaySeconds(kEpoch, kEpoch + 17U, 0.0), 17.e-9, 1.e-20));
        expect(approx(gr::localisation::clockReferencedDelaySeconds(kEpoch + 17U, kEpoch, 0.0), -17.e-9, 1.e-20));
    };
    "arrival tags measure range against the latest emission clock tag"_test = [] {
        Measurement measurement;
        measurement.clock._tags   = {{0U, triggerTime(1'000'000'000U)}, {8U, triggerTime(2'000'000'000U)}};
        measurement.arrival._tags = {{4U, triggerTime(1'003'000'000U)}, {12U, triggerTime(2'005'000'000U)}};
        const auto delays         = measurement.run();
        expect(fatal(eq(delays.size(), 2UZ)));
        expect(approx(delays[0], 0.002, 1.e-14)) << "3 ms after the first emission, less 1 ms latency";
        expect(approx(delays[1], 0.004, 1.e-14)) << "5 ms after the second emission";
    };

    "the first arrival tag of a sample wins"_test = [] {
        Measurement measurement;
        measurement.clock._tags   = {{0U, triggerTime(1'000'000'000U)}};
        measurement.arrival._tags = {{4U, triggerTime(1'003'000'000U)}, {4U, triggerTime(1'009'000'000U)}};
        const auto delays         = measurement.run();
        expect(fatal(eq(delays.size(), 1UZ)));
        expect(approx(delays[0], 0.002, 1.e-14));
    };

    "an arrival before any emission measures nothing"_test = [] {
        Measurement measurement;
        measurement.clock._tags   = {{8U, triggerTime(1'000'000'000U)}};
        measurement.arrival._tags = {{4U, triggerTime(1'003'000'000U)}};
        expect(measurement.run().empty());
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
