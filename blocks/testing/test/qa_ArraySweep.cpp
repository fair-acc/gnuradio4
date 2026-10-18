#include <boost/ut.hpp>

#include <numbers>

#include <gnuradio-4.0/testing/ArrayScene.hpp>

const boost::ut::suite<"stereo sweep"> kSweepTests = [] {
    using namespace boost::ut;
    constexpr double kDegrees = 180.0 / std::numbers::pi;

    "the sweep starts at minus eighty degrees, crosses zero mid-scene, ends just below plus eighty and repeats"_test = [] {
        constexpr std::size_t kScene = 1000UZ;
        expect(approx(gr::testing::sweepBearingRadians(0UZ, kScene) * kDegrees, -80.0, 1e-12));
        expect(approx(gr::testing::sweepBearingRadians(kScene / 2UZ, kScene) * kDegrees, 0.0, 1e-12));
        expect(approx(gr::testing::sweepBearingRadians(kScene - 1UZ, kScene) * kDegrees, 80.0 * (1.0 - 2.0 / 1000.0), 1e-12));
        expect(approx(gr::testing::sweepBearingRadians(250UZ, kScene) * kDegrees, -40.0, 1e-12));
        expect(eq(gr::testing::sweepBearingRadians(kScene + 250UZ, kScene), gr::testing::sweepBearingRadians(250UZ, kScene)));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
