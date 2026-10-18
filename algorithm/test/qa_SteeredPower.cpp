#include <boost/ut.hpp>

#include <array>
#include <limits>

#include <gnuradio-4.0/algorithm/SteeredPower.hpp>

const boost::ut::suite<"SteeredPower"> powerTests = [] {
    using namespace boost::ut;
    using gr::algorithm::computeSteeredPower;

    "fractional lags interpolate adjacent samples and sum distinct pair curves"_test = [] {
        constexpr std::array kCurves{0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0};
        constexpr std::array kLags{-0.25, 0.5};
        expect(eq(computeSteeredPower(kCurves.data(), kLags.data(), 2UZ, 5UZ), 38.5));
    };

    "both curve endpoints contribute and lags outside the window contribute zero"_test = [] {
        constexpr std::array kCurve{1.f, 2.f, 3.f};
        constexpr float      kInfinity = std::numeric_limits<float>::infinity();
        constexpr float      kNaN      = std::numeric_limits<float>::quiet_NaN();
        for (const auto [lag, expected] : std::array{std::array{-1.f, 1.f}, std::array{0.f, 2.f}, std::array{1.f, 3.f}, std::array{-1.01f, 0.f}, std::array{1.01f, 0.f}, std::array{kInfinity, 0.f}, std::array{kNaN, 0.f}}) {
            expect(eq(computeSteeredPower(kCurve.data(), &lag, 1UZ, kCurve.size()), expected));
        }
        expect(eq(computeSteeredPower<float>(nullptr, nullptr, 0UZ, 3UZ), 0.f));
    };

    "interpolation preserves negative correlation and uses the upper endpoint of a two-sample curve"_test = [] {
        constexpr std::array kCurve{-4.f, -2.f};
        constexpr float      kLag = -0.25f;
        expect(eq(computeSteeredPower(kCurve.data(), &kLag, 1UZ, 2UZ), -2.5f));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
