#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gnuradio-4.0/device/DeviceRelocatable.hpp>
#include <gnuradio-4.0/math/Math.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::blocks::math::Clamp;

static_assert(gr::traits::block::can_processOne_simd<Clamp<float>>);
static_assert(gr::traits::block::can_processOne_simd<Clamp<std::int16_t>>);
static_assert(!gr::traits::block::can_processOne_simd<Clamp<std::complex<float>>>, "a complex sample has no SIMD lane type");
static_assert(gr::device::DeviceRelocatable<Clamp<float>>);
static_assert(gr::device::DeviceRelocatable<Clamp<std::complex<float>>>);

template<typename T>
[[nodiscard]] Clamp<T> clampTo(auto low, auto high) {
    Clamp<T> block({{"low", low}, {"high", high}});
    block.init(block.progress);
    return block;
}

const auto definitions = [] {
    "a sample beyond the band is held at its edge, one inside it passes"_test = [] {
        const auto clamp = clampTo<float>(-1.f, 2.f);
        expect(eq(clamp.processOne(-3.f), -1.f));
        expect(eq(clamp.processOne(-1.f), -1.f));
        expect(eq(clamp.processOne(0.5f), 0.5f));
        expect(eq(clamp.processOne(2.f), 2.f));
        expect(eq(clamp.processOne(7.f), 2.f));
    };

    "a reversed band is the same band"_test = [] {
        const auto clamp = clampTo<float>(2.f, -1.f);
        expect(eq(clamp.processOne(-3.f), -1.f));
        expect(eq(clamp.processOne(0.5f), 0.5f));
        expect(eq(clamp.processOne(7.f), 2.f));
    };

    "NaN passes through"_test = [] { expect(std::isnan(clampTo<double>(-1., 1.).processOne(std::numeric_limits<double>::quiet_NaN()))); };

    "integer extremes clamp, and the default band passes every value"_test = [] {
        const auto clamp = clampTo<std::int8_t>(std::int8_t{-100}, std::int8_t{100});
        expect(eq(clamp.processOne(std::numeric_limits<std::int8_t>::lowest()), std::int8_t{-100}));
        expect(eq(clamp.processOne(std::numeric_limits<std::int8_t>::max()), std::int8_t{100}));
        Clamp<std::uint64_t> open;
        open.init(open.progress);
        expect(eq(open.processOne(std::numeric_limits<std::uint64_t>::max()), std::numeric_limits<std::uint64_t>::max()));
    };

    "a vector clamps lane by lane"_test = [] {
        using V            = vir::stdx::native_simd<float>;
        const auto clamp   = clampTo<float>(-1.f, 2.f);
        const V    ramp    = V([](auto lane) { return static_cast<float>(lane) - 2.f; });
        const V    limited = clamp.processOne(ramp);
        for (std::size_t lane = 0UZ; lane < V::size(); ++lane) {
            expect(eq(limited[lane], std::clamp(static_cast<float>(lane) - 2.f, -1.f, 2.f))) << "lane " << lane;
        }
    };

    "an int16 vector clamps lane by lane"_test = [] {
        using V            = vir::stdx::native_simd<std::int16_t>;
        const auto clamp   = clampTo<std::int16_t>(std::int16_t{-3}, std::int16_t{5});
        const V    ramp    = V([](auto lane) { return static_cast<std::int16_t>(static_cast<int>(lane) - 8); });
        const V    limited = clamp.processOne(ramp);
        for (std::size_t lane = 0UZ; lane < V::size(); ++lane) {
            expect(eq(limited[lane], std::clamp(static_cast<std::int16_t>(static_cast<int>(lane) - 8), std::int16_t{-3}, std::int16_t{5}))) << "lane " << lane;
        }
    };

    "a complex sample too large to square passes the default band and is limited by a finite one"_test = [] {
        Clamp<std::complex<float>> open;
        open.init(open.progress);
        const std::complex<float> huge(3e30f, -4e30f);
        expect(eq(open.processOne(huge), huge)) << "|x| = 5e30, its square overflows float";
        const auto limited = clampTo<std::complex<float>>(0.f, 10.f).processOne(huge);
        expect(std::abs(std::abs(limited) - 10.f) < 1e-5f) << "limited to 10";
        expect(std::abs(std::arg(limited) - std::arg(huge)) < 1e-6f) << "phase kept";
    };

    "a complex sample keeps its phase and has its magnitude limited; zero becomes low on the real axis"_test = [] {
        const auto clamp = clampTo<std::complex<float>>(1.f, 2.f);
        for (const auto& [magnitude, phase, expected] : {std::tuple{3.f, 0.7f, 2.f}, std::tuple{0.5f, -2.f, 1.f}, std::tuple{1.5f, 3.f, 1.5f}}) {
            const auto limited = clamp.processOne(std::polar(magnitude, phase));
            expect(std::abs(std::abs(limited) - expected) < 1e-6f) << "magnitude " << magnitude;
            expect(std::abs(std::arg(limited) - phase) < 1e-6f) << "phase " << phase;
        }
        expect(eq(clamp.processOne(std::complex<float>{}), std::complex<float>(1.f, 0.f)));
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "samples clamp the same on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        std::vector<float>               ramp;
        std::vector<std::complex<float>> spiral;
        for (std::size_t i = 0UZ; i < 1024UZ; ++i) {
            ramp.push_back(-4.f + 8.f * static_cast<float>(i) / 1024.f);
            spiral.push_back(std::polar(4.f * static_cast<float>(i) / 1024.f, 0.1f * static_cast<float>(i)));
        }
        constexpr auto kBulk = gr::testing::ProcessFunction::USE_PROCESS_BULK;
        using RealSource     = gr::testing::TagSource<float, kBulk>;
        using ComplexSource  = gr::testing::TagSource<std::complex<float>, kBulk>;
        using RealSink       = gr::testing::TagSink<float, kBulk>;
        using ComplexSink    = gr::testing::TagSink<std::complex<float>, kBulk>;

        const std::string placed        = std::string(domain) + "!";
        auto&             realSource    = fixture.template emplace<RealSource>({{"values", ramp}, {"n_samples_max", static_cast<gr::Size_t>(ramp.size())}, {"mark_tag", false}});
        auto&             complexSource = fixture.template emplace<ComplexSource>({{"values", spiral}, {"n_samples_max", static_cast<gr::Size_t>(spiral.size())}, {"mark_tag", false}});
        auto&             realClamp     = fixture.template emplace<Clamp<float>>({{"low", -1.f}, {"high", 2.f}, {"gr:compute_domain", placed}});
        auto&             complexClamp  = fixture.template emplace<Clamp<std::complex<float>>>({{"low", 1.f}, {"high", 2.f}, {"gr:compute_domain", placed}});
        auto&             realSink      = fixture.template emplace<RealSink>({{"log_samples", true}});
        auto&             complexSink   = fixture.template emplace<ComplexSink>({{"log_samples", true}});
        expect(fixture.template connect<"out", "in">(realSource, realClamp).has_value());
        expect(fixture.template connect<"out", "in">(realClamp, realSink).has_value());
        expect(fixture.template connect<"out", "in">(complexSource, complexClamp).has_value());
        expect(fixture.template connect<"out", "in">(complexClamp, complexSink).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);

        expect(fatal(eq(realSink._samples.size(), ramp.size()))) << domain;
        expect(fatal(eq(complexSink._samples.size(), spiral.size()))) << domain;
        float worst = 0.f;
        for (std::size_t i = 0UZ; i < ramp.size(); ++i) {
            const float               magnitude    = std::clamp(std::abs(spiral[i]), 1.f, 2.f);
            const std::complex<float> expected     = std::abs(spiral[i]) == 0.f ? std::complex<float>(1.f, 0.f) : spiral[i] * (magnitude / std::abs(spiral[i]));
            const float               realError    = std::abs(realSink._samples[i] - std::clamp(ramp[i], -1.f, 2.f));
            const float               complexError = std::abs(complexSink._samples[i] - expected);
            worst                                  = std::max({worst, realError, complexError});
        }
        expect(lt(worst, 1e-5f)) << domain;
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

} // namespace

int main() {
    "Clamp"_test                 = definitions;
    "Clamp on every domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
