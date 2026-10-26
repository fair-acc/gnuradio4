#include <boost/ut.hpp>

#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include <gnuradio-4.0/algorithm/Measurement.hpp>

namespace {
using namespace gr::algorithm::measurement;

[[nodiscard]] std::vector<double> repeated(std::vector<double> period, std::size_t times) {
    std::vector<double> train;
    for (std::size_t i = 0UZ; i < times; ++i) {
        train.insert(train.end(), period.begin(), period.end());
    }
    return train;
}

/** zeros, a linear ramp of `ramp` samples to 1, ones: the 10 % and 90 % levels are crossed exactly 0.8 ramp apart */
[[nodiscard]] std::vector<double> step(std::size_t ramp, std::size_t flat = 20UZ) {
    std::vector<double> samples(flat, 0.);
    for (std::size_t i = 0UZ; i <= ramp; ++i) {
        samples.push_back(static_cast<double>(i) / static_cast<double>(ramp));
    }
    samples.insert(samples.end(), flat, 1.);
    return samples;
}

/** the measurements take a span of const samples, which template deduction does not convert to */
template<typename T>
[[nodiscard]] std::span<const T> view(const std::vector<T>& samples) {
    return std::span<const T>{samples};
}

template<typename TResult>
void expectNear(const TResult& found, double expected, double tolerance, const char* what) {
    using namespace boost::ut;
    expect(found.has_value()) << what << " is valid";
    if (found) {
        expect(approx(*found, expected, tolerance)) << what << ": " << *found << " vs " << expected;
    }
}
} // namespace

const boost::ut::suite<"measurement: amplitude (IEEE 181-2011)"> _amplitude = [] {
    using namespace boost::ut;

    "a constant: mean and RMS are the value, the AC RMS and peak-to-peak zero, top and base undefined"_test = [] {
        const std::vector<double> dc(16UZ, 3.);
        expectNear(mean(view(dc)), 3., 1e-15, "mean");
        expectNear(rms(view(dc)), 3., 1e-15, "rms");
        expectNear(populationStdDev(view(dc)), 0., 1e-15, "ac rms");
        expectNear(peakToPeak(view(dc)), 0., 1e-15, "peak-to-peak");
        expect(top(view(dc)).error() == Invalid::noTwoStates) << "a constant has no two states";
    };

    "a sine of whole cycles: RMS is A/sqrt(2), peak-to-peak 2A"_test = [] {
        constexpr double    kAmplitude = 2.5;
        std::vector<double> sine;
        for (std::size_t i = 0UZ; i < 4UZ * 64UZ; ++i) {
            sine.push_back(kAmplitude * std::sin(2. * std::numbers::pi * static_cast<double>(i) / 64.));
        }
        expectNear(rms(view(sine)), kAmplitude / std::numbers::sqrt2, 1e-12, "rms");
        expectNear(peakToPeak(view(sine)), 2. * kAmplitude, 1e-12, "peak-to-peak (64 samples per cycle hit both peaks)");
        expectNear(mean(view(sine)), 0., 1e-12, "mean");
    };

    "the AC RMS does not cancel under a large offset"_test = [] {
        // a one-pass E[x^2] - E[x]^2 loses everything at 1e8 + sin: the squares are 1e16, double's resolution there is 2
        std::vector<double> offsetSine;
        for (std::size_t i = 0UZ; i < 1024UZ; ++i) {
            offsetSine.push_back(1e8 + std::sin(2. * std::numbers::pi * static_cast<double>(i) / 64.));
        }
        expectNear(populationStdDev(view(offsetSine)), 1. / std::numbers::sqrt2, 1e-8, "ac rms");
    };

    "white noise: the RMS lies within three standard errors of sigma"_test = [] {
        constexpr double                 kSigma = 0.5;
        constexpr std::size_t            kN     = 100'000UZ;
        std::mt19937_64                  engine(42U);
        std::normal_distribution<double> noise(0., kSigma);
        std::vector<double>              samples(kN);
        for (double& x : samples) {
            x = noise(engine);
        }
        expectNear(rms(view(samples)), kSigma, 3. * kSigma / std::sqrt(2. * static_cast<double>(kN)), "rms");
    };

    "median of an odd and an even count"_test = [] {
        const std::vector<double> odd{3., 1., 2.};
        const std::vector<double> even{4., 1., 3., 2.};
        expectNear(median(view(odd)), 2., 1e-15, "odd");
        expectNear(median(view(even)), 2.5, 1e-15, "even");
    };

    "top and base are the two states of a pulse train, as IEEE 181's histogram finds them"_test = [] {
        const auto train = repeated({0., 0., 0., 0., 0., 1., 1., 1.}, 4UZ);
        expectNear(top(view(train)), 1., 1e-15, "top");
        expectNear(base(view(train)), 0., 1e-15, "base");
        expectNear(amplitude(view(train)), 1., 1e-15, "amplitude");
    };

    "the sample and the population standard deviation of a known set differ by n over n - 1"_test = [] {
        const std::vector<double> samples{2., 4., 4., 4., 5., 5., 7., 9.}; // mean 5, sum of squared deviations 32
        expectNear(populationStdDev(view(samples)), 2., 1e-15, "population sd");
        expectNear(sampleStdDev(view(samples)), std::sqrt(32. / 7.), 1e-15, "sample sd");
        expect(sampleStdDev(view(std::vector<double>{3.})).error() == Invalid::tooFewSamples) << "one sample has no sample deviation";
        expectNear(populationStdDev(view(std::vector<double>{3.})), 0., 1e-15, "one sample has a population deviation of zero");
    };

    "a pure ramp has no top and no base, and neither has its amplitude"_test = [] {
        std::vector<double> ramp;
        for (std::size_t i = 0UZ; i < 1000UZ; ++i) {
            ramp.push_back(static_cast<double>(i) / 999.);
        }
        expect(top(view(ramp)).error() == Invalid::noTwoStates);
        expect(base(view(ramp)).error() == Invalid::noTwoStates);
        expect(amplitude(view(ramp)).error() == Invalid::noTwoStates);
        const std::vector<double> sparse{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
        expect(top(view(sparse)).error() == Invalid::noTwoStates) << "ten samples on ten levels are no two states either";
    };

    "a low-duty pulse of 2 % with three-sample edges still reports its top and base"_test = [] {
        std::vector<double> samples(490UZ, 0.);
        samples.insert(samples.end(), {1. / 3., 2. / 3.});
        samples.insert(samples.end(), 18UZ, 1.);
        samples.insert(samples.end(), {2. / 3., 1. / 3.});
        samples.insert(samples.end(), 488UZ, 0.); // 20 of 1000 samples are high
        expectNear(top(view(samples)), 1., 1e-15, "top");
        expectNear(base(view(samples)), 0., 1e-15, "base");
    };

    "a two-level pulse with a slow edge still reports its levels"_test = [] {
        auto train = repeated({1., 1., 1., 1., 1., 1., 1., 1.}, 2UZ);
        train.insert(train.end(), {2., 3., 4., 5.});
        train.insert(train.end(), 16UZ, 5.);
        expectNear(base(view(train)), 1., 1e-15, "base");
        expectNear(top(view(train)), 5., 1e-15, "top");
    };
};

const boost::ut::suite<"measurement: pulse shape (IEEE 181-2011)"> _pulse = [] {
    using namespace boost::ut;

    "a linear ramp over R samples rises from 10 % to 90 % in 0.8 R"_test = [] {
        const auto rising = step(10UZ);
        expectNear(transitionTime(view(rising), true), 8., 1e-12, "rise time");
        std::vector<double> falling(rising.rbegin(), rising.rend());
        expectNear(transitionTime(view(falling), false), 8., 1e-12, "fall time");
        expect(transitionTime(view(falling), true).error() == Invalid::noTransition) << "a falling step has no rising transition";
    };

    "a pulse train of width W and period T: widths W and T - W at the 50 % level, duty W/T, area W per pulse"_test = [] {
        auto train = repeated({0., 0., 0., 0., 0., 1., 1., 1.}, 3UZ);
        train.push_back(0.);
        expectNear(width(view(train), true), 3., 1e-12, "positive width");
        expectNear(width(view(train), false), 5., 1e-12, "negative width");
        expectNear(dutyCycle(view(train)), 37.5, 1e-12, "duty cycle (3 of 8)");
        expectNear(area(view(train)), 9., 1e-12, "area: three pulses of 3, trapezoids included");
    };

    "one sample 10 % of the amplitude above top is a 10 % overshoot"_test = [] {
        std::vector<double> samples(20UZ, 0.);
        samples.push_back(1.1);
        samples.insert(samples.end(), 20UZ, 1.);
        expectNear(overshoot(view(samples)), 10., 1e-9, "overshoot");
        expectNear(undershoot(view(samples)), 0., 1e-12, "undershoot");
    };

    "a single step has a rise but no pulse width or duty cycle"_test = [] {
        const auto rising = step(4UZ);
        expect(width(view(rising), true).error() == Invalid::noTransition);
        expect(dutyCycle(view(rising)).error() == Invalid::noTransition);
    };
};

const boost::ut::suite<"measurement: invalid gates"> _invalid = [] {
    using namespace boost::ut;

    "an empty gate is too short, a gate with a NaN is not finite, and neither yields a number"_test = [] {
        const std::vector<double> empty;
        const std::vector<double> holed{0., std::numeric_limits<double>::quiet_NaN(), 1.};
        expect(mean(view(empty)).error() == Invalid::tooFewSamples);
        expect(rms(view(holed)).error() == Invalid::nonFinite);
        expect(transitionTime(view(holed), true).error() == Invalid::nonFinite);
        expect(area(view(std::vector<double>{1.})).error() == Invalid::tooFewSamples) << "a trapezoid needs two samples";
    };

    "integer samples measure as their values do"_test = [] {
        const std::vector<std::int16_t> train{0, 0, 0, 0, 0, 100, 100, 100, 0, 0, 0, 0, 0, 100, 100, 100, 0};
        expectNear(width(view(train), true), 3., 1e-12, "positive width");
        expectNear(amplitude(view(train)), 100., 1e-12, "amplitude");
    };
};

const boost::ut::suite<"measurement: timing (IEEE 181-2011)"> _timing = [] {
    using namespace boost::ut;

    const auto leader  = repeated({0., 0., 0., 0., 1., 1., 1., 1.}, 4UZ); // rising mid-level crossings at 3.5, 11.5, ...
    const auto shifted = [&](std::size_t lag) {
        std::vector<double> lagging(lag, 0.);
        lagging.insert(lagging.end(), leader.begin(), leader.end() - static_cast<std::ptrdiff_t>(lag));
        return lagging;
    };

    "a square wave of eight samples per cycle has period 8 and frequency 1/8 per sample"_test = [&] {
        expectNear(period(view(leader)), 8., 1e-12, "period");
        expectNear(frequency(view(leader)), 0.125, 1e-12, "frequency");
        expect(period(view(step(10UZ))).error() == Invalid::noTransition) << "one edge is no period";
    };

    "a copy lagging by two samples is delayed by two samples and a quarter cycle in phase"_test = [&] {
        const auto lagging = shifted(2UZ);
        expectNear(delay(view(leader), view(lagging)), 2., 1e-12, "delay");
        expectNear(phase(view(leader), view(lagging)), 90., 1e-9, "phase");
    };

    "a leading copy has a negative delay and phase, and half a cycle is 180 degrees"_test = [&] {
        const auto lagging = shifted(2UZ);
        expectNear(delay(view(lagging), view(leader)), -2., 1e-12, "delay");
        expectNear(phase(view(lagging), view(leader)), -90., 1e-9, "phase");
        const auto opposite = shifted(4UZ);
        expectNear(phase(view(leader), view(opposite)), 180., 1e-9, "phase of half a cycle");
    };

    "a channel without a rising edge has no delay"_test = [&] {
        const std::vector<double> flat(32UZ, 1.);
        expect(delay(view(leader), view(flat)).error() == Invalid::noTwoStates);
        expect(delay(view(flat), view(leader)).error() == Invalid::noTwoStates);
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
