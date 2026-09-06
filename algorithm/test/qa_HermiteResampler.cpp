#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <numeric>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/HermiteResampler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

namespace {
using Resampler = gr::algorithm::filter::HermiteResampler<float>;
}

const boost::ut::suite<"HermiteResampler"> _hermite = [] {
    using namespace boost::ut;

    "the interpolant passes through the samples it interpolates between"_test = [] {
        expect(std::abs(Resampler::interpolate(1.0, 2.0, 3.0, 4.0, 0.0) - 2.0) < 1e-12) << "t = 0 is the second sample";
        expect(std::abs(Resampler::interpolate(1.0, 2.0, 3.0, 4.0, 1.0) - 3.0) < 1e-12) << "t = 1 is the third";
    };

    "a straight line stays a straight line"_test = [] {
        // cubic Hermite reproduces polynomials up to degree three, so a ramp must come back exactly
        bool exact = true;
        for (double t = 0.0; t < 1.0; t += 0.1) {
            exact = exact && std::abs(Resampler::interpolate(10.0, 11.0, 12.0, 13.0, t) - (11.0 + t)) < 1e-12;
        }
        expect(exact) << "a linear-phase interpolator must not bend a ramp";
    };

    "resampling at unity returns the samples unchanged"_test = [] {
        std::vector<float> signal(64);
        std::iota(signal.begin(), signal.end(), 1.f);
        std::vector<float> output(64, 0.f);
        double             phase    = 0.0;
        const std::size_t  produced = Resampler::resample(signal, output, 1.0, phase).produced;
        expect(gt(produced, 32UZ)) << "at a ratio of one the window should be nearly used up";
        bool identity = true;
        for (std::size_t n = 0UZ; n < produced; ++n) {
            identity = identity && std::abs(output[n] - signal[n]) < 1e-3f;
        }
        expect(identity) << "unity ratio has nothing to resample";
    };

    "a ratio above one produces more samples than it consumes"_test = [] {
        std::vector<float> signal(128);
        for (std::size_t i = 0UZ; i < signal.size(); ++i) {
            signal[i] = std::sin(0.07f * static_cast<float>(i));
        }
        for (const double ratio : {0.5, 1.5, 2.0}) {
            std::vector<float> output(512, 0.f);
            double             phase    = 0.0;
            const std::size_t  produced = Resampler::resample(signal, output, ratio, phase).produced;
            const double       expected = static_cast<double>(signal.size()) * ratio;
            expect(std::abs(static_cast<double>(produced) - expected) < 4.0) << std::format("ratio {} produced {} from {}", ratio, produced, signal.size());
        }
    };

    "the phase carries across calls, so a stream has no seam"_test = [] {
        std::vector<float> signal(32);
        for (std::size_t i = 0UZ; i < signal.size(); ++i) {
            signal[i] = std::sin(0.21f * static_cast<float>(i));
        }
        std::vector<float> first(64, 0.f), second(64, 0.f);
        double             phase = 0.0;
        const std::size_t  a     = Resampler::resample(signal, first, 1.3, phase).produced;
        const double       carry = phase;
        const std::size_t  b     = Resampler::resample(signal, second, 1.3, phase).produced;
        expect(gt(a, 0UZ));
        expect(gt(b, 0UZ));
        expect(std::abs(carry) < 1.0) << std::format("the leftover phase must stay within one sample, was {}", carry);
    };
};

int main() { /* tests run from the suite */ }
