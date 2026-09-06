#include <boost/ut.hpp>

#include <cmath>
#include <complex>
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
        expect(std::abs(Resampler::interpolate(1.f, 2.f, 3.f, 4.f, 0.f) - 2.f) < 1e-6f) << "t = 0 is the second sample";
        expect(std::abs(Resampler::interpolate(1.f, 2.f, 3.f, 4.f, 1.f) - 3.f) < 1e-6f) << "t = 1 is the third";
    };

    "a straight line stays a straight line"_test = [] {
        // cubic Hermite reproduces polynomials up to degree three, so a ramp must come back exactly
        bool exact = true;
        for (float t = 0.f; t < 1.f; t += 0.1f) {
            exact = exact && std::abs(Resampler::interpolate(10.f, 11.f, 12.f, 13.f, t) - (11.f + t)) < 1e-5f;
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

const boost::ut::suite<"HermiteResampler complex"> _hermiteComplex = [] {
    using namespace boost::ut;
    using C         = std::complex<float>;
    using Complexes = gr::algorithm::filter::HermiteResampler<C>;

    "a complex ramp interpolates each part independently"_test = [] {
        // real and imaginary parts are two real ramps, so the interpolant must reproduce both exactly
        std::vector<C> signal(16);
        for (std::size_t n = 0UZ; n < signal.size(); ++n) {
            signal[n] = C{static_cast<float>(n), static_cast<float>(-2 * static_cast<int>(n))};
        }
        std::vector<C> output(8);
        double         phase    = 0.0;
        const auto     progress = Complexes::resample(signal, output, 0.5, phase);

        expect(gt(progress.produced, 0UZ));
        bool exact = true;
        for (std::size_t n = 0UZ; n < progress.produced; ++n) {
            const float position = static_cast<float>(n) * 2.f;
            exact                = exact && std::abs(output[n].real() - position) < 1e-4f && std::abs(output[n].imag() + 2.f * position) < 1e-4f;
        }
        expect(exact) << "each part of a complex sample follows the same interpolant as a real one";
    };
};

int main() { /* tests run from the suite */ }
