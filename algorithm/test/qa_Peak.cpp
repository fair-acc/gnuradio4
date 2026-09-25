#include <boost/ut.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/peak/Peak.hpp>

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 16 // GCC 16.2 -O2 false positive on aggregates returned from constant-folded calls
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif

namespace {
using namespace boost::ut;
using namespace gr::algorithm::peak;

constexpr double kInf = std::numeric_limits<double>::infinity();

const auto vertices = [] {
    "the parabola through three samples of -(x - 0.3)^2 + 2 peaks at 0.3, height 2"_test = [] {
        const auto peak = parabolicPeak(-1.69 + 2.0, -0.09 + 2.0, -0.49 + 2.0);
        expect(le(std::abs(peak.offsetSamples - 0.3), 1e-12));
        expect(le(std::abs(peak.value - 2.0), 1e-12));
    };

    "three samples that do not bend downwards keep the middle one"_test = [] {
        const auto flat = parabolicPeak(1.0, 1.0, 1.0);
        const auto dip  = parabolicPeak(2.0, 1.0, 2.0);
        expect(eq(flat.offsetSamples, 0.0) && eq(flat.value, 1.0));
        expect(eq(dip.offsetSamples, 0.0) && eq(dip.value, 1.0));
    };

    "a sampled float Gaussian is placed within 0.05 of its true centre"_test = [] {
        for (const float trueOffset : {0.30f, -0.20f, 0.45f}) {
            constexpr float kSigma = 1.5f;
            const auto      at     = [&](float x) { return std::exp(-0.5f * (x - trueOffset) * (x - trueOffset) / (kSigma * kSigma)); };
            const auto      peak   = parabolicPeak(at(-1.f), at(0.f), at(1.f));
            expect(lt(std::abs(peak.offsetSamples - trueOffset), 0.05f)) << "offset" << trueOffset;
        }
    };

    "a Gaussian of height 3 at 0.3 with sigma 1.2 sampled at -1, 0, 1 is recovered exactly"_test = [] {
        const auto at   = [](double x) { return 3.0 * std::exp(-0.5 * (x - 0.3) * (x - 0.3) / (1.2 * 1.2)); };
        const auto peak = gaussianPeak(at(-1.0), at(0.0), at(1.0));
        expect(fatal(peak.has_value()));
        expect(le(std::abs(peak->offsetSamples - 0.3), 1e-12));
        expect(le(std::abs(peak->value - 3.0), 1e-12));
    };

    "a Gaussian vertex needs three positive samples"_test = [] {
        expect(!gaussianPeak(0.0, 1.0, 0.5).has_value());
        expect(!gaussianPeak(0.5, -1.0, 0.5).has_value());
        expect(!gaussianPeak(0.5, 1.0, std::numeric_limits<double>::quiet_NaN()).has_value());
    };

    "logs that do not bend downwards give the middle sample at offset 0"_test = [] {
        const auto flat = gaussianPeak(2.0, 2.0, 2.0);
        expect(fatal(flat.has_value()));
        expect(eq(flat->offsetSamples, 0.0) && le(std::abs(flat->value - 2.0), 1e-12));
    };
};

const auto localMaxima = [] {
    "a sample strictly above both neighbours is a local maximum"_test = [] {
        constexpr std::array<int, 5> v{0, 2, 1, 3, 3};
        expect(isLocalMaximum(v.data(), 1UZ, v.size()));
        expect(!isLocalMaximum(v.data(), 2UZ, v.size())) << "a dip";
    };

    "a plateau is no peak, on either of its samples"_test = [] {
        constexpr std::array<double, 4> v{0.0, 1.0, 1.0, 0.0};
        expect(!isLocalMaximum(v.data(), 1UZ, v.size()) && !isLocalMaximum(v.data(), 2UZ, v.size()));
    };

    "the first and last sample are never a local maximum, nor is a tiny array's"_test = [] {
        constexpr std::array<double, 3> v{5.0, 1.0, 5.0};
        expect(!isLocalMaximum(v.data(), 0UZ, v.size()) && !isLocalMaximum(v.data(), 2UZ, v.size()));
        expect(!isLocalMaximum(v.data(), 0UZ, 1UZ) && !isLocalMaximum(v.data(), 0UZ, 0UZ));
    };
};

const auto grids = [] {
    "indices, neighbours and the wrap of a periodic dimension, by hand"_test = [] {
        const Grid<double> g{.extents = {3UZ, 4UZ, 1UZ}, .rank = 2UZ, .origin = {}, .step = {1.0, 1.0, 1.0}, .periodic = {false, true, false}};
        expect(eq(g.stride(0UZ), 4UZ) && eq(g.stride(1UZ), 1UZ));
        expect(eq(g.indexOf(6UZ, 0UZ), 1UZ) && eq(g.indexOf(6UZ, 1UZ), 2UZ)) << "cell 6 is (1,2)";
        expect(eq(g.neighbour(6UZ, 1UZ, +1, 12UZ), 7UZ)) << "(1,3)";
        expect(eq(g.neighbour(7UZ, 1UZ, +1, 12UZ), 4UZ)) << "(1,3) + 1 wraps to (1,0)";
        expect(eq(g.neighbour(4UZ, 1UZ, -1, 12UZ), 7UZ)) << "(1,0) - 1 wraps to (1,3)";
        expect(eq(g.neighbour(4UZ, 0UZ, -1, 12UZ), 0UZ)) << "(0,0)";
        expect(eq(g.neighbour(0UZ, 0UZ, -1, 12UZ), 12UZ)) << "past the open dimension's end";
        expect(eq(g.squaredDistance(0UZ, 11UZ), 5UZ)) << "(0,0) to (2,3): 2 rows, 1 column across the seam";
    };

    "a peak is strictly above its neighbours and off an open dimension's ends"_test = [] {
        const std::array<double, 5> line{0, 2, 5, 1, 0};
        const std::array<double, 5> plateau{0, 4, 4, 1, 0};
        const Grid<double>          open{.extents = {5UZ, 1UZ, 1UZ}, .rank = 1UZ, .origin = {}, .step = {1.0, 1.0, 1.0}, .periodic = {}};
        expect(isPeak(line.data(), open, 2UZ, 5UZ));
        expect(!isPeak(line.data(), open, 0UZ, 5UZ)) << "an end";
        expect(!isPeak(plateau.data(), open, 1UZ, 5UZ) && !isPeak(plateau.data(), open, 2UZ, 5UZ)) << "a plateau";
        const std::array<double, 2> pair{1, 3};
        const Grid<double>          ringOfTwo{.extents = {2UZ, 1UZ, 1UZ}, .rank = 1UZ, .origin = {}, .step = {1.0, 1.0, 1.0}, .periodic = {true, false, false}};
        expect(isPeak(pair.data(), ringOfTwo, 1UZ, 2UZ) && !isPeak(pair.data(), ringOfTwo, 0UZ, 2UZ)) << "a ring of two";
        const std::array<double, 1> single{1};
        const Grid<double>          ringOfOne{.extents = {1UZ, 1UZ, 1UZ}, .rank = 1UZ, .origin = {}, .step = {1.0, 1.0, 1.0}, .periodic = {true, false, false}};
        expect(isPeak(single.data(), ringOfOne, 0UZ, 1UZ)) << "a ring of one: its own neighbour";
    };

    "a peak beside a periodic seam is placed by its parabola and wrapped into [origin, origin + period)"_test = [] {
        const std::array<double, 8> ring{5, 3, 0, 0, 0, 0, 0, 4};
        const Grid<double>          g{.extents = {8UZ, 1UZ, 1UZ}, .rank = 1UZ, .origin = {-4.0, 0.0, 0.0}, .step = {1.0, 1.0, 1.0}, .periodic = {true, false, false}};
        std::array<double, 3>       row{};
        describePeak(ring.data(), g, 0UZ, 8UZ, 2.0, row.data());
        // (4, 5, 3): offset (4 - 3) / (2 (4 - 10 + 3)) = -1/6, rise 1/24; -4 - 1/6 wraps by 8 to 3.8333
        expect(le(std::abs(row[0] - (-4.0 - 1.0 / 6.0 + 8.0)), 1e-12));
        expect(le(std::abs(row[1] - (5.0 + 1.0 / 24.0)), 1e-12));
        expect(le(std::abs(row[2] - (5.0 + 1.0 / 24.0) / 2.0), 1e-12));
    };
};

const auto selection = [] {
    // line of 10 cells: a peak at 3 from (2, 5, 4) -> offset 0.25, score 5.125, the reference; at 7 from (1, 3, 1) -> offset 0, score 3
    const std::array<double, 10> values{0, 0, 2, 5, 4, 0, 1, 3, 1, 0};
    const std::array<double, 10> peaks{-kInf, -kInf, -kInf, 5.125, -kInf, -kInf, -kInf, 3.0, -kInf, -kInf};
    const Grid<double>           line{.extents = {10UZ, 1UZ, 1UZ}, .rank = 1UZ, .origin = {}, .step = {1.0, 1.0, 1.0}, .periodic = {}};

    "the strongest peaks come out highest first, each placed by its parabola"_test = [&] {
        std::array<std::uint32_t, 3> accepted{};
        std::array<double, 6>        rows{};
        selectPeaks(values.data(), peaks.data(), line, 10UZ, 2UZ, 0.5, 1UZ, accepted.data(), rows.data(), 3UZ);
        expect(eq(accepted[0], 2U) && eq(accepted[1], 3U) && eq(accepted[2], 7U));
        expect(le(std::abs(rows[0] - 3.25), 1e-12) && le(std::abs(rows[1] - 5.125), 1e-12) && le(std::abs(rows[2] - 1.0), 1e-12));
        expect(le(std::abs(rows[3] - 7.0), 1e-12) && le(std::abs(rows[4] - 3.0), 1e-12) && le(std::abs(rows[5] - 3.0 / 5.125), 1e-12));
    };

    "a peak below the relative threshold, past max_peaks or within the separation is dropped"_test = [&] {
        std::array<std::uint32_t, 3> accepted{};
        std::array<double, 6>        rows{};
        selectPeaks(values.data(), peaks.data(), line, 10UZ, 2UZ, 0.7, 1UZ, accepted.data(), rows.data(), 3UZ);
        expect(eq(accepted[0], 1U)) << "3 < 0.7 * 5.125";
        selectPeaks(values.data(), peaks.data(), line, 10UZ, 1UZ, 0.0, 1UZ, accepted.data(), rows.data(), 3UZ);
        expect(eq(accepted[0], 1U)) << "max_peaks 1";
        selectPeaks(values.data(), peaks.data(), line, 10UZ, 2UZ, 0.0, 16UZ, accepted.data(), rows.data(), 3UZ);
        expect(eq(accepted[0], 1U)) << "4 cells apart is not more than 4";
        selectPeaks(values.data(), peaks.data(), line, 10UZ, 2UZ, 0.0, 15UZ, accepted.data(), rows.data(), 3UZ);
        expect(eq(accepted[0], 2U)) << "4 cells apart is more than sqrt(15)";
    };

    "a map whose reference is not positive gives no peaks"_test = [] {
        const std::array<double, 3>  negative{-3.0, -1.0, -2.0};
        const std::array<double, 3>  scores{-kInf, -1.0, -kInf};
        const Grid<double>           g{.extents = {3UZ, 1UZ, 1UZ}, .rank = 1UZ, .origin = {}, .step = {1.0, 1.0, 1.0}, .periodic = {}};
        std::array<std::uint32_t, 2> accepted{7U, 7U};
        std::array<double, 3>        rows{};
        selectPeaks(negative.data(), scores.data(), g, 3UZ, 1UZ, 0.0, 1UZ, accepted.data(), rows.data(), 3UZ);
        expect(eq(accepted[0], 0U));
    };
};

const auto widths = [] {
    "the half-maximum widths of a triangle of height 4 are 2 and 2, by hand"_test = [] {
        constexpr std::array<double, 9> v{0, 1, 2, 3, 4, 3, 2, 1, 0};
        const auto                      w = halfMaxWidths(std::span<const double>(v), 4UZ);
        expect(le(std::abs(w.left - 2.0), 1e-12) && le(std::abs(w.right - 2.0), 1e-12));
    };

    "a baseline of 1 puts the half level at 2.5 and the crossings at 1.5 either side"_test = [] {
        constexpr std::array<double, 9> v{0, 1, 2, 3, 4, 3, 2, 1, 0};
        const auto                      w = halfMaxWidths(std::span<const double>(v), 4UZ, 1.0);
        expect(le(std::abs(w.left - 1.5), 1e-12) && le(std::abs(w.right - 1.5), 1e-12));
    };

    "the flanks of {1, 2, 3, 2, 1} cross at 1.5, those of the steeper {1, 6, 2, 0} at 0.6 and 0.75"_test = [] {
        constexpr std::array<double, 5> kSymmetric{1, 2, 3, 2, 1};
        const auto                      s = halfMaxWidths(std::span<const double>(kSymmetric), 2UZ);
        expect(le(std::abs(s.left - 1.5), 1e-12) && le(std::abs(s.right - 1.5), 1e-12));
        constexpr std::array<double, 4> kSteep{1, 6, 2, 0};
        const auto                      t = halfMaxWidths(std::span<const double>(kSteep), 1UZ);
        expect(le(std::abs(t.left - (6.0 - 3.0) / (6.0 - 1.0)), 1e-12)) << "half level 3: (6 - 3) / (6 - 1)";
        expect(le(std::abs(t.right - (6.0 - 3.0) / (6.0 - 2.0)), 1e-12)) << "(6 - 3) / (6 - 2)";
    };

    "a flank that ends above the half level gives NaN on that side only"_test = [] {
        constexpr std::array<double, 4> v{3, 4, 3, 2};
        const auto                      w = halfMaxWidths(std::span<const double>(v), 1UZ);
        expect(std::isnan(w.left)) << "data[0] = 3 > 2";
        expect(le(std::abs(w.right - 2.0), 1e-12)) << "one full sample, then (3 - 2) / (3 - 2)";
        expect(std::isnan(halfMaxWidths(std::span<const double>(v), 0UZ).left)) << "no sample to the left at all";
    };
};

const auto estimators = [] {
    "prominence is the height over the higher of the two flanks' lowest points, by hand"_test = [] {
        constexpr std::array<float, 7> v{0, 2, 1, 5, 1, 3, 0};
        const std::span<const float>   s(v);
        expect(eq(estimateProminence(s, 3UZ), 5.f)) << "global maximum: down to 0 either side";
        expect(eq(estimateProminence(s, 1UZ), 1.f)) << "right flank only reaches 1 before the higher 5";
        expect(eq(estimateProminence(s, 5UZ), 2.f)) << "left flank only reaches 1 before the higher 5";
    };

    "isolation is the distance to the nearest equal-or-higher peak, else the length"_test = [] {
        constexpr std::array<float, 7> v{0, 2, 1, 5, 1, 3, 0};
        const std::span<const float>   s(v);
        expect(eq(estimateIsolation(s, 5UZ), 2.f)) << "the 5 at index 3";
        expect(eq(estimateIsolation(s, 3UZ), 7.f)) << "nothing as high: the array length";
    };

    "the noise estimate is the median and 2.50569 times the median absolute deviation"_test = [] {
        constexpr std::array<float, 5> v{5, 1, 3, 2, 4};
        const auto                     noise = estimateMagnitudeNoise(std::span<const float>(v));
        expect(eq(noise.median, 3.f));
        expect(lt(std::abs(noise.robustSigma - kHalfNormalMadToSigma), 1e-6f)) << "absolute deviations {2, 1, 0, 1, 2} have median 1";
        const auto empty = estimateMagnitudeNoise({});
        expect(eq(empty.median, 0.f) && eq(empty.robustSigma, 1.f));
    };

    "the median of an odd array is its middle value"_test = [] {
        std::array<float, 3> v{5.f, 1.f, 3.f};
        expect(eq(medianOf(v), 3.f));
        expect(eq(medianOf({}), 0.f));
    };

    "the excess kurtosis of a single spike among seven samples is 13/6"_test = [] {
        // mean 1/7, m2 = 6/49, m4 = 186/2401, m4 / m2^2 = 31/6
        constexpr std::array<float, 7> v{0, 0, 0, 1, 0, 0, 0};
        expect(lt(std::abs(estimateExcessKurtosis(std::span<const float>(v), 3UZ, 1.f) - 13.f / 6.f), 1e-5f));
        constexpr std::array<float, 3> kTooShort{0, 1, 0};
        expect(eq(estimateExcessKurtosis(std::span<const float>(kTooShort), 1UZ, 1.f), 0.f));
    };

    "the uncertainty falls with the signal-to-noise ratio"_test = [] {
        const auto u = estimateUncertainty(1.f, 10.f, 2.f);
        expect(lt(std::abs(u.position - 2.f / 10.001f), 1e-5f)) << "max(sigma, 1) / (snr + 1e-3)";
        expect(lt(std::abs(u.width - u.position / (std::numbers::sqrt2_v<float> * 2.f)), 1e-5f));
        expect(eq(u.amplitude, 1.f));
    };

    "the containment widths of a Gaussian are 1.406431, 2.904447 and 4.196960 sigma"_test = [] {
        const auto w = containmentWidths(3.f, 3.f, 0.f);
        expect(lt(std::abs(w.w68 / 3.f - 1.406431f), 2e-3f));
        expect(lt(std::abs(w.w96 / 3.f - 2.904447f), 2e-3f));
        expect(lt(std::abs(w.w99 / 3.f - 4.196960f), 3e-3f));
    };
};

const auto results = [] {
    "the half-maximum constant is 1 / sqrt(2 ln 2)"_test = [] {
        const float analytic = 1.f / std::sqrt(2.f * std::numbers::ln2_v<float>);
        expect(lt(std::abs(kGaussianHalfMaxToSigma - analytic), 1e-7f));
    };

    "a peak's full width is the sum of its half-widths and its equivalent sigma follows from it"_test = [] {
        const PeakResult peak{.hwhmLeft = 3.f, .hwhmRight = 5.f};
        expect(eq(peak.fwhm(), 8.f));
        expect(lt(std::abs(peak.gaussianEquivalentSigma() - 4.f * kGaussianHalfMaxToSigma), 1e-6f));
    };

    "the asymmetry is half the log of the flank ratio and 0 for a degenerate flank"_test = [] {
        expect(lt(std::abs(asymmetryOf({.hwhmLeft = 4.f, .hwhmRight = 8.f}) - 0.5f * std::numbers::ln2_v<float>), 1e-6f));
        expect(eq(asymmetryOf({.hwhmLeft = 0.f, .hwhmRight = 8.f}), 0.f));
    };
};

const auto subtraction = [] {
    "subtracting a Gaussian from its own samples leaves nothing"_test = [] {
        std::vector<float> spectrum(7);
        for (std::size_t i = 0UZ; i < spectrum.size(); ++i) {
            const float x = static_cast<float>(i) - 3.f;
            spectrum[i]   = 2.f * std::exp(-0.5f * x * x);
        }
        subtractPeak(spectrum, 3.f, 2.f, 1.f, 1.f, 0.f, SubtractionShape::Gaussian);
        for (const float residual : spectrum) {
            expect(lt(std::abs(residual), 1e-6f));
        }
    };

    "a Lorentzian of height 2 and half-width 1 is 2, 1, 0.4 and 0.2 at 0, 1, 2 and 3 samples"_test = [] {
        std::vector<float> spectrum{0.2f, 0.4f, 1.f, 2.f, 1.f, 0.4f, 0.2f};
        subtractPeak(spectrum, 3.f, 2.f, 1.f, 1.f, 0.f, SubtractionShape::Lorentzian);
        for (const float residual : spectrum) {
            expect(lt(std::abs(residual), 1e-6f));
        }
    };

    "the automatic shape follows the excess kurtosis: Lorentzian above 1.5"_test = [] {
        std::vector<float> spectrum{0.2f, 0.4f, 1.f, 2.f, 1.f, 0.4f, 0.2f};
        subtractPeak(spectrum, 3.f, 2.f, 1.f, 1.f, 2.f, SubtractionShape::Auto);
        for (const float residual : spectrum) {
            expect(lt(std::abs(residual), 1e-6f));
        }
    };

    "a pseudo-Voigt of Lorentzian fraction eta, subtracted with the automatic shape at its excess kurtosis 3 eta, leaves the residual near zero"_test = [] {
        constexpr float       kEta       = 0.3f;
        constexpr float       kKurtosis  = 3.f * kEta;
        constexpr float       kCentre    = 16.3f;
        constexpr float       kAmplitude = 2.f;
        constexpr float       kSigma     = 1.5f;
        constexpr std::size_t kLength    = 33UZ;
        constexpr float       kRadius    = 6.f * kSigma;
        constexpr float       kFloatSum  = 1e-6f;

        const auto         pseudoVoigt = [&](float x) { return kAmplitude * ((1.f - kEta) * std::exp(-0.5f * x * x / (kSigma * kSigma)) + kEta * kSigma * kSigma / (kSigma * kSigma + x * x)); };
        std::vector<float> spectrum(kLength);
        for (std::size_t i = 0UZ; i < kLength; ++i) {
            spectrum[i] = pseudoVoigt(static_cast<float>(i) - kCentre);
        }
        expect(gt(kKurtosis, 0.5f) && lt(kKurtosis, 1.5f)) << "the Voigt band of the automatic shape";

        std::vector<float> subtracted = spectrum;
        subtractPeak(subtracted, kCentre, kAmplitude, kSigma, kSigma, kKurtosis, SubtractionShape::Auto);
        for (std::size_t i = 0UZ; i < kLength; ++i) {
            const float x = static_cast<float>(i) - kCentre;
            if (std::abs(x) <= kRadius) {
                expect(lt(std::abs(subtracted[i]), kFloatSum)) << "sample" << i << "within six sigma";
            } else if (std::abs(x) > kRadius + 1.f) {
                expect(eq(subtracted[i], spectrum[i])) << "sample" << i << "more than a sample beyond six sigma is left alone";
            }
        }

        for (const SubtractionShape single : {SubtractionShape::Gaussian, SubtractionShape::Lorentzian}) {
            std::vector<float> wrongShape = spectrum;
            subtractPeak(wrongShape, kCentre, kAmplitude, kSigma, kSigma, kKurtosis, single);
            float worst = 0.f;
            for (std::size_t i = 0UZ; i < kLength; ++i) {
                worst = std::max(worst, std::abs(wrongShape[i]));
            }
            expect(gt(worst, 0.05f)) << "a single-shape subtraction leaves the mix's difference behind";
        }
    };
};
} // namespace

int main() {
    "parabolic and Gaussian vertices"_test = vertices;
    "local maxima"_test                    = localMaxima;
    "grids and their peaks"_test           = grids;
    "peak selection"_test                  = selection;
    "half-maximum widths"_test             = widths;
    "peak estimators"_test                 = estimators;
    "peak results"_test                    = results;
    "peak subtraction"_test                = subtraction;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
