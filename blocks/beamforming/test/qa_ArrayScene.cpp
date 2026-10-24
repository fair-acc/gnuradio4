#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <format>
#include <numbers>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/basic/StreamToTensor.hpp>
#include <gnuradio-4.0/beamforming/Beamformer.hpp>
#include <gnuradio-4.0/beamforming/CrossSpectralMatrix.hpp>
#include <gnuradio-4.0/fourier/STFT.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/localisation/SourceLocation.hpp>
#include <gnuradio-4.0/math/TensorPeakFinder.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/ArrayScene.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {
using namespace boost::ut;
using gr::testing::TensorCollector;

using Source = gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;

constexpr double      kPi        = std::numbers::pi;
constexpr std::size_t kElements  = gr::testing::kSquareArrayElements;
constexpr std::size_t kSamples   = 16384UZ;
constexpr std::size_t kFrameSize = 256UZ;
constexpr std::size_t kHop       = 128UZ;
constexpr float       kRate      = 1000.f;

const auto scene = [] {
    "sensor noise is reproducible, distinct per microphone and 20 dB below clean mixture power"_test = [] {
        constexpr std::array kClean{-3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f};
        auto                 noisy = kClean;
        gr::testing::addMicrophoneNoise(noisy, 0UZ);
        double addedPower = 0.0;
        for (std::size_t n = 0UZ; n < kClean.size(); ++n) {
            const double added = static_cast<double>(noisy[n]) - static_cast<double>(kClean[n]);
            addedPower += added * added;
        }
        expect(le(std::abs(44.0 / addedPower - 100.0), 1e-4));
        auto repeated = kClean;
        gr::testing::addMicrophoneNoise(repeated, 0UZ);
        expect(std::ranges::equal(noisy, repeated));
        auto otherMicrophone = kClean;
        gr::testing::addMicrophoneNoise(otherMicrophone, 1UZ);
        expect(!std::ranges::equal(noisy, otherMicrophone));
        std::array<float, 8> silent{};
        gr::testing::addMicrophoneNoise(silent, 0UZ);
        expect(std::ranges::all_of(silent, [](float value) { return value == 0.f; }));
        gr::testing::addMicrophoneNoise({}, 0UZ);
        std::array single{2.f};
        gr::testing::addMicrophoneNoise(single, 0UZ);
        expect(le(std::abs(std::abs(single[0] - 2.f) - 0.2f), 1e-6f));
        for (const float amplitude : {1e20f, 1e-20f}) {
            std::array extreme{amplitude};
            gr::testing::addMicrophoneNoise(extreme, 0UZ);
            const double relativeNoise = (static_cast<double>(extreme[0]) - static_cast<double>(amplitude)) / static_cast<double>(amplitude);
            expect(std::isfinite(extreme[0]));
            expect(le(std::abs(std::abs(relativeNoise) - 0.1), 1e-6));
        }
    };

    "arrival sign, fractional interpolation, gain and mixing follow explicit sample fixtures"_test = [] {
        constexpr std::size_t   kCount = 16UZ;
        constexpr std::uint32_t kSeed  = 21U;
        const auto              noise  = gr::testing::whiteNoise(kCount + 2UZ * gr::testing::kMaxArrivalDelaySamples, kSeed);
        constexpr std::array    kDelays{0.0, 1.0, -1.0, 0.25, -0.25};
        for (std::size_t fixture = 0UZ; fixture < kDelays.size(); ++fixture) {
            const double     delay = kDelays[fixture];
            const std::array arrivals{gr::testing::Arrival{delay, -2.0, kSeed}};
            const auto       actual = gr::testing::delayedNoises(arrivals, kCount);
            for (std::size_t n = 0UZ; n < kCount; ++n) {
                const std::size_t centre = n + gr::testing::kMaxArrivalDelaySamples;
                const std::array  expected{noise[centre], noise[centre - 1UZ], noise[centre + 1UZ], 0.75 * noise[centre] + 0.25 * noise[centre - 1UZ], 0.75 * noise[centre] + 0.25 * noise[centre + 1UZ]};
                expect(le(std::abs(static_cast<double>(actual[n]) + 2.0 * expected[fixture]), 1e-7));
            }
            const auto longer = gr::testing::delayedNoises(arrivals, 2UZ * kCount);
            expect(std::ranges::equal(actual, longer | std::views::take(kCount)));
        }
        const std::array cancellation{gr::testing::Arrival{0.25, 1.0, kSeed}, gr::testing::Arrival{0.25, -1.0, kSeed}};
        expect(std::ranges::all_of(gr::testing::delayedNoises(cancellation, kCount), [](float value) { return value == 0.f; }));
        expect(std::ranges::all_of(gr::testing::delayedNoises({}, kCount), [](float value) { return value == 0.f; }));
        expect(gr::testing::delayedNoises(cancellation, 0UZ).empty());
    };

    "clean and noisy sources: SRP-PHAT finds both azimuths, MVDR gives each its own beam at its own level"_test = [] {
        for (const bool addNoise : {false, true}) {
            using namespace gr;
            gr::testing::GraphFixture<> fixture;
            const gr::property_map      geometry = gr::testing::squareArrayGeometry();
            auto&                       frames   = fixture.emplace<gr::basic::StreamsToTensor<float, kElements>>({{"size", gr::Size_t(kFrameSize)}, {"stride", gr::Size_t(kHop)}});
            for (std::size_t m = 0UZ; m < kElements; ++m) {
                std::vector<float> samples = gr::testing::squareArraySamples(m, kSamples);
                if (addNoise) {
                    gr::testing::addMicrophoneNoise(samples, m);
                }
                auto& source = fixture.emplace<Source>({{"values", samples}, {"n_samples_max", gr::Size_t(kSamples)}, {"mark_tag", false}});
                source._tags = {gr::testing::OwningTag(0UZ, {{std::string(tag::SAMPLE_RATE.key()), kRate}})};
                expect(fixture.graph.connect(source, "out", frames, std::format("in#{}", m)).has_value());
            }
            auto&            gcc  = fixture.emplace<gr::localisation::PairCrossCorrelation<float>>({{"max_lag", gr::Size_t(8)}});
            gr::property_map scan = geometry;
            scan.insert_or_assign("grid_steps", gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(360)}));
            auto& srp        = fixture.emplace<gr::localisation::SteeredPowerMap<float>>(scan);
            auto& peaks      = fixture.emplace<gr::blocks::math::TensorPeakFinder<float>>({{"max_peaks", gr::Size_t(2)}, {"threshold", 0.3f}, {"min_separation", 20.f}});
            auto& peakSink   = fixture.emplace<TensorCollector<float>>();
            auto& stft       = fixture.emplace<gr::blocks::fft::STFT<float>>();
            auto& covariance = fixture.emplace<gr::beamforming::CrossSpectralMatrix<float>>({{"alpha", 0.05f}});
            auto& steering   = fixture.emplace<gr::beamforming::SteeringVectors<float>>(geometry);
            auto& mvdr       = fixture.emplace<gr::beamforming::MinimumVarianceBeamformer<float>>({{"diagonal_loading", 0.01f}});
            auto& beams      = fixture.emplace<TensorCollector<std::complex<float>>>();
            expect(fixture.connect<"out", "in">(frames, gcc).has_value());
            expect(fixture.connect<"out", "in">(gcc, srp).has_value());
            expect(fixture.connect<"out", "in">(srp, peaks).has_value());
            expect(fixture.connect<"out", "in">(peaks, peakSink).has_value());
            expect(fixture.connect<"out", "peaks">(peaks, steering).has_value());
            expect(fixture.connect<"out", "in">(frames, stft).has_value());
            expect(fixture.connect<"out", "in">(stft, covariance).has_value());
            expect(fixture.connect<"out", "in">(stft, steering).has_value());
            expect(fixture.connect<"out", "in">(stft, mvdr).has_value());
            expect(fixture.connect<"out", "steering">(steering, mvdr).has_value());
            expect(fixture.connect<"out", "covariance">(covariance, mvdr).has_value());
            expect(fixture.connect<"out", "in">(mvdr, beams).has_value());
            const auto ran = fixture.run();
            expect(ran.has_value()) << (ran.has_value() ? std::string{} : ran.error().message);
            expect(fatal(ran.has_value()));

            expect(fatal(!peakSink._elements.empty()));
            const auto& found = peakSink._elements.back();
            expect(fatal(eq(peakSink._shapes.back()[0], 2UZ))) << "two sources";
            const double step = 2.0 * kPi / 360.0;
            expect(le(std::abs(static_cast<double>(found[0])), step)) << "source 1 at azimuth 0, within a grid step";
            expect(le(std::abs(static_cast<double>(found[3]) - kPi / 2.0), step)) << "source 2 at azimuth pi/2";

            // the beams [2,F] of the last 20 frames: below the spatial-alias frequency c / (2 * 4 m) = 125 Hz, bins 1..31 of 1000 Hz / 256
            expect(fatal(ge(beams._elements.size(), 20UZ)));
            double power0 = 0.0;
            double power1 = 0.0;
            for (std::size_t frame = beams._elements.size() - 20UZ; frame < beams._elements.size(); ++frame) {
                expect(fatal(eq(beams._shapes[frame][0], 2UZ))) << "frame" << frame << "has both beams";
                const std::size_t F = beams._shapes[frame][1];
                for (std::size_t bin = 1UZ; bin < 32UZ; ++bin) {
                    power0 += static_cast<double>(std::norm(beams._elements[frame][bin]));
                    power1 += static_cast<double>(std::norm(beams._elements[frame][F + bin]));
                }
            }
            expect(ge(power0 / power1, 2.5) && le(power0 / power1, 6.5)) << "beam 0 holds source 1, beam 1 source 2, 6 dB below:" << power0 / power1;
        }
    };
};
} // namespace

int main() {
    "the array chain"_test = scene;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
