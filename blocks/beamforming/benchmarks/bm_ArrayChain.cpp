#include <benchmark.hpp>

#include <array>
#include <complex>
#include <format>
#include <print>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/StreamToTensor.hpp>
#include <gnuradio-4.0/beamforming/Beamformer.hpp>
#include <gnuradio-4.0/beamforming/CrossSpectralMatrix.hpp>
#include <gnuradio-4.0/fourier/STFT.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/localisation/SourceLocation.hpp>
#include <gnuradio-4.0/math/TensorPeakFinder.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {
constexpr std::size_t kElements   = 4UZ;
constexpr gr::Size_t  kFrameSize  = 256U;
constexpr gr::Size_t  kHop        = 128U;
constexpr gr::Size_t  kFrames     = 2048U;
constexpr gr::Size_t  kSamples    = kHop * (kFrames - 1U) + kFrameSize;
constexpr float       kSampleRate = 16000.f;

template<typename T>
struct TensorDrop : gr::Block<TensorDrop<T>> {
    gr::TensorPortIn<T> in;
    GR_MAKE_REFLECTABLE(TensorDrop, in);

    gr::work::Status processBulk(gr::InputSpanLike auto& /*input*/) { return gr::work::Status::OK; }
};

[[nodiscard]] gr::property_map geometry() {
    return {{"element_positions", gr::algorithm::geometry::grid(2UZ, 2UZ, 0.1)}, //
        {"propagation_speed", 343.0}, {"field", std::string("far")}, {"sample_rate", kSampleRate}};
}

enum class Stages : std::uint8_t { frames, gcc, srp, peaks, stft, covariance, delayAndSum, mvdr, lcmv };

void runChain(Stages last) {
    gr::Graph graph;
    auto&     frames = graph.emplaceBlock<gr::basic::StreamsToTensor<float, kElements>>({{"size", kFrameSize}, {"stride", kHop}});
    for (std::size_t m = 0UZ; m < kElements; ++m) {
        const auto noise   = gr::testing::whiteNoise(kSamples, 21U + static_cast<std::uint32_t>(m));
        const auto samples = noise | std::views::transform([](double value) { return static_cast<float>(value); }) | std::ranges::to<std::vector<float>>();
        auto&      source  = graph.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"values", samples}, {"n_samples_max", kSamples}, {"mark_tag", false}});
        source._tags       = {gr::testing::OwningTag(0UZ, {{std::string(gr::tag::kSampleRate.key()), kSampleRate}})};
        std::ignore        = graph.connect(source, "out", frames, std::format("in#{}", m));
    }
    const auto through = [&graph]<typename TBlock, typename TOut>(auto& from, gr::property_map settings, bool lastStage) -> TBlock& {
        auto& block = graph.emplaceBlock<TBlock>(std::move(settings));
        std::ignore = graph.connect<"out", "in">(from, block);
        if (lastStage) {
            std::ignore = graph.connect<"out", "in">(block, graph.emplaceBlock<TensorDrop<TOut>>());
        }
        return block;
    };
    using C = std::complex<float>;
    if (last == Stages::frames) {
        std::ignore = graph.connect<"out", "in">(frames, graph.emplaceBlock<TensorDrop<float>>());
    } else if (last <= Stages::peaks) {
        auto& gcc = through.template operator()<gr::localisation::PairCrossCorrelation<float>, float>(frames, {{"max_lag", gr::Size_t(8)}}, last == Stages::gcc);
        if (last >= Stages::srp) {
            auto& srp = through.template operator()<gr::localisation::SteeredPowerMap<float>, float>(gcc, geometry(), last == Stages::srp);
            if (last == Stages::peaks) {
                std::ignore = through.template operator()<gr::blocks::math::TensorPeakFinder<float>, float>(srp, {{"max_peaks", gr::Size_t(2)}}, true);
            }
        }
    } else {
        auto& stft = through.template operator()<gr::blocks::fft::STFT<float>, C>(frames, {}, last == Stages::stft);
        if (last == Stages::covariance) {
            std::ignore = through.template operator()<gr::beamforming::CrossSpectralMatrix<float>, C>(stft, {}, true);
        }
        if (last >= Stages::delayAndSum) {
            auto& peaks    = graph.emplaceBlock<gr::testing::TensorSource<float>>();
            peaks._frames  = {{.extents = {2UZ, 3UZ}, .elements = {0.f, 1.f, 1.f, 1.5f, 1.f, 1.f}}};
            auto& steering = graph.emplaceBlock<gr::beamforming::SteeringVectors<float>>(geometry());
            std::ignore    = graph.connect<"out", "peaks">(peaks, steering);
            std::ignore    = graph.connect<"out", "in">(stft, steering);
            auto& drop     = graph.emplaceBlock<TensorDrop<C>>();
            if (last == Stages::delayAndSum) {
                auto& beams = graph.emplaceBlock<gr::beamforming::ConventionalBeamformer<float>>();
                std::ignore = graph.connect<"out", "in">(stft, beams);
                std::ignore = graph.connect<"out", "steering">(steering, beams);
                std::ignore = graph.connect<"out", "in">(beams, drop);
            } else {
                auto& covariance       = graph.emplaceBlock<gr::beamforming::CrossSpectralMatrix<float>>();
                std::ignore            = graph.connect<"out", "in">(stft, covariance);
                const auto addAdaptive = [&]<typename TBeamformer> {
                    auto& beams = graph.emplaceBlock<TBeamformer>();
                    std::ignore = graph.connect<"out", "in">(stft, beams);
                    std::ignore = graph.connect<"out", "steering">(steering, beams);
                    std::ignore = graph.connect<"out", "covariance">(covariance, beams);
                    std::ignore = graph.connect<"out", "in">(beams, drop);
                };
                if (last == Stages::mvdr) {
                    addAdaptive.template operator()<gr::beamforming::MinimumVarianceBeamformer<float>>();
                } else {
                    addAdaptive.template operator()<gr::beamforming::NullingBeamformer<float>>();
                }
            }
        }
    }
    gr::scheduler::Simple<> scheduler;
    std::ignore    = scheduler.exchange(std::move(graph));
    const auto ran = scheduler.runAndWait();
    boost::ut::expect(ran.has_value()) << (ran ? std::string{} : ran.error().message);
}
} // namespace

inline const boost::ut::suite<"ArrayChain"> _arrayChain = [] {
    using namespace benchmark;
    "4 streams -> frames [4,256]"_benchmark.repeat<3>(kFrames)                                                             = [] { runChain(Stages::frames); };
    "  -> PairCrossCorrelation, max_lag 8: [6,17]"_benchmark.repeat<3>(kFrames)                                            = [] { runChain(Stages::gcc); };
    "  -> SteeredPowerMap over 360 azimuths"_benchmark.repeat<3>(kFrames)                                                  = [] { runChain(Stages::srp); };
    "  -> TensorPeakFinder, 2 peaks"_benchmark.repeat<3>(kFrames)                                                          = [] { runChain(Stages::peaks); };
    "frames -> STFT: [4,129]"_benchmark.repeat<3>(kFrames)                                                                 = [] { runChain(Stages::stft); };
    "  -> CrossSpectralMatrix: [129,4,4]"_benchmark.repeat<3>(kFrames)                                                     = [] { runChain(Stages::covariance); };
    "  -> SteeringVectors, 2 sources, and ConventionalBeamformer: [2,129]"_benchmark.repeat<3>(kFrames)                    = [] { runChain(Stages::delayAndSum); };
    "  -> CrossSpectralMatrix, SteeringVectors and NullingBeamformer (host): [2,129]"_benchmark.repeat<3>(kFrames)         = [] { runChain(Stages::lcmv); };
    "  -> CrossSpectralMatrix, SteeringVectors and MinimumVarianceBeamformer (host): [2,129]"_benchmark.repeat<3>(kFrames) = [] { runChain(Stages::mvdr); };
};

int main() { /* the suite runs at static initialisation */ }
