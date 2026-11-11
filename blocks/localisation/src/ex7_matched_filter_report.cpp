/**
 * Matched filter of a chirp: the chirp SignalGenerator emits, heard `--delay` samples later under noise, against Correlator built from the same
 * settings; prints the lag of the largest coefficient, which is the delay, and the coefficient there.
 * Usage: ex7_matched_filter_report [--delay 300] [--noise 0.5]
 */

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <print>
#include <random>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/algorithm/signal/SignalGeneratorCore.hpp>
#include <gnuradio-4.0/filter/Correlator.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

int main(int argc, char** argv) {
    constexpr float                       kSampleRate = 48000.f;
    constexpr gr::signal::ChirpParameters kChirp{.startHz = 1000.0, .stopHz = 8000.0, .durationSeconds = 0.02, .repeat = false, .periodSeconds = 0.0, .tukeyFraction = 0.25};
    std::size_t                           delay = 300UZ;
    double                                noise = 0.5;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string_view option{argv[i]};
        const std::string_view value{argv[i + 1]};
        if (option == "--delay") {
            std::ignore = std::from_chars(value.data(), value.data() + value.size(), delay);
        } else if (option == "--noise") {
            std::ignore = std::from_chars(value.data(), value.data() + value.size(), noise);
        }
    }

    const std::size_t                      chirpSamples = static_cast<std::size_t>(std::lround(kChirp.durationSeconds * static_cast<double>(kSampleRate)));
    gr::signal::SignalGeneratorCore<float> emitter;
    emitter.configure(gr::signal::SignalType::LinearChirp, 0.0, kSampleRate, 0.f, 1.f, 0.f, 0U);
    emitter.chirp = kChirp;
    std::vector<float> received(delay + chirpSamples + 4096UZ, 0.f);
    emitter.fill(std::span(received).subspan(delay, chirpSamples));
    std::mt19937                     generator(7U);
    std::normal_distribution<double> gaussian(0.0, noise);
    for (float& sample : received) {
        sample += static_cast<float>(gaussian(generator));
    }

    using Source = gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;
    using Sink   = gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;
    gr::testing::GraphFixture<> fixture;
    auto&                       source  = fixture.emplace<Source>({{"values", gr::Tensor<float>(gr::data_from, received)}, {"n_samples_max", static_cast<gr::Size_t>(received.size())}, {"sample_rate", kSampleRate}});
    auto&                       matched = fixture.emplace<gr::filter::Correlator<float>>({{"lags", gr::Size_t(1024)}, {"f_start", kChirp.startHz}, {"f_stop", kChirp.stopHz}, {"duration", kChirp.durationSeconds}, {"taper", kChirp.tukeyFraction}, {"coefficient", true}});
    auto&                       sink    = fixture.emplace<Sink>({{"log_samples", true}});
    if (!fixture.connect<"out", "in">(source, matched).has_value() || !fixture.connect<"out", "in">(matched, sink).has_value() || !fixture.run().has_value()) {
        std::println(stderr, "the graph did not run");
        return 1;
    }
    const std::vector<float> coefficients(sink._samples.begin(), sink._samples.end());
    const auto               peak = std::ranges::max_element(coefficients, {}, [](float c) { return std::abs(c); });
    if (peak == coefficients.end()) {
        std::println(stderr, "no correlation came out");
        return 1;
    }
    const auto found = static_cast<std::size_t>(peak - coefficients.begin());
    std::println("chirp {}..{} Hz over {} ms at {} kHz, noise sigma {}: peak at lag {} ({:.3f} ms), coefficient {:.3f}; true delay {}", kChirp.startHz, kChirp.stopHz, 1e3 * kChirp.durationSeconds, kSampleRate / 1e3f, noise, found, 1e3 * static_cast<double>(found) / static_cast<double>(kSampleRate), *peak, delay);
    return found == delay ? 0 : 2;
}
