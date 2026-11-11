#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <functional>
#include <numbers>
#include <random>
#include <ranges>
#include <span>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/Correlator.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace gr::testing;

[[nodiscard]] std::vector<float> runCorrelator(std::span<const float> reference, gr::Size_t lags, gr::Size_t nSamples) {
    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::Correlator<float>>({{"lags", lags}, {"reference", std::vector<float>(reference.begin(), reference.end())}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());
    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());

    std::vector<float> out(sink._samples.size());
    for (std::size_t i = 0UZ; i < out.size(); ++i) {
        out[i] = sink._samples[i];
    }
    return out;
}
} // namespace

const boost::ut::suite<"Correlator"> _correlator = [] {
    using namespace boost::ut;

    "each lag is the correlation at that lag"_test = [] {
        std::vector<float> reference(16UZ);
        for (std::size_t k = 0UZ; k < reference.size(); ++k) {
            reference[k] = std::sin(0.4f * static_cast<float>(k));
        }
        const std::vector<float> got = runCorrelator(reference, 64U, 4096U);
        expect(gt(got.size(), 64UZ));

        bool matches = true; // the source ramps, so lag l correlates the reference against samples [l, l+K)
        for (std::size_t lag = 0UZ; lag < std::min<std::size_t>(64UZ, got.size()); ++lag) {
            double exact = 0.0;
            for (std::size_t k = 0UZ; k < reference.size(); ++k) {
                exact += static_cast<double>(reference[k]) * static_cast<double>(lag + k);
            }
            matches = matches && std::abs(static_cast<double>(got[lag]) - exact) / std::max(1.0, std::abs(exact)) < 1e-5;
        }
        expect(matches) << "the block must compute the correlation it is named for";
    };

    "a single unit reference passes the stream through"_test = [] {
        const std::vector<float> reference{1.f};
        const std::vector<float> got = runCorrelator(reference, 32U, 512U);
        expect(gt(got.size(), 32UZ));
        bool identity = true;
        for (std::size_t n = 0UZ; n < std::min<std::size_t>(64UZ, got.size()); ++n) {
            identity = identity && std::abs(got[n] - static_cast<float>(n)) < 1e-3f;
        }
        expect(identity) << "correlating against one unit sample is the identity";
    };
    // CLAUDE.md section 8 requires tag propagation coverage: a windowed block must carry a tag through and rescale
    // the sample rate by the ratio it declares, or downstream timing silently drifts
    "a tag survives the block and the sample rate follows the declared ratio"_test = [] {
        constexpr float kInputRate = 48000.f;

        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(4096)}, {"sample_rate", kInputRate}, {"mark_tag", false}});
        source._tags     = {{512UZ, {{"key", "mid_stream"}}}};
        auto& dut        = flow.emplaceBlock<gr::filter::Correlator<float>>({{"lags", gr::Size_t(8)}, {"reference", std::vector<float>{1.f, 0.f, 0.f, 0.f}}});
        auto& sink       = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});

        expect(flow.connect<"out", "in">(source, dut).has_value());
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());

        expect(ge(sink._tags.size(), 1UZ)) << "the tag must reach the sink rather than being dropped by the block";
        expect(std::ranges::any_of(sink._tags, [](const auto& emitted) { return emitted.map.contains("key"); })) << "and carry its payload through unchanged, alongside the settings tags the source emits";

        // a correlation advances by as many samples as it emits, so it does not resample. Deriving the expectation
        // from the chunk sizes instead would make this a tautology over the framework's own formula.
        expect(std::abs(sink.sample_rate - kInputRate) < 1.f) << std::format("sink saw {} Hz; a correlator must pass {} Hz through unchanged", sink.sample_rate, kInputRate);
    };
};

const boost::ut::suite<"Correlator complex"> _correlatorComplex = [] {
    using namespace boost::ut;
    using C = std::complex<float>;

    "a complex signal correlated against itself peaks real and positive"_test = [] {
        // without conjugating the reference the peak would carry the signal's own phase instead of its energy
        gr::filter::Correlator<C> block;
        block.reference.assign({C{1.f, 1.f}, C{0.f, 2.f}, C{-1.f, 0.5f}});
        block.lags = 1U;
        block.settingsChanged({}, {});

        std::vector<C>     inputStorage(block.reference.begin(), block.reference.end());
        std::vector<C>     outputStorage(1);
        std::span<const C> input{inputStorage};
        std::span<C>       output{outputStorage};
        std::ignore = block.processBulk(input, output);

        const float energy = 1.f + 1.f + 4.f + 1.f + 0.25f; // Σ|y[k]|^2
        expect(std::abs(outputStorage[0].real() - energy) < 1e-4f) << "the zero-lag peak is the reference's energy";
        expect(std::abs(outputStorage[0].imag()) < 1e-4f) << "and carries no phase of its own";
    };
};

namespace {
constexpr float       kChirpRate    = 48000.f;
constexpr double      kChirpStartHz = 1000.;
constexpr double      kChirpStopHz  = 8000.;
constexpr double      kChirpSeconds = 0.002;
constexpr std::size_t kChirpSamples = 96UZ; // 2 ms at 48 kHz

/** the analytic linear chirp, sin(2 pi (f0 t + (f1 - f0) t^2 / 2T)), independent of SignalGeneratorCore */
[[nodiscard]] double analyticChirp(std::size_t n) {
    const double t = static_cast<double>(n) / static_cast<double>(kChirpRate);
    return std::sin(2. * std::numbers::pi * (kChirpStartHz * t + 0.5 * (kChirpStopHz - kChirpStartHz) * t * t / kChirpSeconds));
}

[[nodiscard]] gr::property_map chirpSettings(bool coefficient) { return {{"lags", gr::Size_t(64)}, {"f_start", kChirpStartHz}, {"f_stop", kChirpStopHz}, {"duration", kChirpSeconds}, {"sample_rate", kChirpRate}, {"coefficient", coefficient}}; }

struct CorrelatorRun {
    std::vector<float> lags;
    std::vector<float> reference;
};

[[nodiscard]] CorrelatorRun correlate(const std::vector<float>& signal, gr::property_map settings) {
    gr::testing::GraphFixture<> fixture;
    auto&                       source = gr::testing::sourceOf(fixture, signal, kChirpRate);
    auto&                       dut    = fixture.emplace<gr::filter::Correlator<float>>(std::move(settings));
    auto&                       sink   = fixture.emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
    boost::ut::expect(fixture.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(dut, sink).has_value());
    boost::ut::expect(fixture.run().has_value());
    const std::pmr::vector<float>& used = dut.referenceInUse();
    boost::ut::expect(dut.reference.empty() || &used == &dut.reference) << "a user reference is the one in use";
    return {gr::testing::samplesOf(sink), std::vector<float>(used.begin(), used.end())};
}

/** the chirp delayed by `delay` samples, scaled by `gain`, added onto `signal` */
void addChirp(std::vector<float>& signal, std::size_t delay, float gain) {
    for (std::size_t n = 0UZ; n < kChirpSamples && delay + n < signal.size(); ++n) {
        signal[delay + n] += gain * static_cast<float>(analyticChirp(n));
    }
}
} // namespace

const boost::ut::suite<"Correlator chirp reference"> _chirpReference = [] {
    using namespace boost::ut;

    "an empty reference with a duration is the linear chirp of f_start, f_stop, duration at sample_rate"_test = [] {
        const CorrelatorRun run = correlate(std::vector<float>(512UZ, 0.f), chirpSettings(false));
        expect(fatal(eq(run.reference.size(), kChirpSamples))) << "2 ms at 48 kHz";
        for (const std::size_t n : {0UZ, 17UZ, 50UZ, 95UZ}) {
            expect(approx(static_cast<double>(run.reference[n]), analyticChirp(n), 1e-5)) << "sample" << n;
        }
    };

    "the chirp stays out of the reference setting, so a later duration or rate rebuilds it"_test = [] {
        gr::filter::Correlator<float> block;
        block.duration    = kChirpSeconds;
        block.sample_rate = kChirpRate;
        block.settingsChanged({}, {});
        expect(block.reference.empty()) << "a saved flow graph keeps no chirp as a literal";
        expect(eq(block._chirpReference.size(), kChirpSamples));
        block.duration = kChirpSeconds / 2.;
        block.settingsChanged({}, {});
        expect(eq(block._chirpReference.size(), kChirpSamples / 2UZ)) << "half the duration, half the samples";
        expect(eq(block.input_chunk_size.value, static_cast<gr::Size_t>(block.lags.value + kChirpSamples / 2UZ - 1UZ))) << "the window follows";
    };

    "a chirp too long to hold leaves no chirp reference, not the previous one"_test = [] {
        gr::filter::Correlator<float> block;
        block.duration    = kChirpSeconds;
        block.sample_rate = kChirpRate;
        block.settingsChanged({}, {});
        expect(fatal(eq(block._chirpReference.size(), kChirpSamples)));
        block.duration = 3600.;
        block.settingsChanged({}, {});
        expect(block._chirpReference.empty()) << "an hour at 48 kHz exceeds kMaxLength";
    };

    "a reference the user sets wins over the chirp settings"_test = [] {
        gr::property_map settings = chirpSettings(false);
        settings.insert_or_assign(std::string("reference"), gr::pmt::Value(std::vector<float>{1.f, 0.f, 0.f}));
        const CorrelatorRun run = correlate(std::vector<float>(512UZ, 0.f), std::move(settings));
        expect(run.reference == std::vector<float>{1.f, 0.f, 0.f});
    };

    "a chirp 150 samples into noise peaks at lag 150, as a coefficient of 1 without the noise"_test = [] {
        constexpr std::size_t kDelay = 150UZ;
        std::vector<float>    clean(512UZ, 0.f);
        addChirp(clean, kDelay, 0.25f);
        const CorrelatorRun cleanRun = correlate(clean, chirpSettings(true));
        expect(fatal(gt(cleanRun.lags.size(), kDelay)));
        expect(approx(cleanRun.lags[kDelay], 1.f, 1e-4f)) << "the window is the reference scaled by 0.25";
        expect(std::ranges::all_of(cleanRun.lags, [](float c) { return c >= -1.f - 1e-5f && c <= 1.f + 1e-5f; })) << "a coefficient stays within [-1, 1]";
        expect(eq(cleanRun.lags[0], 0.f)) << "a silent window gives 0, not NaN";

        std::mt19937       generator(42U);
        std::vector<float> noisy = clean;
        for (float& sample : noisy) {
            sample += 0.5f * (static_cast<float>(generator()) / static_cast<float>(std::mt19937::max()) - 0.5f);
        }
        const CorrelatorRun noisyRun = correlate(noisy, chirpSettings(false));
        const auto          peak     = std::ranges::max_element(noisyRun.lags, {}, [](float c) { return std::abs(c); });
        expect(eq(static_cast<std::size_t>(peak - noisyRun.lags.begin()), kDelay)) << "the matched filter finds the chirp the noise hides";
    };
};

const boost::ut::suite<"Correlator multipath"> _multipath = [] {
    using namespace boost::ut;

    "a direct chirp and its reflection 110 samples later give two peaks, each its gain times the chirp's energy"_test = [] {
        constexpr std::size_t kDirect = 150UZ;
        constexpr std::size_t kEcho   = 260UZ; // farther than the chirp is long, so the two do not overlap
        std::vector<float>    received(512UZ, 0.f);
        addChirp(received, kDirect, 0.25f);
        addChirp(received, kEcho, 0.125f);
        const CorrelatorRun run = correlate(received, chirpSettings(false));
        expect(fatal(gt(run.lags.size(), kEcho)));

        const double energy = std::ranges::fold_left(std::views::iota(0UZ, kChirpSamples) | std::views::transform([](std::size_t n) { return analyticChirp(n) * analyticChirp(n); }), 0., std::plus{});
        expect(approx(static_cast<double>(run.lags[kDirect]), 0.25 * energy, 1e-3 * energy)) << "direct path";
        expect(approx(static_cast<double>(run.lags[kEcho]), 0.125 * energy, 1e-3 * energy)) << "reflection";
        const auto strongest = std::ranges::max_element(run.lags, {}, [](float c) { return std::abs(c); });
        expect(eq(static_cast<std::size_t>(strongest - run.lags.begin()), kDirect)) << "the direct path is the strongest";
        const auto echoWindow = run.lags | std::views::drop(kDirect + kChirpSamples) | std::views::take(2UZ * (kEcho - kDirect - kChirpSamples) + kChirpSamples);
        const auto echoPeak   = std::ranges::max_element(echoWindow, {}, [](float c) { return std::abs(c); });
        expect(eq(static_cast<std::size_t>(std::ranges::distance(echoWindow.begin(), echoPeak)) + kDirect + kChirpSamples, kEcho)) << "the second peak is the reflection's";
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
