#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <span>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/Correlator.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
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
        dut.settings().autoForwardParameters().insert("key"); // only `gr:`-prefixed keys cross a block boundary unaided
        auto& sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});

        expect(flow.connect<"out", "in">(source, dut).has_value());
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());

        expect(ge(sink._tags.size(), 1UZ)) << "the tag must reach the sink rather than being dropped by the block";
        expect(std::ranges::any_of(sink._tags, [](const auto& emitted) { return emitted.map.contains("key"); })) << "and carry its payload through unchanged, alongside the settings tags the source emits";

        const float expectedRate = kInputRate * static_cast<float>(dut.output_chunk_size) / static_cast<float>(dut.input_chunk_size);
        expect(std::abs(sink.sample_rate - expectedRate) < 1.f) << std::format("sink saw {} Hz, the declared ratio gives {} Hz", sink.sample_rate, expectedRate);
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

int main() { /* tests run from the suite */ }
