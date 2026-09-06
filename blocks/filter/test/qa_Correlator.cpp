#include <boost/ut.hpp>

#include <cmath>
#include <format>
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
};

int main() { /* tests run from the suite */ }
