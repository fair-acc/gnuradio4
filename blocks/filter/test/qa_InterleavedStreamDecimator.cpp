#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <format>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/InterleavedStreamDecimator.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

const boost::ut::suite<"InterleavedStreamDecimator"> decimatorTests = [] {
    using namespace boost::ut;
    using namespace std::string_literals;
    using namespace gr::testing;
    using gr::test::eq;

    "the rate contract counts whole groups"_test = [] {
        gr::filter::InterleavedStreamDecimator<float> block({{"decimation", gr::Size_t(4)}, {"interleave", gr::Size_t(2)}, {"groups_per_frame", gr::Size_t(8)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        expect(eq(static_cast<std::size_t>(block.input_chunk_size), 64UZ)) << "8 groups x 4 decimation x 2 wide";
        expect(eq(static_cast<std::size_t>(block.output_chunk_size), 16UZ)) << "8 groups x 2 wide";
    };

    "an I/Q pair is never split"_test = [] {
        constexpr gr::Size_t kDecimation = 3U;
        constexpr gr::Size_t kInterleave = 2U;

        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        // a ramp: every even sample is its group index, every odd one that plus 1000, so a split pair is visible
        std::vector<float> values;
        for (float g = 0.f; g < 96.f; ++g) {
            values.push_back(g);
            values.push_back(g + 1000.f);
        }
        auto& source = flow.emplaceBlock<TagSource<float>>({{"n_samples_max", gr::Size_t(4096)}, {"values", values}, {"mark_tag", false}});
        auto& dut    = flow.emplaceBlock<gr::filter::InterleavedStreamDecimator<float>>({{"decimation", kDecimation}, {"interleave", kInterleave}, {"groups_per_frame", gr::Size_t(8)}});
        auto& sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

        expect(flow.connect(source, "out"s, dut, "in"s).has_value());
        expect(flow.connect(dut, "out"s, sink, "in"s).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());

        const auto& samples = sink._samples;
        expect(gt(samples.size(), 8UZ)) << "nothing came through";
        expect(eq(samples.size() % 2UZ, 0UZ)) << "an interleaved stream must end on a whole group";
        for (std::size_t g = 0UZ; g + 1UZ < samples.size(); g += 2UZ) {
            expect(approx(samples[g + 1UZ] - samples[g], 1000.f, 1e-3f)) << std::format("group {} came out as ({}, {}): the pair was split", g / 2UZ, samples[g], samples[g + 1UZ]);
        }
    };

    "decimation of one passes every group"_test = [] {
        gr::filter::InterleavedStreamDecimator<float> block({{"decimation", gr::Size_t(1)}, {"interleave", gr::Size_t(4)}, {"groups_per_frame", gr::Size_t(4)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(static_cast<std::size_t>(block.input_chunk_size), static_cast<std::size_t>(block.output_chunk_size))) << "1:1 when nothing is dropped";
    };
    "a complex stream decimates by whole groups too"_test = [] {
        // registered for std::complex<float>, so the type is covered rather than assumed
        gr::filter::InterleavedStreamDecimator<std::complex<float>> block({{"decimation", gr::Size_t(2)}, {"interleave", gr::Size_t(3)}, {"groups_per_frame", gr::Size_t(4)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        expect(eq(static_cast<std::size_t>(block.input_chunk_size), 24UZ)) << "4 groups x 2 decimation x 3 wide";
        expect(eq(static_cast<std::size_t>(block.output_chunk_size), 12UZ)) << "4 groups x 3 wide";
    };
};

int main() { /* not needed for UT */ }
