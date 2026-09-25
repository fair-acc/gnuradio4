#include <gnuradio-4.0/trigger/SampleTest.hpp>

#include <boost/ut.hpp>

#include <deque>
#include <utility>
#include <vector>

#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/DigitalLine.hpp>
#include <gnuradio-4.0/trigger/SamplePredicate.hpp>
#include <gnuradio-4.0/trigger/Segments.hpp>

using namespace gr::blocks::trigger;
namespace primitives = gr::blocks::trigger::detail;

namespace {
struct SegmentProbe : gr::Block<SegmentProbe, gr::NoTagPropagation> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    GR_MAKE_REFLECTABLE(SegmentProbe, in, out);

    std::vector<std::pair<std::size_t, std::size_t>> _runs;
    std::vector<std::size_t>                         _tagIndices;
    std::deque<float>                                _queue;

    gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        const std::size_t first = input.streamIndex;
        primitives::forEachSegment(input, input.size(), [&](std::size_t from, std::size_t to) { _runs.emplace_back(first + from, first + to); }, [&](const gr::property_map_view&, std::size_t at) { _tagIndices.push_back(first + at); });
        _queue.insert(_queue.end(), input.begin(), input.end());
        std::ignore = input.consume(input.size());
        output.publish(primitives::drainInto(_queue, output));
        return gr::work::Status::OK;
    }
};

const auto sampleTests = [] {
    using namespace boost::ut;

    "each comparison holds as its name says"_test = [] {
        static_assert(holds(Comparison::greater, 3, 2) && !holds(Comparison::greater, 2, 2));
        static_assert(holds(Comparison::greaterEqual, 2, 2) && !holds(Comparison::greaterEqual, 1, 2));
        static_assert(holds(Comparison::less, 1, 2) && !holds(Comparison::less, 2, 2));
        static_assert(holds(Comparison::lessEqual, 2, 2) && !holds(Comparison::lessEqual, 3, 2));
        static_assert(holds(Comparison::equal, 2, 2) && !holds(Comparison::equal, 1, 2));
        static_assert(holds(Comparison::notEqual, 1, 2) && !holds(Comparison::notEqual, 2, 2));
        expect(parseComparison("lessEqual") == Comparison::lessEqual);
        expect(parseComparison("sideways") == Comparison::greater) << "an unknown name falls back to greater";
    };

    "a predicate without an expression is the comparison against the threshold"_test = [] {
        primitives::SamplePredicate<float> predicate;
        expect(!predicate.configure("less", "", 0.5f).has_value());
        expect(predicate(0.2f));
        expect(!predicate(0.7f));
        expect(predicate.hostOnlyReason().empty()) << "a comparison runs anywhere";
    };

    "an expression decides on the host, the comparison in a kernel"_test = [] {
        primitives::SamplePredicate<float> predicate;
        expect(!predicate.configure("greater", "x * x > threshold", 0.25f).has_value()) << fatal;
        expect(predicate(-0.6f)) << "0.36 > 0.25, which the comparison 'greater' alone would refuse";
        expect(!predicate(0.4f)) << "0.16 <= 0.25";
        expect(!predicate.hostOnlyReason().empty());
        expect(primitives::passes<true>(predicate, Comparison::greater, -0.6f, 0.25f));
        expect(!primitives::passes<false>(predicate, Comparison::greater, -0.6f, 0.25f)) << "a kernel tests -0.6 > 0.25";
    };

    "an expression that does not compile is refused and the comparison kept"_test = [] {
        primitives::SamplePredicate<float> predicate;
        expect(predicate.configure("greater", "x >", 0.5f).has_value());
        expect(predicate(0.7f));
        expect(!predicate(0.3f));
        expect(predicate.hostOnlyReason().empty());
    };

    "an expression on integer samples is refused and the comparison kept"_test = [] {
        primitives::SamplePredicate<int> predicate;
        expect(predicate.configure("greater", "x > 3", 3).has_value());
        expect(predicate(4));
        expect(!predicate(3));
    };
};

const auto digitalLineTests = [] {
    using namespace boost::ut;

    "a line changes level only beyond the hysteresis band"_test = [] {
        DigitalLine<float> line;
        line.configure(0.1f, 0.5f);
        const std::vector<float>                 samples{0.f, 0.7f, 0.55f, 0.45f, 0.3f};
        const std::vector<std::pair<bool, bool>> expected{{false, false}, {true, true}, {true, false}, {true, false}, {false, true}};
        for (std::size_t i = 0UZ; i < samples.size(); ++i) {
            line.observe(samples[i]);
            expect(eq(line.high, expected[i].first)) << "level after sample" << i;
            expect(eq(line.changed, expected[i].second)) << "change at sample" << i;
        }
    };

    "the first sample sets the level without counting as a change"_test = [] {
        DigitalLine<float> line;
        line.configure(0.1f, 0.5f);
        line.observe(0.7f);
        expect(line.high);
        expect(!line.changed) << "there was no level before it to change from";
    };
};

const auto segmentTests = [] {
    using namespace boost::ut;

    "the runs between tags cover the stream, each tag starting one, and the queue drains in order"_test = [] {
        constexpr std::size_t          kSamples = 20UZ;
        const std::vector<std::size_t> tagged{3UZ, 7UZ, 12UZ};
        gr::testing::GraphFixture      fixture;
        auto&                          source = fixture.emplace<gr::testing::TagSource<float>>({{"n_samples_max", gr::Size_t(kSamples)}, {"mark_tag", false}});
        for (const std::size_t index : tagged) {
            source._tags.push_back({index, {{"at", static_cast<gr::Size_t>(index)}}});
        }
        auto& probe = fixture.emplace<SegmentProbe>();
        auto& sink  = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.connect<"out", "in">(source, probe).has_value());
        expect(fixture.connect<"out", "in">(probe, sink).has_value());
        expect(fixture.run().has_value());

        expect(probe._tagIndices == tagged);
        std::size_t covered = 0UZ;
        for (const auto& [from, to] : probe._runs) {
            expect(eq(from, covered)) << "runs are contiguous";
            covered = to;
        }
        expect(eq(covered, kSamples));
        for (const std::size_t index : tagged) {
            expect(std::ranges::any_of(probe._runs, [index](const auto& run) { return run.first == index; })) << "a run starts at tag" << index;
        }
        expect(eq(sink._samples.size(), kSamples)) << fatal;
        for (std::size_t i = 0UZ; i < kSamples; ++i) {
            expect(eq(sink._samples[i], static_cast<float>(i)));
        }
    };
};
} // namespace

int main() {
    using namespace boost::ut;
    "SampleTest and SamplePredicate"_test = sampleTests;
    "DigitalLine"_test                    = digitalLineTests;
    "Segments"_test                       = segmentTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
