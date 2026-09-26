#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/Predicate.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::testing::ProcessFunction;
using gr::testing::TagSink;
using gr::testing::TagSource;

namespace {
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}, {"7", 7.f}, {"8", 8.f}};
    return map;
}

[[nodiscard]] const gr::property_map& kTags() {
    static const gr::property_map map{{"S", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}}}};
    return map;
}

template<typename TBlock>
[[nodiscard]] std::vector<float> through(std::vector<float> samples, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    const gr::Size_t          n      = static_cast<gr::Size_t>(samples.size());
    auto&                     source = fixture.template emplace<TagSource<float>>({{"n_samples_max", n}, {"values", std::move(samples)}, {"mark_tag", false}});
    auto&                     block  = fixture.template emplace<TBlock>(std::move(settings));
    auto&                     sink   = fixture.template emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();

    boost::ut::expect(fixture.template connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.template connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.run().has_value());

    return std::vector<float>(sink._samples.begin(), sink._samples.end());
}
} // namespace

const boost::ut::suite<"TakeWhile"> _takeWhile = [] {
    using namespace boost::ut;

    "samples pass until the first that fails, and nothing after it"_test = [] {
        const auto found = through<TakeWhile<float>>({1.f, 2.f, 3.f, 7.f, 1.f, 2.f}, {{"predicate", std::string("less")}, {"threshold", 5.f}});

        expect(found == std::vector<float>{1.f, 2.f, 3.f}) << "the 1 and 2 after the 7 are gone too: that is the difference from a filter";
    };

    "a stream that never fails passes whole"_test = [] {
        const auto found = through<TakeWhile<float>>({1.f, 2.f, 3.f}, {{"predicate", std::string("less")}, {"threshold", 5.f}});
        expect(found == std::vector<float>{1.f, 2.f, 3.f});
    };

    "a failure on the first sample emits nothing"_test = [] {
        const auto found = through<TakeWhile<float>>({7.f, 1.f, 2.f}, {{"predicate", std::string("less")}, {"threshold", 5.f}});
        expect(found.empty());
    };

    "an expression overrides the comparison"_test = [] {
        const auto found = through<TakeWhile<float>>({1.f, 2.f, 3.f, 8.f, 6.f, 2.f}, {{"expression", std::string("x < 5 or x > 7")}});
        expect(found == std::vector<float>{1.f, 2.f, 3.f, 8.f}) << "8 passes where 'less than 5' alone would not; 6 fails both halves and ends it";
    };

    "a segment trigger re-arms it, so it is useful on a stream that never ends"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("1 2 7 1 S:1 2 7 1 |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     block  = fixture.emplace<TakeWhile<float>>({{"predicate", std::string("less")}, {"threshold", 5.f}, {"segment_filter", std::string("start")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{1.f, 2.f, 1.f, 2.f}) << "each segment stops at its own 7";
        expect(eq(block.n_segments_ended.value, 2U));
    };
};

const boost::ut::suite<"SkipWhile"> _skipWhile = [] {
    using namespace boost::ut;

    "samples are suppressed until the first that fails, then everything passes"_test = [] {
        const auto found = through<SkipWhile<float>>({7.f, 8.f, 1.f, 7.f, 2.f}, {{"predicate", std::string("greater")}, {"threshold", 5.f}});

        expect(found == std::vector<float>{1.f, 7.f, 2.f}) << "the 7 after the 1 passes: skipping does not resume, which is Rx's asymmetry";
    };

    "a stream that always holds is suppressed entirely"_test = [] {
        const auto found = through<SkipWhile<float>>({7.f, 8.f, 7.f}, {{"predicate", std::string("greater")}, {"threshold", 5.f}});
        expect(found.empty());
    };

    "a segment trigger starts the skipping again"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("7 1 2 S:7 1 2 |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     block  = fixture.emplace<SkipWhile<float>>({{"predicate", std::string("greater")}, {"threshold", 5.f}, {"segment_filter", std::string("start")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{1.f, 2.f, 1.f, 2.f}) << "the 7 opening each segment is skipped again";
    };
};

const boost::ut::suite<"ElementAt"> _elementAt = [] {
    using namespace boost::ut;

    "the nth sample of the stream, counting from zero"_test = [] {
        expect(through<ElementAt<float>>({1.f, 2.f, 3.f, 7.f}, {{"n", 2U}}) == std::vector<float>{3.f});
        expect(through<ElementAt<float>>({1.f, 2.f, 3.f}, {{"n", 0U}}) == std::vector<float>{1.f}) << "zero is the first sample, not a disabled setting";
    };

    "a stream shorter than n emits nothing"_test = [] { expect(through<ElementAt<float>>({1.f, 2.f}, {{"n", 9U}}).empty()); };

    "with a segment filter it emits the nth of every segment"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("S:1 2 3 S:7 8 1 |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     block  = fixture.emplace<ElementAt<float>>({{"n", 1U}, {"segment_filter", std::string("start")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{2.f, 8.f}) << "the second sample of each segment";
        expect(eq(block.n_emitted.value, 2U));
    };

    "what each of the three did, drawn"_test = [] {
        const std::string script = "1 2 3 7 8 1 2 |";
        for (const auto& [title, settings] : std::vector<std::pair<std::string, gr::property_map>>{
                 {"TakeWhile(less than 5)", {{"predicate", std::string("less")}, {"threshold", 5.f}}},
             }) {
            gr::testing::GraphFixture fixture;
            auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
            auto&                     block  = fixture.emplace<TakeWhile<float>>(settings);
            auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
            expect(fixture.connect<"out", "in">(source, block).has_value());
            expect(fixture.connect<"out", "in">(block, sink).has_value());
            expect(fixture.run().has_value());
            gr::trigger_test::drawSubset(std::format("{}: everything after the first failure is gone", title), title, script, sink.script());
        }

        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     block  = fixture.emplace<SkipWhile<float>>({{"predicate", std::string("less")}, {"threshold", 5.f}});
        auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        gr::trigger_test::drawSubset("SkipWhile(less than 5): nothing resumes the skipping", "SkipWhile(less than 5)", script, sink.script());

        expect(ge(block.n_skipped.value, 3U));
    };
};

int main() { /* tests are statically executed */ }
