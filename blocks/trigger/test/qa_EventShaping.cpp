#include "Marble.hpp"
#include "TriggerTest.hpp"
#include <boost/ut.hpp>
#include <cstdint>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/EventShaping.hpp>
#include <numeric>
#include <string>
#include <vector>

namespace qa_debounce {
using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

namespace {
/** a marble script places a trigger tag at an exact sample, which is what makes a quiet period assertable */
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"a", 1.f}};
    return map;
}
[[nodiscard]] const gr::property_map& kTags() {
    static const gr::property_map map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("edge")}}}, //
        {"U", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("other")}}}};
    return map;
}

struct Outcome {
    std::size_t emitted    = 0UZ;
    gr::Size_t  items      = 0U;
    gr::Size_t  suppressed = 0U;
    std::size_t samples    = 0UZ;

    std::vector<std::string> names;
};

[[nodiscard]] Outcome debounce(std::string script, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     clean  = fixture.emplace<Debounce<float>>(std::move(settings));
    auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     tap    = fixture.emplace<EventTap>();

    boost::ut::expect(fixture.connect<"out", "in">(source, clean).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(clean, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(clean, tap).has_value());
    boost::ut::expect(fixture.run().has_value());

    return Outcome{.emitted = tap.size(), .items = clean.n_items.value, .suppressed = clean.n_suppressed.value, .samples = sink._samples.size(), .names = tap.names()};
}
} // namespace

const boost::ut::suite<"Debounce"> _debounce = [] {
    using namespace boost::ut;

    "a burst inside the quiet period leaves one trigger"_test = [] {
        const auto found = debounce("T:a T:a T:a a a a a |", {{"filter", std::string("edge")}, {"n_samples", 3U}});

        expect(eq(found.items, 3U));
        expect(eq(found.emitted, 1UZ)) << "one event for the burst";
        expect(found.names == std::vector<std::string>{"edge"}) << "and it is the trigger itself, held whole";
        expect(eq(found.suppressed, 2U)) << "and the two it replaced are counted";
    };

    "triggers further apart than the quiet period each get through"_test = [] {
        const auto found = debounce("T:a a a a T:a a a a |", {{"filter", std::string("edge")}, {"n_samples", 3U}});

        expect(eq(found.items, 2U));
        expect(eq(found.emitted, 2UZ));
        expect(eq(found.suppressed, 0U));
    };

    "the same burst, cut into work calls of one sample, gives the same answer"_test = [] {
        const std::string script = "T:a T:a a a a T:a a a a |";
        const auto        whole  = debounce(script, {{"filter", std::string("edge")}, {"n_samples", 3U}});

        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     clean  = fixture.emplace<Debounce<float>>({{"filter", std::string("edge")}, {"n_samples", 3U}});
        auto&                     tap    = fixture.emplace<EventTap>();
        clean.in.max_samples             = 1UZ;
        expect(fixture.connect<"out", "in">(source, clean).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(clean, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(whole.emitted, 2UZ));
        expect(eq(whole.suppressed, 1U));
        expect(eq(tap.size(), whole.emitted)) << "the quiet period is judged where each trigger sits, not once per call";
        expect(eq(clean.n_suppressed.value, whole.suppressed));
    };

    "the stream itself is untouched"_test = [] {
        const auto found = debounce("T:a T:a a a a a |", {{"filter", std::string("edge")}, {"n_samples", 2U}});

        expect(eq(found.samples, 6UZ)) << "what is debounced is the trigger, not the signal";
    };

    "with nothing listening on evtOut the stream still passes"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a T:a a a a T:a a a a |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     clean  = fixture.emplace<Debounce<float>>({{"filter", std::string("edge")}, {"n_samples", 3U}});
        auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
        expect(fixture.connect<"out", "in">(source, clean).has_value());
        expect(fixture.connect<"out", "in">(clean, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._samples.size(), 9UZ)) << "an unconnected event output is no back-pressure";
        expect(eq(clean.n_emitted.value, 2U)) << "and what would have been reported is still counted";
    };

    "a trigger the filter does not name is not an item"_test = [] {
        const auto found = debounce("U:a U:a a a a a |", {{"filter", std::string("edge")}, {"n_samples", 2U}});

        expect(eq(found.items, 0U));
        expect(eq(found.emitted, 0UZ));
    };

    "with no quiet period every trigger passes straight through"_test = [] {
        const auto found = debounce("T:a T:a T:a |", {{"filter", std::string("edge")}});

        expect(eq(found.emitted, 3UZ)) << "an unset period is not an infinite one";
        expect(eq(found.suppressed, 0U));
    };

    "seconds work where the rate is known"_test = [] {
        const auto found = debounce("T:a T:a a a a a a a a a |", {{"filter", std::string("edge")}, {"timeout", 0.003f}, {"sample_rate", 1000.f}});

        expect(eq(found.emitted, 1UZ)) << "3 ms at 1 kHz is the same as three samples";
        expect(eq(found.suppressed, 1U));
    };

    "the burst and what survived it, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        const std::string         script = "T:a T:a T:a a a a T:a a a a |";
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     clean  = fixture.emplace<Debounce<float>>({{"filter", std::string("edge")}, {"n_samples", 3U}});
        auto&                     tap    = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, clean).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(clean, tap).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"Debounce: a bouncing trigger, and the one that stuck"};
        diagram.unit = "sample";
        diagram.row("in").at(0U, "edge").at(1U, "edge").at(2U, "edge").at(6U, "edge").completes();
        diagram.condition(std::format("Debounce(filter = \"edge\", quiet = 3 samples), {} suppressed", clean.n_suppressed.value));
        auto& out = diagram.row("evtOut");
        for (std::size_t i = 0UZ; i < tap.size(); ++i) {
            out.at(i == 0UZ ? 2U : 6U, "edge");
        }
        out.completes();
        diagram.print();

        expect(eq(tap.size(), 2UZ));
    };
};
} // namespace qa_debounce

namespace qa_delay_when {
using namespace gr::blocks::trigger;
using gr::testing::ProcessFunction;
using gr::testing::TagSink;
using gr::testing::TagSource;
using gr::trigger_test::EventTap;

namespace {
struct Outcome {
    std::vector<float> samples;
    gr::Size_t         delayed  = 0U;
    gr::Size_t         overflow = 0U;
    std::size_t        messages = 0UZ;
};

[[nodiscard]] Outcome delay(std::vector<float> samples, std::vector<gr::Size_t> delays, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    const gr::Size_t          n        = static_cast<gr::Size_t>(samples.size());
    auto&                     source   = fixture.emplace<TagSource<float>>({{"n_samples_max", n}, {"values", std::move(samples)}, {"mark_tag", false}});
    auto&                     delaySrc = fixture.emplace<TagSource<gr::Size_t>>({{"n_samples_max", n}, {"values", std::move(delays)}, {"mark_tag", false}});
    auto&                     block    = fixture.emplace<DelayWhen<float>>(std::move(settings));
    auto&                     sink     = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();

    boost::ut::expect(fixture.graph.connect(source, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(delaySrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("delayIn")}).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(block, sink).has_value());
    gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
    boost::ut::expect(fixture.run().has_value());

    return Outcome{
        .samples  = std::vector<float>(sink._samples.begin(), sink._samples.end()),
        .delayed  = block.n_delayed.value,
        .overflow = block.n_overflow.value,
        .messages = sent.texts().size(),
    };
}
} // namespace

const boost::ut::suite<"DelayWhen"> _delayWhen = [] {
    using namespace boost::ut;

    "a sample with no delay goes straight through"_test = [] {
        const auto found = delay({1.f, 2.f, 3.f}, {0U, 0U, 0U}, {});
        expect(found.samples == std::vector<float>{1.f, 2.f, 3.f});
        expect(eq(found.delayed, 0U));
    };

    "a delayed sample comes out after the ones that overtake it"_test = [] {
        const auto found = delay({1.f, 2.f, 3.f, 4.f, 5.f}, {3U, 0U, 0U, 0U, 0U}, {});

        expect(found.samples == std::vector<float>{2.f, 3.f, 1.f, 4.f, 5.f}) << "1 is due after sample 3, which is where it appears";
        expect(eq(found.delayed, 1U));
    };

    "different delays come out in the order they fall due"_test = [] {
        const auto found = delay({1.f, 2.f, 3.f, 7.f, 8.f}, {4U, 2U, 0U, 0U, 0U}, {});
        expect(found.samples == std::vector<float>{3.f, 2.f, 7.f, 1.f, 8.f}) << "a per-sample delay is what reorders them, which is the point of the block";
    };

    "a delay reaching past the end of the stream never comes due"_test = [] {
        const auto found = delay({1.f, 2.f, 3.f}, {100U, 0U, 0U}, {});
        expect(found.samples == std::vector<float>{2.f, 3.f}) << "the first sample is still waiting, which is honest: its time never came";
    };

    "past the bound it stops claiming to reorder, and says so once"_test = [] {
        const auto found = delay({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {100U, 100U, 100U, 100U, 100U, 100U}, {{"max_pending", 2U}});

        expect(gt(found.overflow, 0U)) << "everything past the second is forwarded in arrival order";
        expect(eq(found.messages, 1UZ)) << "one coalesced error message, not one per sample";
    };

    "what waited and what did not, drawn"_test = [] {
        const auto found = delay({1.f, 2.f, 3.f, 4.f}, {2U, 0U, 0U, 0U}, {});

        gr::testing::MarbleDiagram diagram{"DelayWhen: the first sample is told to wait two samples"};
        diagram.unit = "sample";
        diagram.row("in").at(0U, "1").at(1U, "2").at(2U, "3").at(3U, "4").completes();
        diagram.condition("DelayWhen(delayIn = 2 0 0 0)");
        auto& out = diagram.row("out");
        for (std::size_t i = 0UZ; i < found.samples.size(); ++i) {
            out.at(i, std::format("{:.0f}", found.samples[i]));
        }
        out.completes();
        diagram.print();

        expect(found.samples == std::vector<float>{2.f, 1.f, 3.f, 4.f});
    };
};
} // namespace qa_delay_when

namespace qa_distinct {
using namespace gr::blocks::trigger;
using gr::testing::ProcessFunction;
using gr::testing::TagSink;
using gr::testing::TagSource;
using gr::trigger_test::EventTap;

namespace {
template<typename TBlock>
[[nodiscard]] std::vector<std::int32_t> through(std::vector<std::int32_t> samples, gr::property_map settings, gr::trigger_test::Kept<TBlock>* kept = nullptr) {
    auto             owner   = std::make_unique<gr::testing::GraphFixture<>>();
    auto&            fixture = *owner;
    const gr::Size_t n       = static_cast<gr::Size_t>(samples.size());
    auto&            source  = fixture.template emplace<TagSource<std::int32_t>>({{"n_samples_max", n}, {"values", std::move(samples)}, {"mark_tag", false}});
    auto&            block   = fixture.template emplace<TBlock>(std::move(settings));
    auto&            sink    = fixture.template emplace<TagSink<std::int32_t, ProcessFunction::USE_PROCESS_ONE>>();

    boost::ut::expect(fixture.template connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.template connect<"out", "in">(block, sink).has_value());
    gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
    boost::ut::expect(fixture.run().has_value());

    if (kept != nullptr) {
        kept->block   = std::addressof(block);
        kept->fixture = std::move(owner);
    }
    return std::vector<std::int32_t>(sink._samples.begin(), sink._samples.end());
}
} // namespace

const boost::ut::suite<"Distinct"> _distinct = [] {
    using namespace boost::ut;

    "a value passes the first time and never again"_test = [] {
        const auto found = through<Distinct<std::int32_t>>({1, 2, 1, 3, 2, 3, 4}, {{"max_values", 16U}});
        expect(found == std::vector<std::int32_t>{1, 2, 3, 4});
    };

    "the repeats are counted, not merely gone"_test = [] {
        gr::trigger_test::Kept<Distinct<std::int32_t>> block;
        std::ignore = through<Distinct<std::int32_t>>({5, 5, 5, 5}, {{"max_values", 16U}}, &block);
        expect(block != nullptr && block->n_passed.value == 1U);
        expect(block != nullptr && block->n_suppressed.value == 3U);
    };

    "once the memory is full the block stops claiming to know"_test = [] {
        gr::trigger_test::Kept<Distinct<std::int32_t>> block;
        const auto                                     found = through<Distinct<std::int32_t>>({1, 2, 3, 1, 2, 3}, {{"max_values", 2U}}, &block);

        expect(found == std::vector<std::int32_t>{1, 2, 3, 1, 2, 3}) << "everything from the third value on is forwarded, repeats included";
        expect(block != nullptr && block->n_overflow.value > 0U) << "and the overflow is counted rather than hidden";
    };

    "a full output stops the call rather than losing samples"_test = [] {
        std::vector<std::int32_t> ramp(100'000UZ);
        std::iota(ramp.begin(), ramp.end(), 0);
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TagSource<std::int32_t>>({{"n_samples_max", static_cast<gr::Size_t>(ramp.size())}, {"values", ramp}, {"mark_tag", false}});
        auto&                     block  = fixture.emplace<Distinct<std::int32_t>>({{"max_values", 16U}});
        auto&                     sink   = fixture.emplace<TagSink<std::int32_t, ProcessFunction::USE_PROCESS_BULK>>();
        sink.in.max_samples              = 7UZ;
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());
        expect(std::ranges::equal(sink._samples, ramp)) << "all distinct: sixteen remembered, the rest forwarded as overflow, none dropped";
    };

    "the overflow says so once, as an error message"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TagSource<std::int32_t>>({{"n_samples_max", 8U}, {"values", std::vector<std::int32_t>{1, 2, 3, 4}}, {"mark_tag", false}});
        auto&                     block  = fixture.emplace<Distinct<std::int32_t>>({{"max_values", 2U}});
        auto&                     tap    = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(block, tap).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());

        expect(eq(sent.countContaining("value memory is full"), 1UZ)) << "one coalesced notice, however many samples follow it";
        expect(eq(tap.size(), 0UZ)) << "an error is no event";
    };

    "a segment trigger clears the memory"_test = [] {
        gr::testing::GraphFixture fixture;
        const gr::property_map    values{{"1", std::int32_t{1}}, {"2", std::int32_t{2}}};
        const gr::property_map    tags{{"S", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}}}};
        auto&                     source = fixture.emplace<MarbleSource<std::int32_t>>({{"script", std::string("1 2 1 S:1 2 1 |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     block  = fixture.emplace<Distinct<std::int32_t>>({{"max_values", 16U}, {"segment_filter", std::string("start")}});
        auto&                     sink   = fixture.emplace<TagSink<std::int32_t, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<std::int32_t> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<std::int32_t>{1, 2, 1, 2}) << "distinct within each cycle, not since the graph started";
    };
};

const boost::ut::suite<"DistinctUntilChanged"> _untilChanged = [] {
    using namespace boost::ut;

    "only a sample that differs from the one before it passes"_test = [] {
        const auto found = through<DistinctUntilChanged<std::int32_t>>({1, 1, 2, 2, 2, 1, 1, 3}, {});
        expect(found == std::vector<std::int32_t>{1, 2, 1, 3}) << "the 1 returning is new again: this asks 'is it new?', not 'have I seen it?'";
    };

    "a constant stream passes once"_test = [] {
        gr::trigger_test::Kept<DistinctUntilChanged<std::int32_t>> block;
        const auto                                                 found = through<DistinctUntilChanged<std::int32_t>>({7, 7, 7, 7, 7}, {}, &block);
        expect(found == std::vector<std::int32_t>{7});
        expect(block != nullptr && block->n_suppressed.value == 4U);
    };

    "what each of the two let through, drawn"_test = [] {
        gr::testing::MarbleDiagram diagram{"Distinct vs DistinctUntilChanged over 1 1 2 2 1"};
        diagram.unit = "sample";
        diagram.row("in").at(0U, "1").at(1U, "1").at(2U, "2").at(3U, "2").at(4U, "1").completes();
        diagram.condition("Distinct: a value once, ever");
        diagram.row("distinct").at(0U, "1").at(2U, "2").completes();
        diagram.condition("DistinctUntilChanged: a value once, until it changes");
        diagram.row("changed").at(0U, "1").at(2U, "2").at(4U, "1").completes();
        diagram.print();

        expect(through<Distinct<std::int32_t>>({1, 1, 2, 2, 1}, {{"max_values", 8U}}) == std::vector<std::int32_t>{1, 2});
        expect(through<DistinctUntilChanged<std::int32_t>>({1, 1, 2, 2, 1}, {}) == std::vector<std::int32_t>{1, 2, 1});
    };
};
} // namespace qa_distinct

namespace qa_repeat {
using namespace gr::blocks::trigger;
using gr::testing::ProcessFunction;
using gr::testing::TagSink;
using gr::testing::TagSource;
using gr::trigger_test::EventTap;

namespace {
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}};
    return map;
}

[[nodiscard]] const gr::property_map& kTags() {
    static const gr::property_map map{{"S", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}}}};
    return map;
}

[[nodiscard]] std::vector<float> replay(std::string script, gr::property_map settings, gr::trigger_test::Kept<Repeat<float>>* kept = nullptr) {
    auto  owner   = std::make_unique<gr::testing::GraphFixture<>>();
    auto& fixture = *owner;
    auto& source  = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto& block   = fixture.emplace<Repeat<float>>(std::move(settings));
    auto& sink    = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();

    boost::ut::expect(fixture.connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(block, sink).has_value());
    gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
    boost::ut::expect(fixture.run().has_value());

    if (kept != nullptr) {
        kept->block   = std::addressof(block);
        kept->fixture = std::move(owner);
    }
    return std::vector<float>(sink._samples.begin(), sink._samples.end());
}
} // namespace

const boost::ut::suite<"Repeat"> _repeat = [] {
    using namespace boost::ut;

    "a segment is emitted as many times as asked"_test = [] {
        const auto found = replay("S:1 2 S:3 |", {{"n_repeats", 3U}, {"segment_filter", std::string("start")}, {"capacity", 64U}});
        expect(found == std::vector<float>{1.f, 2.f, 1.f, 2.f, 1.f, 2.f}) << "three copies of the captured segment";
    };

    "once through is the identity"_test = [] {
        const auto found = replay("S:1 2 S:3 |", {{"n_repeats", 1U}, {"segment_filter", std::string("start")}, {"capacity", 64U}});
        expect(found == std::vector<float>{1.f, 2.f});
    };

    "with no trigger the store's capacity is the segment"_test = [] {
        const auto found = replay("1 2 3 |", {{"n_repeats", 2U}, {"capacity", 2U}});
        expect(found == std::vector<float>{1.f, 2.f, 1.f, 2.f}) << "two samples fill the store, and that is the segment";
    };

    "a segment that outgrows the store is passed through once, and says so"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("S:1 2 3 S:1 |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     block  = fixture.emplace<Repeat<float>>({{"n_repeats", 3U}, {"segment_filter", std::string("start")}, {"capacity", 2U}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());

        expect(gt(block.n_overflow.value, 0U)) << "the segment did not fit, so nothing is repeated";
        expect(eq(sent.countContaining("outgrew the replay store"), 1UZ)) << "one coalesced error message";
        expect(le(sink._samples.size(), 4UZ)) << "what came out is the segment once, not a truncated copy pretending to be data";
    };

    "the segment and its copies, drawn"_test = [] {
        gr::trigger_test::Kept<Repeat<float>> block;
        const auto                            found = replay("S:1 2 S:3 |", {{"n_repeats", 3U}, {"segment_filter", std::string("start")}, {"capacity", 64U}}, &block);

        gr::testing::MarbleDiagram diagram{"Repeat: a captured segment, played three times"};
        diagram.unit = "sample";
        diagram.row("in").at(0U, "start").at(2U, "start").completes();
        diagram.condition("Repeat(n_repeats = 3, segment_filter = \"start\")");
        auto& out = diagram.row("out");
        for (std::size_t i = 0UZ; i < found.size(); ++i) {
            out.at(i, std::format("{:.0f}", found[i]));
        }
        out.completes();
        diagram.print();

        expect(block != nullptr && block->n_segments.value == 1U);
    };
};
} // namespace qa_repeat

namespace qa_sequence_equal {
using namespace gr::blocks::trigger;
using gr::testing::ProcessFunction;
using gr::testing::TagSink;
using gr::testing::TagSource;

namespace {
struct Outcome {
    std::vector<std::uint8_t> verdicts;
    gr::Size_t                mismatches = 0U;
};

[[nodiscard]] Outcome compare(std::vector<float> measured, std::vector<float> reference, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    const gr::Size_t          n     = static_cast<gr::Size_t>(std::min(measured.size(), reference.size()));
    auto&                     left  = fixture.emplace<TagSource<float>>({{"n_samples_max", n}, {"values", std::move(measured)}, {"mark_tag", false}});
    auto&                     right = fixture.emplace<TagSource<float>>({{"n_samples_max", n}, {"values", std::move(reference)}, {"mark_tag", false}});
    auto&                     block = fixture.emplace<SequenceEqual<float>>(std::move(settings));
    auto&                     sink  = fixture.emplace<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_ONE>>();

    boost::ut::expect(fixture.graph.connect(left, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(right, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("reference")}).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.run().has_value());

    return Outcome{.verdicts = std::vector<std::uint8_t>(sink._samples.begin(), sink._samples.end()), .mismatches = block.n_mismatches.value};
}
} // namespace

const boost::ut::suite<"SequenceEqual"> _sequenceEqual = [] {
    using namespace boost::ut;

    "two streams that carried the same samples are equal"_test = [] {
        const auto found = compare({1.f, 2.f, 3.f, 4.f}, {1.f, 2.f, 3.f, 4.f}, {{"n_samples", 4U}});
        expect(found.verdicts == std::vector<std::uint8_t>{1U});
        expect(eq(found.mismatches, 0U));
    };

    "one sample apart is not equal"_test = [] {
        const auto found = compare({1.f, 2.f, 9.f, 4.f}, {1.f, 2.f, 3.f, 4.f}, {{"n_samples", 4U}});
        expect(found.verdicts == std::vector<std::uint8_t>{0U});
        expect(eq(found.mismatches, 1U));
    };

    "a tolerance is what makes the comparison usable on a measured signal"_test = [] {
        const auto strict  = compare({1.f, 2.001f}, {1.f, 2.f}, {{"n_samples", 2U}});
        const auto lenient = compare({1.f, 2.001f}, {1.f, 2.f}, {{"n_samples", 2U}, {"tolerance", 0.01f}});
        expect(strict.verdicts == std::vector<std::uint8_t>{0U});
        expect(lenient.verdicts == std::vector<std::uint8_t>{1U}) << "within the tolerance the two streams are the same";
    };

    "a verdict per segment, not one for the whole run"_test = [] {
        const auto found = compare({1.f, 2.f, 9.f, 4.f, 5.f, 6.f}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {{"n_samples", 3U}});
        expect(found.verdicts == std::vector<std::uint8_t>{0U, 1U}) << "the first three differ, the last three do not";
    };

    "the verdicts a run produced, drawn"_test = [] {
        const auto found = compare({1.f, 2.f, 9.f, 4.f, 5.f, 6.f}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {{"n_samples", 3U}});

        gr::testing::MarbleDiagram diagram{"SequenceEqual: a verdict per segment of three samples"};
        diagram.unit = "sample";
        diagram.row("in").at(2U, "9 where 3 was expected").completes();
        diagram.condition("SequenceEqual(n_samples = 3)");
        auto& out = diagram.row("out");
        for (std::size_t i = 0UZ; i < found.verdicts.size(); ++i) {
            out.at(2U + 3U * i, found.verdicts[i] == 1U ? "equal" : "differs");
        }
        out.completes();
        diagram.print();

        expect(eq(found.verdicts.size(), 2UZ));
    };
};
} // namespace qa_sequence_equal

namespace qa_throttle {
using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

namespace {
/** a marble script places a trigger tag at an exact sample, which is what makes an inhibit assertable */
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"a", 1.f}};
    return map;
}
[[nodiscard]] const gr::property_map& kTags() {
    static const gr::property_map map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("edge")}}}, //
        {"U", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("other")}}}};
    return map;
}

struct Outcome {
    std::size_t emitted    = 0UZ;
    gr::Size_t  items      = 0U;
    gr::Size_t  suppressed = 0U;
    std::size_t samples    = 0UZ;
};

[[nodiscard]] Outcome throttle(std::string script, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     clean  = fixture.emplace<Throttle<float>>(std::move(settings));
    auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     tap    = fixture.emplace<EventTap>();

    boost::ut::expect(fixture.connect<"out", "in">(source, clean).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(clean, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(clean, tap).has_value());
    boost::ut::expect(fixture.run().has_value());

    return Outcome{.emitted = tap.size(), .items = clean.n_items.value, .suppressed = clean.n_suppressed.value, .samples = sink._samples.size()};
}
} // namespace

const boost::ut::suite<"Throttle"> _debounce = [] {
    using namespace boost::ut;

    "a burst inside the inhibit leaves its first trigger"_test = [] {
        const auto found = throttle("T:a T:a T:a a a a a |", {{"filter", std::string("edge")}, {"n_samples", 3U}});

        expect(eq(found.items, 3U));
        expect(eq(found.emitted, 1UZ)) << "one event for the burst, reported immediately";
        expect(eq(found.suppressed, 2U)) << "and the two inside the inhibit are counted";
    };

    "triggers further apart than the inhibit each get through"_test = [] {
        const auto found = throttle("T:a a a a T:a a a a |", {{"filter", std::string("edge")}, {"n_samples", 3U}});

        expect(eq(found.items, 2U));
        expect(eq(found.emitted, 2UZ));
        expect(eq(found.suppressed, 0U));
    };

    "the same burst, cut into work calls of one sample, gives the same answer"_test = [] {
        const std::string script = "T:a T:a a a a T:a a a a |";
        const auto        whole  = throttle(script, {{"filter", std::string("edge")}, {"n_samples", 3U}});

        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     clean  = fixture.emplace<Throttle<float>>({{"filter", std::string("edge")}, {"n_samples", 3U}});
        auto&                     tap    = fixture.emplace<EventTap>();
        clean.in.max_samples             = 1UZ;
        expect(fixture.connect<"out", "in">(source, clean).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(clean, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(whole.emitted, 2UZ));
        expect(eq(whole.suppressed, 1U));
        expect(eq(tap.size(), whole.emitted)) << "the inhibit is judged where each trigger sits, not once per call";
        expect(eq(clean.n_suppressed.value, whole.suppressed));
    };

    "the stream itself is untouched"_test = [] {
        const auto found = throttle("T:a T:a a a a a |", {{"filter", std::string("edge")}, {"n_samples", 2U}});

        expect(eq(found.samples, 6UZ)) << "what is limited is the trigger, not the signal";
    };

    "with nothing listening on evtOut the stream still passes"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a T:a a a a T:a a a a |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     clean  = fixture.emplace<Throttle<float>>({{"filter", std::string("edge")}, {"n_samples", 3U}});
        auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
        expect(fixture.connect<"out", "in">(source, clean).has_value());
        expect(fixture.connect<"out", "in">(clean, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._samples.size(), 9UZ)) << "an unconnected event output is no back-pressure";
        expect(eq(clean.n_emitted.value, 2U)) << "and what would have been reported is still counted";
    };

    "a trigger the filter does not name is not an item"_test = [] {
        const auto found = throttle("U:a U:a a a a a |", {{"filter", std::string("edge")}, {"n_samples", 2U}});

        expect(eq(found.items, 0U));
        expect(eq(found.emitted, 0UZ));
    };

    "with no inhibit every trigger passes straight through"_test = [] {
        const auto found = throttle("T:a T:a T:a |", {{"filter", std::string("edge")}});

        expect(eq(found.emitted, 3UZ)) << "an unset inhibit is not an infinite one";
        expect(eq(found.suppressed, 0U));
    };

    "seconds work where the rate is known"_test = [] {
        const auto found = throttle("T:a T:a a a a a a a a a |", {{"filter", std::string("edge")}, {"timeout", 0.003f}, {"sample_rate", 1000.f}});

        expect(eq(found.emitted, 1UZ)) << "3 ms at 1 kHz is the same as three samples";
        expect(eq(found.suppressed, 1U));
    };

    "the burst and what got through it, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        const std::string         script = "T:a T:a T:a a a a T:a a a a |";
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     clean  = fixture.emplace<Throttle<float>>({{"filter", std::string("edge")}, {"n_samples", 3U}});
        auto&                     tap    = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, clean).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(clean, tap).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"Throttle: a fast trigger, and the ones the inhibit let through"};
        diagram.unit = "sample";
        diagram.row("in").at(0U, "edge").at(1U, "edge").at(2U, "edge").at(6U, "edge").completes();
        diagram.condition(std::format("Throttle(filter = \"edge\", inhibit = 3 samples), {} suppressed", clean.n_suppressed.value));
        auto& out = diagram.row("evtOut");
        for (std::size_t i = 0UZ; i < tap.size(); ++i) {
            out.at(i == 0UZ ? 0U : 6U, "edge");
        }
        out.completes();
        diagram.print();

        expect(eq(tap.size(), 2UZ));
    };
};
} // namespace qa_throttle

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
