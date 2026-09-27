#include "TriggerTest.hpp"
#include <boost/ut.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/StreamOps.hpp>
#include <string>
#include <vector>

namespace qaMux {
using namespace gr::blocks::trigger;
using gr::trigger_test::EventScript;

namespace {
[[nodiscard]] const gr::property_map& kLow() {
    static const gr::property_map map{{"a", 1.f}, {"b", 2.f}, {"c", 3.f}, {"d", 4.f}};
    return map;
}

[[nodiscard]] const gr::property_map& kHigh() {
    static const gr::property_map map{{"a", 11.f}, {"b", 12.f}, {"c", 13.f}, {"d", 14.f}};
    return map;
}

[[nodiscard]] gr::property_map contextNamed(std::string context) { //
    return gr::property_map{{std::string(gr::tag::CONTEXT.key()), std::move(context)}, {std::string(gr::tag::TRIGGER_NAME.key()), std::string("switch")}};
}
} // namespace

const boost::ut::suite<"Mux"> _mux = [] {
    using namespace boost::ut;

    "the selected input reaches the output and the other does not"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     ramp    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b c d |")}, {"sample_values", kLow()}});
        auto&                     flatTop = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b c d |")}, {"sample_values", kHigh()}});
        auto&                     control = fixture.emplace<EventScript>();
        auto&                     mux     = fixture.emplace<Mux<float>>({{"contexts", std::vector<std::string>{"RAMP", "FLATTOP"}}});
        auto&                     sink    = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
        control._events.push_back(contextNamed("RAMP"));
        control._stopWhenDone = false; // the streams decide when the graph ends

        expect(fixture.graph.connect(control, gr::PortDefinition{std::string("evtOut")}, mux, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(ramp, gr::PortDefinition{std::string("out")}, mux, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(flatTop, gr::PortDefinition{std::string("out")}, mux, gr::PortDefinition{std::string("in#1")}).has_value());
        expect(fixture.graph.connect(mux, gr::PortDefinition{std::string("out")}, sink, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(!sink._collected.empty()) << "the selected input was forwarded";
        expect(std::ranges::all_of(sink._collected, [](float sample) { return sample < 10.f; })) << "every sample came from the low source, which is the one RAMP names";
        expect(eq(mux.n_switches.value, gr::Size_t{1}));
        expect(gt(mux.n_dropped.value, gr::Size_t{0})) << "the un-selected input was read and dropped, as Demux does with an unclaimed context";
    };

    "a context naming no input is counted and changes nothing"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     ramp    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b |")}, {"sample_values", kLow()}});
        auto&                     control = fixture.emplace<EventScript>();
        auto&                     mux     = fixture.emplace<Mux<float>>({{"contexts", std::vector<std::string>{"RAMP"}}});
        auto&                     sink    = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
        control._events.push_back(contextNamed("CALIBRATION"));
        control._stopWhenDone = false;

        expect(fixture.graph.connect(control, gr::PortDefinition{std::string("evtOut")}, mux, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(ramp, gr::PortDefinition{std::string("out")}, mux, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(mux, gr::PortDefinition{std::string("out")}, sink, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(eq(mux.n_unmatched.value, gr::Size_t{1}));
        expect(eq(mux.n_switches.value, gr::Size_t{0}));
        expect(sink._collected.empty()) << "nothing is selected, so nothing is forwarded";
    };

    "with no context yet nothing is forwarded"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     ramp = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b |")}, {"sample_values", kLow()}});
        auto&                     mux  = fixture.emplace<Mux<float>>({{"contexts", std::vector<std::string>{"RAMP"}}});
        auto&                     sink = fixture.emplace<gr::trigger_test::CollectingSink<float>>();

        expect(fixture.graph.connect(ramp, gr::PortDefinition{std::string("out")}, mux, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(mux, gr::PortDefinition{std::string("out")}, sink, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(sink._collected.empty()) << "a stream whose state is unknown is not a stream in the first state";
    };

    "the input list follows the context list"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     mux = fixture.emplace<Mux<float>>({{"contexts", std::vector<std::string>{"RAMP", "FLATTOP", "CALIBRATION"}}});
        expect(eq(mux.in.size(), 3UZ));
    };
};
} // namespace qaMux

namespace qaPairwise {
using namespace gr::blocks::trigger;

namespace {
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}, {"4", 4.f}};
    return map;
}

struct Pairs {
    std::vector<float> previous;
    std::vector<float> current;
    gr::Size_t         published = 0U;
};

[[nodiscard]] Pairs paired(std::string script, std::size_t chunk = 0UZ, std::size_t room = 0UZ) {
    gr::testing::GraphFixture fixture;
    auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", kValues()}});
    auto&                     block   = fixture.emplace<Pairwise<float>>();
    auto&                     earlier = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
    auto&                     later   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
    if (chunk > 0UZ) {
        block.in.max_samples = chunk;
    }
    if (room > 0UZ) {
        block.previous.max_samples = room;
        block.current.max_samples  = room;
    }
    boost::ut::expect(fixture.connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.connect<"previous", "in">(block, earlier).has_value());
    boost::ut::expect(fixture.connect<"current", "in">(block, later).has_value());
    boost::ut::expect(fixture.run().has_value());
    return Pairs{earlier._collected, later._collected, block.n_pairs.value};
}
} // namespace

const boost::ut::suite<"Pairwise"> _pairwise = [] {
    using namespace boost::ut;

    "each sample arrives beside the one before it"_test = [] {
        const Pairs found = paired("1 2 3 4 |");
        expect(found.previous == std::vector<float>{1.f, 2.f, 3.f});
        expect(found.current == std::vector<float>{2.f, 3.f, 4.f});
        expect(eq(found.published, gr::Size_t{3})) << "n samples make n-1 pairs";
    };

    "the two outputs stay aligned"_test = [] {
        const Pairs found = paired("1 2 3 4 |");
        expect(eq(found.previous.size(), found.current.size())) << "a downstream block reads them together, so a difference in length would be a defect";
    };

    "the first sample starts no pair"_test = [] {
        const Pairs found = paired("3 |");
        expect(found.previous.empty());
        expect(found.current.empty());
        expect(eq(found.published, gr::Size_t{0}));
    };

    "two samples make exactly one pair"_test = [] {
        const Pairs found = paired("2 4 |");
        expect(found.previous == std::vector<float>{2.f});
        expect(found.current == std::vector<float>{4.f});
    };

    "the pairs do not depend on where the stream was cut"_test = [] {
        for (const std::size_t chunk : {1UZ, 2UZ, 3UZ}) {
            const Pairs found = paired("1 2 3 4 |", chunk);
            expect(found.previous == std::vector<float>{1.f, 2.f, 3.f}) << std::format("at a work-call size of {}", chunk);
            expect(found.current == std::vector<float>{2.f, 3.f, 4.f}) << std::format("at a work-call size of {}", chunk);
        }
    };
};
} // namespace qaPairwise

namespace qaSampleFilter {
using namespace gr::blocks::trigger;
using gr::trigger_test::completed;

namespace {
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"a", 2.f}, {"b", 30.f}, {"c", 22.f}, {"d", 5.f}, {"e", 60.f}, {"f", 1.f}};
    return map;
}

[[nodiscard]] const gr::property_map& kTags() {
    static const gr::property_map map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("mark")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};
    return map;
}

[[nodiscard]] std::string filtered(std::string script, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     block  = fixture.emplace<SampleFilter<float>>(std::move(settings));
    auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
    boost::ut::expect(fixture.connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.run().has_value());
    return sink.script();
}
} // namespace

const boost::ut::suite<"SampleFilter"> _sampleFilter = [] {
    using namespace boost::ut;

    "a sample above the threshold passes and one below it does not"_test = [] { //
        expect(eq(filtered("a b c d e f |", {{"predicate", std::string("greater")}, {"threshold", 10.}}), completed("b c e")));
    };

    "every comparison the family spells is honoured"_test = [] {
        expect(eq(filtered("a b c d e f |", {{"predicate", std::string("less")}, {"threshold", 10.}}), completed("a d f"))) << "less keeps what greater dropped";
        expect(eq(filtered("a b c d e f |", {{"predicate", std::string("equal")}, {"threshold", 22.}}), completed("c")));
        expect(eq(filtered("a b c d e f |", {{"predicate", std::string("not_equal")}, {"threshold", 22.}}), completed("a b d e f")));
        expect(eq(filtered("a b c d e f |", {{"predicate", std::string("greater_equal")}, {"threshold", 22.}}), completed("b c e")));
        expect(eq(filtered("a b c d e f |", {{"predicate", std::string("less_equal")}, {"threshold", 2.}}), completed("a f")));
    };

    "an expression replaces the comparison where the test is not one"_test = [] { //
        expect(eq(filtered("a b c d e f |", {{"expression", std::string("x > 10 and x < 40")}}), completed("b c"))) << "a band, which no single comparison expresses";
    };

    "a tag on a passed sample travels with it, one on a dropped sample does not"_test = [] {
        expect(eq(filtered("a T:b c d |", {{"predicate", std::string("greater")}, {"threshold", 10.}}), completed("T:b c"))) << "the tag kept its sample";
        expect(eq(filtered("T:a b c d |", {{"predicate", std::string("greater")}, {"threshold", 10.}}), completed("b c"))) << "a suppressed sample takes its tags with it, as everywhere in this family";
    };

    "a stream where nothing passes still ends"_test = [] { //
        expect(eq(filtered("a d f |", {{"predicate", std::string("greater")}, {"threshold", 100.}}), std::string("|")));
    };

    "the counters add up to what arrived"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b c d e f |")}, {"sample_values", kValues()}});
        auto&                     block  = fixture.emplace<SampleFilter<float>>({{"predicate", std::string("greater")}, {"threshold", 10.}});
        auto&                     sink   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(block.n_passed.value, gr::Size_t{3}));
        expect(eq(block.n_dropped.value, gr::Size_t{3}));
        expect(eq(sink._collected.size(), 3UZ));
        expect(eq(sink._collected[0], 30.f));
    };

    "a single sample is enough"_test = [] { //
        expect(eq(filtered("e |", {{"predicate", std::string("greater")}, {"threshold", 10.}}), completed("e")));
    };
};
} // namespace qaSampleFilter

namespace qaScan {
using namespace gr::blocks::trigger;

namespace {
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}, {"4", 4.f}};
    return map;
}

[[nodiscard]] const gr::property_map& kTags() {
    static const gr::property_map map{{"R", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("restart")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}, //
        {"X", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("other")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};
    return map;
}

struct Run {
    std::vector<float> values;
    gr::Size_t         resets = 0U;
};

[[nodiscard]] Run scanned(std::string script, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     block  = fixture.emplace<Scan<float>>(std::move(settings));
    auto&                     sink   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
    boost::ut::expect(fixture.connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.run().has_value());
    return Run{sink._collected, block.n_resets.value};
}
} // namespace

const boost::ut::suite<"Scan"> _scan = [] {
    using namespace boost::ut;

    "a running sum publishes on every sample, not once per segment"_test = [] {
        const Run found = scanned("1 2 3 |", {{"operation", std::string("sum")}});
        expect(found.values == std::vector<float>{1.f, 3.f, 6.f}) << "the running value, which is what separates scan from reduce";
    };

    "every operation the family spells"_test = [] {
        expect(scanned("1 2 3 |", {{"operation", std::string("product")}}).values == std::vector<float>{1.f, 2.f, 6.f});
        expect(scanned("2 1 3 |", {{"operation", std::string("minimum")}}).values == std::vector<float>{2.f, 1.f, 1.f});
        expect(scanned("2 1 3 |", {{"operation", std::string("maximum")}}).values == std::vector<float>{2.f, 2.f, 3.f});
        expect(scanned("1 2 3 |", {{"operation", std::string("count")}}).values == std::vector<float>{1.f, 2.f, 3.f});
        expect(scanned("1 2 3 |", {{"operation", std::string("mean")}}).values == std::vector<float>{1.f, 1.5f, 2.f});
        expect(scanned("1 2 3 |", {{"operation", std::string("last")}}).values == std::vector<float>{1.f, 2.f, 3.f}) << "last is the identity, which is the honest answer for a running 'last'";
    };

    "a matching trigger restarts the accumulation at the sample it sits on"_test = [] {
        const Run found = scanned("1 2 3 R:1 2 |", {{"operation", std::string("sum")}, {"reset_filter", std::string("restart")}});
        expect(found.values == std::vector<float>{1.f, 3.f, 6.f, 1.f, 3.f}) << "the tagged sample starts the new total rather than ending the old one";
        expect(eq(found.resets, gr::Size_t{1}));
    };

    "a trigger the filter does not name changes nothing"_test = [] {
        const Run found = scanned("1 2 X:3 |", {{"operation", std::string("sum")}, {"reset_filter", std::string("restart")}});
        expect(found.values == std::vector<float>{1.f, 3.f, 6.f});
        expect(eq(found.resets, gr::Size_t{0}));
    };

    "with no filter named nothing restarts it"_test = [] {
        const Run found = scanned("1 2 R:3 |", {{"operation", std::string("sum")}});
        expect(found.values == std::vector<float>{1.f, 3.f, 6.f});
        expect(eq(found.resets, gr::Size_t{0}));
    };

    "a single sample is its own running value"_test = [] { //
        expect(scanned("3 |", {{"operation", std::string("sum")}}).values == std::vector<float>{3.f});
    };

    "the answer does not depend on where the stream was cut"_test = [] {
        for (const std::size_t chunk : {1UZ, 2UZ, 3UZ, 5UZ}) {
            gr::testing::GraphFixture fixture;
            auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("1 2 3 R:1 2 |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
            auto&                     block  = fixture.emplace<Scan<float>>({{"operation", std::string("sum")}, {"reset_filter", std::string("restart")}});
            auto&                     sink   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
            block.in.max_samples             = chunk;
            expect(fixture.connect<"out", "in">(source, block).has_value());
            expect(fixture.connect<"out", "in">(block, sink).has_value());
            expect(fixture.run().has_value());
            expect(sink._collected == std::vector<float>{1.f, 3.f, 6.f, 1.f, 3.f}) << std::format("at a work-call size of {}", chunk);
        }
    };
};
} // namespace qaScan

namespace qaTail {
using namespace gr::blocks::trigger;

namespace {
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}, {"4", 4.f}, {"5", 5.f}, {"6", 6.f}};
    return map;
}

[[nodiscard]] const gr::property_map& kTags() {
    static const gr::property_map map{{"E", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("end")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}, //
        {"X", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("other")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};
    return map;
}

template<typename TBlock>
[[nodiscard]] std::vector<float> tailed(std::string script, gr::property_map settings, std::size_t chunk = 0UZ, std::size_t room = 0UZ) {
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     block  = fixture.emplace<TBlock>(std::move(settings));
    auto&                     sink   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
    if (chunk > 0UZ) {
        block.in.max_samples = chunk;
    }
    if (room > 0UZ) {
        block.out.max_samples = room;
    }
    boost::ut::expect(fixture.connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.run().has_value());
    return sink._collected;
}
} // namespace

const boost::ut::suite<"Tail"> _tail = [] {
    using namespace boost::ut;

    "TakeLast keeps the tail of the stream, SkipLast keeps everything before it"_test = [] {
        expect(tailed<TakeLast<float>>("1 2 3 4 |", {{"n", 2U}}) == std::vector<float>{3.f, 4.f});
        expect(tailed<SkipLast<float>>("1 2 3 4 |", {{"n", 2U}}) == std::vector<float>{1.f, 2.f}) << "the two together are the whole stream, which is what makes them a pair";
    };

    "a matching trigger ends a segment, and each segment has its own tail"_test = [] {
        expect(tailed<TakeLast<float>>("1 2 3 E:4 5 6 |", {{"filter", std::string("end")}, {"n", 2U}}) == std::vector<float>{2.f, 3.f, 5.f, 6.f}) << "the tag ends the first segment before its own sample, and the stream ends the second";
        expect(tailed<SkipLast<float>>("1 2 3 E:4 5 6 |", {{"filter", std::string("end")}, {"n", 2U}}) == std::vector<float>{1.f, 4.f});
    };

    "a trigger the filter does not name does not end anything"_test = [] { //
        expect(tailed<TakeLast<float>>("1 2 X:3 4 |", {{"filter", std::string("end")}, {"n", 2U}}) == std::vector<float>{3.f, 4.f});
    };

    "a tail longer than the segment is the whole segment"_test = [] {
        expect(tailed<TakeLast<float>>("1 2 |", {{"n", 5U}}) == std::vector<float>{1.f, 2.f});
        expect(tailed<SkipLast<float>>("1 2 |", {{"n", 5U}}) == std::vector<float>{}) << "nothing is old enough to be released";
    };

    "a tail of one"_test = [] {
        expect(tailed<TakeLast<float>>("1 2 3 |", {{"n", 1U}}) == std::vector<float>{3.f});
        expect(tailed<SkipLast<float>>("1 2 3 |", {{"n", 1U}}) == std::vector<float>{1.f, 2.f});
    };

    "a single sample"_test = [] {
        expect(tailed<TakeLast<float>>("4 |", {{"n", 2U}}) == std::vector<float>{4.f});
        expect(tailed<SkipLast<float>>("4 |", {{"n", 2U}}) == std::vector<float>{});
    };

    "the counters account for every sample"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("1 2 3 4 |")}, {"sample_values", kValues()}});
        auto&                     block  = fixture.emplace<TakeLast<float>>({{"n", 2U}});
        auto&                     sink   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(block.n_passed.value, gr::Size_t{2}));
        expect(eq(block.n_dropped.value, gr::Size_t{2}));
        expect(eq(block.n_segments.value, gr::Size_t{1})) << "the end of the stream ends the last segment";
    };

    "the tail does not depend on where the stream was cut"_test = [] {
        for (const std::size_t chunk : {1UZ, 2UZ, 3UZ, 5UZ}) {
            expect(tailed<TakeLast<float>>("1 2 3 E:4 5 6 |", {{"filter", std::string("end")}, {"n", 2U}}, chunk) == std::vector<float>{2.f, 3.f, 5.f, 6.f}) << std::format("TakeLast at a work-call size of {}", chunk);
            expect(tailed<SkipLast<float>>("1 2 3 E:4 5 6 |", {{"filter", std::string("end")}, {"n", 2U}}, chunk) == std::vector<float>{1.f, 4.f}) << std::format("SkipLast at a work-call size of {}", chunk);
        }
    };
};
} // namespace qaTail

int main() { /* tests are statically executed */ }
