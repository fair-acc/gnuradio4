#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/DigitalLine.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/SamplePredicate.hpp>
#include <gnuradio-4.0/trigger/SampleTest.hpp>
#include <gnuradio-4.0/trigger/Segments.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <gnuradio-4.0/trigger/WindowCollector.hpp>

namespace qa_primitives {
using namespace gr::blocks::trigger;
namespace primitives = gr::blocks::trigger::detail;

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

const boost::ut::suite<"SampleTest and SamplePredicate"> _sampleTests = [] {
    using namespace boost::ut;

    "each comparison holds as its name says"_test = [] {
        static_assert(holds(Comparison::greater, 3, 2) && !holds(Comparison::greater, 2, 2));
        static_assert(holds(Comparison::greaterEqual, 2, 2) && !holds(Comparison::greaterEqual, 1, 2));
        static_assert(holds(Comparison::less, 1, 2) && !holds(Comparison::less, 2, 2));
        static_assert(holds(Comparison::lessEqual, 2, 2) && !holds(Comparison::lessEqual, 3, 2));
        static_assert(holds(Comparison::equal, 2, 2) && !holds(Comparison::equal, 1, 2));
        static_assert(holds(Comparison::notEqual, 1, 2) && !holds(Comparison::notEqual, 2, 2));
        expect(parseComparison("lessEqual") == Comparison::lessEqual);
        expect(!holds(Comparison::notEqual, std::numeric_limits<float>::quiet_NaN(), 1.f)) << "a NaN holds nothing, so invalid data never passes";
        expect(holds(Comparison::inside, 2.f, 2.f, 3.f) && holds(Comparison::inside, 3.f, 2.f, 3.f) && !holds(Comparison::outside, 3.f, 2.f, 3.f)) << "closed at both ends";
        expect(!parseComparison("sideways").has_value()) << "an unknown name is refused, never read as greater";
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

const boost::ut::suite<"DigitalLine"> _digitalLine = [] {
    using namespace boost::ut;

    "a line changes level only beyond the hysteresis band"_test = [] {
        DigitalLine<float> line;
        line.configure({.level = 0.5f, .hysteresis = 0.25f});
        const std::vector<float>                 samples{0.f, 0.75f, 0.55f, 0.45f, 0.25f};
        const std::vector<std::pair<bool, bool>> expected{{false, false}, {true, true}, {true, false}, {true, false}, {false, true}};
        for (std::size_t i = 0UZ; i < samples.size(); ++i) {
            line.observe(samples[i]);
            expect(eq(line.high(), expected[i].first)) << "level after sample" << i;
            expect(eq(line.changed, expected[i].second)) << "change at sample" << i;
        }
    };

    "the first sample sets the level without counting as a change"_test = [] {
        DigitalLine<float> line;
        line.configure({.level = 0.5f, .hysteresis = 0.25f});
        line.observe(0.75f);
        expect(line.high());
        expect(!line.changed) << "there was no level before it to change from";
    };

    "a line is unknown, not low, until a sample leaves the band"_test = [] {
        DigitalLine<float> line;
        line.configure({.level = 0.5f, .hysteresis = 0.25f});
        line.observe(0.45f);
        expect(line.state() == gr::trigger::LineState::unknown) << "inside the band after the start";
        line.observe(0.25f);
        expect(line.state() == gr::trigger::LineState::low);
        expect(!line.changed) << "leaving the unknown state is no edge";
        line.observe(std::numeric_limits<float>::quiet_NaN());
        expect(line.state() == gr::trigger::LineState::unknown) << "a non-finite sample says nothing about the line";
    };
};

const boost::ut::suite<"Segments"> _segments = [] {
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
} // namespace qa_primitives

namespace qa_events {
namespace primitives = gr::blocks::trigger::detail;

/** the part of a block `reportError` uses: a name and a message output */
struct Refusals {
    struct {
        std::string value = "test";
    } name;
    std::vector<std::string> reasons;

    void emitMessage(std::string_view /*endpoint*/, const gr::property_map& message) { reasons.emplace_back(gr::property_map_view{message}.value_or<std::string_view>(primitives::kErrorKey, std::string_view{})); }
};

struct TaggedAt {
    std::size_t      index;
    gr::property_map map;
};

const boost::ut::suite<"Events"> _events = [] {
    using namespace boost::ut;

    "an inline text keeps what fits and forgets what it held before"_test = [] {
        primitives::InlineText text;
        text.assign("a long name");
        text.assign("short");
        expect(eq(text.view(), std::string_view{"short"}));
        text.assign(std::string(1000UZ, 'x'));
        expect(eq(text.view().size(), text.chars.size() - 1UZ)) << "truncated, still terminated";
    };

    "a held event is a copy of its image, and one beyond a device slot is refused there"_test = [] {
        const gr::property_map small = primitives::makeEvent("edge");
        primitives::HeldEvent  held;
        expect(held.empty());
        expect(held.assign<false>(gr::property_map_view{small}));
        expect(!held.empty());
        expect(eq(held.view().get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()).value_or(std::string_view{}), std::string_view{"edge"}));
        held.clear();
        expect(held.empty());

        gr::property_map large = primitives::makeEvent("edge");
        large["payload"]       = std::pmr::string(2UZ * gr::kDeviceEventSlotBytes, 'x');
        expect(!held.assign<false>(gr::property_map_view{large})) << "a device cannot grow";
        expect(held.empty());
        expect(!held.assign<true>(gr::property_map_view{large})) << "compacting does not make a payload fit";
    };

    "the tags on one sample merge with the latest winning, short keys read as long ones"_test = [] {
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, 1024UZ> scratch{};
        const std::vector<TaggedAt>                                    none;
        expect(primitives::mergedAt(none.begin(), none.end(), scratch).empty());

        const std::vector<TaggedAt> one{{3UZ, {{std::string(gr::tag::TRIGGER_NAME.key()), "only"}}}};
        expect(eq(primitives::mergedAt(one.begin(), one.end(), scratch).get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()).value_or(std::string_view{}), std::string_view{"only"})) << "a single tag is viewed as it is";

        const std::vector<TaggedAt> several{{3UZ, {{std::string(gr::tag::TRIGGER_NAME.key()), "early"}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{7U}}}}, {3UZ, {{std::string(gr::tag::TRIGGER_NAME.shortKey()), "late"}}}};
        const auto                  merged = primitives::mergedAt(several.begin(), several.end(), scratch);
        expect(eq(merged.get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()).value_or(std::string_view{}), std::string_view{"late"})) << "the later tag, under its short key";
        const auto* time = merged.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        expect(time != nullptr && *time == 7U) << "what the later tag lacks stays";
    };

    "a trigger is accepted when named and, with a filter, when it matches"_test = [] {
        const gr::property_map start{{std::string(gr::tag::TRIGGER_NAME.key()), "start"}};
        const gr::property_map other{{std::string(gr::tag::TRIGGER_NAME.key()), "other"}};
        const gr::property_map unnamed{{"x", 1}};

        primitives::MatchState state;
        Refusals               refusals;
        expect(!primitives::compileOptionalFilter(refusals, "", "pulse", state)) << "no definition, no filter";
        expect(primitives::acceptsTrigger(state, false, gr::property_map_view{other}));
        expect(!primitives::acceptsTrigger(state, false, gr::property_map_view{unnamed})) << "a tag without a trigger name is no trigger";

        expect(primitives::compileOptionalFilter(refusals, "start", "pulse", state));
        expect(primitives::acceptsTrigger(state, true, gr::property_map_view{start}));
        expect(!primitives::acceptsTrigger(state, true, gr::property_map_view{other}));

        expect(!primitives::compileOptionalFilter(refusals, "[start", "pulse", state)) << "an unmatched bracket is refused";
        expect(eq(refusals.reasons.size(), 1UZ)) << "and the refusal is reported as an error message";
        expect(!primitives::acceptsTrigger(state, true, gr::property_map_view{start})) << "and leaves no filter behind";
    };
};
} // namespace qa_events

namespace qa_event_store {
using namespace gr::blocks::trigger;

[[nodiscard]] StoredEvent dated(std::uint64_t at, std::string name) {
    gr::property_map event{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), at}};
    return StoredEvent{.event = std::move(event), .at = at};
}

[[nodiscard]] StoredEvent simultaneous(std::uint64_t at, float offset, std::string name, std::string context) {
    gr::property_map event{
        {std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)},
        {std::string(gr::tag::TRIGGER_TIME.key()), at},
        {std::string(gr::tag::TRIGGER_OFFSET.key()), offset},
        {std::string(gr::tag::CONTEXT.key()), std::move(context)},
    };
    return StoredEvent{.event = std::move(event), .at = at};
}

[[nodiscard]] StoredEvent undated(std::string name) { //
    return StoredEvent{.event = gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)}}, .at = std::nullopt};
}

[[nodiscard]] std::string nameOf(const StoredEvent& stored) { //
    return std::string(gr::property_map_view{stored.event}.get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()).value_or(std::string_view{}));
}

const boost::ut::suite<"EventStore"> _eventStore = [] {
    using namespace boost::ut;

    "events arriving out of order are read back in time order"_test = [] {
        EventStore store;
        store.push(dated(300U, "third"));
        store.push(dated(100U, "first"));
        store.push(dated(200U, "second"));
        store.order();

        expect(eq(store.size(), 3UZ));
        expect(eq(nameOf(store.ordered()[0]), std::string("first")));
        expect(eq(nameOf(store.ordered()[1]), std::string("second")));
        expect(eq(nameOf(store.ordered()[2]), std::string("third")));
    };

    "an undated event keeps its arrival order, after the dated ones"_test = [] {
        EventStore store;
        store.push(undated("no time"));
        store.push(dated(500U, "dated"));
        store.push(undated("also no time"));
        store.order();

        expect(eq(nameOf(store.ordered()[0]), std::string("dated"))) << "a time orders, the absence of one does not";
        expect(eq(nameOf(store.ordered()[1]), std::string("no time")));
        expect(eq(nameOf(store.ordered()[2]), std::string("also no time")));
    };

    "simultaneous events are ordered by offset, name and context whichever arrived first"_test = [] {
        const auto  makeAll      = [] { return std::array{simultaneous(100U, 0.f, "b", "x"), simultaneous(100U, 0.f, "a", "y"), simultaneous(100U, 0.f, "a", "x"), simultaneous(100U, -0.25f, "z", "x")}; };
        const auto  expected     = std::vector<std::string>{"z/x", "a/x", "a/y", "b/x"};
        auto        arrival      = std::array<std::size_t, 4UZ>{0UZ, 1UZ, 2UZ, 3UZ};
        std::size_t permutations = 0UZ;
        do {
            auto       all = makeAll();
            EventStore store;
            for (const std::size_t index : arrival) {
                store.push(std::move(all[index]));
            }
            store.order();
            std::vector<std::string> order;
            for (const StoredEvent& stored : store.ordered()) {
                const gr::property_map_view event{stored.event};
                order.push_back(std::string(*event.get_if<std::string_view>(gr::tag::TRIGGER_NAME.key())) + "/" + std::string(*event.get_if<std::string_view>(gr::tag::CONTEXT.key())));
            }
            expect(order == expected) << "arrival permutation " << permutations;
            ++permutations;
        } while (std::ranges::next_permutation(arrival).found);
        expect(eq(permutations, 24UZ));
    };

    "events of equal time and equal tie-break keep their arrival order"_test = [] {
        EventStore store;
        auto       first  = simultaneous(100U, 0.f, "a", "x");
        auto       second = simultaneous(100U, 0.f, "a", "x");
        first.event.insert_or_assign("n", std::int64_t{1});
        second.event.insert_or_assign("n", std::int64_t{2});
        store.push(std::move(second));
        store.push(std::move(first));
        store.order();
        expect(eq(store.ordered()[0].event.template value_or<std::int64_t>("n", 0), std::int64_t{2})) << "stable: indistinguishable events are not reordered";
    };

    "an event is ready once the watermark has passed it by the lateness"_test = [] {
        EventStore store;
        store.lateness = 100U;
        store.push(dated(1'000U, "first"));
        store.order();
        expect(eq(store.ready(), 0UZ)) << "nothing yet shows that no earlier event can still come";
        store.push(dated(1'200U, "heartbeat"));
        store.order();
        expect(eq(store.ready(), 1UZ)) << "a later event moves the watermark: 1000 + 100 <= 1200";
    };

    "the watermark is the latest trigger_time seen, and a later arrival of an earlier time does not move it"_test = [] {
        EventStore store;
        store.lateness = 300U; // how far one producer's events may trail another's
        store.push(dated(1'200U, "edge"));
        store.push(dated(1'000U, "pulse")); // arrives second, dated earlier
        store.order();
        expect(eq(store.watermark.value_or(0U), std::uint64_t{1'200U})) << "the earlier time leaves the watermark at 1200";
        expect(eq(nameOf(store.ordered()[0]), std::string("pulse"))) << "ordered by trigger_time, not arrival";
        expect(eq(store.ready(), 0UZ)) << "1000 + 300 > 1200: an event dated before 1000 could still come";

        store.push(dated(1'500U, "later"));
        store.order();
        expect(eq(store.watermark.value_or(0U), std::uint64_t{1'500U}));
        expect(eq(store.ready(), 2UZ)) << "watermark 1500: pulse (1000 + 300) and edge (1200 + 300) are ready, later (1500 + 300) waits";
    };

    "without a lateness every drained event is ready, as before"_test = [] {
        EventStore store;
        store.push(dated(300U, "late"));
        store.push(dated(100U, "early"));
        store.order();
        expect(eq(store.ready(), 2UZ));
    };

    "a full store loses its oldest entry and counts it"_test = [] {
        EventStore store;
        store.capacity = 2UZ;
        store.push(dated(100U, "first"));
        store.push(dated(200U, "second"));
        store.push(dated(300U, "third"));

        expect(eq(store.size(), 2UZ));
        expect(eq(store.dropped, 1U)) << "the loss is counted, never silent";
        expect(eq(nameOf(store.ordered()[0]), std::string("second"))) << "the oldest went, not the newest";
    };

    "retiring drops what the block has finished with"_test = [] {
        EventStore store;
        store.push(dated(100U, "first"));
        store.push(dated(200U, "second"));
        store.retire(1UZ);

        expect(eq(store.size(), 1UZ));
        expect(eq(nameOf(store.ordered()[0]), std::string("second")));
        store.retire(99UZ);
        expect(store.empty()) << "retiring more than it holds empties it rather than overrunning";
    };

    "clearing forgets the entries and the loss count"_test = [] {
        EventStore store;
        store.capacity = 1UZ;
        store.push(dated(100U, "first"));
        store.push(dated(200U, "second"));
        store.clear();

        expect(store.empty());
        expect(eq(store.dropped, 0U));
    };
};
} // namespace qa_event_store

namespace qa_time_base {
using namespace gr::blocks::trigger;

const boost::ut::suite<"TimeBase"> _timeBase = [] {
    using namespace boost::ut;

    "with neither an anchor nor a rate nothing can be dated"_test = [] {
        TimeBase time;
        expect(!time.measuresDuration());
        expect(!time.datesSamples());
        expect(!time.seconds(1000UZ).has_value());
        expect(!time.at(0UZ).has_value());
    };

    "a rate alone measures a duration but dates no sample"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        expect(time.measuresDuration());
        expect(!time.datesSamples()) << "without an origin there is no absolute time";
        expect(approx(time.seconds(500UZ).value(), 0.5, 1e-12));
        expect(eq(time.samples(0.25).value(), 250UZ));
    };

    "a duration is the shorter of its sample count and its seconds, unset ones ignored"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        expect(!time.samples(-1.).has_value()) << "a negative duration has no sample count";
        expect(!TimeBase{}.samples(1.).has_value()) << "without a rate seconds cannot be counted";

        struct Case {
            gr::Size_t  samples;
            float       seconds;
            std::size_t expected;
        };
        for (const auto& [samples, seconds, expected] : {Case{0U, 0.f, 0UZ}, Case{40U, 0.f, 40UZ}, Case{0U, 0.05f, 50UZ}, Case{40U, 0.05f, 40UZ}, Case{80U, 0.05f, 50UZ}}) {
            const auto resolved = resolveDuration(time, samples, seconds);
            expect(eq(resolved.value_or(0UZ), expected)) << samples << seconds;
            expect(eq(resolved.has_value(), expected != 0UZ));
        }
        expect(!resolveDuration(TimeBase{}, 0U, 0.05f).has_value()) << "seconds need a rate";

        expect(withinDuration(time, 30UZ, 20U, 0.f, 0U, 0.05f) && withinDuration(time, 20UZ, 20U, 0.f, 0U, 0.f) && withinDuration(time, 50UZ, 0U, 0.f, 0U, 0.05f));
        expect(!withinDuration(time, 19UZ, 20U, 0.f, 0U, 0.f) && !withinDuration(time, 51UZ, 0U, 0.f, 0U, 0.05f));
        expect(withinDuration(time, 123456UZ, 0U, 0.f, 0U, 0.f)) << "no bound accepts any width";
    };

    "one anchor plus the rate setting dates every sample"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.f, 10UZ);
        expect(time.datesSamples());
        expect(eq(time.at(10UZ).value(), std::uint64_t{1'000'000'000U}));
        expect(eq(time.at(11UZ).value(), std::uint64_t{1'001'000'000U}));
        expect(eq(time.at(0UZ).value(), std::uint64_t{990'000'000U})) << "ten samples before the anchor is ten milliseconds earlier";
    };

    "a second anchor re-anchors, and the rate stays the nominal one"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);
        time.anchorAt(1'003'000'000U, 0.f, 2UZ);
        expect(approx(time.rate(), 1000., 1e-9)) << "a rate measured between anchors would take the jitter for a rate";
        expect(eq(time.at(3UZ).value(), std::uint64_t{1'004'000'000U})) << "counted on from the newest anchor";
    };

    "the anchor offset shifts the origin, in seconds"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.5e-3f, 0UZ);
        expect(eq(time.at(0UZ).value(), std::uint64_t{1'000'500'000U}));
    };

    "a rate that does not divide a nanosecond does not drift"_test = [] {
        TimeBase time;
        time.setRate(3000.);
        time.anchorAt(0U, 0.f, 0UZ);
        expect(eq(time.at(3'000'000UZ).value(), std::uint64_t{1'000'000'000'000U})) << "1e6 samples at 3 kHz is exactly 1000 s";
        expect(eq(time.at(3UZ).value(), std::uint64_t{1'000'000U})) << "and three samples are exactly one millisecond";
    };

    "a time near the present epoch finds its sample to the nanosecond"_test = [] {
        constexpr std::uint64_t kAnchorNs = 1'800'000'000'000'000'000U;
        TimeBase                time;
        time.setRate(1e8);
        time.anchorAt(kAnchorNs, 0.f, 100UZ);
        expect(eq(time.indexAt(kAnchorNs + 130U).value(), 113UZ)) << "130 ns at 100 MHz is 13 samples; a double at 1.8e18 ns resolves only 256 ns";
        expect(eq(time.indexAt(kAnchorNs - 130U).value(), 87UZ)) << "and as many before the anchor";
    };

    "an anchor tag with gr:trigger_offset -0.5 s dates sample 0 half a second earlier"_test = [] {
        TimeBase               time;
        const gr::property_map anchorTag{{std::string(gr::tag::SAMPLE_RATE.key()), 1000.f}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{2'000'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), -0.5f}};
        std::ignore = time.adopt(gr::property_map_view{anchorTag}, 0UZ);
        expect(eq(time.at(0UZ).value(), std::uint64_t{1'500'000'000U})) << "2.0 s - 0.5 s";
        expect(eq(time.at(1UZ).value(), std::uint64_t{1'501'000'000U})) << "one sample at 1 kHz later";
        expect(eq(time.indexAt(1'500'000'000U).value(), 0UZ));
    };

    "without a dated rate indexAt has no answer"_test = [] {
        TimeBase anchorOnly;
        anchorOnly.anchorAt(1'000'000'000U, 0.f, 0UZ);
        expect(!anchorOnly.indexAt(1'000'000'000U).has_value()) << "an anchor without a rate cannot convert time to samples";

        TimeBase rateOnly;
        rateOnly.setRate(1000.);
        expect(!rateOnly.indexAt(1'000'000'000U).has_value()) << "a rate without an anchor has no origin";
    };

    "a reset forgets the anchor and the rate it derived, not the setting"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);
        time.reset();
        expect(!time.datesSamples());
        expect(time.measuresDuration()) << "the sample_rate setting survives, an anchor does not";
    };

    "a sample_rate tag is adopted under its wire key gr:sample_rate and its setting key sample_rate"_test = [] {
        TimeBase               wire;
        const gr::property_map wireTag{{std::string(gr::tag::SAMPLE_RATE.key()), 48'000.f}};
        std::ignore = wire.adopt(gr::property_map_view{wireTag}, 0UZ);
        expect(approx(wire.rate(), 48'000., 1e-9)) << "the wire key is how a source publishes the rate";

        TimeBase               setting;
        const gr::property_map settingTag{{std::string(gr::tag::SAMPLE_RATE.shortKey()), 12'000.f}};
        std::ignore = setting.adopt(gr::property_map_view{settingTag}, 0UZ);
        expect(approx(setting.rate(), 12'000., 1e-9));
    };

    "the merged condition of several tags carries the latest gr:sample_rate"_test = [] {
        const gr::property_map                                                            first{{std::string(gr::tag::SAMPLE_RATE.key()), 1'000.f}};
        const gr::property_map                                                            second{{std::string(gr::tag::SAMPLE_RATE.key()), 2'000.f}, {std::string(gr::tag::TRIGGER_NAME.key()), std::string("edge")}};
        const std::vector<gr::Tag>                                                        tags{gr::Tag{3UZ, gr::property_map_view{first}}, gr::Tag{3UZ, gr::property_map_view{second}}};
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> scratch{};

        const gr::property_map_view merged = gr::blocks::trigger::detail::mergedAt(tags.begin(), tags.end(), scratch);
        const auto                  rate   = merged.get_if<float>(gr::tag::SAMPLE_RATE.key());
        expect(rate != nullptr) << "a rate published under the wire key must not be dropped";
        if (rate != nullptr) {
            expect(eq(*rate, 2'000.f));
        }
    };

    "a dated trigger carries its event as trigger_time, the tag sitting on the first sample at or after it"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);

        const auto                  edge = datedTrigger(time, 4UZ, -0.5, 0.f, "edge", "ctx");
        const gr::property_map_view view{edge};
        const auto                  stamp  = view.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto                  offset = view.get_if<float>(gr::tag::TRIGGER_OFFSET.key());
        expect(stamp != nullptr and offset != nullptr) << fatal;
        expect(eq(*stamp, std::uint64_t{1'003'500'000U})) << "sample 4 is at 1.004 s; half a sample (0.5 ms at 1 kHz) earlier";
        expect(eq(*offset, 0.f)) << "a whole number of ns leaves no residual";
    };

    "trigger_offset holds the sub-ns residual of the event time"_test = [] {
        TimeBase time;
        time.setRate(3000.);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);

        // sample 4: 4e9/3000 = 1'333'333.33 ns after the anchor, rounded by at() to 1'333'333; half a sample is 1e9/6000 = 166'666.667 ns
        const auto                  edge = datedTrigger(time, 4UZ, -0.5, 0.f, "edge", "ctx");
        const gr::property_map_view view{edge};
        const auto                  stamp  = view.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto                  offset = view.get_if<float>(gr::tag::TRIGGER_OFFSET.key());
        expect(stamp != nullptr and offset != nullptr) << fatal;
        expect(eq(*stamp, std::uint64_t{1'001'166'666U})) << "1'001'333'333 - 166'666.667 = 1'001'166'666.333, floored";
        expect(approx(*offset, 0.333333e-9f, 1e-12f)) << "the third of a nanosecond that the floor dropped, in seconds";
    };

    "a time base anchors on the event time of its first tag, up to a sample before the tag's sample"_test = [] {
        TimeBase source;
        source.setRate(1000.);
        source.anchorAt(1'000'000'000U, 0.f, 0UZ);
        const auto pulse = datedTrigger(source, 40UZ, -0.5, 0.f, "pulse", "");

        TimeBase downstream;
        downstream.setRate(1000.);
        expect(downstream.adopt(gr::property_map_view{pulse}, 40UZ) == std::optional{AnchorChange::first});
        expect(downstream.at(40UZ) == std::optional<std::uint64_t>{1'039'500'000U}) << "the first anchor is adopted as it is: the sample is 40 ms after the source's origin, less the half sample";
        expect(downstream.at(0UZ) == std::optional<std::uint64_t>{999'500'000U});
    };

    "a derived trigger within one sample does not move the anchor, a jump of one sample or more does"_test = [] {
        TimeBase source;
        source.setRate(1000.);
        source.anchorAt(1'000'000'000U, 0.f, 0UZ);

        TimeBase downstream;
        downstream.setRate(1000.);
        const gr::property_map origin{{std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'000'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}};
        expect(downstream.adopt(gr::property_map_view{origin}, 0UZ) == std::optional{AnchorChange::first});

        const auto derived = datedTrigger(source, 40UZ, -0.5, 0.f, "derived", ""); // event at 1.0395 s, the running time at sample 40 is 1.040 s: 0.5 ms apart
        expect(!downstream.adopt(gr::property_map_view{derived}, 40UZ).has_value()) << "0.5 sample apart is the tag's own sample lying after its event";
        expect(eq(downstream.anchorIndex, 0UZ));
        expect(eq(downstream.anchorTimeNs.value_or(0U), std::uint64_t{1'000'000'000U})) << "unmoved";

        const gr::property_map exactlyOneSample{{std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'041'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}};
        expect(downstream.adopt(gr::property_map_view{exactlyOneSample}, 40UZ) == std::optional{AnchorChange::forward}) << "1.041 s against the running 1.040 s is 1 ms = one sample period";
        expect(downstream.at(40UZ) == std::optional<std::uint64_t>{1'041'000'000U});

        const gr::property_map jumpBack{{std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'030'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}};
        expect(downstream.adopt(gr::property_map_view{jumpBack}, 50UZ) == std::optional{AnchorChange::backward}) << "running time at 50 is 1.051 s, the tag says 1.030 s";
        expect(downstream.at(50UZ) == std::optional<std::uint64_t>{1'030'000'000U});
    };

    "a tag dates its sample at trigger_time + trigger_offset"_test = [] {
        const gr::property_map hardware{{std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{2'000'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 1e-3f}};
        TimeBase               time;
        time.setRate(1000.);
        std::ignore = time.adopt(gr::property_map_view{hardware}, 10UZ);
        expect(time.at(10UZ) == std::optional<std::uint64_t>{2'001'000'000U}) << "the sample is 1 ms after the hardware trigger";
    };

    "an undatable trigger carries a name and no time at all"_test = [] {
        const TimeBase time;
        const auto     found = datedTrigger(time, 0UZ, 0., 0.f, "edge", "ctx");
        expect(!gr::property_map_view{found}.contains(gr::tag::TRIGGER_TIME.key())) << "a fabricated time would be worse than none";
        expect(gr::property_map_view{found}.contains(gr::tag::TRIGGER_NAME.key()));
    };
};
} // namespace qa_time_base

namespace qa_window_collector {
using gr::blocks::trigger::WindowCollector;

[[nodiscard]] std::vector<std::vector<float>> drain(WindowCollector<float>& windows) {
    std::vector<std::vector<float>> found;
    while (auto window = windows.take()) {
        found.push_back(std::move(window->samples));
    }
    return found;
}

const boost::ut::suite<"WindowCollector"> _windowCollector = [] {
    using namespace boost::ut;

    const std::array<float, 9> ramp{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};

    "windows that follow one another hold every sample once"_test = [&] {
        WindowCollector<float> windows;
        for (std::size_t opening = 0UZ; opening < 9UZ; opening += 3UZ) {
            expect(windows.open(opening, opening + 3UZ));
        }
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 3UZ));
        if (found.size() == 3UZ) {
            expect(eq(found[0].size(), 3UZ));
            expect(approx(found[1][0], 3.f, 1e-6f)) << "the second window starts where the first ended";
            expect(approx(found[2][2], 8.f, 1e-6f));
        }
    };

    "overlapping windows share the samples they both cover"_test = [&] {
        WindowCollector<float> windows;
        for (std::size_t opening = 0UZ; opening + 3UZ <= 9UZ; opening += 2UZ) {
            expect(windows.open(opening, opening + 3UZ));
        }
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 4UZ)) << "openings at 0, 2, 4 and 6";
        if (found.size() == 4UZ) {
            expect(approx(found[0][2], 2.f, 1e-6f));
            expect(approx(found[1][0], 2.f, 1e-6f)) << "sample 2 belongs to two windows at once";
        }
    };

    "a window that closes where it opened is delivered empty, not skipped"_test = [&] {
        WindowCollector<float> windows;
        expect(windows.open(0UZ, 0UZ));
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 1UZ)) << "a quiet interval is a window of length zero, and a consumer counts windows";
        if (!found.empty()) {
            expect(found[0].empty());
        }
    };

    "windows closing out of the order they opened still come out in closing order"_test = [&] {
        WindowCollector<float> windows;
        expect(windows.open(0UZ, 6UZ));
        expect(windows.open(0UZ, 2UZ));
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 2UZ));
        if (found.size() == 2UZ) {
            expect(eq(found[0].size(), 2UZ)) << "the one that closed first comes first";
            expect(eq(found[1].size(), 6UZ));
        }
    };

    "a collector already full refuses a window and counts it"_test = [&] {
        WindowCollector<float> windows;
        windows.maxOpen = 2UZ;
        expect(windows.open(0UZ, 100UZ));
        expect(windows.open(0UZ, 100UZ));
        expect(!windows.open(0UZ, 100UZ)) << "a notifier that never closes anything would otherwise grow without limit";
        expect(eq(windows.refused, 1U));
        expect(eq(windows.nOpen(), 2UZ));
    };

    "closing everything ends the open windows where the stream did"_test = [&] {
        WindowCollector<float> windows;
        expect(windows.openHere());
        windows.push(ramp);
        expect(eq(windows.nClosed(), 0UZ)) << "a window with no closing index waits for a notifier";
        windows.closeAll();

        const auto found = drain(windows);
        expect(eq(found.size(), 1UZ));
        if (!found.empty()) {
            expect(eq(found[0].size(), 9UZ));
        }
    };

    "a window planned for a later position takes only the samples from there"_test = [&] {
        WindowCollector<float> windows;
        expect(windows.open(4UZ, 7UZ));
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 1UZ));
        if (!found.empty()) {
            expect(eq(found[0].size(), 3UZ));
            expect(approx(found[0][0], 4.f, 1e-6f)) << "it starts where it was told to, not where the stream was";
        }
    };

    "a window reaches back before its opening into the history and forward past it"_test = [&] {
        WindowCollector<float> windows;
        windows.reachBack(16UZ);
        windows.push(std::span{ramp}.first(4UZ));
        expect(windows.open(1UZ, 7UZ)) << "two samples before the decision at sample 3, three after";
        expect(eq(windows.nClosed(), 0UZ)) << "the samples after it have not arrived yet";

        windows.push(std::span{ramp}.subspan(4UZ));
        const auto window = windows.take();
        expect(window.has_value());
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{1.f, 2.f, 3.f, 4.f, 5.f, 6.f})) << "the window spans the decision, not just what followed it";
            expect(!window->partial);
        }
    };

    "a history deepened late keeps what it held and never shrinks; closing the oldest of none does nothing"_test = [&] {
        WindowCollector<float> windows;
        windows.reachBack(2UZ);
        windows.push(std::span{ramp}.first(4UZ));
        windows.reachBack(16UZ);
        windows.reachBack(4UZ);
        windows.reachBack(0UZ);
        expect(eq(windows.history->capacity(), 16UZ));
        windows.closeOldest();
        expect(eq(windows.nOpen() + windows.nClosed(), 0UZ));

        expect(windows.open(2UZ, 6UZ));
        windows.push(std::span{ramp}.subspan(4UZ, 2UZ));
        const auto window = windows.take();
        expect(window.has_value());
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{2.f, 3.f, 4.f, 5.f})) << "the two samples held before deepening lead the window";
            expect(!window->partial);
        }
    };

    "a window reaching back further than the history is cut where the history ends and marked partial"_test = [&] {
        WindowCollector<float> windows;
        windows.reachBack(2UZ);
        windows.push(std::span{ramp}.first(6UZ));
        expect(windows.open(1UZ, 6UZ)) << "samples 1 to 3 have left a history of two";

        const auto window = windows.take();
        expect(window.has_value());
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{4.f, 5.f})) << "only what the history still held, never padding";
            expect(eq(window->startsAt, 4UZ)) << "the window says where it really starts";
            expect(window->partial) << "and the caller is told it is short";
        }
    };

    "what travels with a window comes back with it, in the order the windows closed"_test = [&] {
        WindowCollector<float, std::uint64_t> windows;
        windows.reachBack(16UZ);
        windows.push(std::span{ramp}.first(3UZ));
        expect(windows.open(0UZ, 3UZ, 111U)) << "already complete when opened";
        expect(windows.open(1UZ, 4UZ, 222U));

        windows.push(std::span{ramp}.subspan(3UZ));
        const auto first  = windows.take();
        const auto second = windows.take();
        expect(first.has_value() and second.has_value());
        if (first and second) {
            expect(eq(first->info, std::uint64_t{111U})) << "each window carries the decision it belongs to";
            expect(eq(second->info, std::uint64_t{222U}));
            expect(std::ranges::equal(first->samples, std::vector{0.f, 1.f, 2.f}));
            expect(std::ranges::equal(second->samples, std::vector{1.f, 2.f, 3.f}));
        }
    };

    "a window keeps its samples however long it waits to be read"_test = [&] {
        WindowCollector<float> windows;
        windows.reachBack(2UZ);
        windows.push(std::span{ramp}.first(2UZ));
        expect(windows.open(1UZ, 3UZ));
        windows.push(std::span{ramp}.subspan(2UZ));
        windows.push(std::span{ramp});

        const auto window = windows.take();
        expect(window.has_value()) << "the history of two has turned over many times";
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{1.f, 2.f}));
        }
    };

    "a closing position moved before the current one shortens the window to it"_test = [&] {
        WindowCollector<float> windows;
        expect(windows.open(0UZ, 100UZ));
        windows.push(std::span{ramp}.first(6UZ));
        windows.openWindows.front().closesAt = 2UZ;

        const auto window = windows.take();
        expect(window.has_value());
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{0.f, 1.f}));
        }
    };

    "a reset forgets the windows and the history"_test = [&] {
        WindowCollector<float> windows;
        windows.reachBack(4UZ);
        windows.push(std::span{ramp}.first(3UZ));
        expect(windows.open(1UZ));
        windows.reset();
        expect(eq(windows.nOpen(), 0UZ));
        expect(eq(windows.streamIndex, 0UZ));

        windows.push(std::span{ramp}.subspan(8UZ));
        expect(windows.open(0UZ, 1UZ));
        const auto window = windows.take();
        expect(window.has_value());
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{8.f})) << "the history holds only what came after the reset";
        }
    };

    "what a window cuts out of the history, drawn"_test = [] {
        constexpr std::size_t kLength = 60UZ;
        constexpr std::size_t kPeak   = 30UZ;
        std::vector<float>    stream(kLength, 0.f);
        for (std::size_t i = kPeak; i < kLength && i - kPeak < 15UZ; ++i) {
            stream[i] = 5.f * std::exp(-static_cast<float>(i - kPeak) / 4.f);
        }

        WindowCollector<float> windows;
        windows.reachBack(kLength);
        windows.push(stream);
        expect(windows.open(kPeak - 10UZ, kPeak + 21UZ));
        const auto window = windows.take();
        expect(window.has_value());
        if (!window) {
            return;
        }
        expect(eq(window->samples.size(), 31UZ)) << "ten before the decision, the decision, twenty after";
        expect(eq(window->samples[10UZ], 5.f)) << "the decision's own sample";

        std::vector<double> xStream(kLength);
        std::vector<double> yStream(kLength);
        for (std::size_t i = 0UZ; i < kLength; ++i) {
            xStream[i] = static_cast<double>(i);
            yStream[i] = static_cast<double>(stream[i]);
        }
        auto whole = gr::graphs::ImChart<90, 12>({{0., static_cast<double>(kLength - 1UZ)}, {-0.5, 6.}});
        whole.draw(xStream, yStream, "stream");
        std::println("\nthe stream, with the decision at sample {} of {}:", kPeak, kLength);
        whole.draw();

        std::vector<double> xCut(window->samples.size());
        std::vector<double> yCut(window->samples.size());
        for (std::size_t i = 0UZ; i < window->samples.size(); ++i) {
            xCut[i] = static_cast<double>(i) - 10.;
            yCut[i] = static_cast<double>(window->samples[i]);
        }
        auto cut = gr::graphs::ImChart<90, 12>({{xCut.front(), xCut.back()}, {-0.5, 6.}});
        cut.draw(xCut, yCut, "window");
        std::println("\nand the window it cut, zero at the decision -- the ten samples left of it are the history:");
        cut.draw();
    };

    "what the windows held, drawn"_test = [&] {
        WindowCollector<float> windows;
        for (std::size_t opening = 0UZ; opening + 3UZ <= 9UZ; opening += 2UZ) {
            expect(windows.open(opening, opening + 3UZ));
        }
        windows.push(ramp);
        const auto found = drain(windows);

        gr::testing::MarbleDiagram diagram{"WindowCollector: length 3 every 2 samples, so each sample is in two windows"};
        diagram.unit = "sample";
        for (std::size_t i = 0UZ; i < found.size(); ++i) {
            auto& row = diagram.row(std::format("w{}", i));
            for (std::size_t j = 0UZ; j < found[i].size(); ++j) {
                row.at(2UZ * i + j, std::format("{:.0f}", found[i][j]));
            }
            row.completes();
        }
        diagram.print();
        expect(eq(found.size(), 4UZ));
    };
};
} // namespace qa_window_collector

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
