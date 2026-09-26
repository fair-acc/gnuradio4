#include "TriggerTest.hpp"
#include <boost/ut.hpp>
#include <cstdint>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/EventReduce.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>
#include <string>
#include <vector>

namespace qaAccumulate {
using namespace gr::blocks::trigger;
using gr::testing::ProcessFunction;
using gr::testing::TagSink;

namespace {
/// 1 2 3 | 4 5 | 6, where the bars are the segment-ending triggers at samples 3 and 5
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}, {"4", 4.f}, {"5", 5.f}, {"6", 6.f}};
    return map;
}
[[nodiscard]] const gr::property_map& kTags() {
    static const gr::property_map map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("cut")}}}};
    return map;
}
constexpr std::string_view kScript = "1 2 3 T:4 5 T:6 |";

template<Accumulation mode, typename TOut = float>
[[nodiscard]] std::vector<TOut> segmentsOf(gr::property_map settings) {
    settings[std::string("segment_filter")] = std::string("cut");
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     fold   = fixture.emplace<Accumulate<float, mode>>(std::move(settings));
    auto&                     sink   = fixture.emplace<TagSink<TOut, ProcessFunction::USE_PROCESS_ONE>>();

    boost::ut::expect(fixture.connect<"out", "in">(source, fold).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(fold, sink).has_value());
    boost::ut::expect(fixture.run().has_value());

    return std::vector<TOut>(sink._samples.begin(), sink._samples.end());
}
} // namespace

const boost::ut::suite<"Accumulate"> _accumulate = [] {
    using namespace boost::ut;

    "the last sample of each segment"_test = [] {
        const auto found = segmentsOf<Accumulation::last>({});
        expect(found == std::vector<float>{3.f, 5.f}) << "the sample before each cut; the third segment never ended";
    };

    "the sum of each segment"_test = [] {
        const auto found = segmentsOf<Accumulation::AUTO>({{"operation", std::string("sum")}});
        expect(found == std::vector<float>{6.f, 9.f}) << "1+2+3, then 4+5";
    };

    "the mean of each segment"_test = [] {
        const auto found = segmentsOf<Accumulation::AUTO>({{"operation", std::string("mean")}});
        expect(found == std::vector<float>{2.f, 4.5f});
    };

    "the largest and the smallest of each segment"_test = [] {
        expect(segmentsOf<Accumulation::AUTO>({{"operation", std::string("maximum")}}) == std::vector<float>{3.f, 5.f});
        expect(segmentsOf<Accumulation::AUTO>({{"operation", std::string("minimum")}}) == std::vector<float>{1.f, 4.f});
    };

    "how many samples each segment held"_test = [] {
        const auto found = segmentsOf<Accumulation::AUTO>({{"operation", std::string("count")}});
        expect(found == std::vector<float>{3.f, 2.f}) << "three samples before the first cut, two between the cuts";
    };

    "whether every sample of the segment compared true"_test = [] {
        const auto all = segmentsOf<Accumulation::all, std::uint8_t>({{"predicate", std::string("less")}, {"threshold", 4.f}});
        expect(all == std::vector<std::uint8_t>{1U, 0U}) << "1 2 3 are all below 4; the segment holding 4 and 5 is not";
    };

    "whether any sample of the segment compared true"_test = [] {
        const auto any = segmentsOf<Accumulation::any, std::uint8_t>({{"predicate", std::string("greater")}, {"threshold", 2.5f}});
        expect(any == std::vector<std::uint8_t>{1U, 1U}) << "3 in the first segment, 4 and 5 in the second";
    };

    "with no trigger named the cadence ends the segments"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::AUTO>>({{"operation", std::string("sum")}, {"n_samples", 2U}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{3.f, 7.f, 11.f}) << "1+2, 3+4, 5+6: a block with no trigger source is not a silent one";
    };

    "a cadence in seconds is the same limit where the rate is known"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::AUTO>>({{"operation", std::string("count")}, {"n_samples", 0U}, {"timeout", 0.003f}, {"sample_rate", 1000.f}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{3.f, 3.f}) << "3 ms at 1 kHz is three samples, twice over";
    };

    "a trigger ends a segment before the cadence does"_test = [] {
        // the cut sits at sample 3, well inside a cadence of 100
        const auto found = segmentsOf<Accumulation::AUTO>({{"operation", std::string("count")}, {"n_samples", 100U}});
        expect(found == std::vector<float>{3.f, 2.f}) << "whichever comes first, and here it is the trigger";
    };

    "the segments and what each amounted to, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::AUTO>>({{"segment_filter", std::string("cut")}, {"operation", std::string("sum")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"Reduce(sum): a trigger ends a segment, and the sum is published where it sat"};
        diagram.unit  = "sample";
        auto& samples = diagram.row("in");
        for (std::size_t i = 0UZ; i < 6UZ; ++i) {
            samples.at(i, std::format("{}", i + 1UZ));
        }
        samples.completes();
        diagram.condition("Reduce(operation = sum, segment_filter = \"cut\")");
        auto&                            out = diagram.row("out");
        const std::vector<std::uint64_t> cuts{3U, 5U};
        for (std::size_t i = 0UZ; i < sink._samples.size() && i < cuts.size(); ++i) {
            out.at(cuts[i], std::format("{:.0f}", sink._samples[i]));
        }
        out.completes();
        diagram.print();

        expect(eq(sink._samples.size(), 2UZ));
    };
};
} // namespace qaAccumulate

namespace qaCount {
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;
using gr::trigger_test::undatedEvent;

namespace {
/// events one millisecond apart, so the rate is exactly 1 kHz and can be asserted rather than approximated loosely
[[nodiscard]] std::vector<gr::property_map> ticks(std::size_t count, std::string name = "edge") {
    std::vector<gr::property_map> events;
    for (std::size_t i = 0UZ; i < count; ++i) {
        events.push_back(eventNamed(name, 1'000'000'000U + static_cast<std::uint64_t>(i) * 1'000'000U));
    }
    return events;
}
} // namespace

const boost::ut::suite<"Count"> _count = [] {
    using namespace boost::ut;

    "every nth match is reported, and carries the running total"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks(10UZ);
        auto& counter                    = fixture.emplace<Count>({{"filter", std::string("edge")}, {"mode", std::string("every")}, {"n", 3U}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(counter.n_matches.value, 10U));
        expect(eq(tap.size(), 3UZ)) << "the third, sixth and ninth of ten";
        if (tap.size() == 3UZ) {
            const std::uint64_t* third = gr::property_map_view{tap.metaOf(0)}.get_if<std::uint64_t>(std::string_view{"count"});
            expect(third != nullptr && *third == 3U) << "the report says which occurrence it was";
            const std::uint64_t* ninth = gr::property_map_view{tap.metaOf(2)}.get_if<std::uint64_t>(std::string_view{"count"});
            expect(ninth != nullptr && *ninth == 9U);
        }
    };

    "nth reports exactly once, however long the stream runs"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks(10UZ);
        auto& counter                    = fixture.emplace<Count>({{"mode", std::string("nth")}, {"n", 4U}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(tap.size(), 1UZ));
        expect(eq(counter.n_matches.value, 10U)) << "it keeps counting after it has reported";
    };

    "the rate comes from the events' own times, not from a clock of its own"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks(11UZ); // ten intervals of 1 ms
        auto& counter                    = fixture.emplace<Count>({{"n", 100U}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(counter.n_matches.value, 11U));
        expect(approx(counter.rate.value, 1000., 1e-6)) << "eleven events, ten milliseconds apart in total";
        expect(eq(tap.size(), 0UZ)) << "a prescale larger than the stream reports nothing";
    };

    "an undated event is counted but cannot inform the rate"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {undatedEvent("edge"), undatedEvent("edge")};
        auto& counter                    = fixture.emplace<Count>({{"n", 1U}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(counter.n_matches.value, 2U));
        expect(eq(counter.n_undated.value, 2U));
        expect(approx(counter.rate.value, 0., 1e-12)) << "a rate from undated events would be invented";
    };

    "a prescale of zero is refused"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     counter = fixture.emplace<Count>({{"n", 5U}});
        counter.settings().init();
        std::ignore = counter.settings().applyStagedParameters();
        expect(eq(counter.n.value, 5U));

        expect(counter.settings().set({{"n", gr::Size_t(0)}}).empty());
        std::ignore = counter.settings().activateContext();
        std::ignore = counter.settings().applyStagedParameters();
        expect(eq(counter.n.value, 5U)) << "a prescale of no events has no meaning, so the previous one stays";
    };

    "what the prescaler let through, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks(9UZ);
        auto& counter                    = fixture.emplace<Count>({{"n", 3U}, {"trigger_name", std::string("every3rd")}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"Count: one event in three, with the rate the stream itself gives"};
        auto&                      in = diagram.row("evtIn");
        for (const gr::property_map& event : script._events) {
            in.at(*gr::property_map_view{event}.get_if<std::uint64_t>(std::string_view{gr::tag::TRIGGER_TIME.key()}), "edge");
        }
        in.completes();
        diagram.condition(std::format("Count(every 3rd), rate {:.0f} Hz", counter.rate.value));
        auto& out = diagram.row("evtOut");
        for (const std::uint64_t at : tap.times()) {
            out.at(at, "every3rd");
        }
        out.completes();
        diagram.print();

        expect(eq(tap.size(), 3UZ));
    };
};
} // namespace qaCount

namespace qaSequence {
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

namespace {
constexpr std::uint64_t kStart = 1'000'000'000U; // 1 s, so every time below reads as a round offset from it

struct Outcome {
    std::vector<std::string> names;
    gr::Size_t               sequences = 0U;
    gr::Size_t               timeouts  = 0U;
    gr::Size_t               disarmed  = 0U;
    gr::Size_t               ignored   = 0U;
};

[[nodiscard]] Outcome gate(std::vector<gr::property_map> events, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    auto& sequence                   = fixture.emplace<Sequence>(std::move(settings));
    auto& tap                        = fixture.emplace<EventTap>();

    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, sequence).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(sequence, tap).has_value());
    boost::ut::expect(fixture.run().has_value());

    return Outcome{.names = tap.names(), .sequences = sequence.n_sequences.value, .timeouts = sequence.n_timeouts.value, .disarmed = sequence.n_disarmed.value, .ignored = sequence.n_ignored.value};
}

[[nodiscard]] gr::property_map settingsFor(std::string rearm, double timeout = 0.) {
    return gr::property_map{{"arm_filter", std::string("arm")}, {"trigger_filter", std::string("edge")}, {"disarm_filter", std::string("disarm")}, //
        {"rearm", std::move(rearm)}, {"timeout", timeout}};
}
} // namespace

const boost::ut::suite<"Sequence"> _sequence = [] {
    using namespace boost::ut;

    "a trigger inside the window is reported, one before it is not"_test = [] {
        const auto found = gate({eventNamed("edge", kStart), eventNamed("arm", kStart + 10U), eventNamed("edge", kStart + 20U)}, settingsFor("auto"));

        expect(eq(found.sequences, 1U)) << "only the trigger that followed the arming event";
        expect(eq(found.ignored, 1U)) << "and the one before it is counted, not silently dropped";
        expect(eq(found.names.size(), 1UZ));
        if (!found.names.empty()) {
            expect(eq(found.names[0], std::string("sequence")));
        }
    };

    "rearm auto keeps the window open, so every trigger in it is reported"_test = [] {
        const auto found = gate({eventNamed("arm", kStart), eventNamed("edge", kStart + 10U), eventNamed("edge", kStart + 20U), eventNamed("edge", kStart + 30U)}, settingsFor("auto"));

        expect(eq(found.sequences, 3U));
    };

    "rearm manual closes the window on the first trigger"_test = [] {
        const auto found = gate({eventNamed("arm", kStart), eventNamed("edge", kStart + 10U), eventNamed("edge", kStart + 20U)}, settingsFor("manual"));

        expect(eq(found.sequences, 1U));
        expect(eq(found.ignored, 1U)) << "the second trigger found the window shut";
    };

    "a disarming event closes the window"_test = [] {
        const auto found = gate({eventNamed("arm", kStart), eventNamed("disarm", kStart + 10U), eventNamed("edge", kStart + 20U)}, settingsFor("auto"));

        expect(eq(found.sequences, 0U));
        expect(eq(found.disarmed, 1U));
        expect(eq(found.ignored, 1U));
    };

    "a window whose timeout passes expires, and says so"_test = [] {
        // the arming event opens a 1 ms window; the edge arrives 2 ms later, so the window expired before it
        const auto found = gate({eventNamed("arm", kStart), eventNamed("edge", kStart + 2'000'000U)}, settingsFor("auto", 0.001));

        expect(eq(found.timeouts, 1U));
        expect(eq(found.sequences, 0U)) << "a trigger after the window closed is not a sequence";
        expect(eq(found.ignored, 1U));
        expect(eq(found.names.size(), 1UZ));
        if (!found.names.empty()) {
            expect(eq(found.names[0], std::string("sequence_timeout"))) << "which is the event a supervisor watches for";
        }
    };

    "a trigger inside the timeout is still a sequence"_test = [] {
        const auto found = gate({eventNamed("arm", kStart), eventNamed("edge", kStart + 500'000U)}, settingsFor("auto", 0.001));

        expect(eq(found.sequences, 1U));
        expect(eq(found.timeouts, 0U));
    };

    "an arming event while the window is open restarts its timeout"_test = [] {
        const auto found = gate({eventNamed("arm", kStart), eventNamed("arm", kStart + 900'000U), eventNamed("edge", kStart + 1'500'000U)}, settingsFor("auto", 0.001));

        expect(eq(found.timeouts, 0U)) << "the second arming event moved the deadline out to 2.5 ms";
        expect(eq(found.sequences, 1U));
    };

    "the reported event carries how long the sequence took"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("arm", kStart), eventNamed("edge", kStart + 250'000U)};
        auto& sequence                   = fixture.emplace<Sequence>(settingsFor("auto"));
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, sequence).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(sequence, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(tap.size(), 1UZ));
        if (tap.size() > 0UZ) {
            const double* elapsed = gr::property_map_view{tap.metaOf(0)}.get_if<double>(std::string_view{"elapsed"});
            expect(elapsed != nullptr) << "the delay from arming to trigger is the measurement worth keeping";
            if (elapsed != nullptr) {
                expect(approx(*elapsed, 250e-6, 1e-12));
            }
        }
    };

    "the window, what fired in it and what expired, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("arm", kStart), eventNamed("edge", kStart + 300'000U), eventNamed("disarm", kStart + 600'000U), //
                              eventNamed("edge", kStart + 700'000U), eventNamed("arm", kStart + 900'000U), eventNamed("edge", kStart + 3'000'000U)};
        auto& sequence                   = fixture.emplace<Sequence>(settingsFor("auto", 0.001));
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, sequence).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(sequence, tap).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"Sequence: a trigger only counts between arm and disarm, and a window that expires says so"};
        auto&                      in = diagram.row("evtIn");
        for (const gr::property_map& event : script._events) {
            const gr::property_map_view view{event};
            in.at(*view.get_if<std::uint64_t>(std::string_view{gr::tag::TRIGGER_TIME.key()}), std::string(*view.get_if<std::string_view>(std::string_view{gr::tag::TRIGGER_NAME.key()})));
        }
        in.completes();
        diagram.condition("Sequence(arm -> edge -> disarm, timeout 1 ms, rearm auto)");
        auto& out = diagram.row("evtOut");
        for (const auto& [at, name] : tap.dated()) {
            out.at(at, name);
        }
        out.completes();
        diagram.print();

        expect(eq(sequence.n_sequences.value, 1U)) << "the edge inside the first window";
        expect(eq(sequence.n_disarmed.value, 1U));
        expect(eq(sequence.n_timeouts.value, 1U)) << "the second window expired before the last edge";
    };
};
} // namespace qaSequence

namespace qaEventBuilder {
using namespace gr::blocks::trigger;
using gr::trigger_test::CollectingSink;
using gr::trigger_test::EventTap;

namespace {
/// publishes a written list of fragments, so what the builder sees does not depend on work-call granularity
struct FragmentSource : gr::Block<FragmentSource> {
    gr::PortOut<gr::DataSet<float>, gr::Async> out;

    GR_MAKE_REFLECTABLE(FragmentSource, out);

    std::vector<gr::DataSet<float>> _fragments;
    std::size_t                     _published = 0UZ;

    /// one fragment per work call, so two of these interleave the way two front-ends do: a source that emptied itself
    /// first would make its own events look overdue before the other had answered
    gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        std::size_t emitted = 0UZ;
        if (_published < _fragments.size() && !outSpan.empty()) {
            outSpan[0] = _fragments[_published++];
            emitted    = 1UZ;
        }
        outSpan.publish(emitted);
        if (_published >= _fragments.size()) {
            this->requestStop();
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }
};

/// one fragment: a named signal of `length` samples, stamped with the event it belongs to
[[nodiscard]] gr::DataSet<float> fragmentOf(std::int64_t at, std::string name, std::size_t length, std::uint64_t id = 0U) {
    gr::DataSet<float> fragment;
    fragment.timestamp = at;
    fragment.axis_names.emplace_back("time");
    fragment.axis_units.emplace_back("s");
    fragment.axis_values.resize(1UZ);
    for (std::size_t i = 0UZ; i < length; ++i) {
        fragment.axis_values[0].push_back(static_cast<float>(i));
    }
    fragment.extents.push_back(static_cast<std::int32_t>(length));
    fragment.signal_names.emplace_back(std::move(name));
    fragment.signal_quantities.emplace_back("voltage");
    fragment.signal_units.emplace_back("V");
    for (std::size_t i = 0UZ; i < length; ++i) {
        fragment.signal_values.push_back(static_cast<float>(i));
    }
    fragment.signal_ranges.resize(1UZ);
    fragment.meta_information.resize(1UZ);
    if (id != 0U) {
        fragment.meta_information[0].insert_or_assign(std::string_view{"event_id"}, id);
    }
    fragment.timing_events.resize(1UZ);
    return fragment;
}

struct Outcome {
    std::size_t events     = 0UZ;
    std::size_t signals    = 0UZ;
    gr::Size_t  incomplete = 0U;
    gr::Size_t  dropped    = 0U;
    gr::Size_t  mismatched = 0U;
    std::size_t notices    = 0UZ;
};

[[nodiscard]] Outcome build(std::vector<gr::DataSet<float>> left, std::vector<gr::DataSet<float>> right, gr::property_map settings) {
    settings[std::string("n_inputs")] = gr::Size_t(2);
    gr::testing::GraphFixture fixture;
    auto&                     a = fixture.emplace<FragmentSource>();
    a._fragments                = std::move(left);
    auto& b                     = fixture.emplace<FragmentSource>();
    b._fragments                = std::move(right);
    auto& builder               = fixture.emplace<EventBuilder<float>>(std::move(settings));
    auto& sink                  = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
    auto& tap                   = fixture.emplace<EventTap>();

    boost::ut::expect(fixture.graph.connect(a, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#0")}).has_value());
    boost::ut::expect(fixture.graph.connect(b, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#1")}).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(builder, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(builder, tap).has_value());
    boost::ut::expect(fixture.run().has_value());

    return Outcome{.events = sink._collected.size(), .signals = sink._collected.empty() ? 0UZ : sink._collected.front().signal_names.size(), .incomplete = builder.n_incomplete.value, .dropped = builder.n_dropped.value, .mismatched = builder.n_mismatched.value, .notices = tap.size()};
}

constexpr std::int64_t kAt = 1'000'000'000;
} // namespace

const boost::ut::suite<"EventBuilder"> _eventBuilder = [] {
    using namespace boost::ut;

    "fragments sharing a timestamp become one event"_test = [] {
        const auto found = build({fragmentOf(kAt, "ch0", 4UZ)}, {fragmentOf(kAt, "ch1", 4UZ)}, {});

        expect(eq(found.events, 1UZ));
        expect(eq(found.signals, 2UZ)) << "one signal per fragment, in input order";
        expect(eq(found.incomplete, 0U));
    };

    "timestamps within the tolerance are the same event"_test = [] {
        const auto found = build({fragmentOf(kAt, "ch0", 4UZ)}, {fragmentOf(kAt + 30, "ch1", 4UZ)}, {{"tolerance", 50e-9}});

        expect(eq(found.events, 1UZ)) << "30 ns apart, inside a 50 ns tolerance";
    };

    "timestamps beyond the tolerance are different events"_test = [] {
        const auto found = build({fragmentOf(kAt, "ch0", 4UZ)}, {fragmentOf(kAt + 500'000, "ch1", 4UZ)}, {{"tolerance", 50e-9}, {"timeout", 1e-6}});

        expect(eq(found.events, 0UZ)) << "neither event was ever complete";
        expect(ge(found.incomplete, 1U)) << "and the older one is judged when the newer arrives from beyond the timeout";
        expect(ge(found.dropped, 1U));
    };

    "an incomplete event can be published as it stands"_test = [] {
        const auto found = build({fragmentOf(kAt, "ch0", 4UZ)}, {fragmentOf(kAt + 500'000, "ch1", 4UZ)}, //
            {{"tolerance", 50e-9}, {"timeout", 1e-6}, {"on_incomplete", std::string("emit_partial")}});

        expect(ge(found.events, 1UZ)) << "what was there is published rather than thrown away";
        expect(eq(found.dropped, 0U));
        expect(ge(found.incomplete, 1U)) << "and it is still counted as incomplete";
    };

    "an identifier can key the event instead of a time"_test = [] {
        const auto found = build({fragmentOf(0, "ch0", 4UZ, 42U)}, {fragmentOf(0, "ch1", 4UZ, 42U)}, {{"key", std::string("event_id")}});

        expect(eq(found.events, 1UZ)) << "neither fragment carries a usable timestamp, and neither needs to";
        expect(eq(found.signals, 2UZ));
    };

    "a fragment of another length is refused, and says why"_test = [] {
        const auto found = build({fragmentOf(kAt, "ch0", 4UZ)}, {fragmentOf(kAt, "ch1", 7UZ)}, {});

        expect(eq(found.mismatched, 1U)) << "a DataSet carries one extent per signal, so the seven cannot join the four";
        expect(eq(found.events, 0UZ));
        expect(ge(found.notices, 1UZ)) << "and the refusal reaches the event output";
    };

    "a fragment carrying nothing to match on is counted"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     a = fixture.emplace<FragmentSource>();
        a._fragments                = {fragmentOf(0, "ch0", 4UZ)}; // no timestamp, no identifier
        auto& b                     = fixture.emplace<FragmentSource>();
        b._fragments                = {fragmentOf(kAt, "ch1", 4UZ)};
        auto& builder               = fixture.emplace<EventBuilder<float>>({{"n_inputs", 2U}});
        auto& sink                  = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.graph.connect(a, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(b, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#1")}).has_value());
        expect(fixture.connect<"out", "in">(builder, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(builder.n_unkeyed.value, 1U)) << "matching by arrival order would invent an event";
        expect(eq(sink._collected.size(), 0UZ));
    };

    "what was assembled, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     a = fixture.emplace<FragmentSource>();
        a._fragments                = {fragmentOf(kAt, "ch0", 4UZ), fragmentOf(kAt + 1'000'000, "ch0", 4UZ)};
        auto& b                     = fixture.emplace<FragmentSource>();
        b._fragments                = {fragmentOf(kAt + 20, "ch1", 4UZ), fragmentOf(kAt + 1'000'020, "ch1", 4UZ)};
        auto& builder               = fixture.emplace<EventBuilder<float>>({{"n_inputs", 2U}, {"tolerance", 50e-9}, {"timeout", 1e-4}});
        auto& sink                  = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.graph.connect(a, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(b, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#1")}).has_value());
        expect(fixture.connect<"out", "in">(builder, sink).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"EventBuilder: two front-ends, two events, matched within 50 ns"};
        diagram.row("in#0").at(static_cast<std::uint64_t>(kAt), "ch0").at(static_cast<std::uint64_t>(kAt) + 1'000'000U, "ch0").completes();
        diagram.row("in#1").at(static_cast<std::uint64_t>(kAt) + 20U, "ch1").at(static_cast<std::uint64_t>(kAt) + 1'000'020U, "ch1").completes();
        diagram.condition("EventBuilder(n_inputs = 2, tolerance = 50 ns)");
        auto& out = diagram.row("out");
        for (const gr::DataSet<float>& event : sink._collected) {
            out.at(static_cast<std::uint64_t>(event.timestamp), std::format("{} signals", event.signal_names.size()));
        }
        out.completes();
        diagram.print();

        expect(eq(sink._collected.size(), 2UZ));
        expect(eq(builder.n_events.value, 2U));
    };
};
} // namespace qaEventBuilder

int main() { /* tests are statically executed */ }
