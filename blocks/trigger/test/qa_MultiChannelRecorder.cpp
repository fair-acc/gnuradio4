#include <boost/ut.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/MultiChannelRecorder.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

namespace qa_multi_channel_recorder {
using gr::trigger_test::kSampleRate;
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::trigger;
using gr::trigger_test::CollectingSink;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;

[[nodiscard]] gr::property_map tagsFor() {
    return gr::property_map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, //
                                      {std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart},                      //
                                      {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},                             //
                                      {std::string("sample_rate"), kSampleRate}}},
        {"R", gr::property_map{{std::string("sample_rate"), 2.f * kSampleRate}}}, //
        {"U", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("resync")}, {std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart + 2'000'000U}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}, {std::string("sample_rate"), kSampleRate}}}, {"S", gr::property_map{{std::string("sample_rate"), kSampleRate}}}};
}

struct Recorded {
    std::vector<gr::DataSet<float>> sets;
    gr::Size_t                      lost;
    std::size_t                     errors;
    gr::Size_t                      unmatched;
    gr::Size_t                      suppressed = 0U;
    std::string                     stage;
};

/** one channel, its decisions, and whatever the recorder makes of them */
[[nodiscard]] Recorded record(const std::string& script, const gr::property_map& values, std::vector<gr::property_map> decisions, gr::property_map settings, bool rateFromTags = false, std::size_t maxSamples = 0UZ) {
    gr::testing::GraphFixture fixture;
    auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", values}, {"sample_tags", tagsFor()}});
    auto&                     events  = fixture.emplace<EventScript>();
    events._events                    = std::move(decisions);
    events._stopWhenDone              = false;
    settings[std::string("n_inputs")] = 1U;
    if (!rateFromTags) {
        settings[std::string("sample_rate")] = kSampleRate;
    }
    auto& recorder = fixture.emplace<MultiChannelRecorder<float>>(std::move(settings));
    if (maxSamples > 0UZ) {
        recorder.in[0].max_samples = maxSamples;
    }
    auto& sets = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
    boost::ut::expect(fixture.graph.connect(source, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
    boost::ut::expect(fixture.graph.connect(events, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
    boost::ut::expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
    gr::trigger_test::ErrorMessages errors{fixture.scheduler()};
    boost::ut::expect(fixture.run().has_value());
    return {
        sets._collected,
        recorder.n_lost.value,
        errors.texts().size(),
        recorder.n_unmatched.value,
        recorder.n_suppressed.value,
        std::string(recorder.stage.value),
    };
}

template<typename TValue>
[[nodiscard]] std::optional<TValue> metaOf(const gr::DataSet<float>& set, std::string_view key) {
    if (set.meta_information.empty()) {
        return std::nullopt;
    }
    const auto found = gr::property_map_view{set.meta_information[0]}.get_if<TValue>(key);
    return found ? std::optional<TValue>{TValue(*found)} : std::nullopt;
}

/** the recorder driven one work call at a time, so the stage it reports after each sample can be read */
struct StepRig {
    MultiChannelRecorder<float>    recorder;
    gr::PortOut<float>             source;
    gr::EventPortOut               events;
    gr::PortIn<gr::DataSet<float>> sink;

    explicit StepRig(gr::property_map settings) : recorder(std::move(settings)) {
        recorder.init(recorder.progress);
        boost::ut::expect(source.connect(recorder.in[0]).has_value()) << boost::ut::fatal;
        boost::ut::expect(events.connect(recorder.evtIn).has_value()) << boost::ut::fatal;
        boost::ut::expect(recorder.out.connect(sink).has_value()) << boost::ut::fatal;
        boost::ut::expect(recorder.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << boost::ut::fatal;
    }

    ~StepRig() { // stopped as a scheduler would, while the recorder's members still live: ~Block would call stop() after they are gone
        std::ignore = recorder.changeStateTo(gr::lifecycle::State::REQUESTED_STOP);
        std::ignore = recorder.changeStateTo(gr::lifecycle::State::STOPPED);
    }

    void publishEvent(const gr::property_map& event) {
        auto span = events.reserve<gr::SpanReleasePolicy::ProcessNone>(1UZ);
        boost::ut::expect(gr::emit(span, 0UZ, gr::property_map_view{event}).has_value()) << boost::ut::fatal;
        span.publish(1UZ);
    }

    /** one sample per work call, the first carrying the dated tag; the distinct stages in the order they were reported */
    [[nodiscard]] std::vector<std::string> walk(std::size_t nSamples) {
        std::vector<std::string> stages;
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            if (i == 0UZ) {
                const gr::property_map anchor{
                    {std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")},
                    {std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart},
                    {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},
                    {std::string("sample_rate"), kSampleRate},
                };
                source.publishTag(gr::property_map_view{anchor}, 0UZ);
            }
            auto samples = source.streamWriter().reserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            samples[0]   = static_cast<float>(i);
            samples.publish(1UZ);
            std::ignore = recorder.work(1UZ);
            if (stages.empty() || stages.back() != std::string_view{recorder.stage.value}) {
                stages.emplace_back(recorder.stage.value);
            }
        }
        return stages;
    }
};

[[nodiscard]] gr::property_map delayedDecision(std::string name, std::uint64_t decisionAt, std::uint64_t startAt) {
    gr::property_map decision                               = eventNamed(std::move(name), decisionAt);
    decision[std::string(gr::tag::TRIGGER_META_INFO.key())] = gr::property_map{
        {std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")},
        {std::string(gr::tag::TRIGGER_TIME.key()), startAt},
        {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},
    };
    return decision;
}

/** the stream's own sample index as its value, 'a' = 0 .. 't' = 19 */
[[nodiscard]] gr::property_map indexValues() {
    gr::property_map values;
    for (int i = 0; i < 20; ++i) {
        values[std::string(1UZ, static_cast<char>('a' + i))] = static_cast<float>(i);
    }
    return values;
}

[[nodiscard]] std::vector<float> firstSamplesOf(const Recorded& found) {
    std::vector<float> firsts;
    for (const auto& set : found.sets) {
        firsts.push_back(set.signal_values.empty() ? -1.f : set.signal_values.front());
    }
    return firsts;
}

const boost::ut::suite<"MultiChannelRecorder"> _multiChannelRecorder = [] {
    using namespace boost::ut;

    const gr::property_map lowValues{{"a", 1.f}, {"b", 2.f}, {"c", 3.f}, {"d", 4.f}, {"e", 5.f}, {"f", 6.f}, {"g", 7.f}};
    const gr::property_map highValues{{"a", 11.f}, {"b", 12.f}, {"c", 13.f}, {"d", 14.f}, {"e", 15.f}, {"f", 16.f}, {"g", 17.f}};

    "one decision extracts the same window from every channel"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     low    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f g |")}, {"sample_values", lowValues}, {"sample_tags", tagsFor()}});
        auto&                     high   = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f g |")}, {"sample_values", highValues}, {"sample_tags", tagsFor()}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("coincidence", kStreamStart)};
        script._stopWhenDone             = false;
        auto& recorder                   = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", 2U}, {"n_pre", 2U}, {"n_post", 2U}, {"sample_rate", kSampleRate}});
        auto& sets                       = fixture.emplace<CollectingSink<gr::DataSet<float>>>();

        expect(fixture.graph.connect(low, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(high, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#1")}).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(eq(recorder.n_recorded.value, 1U)) << "one decision, one segment";
        expect(eq(sets._collected.size(), 1UZ));
        if (sets._collected.size() == 1UZ) {
            const gr::DataSet<float>& set = sets._collected.front();
            expect(eq(set.signal_names.size(), 2UZ)) << "both channels in one set, so their extents match by construction";
            expect(eq(static_cast<std::size_t>(set.extents.at(0)), 4UZ)) << "two before the anchor, two from it on: [anchor - n_pre, anchor + n_post)";
            expect(eq(set.signal_values.size(), 8UZ));
            if (set.signal_values.size() == 8UZ) {
                expect(eq(set.signal_values[0], 1.f)) << "the window reaches back before the trigger";
                expect(eq(set.signal_values[2], 3.f)) << "and the trigger's own sample sits in the middle";
                expect(eq(set.signal_values[4], 11.f)) << "the second channel follows, same window";
                expect(eq(set.signal_values[6], 13.f));
            }

            gr::testing::MarbleDiagram diagram{"MultiChannelRecorder: one decision, one window from each channel"};
            diagram.row("ch0 tag").at(kStreamStart, "start");
            diagram.row("ch1 tag").at(kStreamStart, "start");
            diagram.row("decision").at(kStreamStart, "coincidence");
            diagram.row("segment").at(static_cast<std::uint64_t>(set.timestamp), "DataSet");
            diagram.print();

            std::vector<double> x(static_cast<std::size_t>(set.extents.at(0)));
            std::vector<double> ch0(x.size());
            std::vector<double> ch1(x.size());
            for (std::size_t i = 0UZ; i < x.size(); ++i) {
                x[i]   = static_cast<double>(set.axis_values[0][i]);
                ch0[i] = static_cast<double>(set.signal_values[i]);
                ch1[i] = static_cast<double>(set.signal_values[x.size() + i]);
            }
            auto chart = gr::graphs::ImChart<80, 16>({{x.front(), x.back()}, {0., 20.}});
            chart.draw(x, ch0, "ch0");
            chart.draw(x, ch1, "ch1");
            chart.draw();
        }
    };

    "a decision that carries no time cannot place a window"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     low    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e |")}, {"sample_values", lowValues}, {"sample_tags", tagsFor()}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("coincidence")}}};
        script._stopWhenDone             = false;
        auto& recorder                   = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", 1U}, {"n_pre", 1U}, {"n_post", 1U}, {"sample_rate", kSampleRate}});
        auto& sets                       = fixture.emplace<CollectingSink<gr::DataSet<float>>>();

        expect(fixture.graph.connect(low, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());

        expect(eq(recorder.n_recorded.value, 0U));
        expect(eq(recorder.n_undated.value, 1U)) << "a window has to be placed somewhere, and nothing said where";
        expect(eq(sent.countContaining("undated"), 1UZ)) << "and it is reported as an error message";
    };

    "pre and post in nanoseconds are counted in samples at the stream's rate"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("decision", kStreamStart)}, {{"trigger_pre", std::uint64_t{2'000'000U}}, {"trigger_post", std::uint64_t{1'000'000U}}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        const gr::DataSet<float>& set = found.sets.front();
        expect(std::ranges::equal(set.signal_values, std::vector{1.f, 2.f, 3.f})) << "2 ms before at 1 kHz, then 1 ms from the anchor on";
        expect(eq(metaOf<std::uint64_t>(set, gr::tag::TRIGGER_PRE.key()).value_or(0U), std::uint64_t{2'000'000U}));
        expect(eq(metaOf<std::uint64_t>(set, gr::tag::TRIGGER_TIME.key()).value_or(0U), kStreamStart));
        expect(eq(set.axis_values[0][2], 0.f)) << "the time axis is zero at the window's start";
    };

    "the event's own pre, post and stop time win over the settings"_test = [&] {
        gr::property_map decision                               = eventNamed("window", kStreamStart);
        decision[std::string(gr::tag::TRIGGER_STOP_TIME.key())] = kStreamStart + 2'000'000U;
        decision[std::string(gr::tag::TRIGGER_PRE.key())]       = std::uint64_t{1'000'000U};
        decision[std::string(gr::tag::TRIGGER_POST.key())]      = std::uint64_t{1'000'000U};
        const auto found                                        = record("a b T:c d e f g |", lowValues, {std::move(decision)}, {{"trigger_pre", std::uint64_t{2'000'000U}}, {"n_post", 3U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        const gr::DataSet<float>& set = found.sets.front();
        expect(std::ranges::equal(set.signal_values, std::vector{2.f, 3.f, 4.f, 5.f})) << "1 ms before the start to 1 ms from the stop on";
        expect(eq(metaOf<std::uint64_t>(set, gr::tag::TRIGGER_STOP_TIME.key()).value_or(0U), kStreamStart + 2'000'000U));
    };

    "a window cut at the stream's start is flagged, or dropped and reported when asked"_test = [&] {
        const gr::property_map early   = eventNamed("decision", kStreamStart);
        const auto             flagged = record("T:a b c d |", lowValues, {early}, {{"n_pre", 2U}, {"n_post", 1U}});
        expect(eq(flagged.sets.size(), 1UZ)) << fatal;
        expect(eq(flagged.sets.front().signal_values.size(), 1UZ)) << "only what the stream had, never padding";
        const auto meta       = metaOf<gr::property_map>(flagged.sets.front(), gr::tag::TRIGGER_META_INFO.key());
        bool       incomplete = false;
        if (meta) {
            if (const auto flags = gr::property_map_view{*meta}.get_if<gr::property_map>(std::string_view{gr::tag::ACQUISITION_FLAGS.key()})) {
                incomplete = gr::property_map_view{*flags}.contains("incomplete");
            }
        }
        expect(incomplete) << "and it says so";

        const auto dropped = record("T:a b c d |", lowValues, {early}, {{"n_pre", 2U}, {"n_post", 1U}, {"emit_partial", false}});
        expect(dropped.sets.empty());
        expect(eq(dropped.lost, 1U));
        expect(eq(dropped.errors, 1UZ)) << "a dropped window is reported as an error message";
    };

    "an open event and the next close event bound the window, oldest first"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("open", kStreamStart), eventNamed("close", kStreamStart + 2'000'000U)}, //
            {{"open_filter", std::string("open")}, {"close_filter", std::string("close")}, {"n_pre", 1U}, {"n_post", 1U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values, std::vector{2.f, 3.f, 4.f, 5.f})) << "one before the open, one from the close on";
        expect(eq(found.unmatched, 0U));
    };

    "a close event's own post wins over the one its window opened with"_test = [&] {
        gr::property_map close                          = eventNamed("close", kStreamStart + 2'000'000U);
        close[std::string(gr::tag::TRIGGER_POST.key())] = std::uint64_t{2'000'000U};
        const auto found                                = record("a b T:c d e f g |", lowValues, {eventNamed("open", kStreamStart), std::move(close)}, //
                                           {{"open_filter", std::string("open")}, {"close_filter", std::string("close")}, {"n_pre", 1U}, {"n_post", 1U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values, std::vector{2.f, 3.f, 4.f, 5.f, 6.f})) << "one before the open, two from the close on";
    };

    "a rate the stream's tags supply leaves what is being recorded intact"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("decision", kStreamStart)}, {{"trigger_pre", std::uint64_t{2'000'000U}}, {"trigger_post", std::uint64_t{1'000'000U}}}, true);
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values,
            std::vector{
                1.f,
                2.f,
                3.f,
            }))
            << "the same window as with the rate set, though the rate arrived with the trigger";
        expect(eq(found.lost, 0U));
    };

    "a close with no open window is ignored and counted"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("close", kStreamStart)}, {{"open_filter", std::string("open")}, {"close_filter", std::string("close")}});
        expect(found.sets.empty());
        expect(eq(found.unmatched, 1U));
        expect(eq(found.errors, 0UZ)) << "a lone close is not an error";
    };

    "a window never closed ends after max_duration, flagged truncated"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("open", kStreamStart)}, //
            {{"open_filter", std::string("open")}, {"close_filter", std::string("close")}, {"max_duration", 0.002f}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values, std::vector{3.f, 4.f})) << "2 ms at 1 kHz from the open";
        const auto meta = metaOf<gr::property_map>(found.sets.front(), gr::tag::TRIGGER_META_INFO.key());
        expect(meta.has_value() && gr::property_map_view{*meta}.contains(gr::tag::ACQUISITION_FLAGS.key())) << "and it says it was cut";
    };

    "retrigger decides what a second decision does to an open window"_test = [&] {
        const std::vector<gr::property_map> twice{eventNamed("decision", kStreamStart), eventNamed("decision", kStreamStart + 1'000'000U)};
        auto                                with = [&](std::string mode) { return record("a b T:c d e f g |", lowValues, twice, {{"n_pre", 0U}, {"n_post", 2U}, {"retrigger", std::move(mode)}}).sets; };
        expect(eq(with("parallel").size(), 2UZ)) << "both windows";
        const auto ignored = with("ignore");
        expect(ignored.size() == 1UZ && std::ranges::equal(ignored.front().signal_values, std::vector{3.f, 4.f})) << "the first window only";
        const auto restarted = with("restart");
        expect(restarted.size() == 1UZ && std::ranges::equal(restarted.front().signal_values, std::vector{4.f, 5.f})) << "the second replaces it";
        const auto extended = with("extend");
        expect(extended.size() == 1UZ && std::ranges::equal(extended.front().signal_values, std::vector{3.f, 4.f, 5.f})) << "the first, lengthened";
    };

    "a rate change stops an open window there and says so"_test = [&] {
        const auto found = record("a b T:c d R:e f g |", lowValues, {eventNamed("decision", kStreamStart)}, {{"n_pre", 0U}, {"n_post", 3U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values, std::vector{3.f, 4.f})) << "the samples before the change only";
        bool rateChanged = false;
        if (const auto meta = metaOf<gr::property_map>(found.sets.front(), gr::tag::TRIGGER_META_INFO.key())) {
            if (const auto flags = gr::property_map_view{*meta}.get_if<gr::property_map>(std::string_view{gr::tag::ACQUISITION_FLAGS.key()})) {
                rateChanged = gr::property_map_view{*flags}.contains("rate_changed");
            }
        }
        expect(rateChanged);
    };

    "a tag repeating the same rate after a jittery re-anchor leaves an open window alone"_test = [&] {
        const auto found = record("T:a U:b c d S:e f g |", lowValues, {eventNamed("decision", kStreamStart + 3'000'000U)}, {{"n_pre", 0U}, {"n_post", 3U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(eq(found.sets.front().signal_values.size(), 3UZ)) << "not truncated: the nominal rate did not change";
        expect(!metaOf<gr::property_map>(found.sets.front(), gr::tag::TRIGGER_META_INFO.key()).has_value()) << "and no flag is raised";
    };

    "n_segments stops the recorder after the count it was given"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     low    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f g |")}, {"sample_values", lowValues}, {"sample_tags", tagsFor()}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("coincidence", kStreamStart), eventNamed("coincidence", kStreamStart + 1'000'000U)};
        script._stopWhenDone             = false;
        auto& recorder                   = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", 1U}, {"n_pre", 1U}, {"n_post", 1U}, {"sample_rate", kSampleRate}, {"n_segments", 1U}});
        auto& sets                       = fixture.emplace<CollectingSink<gr::DataSet<float>>>();

        expect(fixture.graph.connect(low, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(eq(recorder.n_recorded.value, 1U)) << "two decisions, one segment asked for";
    };

    /** decisions 0, 1 and 2 ms after the anchor at sample 2 ('c'), so a window's time says which decision opened it */
    const auto three    = [] { return std::vector<gr::property_map>{eventNamed("start", kStreamStart), eventNamed("start", kStreamStart + 1'000'000U), eventNamed("start", kStreamStart + 2'000'000U)}; };
    const auto startsOf = [](const Recorded& found) {
        std::vector<std::uint64_t> starts;
        for (const auto& set : found.sets) {
            starts.push_back(static_cast<std::uint64_t>(set.timestamp));
        }
        return starts;
    };

    "a decision early in a long work call is recorded, however the stream is cut"_test = [&] {
        std::string script = "T:a ";
        for (int i = 0; i < 199; ++i) {
            script += "b ";
        }
        script += "|";
        const auto whole = record(script, lowValues, {eventNamed("start", kStreamStart + 2'000'000U)}, {{"n_pre", 1U}, {"n_post", 1U}});
        const auto cut   = record(script, lowValues, {eventNamed("start", kStreamStart + 2'000'000U)}, {{"n_pre", 1U}, {"n_post", 1U}}, false, 7UZ);
        expect(eq(whole.sets.size(), 1UZ)) << "200 samples in one call: the decision at sample 2 is still within reach";
        expect(eq(cut.sets.size(), 1UZ));
    };

    "auto's deadlines and real decisions are taken in the order of their samples, however the stream is cut"_test = [&] {
        // a deadline at sample 5 (5 ms after the start) and a decision at sample 7 give the same windows in one call or in many
        const std::vector<gr::property_map> decision{eventNamed("start", kStreamStart + 7'000'000U)};
        const gr::property_map              settings{{"n_pre", 0U}, {"n_post", 1U}, {"mode", std::string("auto")}, {"auto_timeout", 5e-3f}};
        const auto                          whole = record("T:a b c d e f g a b c d e f |", lowValues, decision, settings);
        const auto                          cut   = record("T:a b c d e f g a b c d e f |", lowValues, decision, settings, false, 3UZ);
        expect(startsOf(whole) == startsOf(cut));
        expect(!startsOf(whole).empty() && startsOf(whole).front() == kStreamStart + 5'000'000U) << "the deadline comes first";
    };

    "a decision a retrigger ignores is suppressed, and does not move the holdoff"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, three(), {{"n_pre", 0U}, {"n_post", 3U}, {"retrigger", std::string("ignore")}});
        expect(eq(found.sets.size(), 1UZ));
        expect(eq(found.suppressed, 2U)) << "the two decisions inside the open window";
    };

    "single records the first accepted decision and then stops"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, three(), {{"n_pre", 1U}, {"n_post", 1U}, {"mode", std::string("single")}});
        expect(startsOf(found) == std::vector<std::uint64_t>{kStreamStart});
        expect(eq(found.suppressed, 2U)) << "detection went on, acceptance did not";
        expect(eq(found.stage, std::string("stopped")));
    };

    "holdoff refers to the last accepted decision, accepts equality and is not extended by what it suppresses"_test = [&] {
        const auto extended = record("a b T:c d e f g |", lowValues, three(), {{"n_pre", 1U}, {"n_post", 1U}, {"holdoff", 1.5e-3f}});
        expect(startsOf(extended) == std::vector<std::uint64_t>{kStreamStart, kStreamStart + 2'000'000U}) << "1 ms is held off, 2 ms is 2 ms after the last accepted one";
        const auto equal = record("a b T:c d e f g |", lowValues, three(), {{"n_pre", 1U}, {"n_post", 1U}, {"holdoff", 2e-3f}});
        expect(startsOf(equal) == std::vector<std::uint64_t>{kStreamStart, kStreamStart + 2'000'000U}) << "exactly the holdoff later is accepted";
        expect(eq(equal.suppressed, 1U));
    };

    "a count holdoff suppresses the next n decisions after each accepted one"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, three(), {{"n_pre", 1U}, {"n_post", 1U}, {"holdoff_count", 1U}});
        expect(startsOf(found) == std::vector<std::uint64_t>{kStreamStart, kStreamStart + 2'000'000U});
        expect(eq(found.suppressed, 1U));
    };

    "a forced decision opens a window whatever the open filter says, flagged forced"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("other", kStreamStart), eventNamed("forced", kStreamStart + 1'000'000U)}, {{"n_pre", 1U}, {"n_post", 1U}, {"open_filter", std::string("start")}});
        expect(startsOf(found) == std::vector<std::uint64_t>{kStreamStart + 1'000'000U});
        if (!found.sets.empty()) {
            const auto meta  = metaOf<gr::property_map>(found.sets[0], gr::tag::TRIGGER_META_INFO.key());
            const auto flags = meta ? gr::property_map_view{*meta}.get_if<gr::property_map>(gr::tag::ACQUISITION_FLAGS.key()) : nullptr;
            expect(flags != nullptr && gr::property_map_view{*flags}.contains(std::string_view{"forced"})) << "a forced window is not taken for a match";
        }
    };

    "auto forces a window at each deadline without a decision, flagged auto_timeout"_test = [&] {
        // nothing to accept: deadlines every 2 ms from the stream's start, the first dated one at the anchor 'c'
        const auto found = record("a b T:c d e f g |", lowValues, {}, {{"n_pre", 1U}, {"n_post", 1U}, {"mode", std::string("auto")}, {"auto_timeout", 2e-3f}});
        expect(ge(found.sets.size(), 2UZ)) << fatal;
        expect(eq(static_cast<std::uint64_t>(found.sets[0].timestamp), kStreamStart)) << "sample 2, two samples after the start";
        expect(eq(static_cast<std::uint64_t>(found.sets[1].timestamp), kStreamStart + 2'000'000U));
        for (const auto& set : found.sets) {
            const auto meta  = metaOf<gr::property_map>(set, gr::tag::TRIGGER_META_INFO.key());
            const auto flags = meta ? gr::property_map_view{*meta}.get_if<gr::property_map>(gr::tag::ACQUISITION_FLAGS.key()) : nullptr;
            expect(flags != nullptr && gr::property_map_view{*flags}.contains(std::string_view{"auto_timeout"})) << "it is not evidence that anything matched";
        }
    };

    /** index time: sample i is dated kStreamStart + i ms; the script is the stream 'a'..'t' = values 0..19 anchored at sample 0 */
    const std::string ramp = "T:a b c d e f g h i j k l m n o p q r s t |";

    "the window holds the indices [anchor - n_pre, anchor + n_post) exactly"_test = [&] {
        const auto found = record(ramp, indexValues(), {eventNamed("decision", kStreamStart + 10'000'000U)}, {{"n_pre", 3U}, {"n_post", 4U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values,
            std::vector{
                7.f,
                8.f,
                9.f,
                10.f,
                11.f,
                12.f,
                13.f,
            }))
            << "indices 7..13 around the anchor 10";
        expect(eq(metaOf<gr::Size_t>(found.sets.front(), "n_pre").value_or(99U), 3U));
        expect(eq(metaOf<gr::Size_t>(found.sets.front(), "n_post").value_or(99U), 4U));
    };

    "a window reaching before the first sample is flagged incomplete and holds only real samples"_test = [&] {
        const auto found = record(ramp, indexValues(), {eventNamed("decision", kStreamStart + 1'000'000U)}, {{"n_pre", 3U}, {"n_post", 4U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values,
            std::vector{
                0.f,
                1.f,
                2.f,
                3.f,
                4.f,
            }))
            << "the two pre samples that never existed are absent, not zero";
        const auto meta  = metaOf<gr::property_map>(found.sets.front(), gr::tag::TRIGGER_META_INFO.key());
        const auto flags = meta ? gr::property_map_view{*meta}.get_if<gr::property_map>(gr::tag::ACQUISITION_FLAGS.key()) : nullptr;
        expect(flags != nullptr && gr::property_map_view{*flags}.contains(std::string_view{"incomplete"}));
    };

    "reference = start anchors the window at the start a delayed decision refers to"_test = [&] {
        const auto pulse    = delayedDecision("width", kStreamStart + 6'000'000U, kStreamStart + 2'000'000U);
        const auto byStart  = record(ramp, indexValues(), {pulse}, {{"n_pre", 1U}, {"n_post", 2U}, {"reference", std::string("start")}});
        const auto byDecide = record(ramp, indexValues(), {pulse}, {{"n_pre", 1U}, {"n_post", 2U}});
        expect(firstSamplesOf(byStart) == std::vector{1.f}) << "start at sample 2: [1, 4)";
        expect(firstSamplesOf(byDecide) == std::vector{5.f}) << "decision at sample 6: [5, 8)";
        expect(eq(static_cast<std::uint64_t>(byStart.sets.at(0).timestamp), kStreamStart + 2'000'000U));
    };

    "reference = start falls back to the decision time when the decision carries no start tuple"_test = [&] {
        const auto found = record(ramp, indexValues(), {eventNamed("width", kStreamStart + 6'000'000U)},
            {
                {"n_pre", 1U},
                {"n_post", 2U},
                {"reference", std::string("start")},
            });
        expect(firstSamplesOf(found) == std::vector{5.f});
    };

    "holdoff counts from the start times when reference = start"_test = [&] {
        const std::vector<gr::property_map> decisions{delayedDecision("width", kStreamStart + 6'000'000U, kStreamStart + 2'000'000U), delayedDecision("width", kStreamStart + 7'000'000U, kStreamStart + 5'000'000U)};
        const auto                          byStart  = record(ramp, indexValues(), decisions,
                                      {
                {"n_pre", 0U},
                {"n_post", 1U},
                {"holdoff", 2.5e-3f},
                {"reference", std::string("start")},
            });
        const auto                          byDecide = record(ramp, indexValues(), decisions, {{"n_pre", 0U}, {"n_post", 1U}, {"holdoff", 2.5e-3f}});
        expect(firstSamplesOf(byStart) == std::vector{2.f, 5.f}) << "the starts are 3 ms apart: both accepted";
        expect(firstSamplesOf(byDecide) == std::vector{6.f}) << "the decisions are 1 ms apart: the second is held off";
    };

    "single takes the first accepted decision and anchors it at its start when reference = start"_test = [&] {
        const std::vector<gr::property_map> decisions{delayedDecision("width", kStreamStart + 6'000'000U, kStreamStart + 5'000'000U), delayedDecision("width", kStreamStart + 8'000'000U, kStreamStart + 2'000'000U)};
        const auto                          found = record(ramp, indexValues(), decisions,
                                     {
                {"n_pre", 0U},
                {"n_post", 1U},
                {"mode", std::string("single")},
                {"reference", std::string("start")},
            });
        expect(firstSamplesOf(found) == std::vector{5.f}) << "the decision at 6 ms is the first accepted, and its start is sample 5";
    };

    "an arming event is required before a window opens, and opens nothing itself"_test = [&] {
        const std::vector<gr::property_map> events{
            eventNamed("start", kStreamStart + 2'000'000U),
            eventNamed("arm", kStreamStart + 3'000'000U),
            eventNamed("start", kStreamStart + 5'000'000U),
            eventNamed("start", kStreamStart + 7'000'000U),
        };
        const auto autoRearm   = record(ramp, indexValues(), events,
              {
                {"n_pre", 0U},
                {"n_post", 1U},
                {"open_filter", std::string("start")},
                {"arm_filter", std::string("arm")},
            });
        const auto manualRearm = record(ramp, indexValues(), events,
            {
                {"n_pre", 0U},
                {"n_post", 1U},
                {"open_filter", std::string("start")},
                {"arm_filter", std::string("arm")},
                {"rearm", std::string("manual")},
            });
        expect(firstSamplesOf(autoRearm) == std::vector{5.f, 7.f}) << "once armed, every decision may open a window";
        expect(eq(autoRearm.suppressed, 1U)) << "the start before the arming";
        expect(firstSamplesOf(manualRearm) == std::vector{5.f}) << "the arming is spent on one window";
        expect(eq(manualRearm.suppressed, 2U));
    };

    "manual rearming re-opens a stopped single recorder, auto does not"_test = [&] {
        const std::vector<gr::property_map> events{
            eventNamed("arm", kStreamStart),
            eventNamed("start", kStreamStart + 2'000'000U),
            eventNamed("arm", kStreamStart + 4'000'000U),
            eventNamed("start", kStreamStart + 6'000'000U),
        };
        const gr::property_map base{
            {"n_pre", 0U},
            {"n_post", 1U},
            {"open_filter", std::string("start")},
            {"arm_filter", std::string("arm")},
            {"mode", std::string("single")},
        };
        gr::property_map manual      = base;
        manual[std::string("rearm")] = std::string("manual");
        expect(firstSamplesOf(record(ramp, indexValues(), events, manual)) == std::vector{2.f, 6.f}) << "the second arming allows a second single window";
        expect(firstSamplesOf(record(ramp, indexValues(), events, base)) == std::vector{2.f}) << "auto: a single recorder stays stopped until reset";
    };

    "an unknown rearm or reference is refused and the previous kept"_test = [&] {
        MultiChannelRecorder<float> recorder({{"n_inputs", 1U}, {"rearm", std::string("manual")}, {"reference", std::string("start")}});
        recorder.settings().init();
        std::ignore = recorder.settings().applyStagedParameters();
        expect(eq(std::string(recorder.rearm.value), std::string("manual"))) << fatal;
        expect(recorder.settings().set({{"rearm", std::string("sometimes")}, {"reference", std::string("middle")}}).empty());
        std::ignore = recorder.settings().activateContext();
        std::ignore = recorder.settings().applyStagedParameters();
        expect(eq(std::string(recorder.rearm.value), std::string("manual")));
        expect(eq(std::string(recorder.reference.value), std::string("start")));
    };

    "the stage walks priming, armed, awaiting samples and holdoff as the stream passes"_test = [&] {
        StepRig rig({{"n_inputs", 1U}, {"sample_rate", kSampleRate}, {"n_pre", 2U}, {"n_post", 2U}, {"holdoff_count", 1U}});
        rig.publishEvent(eventNamed("decision", kStreamStart + 3'000'000U));
        const std::vector<std::string> expected{"priming", "armed", "awaiting_samples", "holdoff"};
        expect(rig.walk(9UZ) == expected);
        expect(eq(rig.recorder.n_recorded.value, 1U));
    };

    "the stage of a single recorder with an arming filter walks unarmed, armed, awaiting samples, stopped"_test = [&] {
        StepRig rig({
            {"n_inputs", 1U},
            {"sample_rate", kSampleRate},
            {"n_pre", 0U},
            {"n_post", 2U},
            {"mode", std::string("single")},
            {"arm_filter", std::string("arm")},
        });
        rig.publishEvent(eventNamed("arm", kStreamStart + 1'000'000U));
        rig.publishEvent(eventNamed("decision", kStreamStart + 3'000'000U));
        const std::vector<std::string> expected{"unarmed", "armed", "awaiting_samples", "stopped"};
        expect(rig.walk(9UZ) == expected);
    };

    "windows beyond max_open are lost and counted exactly"_test = [&] {
        const std::vector<gr::property_map> events{
            eventNamed("d", kStreamStart + 2'000'000U),
            eventNamed("d", kStreamStart + 3'000'000U),
            eventNamed("d", kStreamStart + 4'000'000U),
            eventNamed("d", kStreamStart + 5'000'000U),
        };
        const auto found = record(ramp, indexValues(), events, {{"n_pre", 0U}, {"n_post", 6U}, {"max_open", 2U}}, false, 1UZ);
        expect(eq(found.sets.size(), 2UZ)) << "two fit";
        expect(eq(found.lost, 2U)) << "and the other two are counted, one each";
        expect(ge(found.errors, 1UZ)) << "and reported as an error message";
    };

    "partial windows dropped under emit_partial = false are counted one each"_test = [&] {
        const std::vector<gr::property_map> events{
            eventNamed("d", kStreamStart),
            eventNamed("d", kStreamStart + 1'000'000U),
            eventNamed("d", kStreamStart + 10'000'000U),
        };
        const auto found = record(ramp, indexValues(), events, {{"n_pre", 3U}, {"n_post", 1U}, {"emit_partial", false}});
        expect(eq(found.sets.size(), 1UZ)) << "only the complete window";
        expect(eq(found.lost, 2U));
    };

    "closed windows the output had no room for at the end of the stream are lost and counted"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", ramp}, {"sample_values", indexValues()}, {"sample_tags", tagsFor()}});
        auto&                     events = fixture.emplace<EventScript>();
        for (std::uint64_t ms = 2U; ms <= 18U; ms += 2U) {
            events._events.push_back(eventNamed("d", kStreamStart + ms * 1'000'000U));
        }
        events._stopWhenDone = false;
        auto& recorder       = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", 1U}, {"n_pre", 0U}, {"n_post", 1U}, {"sample_rate", kSampleRate}});
        auto& sets           = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.graph.connect(source, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(events, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}, gr::EdgeParameters{.minBufferSize = 2UZ}).has_value());
        expect(fixture.run().has_value());
        expect(eq(sets._collected.size(), 2UZ)) << "the output holds two";
        expect(eq(recorder.n_lost.value, 7U)) << "and the other seven of the nine are counted, none vanishes silently";
    };

    "a consumer that stops reading makes windows wait in the recorder, and none is lost"_test = [&] {
        StepRig                rig({{"n_inputs", 1U}, {"sample_rate", kSampleRate}, {"n_pre", 0U}, {"n_post", 1U}, {"max_open", 1U}});
        const gr::property_map anchor{
            {std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")},
            {std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart},
            {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},
            {std::string("sample_rate"), kSampleRate},
        };
        std::size_t published = 0UZ;
        const auto  step      = [&](bool withDecision) {
            if (published == 0UZ) {
                rig.source.publishTag(gr::property_map_view{anchor}, 0UZ);
            }
            if (withDecision) {
                rig.publishEvent(eventNamed("d", kStreamStart + static_cast<std::uint64_t>(published) * 1'000'000U));
            }
            auto samples = rig.source.streamWriter().reserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            samples[0]   = static_cast<float>(published);
            samples.publish(1UZ);
            ++published;
            std::ignore = rig.recorder.work(1UZ);
        };
        const std::size_t capacity   = rig.sink.streamReader().buffer().size();
        const std::size_t nDecisions = capacity + capacity / 2UZ;
        for (std::size_t i = 0UZ; i < nDecisions; ++i) {
            step(true);
        }
        expect(eq(rig.sink.streamReader().available(), capacity)) << "the unread output port is full";
        expect(eq(rig.recorder.n_lost.value, 0U)) << "and the windows beyond it wait instead of being dropped";

        std::size_t delivered = 0UZ;
        for (std::size_t round = 0UZ; round < 8UZ && delivered < nDecisions; ++round) {
            delivered += rig.sink.streamReader().get<gr::SpanReleasePolicy::ProcessAll>(rig.sink.streamReader().available()).size();
            step(false);
        }
        delivered += rig.sink.streamReader().get<gr::SpanReleasePolicy::ProcessAll>(rig.sink.streamReader().available()).size();
        expect(eq(delivered, nDecisions)) << "once the consumer reads again, every decision's window arrives";
        expect(eq(rig.recorder.n_lost.value, 0U));
    };
};
} // namespace qa_multi_channel_recorder

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
