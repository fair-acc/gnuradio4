#include <boost/ut.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Marble.hpp"
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/MultiChannelRecorder.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::CollectingSink;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;

namespace {
constexpr std::uint64_t kTriggerTime = 1'000'000'000U;
constexpr float         kSampleRate  = 1000.f;

[[nodiscard]] gr::property_map tagsFor() {
    return gr::property_map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, //
                                      {std::string(gr::tag::TRIGGER_TIME.key()), kTriggerTime},                      //
                                      {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},                             //
                                      {std::string("sample_rate"), kSampleRate}}},
        {"R", gr::property_map{{std::string("sample_rate"), 2.f * kSampleRate}}}, //
        {"U", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("resync")}, {std::string(gr::tag::TRIGGER_TIME.key()), kTriggerTime + 2'000'000U}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}, {std::string("sample_rate"), kSampleRate}}}, {"S", gr::property_map{{std::string("sample_rate"), kSampleRate}}}};
}

struct Recorded {
    std::vector<gr::DataSet<float>> sets;
    gr::Size_t                      lost;
    std::size_t                     errors;
    gr::Size_t                      unmatched;
};

/** one channel, its decisions, and whatever the recorder makes of them */
[[nodiscard]] Recorded record(const std::string& script, const gr::property_map& values, std::vector<gr::property_map> decisions, gr::property_map settings, bool rateFromTags = false) {
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
    auto& sets     = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
    auto& errors   = fixture.emplace<gr::trigger_test::EventTap>();
    boost::ut::expect(fixture.graph.connect(source, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
    boost::ut::expect(fixture.graph.connect(events, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
    boost::ut::expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("evtOut")}, errors, gr::PortDefinition{std::string("evtIn")}).has_value());
    boost::ut::expect(fixture.run().has_value());
    return {sets._collected, recorder.n_lost.value, errors.size(), recorder.n_unmatched.value};
}

template<typename TValue>
[[nodiscard]] std::optional<TValue> metaOf(const gr::DataSet<float>& set, std::string_view key) {
    if (set.meta_information.empty()) {
        return std::nullopt;
    }
    const auto found = gr::property_map_view{set.meta_information[0]}.get_if<TValue>(key);
    return found ? std::optional<TValue>{TValue(*found)} : std::nullopt;
}
} // namespace

const boost::ut::suite<"MultiChannelRecorder"> _recorder = [] {
    using namespace boost::ut;

    const gr::property_map lowValues{{"a", 1.f}, {"b", 2.f}, {"c", 3.f}, {"d", 4.f}, {"e", 5.f}, {"f", 6.f}, {"g", 7.f}};
    const gr::property_map highValues{{"a", 11.f}, {"b", 12.f}, {"c", 13.f}, {"d", 14.f}, {"e", 15.f}, {"f", 16.f}, {"g", 17.f}};

    "one decision extracts the same window from every channel"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     low    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f g |")}, {"sample_values", lowValues}, {"sample_tags", tagsFor()}});
        auto&                     high   = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f g |")}, {"sample_values", highValues}, {"sample_tags", tagsFor()}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("coincidence", kTriggerTime)};
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
            expect(eq(static_cast<std::size_t>(set.extents.at(0)), 5UZ)) << "two before the decision, the decision, two after";
            expect(eq(set.signal_values.size(), 10UZ));
            if (set.signal_values.size() == 10UZ) {
                expect(eq(set.signal_values[0], 1.f)) << "the window reaches back before the trigger";
                expect(eq(set.signal_values[2], 3.f)) << "and the trigger's own sample sits in the middle";
                expect(eq(set.signal_values[5], 11.f)) << "the second channel follows, same window";
                expect(eq(set.signal_values[7], 13.f));
            }

            gr::testing::MarbleDiagram diagram{"MultiChannelRecorder: one decision, one window from each channel"};
            diagram.row("ch0 tag").at(kTriggerTime, "start");
            diagram.row("ch1 tag").at(kTriggerTime, "start");
            diagram.row("decision").at(kTriggerTime, "coincidence");
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
        expect(fixture.run().has_value());

        expect(eq(recorder.n_recorded.value, 0U));
        expect(eq(recorder.n_undated.value, 1U)) << "a window has to be placed somewhere, and nothing said where";
    };

    "pre and post in nanoseconds are counted in samples at the stream's rate"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("decision", kTriggerTime)}, {{"trigger_pre", std::uint64_t{2'000'000U}}, {"trigger_post", std::uint64_t{1'000'000U}}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        const gr::DataSet<float>& set = found.sets.front();
        expect(std::ranges::equal(set.signal_values, std::vector{1.f, 2.f, 3.f, 4.f})) << "2 ms before at 1 kHz, the decision, 1 ms after";
        expect(eq(metaOf<std::uint64_t>(set, gr::tag::TRIGGER_PRE.key()).value_or(0U), std::uint64_t{2'000'000U}));
        expect(eq(metaOf<std::uint64_t>(set, gr::tag::TRIGGER_TIME.key()).value_or(0U), kTriggerTime));
        expect(eq(set.axis_values[0][2], 0.f)) << "the time axis is zero at the window's start";
    };

    "the event's own pre, post and stop time win over the settings"_test = [&] {
        gr::property_map decision                               = eventNamed("window", kTriggerTime);
        decision[std::string(gr::tag::TRIGGER_STOP_TIME.key())] = kTriggerTime + 2'000'000U;
        decision[std::string(gr::tag::TRIGGER_PRE.key())]       = std::uint64_t{1'000'000U};
        decision[std::string(gr::tag::TRIGGER_POST.key())]      = std::uint64_t{1'000'000U};
        const auto found                                        = record("a b T:c d e f g |", lowValues, {std::move(decision)}, {{"trigger_pre", std::uint64_t{2'000'000U}}, {"n_post", 3U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        const gr::DataSet<float>& set = found.sets.front();
        expect(std::ranges::equal(set.signal_values, std::vector{2.f, 3.f, 4.f, 5.f, 6.f})) << "1 ms before the start to 1 ms after the stop";
        expect(eq(metaOf<std::uint64_t>(set, gr::tag::TRIGGER_STOP_TIME.key()).value_or(0U), kTriggerTime + 2'000'000U));
    };

    "a window cut at the stream's start is flagged, or dropped and reported when asked"_test = [&] {
        const gr::property_map early   = eventNamed("decision", kTriggerTime);
        const auto             flagged = record("T:a b c d |", lowValues, {early}, {{"n_pre", 2U}, {"n_post", 1U}});
        expect(eq(flagged.sets.size(), 1UZ)) << fatal;
        expect(eq(flagged.sets.front().signal_values.size(), 2UZ)) << "only what the stream had, never padding";
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
        expect(eq(dropped.errors, 1UZ)) << "a dropped window is reported on evtOut";
    };

    "an open event and the next close event bound the window, oldest first"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("open", kTriggerTime), eventNamed("close", kTriggerTime + 2'000'000U)}, //
            {{"open_filter", std::string("open")}, {"close_filter", std::string("close")}, {"n_pre", 1U}, {"n_post", 1U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values, std::vector{2.f, 3.f, 4.f, 5.f, 6.f})) << "one before the open, one after the close";
        expect(eq(found.unmatched, 0U));
    };

    "a close event's own post wins over the one its window opened with"_test = [&] {
        gr::property_map close                          = eventNamed("close", kTriggerTime + 2'000'000U);
        close[std::string(gr::tag::TRIGGER_POST.key())] = std::uint64_t{2'000'000U};
        const auto found                                = record("a b T:c d e f g |", lowValues, {eventNamed("open", kTriggerTime), std::move(close)}, //
                                           {{"open_filter", std::string("open")}, {"close_filter", std::string("close")}, {"n_pre", 1U}, {"n_post", 1U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values, std::vector{2.f, 3.f, 4.f, 5.f, 6.f, 7.f})) << "one before the open, two after the close";
    };

    "a rate the stream's tags supply leaves what is being recorded intact"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("decision", kTriggerTime)}, {{"trigger_pre", std::uint64_t{2'000'000U}}, {"trigger_post", std::uint64_t{1'000'000U}}}, true);
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values, std::vector{1.f, 2.f, 3.f, 4.f})) << "the same window as with the rate set, though the rate arrived with the trigger";
        expect(eq(found.lost, 0U));
    };

    "a close with no open window is ignored and counted"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("close", kTriggerTime)}, {{"open_filter", std::string("open")}, {"close_filter", std::string("close")}});
        expect(found.sets.empty());
        expect(eq(found.unmatched, 1U));
        expect(eq(found.errors, 0UZ)) << "a lone close is not an error";
    };

    "a window never closed ends after max_duration, flagged truncated"_test = [&] {
        const auto found = record("a b T:c d e f g |", lowValues, {eventNamed("open", kTriggerTime)}, //
            {{"open_filter", std::string("open")}, {"close_filter", std::string("close")}, {"max_duration", 0.002f}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found.sets.front().signal_values, std::vector{3.f, 4.f, 5.f})) << "2 ms at 1 kHz past the open";
        const auto meta = metaOf<gr::property_map>(found.sets.front(), gr::tag::TRIGGER_META_INFO.key());
        expect(meta.has_value() && gr::property_map_view{*meta}.contains(gr::tag::ACQUISITION_FLAGS.key())) << "and it says it was cut";
    };

    "retrigger decides what a second decision does to an open window"_test = [&] {
        const std::vector<gr::property_map> twice{eventNamed("decision", kTriggerTime), eventNamed("decision", kTriggerTime + 1'000'000U)};
        auto                                with = [&](std::string mode) { return record("a b T:c d e f g |", lowValues, twice, {{"n_pre", 0U}, {"n_post", 2U}, {"retrigger", std::move(mode)}}).sets; };
        expect(eq(with("parallel").size(), 2UZ)) << "both windows";
        const auto ignored = with("ignore");
        expect(ignored.size() == 1UZ && std::ranges::equal(ignored.front().signal_values, std::vector{3.f, 4.f, 5.f})) << "the first window only";
        const auto restarted = with("restart");
        expect(restarted.size() == 1UZ && std::ranges::equal(restarted.front().signal_values, std::vector{4.f, 5.f, 6.f})) << "the second replaces it";
        const auto extended = with("extend");
        expect(extended.size() == 1UZ && std::ranges::equal(extended.front().signal_values, std::vector{3.f, 4.f, 5.f, 6.f})) << "the first, lengthened";
    };

    "a rate change stops an open window there and says so"_test = [&] {
        const auto found = record("a b T:c d R:e f g |", lowValues, {eventNamed("decision", kTriggerTime)}, {{"n_pre", 0U}, {"n_post", 3U}});
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
        const auto found = record("T:a U:b c d S:e f g |", lowValues, {eventNamed("decision", kTriggerTime + 3'000'000U)}, {{"n_pre", 0U}, {"n_post", 3U}});
        expect(eq(found.sets.size(), 1UZ)) << fatal;
        expect(eq(found.sets.front().signal_values.size(), 4UZ)) << "not truncated: the nominal rate did not change";
        expect(!metaOf<gr::property_map>(found.sets.front(), gr::tag::TRIGGER_META_INFO.key()).has_value()) << "and no flag is raised";
    };

    "n_segments stops the recorder after the count it was given"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     low    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f g |")}, {"sample_values", lowValues}, {"sample_tags", tagsFor()}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("coincidence", kTriggerTime), eventNamed("coincidence", kTriggerTime + 1'000'000U)};
        script._stopWhenDone             = false;
        auto& recorder                   = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", 1U}, {"n_pre", 1U}, {"n_post", 1U}, {"sample_rate", kSampleRate}, {"n_segments", 1U}});
        auto& sets                       = fixture.emplace<CollectingSink<gr::DataSet<float>>>();

        expect(fixture.graph.connect(low, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(eq(recorder.n_recorded.value, 1U)) << "two decisions, one segment asked for";
    };
};

int main() { /* tests are statically executed */ }
