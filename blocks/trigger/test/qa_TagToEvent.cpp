#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>

namespace {
using namespace boost::ut;
using gr::blocks::trigger::matchesGlob;
using gr::blocks::trigger::TagToEvent;

using Source = gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;
using Sink   = gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;

const std::string kTriggerKey{gr::tag::TRIGGER_NAME.key()};

struct EventCollector : gr::Block<EventCollector> {
    gr::EventPortIn in;
    GR_MAKE_REFLECTABLE(EventCollector, in);

    std::vector<gr::property_map> _events;

    gr::work::Status processBulk(gr::InputSpanLike auto& events) {
        for (const gr::property_map_view& event : events) {
            _events.emplace_back(event);
        }
        std::ignore = events.consume(events.size());
        return gr::work::Status::OK;
    }
};

struct Filtered {
    gr::testing::GraphFixture<> fixture;
    Source*                     source;
    TagToEvent<float>*          filter;
    Sink*                       sink;
    EventCollector*             collector;

    Filtered(std::size_t nSamples, std::vector<gr::testing::OwningTag> tags, gr::property_map settings) {
        source        = &fixture.emplace<Source>({{"n_samples_max", static_cast<gr::Size_t>(nSamples)}, {"mark_tag", false}});
        source->_tags = std::move(tags);
        filter        = &fixture.emplace<TagToEvent<float>>(std::move(settings));
        sink          = &fixture.emplace<Sink>({{"log_samples", true}});
        collector     = &fixture.emplace<EventCollector>();
        expect(fixture.connect<"out", "in">(*source, *filter).has_value());
        expect(fixture.connect<"out", "in">(*filter, *sink).has_value());
        expect(fixture.connect<"evtOut", "in">(*filter, *collector).has_value());
        expect(fixture.run().has_value());
    }
};

const auto definitions = [] {
    "matchesGlob against hand-written cases"_test = [] {
        expect(matchesGlob("gr:sample_rate", "gr:*"));
        expect(!matchesGlob("local_key", "gr:*"));
        expect(matchesGlob("abc", "a?c"));
        expect(!matchesGlob("ac", "a?c")) << "? is exactly one character";
        expect(matchesGlob("abc", "*b*"));
        expect(!matchesGlob("abc", "*d"));
        expect(matchesGlob("gr:trigger_name", "gr:trigger_*"));
        expect(matchesGlob("aXbXc", "a*c")) << "* backtracks over repeats";
        expect(matchesGlob("", "*")) << "* matches the empty run";
        expect(matchesGlob("", ""));
        expect(!matchesGlob("x", ""));
        expect(!matchesGlob("", "?"));
    };

    "the stream passes untouched; tags with a key outside gr: also go out as events"_test = [] {
        Filtered run(10UZ, {{2UZ, {{kTriggerKey, "two"}}}, {5UZ, {{kTriggerKey, "five"}, {"local_key", 5}}}, {7UZ, {{"other", 7}}}}, {{"key_patterns", std::vector<std::string>{"!gr:*"}}});
        expect(eq(run.sink->_samples.size(), 10UZ));
        expect(eq(std::ranges::count_if(run.sink->_tags, [](const auto& tag) { return tag.index == 2UZ || tag.index == 5UZ || tag.index == 7UZ; }), 3)) << "every tag still goes downstream";
        expect(eq(run.collector->_events.size(), 2UZ)) << "the tag at 2 has gr: keys only";
        expect(run.collector->_events.size() == 2UZ && run.collector->_events[0].contains("local_key") && run.collector->_events[1].contains("other"));
        expect(run.collector->_events.size() == 2UZ && run.collector->_events[0].contains(kTriggerKey)) << "the whole tag, not only the matching key";
    };

    "a list of patterns matches when any one does"_test = [] {
        Filtered run(10UZ, {{1UZ, {{kTriggerKey, "one"}}}, {3UZ, {{"gr:signal_name", "x"}}}, {6UZ, {{"other", 6}}}}, {{"key_patterns", std::vector<std::string>{"gr:trigger_*", "oth?r"}}});
        expect(eq(run.collector->_events.size(), 2UZ));
        expect(run.collector->_events.size() == 2UZ && run.collector->_events[0].contains(kTriggerKey) && run.collector->_events[1].contains("other"));
    };

    "more matching tags than event slots per call: none is lost while the queue holds them"_test = [] {
        std::vector<gr::testing::OwningTag> tags;
        for (std::size_t i = 0UZ; i < 200UZ; ++i) {
            tags.push_back({i, {{"n", static_cast<std::int64_t>(i)}}});
        }
        Filtered run(256UZ, std::move(tags), {{"key_patterns", std::vector<std::string>{"n"}}, {"max_pending_events", gr::Size_t(256)}});
        expect(eq(run.sink->_samples.size(), 256UZ));
        expect(eq(run.collector->_events.size(), 200UZ));
        bool inOrder = true;
        for (std::size_t i = 0UZ; i < run.collector->_events.size(); ++i) {
            inOrder = inOrder && run.collector->_events[i].template value_or<std::int64_t>("n", -1) == static_cast<std::int64_t>(i);
        }
        expect(inOrder) << "in stream order";
    };

    "a tag the trigger filter accepts becomes a dated trigger event; neither a filter nor a pattern selects nothing"_test = [] {
        const std::vector<gr::testing::OwningTag> tags{{2UZ, {{kTriggerKey, "start"}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{42}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}, {6UZ, {{kTriggerKey, "other"}}}};
        Filtered                                  run(10UZ, tags, {{"filter", std::string("start")}});
        expect(eq(run.sink->_samples.size(), 10UZ));
        expect(eq(run.collector->_events.size(), 1UZ)) << "only the accepted trigger";
        expect(run.collector->_events.size() == 1UZ && run.collector->_events[0].template value_or<std::uint64_t>(gr::tag::TRIGGER_TIME.key(), 0U) == 42U) << "dated as the tag was";
        Filtered none(10UZ, tags, {});
        expect(none.collector->_events.empty());
    };

    "a full pending queue drops its oldest event and the next work call publishes exactly one error event naming the drop count"_test = [] {
        constexpr std::size_t               kTriggers = 5UZ;
        std::vector<gr::testing::OwningTag> tags;
        for (std::size_t i = 0UZ; i < kTriggers; ++i) {
            tags.push_back({2UZ * i + 1UZ, {{kTriggerKey, "start"}}});
        }
        Filtered run(16UZ, std::move(tags), {{"filter", std::string("start")}, {"max_pending_events", gr::Size_t(1)}});

        std::size_t              nKept = 0UZ;
        std::vector<std::string> dropReports;
        for (const auto& event : run.collector->_events) {
            const auto details = gr::property_map_view{event}.template get_if<gr::property_map>(gr::tag::TRIGGER_META_INFO.key());
            const auto reason  = details != nullptr ? details->template value_or<std::string_view>("reason", "") : std::string_view{};
            if (reason.contains("dropped")) {
                dropReports.emplace_back(reason);
            } else {
                ++nKept;
            }
        }
        expect(gt(run.filter->n_events_dropped, 0U)) << "an undated trigger queues an error event and the trigger, which cannot both fit one slot";
        expect(eq(nKept + run.filter->n_events_dropped, 2UZ * kTriggers)) << "every queued event is either published or counted as dropped";
        expect(!dropReports.empty() && dropReports.size() <= static_cast<std::size_t>(run.filter->n_events_dropped)) << "a drop report per loss, never more than one per lost event";
        expect(!dropReports.empty() && dropReports.back().starts_with(std::format("{} event(s) dropped", run.filter->n_events_dropped))) << "the last report names the total";
    };

    "an incomplete trigger is counted in n_invalid_triggers and also publishes an error event"_test = [] {
        const std::vector<gr::testing::OwningTag> tags{{3UZ, {{kTriggerKey, "start"}}}};
        Filtered                                  run(10UZ, tags, {{"filter", std::string("start")}});
        expect(eq(run.filter->n_invalid_triggers, 1U));
        expect(eq(run.collector->_events.size(), 2UZ)) << "the error report and the trigger itself";
        const auto nErrors = std::ranges::count_if(run.collector->_events, [](const auto& event) { return event.template value_or<std::string_view>(kTriggerKey, "") == "error"; });
        const auto nStarts = std::ranges::count_if(run.collector->_events, [](const auto& event) { return event.template value_or<std::string_view>(kTriggerKey, "") == "start"; });
        expect(eq(nErrors, 1));
        expect(eq(nStarts, 1)) << "an undated trigger is still delivered, undated";
        expect(std::ranges::none_of(run.collector->_events, [](const auto& event) { return event.contains(gr::tag::TRIGGER_TIME.key()); }));
    };

    "interval mode publishes state active on the match and inactive on the unmatch, the trigger names taken from the tags"_test = [] {
        const auto                                dated = [](std::string_view name, std::uint64_t timeNs) { return gr::property_map{{kTriggerKey, std::string(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), timeNs}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}; };
        const std::vector<gr::testing::OwningTag> tags{{2UZ, dated("start", 20U)}, {6UZ, dated("stop", 60U)}};
        Filtered                                  run(10UZ, tags, {{"filter", std::string("start,stop")}, {"match_mode", std::string("interval")}});
        expect(eq(run.collector->_events.size(), 2UZ));
        if (run.collector->_events.size() == 2UZ) {
            expect(eq(run.collector->_events[0].template value_or<std::string_view>("state", ""), std::string_view{"active"}));
            expect(eq(run.collector->_events[0].template value_or<std::uint64_t>(gr::tag::TRIGGER_TIME.key(), 0U), std::uint64_t{20U}));
            expect(eq(run.collector->_events[1].template value_or<std::string_view>("state", ""), std::string_view{"inactive"}));
            expect(eq(run.collector->_events[1].template value_or<std::uint64_t>(gr::tag::TRIGGER_TIME.key(), 0U), std::uint64_t{60U}));
        }
    };

    "include_tag keeps the tag's extra fields without overwriting the event keys"_test = [] {
        const std::vector<gr::testing::OwningTag> tags{{2UZ, {{kTriggerKey, "start"}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{42}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}, {"extra", 7}}}};
        Filtered                                  withTag(10UZ, tags, {{"filter", std::string("start")}, {"trigger_name", std::string("renamed")}, {"include_tag", true}});
        expect(eq(withTag.collector->_events.size(), 1UZ));
        if (withTag.collector->_events.size() == 1UZ) {
            const auto& event = withTag.collector->_events[0];
            expect(eq(event.template value_or<int>("extra", -1), 7)) << "the tag's own field is kept";
            expect(eq(event.template value_or<std::string_view>(kTriggerKey, ""), std::string_view{"renamed"})) << "the event's name is not overwritten by the tag's";
            expect(eq(event.template value_or<std::uint64_t>(gr::tag::TRIGGER_TIME.key(), 0U), std::uint64_t{42U}));
        }
        Filtered without(10UZ, tags, {{"filter", std::string("start")}});
        expect(eq(without.collector->_events.size(), 1UZ));
        expect(without.collector->_events.size() == 1UZ && !without.collector->_events[0].contains("extra")) << "off by default";
    };
};

} // namespace

int main() {
    "TagToEvent"_test = definitions;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
