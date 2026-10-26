#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>

#include "TriggerTest.hpp"

namespace {
using namespace boost::ut;
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
    std::vector<std::string>    errors;

    Filtered(std::size_t nSamples, std::vector<gr::testing::OwningTag> tags, gr::property_map settings) {
        source        = &fixture.emplace<Source>({{"n_samples_max", static_cast<gr::Size_t>(nSamples)}, {"mark_tag", false}});
        source->_tags = std::move(tags);
        filter        = &fixture.emplace<TagToEvent<float>>(std::move(settings));
        sink          = &fixture.emplace<Sink>({{"log_samples", true}});
        collector     = &fixture.emplace<EventCollector>();
        expect(fixture.connect<"out", "in">(*source, *filter).has_value());
        expect(fixture.connect<"out", "in">(*filter, *sink).has_value());
        expect(fixture.connect<"evtOut", "in">(*filter, *collector).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());
        errors = sent.texts();
    }
};

const auto definitions = [] {
    "without a filter every tag goes out whole as an event and the stream passes untouched"_test = [] {
        const auto dated = [](std::uint64_t at) { return gr::property_map{{kTriggerKey, "t"}, {std::string(gr::tag::TRIGGER_TIME.key()), at}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}; };
        auto       two   = dated(2U);
        auto       five  = dated(5U);
        five.insert_or_assign("local_key", 5);
        Filtered run(10UZ, {{2UZ, std::move(two)}, {5UZ, std::move(five)}, {7UZ, {{"other", 7}}}}, {});
        expect(eq(run.sink->_samples.size(), 10UZ));
        expect(eq(std::ranges::count_if(run.sink->_tags, [](const auto& tag) { return tag.index == 2UZ || tag.index == 5UZ || tag.index == 7UZ; }), 3)) << "every tag still goes downstream";
        expect(eq(run.collector->_events.size(), 3UZ)) << "one event per tag, also a tag that names no trigger";
        expect(run.collector->_events.size() == 3UZ && run.collector->_events[1].contains("local_key") && run.collector->_events[1].contains(kTriggerKey)) << "the whole tag";
        expect(run.collector->_events.size() == 3UZ && run.collector->_events[2].contains("other"));
    };

    "more matching tags than event slots per call: none is lost while the queue holds them"_test = [] {
        std::vector<gr::testing::OwningTag> tags;
        for (std::size_t i = 0UZ; i < 200UZ; ++i) {
            tags.push_back({i, {{"n", static_cast<std::int64_t>(i)}}});
        }
        Filtered run(256UZ, std::move(tags), {{"max_pending_events", gr::Size_t(256)}});
        expect(eq(run.sink->_samples.size(), 256UZ));
        expect(eq(run.collector->_events.size(), 200UZ));
        bool inOrder = true;
        for (std::size_t i = 0UZ; i < run.collector->_events.size(); ++i) {
            inOrder = inOrder && run.collector->_events[i].template value_or<std::int64_t>("n", -1) == static_cast<std::int64_t>(i);
        }
        expect(inOrder) << "in stream order";
    };

    "a tag the trigger filter names becomes an event, dated as the tag was; other tags, and a filter that does not compile, select nothing"_test = [] {
        const std::vector<gr::testing::OwningTag> tags{{2UZ, {{kTriggerKey, "start"}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{42}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}, {6UZ, {{kTriggerKey, "other"}}}};
        Filtered                                  run(10UZ, tags, {{"filter", std::string("start")}});
        expect(eq(run.sink->_samples.size(), 10UZ));
        expect(eq(run.collector->_events.size(), 1UZ)) << "only the accepted trigger";
        expect(run.collector->_events.size() == 1UZ && run.collector->_events[0].template value_or<std::uint64_t>(gr::tag::TRIGGER_TIME.key(), 0U) == 42U) << "dated as the tag was";
        Filtered invalid(10UZ, tags, {{"filter", std::string("[start")}});
        expect(invalid.collector->_events.empty()) << "a filter that does not compile selects nothing";
    };

    "a full pending queue drops its oldest event and the next work call sends exactly one error message naming the drop count"_test = [] {
        constexpr std::size_t               kTriggers = 5UZ;
        std::vector<gr::testing::OwningTag> tags;
        for (std::size_t i = 0UZ; i < kTriggers; ++i) {
            tags.push_back({1UZ, {{kTriggerKey, "start"}, {"n", static_cast<std::int64_t>(i)}}});
        }
        Filtered run(16UZ, std::move(tags), {{"filter", std::string("start")}, {"max_pending_events", gr::Size_t(1)}});

        std::vector<std::string> dropReports;
        std::ranges::copy_if(run.errors, std::back_inserter(dropReports), [](const std::string& text) { return text.contains("dropped"); });
        expect(gt(run.filter->n_events_dropped, 0U)) << "a queue of one holds one of the triggers";
        expect(eq(run.collector->_events.size() + run.filter->n_events_dropped, kTriggers)) << "every queued event is either published or counted as dropped";
        expect(!dropReports.empty() && dropReports.size() <= static_cast<std::size_t>(run.filter->n_events_dropped)) << "a drop report per loss, never more than one per lost event";
        expect(!dropReports.empty() && dropReports.back().starts_with(std::format("{} event(s) dropped", run.filter->n_events_dropped))) << "the last report names the total";
    };

    "an incomplete trigger is counted in n_invalid_triggers and also sends an error message"_test = [] {
        const std::vector<gr::testing::OwningTag> tags{{3UZ, {{kTriggerKey, "start"}}}};
        Filtered                                  run(10UZ, tags, {{"filter", std::string("start")}});
        expect(eq(run.filter->n_invalid_triggers, 1U));
        expect(eq(run.collector->_events.size(), 1UZ)) << "the trigger itself";
        expect(eq(std::ranges::count_if(run.errors, [](const std::string& text) { return text.contains("lacking"); }), 1));
        const auto nStarts = std::ranges::count_if(run.collector->_events, [](const auto& event) { return event.template value_or<std::string_view>(kTriggerKey, "") == "start"; });
        expect(eq(nStarts, 1)) << "an undated trigger is still delivered, undated";
        expect(std::ranges::none_of(run.collector->_events, [](const auto& event) { return event.contains(gr::tag::TRIGGER_TIME.key()); }));
    };

    "an incomplete trigger is reported without a filter as well, and a tag naming no trigger is not one"_test = [] {
        const std::vector<gr::testing::OwningTag> tags{{3UZ, {{kTriggerKey, "start"}}}, {6UZ, {{"other", 7}}}};
        Filtered                                  run(10UZ, tags, {});
        expect(eq(run.filter->n_invalid_triggers, 1U)) << "the named tag lacks its time and offset; the other tag is no trigger";
        expect(eq(std::ranges::count_if(run.errors, [](const std::string& text) { return text.contains("lacking"); }), 1));
        expect(eq(run.collector->_events.size(), 2UZ)) << "the trigger and the other tag";
    };

    "an interval filter routes out its start and its stop, each unchanged"_test = [] {
        const auto trigger = [](std::string name, std::uint64_t at) { return gr::property_map{{kTriggerKey, std::move(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), at}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}; };
        Filtered   run(10UZ,
              {
                {2UZ, trigger("start", 42U)},
                {3UZ, trigger("other", 43U)},
                {5UZ, trigger("stop", 45U)},
            },
              {{"filter", std::string("[start, stop]")}});
        expect(fatal(eq(run.collector->_events.size(), 2UZ))) << "the start and the stop, not the tag between";
        const char* expected[] = {"start", "stop"};
        for (std::size_t i = 0UZ; i < 2UZ; ++i) {
            const gr::property_map_view event{run.collector->_events[i]};
            expect(event.get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()) == std::optional<std::string_view>{expected[i]}) << "event " << i << " keeps the tag's name";
            expect(!event.contains(gr::tag::TRIGGER_META_INFO.key())) << "event " << i << " is the tag, nothing added";
        }
    };

    "a '*' and a leading '!' select by pattern"_test = [] {
        const auto                                trigger = [](std::string name, std::uint64_t at) { return gr::property_map{{kTriggerKey, std::move(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), at}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}; };
        const std::vector<gr::testing::OwningTag> tags{{1UZ, trigger("beam_in", 41U)}, {3UZ, trigger("noise", 43U)}, {5UZ, trigger("beam_out", 45U)}};
        Filtered                                  beams(10UZ, tags, {{"filter", std::string("beam_*")}});
        expect(eq(beams.collector->_events.size(), 2UZ)) << "beam_in and beam_out";
        Filtered quiet(10UZ, tags, {{"filter", std::string("!noise")}});
        expect(eq(quiet.collector->_events.size(), 2UZ)) << "all but noise";
    };

    "a delay re-dates the forwarded trigger: trigger_time plus the delay, the offset kept"_test = [] {
        const std::vector<gr::testing::OwningTag> tags{{2UZ, {{kTriggerKey, "start"}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'000'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.25f}}}, {6UZ, {{"other", 1}}}};
        Filtered                                  run(10UZ, tags, {{"delay", 2.5e-6}});
        expect(eq(run.collector->_events.size(), 2UZ));
        expect(run.collector->_events.size() == 2UZ && run.collector->_events[0].template value_or<std::uint64_t>(gr::tag::TRIGGER_TIME.key(), 0U) == 1'000'002'500U) << "1 s + 2.5 us";
        expect(run.collector->_events.size() == 2UZ && run.collector->_events[0].template value_or<float>(gr::tag::TRIGGER_OFFSET.key(), 0.f) == 0.25f) << "the offset is not part of the delay";
        expect(run.collector->_events.size() == 2UZ && !run.collector->_events[1].contains(gr::tag::TRIGGER_TIME.key())) << "a tag without a time is forwarded as it was";
    };
};

} // namespace

int main() {
    "TagToEvent"_test = definitions;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
