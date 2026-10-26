#include <boost/ut.hpp>

#include <memory>
#include <string>
#include <vector>

#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Gate.hpp>
#include <gnuradio-4.0/trigger/TagBridge.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>
#include <gnuradio-4.0/trigger/TriggerWatchdog.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

namespace qa_tag_to_event {
using namespace boost::ut;
using gr::blocks::trigger::TagToEvent;
using gr::trigger_test::EventTap;

using Source = gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;
using Sink   = gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;

const std::string kTriggerKey{gr::tag::TRIGGER_NAME.key()};

struct Filtered {
    gr::testing::GraphFixture<> fixture;
    Source*                     source;
    TagToEvent<float>*          filter;
    Sink*                       sink;
    EventTap*                   collector;
    std::vector<std::string>    errors;

    Filtered(std::size_t nSamples, std::vector<gr::testing::OwningTag> tags, gr::property_map settings) {
        source        = &fixture.emplace<Source>({{"n_samples_max", static_cast<gr::Size_t>(nSamples)}, {"mark_tag", false}});
        source->_tags = std::move(tags);
        filter        = &fixture.emplace<TagToEvent<float>>(std::move(settings));
        sink          = &fixture.emplace<Sink>({{"log_samples", true}});
        collector     = &fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(*source, *filter).has_value());
        expect(fixture.connect<"out", "in">(*filter, *sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(*filter, *collector).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());
        errors = sent.texts();
    }
};

const boost::ut::suite<"TagToEvent"> _tagToEvent = [] {
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
        constexpr std::size_t               kTriggers = 20UZ; // more than the 8 slots evtOut offers a call plus the queue of one
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

    "the event keeps the tag's context, as a value trigger's own event does"_test = [] {
        const std::vector<gr::testing::OwningTag> tags{{2UZ, {{kTriggerKey, "start"}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}, {std::string(gr::tag::CONTEXT.key()), "ring A"}}}};
        Filtered                                  run(10UZ, tags, {{"filter", std::string("start")}});
        expect(eq(run.collector->_events.size(), 1UZ)) << fatal;
        const auto context = gr::property_map_view{run.collector->_events.front()}.get_if<std::string_view>(gr::tag::CONTEXT.key());
        expect(context.has_value() && *context == "ring A");
    };

    "which tags became events, drawn"_test = [] {
        const auto                                trigger = [](std::string name, std::uint64_t at) { return gr::property_map{{kTriggerKey, std::move(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), at}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}; };
        const std::vector<gr::testing::OwningTag> tags{{1UZ, trigger("start", 41U)}, {3UZ, trigger("other", 43U)}, {4UZ, trigger("start", 44U)}};
        Filtered                                  run(6UZ, tags, {{"filter", std::string("start")}});

        gr::testing::MarbleDiagram diagram{"TagToEvent: the tags a filter accepts leave as events, the stream untouched"};
        diagram.unit = "sample";
        diagram.row("in").at(1U, "start").at(3U, "other").at(4U, "start").completes();
        diagram.condition("TagToEvent(filter = \"start\")");
        auto&                            out = diagram.row("evtOut");
        const std::vector<std::uint64_t> matched{1U, 4U};
        for (std::size_t i = 0UZ; i < run.collector->_events.size() && i < matched.size(); ++i) {
            out.at(matched[i], "start");
        }
        out.completes();
        diagram.print();

        expect(eq(run.collector->_events.size(), 2UZ)) << "the two accepted tags, not the rejected one";
    };
};
} // namespace qa_tag_to_event

namespace qa_message_to_tag {
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::trigger;
using gr::trigger_test::EventScript;

/** a tag view is only valid while its work call is, so what the test asserts on is read out as it arrives */
struct TagWitness : gr::Block<TagWitness> {
    gr::PortIn<float> in;

    GR_MAKE_REFLECTABLE(TagWitness, in);

    std::vector<std::string>   _triggerNames;
    std::vector<std::string>   _sharedValues;
    std::size_t                _tagsSeen   = 0UZ;
    std::size_t                _injected   = 0UZ;
    std::size_t                _withFirst  = 0UZ;
    std::size_t                _withSecond = 0UZ;
    std::size_t                _bothFields = 0UZ;
    std::size_t                _lateFlags  = 0UZ;
    std::size_t                _samples    = 0UZ;
    std::vector<std::size_t>   _injectedAt;
    std::vector<std::uint64_t> _triggerTimes;

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan) {
        for (const auto& tag : inSpan.rawTags()) {
            ++_tagsSeen;
            if (const auto triggerName = tag.map.template get_if<std::string_view>(gr::tag::TRIGGER_NAME.key())) {
                _triggerNames.emplace_back(*triggerName);
            }
            if (const auto shared = tag.map.template get_if<std::string_view>(std::string_view{"shared"})) {
                _sharedValues.emplace_back(*shared);
            }
            const bool hasFirst  = tag.map.contains(std::string_view{"first"});
            const bool hasSecond = tag.map.contains(std::string_view{"second"});
            _withFirst += hasFirst ? 1UZ : 0UZ;
            _withSecond += hasSecond ? 1UZ : 0UZ;
            if (hasFirst && hasSecond) {
                ++_bothFields;
            }
            if (hasFirst || hasSecond || tag.map.contains(gr::tag::TRIGGER_NAME.key())) {
                ++_injected;
                _injectedAt.push_back(tag.index);
                if (const auto stamp = tag.map.template get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key())) {
                    _triggerTimes.push_back(*stamp);
                }
            }
            if (tag.map.contains(std::string_view{kInjectionLate})) {
                ++_lateFlags;
            }
        }
        _samples += inSpan.size();
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

[[nodiscard]] gr::property_map injection(gr::property_map tag) { return gr::property_map{{std::string(kInjectionTag), std::move(tag)}}; }

[[nodiscard]] gr::property_map namedTag(std::string name) { //
    return gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}};
}

const boost::ut::suite<"EventToTag"> _eventToTag = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}};

    const std::string longScript = [] {
        std::string script;
        for (int i = 0; i < 32; ++i) {
            script += "a ";
        }
        return script + "|";
    }();

    /** the scheduler owns the graph once started, so the fixture must outlive every assertion on a block */
    struct Chain {
        std::unique_ptr<gr::testing::GraphFixture<>> fixture;
        EventToTag<float>*                           bridge   = nullptr;
        EventScript*                                 injector = nullptr;
        TagWitness*                                  witness  = nullptr;
    };

    /** the stream's first sample is dated 1 s by a `trigger_time` tag, so a request's time names a sample */
    const gr::property_map anchored{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart}}}};

    auto build = [&](gr::property_map bridgeSettings, std::string script = {}, gr::property_map sampleTags = {}) {
        Chain chain;
        chain.fixture                   = std::make_unique<gr::testing::GraphFixture<>>();
        auto& source                    = chain.fixture->emplace<MarbleSource<float>>({{"script", script.empty() ? longScript : script}, {"sample_values", values}, {"sample_tags", std::move(sampleTags)}});
        chain.injector                  = &chain.fixture->emplace<EventScript>();
        chain.injector->_stopWhenDone   = false;
        chain.injector->_maxPerWorkCall = 1UZ;
        chain.bridge                    = &chain.fixture->emplace<EventToTag<float>>(std::move(bridgeSettings));
        chain.witness                   = &chain.fixture->emplace<TagWitness>();
        chain.bridge->in.max_samples    = 4UZ;
        expect(chain.fixture->connect<"out", "in">(source, *chain.bridge).has_value());
        expect(chain.fixture->connect<"evtOut", "evtIn">(*chain.injector, *chain.bridge).has_value());
        expect(chain.fixture->connect<"out", "in">(*chain.bridge, *chain.witness).has_value());
        return chain;
    };

    "an event carrying a tag publishes it on the stream"_test = [&] {
        auto chain = build({});
        chain.injector->_events.push_back(injection(namedTag("start")));
        expect(chain.fixture->run().has_value());

        expect(eq(chain.witness->_samples, 32UZ)) << "the stream passes through untouched";
        expect(ge(chain.witness->_tagsSeen, 1UZ)) << "the requested tag must reach the stream";
        expect(!chain.witness->_triggerNames.empty()) << "and carry the map the event asked for";
        if (!chain.witness->_triggerNames.empty()) {
            expect(eq(chain.witness->_triggerNames.front(), std::string("start")));
        }
    };

    "the requests and the tags they became, drawn"_test = [&] {
        auto chain = build({});
        chain.injector->_events.push_back(injection(namedTag("start")));
        chain.injector->_events.push_back(injection(namedTag("stop")));
        expect(chain.fixture->run().has_value());

        gr::testing::MarbleDiagram diagram{"EventToTag: an event asks for a tag, the bridge puts it on the stream"};
        diagram.unit   = "sample";
        auto& requests = diagram.row("evtIn");
        for (std::size_t i = 0UZ; i < chain.injector->_published; ++i) {
            requests.at(i, i == 0UZ ? "start" : "stop");
        }
        requests.completes();
        diagram.condition(std::format("EventToTag, {} late of {} seen", chain.bridge->n_late.value, chain.witness->_tagsSeen));
        auto& published = diagram.row("out");
        for (std::size_t i = 0UZ; i < chain.witness->_triggerNames.size(); ++i) {
            published.at(i, chain.witness->_triggerNames[i]);
        }
        published.completes();
        diagram.print();

        expect(ge(chain.witness->_tagsSeen, 2UZ)) << "both requests reached the stream";
    };

    "a request naming no time is reported late"_test = [&] {
        auto chain = build({});
        chain.injector->_events.push_back(injection(namedTag("start")));
        expect(chain.fixture->run().has_value());

        expect(ge(chain.bridge->n_late.value, 1U)) << "a request naming no time has nothing to wait for";
        expect(ge(chain.witness->_lateFlags, 1UZ)) << "and says so in the tag it publishes";
    };

    "requests landing on one sample merge, the later value winning"_test = [&] {
        auto chain = build({});
        chain.injector->_events.push_back(injection(gr::property_map{{std::string("first"), true}, {std::string("shared"), std::string("earlier")}}));
        chain.injector->_events.push_back(injection(gr::property_map{{std::string("second"), true}, {std::string("shared"), std::string("later")}}));
        expect(chain.fixture->run().has_value());

        expect(eq(chain.witness->_withFirst, 1UZ)) << "the first request reaches the stream";
        expect(eq(chain.witness->_withSecond, 1UZ)) << "so does the second";
        expect(!chain.witness->_sharedValues.empty()) << "the shared key must survive";
        if (chain.witness->_bothFields == 1UZ) {
            expect(eq(chain.witness->_sharedValues.back(), std::string("later"))) << "merged into one tag, the later request wins a shared key";
        } else {
            expect(eq(chain.witness->_sharedValues.size(), 2UZ));
            expect(eq(chain.witness->_sharedValues.front(), std::string("earlier")));
            expect(eq(chain.witness->_sharedValues.back(), std::string("later")));
        }
    };

    "a full queue drops the oldest request and counts it"_test = [&] {
        auto chain = build({{"max_pending_events", 1U}, {"sample_rate", 1.0f}});
        for (int i = 0; i < 2; ++i) {
            gr::property_map request             = injection(namedTag("start"));
            request[std::string(kInjectionTime)] = std::uint64_t{1'000'000'000};
            chain.injector->_events.push_back(std::move(request));
        }
        expect(chain.fixture->run().has_value());

        expect(eq(chain.bridge->n_events_dropped.value, 1U)) << "the queue holds one, so the second drops the first";
        expect(eq(chain.witness->_injected, 0UZ)) << "a time no anchor can resolve publishes nothing";
    };

    "a dated request lands on the sample its time names"_test = [&] {
        auto             chain               = build({{"sample_rate", 1000.f}}, "T:" + longScript, anchored);
        gr::property_map request             = injection(namedTag("start"));
        request[std::string(kInjectionTime)] = kStreamStart + 20'000'000U;
        chain.injector->_events.push_back(std::move(request));
        expect(chain.fixture->run().has_value());

        expect(chain.witness->_injectedAt == std::vector<std::size_t>{20UZ}) << "20 ms after the anchor at 1 kHz is sample 20";
        expect(eq(chain.bridge->n_late.value, 0U)) << "the stream had not reached that time yet";
        expect(eq(chain.witness->_lateFlags, 0UZ));
    };

    "a request for a time the stream has passed is published at once and marked late"_test = [&] {
        auto             chain               = build({{"sample_rate", 1000.f}}, "T:" + longScript, anchored);
        gr::property_map request             = injection(namedTag("start"));
        request[std::string(kInjectionTime)] = kStreamStart - 1'000'000U;
        chain.injector->_events.push_back(std::move(request));
        expect(chain.fixture->run().has_value());

        expect(eq(chain.witness->_injected, 1UZ)) << "an overdue request is not lost";
        expect(eq(chain.bridge->n_late.value, 1U)) << "1 ms before the first sample can only be late";
        expect(eq(chain.witness->_lateFlags, 1UZ)) << "and the tag says so";
    };

    "a delay moves the sample a request lands on and re-dates the trigger its tag carries"_test = [&] {
        auto             chain               = build({{"sample_rate", 1000.f}, {"delay", 5e-3}}, "T:" + longScript, anchored);
        gr::property_map request             = injection(gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart + 10'000'000U}});
        request[std::string(kInjectionTime)] = kStreamStart + 10'000'000U;
        chain.injector->_events.push_back(std::move(request));
        expect(chain.fixture->run().has_value());

        expect(chain.witness->_injectedAt == std::vector<std::size_t>{15UZ}) << "10 ms + 5 ms at 1 kHz is sample 15";
        expect(chain.witness->_triggerTimes == std::vector<std::uint64_t>{kStreamStart + 15'000'000U}) << "the carried trigger_time moves by the delay too";
    };

    "an event that names no tag publishes nothing"_test = [&] {
        auto chain = build({});
        chain.injector->_events.push_back(gr::property_map{{std::string(kInjectionTime), kStreamStart}});
        expect(chain.fixture->run().has_value());

        expect(eq(chain.witness->_injected, 0UZ)) << "there is no map to publish";
        expect(eq(chain.bridge->n_late.value, 0U)) << "and so nothing to be late with";
    };
};
} // namespace qa_message_to_tag

namespace qa_event_schema {
using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

/** anything meaningful only inside the sending block, never for an outside comparison */
[[nodiscard]] bool carriesPrivateKey(const gr::property_map& event) {
    const gr::property_map_view view{event};
    return view.contains(std::string_view{"stream_index"}) || view.contains(gr::tag::LOCAL_TIME.key());
}

[[nodiscard]] const gr::property_map& values() {
    static const gr::property_map map{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}};
    return map;
}
/** a well-formed trigger: named, dated, and with an offset */
[[nodiscard]] const gr::property_map& tags() {
    static const gr::property_map map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, //
                                                {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'000'000'000U}},     //
                                                {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},                             //
                                                {std::string("sample_rate"), 1000.f}}}};
    return map;
}

template<typename TBlock>
void checkEmitter(std::string_view what, gr::property_map settings, std::string_view script = "a T:b c T:d e f |") {
    using namespace boost::ut;

    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(script)}, {"sample_values", values()}, {"sample_tags", tags()}});
    auto&                     block  = fixture.emplace<TBlock>(std::move(settings));
    auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", tags()}});
    auto&                     schema = fixture.emplace<EventTap>();
    expect(fixture.connect<"out", "in">(source, block).has_value()) << what;
    expect(fixture.connect<"out", "in">(block, sink).has_value()) << what;
    expect(fixture.connect<"evtOut", "evtIn">(block, schema).has_value()) << what;
    expect(fixture.run().has_value()) << what;

    const std::size_t seen    = schema.size();
    const std::size_t named   = schema.countCarrying(gr::tag::TRIGGER_NAME.key());
    const std::size_t sourced = static_cast<std::size_t>(std::ranges::count_if(schema._events, gr::trigger_test::carriesSource));
    const std::size_t hollow  = static_cast<std::size_t>(std::ranges::count_if(schema._events, gr::trigger_test::carriesEmptyMeta));
    const std::size_t dated   = schema.countCarrying(gr::tag::TRIGGER_TIME.key());
    const std::size_t priv    = static_cast<std::size_t>(std::ranges::count_if(schema._events, carriesPrivateKey));

    expect(gt(seen, 0UZ)) << what << ": emitted no event at all, so the schema is untested";
    expect(eq(named, seen)) << what << ": every event names its trigger";
    expect(eq(sourced, 0UZ)) << what << ": no event carries a source, at the root or in trigger_meta_info";
    expect(eq(hollow, 0UZ)) << what << ": a trigger_meta_info, when present, holds something";
    expect(eq(priv, 0UZ)) << what << ": an event carries nothing private to its sender";
    expect(eq(dated, seen)) << what << ": every event carries the time it refers to";
}

const boost::ut::suite<"event schema"> _eventSchema = [] {
    using namespace boost::ut;

    "a gate's state changes carry a time"_test = [] { //
        checkEmitter<Gate<float>>("Gate", {{"mode", std::string("toggle")}, {"open_filter", std::string("start")}});
    };

    "a run report carries the time of the tag that started it"_test = [] { //
        checkEmitter<TakeN<float>>("TakeN", {{"filter", std::string("start")}, {"n", 2U}});
    };

    "a skipped run reports the same way"_test = [] { //
        checkEmitter<SkipN<float>>("SkipN", {{"filter", std::string("start")}, {"n", 2U}});
    };

    "a tag turned into an event keeps the tag's own time"_test = [] { //
        checkEmitter<TagToEvent<float>>("TagToEvent", {{"filter", std::string("start")}});
    };

    "a watchdog dates its outage and its recovery"_test = [] { //
        checkEmitter<TriggerWatchdog<float>>("TriggerWatchdog", {{"filter", std::string("start")}, {"timeout_samples", 2U}, {"sample_rate", 1000.f}}, "T:a b c d T:e f |");
    };
};
} // namespace qa_event_schema

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
