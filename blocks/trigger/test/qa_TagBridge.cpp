#include "Marble.hpp"
#include "TriggerTest.hpp"
#include <boost/ut.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/TagBridge.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>
#include <memory>
#include <string>
#include <vector>

namespace qa_tag_to_event {
using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

const boost::ut::suite<"TagToEvent"> _tagToEvent = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}};
    const gr::property_map tags{{"S", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}, {"X", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("other")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}, {"R", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

    "a tag the filter accepts becomes an event"_test = [values, tags] {
        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("a S:b c |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&     converter = graph.emplaceBlock<TagToEvent<float>>({{"filter", std::string("start")}});
        auto&     sink      = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&     events    = graph.emplaceBlock<EventTap>();
        expect(graph.connect<"out", "in">(source, converter).has_value());
        expect(graph.connect<"out", "in">(converter, sink).has_value());
        expect(graph.connect<"evtOut", "evtIn">(converter, events).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(eq(sink._samples.size(), 3UZ)) << "the stream passes through untouched";
        expect(eq(events._events.size(), 1UZ)) << "one tag matched, so one event";
        if (!events._events.empty()) {
            const auto& event = events._events.front();
            expect(eq(event.template get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()).value_or(std::string_view{}), std::string_view{"start"}));
            const std::uint64_t* stamp = event.template get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
            expect(stamp != nullptr) << "carrying the time of the tag it came from, which is what relates it to other blocks' events";
        }
    };

    "which tags became events, drawn"_test = [values, tags] {
        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("a S:b c X:a S:b c |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&     converter = graph.emplaceBlock<TagToEvent<float>>({{"filter", std::string("start")}});
        auto&     events    = graph.emplaceBlock<EventTap>();
        expect(graph.connect<"out", "in">(source, converter).has_value());
        expect(graph.connect<"evtOut", "evtIn">(converter, events).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        gr::testing::MarbleDiagram diagram{"TagToEvent: the tags a filter accepts leave as events, the stream untouched"};
        diagram.unit = "sample";
        diagram.row("in").at(1U, "start").at(3U, "other").at(4U, "start").completes();
        diagram.condition("TagToEvent(filter = \"start\")");
        auto&                            out = diagram.row("evtOut");
        const std::vector<std::uint64_t> matched{1U, 4U};
        for (std::size_t i = 0UZ; i < events._events.size() && i < matched.size(); ++i) {
            out.at(matched[i], "start");
        }
        out.completes();
        diagram.print();

        expect(eq(events._events.size(), 2UZ)) << "the two accepted tags, not the rejected one";
    };

    "a tag the filter rejects produces no event"_test = [values, tags] {
        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("a X:b c |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&     converter = graph.emplaceBlock<TagToEvent<float>>({{"filter", std::string("start")}});
        auto&     sink      = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&     events    = graph.emplaceBlock<EventTap>();
        expect(graph.connect<"out", "in">(source, converter).has_value());
        expect(graph.connect<"out", "in">(converter, sink).has_value());
        expect(graph.connect<"evtOut", "evtIn">(converter, events).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(eq(sink._samples.size(), 3UZ));
        expect(events._events.empty()) << "a tag naming another trigger must not be reported";
    };

    "the event is the tag itself, nothing added"_test = [values, tags] {
        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("S:a |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&     converter = graph.emplaceBlock<TagToEvent<float>>({{"filter", std::string("start")}});
        auto&     sink      = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&     events    = graph.emplaceBlock<EventTap>();
        expect(graph.connect<"out", "in">(source, converter).has_value());
        expect(graph.connect<"out", "in">(converter, sink).has_value());
        expect(graph.connect<"evtOut", "evtIn">(converter, events).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(fatal(eq(events._events.size(), 1UZ)));
        const gr::property_map_view event{events._events.front()};
        expect(event.contains(gr::tag::TRIGGER_NAME.key()) and event.contains(gr::tag::TRIGGER_TIME.key()) and event.contains(gr::tag::TRIGGER_OFFSET.key())) << "the tag's own fields";
        expect(!event.contains(gr::tag::TRIGGER_META_INFO.key())) << "no meta-info the tag did not carry";
    };

    "the event keeps the tag's context, as a value trigger's own event does"_test = [values] {
        const gr::property_map withContext{{"S", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, //
                                                     {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}, {std::string(gr::tag::CONTEXT.key()), std::string("ring A")}}}};
        gr::Graph              graph;
        auto&                  source    = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("S:a |")}, {"sample_values", values}, {"sample_tags", withContext}});
        auto&                  converter = graph.emplaceBlock<TagToEvent<float>>({{"filter", std::string("start")}});
        auto&                  events    = graph.emplaceBlock<EventTap>();
        expect(graph.connect<"out", "in">(source, converter).has_value());
        expect(graph.connect<"evtOut", "evtIn">(converter, events).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(eq(events._events.size(), 1UZ)) << fatal;
        const auto context = gr::property_map_view{events._events.front()}.get_if<std::string_view>(gr::tag::CONTEXT.key());
        expect(context.has_value() && *context == "ring A");
    };

    "a full queue loses its oldest event and reports the loss once"_test = [values, tags] {
        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("S+R:a b |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&     converter = graph.emplaceBlock<TagToEvent<float>>({{"filter", std::string("start")}, {"max_pending_events", 1U}});
        auto&     sink      = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&     events    = graph.emplaceBlock<EventTap>();
        expect(graph.connect<"out", "in">(source, converter).has_value());
        expect(graph.connect<"out", "in">(converter, sink).has_value());
        expect(graph.connect<"evtOut", "evtIn">(converter, events).has_value());

        gr::scheduler::Simple<>         scheduler;
        gr::trigger_test::ErrorMessages sent{scheduler};
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(eq(converter.n_events_dropped.value, 1U)) << "two tags land on one sample, and the queue holds one";
        expect(eq(sent.countContaining("dropped"), 1UZ)) << "a loss is never silent";
    };
};
} // namespace qa_tag_to_event

namespace qa_message_to_tag {
using namespace gr::blocks::trigger;
using gr::trigger_test::EventScript;

namespace {
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
} // namespace

const boost::ut::suite<"EventToTag"> _messageToTag = [] {
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
    const std::uint64_t    kAnchorNs = 1'000'000'000U;
    const gr::property_map anchored{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_TIME.key()), kAnchorNs}}}};

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

    "a full queue refuses the newest request"_test = [&] {
        auto chain = build({{"queue_depth", 1U}, {"sample_rate", 1.0f}});
        for (int i = 0; i < 2; ++i) {
            gr::property_map request             = injection(namedTag("start"));
            request[std::string(kInjectionTime)] = std::uint64_t{1'000'000'000};
            chain.injector->_events.push_back(std::move(request));
        }
        expect(chain.fixture->run().has_value());

        expect(ge(chain.bridge->n_refused.value, 1U)) << "the queue holds one, so the second is refused";
        expect(eq(chain.witness->_injected, 0UZ)) << "a time no anchor can resolve publishes nothing";
    };

    "a dated request lands on the sample its time names"_test = [&] {
        auto             chain               = build({{"sample_rate", 1000.f}}, "T:" + longScript, anchored);
        gr::property_map request             = injection(namedTag("start"));
        request[std::string(kInjectionTime)] = kAnchorNs + 20'000'000U;
        chain.injector->_events.push_back(std::move(request));
        expect(chain.fixture->run().has_value());

        expect(chain.witness->_injectedAt == std::vector<std::size_t>{20UZ}) << "20 ms after the anchor at 1 kHz is sample 20";
        expect(eq(chain.bridge->n_late.value, 0U)) << "the stream had not reached that time yet";
        expect(eq(chain.witness->_lateFlags, 0UZ));
    };

    "a request for a time the stream has passed is published at once and marked late"_test = [&] {
        auto             chain               = build({{"sample_rate", 1000.f}}, "T:" + longScript, anchored);
        gr::property_map request             = injection(namedTag("start"));
        request[std::string(kInjectionTime)] = kAnchorNs - 1'000'000U;
        chain.injector->_events.push_back(std::move(request));
        expect(chain.fixture->run().has_value());

        expect(eq(chain.witness->_injected, 1UZ)) << "an overdue request is not lost";
        expect(eq(chain.bridge->n_late.value, 1U)) << "1 ms before the first sample can only be late";
        expect(eq(chain.witness->_lateFlags, 1UZ)) << "and the tag says so";
    };

    "a delay moves the sample a request lands on and re-dates the trigger its tag carries"_test = [&] {
        auto             chain               = build({{"sample_rate", 1000.f}, {"delay", 5e-3}}, "T:" + longScript, anchored);
        gr::property_map request             = injection(gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), kAnchorNs + 10'000'000U}});
        request[std::string(kInjectionTime)] = kAnchorNs + 10'000'000U;
        chain.injector->_events.push_back(std::move(request));
        expect(chain.fixture->run().has_value());

        expect(chain.witness->_injectedAt == std::vector<std::size_t>{15UZ}) << "10 ms + 5 ms at 1 kHz is sample 15";
        expect(chain.witness->_triggerTimes == std::vector<std::uint64_t>{kAnchorNs + 15'000'000U}) << "the carried trigger_time moves by the delay too";
    };

    "an event that names no tag publishes nothing"_test = [&] {
        auto chain = build({});
        chain.injector->_events.push_back(gr::property_map{{std::string(kInjectionTime), kAnchorNs}});
        expect(chain.fixture->run().has_value());

        expect(eq(chain.witness->_injected, 0UZ)) << "there is no map to publish";
        expect(eq(chain.bridge->n_late.value, 0U)) << "and so nothing to be late with";
    };
};
} // namespace qa_message_to_tag

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
