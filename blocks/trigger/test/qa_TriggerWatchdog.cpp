#include <boost/ut.hpp>

#include <optional>
#include <string>
#include <vector>

#include "Marble.hpp"
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/Gate.hpp>
#include <gnuradio-4.0/trigger/TriggerWatchdog.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

const boost::ut::suite<"TriggerWatchdog"> _watchdog = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

    "an outage is reported once, and so is the recovery"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}});
        auto&                     sink      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"out", "in">(watchdog, sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._samples.size(), 6UZ)) << "the stream passes through untouched";
        expect(eq(watchdog.n_timeouts.value, 1U)) << "one outage, reported once however long it lasts";
        const std::vector<std::string> names = collector.names();
        expect(eq(names.size(), 2UZ)) << "one timeout event and one recovery event";
        if (names.size() == 2UZ) {
            expect(eq(names[0], std::string("timeout")));
            expect(eq(names[1], std::string("recovered")));
        }
    };

    "an outage closes a gate downstream, and the recovery opens it again"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}, {"timeout_action", std::string("close")}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        const std::vector<std::string> names = collector.names();
        expect(eq(names.size(), 4UZ)) << "each report is followed by what the gate should do";
        if (names.size() == 4UZ) {
            expect(eq(names[0], std::string("timeout")));
            expect(eq(names[1], std::string("close"))) << "the action names what the outage does to the gate";
            expect(eq(names[2], std::string("recovered")));
            expect(eq(names[3], std::string("open"))) << "and the recovery does the opposite";
        }
    };

    "the action can be the other way round, for a chain that must run when the trigger is lost"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}, {"timeout_action", std::string("open")}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        const std::vector<std::string> names = collector.names();
        expect(eq(names.size(), 4UZ));
        if (names.size() == 4UZ) {
            expect(eq(names[1], std::string("open")));
            expect(eq(names[3], std::string("close")));
        }
    };

    "report says nothing to a gate, which is the default"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        expect(eq(collector.names().size(), 2UZ)) << "the two reports and nothing else";
    };

    "an outage closing a real gate stops the stream, drawn"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}, {"timeout_action", std::string("close")}});
        auto&                     gate      = fixture.emplace<Gate<float>>({{"mode", std::string("toggle")}, {"open_filter", std::string("open")}, {"close_filter", std::string("close")}, {"initial_state", true}});
        auto&                     sink      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"out", "in">(watchdog, gate).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, gate).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.connect<"out", "in">(gate, sink).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"TriggerWatchdog: an outage closes the gate, the recovery opens it"};
        diagram.unit = "sample";
        diagram.row("in").at(0U, "start").at(4U, "start").completes();
        diagram.condition("TriggerWatchdog(timeout = 2 samples, action = close) -> Gate(toggle)");
        const std::vector<std::string> names   = collector.names();
        auto&                          reports = diagram.row("evtOut");
        for (std::size_t i = 0UZ; i < names.size(); ++i) {
            reports.at(i < 2UZ ? 2U : 4U, names[i]);
        }
        reports.completes();
        diagram.print();
        std::println("the samples the gate let through: '{}'", sink.script());

        expect(ge(names.size(), 2UZ)) << "the outage is reported whether or not a gate acts on it";
    };

    "a disabled watchdog reports nothing"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b c d |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}});
        auto&                     sink      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"out", "in">(watchdog, sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        expect(eq(watchdog.n_timeouts.value, 0U)) << "both limits are zero, so the watchdog is unarmed";
        expect(collector.names().empty());
    };

    "a seconds limit with no rate to convert it stays unarmed and says so"_test = [&] {
        const gr::property_map untimedTags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}}}};

        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d e f |")}, {"sample_values", values}, {"sample_tags", untimedTags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_seconds", 1.f}});
        auto&                     sink      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", untimedTags}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"out", "in">(watchdog, sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        expect(eq(watchdog.n_timeouts.value, 0U)) << "an unconvertible limit must not expire";
        expect(eq(watchdog.n_invalid_triggers.value, 1U)) << "and the trigger that carried no time is counted";
        const std::vector<std::string> names = collector.names();
        expect(ge(names.size(), 1UZ)) << "and must not stay silent either";
        if (!names.empty()) {
            expect(eq(names.front(), std::string("error")));
        }
    };

    "a seconds limit needs the nominal rate, which timed tags alone do not give"_test = [&] {
        const auto timed = [](std::uint64_t at, std::optional<float> rate) {
            gr::property_map anchorTag{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), at}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}};
            if (rate) {
                anchorTag[std::string("sample_rate")] = *rate;
            }
            return anchorTag;
        };
        const auto run = [&](std::optional<float> rate) {
            const gr::property_map    timedTags{{"T", timed(1'000'000'000U, rate)}, {"U", timed(1'002'000'000U, rate)}};
            gr::testing::GraphFixture fixture;
            auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b U:c d e f |")}, {"sample_values", values}, {"sample_tags", timedTags}});
            auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_seconds", 0.002f}});
            auto&                     sink      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", timedTags}});
            auto&                     collector = fixture.emplace<EventTap>();
            expect(fixture.connect<"out", "in">(source, watchdog).has_value());
            expect(fixture.connect<"out", "in">(watchdog, sink).has_value());
            expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
            expect(fixture.run().has_value());
            return std::pair{watchdog.n_timeouts.value, collector.names()};
        };

        const auto [withoutRate, unarmed] = run(std::nullopt);
        expect(eq(withoutRate, 0U)) << "two anchors no longer imply a rate";
        expect(eq(unarmed.empty() ? std::string() : unarmed.front(), std::string("error"))) << "so the limit reports that it cannot arm";

        const auto [withRate, expired] = run(1000.f);
        expect(eq(withRate, 1U)) << "with the stream's sample_rate the limit expires";
        expect(eq(expired.empty() ? std::string() : expired.front(), std::string("timeout")));
    };
};

int main() { /* tests are statically executed */ }
