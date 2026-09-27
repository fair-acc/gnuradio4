#include <boost/ut.hpp>

#include <format>
#include <string>
#include <tuple>
#include <vector>

#include "Marble.hpp"
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/ClockedTriggers.hpp>
#include <gnuradio-4.0/trigger/Coincidence.hpp>
#include <gnuradio-4.0/trigger/EventFilter.hpp>
#include <gnuradio-4.0/trigger/EventReduce.hpp>
#include <gnuradio-4.0/trigger/EventShaping.hpp>
#include <gnuradio-4.0/trigger/Gate.hpp>
#include <gnuradio-4.0/trigger/Predicate.hpp>
#include <gnuradio-4.0/trigger/SampleAndHold.hpp>
#include <gnuradio-4.0/trigger/SchmittTrigger.hpp>
#include <gnuradio-4.0/trigger/StreamOps.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>
#include <gnuradio-4.0/trigger/TimeInterval.hpp>
#include <gnuradio-4.0/trigger/TriggerWatchdog.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::testing::operator""_domain_test;
using gr::trigger_test::completed;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;

namespace {
struct EdgeRecord {
    std::size_t   index;
    std::string   name;
    std::uint64_t time;

    bool operator==(const EdgeRecord&) const = default;
};

struct SchmittRun {
    std::size_t                                        nSamples = 0UZ;
    std::vector<EdgeRecord>                            tagged;
    std::vector<std::pair<std::uint64_t, std::string>> events;
    std::size_t                                        nSourced = 0UZ;

    bool operator==(const SchmittRun&) const = default;
};

/**
 * a 0/5 square wave of period 16, dated at 0, re-dated at 40, and its threshold raised out of reach at 72 -- left at
 * its default, since a setting given at construction no longer follows the tags
 */
template<gr::trigger::InterpolationMethod kMethod>
[[nodiscard]] SchmittRun runSchmitt(auto& fixture, const std::string& placed) {
    const gr::property_map values{{"l", 0.f}, {"h", 5.f}};
    const auto             anchor = [](std::uint64_t at) { return gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("anchor")}, {std::string(gr::tag::TRIGGER_TIME.key()), at}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}; };
    const gr::property_map tags{{"T", anchor(1U)}, {"U", anchor(1'000'000'000U)}, {"H", gr::property_map{{"threshold", 10.f}}}};
    std::string            script;
    for (std::size_t i = 0UZ; i < 128UZ; ++i) {
        script += i == 0UZ ? "T:" : (i == 40UZ ? "U:" : (i == 72UZ ? "H:" : ""));
        script += (i / 8UZ) % 2UZ == 0UZ ? "l " : "h ";
    }
    script += "|";

    auto& source  = fixture.template emplace<MarbleSource<float>>({{"script", script}, {"sample_values", values}, {"sample_tags", tags}});
    auto& trigger = fixture.template emplace<SchmittTrigger<float, kMethod>>({{"offset", 2.5f}, {"sample_rate", 1000.f}, {"gr:compute_domain", placed}});
    auto& sink    = fixture.template emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
    auto& edges   = fixture.template emplace<gr::trigger_test::EventTap>();
    boost::ut::expect(fixture.template connect<"out", "in">(source, trigger).has_value());
    boost::ut::expect(fixture.template connect<"out", "in">(trigger, sink).has_value());
    boost::ut::expect(fixture.template connect<"evtOut", "evtIn">(trigger, edges).has_value());
    boost::ut::expect(fixture.run().has_value()) << placed;

    SchmittRun found;
    found.nSamples = sink._samples.size();
    for (const auto& recorded : sink._tags) {
        const gr::property_map_view tagMap{recorded.map};
        const auto                  name = tagMap.get_if<std::string_view>(std::string_view{gr::tag::TRIGGER_NAME.key()});
        const auto                  at   = tagMap.get_if<std::uint64_t>(std::string_view{gr::tag::TRIGGER_TIME.key()});
        if (name && (*name == "RISING" || *name == "FALLING")) {
            found.tagged.push_back({static_cast<std::size_t>(recorded.index), std::string(*name), at ? *at : 0U});
        }
    }
    found.events   = edges.dated();
    found.nSourced = static_cast<std::size_t>(std::ranges::count_if(edges._events, [&](const gr::property_map& event) {
        const auto source_ = gr::property_map_view{event}.get_if<std::string_view>(std::string_view{"source"});
        return source_ && *source_ == trigger.unique_name.value();
    }));
    return found;
}

/** a 1/5 alternation of 40 samples, dated at its start: 39 crossings, far more than one work call's event slots */
template<ValueCondition kCondition>
[[nodiscard]] SchmittRun runValueTrigger(auto& fixture, const std::string& placed, gr::property_map settings) {
    const gr::property_map values{{"a", 1.f}, {"e", 5.f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("anchor")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};
    std::string            script = "T:";
    for (std::size_t i = 0UZ; i < 40UZ; ++i) {
        script += i % 2UZ == 0UZ ? "a " : "e ";
    }
    script += "|";
    settings[std::string("threshold")]         = 3.f;
    settings[std::string("hysteresis")]        = 0.5f;
    settings[std::string("sample_rate")]       = 1000.f;
    settings[std::string("trigger_name")]      = std::string("edge");
    settings[std::string("gr:compute_domain")] = placed;

    auto& source  = fixture.template emplace<MarbleSource<float>>({{"script", script}, {"sample_values", values}, {"sample_tags", tags}});
    auto& trigger = fixture.template emplace<ValueTrigger<float, kCondition>>(std::move(settings));
    auto& sink    = fixture.template emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
    auto& found   = fixture.template emplace<gr::trigger_test::EventTap>();
    boost::ut::expect(fixture.template connect<"out", "in">(source, trigger).has_value());
    boost::ut::expect(fixture.template connect<"out", "in">(trigger, sink).has_value());
    boost::ut::expect(fixture.template connect<"evtOut", "evtIn">(trigger, found).has_value());
    boost::ut::expect(fixture.run().has_value()) << placed;

    SchmittRun run;
    run.nSamples = sink._samples.size();
    for (const auto& recorded : sink._tags) {
        const gr::property_map_view tagMap{recorded.map};
        const auto                  name = tagMap.get_if<std::string_view>(std::string_view{gr::tag::TRIGGER_NAME.key()});
        const auto                  at   = tagMap.get_if<std::uint64_t>(std::string_view{gr::tag::TRIGGER_TIME.key()});
        if (name && *name == "edge") {
            run.tagged.push_back({static_cast<std::size_t>(recorded.index), std::string(*name), at ? *at : 0U});
        }
    }
    run.events   = found.dated();
    run.nSourced = static_cast<std::size_t>(std::ranges::count_if(found._events, [&](const gr::property_map& event) {
        const auto source_ = gr::property_map_view{event}.get_if<std::string_view>(std::string_view{"source"});
        return source_ && *source_ == trigger.unique_name.value();
    }));
    return run;
}

/** a data line and a square clock (low, high: an edge on every second sample) into a clocked trigger */
template<typename TBlock>
[[nodiscard]] std::vector<gr::property_map> runClocked(auto& fixture, const std::string& placed, std::vector<float> data, gr::property_map settings) {
    std::vector<float> clock;
    for (std::size_t i = 0UZ; i < data.size(); ++i) {
        clock.push_back(i % 2UZ == 0UZ ? 0.f : 5.f);
    }
    const auto nSamples                        = static_cast<gr::Size_t>(data.size());
    settings[std::string("gr:compute_domain")] = placed;
    auto& dataSrc                              = fixture.template emplace<gr::testing::TagSource<float>>({{"n_samples_max", nSamples}, {"values", std::move(data)}, {"mark_tag", false}});
    auto& clkSrc                               = fixture.template emplace<gr::testing::TagSource<float>>({{"n_samples_max", nSamples}, {"values", std::move(clock)}, {"mark_tag", false}});
    auto& block                                = fixture.template emplace<TBlock>(std::move(settings));
    auto& sink                                 = fixture.template emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
    auto& found                                = fixture.template emplace<gr::trigger_test::EventTap>();
    boost::ut::expect(fixture.graph.connect(dataSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(clkSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("clk_in")}).has_value());
    boost::ut::expect(fixture.template connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.template connect<"evtOut", "evtIn">(block, found).has_value());
    boost::ut::expect(fixture.run().has_value()) << placed;
    boost::ut::expect(boost::ut::eq(sink._samples.size(), static_cast<std::size_t>(nSamples))) << placed << ": the data line passes through";
    return found._events;
}
} // namespace

static_assert(gr::device::DeviceRelocatable<Coincidence>);
static_assert(gr::device::DeviceRelocatable<Debounce<float>> && gr::device::DeviceRelocatable<Throttle<float>>);
static_assert(gr::device::DeviceRelocatable<DelayWhen<float>> && gr::device::DeviceRelocatable<SequenceEqual<float>> && gr::device::DeviceRelocatable<Repeat<float>>);
static_assert(gr::device::DeviceRelocatable<Gate<float>> && gr::device::DeviceRelocatable<Distinct<float>> && gr::device::DeviceRelocatable<Count>);

int main() {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

    "a gate answers the same on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& source = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto& gate   = fixture.template emplace<Gate<float>>({{"mode", std::string("skip_until")}, {"open_filter", std::string("start")}});
        auto& sink   = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, gate).has_value());
        expect(fixture.template connect<"out", "in">(gate, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink.script(), completed("T:c d e f"))) << domain;
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a take/hold chain answers the same on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& source = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a T:b c d e |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto& take   = fixture.template emplace<TakeN<float>>({{"filter", std::string("start")}, {"n", 3U}});
        auto& hold   = fixture.template emplace<SampleAndHold<float>>({{"filter", std::string("start")}, {"initial_value", 1.0f}});
        auto& sink   = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, take).has_value());
        expect(fixture.template connect<"out", "in">(take, hold).has_value());
        expect(fixture.template connect<"out", "in">(hold, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink.script(), completed("T:b b b"))) << domain << ": the captured sample is held for the whole run";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    /** the blocks below are given the domain itself, with '!' so a fallback to the host fails instead of passing */
    "a sample-and-hold runs on the domain it is given"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        auto&             source = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a T:b c d e |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&             hold   = fixture.template emplace<SampleAndHold<float>>({{"filter", std::string("start")}, {"initial_value", 1.0f}, {"gr:compute_domain", placed}});
        auto&             sink   = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, hold).has_value());
        expect(fixture.template connect<"out", "in">(hold, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink.script(), completed("a T:b b b b"))) << domain << ": the initial value, then the captured sample held";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a Schmitt trigger finds on the domain it is given the edges it finds on the host"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const auto        check  = [&]<gr::trigger::InterpolationMethod kMethod>(auto& graph) {
            gr::testing::GraphFixture reference;
            const SchmittRun          onHost    = runSchmitt<kMethod>(reference, "host");
            const SchmittRun          placedRun = runSchmitt<kMethod>(graph, placed);
            const std::size_t         lookAhead = kMethod == gr::trigger::InterpolationMethod::NO_INTERPOLATION ? 0UZ : SchmittTrigger<float, kMethod>::N_HISTORY;
            expect(ge(onHost.tagged.size(), 8UZ)) << "four pulses before the threshold is raised";
            expect(std::ranges::none_of(onHost.tagged, [&](const EdgeRecord& edge) { return edge.index >= 72UZ + lookAhead; })) << "and none once it has reached the comparator";
            if constexpr (kMethod == gr::trigger::InterpolationMethod::NO_INTERPOLATION) {
                expect(std::ranges::all_of(onHost.tagged, [](const EdgeRecord& edge) { return (edge.index >= 40UZ) == (edge.time >= 1'000'000'000U); })) << "an edge is dated from the anchor before it";
            }
            expect(eq(onHost.events.size(), onHost.tagged.size())) << "every edge also leaves as an event";
            expect(eq(onHost.nSourced, onHost.events.size())) << "and names the block that found it";
            expect(placedRun == onHost) << domain << gr::meta::enumName(kMethod).value_or("") << ": samples, edge tags, events and their source agree";
        };
        check.template            operator()<gr::trigger::InterpolationMethod::NO_INTERPOLATION>(fixture);
        gr::testing::GraphFixture interpolating;
        check.template            operator()<gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION>(interpolating);
        gr::testing::GraphFixture polynomial;
        check.template            operator()<gr::trigger::InterpolationMethod::POLYNOMIAL_INTERPOLATION>(polynomial);
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "an interpolating Schmitt trigger releases its held-back window at the end of the stream on the domain it is given"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string      placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const gr::property_map flat{{"l", 0.f}};
        std::string            script;
        for (std::size_t i = 0UZ; i < 64UZ; ++i) {
            script += "l ";
        }
        script += "|";
        auto& source  = fixture.template emplace<MarbleSource<float>>({{"script", script}, {"sample_values", flat}});
        auto& trigger = fixture.template emplace<SchmittTrigger<float, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION>>({{"offset", 2.5f}, {"gr:compute_domain", placed}});
        auto& sink    = fixture.template emplace<MarbleSink<float>>({{"sample_values", flat}});
        expect(fixture.template connect<"out", "in">(source, trigger).has_value());
        expect(fixture.template connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink._samples.size(), 64UZ)) << domain << ": no edge releases the last window, only the end of the stream does";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a take and a skip run on the domain they are given and report their runs"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string      placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const auto             named  = [](std::uint64_t at) { return at == 0U ? gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}} : gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), at}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}; };
        const gr::property_map runTags{{"T", named(1U)}, {"U", named(0U)}};
        auto&                  source = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a T:b c d U:e f |")}, {"sample_values", values}, {"sample_tags", runTags}});
        auto&                  take   = fixture.template emplace<TakeN<float>>({{"filter", std::string("start")}, {"n", 2U}, {"gr:compute_domain", placed}});
        auto&                  skipN  = fixture.template emplace<SkipN<float>>({{"filter", std::string("start")}, {"n", 1U}, {"gr:compute_domain", placed}});
        auto&                  sink   = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", runTags}});
        auto&                  taken  = fixture.template emplace<gr::trigger_test::EventTap>();
        auto&                  missed = fixture.template emplace<gr::trigger_test::EventTap>();
        expect(fixture.template connect<"out", "in">(source, take).has_value());
        expect(fixture.template connect<"out", "in">(take, skipN).has_value());
        expect(fixture.template connect<"out", "in">(skipN, sink).has_value());
        expect(fixture.template connect<"evtOut", "evtIn">(take, taken).has_value());
        expect(fixture.template connect<"evtOut", "evtIn">(skipN, missed).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink.script(), completed("c f"))) << domain << ": take keeps b c and e f, skip drops the first of each";
        expect(taken.names() == std::vector<std::string>{"take", "error", "take"}) << domain << ": a run each, and the undated trigger reported";
        expect(eq(taken.dated().size(), 1UZ)) << domain << ": the dated trigger's run carries its time";
        expect(std::ranges::all_of(taken._events,
            [&](const gr::property_map& event) {
                const gr::property_map_view view{event};
                const auto                  meta    = view.get_if<gr::property_map_view>(std::string_view{gr::tag::TRIGGER_META_INFO.key()});
                const auto                  source_ = meta ? meta->get_if<std::string_view>(std::string_view{"source"}) : view.get_if<std::string_view>(std::string_view{"source"});
                return source_ && *source_ == take.unique_name.value();
            }))
            << domain << ": every event names the block that sent it";
        expect(eq(missed.names().size(), 3UZ)) << domain;
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a value trigger finds on the domain it is given the triggers it finds on the host"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const auto        check  = [&]<ValueCondition kCondition>(auto& graph, const gr::property_map& settings, std::size_t expected) {
            gr::testing::GraphFixture reference;
            const SchmittRun          onHost    = runValueTrigger<kCondition>(reference, "host", settings);
            const SchmittRun          placedRun = runValueTrigger<kCondition>(graph, placed, settings);
            expect(eq(onHost.tagged.size(), expected)) << gr::meta::enumName(kCondition).value_or("");
            expect(eq(onHost.events.size(), expected)) << "every trigger also leaves as an event, however many one call finds";
            expect(eq(onHost.nSourced, expected)) << "and names the block that found it";
            expect(placedRun == onHost) << domain << gr::meta::enumName(kCondition).value_or("") << ": samples, trigger tags, events and their source agree";
        };
        check.template            operator()<ValueCondition::level>(fixture, gr::property_map{{"edge", std::string("both")}}, 39UZ);
        gr::testing::GraphFixture pulses;
        check.template            operator()<ValueCondition::pulse_width>(pulses, gr::property_map{{"width_max_samples", 2U}}, 19UZ);
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a watchdog reports an outage and its end on the domain it is given"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string      placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const gr::property_map dated{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}};
        const gr::property_map beats{{"T", dated}, {"U", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}}}};
        auto&                  source   = fixture.template emplace<MarbleSource<float>>({{"script", std::string("T:a b c d e U:f a |")}, {"sample_values", values}, {"sample_tags", beats}});
        auto&                  watchdog = fixture.template emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}, {"timeout_action", std::string("close")}, {"gr:compute_domain", placed}});
        auto&                  sink     = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", beats}});
        auto&                  reports  = fixture.template emplace<gr::trigger_test::EventTap>();
        expect(fixture.template connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.template connect<"out", "in">(watchdog, sink).has_value());
        expect(fixture.template connect<"evtOut", "evtIn">(watchdog, reports).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink._samples.size(), 7UZ)) << domain << ": the stream passes through untouched";
        expect(reports.names() == std::vector<std::string>{"timeout", "close", "error", "recovered", "open"}) << domain << ": the outage, its action, the undated trigger that ended it, the recovery and its action";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "the clocked triggers find on the domain they are given what they find on the host"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string      placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const gr::property_map common{{"threshold", 2.5f}, {"hysteresis", 1.f}, {"clock_threshold", 2.5f}, {"sample_rate", 1000.f}};

        gr::property_map serial        = common;
        serial[std::string("pattern")] = std::string("1");
        const std::vector<float>  high(40UZ, 5.f);
        gr::testing::GraphFixture serialHost;
        const auto                serialOnHost = runClocked<SerialPatternTrigger<float>>(serialHost, "host", high, serial);
        expect(eq(serialOnHost.size(), 20UZ)) << "every bit matches a one-bit pattern";
        expect(runClocked<SerialPatternTrigger<float>>(fixture, placed, high, serial) == serialOnHost) << domain << ": the same matches, dated alike";

        gr::property_map setupHold              = common;
        setupHold[std::string("setup_samples")] = 2U;
        setupHold[std::string("hold_samples")]  = 2U;
        const std::vector<float>  moving{0.f, 0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
        gr::testing::GraphFixture setupHost;
        gr::testing::GraphFixture setupPlaced;
        const auto                setupOnHost = runClocked<SetupHoldTrigger<float>>(setupHost, "host", moving, setupHold);
        expect(ge(setupOnHost.size(), 1UZ)) << "the change violates at least one window";
        expect(runClocked<SetupHoldTrigger<float>>(setupPlaced, placed, moving, setupHold) == setupOnHost) << domain << ": the same violations, dated alike";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a running sum and a pairing run on the domain they are given"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed   = domain == "host" ? std::string("host") : std::string(domain) + "!";
        auto&             source   = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a b c T:a b |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&             scan     = fixture.template emplace<Scan<float>>({{"operation", std::string("sum")}, {"reset_filter", std::string("start")}, {"gr:compute_domain", placed}});
        auto&             pairs    = fixture.template emplace<Pairwise<float>>({{"gr:compute_domain", placed}});
        auto&             previous = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&             current  = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, scan).has_value());
        expect(fixture.template connect<"out", "in">(scan, pairs).has_value());
        expect(fixture.template connect<"previous", "in">(pairs, previous).has_value());
        expect(fixture.template connect<"current", "in">(pairs, current).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(previous.script(), completed("a c f a"))) << domain << ": the sums 1 3 6, restarted at the trigger, each beside its successor";
        expect(eq(current.script(), completed("c f T:a c"))) << domain;
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a sample filter runs on the domain it is given, and refuses a device for an expression"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        auto&             source = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a T:e b f c |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&             filter = fixture.template emplace<SampleFilter<float>>({{"predicate", std::string("greater")}, {"threshold", 2.5}, {"gr:compute_domain", placed}});
        auto&             sink   = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, filter).has_value());
        expect(fixture.template connect<"out", "in">(filter, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink.script(), completed("T:e f c"))) << domain << ": the samples above 2.5, each with its own tag";

        gr::testing::GraphFixture scripted;
        auto&                     scriptedSource = scripted.emplace<MarbleSource<float>>({{"script", std::string("a e b f c |")}, {"sample_values", values}});
        auto&                     scriptedFilter = scripted.emplace<SampleFilter<float>>({{"expression", std::string("x > 2.5")}, {"gr:compute_domain", placed}});
        auto&                     scriptedSink   = scripted.emplace<MarbleSink<float>>({{"sample_values", values}});
        expect(scripted.connect<"out", "in">(scriptedSource, scriptedFilter).has_value());
        expect(scripted.connect<"out", "in">(scriptedFilter, scriptedSink).has_value());
        expect(eq(scripted.run().has_value(), domain == "host")) << domain << ": an ExprTk expression is refused a device rather than silently ignored";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "take-while, skip-while and element-at run on the domain they are given"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const auto        play   = [&]<typename TBlock>(auto& graph, gr::property_map settings) {
            settings[std::string("gr:compute_domain")] = placed;
            auto& source                               = graph.template emplace<MarbleSource<float>>({{"script", std::string("a b c f a b |")}, {"sample_values", values}});
            auto& block                                = graph.template emplace<TBlock>(std::move(settings));
            auto& sink                                 = graph.template emplace<MarbleSink<float>>({{"sample_values", values}});
            expect(graph.template connect<"out", "in">(source, block).has_value());
            expect(graph.template connect<"out", "in">(block, sink).has_value());
            expect(graph.run().has_value()) << domain;
            return std::string(sink.script());
        };
        expect(eq(play.template operator()<TakeWhile<float>>(fixture, {{"predicate", std::string("less")}, {"threshold", 4.f}}), completed("a b c"))) << domain << ": taken until the first failing sample";
        gr::testing::GraphFixture skipping;
        expect(eq(play.template operator()<SkipWhile<float>>(skipping, {{"predicate", std::string("less")}, {"threshold", 4.f}}), completed("f a b"))) << domain << ": skipped until the first failing sample";
        gr::testing::GraphFixture picking;
        expect(eq(play.template operator()<ElementAt<float>>(picking, {{"n", 3U}}), completed("f"))) << domain << ": the fourth sample, counting from zero";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a tail is kept or skipped on the domain it is given, at a trigger and at the stream's end"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const auto        play   = [&]<typename TBlock>(auto& graph, gr::property_map settings) {
            settings[std::string("gr:compute_domain")] = placed;
            settings[std::string("n")]                 = 2U;
            auto& source                               = graph.template emplace<MarbleSource<float>>({{"script", std::string("a b c T:d e f |")}, {"sample_values", values}, {"sample_tags", tags}});
            auto& block                                = graph.template emplace<TBlock>(std::move(settings));
            auto& sink                                 = graph.template emplace<MarbleSink<float>>({{"sample_values", values}});
            expect(graph.template connect<"out", "in">(source, block).has_value());
            expect(graph.template connect<"out", "in">(block, sink).has_value());
            expect(graph.run().has_value()) << domain;
            return std::string(sink.script());
        };
        expect(eq(play.template operator()<TakeLast<float>>(fixture, {{"filter", std::string("start")}}), completed("b c e f"))) << domain << ": the last two before the trigger, and the last two before the end";
        gr::testing::GraphFixture skipping;
        expect(eq(play.template operator()<SkipLast<float>>(skipping, {{"filter", std::string("start")}}), completed("a d"))) << domain << ": everything but those";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "more tags in one span than a kernel's slots are all seen, over several calls"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        std::string       script;
        for (std::size_t i = 0UZ; i < 100UZ; ++i) {
            script += i % 2UZ == 0UZ ? "T:a " : "T:e ";
        }
        script += "|";
        auto& source  = fixture.template emplace<MarbleSource<float>>({{"script", script}, {"sample_values", values}, {"sample_tags", tags}});
        auto& trigger = fixture.template emplace<ValueTrigger<float, ValueCondition::level>>({{"threshold", 3.f}, {"hysteresis", 0.5f}, {"edge", std::string("both")}, {"trigger_name", std::string("edge")}, {"gr:compute_domain", placed}});
        auto& sink    = fixture.template emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.template connect<"out", "in">(source, trigger).has_value());
        expect(fixture.template connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        const auto named = [&](std::string_view name) { return std::ranges::count_if(sink._tags, [&](const auto& recorded) { return gr::property_map_view{recorded.map}.template get_if<std::string_view>(std::string_view{gr::tag::TRIGGER_NAME.key()}) == std::optional{name}; }); };
        expect(eq(sink._samples.size(), 100UZ)) << domain;
        expect(eq(named("start"), 100)) << domain << ": every forwarded tag arrives";
        expect(eq(named("edge"), 99)) << domain << ": and every crossing";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "an event filter on a device passes, blocks and orders events as on the host"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const auto        run    = [&](auto& graph, const std::string& where, gr::property_map settings) {
            settings[std::string("gr:compute_domain")] = where;
            auto& script                               = graph.template emplace<EventScript>();
            script._events                             = {eventNamed("A", 3'000U), gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("A")}}, eventNamed("B", 2'000U), eventNamed("A", 1'000U), eventNamed("A", 2'500U)};
            auto& filter                               = graph.template emplace<EventFilter>(std::move(settings));
            auto& tap                                  = graph.template emplace<gr::trigger_test::EventTap>();
            expect(graph.template connect<"evtOut", "evtIn">(script, filter).has_value());
            expect(graph.template connect<"evtOut", "evtIn">(filter, tap).has_value());
            const bool ran = graph.run().has_value();
            return std::pair{ran, tap.dated(std::string_view{"undated"})};
        };
        gr::testing::GraphFixture reference;
        const auto                onHost = run(reference, "host", {{"filter", std::string("A")}});
        expect(onHost.first);
        expect(eq(onHost.second.size(), 3UZ)) << "the three dated A events, in time order";
        expect(run(fixture, placed, {{"filter", std::string("A")}}) == onHost) << domain << ": the same events, in the same order";
        gr::testing::GraphFixture renaming;
        expect(eq(run(renaming, placed, {{"filter", std::string("A")}, {"trigger_name", std::string("renamed")}}).first, domain == "host")) << domain << ": a rename is refused a device rather than run differently";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "an interval measurement runs on the domain it is given"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        auto&             script = fixture.template emplace<EventScript>();
        script._events           = {eventNamed("tick", 1'000'000'000U), eventNamed("tick", 1'003'000'000U), gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("tick")}}, eventNamed("tick", 1'002'000'000U)};
        auto& interval           = fixture.template emplace<TimeInterval<double>>({{"mode", std::string("to_previous")}, {"gr:compute_domain", placed}});
        auto& measured           = fixture.template emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.template connect<"evtOut", "evtIn">(script, interval).has_value());
        expect(fixture.template connect<"out", "in">(interval, measured).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(measured._samples.size(), 2UZ)) << domain << ": three dated ticks, ordered in time, give two periods";
        if (measured._samples.size() == 2UZ) {
            expect(approx(measured._samples[0], 2e-3, 1e-12)) << domain;
            expect(approx(measured._samples[1], 1e-3, 1e-12)) << domain;
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a counter on a device reports every nth of more events than one call takes"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        auto&             script = fixture.template emplace<EventScript>();
        for (std::uint64_t i = 0U; i < 10U; ++i) {
            script._events.push_back(eventNamed("edge", 1'000'000'000U + i * 1'000'000U));
        }
        auto& counter = fixture.template emplace<Count>({{"filter", std::string("edge")}, {"n", 3U}, {"gr:compute_domain", placed}});
        auto& tap     = fixture.template emplace<gr::trigger_test::EventTap>();
        expect(fixture.template connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.template connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(tap.size(), 3UZ)) << domain << ": the third, sixth and ninth of ten";
        if (tap.size() == 3UZ) {
            const auto* ninth = gr::property_map_view{tap.metaOf(2)}.get_if<std::uint64_t>(std::string_view{"count"});
            expect(ninth != nullptr && *ninth == 9U) << domain << ": each report carries its occurrence";
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "distinct and distinct-until-changed run on the domain they are given"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const auto        play   = [&]<typename TBlock>(auto& graph, gr::property_map settings) {
            settings[std::string("gr:compute_domain")] = placed;
            auto& source                               = graph.template emplace<MarbleSource<float>>({{"script", std::string("a a b a c b c c |")}, {"sample_values", values}});
            auto& block                                = graph.template emplace<TBlock>(std::move(settings));
            auto& sink                                 = graph.template emplace<MarbleSink<float>>({{"sample_values", values}});
            expect(graph.template connect<"out", "in">(source, block).has_value());
            expect(graph.template connect<"out", "in">(block, sink).has_value());
            expect(graph.run().has_value()) << domain;
            return std::string(sink.script());
        };
        expect(eq(play.template operator()<Distinct<float>>(fixture, {}), completed("a b c"))) << domain << ": each value once";
        gr::testing::GraphFixture changes;
        expect(eq(play.template operator()<DistinctUntilChanged<float>>(changes, {}), completed("a b a c b c"))) << domain << ": each change once";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "debounce and throttle run on the domain they are given, on tags and on injected events"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const auto        play   = [&]<typename TBlock>(auto& graph, std::vector<gr::property_map> injected) {
            auto& script   = graph.template emplace<EventScript>();
            script._events = std::move(injected);
            auto& source   = graph.template emplace<MarbleSource<float>>({{"script", std::string(script._events.empty() ? "T:a T:a a a a T:a a a a |" : "a a a a a a a a a |")}, {"sample_values", values}, {"sample_tags", tags}});
            auto& block    = graph.template emplace<TBlock>({{"filter", std::string("start")}, {"n_samples", 3U}, {"gr:compute_domain", placed}});
            auto& sink     = graph.template emplace<MarbleSink<float>>({{"sample_values", values}});
            auto& found    = graph.template emplace<gr::trigger_test::EventTap>();
            expect(graph.template connect<"out", "in">(source, block).has_value());
            expect(graph.template connect<"out", "in">(block, sink).has_value());
            expect(graph.template connect<"evtOut", "evtIn">(block, found).has_value());
            if (!script._events.empty()) {
                expect(graph.template connect<"evtOut", "evtIn">(script, block).has_value());
            }
            expect(graph.run().has_value()) << domain;
            expect(eq(sink._samples.size(), 9UZ)) << domain << ": the stream passes untouched";
            return found.names();
        };
        const std::vector<std::string> twice{"start", "start"};
        const auto debounced = play.template operator()<Debounce<float>>(fixture, {});
        expect(debounced == twice) << domain << ": a burst and a lone trigger, one event each, got" << std::format("{}", debounced);
        gr::testing::GraphFixture throttled;
        const auto throttledNames = play.template operator()<Throttle<float>>(throttled, {});
        expect(throttledNames == twice) << domain << ": the first of the burst, then the lone one, got" << std::format("{}", throttledNames);
        gr::testing::GraphFixture injected;
        const auto injectedNames = play.template operator()<Throttle<float>>(injected, {eventNamed("start", 1U), eventNamed("start", 2U)});
        expect(eq(injectedNames.size(), 1UZ)) << domain << ": two injected at one sample, the inhibit passes one, got" << std::format("{}", injectedNames);
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a coincidence unit runs on the domain it is given"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        auto&             script = fixture.template emplace<EventScript>();
        script._events           = {eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'030U), eventNamed("a", 2'000'000'000U), eventNamed("b", 2'000'000'080U)};
        auto& unit               = fixture.template emplace<Coincidence>({{"filters", std::vector<std::string>{"a", "b"}}, {"logic", std::string("all")}, {"window", 50e-9}, {"gr:compute_domain", placed}});
        auto& found              = fixture.template emplace<gr::trigger_test::EventTap>();
        expect(fixture.template connect<"evtOut", "evtIn">(script, unit).has_value());
        expect(fixture.template connect<"evtOut", "evtIn">(unit, found).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(found.names() == std::vector<std::string>{"coincidence"}) << domain << ": 30 ns apart coincide, 80 ns apart do not";
        expect(found.times() == std::vector<std::uint64_t>{1'000'000'000U}) << domain << ": dated by the first contribution";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "delay-when, sequence-equal and repeat run on the domain they are given"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        {
            auto& source   = fixture.template emplace<gr::testing::TagSource<float>>({{"n_samples_max", 5U}, {"values", std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f}}, {"mark_tag", false}});
            auto& delays   = fixture.template emplace<gr::testing::TagSource<gr::Size_t>>({{"n_samples_max", 5U}, {"values", std::vector<gr::Size_t>{3U, 0U, 0U, 0U, 0U}}, {"mark_tag", false}});
            auto& delaying = fixture.template emplace<DelayWhen<float>>({{"gr:compute_domain", placed}});
            auto& sink     = fixture.template emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            expect(fixture.graph.connect(source, gr::PortDefinition{std::string("out")}, delaying, gr::PortDefinition{std::string("in")}).has_value());
            expect(fixture.graph.connect(delays, gr::PortDefinition{std::string("out")}, delaying, gr::PortDefinition{std::string("delay_in")}).has_value());
            expect(fixture.template connect<"out", "in">(delaying, sink).has_value());
            expect(fixture.run().has_value()) << domain;
            expect(std::vector<float>(sink._samples.begin(), sink._samples.end()) == std::vector<float>{2.f, 3.f, 1.f, 4.f, 5.f}) << domain << ": the first waits three samples";
        }
        {
            gr::testing::GraphFixture compared;
            auto&                     measured  = compared.emplace<gr::testing::TagSource<float>>({{"n_samples_max", 6U}, {"values", std::vector<float>{1.f, 2.f, 9.f, 4.f, 5.f, 6.f}}, {"mark_tag", false}});
            auto&                     reference = compared.emplace<gr::testing::TagSource<float>>({{"n_samples_max", 6U}, {"values", std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}}, {"mark_tag", false}});
            auto&                     equal     = compared.emplace<SequenceEqual<float>>({{"n_samples", 3U}, {"gr:compute_domain", placed}});
            auto&                     verdicts  = compared.emplace<gr::testing::TagSink<std::uint8_t, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            expect(compared.graph.connect(measured, gr::PortDefinition{std::string("out")}, equal, gr::PortDefinition{std::string("in")}).has_value());
            expect(compared.graph.connect(reference, gr::PortDefinition{std::string("out")}, equal, gr::PortDefinition{std::string("reference")}).has_value());
            expect(compared.connect<"out", "in">(equal, verdicts).has_value());
            expect(compared.run().has_value()) << domain;
            expect(std::vector<std::uint8_t>(verdicts._samples.begin(), verdicts._samples.end()) == std::vector<std::uint8_t>{0U, 1U}) << domain << ": a verdict per segment of three";
        }
        {
            gr::testing::GraphFixture replayed;
            auto&                     source = replayed.emplace<MarbleSource<float>>({{"script", std::string("T:a b T:c |")}, {"sample_values", values}, {"sample_tags", tags}});
            auto&                     repeat = replayed.emplace<Repeat<float>>({{"n_repeats", 3U}, {"segment_filter", std::string("start")}, {"capacity", 64U}, {"gr:compute_domain", placed}});
            auto&                     sink   = replayed.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            expect(replayed.connect<"out", "in">(source, repeat).has_value());
            expect(replayed.connect<"out", "in">(repeat, sink).has_value());
            expect(replayed.run().has_value()) << domain;
            expect(std::vector<float>(sink._samples.begin(), sink._samples.end()) == std::vector<float>{1.f, 2.f, 1.f, 2.f, 1.f, 2.f}) << domain << ": the segment between two triggers, three times";
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a gate runs on the domain it is given, opened by a tag or by an event"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        auto&             source = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&             gate   = fixture.template emplace<Gate<float>>({{"mode", std::string("skip_until")}, {"open_filter", std::string("start")}, {"gr:compute_domain", placed}});
        auto&             sink   = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&             states = fixture.template emplace<gr::trigger_test::EventTap>();
        expect(fixture.template connect<"out", "in">(source, gate).has_value());
        expect(fixture.template connect<"out", "in">(gate, sink).has_value());
        expect(fixture.template connect<"evtOut", "evtIn">(gate, states).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink.script(), completed("T:c d e f"))) << domain << ": opened by the tag on c";
        expect(states.names() == std::vector<std::string>{"opened"}) << domain << ": and says so once";

        gr::testing::GraphFixture injected;
        auto&                     stream = injected.emplace<MarbleSource<float>>({{"script", std::string("a b c d e f |")}, {"sample_values", values}});
        auto&                     script = injected.emplace<EventScript>();
        script._events                   = {eventNamed("start", 1U)};
        auto& eventGate                  = injected.emplace<Gate<float>>({{"mode", std::string("skip_until")}, {"open_filter", std::string("start")}, {"gr:compute_domain", placed}});
        auto& passed                     = injected.emplace<MarbleSink<float>>({{"sample_values", values}});
        expect(injected.connect<"out", "in">(stream, eventGate).has_value());
        expect(injected.connect<"evtOut", "evtIn">(script, eventGate).has_value());
        expect(injected.connect<"out", "in">(eventGate, passed).has_value());
        expect(injected.run().has_value()) << domain;
        expect(eq(passed._samples.empty(), false)) << domain << ": an injected event opens it too";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "the tag/event bridge and the watchdog run on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& source   = fixture.template emplace<MarbleSource<float>>({{"script", std::string("T:a b c d |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto& watchdog = fixture.template emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}});
        auto& bridge   = fixture.template emplace<TagToEvent<float>>({{"filter", std::string("start")}});
        auto& sink     = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.template connect<"out", "in">(watchdog, bridge).has_value());
        expect(fixture.template connect<"out", "in">(bridge, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink._samples.size(), 4UZ)) << domain << ": the stream passes through untouched";
        expect(eq(watchdog.n_timeouts.value, 1U)) << domain << ": one outage after the trigger stops";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a value trigger forms the same triggers on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& source  = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a e a e a |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto& trigger = fixture.template emplace<ValueTrigger<float, ValueCondition::level>>({{"threshold", 3.f}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}});
        auto& sink    = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, trigger).has_value());
        expect(fixture.template connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink._samples.size(), 5UZ)) << domain << ": the stream passes through untouched";
        expect(eq(trigger.n_triggers.value, 2U)) << domain << ": two crossings of the threshold, whichever domain ran it";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "an interval measurement answers the same on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& script   = fixture.template emplace<EventScript>();
        script._events = {eventNamed("tick", 1'000'000'000U), eventNamed("tick", 1'002'000'000U), eventNamed("tick", 1'004'000'000U)};
        auto& interval = fixture.template emplace<TimeInterval<double>>({{"mode", std::string("to_previous")}});
        auto& measured = fixture.template emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.template connect<"evtOut", "evtIn">(script, interval).has_value());
        expect(fixture.template connect<"out", "in">(interval, measured).has_value());
        expect(fixture.run().has_value()) << domain;

        expect(eq(interval.n_undated.value, 0U)) << domain << ": every event arrived dated";
        expect(eq(interval.n_intervals.value, 2U)) << domain << ": three ticks give two periods";
        expect(eq(measured._samples.size(), 2UZ)) << domain;
        if (measured._samples.size() == 2UZ) {
            expect(approx(measured._samples[0], 2e-3, 1e-12)) << domain;
            expect(approx(measured._samples[1], 2e-3, 1e-12)) << domain;
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "an edge detector feeding an interval measurement answers the same on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& source   = fixture.template emplace<MarbleSource<float>>({{"script", std::string("T:a e a e a |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto& trigger  = fixture.template emplace<ValueTrigger<float, ValueCondition::level>>({{"threshold", 3.f}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}});
        auto& spent    = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto& interval = fixture.template emplace<TimeInterval<double>>({{"mode", std::string("to_previous")}});
        auto& measured = fixture.template emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        /** an event output grants one slot per work call, so each edge needs a call of its own */
        trigger.in.max_samples = 2UZ;
        expect(fixture.template connect<"out", "in">(source, trigger).has_value());
        expect(fixture.template connect<"out", "in">(trigger, spent).has_value());
        expect(fixture.template connect<"evtOut", "evtIn">(trigger, interval).has_value());
        expect(fixture.template connect<"out", "in">(interval, measured).has_value());
        expect(fixture.run().has_value()) << domain;

        expect(eq(trigger.n_triggers.value, 2U)) << domain << ": the trigger found both edges";
        expect(eq(interval.n_undated.value, 0U)) << domain << ": every edge arrived dated";
        expect(eq(interval.n_intervals.value, 1U)) << domain << ": two edges give one period between them";
        expect(eq(measured._samples.size(), 1UZ)) << domain;
        if (measured._samples.size() == 1UZ) {
            expect(approx(measured._samples[0], 2e-3, 1e-9)) << domain << ": two samples apart at 1 kHz";
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    return 0;
}
