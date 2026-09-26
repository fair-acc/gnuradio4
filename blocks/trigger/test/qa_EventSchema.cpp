#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/Gate.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/TagBridge.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>
#include <gnuradio-4.0/trigger/TriggerWatchdog.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

namespace {
/// anything meaningful only inside the sending block, never for an outside comparison
[[nodiscard]] bool carriesPrivateKey(const gr::property_map& event) {
    const gr::property_map_view view{event};
    return view.contains(std::string_view{"stream_index"}) || view.contains(gr::tag::LOCAL_TIME.key());
}

[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}};
    return map;
}
/// a well-formed trigger: named, dated, and with an offset
[[nodiscard]] const gr::property_map& kTags() {
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
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(script)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     block  = fixture.emplace<TBlock>(std::move(settings));
    auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     schema = fixture.emplace<EventTap>();
    expect(fixture.connect<"out", "in">(source, block).has_value()) << what;
    expect(fixture.connect<"out", "in">(block, sink).has_value()) << what;
    expect(fixture.connect<"evtOut", "evtIn">(block, schema).has_value()) << what;
    expect(fixture.run().has_value()) << what;

    const std::size_t seen    = schema.size();
    const std::size_t named   = schema.countCarrying(gr::tag::TRIGGER_NAME.key());
    const std::size_t sourced = schema.countCarrying(std::string_view{"source"});
    const std::size_t dated   = schema.countCarrying(gr::tag::TRIGGER_TIME.key());
    const std::size_t priv    = static_cast<std::size_t>(std::ranges::count_if(schema._events, carriesPrivateKey));

    expect(gt(seen, 0UZ)) << what << ": emitted no event at all, so the schema is untested";
    expect(eq(named, seen)) << what << ": every event names its trigger";
    expect(eq(sourced, seen)) << what << ": every event names its sender";
    expect(eq(priv, 0UZ)) << what << ": an event carries nothing private to its sender";
    expect(eq(dated, seen)) << what << ": every event carries the time it refers to";
}
} // namespace

const boost::ut::suite<"event schema"> _schema = [] {
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
        checkEmitter<TagToMessage<float>>("TagToMessage", {{"filter", std::string("start")}});
    };

    "a watchdog dates its outage and its recovery"_test = [] { //
        checkEmitter<TriggerWatchdog<float>>("TriggerWatchdog", {{"filter", std::string("start")}, {"timeout_samples", 2U}, {"sample_rate", 1000.f}}, "T:a b c d T:e f |");
    };
};

int main() { /* tests are statically executed */ }
