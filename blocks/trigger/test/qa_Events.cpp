#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/trigger/Events.hpp>

namespace primitives = gr::blocks::trigger::detail;

namespace {
/** the part of a block `reportError` uses: a name and a message output */
struct Refusals {
    struct {
        std::string value = "test";
    } name;
    std::vector<std::string> reasons;

    void emitMessage(std::string_view /*endpoint*/, const gr::property_map& message) { reasons.emplace_back(gr::property_map_view{message}.value_or<std::string_view>(primitives::kErrorKey, std::string_view{})); }
};

struct TaggedAt {
    std::size_t      index;
    gr::property_map map;
};

const auto eventTests = [] {
    using namespace boost::ut;

    "an inline text keeps what fits and forgets what it held before"_test = [] {
        primitives::InlineText text;
        text.assign("a long name");
        text.assign("short");
        expect(eq(text.view(), std::string_view{"short"}));
        text.assign(std::string(1000UZ, 'x'));
        expect(eq(text.view().size(), text.chars.size() - 1UZ)) << "truncated, still terminated";
    };

    "a held event is a copy of its image, and one beyond a device slot is refused there"_test = [] {
        const gr::property_map small = primitives::makeEvent("edge");
        primitives::HeldEvent  held;
        expect(held.empty());
        expect(held.assign<false>(gr::property_map_view{small}));
        expect(!held.empty());
        expect(eq(held.view().get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()).value_or(std::string_view{}), std::string_view{"edge"}));
        held.clear();
        expect(held.empty());

        gr::property_map large = primitives::makeEvent("edge");
        large["payload"]       = std::pmr::string(2UZ * gr::kDeviceEventSlotBytes, 'x');
        expect(!held.assign<false>(gr::property_map_view{large})) << "a device cannot grow";
        expect(held.empty());
        expect(!held.assign<true>(gr::property_map_view{large})) << "compacting does not make a payload fit";
    };

    "the tags on one sample merge with the latest winning, short keys read as long ones"_test = [] {
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, 1024UZ> scratch{};
        const std::vector<TaggedAt>                                    none;
        expect(primitives::mergedAt(none.begin(), none.end(), scratch).empty());

        const std::vector<TaggedAt> one{{3UZ, {{std::string(gr::tag::TRIGGER_NAME.key()), "only"}}}};
        expect(eq(primitives::mergedAt(one.begin(), one.end(), scratch).get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()).value_or(std::string_view{}), std::string_view{"only"})) << "a single tag is viewed as it is";

        const std::vector<TaggedAt> several{{3UZ, {{std::string(gr::tag::TRIGGER_NAME.key()), "early"}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{7U}}}}, {3UZ, {{std::string(gr::tag::TRIGGER_NAME.shortKey()), "late"}}}};
        const auto                  merged = primitives::mergedAt(several.begin(), several.end(), scratch);
        expect(eq(merged.get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()).value_or(std::string_view{}), std::string_view{"late"})) << "the later tag, under its short key";
        const auto* time = merged.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        expect(time != nullptr && *time == 7U) << "what the later tag lacks stays";
    };

    "a trigger is accepted when named and, with a filter, when it matches"_test = [] {
        const gr::property_map start{{std::string(gr::tag::TRIGGER_NAME.key()), "start"}};
        const gr::property_map other{{std::string(gr::tag::TRIGGER_NAME.key()), "other"}};
        const gr::property_map unnamed{{"x", 1}};

        primitives::MatchState state;
        Refusals               refusals;
        expect(!primitives::compileOptionalFilter(refusals, "", "pulse", state)) << "no definition, no filter";
        expect(primitives::acceptsTrigger(state, false, gr::property_map_view{other}));
        expect(!primitives::acceptsTrigger(state, false, gr::property_map_view{unnamed})) << "a tag without a trigger name is no trigger";

        expect(primitives::compileOptionalFilter(refusals, "start", "pulse", state));
        expect(primitives::acceptsTrigger(state, true, gr::property_map_view{start}));
        expect(!primitives::acceptsTrigger(state, true, gr::property_map_view{other}));

        expect(!primitives::compileOptionalFilter(refusals, "[start", "pulse", state)) << "an unmatched bracket is refused";
        expect(eq(refusals.reasons.size(), 1UZ)) << "and the refusal is reported as an error message";
        expect(!primitives::acceptsTrigger(state, true, gr::property_map_view{start})) << "and leaves no filter behind";
    };
};
} // namespace

int main() {
    using namespace boost::ut;
    "Events"_test = eventTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
