#include <gnuradio-4.0/TriggerMatcher.hpp>

#include <array>
#include <iostream>

std::ostream& operator<<(std::ostream& os, const gr::trigger::MatchResult& result) { return os << std::format("{}", result); }

#include <boost/ut.hpp>

const boost::ut::suite<"BasicTriggerNameCtxMatcher"> triggerTest = [] {
    using namespace boost::ut;
    using namespace gr;

    "trigger parser"_test = [] {
        using namespace std::string_literals;
        std::string triggerName;
        bool        triggerNameEnds = false;
        std::string triggerCtx;
        bool        triggerCtxEnds = false;

        "full <trigger name>/<ctx>"_test = [&] {
            expect(nothrow([&] { trigger::detail::parse("alarm/kitchen", triggerName, triggerNameEnds, triggerCtx, triggerCtxEnds); }));
            expect(eq(triggerName, "alarm"s));
            expect(eq(triggerCtx, "kitchen"s));
            expect(!triggerNameEnds);
            expect(!triggerCtxEnds);

            expect(nothrow([&] { trigger::detail::parse("^alarm/kitchen", triggerName, triggerNameEnds, triggerCtx, triggerCtxEnds); }));
            expect(eq(triggerName, "alarm"s));
            expect(eq(triggerCtx, "kitchen"s));
            expect(triggerNameEnds);
            expect(!triggerCtxEnds);

            expect(nothrow([&] { trigger::detail::parse("alarm/^kitchen", triggerName, triggerNameEnds, triggerCtx, triggerCtxEnds); }));
            expect(eq(triggerName, "alarm"s));
            expect(eq(triggerCtx, "kitchen"s));
            expect(!triggerNameEnds);
            expect(triggerCtxEnds);

            expect(nothrow([&] { trigger::detail::parse("^alarm/^kitchen", triggerName, triggerNameEnds, triggerCtx, triggerCtxEnds); }));
            expect(eq(triggerName, "alarm"s));
            expect(eq(triggerCtx, "kitchen"s));
            expect(triggerNameEnds);
            expect(triggerCtxEnds);
        };

        "<trigger name> only"_test = [&] {
            expect(nothrow([&] { trigger::detail::parse("alarm", triggerName, triggerNameEnds, triggerCtx, triggerCtxEnds); }));
            expect(eq(triggerName, "alarm"s));
            expect(eq(triggerCtx, ""s));
        };

        "/<ctx> only"_test = [&] {
            expect(nothrow([&] { trigger::detail::parse("/kitchen", triggerName, triggerNameEnds, triggerCtx, triggerCtxEnds); }));
            expect(eq(triggerName, ""s));
            expect(eq(triggerCtx, "kitchen"s));
        };

        "extraneous separator <trigger name>/<ctx>/<..>"_test = [&] {
            expect(throws([&] { trigger::detail::parse("alarm/kitchen/cabinet", triggerName, triggerNameEnds, triggerCtx, triggerCtxEnds); })); //
        };
    };

    "BasicTriggerNameCtxMatcher Tests"_test = [] {
        using namespace std::string_literals;
        using enum gr::trigger::MatchResult;
        struct OwningTestTag {
            std::size_t  index{0UZ};
            property_map map{};

            [[nodiscard]] operator Tag() const noexcept { return Tag{index, map}; }
        };
        auto createTag = [](std::string triggerName, std::string cxt) noexcept { return OwningTestTag{0UZ, {{tag::TRIGGER_NAME.shortKey(), std::move(triggerName)}, {tag::CONTEXT.shortKey(), std::move(cxt)}, {tag::TRIGGER_META_INFO.shortKey(), property_map{}}}}; };

        "trigger on room1-room3 (exclusive)"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[alarm/room1, alarm/room3]";
            property_map   state;

            expect(nothrow([&state] { expect(eq(matcher(filter, Tag{}, state), Ignore)); }));
            expect(!state.find_value("isSingleTrigger").value().value_or(true));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room2"), state), Ignore));
            expect(eq(matcher(filter, createTag("info", "room2"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), NotMatching));
            expect(eq(matcher(filter, createTag("alarm", "room4"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("info", "room2"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room2"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), NotMatching));
            expect(eq(matcher(filter, createTag("alarm", "room4"), state), Ignore));

            expect(nothrow([&state] { trigger::BasicTriggerNameCtxMatcher::reset(state); })) << "reset matcher for next scenario";
        };

        "trigger on room1-^room3 (inclusive)"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[alarm/room1, alarm/^room3]";
            property_map   state;

            expect(nothrow([&state] { expect(eq(matcher(filter, Tag{}, state), Ignore)); }));
            expect(!state.find_value("isSingleTrigger").value().value_or(true));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room2"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room4"), state), NotMatching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room2"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room4"), state), NotMatching));
        };

        "trigger on room1-^room3 (inclusive)"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[alarm/^room1, alarm/^room3]"; // implicitly resets
            property_map   state;

            expect(nothrow([&state] { expect(eq(matcher(filter, Tag{}, state), Ignore)); }));
            expect(!state.find_value("isSingleTrigger").value().value_or(true));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("info", "room2"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room4"), state), NotMatching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room2"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room4"), state), NotMatching));
        };

        "trigger on ^alarm/room1 to alarm/room3"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[^alarm/room1, alarm/room3]";
            property_map   state;

            expect(nothrow([&state] { expect(eq(matcher(filter, Tag{}, state), Ignore)); }));
            expect(!state.find_value("isSingleTrigger").value().value_or(true));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore)); // skipped due to ^alarm
            expect(eq(matcher(filter, createTag("other", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room2"), state), Ignore));
            expect(eq(matcher(filter, createTag("other", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), NotMatching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("other", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), NotMatching));
        };

        "trigger with ^alarm/^room1 to ^alarm/room3"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[^alarm/^room1, ^alarm/room3]";
            property_map   state;

            expect(nothrow([&state] { expect(eq(matcher(filter, Tag{}, state), Ignore)); }));
            expect(!state.find_value("isSingleTrigger").value().value_or(true));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore)); // skipped due to ^alarm/^room1
            expect(eq(matcher(filter, createTag("other", "room2"), state), Matching));
            expect(eq(matcher(filter, createTag("other", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room2"), state), Ignore));
            expect(eq(matcher(filter, createTag("other", "room3"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), Ignore)); // skipped due to ^alarm/^room3
            expect(eq(matcher(filter, createTag("other", "room4"), state), NotMatching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore)); // skipped due to ^alarm/^room1
            expect(eq(matcher(filter, createTag("other", "room2"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), Ignore)); // skipped due to ^alarm/^room3
            expect(eq(matcher(filter, createTag("other", "room4"), state), NotMatching));
        };

        "trigger with alarm/^room1 to alarm/^room3"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[alarm/^room1, alarm/^room3]";
            property_map   state;

            expect(nothrow([&state] { expect(eq(matcher(filter, Tag{}, state), Ignore)); }));
            expect(!state.find_value("isSingleTrigger").value().value_or(true));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room2"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room4"), state), NotMatching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room2"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room4"), state), NotMatching));
        };

        "mixed trigger conditions"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[^alarm/room1, alarm/room3]";
            property_map   state;

            expect(nothrow([&state] { expect(eq(matcher(filter, Tag{}, state), Ignore)); }));
            expect(!state.find_value("isSingleTrigger").value().value_or(true));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("other", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room2"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), NotMatching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("other", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room3"), state), NotMatching));
        };

        "single trigger 1"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[alarm/room1]";
            property_map   state;

            expect(nothrow([&state] { expect(eq(matcher(filter, Tag{}, state), Ignore)); }));
            expect(state.find_value("isSingleTrigger").value().value_or(true));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("other", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
        };

        "single trigger 2"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[, alarm/room1]"; // note: extra ',' separator
            property_map   state;

            expect(nothrow([&state] { expect(eq(matcher(filter, Tag{}, state), Ignore)); }));
            expect(state.find_value("isSingleTrigger").value().value_or(true));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("other", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(trigger::BasicTriggerNameCtxMatcher::isSingleTrigger(state));
        };

        "single trigger 3"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[alarm/room1, alarm/room1]";
            property_map   state;

            expect(nothrow([&state] { expect(eq(matcher(filter, Tag{}, state), Ignore)); }));
            expect(state.find_value("isSingleTrigger").value().value_or(true));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("other", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
        };

        "a '*' matches any run of a name and a leading '!' selects what the rest does not, alike in both matchers"_test = [&] {
            struct Case {
                std::string_view     filter;
                std::string_view     name;
                std::string_view     context;
                trigger::MatchResult expected;
            };
            constexpr std::array kCases{
                Case{"beam_*", "beam_in", "", Matching}, Case{"beam_*", "beam_", "", Matching}, Case{"beam_*", "rf_on", "", Ignore},          //
                Case{"*_in", "beam_in", "", Matching}, Case{"b*m*n", "beam_in", "", Matching}, Case{"b*m*n", "beam_out", "", Ignore},         //
                Case{"!beam_in", "beam_out", "", Matching}, Case{"!beam_in", "beam_in", "", Ignore}, Case{"!beam_*", "beam_out", "", Ignore}, //
                Case{"beam_*/!bad", "beam_in", "good", Matching}, Case{"beam_*/!bad", "beam_in", "bad", Ignore},                              //
                Case{"^beam_*", "beam_in", "", Matching}, Case{"beam_in", "beam_in", "", Matching}, Case{"beam_in", "beam_inx", "", Ignore},  //
            };
            for (const Case& c : kCases) {
                property_map legacyState;
                const auto   legacy   = trigger::BasicTriggerNameCtxMatcher::filter(c.filter, createTag(std::string(c.name), std::string(c.context)), legacyState);
                auto         compiled = trigger::BasicTriggerNameCtxMatcher::compile(c.filter);
                expect(fatal(compiled.has_value())) << c.filter;
                const OwningTestTag tag = createTag(std::string(c.name), std::string(c.context));
                expect(eq(legacy, c.expected)) << "filter" << c.filter << "on" << c.name << "/" << c.context << ": property-map matcher";
                expect(eq(trigger::BasicTriggerNameCtxMatcher::match(*compiled, tag.map), c.expected)) << "filter" << c.filter << "on" << c.name << "/" << c.context << ": compiled matcher";
            }
        };

        "an interval opens on any name its start pattern matches and closes on its stop"_test = [&] {
            auto compiled = trigger::BasicTriggerNameCtxMatcher::compile("[start_*, stop]");
            expect(fatal(compiled.has_value()));
            property_map legacyState;
            for (const auto& [name, expected] : std::array{std::pair{"start_a", Matching}, std::pair{"start_b", Ignore}, std::pair{"stop", NotMatching}, std::pair{"other", Ignore}, std::pair{"start_b", Matching}}) {
                const OwningTestTag tag = createTag(name, "");
                expect(eq(trigger::BasicTriggerNameCtxMatcher::filter("[start_*, stop]", tag, legacyState), expected)) << name << ": property-map matcher";
                expect(eq(trigger::BasicTriggerNameCtxMatcher::match(*compiled, tag.map), expected)) << name << ": compiled matcher";
            }
        };

        "only a bare trigger name is a plain name a kernel can compare by hash"_test = [] {
            for (const auto& [filter, plain] : std::array{std::pair{"edge", true}, std::pair{"edge/calibration", false}, std::pair{"beam_*", false}, std::pair{"!noise", false}, std::pair{"[start, stop]", false}, std::pair{"^edge", false}}) {
                const auto compiled = trigger::BasicTriggerNameCtxMatcher::compile(filter);
                expect(fatal(compiled.has_value())) << filter;
                expect(eq(compiled->isPlainName(), plain)) << filter;
            }
        };
    };
};

int main() { /* not needed for UT */ }
