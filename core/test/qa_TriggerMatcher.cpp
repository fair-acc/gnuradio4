#include <gnuradio-4.0/TriggerMatcher.hpp>

#include <iostream>

std::ostream& operator<<(std::ostream& os, const gr::trigger::MatchResult& result) { return os << fmt::format("{}", result); }

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
        constexpr auto createTag = [](std::string triggerName, std::string cxt) noexcept {
            auto meta = property_map{{tag::CONTEXT.shortKey(), cxt}};
            return Tag(0, {{tag::TRIGGER_NAME.shortKey(), triggerName}, {tag::TRIGGER_META_INFO.shortKey(), meta}});
        };

        "trigger on room1-room3 (exclusive)"_test = [&] {
            auto&          matcher = trigger::BasicTriggerNameCtxMatcher::filter;
            constexpr auto filter  = "[alarm/room1, alarm/room3]";
            property_map   state;

            expect(nothrow([&state] { matcher(filter, Tag{}, state); }));
            expect(!std::get<bool>(state.at("isSingleTrigger")));

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

            expect(nothrow([&state] { matcher(filter, Tag{}, state); }));
            expect(!std::get<bool>(state.at("isSingleTrigger")));

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

            expect(nothrow([&state] { matcher(filter, Tag{}, state); }));
            expect(!std::get<bool>(state.at("isSingleTrigger")));

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

            expect(nothrow([&state] { matcher(filter, Tag{}, state); }));
            expect(!std::get<bool>(state.at("isSingleTrigger")));

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

            expect(nothrow([&state] { matcher(filter, Tag{}, state); }));
            expect(!std::get<bool>(state.at("isSingleTrigger")));

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

            expect(nothrow([&state] { matcher(filter, Tag{}, state); }));
            expect(!std::get<bool>(state.at("isSingleTrigger")));

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

            expect(nothrow([&state] { matcher(filter, Tag{}, state); }));
            expect(!std::get<bool>(state.at("isSingleTrigger")));

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

            expect(nothrow([&state] { matcher(filter, Tag{}, state); }));
            expect(std::get<bool>(state.at("isSingleTrigger")));

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

            expect(nothrow([&state] { matcher(filter, Tag{}, state); }));
            expect(std::get<bool>(state.at("isSingleTrigger")));

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

            expect(nothrow([&state] { matcher(filter, Tag{}, state); }));
            expect(std::get<bool>(state.at("isSingleTrigger")));

            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("other", "room1"), state), Ignore));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
            expect(eq(matcher(filter, createTag("alarm", "room1"), state), Matching));
        };
    };
};

int main() { /* not needed for UT */ }
