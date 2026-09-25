#include <gnuradio-4.0/TriggerMatcher.hpp>

#include <boost/ut.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

#include <gnuradio-4.0/ValueMap.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>

using namespace gr::testing;
namespace matcher = gr::trigger::BasicTriggerNameCtxMatcher;

int main() {
    constexpr std::uint32_t kKeys    = 4U;
    constexpr std::uint32_t kPayload = 128U;
    constexpr std::size_t   kSlot    = ((gr::pmt::blobBytesForKeys(kKeys, kPayload) + gr::pmt::kBlobAlignment - 1UZ) / gr::pmt::kBlobAlignment) * gr::pmt::kBlobAlignment;

    // a filter that accepts and one that does not, so a kernel answering a constant would be caught
    const std::array<std::pair<std::string_view, gr::trigger::MatchResult>, 2UZ> cases{{
        {"alarm", gr::trigger::MatchResult::Matching},
        {"other", gr::trigger::MatchResult::Ignore},
    }};

    "a compiled filter decides inside a kernel exactly as it does on the host"_domain_test = [&](auto& ctx) {
        const auto compiled = matcher::compile(std::string_view{"alarm/room1"});
        boost::ut::expect(compiled.has_value()) << "the filter under test must compile on the host";
        if (!compiled) {
            return;
        }

        std::byte*           slot  = ctx.template alloc<std::byte>(kSlot);
        matcher::MatchState* state = ctx.template alloc<matcher::MatchState>(1UZ);

        for (const auto& [triggerName, expected] : cases) {
            std::ranges::fill(std::span{slot, kSlot}, std::byte{});
            auto tagMap = gr::pmt::ValueMapView::formatAt({slot, kSlot}, kPayload, gr::pmt::entryCapacityForKeys(kKeys));
            boost::ut::expect(tagMap.try_emplace(std::string_view{gr::tag::TRIGGER_NAME.shortKey()}, triggerName));
            boost::ut::expect(tagMap.try_emplace(std::string_view{gr::tag::CONTEXT.shortKey()}, std::string_view{"room1"}));

            matcher::MatchState onHostState = compiled.value();
            const auto          onHost      = matcher::match(onHostState, gr::pmt::ValueMap::makeView(std::span<const std::byte>{slot, kSlot}));
            boost::ut::expect(boost::ut::eq(static_cast<int>(onHost), static_cast<int>(expected))) << "the host answer is the scenario's premise";

            *state             = compiled.value();
            const int wantedAs = static_cast<int>(expected);
            ctx.launch([slot, slotSize = kSlot, state, wantedAs](const DeviceTestHandle& device) {
                const auto view   = gr::pmt::ValueMap::makeView(std::span<const std::byte>{slot, slotSize});
                const int  answer = static_cast<int>(matcher::match(*state, view));
                expect(device, answer == wantedAs, "the kernel decided {}, the host {}", answer, wantedAs);
            });
        }
    } | kAllDomains;

    return 0;
}
