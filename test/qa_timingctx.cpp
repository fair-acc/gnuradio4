#include <boost/ut.hpp>

#include "pmtv/pmt.hpp"
#include "timingctx.hpp"

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

namespace fair::graph::timingctx_test {

static inline std::string
toUpper(const std::string_view &mixedCase) noexcept {
    std::string retval;
    retval.resize(mixedCase.size());
    std::transform(mixedCase.begin(), mixedCase.end(), retval.begin(), [](unsigned char c) noexcept { return std::toupper(c); });
    return retval;
}

constexpr static auto WILDCARD        = std::string_view("ALL");
constexpr static auto WILDCARD_VALUE  = -1;
constexpr static auto SELECTOR_PREFIX = std::string_view("FAIR.SELECTOR.");

[[nodiscard]] static constexpr bool
isWildcard(int x) noexcept {
    return x == WILDCARD_VALUE;
}

[[nodiscard]] static constexpr bool
wildcardMatch(auto lhs, auto rhs) {
    const auto l = pmtv::cast<int>(lhs);
    const auto r = pmtv::cast<int>(rhs);
    return isWildcard(r) || l == r;
}

auto parsePred = [](auto t) {
    t->_identifier["cid"]        = WILDCARD_VALUE;
    t->_identifier["sid"]        = WILDCARD_VALUE;
    t->_identifier["pid"]        = WILDCARD_VALUE;
    t->_identifier["gid"]        = WILDCARD_VALUE;
    const auto upperCaseSelector = toUpper(t->selector.value);
    if (upperCaseSelector.empty() || upperCaseSelector == WILDCARD) {
        return;
    }
    if (!upperCaseSelector.starts_with(SELECTOR_PREFIX)) {
        throw std::invalid_argument(fmt::format("Invalid tag '{}'", t->selector.value));
    }
    auto upperCaseSelectorView = std::string_view{ upperCaseSelector.data() + SELECTOR_PREFIX.length(), upperCaseSelector.size() - SELECTOR_PREFIX.length() };
    if (upperCaseSelectorView == WILDCARD) {
        return;
    }

    while (true) {
        const auto posColon = upperCaseSelectorView.find(':');
        const auto tag      = posColon != std::string_view::npos ? upperCaseSelectorView.substr(0, posColon) : upperCaseSelectorView;

        if (tag.length() < 3) {
            t->_hash = 0;
            throw std::invalid_argument(fmt::format("Invalid tag '{}'", tag));
        }

        const auto posEqual = tag.find('=');

        // there must be one char left of the '=', at least one after, and there must be only one '='
        if (posEqual != 1 || tag.find('=', posEqual + 1) != std::string_view::npos) {
            t->_hash = 0;
            throw std::invalid_argument(fmt::format("Tag has invalid format: '{}'", tag));
        }

        const auto key         = tag.substr(0, posEqual);
        const auto valueString = tag.substr(posEqual + 1, tag.length() - posEqual - 1);

        int32_t    value       = -1;

        if (WILDCARD != valueString) {
            int32_t intValue = 0;
            if (const auto result = std::from_chars(valueString.begin(), valueString.end(), intValue); result.ec == std::errc::invalid_argument) {
                t->_hash = 0;
                throw std::invalid_argument(fmt::format("Value: '{}' in '{}' is not a valid integer", valueString, tag));
            }

            value = intValue;
        }

        switch (key[0]) {
        case 'C': t->_identifier["cid"] = value; break;
        case 'S': t->_identifier["sid"] = value; break;
        case 'P': t->_identifier["pid"] = value; break;
        case 'T': t->_identifier["gid"] = value; break;
        default: t->_hash = 0; throw std::invalid_argument(fmt::format("Unknown key '{}' in '{}'.", key[0], tag));
        }

        if (posColon == std::string_view::npos) {
            // if there's no other segment, we're done
            return;
        }

        // otherwise, advance to after the ":"
        upperCaseSelectorView.remove_prefix(posColon + 1);
    }
};

auto matchPred = [](const auto lhs, const auto rhs) {
    return wildcardMatch(lhs._identifier["cid"], rhs._identifier["cid"]) && wildcardMatch(lhs._identifier["sid"], rhs._identifier["sid"])
        && wildcardMatch(lhs._identifier["pid"], rhs._identifier["pid"]) && wildcardMatch(lhs._identifier["gid"], rhs._identifier["gid"]);
};

const boost::ut::suite timingTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;

    "SimpleTimingCtx"_test = [] {
        expect(nothrow([] { TimingCtx(parsePred, matchPred); }));
        expect(nothrow([] { TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.ALL"); }));
        expect(eq(TimingCtx(parsePred, matchPred), TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.ALL")));
        expect(eq(TimingCtx(parsePred, matchPred, "ALL"), std::string("ALL")));
        expect(eq(std::string("ALL"), TimingCtx(parsePred, matchPred, "ALL")));
        expect(eq(TimingCtx(parsePred, matchPred, "all"), std::string("ALL")));
        expect(eq(TimingCtx(parsePred, matchPred, "ALL").bpcts.value, 0));
        expect(neq(TimingCtx(parsePred, matchPred, "ALL").hash(), static_cast<std::size_t>(0)));
        expect(eq(TimingCtx(parsePred, matchPred, "ALL").hash(), TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.ALL").hash()));
        expect(eq(TimingCtx(parsePred, matchPred, "ALL").hash(), TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=-1").hash()));
        expect(eq(TimingCtx(parsePred, matchPred, "ALL").hash(), TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=-1:S=-1").hash()));
        expect(eq(TimingCtx(parsePred, matchPred, "ALL").hash(), TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=-1:S=-1:P=-1").hash()));
        expect(eq(TimingCtx(parsePred, matchPred, "ALL").hash(), TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=-1:S=-1:P=-1:T=-1").hash()));
        expect(neq(TimingCtx(parsePred, matchPred, "ALL").hash(), TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=-1:S=-1:P=-1:T=0").hash()));

        auto changeMyFields = TimingCtx(parsePred, matchPred, "ALL");
        expect(changeMyFields.identifier()["cid"] == WILDCARD_VALUE);
        expect(changeMyFields.identifier()["sid"] == WILDCARD_VALUE);
        expect(changeMyFields.identifier()["pid"] == WILDCARD_VALUE);
        expect(changeMyFields.identifier()["gid"] == WILDCARD_VALUE);
        changeMyFields.selector = "FAIR.SELECTOR.C=1:S=2:P=3:T=4";
        expect(changeMyFields.identifier()["cid"] == 1);
        expect(changeMyFields.identifier()["sid"] == 2);
        expect(changeMyFields.identifier()["pid"] == 3);
        expect(changeMyFields.identifier()["gid"] == 4);
        changeMyFields.selector = "FAIR.SELECTOR.ALL";
        expect(changeMyFields.identifier()["cid"] == WILDCARD_VALUE);
        expect(changeMyFields.identifier()["sid"] == WILDCARD_VALUE);
        expect(changeMyFields.identifier()["pid"] == WILDCARD_VALUE);
        expect(changeMyFields.identifier()["gid"] == WILDCARD_VALUE);

        const auto timestamp = std::chrono::microseconds(1234);

        auto       ctx       = TimingCtx(parsePred, matchPred);
        expect(nothrow([&] { ctx = TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1:P=2:T=3", timestamp); }));
        expect(eq(ctx, TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1:P=2:T=3", timestamp)));
        expect(neq(ctx, TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1:P=2:T=3")));
        expect(neq(ctx, TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1:P=2:T=3", timestamp + std::chrono::microseconds(1))));
    };

    "MatchingTimingCtx"_test = [] {
        constexpr auto timestamp = std::chrono::microseconds(1234);
        const auto     ctx       = TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1:P=2:T=3", timestamp);
        expect(ctx.matches(ctx));
        expect(ctx.matches(TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.ALL")));
        expect(ctx.matchesWithBpcts(TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0")));
        expect(ctx.matchesWithBpcts(TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1")));
        expect(ctx.matchesWithBpcts(TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1:P=2")));
        expect(ctx.matches(TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1:P=2")));
        expect(!ctx.matches(TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=0:P=2")));
        expect(!ctx.matches(TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1:P=0")));

        expect(!TimingCtx(parsePred, matchPred).matches(TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1:P=2", timestamp)));
        expect(TimingCtx(parsePred, matchPred, "FAIR.SELECTOR.C=0:S=1:P=2", timestamp).matches(TimingCtx(parsePred, matchPred)));
    };
};

} // namespace fair::graph::timingctx_test

int
main() { /* tests are statically executed */
}
