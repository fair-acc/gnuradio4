#ifndef GRAPH_PROTOTYPE_TIMINGCTX_HPP
#define GRAPH_PROTOTYPE_TIMINGCTX_HPP

#include <algorithm>
#include <annotated.hpp>
#include <cctype>
#include <charconv>
#include <chrono>
#include <exception>
#include <functional>
#include <map>
#include <pmtv/pmt.hpp>
#include <ranges>
#include <string>
#include <string_view>

namespace fair::graph {

namespace detail {
template<class T>
inline constexpr void
hash_combine(std::size_t &seed, const T &v) noexcept {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

constexpr uint32_t
const_hash(char const *input) noexcept {
    return *input ? static_cast<uint32_t>(*input) + 33 * const_hash(input + 1) : 5381;
} // NOLINT

constexpr std::string_view TIMING_DESCRIPTION = "Dummy timing selector";
} // namespace detail

static auto nullParsePred = [](auto) {};
static auto nullMatchPred = [](auto, auto) { return true; };

class TimingCtx {
    using timestamp                  = std::chrono::system_clock::time_point;
    using ParsePredicate             = std::function<void(const TimingCtx *)>;
    using MatchPredicate             = std::function<bool(const TimingCtx &, const TimingCtx &)>;
    constexpr static auto DEFAULT    = "";
    constexpr static auto EMPTY_HASH = 0;
    ParsePredicate        _parse_pred;
    MatchPredicate        _match_pred;

public:
    Annotated<std::string, "Timing selector">       selector = static_cast<std::string>(DEFAULT);
    Annotated<long, "start time-stamp", Unit<"µs">> bpcts    = 0;

    TimingCtx(ParsePredicate parsePred, MatchPredicate matchPred, const std::string_view &selectorToken, std::chrono::microseconds bpcTimeStamp = {})
        : _parse_pred(parsePred), _match_pred(matchPred), selector(toUpper(selectorToken)), bpcts(bpcTimeStamp.count()) {
        parse();
    }

    explicit TimingCtx(ParsePredicate parsePred = nullParsePred, MatchPredicate matchPred = nullMatchPred, pmtv::map_t identifier = {}, std::chrono::microseconds bpcTimeStamp = {})
        : _parse_pred(parsePred), _match_pred(matchPred), selector(), bpcts(bpcTimeStamp.count()), _identifier(identifier) {
        parse();
    }

    mutable std::size_t _hash = EMPTY_HASH; // to distinguish already parsed selectors
    mutable pmtv::map_t _identifier;

    // clang-format off
    [[nodiscard]] pmtv::map_t identifier() { parse(); return _identifier; }

    // these are not commutative, and must not be confused with operator==
    [[nodiscard]] bool matches(const TimingCtx &other) const { parse(); return _match_pred(*this, other); }

    [[nodiscard]] bool matchesWithBpcts(const TimingCtx &other) const {
        auto match = [](const long lhs, const long rhs) { return rhs == 0 || lhs == rhs; };
        parse();
        return match(bpcts, other.bpcts) && matches(other);
    }

    // clang-format on

    [[nodiscard]] bool
    operator==(const TimingCtx &other) const {
        parse();
        return bpcts == other.bpcts && _identifier == other._identifier;
    }

    template<bool forceParse = true>
    [[nodiscard]] std::string
    toString() const noexcept(forceParse) {
        if constexpr (forceParse) {
            parse();
        }

        std::vector<std::string> segments;
        segments.reserve(_identifier.size());
        for (const auto &[key, val] : _identifier) {
            segments.emplace_back(fmt::format("{}={}", key, val));
        }
        return fmt::format("{}", fmt::join(segments, ":"));
    }

    [[nodiscard]] std::size_t
    hash() const noexcept {
        parse();
        std::size_t seed = 0;
        for (const auto &[key, val] : _identifier) {
            detail::hash_combine(seed, key);
            detail::hash_combine(seed, pmtv::to_base64(val));
        }
        detail::hash_combine(seed, bpcts.value);
        return seed;
    }

    void
    parse() const {
        // lazy revaluation in case selector changed -- not mathematically perfect but should be sufficient given the limited/constraint selector syntax
        const size_t selectorHash = detail::const_hash(selector.value.data());
        if (_hash == selectorHash) {
            return;
        }

        _parse_pred(this);

        _hash = selectorHash;
    }

private:
    static inline std::string
    toUpper(const std::string_view &mixedCase) noexcept {
        std::string retval;
        retval.resize(mixedCase.size());
        std::transform(mixedCase.begin(), mixedCase.end(), retval.begin(), [](unsigned char c) noexcept { return std::toupper(c); });
        return retval;
    }
};

inline static const auto NullTimingCtx = TimingCtx{ nullParsePred, nullMatchPred };

[[nodiscard]] inline bool
operator==(const TimingCtx &lhs, const std::string_view &rhs) {
    return (lhs.bpcts == 0) && (lhs.selector.value == rhs);
}

} // namespace fair::graph

namespace std {
template<>
struct hash<fair::graph::TimingCtx> {
    [[nodiscard]] size_t
    operator()(const fair::graph::TimingCtx &ctx) const noexcept {
        return ctx.hash();
    }
};
} // namespace std

template<>
struct fmt::formatter<fair::graph::TimingCtx> {
    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        return ctx.begin(); // not (yet) implemented
    }

    template<typename FormatContext>
    auto
    format(const fair::graph::TimingCtx &v, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "{}", v.toString());
    }
};

namespace fair::graph {
inline std::ostream &
operator<<(std::ostream &os, const fair::graph::TimingCtx &v) {
    return os << fmt::format("{}", v);
}
} // namespace fair::graph

#endif // GRAPH_PROTOTYPE_TIMINGCTX_HPP
