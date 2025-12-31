#ifndef GNURADIO_TRIGGERMATCHER_HPP
#define GNURADIO_TRIGGERMATCHER_HPP

#include "gnuradio-4.0/Message.hpp"
#include "gnuradio-4.0/Tag.hpp"
#include "gnuradio-4.0/meta/formatter.hpp"

namespace gr::trigger {

constexpr inline std::string_view SEPARATOR = "/";
constexpr inline std::string_view RANGE_OP  = ",";

enum class MatchResult {
    Matching,    ///< tag matches
    NotMatching, ///< tag does not match
    Ignore       ///< Ignore tag
};

/**
 * Used for testing whether a tag should trigger data acquisition.
 *
 * For the 'Triggered' (data window) and 'Snapshot' (single sample) acquisition modes:
 * Stateless predicate to check whether a tag matches the trigger criteria.
 *
 * @code
 * auto matcher = [](std::string_view optionalCriteria, const auto &tag, property_map& filterState) {
 *     const auto isTrigger = ...check if tag is trigger...;
 *     return isTrigger ? trigger::MatchResult::Matching : trigger::MatchResult::Ignore;
 * };
 * @endcode
 *
 * For the 'Multiplexed' acquisition mode: Possibly stateful object checking all incoming tags to control which data should be sent
 * to the listener.
 *
 * A new dataset is started when the matcher returns @c Start or @c StopAndStart.
 * A dataset is closed and sent when @c Stop or @StopAndStart is returned.
 *
 * For the multiplexed case, the matcher might be stateful and can rely on being called with each incoming tag exactly once, in the order they arrive.
 *
 * Example:
 *
 * @code
 * // matcher observing three possible tag values, "green", "yellow", "red".
 * // starting a dataset when seeing "green", stopping on "red", starting a new dataset on "yellow"
 * struct ColorMatcher {
 *     trigger::MatcherResult operator()(std::string_view optionalCriteria, const auto &tag, property_map& filterState) {
 *         if (tag == green || tag == yellow) {
 *             return trigger::MatchResult::Matching;
 *         }
 *         if (tag == red) {
 *             return trigger::MatchResult::NotMatching;
 *         }
 *
 *         return trigger::MatchResult::Ignore;
 *     }
 * };
 * @endcode
 *
 * @see trigger::MatchResult
 */
template<typename T>
concept Matcher = requires(T matcher, std::string_view filterDefinition, const Tag& tag, property_map& filterState) {
    { matcher(filterDefinition, tag, filterState) } -> std::convertible_to<trigger::MatchResult>;
} or requires(T matcher, std::string_view filterDefinition, const Tag& tag, const property_map& filterState) {
    { matcher(filterDefinition, tag, filterState) } -> std::convertible_to<trigger::MatchResult>;
};

namespace detail {
[[nodiscard]] inline constexpr std::string_view trim(std::string_view str) noexcept {
    const auto first = str.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string_view::npos) {
        return ""; // all whitespace
    }

    const auto last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, last - first + 1);
}

inline void parse(const std::string_view& trigger, std::string& triggerName, bool& triggerNameEnds, std::string& triggerCtx, bool& triggerCtxEnds) {
    if (const size_t first_pos = trigger.find(SEPARATOR); first_pos != std::string::npos) {
        if (trigger.find(SEPARATOR, first_pos + SEPARATOR.size()) != std::string::npos) {
            throw gr::exception(std::format("invalid trigger input: multiple '{}' separators found: '{}'", SEPARATOR, trigger));
        }

        // separator found, split the trigger name and context
        triggerName = trim(trigger.substr(0, first_pos));
        triggerCtx  = trim(trigger.substr(first_pos + SEPARATOR.size()));
        if (triggerNameEnds = triggerName.starts_with('^'); triggerNameEnds) {
            triggerName = trim(triggerName.substr(1));
        }
        if (triggerCtxEnds = triggerCtx.starts_with('^'); triggerCtxEnds) {
            triggerCtx = trim(triggerCtx.substr(1));
        }
    } else { // no separator, the whole string is considered as trigger name
        triggerName = trim(trigger);
        if (triggerNameEnds = triggerName.starts_with('^'); triggerNameEnds) {
            triggerName = trim(triggerName.substr(1));
        }
        triggerCtx = "";
    }
};
} // namespace detail

namespace BasicTriggerNameCtxMatcher {

namespace key {
constexpr const char* kFilter                  = "filter";
constexpr const char* kStartDefined            = "startDefined";
constexpr const char* kStopDefined             = "stopDefined";
constexpr const char* kStartTriggerName        = "startTriggerName";
constexpr const char* kStartCtx                = "startCtx";
constexpr const char* kStopTriggerName         = "stopTriggerName";
constexpr const char* kStopCtx                 = "stopCtx";
constexpr const char* kStartTriggerNameEnds    = "startTriggerNameEnds";
constexpr const char* kStartCtxEnds            = "startCtxEnds";
constexpr const char* kStopTriggerNameEnds     = "stopTriggerNameEnds";
constexpr const char* kStopCtxEnds             = "stopCtxEnds";
constexpr const char* kTriggerActive           = "triggerActive";
constexpr const char* kWaitingForStartNonMatch = "waitingForStartNonMatch";
constexpr const char* kWaitingForStopNonMatch  = "waitingForStopNonMatch";
constexpr const char* kIsSingleTrigger         = "isSingleTrigger";
} // namespace key

inline void reset(property_map& state) noexcept {
    state[key::kTriggerActive]           = false;
    state[key::kWaitingForStartNonMatch] = false;
    state[key::kWaitingForStopNonMatch]  = false;
}

[[nodiscard]] inline bool isSingleTrigger(const property_map& state) noexcept {
    if (state.contains(key::kStartDefined) && state.contains(key::kStopDefined)) {
        const auto startDefined = CAP{state.at(key::kStartDefined).get_if<bool>()};
        const auto endDefined   = CAP{state.at(key::kStopDefined).get_if<bool>()};
        return startDefined != nullptr && endDefined != nullptr && (*startDefined xor *endDefined);
    } else {
        return false;
    }
}

inline void verifyFilterState(std::string_view matchCriteria, property_map& state) {
    using namespace std::string_literals;
    using namespace gr::trigger::detail;
    if (state.contains(key::kFilter) && //
        (state.at(key::kFilter).value_or(std::string_view{}) == matchCriteria)) {
        return;
    }

    state[key::kFilter] = std::string(matchCriteria);

    // default config
    state[key::kStartDefined]         = false;
    state[key::kStopDefined]          = false;
    state[key::kStartTriggerName]     = ""s;
    state[key::kStartCtx]             = ""s;
    state[key::kStopTriggerName]      = ""s;
    state[key::kStopCtx]              = ""s;
    state[key::kStartTriggerNameEnds] = false;
    state[key::kStartCtxEnds]         = false;
    state[key::kStopTriggerNameEnds]  = false;
    state[key::kStopCtxEnds]          = false;

    // reset state
    reset(state);

    if (matchCriteria.empty()) {
        return;
    }
    std::string_view criteria = matchCriteria;
    if ((criteria.front() == '[') && (criteria.back() == ']')) { // strip surrounding brackets if needed
        criteria = criteria.substr(1, criteria.size() - 2);
    } else if ((criteria.front() == '[') xor (criteria.back() == ']')) {
        throw gr::exception(std::format("unmatched bracket pair: '{}'", criteria));
    }

    std::string_view startPart;
    std::string_view stopPart;
    if (const size_t arrowPos = criteria.find(RANGE_OP); arrowPos != std::string_view::npos) {
        startPart = detail::trim(criteria.substr(0, arrowPos));
        stopPart  = detail::trim(criteria.substr(arrowPos + RANGE_OP.size()));
    } else {
        startPart = criteria;
    }

    if (!startPart.empty()) {
        const auto _startTriggerName    = state[key::kStartTriggerName].value_or(std::string_view{});
        const auto startTriggerNameEnds = CAP{state[key::kStartTriggerNameEnds].get_if<bool>()};
        const auto _startCtx            = state[key::kStartCtx].value_or(std::string_view{});
        const auto startCtxEnds         = CAP{state[key::kStartCtxEnds].get_if<bool>()};
        if (_startTriggerName.data() == nullptr || startTriggerNameEnds == nullptr || _startCtx.data() == nullptr || startCtxEnds == nullptr) {
            return;
        }
        std::string startTriggerName(_startTriggerName);
        std::string startCtx(_startCtx);
        detail::parse(startPart, startTriggerName, *startTriggerNameEnds, startCtx, *startCtxEnds);
        state[key::kStartTriggerName] = std::move(startTriggerName);
        state[key::kStartCtx]         = std::move(startCtx);
        state[key::kStartDefined]     = true;
    } else {
        state[key::kStartTriggerName] = ""s;
        state[key::kStartCtx]         = ""s;
    }

    if (!stopPart.empty()) {
        const auto _stopTriggerName    = state[key::kStopTriggerName].value_or(std::string_view{});
        const auto stopTriggerNameEnds = CAP{state[key::kStopTriggerNameEnds].get_if<bool>()};
        const auto _stopCtx            = state[key::kStopCtx].value_or(std::string_view{});
        const auto stopCtxEnds         = CAP{state[key::kStopCtxEnds].get_if<bool>()};
        if (_stopTriggerName.data() == nullptr || stopTriggerNameEnds == nullptr || _stopCtx.data() == nullptr || stopCtxEnds == nullptr) {
            return;
        }
        std::string stopTriggerName(_stopTriggerName);
        std::string stopCtx(_stopCtx);
        detail::parse(stopPart, stopTriggerName, *stopTriggerNameEnds, stopCtx, *stopCtxEnds);
        state[key::kStopTriggerName] = std::move(stopTriggerName);
        state[key::kStopCtx]         = std::move(stopCtx);
        state[key::kStopDefined]     = true;
    } else {
        state[key::kStopTriggerName] = ""s;
        state[key::kStopCtx]         = ""s;
    }

    if (isSingleTrigger(state) && /* checked in isSingleTrigger*/ *state[key::kStopDefined].get_if<bool>()) {
        std::swap(state[key::kStartTriggerName], state[key::kStopTriggerName]);
        std::swap(state[key::kStartCtx], state[key::kStopCtx]);
        state[key::kStopTriggerName] = ""s;
        state[key::kStopCtx]         = ""s;
    }

    const auto startTriggerName = state[key::kStartTriggerName].value_or(std::string_view{});
    const auto stopTriggerName  = state[key::kStopTriggerName].value_or(std::string_view{});
    const auto startCtx         = state[key::kStartCtx].value_or(std::string_view{});
    const auto stopCtx          = state[key::kStopCtx].value_or(std::string_view{});
    assert(startTriggerName.data() && stopTriggerName.data() && startCtx.data() && stopCtx.data());
    if (startTriggerName == stopTriggerName && startCtx == stopCtx) {
        state[key::kStartDefined]    = true;
        state[key::kStopDefined]     = false;
        state[key::kStopTriggerName] = ""s;
        state[key::kStopCtx]         = ""s;
    }

    const auto startDefined      = CAP{state[key::kStartDefined].get_if<bool>()};
    const auto stopDefined       = CAP{state[key::kStopDefined].get_if<bool>()};
    state[key::kIsSingleTrigger] = startDefined != nullptr && stopDefined != nullptr && (*startDefined xor *stopDefined);
}

[[nodiscard]] inline trigger::MatchResult filter(std::string_view filterDefinition, const Tag& tag, property_map& filterState) {
    using namespace gr::trigger::detail;
    verifyFilterState(filterDefinition, filterState); // N.B. automatically generates config and state variables if needed

    const auto startDefined = CAP{filterState[key::kStartDefined].get_if<bool>()};
    const auto stopDefined  = CAP{filterState[key::kStopDefined].get_if<bool>()};
    if (startDefined == nullptr || stopDefined == nullptr || (!*startDefined && !*stopDefined) || tag.map.empty()) {
        return trigger::MatchResult::Ignore;
    }
    const auto triggerActive    = CAP{filterState[key::kTriggerActive].get_if<bool>()};
    const auto startTriggerName = filterState[key::kStartTriggerName].value_or(std::string_view{});
    const auto startCtx         = filterState[key::kStartCtx].value_or(std::string_view{});
    const auto stopTriggerName  = filterState[key::kStopTriggerName].value_or(std::string_view{});
    const auto stopCtx          = filterState[key::kStopCtx].value_or(std::string_view{});
    if (triggerActive == nullptr || startTriggerName.data() == nullptr || startCtx.data() == nullptr || stopTriggerName.data() == nullptr || stopCtx.data() == nullptr) {
        return trigger::MatchResult::Ignore;
    }
    /// std::println("filter {} -> '{}'/'{}' & '{}'/'{}'\nfilter state: {}", filterDefinition, startTriggerName, startCtx, stopTriggerName, stopCtx, filterState);

    std::string triggerName;
    std::string triggerCtx;
    if (tag.map.contains(tag::TRIGGER_NAME.shortKey()) && tag.map.at(tag::TRIGGER_NAME.shortKey()).holds<std::string>()) {
        const auto str = tag.map.at(tag::TRIGGER_NAME.shortKey()).value_or(std::string_view{});
        if (str.data() != nullptr) {
            triggerName = str;
        }
    }
    if (tag.map.contains(tag::CONTEXT.shortKey()) && tag.map.at(tag::CONTEXT.shortKey()).holds<std::string>()) {
        const auto str = tag.map.at(tag::CONTEXT.shortKey()).value_or(std::string_view{});
        if (str.data()) {
            triggerCtx = str;
        }
    }

    if (isSingleTrigger(filterState)) {
        bool triggerMatch = startTriggerName.empty() || triggerName == startTriggerName;
        bool contextMatch = startCtx.empty() || startCtx.contains(triggerCtx);
        if (triggerMatch && contextMatch) {
            const auto startTriggerNameEnds            = CAP{filterState[key::kStartTriggerNameEnds].get_if<bool>()};
            const auto startCtxEnds                    = CAP{filterState[key::kStartCtxEnds].get_if<bool>()};
            filterState[key::kWaitingForStartNonMatch] = startTriggerNameEnds != nullptr && startCtxEnds != nullptr && (*startTriggerNameEnds || *startCtxEnds);
            return MatchResult::Matching;
        }
    }

    if (*startDefined && *stopDefined) {
        auto waitingForStartNonMatch = [&] {
            auto ptr = CAP{filterState[key::kWaitingForStartNonMatch].get_if<bool>()};
            return ptr != nullptr && *ptr;
        };
        auto waitingForStopNonMatch = [&] {
            auto ptr = CAP{filterState[key::kWaitingForStopNonMatch].get_if<bool>()};
            return ptr != nullptr && *ptr;
        };
        if (!*triggerActive || waitingForStartNonMatch()) {
            const bool triggerMatch = startTriggerName.empty() || triggerName == startTriggerName;
            const bool contextMatch = startCtx.empty() || triggerCtx.contains(startCtx);

            if (triggerMatch && contextMatch) {
                filterState[key::kTriggerActive]           = true;
                const auto startTriggerNameEnds            = CAP{filterState[key::kStartTriggerNameEnds].get_if<bool>()};
                const auto startCtxEnds                    = CAP{filterState[key::kStartCtxEnds].get_if<bool>()};
                filterState[key::kWaitingForStartNonMatch] = startTriggerNameEnds != nullptr && startCtxEnds != nullptr && (*startTriggerNameEnds || *startCtxEnds);
                return waitingForStartNonMatch() ? MatchResult::Ignore : MatchResult::Matching;
            } else if (waitingForStartNonMatch()) {
                filterState[key::kWaitingForStartNonMatch] = false;
                return MatchResult::Matching;
            }
        } else {
            const bool triggerMatch = stopTriggerName.empty() || triggerName == stopTriggerName;
            const bool contextMatch = stopCtx.empty() || triggerCtx.contains(stopCtx);

            if ((triggerMatch && contextMatch) || waitingForStopNonMatch()) {
                const auto stopTriggerNameEnds            = CAP{filterState[key::kStopTriggerNameEnds].get_if<bool>()};
                const auto stopCtxEnds                    = CAP{filterState[key::kStopCtxEnds].get_if<bool>()};
                filterState[key::kWaitingForStopNonMatch] = stopTriggerNameEnds != nullptr && stopCtxEnds != nullptr && (*stopTriggerNameEnds || *stopCtxEnds);
                if (!waitingForStopNonMatch()) {
                    reset(filterState);
                    return MatchResult::NotMatching;
                } else if (!triggerMatch || !contextMatch) {
                    reset(filterState);
                    return MatchResult::NotMatching;
                }
                return MatchResult::Ignore;
            }
        }
    }

    return trigger::MatchResult::Ignore;
}

static_assert(Matcher<decltype(&filter)>);

struct Filter {
    [[nodiscard]] inline trigger::MatchResult operator()(std::string_view filterDefinition, const Tag& tag, property_map& filterState) const { return filter(filterDefinition, tag, filterState); }
};

static_assert(Matcher<Filter>);

} // namespace BasicTriggerNameCtxMatcher

} // namespace gr::trigger

template<>
struct std::formatter<gr::trigger::MatchResult> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.begin(); }

    // Formats the source_location, using 'f' for file and 'l' for line
    template<typename FormatContext>
    auto format(const gr::trigger::MatchResult& ret, FormatContext& ctx) const -> decltype(ctx.out()) {
        return std::format_to(ctx.out(), "{}", magic_enum::enum_name(ret));
    }
};

#endif // GNURADIO_TRIGGERMATCHER_HPP
