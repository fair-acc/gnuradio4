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
            throw gr::exception(fmt::format("invalid trigger input: multiple '{}' separators found: '{}'", SEPARATOR, trigger));
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
        return std::get<bool>(state.at(key::kStartDefined)) xor std::get<bool>(state.at(key::kStopDefined));
    } else {
        return false;
    }
}

inline void verifyFilterState(std::string_view matchCriteria, property_map& state) {
    using namespace std::string_literals;
    using namespace gr::trigger::detail;
    if (state.contains(key::kFilter) && std::holds_alternative<std::string>(state.at(key::kFilter)) && (std::get<std::string>(state.at(key::kFilter)) == matchCriteria)) {
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
        throw gr::exception(fmt::format("unmatched bracket pair: '{}'", criteria));
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
        detail::parse(startPart, std::get<std::string>(state[key::kStartTriggerName]), std::get<bool>(state[key::kStartTriggerNameEnds]), std::get<std::string>(state[key::kStartCtx]), std::get<bool>(state[key::kStartCtxEnds]));
        state[key::kStartDefined] = true;
    } else {
        state[key::kStartTriggerName] = ""s;
        state[key::kStartCtx]         = ""s;
    }

    if (!stopPart.empty()) {
        detail::parse(stopPart, std::get<std::string>(state[key::kStopTriggerName]), std::get<bool>(state[key::kStopTriggerNameEnds]), std::get<std::string>(state[key::kStopCtx]), std::get<bool>(state[key::kStopCtxEnds]));
        state[key::kStopDefined] = true;
    } else {
        state[key::kStopTriggerName] = ""s;
        state[key::kStopCtx]         = ""s;
    }

    if (isSingleTrigger(state) && std::get<bool>(state[key::kStopDefined])) {
        std::swap(state[key::kStartTriggerName], state[key::kStopTriggerName]);
        std::swap(state[key::kStartCtx], state[key::kStopCtx]);
        state[key::kStopTriggerName] = ""s;
        state[key::kStopCtx]         = ""s;
    }

    if (std::get<std::string>(state[key::kStartTriggerName]) == std::get<std::string>(state[key::kStopTriggerName]) && std::get<std::string>(state[key::kStartCtx]) == std::get<std::string>(state[key::kStopCtx])) {
        state[key::kStartDefined]    = true;
        state[key::kStopDefined]     = false;
        state[key::kStopTriggerName] = ""s;
        state[key::kStopCtx]         = ""s;
    }

    state[key::kIsSingleTrigger] = bool(std::get<bool>(state.at(key::kStartDefined)) xor std::get<bool>(state.at(key::kStopDefined)));
}

[[nodiscard]] inline trigger::MatchResult filter(std::string_view filterDefinition, const Tag& tag, property_map& filterState) {
    using namespace gr::trigger::detail;
    verifyFilterState(filterDefinition, filterState); // N.B. automatically generates config and state variables if needed

    const auto startDefined = std::get<bool>(filterState[key::kStartDefined]);
    const auto stopDefined  = std::get<bool>(filterState[key::kStopDefined]);
    if ((!startDefined && !stopDefined) || tag.map.empty()) {
        return trigger::MatchResult::Ignore;
    }
    const auto triggerActive    = std::get<bool>(filterState[key::kTriggerActive]);
    const auto startTriggerName = std::get<std::string>(filterState[key::kStartTriggerName]);
    const auto startCtx         = std::get<std::string>(filterState[key::kStartCtx]);
    const auto stopTriggerName  = std::get<std::string>(filterState[key::kStopTriggerName]);
    const auto stopCtx          = std::get<std::string>(filterState[key::kStopCtx]);
    /// fmt::println("filter {} -> '{}'/'{}' & '{}'/'{}'\nfilter state: {}", filterDefinition, startTriggerName, startCtx, stopTriggerName, stopCtx, filterState);

    std::string triggerName;
    std::string triggerCtx;
    if (tag.map.contains(tag::TRIGGER_NAME.shortKey()) && std::holds_alternative<std::string>(tag.map.at(tag::TRIGGER_NAME.shortKey()))) {
        triggerName = std::get<std::string>(tag.map.at(tag::TRIGGER_NAME.shortKey()));
    }
    if (tag.map.contains(tag::CONTEXT.shortKey()) && std::holds_alternative<std::string>(tag.map.at(tag::CONTEXT.shortKey()))) {
        triggerCtx = std::get<std::string>(tag.map.at(tag::CONTEXT.shortKey()));
    }

    if (isSingleTrigger(filterState)) {
        bool triggerMatch = startTriggerName.empty() || triggerName == startTriggerName;
        bool contextMatch = startCtx.empty() || startCtx.contains(triggerCtx);
        if (triggerMatch && contextMatch) {
            filterState[key::kWaitingForStartNonMatch] = std::get<bool>(filterState[key::kStartTriggerNameEnds]) || std::get<bool>(filterState[key::kStartCtxEnds]);
            return MatchResult::Matching;
        }
    }

    if (startDefined && stopDefined) {
        if (!triggerActive || std::get<bool>(filterState[key::kWaitingForStartNonMatch])) {
            const bool triggerMatch = startTriggerName.empty() || triggerName == startTriggerName;
            const bool contextMatch = startCtx.empty() || triggerCtx.contains(startCtx);

            if (triggerMatch && contextMatch) {
                filterState[key::kTriggerActive]           = true;
                filterState[key::kWaitingForStartNonMatch] = std::get<bool>(filterState[key::kStartTriggerNameEnds]) || std::get<bool>(filterState[key::kStartCtxEnds]);
                return std::get<bool>(filterState[key::kWaitingForStartNonMatch]) ? MatchResult::Ignore : MatchResult::Matching;
            } else if (std::get<bool>(filterState[key::kWaitingForStartNonMatch])) {
                filterState[key::kWaitingForStartNonMatch] = false;
                return MatchResult::Matching;
            }
        } else {
            const bool triggerMatch = stopTriggerName.empty() || triggerName == stopTriggerName;
            const bool contextMatch = stopCtx.empty() || triggerCtx.contains(stopCtx);

            if ((triggerMatch && contextMatch) || std::get<bool>(filterState[key::kWaitingForStopNonMatch])) {
                filterState[key::kWaitingForStopNonMatch] = std::get<bool>(filterState[key::kStopTriggerNameEnds]) || std::get<bool>(filterState[key::kStopCtxEnds]);
                if (!std::get<bool>(filterState[key::kWaitingForStopNonMatch])) {
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
struct fmt::formatter<gr::trigger::MatchResult> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.begin(); }

    // Formats the source_location, using 'f' for file and 'l' for line
    template<typename FormatContext>
    auto format(const gr::trigger::MatchResult& ret, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{}", magic_enum::enum_name(ret));
    }
};

#endif // GNURADIO_TRIGGERMATCHER_HPP
