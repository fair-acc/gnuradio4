#ifndef GNURADIO_TRIGGERMATCHER_HPP
#define GNURADIO_TRIGGERMATCHER_HPP

#include <algorithm>

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

struct MatchToken {
    static constexpr std::size_t kCapacity = 64UZ;

    std::array<char, kCapacity> characters{};
    std::uint8_t                length  = 0U;
    bool                        endsRun = false;

    [[nodiscard]] constexpr std::string_view view() const noexcept { return {characters.data(), length}; }
    [[nodiscard]] constexpr bool             empty() const noexcept { return length == 0U; }

    [[nodiscard]] constexpr bool matches(std::string_view text) const noexcept { return std::ranges::equal(text, view(), gr::pmt::detail::charEquals); }
    [[nodiscard]] constexpr bool contains(std::string_view text) const noexcept { return !std::ranges::search(view(), text, gr::pmt::detail::charEquals).empty(); }
    [[nodiscard]] constexpr bool isContainedIn(std::string_view text) const noexcept { return !std::ranges::search(text, view(), gr::pmt::detail::charEquals).empty(); }

    [[nodiscard]] constexpr bool assign(std::string_view text) noexcept {
        if (text.size() > kCapacity) {
            return false;
        }
        length = static_cast<std::uint8_t>(text.size());
        for (std::size_t i = 0UZ; i < text.size(); ++i) {
            characters[i] = text[i];
        }
        return true;
    }
};

enum class MatchMode : std::uint8_t { pulse, interval };

struct MatchState {
    MatchToken start{};
    MatchToken startContext{};
    MatchToken stop{};
    MatchToken stopContext{};

    bool startDefined    = false;
    bool stopDefined     = false;
    bool isSingleTrigger = false;

    bool triggerActive           = false;
    bool waitingForStartNonMatch = false;
    bool waitingForStopNonMatch  = false;

    constexpr void reset() noexcept {
        triggerActive           = false;
        waitingForStartNonMatch = false;
        waitingForStopNonMatch  = false;
    }
};

static_assert(std::is_trivially_copyable_v<MatchState>, "a compiled filter crosses to a device by value");

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

[[nodiscard]] inline std::expected<void, Error> parseToken(std::string_view text, MatchToken& name, MatchToken& context) noexcept {
    const auto separator = text.find(SEPARATOR);
    if (separator != std::string_view::npos && text.find(SEPARATOR, separator + SEPARATOR.size()) != std::string_view::npos) {
        return std::unexpected(Error(std::format("invalid trigger input: multiple '{}' separators found: '{}'", SEPARATOR, text)));
    }

    std::string_view namePart    = separator == std::string_view::npos ? detail::trim(text) : detail::trim(text.substr(0UZ, separator));
    std::string_view contextPart = separator == std::string_view::npos ? std::string_view{} : detail::trim(text.substr(separator + SEPARATOR.size()));

    name.endsRun = namePart.starts_with('^');
    if (name.endsRun) {
        namePart = detail::trim(namePart.substr(1UZ));
    }
    context.endsRun = contextPart.starts_with('^');
    if (context.endsRun) {
        contextPart = detail::trim(contextPart.substr(1UZ));
    }
    if (!name.assign(namePart) || !context.assign(contextPart)) {
        return std::unexpected(Error(std::format("trigger name or context exceeds {} characters: '{}'", MatchToken::kCapacity, text)));
    }
    return {};
}

[[nodiscard]] inline std::expected<MatchState, Error> compile(std::string_view filterDefinition, std::optional<MatchMode> mode = std::nullopt) noexcept {
    MatchState compiled{};
    if (filterDefinition.empty()) {
        return compiled;
    }

    std::string_view criteria = filterDefinition;
    if (criteria.front() == '[' && criteria.back() == ']') {
        criteria = criteria.substr(1UZ, criteria.size() - 2UZ);
    } else if ((criteria.front() == '[') xor (criteria.back() == ']')) {
        return std::unexpected(Error(std::format("unmatched bracket pair: '{}'", criteria)));
    }

    std::string_view startPart = criteria;
    std::string_view stopPart;
    if (const auto rangeOperator = criteria.find(RANGE_OP); rangeOperator != std::string_view::npos) {
        startPart = detail::trim(criteria.substr(0UZ, rangeOperator));
        stopPart  = detail::trim(criteria.substr(rangeOperator + RANGE_OP.size()));
    }

    if (!startPart.empty()) {
        if (const auto parsed = parseToken(startPart, compiled.start, compiled.startContext); !parsed) {
            return std::unexpected(parsed.error());
        }
        compiled.startDefined = true;
    }
    if (!stopPart.empty()) {
        if (const auto parsed = parseToken(stopPart, compiled.stop, compiled.stopContext); !parsed) {
            return std::unexpected(parsed.error());
        }
        compiled.stopDefined = true;
    }

    if ((compiled.startDefined xor compiled.stopDefined) && compiled.stopDefined) { // a lone stop trigger reads as a start
        compiled.start        = compiled.stop;
        compiled.startContext = compiled.stopContext;
        compiled.stop         = MatchToken{};
        compiled.stopContext  = MatchToken{};
    }
    if (compiled.start.view() == compiled.stop.view() && compiled.startContext.view() == compiled.stopContext.view()) {
        compiled.startDefined = true;
        compiled.stopDefined  = false;
        compiled.stop         = MatchToken{};
        compiled.stopContext  = MatchToken{};
    }
    compiled.isSingleTrigger = compiled.startDefined xor compiled.stopDefined;

    if (mode == MatchMode::pulse && !compiled.isSingleTrigger) {
        return std::unexpected(Error(std::format("match_mode 'pulse' accepts only a single-trigger filter, got '{}'", filterDefinition)));
    }
    if (mode == MatchMode::interval && !(compiled.startDefined && compiled.stopDefined)) {
        return std::unexpected(Error(std::format("match_mode 'interval' requires a start and a stop trigger, got '{}'", filterDefinition)));
    }
    return compiled;
}

[[nodiscard]] inline MatchResult match(MatchState& state, const property_map_view& tagMap) noexcept {
    if ((!state.startDefined && !state.stopDefined) || tagMap.empty()) {
        return MatchResult::Ignore;
    }

    const auto readTagString = [&tagMap](const auto& tagKey) -> std::string_view {
        if (const auto prefixed = tagMap.get_if<std::string_view>(std::string_view{tagKey}); prefixed.has_value()) {
            return *prefixed;
        }
        if (const auto bare = tagMap.get_if<std::string_view>(std::string_view{tagKey.shortKey()}); bare.has_value()) {
            return *bare;
        }
        return {};
    };
    const std::string_view triggerName    = readTagString(tag::TRIGGER_NAME);
    const std::string_view triggerContext = readTagString(tag::CONTEXT);

    if (state.isSingleTrigger) {
        const bool nameMatches    = state.start.empty() || state.start.matches(triggerName);
        const bool contextMatches = state.startContext.empty() || state.startContext.contains(triggerContext);
        if (nameMatches && contextMatches) {
            state.waitingForStartNonMatch = state.start.endsRun || state.startContext.endsRun;
            return MatchResult::Matching;
        }
    }

    if (state.startDefined && state.stopDefined) {
        if (!state.triggerActive || state.waitingForStartNonMatch) {
            const bool nameMatches    = state.start.empty() || state.start.matches(triggerName);
            const bool contextMatches = state.startContext.empty() || state.startContext.isContainedIn(triggerContext);
            if (nameMatches && contextMatches) {
                state.triggerActive           = true;
                state.waitingForStartNonMatch = state.start.endsRun || state.startContext.endsRun;
                return state.waitingForStartNonMatch ? MatchResult::Ignore : MatchResult::Matching;
            }
            if (state.waitingForStartNonMatch) {
                state.waitingForStartNonMatch = false;
                return MatchResult::Matching;
            }
        } else {
            const bool nameMatches    = state.stop.empty() || state.stop.matches(triggerName);
            const bool contextMatches = state.stopContext.empty() || state.stopContext.isContainedIn(triggerContext);
            if ((nameMatches && contextMatches) || state.waitingForStopNonMatch) {
                state.waitingForStopNonMatch = state.stop.endsRun || state.stopContext.endsRun;
                if (!state.waitingForStopNonMatch || !nameMatches || !contextMatches) {
                    state.reset();
                    return MatchResult::NotMatching;
                }
                return MatchResult::Ignore;
            }
        }
    }

    return MatchResult::Ignore;
}

inline void reset(property_map& state) noexcept {
    state.insert_or_assign(key::kTriggerActive, false);
    state.insert_or_assign(key::kWaitingForStartNonMatch, false);
    state.insert_or_assign(key::kWaitingForStopNonMatch, false);
}

[[nodiscard]] inline bool isSingleTrigger(const property_map& state) noexcept {
    const auto startIt = state.find(key::kStartDefined);
    const auto endIt   = state.find(key::kStopDefined);
    if (startIt == state.end() || endIt == state.end()) {
        return false;
    }
    // bind iter-yielded Values to lvalues — get_if<bool>() yields pointers into the Value's
    // inline storage, which would dangle if the temporaries died at the end of the init expression.
    const Value startVal     = (*startIt).second;
    const Value endVal       = (*endIt).second;
    const auto  startDefined = checked_access_ptr{startVal.get_if<bool>()};
    const auto  endDefined   = checked_access_ptr{endVal.get_if<bool>()};
    return startDefined != nullptr && endDefined != nullptr && (*startDefined xor *endDefined);
}

inline void verifyFilterState(std::string_view matchCriteria, property_map& state) {
    using namespace std::string_literals;
    using namespace gr::trigger::detail;
    if (auto it = state.find(key::kFilter); it != state.end() && (*it).second.value_or(std::string_view{}) == matchCriteria) {
        return;
    }

    state.insert_or_assign(key::kFilter, std::string(matchCriteria));

    // default config
    state.insert_or_assign(key::kStartDefined, false);
    state.insert_or_assign(key::kStopDefined, false);
    state.insert_or_assign(key::kStartTriggerName, std::string{});
    state.insert_or_assign(key::kStartCtx, std::string{});
    state.insert_or_assign(key::kStopTriggerName, std::string{});
    state.insert_or_assign(key::kStopCtx, std::string{});
    state.insert_or_assign(key::kStartTriggerNameEnds, false);
    state.insert_or_assign(key::kStartCtxEnds, false);
    state.insert_or_assign(key::kStopTriggerNameEnds, false);
    state.insert_or_assign(key::kStopCtxEnds, false);

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
        std::string startTriggerName     = state[key::kStartTriggerName].value_or(std::string{});
        std::string startCtx             = state[key::kStartCtx].value_or(std::string{});
        bool        startTriggerNameEnds = state[key::kStartTriggerNameEnds].value_or(false);
        bool        startCtxEnds         = state[key::kStartCtxEnds].value_or(false);
        detail::parse(startPart, startTriggerName, startTriggerNameEnds, startCtx, startCtxEnds);
        state.insert_or_assign(key::kStartTriggerName, std::move(startTriggerName));
        state.insert_or_assign(key::kStartCtx, std::move(startCtx));
        state.insert_or_assign(key::kStartTriggerNameEnds, startTriggerNameEnds);
        state.insert_or_assign(key::kStartCtxEnds, startCtxEnds);
        state.insert_or_assign(key::kStartDefined, true);
    } else {
        state.insert_or_assign(key::kStartTriggerName, std::string{});
        state.insert_or_assign(key::kStartCtx, std::string{});
    }

    if (!stopPart.empty()) {
        std::string stopTriggerName     = state[key::kStopTriggerName].value_or(std::string{});
        std::string stopCtx             = state[key::kStopCtx].value_or(std::string{});
        bool        stopTriggerNameEnds = state[key::kStopTriggerNameEnds].value_or(false);
        bool        stopCtxEnds         = state[key::kStopCtxEnds].value_or(false);
        detail::parse(stopPart, stopTriggerName, stopTriggerNameEnds, stopCtx, stopCtxEnds);
        state.insert_or_assign(key::kStopTriggerName, std::move(stopTriggerName));
        state.insert_or_assign(key::kStopCtx, std::move(stopCtx));
        state.insert_or_assign(key::kStopTriggerNameEnds, stopTriggerNameEnds);
        state.insert_or_assign(key::kStopCtxEnds, stopCtxEnds);
        state.insert_or_assign(key::kStopDefined, true);
    } else {
        state.insert_or_assign(key::kStopTriggerName, std::string{});
        state.insert_or_assign(key::kStopCtx, std::string{});
    }

    {
        const Value stopDefSnap    = state[key::kStopDefined];
        const auto  stopDefBoolPtr = stopDefSnap.get_if<bool>();
        if (isSingleTrigger(state) && stopDefBoolPtr != nullptr && *stopDefBoolPtr) {
            // Equivalent of: swap(start, stop), then stop = "". Net effect: start=originalStop, stop=""
            std::string newStart    = state[key::kStopTriggerName].value_or(std::string{});
            std::string newStartCtx = state[key::kStopCtx].value_or(std::string{});
            state.insert_or_assign(key::kStartTriggerName, std::move(newStart));
            state.insert_or_assign(key::kStartCtx, std::move(newStartCtx));
            state.insert_or_assign(key::kStopTriggerName, std::string{});
            state.insert_or_assign(key::kStopCtx, std::string{});
        }
    }

    // Bind to owning std::string: ValueMap iter / operator[] return Value by value;
    // .value_or<string_view>() on a temporary yields a dangling view.
    const std::string startTriggerName = state[key::kStartTriggerName].value_or(std::string{});
    const std::string stopTriggerName  = state[key::kStopTriggerName].value_or(std::string{});
    const std::string startCtx         = state[key::kStartCtx].value_or(std::string{});
    const std::string stopCtx          = state[key::kStopCtx].value_or(std::string{});
    if (startTriggerName == stopTriggerName && startCtx == stopCtx) {
        state.insert_or_assign(key::kStartDefined, true);
        state.insert_or_assign(key::kStopDefined, false);
        state.insert_or_assign(key::kStopTriggerName, std::string{});
        state.insert_or_assign(key::kStopCtx, std::string{});
    }

    // Bind to lvalue Values: state[k] returns Value by value (temporary); the bool* below
    // would otherwise alias a destroyed temporary's storage.
    const Value startDefVal  = state[key::kStartDefined];
    const Value stopDefVal   = state[key::kStopDefined];
    const auto  startDefined = checked_access_ptr{startDefVal.get_if<bool>()};
    const auto  stopDefined  = checked_access_ptr{stopDefVal.get_if<bool>()};
    state.insert_or_assign(key::kIsSingleTrigger, startDefined.ptr != nullptr && stopDefined.ptr != nullptr && (*startDefined xor *stopDefined));
}

[[nodiscard]] inline trigger::MatchResult filter(std::string_view filterDefinition, const Tag& tag, property_map& filterState) {
    using namespace gr::trigger::detail;
    verifyFilterState(filterDefinition, filterState); // N.B. automatically generates config and state variables if needed

    // ValueMap: state[k] returns Value by value; copy-out the bool flags + strings before use.
    const bool startDefined = filterState[key::kStartDefined].value_or(false);
    const bool stopDefined  = filterState[key::kStopDefined].value_or(false);
    if ((!startDefined && !stopDefined) || tag.map.empty()) {
        return trigger::MatchResult::Ignore;
    }
    const bool        triggerActive    = filterState[key::kTriggerActive].value_or(false);
    const std::string startTriggerName = filterState[key::kStartTriggerName].value_or(std::string{});
    const std::string startCtx         = filterState[key::kStartCtx].value_or(std::string{});
    const std::string stopTriggerName  = filterState[key::kStopTriggerName].value_or(std::string{});
    const std::string stopCtx          = filterState[key::kStopCtx].value_or(std::string{});
    /// std::println("filter {} -> '{}'/'{}' & '{}'/'{}'\nfilter state: {}", filterDefinition, startTriggerName, startCtx, stopTriggerName, stopCtx, filterState);

    // Trigger tags travel gr:-prefixed on the wire (auto-forwarded by Block); legacy producers may still emit the bare
    // form. Read the wire form first and fall back to the bare form — matches CtxSettingsBase::contextInTag in Settings.cpp.
    auto readTagString = [&tag](const auto& defaultTag) -> std::string {
        auto it = tag.map.find(std::string_view{defaultTag});
        if (it == tag.map.end()) {
            it = tag.map.find(defaultTag.shortKey());
        }
        return it != tag.map.end() ? (*it).second.value_or(std::string{}) : std::string{}; // owning copy — ValueMap iter yields by value
    };
    const std::string triggerName = readTagString(tag::TRIGGER_NAME);
    const std::string triggerCtx  = readTagString(tag::CONTEXT);

    auto readFlag = [&](const char* k) -> bool { return filterState[k].value_or(false); };

    if (isSingleTrigger(filterState)) {
        bool triggerMatch = startTriggerName.empty() || triggerName == startTriggerName;
        bool contextMatch = startCtx.empty() || startCtx.contains(triggerCtx);
        if (triggerMatch && contextMatch) {
            const bool startTriggerNameEnds = readFlag(key::kStartTriggerNameEnds);
            const bool startCtxEnds         = readFlag(key::kStartCtxEnds);
            filterState.insert_or_assign(key::kWaitingForStartNonMatch, startTriggerNameEnds || startCtxEnds);
            return MatchResult::Matching;
        }
    }

    if (startDefined && stopDefined) {
        auto waitingForStartNonMatch = [&] { return readFlag(key::kWaitingForStartNonMatch); };
        auto waitingForStopNonMatch  = [&] { return readFlag(key::kWaitingForStopNonMatch); };
        if (!triggerActive || waitingForStartNonMatch()) {
            const bool triggerMatch = startTriggerName.empty() || triggerName == startTriggerName;
            const bool contextMatch = startCtx.empty() || triggerCtx.contains(startCtx);

            if (triggerMatch && contextMatch) {
                filterState.insert_or_assign(key::kTriggerActive, true);
                const bool startTriggerNameEnds = readFlag(key::kStartTriggerNameEnds);
                const bool startCtxEnds         = readFlag(key::kStartCtxEnds);
                filterState.insert_or_assign(key::kWaitingForStartNonMatch, startTriggerNameEnds || startCtxEnds);
                return waitingForStartNonMatch() ? MatchResult::Ignore : MatchResult::Matching;
            } else if (waitingForStartNonMatch()) {
                filterState.insert_or_assign(key::kWaitingForStartNonMatch, false);
                return MatchResult::Matching;
            }
        } else {
            const bool triggerMatch = stopTriggerName.empty() || triggerName == stopTriggerName;
            const bool contextMatch = stopCtx.empty() || triggerCtx.contains(stopCtx);

            if ((triggerMatch && contextMatch) || waitingForStopNonMatch()) {
                const bool stopTriggerNameEnds = readFlag(key::kStopTriggerNameEnds);
                const bool stopCtxEnds         = readFlag(key::kStopCtxEnds);
                filterState.insert_or_assign(key::kWaitingForStopNonMatch, stopTriggerNameEnds || stopCtxEnds);
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
        return std::format_to(ctx.out(), "{}", gr::meta::enumName(ret).value_or(""));
    }
};

#endif // GNURADIO_TRIGGERMATCHER_HPP
