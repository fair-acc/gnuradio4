#ifndef GNURADIO_TRIGGER_COINCIDENCE_HPP
#define GNURADIO_TRIGGER_COINCIDENCE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/CRC.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/EventReference.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>

namespace gr::blocks::trigger {

enum class CoincidenceLogic : std::uint8_t { automatic, any, all, atLeast, exactly, exclusive };

enum class CoincidenceTime : std::uint8_t { first, last };

GR_REGISTER_BLOCK(gr::blocks::trigger::Coincidence)

struct Coincidence : gr::Block<Coincidence> {
    using Description = Doc<R"(@brief report when N conditions occur within a window of one another

    evtIn   ─A────B──C──────A──B─▶    (no RxMarbles equivalent: Rx joins by arrival, not by carried time)
    evtOut  ───────────X──────────▶   logic = all, window = Δt
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::vector<std::pmr::string>, "filters", Doc<"one trigger filter per condition; at most 8">>                                             filters;
    A<std::pmr::string, "veto filter", Doc<"an event matching this suppresses a report">>                                                            veto_filter;
    A<std::pmr::string, "logic", Doc<"any|all|atLeast|exactly|exclusive">>                                                                           logic         = std::pmr::string("all");
    A<gr::Size_t, "k", Doc<"how many conditions 'atLeast' and 'exactly' ask for">>                                                                   k             = 1U;
    A<double, "window", Doc<"s, how far apart contributing events may be">>                                                                          window        = 0.;
    A<double, "holdoff", Doc<"s, dead time after a report, 0 = none">>                                                                               holdoff       = 0.;
    A<double, "max lateness", Doc<"s, how long after its trigger_time an event may be decided; earlier ones wait as long">>                          max_lateness  = 0.;
    A<std::pmr::string, "resolve", Doc<"first|last: which event dates the composite">>                                                               resolve       = std::pmr::string("first");
    A<std::pmr::string, "on incomplete", Doc<"drop|emit_partial for an incomplete group">>                                                           on_incomplete = std::pmr::string("drop");
    A<std::pmr::string, "reference", Doc<"decision|start: the time events are dated by, the start tuple of a delayed decision or its trigger_time">> reference     = std::pmr::string("decision");
    gr::TriggerName                                                                                                                                  trigger_name;
    gr::Context                                                                                                                                      context;

    mutable A<gr::Size_t, "n coincidences", Doc<"composites reported">>                                                                           n_coincidences   = 0U;
    mutable A<gr::Size_t, "n vetoed", Doc<"composites a veto suppressed">>                                                                        n_vetoed         = 0U;
    mutable A<gr::Size_t, "n undated", Doc<"events with no trigger_time, unrelatable">>                                                           n_undated        = 0U;
    mutable A<gr::Size_t, "n incomplete", Doc<"groups that closed without every condition">>                                                      n_incomplete     = 0U;
    mutable A<gr::Size_t, "n pending", Doc<"contributions not yet judged, and events held back for max_lateness, as of the last work call">>      n_pending        = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events the store or the open window had no room for, or that came after their window closed">> n_events_dropped = 0U;

    using MatchState                            = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;
    constexpr static std::size_t kMaxConditions = 8UZ;
    constexpr static std::size_t kMaxWithin     = 64UZ;

    CoincidenceLogic _logic     = CoincidenceLogic::all;
    CoincidenceTime  _resolve   = CoincidenceTime::first;
    bool             _partial   = false;
    EventReference   _reference = EventReference::decision;
    /** CRC32C of each condition's trigger name: a kernel holds no strings, and a fixed array of hashes crosses by value */
    std::array<std::uint32_t, kMaxConditions> _conditions{};
    gr::Size_t                                _nConditions = 0U;
    bool                                      _byNameOnly  = true;
    MatchState                                _veto{};
    bool                                      _hasVeto = false;
    detail::InlineText                        _name{};
    detail::InlineText                        _context{};
    gr::Size_t                                _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Coincidence, evtIn, evtOut, filters, veto_filter, logic, k, window, holdoff, max_lateness, resolve, on_incomplete, reference, trigger_name, context, n_coincidences, n_vetoed, n_undated, n_incomplete, n_pending, n_events_dropped, _logic, _resolve, _partial, _reference, _conditions, _nConditions, _byNameOnly, _veto, _hasVeto, _name, _context, _settings_epoch);

    struct Contribution {
        std::size_t   condition;
        std::uint64_t at;
        bool          veto;
    };

    std::array<MatchState, kMaxConditions>         _compiled{};
    mutable std::array<MatchState, kMaxConditions> _matching{};
    mutable MatchState                             _vetoing{};
    /** the first `_nWithin`, by time, the contributions no group has consumed or given up on */
    mutable std::array<Contribution, kMaxWithin> _within{};
    mutable std::size_t                          _nWithin = 0UZ;
    /** [ns] the latest trigger_time seen: windows close in event time, not in arrival order */
    mutable std::uint64_t _eventTime = 0U;
    /** ns in event time, the holdoff's end */
    mutable std::optional<std::uint64_t> _reportedUntil;
    mutable gr::Size_t                   _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable EventStore                   _events;
    detail::ErrorMemory                  _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        detail::nameTriggerAfterBlock(*this);
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    void reset() {
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
        _events.clear();
        n_coincidences   = 0U;
        n_vetoed         = 0U;
        n_undated        = 0U;
        n_incomplete     = 0U;
        n_pending        = 0U;
        n_events_dropped = 0U;
    }

    [[nodiscard]] std::string_view hostOnlyReason() const noexcept {
        if (_hasVeto) {
            return "a veto filter needs the host matcher";
        }
        constexpr std::string_view kNeedsHostMatcher = "a condition with a context, a wildcard, a negation, a range or a run marker needs the host matcher; a kernel compares trigger names by hash";
        return _byNameOnly ? std::string_view{} : kNeedsHostMatcher;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _logic     = parseLogic(logic.value);
        _resolve   = gr::meta::parseEnum<CoincidenceTime>(resolve.value).value_or(CoincidenceTime::first);
        _partial   = on_incomplete == "emit_partial";
        _reference = gr::meta::parseEnum<EventReference>(reference.value).value_or(EventReference::decision);

        if (filters.value.size() > kMaxConditions) {
            const std::string message = std::format("{} filters exceed the {} the block holds; the rest are ignored", filters.value.size(), kMaxConditions);
            detail::reportError(*this, "settingsChanged()", message);
        }
        _nConditions = 0U;
        _byNameOnly  = true;
        for (const std::pmr::string& definition : filters.value | std::views::take(kMaxConditions)) {
            if (auto compiled = detail::compileFilter(definition, std::string_view{}); compiled) {
                _byNameOnly               = _byNameOnly && compiled.value().isPlainName();
                _compiled[_nConditions]   = compiled.value();
                _conditions[_nConditions] = nameHash(compiled.value().start.view());
                _nConditions              = _nConditions + 1U;
            } else {
                detail::reportError(*this, "settingsChanged()", std::format("{}", compiled.error().message));
            }
        }
        _hasVeto = detail::compileOptionalFilter(*this, veto_filter.value, std::string_view{}, _veto);
        _name.assign(trigger_name.value);
        _context.assign(context.value);
        _events.clear();
        _events.lateness = detail::nanoseconds(max_lateness.value);
        _settings_epoch  = _settings_epoch + 1U;
        n_pending        = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(evtSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _matching      = _compiled;
            _vetoing       = _veto;
            _nWithin       = 0UZ;
            _eventTime     = 0U;
            _reportedUntil = std::nullopt;
            _stateEpoch    = _settings_epoch;
        }
        constexpr std::size_t kReportsPerEvent = 2UZ;
        std::size_t           nEmitted         = 0UZ;
        std::ignore                            = forEachEvent<kOnHost>(_events, evtSpan, kOnHost ? std::numeric_limits<std::size_t>::max() : evtOutSpan.size() / kReportsPerEvent, [&](const gr::property_map_view& event) {
            if (evtOutSpan.isConnected && nEmitted + kReportsPerEvent > evtOutSpan.size()) {
                return false;
            }
            admit<kOnHost>(event, evtOutSpan, nEmitted);
            return true;
        });
        if constexpr (kOnHost) {
            n_events_dropped = n_events_dropped + static_cast<gr::Size_t>(_events.dropped);
            _events.dropped  = 0U;
        }
        n_pending = static_cast<gr::Size_t>(_nWithin + (kOnHost ? _events.size() : 0UZ));
        evtOutSpan.publish(nEmitted);
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] static CoincidenceLogic parseLogic(std::string_view text) noexcept {
        const auto named = gr::meta::parseEnum<CoincidenceLogic>(text);
        return named && *named != CoincidenceLogic::automatic ? *named : CoincidenceLogic::all;
    }

    [[nodiscard]] bool decidesOnArrival() const noexcept { return _logic == CoincidenceLogic::any || _logic == CoincidenceLogic::all || _logic == CoincidenceLogic::atLeast || _logic == CoincidenceLogic::automatic; }

    [[nodiscard]] static std::uint32_t nameHash(std::string_view triggerName) noexcept { return gr::crc::computeScalar<gr::crc::Flavour::CRC32C_CASTAGNOLI>(std::as_bytes(std::span(triggerName.data(), triggerName.size()))); }

    template<bool onHost>
    [[nodiscard]] std::optional<std::size_t> conditionOf(const gr::property_map_view& event) const {
        const auto conditions = std::views::iota(0UZ, static_cast<std::size_t>(_nConditions));
        if constexpr (onHost) {
            const auto found = std::ranges::find_if(conditions, [&](std::size_t condition) { return detail::matches(_matching[condition], event); });
            return found == conditions.end() ? std::nullopt : std::optional{*found};
        } else {
            auto triggerName = event.template get_if<std::string_view>(gr::tag::kTriggerName.key());
            if (!triggerName) {
                triggerName = event.template get_if<std::string_view>(gr::tag::kTriggerName.shortKey());
            }
            const std::uint32_t hash  = nameHash(triggerName ? *triggerName : std::string_view{});
            const auto          found = std::ranges::find_if(conditions, [&](std::size_t condition) { return _conditions[condition] == hash; });
            return found == conditions.end() ? std::nullopt : std::optional{*found};
        }
    }

    /** the group an anchor opens: the contributions from `first` on that lie within `window` of it; `_within` is sorted by time */
    struct Group {
        std::size_t   satisfied = 0UZ;
        bool          vetoed    = false;
        std::uint64_t members   = 0U; /// bit i: `_within[i]` belongs to the group
        std::uint64_t picks     = 0U; /// bit i: the earliest contribution of its condition, the one a report consumes
    };

    [[nodiscard]] Group groupAt(std::size_t first) const {
        Group                            group;
        std::array<bool, kMaxConditions> held{};
        const std::uint64_t              horizon = _within[first].at + detail::nanoseconds(window.value);
        for (std::size_t i = first; i < _nWithin && _within[i].at <= horizon; ++i) {
            const Contribution& contribution = _within[i];
            group.members |= std::uint64_t{1U} << i;
            if (contribution.veto) {
                group.vetoed = true;
            } else if (contribution.condition < held.size() && !held[contribution.condition]) {
                held[contribution.condition] = true;
                ++group.satisfied;
                group.picks |= std::uint64_t{1U} << i;
            }
        }
        return group;
    }

    void consume(std::uint64_t removed) const {
        std::size_t kept = 0UZ;
        for (std::size_t i = 0UZ; i < _nWithin; ++i) {
            if (((removed >> i) & 1U) == 0U) {
                _within[kept++] = _within[i];
            }
        }
        _nWithin = kept;
    }

    void hold(Contribution contribution) const {
        if (_nWithin == kMaxWithin) {
            consume(1U);
            n_events_dropped = n_events_dropped + 1U;
        }
        std::size_t at = _nWithin++;
        for (; at > 0UZ && _within[at - 1UZ].at > contribution.at; --at) {
            _within[at] = _within[at - 1UZ];
        }
        _within[at] = contribution;
    }

    template<bool onHost>
    void admit(const gr::property_map_view& event, auto& evtOutSpan, std::size_t& nEmitted) const {
        const auto stamp = event.template get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
        if (!stamp) {
            n_undated = n_undated + 1U;
            return;
        }
        const std::uint64_t at = referenceTimeOf(event, *stamp, _reference);
        _eventTime             = std::max(_eventTime, at);

        /** on the host the store has held events back for `max_lateness`; a kernel has no store and keeps its groups open that long instead */
        const std::uint64_t slack = onHost ? 0U : detail::nanoseconds(max_lateness.value);
        closeExpired(slack, evtOutSpan, nEmitted);
        const bool                       vetoes    = _hasVeto && detail::matches(_vetoing, event);
        const std::optional<std::size_t> condition = vetoes ? std::nullopt : conditionOf<onHost>(event);
        const bool                       closed    = _eventTime - at > detail::nanoseconds(window.value) + slack;
        if (closed && (vetoes || condition)) {
            n_events_dropped = n_events_dropped + 1U;
            return;
        }
        if (vetoes) {
            hold(Contribution{.condition = 0UZ, .at = at, .veto = true});
            return;
        }
        if (condition) {
            hold(Contribution{.condition = *condition, .at = at, .veto = false});
        }
        decide(evtOutSpan, nEmitted);
    }

    /** the earliest contribution anchors a group; once event time is past its window no later arrival can join it */
    void closeExpired(std::uint64_t slack, auto& evtOutSpan, std::size_t& nEmitted) const {
        while (_nWithin > 0UZ && _eventTime > _within[0].at + detail::nanoseconds(window.value) + slack) {
            if (decidesOnArrival() && !_partial) {
                consume(1U); // judged on arrival; the anchor found no partner
                continue;
            }
            judgeClosed(groupAt(0UZ), evtOutSpan, nEmitted);
        }
    }

    /** consumes at least the anchor, so the caller's loop ends */
    void judgeClosed(const Group& group, auto& evtOutSpan, std::size_t& nEmitted) const {
        if (group.satisfied == 0UZ) {
            consume(1U);
            return;
        }
        const bool consumesAll = !decidesOnArrival(); // exclusive and exactly: every member had its partners
        if (group.vetoed) {
            n_vetoed = n_vetoed + 1U;
            consume(group.members);
            return;
        }
        const bool exclusiveBreached = _logic == CoincidenceLogic::exclusive && group.satisfied > 1UZ;
        const bool exactlyBreached   = _logic == CoincidenceLogic::exactly && group.satisfied > static_cast<std::size_t>(k);
        const bool tooMany           = exclusiveBreached || exactlyBreached;
        const bool complete          = satisfies(group.satisfied);
        const bool partial           = !complete && !tooMany && _partial && group.satisfied < static_cast<std::size_t>(_nConditions);
        if (!complete && !partial) {
            consume(consumesAll ? group.members : 1U);
            return;
        }
        if (inHoldoff(group)) {
            consume(consumesAll ? group.members : group.picks);
            return;
        }
        if (partial) {
            n_incomplete = n_incomplete + 1U;
        }
        report(group, consumesAll ? group.members : group.picks, evtOutSpan, nEmitted);
    }

    /** event time when the group holds no condition */
    [[nodiscard]] std::uint64_t compositeTime(const Group& group) const {
        std::uint64_t earliest = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t latest   = 0U;
        for (std::size_t i = 0UZ; i < _nWithin; ++i) {
            if (((group.picks >> i) & 1U) != 0U) {
                earliest = std::min(earliest, _within[i].at);
                latest   = std::max(latest, _within[i].at);
            }
        }
        if (group.picks == 0U) {
            return _eventTime;
        }
        return _resolve == CoincidenceTime::first ? earliest : latest;
    }

    [[nodiscard]] bool inHoldoff(const Group& group) const { return _reportedUntil && compositeTime(group) < *_reportedUntil; }

    /** the earliest anchor whose group holds enough conditions reports; a group that only a later anchor completes waits for it */
    void decide(auto& evtOutSpan, std::size_t& nEmitted) const {
        if (!decidesOnArrival()) {
            return;
        }
        for (std::size_t first = 0UZ; first < _nWithin; ++first) {
            const Group group = groupAt(first);
            if (!satisfies(group.satisfied)) {
                continue;
            }
            if (group.vetoed) {
                n_vetoed = n_vetoed + 1U;
                consume(group.members);
            } else if (inHoldoff(group)) {
                consume(group.picks);
            } else {
                report(group, group.picks, evtOutSpan, nEmitted);
            }
            return;
        }
    }

    [[nodiscard]] bool satisfies(std::size_t satisfied) const noexcept {
        switch (_logic) {
        case CoincidenceLogic::any: return satisfied >= 1UZ;
        case CoincidenceLogic::atLeast: return satisfied >= static_cast<std::size_t>(k);
        case CoincidenceLogic::exactly: return satisfied == static_cast<std::size_t>(k);
        case CoincidenceLogic::exclusive: return satisfied == 1UZ;
        case CoincidenceLogic::all:
        case CoincidenceLogic::automatic: return _nConditions > 0U && satisfied == static_cast<std::size_t>(_nConditions);
        }
        return false;
    }

    void report(const Group& group, std::uint64_t consumed, auto& evtOutSpan, std::size_t& nEmitted) const {
        const std::uint64_t at = compositeTime(group);

        const auto writeComposite = [&](gr::property_map_view& composite) {
            const auto presence = [&](gr::property_map_view& details) { // so a partial group is recognisable
                if (!details.try_emplace(std::string_view{"n_present"}, static_cast<gr::Size_t>(group.satisfied))) {
                    return false;
                }
                return details.try_emplace(std::string_view{"n_expected"}, _nConditions);
            };
            return detail::writeEvent(composite, _name.view(), gr::property_map_view{}, 2U, presence) //
                   && composite.try_emplace(gr::tag::kTriggerTime.key(), at) && composite.try_emplace(gr::tag::kContext.key(), _context.view());
        };
        if (evtOutSpan.isConnected && gr::emit(evtOutSpan, nEmitted, detail::kEventKeys, writeComposite)) {
            ++nEmitted;
        }

        n_coincidences = n_coincidences + 1U;
        consume(consumed);
        if (holdoff > 0.) {
            _reportedUntil = at + detail::nanoseconds(holdoff.value);
        }
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_COINCIDENCE_HPP
