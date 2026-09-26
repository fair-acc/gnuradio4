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
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>

namespace gr::blocks::trigger {

enum class CoincidenceLogic : std::uint8_t { AUTO, any, all, at_least, exactly, exclusive };

enum class CoincidenceTime : std::uint8_t { first, last };

GR_REGISTER_BLOCK(gr::blocks::trigger::Coincidence)

struct Coincidence : gr::Block<Coincidence> {
    using Description = Doc<R"(@brief report when N conditions occur within a window of one another

    evtIn   ─A────B──C──────A──B─▶    (no RxMarbles equivalent: Rx joins by arrival, not by carried time)
    evtOut  ───────────X──────────▶   logic = all, window = Δt
                                                 {"logic", "all"}, {"window", 50e-6}, {"holdoff", 1e-3}});

The coincidence unit of a detector, and the AND of a scope's logic trigger -- over *event* streams rather than sample streams.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::vector<std::pmr::string>, "filters", Doc<"one trigger filter per condition; at most 8">> filters;
    A<std::pmr::string, "veto filter", Doc<"an event matching this suppresses a report">>                veto_filter;
    A<std::pmr::string, "logic", Doc<"any|all|at_least|exactly|exclusive">>                              logic         = std::pmr::string("all");
    A<gr::Size_t, "k", Doc<"how many conditions 'at_least' and 'exactly' ask for">>                      k             = 1U;
    A<double, "window", Doc<"s, how far apart contributing events may be">>                              window        = 0.;
    A<double, "holdoff", Doc<"s, dead time after a report, 0 = none">>                                   holdoff       = 0.;
    A<std::pmr::string, "resolve", Doc<"first|last: which event dates the composite">>                   resolve       = std::pmr::string("first");
    A<std::pmr::string, "on incomplete", Doc<"drop|emit_partial for an incomplete group">>               on_incomplete = std::pmr::string("drop");
    gr::TriggerName                                                                                      trigger_name  = std::pmr::string("coincidence");
    gr::Context                                                                                          context;

    mutable A<gr::Size_t, "n coincidences", Doc<"composites reported; host path only">>                                                                           n_coincidences   = 0U;
    mutable A<gr::Size_t, "n vetoed", Doc<"composites a veto suppressed; host path only">>                                                                        n_vetoed         = 0U;
    mutable A<gr::Size_t, "n undated", Doc<"events with no trigger_time, unrelatable; host path only">>                                                           n_undated        = 0U;
    mutable A<gr::Size_t, "n incomplete", Doc<"groups that closed without every condition; host path only">>                                                      n_incomplete     = 0U;
    mutable A<gr::Size_t, "n pending", Doc<"contributions not yet judged; host path only">>                                                                       n_pending        = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events the store or the open window had no room for, or that came after their window closed; host path only">> n_events_dropped = 0U;

    using MatchState                            = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;
    constexpr static std::size_t kMaxConditions = 8UZ;
    constexpr static std::size_t kMaxWithin     = 64UZ;

    CoincidenceLogic _logic   = CoincidenceLogic::all;
    CoincidenceTime  _resolve = CoincidenceTime::first;
    bool             _partial = false;
    /** CRC32C of each condition's trigger name: a kernel holds no strings, and a fixed array of hashes crosses by value */
    std::array<std::uint32_t, kMaxConditions> _conditions{};
    gr::Size_t                                _nConditions = 0U;
    bool                                      _byNameOnly  = true;
    MatchState                                _veto{};
    bool                                      _hasVeto = false;
    detail::InlineText                        _source{};
    detail::InlineText                        _name{};
    detail::InlineText                        _context{};
    gr::Size_t                                _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Coincidence, evtIn, evtOut, filters, veto_filter, logic, k, window, holdoff, resolve, on_incomplete, trigger_name, context, n_coincidences, n_vetoed, n_undated, n_incomplete, n_pending, n_events_dropped, _logic, _resolve, _partial, _conditions, _nConditions, _byNameOnly, _veto, _hasVeto, _source, _name, _context, _settings_epoch);

    struct Contribution {
        std::size_t   condition;
        std::uint64_t at;
        bool          veto;
    };

    std::array<MatchState, kMaxConditions>         _compiled{};
    mutable std::array<MatchState, kMaxConditions> _matching{};
    mutable MatchState                             _vetoing{};
    /** the first `_nWithin`, in arrival order; all within `window` of `_eventTime`, hence of one another */
    mutable std::array<Contribution, kMaxWithin> _within{};
    mutable std::size_t                          _nWithin = 0UZ;
    /** [ns] the latest trigger_time seen: windows close in event time, not in arrival order */
    mutable std::uint64_t _eventTime = 0U;
    /** ns in event time, the holdoff's end */
    mutable std::optional<std::uint64_t> _reportedUntil;
    mutable gr::Size_t                   _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable EventStore                   _events;

    void start() { _stateEpoch = std::numeric_limits<gr::Size_t>::max(); }

    [[nodiscard]] std::string_view hostOnlyReason() const noexcept { return _byNameOnly ? std::string_view{} : std::string_view{"a condition with a context, a range or a run marker needs the host matcher; a kernel compares trigger names by hash"}; }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _logic   = parseLogic(logic.value);
        _resolve = resolve == "last" ? CoincidenceTime::last : CoincidenceTime::first;
        _partial = on_incomplete == "emit_partial";

        if (filters.value.size() > kMaxConditions) {
            gr::log::warning("Coincidence: {} filters exceed the {} the block holds; the rest are ignored", filters.value.size(), kMaxConditions);
        }
        _nConditions = 0U;
        _byNameOnly  = true;
        for (const std::pmr::string& definition : filters.value | std::views::take(kMaxConditions)) {
            if (auto compiled = detail::compileFilter(definition, std::string_view{}); compiled) {
                constexpr std::string_view kGrammar = "/,^[]";
                _byNameOnly                         = _byNameOnly && !definition.empty() && std::string_view{definition}.find_first_of(kGrammar) == std::string_view::npos;
                _compiled[_nConditions]             = compiled.value();
                _conditions[_nConditions]           = nameHash(compiled.value().start.view());
                _nConditions                        = _nConditions + 1U;
            } else {
                gr::log::warning("Coincidence: {}", compiled.error().message);
            }
        }
        _hasVeto = detail::compileOptionalFilter(veto_filter.value, std::string_view{}, _veto, "Coincidence");
        _source.assign(this->unique_name.value());
        _name.assign(trigger_name.value);
        _context.assign(context.value);
        _events.clear();
        _settings_epoch  = _settings_epoch + 1U;
        n_coincidences   = 0U;
        n_vetoed         = 0U;
        n_undated        = 0U;
        n_incomplete     = 0U;
        n_pending        = 0U;
        n_events_dropped = 0U;
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
        n_pending = static_cast<gr::Size_t>(_nWithin);
        evtOutSpan.publish(nEmitted);
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] static CoincidenceLogic parseLogic(std::string_view text) noexcept {
        const auto named = gr::meta::parseEnum<CoincidenceLogic>(text);
        return named && *named != CoincidenceLogic::AUTO ? *named : CoincidenceLogic::all;
    }

    [[nodiscard]] bool decidesOnArrival() const noexcept { return _logic == CoincidenceLogic::any || _logic == CoincidenceLogic::all || _logic == CoincidenceLogic::at_least || _logic == CoincidenceLogic::AUTO; }

    [[nodiscard]] static std::uint64_t nanoseconds(double seconds) noexcept { return static_cast<std::uint64_t>(std::round(std::max(seconds, 0.) * 1e9)); }

    [[nodiscard]] static std::uint32_t nameHash(std::string_view triggerName) noexcept { return gr::crc::computeScalar<gr::crc::Flavour::CRC32C_CASTAGNOLI>(std::as_bytes(std::span(triggerName.data(), triggerName.size()))); }

    template<bool kOnHost>
    [[nodiscard]] std::optional<std::size_t> conditionOf(const gr::property_map_view& event) const {
        const auto conditions = std::views::iota(0UZ, static_cast<std::size_t>(_nConditions));
        if constexpr (kOnHost) {
            const auto found = std::ranges::find_if(conditions, [&](std::size_t condition) { return detail::matches(_matching[condition], event); });
            return found == conditions.end() ? std::nullopt : std::optional{*found};
        } else {
            auto triggerName = event.template get_if<std::string_view>(gr::tag::TRIGGER_NAME.key());
            if (!triggerName) {
                triggerName = event.template get_if<std::string_view>(gr::tag::TRIGGER_NAME.shortKey());
            }
            const std::uint32_t hash  = nameHash(triggerName ? *triggerName : std::string_view{});
            const auto          found = std::ranges::find_if(conditions, [&](std::size_t condition) { return _conditions[condition] == hash; });
            return found == conditions.end() ? std::nullopt : std::optional{*found};
        }
    }

    [[nodiscard]] std::span<Contribution> within() const noexcept { return {_within.data(), _nWithin}; }

    void hold(Contribution contribution) const {
        if (_nWithin == kMaxWithin) {
            std::ranges::move(_within.begin() + 1, _within.end(), _within.begin());
            --_nWithin;
            n_events_dropped = n_events_dropped + 1U;
        }
        _within[_nWithin++] = contribution;
    }

    template<bool kOnHost>
    void admit(const gr::property_map_view& event, auto& evtOutSpan, std::size_t& nEmitted) const {
        const auto stamp = event.template get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        if (!stamp) {
            n_undated = n_undated + 1U;
            return;
        }
        const std::uint64_t at = *stamp;
        _eventTime             = std::max(_eventTime, at);

        closeExpired(evtOutSpan, nEmitted);
        const bool                       vetoes    = _hasVeto && detail::matches(_vetoing, event);
        const std::optional<std::size_t> condition = vetoes ? std::nullopt : conditionOf<kOnHost>(event);
        const bool                       closed    = _eventTime - at > nanoseconds(window.value);
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

    void closeExpired(auto& evtOutSpan, std::size_t& nEmitted) const {
        const std::uint64_t now     = _eventTime;
        const std::uint64_t span    = nanoseconds(window.value);
        const auto          expired = [now, span](const Contribution& held) { return now > held.at && now - held.at > span; };
        if (std::ranges::none_of(within(), expired)) {
            return;
        }
        if (!decidesOnArrival() || _partial) {
            judgeClosed(evtOutSpan, nEmitted);
        }
        const auto held = within();
        const auto kept = std::ranges::remove_if(held, expired);
        _nWithin        = static_cast<std::size_t>(kept.begin() - held.begin());
    }

    void judgeClosed(auto& evtOutSpan, std::size_t& nEmitted) const {
        const auto [satisfied, vetoed] = countSatisfied();
        if (satisfied == 0UZ || inHoldoff()) {
            return;
        }
        if (vetoed) {
            n_vetoed = n_vetoed + 1U;
            _nWithin = 0UZ;
            return;
        }
        if (satisfies(satisfied)) {
            report(satisfied, evtOutSpan, nEmitted);
            return;
        }
        if (_partial && satisfied < static_cast<std::size_t>(_nConditions)) {
            n_incomplete = n_incomplete + 1U;
            report(satisfied, evtOutSpan, nEmitted);
        }
    }

    [[nodiscard]] std::pair<std::size_t, bool> countSatisfied() const {
        std::array<bool, kMaxConditions> held{};
        bool                             vetoed = false;
        for (const Contribution& contribution : within()) {
            if (contribution.veto) {
                vetoed = true;
            } else if (contribution.condition < held.size()) {
                held[contribution.condition] = true;
            }
        }
        return {static_cast<std::size_t>(std::ranges::count(held, true)), vetoed};
    }

    /** event time when none is held */
    [[nodiscard]] std::uint64_t compositeTime() const {
        auto times = within() | std::views::filter([](const Contribution& held) { return !held.veto; }) | std::views::transform(&Contribution::at);
        if (std::ranges::empty(times)) {
            return _eventTime;
        }
        const auto [earliest, latest] = std::ranges::minmax(times);
        return _resolve == CoincidenceTime::first ? earliest : latest;
    }

    [[nodiscard]] bool inHoldoff() const { return _reportedUntil && compositeTime() < *_reportedUntil; }

    void decide(auto& evtOutSpan, std::size_t& nEmitted) const {
        if (!decidesOnArrival() || inHoldoff()) {
            return;
        }
        const auto [satisfied, vetoed] = countSatisfied();
        if (!satisfies(satisfied)) {
            return;
        }
        if (vetoed) {
            n_vetoed = n_vetoed + 1U;
            _nWithin = 0UZ;
            return;
        }
        report(satisfied, evtOutSpan, nEmitted);
    }

    [[nodiscard]] bool satisfies(std::size_t satisfied) const noexcept {
        switch (_logic) {
        case CoincidenceLogic::any: return satisfied >= 1UZ;
        case CoincidenceLogic::at_least: return satisfied >= static_cast<std::size_t>(k);
        case CoincidenceLogic::exactly: return satisfied == static_cast<std::size_t>(k);
        case CoincidenceLogic::exclusive: return satisfied == 1UZ;
        case CoincidenceLogic::all:
        case CoincidenceLogic::AUTO: return _nConditions > 0U && satisfied == static_cast<std::size_t>(_nConditions);
        }
        return false;
    }

    void report(std::size_t satisfied, auto& evtOutSpan, std::size_t& nEmitted) const {
        const std::uint64_t at = compositeTime();

        const auto writeComposite = [&](gr::property_map_view& composite) {
            return detail::writeEvent(composite, _name.view(), _source.view(), gr::property_map_view{})                                        //
                   && composite.try_emplace(gr::tag::TRIGGER_TIME.key(), at) && composite.try_emplace(gr::tag::CONTEXT.key(), _context.view()) //
                   && composite.try_emplace(std::string_view{"n_present"}, static_cast<gr::Size_t>(satisfied)) /* so a partial group is recognisable */ && composite.try_emplace(std::string_view{"n_expected"}, _nConditions);
        };
        if (evtOutSpan.isConnected && gr::emit(evtOutSpan, nEmitted, detail::kEventKeys, writeComposite)) {
            ++nEmitted;
        }

        n_coincidences = n_coincidences + 1U;
        _nWithin       = 0UZ;
        if (holdoff > 0.) {
            _reportedUntil = at + nanoseconds(holdoff.value);
        }
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_COINCIDENCE_HPP
