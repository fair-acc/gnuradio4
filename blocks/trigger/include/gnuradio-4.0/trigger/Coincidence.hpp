#ifndef GNURADIO_TRIGGER_COINCIDENCE_HPP
#define GNURADIO_TRIGGER_COINCIDENCE_HPP

#include <algorithm>
#include <cstdint>
#include <memory_resource>
#include <optional>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/MonotonicClock.hpp>
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

    A<std::vector<std::string>, "filters", Doc<"one trigger filter per condition">>        filters;
    A<std::pmr::string, "veto filter", Doc<"an event matching this suppresses a report">>  veto_filter;
    A<std::pmr::string, "logic", Doc<"any|all|at_least|exactly|exclusive">>                logic         = std::pmr::string("all");
    A<gr::Size_t, "k", Doc<"how many conditions 'at_least' and 'exactly' ask for">>        k             = 1U;
    A<double, "window", Doc<"s, how far apart contributing events may be">>                window        = 0.;
    A<double, "holdoff", Doc<"s, dead time after a report, 0 = none">>                     holdoff       = 0.;
    A<std::pmr::string, "resolve", Doc<"first|last: which event dates the composite">>     resolve       = std::pmr::string("first");
    A<std::pmr::string, "on incomplete", Doc<"drop|emit_partial for an incomplete group">> on_incomplete = std::pmr::string("drop");
    A<double, "flush after", Doc<"s, idle flush; 0 = off">>                                flush_after   = 0.1;
    A<std::pmr::string, "trigger name", Doc<"the name the composite carries">>             trigger_name  = std::pmr::string("coincidence");
    A<std::pmr::string, "context", Doc<"context the composite carries">>                   context;
    A<gr::Size_t, "n coincidences", Doc<"composites reported">>                            n_coincidences   = 0U;
    A<gr::Size_t, "n vetoed", Doc<"composites a veto suppressed">>                         n_vetoed         = 0U;
    A<gr::Size_t, "n undated", Doc<"events with no trigger_time, unrelatable">>            n_undated        = 0U;
    A<gr::Size_t, "n incomplete", Doc<"groups that closed without every condition">>       n_incomplete     = 0U;
    A<gr::Size_t, "n pending", Doc<"contributions not yet judged">>                        n_pending        = 0U;
    A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">>             n_events_dropped = 0U;

    GR_MAKE_REFLECTABLE(Coincidence, evtIn, evtOut, filters, veto_filter, logic, k, window, holdoff, resolve, on_incomplete, flush_after, trigger_name, context, n_coincidences, n_vetoed, n_undated, n_incomplete, n_pending, n_events_dropped);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    CoincidenceLogic        _logic   = CoincidenceLogic::all;
    CoincidenceTime         _resolve = CoincidenceTime::first;
    bool                    _partial = false;
    std::vector<MatchState> _conditions;
    MatchState              _veto{};
    bool                    _hasVeto = false;
    EventStore              _events;
    detail::PendingEvents   _pending;

    std::optional<std::uint64_t> _reportedUntil; // ns in event time, the holdoff's end
    std::uint64_t                _lastArrival = kUnknownTime;

    struct Contribution {
        std::size_t   condition;
        std::uint64_t at;
        bool          veto;
    };
    std::vector<Contribution> _within;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _logic   = parseLogic(logic.value);
        _resolve = resolve == "last" ? CoincidenceTime::last : CoincidenceTime::first;
        _partial = on_incomplete == "emit_partial";

        _conditions.clear();
        for (const std::string& definition : filters.value) {
            if (auto compiled = detail::compileFilter(definition, std::string_view{}); compiled) {
                _conditions.push_back(compiled.value());
            } else {
                gr::log::warning("Coincidence: {}", compiled.error().message);
            }
        }
        _hasVeto = detail::compileOptionalFilter(veto_filter.value, std::string_view{}, _veto, "Coincidence");

        _events.clear();
        _pending.clear();
        _within.clear();
        _reportedUntil   = std::nullopt;
        n_coincidences   = 0U;
        n_vetoed         = 0U;
        n_undated        = 0U;
        n_incomplete     = 0U;
        n_pending        = 0U;
        n_events_dropped = 0U;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) {
        _events.drain(evtSpan);
        n_events_dropped = static_cast<gr::Size_t>(_events.dropped);

        const std::size_t taken = _events.size();
        for (const StoredEvent& stored : _events.ordered()) {
            admit(stored);
        }
        _events.retire(taken);

        flushIdle();
        n_pending = static_cast<gr::Size_t>(_within.size());

        const std::size_t published = _pending.drainInto(evtOutSpan, 0UZ, this->name, this->unique_name);
        evtOutSpan.publish(published);
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] static CoincidenceLogic parseLogic(std::string_view text) noexcept {
        const auto named = gr::meta::parseEnum<CoincidenceLogic>(text);
        return named && *named != CoincidenceLogic::AUTO ? *named : CoincidenceLogic::all;
    }

    [[nodiscard]] bool decidesOnArrival() const noexcept { return _logic == CoincidenceLogic::any || _logic == CoincidenceLogic::all || _logic == CoincidenceLogic::at_least || _logic == CoincidenceLogic::AUTO; }

    [[nodiscard]] std::uint64_t windowNs() const noexcept { return static_cast<std::uint64_t>(std::llround(std::max(window.value, 0.) * 1e9)); }

    void admit(const StoredEvent& stored) {
        if (!stored.dated()) {
            n_undated = n_undated + 1U;
            return;
        }
        const gr::property_map_view event{stored.event};
        const std::uint64_t         at = *stored.at;

        _lastArrival = monotonicNowNs();
        closeExpired(at);
        if (_hasVeto && gr::trigger::BasicTriggerNameCtxMatcher::match(_veto, event) == gr::trigger::MatchResult::Matching) {
            _within.push_back(Contribution{.condition = 0UZ, .at = at, .veto = true});
            return;
        }
        for (std::size_t condition = 0UZ; condition < _conditions.size(); ++condition) {
            if (gr::trigger::BasicTriggerNameCtxMatcher::match(_conditions[condition], event) == gr::trigger::MatchResult::Matching) {
                _within.push_back(Contribution{.condition = condition, .at = at, .veto = false});
                break;
            }
        }
        decide(at);
    }

    void flushIdle() {
        if (flush_after <= 0. || _within.empty()) {
            return;
        }
        const std::uint64_t now = monotonicNowNs();
        if (_lastArrival == kUnknownTime || now == kUnknownTime) {
            return;
        }
        if (now > _lastArrival && now - _lastArrival >= static_cast<std::uint64_t>(std::llround(flush_after.value * 1e9))) {
            judgeClosed();
            _within.clear();
        }
    }

    void closeExpired(std::uint64_t now) {
        const std::uint64_t span    = windowNs();
        const auto          expired = [now, span](const Contribution& held) { return now > held.at && now - held.at > span; };
        if (!std::ranges::any_of(_within, expired)) {
            return;
        }
        if (!decidesOnArrival() || _partial) {
            judgeClosed();
        }
        std::erase_if(_within, expired);
    }

    void judgeClosed() {
        const auto [satisfied, vetoed] = countSatisfied();
        if (satisfied == 0UZ) {
            return;
        }
        if (vetoed) {
            n_vetoed = n_vetoed + 1U;
            _within.clear();
            return;
        }
        if (satisfies(satisfied)) {
            report(newestContributionTime(), satisfied);
            return;
        }
        if (_partial && satisfied < _conditions.size()) {
            n_incomplete = n_incomplete + 1U;
            report(newestContributionTime(), satisfied);
        }
    }

    [[nodiscard]] std::pair<std::size_t, bool> countSatisfied() const {
        std::vector<bool> held(_conditions.size(), false);
        bool              vetoed = false;
        std::ranges::for_each(_within, [&](const Contribution& contribution) {
            if (contribution.veto) {
                vetoed = true;
            } else if (contribution.condition < held.size()) {
                held[contribution.condition] = true;
            }
        });
        return {static_cast<std::size_t>(std::ranges::count(held, true)), vetoed};
    }

    [[nodiscard]] std::uint64_t newestContributionTime() const {
        const auto newest = std::ranges::max_element(_within, {}, &Contribution::at);
        return newest == _within.end() ? 0U : newest->at;
    }

    void decide(std::uint64_t now) {
        if (_reportedUntil && now < *_reportedUntil) {
            return;
        }
        if (!decidesOnArrival()) {
            return;
        }
        const auto [satisfied, vetoed] = countSatisfied();
        if (!satisfies(satisfied)) {
            return;
        }
        if (vetoed) {
            n_vetoed = n_vetoed + 1U;
            _within.clear();
            return;
        }
        report(now, satisfied);
    }

    [[nodiscard]] bool satisfies(std::size_t satisfied) const noexcept {
        switch (_logic) {
        case CoincidenceLogic::any: return satisfied >= 1UZ;
        case CoincidenceLogic::at_least: return satisfied >= static_cast<std::size_t>(k);
        case CoincidenceLogic::exactly: return satisfied == static_cast<std::size_t>(k);
        case CoincidenceLogic::exclusive: return satisfied == 1UZ;
        case CoincidenceLogic::all:
        case CoincidenceLogic::AUTO: return !_conditions.empty() && satisfied == _conditions.size();
        }
        return false;
    }

    void report(std::uint64_t now, std::size_t satisfied) {
        const auto          earliest = std::ranges::min_element(_within, {}, &Contribution::at);
        const std::uint64_t at       = (_resolve == CoincidenceTime::first && earliest != _within.end()) ? earliest->at : now;

        gr::property_map composite                          = detail::makeEvent(trigger_name.value, this->unique_name.value());
        composite[std::string(gr::tag::TRIGGER_TIME.key())] = at;
        composite[std::string(gr::tag::CONTEXT.key())]      = context.value;
        composite[std::string("n_present")]                 = static_cast<gr::Size_t>(satisfied); // so a partial group is recognisable
        composite[std::string("n_expected")]                = static_cast<gr::Size_t>(_conditions.size());
        _pending.push(std::move(composite));

        n_coincidences = n_coincidences + 1U;
        _within.clear();
        if (holdoff > 0.) {
            _reportedUntil = now + static_cast<std::uint64_t>(std::llround(holdoff.value * 1e9));
        }
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_COINCIDENCE_HPP
