#ifndef GNURADIO_TRIGGER_TIMEINTERVAL_HPP
#define GNURADIO_TRIGGER_TIMEINTERVAL_HPP

#include <cmath>
#include <cstdint>
#include <deque>
#include <memory_resource>
#include <optional>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/Segments.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>

namespace gr::blocks::trigger {

enum class IntervalMode : std::uint8_t { AUTO, to_reference, nearest, paired, to_previous, tie };

GR_REGISTER_BLOCK("gr::blocks::trigger::TimeInterval", gr::blocks::trigger::TimeInterval, [T], [ double, gr::UncertainValue<double> ])

template<typename T = double>
requires(std::is_same_v<T, double> || std::is_same_v<T, gr::UncertainValue<double>>)
struct TimeInterval : gr::Block<TimeInterval<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief measure the time between triggers as a sample a signal chain can process

    evtIn  ─clk──pulse────clk──pulse─▶   (no RxMarbles equivalent)
    out    ──────0.2ms─────────0.1ms─▶   mode = to_reference

The distribution of that signal is a jitter histogram, its mean a phase error, its deviation a wander figure -- feed it to
`gr::blocks::math::Histogram`.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn evtIn;
    gr::PortOut<T>  out;

    A<std::pmr::string, "mode", Doc<"to_reference|nearest|paired|to_previous|tie">>              mode = std::pmr::string("to_reference");
    A<std::pmr::string, "reference filter", Doc<"trigger filter naming the reference events">>   reference_filter;
    A<std::pmr::string, "measure filter", Doc<"measured-event filter; empty = everything else">> measure_filter;
    A<double, "nominal period", Doc<"s, the ideal grid 'tie' measures against">>                 nominal_period   = 0.;
    A<double, "max interval", Doc<"s, longer counts unmatched; 0 = no limit">>                   max_interval     = 0.;
    A<gr::Size_t, "n intervals", Doc<"differences emitted">>                                     n_intervals      = 0U;
    A<gr::Size_t, "n unmatched", Doc<"measured events with no reference to measure against">>    n_unmatched      = 0U;
    A<gr::Size_t, "n undated", Doc<"events carrying no trigger_time">>                           n_undated        = 0U;
    A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">>                   n_events_dropped = 0U;

    GR_MAKE_REFLECTABLE(TimeInterval, evtIn, out, mode, reference_filter, measure_filter, nominal_period, max_interval, n_intervals, n_unmatched, n_undated, n_events_dropped);

    IntervalMode                                        _mode = IntervalMode::to_reference;
    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _reference{};
    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _measured{};
    bool                                                _hasReferenceFilter = false;
    bool                                                _hasMeasureFilter   = false;
    EventStore                                          _events;

    std::optional<std::uint64_t> _lastReference; // ns, carried across work calls
    std::uint64_t                _lastReferenceError = 0U;
    std::optional<std::uint64_t> _lastMeasured;
    std::uint64_t                _tieCount = 0U;
    struct Awaiting {
        std::uint64_t                at;
        std::uint64_t                error;
        std::optional<std::uint64_t> before;
        std::uint64_t                beforeError;
    };
    std::deque<Awaiting> _awaiting;
    std::deque<T>        _results;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _mode               = parseMode(mode.value);
        _hasReferenceFilter = detail::compileOptionalFilter(reference_filter.value, std::string_view{}, _reference, "TimeInterval");
        _hasMeasureFilter   = detail::compileOptionalFilter(measure_filter.value, std::string_view{}, _measured, "TimeInterval");
        _events.clear();
        _lastReference      = std::nullopt;
        _lastReferenceError = 0U;
        _lastMeasured       = std::nullopt;
        _tieCount           = 0U;
        _awaiting.clear();
        _results.clear();
        n_intervals      = 0U;
        n_unmatched      = 0U;
        n_undated        = 0U;
        n_events_dropped = 0U;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& outSpan) {
        _events.drain(evtSpan);
        n_events_dropped = static_cast<gr::Size_t>(_events.dropped);

        for (const StoredEvent& stored : _events.ordered()) {
            if (!stored.dated()) {
                n_undated = n_undated + 1U;
                continue;
            }
            if (const auto measured = measure(stored); measured) {
                _results.push_back(*measured);
            }
        }
        _events.retire(_events.size());

        const std::size_t emitted = detail::drainInto(_results, outSpan);
        n_intervals               = n_intervals + static_cast<gr::Size_t>(emitted);
        outSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    constexpr static std::size_t kMaxAwaiting = 64UZ;

    void resolveAwaiting(std::uint64_t referenceAt, std::uint64_t referenceError) {
        while (!_awaiting.empty()) {
            const Awaiting waiting = _awaiting.front();
            if (waiting.at > referenceAt) {
                return;
            }
            _awaiting.pop_front();
            const double after = secondsBetween(referenceAt, waiting.at); // positive: the reference comes later
            if (!waiting.before.has_value()) {
                if (withinLimit(after)) {
                    _results.push_back(asOutput(-after, waiting.error, referenceError));
                } else {
                    std::ignore = countUnmatched();
                }
                continue;
            }
            const double before    = secondsBetween(waiting.at, *waiting.before);
            const bool   isEarlier = before <= after;
            const double interval  = isEarlier ? before : -after;
            if (withinLimit(interval)) {
                _results.push_back(asOutput(interval, waiting.error, isEarlier ? waiting.beforeError : referenceError));
            } else {
                std::ignore = countUnmatched();
            }
        }
    }

    [[nodiscard]] static IntervalMode parseMode(std::string_view text) noexcept {
        const auto named = gr::meta::parseEnum<IntervalMode>(text);
        return named && *named != IntervalMode::AUTO ? *named : IntervalMode::to_reference;
    }

    [[nodiscard]] bool isReference(const gr::property_map_view& event) { return _hasReferenceFilter && gr::trigger::BasicTriggerNameCtxMatcher::match(_reference, event) == gr::trigger::MatchResult::Matching; }

    [[nodiscard]] bool isMeasured(const gr::property_map_view& event) {
        if (!_hasMeasureFilter) {
            return true;
        }
        return gr::trigger::BasicTriggerNameCtxMatcher::match(_measured, event) == gr::trigger::MatchResult::Matching;
    }

    [[nodiscard]] static double secondsBetween(std::uint64_t later, std::uint64_t earlier) noexcept { //
        return static_cast<double>(static_cast<std::int64_t>(later) - static_cast<std::int64_t>(earlier)) * 1e-9;
    }

    [[nodiscard]] bool withinLimit(double interval) const noexcept { return max_interval <= 0. || std::abs(interval) <= max_interval.value; }

    [[nodiscard]] static T asOutput(double interval, std::uint64_t errorNsA, std::uint64_t errorNsB) noexcept {
        if constexpr (std::is_same_v<T, gr::UncertainValue<double>>) { // errors add in quadrature, being independent
            const double a = static_cast<double>(errorNsA) * 1e-9;
            const double b = static_cast<double>(errorNsB) * 1e-9;
            return gr::UncertainValue<double>{interval, std::sqrt(a * a + b * b)};
        } else {
            return interval;
        }
    }

    [[nodiscard]] std::optional<T> measure(const StoredEvent& stored) {
        const gr::property_map_view event{stored.event};
        const std::uint64_t         at     = *stored.at;
        const auto                  stated = event.template get_if<std::uint64_t>(gr::tag::TRIGGER_TIME_ERROR.key());
        const std::uint64_t         error  = stated ? *stated : 0U;

        if (isReference(event)) {
            if (_mode == IntervalMode::nearest) {
                resolveAwaiting(at, error);
            }
            _lastReference      = at;
            _lastReferenceError = error;
            return std::nullopt;
        }
        if (!isMeasured(event)) {
            return std::nullopt;
        }

        switch (_mode) {
        case IntervalMode::nearest: {
            if (_awaiting.size() >= kMaxAwaiting) {
                _awaiting.pop_front();
                n_unmatched = n_unmatched + 1U;
            }
            _awaiting.push_back(Awaiting{.at = at, .error = error, .before = _lastReference, .beforeError = _lastReferenceError});
            return std::nullopt;
        }
        case IntervalMode::tie: {
            if (nominal_period <= 0.) {
                n_unmatched = n_unmatched + 1U;
                return std::nullopt;
            }
            if (!_lastReference) {
                _lastReference      = at; // the first event fixes the grid's origin
                _lastReferenceError = error;
                _tieCount           = 0U;
                return std::nullopt;
            }
            ++_tieCount;
            const double ideal    = static_cast<double>(_tieCount) * nominal_period.value;
            const double interval = secondsBetween(at, *_lastReference) - ideal;
            return withinLimit(interval) ? std::optional{asOutput(interval, error, _lastReferenceError)} : countUnmatched();
        }
        case IntervalMode::to_previous: {
            const auto previous = _lastMeasured;
            _lastMeasured       = at;
            if (!previous) {
                return std::nullopt;
            }
            const double interval = secondsBetween(at, *previous);
            return withinLimit(interval) ? std::optional{asOutput(interval, error, error)} : countUnmatched();
        }
        case IntervalMode::paired: {
            if (!_lastReference) {
                n_unmatched = n_unmatched + 1U;
                return std::nullopt;
            }
            const double interval = secondsBetween(at, *_lastReference);
            _lastReference        = std::nullopt; // one reference serves one measurement
            return withinLimit(interval) ? std::optional{asOutput(interval, error, _lastReferenceError)} : countUnmatched();
        }
        case IntervalMode::to_reference:
        case IntervalMode::AUTO: {
            if (!_lastReference) {
                n_unmatched = n_unmatched + 1U;
                return std::nullopt;
            }
            const double interval = secondsBetween(at, *_lastReference);
            return withinLimit(interval) ? std::optional{asOutput(interval, error, _lastReferenceError)} : countUnmatched();
        }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<T> countUnmatched() {
        n_unmatched = n_unmatched + 1U;
        return std::nullopt;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TIMEINTERVAL_HPP
