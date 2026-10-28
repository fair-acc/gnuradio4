#ifndef GNURADIO_TRIGGER_CLOCKEDTRIGGERS_HPP
#define GNURADIO_TRIGGER_CLOCKEDTRIGGERS_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/trigger/DigitalLine.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::SerialPatternTrigger, [T], [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK(gr::blocks::trigger::SetupHoldTrigger, [T], [ int16_t, int32_t, float, double ])

namespace detail {

[[nodiscard]] constexpr bool wantsClockEdge(EdgeSelection edge, bool rising, bool changed) noexcept { //
    return changed && (edge == EdgeSelection::both || rising == (edge != EdgeSelection::falling));
}

[[nodiscard]] inline EdgeSelection parseEdgeSelection(std::string_view edge) noexcept { //
    return edge == "both" ? EdgeSelection::both : (edge == "falling" ? EdgeSelection::falling : EdgeSelection::rising);
}

/** the tags on one sample carry a gap: samples were dropped before it */
template<typename TIterator>
[[nodiscard]] bool hasGap(TIterator first, TIterator last) noexcept {
    return std::any_of(first, last, [](const auto& tag) { return isGap(gr::property_map_view{tag.map}); });
}

/** one event per sample at most: the shared emit-or-count step of both clocked triggers */
[[nodiscard]] inline bool emitDated(auto& evtOutSpan, std::size_t& nEmitted, const TimeBase& time, std::size_t at, std::string_view name) noexcept { //
    if (!gr::emit(evtOutSpan, nEmitted, kDatedTriggerKeys, [&](gr::property_map_view& event) { return writeDatedTrigger(event, time, at, 0., 0.f, name, std::string_view{}); })) {
        return false;
    }
    ++nEmitted;
    return true;
}

} // namespace detail

/** the data and the clock line of a clocked trigger, read with one band, dated by one time base */
template<typename T>
struct ClockedLines {
    DigitalLine<T> data;
    DigitalLine<T> clock;
    TimeBase       time;

    void configure(T threshold, T clockThreshold, T hysteresis) {
        data.configure({.level = threshold, .hysteresis = hysteresis});
        clock.configure({.level = clockThreshold, .hysteresis = hysteresis});
    }

    void adoptTimes(const gr::InputSpanLike auto& inSpan) {
        for (const auto& tag : inSpan.rawTags()) {
            std::ignore = time.adopt(gr::property_map_view{tag.map}, tag.index);
        }
    }

    void observe(T dataSample, T clockSample) {
        data.observe(dataSample);
        clock.observe(clockSample);
    }
};

template<typename T>
struct SerialPatternTrigger : gr::Block<SerialPatternTrigger<T>> {
    using Description = Doc<R"(@brief report when the bits sampled on a clock's edges spell a pattern

    clk     ─0─1─0─1─0─1─0─1─▶     (no RxMarbles equivalent: a fault, not a transformation)
    in      ─0─1─1─0─1─0─1─1─▶     pattern = "101X"
    evtOut  ─────────────T───▶     the last four bits sampled read 1,0,1,1 -- X accepts either
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortIn<T>    clk;
    gr::PortOut<T>   out;

    A<std::pmr::string, "pattern", Doc<"bits in the order they arrive, oldest first; 1, 0 or X; at most 64">> pattern;
    A<T, "threshold", Doc<"where the data line reads as a one">>                                              threshold       = T{};
    A<T, "hysteresis", Doc<"full band width about both thresholds">>                                          hysteresis      = T{};
    A<T, "clock threshold", Doc<"where the clock line reads as a one">>                                       clock_threshold = T{};
    A<std::pmr::string, "clock edge", Doc<"rising|falling|both, which edge samples the data">>                clock_edge      = std::pmr::string("rising");
    gr::SampleRate                                                                                            sample_rate     = 0.f;
    gr::TriggerName                                                                                           trigger_name;

    mutable A<gr::Size_t, "n bits", Doc<"clock edges sampled">>                            n_bits           = 0U;
    mutable A<gr::Size_t, "n triggers", Doc<"patterns reported">>                          n_triggers       = 0U;
    mutable A<gr::Size_t, "n gaps", Doc<"gaps in the data; each restarts the register">>   n_gaps           = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"triggers a full event output refused">> n_events_dropped = 0U;

    /** the newest bit is bit 0 */
    std::uint64_t _pattern_bits = 0U;
    /** clear where the pattern says X */
    std::uint64_t _pattern_care = 0U;
    /** 0 never matches */
    gr::Size_t    _pattern_length = 0U;
    EdgeSelection _clock_edge     = EdgeSelection::rising;
    gr::Size_t    _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(SerialPatternTrigger, evtOut, in, clk, out, pattern, threshold, hysteresis, clock_threshold, clock_edge, sample_rate, trigger_name, n_bits, n_triggers, n_gaps, n_events_dropped, _pattern_bits, _pattern_care, _pattern_length, _clock_edge, _settings_epoch);

    mutable ClockedLines<T> _lines;
    mutable std::uint64_t   _shifted    = 0U;
    mutable std::size_t     _nShifted   = 0UZ;
    mutable gr::Size_t      _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    detail::ErrorMemory     _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        detail::nameTriggerAfterBlock(*this);
        _lines.time.reset();
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _pattern_bits   = 0U;
        _pattern_care   = 0U;
        _pattern_length = static_cast<gr::Size_t>(pattern.value.size());
        if (pattern.value.size() > 64UZ || !std::ranges::all_of(pattern.value, [](char c) { return c == '0' || c == '1' || c == 'X' || c == 'x'; })) {
            detail::reportError(*this, "settingsChanged()", std::format("pattern '{}' is not up to 64 of 1, 0 and X, and never matches", pattern.value));
            _pattern_length = 0U;
        }
        /** pattern[0] is the oldest bit, the highest of the register */
        for (std::size_t k = 0UZ; k < _pattern_length; ++k) {
            const std::uint64_t bit = std::uint64_t{1} << (_pattern_length - 1UZ - k);
            _pattern_care |= pattern.value[k] == '0' || pattern.value[k] == '1' ? bit : 0U;
            _pattern_bits |= pattern.value[k] == '1' ? bit : 0U;
        }
        _clock_edge     = detail::parseEdgeSelection(clock_edge.value);
        _settings_epoch = _settings_epoch + 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& clkSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) const {
        if (_stateEpoch != _settings_epoch) {
            _lines.configure(threshold, clock_threshold, hysteresis);
            _shifted  = 0U;
            _nShifted = 0UZ;
            _lines.time.setRate(static_cast<double>(sample_rate));
            _stateEpoch = _settings_epoch;
        }
        _lines.adoptTimes(inSpan);

        const std::size_t nSpan    = std::min({inSpan.size(), clkSpan.size(), outSpan.size()});
        std::size_t       nEmitted = 0UZ;
        std::size_t       n        = 0UZ;
        auto              tags     = inSpan.rawTags();
        auto              next     = std::ranges::begin(tags);
        for (; n < nSpan; ++n) {
            if (evtOutSpan.isConnected && nEmitted > 0UZ && nEmitted == evtOutSpan.size()) {
                break;
            }
            const auto first = detail::tagsOn(next, std::ranges::end(tags), inSpan.streamIndex + n);
            if (detail::hasGap(first, next)) { // nothing is known across a gap: the register and both lines restart
                _lines.configure(threshold, clock_threshold, hysteresis);
                _shifted  = 0U;
                _nShifted = 0UZ;
                n_gaps    = n_gaps + 1U;
            }
            _lines.observe(inSpan[n], clkSpan[n]);
            outSpan[n] = inSpan[n];
            if (!detail::wantsClockEdge(_clock_edge, _lines.clock.high(), _lines.clock.changed)) {
                continue;
            }
            n_bits = n_bits + 1U;
            if (_lines.data.state() == gr::trigger::LineState::unknown) { // an unknown bit is no bit: the register restarts
                _shifted  = 0U;
                _nShifted = 0UZ;
                continue;
            }
            _shifted  = (_shifted << 1U) | (_lines.data.high() ? 1U : 0U);
            _nShifted = std::min(_nShifted + 1UZ, 64UZ);
            if (_pattern_length > 0U && _nShifted >= _pattern_length && (_shifted & _pattern_care) == _pattern_bits) {
                n_triggers = n_triggers + 1U;
                if (evtOutSpan.isConnected && !detail::emitDated(evtOutSpan, nEmitted, _lines.time, inSpan.streamIndex + n, trigger_name.value)) {
                    n_events_dropped = n_events_dropped + 1U;
                }
            }
        }

        evtOutSpan.publish(nEmitted);
        outSpan.publish(n);
        if (!inSpan.consume(n) || !clkSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
struct SetupHoldTrigger : gr::Block<SetupHoldTrigger<T>> {
    using Description = Doc<R"(@brief report a data change too close to a clock edge, before it or after it

    clk     ─0─1─0─1─0─1─0─1─▶     (no RxMarbles equivalent: a fault, not a transformation)
    in      ─0─0─0─0─1─1─1─1─▶     setup = 2, hold = 2
    evtOut  ───────────T─────▶     the change at 4 sits inside the window of the edge at 3
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortIn<T>    clk;
    gr::PortOut<T>   out;

    A<gr::Size_t, "setup samples", Doc<"data settled before the edge">>                            setup_samples   = 1U;
    A<gr::Size_t, "hold samples", Doc<"how long it must stay settled after the edge; at most 63">> hold_samples    = 1U;
    A<T, "threshold", Doc<"where the data line reads as a one">>                                   threshold       = T{};
    A<T, "hysteresis", Doc<"full band width about both thresholds">>                               hysteresis      = T{};
    A<T, "clock threshold", Doc<"where the clock line reads as a one">>                            clock_threshold = T{};
    A<std::pmr::string, "clock edge", Doc<"rising|falling|both, which edge the window is about">>  clock_edge      = std::pmr::string("rising");
    gr::SampleRate                                                                                 sample_rate     = 0.f;
    gr::TriggerName                                                                                trigger_name;

    mutable A<gr::Size_t, "n checks", Doc<"clock edges judged">>                           n_checks         = 0U;
    mutable A<gr::Size_t, "n violations", Doc<"changes inside a setup or hold window">>    n_violations     = 0U;
    mutable A<gr::Size_t, "n gaps", Doc<"gaps in the data; each drops the open windows">>  n_gaps           = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"triggers a full event output refused">> n_events_dropped = 0U;

    constexpr static std::size_t kMaxAwaiting = 64UZ;

    EdgeSelection _clock_edge     = EdgeSelection::rising;
    gr::Size_t    _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(SetupHoldTrigger, evtOut, in, clk, out, setup_samples, hold_samples, threshold, hysteresis, clock_threshold, clock_edge, sample_rate, trigger_name, n_checks, n_violations, n_gaps, n_events_dropped, _clock_edge, _settings_epoch);

    mutable ClockedLines<T> _lines;
    mutable std::size_t     _lastChange  = 0UZ;
    mutable bool            _everChanged = false;
    /** clock edges whose hold window is still open, oldest at _firstAwaiting */
    mutable std::array<std::size_t, kMaxAwaiting> _awaiting{};
    mutable std::size_t                           _firstAwaiting = 0UZ;
    mutable std::size_t                           _nAwaiting     = 0UZ;
    mutable gr::Size_t                            _stateEpoch    = std::numeric_limits<gr::Size_t>::max();
    detail::ErrorMemory                           _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        detail::nameTriggerAfterBlock(*this);
        _lines.time.reset();
        _lastChange  = 0UZ;
        _everChanged = false;
        _nAwaiting   = 0UZ;
        _stateEpoch  = std::numeric_limits<gr::Size_t>::max();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if (hold_samples >= kMaxAwaiting) {
            const std::string message = std::format("hold_samples {} exceeds the {} the block holds and is lowered to it", hold_samples.value, kMaxAwaiting - 1UZ);
            detail::reportError(*this, "settingsChanged()", message);
            hold_samples = static_cast<gr::Size_t>(kMaxAwaiting - 1UZ);
        }
        _clock_edge     = detail::parseEdgeSelection(clock_edge.value);
        _settings_epoch = _settings_epoch + 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& clkSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) const {
        if (_stateEpoch != _settings_epoch) {
            _lines.configure(threshold, clock_threshold, hysteresis);
            _lines.time.setRate(static_cast<double>(sample_rate));
            _stateEpoch = _settings_epoch;
        }
        _lines.adoptTimes(inSpan);

        const std::size_t nSpan    = std::min({inSpan.size(), clkSpan.size(), outSpan.size()});
        std::size_t       nEmitted = 0UZ;
        std::size_t       n        = 0UZ;
        auto              tags     = inSpan.rawTags();
        auto              next     = std::ranges::begin(tags);
        for (; n < nSpan; ++n) {
            if (evtOutSpan.isConnected && nEmitted > 0UZ && nEmitted == evtOutSpan.size()) {
                break;
            }
            const std::size_t at    = inSpan.streamIndex + n;
            const auto        first = detail::tagsOn(next, std::ranges::end(tags), at);
            if (detail::hasGap(first, next)) { // a window across a gap proves nothing: the open ones are dropped
                _lines.configure(threshold, clock_threshold, hysteresis);
                _everChanged = false;
                _nAwaiting   = 0UZ;
                n_gaps       = n_gaps + 1U;
            }
            _lines.observe(inSpan[n], clkSpan[n]);
            outSpan[n] = inSpan[n];
            if (_lines.data.changed) {
                _lastChange  = at;
                _everChanged = true;
            }
            if (detail::wantsClockEdge(_clock_edge, _lines.clock.high(), _lines.clock.changed) && _nAwaiting < kMaxAwaiting) {
                _awaiting[(_firstAwaiting + _nAwaiting) % kMaxAwaiting] = at;
                ++_nAwaiting;
                n_checks = n_checks + 1U;
            }
            for (; _nAwaiting > 0UZ && _awaiting[_firstAwaiting] + hold_samples <= at; _firstAwaiting = (_firstAwaiting + 1UZ) % kMaxAwaiting, --_nAwaiting) {
                const std::size_t edge     = _awaiting[_firstAwaiting];
                const std::size_t earliest = edge >= setup_samples ? edge - setup_samples : 0UZ;
                if (!_everChanged || _lastChange < earliest || _lastChange > edge + hold_samples) {
                    continue;
                }
                n_violations = n_violations + 1U;
                if (evtOutSpan.isConnected && !detail::emitDated(evtOutSpan, nEmitted, _lines.time, edge, trigger_name.value)) {
                    n_events_dropped = n_events_dropped + 1U;
                }
            }
        }

        evtOutSpan.publish(nEmitted);
        outSpan.publish(n);
        if (!inSpan.consume(n) || !clkSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_CLOCKEDTRIGGERS_HPP
