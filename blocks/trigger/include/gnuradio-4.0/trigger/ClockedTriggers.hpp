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

/** one event per sample at most: the shared emit-or-count step of both clocked triggers */
[[nodiscard]] inline bool emitDated(auto& evtOutSpan, std::size_t& nEmitted, const TimeBase& time, std::size_t at, std::string_view name) noexcept { //
    if (!gr::emit(evtOutSpan, nEmitted, kEventKeys, [&](gr::property_map_view& event) { return writeDatedTrigger(event, time, at, gr::UncertainValue<float>{}, name, std::string_view{}); })) {
        return false;
    }
    ++nEmitted;
    return true;
}

} // namespace detail

template<typename T>
struct SerialPatternTrigger : gr::Block<SerialPatternTrigger<T>> {
    using Description = Doc<R"(@brief report when the bits sampled on a clock's edges spell a pattern

    clk_in  ─0─1─0─1─0─1─0─1─▶     (no RxMarbles equivalent: a fault, not a transformation)
    in      ─0─1─1─0─1─0─1─1─▶     pattern = "101X"
    evtOut  ─────────────T───▶     the last four bits sampled read 1,0,1,1 -- X accepts either

The data line is sampled on the clock edge `clock_edge` selects, one bit per edge, oldest first. A `sample_rate` of 0 means not yet known.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortIn<T>    clk_in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "pattern", Doc<"bits in the order they arrive, oldest first; 1, 0 or X; at most 64">> pattern;
    A<T, "threshold", Doc<"where the data line reads as a one">>                                              threshold       = T{};
    A<T, "hysteresis", Doc<"guard band about both thresholds">>                                               hysteresis      = T{};
    A<T, "clock threshold", Doc<"where the clock line reads as a one">>                                       clock_threshold = T{};
    A<std::pmr::string, "clock edge", Doc<"rising|falling|both, which edge samples the data">>                clock_edge      = std::pmr::string("rising");
    gr::SampleRate                                                                                            sample_rate     = 0.f;
    gr::TriggerName                                                                                           trigger_name    = std::pmr::string("serial_pattern");

    mutable A<gr::Size_t, "n bits", Doc<"clock edges sampled; host path only">>                            n_bits           = 0U;
    mutable A<gr::Size_t, "n triggers", Doc<"host path only">>                                             n_triggers       = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"triggers a full event output refused; host path only">> n_events_dropped = 0U;

    /** the newest bit is bit 0 */
    std::uint64_t _pattern_bits = 0U;
    /** clear where the pattern says X */
    std::uint64_t _pattern_care = 0U;
    /** 0 never matches */
    gr::Size_t    _pattern_length = 0U;
    EdgeSelection _clock_edge     = EdgeSelection::rising;
    gr::Size_t    _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(SerialPatternTrigger, evtOut, in, clk_in, out, pattern, threshold, hysteresis, clock_threshold, clock_edge, sample_rate, trigger_name, n_bits, n_triggers, n_events_dropped, _pattern_bits, _pattern_care, _pattern_length, _clock_edge, _settings_epoch);

    mutable DigitalLine<T> _data;
    mutable DigitalLine<T> _clock;
    mutable TimeBase       _time;
    mutable std::uint64_t  _shifted    = 0U;
    mutable std::size_t    _nShifted   = 0UZ;
    mutable gr::Size_t     _stateEpoch = std::numeric_limits<gr::Size_t>::max();

    void start() {
        _time.reset();
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _pattern_bits   = 0U;
        _pattern_care   = 0U;
        _pattern_length = static_cast<gr::Size_t>(pattern.value.size());
        if (pattern.value.size() > 64UZ || !std::ranges::all_of(pattern.value, [](char c) { return c == '0' || c == '1' || c == 'X' || c == 'x'; })) {
            gr::log::warning("SerialPatternTrigger: pattern '{}' is not up to 64 of 1, 0 and X, and never matches", pattern.value);
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
            _data.configure(hysteresis, threshold);
            _clock.configure(hysteresis, clock_threshold);
            _shifted  = 0U;
            _nShifted = 0UZ;
            _time.setRate(static_cast<double>(sample_rate));
            _stateEpoch = _settings_epoch;
        }
        for (const auto& tag : inSpan.rawTags()) {
            std::ignore = _time.adopt(gr::property_map_view{tag.map}, tag.index);
        }

        const std::size_t nSpan    = std::min({inSpan.size(), clkSpan.size(), outSpan.size()});
        std::size_t       nEmitted = 0UZ;
        std::size_t       n        = 0UZ;
        for (; n < nSpan; ++n) {
            if (evtOutSpan.isConnected && nEmitted > 0UZ && nEmitted == evtOutSpan.size()) {
                break;
            }
            _data.observe(inSpan[n]);
            _clock.observe(clkSpan[n]);
            outSpan[n] = inSpan[n];
            if (!detail::wantsClockEdge(_clock_edge, _clock.high, _clock.changed)) {
                continue;
            }
            n_bits    = n_bits + 1U;
            _shifted  = (_shifted << 1U) | (_data.high ? 1U : 0U);
            _nShifted = std::min(_nShifted + 1UZ, 64UZ);
            if (_pattern_length > 0U && _nShifted >= _pattern_length && (_shifted & _pattern_care) == _pattern_bits) {
                n_triggers = n_triggers + 1U;
                if (evtOutSpan.isConnected && !detail::emitDated(evtOutSpan, nEmitted, _time, inSpan.streamIndex + n, trigger_name.value)) {
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

    clk_in  ─0─1─0─1─0─1─0─1─▶     (no RxMarbles equivalent: a fault, not a transformation)
    in      ─0─0─0─0─1─1─1─1─▶     setup = 2, hold = 2
    evtOut  ───────────T─────▶     the change at 4 sits inside the window of the edge at 3

This is the trigger that catches the fault rather than the signal. A `sample_rate` of 0 means not yet known.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortIn<T>    clk_in;
    gr::PortOut<T>   out;

    A<gr::Size_t, "setup samples", Doc<"data settled before the edge">>                            setup_samples   = 1U;
    A<gr::Size_t, "hold samples", Doc<"how long it must stay settled after the edge; at most 63">> hold_samples    = 1U;
    A<T, "threshold", Doc<"where the data line reads as a one">>                                   threshold       = T{};
    A<T, "hysteresis", Doc<"guard band about both thresholds">>                                    hysteresis      = T{};
    A<T, "clock threshold", Doc<"where the clock line reads as a one">>                            clock_threshold = T{};
    A<std::pmr::string, "clock edge", Doc<"rising|falling|both, which edge the window is about">>  clock_edge      = std::pmr::string("rising");
    gr::SampleRate                                                                                 sample_rate     = 0.f;
    gr::TriggerName                                                                                trigger_name    = std::pmr::string("setup_hold_violation");

    mutable A<gr::Size_t, "n checks", Doc<"clock edges judged; host path only">>                           n_checks         = 0U;
    mutable A<gr::Size_t, "n violations", Doc<"host path only">>                                           n_violations     = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"triggers a full event output refused; host path only">> n_events_dropped = 0U;

    constexpr static std::size_t kMaxAwaiting = 64UZ;

    EdgeSelection _clock_edge     = EdgeSelection::rising;
    gr::Size_t    _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(SetupHoldTrigger, evtOut, in, clk_in, out, setup_samples, hold_samples, threshold, hysteresis, clock_threshold, clock_edge, sample_rate, trigger_name, n_checks, n_violations, n_events_dropped, _clock_edge, _settings_epoch);

    mutable DigitalLine<T> _data;
    mutable DigitalLine<T> _clock;
    mutable TimeBase       _time;
    mutable std::size_t    _lastChange  = 0UZ;
    mutable bool           _everChanged = false;
    /** clock edges whose hold window is still open, oldest at _firstAwaiting */
    mutable std::array<std::size_t, kMaxAwaiting> _awaiting{};
    mutable std::size_t                           _firstAwaiting = 0UZ;
    mutable std::size_t                           _nAwaiting     = 0UZ;
    mutable gr::Size_t                            _stateEpoch    = std::numeric_limits<gr::Size_t>::max();

    void start() {
        _time.reset();
        _lastChange  = 0UZ;
        _everChanged = false;
        _nAwaiting   = 0UZ;
        _stateEpoch  = std::numeric_limits<gr::Size_t>::max();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if (hold_samples >= kMaxAwaiting) {
            gr::log::warning("SetupHoldTrigger: hold_samples {} exceeds the {} the block holds and is lowered to it", hold_samples.value, kMaxAwaiting - 1UZ);
            hold_samples = static_cast<gr::Size_t>(kMaxAwaiting - 1UZ);
        }
        _clock_edge     = detail::parseEdgeSelection(clock_edge.value);
        _settings_epoch = _settings_epoch + 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& clkSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) const {
        if (_stateEpoch != _settings_epoch) {
            _data.configure(hysteresis, threshold);
            _clock.configure(hysteresis, clock_threshold);
            _time.setRate(static_cast<double>(sample_rate));
            _stateEpoch = _settings_epoch;
        }
        for (const auto& tag : inSpan.rawTags()) {
            std::ignore = _time.adopt(gr::property_map_view{tag.map}, tag.index);
        }

        const std::size_t nSpan    = std::min({inSpan.size(), clkSpan.size(), outSpan.size()});
        std::size_t       nEmitted = 0UZ;
        std::size_t       n        = 0UZ;
        for (; n < nSpan; ++n) {
            if (evtOutSpan.isConnected && nEmitted > 0UZ && nEmitted == evtOutSpan.size()) {
                break;
            }
            const std::size_t at = inSpan.streamIndex + n;
            _data.observe(inSpan[n]);
            _clock.observe(clkSpan[n]);
            outSpan[n] = inSpan[n];
            if (_data.changed) {
                _lastChange  = at;
                _everChanged = true;
            }
            if (detail::wantsClockEdge(_clock_edge, _clock.high, _clock.changed) && _nAwaiting < kMaxAwaiting) {
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
                if (evtOutSpan.isConnected && !detail::emitDated(evtOutSpan, nEmitted, _time, edge, trigger_name.value)) {
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
