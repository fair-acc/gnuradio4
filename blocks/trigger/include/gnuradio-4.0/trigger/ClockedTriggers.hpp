#ifndef GNURADIO_TRIGGER_CLOCKEDTRIGGERS_HPP
#define GNURADIO_TRIGGER_CLOCKEDTRIGGERS_HPP

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory_resource>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/trigger/DigitalLine.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::SerialPatternTrigger, [T], [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK(gr::blocks::trigger::SetupHoldTrigger, [T], [ int16_t, int32_t, float, double ])

namespace detail {

[[nodiscard]] inline bool wantsClockEdge(std::string_view edge, bool rising, bool changed) noexcept {
    if (!changed) {
        return false;
    }
    if (edge == "both") {
        return true;
    }
    return rising == (edge != "falling");
}

} // namespace detail

template<typename T>
struct SerialPatternTrigger : gr::Block<SerialPatternTrigger<T>> {
    using Description = Doc<R"(@brief report when the bits sampled on a clock's edges spell a pattern

    clk_in  ─0─1─0─1─0─1─0─1─▶     (no RxMarbles equivalent: a fault, not a transformation)
    in      ─0─1─1─0─1─0─1─1─▶     pattern = "101X"
    evtOut  ─────────────T───▶     the last four bits sampled read 1,0,1,1 -- X accepts either

The data line is sampled on the clock edge `clock_edge` selects, one bit per edge, oldest first.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortIn<T>    clk_in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "pattern", Doc<"bits in the order they arrive, oldest first; 1, 0 or X">> pattern;
    A<T, "threshold", Doc<"where the data line reads as a one">>                                  threshold       = T{};
    A<T, "hysteresis", Doc<"guard band about both thresholds">>                                   hysteresis      = T{};
    A<T, "clock threshold", Doc<"where the clock line reads as a one">>                           clock_threshold = T{};
    A<std::pmr::string, "clock edge", Doc<"rising|falling|both, which edge samples the data">>    clock_edge      = std::pmr::string("rising");
    A<float, "sample rate", Doc<"Hz, dates a trigger, 0 = unknown">>                              sample_rate     = 0.f;
    A<std::pmr::string, "trigger name">                                                           trigger_name    = std::pmr::string("serial_pattern");

    A<gr::Size_t, "n bits", Doc<"clock edges sampled">>                                 n_bits           = 0U;
    A<gr::Size_t, "n triggers">                                                         n_triggers       = 0U;
    A<gr::Size_t, "n events dropped", Doc<"triggers the event output had no room for">> n_events_dropped = 0U;

    GR_MAKE_REFLECTABLE(SerialPatternTrigger, evtOut, in, clk_in, out, pattern, threshold, hysteresis, clock_threshold, clock_edge, sample_rate, trigger_name, n_bits, n_triggers, n_events_dropped);

    DigitalLine<T>        _data;
    DigitalLine<T>        _clock;
    TimeBase              _time;
    std::size_t           _streamIndex = 0UZ;
    std::pmr::string      _shifted;
    detail::PendingEvents _reports;

    void start() {
        _data.configure(hysteresis, threshold);
        _clock.configure(hysteresis, clock_threshold);
        _shifted.clear();
        _streamIndex = 0UZ;
        _reports.clear();
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _data.configure(hysteresis, threshold);
        _clock.configure(hysteresis, clock_threshold);
        _shifted.clear();
        _time.setRate(static_cast<double>(sample_rate));
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& clkSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples = std::min({inSpan.size(), clkSpan.size(), outSpan.size()});
        for (const auto& tag : inSpan.rawTags()) {
            _time.adopt(gr::property_map_view{tag.map}, _streamIndex + (tag.index - inSpan.streamIndex));
        }

        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            _data.observe(inSpan[i]);
            _clock.observe(clkSpan[i]);
            outSpan[i] = inSpan[i];
            if (!detail::wantsClockEdge(clock_edge.value, _clock.high, _clock.changed)) {
                continue;
            }
            n_bits = n_bits + 1U;
            shift(_data.high);
            if (matches()) {
                _reports.push(datedTrigger(_time, _streamIndex + i, gr::UncertainValue<float>{}, trigger_name.value, std::string_view{}));
                n_triggers = n_triggers + 1U;
            }
        }
        _streamIndex += nSamples;

        const std::size_t emitted = _reports.drainInto(evtOutSpan, 0UZ, "SerialPatternTrigger", this->unique_name.value());
        n_events_dropped          = static_cast<gr::Size_t>(_reports.dropped);
        evtOutSpan.publish(emitted);
        outSpan.publish(nSamples);
        if (!inSpan.consume(nSamples) || !clkSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void shift(bool bit) {
        _shifted.push_back(bit ? '1' : '0');
        if (_shifted.size() > pattern.value.size()) {
            _shifted.erase(_shifted.begin());
        }
    }

    [[nodiscard]] bool matches() const {
        if (pattern.value.empty() || _shifted.size() != pattern.value.size()) {
            return false;
        }
        return std::ranges::equal(pattern.value, _shifted, [](char wanted, char seen) { return wanted == 'X' || wanted == 'x' || wanted == seen; });
    }
};

template<typename T>
struct SetupHoldTrigger : gr::Block<SetupHoldTrigger<T>> {
    using Description = Doc<R"(@brief report a data change too close to a clock edge, before it or after it

    clk_in  ─0─1─0─1─0─1─0─1─▶     (no RxMarbles equivalent: a fault, not a transformation)
    in      ─0─0─0─0─1─1─1─1─▶     setup = 2, hold = 2
    evtOut  ───────────T─────▶     the change at 4 sits inside the window of the edge at 3

This is the trigger that catches the fault rather than the signal.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortIn<T>    clk_in;
    gr::PortOut<T>   out;

    A<gr::Size_t, "setup samples", Doc<"data settled before the edge">>                           setup_samples   = 1U;
    A<gr::Size_t, "hold samples", Doc<"how long it must stay settled after the edge">>            hold_samples    = 1U;
    A<T, "threshold", Doc<"where the data line reads as a one">>                                  threshold       = T{};
    A<T, "hysteresis", Doc<"guard band about both thresholds">>                                   hysteresis      = T{};
    A<T, "clock threshold", Doc<"where the clock line reads as a one">>                           clock_threshold = T{};
    A<std::pmr::string, "clock edge", Doc<"rising|falling|both, which edge the window is about">> clock_edge      = std::pmr::string("rising");
    A<float, "sample rate", Doc<"Hz, dates a trigger, 0 = unknown">>                              sample_rate     = 0.f;
    A<std::pmr::string, "trigger name">                                                           trigger_name    = std::pmr::string("setup_hold_violation");

    A<gr::Size_t, "n checks", Doc<"clock edges judged">>                                n_checks         = 0U;
    A<gr::Size_t, "n violations">                                                       n_violations     = 0U;
    A<gr::Size_t, "n events dropped", Doc<"triggers the event output had no room for">> n_events_dropped = 0U;

    GR_MAKE_REFLECTABLE(SetupHoldTrigger, evtOut, in, clk_in, out, setup_samples, hold_samples, threshold, hysteresis, clock_threshold, clock_edge, sample_rate, trigger_name, n_checks, n_violations, n_events_dropped);

    DigitalLine<T>          _data;
    DigitalLine<T>          _clock;
    TimeBase                _time;
    std::size_t             _streamIndex = 0UZ;
    std::size_t             _lastChange  = 0UZ; // absolute index of the newest data transition
    bool                    _everChanged = false;
    std::deque<std::size_t> _awaiting;
    detail::PendingEvents   _reports;

    void start() {
        _data.configure(hysteresis, threshold);
        _clock.configure(hysteresis, clock_threshold);
        _streamIndex = 0UZ;
        _lastChange  = 0UZ;
        _everChanged = false;
        _awaiting.clear();
        _reports.clear();
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _data.configure(hysteresis, threshold);
        _clock.configure(hysteresis, clock_threshold);
        _time.setRate(static_cast<double>(sample_rate));
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& clkSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples = std::min({inSpan.size(), clkSpan.size(), outSpan.size()});
        for (const auto& tag : inSpan.rawTags()) {
            _time.adopt(gr::property_map_view{tag.map}, _streamIndex + (tag.index - inSpan.streamIndex));
        }

        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            const std::size_t at = _streamIndex + i;
            _data.observe(inSpan[i]);
            _clock.observe(clkSpan[i]);
            outSpan[i] = inSpan[i];
            if (_data.changed) {
                _lastChange  = at;
                _everChanged = true;
            }
            if (detail::wantsClockEdge(clock_edge.value, _clock.high, _clock.changed)) {
                _awaiting.push_back(at);
                n_checks = n_checks + 1U;
            }
            judgeThoseDueAt(at);
        }
        _streamIndex += nSamples;

        const std::size_t emitted = _reports.drainInto(evtOutSpan, 0UZ, "SetupHoldTrigger", this->unique_name.value());
        n_events_dropped          = static_cast<gr::Size_t>(_reports.dropped);
        evtOutSpan.publish(emitted);
        outSpan.publish(nSamples);
        if (!inSpan.consume(nSamples) || !clkSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void judgeThoseDueAt(std::size_t position) {
        while (!_awaiting.empty() && _awaiting.front() + hold_samples <= position) {
            const std::size_t edge = _awaiting.front();
            _awaiting.pop_front();
            const std::size_t earliest = edge >= setup_samples ? edge - setup_samples : 0UZ;
            if (_everChanged && _lastChange >= earliest && _lastChange <= edge + hold_samples) {
                _reports.push(datedTrigger(_time, edge, gr::UncertainValue<float>{}, trigger_name.value, std::string_view{}));
                n_violations = n_violations + 1U;
            }
        }
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_CLOCKEDTRIGGERS_HPP
