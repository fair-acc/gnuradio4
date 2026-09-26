#ifndef GNURADIO_TRIGGER_GATE_HPP
#define GNURADIO_TRIGGER_GATE_HPP

#include <algorithm>
#include <cstdint>
#include <format>
#include <map>
#include <memory_resource>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/ConditionSource.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

GR_REGISTER_BLOCK(gr::blocks::trigger::Gate, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

namespace gr::blocks::trigger {

enum class GateMode : std::uint8_t { once, toggle, cooldown, wait, take_until, skip_until };
enum class ClosedPolicy : std::uint8_t { drop, hold };
enum class Retrigger : std::uint8_t { ignore, restart, extend };

template<typename T>
struct Gate : gr::Block<Gate<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief forward a stream only while a trigger says it may: six modes [takeUntil, skipUntil, throttle]

    once, n_open=3   in: a b T:c d e f T:g h   out: . . T:c d e . . .
    toggle           in: a b T:c d T:e f T:g   out: . . T:c d . . T:g
    cooldown 2/3     in: a T:b c T:d e f T:g   out: . T:b . . . f T:g
    wait, n_delay=2  in: a T:b c d e f         out: . . . d e f
    take_until       in: a b c T:d e f         out: a b c . . .
    skip_until       in: a b c T:d e f         out: . . . T:d e f

`closed_policy`: `drop` consumes a suppressed sample, `hold` leaves it for when the gate reopens -- which needs `control` or
`evtIn`, since a held input never advances to a reopening stream tag.

 [1] example: https://rxmarbles.com/#takeUntil, https://rxmarbles.com/#skipUntil, https://rxmarbles.com/#throttle
 [2] detailed documentation: https://reactivex.io/documentation/operators/takeuntil.html, https://reactivex.io/documentation/operators/skipuntil.html, https://reactivex.io/documentation/operators/throttle.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                                   evtIn;
    gr::EventPortOut                                  evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>                                     in;
    gr::PortOut<T>                                    out;
    gr::PortIn<std::uint8_t, gr::Async, gr::Optional> control;

    A<std::pmr::string, "mode", Doc<"once|toggle|cooldown|wait|take_until|skip_until">> mode = std::pmr::string("once");
    A<std::pmr::string, "open filter", Doc<"trigger filter that opens the gate">>       open_filter;
    A<std::pmr::string, "close filter", Doc<"trigger filter that closes it">>           close_filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                     match_mode;
    A<gr::Size_t, "n open", Doc<"samples forwarded per open run">>                      n_open             = 1U;
    A<gr::Size_t, "n cooldown", Doc<"samples suppressed between open runs">>            n_cooldown         = 0U;
    A<gr::Size_t, "n delay", Doc<"samples excluded after the trigger, 'wait' only">>    n_delay            = 1U;
    A<std::pmr::string, "retrigger", Doc<"ignore|restart|extend">>                      retrigger          = std::pmr::string("ignore");
    A<std::pmr::string, "closed policy", Doc<"drop|hold">>                              closed_policy      = std::pmr::string("drop");
    A<bool, "initial state", Doc<"true = the gate starts open">>                        initial_state      = false;
    A<bool, "is open", Doc<"reflected gate state">>                                     is_open            = false;
    A<gr::Size_t, "n passed", Doc<"samples forwarded">>                                 n_passed           = 0U;
    A<gr::Size_t, "n suppressed", Doc<"samples consumed but not forwarded">>            n_suppressed       = 0U;
    A<gr::Size_t, "n triggers ignored">                                                 n_triggers_ignored = 0U;
    A<gr::Size_t, "n tags dropped", Doc<"tags dropped with samples">>                   n_tags_dropped     = 0U;
    A<gr::Size_t, "n invalid triggers", Doc<"undated triggers">>                        n_invalid_triggers = 0U;

    GR_MAKE_REFLECTABLE(Gate, evtIn, evtOut, in, out, control, mode, open_filter, close_filter, match_mode, n_open, n_cooldown, n_delay, retrigger, closed_policy, initial_state, is_open, n_passed, n_suppressed, n_triggers_ignored, n_tags_dropped, n_invalid_triggers);

    GateMode                                            _mode        = GateMode::once;
    std::size_t                                         _streamIndex = 0UZ;
    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _open{};
    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _close{};
    bool                                                _hasClose          = false;
    bool                                                _gateOpen          = false;
    std::uint64_t                                       _samplesLeftOpen   = 0UZ;
    std::uint64_t                                       _samplesLeftClosed = 0UZ; // samples left to suppress before reopening
    bool                                                _armed             = true;
    bool                                                _reportedOpen      = false;
    ClosedPolicy                                        _closedPolicy      = ClosedPolicy::drop;
    Retrigger                                           _retrigger         = Retrigger::ignore;
    bool                                                _reopenPathChecked = false;
    TimeBase                                            _time;
    EventStore                                          _events;
    std::pmr::string                                    _pendingError;

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& /*newSettings*/) {
        _mode = parseMode(mode.value);
        if (parseRetrigger(retrigger.value) != Retrigger::ignore && !hasWindow(_mode)) {
            refuseSetting<std::string>(retrigger, "retrigger", std::string("ignore"), oldSettings, std::format("mode '{}' has no open run to restart or extend", mode.value));
        }
        _retrigger = parseRetrigger(retrigger.value);
        if (match_mode == "interval" && _mode != GateMode::toggle) {
            refuseSetting<std::string>(match_mode, "match_mode", std::string(), oldSettings, std::format("only 'toggle' acts on both edges of an interval, not '{}'", mode.value));
        }
        if (n_open == 0U) {
            refuseSetting<gr::Size_t>(n_open, "n_open", 1U, oldSettings, "a run of no samples cannot be told from an unbounded one");
        }
        if (n_delay == 0U) {
            refuseSetting<gr::Size_t>(n_delay, "n_delay", 1U, oldSettings, "'wait' excludes at least the trigger sample");
        }
        _closedPolicy      = closed_policy == "hold" ? ClosedPolicy::hold : ClosedPolicy::drop;
        _reopenPathChecked = false;
        _time.reset();
        _events.clear();

        detail::compileFilterInto(open_filter.value, match_mode.value, _open, "Gate");
        _hasClose          = detail::compileOptionalFilter(close_filter.value, match_mode.value, _close, "Gate");
        _gateOpen          = initial_state || _mode == GateMode::take_until;
        _reportedOpen      = _gateOpen;
        _streamIndex       = 0UZ;
        _samplesLeftOpen   = 0UZ;
        _samplesLeftClosed = 0UZ;
        _armed             = true;
        is_open            = _gateOpen;
        n_passed           = 0U;
        n_suppressed       = 0U;
        n_triggers_ignored = 0U;
        n_tags_dropped     = 0U;
        n_invalid_triggers = 0U;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& controlSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        refuseHoldWithoutReopenPath(evtSpan, controlSpan);
        _events.drain(evtSpan);
        const std::size_t n          = std::min(inSpan.size(), outSpan.size());
        const auto        conditions = detail::collectConditions(inSpan, n);
        const bool        holds      = _closedPolicy == ClosedPolicy::hold && !conditions.endsStream();
        std::size_t       emitted    = 0UZ;
        std::size_t       consumed   = 0UZ;
        std::size_t       reported   = 0UZ;
        bool              applied    = false;

        for (std::size_t i = 0UZ; i < n; ++i) {
            const gr::property_map* tag = conditions.tagAt(i);
            if (tag != nullptr) {
                _time.adopt(gr::property_map_view{*tag}, _streamIndex + i);
            }
            if (tag != nullptr && applyMatch(gr::property_map_view{*tag})) {
                reported += detail::auditTrigger(gr::property_map_view{*tag}, n_invalid_triggers, evtOutSpan, reported, this->name, this->unique_name);
            }
            if (!applied) {
                applyControl(controlSpan);
                for (const StoredEvent& injected : _events.ordered()) {
                    if (applyMatch(gr::property_map_view{injected.event})) {
                        reported += detail::auditTrigger(gr::property_map_view{injected.event}, n_invalid_triggers, evtOutSpan, reported, this->name, this->unique_name);
                    }
                }
                _events.retire(_events.size());
                applied = true;
            }
            if (_gateOpen) {
                if (tag != nullptr) {
                    outSpan.publishTag(*tag, emitted);
                }
                outSpan[emitted++] = inSpan[i];
                ++consumed;
                ++n_passed;
                countDownOpenRun();
            } else {
                if (holds) {
                    break;
                }
                ++consumed;
                ++n_suppressed;
                if (tag != nullptr) {
                    n_tags_dropped = n_tags_dropped + 1U;
                }
                countDownCooldown();
            }
        }

        if (!applied) {
            applyControl(controlSpan);
            for (const StoredEvent& injected : _events.ordered()) {
                static_cast<void>(applyMatch(gr::property_map_view{injected.event}));
            }
            _events.retire(_events.size());
        }

        reported += reportStateChanges(evtOutSpan, reported);
        if (!_pendingError.empty() && reported < evtOutSpan.size() && gr::emitEvent(evtOutSpan, reported, detail::makeErrorEvent(_pendingError, this->unique_name.value()))) {
            ++reported;
            _pendingError.clear();
        }
        _streamIndex += consumed;
        is_open = _gateOpen;
        outSpan.publish(emitted);
        evtOutSpan.publish(reported);
        if (!controlSpan.consume(controlSpan.size()) || !inSpan.consume(consumed)) {
            return gr::work::Status::ERROR;
        }
        return consumed == 0UZ && n > 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
    }

private:
    [[nodiscard]] std::size_t reportStateChanges(auto& evtOutSpan, std::size_t index) {
        if (_gateOpen == _reportedOpen || index >= evtOutSpan.size()) {
            return 0UZ;
        }
        _reportedOpen = _gateOpen;

        gr::property_map event = detail::makeEvent(_gateOpen ? "opened" : "closed", this->unique_name.value());
        if (const auto stamp = _time.at(_streamIndex)) {
            event[std::string(gr::tag::TRIGGER_TIME.key())] = *stamp;
        }
        return gr::emitEvent(evtOutSpan, index, event).has_value() ? 1UZ : 0UZ;
    }

    [[nodiscard]] static GateMode parseMode(std::string_view text) noexcept { return gr::meta::parseEnum<GateMode>(text).value_or(GateMode::once); }

    [[nodiscard]] static Retrigger parseRetrigger(std::string_view text) noexcept { return gr::meta::parseEnum<Retrigger>(text).value_or(Retrigger::ignore); }

    [[nodiscard]] static bool hasWindow(GateMode gateMode) noexcept { return gateMode == GateMode::once || gateMode == GateMode::cooldown || gateMode == GateMode::wait; }

    template<typename TValue>
    void refuseSetting(auto& setting, std::string_view key, TValue fallback, const gr::property_map& oldSettings, std::string_view reason) {
        TValue previous = oldSettings.value_or<TValue>(std::string(key), fallback);
        if constexpr (std::is_arithmetic_v<TValue>) {
            if (previous == TValue{}) {
                previous = fallback;
            }
        }
        _pendingError = std::format("'{}' = '{}' refused ({}); keeping '{}'", key, setting.value, reason, previous);
        gr::log::warning("Gate: {}", _pendingError);
        setting = previous;
    }

    void refuseHoldWithoutReopenPath(const gr::InputSpanLike auto& evtSpan, const gr::InputSpanLike auto& controlSpan) {
        if (_reopenPathChecked || _closedPolicy != ClosedPolicy::hold) {
            return;
        }
        _reopenPathChecked = true;
        if (!evtSpan.isConnected && !controlSpan.isConnected) {
            _pendingError = std::pmr::string("'closed_policy' = 'hold' refused (neither 'evtIn' nor 'control' is connected to reopen the gate); keeping 'drop'");
            gr::log::warning("Gate: {}", _pendingError);
            closed_policy = std::pmr::string("drop");
            _closedPolicy = ClosedPolicy::drop;
        }
    }

    void retriggerRun(std::uint64_t window) noexcept {
        switch (_retrigger) {
        case Retrigger::restart: openFor(window); return;
        case Retrigger::extend:
            if (_gateOpen) {
                _samplesLeftOpen += window;
            } else {
                openFor(window);
            }
            return;
        case Retrigger::ignore: n_triggers_ignored = n_triggers_ignored + 1U; return;
        }
    }

    void openFor(std::uint64_t nSamples) noexcept {
        _gateOpen          = true;
        _samplesLeftOpen   = nSamples;
        _samplesLeftClosed = 0UZ;
    }

    void closeGate() noexcept {
        _gateOpen        = false;
        _samplesLeftOpen = 0UZ;
    }

    void countDownOpenRun() noexcept {
        if (_samplesLeftOpen > 0UZ && --_samplesLeftOpen == 0UZ) {
            closeGate();
            if (_mode == GateMode::cooldown) {
                _samplesLeftClosed = n_cooldown;
                if (_samplesLeftClosed == 0UZ) {
                    openFor(n_open);
                }
            }
        }
    }

    void countDownCooldown() noexcept {
        if (_samplesLeftClosed > 0UZ && --_samplesLeftClosed == 0UZ) {
            openFor(_mode == GateMode::cooldown ? static_cast<std::uint64_t>(n_open) : 0UZ);
        }
    }

    [[nodiscard]] bool applyMatch(const gr::property_map_view& view) {
        const bool closes = _hasClose && matches(_close, view);
        const bool opens  = !closes && matches(_open, view); // close wins where one sample matches both

        if (closes) {
            closeGate();
            if (_mode == GateMode::take_until) {
                _armed = false;
            }
            return true;
        }
        if (!opens) {
            return false;
        }

        switch (_mode) {
        case GateMode::once:
            if (_armed) {
                openFor(n_open);
                _armed = false;
            } else {
                retriggerRun(n_open);
            }
            break;
        case GateMode::toggle:
            if (_gateOpen) {
                closeGate(); // a closing trigger excludes its own sample
            } else {
                openFor(0UZ);
            }
            break;
        case GateMode::cooldown:
            if (_samplesLeftClosed == 0UZ) {
                openFor(1UZ); // the trigger sample, then the cooldown/open alternation
            } else {
                retriggerRun(1UZ);
            }
            break;
        case GateMode::wait:
            if (_samplesLeftClosed == 0UZ) {
                closeGate();
                _samplesLeftClosed = n_delay;
            } else if (_retrigger == Retrigger::restart) {
                _samplesLeftClosed = n_delay;
            } else if (_retrigger == Retrigger::extend) {
                _samplesLeftClosed += n_delay;
            } else {
                n_triggers_ignored = n_triggers_ignored + 1U;
            }
            break;
        case GateMode::take_until:
            closeGate();
            _armed = false;
            break;
        case GateMode::skip_until:
            openFor(0UZ); // the trigger sample is included
            break;
        }
        return true;
    }

    [[nodiscard]] static bool matches(gr::trigger::BasicTriggerNameCtxMatcher::MatchState& state, const gr::property_map_view& view) { return gr::trigger::BasicTriggerNameCtxMatcher::match(state, view) == gr::trigger::MatchResult::Matching; }

    void applyControl(gr::InputSpanLike auto& controlSpan) {
        for (const std::uint8_t wanted : controlSpan) {
            if (wanted != 0U) {
                openFor(0UZ);
            } else {
                closeGate();
            }
        }
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_GATE_HPP
