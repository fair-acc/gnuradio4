#ifndef GNURADIO_TRIGGER_GATE_HPP
#define GNURADIO_TRIGGER_GATE_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <limits>
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
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
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

    A<std::pmr::string, "mode", Doc<"once|toggle|cooldown|wait|take_until|skip_until">>              mode = std::pmr::string("once");
    A<std::pmr::string, "open filter", Doc<"trigger filter that opens the gate">>                    open_filter;
    A<std::pmr::string, "close filter", Doc<"trigger filter that closes it">>                        close_filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                                  match_mode;
    A<gr::Size_t, "n open", Doc<"samples forwarded per open run">>                                   n_open             = 1U;
    A<gr::Size_t, "n cooldown", Doc<"samples suppressed between open runs">>                         n_cooldown         = 0U;
    A<gr::Size_t, "n delay", Doc<"samples excluded after the trigger, 'wait' only">>                 n_delay            = 1U;
    A<std::pmr::string, "retrigger", Doc<"ignore|restart|extend">>                                   retrigger          = std::pmr::string("ignore");
    A<std::pmr::string, "closed policy", Doc<"drop|hold">>                                           closed_policy      = std::pmr::string("drop");
    A<bool, "initial state", Doc<"true = the gate starts open">>                                     initial_state      = false;
    mutable A<bool, "is open", Doc<"gate state; host path only">>                                    is_open            = false;
    mutable A<gr::Size_t, "n passed", Doc<"samples forwarded; host path only">>                      n_passed           = 0U;
    mutable A<gr::Size_t, "n suppressed", Doc<"samples consumed but not forwarded; host path only">> n_suppressed       = 0U;
    mutable A<gr::Size_t, "n triggers ignored", Doc<"host path only">>                               n_triggers_ignored = 0U;
    mutable A<gr::Size_t, "n tags dropped", Doc<"tags dropped with samples; host path only">>        n_tags_dropped     = 0U;
    mutable A<gr::Size_t, "n invalid triggers", Doc<"undated triggers; host path only">>             n_invalid_triggers = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    GateMode           _mode         = GateMode::once;
    Retrigger          _retrigger    = Retrigger::ignore;
    ClosedPolicy       _closedPolicy = ClosedPolicy::drop;
    MatchState         _open{};
    MatchState         _close{};
    bool               _hasClose = false;
    detail::InlineText _source{};
    detail::InlineText _pending_error{};
    gr::Size_t         _error_epoch    = 0U;
    gr::Size_t         _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Gate, evtIn, evtOut, in, out, control, mode, open_filter, close_filter, match_mode, n_open, n_cooldown, n_delay, retrigger, closed_policy, initial_state, is_open, n_passed, n_suppressed, n_triggers_ignored, n_tags_dropped, n_invalid_triggers, _mode, _retrigger, _closedPolicy, _open, _close, _hasClose, _source, _pending_error, _error_epoch, _settings_epoch);

    mutable MatchState    _opening{};
    mutable MatchState    _closing{};
    mutable bool          _gateOpen          = false;
    mutable std::uint64_t _samplesLeftOpen   = 0UZ;
    mutable std::uint64_t _samplesLeftClosed = 0UZ;
    mutable bool          _armed             = true;
    mutable bool          _reportedOpen      = false;
    mutable TimeBase      _time;
    mutable gr::Size_t    _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable gr::Size_t    _errorEpoch = 0U;
    mutable EventStore    _events;

    void start() {
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
        refuseHoldWithoutReopenPath();
    }

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
        _closedPolicy = closed_policy == "hold" ? ClosedPolicy::hold : ClosedPolicy::drop;
        if (lifecycle::isActive(this->state())) {
            refuseHoldWithoutReopenPath();
        }
        _events.clear();

        detail::compileFilterInto(open_filter.value, match_mode.value, _open, "Gate");
        _hasClose = detail::compileOptionalFilter(close_filter.value, match_mode.value, _close, "Gate");
        _source.assign(this->unique_name.value());
        _settings_epoch    = _settings_epoch + 1U;
        is_open            = initial_state || _mode == GateMode::take_until;
        n_passed           = 0U;
        n_suppressed       = 0U;
        n_triggers_ignored = 0U;
        n_tags_dropped     = 0U;
        n_invalid_triggers = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& controlSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _opening           = _open;
            _closing           = _close;
            _gateOpen          = initial_state || _mode == GateMode::take_until;
            _reportedOpen      = _gateOpen;
            _samplesLeftOpen   = 0UZ;
            _samplesLeftClosed = 0UZ;
            _armed             = true;
            _time.reset();
            _stateEpoch = _settings_epoch;
        }

        std::size_t nEmitted = 0UZ;
        const auto  report   = [&](auto writeInto) {
            const bool emitted = evtOutSpan.isConnected && gr::emit(evtOutSpan, nEmitted, detail::kEventKeys, writeInto).has_value();
            nEmitted += emitted ? 1UZ : 0UZ;
            return emitted;
        };
        const auto audit = [&](const gr::property_map_view& trigger) {
            if (!detail::isIncompleteTrigger(trigger)) {
                return;
            }
            n_invalid_triggers = n_invalid_triggers + 1U;
            if constexpr (kOnHost) {
                gr::log::warning("Gate: {}", detail::kIncompleteTrigger);
            }
            std::ignore = report([&](gr::property_map_view& event) { return detail::writeErrorEvent(event, detail::kIncompleteTrigger, _source.view()); });
        };
        const auto applyInjected = [&](bool audited) {
            applyControl(controlSpan);
            std::ignore = forEachEvent<kOnHost>(_events, evtSpan, kOnHost ? std::numeric_limits<std::size_t>::max() : evtOutSpan.size(), [&](const gr::property_map_view& event) {
                if (applyMatch(event) && audited) {
                    audit(event);
                }
                return true;
            });
        };

        const std::size_t                                                                 n          = std::min(inSpan.size(), outSpan.size());
        auto                                                                              tags       = inSpan.rawTags();
        const bool                                                                        endsStream = std::ranges::any_of(tags, [&](const auto& tag) {
            const auto ends = gr::property_map_view{tag.map}.template get_if<bool>(gr::tag::END_OF_STREAM.key());
            return tag.index >= inSpan.streamIndex && tag.index - inSpan.streamIndex < n && ends && *ends;
        });
        const bool                                                                        holds      = _closedPolicy == ClosedPolicy::hold && !endsStream;
        auto                                                                              next       = std::ranges::begin(tags);
        std::size_t                                                                       emitted    = 0UZ;
        std::size_t                                                                       consumed   = 0UZ;
        bool                                                                              applied    = false;
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> scratch{};

        for (std::size_t i = 0UZ; i < n; ++i) {
            const std::size_t           at        = inSpan.streamIndex + i;
            const auto                  first     = detail::tagsOn(next, std::ranges::end(tags), at);
            const gr::property_map_view condition = detail::mergedAt(first, next, scratch);
            if (first != next) {
                std::ignore = _time.adopt(condition, at);
                if (applyMatch(condition)) {
                    audit(condition);
                }
            }
            if (!applied) {
                applyInjected(true);
                applied = true;
            }
            if (_gateOpen) {
                for (auto tag = first; tag != next; ++tag) {
                    outSpan.publishTag(gr::property_map_view{(*tag).map}, emitted);
                }
                outSpan[emitted++] = inSpan[i];
                ++consumed;
                n_passed = n_passed + 1U;
                countDownOpenRun();
            } else {
                if (holds) {
                    break;
                }
                ++consumed;
                n_suppressed = n_suppressed + 1U;
                if (first != next) {
                    n_tags_dropped = n_tags_dropped + 1U;
                }
                countDownCooldown();
            }
        }
        if (!applied) {
            applyInjected(false);
        }

        if (_gateOpen != _reportedOpen && report([&](gr::property_map_view& event) {
                const auto stamp = _time.at(inSpan.streamIndex);
                return detail::writeEvent(event, _gateOpen ? "opened" : "closed", _source.view(), gr::property_map_view{}) && (!stamp || event.try_emplace(gr::tag::TRIGGER_TIME.key(), *stamp));
            })) {
            _reportedOpen = _gateOpen;
        }
        if (_errorEpoch != _error_epoch && report([&](gr::property_map_view& event) { return detail::writeErrorEvent(event, _pending_error.view(), _source.view()); })) {
            _errorEpoch = _error_epoch;
        }
        if constexpr (kOnHost) {
            is_open = _gateOpen;
        }
        outSpan.publish(emitted);
        evtOutSpan.publish(nEmitted);
        if (!controlSpan.consume(controlSpan.size()) || !inSpan.consume(consumed)) {
            return gr::work::Status::ERROR;
        }
        return consumed == 0UZ && n > 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
    }

private:
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
        const std::string message = std::format("'{}' = '{}' refused ({}); keeping '{}'", key, setting.value, reason, previous);
        gr::log::warning("Gate: {}", message);
        _pending_error.assign(message);
        _error_epoch = _error_epoch + 1U;
        setting      = previous;
    }

    void refuseHoldWithoutReopenPath() {
        if (_closedPolicy != ClosedPolicy::hold || evtIn.isConnected() || control.isConnected()) {
            return;
        }
        constexpr std::string_view kNoReopenPath = "'closed_policy' = 'hold' refused (neither 'evtIn' nor 'control' is connected to reopen the gate); keeping 'drop'";
        gr::log::warning("Gate: {}", kNoReopenPath);
        _pending_error.assign(kNoReopenPath);
        _error_epoch  = _error_epoch + 1U;
        closed_policy = std::pmr::string("drop");
        _closedPolicy = ClosedPolicy::drop;
    }

    void retriggerRun(std::uint64_t window) const noexcept {
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

    void openFor(std::uint64_t nSamples) const noexcept {
        _gateOpen          = true;
        _samplesLeftOpen   = nSamples;
        _samplesLeftClosed = 0UZ;
    }

    void closeGate() const noexcept {
        _gateOpen        = false;
        _samplesLeftOpen = 0UZ;
    }

    void countDownOpenRun() const noexcept {
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

    void countDownCooldown() const noexcept {
        if (_samplesLeftClosed > 0UZ && --_samplesLeftClosed == 0UZ) {
            openFor(_mode == GateMode::cooldown ? static_cast<std::uint64_t>(n_open) : 0UZ);
        }
    }

    [[nodiscard]] bool applyMatch(const gr::property_map_view& view) const {
        const bool closes = _hasClose && detail::matches(_closing, view);
        const bool opens  = !closes && detail::matches(_opening, view);

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
                closeGate();
            } else {
                openFor(0UZ);
            }
            break;
        case GateMode::cooldown:
            if (_samplesLeftClosed == 0UZ) {
                openFor(1UZ);
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
            openFor(0UZ); //
            break;
        }
        return true;
    }

    void applyControl(gr::InputSpanLike auto& controlSpan) const {
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
