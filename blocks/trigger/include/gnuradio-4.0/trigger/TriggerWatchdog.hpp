#ifndef GNURADIO_TRIGGER_TRIGGERWATCHDOG_HPP
#define GNURADIO_TRIGGER_TRIGGERWATCHDOG_HPP

#include <algorithm>
#include <array>
#include <limits>
#include <memory_resource>
#include <ranges>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

GR_REGISTER_BLOCK(gr::blocks::trigger::TriggerWatchdog, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

namespace gr::blocks::trigger {

template<typename T>
struct TriggerWatchdog : gr::Block<TriggerWatchdog<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief report when triggers stop arriving and when they resume, and optionally act on it

    in      ─T───T─────────────T─▶   (Resilience4j's TimeLimiter in all but name)
    evtOut  ───────────timeout──recovered─▶
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "filter", Doc<"trigger filter the watchdog waits for">>                                                                              filter;
    A<std::pmr::string, "arm filter", Doc<"trigger filter that opens a window, which ends at the first match or at its end, once; empty = always watching">> arm_filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                                                                                          match_mode;
    A<gr::Size_t, "timeout samples", Doc<"samples without a match, 0 disables">>                                                                             timeout_samples    = 0U;
    A<float, "timeout seconds", Doc<"seconds without a match, <= 0 disables">>                                                                               timeout_seconds    = 0.f;
    gr::SampleRate                                                                                                                                           sample_rate        = 0.f;
    A<std::pmr::string, "timeout action", Doc<"report|open|close">>                                                                                          timeout_action     = std::pmr::string("report");
    A<std::pmr::string, "open trigger name", Doc<"name of the event that opens a gate">>                                                                     open_trigger_name  = std::pmr::string("open");
    A<std::pmr::string, "close trigger name", Doc<"name of the event that closes a gate">>                                                                   close_trigger_name = std::pmr::string("close");
    mutable A<bool, "timed out", Doc<"outage state">>                                                                                                        timed_out          = false;
    mutable A<gr::Size_t, "n timeouts", Doc<"outages reported">>                                                                                             n_timeouts         = 0U;
    mutable A<gr::Size_t, "n invalid triggers", Doc<"undated triggers">>                                                                                     n_invalid_triggers = 0U;
    mutable A<gr::Size_t, "n inconclusive", Doc<"windows a gap in the data ended: dropped samples are no proof of absence">>                                 n_inconclusive     = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"reports a full event output refused">>                                                                    n_events_dropped   = 0U;

    enum class Action : std::uint8_t { report, open, close };

    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _matcher{};
    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _armMatcher{};
    bool                                                _anchored       = false;
    Action                                              _action         = Action::report;
    gr::Size_t                                          _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(TriggerWatchdog, evtOut, in, out, filter, arm_filter, match_mode, timeout_samples, timeout_seconds, sample_rate, timeout_action, open_trigger_name, close_trigger_name, timed_out, n_timeouts, n_invalid_triggers, n_inconclusive, n_events_dropped, _matcher, _armMatcher, _anchored, _action, _settings_epoch);

    mutable std::uint64_t      _sinceMatch = 0UZ;
    mutable TimeBase           _time;
    mutable detail::MatchState _matching{};
    mutable detail::MatchState _arming{};
    mutable bool               _watching        = true;
    mutable bool               _outage          = false;
    mutable bool               _timeoutReported = false;
    mutable bool               _unarmedReported = false;
    mutable gr::Size_t         _stateEpoch      = std::numeric_limits<gr::Size_t>::max();

    void start() { _stateEpoch = std::numeric_limits<gr::Size_t>::max(); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        detail::compileFilterInto(*this, filter.value, match_mode.value, _matcher);
        _anchored = !arm_filter.value.empty();
        if (_anchored) {
            detail::compileFilterInto(*this, arm_filter.value, match_mode.value, _armMatcher);
        }
        _action         = gr::meta::parseEnum<Action>(timeout_action.value).value_or(Action::report);
        _settings_epoch = _settings_epoch + 1U;
    }

    void reset() {
        _stateEpoch        = std::numeric_limits<gr::Size_t>::max();
        timed_out          = false;
        n_timeouts         = 0U;
        n_invalid_triggers = 0U;
        n_inconclusive     = 0U;
        n_events_dropped   = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool        kOnHost          = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        constexpr std::size_t kEventsPerSample = 3UZ;
        if (_stateEpoch != _settings_epoch) {
            _sinceMatch      = 0UZ;
            _matching        = _matcher;
            _arming          = _armMatcher;
            _watching        = !_anchored;
            _outage          = false;
            _timeoutReported = false;
            _unarmedReported = false;
            _time.reset();
            _time.setRate(static_cast<double>(sample_rate));
            _stateEpoch = _settings_epoch;
        }

        std::size_t nEmitted = 0UZ;
        const auto  report   = [&](auto writeInto) {
            const bool emitted = evtOutSpan.isConnected && gr::emit(evtOutSpan, nEmitted, detail::kEventKeys, writeInto).has_value();
            nEmitted += emitted ? 1UZ : 0UZ;
            if (evtOutSpan.isConnected && !emitted) {
                n_events_dropped = n_events_dropped + 1U;
            }
            return emitted;
        };
        const auto reportState = [&](std::string_view state, std::size_t at) {
            return report([&](gr::property_map_view& event) {
                const auto stamp = _time.at(at);
                return detail::writeEvent(event, state, gr::property_map_view{}) && (!stamp || event.try_emplace(gr::tag::TRIGGER_TIME.key(), *stamp));
            });
        };
        const auto act = [&](bool onTimeout, std::size_t at) {
            if (_action != Action::report) {
                std::ignore = reportState((_action == Action::close) == onTimeout ? std::string_view{close_trigger_name.value} : std::string_view{open_trigger_name.value}, at);
            }
        };

        const std::size_t                                                                 nSpan = std::min(inSpan.size(), outSpan.size());
        auto                                                                              tags  = inSpan.rawTags();
        auto                                                                              next  = std::ranges::begin(tags);
        std::size_t                                                                       n     = 0UZ;
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> scratch{};
        for (; n < nSpan; ++n) {
            if (evtOutSpan.isConnected && nEmitted > 0UZ && nEmitted + kEventsPerSample > evtOutSpan.size()) {
                break;
            }
            const std::size_t at    = inSpan.streamIndex + n;
            const auto        first = detail::tagsOn(next, std::ranges::end(tags), at);
            for (auto tag = first; tag != next; ++tag) {
                outSpan.publishTag(gr::property_map_view{(*tag).map}, n);
            }
            const gr::property_map_view condition = detail::mergedAt(first, next, scratch);
            if (first != next) {
                std::ignore = _time.adopt(condition, at);
                if (detail::isGap(condition) && _watching && !_outage) {
                    n_inconclusive = n_inconclusive + 1U; // a gap is no proof of absence
                    _sinceMatch    = 0UZ;
                    _watching      = !_anchored;
                }
                if (_anchored && detail::matches(_arming, condition)) {
                    _watching        = true;
                    _sinceMatch      = 0UZ;
                    _outage          = false;
                    _timeoutReported = false;
                }
            }
            if (first != next && detail::matches(_matching, condition)) {
                if (detail::isIncompleteTrigger(condition)) {
                    n_invalid_triggers = n_invalid_triggers + 1U;
                    if constexpr (kOnHost) {
                        detail::reportError(*this, "processBulk()", detail::kIncompleteTrigger);
                    }
                }
                if (_outage && _timeoutReported) {
                    std::ignore = reportState("recovered", at);
                    act(false, at);
                }
                _outage          = false;
                _timeoutReported = false;
                _sinceMatch      = 0UZ;
                _watching        = !_anchored; // an anchored window closes at its first match
                if constexpr (kOnHost) {
                    timed_out = false;
                }
                continue;
            }
            if (!_watching) {
                continue;
            }
            ++_sinceMatch;
            const auto limit = resolveDuration(_time, timeout_samples, timeout_seconds);
            if (!_outage && limit && _sinceMatch >= *limit) {
                _outage          = true;
                _watching        = !_anchored; // and at its end, once
                n_timeouts       = n_timeouts + 1U;
                _timeoutReported = reportState("timeout", at);
                act(true, at);
                if constexpr (kOnHost) {
                    timed_out = true;
                }
            }
        }

        if (!_unarmedReported && timeout_seconds > 0.f && !_time.measuresDuration()) {
            _unarmedReported = true;
            if constexpr (kOnHost) {
                detail::reportError(*this, "processBulk()", "timeout_seconds needs a sample_rate: no 'sample_rate' setting or tag gives a rate, so it stays unarmed");
            }
        }

        std::copy_n(inSpan.begin(), n, outSpan.begin());
        outSpan.publish(n);
        evtOutSpan.publish(nEmitted);
        if (!inSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TRIGGERWATCHDOG_HPP
