#ifndef GNURADIO_TRIGGER_VALUETRIGGER_HPP
#define GNURADIO_TRIGGER_VALUETRIGGER_HPP

#include <cmath>
#include <cstdint>
#include <memory_resource>
#include <optional>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/ConditionSource.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

namespace gr::blocks::trigger {

enum class ValueCondition : std::uint8_t { AUTO, level, window, pulse_width, runt, slew, slew_rate, dropout };

GR_REGISTER_BLOCK("gr::blocks::trigger::LevelTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::level), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::WindowTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::window), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::PulseWidthTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::pulse_width), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::RuntTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::runt), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SlewTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::slew), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SlewRateTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::slew_rate), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::DropoutTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::dropout), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::ValueTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::AUTO), [ int16_t, int32_t, float, double ])

template<typename T, ValueCondition condition = ValueCondition::AUTO>
requires(std::is_arithmetic_v<T>)
struct ValueTrigger : gr::Block<ValueTrigger<T, condition>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief report a level, window, pulse-width, runt, slew, slew-rate or dropout condition

    in      ──╱‾╲_╱‾‾‾╲__▏╲_▏╱▔▔▔▔   (no RxMarbles equivalent)
    evtOut  ───T───────────T──────▶  condition_mode picks which question is asked

`level` (an edge), `window` (leaving a band), `pulse_width` (a pulse inside or outside a width band), `runt` (a pulse that never
reached the upper threshold), `slew` (the interval between two thresholds), `slew_rate` (volts per second over `slew_samples`,
once per excursion), `dropout` (no crossing for a duration).

 [1] specification: IEEE Std 181-2011, transition and pulse parameters
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::size_t kHistory = 32UZ;
    using Comparator                      = gr::trigger::SchmittTrigger<T, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION, kHistory>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "condition", Doc<"level|window|pulse_width|runt|slew|dropout (if AUTO)">> condition_mode = std::pmr::string("level");
    A<T, "threshold", Doc<"the level a crossing is measured against">>                            threshold{};
    A<T, "upper threshold", Doc<"the second level, for window, runt and slew">>                   threshold_upper{};
    A<T, "hysteresis", Doc<"widens each comparator against chatter">>                             hysteresis{};
    A<std::pmr::string, "edge", Doc<"rising|falling|both">>                                       edge              = std::pmr::string("rising");
    A<gr::Size_t, "width min samples", Doc<"shortest qualifying interval, 0 = unset">>            width_min_samples = 0U;
    A<gr::Size_t, "width max samples", Doc<"longest qualifying interval, 0 = unset">>             width_max_samples = 0U;
    A<float, "width min seconds", Doc<"seconds, 0 = unset">>                                      width_min_seconds = 0.f;
    A<float, "width max seconds", Doc<"seconds, 0 = unset">>                                      width_max_seconds = 0.f;
    A<gr::Size_t, "slew samples", Doc<"samples the rate is measured over; at least 1">>           slew_samples      = 1U;
    A<float, "slew min rate", Doc<"min rate, units/s">>                                           slew_min_rate     = 0.f;
    A<float, "slew max rate", Doc<"max rate, units/s">>                                           slew_max_rate     = 0.f;
    A<float, "sample rate", Doc<"Hz, seconds limits + dating; 0 = unknown">>                      sample_rate       = 0.f;
    A<std::pmr::string, "trigger name", Doc<"the name the emitted trigger carries">>              trigger_name      = std::pmr::string("edge");
    A<std::pmr::string, "context", Doc<"context the emitted trigger carries">>                    context;
    A<gr::Size_t, "n triggers", Doc<"triggers emitted">>                                          n_triggers       = 0U;
    A<gr::Size_t, "n rejected", Doc<"crossings the qualifier turned down">>                       n_rejected       = 0U;
    A<gr::Size_t, "n events dropped", Doc<"triggers the event output had no room for">>           n_events_dropped = 0U;

    GR_MAKE_REFLECTABLE(ValueTrigger, evtOut, in, out, condition_mode, threshold, threshold_upper, hysteresis, edge, width_min_samples, width_max_samples, width_min_seconds, width_max_seconds, slew_samples, slew_min_rate, slew_max_rate, sample_rate, trigger_name, context, n_triggers, n_rejected, n_events_dropped);

    ValueCondition        _condition = condition == ValueCondition::AUTO ? ValueCondition::level : condition;
    Comparator            _lower{1, 0};
    Comparator            _upper{1, 0};
    TimeBase              _time;
    std::size_t           _streamIndex = 0UZ;
    detail::PendingEvents _pending;

    std::optional<std::size_t> _pulseStart; // sample the current pulse began on
    bool                       _reachedUpper  = false;
    std::size_t                _sinceCrossing = 0UZ;
    bool                       _droppedOut    = false;
    bool                       _slewOutOfBand = false;
    gr::HistoryBuffer<T>       _recent{2UZ};
    bool                       _inBand = false;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if constexpr (condition == ValueCondition::AUTO) {
            _condition = parseCondition(condition_mode.value);
        }
        _time.setRate(static_cast<double>(sample_rate));
        const T guard = hysteresis.value == T{} ? T(1) : hysteresis.value;
        _lower        = Comparator{guard, threshold.value};
        _upper        = Comparator{guard, threshold_upper.value};
        reset();
    }

    void reset() {
        _lower.reset();
        _upper.reset();
        _streamIndex   = 0UZ;
        _pulseStart    = std::nullopt;
        _slewOutOfBand = false;
        _recent        = gr::HistoryBuffer<T>(static_cast<std::size_t>(slew_samples == 0U ? 1U : slew_samples.value) + 2UZ);
        _reachedUpper  = false;
        _sinceCrossing = 0UZ;
        _droppedOut    = false;
        _inBand        = false;
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
        _pending.clear();
        n_triggers       = 0U;
        n_rejected       = 0U;
        n_events_dropped = 0U;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t n          = std::min(inSpan.size(), outSpan.size());
        const auto        conditions = detail::collectConditions(inSpan, n);
        std::size_t       published  = 0UZ;

        for (std::size_t i = 0UZ; i < n; ++i) {
            if (const gr::property_map* tag = conditions.tagAt(i); tag != nullptr) {
                _time.adopt(gr::property_map_view{*tag}, _streamIndex + i);
                outSpan.publishTag(*tag, i);
            }
            if (const auto found = qualify(inSpan[i], i); found) {
                enqueue(outSpan, *found, i);
            }
            outSpan[i] = inSpan[i];
        }
        published += _pending.drainInto(evtOutSpan, published, this->name, this->unique_name);
        n_events_dropped = static_cast<gr::Size_t>(_pending.dropped);

        _streamIndex += n;
        outSpan.publish(n);
        evtOutSpan.publish(published);
        if (!inSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] static ValueCondition parseCondition(std::string_view text) noexcept {
        const auto named = gr::meta::parseEnum<ValueCondition>(text);
        return named && *named != ValueCondition::AUTO ? *named : ValueCondition::level;
    }

    [[nodiscard]] bool wantsEdge(gr::trigger::EdgeDetection detected) const noexcept {
        using enum gr::trigger::EdgeDetection;
        if (edge == "both") {
            return detected != NONE;
        }
        return detected == (edge == "falling" ? FALLING : RISING);
    }

    [[nodiscard]] std::optional<std::size_t> qualify(T sample, std::size_t offset) {
        using enum gr::trigger::EdgeDetection;
        const auto lowerEdge = _lower.processOne(sample);
        const auto upperEdge = _upper.processOne(sample);
        ++_sinceCrossing;

        switch (_condition) {
        case ValueCondition::level: return wantsEdge(lowerEdge) ? std::optional{offset} : std::nullopt;

        case ValueCondition::window: {
            const bool above   = sample > threshold_upper;
            const bool below   = sample < threshold;
            const bool band    = !above && !below;
            const bool entered = band && !_inBand;
            const bool left    = !band && _inBand;
            _inBand            = band;
            if ((entered && edge != "falling") || (left && edge != "rising")) {
                return offset;
            }
            return std::nullopt;
        }

        case ValueCondition::pulse_width:
            if (lowerEdge == RISING) {
                _pulseStart = _streamIndex + offset;
            } else if (lowerEdge == FALLING && _pulseStart) {
                const std::size_t width = (_streamIndex + offset) - *_pulseStart;
                _pulseStart             = std::nullopt;
                if (withinDuration(_time, width, width_min_samples, width_min_seconds, width_max_samples, width_max_seconds)) {
                    return offset;
                }
                ++n_rejected;
            }
            return std::nullopt;

        case ValueCondition::runt:
            if (lowerEdge == RISING) {
                _pulseStart   = _streamIndex + offset;
                _reachedUpper = false;
            }
            if (upperEdge == RISING) {
                _reachedUpper = true;
            }
            if (lowerEdge == FALLING && _pulseStart) {
                const bool runt = !_reachedUpper;
                _pulseStart     = std::nullopt;
                if (runt) {
                    return offset;
                }
                ++n_rejected;
            }
            return std::nullopt;

        case ValueCondition::slew:
            if (lowerEdge == RISING) {
                _pulseStart = _streamIndex + offset;
            }
            if (upperEdge == RISING && _pulseStart) {
                const std::size_t rise = (_streamIndex + offset) - *_pulseStart;
                _pulseStart            = std::nullopt;
                if (!withinDuration(_time, rise, width_min_samples, width_min_seconds, width_max_samples, width_max_seconds)) {
                    return offset;
                }
                ++n_rejected;
            }
            return std::nullopt;

        case ValueCondition::slew_rate: {
            const std::size_t span = slew_samples == 0U ? 1UZ : static_cast<std::size_t>(slew_samples.value);
            _recent.push_front(sample);
            if (_recent.size() <= span) {
                return std::nullopt;
            }
            const double rate      = _time.rate() > 0. ? _time.rate() : 1.;
            const double perSecond = static_cast<double>(sample - _recent[span]) * rate / static_cast<double>(span);
            const bool   tooSlow   = slew_min_rate > 0.f && std::abs(perSecond) < static_cast<double>(slew_min_rate);
            const bool   tooFast   = slew_max_rate > 0.f && std::abs(perSecond) > static_cast<double>(slew_max_rate);
            if (tooSlow || tooFast) {
                if (_slewOutOfBand) {
                    return std::nullopt;
                }
                _slewOutOfBand = true;
                return offset;
            }
            if (_slewOutOfBand) {
                _slewOutOfBand = false;
                ++n_rejected;
            }
            return std::nullopt;
        }

        case ValueCondition::dropout: {
            if (wantsEdge(lowerEdge)) {
                _sinceCrossing = 0UZ;
                _droppedOut    = false;
                return std::nullopt;
            }
            const auto high = resolveDuration(_time, width_max_samples, width_max_seconds);
            if (!_droppedOut && high && _sinceCrossing >= *high) {
                _droppedOut = true;
                return offset;
            }
            return std::nullopt;
        }

        case ValueCondition::AUTO: return std::nullopt;
        }
        return std::nullopt;
    }

    void enqueue(auto& outSpan, std::size_t offset, std::size_t tagOffset) {
        ++n_triggers;
        gr::property_map found = datedTrigger(_time, _streamIndex + offset, _lower.lastEdgeOffset, trigger_name.value, context.value);
        outSpan.publishTag(found, tagOffset);
        found[std::string("source")] = std::pmr::string(this->unique_name.value());
        _pending.push(std::move(found));
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_VALUETRIGGER_HPP
