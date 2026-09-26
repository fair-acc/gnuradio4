#ifndef GNURADIO_TRIGGER_VALUETRIGGER_HPP
#define GNURADIO_TRIGGER_VALUETRIGGER_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <optional>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
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
once per excursion), `dropout` (no crossing for a duration). A `sample_rate` of 0 means not yet known.

 [1] specification: IEEE Std 181-2011, transition and pulse parameters
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::size_t kHistory        = 32UZ;
    constexpr static std::size_t kMaxSlewSamples = 64UZ;
    using Comparator                             = gr::trigger::SchmittTrigger<T, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION, kHistory>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "condition", Doc<"level|window|pulse_width|runt|slew|dropout (if AUTO)">>          condition_mode = std::pmr::string("level");
    A<T, "threshold", Doc<"the level a crossing is measured against">>                                     threshold{};
    A<T, "upper threshold", Doc<"the second level, for window, runt and slew">>                            threshold_upper{};
    A<T, "hysteresis", Doc<"widens each comparator against chatter">>                                      hysteresis{};
    A<std::pmr::string, "edge", Doc<"rising|falling|both">>                                                edge              = std::pmr::string("rising");
    A<gr::Size_t, "width min samples", Doc<"shortest qualifying interval, 0 = unset">>                     width_min_samples = 0U;
    A<gr::Size_t, "width max samples", Doc<"longest qualifying interval, 0 = unset">>                      width_max_samples = 0U;
    A<float, "width min seconds", Doc<"seconds, 0 = unset">>                                               width_min_seconds = 0.f;
    A<float, "width max seconds", Doc<"seconds, 0 = unset">>                                               width_max_seconds = 0.f;
    A<gr::Size_t, "slew samples", Doc<"samples the rate is measured over; 1 to 64">>                       slew_samples      = 1U;
    A<float, "slew min rate", Doc<"min rate, units/s">>                                                    slew_min_rate     = 0.f;
    A<float, "slew max rate", Doc<"max rate, units/s">>                                                    slew_max_rate     = 0.f;
    gr::SampleRate                                                                                         sample_rate       = 0.f;
    gr::TriggerName                                                                                        trigger_name      = std::pmr::string("edge");
    gr::Context                                                                                            context;
    mutable A<gr::Size_t, "n triggers", Doc<"triggers emitted; host path only">>                           n_triggers       = 0U;
    mutable A<gr::Size_t, "n rejected", Doc<"crossings the qualifier turned down; host path only">>        n_rejected       = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"triggers a full event output refused; host path only">> n_events_dropped = 0U;

    ValueCondition     _condition = condition == ValueCondition::AUTO ? ValueCondition::level : condition;
    EdgeSelection      _edge      = EdgeSelection::rising;
    detail::InlineText _source{};
    gr::Size_t         _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(ValueTrigger, evtOut, in, out, condition_mode, threshold, threshold_upper, hysteresis, edge, width_min_samples, width_max_samples, width_min_seconds, width_max_seconds, slew_samples, slew_min_rate, slew_max_rate, sample_rate, trigger_name, context, n_triggers, n_rejected, n_events_dropped, _condition, _edge, _source, _settings_epoch);

    mutable Comparator                                  _lower{1, 0};
    mutable Comparator                                  _upper{1, 0};
    mutable TimeBase                                    _time;
    mutable std::optional<std::size_t>                  _pulseStart;
    mutable bool                                        _reachedUpper  = false;
    mutable std::size_t                                 _sinceCrossing = 0UZ;
    mutable bool                                        _droppedOut    = false;
    mutable bool                                        _slewOutOfBand = false;
    mutable gr::HistoryBuffer<T, kMaxSlewSamples + 2UZ> _recent{};
    mutable bool                                        _inBand     = false;
    mutable gr::Size_t                                  _stateEpoch = std::numeric_limits<gr::Size_t>::max();

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if constexpr (condition == ValueCondition::AUTO) {
            _condition = parseCondition(condition_mode.value);
        }
        _edge = edge.value == "both" ? EdgeSelection::both : (edge.value == "falling" ? EdgeSelection::falling : EdgeSelection::rising);
        if (slew_samples > kMaxSlewSamples) {
            gr::log::warning("ValueTrigger: slew_samples {} exceeds the {} the block holds and is lowered to it", slew_samples.value, kMaxSlewSamples);
            slew_samples = static_cast<gr::Size_t>(kMaxSlewSamples);
        }
        _source.assign(this->unique_name.value());
        _settings_epoch  = _settings_epoch + 1U;
        n_triggers       = 0U;
        n_rejected       = 0U;
        n_events_dropped = 0U;
    }

    void reset() {
        _stateEpoch      = std::numeric_limits<gr::Size_t>::max();
        n_triggers       = 0U;
        n_rejected       = 0U;
        n_events_dropped = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) const {
        if (_stateEpoch != _settings_epoch) {
            restart();
        }
        const std::size_t nSpan    = std::min(inSpan.size(), outSpan.size());
        auto              tags     = inSpan.rawTags();
        auto              next     = std::ranges::begin(tags);
        std::size_t       nEmitted = 0UZ;
        std::size_t       n        = 0UZ;
        std::size_t       nTagsOut = 0UZ;
        for (; n < nSpan; ++n) {
            if (evtOutSpan.isConnected && nEmitted > 0UZ && nEmitted == evtOutSpan.size()) {
                break;
            }
            const std::size_t at    = inSpan.streamIndex + n;
            const auto        first = detail::tagsOn(next, std::ranges::end(tags), at);
            if (gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>> && nTagsOut + static_cast<std::size_t>(std::ranges::distance(first, next)) + 1UZ > gr::device::kDeviceTagSlots) {
                break;
            }
            for (auto tag = first; tag != next; ++tag, ++nTagsOut) {
                const gr::property_map_view tagMap{(*tag).map};
                std::ignore = _time.adopt(tagMap, at);
                outSpan.publishTag(tagMap, n);
            }
            if (qualifies(inSpan[n], at)) {
                ++nTagsOut;
                n_triggers = n_triggers + 1U;
                alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> blob{};
                gr::property_map_view                                                             found = gr::formatEvent(blob, kDatedTriggerKeys + 1U);
                std::ignore                                                                             = writeDatedTrigger(found, _time, at, _lower.lastEdgeOffset, trigger_name.value, context.value);
                outSpan.publishTag(found, n);
                if (evtOutSpan.isConnected) {
                    if (found.try_emplace(std::string_view{"source"}, _source.view()) && gr::emit(evtOutSpan, nEmitted, found)) {
                        ++nEmitted;
                    } else {
                        n_events_dropped = n_events_dropped + 1U;
                    }
                }
            }
            outSpan[n] = inSpan[n];
        }

        outSpan.publish(n);
        evtOutSpan.publish(nEmitted);
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

    void restart() const {
        const T guard  = hysteresis.value == T{} ? T(1) : hysteresis.value;
        _lower         = Comparator{guard, threshold.value};
        _upper         = Comparator{guard, threshold_upper.value};
        _pulseStart    = std::nullopt;
        _reachedUpper  = false;
        _sinceCrossing = 0UZ;
        _droppedOut    = false;
        _slewOutOfBand = false;
        _recent.reset();
        _inBand = false;
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
        _stateEpoch = _settings_epoch;
    }

    [[nodiscard]] bool wantsEdge(gr::trigger::EdgeDetection detected) const noexcept {
        using enum gr::trigger::EdgeDetection;
        if (_edge == EdgeSelection::both) {
            return detected != NONE;
        }
        return detected == (_edge == EdgeSelection::falling ? FALLING : RISING);
    }

    [[nodiscard]] bool qualifies(T sample, std::size_t at) const {
        using enum gr::trigger::EdgeDetection;
        const auto lowerEdge = _lower.processOne(sample);
        const auto upperEdge = _upper.processOne(sample);
        ++_sinceCrossing;

        switch (_condition) {
        case ValueCondition::level: return wantsEdge(lowerEdge);

        case ValueCondition::window: {
            const bool above   = sample > threshold_upper;
            const bool below   = sample < threshold;
            const bool band    = !above && !below;
            const bool entered = band && !_inBand;
            const bool left    = !band && _inBand;
            _inBand            = band;
            if ((entered && _edge != EdgeSelection::falling) || (left && _edge != EdgeSelection::rising)) {
                return true;
            }
            return false;
        }

        case ValueCondition::pulse_width:
            if (lowerEdge == RISING) {
                _pulseStart = at;
            } else if (lowerEdge == FALLING && _pulseStart) {
                const std::size_t width = (at) - *_pulseStart;
                _pulseStart             = std::nullopt;
                if (withinDuration(_time, width, width_min_samples, width_min_seconds, width_max_samples, width_max_seconds)) {
                    return true;
                }
                n_rejected = n_rejected + 1U;
            }
            return false;

        case ValueCondition::runt:
            if (lowerEdge == RISING) {
                _pulseStart   = at;
                _reachedUpper = false;
            }
            if (upperEdge == RISING) {
                _reachedUpper = true;
            }
            if (lowerEdge == FALLING && _pulseStart) {
                const bool runt = !_reachedUpper;
                _pulseStart     = std::nullopt;
                if (runt) {
                    return true;
                }
                n_rejected = n_rejected + 1U;
            }
            return false;

        case ValueCondition::slew:
            if (lowerEdge == RISING) {
                _pulseStart = at;
            }
            if (upperEdge == RISING && _pulseStart) {
                const std::size_t rise = (at) - *_pulseStart;
                _pulseStart            = std::nullopt;
                if (!withinDuration(_time, rise, width_min_samples, width_min_seconds, width_max_samples, width_max_seconds)) {
                    return true;
                }
                n_rejected = n_rejected + 1U;
            }
            return false;

        case ValueCondition::slew_rate: {
            const std::size_t span = slew_samples == 0U ? 1UZ : static_cast<std::size_t>(slew_samples.value);
            _recent.push_front(sample);
            if (_recent.size() <= span) {
                return false;
            }
            const double rate      = _time.rate() > 0. ? _time.rate() : 1.;
            const double perSecond = static_cast<double>(sample - _recent[span]) * rate / static_cast<double>(span);
            const bool   tooSlow   = slew_min_rate > 0.f && std::abs(perSecond) < static_cast<double>(slew_min_rate);
            const bool   tooFast   = slew_max_rate > 0.f && std::abs(perSecond) > static_cast<double>(slew_max_rate);
            if (tooSlow || tooFast) {
                if (_slewOutOfBand) {
                    return false;
                }
                _slewOutOfBand = true;
                return true;
            }
            if (_slewOutOfBand) {
                _slewOutOfBand = false;
                n_rejected     = n_rejected + 1U;
            }
            return false;
        }

        case ValueCondition::dropout: {
            if (wantsEdge(lowerEdge)) {
                _sinceCrossing = 0UZ;
                _droppedOut    = false;
                return false;
            }
            const auto high = resolveDuration(_time, width_max_samples, width_max_seconds);
            if (!_droppedOut && high && _sinceCrossing >= *high) {
                _droppedOut = true;
                return true;
            }
            return false;
        }

        case ValueCondition::AUTO: return false;
        }
        return false;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_VALUETRIGGER_HPP
