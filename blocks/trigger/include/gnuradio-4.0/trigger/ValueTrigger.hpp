#ifndef GNURADIO_TRIGGER_VALUETRIGGER_HPP
#define GNURADIO_TRIGGER_VALUETRIGGER_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <memory_resource>
#include <optional>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/SampleTest.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

namespace gr::blocks::trigger {

enum class ValueCondition : std::uint8_t { automatic, level, window, pulseWidth, runt, transition, slewRate, levelTimeout, edgeTimeout, interval };
enum class DurationUnit : std::uint8_t { samples, seconds };

GR_REGISTER_BLOCK("gr::blocks::trigger::LevelTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::level), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::WindowTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::window), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::PulseWidthTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::pulseWidth), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::RuntTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::runt), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::TransitionTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::transition), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SlewRateTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::slewRate), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::LevelTimeoutTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::levelTimeout), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::EdgeTimeoutTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::edgeTimeout), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::IntervalTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::interval), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::ValueTrigger", gr::blocks::trigger::ValueTrigger, ([T], gr::blocks::trigger::ValueCondition::automatic), [ int16_t, int32_t, float, double ])

template<typename T, ValueCondition condition = ValueCondition::automatic>
requires(std::is_arithmetic_v<T> && std::is_signed_v<T>)
struct ValueTrigger : gr::Block<ValueTrigger<T, condition>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief report an edge, a window crossing, a pulse, a runt, a transition, a slew rate, a timeout or an interval

    in      ──╱‾╲_╱‾‾‾╲__▏╲_▏╱▔▔▔▔   (no RxMarbles equivalent)
    evtOut  ───T───────────T──────▶  condition_mode picks which question is asked

A transition that turns back is reported with class `reversed`, a level timeout still held with class `ongoing`.
`trigger_meta_info` also holds, while the event slot has room, the lower comparator's boundaries and the estimator.

 [1] specification: IEEE Std 181-2011, transition and pulse parameters
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::size_t   kHistory        = 32UZ;
    constexpr static std::size_t   kMaxSlewSamples = 64UZ;
    constexpr static std::uint32_t kEventKeys      = kDatedTriggerKeys + 1U; /// + trigger_meta_info
    using Comparator                               = gr::trigger::SchmittTrigger<T, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION, kHistory>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "condition", Doc<"level: edge; window: enter or leave [threshold, threshold_upper]; pulseWidth: completed pulse; runt: pulse back across its first threshold; transition: threshold to threshold_upper; slewRate: units/s over slew_samples; levelTimeout: level held for a duration; edgeTimeout: no edge for a duration; interval: between same-polarity edges (if automatic)">> condition_mode = std::pmr::string("level");
    A<T, "threshold", Doc<"the level a crossing is measured against; the lower one of a window, runt or transition">>                                                                                                                                                                                                                                                                                      threshold{};
    A<T, "upper threshold", Doc<"the second level, for window, runt and transition">>                                                                                                                                                                                                                                                                                                                      threshold_upper{};
    A<T, "hysteresis", Doc<"full width of each comparator's band, centred on its level; >= 0; zero: high is x >= level">>                                                                                                                                                                                                                                                                                  hysteresis{};
    A<std::pmr::string, "edge", Doc<"rising|falling|both: the edge, the pulse polarity, the level a timeout watches">>                                                                                                                                                                                                                                                                                     edge = std::pmr::string("rising");
    A<std::pmr::string, "duration condition", Doc<"less|lessEqual|greater|greaterEqual|inside|outside; empty: no duration test">>                                                                                                                                                                                                                                                                          duration_condition;
    A<double, "duration min", Doc<"read by greater, greaterEqual, inside and outside">>                                                                                                                                                                                                                                                                                                                    duration_min  = 0.;
    A<double, "duration max", Doc<"read by less, lessEqual, inside and outside">>                                                                                                                                                                                                                                                                                                                          duration_max  = 0.;
    A<std::pmr::string, "duration unit", Doc<"samples|seconds">>                                                                                                                                                                                                                                                                                                                                           duration_unit = std::pmr::string("samples");
    A<gr::Size_t, "slew samples", Doc<"samples the rate is measured over; 1 to 64">>                                                                                                                                                                                                                                                                                                                       slew_samples  = 1U;
    A<float, "slew min rate", Doc<"min rate, units/s">>                                                                                                                                                                                                                                                                                                                                                    slew_min_rate = 0.f;
    A<float, "slew max rate", Doc<"max rate, units/s">>                                                                                                                                                                                                                                                                                                                                                    slew_max_rate = 0.f;
    A<T, "clip level", Doc<"|sample| at or beyond it is clipped, so unknown for amplitude and width; the span is reported at once, flagged clipped in acquisition_flags, without a duration test; 0: off">>                                                                                                                                                                                                clip_level{};
    gr::SampleRate                                                                                                                                                                                                                                                                                                                                                                                         sample_rate = 0.f;
    gr::TriggerName                                                                                                                                                                                                                                                                                                                                                                                        trigger_name;
    gr::Context                                                                                                                                                                                                                                                                                                                                                                                            context;
    mutable A<gr::Size_t, "n triggers", Doc<"triggers emitted">>                                                                                                                                                                                                                                                                                                                                           n_triggers       = 0U;
    mutable A<gr::Size_t, "n rejected", Doc<"completed measurements the duration test turned down">>                                                                                                                                                                                                                                                                                                       n_rejected       = 0U;
    mutable A<gr::Size_t, "n unmeasurable", Doc<"decisions that needed a sample_rate the block does not have: a duration in seconds, a slew rate">>                                                                                                                                                                                                                                                        n_unmeasurable   = 0U;
    mutable A<gr::Size_t, "n gaps", Doc<"gap tags seen; each discards what was partly measured">>                                                                                                                                                                                                                                                                                                          n_gaps           = 0U;
    mutable A<gr::Size_t, "n abandoned", Doc<"pulses, runts and transitions cut short by a reversal or by unknown data">>                                                                                                                                                                                                                                                                                  n_abandoned      = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"triggers a full event output refused">>                                                                                                                                                                                                                                                                                                                 n_events_dropped = 0U;

    ValueCondition      _condition      = condition == ValueCondition::automatic ? ValueCondition::level : condition;
    EdgeSelection       _edge           = EdgeSelection::rising;
    bool                _timed          = false;
    Comparison          _durationTest   = Comparison::greaterEqual;
    double              _durationMin    = 0.;
    double              _durationMax    = 0.;
    DurationUnit        _durationUnit   = DurationUnit::samples;
    bool                _inverted       = false; /// threshold_upper below threshold: refused
    gr::Size_t          _settings_epoch = 0U;
    detail::ErrorMemory _deferredError{};

    GR_MAKE_REFLECTABLE(ValueTrigger, evtOut, in, out, condition_mode, threshold, threshold_upper, hysteresis, edge, duration_condition, duration_min, duration_max, duration_unit, slew_samples, slew_min_rate, slew_max_rate, clip_level, sample_rate, trigger_name, context, n_triggers, n_rejected, n_unmeasurable, n_gaps, n_abandoned, n_events_dropped, _condition, _edge, _timed, _durationTest, _durationMin, _durationMax, _durationUnit, _inverted, _settings_epoch);

    /** what one sample decided, in absolute fractional sample positions: the decision and, for a delayed one, the moment it is about */
    struct Found {
        double                moment{};
        std::optional<double> reference;
        bool                  referenceRising = false;
        float                 error{}; /// samples, 1σ
        bool                  clipped = false;
        std::string_view      label{}; /// `class` in trigger_meta_info: reversed, ongoing
    };

    mutable Comparator                                  _lower{};
    mutable Comparator                                  _upper{};
    mutable Comparator                                  _upperFromAbove{}; /// on -x: the window's closed upper bound
    mutable TimeBase                                    _time;
    mutable std::optional<double>                       _start;               /// a pulse's, runt's, transition's or episode's first crossing
    mutable bool                                        _startRising = false; /// the polarity of an edge timeout's `_start`
    mutable bool                                        _touched     = false; /// a runt reached its second threshold
    mutable bool                                        _reported    = false; /// a timeout's episode was reported
    mutable std::optional<bool>                         _inside;
    mutable bool                                        _slewOutOfBand        = false;
    mutable bool                                        _spanClipped          = false; /// a clipped sample lies in what is being measured
    mutable bool                                        _sampleClipped        = false; /// the sample being decided
    mutable bool                                        _previousClipped      = false; /// the one before it, which a crossing interpolates from
    mutable bool                                        _unmeasurableReported = false;
    mutable gr::HistoryBuffer<T, kMaxSlewSamples + 2UZ> _recent{};
    mutable gr::Size_t                                  _stateEpoch = std::numeric_limits<gr::Size_t>::max();

    void start() {
        detail::sendDeferredError(*this);
        detail::nameTriggerAfterBlock(*this);
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if constexpr (condition == ValueCondition::automatic) {
            if (const auto named = gr::meta::parseEnum<ValueCondition>(condition_mode.value); named && *named != ValueCondition::automatic) {
                _condition = *named;
            } else {
                refuse(std::format("condition_mode '{}' is not a condition; keeping '{}'", condition_mode.value, gr::meta::enumName(_condition).value_or("")));
                condition_mode = std::pmr::string(gr::meta::enumName(_condition).value_or(""));
            }
        }
        if (const auto named = gr::meta::parseEnum<EdgeSelection>(edge.value)) {
            _edge = singleSecondThreshold(*named, _condition);
        } else {
            refuse(std::format("edge '{}' is not rising, falling or both; keeping '{}'", edge.value, gr::meta::enumName(_edge).value_or("")));
            edge = std::pmr::string(gr::meta::enumName(_edge).value_or(""));
        }
        if (clip_level.value < T{}) {
            refuse(std::format("clip_level {} refused (a magnitude cannot be negative); keeping 0", clip_level.value));
            clip_level = T{};
        }
        if (slew_samples > kMaxSlewSamples) {
            const std::string message = std::format("slew_samples {} exceeds the {} the block holds and is lowered to it", slew_samples.value, kMaxSlewSamples);
            detail::reportError(*this, "settingsChanged()", message);
            slew_samples = static_cast<gr::Size_t>(kMaxSlewSamples);
        }
        acceptDurationTest();
        if (hysteresis.value < T{}) {
            const std::string message = std::format("'hysteresis' = {} refused (a band cannot be narrower than nothing); keeping 0", hysteresis.value);
            detail::reportError(*this, "settingsChanged()", message);
            hysteresis = T{};
        }
        const bool twoLevels = _condition == ValueCondition::window || _condition == ValueCondition::runt || _condition == ValueCondition::transition;
        _inverted            = twoLevels && threshold_upper.value < threshold.value;
        if (_inverted) {
            const std::string message = std::format("threshold_upper {} below threshold {} refused (an inverted range); nothing is reported until it is set right", threshold_upper.value, threshold.value);
            detail::reportError(*this, "settingsChanged()", message);
        }
        // a renamed trigger or a new context, which upstream tags may carry, leaves what is being measured alone
        constexpr std::array kMeasuring{"condition_mode", "threshold", "threshold_upper", "hysteresis", "edge", "duration_condition", "duration_min", "duration_max", "duration_unit", "slew_samples", "slew_min_rate", "slew_max_rate", "clip_level"};
        if (std::ranges::any_of(kMeasuring, [&](const char* key) { return newSettings.contains(key); })) {
            _settings_epoch = _settings_epoch + 1U;
        }
    }

    void reset() {
        _stateEpoch      = std::numeric_limits<gr::Size_t>::max();
        n_triggers       = 0U;
        n_rejected       = 0U;
        n_unmeasurable   = 0U;
        n_gaps           = 0U;
        n_abandoned      = 0U;
        n_events_dropped = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) const {
        if (_stateEpoch != _settings_epoch) {
            restart();
        }
        _time.setRate(static_cast<double>(sample_rate)); // a new rate rescales durations; it is no reason to forget the anchor
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
                if (detail::isGap(tagMap)) {
                    n_gaps = n_gaps + 1U;
                    forgetState(); // a gap: nothing measured across it is a measurement
                }
                outSpan.publishTag(tagMap, n);
            }
            if (const auto found = decide(inSpan[n], at)) {
                ++nTagsOut;
                n_triggers = n_triggers + 1U;
                alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> blob{};
                gr::property_map_view                                                             tag = gr::formatEvent(blob, kEventKeys);
                std::ignore                                                                           = writeEvent(tag, *found, at);
                outSpan.publishTag(tag, n);
                if (evtOutSpan.isConnected) {
                    gr::property_map_view event = gr::formatEvent(blob, kEventKeys); // the tag is published, its blob free again
                    if (writeEvent(event, *found, at) && gr::emit(evtOutSpan, nEmitted, event)) {
                        ++nEmitted;
                    } else {
                        n_events_dropped = n_events_dropped + 1U;
                    }
                }
            }
            outSpan[n] = inSpan[n];
        }

        if constexpr (!gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>) {
            if (n_unmeasurable > 0U && !_unmeasurableReported) {
                _unmeasurableReported = true;
                detail::reportError(*this, "processBulk()", "a duration in seconds and a slew rate need a sample_rate, which is 0: those decisions are counted in n_unmeasurable and not made");
            }
        }
        outSpan.publish(n);
        evtOutSpan.publish(nEmitted);
        if (!inSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    /** -x, saturated at the type's lowest value, which has no negative */
    /// the condition measured: fixed at compile time for the nine named ids, so their kernel keeps only its own branch
    [[nodiscard]] constexpr ValueCondition measuredCondition() const noexcept {
        if constexpr (condition == ValueCondition::automatic) {
            return _condition;
        } else {
            return condition;
        }
    }

    [[nodiscard]] static constexpr T negated(T x) noexcept {
        if constexpr (std::is_integral_v<T>) {
            return x == std::numeric_limits<T>::lowest() ? std::numeric_limits<T>::max() : static_cast<T>(-x);
        } else {
            return -x;
        }
    }

    [[nodiscard]] EdgeSelection singleSecondThreshold(EdgeSelection named, ValueCondition asked) {
        if (named == EdgeSelection::both && (asked == ValueCondition::runt || asked == ValueCondition::transition)) {
            refuse("edge 'both' for a runt or a transition has no single second threshold; taking rising");
            return EdgeSelection::rising;
        }
        return named;
    }

    /** the setting itself is put back by the caller */
    void refuse(const std::string& message) { detail::reportError(*this, "settingsChanged()", message); }

    /** one test per duration; a refused one keeps the test accepted before it */
    void acceptDurationTest() {
        const auto       named   = gr::meta::parseEnum<Comparison>(duration_condition.value);
        const bool       none    = duration_condition.value.empty();
        const bool       known   = none || (named && *named != Comparison::equal && *named != Comparison::notEqual);
        const bool       range   = named && (*named == Comparison::inside || *named == Comparison::outside);
        const bool       timeout = _condition == ValueCondition::levelTimeout || _condition == ValueCondition::edgeTimeout;
        const bool       onset   = named && (*named == Comparison::greater || *named == Comparison::greaterEqual);
        std::string_view refused;
        if (!known) {
            refused = "a duration is compared by less, lessEqual, greater, greaterEqual, inside or outside";
        } else if (range && duration_min.value > duration_max.value) {
            refused = "an inverted range";
        } else if (timeout && !none && !onset) {
            refused = "a timeout is decided once its duration is reached: greater or greaterEqual";
        }
        if (!refused.empty()) {
            refuse(std::format("duration_condition '{}' over [{}, {}] refused ({}); keeping the previous test", duration_condition.value, duration_min.value, duration_max.value, refused));
            duration_condition = _timed ? std::pmr::string(gr::meta::enumName(_durationTest).value_or("")) : std::pmr::string();
            duration_min       = _durationMin;
            duration_max       = _durationMax;
        } else {
            _timed        = !none;
            _durationTest = named.value_or(Comparison::greaterEqual);
            _durationMin  = duration_min.value;
            _durationMax  = duration_max.value;
        }
        _durationUnit = duration_unit.value == "seconds" ? DurationUnit::seconds : DurationUnit::samples;
    }

    void restart() const {
        _lower          = Comparator{{.level = threshold.value, .hysteresis = hysteresis.value}};
        _upper          = Comparator{{.level = threshold_upper.value, .hysteresis = hysteresis.value}};
        _upperFromAbove = Comparator{{.level = negated(threshold_upper.value), .hysteresis = hysteresis.value}};
        forgetPartial();
        _slewOutOfBand = false;
        _recent.reset();
        _spanClipped          = false;
        _previousClipped      = false;
        _unmeasurableReported = false;
        _stateEpoch           = _settings_epoch;
    }

    /** what was partly measured is discarded, never completed across unknown data */
    void forgetPartial() const {
        if (_start && (measuredCondition() == ValueCondition::pulseWidth || measuredCondition() == ValueCondition::runt || measuredCondition() == ValueCondition::transition)) {
            n_abandoned = n_abandoned + 1U;
        }
        _start       = std::nullopt;
        _touched     = false;
        _reported    = false;
        _inside      = std::nullopt;
        _spanClipped = false;
    }

    void forgetState() const {
        _lower.reset();
        _upper.reset();
        _upperFromAbove.reset();
        _recent.reset();
        _previousClipped = false;
        forgetPartial();
    }

    [[nodiscard]] bool wants(bool rising) const noexcept { return _edge == EdgeSelection::both || rising == (_edge == EdgeSelection::rising); }

    /** a duration in samples, as `duration_unit` states its limits */
    [[nodiscard]] std::optional<double> limitInSamples(double limit) const noexcept {
        if (_durationUnit == DurationUnit::samples) {
            return limit;
        }
        return _time.measuresDuration() ? std::optional{limit * _time.rate()} : std::nullopt;
    }

    /** a duration in seconds against a block that does not know its rate decides nothing */
    [[nodiscard]] bool durationUnmeasurable() const noexcept { return _timed && _durationUnit == DurationUnit::seconds && !_time.measuresDuration(); }

    void rejectDuration() const noexcept {
        if (durationUnmeasurable()) {
            n_unmeasurable = n_unmeasurable + 1U;
        } else {
            n_rejected = n_rejected + 1U;
        }
    }

    /** a clipped span has no confirmed width: it is reported, flagged, whatever the duration test says */
    [[nodiscard]] bool spanUnknown() const noexcept { return _spanClipped || _sampleClipped || _previousClipped; }

    [[nodiscard]] bool passes(double durationSamples) const noexcept {
        if (!_timed) {
            return true;
        }
        const auto low  = limitInSamples(_durationMin);
        const auto high = limitInSamples(_durationMax);
        return low && high && holds(_durationTest, durationSamples, *low, *high);
    }

    [[nodiscard]] static double crossingOf(const Comparator& comparator, std::size_t at) noexcept { return static_cast<double>(at) + static_cast<double>(gr::value(comparator.lastEdgePosition)); }

    /** the pulse-like conditions measure from a first crossing to a second; a completed one passes or is rejected */
    [[nodiscard]] std::optional<Found> completed(double stop, bool startRising, float error) const {
        const double width = stop - *_start;
        const Found  found{.moment = stop, .reference = *_start, .referenceRising = startRising, .error = error};
        _start = std::nullopt;
        if (spanUnknown() || passes(width)) {
            return found;
        }
        rejectDuration();
        return std::nullopt;
    }

    /** `measure` with the clipped samples tracked: a clipped sample inside a measured span, or one a crossing interpolates from, leaves the result flagged */
    [[nodiscard]] std::optional<Found> decide(T sample, std::size_t at) const {
        _sampleClipped         = clip_level.value > T{} && (sample >= clip_level.value || sample <= negated(clip_level.value));
        const auto startBefore = _start;
        auto       found       = measure(sample, at);
        if (found) {
            found->clipped = spanUnknown();
        }
        if (_start && _start != startBefore) {
            _spanClipped = _sampleClipped || _previousClipped;
        } else if (!_start) {
            _spanClipped = false;
        } else {
            _spanClipped = _spanClipped || _sampleClipped;
        }
        _previousClipped = _sampleClipped;
        return found;
    }

    [[nodiscard]] std::optional<Found> measure(T sample, std::size_t at) const {
        using enum gr::trigger::EdgeDetection;
        if constexpr (std::is_floating_point_v<T>) {
            if (!std::isfinite(sample)) [[unlikely]] {
                forgetState();
                return std::nullopt;
            }
        }
        if (_inverted) {
            return std::nullopt;
        }
        const bool  twoLevels = measuredCondition() == ValueCondition::runt || measuredCondition() == ValueCondition::transition;
        const auto  lowerEdge = _lower.processOne(sample);
        const auto  upperEdge = twoLevels ? _upper.processOne(sample) : NONE;
        const auto  fromAbove = measuredCondition() == ValueCondition::window ? _upperFromAbove.processOne(negated(sample)) : NONE;
        const float error     = gr::uncertainty(_lower.lastEdgePosition);

        switch (measuredCondition()) {
        case ValueCondition::level:
            if (lowerEdge != NONE && wants(lowerEdge == RISING)) {
                return Found{.moment = crossingOf(_lower, at), .reference = std::nullopt, .referenceRising = false, .error = error};
            }
            return std::nullopt;

        case ValueCondition::window: {
            using enum gr::trigger::LineState;
            const bool insideNow  = _lower.state() == high && _upperFromAbove.state() == high;
            const bool outsideNow = _lower.state() == low || _upperFromAbove.state() == low;
            if (!insideNow && !outsideNow) {
                return std::nullopt;
            }
            const bool changed = _inside.has_value() && *_inside != insideNow && (lowerEdge != NONE || fromAbove != NONE);
            _inside            = insideNow;
            if (!changed) {
                return std::nullopt;
            }
            const double                crossing = crossingOf(lowerEdge != NONE ? _lower : _upperFromAbove, at);
            const std::optional<double> previous = _start; /// the crossing that began the dwell: an exit before an entry, an entry before an exit
            _start                               = crossing;
            if (!wants(insideNow)) {
                return std::nullopt;
            }
            if (_timed && !spanUnknown() && !(previous && passes(crossing - *previous))) {
                rejectDuration();
                return std::nullopt;
            }
            return Found{.moment = crossing, .reference = _timed ? previous : std::nullopt, .referenceRising = !insideNow, .error = error};
        }

        case ValueCondition::pulseWidth:
            if (lowerEdge == NONE) {
                return std::nullopt;
            }
            if (!_start) {
                if (wants(lowerEdge == RISING)) {
                    _start = crossingOf(_lower, at);
                }
                return std::nullopt;
            }
            {
                const double crossing = crossingOf(_lower, at);
                const auto   found    = completed(crossing, lowerEdge != RISING, error); // a pulse starts with the other edge
                if (_edge == EdgeSelection::both) {                                      // the trailing edge of one pulse leads the next, of the other polarity
                    _start = crossing;
                }
                return found;
            }

        case ValueCondition::runt: {
            // positive: up across threshold and back without touching threshold_upper; negative: the mirror image
            const bool  positive = _edge != EdgeSelection::falling;
            const auto& first    = positive ? _lower : _upper;
            const auto  edgeOf   = positive ? lowerEdge : upperEdge;
            const bool  away     = edgeOf == (positive ? RISING : FALLING);
            if (away) {
                _start   = crossingOf(first, at);
                _touched = false;
            }
            if (!_start) {
                return std::nullopt;
            }
            _touched = _touched || (positive ? sample >= threshold_upper.value : sample <= threshold.value);
            if (edgeOf == NONE || away) {
                return std::nullopt;
            }
            if (_touched) {
                _start = std::nullopt;
                return std::nullopt;
            }
            return completed(crossingOf(first, at), positive, error);
        }

        case ValueCondition::transition: {
            // rising: from threshold up to threshold_upper; falling: from threshold_upper down to threshold
            const bool  rising   = _edge != EdgeSelection::falling;
            const auto& from     = rising ? _lower : _upper;
            const auto& to       = rising ? _upper : _lower;
            const auto  fromEdge = rising ? lowerEdge : upperEdge;
            const auto  toEdge   = rising ? upperEdge : lowerEdge;
            const auto  onward   = rising ? RISING : FALLING;
            if (fromEdge == onward) {
                _start = crossingOf(from, at);
            } else if (fromEdge != NONE && _start) { // reversed before reaching the second threshold
                const double began = *_start;
                _start             = std::nullopt;
                n_abandoned        = n_abandoned + 1U;
                return Found{.moment = crossingOf(from, at), .reference = began, .referenceRising = rising, .error = error, .label = "reversed"};
            }
            if (_start && toEdge == onward) {
                return completed(crossingOf(to, at), rising, error);
            }
            return std::nullopt;
        }

        case ValueCondition::slewRate: {
            const std::size_t span = slew_samples == 0U ? 1UZ : static_cast<std::size_t>(slew_samples.value);
            _recent.push_front(sample);
            if (_recent.size() <= span) {
                return std::nullopt;
            }
            if (!_time.measuresDuration()) {
                n_unmeasurable = n_unmeasurable + 1U;
                return std::nullopt;
            }
            const double rate      = _time.rate();
            const double perSecond = static_cast<double>(sample - _recent[span]) * rate / static_cast<double>(span);
            const bool   tooSlow   = slew_min_rate > 0.f && std::abs(perSecond) < static_cast<double>(slew_min_rate);
            const bool   tooFast   = slew_max_rate > 0.f && std::abs(perSecond) > static_cast<double>(slew_max_rate);
            if (tooSlow || tooFast) {
                if (_slewOutOfBand) {
                    return std::nullopt;
                }
                _slewOutOfBand = true;
                return Found{.moment = static_cast<double>(at), .reference = std::nullopt, .referenceRising = false, .error = 0.f};
            }
            _slewOutOfBand = false;
            return std::nullopt;
        }

        case ValueCondition::levelTimeout: {
            using enum gr::trigger::LineState;
            if (lowerEdge != NONE) { // an episode starts at every edge, a watched one counts
                _start    = wants(lowerEdge == RISING) ? std::optional{crossingOf(_lower, at)} : std::nullopt;
                _reported = false;
            } else if (!_start && _lower.state() != unknown && wants(_lower.state() == high) && !_reported) {
                _start = static_cast<double>(at); // the level was already held when the state became known: a lower bound
            }
            return timedOut(at, _lower.state() == high, "ongoing");
        }

        case ValueCondition::edgeTimeout:
            if (lowerEdge != NONE && wants(lowerEdge == RISING)) {
                _start       = crossingOf(_lower, at);
                _startRising = lowerEdge == RISING;
                _reported    = false;
                return std::nullopt;
            }
            return timedOut(at, _startRising, {});

        case ValueCondition::interval:
            if (lowerEdge == NONE || !wants(lowerEdge == RISING)) {
                return std::nullopt;
            }
            if (!_start) {
                _start = crossingOf(_lower, at);
                return std::nullopt;
            }
            {
                const double stop  = crossingOf(_lower, at);
                const double start = *_start;
                _start             = stop;
                const double width = stop - start;
                if (!spanUnknown() && !passes(width)) {
                    rejectDuration();
                    return std::nullopt;
                }
                return Found{.moment = stop, .reference = start, .referenceRising = lowerEdge == RISING, .error = error};
            }

        case ValueCondition::automatic: return std::nullopt;
        }
        return std::nullopt;
    }

    /** a timeout is decided once its duration is reached (or exceeded) and dated at that deadline, start + duration, however late the deciding sample;
       `startRising` names the crossing that began it */
    [[nodiscard]] std::optional<Found> timedOut(std::size_t at, bool startRising, std::string_view label) const {
        if (!_timed || !_start || _reported) { // no duration, no timeout
            return std::nullopt;
        }
        const double elapsed = static_cast<double>(at) - *_start;
        if (!passes(elapsed)) {
            return std::nullopt;
        }
        _reported             = true;
        const double deadline = *_start + limitInSamples(_durationMin).value_or(elapsed);
        return Found{.moment = deadline, .reference = *_start, .referenceRising = startRising, .error = 0.f, .label = label};
    }

    /**
     * `trigger_meta_info`: the reference trigger tuple of a delayed decision, the class of an unusual outcome and the
     * flags of a clipped span, which are always written; then, while the event slot has room, a transition's equivalent
     * slope, the estimator that dated the crossings and the lower comparator's boundaries
     */
    [[nodiscard]] bool writeEvent(gr::property_map_view& event, const Found& found, std::size_t at) const noexcept {
        using enum gr::trigger::EdgeDetection;
        constexpr std::uint32_t kScalarBytes = gr::pmt::alignToRecord(gr::pmt::kRecHeaderBytes + static_cast<std::uint32_t>(sizeof(double)));
        constexpr std::uint32_t kStringBytes = 16U; /// a record: 8 B header, the characters and a NUL, to 16 B
        constexpr std::uint32_t kLabelBytes  = 32U;
        constexpr std::uint32_t kFlagsBytes  = 176U;
        const double            tagSample    = static_cast<double>(at);
        bool                    complete     = writeDatedTrigger(event, _time, at, found.moment - tagSample, found.error, trigger_name.value, context.value);

        const bool hasReference      = found.reference && _time.datesSamples();
        const bool hasLabel          = !found.label.empty();
        const bool isTransitionDated = measuredCondition() == ValueCondition::transition && hasReference && !hasLabel;
        const bool hasSlope          = isTransitionDated && found.moment > *found.reference && _time.measuresDuration();
        /** the root tuple always fits; the meta degrades tier by tier: slope, estimator, boundaries, flags, class, reference */
        gr::property_map_view details;
        bool                  withSlope      = false;
        bool                  withEstimator  = false;
        bool                  withBoundaries = false;
        bool                  withFlags      = false;
        bool                  withLabel      = false;
        bool                  withReference  = false;
        for (int tier = 0; tier < 7 && details.blob().empty(); ++tier) {
            withSlope                      = hasSlope && tier < 1;
            withEstimator                  = tier < 2;
            withBoundaries                 = tier < 3;
            withFlags                      = found.clipped && tier < 4;
            withLabel                      = hasLabel && tier < 5;
            withReference                  = hasReference && tier < 6;
            const std::uint32_t nHeadKeys  = (withReference ? 3U : 0U) + (withLabel ? 1U : 0U) + (withFlags ? 1U : 0U);
            const std::uint32_t nTailKeys  = (withSlope ? 1U : 0U) + (withEstimator ? 1U : 0U) + (withBoundaries ? 2U : 0U);
            const std::uint32_t nKeys      = nHeadKeys + nTailKeys;
            const std::uint32_t nHeadBytes = (withReference ? 2U * kScalarBytes + kStringBytes : 0U) + (withLabel ? kLabelBytes : 0U) + (withFlags ? kFlagsBytes : 0U);
            const std::uint32_t nTailBytes = (withSlope ? kScalarBytes : 0U) + (withEstimator ? kStringBytes : 0U) + (withBoundaries ? 2U * kScalarBytes : 0U);
            const std::uint32_t nBytes     = nHeadBytes + nTailBytes;
            if (nKeys > 0U) {
                details = event.try_emplace_map(gr::tag::kTriggerMetaInfo.key(), nKeys, nBytes);
            }
        }
        if (details.blob().empty()) {
            return complete;
        }
        if (withReference) {
            const std::string_view referenceName = gr::meta::enumName(found.referenceRising ? RISING : FALLING).value_or("");
            complete                             = complete && details.try_emplace(gr::tag::kTriggerName.key(), referenceName);
            complete                             = complete && writeTriggerTime(details, _time, at, *found.reference - tagSample);
        }
        if (withBoundaries) {
            complete = complete && details.try_emplace(std::string_view{"boundary_rising"}, _lower.boundary(true));
            complete = complete && details.try_emplace(std::string_view{"boundary_falling"}, _lower.boundary(false));
        }
        if (withEstimator) {
            complete = complete && details.try_emplace(std::string_view{"estimator"}, std::string_view{"linear"});
        }
        if (withSlope) {
            const double riseSeconds = (found.moment - *found.reference) / _time.rate();
            const double span        = static_cast<double>(threshold_upper.value) - static_cast<double>(threshold.value);
            complete                 = complete && details.try_emplace(std::string_view{"slope"}, (found.referenceRising ? span : -span) / riseSeconds);
        }
        if (withLabel) {
            complete = complete && details.try_emplace(std::string_view{"class"}, found.label);
        }
        if (withFlags) {
            gr::property_map_view flags = details.try_emplace_map(gr::tag::kAcquisitionFlags.key(), 1U, kScalarBytes);
            complete                    = complete && !flags.blob().empty() && flags.try_emplace(std::string_view{"clipped"}, true);
        }
        return complete;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_VALUETRIGGER_HPP
