#ifndef GNURADIO_TRIGGER_SCHMITTTRIGGER_HPP
#define GNURADIO_TRIGGER_SCHMITTTRIGGER_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory_resource>
#include <string_view>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK("gr::blocks::trigger::SchmittTriggerNoInterpolation", gr::blocks::trigger::SchmittTrigger, ([T], gr::trigger::InterpolationMethod::NO_INTERPOLATION), [ std::int16_t, std::int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SchmittTriggerBasic", gr::blocks::trigger::SchmittTrigger, ([T], gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION), [ std::int16_t, std::int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SchmittTrigger", gr::blocks::trigger::SchmittTrigger, ([T], gr::trigger::InterpolationMethod::LINEAR_INTERPOLATION), [ std::int16_t, std::int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SchmittTriggerPolynomial", gr::blocks::trigger::SchmittTrigger, ([T], gr::trigger::InterpolationMethod::POLYNOMIAL_INTERPOLATION), [ std::int16_t, std::int32_t, float, double ])

template<typename T, gr::trigger::InterpolationMethod Method>
requires(std::is_arithmetic_v<T> or (UncertainValueLike<T> && std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>))
struct SchmittTrigger : public gr::Block<SchmittTrigger<T, Method>, NoTagPropagation> {
    using Description = Doc<R"(@brief report an edge with hysteresis, interpolated to a fraction of a sample

    in      ──╱‾‾‾‾╲__╱‾‾‾▶          (no RxMarbles equivalent: a condition on the waveform)
    evtOut  ───R────F───R──▶         R = rising, F = falling; a band of width hysteresis about threshold

The first boundary reached after a start, a reset or a non-finite sample settles the state without an edge.

 [1] specification: IEEE Std 181-2011, transition and pulse parameters
)">;
    using enum gr::trigger::EdgeDetection;
    using value_t = meta::fundamental_base_value_type_t<T>;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::size_t kNHistory = 32UZ;

    PortIn<T>        in;
    PortOut<T>       out;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<value_t, "threshold", Doc<"the level edges are taken about">, Visible>                                             threshold{value_t(0)};
    A<value_t, "hysteresis", Doc<"full width of the band about threshold, >= 0; zero: high is x >= threshold">, Visible> hysteresis{value_t(0)};
    A<std::pmr::string, "edge", Doc<"rising|falling|both: the edges that are reported">, Visible>                        edge           = std::pmr::string("both");
    A<double, "dead time", Doc<"minimum distance between reported edges, in dead_time_unit">, Visible>                   dead_time      = 0.;
    A<std::pmr::string, "dead time unit", Doc<"samples|seconds">, Visible>                                               dead_time_unit = std::pmr::string("samples");
    gr::SampleRate                                                                                                       sample_rate    = 0.f;

    A<bool, "forward tags", Doc<"false: emit only tags for detected edges">>                                                        forward_tags{true};
    gr::TriggerName                                                                                                                 trigger_name = "";
    A<std::pmr::string, "rising trigger", Doc<"name of a rising edge; empty = trigger_name">, Visible>                              trigger_name_rising;
    A<std::pmr::string, "falling trigger", Doc<"name of a falling edge; empty = trigger_name">, Visible>                            trigger_name_falling;
    gr::TriggerTime                                                                                                                 trigger_time{0U};
    gr::TriggerOffset                                                                                                               trigger_offset{0.0f};
    gr::Context                                                                                                                     context          = "";
    mutable A<gr::Size_t, "n events dropped", Doc<"edges a full evtOut could not take">>                                            n_events_dropped = 0U;
    mutable A<gr::Size_t, "n unmeasurable", Doc<"edges whose dead time in seconds could not be applied for want of a sample_rate">> n_unmeasurable   = 0U;
    mutable A<gr::Size_t, "n gaps", Doc<"gap tags that reset the detection">>                                                       n_gaps           = 0U;

    using TriggerType = gr::trigger::SchmittTrigger<T, Method, kNHistory>;

    gr::Size_t                                                _comparator_epoch  = 0U;
    gr::Size_t                                                _anchor_epoch      = 0U;
    EdgeSelection                                             _edge              = EdgeSelection::both;
    bool                                                      _deadTimeInSeconds = false;
    std::array<typename TriggerType::sg_compute_t, kNHistory> _sg_coefficients{};

    GR_MAKE_REFLECTABLE(SchmittTrigger, in, out, evtOut, threshold, hysteresis, edge, dead_time, dead_time_unit, sample_rate, forward_tags, trigger_name, trigger_name_rising, trigger_name_falling, trigger_time, trigger_offset, context, n_events_dropped, n_unmeasurable, n_gaps, _comparator_epoch, _anchor_epoch, _edge, _deadTimeInSeconds, _sg_coefficients);

    mutable TriggerType _trigger{};
    mutable TimeBase    _time;
    mutable std::size_t _triggerCursor          = 0UZ;
    mutable std::size_t _deadUntil              = 0UZ;
    mutable gr::Size_t  _appliedComparatorEpoch = 0U;
    mutable gr::Size_t  _appliedAnchorEpoch     = 0U;
    mutable bool        _undatedReported        = false;
    mutable bool        _deadTimeReported       = false;

    detail::ErrorMemory _deferredError{};

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (const auto named = gr::meta::parseEnum<EdgeSelection>(edge.value)) {
            _edge = *named;
        } else {
            const std::string message = std::format("edge '{}' is not rising, falling or both; keeping '{}'", edge.value, gr::meta::enumName(_edge).value_or(""));
            detail::reportError(*this, "settingsChanged()", message);
            edge = std::pmr::string(gr::meta::enumName(_edge).value_or(""));
        }
        _deadTimeInSeconds = dead_time_unit.value == "seconds";
        if (newSettings.contains("trigger_time")) {
            _anchor_epoch    = _anchor_epoch + 1U;
            _undatedReported = false;
        }
        if (newSettings.contains("threshold") || newSettings.contains("hysteresis")) {
            _comparator_epoch = _comparator_epoch + 1U;
        }
    }

    void start() {
        detail::sendDeferredError(*this);
        if constexpr (Method != gr::trigger::InterpolationMethod::NO_INTERPOLATION) {
            /** the window stays unpublished, so a span cut at a tag must exceed it */
            constexpr std::size_t kMinSpan = kNHistory + 1UZ;
            in.min_samples                 = kMinSpan;
            if (in.max_samples < kMinSpan) {
                const std::string message = std::format("max_samples {} is below the {}-sample interpolation window and is raised to it", in.max_samples, kMinSpan);
                detail::reportError(*this, "processBulk()", message);
                in.max_samples = kMinSpan;
            }
        }
        if constexpr (Method == gr::trigger::InterpolationMethod::POLYNOMIAL_INTERPOLATION) {
            std::ranges::copy(TriggerType::computeCoefficients(std::pmr::get_default_resource()), _sg_coefficients.begin());
        }
        detail::nameTriggerAfterBlock(*this);
        reset();
    }

    void reset() {
        _trigger = TriggerType{{.level = threshold, .hysteresis = hysteresis}};
        _time.reset();
        _triggerCursor          = 0UZ;
        _deadUntil              = 0UZ;
        _appliedComparatorEpoch = _comparator_epoch;
        _appliedAnchorEpoch     = _anchor_epoch;
        _undatedReported        = false;
        n_events_dropped        = 0U;
        n_gaps                  = 0U;
        n_unmeasurable          = 0U;
        _deadTimeReported       = false;
    }

    gr::work::Status processBulk(InputSpanLike auto& inputSpan, OutputSpanLike auto& outputSpan, OutputSpanLike auto& evtOutSpan) const {
        constexpr bool    kOnHost     = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inputSpan)>>;
        const std::size_t streamIndex = inputSpan.streamIndex;
        const std::size_t nSpan       = std::min(inputSpan.size(), outputSpan.size());

        _time.setRate(static_cast<double>(sample_rate));
        if (_appliedComparatorEpoch != _comparator_epoch) {
            _trigger                = TriggerType{{.level = threshold, .hysteresis = hysteresis}};
            _appliedComparatorEpoch = _comparator_epoch;
        }
        if constexpr (Method == gr::trigger::InterpolationMethod::POLYNOMIAL_INTERPOLATION) {
            _trigger.setCoefficients(_sg_coefficients);
        }
        if (_appliedAnchorEpoch != _anchor_epoch) {
            std::ignore         = _time.anchorAt(trigger_time, trigger_offset, streamIndex);
            _appliedAnchorEpoch = _anchor_epoch;
        }

        auto tags = inputSpan.rawTags();
        auto next = std::ranges::begin(tags);

        std::size_t      edgeAt = nSpan;
        double           edgeReference{}; /// samples from the tag's sample back to the crossing
        std::string_view edgeName;
        for (std::size_t i = _triggerCursor > streamIndex ? _triggerCursor - streamIndex : 0UZ; i < nSpan && edgeAt == nSpan; ++i) {
            _triggerCursor = streamIndex + i + 1UZ;
            for (auto tag = detail::tagsOn(next, std::ranges::end(tags), streamIndex + i); tag != next; ++tag) {
                if (detail::isGap(gr::property_map_view{(*tag).map})) {
                    _trigger.reset(); // a gap: no state, and no history to interpolate across it
                    n_gaps = n_gaps + 1U;
                }
            }
            if (_trigger.processOne(inputSpan[i]) == NONE) {
                continue;
            }
            const double           crossing     = static_cast<double>(i) + static_cast<double>(gr::value(_trigger.lastEdgePosition));
            const std::size_t      edgePosition = static_cast<std::size_t>(std::max(0., std::ceil(crossing))); // the first sample at or after it
            const bool             wanted       = _edge == EdgeSelection::both || (_edge == EdgeSelection::rising) == (_trigger.lastEdge == RISING);
            const std::string_view risingName   = trigger_name_rising.value;
            const std::string_view fallingName  = trigger_name_falling.value;
            const std::string_view named        = _trigger.lastEdge == RISING ? risingName : fallingName;
            const std::string_view chosen       = named.empty() ? std::string_view{trigger_name.value} : named;
            if (wanted && streamIndex + edgePosition >= _deadUntil && !chosen.empty()) {
                edgeName      = chosen;
                edgeAt        = edgePosition;
                edgeReference = crossing - static_cast<double>(edgePosition);
            }
        }

        const bool        edgeFound = edgeAt < nSpan;
        const std::size_t nPublish  = edgeFound ? edgeAt + 1UZ : std::min(publishableInput(inputSpan), nSpan);
        if (nPublish == 0UZ) {
            evtOutSpan.publish(0UZ);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        if (forward_tags) {
            for (const auto& tag : inputSpan.rawTags()) {
                if (tag.index >= streamIndex && tag.index - streamIndex < nPublish) {
                    outputSpan.publishTag(gr::property_map_view{tag.map}, tag.index - streamIndex);
                }
            }
        }

        std::size_t nEmitted = 0UZ;
        if (edgeFound) {
            alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> blob{};
            gr::property_map_view                                                             reported   = gr::formatEvent(blob, kDatedTriggerKeys + 1U);
            const auto                                                                        edgeSample = streamIndex + edgeAt;
            const auto                                                                        edgeError  = gr::uncertainty(_trigger.lastEdgePosition);
            const bool                                                                        complete   = writeDatedTrigger(reported, _time, edgeSample, edgeReference, edgeError, edgeName, context.value);
            if constexpr (kOnHost) {
                if (!_time.datesSamples() && !_undatedReported) {
                    _undatedReported          = true;
                    const std::string message = std::format("no anchor and no sample_rate, so an edge cannot be dated; emitting '{}' without a time", edgeName);
                    detail::reportError(*this, "processBulk()", message);
                }
                if (!complete) {
                    const std::string message = std::format("edge '{}' left without some fields: its names exceed the {} B an edge may carry", edgeName, gr::kDeviceEventSlotBytes);
                    detail::reportError(*this, "processBulk()", message);
                }
            }
            outputSpan.publishTag(reported, edgeAt);
            if (evtOutSpan.isConnected) {
                if (gr::emit(evtOutSpan, 0UZ, reported)) {
                    nEmitted = 1UZ;
                } else {
                    n_events_dropped = n_events_dropped + 1U;
                }
            }
            const bool unmeasurable = _deadTimeInSeconds && dead_time > 0. && !_time.measuresDuration();
            if (unmeasurable) {
                n_unmeasurable = n_unmeasurable + 1U;
                if constexpr (kOnHost) {
                    if (!_deadTimeReported) {
                        _deadTimeReported = true;
                        detail::reportError(*this, "processBulk()", "dead_time in seconds needs a sample_rate; no dead time is applied");
                    }
                }
            }
            const double deadSamples = _deadTimeInSeconds ? dead_time * static_cast<double>(sample_rate) : dead_time.value;
            _deadUntil               = streamIndex + edgeAt + (unmeasurable ? 0UZ : static_cast<std::size_t>(std::ceil(deadSamples)));
        }
        evtOutSpan.publish(nEmitted);

        std::copy_n(inputSpan.begin(), nPublish, outputSpan.begin());
        std::ignore = inputSpan.consume(nPublish);
        outputSpan.publish(nPublish);
        return gr::work::Status::OK;
    }

    /** an interpolating trigger holds back its look-ahead window until the stream ends */
    [[nodiscard]] std::size_t publishableInput(const auto& inputSpan) const {
        if constexpr (Method == gr::trigger::InterpolationMethod::NO_INTERPOLATION) {
            return inputSpan.size();
        } else {
            const bool streamEnds = [&] {
                if constexpr (gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inputSpan)>>) {
                    return inputSpan.streamEnds();
                } else {
                    /** a read-only query the port API offers only non-const */
                    return gr::endsStreamAt(const_cast<PortIn<T>&>(in), inputSpan.streamIndex + inputSpan.size());
                }
            }();
            return streamEnds ? inputSpan.size() : (inputSpan.size() > kNHistory ? inputSpan.size() - kNHistory : 0UZ);
        }
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_SCHMITTTRIGGER_HPP
