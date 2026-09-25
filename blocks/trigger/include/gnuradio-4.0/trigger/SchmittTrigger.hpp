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
    evtOut  ───R────F───R──▶         R = rising, F = falling; threshold about offset

`Method` picks the interpolation: none (the crossing sample), basic linear (no look-ahead), linear, or a Savitzky-Golay
polynomial fit. `trigger_name`, `trigger_time` and `trigger_offset` hold the last trigger the edges are dated from, by
counting samples since.

 [1] specification: IEEE Std 181-2011, transition and pulse parameters
)">;
    using enum gr::trigger::EdgeDetection;
    using value_t = meta::fundamental_base_value_type_t<T>;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::size_t N_HISTORY = 32UZ;

    PortIn<T>        in;
    PortOut<T>       out;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<value_t, "offset", Visible>                                                                    offset{value_t(0)};
    A<value_t, "threshold", Visible>                                                                 threshold{value_t(1)};
    A<gr::Size_t, "dead time", Doc<"minimum sample distance between emitted edges">, Visible>        n_dead_time = 0U;
    A<std::pmr::string, "rising trigger", Doc<"rising-edge trigger name; \"\" omits it">, Visible>   trigger_name_rising_edge{std::string(gr::meta::enumName(RISING).value_or(""))};
    A<std::pmr::string, "falling trigger", Doc<"falling-edge trigger name; \"\" omits it">, Visible> trigger_name_falling_edge{std::string(gr::meta::enumName(FALLING).value_or(""))};
    gr::SampleRate                                                                                   sample_rate = 1.f;

    A<bool, "forward tags ", Doc<"false: emit only tags for detected edges">>                            forward_tag{true};
    gr::TriggerName                                                                                      trigger_name = "";
    gr::TriggerTime                                                                                      trigger_time{0U};
    gr::TriggerOffset                                                                                    trigger_offset{0.0f};
    gr::Context                                                                                          context          = "";
    mutable A<gr::Size_t, "n events dropped", Doc<"edges a full evtOut could not take; host path only">> n_events_dropped = 0U;

    using TriggerType = gr::trigger::SchmittTrigger<T, Method, N_HISTORY>;

    gr::Size_t                                                _comparator_epoch = 0U;
    gr::Size_t                                                _anchor_epoch     = 0U;
    std::array<typename TriggerType::sg_compute_t, N_HISTORY> _sg_coefficients{};
    detail::InlineText                                        _source{};

    GR_MAKE_REFLECTABLE(SchmittTrigger, in, out, evtOut, offset, threshold, n_dead_time, trigger_name_rising_edge, trigger_name_falling_edge, sample_rate, forward_tag, trigger_name, trigger_time, trigger_offset, context, n_events_dropped, _comparator_epoch, _anchor_epoch, _sg_coefficients, _source);

    mutable TriggerType _trigger{0, 1};
    mutable TimeBase    _time;
    mutable std::size_t _triggerCursor   = 0UZ;
    mutable std::size_t _deadUntil       = 0UZ;
    mutable gr::Size_t  _comparatorEpoch = 0U;
    mutable gr::Size_t  _anchorEpoch     = 0U;
    mutable bool        _undatedReported = false;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("trigger_time")) {
            _anchor_epoch    = _anchor_epoch + 1U;
            _undatedReported = false;
        }
        if (newSettings.contains("offset") || newSettings.contains("threshold")) {
            _comparator_epoch = _comparator_epoch + 1U;
        }
        _source.assign(this->unique_name.value());
    }

    void start() {
        if constexpr (Method != gr::trigger::InterpolationMethod::NO_INTERPOLATION) {
            /** the window stays unpublished, so a span cut at a tag must exceed it */
            constexpr std::size_t kMinSpan = N_HISTORY + 1UZ;
            in.min_samples                 = kMinSpan;
            if (in.max_samples < kMinSpan) {
                gr::log::warning("SchmittTrigger: max_samples {} is below the {}-sample interpolation window and is raised to it", in.max_samples, kMinSpan);
                in.max_samples = kMinSpan;
            }
        }
        if constexpr (Method == gr::trigger::InterpolationMethod::POLYNOMIAL_INTERPOLATION) {
            std::ranges::copy(TriggerType::computeCoefficients(std::pmr::get_default_resource()), _sg_coefficients.begin());
        }
        reset();
    }

    void reset() {
        _trigger.reset();
        _time.reset();
        _triggerCursor   = 0UZ;
        _deadUntil       = 0UZ;
        _comparatorEpoch = _comparator_epoch;
        _anchorEpoch     = _anchor_epoch;
        _undatedReported = false;
        n_events_dropped = 0U;
    }

    gr::work::Status processBulk(InputSpanLike auto& inputSpan, OutputSpanLike auto& outputSpan, OutputSpanLike auto& evtOutSpan) const {
        constexpr bool    kOnHost     = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inputSpan)>>;
        const std::size_t streamIndex = inputSpan.streamIndex;
        const std::size_t nSpan       = std::min(inputSpan.size(), outputSpan.size());

        _time.setRate(static_cast<double>(sample_rate));
        _trigger.setOffset(offset);
        _trigger.setThreshold(threshold);
        if constexpr (Method == gr::trigger::InterpolationMethod::POLYNOMIAL_INTERPOLATION) {
            _trigger.setCoefficients(_sg_coefficients);
        }
        if (_comparatorEpoch != _comparator_epoch) {
            _trigger.reset();
            _comparatorEpoch = _comparator_epoch;
        }
        if (_anchorEpoch != _anchor_epoch) {
            std::ignore  = _time.anchorAt(trigger_time, trigger_offset, streamIndex);
            _anchorEpoch = _anchor_epoch;
        }

        std::size_t      edgeAt = nSpan;
        std::string_view edgeName;
        for (std::size_t i = _triggerCursor > streamIndex ? _triggerCursor - streamIndex : 0UZ; i < nSpan && edgeAt == nSpan; ++i) {
            _triggerCursor = streamIndex + i + 1UZ;
            if (_trigger.processOne(inputSpan[i]) == NONE) {
                continue;
            }
            const std::size_t edgePosition = static_cast<std::size_t>(std::max<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(i) + _trigger.lastEdgeIdx));
            edgeName                       = _trigger.lastEdge == RISING ? std::string_view{trigger_name_rising_edge.value} : std::string_view{trigger_name_falling_edge.value};
            if (streamIndex + edgePosition >= _deadUntil && !edgeName.empty()) {
                edgeAt = edgePosition;
            }
        }

        const bool        edgeFound = edgeAt < nSpan;
        const std::size_t nPublish  = edgeFound ? edgeAt + 1UZ : std::min(publishableInput(inputSpan), nSpan);
        if (nPublish == 0UZ) {
            evtOutSpan.publish(0UZ);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        if (forward_tag) {
            for (const auto& tag : inputSpan.rawTags()) {
                if (tag.index >= streamIndex && tag.index - streamIndex < nPublish) {
                    outputSpan.publishTag(gr::property_map_view{tag.map}, tag.index - streamIndex);
                }
            }
        }

        std::size_t nEmitted = 0UZ;
        if (edgeFound) {
            alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> blob{};
            gr::property_map_view                                                             edge     = gr::formatEvent(blob, kDatedTriggerKeys + 1U);
            const bool                                                                        complete = writeDatedTrigger(edge, _time, streamIndex + edgeAt, _trigger.lastEdgeOffset, edgeName, context.value);
            if constexpr (kOnHost) {
                if (!_time.datesSamples() && !_undatedReported) {
                    _undatedReported = true;
                    gr::log::warning("SchmittTrigger: no anchor and no sample_rate, so an edge cannot be dated; emitting '{}' without a time", edgeName);
                }
                if (!complete) {
                    gr::log::warning("SchmittTrigger: edge '{}' left without some fields: its names exceed the {} B an edge may carry", edgeName, gr::kDeviceEventSlotBytes);
                }
            }
            outputSpan.publishTag(edge, edgeAt);
            if (evtOutSpan.isConnected) {
                std::ignore = edge.try_emplace(std::string_view{"source"}, _source.view());
                if (gr::emit(evtOutSpan, 0UZ, edge)) {
                    nEmitted = 1UZ;
                } else {
                    n_events_dropped = n_events_dropped + 1U;
                }
            }
            _deadUntil = streamIndex + edgeAt + static_cast<std::size_t>(n_dead_time);
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
            return streamEnds ? inputSpan.size() : (inputSpan.size() > N_HISTORY ? inputSpan.size() - N_HISTORY : 0UZ);
        }
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_SCHMITTTRIGGER_HPP
