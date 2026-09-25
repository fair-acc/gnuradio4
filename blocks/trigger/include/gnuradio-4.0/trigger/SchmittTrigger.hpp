#ifndef GNURADIO_TRIGGER_SCHMITTTRIGGER_HPP
#define GNURADIO_TRIGGER_SCHMITTTRIGGER_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

#include <memory_resource>

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
polynomial fit.

 [1] specification: IEEE Std 181-2011, transition and pulse parameters
)">;
    using enum gr::trigger::EdgeDetection;
    using value_t = meta::fundamental_base_value_type_t<T>;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::size_t N_HISTORY = 32UZ;

    PortIn<T>  in;
    PortOut<T> out;

    A<value_t, "offset", Visible>                                                                    offset{value_t(0)};
    A<value_t, "threshold", Visible>                                                                 threshold{value_t(1)};
    A<std::pmr::string, "rising trigger", Doc<"rising-edge trigger name; \"\" omits it">, Visible>   trigger_name_rising_edge{std::string(gr::meta::enumName(RISING).value_or(""))};
    A<std::pmr::string, "falling trigger", Doc<"falling-edge trigger name; \"\" omits it">, Visible> trigger_name_falling_edge{std::string(gr::meta::enumName(FALLING).value_or(""))};
    A<float, "avg. sample rate", Visible>                                                            sample_rate = 1.f;

    A<bool, "forward tags ", Doc<"false: emit only tags for detected edges">>                        forward_tag{true};
    A<std::pmr::string, "trigger name", Doc<"last trigger used to synchronise time">>                trigger_name = "";
    A<std::uint64_t, "trigger time", Doc<"last UTC trigger time; then sample counting">, Unit<"ns">> trigger_time{0U};
    A<float, "trigger offset", Doc<"last trigger offset; then sample counting">, Unit<"s">>          trigger_offset{0.0f};
    std::pmr::string                                                                                 context = "";

    GR_MAKE_REFLECTABLE(SchmittTrigger, in, out, offset, threshold, trigger_name_rising_edge, trigger_name_falling_edge, sample_rate, forward_tag, trigger_name, trigger_time, trigger_offset, context);

    using TriggerType = gr::trigger::SchmittTrigger<T, Method, N_HISTORY>;
    TriggerType                                          _trigger{0, 1};
    std::pmr::vector<typename TriggerType::sg_compute_t> _sgCoefficients;
    TimeBase                                             _time;
    std::size_t                                          _streamIndex     = 0UZ;
    bool                                                 _undatedReported = false;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("sample_rate")) {
            _time.setRate(static_cast<double>(sample_rate));
        }
        if (newSettings.contains("trigger_time")) {
            std::ignore      = _time.anchorAt(trigger_time, trigger_offset, _streamIndex);
            _undatedReported = false;
        }

        if (newSettings.contains("offset") || newSettings.contains("threshold")) {
            _trigger.setOffset(offset);
            _trigger.setThreshold(threshold);
            _trigger.reset();
        }
    }

    [[nodiscard]] property_map dateEdge(std::string_view triggerName, std::size_t edgePos) {
        if (!_time.datesSamples() && !_undatedReported) {
            _undatedReported = true;
            gr::log::warning("SchmittTrigger: no anchor and no sample_rate, so an edge cannot be dated; emitting '{}' without a time", triggerName);
        }
        return datedTrigger(_time, _streamIndex + edgePos, _trigger.lastEdgeOffset, triggerName, context);
    }

    void start() {
        if constexpr (Method != gr::trigger::InterpolationMethod::NO_INTERPOLATION) {
            in.min_samples = N_HISTORY;
            if (in.max_samples < N_HISTORY) {
                gr::log::warning("SchmittTrigger: max_samples {} is below the {}-sample interpolation window and is raised to it", in.max_samples, N_HISTORY);
                in.max_samples = N_HISTORY;
            }
        }
        reset();
    }

    void reset() {
        if constexpr (Method == gr::trigger::InterpolationMethod::POLYNOMIAL_INTERPOLATION) {
            if (_sgCoefficients.empty()) {
                _sgCoefficients = TriggerType::computeCoefficients(std::pmr::get_default_resource());
            }
            _trigger.setCoefficients(_sgCoefficients);
        }
        _trigger.reset();
        _streamIndex = 0UZ;
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
        _undatedReported = false;
    }

    gr::work::Status processBulk(InputSpanLike auto& inputSpan, OutputSpanLike auto& outputSpan) {
        const std::optional<std::size_t> nextEoSTag = samples_to_eos_tag(in);
        if (inputSpan.size() < N_HISTORY && !nextEoSTag.has_value()) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        const std::size_t nProcessInput = nextEoSTag.has_value() ? inputSpan.size() : (inputSpan.size() > N_HISTORY ? inputSpan.size() - N_HISTORY : 0UZ);
        const std::size_t nProcess      = std::min(nProcessInput, outputSpan.size());

        if (nProcess == 0) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        auto forwardTags = [&](std::size_t maxRelIndex) {
            if (!forward_tag) {
                return;
            }
            for (const auto& tag : inputSpan.rawTags()) {
                const auto relIndex = tag.index >= inputSpan.streamIndex                                   //
                                          ? static_cast<std::ptrdiff_t>(tag.index - inputSpan.streamIndex) //
                                          : -static_cast<std::ptrdiff_t>(inputSpan.streamIndex - tag.index);
                if (relIndex >= 0 && static_cast<std::size_t>(relIndex) <= maxRelIndex) {
                    outputSpan.publishTag(tag.map, static_cast<std::size_t>(relIndex));
                }
            }
        };

        auto publishEdge = [&](std::string_view triggerName, std::size_t edgePos) {
            forwardTags(edgePos);

            outputSpan.publishTag(dateEdge(triggerName, edgePos), edgePos);

            const std::size_t nPublish = edgePos + 1; // include edge sample
            std::copy_n(inputSpan.begin(), nPublish, outputSpan.begin());
            std::ignore = inputSpan.consume(nPublish);
            outputSpan.publish(nPublish);
            _streamIndex += nPublish;
            return gr::work::Status::OK;
        };

        for (std::size_t i = 0; i < nProcess; ++i) {
            if (_trigger.processOne(inputSpan[i]) != NONE) {
                const std::ptrdiff_t edgePosition = std::max<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(i) + _trigger.lastEdgeIdx);

                if (static_cast<std::size_t>(edgePosition) < nProcess) {
                    if (_trigger.lastEdge == RISING && !trigger_name_rising_edge.value.empty()) {
                        return publishEdge(trigger_name_rising_edge, static_cast<std::size_t>(edgePosition));
                    }
                    if (_trigger.lastEdge == FALLING && !trigger_name_falling_edge.value.empty()) {
                        return publishEdge(trigger_name_falling_edge, static_cast<std::size_t>(edgePosition));
                    }
                }
            }
        }

        std::copy_n(inputSpan.begin(), nProcess, outputSpan.begin());
        forwardTags(nProcess - 1);
        std::ignore = inputSpan.consume(nProcess);
        outputSpan.publish(nProcess);
        _streamIndex += nProcess;
        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_SCHMITTTRIGGER_HPP
