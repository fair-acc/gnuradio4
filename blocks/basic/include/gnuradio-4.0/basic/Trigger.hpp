#ifndef TRIGGER_HPP
#define TRIGGER_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::basic {

template<typename T, gr::trigger::InterpolationMethod Method>
requires(std::is_arithmetic_v<T> or (UncertainValueLike<T> && std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>))
struct SchmittTrigger : public gr::Block<SchmittTrigger<T, Method>, NoDefaultTagForwarding> {
    using Description = Doc<R""(@brief Digital Schmitt trigger implementation with optional intersample interpolation

@see https://en.wikipedia.org/wiki/Schmitt_trigger

The following sub-sample interpolation methods are supported:
  * NO_INTERPOLATION: nomen est omen
  * BASIC_LINEAR_INTERPOLATION: basic linear interpolation based on the new and previous sample
  * LINEAR_INTERPOLATION: interpolation via linear regression over the samples between when
    the lower and upper threshold has been crossed and vice versa
  * POLYNOMIAL_INTERPOLATION: Savitzky–Golay filter-based methods
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    (TODO: WIP needs Tensor<T> and SVD implementation)

The block generates the following optional trigger that are controlled by:
 * trigger_name_rising_edge  -> trigger name that is used when a rising edge is detected
 * trigger_name_falling_edge -> trigger name that is used when a falling edge is detected

The trigger time and offset is calculated through sample counting from the last synchronising trigger.
The information is stored (info only) in `trigger_name`, `trigger_time`, `trigger_offset`.
)"">;
    using enum gr::trigger::EdgeDetection;
    using ClockSourceType = std::chrono::system_clock;
    using value_t         = meta::fundamental_base_value_type_t<T>;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::size_t N_HISTORY = 16UZ;

    PortIn<T>  in;
    PortOut<T> out;

    A<value_t, "offset", Doc<"trigger offset">, Visible>                                                                         offset{value_t(0)};
    A<value_t, "threshold", Doc<"trigger threshold">, Visible>                                                                   threshold{value_t(1)};
    A<std::string, "rising trigger", Doc<"trigger name generated on detected rising edge (N.B. \"\" omits trigger)">, Visible>   trigger_name_rising_edge{magic_enum::enum_name(RISING)};
    A<std::string, "falling trigger", Doc<"trigger name generated on detected falling edge (N.B. \"\" omits trigger)">, Visible> trigger_name_falling_edge{magic_enum::enum_name(FALLING)};
    A<float, "avg. sample rate", Visible>                                                                                        sample_rate = 1.f;

    A<bool, "forward tags ", Doc<"false: emit only tags for detected edges">>                                                  forward_tag{true};
    A<std::string, "trigger name", Doc<"last trigger used to synchronise time">>                                               trigger_name = "";
    A<std::uint64_t, "trigger time", Doc<"last trigger UTC time used for synchronisation (then sample counting)">, Unit<"ns">> trigger_time{0U};
    A<float, "trigger offset", Doc<"last trigger offset time used for synchronisation (then sample counting)">, Unit<"s">>     trigger_offset{0.0f};
    std::string                                                                                                                context = "";

    GR_MAKE_REFLECTABLE(SchmittTrigger, in, out, offset, threshold, trigger_name_rising_edge, trigger_name_falling_edge, sample_rate, forward_tag, trigger_name, trigger_time, trigger_offset, context);

    gr::trigger::SchmittTrigger<T, Method, N_HISTORY> _trigger{0, 1};
    std::uint64_t                                     _period{1U};
    std::uint64_t                                     _now{0U};

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("sample_rate")) {
            _period = static_cast<std::uint64_t>(1e6f / sample_rate);
        }
        if (newSettings.contains("trigger_time")) {
            _now = trigger_time + static_cast<std::uint64_t>(1e6f * trigger_offset);
        }

        if (newSettings.contains("offset") || newSettings.contains("threshold")) {
            _trigger.setOffset(offset);
            _trigger.setThreshold(threshold);
            _trigger.reset();
        }
    }

    void start() { reset(); }

    void reset() {
        _trigger.reset();
        _now         = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ClockSourceType::now().time_since_epoch()).count());
        trigger_time = _now;
    }

    gr::work::Status processBulk(InputSpanLike auto& inputSpan, OutputSpanLike auto& outputSpan) {
        const std::optional<std::size_t> nextEoSTag = samples_to_eos_tag(in);
        if (inputSpan.size() < N_HISTORY && nextEoSTag.value_or(0) >= N_HISTORY) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const std::size_t nProcess = inputSpan.size() - N_HISTORY * nextEoSTag.has_value(); // limit input data to have enough samples to back-date detected trigger if necessary

        auto forwardTags = [&](std::size_t maxRelIndex) {
            if (!forward_tag) {
                return;
            }
            for (const auto& tag : inputSpan.rawTags) {
                const auto relIndex = tag.index >= inputSpan.streamIndex                                   //
                                          ? static_cast<std::ptrdiff_t>(tag.index - inputSpan.streamIndex) //
                                          : -static_cast<std::ptrdiff_t>(inputSpan.streamIndex - tag.index);
                if (relIndex >= 0 && static_cast<std::size_t>(relIndex) <= maxRelIndex) {
                    outputSpan.publishTag(tag.map, static_cast<std::size_t>(relIndex));
                }
            }
        };

        auto publishEdge = [&](const std::string& triggerName, std::ptrdiff_t edgePosition) {
            const std::size_t edgePosUnsigned = std::max(0UZ, static_cast<std::size_t>(edgePosition));

            forwardTags(edgePosUnsigned);

            const UncertainValue<float> edgeIdxOffset = UncertainValue<float>{static_cast<float>(_trigger.lastEdgeIdx)} + _trigger.lastEdgeOffset;
            const float                 relOffset     = gr::value(edgeIdxOffset) * static_cast<float>(_period);
            outputSpan.publishTag(
                property_map{
                    //
                    {gr::tag::TRIGGER_NAME.shortKey(), triggerName},                                                             //
                    {gr::tag::TRIGGER_TIME.shortKey(), _now - static_cast<uint64_t>(relOffset)},                                 //
                    {"trigger_time_error", static_cast<uint64_t>(gr::uncertainty(edgeIdxOffset) * static_cast<float>(_period))}, //
                    {gr::tag::TRIGGER_OFFSET.shortKey(), relOffset},                                                             //
                    {gr::tag::CONTEXT.shortKey(), context}                                                                       //
                },
                edgePosUnsigned);

            // copy samples up to nPublish
            std::copy_n(inputSpan.begin(), edgePosUnsigned, outputSpan.begin());
            std::ignore = inputSpan.consume(edgePosUnsigned);
            outputSpan.publish(edgePosUnsigned);
            return gr::work::Status::OK;
        };

        for (std::size_t i = 0; i < nProcess; ++i) {
            const T sample = inputSpan[i];
            _now += _period;

            if (_trigger.processOne(sample) != NONE) { // edge detected
                const std::ptrdiff_t edgePosition = static_cast<std::ptrdiff_t>(i) + static_cast<std::ptrdiff_t>(gr::value(_trigger.lastEdgeIdx));

                if (_trigger.lastEdge == RISING && !trigger_name_rising_edge.value.empty()) {
                    return publishEdge(trigger_name_rising_edge, edgePosition);
                }

                if (_trigger.lastEdge == FALLING && !trigger_name_falling_edge.value.empty()) {
                    return publishEdge(trigger_name_falling_edge, edgePosition);
                }
            }
        }

        // no trigger found - just copy samples & tags
        std::copy_n(inputSpan.begin(), nProcess, outputSpan.begin());
        forwardTags(nProcess - 1UZ);
        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::basic

const inline auto registerTrigger = gr::registerBlock<gr::blocks::basic::SchmittTrigger, gr::trigger::InterpolationMethod::NO_INTERPOLATION, std::int16_t, std::int32_t, float, gr::UncertainValue<float>>(gr::globalBlockRegistry())             //
                                    + gr::registerBlock<gr::blocks::basic::SchmittTrigger, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION, std::int16_t, std::int32_t, float, gr::UncertainValue<float>>(gr::globalBlockRegistry()) //
                                    + gr::registerBlock<gr::blocks::basic::SchmittTrigger, gr::trigger::InterpolationMethod::LINEAR_INTERPOLATION, std::int16_t, std::int32_t, float, gr::UncertainValue<float>>(gr::globalBlockRegistry());

#endif // TRIGGER_HPP
