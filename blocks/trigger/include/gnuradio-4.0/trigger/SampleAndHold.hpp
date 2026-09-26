#ifndef GNURADIO_TRIGGER_SAMPLEANDHOLD_HPP
#define GNURADIO_TRIGGER_SAMPLEANDHOLD_HPP

#include <memory_resource>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/ConditionSource.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>

GR_REGISTER_BLOCK(gr::blocks::trigger::SampleAndHold, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

namespace gr::blocks::trigger {

template<typename T>
struct SampleAndHold : gr::Block<SampleAndHold<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief holds the sample the filter last matched, one output per input

    in   ─a──b──T:c──d──e─▶        sample
    out  ─x──x──c────c──c─▶        x = initial_value

Rx `sample` emits the newest item when a notifier fires; this emits the newest *matched* item on every sample, which is the
zero-order hold an instrument wants.

 [1] example: https://rxmarbles.com/#sample
 [2] detailed documentation: https://reactivex.io/documentation/operators/sample.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    A<std::pmr::string, "filter", Doc<"trigger filter that captures a sample">>    filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                match_mode;
    A<T, "initial value", Doc<"emitted until the first match">>                    initial_value{};
    A<bool, "tag output", Doc<"republish the capture tag on the captured output">> tag_output         = true;
    A<gr::Size_t, "n captured">                                                    n_captured         = 0U;
    A<gr::Size_t, "n invalid triggers", Doc<"undated triggers">>                   n_invalid_triggers = 0U;

    GR_MAKE_REFLECTABLE(SampleAndHold, in, out, filter, match_mode, initial_value, tag_output, n_captured, n_invalid_triggers);

    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _matcher{};
    T                                                   _held{};

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        detail::compileFilterInto(filter.value, match_mode.value, _matcher, "SampleAndHold");
        _held              = initial_value;
        n_captured         = 0U;
        n_invalid_triggers = 0U;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples   = std::min(inSpan.size(), outSpan.size());
        const auto        conditions = detail::collectConditions(inSpan, nSamples);

        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            bool captured = false;
            if (const gr::property_map* tag = conditions.tagAt(i); tag != nullptr) {
                captured = gr::trigger::BasicTriggerNameCtxMatcher::match(_matcher, gr::property_map_view{*tag}) == gr::trigger::MatchResult::Matching;
                if (captured) {
                    _held = inSpan[i];
                    ++n_captured;
                    detail::auditTrigger(gr::property_map_view{*tag}, n_invalid_triggers, this->name);
                }
                if (captured ? tag_output.value : true) {
                    outSpan.publishTag(*tag, i);
                }
            }
            outSpan[i] = _held;
        }

        outSpan.publish(nSamples);
        if (!inSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_SAMPLEANDHOLD_HPP
