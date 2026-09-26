#ifndef GNURADIO_TRIGGER_SAMPLEANDHOLD_HPP
#define GNURADIO_TRIGGER_SAMPLEANDHOLD_HPP

#include <array>
#include <limits>
#include <memory_resource>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>

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

    A<std::pmr::string, "filter", Doc<"trigger filter that captures a sample">>                         filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                                     match_mode;
    A<T, "initial value", Doc<"emitted until the first match">>                                         initial_value{};
    A<bool, "tag output", Doc<"republish the capture tag on the captured output">>                      tag_output         = true;
    mutable A<gr::Size_t, "n captured", Doc<"host path only: a device body does not report it back">>   n_captured         = 0U;
    mutable A<gr::Size_t, "n invalid triggers", Doc<"undated triggers; host path only, as n_captured">> n_invalid_triggers = 0U;

    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _matcher{};
    gr::Size_t                                          _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(SampleAndHold, in, out, filter, match_mode, initial_value, tag_output, n_captured, n_invalid_triggers, _matcher, _settings_epoch);

    mutable T                  _held{};
    mutable detail::MatchState _matching{};
    mutable gr::Size_t         _heldEpoch = std::numeric_limits<gr::Size_t>::max();

    void start() { _heldEpoch = std::numeric_limits<gr::Size_t>::max(); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        detail::compileFilterInto(filter.value, match_mode.value, _matcher, "SampleAndHold");
        _settings_epoch    = _settings_epoch + 1U;
        n_captured         = 0U;
        n_invalid_triggers = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) const {
        if (_heldEpoch != _settings_epoch) {
            _held      = initial_value;
            _matching  = _matcher;
            _heldEpoch = _settings_epoch;
        }
        const std::size_t                                                                 nSamples = std::min(inSpan.size(), outSpan.size());
        auto                                                                              tags     = inSpan.rawTags();
        auto                                                                              next     = std::ranges::begin(tags);
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> scratch{};
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            const auto                  first     = detail::tagsOn(next, std::ranges::end(tags), inSpan.streamIndex + i);
            const gr::property_map_view condition = detail::mergedAt(first, next, scratch);
            const bool                  captured  = first != next && detail::matches(_matching, condition);
            if (captured) {
                _held      = inSpan[i];
                n_captured = n_captured + 1U;
                if (detail::isIncompleteTrigger(condition)) {
                    n_invalid_triggers = n_invalid_triggers + 1U;
                    if constexpr (!gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>) {
                        gr::log::warning("{}: {}", this->name.value, detail::kIncompleteTrigger);
                    }
                }
            }
            for (auto tag = first; tag != next && (!captured || tag_output); ++tag) {
                outSpan.publishTag(gr::property_map_view{(*tag).map}, i);
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
