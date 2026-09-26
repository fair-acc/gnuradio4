#ifndef GNURADIO_TRIGGER_TRIGGERWATCHDOG_HPP
#define GNURADIO_TRIGGER_TRIGGERWATCHDOG_HPP

#include <memory_resource>
#include <optional>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

GR_REGISTER_BLOCK(gr::blocks::trigger::TriggerWatchdog, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

namespace gr::blocks::trigger {

template<typename T>
struct TriggerWatchdog : gr::Block<TriggerWatchdog<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief report when triggers stop arriving and when they resume, and optionally act on it

    in      ─T───T─────────────T─▶   (Resilience4j's TimeLimiter in all but name)
    evtOut  ───────────timeout──recovered─▶

The absence of a trigger is the fault worth reporting, and nothing else in the family reports it.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "filter", Doc<"trigger filter the watchdog waits for">>            filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                        match_mode;
    A<gr::Size_t, "timeout samples", Doc<"samples without a match, 0 disables">>           timeout_samples    = 0U;
    A<float, "timeout seconds", Doc<"seconds without a match, <= 0 disables">>             timeout_seconds    = 0.f;
    A<float, "sample rate", Doc<"Hz, for seconds limit without anchors; 0 = unknown">>     sample_rate        = 0.f;
    A<std::pmr::string, "timeout action", Doc<"report|open|close">>                        timeout_action     = std::pmr::string("report");
    A<std::pmr::string, "open trigger name", Doc<"name of the event that opens a gate">>   open_trigger_name  = std::pmr::string("open");
    A<std::pmr::string, "close trigger name", Doc<"name of the event that closes a gate">> close_trigger_name = std::pmr::string("close");
    A<bool, "timed out", Doc<"reflected outage state">>                                    timed_out          = false;
    A<gr::Size_t, "n timeouts", Doc<"outages reported">>                                   n_timeouts         = 0U;
    A<gr::Size_t, "n invalid triggers", Doc<"undated triggers">>                           n_invalid_triggers = 0U;

    GR_MAKE_REFLECTABLE(TriggerWatchdog, evtOut, in, out, filter, match_mode, timeout_samples, timeout_seconds, sample_rate, timeout_action, open_trigger_name, close_trigger_name, timed_out, n_timeouts, n_invalid_triggers);

    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _matcher{};
    std::uint64_t                                       _sinceMatch  = 0UZ;
    std::size_t                                         _streamIndex = 0UZ;
    TimeBase                                            _time;
    bool                                                _unarmedReported = false;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        detail::compileFilterInto(filter.value, match_mode.value, _matcher, "TriggerWatchdog");
        _sinceMatch        = 0UZ;
        timed_out          = false;
        n_timeouts         = 0U;
        n_invalid_triggers = 0U;
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
        _unarmedReported = false;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples   = std::min(inSpan.size(), outSpan.size());
        const auto        conditions = detail::collectConditions(inSpan, nSamples);
        std::size_t       published  = 0UZ;

        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            if (const gr::property_map* tag = conditions.tagAt(i); tag != nullptr) {
                outSpan.publishTag(*tag, i);
                _time.adopt(gr::property_map_view{*tag}, _streamIndex + i);
                if (gr::trigger::BasicTriggerNameCtxMatcher::match(_matcher, gr::property_map_view{*tag}) == gr::trigger::MatchResult::Matching) {
                    published += detail::auditTrigger(gr::property_map_view{*tag}, n_invalid_triggers, evtOutSpan, published, this->name, this->unique_name);
                    if (timed_out) {
                        published += emit(evtOutSpan, published, "recovered", _streamIndex + i) ? 1UZ : 0UZ;
                        published += act(evtOutSpan, published, false, _streamIndex + i) ? 1UZ : 0UZ;
                        timed_out = false;
                    }
                    _sinceMatch = 0UZ;
                    continue;
                }
            }
            ++_sinceMatch;
            if (!timed_out && expired()) {
                timed_out = true;
                ++n_timeouts;
                published += emit(evtOutSpan, published, "timeout", _streamIndex + i) ? 1UZ : 0UZ;
                published += act(evtOutSpan, published, true, _streamIndex + i) ? 1UZ : 0UZ;
            }
        }

        published += reportUnarmedSeconds(evtOutSpan, published) ? 1UZ : 0UZ;

        std::ranges::copy(inSpan | std::views::take(nSamples), outSpan.begin());
        _streamIndex += nSamples;
        outSpan.publish(nSamples);
        evtOutSpan.publish(published);
        if (!inSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] bool reportUnarmedSeconds(auto& evtOutSpan, std::size_t index) {
        if (_unarmedReported || timeout_seconds <= 0.f || sample_rate > 0.f || _time.anchor.has_value()) {
            return false;
        }
        _unarmedReported = true;
        gr::log::warning("TriggerWatchdog: 'timeout_seconds' stays unarmed, neither a trigger_time anchor pair nor 'sample_rate' gives a rate");
        return gr::emitEvent(evtOutSpan, index, detail::makeErrorEvent("timeout_seconds needs a trigger_time anchor pair or a sample_rate", this->unique_name.value())).has_value();
    }

    [[nodiscard]] bool expired() const noexcept {
        const auto limit = resolveDuration(_time, timeout_samples, timeout_seconds);
        return limit && _sinceMatch >= *limit;
    }

    [[nodiscard]] bool act(auto& evtOutSpan, std::size_t index, bool onTimeout, std::size_t streamIndex) const {
        if (timeout_action.value == "report") {
            return false;
        }
        const bool closes = (timeout_action.value == "close") == onTimeout;
        return emit(evtOutSpan, index, closes ? close_trigger_name.value : open_trigger_name.value, streamIndex);
    }

    [[nodiscard]] bool emit(auto& evtOutSpan, std::size_t index, std::string_view state, std::size_t streamIndex) const {
        if (index >= evtOutSpan.size()) {
            return false;
        }
        gr::property_map event = detail::makeEvent(state, this->unique_name.value());
        if (const auto stamp = _time.at(streamIndex)) {
            event[std::string(gr::tag::TRIGGER_TIME.key())] = *stamp;
        }
        return gr::emitEvent(evtOutSpan, index, event).has_value();
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TRIGGERWATCHDOG_HPP
