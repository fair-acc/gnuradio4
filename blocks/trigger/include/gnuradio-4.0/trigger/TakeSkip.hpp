#ifndef GNURADIO_TRIGGER_TAKESKIP_HPP
#define GNURADIO_TRIGGER_TAKESKIP_HPP

#include <map>
#include <memory_resource>
#include <optional>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/ConditionSource.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

GR_REGISTER_BLOCK("gr::blocks::trigger::TakeN", gr::blocks::trigger::Run, ([T], true), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SkipN", gr::blocks::trigger::Run, ([T], false), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

namespace gr::blocks::trigger {

namespace detail {

inline void refuseHold(auto& policy, std::string_view blockName) {
    if (policy.value == "hold") {
        gr::log::warning("{}: 'policy' = 'hold' refused (no asynchronous input could release a held sample); keeping 'drop'", blockName);
        policy = std::pmr::string("drop");
    }
}

template<bool forwardsTheRun>
[[nodiscard]] gr::work::Status walkRun(auto& self, auto& inSpan, auto& evtOutSpan, auto& outSpan) {
    const std::size_t nSamples   = std::min(inSpan.size(), outSpan.size());
    const auto        conditions = collectConditions(inSpan, nSamples);

    std::size_t emitted   = 0UZ;
    std::size_t published = 0UZ;
    for (std::size_t i = 0UZ; i < nSamples; ++i) {
        const gr::property_map* tag = conditions.tagAt(i);
        if (tag != nullptr) {
            self._time.adopt(gr::property_map_view{*tag}, self._streamIndex + i);
        }
        if (tag != nullptr && gr::trigger::BasicTriggerNameCtxMatcher::match(self._matcher, gr::property_map_view{*tag}) == gr::trigger::MatchResult::Matching) {
            published += auditTrigger(gr::property_map_view{*tag}, self.n_invalid_triggers, evtOutSpan, published, self.name, self.unique_name);
            self.startRun();
            published += self.reportRunStart(evtOutSpan, published, gr::property_map_view{*tag}) ? 1UZ : 0UZ;
        }

        const bool insideRun = self._remaining > 0UZ;
        if (insideRun) {
            --self._remaining;
        }
        if (insideRun == forwardsTheRun) {
            if (tag != nullptr) {
                outSpan.publishTag(*tag, emitted);
            }
            outSpan[emitted++] = inSpan[i];
            ++self.n_passed;
        } else {
            ++self.n_suppressed;
        }
    }

    self._streamIndex += nSamples;
    outSpan.publish(emitted);
    evtOutSpan.publish(published);
    if (!inSpan.consume(nSamples)) {
        return gr::work::Status::ERROR;
    }
    return gr::work::Status::OK;
}

} // namespace detail

template<typename T, bool forwardsTheRun>
struct Run : gr::Block<Run<T, forwardsTheRun>, gr::NoTagPropagation> {
    using Description = std::conditional_t<forwardsTheRun, Doc<R"(@brief emit n samples beginning with the one the filter matched [take, from a trigger]

    in   ─a──b──T:c──d──e─▶        take, started by a trigger
    out  ───────T:c──d────▶        n = 2

Rx `take(n)` counts from the start of the stream; this counts from a trigger, which is the superset. `policy`
decides what a closed run does with a sample: `drop` consumes it, `hold` leaves it.

@code
auto& firstTwo = graph.emplaceBlock<TakeN<float>>({{"filter", "start"}, {"n", 2U}});
@endcode
)">,
        Doc<R"(@brief discard n samples beginning with the one the filter matched [skip, from a trigger]

    in   ─a──b──T:c──d──e─▶        skip, started by a trigger
    out  ─a──b──────────e─▶        n = 2

The mirror of `TakeN`, and the same relation to Rx `skip(n)`.

@code
auto& without = graph.emplaceBlock<SkipN<float>>({{"filter", "start"}, {"n", 2U}});
@endcode
)">>;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::string_view kName = forwardsTheRun ? "TakeN" : "SkipN";

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "filter", Doc<"trigger filter that starts a run">>   filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>          match_mode;
    A<gr::Size_t, "n", Doc<"length of the run, in samples">>                 n                  = 1U;
    A<std::pmr::string, "retrigger", Doc<"ignore|restart|extend">>           retrigger          = std::pmr::string("restart");
    A<std::pmr::string, "policy", Doc<"drop|hold">>                          policy             = std::pmr::string("drop");
    A<bool, "initial state", Doc<"true = a run is already in progress">>     initial_state      = false;
    A<gr::Size_t, "n passed", Doc<"samples forwarded">>                      n_passed           = 0U;
    A<gr::Size_t, "n suppressed", Doc<"samples consumed but not forwarded">> n_suppressed       = 0U;
    A<gr::Size_t, "n invalid triggers", Doc<"undated triggers">>             n_invalid_triggers = 0U;

    GR_MAKE_REFLECTABLE(Run, evtOut, in, out, filter, match_mode, n, retrigger, policy, initial_state, n_passed, n_suppressed, n_invalid_triggers);

    detail::MatchState _matcher{};
    std::uint64_t      _remaining   = 0UZ;
    std::size_t        _streamIndex = 0UZ;
    TimeBase           _time;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _time.reset();
        detail::compileFilterInto(filter.value, match_mode.value, _matcher, kName);
        detail::refuseHold(policy, kName);
        _remaining         = initial_state ? static_cast<std::uint64_t>(n) : 0UZ;
        n_passed           = 0U;
        n_suppressed       = 0U;
        n_invalid_triggers = 0U;
    }

    [[nodiscard]] bool reportRunStart(auto& evtOutSpan, std::size_t index, const gr::property_map_view& from) const {
        if (index >= evtOutSpan.size()) {
            return false;
        }
        gr::property_map event = detail::makeEvent(forwardsTheRun ? "take" : "skip", this->unique_name.value(), from);
        return gr::emitEvent(evtOutSpan, index, event).has_value();
    }

    void startRun() noexcept {
        if (_remaining == 0UZ || retrigger.value == "restart") {
            _remaining = n;
        } else if (retrigger.value == "extend") {
            _remaining += n;
        }
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) { return detail::walkRun<forwardsTheRun>(*this, inSpan, evtOutSpan, outSpan); }
};

template<typename T>
using TakeN = Run<T, true>;

template<typename T>
using SkipN = Run<T, false>;

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TAKESKIP_HPP
