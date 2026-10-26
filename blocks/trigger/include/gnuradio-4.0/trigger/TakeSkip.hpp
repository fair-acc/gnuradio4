#ifndef GNURADIO_TRIGGER_TAKESKIP_HPP
#define GNURADIO_TRIGGER_TAKESKIP_HPP

#include <algorithm>
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

GR_REGISTER_BLOCK("gr::blocks::trigger::TakeN", gr::blocks::trigger::Run, ([T], true), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SkipN", gr::blocks::trigger::Run, ([T], false), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

namespace gr::blocks::trigger {

template<typename T, bool forwardsTheRun>
struct Run : gr::Block<Run<T, forwardsTheRun>, gr::NoTagPropagation> {
    using Description = std::conditional_t<forwardsTheRun, Doc<R"(@brief emit n samples beginning with the one the filter matched [take, from a trigger]

    in   ─a──b──T:c──d──e─▶        take, started by a trigger
    out  ───────T:c──d────▶        n = 2
)">,
        Doc<R"(@brief discard n samples beginning with the one the filter matched [skip, from a trigger]

    in   ─a──b──T:c──d──e─▶        skip, started by a trigger
    out  ─a──b──────────e─▶        n = 2
)">>;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "filter", Doc<"trigger filter that starts a run">>                                                  filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                                                         match_mode;
    A<gr::Size_t, "n", Doc<"length of the run, in samples">>                                                                n             = 1U;
    A<std::pmr::string, "retrigger", Doc<"ignore|restart|extend">>                                                          retrigger     = std::pmr::string("restart");
    A<std::pmr::string, "policy", Doc<"drop|hold: what a closed run does with a sample, drop consumes it, hold leaves it">> policy        = std::pmr::string("drop");
    A<bool, "initial state", Doc<"true = a run is already in progress">>                                                    initial_state = false;

    mutable A<gr::Size_t, "n passed", Doc<"samples forwarded">>                           n_passed           = 0U;
    mutable A<gr::Size_t, "n suppressed", Doc<"samples consumed but not forwarded">>      n_suppressed       = 0U;
    mutable A<gr::Size_t, "n invalid triggers", Doc<"undated triggers">>                  n_invalid_triggers = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"reports a full event output refused">> n_events_dropped   = 0U;

    detail::MatchState _matcher{};
    Retrigger          _retrigger      = Retrigger::restart;
    gr::Size_t         _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Run, evtOut, in, out, filter, match_mode, n, retrigger, policy, initial_state, n_passed, n_suppressed, n_invalid_triggers, n_events_dropped, _matcher, _retrigger, _settings_epoch);

    mutable std::uint64_t      _remaining = 0U;
    mutable detail::MatchState _matching{};
    mutable gr::Size_t         _remainingEpoch = std::numeric_limits<gr::Size_t>::max();

    detail::ErrorMemory _deferredError{};

    void start() {
        _remainingEpoch = std::numeric_limits<gr::Size_t>::max();
        detail::sendDeferredError(*this);
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        detail::compileFilterInto(*this, filter.value, match_mode.value, _matcher);
        if (policy.value == "hold") {
            detail::reportError(*this, "settingsChanged()", "'policy' = 'hold' refused (no asynchronous input could release a held sample); keeping 'drop'");
            policy = std::pmr::string("drop");
        }
        _retrigger      = gr::meta::parseEnum<Retrigger>(retrigger.value).value_or(Retrigger::ignore);
        _settings_epoch = _settings_epoch + 1U;
    }

    void reset() {
        _remainingEpoch    = std::numeric_limits<gr::Size_t>::max();
        n_passed           = 0U;
        n_suppressed       = 0U;
        n_invalid_triggers = 0U;
        n_events_dropped   = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        if (_remainingEpoch != _settings_epoch) {
            _remaining      = initial_state ? static_cast<std::uint64_t>(n) : 0U;
            _matching       = _matcher;
            _remainingEpoch = _settings_epoch;
        }

        std::size_t nEmitted = 0UZ;
        const auto  report   = [&](auto writeInto) {
            if (!evtOutSpan.isConnected) {
                return;
            }
            if (gr::emit(evtOutSpan, nEmitted, detail::kEventKeys, writeInto)) {
                ++nEmitted;
            } else {
                n_events_dropped = n_events_dropped + 1U;
            }
        };

        const std::size_t                                                                 nSamples  = std::min(inSpan.size(), outSpan.size());
        auto                                                                              tags      = inSpan.rawTags();
        auto                                                                              next      = std::ranges::begin(tags);
        std::size_t                                                                       published = 0UZ;
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> scratch{};
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            const auto                  first     = detail::tagsOn(next, std::ranges::end(tags), inSpan.streamIndex + i);
            const gr::property_map_view condition = detail::mergedAt(first, next, scratch);
            if (first != next && detail::matches(_matching, condition)) {
                if (detail::isIncompleteTrigger(condition)) {
                    n_invalid_triggers = n_invalid_triggers + 1U;
                    if constexpr (kOnHost) {
                        detail::reportError(*this, "processBulk()", detail::kIncompleteTrigger);
                    }
                }
                if (_remaining == 0U || _retrigger == Retrigger::restart) {
                    _remaining = n;
                } else if (_retrigger == Retrigger::extend) {
                    _remaining += n;
                }
                report([&](gr::property_map_view& event) { return detail::writeEvent(event, forwardsTheRun ? "take" : "skip", condition); });
            }

            const bool insideRun = _remaining > 0U;
            if (insideRun) {
                --_remaining;
            }
            if (insideRun != forwardsTheRun) {
                n_suppressed = n_suppressed + 1U;
                continue;
            }
            for (auto tag = first; tag != next; ++tag) {
                outSpan.publishTag(gr::property_map_view{(*tag).map}, published);
            }
            outSpan[published++] = inSpan[i];
            n_passed             = n_passed + 1U;
        }

        outSpan.publish(published);
        evtOutSpan.publish(nEmitted);
        if (!inSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
using TakeN = Run<T, true>;

template<typename T>
using SkipN = Run<T, false>;

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TAKESKIP_HPP
