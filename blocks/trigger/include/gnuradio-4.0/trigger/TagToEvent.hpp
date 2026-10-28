#ifndef GNURADIO_TRIGGER_TAGTOEVENT_HPP
#define GNURADIO_TRIGGER_TAGTOEVENT_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/Segments.hpp>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::TagToEvent, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct TagToEvent : gr::Block<TagToEvent<T>> {
    using Description = Doc<R"(@brief route the tags a trigger filter selects to `evtOut` as events, unchanged, leaving the stream untouched

    in      ─a──T:b──c──U:d─▶        (no RxMarbles equivalent: a change of carrier, not of content)
    out     ─a──T:b──c──U:d─▶
    evtOut  ─────T──────────▶        filter = "T"
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "filter", Doc<"trigger filter, e.g. 'start/ctx', '[start, ^stop]', 'beam_*', '!noise'; empty: every tag">>                  filter;
    A<double, "delay", Doc<"s, added to the trigger_time of each forwarded trigger, trigger_offset kept; 0 = as tagged">>                           delay              = 0.;
    A<gr::Size_t, "max pending events", Doc<"events held while evtOut is full">, gr::Limits<1U, static_cast<gr::Size_t>(detail::kMaxQueuedEvents)>> max_pending_events = 64U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events the queue lost, oldest first, or that a held slot could not take even compacted">>        n_events_dropped   = 0U;
    mutable A<gr::Size_t, "n invalid triggers", Doc<"undated triggers">>                                                                            n_invalid_triggers = 0U;

    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _matcher{};
    bool                                                _filtered       = false;
    gr::Size_t                                          _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(TagToEvent, evtOut, in, out, filter, delay, max_pending_events, n_events_dropped, n_invalid_triggers, _matcher, _filtered, _settings_epoch);

    constexpr static std::size_t kMaxPending = detail::kMaxQueuedEvents;

    mutable gr::trigger::BasicTriggerNameCtxMatcher::MatchState _matching{};
    mutable std::array<detail::HeldEvent, kMaxPending>          _pending{}; /// what `evtOut` had no room for, oldest at `_pendingFirst`
    mutable std::size_t                                         _pendingFirst = 0UZ;
    mutable std::size_t                                         _nPending     = 0UZ;
    mutable gr::Size_t                                          _stateEpoch   = std::numeric_limits<gr::Size_t>::max();

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("filter")) {
            std::ignore     = detail::compileOptionalFilter(*this, filter.value, std::string_view{}, _matcher);
            _filtered       = !filter.value.empty(); // a filter that does not compile selects nothing
            _settings_epoch = _settings_epoch + 1U;
        }
    }

    void stop() { // no later work call publishes what is still held: it is lost, and said so
        if (_nPending == 0UZ) {
            return;
        }
        n_events_dropped = n_events_dropped + static_cast<gr::Size_t>(_nPending);
        _nPending        = 0UZ;
        detail::reportError(*this, "stop()", std::format("{} event(s) dropped: still held when the block stopped", n_events_dropped.value));
    }

    void reset() {
        _stateEpoch        = std::numeric_limits<gr::Size_t>::max();
        n_events_dropped   = 0U;
        n_invalid_triggers = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        if (_stateEpoch != _settings_epoch) { // a new filter: what the old one selected is no longer wanted
            _matching   = _matcher;
            _nPending   = 0UZ;
            _stateEpoch = _settings_epoch;
        }
        const gr::Size_t droppedBefore = n_events_dropped.value;
        std::size_t      published     = 0UZ;
        for (; published < evtSpan.size() && _nPending > 0UZ; _pendingFirst = (_pendingFirst + 1UZ) % kMaxPending, --_nPending) {
            if (gr::emit(evtSpan, published, _pending[_pendingFirst].view()).has_value()) {
                ++published;
            } else {
                n_events_dropped = n_events_dropped + 1U;
            }
        }

        const std::size_t n = std::min(inSpan.size(), outSpan.size());
        for (const auto& tag : detail::tagsWithin(inSpan, n)) {
            const gr::property_map_view tagMap{tag.map};
            if (_filtered && gr::trigger::BasicTriggerNameCtxMatcher::match(_matching, tagMap) == gr::trigger::MatchResult::Ignore) {
                continue;
            }
            const bool isTrigger = _filtered || tagMap.contains(gr::tag::kTriggerName.key());
            if (isTrigger && detail::isIncompleteTrigger(tagMap)) {
                if constexpr (kOnHost) {
                    if (n_invalid_triggers.value == 0U) {
                        detail::reportError(*this, "processBulk()", detail::kIncompleteTrigger);
                    }
                }
                n_invalid_triggers = n_invalid_triggers + 1U;
            }
            if (_nPending == 0UZ && published < evtSpan.size() && gr::emit(evtSpan, published, tagMap).has_value()) {
                redate(evtSpan[published]);
                ++published;
            } else {
                hold<kOnHost>(tagMap);
            }
        }
        if constexpr (kOnHost) {
            if (n_events_dropped.value != droppedBefore) {
                detail::reportError(*this, "processBulk()", std::format("{} event(s) dropped: the queue of {} was full", n_events_dropped.value, max_pending_events.value));
            }
        }

        for (std::size_t i = 0UZ; i < n; ++i) {
            outSpan[i] = inSpan[i];
        }
        outSpan.publish(n);
        evtSpan.publish(published);
        std::ignore = inSpan.consume(n);
        return gr::work::Status::OK;
    }

private:
    /** `trigger_time` + `delay` in the forwarded copy, in place: the field keeps its type and size */
    void redate(gr::property_map_view forwarded) const {
        const auto stamp = forwarded.get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
        if (!stamp || delay.value == 0.) {
            return;
        }
        const auto delayNs = static_cast<std::int64_t>(std::round(delay.value * 1e9)); // not llround: a kernel has none
        std::ignore        = forwarded.insert_or_assign(gr::tag::kTriggerTime.key(), static_cast<std::uint64_t>(static_cast<std::int64_t>(*stamp) + delayNs));
    }

    template<bool onHost>
    void hold(const gr::property_map_view& tagMap) const {
        const std::size_t capacity = std::clamp(static_cast<std::size_t>(max_pending_events.value), 1UZ, kMaxPending);
        if (_nPending >= capacity) { // the oldest go, more than one when the capacity was lowered
            const std::size_t lost = _nPending - capacity + 1UZ;
            _pendingFirst          = (_pendingFirst + lost) % kMaxPending;
            _nPending              = _nPending - lost;
            n_events_dropped       = n_events_dropped + static_cast<gr::Size_t>(lost);
        }
        detail::HeldEvent& slot = _pending[(_pendingFirst + _nPending) % kMaxPending];
        if (!slot.template assign<onHost>(tagMap)) {
            n_events_dropped = n_events_dropped + 1U;
            return;
        }
        redate(slot.view());
        ++_nPending;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TAGTOEVENT_HPP
