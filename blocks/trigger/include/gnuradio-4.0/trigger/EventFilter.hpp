#ifndef GNURADIO_TRIGGER_EVENTFILTER_HPP
#define GNURADIO_TRIGGER_EVENTFILTER_HPP

#include <limits>
#include <memory_resource>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::EventFilter)

struct EventFilter : gr::Block<EventFilter, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief forward the events a trigger filter names [filter]

    evtIn   ─A──B──A──C─▶          filter
    evtOut  ─A─────A────▶          filter = "A"

 [1] example: https://rxmarbles.com/#filter
 [2] detailed documentation: https://reactivex.io/documentation/operators/filter.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::string, "filter", Doc<"trigger filter, e.g. 'edge', 'edge/!calibration', 'beam_*', '!noise'; an interval '[start, stop]' passes its start and its stop; empty: every event">> filter;

    mutable A<gr::Size_t, "n passed", Doc<"events forwarded">>                         n_passed         = 0U;
    mutable A<gr::Size_t, "n blocked", Doc<"events the filter did not accept">>        n_blocked        = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">> n_events_dropped = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState _accept{};
    bool       _filtered       = false;
    gr::Size_t _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(EventFilter, evtIn, evtOut, filter, n_passed, n_blocked, n_events_dropped, _accept, _filtered, _settings_epoch);

    mutable MatchState _accepting{};
    mutable gr::Size_t _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable EventStore _events;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered       = detail::compileOptionalFilter(*this, filter.value, std::string_view{}, _accept);
        _settings_epoch = _settings_epoch + 1U;
        _events.clear();
    }

    void start() { _stateEpoch = std::numeric_limits<gr::Size_t>::max(); }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(evtSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _accepting  = _accept;
            _stateEpoch = _settings_epoch;
        }
        std::size_t emitted = 0UZ;
        std::ignore         = forEachEvent<kOnHost>(_events, evtSpan, kOnHost ? std::numeric_limits<std::size_t>::max() : evtOutSpan.size(), [&](const gr::property_map_view& event) {
            if (emitted >= evtOutSpan.size()) {
                return false;
            }
            if (_filtered && gr::trigger::BasicTriggerNameCtxMatcher::match(_accepting, event) == gr::trigger::MatchResult::Ignore) {
                n_blocked = n_blocked + 1U;
                return true;
            }
            if (gr::emit(evtOutSpan, emitted, event).has_value()) {
                ++emitted;
                n_passed = n_passed + 1U;
            }
            return true;
        });
        if constexpr (kOnHost) {
            n_events_dropped = static_cast<gr::Size_t>(_events.dropped);
        }
        evtOutSpan.publish(emitted);
        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_EVENTFILTER_HPP
