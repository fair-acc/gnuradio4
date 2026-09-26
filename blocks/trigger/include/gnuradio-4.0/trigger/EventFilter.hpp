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
    using Description = Doc<R"(@brief forward the events a filter accepts, optionally under a new name [filter]

    evtIn   ─A──B──A──C─▶          filter
    evtOut  ─A─────A────▶          filter = "A"

`veto_filter` wins over `filter`: an event both accept is dropped, which is how a wanted event is excluded while the machine is
in a state that invalidates it. An empty `trigger_name` keeps each event's own.

 [1] example: https://rxmarbles.com/#filter
 [2] detailed documentation: https://reactivex.io/documentation/operators/filter.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::string, "filter", Doc<"filter the event must match, empty = all">> filter;
    A<std::pmr::string, "veto filter", Doc<"filter that overrides 'filter'">>      veto_filter;
    gr::TriggerName                                                                trigger_name;

    mutable A<gr::Size_t, "n passed", Doc<"host path only">>                                    n_passed         = 0U;
    mutable A<gr::Size_t, "n blocked", Doc<"events the filter did not accept; host path only">> n_blocked        = 0U;
    mutable A<gr::Size_t, "n vetoed", Doc<"host path only">>                                    n_vetoed         = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">>          n_events_dropped = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState _accept{};
    MatchState _veto{};
    bool       _filtered       = false;
    bool       _hasVeto        = false;
    gr::Size_t _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(EventFilter, evtIn, evtOut, filter, veto_filter, trigger_name, n_passed, n_blocked, n_vetoed, n_events_dropped, _accept, _veto, _filtered, _hasVeto, _settings_epoch);

    mutable MatchState _accepting{};
    mutable MatchState _vetoing{};
    mutable gr::Size_t _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable EventStore _events;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered       = detail::compileOptionalFilter(filter.value, std::string_view{}, _accept, "EventFilter");
        _hasVeto        = detail::compileOptionalFilter(veto_filter.value, std::string_view{}, _veto, "EventFilter");
        _settings_epoch = _settings_epoch + 1U;
        _events.clear();
    }

    void start() { _stateEpoch = std::numeric_limits<gr::Size_t>::max(); }

    [[nodiscard]] std::string_view hostOnlyReason() const noexcept { return trigger_name.value.empty() ? std::string_view{} : std::string_view{"renaming an event (trigger_name) copies its whole map, which runs on the host only"}; }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(evtSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _accepting  = _accept;
            _vetoing    = _veto;
            _stateEpoch = _settings_epoch;
        }
        std::size_t emitted = 0UZ;
        std::ignore         = forEachEvent<kOnHost>(_events, evtSpan, kOnHost ? std::numeric_limits<std::size_t>::max() : evtOutSpan.size(), [&](const gr::property_map_view& event) {
            if (emitted >= evtOutSpan.size()) {
                return false;
            }
            if (_hasVeto && detail::matches(_vetoing, event)) {
                n_vetoed = n_vetoed + 1U;
                return true;
            }
            if (_filtered && !detail::matches(_accepting, event)) {
                n_blocked = n_blocked + 1U;
                return true;
            }
            const bool passed = [&] {
                if constexpr (kOnHost) {
                    return (trigger_name.value.empty() ? gr::emit(evtOutSpan, emitted, event) : gr::emit(evtOutSpan, emitted, gr::property_map_view{renamed(event)})).has_value();
                } else {
                    return gr::emit(evtOutSpan, emitted, event).has_value();
                }
            }();
            if (passed) {
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

private:
    [[nodiscard]] gr::property_map renamed(const gr::property_map_view& event) const {
        gr::property_map copy{event};
        copy.insert_or_assign(std::string(gr::tag::TRIGGER_NAME.key()), trigger_name.value);
        return copy;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_EVENTFILTER_HPP
