#ifndef GNURADIO_TRIGGER_EVENTFILTER_HPP
#define GNURADIO_TRIGGER_EVENTFILTER_HPP

#include <memory_resource>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::EventFilter)

struct EventFilter : gr::Block<EventFilter, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief forward the events a filter accepts, optionally under a new name [filter]

    evtIn   ─A──B──A──C─▶          filter
    evtOut  ─A─────A────▶          filter = "A"

`veto_filter` wins over `filter`: an event both accept is dropped, which is how a wanted event is excluded while the machine is
in a state that invalidates it.

 [1] example: https://rxmarbles.com/#filter
 [2] detailed documentation: https://reactivex.io/documentation/operators/filter.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::string, "filter", Doc<"filter the event must match, empty = all">>  filter;
    A<std::pmr::string, "veto filter", Doc<"filter that overrides 'filter'">>       veto_filter;
    A<std::pmr::string, "trigger name", Doc<"rename what passes, empty = keep it">> trigger_name;

    A<gr::Size_t, "n passed">                                                  n_passed         = 0U;
    A<gr::Size_t, "n blocked", Doc<"events the filter did not accept">>        n_blocked        = 0U;
    A<gr::Size_t, "n vetoed">                                                  n_vetoed         = 0U;
    A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">> n_events_dropped = 0U;

    GR_MAKE_REFLECTABLE(EventFilter, evtIn, evtOut, filter, veto_filter, trigger_name, n_passed, n_blocked, n_vetoed, n_events_dropped);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState _accept{};
    MatchState _veto{};
    bool       _filtered = false;
    bool       _hasVeto  = false;
    EventStore _events;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered = detail::compileOptionalFilter(filter.value, std::string_view{}, _accept, "EventFilter");
        _hasVeto  = detail::compileOptionalFilter(veto_filter.value, std::string_view{}, _veto, "EventFilter");
        _events.clear();
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) {
        _events.drain(evtSpan);
        n_events_dropped = static_cast<gr::Size_t>(_events.dropped);

        std::size_t emitted  = 0UZ;
        std::size_t consumed = 0UZ;
        for (const StoredEvent& stored : _events.ordered()) {
            if (emitted >= evtOutSpan.size()) {
                break;
            }
            ++consumed;
            const gr::property_map_view event{stored.event};
            if (_hasVeto && detail::matches(_veto, event)) {
                n_vetoed = n_vetoed + 1U;
                continue;
            }
            if (_filtered && !detail::matches(_accept, event)) {
                n_blocked = n_blocked + 1U;
                continue;
            }
            if (gr::emitEvent(evtOutSpan, emitted, gr::property_map_view{renamed(stored.event)})) {
                ++emitted;
                n_passed = n_passed + 1U;
            }
        }
        _events.retire(consumed);

        evtOutSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] gr::property_map renamed(const gr::property_map& event) const {
        if (trigger_name.value.empty()) {
            return event;
        }
        gr::property_map copy = event;
        copy.insert_or_assign(std::string(gr::tag::TRIGGER_NAME.key()), trigger_name.value);
        return copy;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_EVENTFILTER_HPP
