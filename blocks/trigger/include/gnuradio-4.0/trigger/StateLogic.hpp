#ifndef GNURADIO_TRIGGER_STATELOGIC_HPP
#define GNURADIO_TRIGGER_STATELOGIC_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>

namespace gr::blocks::trigger {

enum class Truth : std::uint8_t { no, yes, unknown };

enum class StateOperation : std::uint8_t { allOf, anyOf, notOf };

/** Kleene's three-valued logic: `allOf` is no when any is no, `anyOf` is yes when any is yes, `notOf` negates the first; an empty set is unknown */
[[nodiscard]] constexpr Truth combine(StateOperation operation, std::span<const Truth> states) noexcept {
    if (states.empty()) {
        return Truth::unknown;
    }
    const auto has = [&](Truth wanted) { return std::ranges::any_of(states, [wanted](Truth state) { return state == wanted; }); }; // not `contains`: a byte-sized find may become a memchr a kernel cannot link
    switch (operation) {
    case StateOperation::allOf:
        if (has(Truth::no)) {
            return Truth::no;
        }
        return has(Truth::unknown) ? Truth::unknown : Truth::yes;
    case StateOperation::anyOf:
        if (has(Truth::yes)) {
            return Truth::yes;
        }
        return has(Truth::unknown) ? Truth::unknown : Truth::no;
    case StateOperation::notOf:
        if (states.front() == Truth::unknown) {
            return Truth::unknown;
        }
        return states.front() == Truth::yes ? Truth::no : Truth::yes;
    }
    return Truth::unknown;
}

GR_REGISTER_BLOCK(gr::blocks::trigger::StateLogic)

struct StateLogic : gr::Block<StateLogic, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief combine states that events open and close with AllOf, AnyOf or Not, three-valued, and report when the combination holds

    evtIn   ─a_on──b_on──a_off──▶       (no RxMarbles equivalent: a state, not a stream of items)
    evtOut  ───────entry──exit──▶       operation = allOf, filters = ['[a_on, a_off]', '[b_on, b_off]']

False AND unknown is false, true OR unknown is true, otherwise unknown propagates.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::size_t kMaxStates = 8UZ;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::vector<std::pmr::string>, "filters", Doc<"one interval filter per state, '[enter, exit]': holds from enter to exit, unknown until the first of either; at most 8">> filters;
    A<std::pmr::string, "operation", Doc<"allOf|anyOf|notOf, which negates the first filter">>                                                                                      operation        = std::pmr::string("allOf");
    A<std::pmr::string, "entry name", Doc<"event name when the combination becomes true, dated by the event that made it so">>                                                      entry_name       = std::pmr::string("state_entry");
    A<std::pmr::string, "exit name", Doc<"event name when the combination stops being true (false or unknown), dated by the event that ended it">>                                  exit_name        = std::pmr::string("state_exit");
    mutable A<gr::Size_t, "n undated", Doc<"events with no trigger_time">>                                                                                                          n_undated        = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">>                                                                                              n_events_dropped = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    StateOperation                     _operation = StateOperation::allOf;
    std::array<MatchState, kMaxStates> _states{};
    gr::Size_t                         _nStates        = 0U;
    gr::Size_t                         _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(StateLogic, evtIn, evtOut, filters, operation, entry_name, exit_name, n_undated, n_events_dropped, _operation, _states, _nStates, _settings_epoch);

    mutable std::array<MatchState, kMaxStates> _matching{};
    mutable std::array<Truth, kMaxStates>      _truths{};
    mutable Truth                              _combined   = Truth::unknown;
    mutable gr::Size_t                         _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable EventStore                         _events;

    detail::ErrorMemory _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _operation = gr::meta::parseEnum<StateOperation>(operation.value).value_or(StateOperation::allOf);
        if (filters.value.size() > kMaxStates) {
            const std::string message = std::format("{} filters exceed the {} the block holds; the rest are ignored", filters.value.size(), kMaxStates);
            detail::reportError(*this, "settingsChanged()", message);
        }
        _nStates = 0U;
        for (const std::pmr::string& definition : filters.value | std::views::take(kMaxStates)) {
            if (auto compiled = detail::compileFilter(definition, std::string_view{"interval"}); compiled) {
                _states[_nStates] = *compiled;
                _nStates          = _nStates + 1U;
            } else {
                detail::reportError(*this, "settingsChanged()", std::format("{}", compiled.error().message));
            }
        }
        const auto dropped = _events.dropped; // a lifetime figure, which clear() would zero
        _events.clear();
        _events.dropped = dropped;
        _settings_epoch = _settings_epoch + 1U;
    }

    void reset() {
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
        _events.clear();
        n_undated        = 0U;
        n_events_dropped = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(evtSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _matching = _states;
            _truths.fill(Truth::unknown);
            _combined   = Truth::unknown;
            _stateEpoch = _settings_epoch;
        }
        std::size_t emitted = 0UZ;
        std::ignore         = forEachEvent<kOnHost>(_events, evtSpan, kOnHost ? std::numeric_limits<std::size_t>::max() : evtOutSpan.size(), [&](const gr::property_map_view& event) {
            if (emitted >= evtOutSpan.size()) {
                return false;
            }
            judge(event, evtOutSpan, emitted);
            return true;
        });
        if constexpr (kOnHost) {
            n_events_dropped = static_cast<gr::Size_t>(_events.dropped);
        }
        evtOutSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    void judge(const gr::property_map_view& event, auto& evtOutSpan, std::size_t& emitted) const {
        if (!event.template get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key())) {
            n_undated = n_undated + 1U;
            return;
        }
        for (std::size_t i = 0UZ; i < _nStates; ++i) {
            switch (gr::trigger::BasicTriggerNameCtxMatcher::match(_matching[i], event)) {
            case gr::trigger::MatchResult::Matching: _truths[i] = Truth::yes; break;
            case gr::trigger::MatchResult::NotMatching: _truths[i] = Truth::no; break;
            case gr::trigger::MatchResult::Ignore: break;
            }
        }
        const Truth before = _combined;
        _combined          = combine(_operation, std::span<const Truth>(_truths.data(), _nStates));
        const bool entered = before != Truth::yes && _combined == Truth::yes;
        const bool exited  = before == Truth::yes && _combined != Truth::yes;
        if (!entered && !exited) {
            return;
        }
        const std::string_view reported = entered ? std::string_view{entry_name.value} : std::string_view{exit_name.value};
        if (gr::emit(evtOutSpan, emitted, detail::kEventKeys, [&](gr::property_map_view& report) { return detail::writeEvent(report, reported, event); }).has_value()) {
            ++emitted;
        }
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_STATELOGIC_HPP
