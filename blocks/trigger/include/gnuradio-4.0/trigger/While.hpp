#ifndef GNURADIO_TRIGGER_WHILE_HPP
#define GNURADIO_TRIGGER_WHILE_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/trigger/DigitalLine.hpp>
#include <gnuradio-4.0/trigger/EventReference.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::While, [T], [ int16_t, int32_t, float, double ])

template<typename T>
struct While : gr::Block<While<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief pass an event only if a line was at the required level at the event's trigger_time

    in      ──▁▁▁▔▔▔▔▔▔▁▁▁▁▔▔▔──▶   (no RxMarbles equivalent: a state, not a stream of items)
    evtIn   ─E────E──────E───E─▶
    evtOut  ──────E──────────E─▶    level = high
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::size_t kHistory = 64UZ; /// level changes retained
    constexpr static std::size_t kPending = 64UZ; /// events awaiting the line

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<T, "threshold", Doc<"the line's level">>                                                                                                       threshold{};
    A<T, "hysteresis", Doc<"full width of the band about threshold">, gr::Limits<T{}, std::numeric_limits<T>::max()>>                                hysteresis{};
    A<std::pmr::string, "level", Doc<"high|low: the level an event needs">>                                                                          level            = std::pmr::string("high");
    A<std::pmr::string, "reference", Doc<"decision|start: the time the line is read at, the start tuple of a delayed decision or its trigger_time">> reference        = std::pmr::string("decision");
    gr::SampleRate                                                                                                                                   sample_rate      = 0.f;
    mutable A<gr::Size_t, "n passed", Doc<"events the line qualified">>                                                                              n_passed         = 0U;
    mutable A<gr::Size_t, "n rejected", Doc<"events at the other level">>                                                                            n_rejected       = 0U;
    mutable A<gr::Size_t, "n unknown", Doc<"events whose level is not known: undated, before the history, unknown line">>                            n_unknown        = 0U;
    mutable A<gr::Size_t, "n gaps", Doc<"gaps in the line, each leaving its dropped samples unknown">>                                               n_gaps           = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events beyond the pending queue, or larger than an event slot even compacted">>                   n_events_dropped = 0U;

    EventReference         _reference      = EventReference::decision;
    gr::trigger::LineState _wanted         = gr::trigger::LineState::high;
    gr::Size_t             _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(While, evtIn, evtOut, in, out, threshold, hysteresis, level, reference, sample_rate, n_passed, n_rejected, n_unknown, n_gaps, n_events_dropped, _reference, _wanted, _settings_epoch);

    struct Change {
        std::uint64_t          at;
        gr::trigger::LineState state;
    };

    mutable DigitalLine<T>                          _line;
    mutable TimeBase                                _time;
    mutable std::array<Change, kHistory>            _changes{};
    mutable std::size_t                             _nChanges = 0UZ;
    mutable std::optional<std::uint64_t>            _historyFrom;  /// the history is complete from this time on
    mutable std::uint64_t                           _reached = 0U; /// the latest sample time the line has shown
    mutable std::array<detail::HeldEvent, kPending> _pending{};
    mutable std::size_t                             _nPending   = 0UZ;
    mutable gr::Size_t                              _stateEpoch = std::numeric_limits<gr::Size_t>::max();

    detail::ErrorMemory _deferredError{};

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _reference      = gr::meta::parseEnum<EventReference>(reference.value).value_or(EventReference::decision);
        _wanted         = gr::meta::parseEnum<gr::trigger::LineState>(level.value) == gr::trigger::LineState::low ? gr::trigger::LineState::low : gr::trigger::LineState::high;
        _settings_epoch = _settings_epoch + 1U;
    }

    void start() {
        detail::sendDeferredError(*this);
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            restartLine();
            _stateEpoch = _settings_epoch;
        }
        for (const gr::property_map_view& event : evtSpan) {
            if (!event.empty()) {
                hold<kOnHost>(event);
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        const std::size_t n    = std::min(inSpan.size(), outSpan.size());
        auto              tags = inSpan.rawTags();
        auto              next = std::ranges::begin(tags);
        for (std::size_t i = 0UZ; i < n; ++i) {
            const std::size_t at    = inSpan.streamIndex + i;
            const auto        first = detail::tagsOn(next, std::ranges::end(tags), at);
            bool              gap   = false;
            for (auto tag = first; tag != next; ++tag) {
                const gr::property_map_view tagMap{(*tag).map};
                std::ignore = _time.adopt(tagMap, at);
                gap         = gap || detail::isGap(tagMap);
                outSpan.publishTag(tagMap, i);
            }
            observe(inSpan[i], at, gap);
            outSpan[i] = inSpan[i];
        }
        outSpan.publish(n);
        if (!inSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }

        const std::size_t emitted = judgePending(evtOutSpan);
        evtOutSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    template<bool onHost>
    void hold(const gr::property_map_view& event) const {
        if (_nPending == kPending) {
            n_events_dropped = n_events_dropped + 1U;
            return;
        }
        if (!_pending[_nPending].template assign<onHost>(event)) {
            n_events_dropped = n_events_dropped + 1U;
            if constexpr (onHost) {
                detail::reportError(*this, "processBulk()", detail::kEventTooLarge);
            }
            return;
        }
        ++_nPending;
    }

    void record(std::uint64_t at, gr::trigger::LineState state) const {
        if (_nChanges == kHistory) { // the oldest change goes; what lay before it is no longer known
            _historyFrom = _changes[1].at;
            std::ranges::move(_changes.begin() + 1, _changes.end(), _changes.begin());
            --_nChanges;
        }
        _changes[_nChanges++] = Change{.at = at, .state = state};
    }

    void observe(T sample, std::size_t at, bool gap) const {
        if (gap) { // the dropped samples are unknown, and no edge is interpolated across them
            _line.configure({.level = threshold.value, .hysteresis = hysteresis.value});
            n_gaps = n_gaps + 1U;
            if (_historyFrom) {
                record(_reached + 1U, gr::trigger::LineState::unknown);
            }
        }
        const gr::trigger::LineState before = _line.state();
        _line.observe(sample);
        const auto stamp = _time.at(at);
        if (!stamp) {
            return;
        }
        _reached = *stamp;
        if (!_historyFrom) {
            _historyFrom = *stamp;
        }
        if (_line.state() == before) {
            return;
        }
        // an edge changed the level at its crossing; a settle or an invalid sample at the sample itself
        const double crossing = _line.changed ? static_cast<double>(gr::value(_line.comparator.lastEdgePosition)) * 1e9 / _time.rate() : 0.;
        record(static_cast<std::uint64_t>(static_cast<std::int64_t>(*stamp) + static_cast<std::int64_t>(std::round(crossing))), _line.state()); // not llround: a kernel has none
    }

    [[nodiscard]] std::optional<gr::trigger::LineState> stateAt(std::uint64_t at) const {
        if (!_historyFrom || at < *_historyFrom) {
            return std::nullopt;
        }
        gr::trigger::LineState found = gr::trigger::LineState::unknown;
        for (std::size_t i = 0UZ; i < _nChanges && _changes[i].at <= at; ++i) {
            found = _changes[i].state;
        }
        return found;
    }

    /** an event waits while the line has not shown its time, and while it would pass but the output has no room */
    [[nodiscard]] std::size_t judgePending(auto& evtOutSpan) const {
        std::size_t kept    = 0UZ;
        std::size_t emitted = 0UZ;
        for (std::size_t i = 0UZ; i < _nPending; ++i) {
            const gr::property_map_view event = _pending[i].view();
            const auto                  stamp = event.get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
            const auto                  at    = stamp ? std::optional{referenceTimeOf(event, *stamp, _reference)} : std::nullopt;
            const auto                  state = at && *at <= _reached ? stateAt(*at) : std::nullopt;
            if ((at && *at > _reached) || (state == _wanted && emitted >= evtOutSpan.size())) {
                if (kept != i) {
                    _pending[kept] = _pending[i];
                }
                ++kept;
                continue;
            }
            if (!state || *state == gr::trigger::LineState::unknown) {
                n_unknown = n_unknown + 1U;
            } else if (*state == _wanted) {
                if (gr::emit(evtOutSpan, emitted, event).has_value()) {
                    ++emitted;
                }
                n_passed = n_passed + 1U;
            } else {
                n_rejected = n_rejected + 1U;
            }
        }
        _nPending = kept;
        return emitted;
    }

    /** a new threshold, band or rate: the line's level history no longer applies */
    void restartLine() const {
        _line.configure({.level = threshold.value, .hysteresis = hysteresis.value});
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
        _nChanges    = 0UZ;
        _historyFrom = std::nullopt;
        _reached     = 0U;
        _nPending    = 0UZ;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_WHILE_HPP
