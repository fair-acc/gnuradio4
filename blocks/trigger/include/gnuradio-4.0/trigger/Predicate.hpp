#ifndef GNURADIO_TRIGGER_PREDICATE_HPP
#define GNURADIO_TRIGGER_PREDICATE_HPP

#include <limits>
#include <memory_resource>
#include <optional>
#include <string>
#include <type_traits>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/SamplePredicate.hpp>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK("gr::blocks::trigger::TakeWhile", gr::blocks::trigger::WhileHolds, ([T], true), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SkipWhile", gr::blocks::trigger::WhileHolds, ([T], false), [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK(gr::blocks::trigger::ElementAt, [T], [ int16_t, int32_t, float, double ])

namespace detail {

/** whether an event or, in a segmented block, a tag of this call starts a new segment; the events are consumed either way */
[[nodiscard]] bool restartsSegment(gr::trigger::BasicTriggerNameCtxMatcher::MatchState& starts, bool segmented, gr::InputSpanLike auto& evtSpan, const gr::InputSpanLike auto& inSpan) {
    bool again = false;
    for (const gr::property_map_view& event : evtSpan) {
        again = again || (segmented && !event.empty() && matches(starts, event));
    }
    std::ignore = evtSpan.consume(evtSpan.size());
    if (segmented) {
        for (const auto& tag : inSpan.rawTags()) {
            again = again || matches(starts, gr::property_map_view{tag.map});
        }
    }
    return again;
}

} // namespace detail

template<typename T, bool takes>
struct WhileHolds : gr::Block<WhileHolds<T, takes>> {
    using Description = std::conditional_t<takes, Doc<R"(@brief emit items while a test holds, then end the stream or the segment [takeWhile]

Rx's `takeWhile` completes its output at the first failure, which is the difference from a filter: everything after the failure
is gone, not merely the samples that fail.

 [1] example: https://rxmarbles.com/#takeWhile
 [2] detailed documentation: https://reactivex.io/documentation/operators/takewhile.html
)">,
        Doc<R"(@brief discard items while a test holds, then emit everything [skipWhile]

The asymmetry with `TakeWhile` is Rx's, and it is deliberate: `skipWhile` does not resume skipping if the test holds again
later.

 [1] example: https://rxmarbles.com/#skipWhile
 [2] detailed documentation: https://reactivex.io/documentation/operators/skipwhile.html
)">>;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    constexpr static std::string_view kName = takes ? "TakeWhile" : "SkipWhile";

    gr::EventPortIn           evtIn;
    gr::PortIn<T>             in;
    gr::PortOut<T, gr::Async> out;

    A<std::pmr::string, "predicate", Doc<"greater|greater_equal|less|less_equal|equal|not_equal">> predicate = std::pmr::string(takes ? "less" : "greater");
    A<T, "threshold", Doc<"what each sample is compared against">>                                 threshold = T{};
    A<std::pmr::string, "expression", Doc<"ExprTk in 'x' and 'threshold'; host only">>             expression;
    A<std::pmr::string, "segment filter", Doc<"starts a segment, empty = one per stream">>         segment_filter;

    mutable A<gr::Size_t, "n passed", Doc<"samples emitted; host path only">>                     n_passed         = 0U;
    mutable A<gr::Size_t, "n skipped", Doc<"samples discarded; host path only">>                  n_skipped        = 0U;
    mutable A<gr::Size_t, "n segments ended", Doc<"runs a failing sample ended; host path only">> n_segments_ended = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    Comparison _comparison = takes ? Comparison::less : Comparison::greater;
    MatchState _starts{};
    bool       _segmented      = false;
    gr::Size_t _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(WhileHolds, evtIn, in, out, predicate, threshold, expression, segment_filter, n_passed, n_skipped, n_segments_ended, _comparison, _starts, _segmented, _settings_epoch);

    mutable detail::SamplePredicate<T> _test;
    mutable bool                       _inRun = true;
    mutable MatchState                 _startsMatching{};
    mutable gr::Size_t                 _stateEpoch = std::numeric_limits<gr::Size_t>::max();

    void start() { _stateEpoch = std::numeric_limits<gr::Size_t>::max(); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if (const auto refused = _test.configure(predicate.value, expression.value, threshold.value)) {
            gr::log::warning("{}: {}", kName, *refused);
        }
        _comparison     = _test.comparison;
        _segmented      = detail::compileOptionalFilter(segment_filter.value, std::string_view{}, _starts, kName);
        _settings_epoch = _settings_epoch + 1U;
    }

    [[nodiscard]] std::string_view hostOnlyReason() const noexcept { return _test.hostOnlyReason(); }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _inRun          = true;
            _startsMatching = _starts;
            _stateEpoch     = _settings_epoch;
        }
        if (detail::restartsSegment(_startsMatching, _segmented, evtSpan, inSpan)) {
            _inRun = true;
        }

        std::size_t passed = 0UZ;
        std::size_t seen   = 0UZ;
        while (seen < inSpan.size() && passed < outSpan.size() && (_inRun || !takes)) {
            const bool holds = _inRun && detail::passes<kOnHost>(_test, _comparison, inSpan[seen], threshold.value);
            if (_inRun && !holds) {
                _inRun           = false;
                n_segments_ended = n_segments_ended + 1U;
                if constexpr (takes) {
                    break;
                }
            }
            if (!takes && holds) {
                ++seen;
                n_skipped = n_skipped + 1U;
                continue;
            }
            outSpan[passed++] = inSpan[seen++];
        }
        n_passed = n_passed + static_cast<gr::Size_t>(passed);

        const std::size_t consumed = takes && !_inRun ? inSpan.size() : seen;
        n_skipped                  = n_skipped + static_cast<gr::Size_t>(consumed - seen);
        outSpan.publish(passed);
        if (!inSpan.consume(consumed)) {
            return gr::work::Status::ERROR;
        }
        if (takes && !_inRun && !_segmented) {
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
using TakeWhile = WhileHolds<T, true>;

template<typename T>
using SkipWhile = WhileHolds<T, false>;

template<typename T>
struct ElementAt : gr::Block<ElementAt<T>> {
    using Description = Doc<R"(@brief emit item number n of the stream, or of every segment [elementAt]

    in   ─1──2──3──7─▶             elementAt(2)
    out  ───────3────│

`n` counts from zero, as Rx's `elementAt` does, so `n = 0` is the first sample and not a disabled setting.

 [1] example: https://rxmarbles.com/#elementAt
 [2] detailed documentation: https://reactivex.io/documentation/operators/elementat.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn           evtIn;
    gr::PortIn<T>             in;
    gr::PortOut<T, gr::Async> out;

    A<gr::Size_t, "n", Doc<"which sample, counting from zero">>                            n = 0U;
    A<std::pmr::string, "segment filter", Doc<"starts a segment, empty = one per stream">> segment_filter;

    mutable A<gr::Size_t, "n emitted", Doc<"host path only">>                                             n_emitted = 0U;
    mutable A<gr::Size_t, "n missed", Doc<"segments that ended before their nth sample; host path only">> n_missed  = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState _starts{};
    bool       _segmented      = false;
    gr::Size_t _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(ElementAt, evtIn, in, out, n, segment_filter, n_emitted, n_missed, _starts, _segmented, _settings_epoch);

    mutable std::size_t _position = 0UZ;
    mutable MatchState  _startsMatching{};
    mutable bool        _done       = false;
    mutable gr::Size_t  _stateEpoch = std::numeric_limits<gr::Size_t>::max();

    void start() { _stateEpoch = std::numeric_limits<gr::Size_t>::max(); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _segmented      = detail::compileOptionalFilter(segment_filter.value, std::string_view{}, _starts, "ElementAt");
        _settings_epoch = _settings_epoch + 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) const {
        if (_stateEpoch != _settings_epoch) {
            _position       = 0UZ;
            _done           = false;
            _startsMatching = _starts;
            _stateEpoch     = _settings_epoch;
        }
        bool restart = false;
        for (const gr::property_map_view& event : evtSpan) {
            restart = restart || (_segmented && !event.empty() && detail::matches(_startsMatching, event));
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        std::size_t emitted  = 0UZ;
        std::size_t consumed = 0UZ;
        for (const auto& tag : inSpan.rawTags()) {
            const std::size_t at = tag.index - inSpan.streamIndex;
            emitted += walk(inSpan, consumed, at, outSpan, emitted);
            consumed = at;
            restart  = restart || (_segmented && detail::matches(_startsMatching, gr::property_map_view{tag.map}));
            if (restart) {
                beginSegment();
                restart = false;
            }
        }
        if (restart) {
            beginSegment();
        }
        emitted += walk(inSpan, consumed, inSpan.size(), outSpan, emitted);

        outSpan.publish(emitted);
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        if (_done && !_segmented) {
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }

private:
    void beginSegment() const {
        if (!_done && _position <= n) {
            n_missed = n_missed + 1U;
        }
        _position = 0UZ;
        _done     = false;
    }

    [[nodiscard]] std::size_t walk(const gr::InputSpanLike auto& inSpan, std::size_t from, std::size_t until, gr::OutputSpanLike auto& outSpan, std::size_t emitted) const {
        std::size_t published = 0UZ;
        for (std::size_t i = from; i < until; ++i) {
            if (!_done && _position == n && emitted + published < outSpan.size()) {
                outSpan[emitted + published] = inSpan[i];
                ++published;
                _done     = true;
                n_emitted = n_emitted + 1U;
            }
            ++_position;
        }
        return published;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_PREDICATE_HPP
