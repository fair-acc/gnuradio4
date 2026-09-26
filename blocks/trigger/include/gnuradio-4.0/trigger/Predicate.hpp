#ifndef GNURADIO_TRIGGER_PREDICATE_HPP
#define GNURADIO_TRIGGER_PREDICATE_HPP

#include <memory_resource>
#include <optional>
#include <string>
#include <type_traits>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/SamplePredicate.hpp>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::TakeWhile, [T], [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK(gr::blocks::trigger::SkipWhile, [T], [ int16_t, int32_t, float, double ])
GR_REGISTER_BLOCK(gr::blocks::trigger::ElementAt, [T], [ int16_t, int32_t, float, double ])

template<typename T>
struct TakeWhile : gr::Block<TakeWhile<T>> {
    using Description = Doc<R"(@brief emit items while a test holds, then end the stream or the segment [takeWhile]

Rx's `takeWhile` completes its output at the first failure, which is the difference from a filter: everything after the failure
is gone, not merely the samples that fail.

 [1] example: https://rxmarbles.com/#takeWhile
 [2] detailed documentation: https://reactivex.io/documentation/operators/takewhile.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn           evtIn;
    gr::PortIn<T>             in;
    gr::PortOut<T, gr::Async> out;

    A<std::pmr::string, "predicate", Doc<"greater|greater_equal|less|less_equal|equal|not_equal">> predicate = std::pmr::string("less");
    A<T, "threshold", Doc<"what each sample is compared against">>                                 threshold = T{};
    A<std::pmr::string, "expression", Doc<"ExprTk in 'x' and 'threshold'; host only">>             expression;
    A<std::pmr::string, "segment filter", Doc<"starts a segment, empty = one per stream">>         segment_filter;

    A<gr::Size_t, "n passed">                                                     n_passed         = 0U;
    A<gr::Size_t, "n segments ended", Doc<"segments a failing sample cut short">> n_segments_ended = 0U;

    GR_MAKE_REFLECTABLE(TakeWhile, evtIn, in, out, predicate, threshold, expression, segment_filter, n_passed, n_segments_ended);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    detail::SamplePredicate<T> _test;
    MatchState                 _starts{};
    bool                       _segmented = false;
    bool                       _taking    = true;

    void start() { _taking = true; }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if (const auto refused = _test.configure(predicate.value, expression.value, threshold.value)) {
            gr::log::warning("TakeWhile: {}", *refused);
        }
        _segmented = detail::compileOptionalFilter(segment_filter.value, std::string_view{}, _starts, "TakeWhile");
        _taking    = true;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        if (restarted(evtSpan, inSpan)) {
            _taking = true;
        }

        std::size_t taken = 0UZ;
        std::size_t seen  = 0UZ;
        while (seen < inSpan.size() && taken < outSpan.size() && _taking) {
            if (!_test(inSpan[seen])) {
                _taking          = false;
                n_segments_ended = n_segments_ended + 1U;
                break;
            }
            outSpan[taken++] = inSpan[seen++];
        }
        n_passed = n_passed + static_cast<gr::Size_t>(taken);

        outSpan.publish(taken);
        const std::size_t consumed = _taking ? seen : inSpan.size();
        if (!inSpan.consume(consumed)) {
            return gr::work::Status::ERROR;
        }
        if (!_taking && !_segmented) {
            this->requestStop();
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] bool restarted(gr::InputSpanLike auto& evtSpan, const gr::InputSpanLike auto& inSpan) {
        bool again = false;
        for (const gr::property_map_view& event : evtSpan) {
            again = again || (_segmented && !event.empty() && gr::trigger::BasicTriggerNameCtxMatcher::match(_starts, event) == gr::trigger::MatchResult::Matching);
        }
        std::ignore = evtSpan.consume(evtSpan.size());
        if (!_segmented) {
            return false;
        }
        for (const auto& tag : inSpan.rawTags()) {
            again = again || gr::trigger::BasicTriggerNameCtxMatcher::match(_starts, gr::property_map_view{tag.map}) == gr::trigger::MatchResult::Matching;
        }
        return again;
    }
};

template<typename T>
struct SkipWhile : gr::Block<SkipWhile<T>> {
    using Description = Doc<R"(@brief discard items while a test holds, then emit everything [skipWhile]

The asymmetry with `TakeWhile` is Rx's, and it is deliberate: `skipWhile` does not resume skipping if the test holds again
later.

 [1] example: https://rxmarbles.com/#skipWhile
 [2] detailed documentation: https://reactivex.io/documentation/operators/skipwhile.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn           evtIn;
    gr::PortIn<T>             in;
    gr::PortOut<T, gr::Async> out;

    A<std::pmr::string, "predicate", Doc<"greater|greater_equal|less|less_equal|equal|not_equal">> predicate = std::pmr::string("greater");
    A<T, "threshold", Doc<"what each sample is compared against">>                                 threshold = T{};
    A<std::pmr::string, "expression", Doc<"ExprTk in 'x' and 'threshold'; host only">>             expression;
    A<std::pmr::string, "segment filter", Doc<"starts a segment, empty = one per stream">>         segment_filter;

    A<gr::Size_t, "n skipped"> n_skipped = 0U;
    A<gr::Size_t, "n passed">  n_passed  = 0U;

    GR_MAKE_REFLECTABLE(SkipWhile, evtIn, in, out, predicate, threshold, expression, segment_filter, n_skipped, n_passed);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    detail::SamplePredicate<T> _test;
    MatchState                 _starts{};
    bool                       _segmented = false;
    bool                       _skipping  = true;

    void start() { _skipping = true; }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if (const auto refused = _test.configure(predicate.value, expression.value, threshold.value)) {
            gr::log::warning("SkipWhile: {}", *refused);
        }
        _segmented = detail::compileOptionalFilter(segment_filter.value, std::string_view{}, _starts, "SkipWhile");
        _skipping  = true;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (_segmented && !event.empty() && gr::trigger::BasicTriggerNameCtxMatcher::match(_starts, event) == gr::trigger::MatchResult::Matching) {
                _skipping = true;
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());
        if (_segmented) {
            for (const auto& tag : inSpan.rawTags()) {
                if (gr::trigger::BasicTriggerNameCtxMatcher::match(_starts, gr::property_map_view{tag.map}) == gr::trigger::MatchResult::Matching) {
                    _skipping = true;
                }
            }
        }

        std::size_t passed = 0UZ;
        std::size_t seen   = 0UZ;
        while (seen < inSpan.size() && passed < outSpan.size()) {
            if (_skipping && _test(inSpan[seen])) {
                ++seen;
                n_skipped = n_skipped + 1U;
                continue;
            }
            _skipping         = false;
            outSpan[passed++] = inSpan[seen++];
        }
        n_passed = n_passed + static_cast<gr::Size_t>(passed);

        outSpan.publish(passed);
        if (!inSpan.consume(seen)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

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

    A<gr::Size_t, "n emitted">                                                    n_emitted = 0U;
    A<gr::Size_t, "n missed", Doc<"segments that ended before their nth sample">> n_missed  = 0U;

    GR_MAKE_REFLECTABLE(ElementAt, evtIn, in, out, n, segment_filter, n_emitted, n_missed);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState  _starts{};
    bool        _segmented = false;
    std::size_t _position  = 0UZ;
    bool        _done      = false;

    void start() {
        _position = 0UZ;
        _done     = false;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _segmented = detail::compileOptionalFilter(segment_filter.value, std::string_view{}, _starts, "ElementAt");
        _position  = 0UZ;
        _done      = false;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        bool restart = false;
        for (const gr::property_map_view& event : evtSpan) {
            restart = restart || (_segmented && !event.empty() && gr::trigger::BasicTriggerNameCtxMatcher::match(_starts, event) == gr::trigger::MatchResult::Matching);
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        std::size_t emitted  = 0UZ;
        std::size_t consumed = 0UZ;
        for (const auto& tag : inSpan.rawTags()) {
            const std::size_t at = tag.index - inSpan.streamIndex;
            emitted += walk(inSpan, consumed, at, outSpan, emitted);
            consumed = at;
            if (_segmented && gr::trigger::BasicTriggerNameCtxMatcher::match(_starts, gr::property_map_view{tag.map}) == gr::trigger::MatchResult::Matching) {
                restart = true;
            }
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
            this->requestStop();
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }

private:
    void beginSegment() {
        if (!_done && _position <= n) {
            n_missed = n_missed + 1U;
        }
        _position = 0UZ;
        _done     = false;
    }

    [[nodiscard]] std::size_t walk(const gr::InputSpanLike auto& inSpan, std::size_t from, std::size_t until, gr::OutputSpanLike auto& outSpan, std::size_t emitted) {
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
