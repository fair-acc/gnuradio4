#ifndef GNURADIO_TRIGGER_EVENTSHAPING_HPP
#define GNURADIO_TRIGGER_EVENTSHAPING_HPP

#include <algorithm>
#include <cstdint>
#include <deque>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/Segments.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <memory_resource>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::Debounce, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct Debounce : gr::Block<Debounce<T>> {
    using Description = Doc<R"(@brief report one trigger per burst, the last one, after the burst has gone quiet [debounce]

    in      ─T─T─T─────────T──▶    debounce
    evtOut  ───────T─────────T─▶   quiet = 3 samples

A bouncing switch, a discriminator on a noisy baseline, an operator's finger.

 [1] example: https://rxmarbles.com/#debounce
 [2] detailed documentation: https://reactivex.io/documentation/operators/debounce.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::PortIn<T>    in;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortOut<T>   out;

    A<std::pmr::string, "filter", Doc<"filter naming what is debounced, empty = all">> filter;
    A<gr::Size_t, "n samples", Doc<"quiet period, samples">>                           n_samples   = 0U;
    A<float, "timeout", Doc<"quiet period, seconds">>                                  timeout     = 0.f;
    A<float, "sample rate", Doc<"Hz, converts 'timeout' into samples, 0 = unknown">>   sample_rate = 0.f;

    A<gr::Size_t, "n items", Doc<"triggers seen">>                                        n_items      = 0U;
    A<gr::Size_t, "n emitted", Doc<"bursts reported, one event each">>                    n_emitted    = 0U;
    A<gr::Size_t, "n suppressed", Doc<"triggers a later one of the same burst replaced">> n_suppressed = 0U;

    GR_MAKE_REFLECTABLE(Debounce, evtIn, in, evtOut, out, filter, n_samples, timeout, sample_rate, n_items, n_emitted, n_suppressed);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState                      _accept{};
    bool                            _filtered = false;
    TimeBase                        _time;
    std::size_t                     _streamIndex = 0UZ;
    std::optional<gr::property_map> _pendingItem;
    std::size_t                     _pendingAt = 0UZ;
    detail::PendingEvents           _reports;
    std::vector<detail::TriggerAt>  _items;

    void start() {
        _streamIndex = 0UZ;
        _pendingItem.reset();
        _reports.clear();
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples = std::min(inSpan.size(), outSpan.size());

        detail::collectTriggers(
            evtSpan, inSpan, _streamIndex, _items, //
            [this](const gr::property_map_view& candidate) { return detail::acceptsTrigger(_accept, _filtered, candidate); }, [this](const gr::property_map_view& carried, std::size_t at) { _time.adopt(carried, at); });
        for (auto& [at, item] : _items) {
            releaseBefore(at);
            hold(std::move(item), at);
        }

        std::ranges::copy(inSpan | std::views::take(nSamples), outSpan.begin());
        _streamIndex += nSamples;
        releaseBefore(_streamIndex);

        const std::size_t emitted = _reports.drainInto(evtOutSpan, 0UZ, "Debounce", this->unique_name.value());
        evtOutSpan.publish(emitted);
        outSpan.publish(nSamples);
        if (!inSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered = detail::compileOptionalFilter(filter.value, std::string_view{}, _accept, "Debounce");
        _time.setRate(static_cast<double>(sample_rate));
    }

private:
    void hold(gr::property_map&& item, std::size_t at) {
        if (_pendingItem.has_value()) {
            n_suppressed = n_suppressed + 1U;
        }
        n_items      = n_items + 1U;
        _pendingItem = std::move(item);
        _pendingAt   = at;
    }

    void releaseBefore(std::size_t position) {
        if (!_pendingItem.has_value()) {
            return;
        }
        const auto quiet = resolveDuration(_time, n_samples, timeout);
        if (!quiet.has_value()) {
            emit();
            return;
        }
        if (position >= _pendingAt + *quiet) {
            emit();
        }
    }

    void emit() {
        _reports.push(std::move(_pendingItem.value()));
        _pendingItem.reset();
        n_emitted = n_emitted + 1U;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::DelayWhen, [T], [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])

template<typename T>
struct DelayWhen : gr::Block<DelayWhen<T>> {
    using Description = Doc<R"(@brief delay each item by its own delay, from a parallel input, and emit them as they fall due [delayWhen]

    in        ─1──2──3─▶           delayWhen
    delay_in  ─4──2──0─▶           (samples)
    out       ─3──2──1─▶           due at 2, 3, 4

Rx computes the delay from the item with a lambda; GR4 has none that also runs on a device, so the number arrives on a parallel
input -- a cable length, a time of flight, a correction curve computed upstream.

 [1] example: https://rxmarbles.com/#delayWhen
 [2] detailed documentation: https://reactivex.io/documentation/operators/delaywhen.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>             in;
    gr::PortIn<gr::Size_t>    delay_in;
    gr::EventPortOut          evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortOut<T, gr::Async> out;

    A<gr::Size_t, "max pending", Doc<"samples held pending">> max_pending = 1024U;

    A<gr::Size_t, "n delayed", Doc<"samples that waited">>                              n_delayed  = 0U;
    A<gr::Size_t, "n overflow", Doc<"samples forwarded in arrival order, buffer full">> n_overflow = 0U;
    A<gr::Size_t, "n pending", Doc<"samples waiting to come due">>                      n_pending  = 0U;

    GR_MAKE_REFLECTABLE(DelayWhen, in, delay_in, evtOut, out, max_pending, n_delayed, n_overflow, n_pending);

    struct Waiting {
        T           sample;
        std::size_t dueAt;
    };

    std::deque<Waiting>   _waiting;
    std::size_t           _streamIndex = 0UZ;
    bool                  _full        = false;
    detail::PendingEvents _reports;

    void start() {
        _waiting.clear();
        _streamIndex = 0UZ;
        _full        = false;
        _reports.clear();
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& delaySpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples = std::min(inSpan.size(), delaySpan.size());
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            hold(inSpan[i], static_cast<std::size_t>(delaySpan[i]));
            ++_streamIndex;
        }
        n_pending = static_cast<gr::Size_t>(_waiting.size());

        std::size_t emitted = 0UZ;
        while (emitted < outSpan.size() && !_waiting.empty() && _waiting.front().dueAt <= _streamIndex) {
            outSpan[emitted++] = _waiting.front().sample;
            _waiting.pop_front();
        }

        const std::size_t reported = _reports.drainInto(evtOutSpan, 0UZ, "DelayWhen", this->unique_name.value());
        evtOutSpan.publish(reported);
        outSpan.publish(emitted);
        if (!inSpan.consume(nSamples) || !delaySpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void hold(const T& sample, std::size_t delay) {
        if (_waiting.size() >= static_cast<std::size_t>(max_pending.value)) {
            if (!_full) {
                _full = true;
                _reports.push(detail::makeErrorEvent(std::format("{} samples are already waiting; the order is arrival order from here", max_pending.value), this->unique_name.value()));
            }
            _waiting.push_back(Waiting{.sample = sample, .dueAt = _streamIndex}); // due at once: no claim about order
            n_overflow = n_overflow + 1U;
            return;
        }
        _full                   = false;
        const std::size_t dueAt = _streamIndex + delay;
        const auto        after = std::ranges::upper_bound(_waiting, dueAt, {}, [](const Waiting& held) { return held.dueAt; });
        _waiting.insert(after, Waiting{.sample = sample, .dueAt = dueAt});
        if (delay > 0UZ) {
            n_delayed = n_delayed + 1U;
        }
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::Distinct, [T], [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])
GR_REGISTER_BLOCK(gr::blocks::trigger::DistinctUntilChanged, [T], [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])

template<typename T>
struct Distinct : gr::Block<Distinct<T>> {
    using Description = Doc<R"(@brief emit each value the first time it is seen, within a bounded memory [distinct]

    in   ─1──2──1──3──2──4─▶       distinct
    out  ─1──2─────3─────4─▶       max_values bounds what is remembered

Every value ever seen is remembered, up to `max_values`; past that the block stops claiming to know and forwards everything,
counted as `n_overflow`.

 [1] example: https://rxmarbles.com/#distinct
 [2] detailed documentation: https://reactivex.io/documentation/operators/distinct.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn           evtIn;
    gr::PortIn<T>             in;
    gr::EventPortOut          evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortOut<T, gr::Async> out;

    A<gr::Size_t, "max values", Doc<"values remembered">>                                   max_values = 1024U;
    A<std::pmr::string, "segment filter", Doc<"clears memory on a trigger, empty = never">> segment_filter;

    A<gr::Size_t, "n passed", Doc<"values seen for the first time">>                  n_passed     = 0U;
    A<gr::Size_t, "n suppressed", Doc<"repeats of a value already seen">>             n_suppressed = 0U;
    A<gr::Size_t, "n overflow", Doc<"samples forwarded because the memory was full">> n_overflow   = 0U;

    GR_MAKE_REFLECTABLE(Distinct, evtIn, in, evtOut, out, max_values, segment_filter, n_passed, n_suppressed, n_overflow);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    std::unordered_set<T> _seen;
    MatchState            _clears{};
    bool                  _segmented = false;
    bool                  _full      = false;
    detail::PendingEvents _reports;

    void start() { forget(); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _segmented = detail::compileOptionalFilter(segment_filter.value, std::string_view{}, _clears, "Distinct");
        forget();
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (_segmented && !event.empty() && gr::trigger::BasicTriggerNameCtxMatcher::match(_clears, event) == gr::trigger::MatchResult::Matching) {
                forget();
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        std::size_t passed = 0UZ;
        detail::forEachSegment(
            inSpan, inSpan.size(), [&](std::size_t from, std::size_t until) { passed += processRun(inSpan, from, until, outSpan, passed); },
            [&](const gr::property_map_view& tag, std::size_t) {
                if (_segmented && detail::matches(_clears, tag)) {
                    forget();
                }
            });

        const std::size_t reported = _reports.drainInto(evtOutSpan, 0UZ, "Distinct", this->unique_name.value());
        evtOutSpan.publish(reported);
        outSpan.publish(passed);
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void forget() {
        _seen.clear();
        _seen.reserve(max_values == 0U ? 1UZ : static_cast<std::size_t>(max_values.value)); // reserved once, so nothing rehashes mid-stream
        _full = false;
    }

    [[nodiscard]] std::size_t processRun(const gr::InputSpanLike auto& inSpan, std::size_t from, std::size_t until, gr::OutputSpanLike auto& outSpan, std::size_t written) {
        std::size_t published = 0UZ;
        for (std::size_t i = from; i < until && written + published < outSpan.size(); ++i) {
            if (!admits(inSpan[i])) {
                continue;
            }
            outSpan[written + published] = inSpan[i];
            ++published;
        }
        return published;
    }

    [[nodiscard]] bool admits(const T& value) {
        if (_full) {
            n_overflow = n_overflow + 1U;
            return true;
        }
        if (_seen.contains(value)) {
            n_suppressed = n_suppressed + 1U;
            return false;
        }
        if (_seen.size() >= static_cast<std::size_t>(max_values.value)) {
            _full = true;
            _reports.push(detail::makeErrorEvent(std::format("remembered {} values and can take no more; everything is forwarded from here", max_values.value), this->unique_name.value()));
            n_overflow = n_overflow + 1U;
            return true;
        }
        _seen.insert(value);
        n_passed = n_passed + 1U;
        return true;
    }
};

template<typename T>
struct DistinctUntilChanged : gr::Block<DistinctUntilChanged<T>> {
    using Description = Doc<R"(@brief emit an item only where it differs from the one before it [distinctUntilChanged]

    in   ─1──1──2──2──1──3─▶       distinctUntilChanged
    out  ─1─────2─────1──3─▶

This is the one that needs no memory at all -- the previous value is the whole state -- and it is what is usually wanted on a
measured signal: a state code sampled at 1 MS/s becomes one sample per change.

 [1] example: https://rxmarbles.com/#distinctUntilChanged
 [2] detailed documentation: https://reactivex.io/documentation/operators/distinctuntilchanged.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>             in;
    gr::PortOut<T, gr::Async> out;

    A<gr::Size_t, "n passed">     n_passed     = 0U;
    A<gr::Size_t, "n suppressed"> n_suppressed = 0U;

    GR_MAKE_REFLECTABLE(DistinctUntilChanged, in, out, n_passed, n_suppressed);

    T    _previous = T{};
    bool _started  = false;

    void start() { _started = false; }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        std::size_t passed = 0UZ;
        std::size_t seen   = 0UZ;
        while (seen < inSpan.size() && passed < outSpan.size()) {
            const T& sample = inSpan[seen++];
            if (_started && sample == _previous) {
                n_suppressed = n_suppressed + 1U;
                continue;
            }
            _previous         = sample;
            _started          = true;
            outSpan[passed++] = sample;
            n_passed          = n_passed + 1U;
        }
        outSpan.publish(passed);
        if (!inSpan.consume(seen)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::Repeat, [T], [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])

template<typename T>
struct Repeat : gr::Block<Repeat<T>> {
    using Description = Doc<R"(@brief emit a captured segment again, as many times as asked [repeat, per segment]

    in   ─T:1──2──T:3─▶            repeat(3)
    out  ─1──2──1──2──1──2─▶       segment = what lies between two triggers

Rx re-subscribes a completed source; a GR4 block cannot, and end of stream is not observable from inside one (§10 of the
family's review).

 [1] example: https://rxmarbles.com/#repeat
 [2] detailed documentation: https://reactivex.io/documentation/operators/repeat.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn           evtIn;
    gr::PortIn<T>             in;
    gr::EventPortOut          evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortOut<T, gr::Async> out;

    A<gr::Size_t, "n repeats", Doc<"plays per segment">>                                n_repeats = 1U;
    A<gr::Size_t, "capacity", Doc<"replay store, samples">>                             capacity  = 4096U;
    A<std::pmr::string, "segment filter", Doc<"ends the segment, empty = by capacity">> segment_filter;

    A<gr::Size_t, "n segments">                                                               n_segments = 0U;
    A<gr::Size_t, "n overflow", Doc<"segments passed through once because they did not fit">> n_overflow = 0U;
    A<gr::Size_t, "n captured", Doc<"samples in the segment being captured">>                 n_captured = 0U;

    GR_MAKE_REFLECTABLE(Repeat, evtIn, in, evtOut, out, n_repeats, capacity, segment_filter, n_segments, n_overflow, n_captured);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState            _ends{};
    bool                  _segmented = false;
    std::vector<T>        _store;
    bool                  _tooLong = false;
    std::vector<T>        _playing;
    std::size_t           _position  = 0UZ;
    gr::Size_t            _playsLeft = 0U;
    detail::PendingEvents _reports;

    void start() {
        _store.clear();
        _store.reserve(capacity == 0U ? 1UZ : static_cast<std::size_t>(capacity.value));
        _playing.clear();
        _position  = 0UZ;
        _playsLeft = 0U;
        _tooLong   = false;
        _reports.clear();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _segmented = detail::compileOptionalFilter(segment_filter.value, std::string_view{}, _ends, "Repeat");
        start();
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (_segmented && !event.empty() && gr::trigger::BasicTriggerNameCtxMatcher::match(_ends, event) == gr::trigger::MatchResult::Matching) {
                close();
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        detail::forEachSegment(
            inSpan, inSpan.size(), [&](std::size_t from, std::size_t until) { processRun(inSpan, from, until); },
            [&](const gr::property_map_view& tag, std::size_t) {
                if (_segmented && detail::matches(_ends, tag)) {
                    close();
                }
            });
        n_captured = static_cast<gr::Size_t>(_store.size());

        std::size_t emitted = 0UZ;
        while (emitted < outSpan.size() && _playsLeft > 0U) {
            if (_position >= _playing.size()) {
                --_playsLeft;
                _position = 0UZ;
                if (_playing.empty()) {
                    break;
                }
                continue;
            }
            outSpan[emitted++] = _playing[_position++];
        }

        const std::size_t reported = _reports.drainInto(evtOutSpan, 0UZ, "Repeat", this->unique_name.value());
        evtOutSpan.publish(reported);
        outSpan.publish(emitted);
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void processRun(const gr::InputSpanLike auto& inSpan, std::size_t from, std::size_t until) {
        const std::size_t room = capacity == 0U ? 1UZ : static_cast<std::size_t>(capacity.value);
        for (std::size_t i = from; i < until; ++i) {
            if (!_tooLong && _store.size() >= room) {
                if (!_segmented) {
                    close();
                } else {
                    overgrown();
                }
            }
            if (_tooLong) {
                _playing.push_back(inSpan[i]);
                _playsLeft = _playsLeft == 0U ? 1U : _playsLeft;
                continue;
            }
            _store.push_back(inSpan[i]);
        }
    }

    void overgrown() {
        _tooLong = true;
        _reports.push(detail::makeErrorEvent(std::format("a segment outgrew the replay store of {}; it is passed through once instead", capacity.value), this->unique_name.value()));
        n_overflow = n_overflow + 1U;
        _playing.insert(_playing.end(), _store.begin(), _store.end());
        _store.clear();
        _playsLeft = _playsLeft == 0U ? 1U : _playsLeft;
    }

    void close() {
        if (_tooLong) {
            _store.clear();
            _tooLong = false;
            return;
        }
        if (_store.empty()) {
            return;
        }
        _playing = _store;
        _store.clear();
        _position  = 0UZ;
        _playsLeft = n_repeats == 0U ? 1U : n_repeats.value;
        n_segments = n_segments + 1U;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::SequenceEqual, [T], [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])

template<typename T>
struct SequenceEqual : gr::Block<SequenceEqual<T>> {
    using Description = Doc<R"(@brief emit one truth value per segment: whether two streams carried the same items [sequenceEqual]

    in         ─1──2──9──4─▶       sequenceEqual
    reference  ─1──2──3──4─▶
    out        ──────────0─▶       0 = differ, 1 = equal (uint8_t: GR4 carries no bool stream)

A comparison against a golden waveform, a redundant channel, the same signal down two cables.

 [1] example: https://rxmarbles.com/#sequenceEqual
 [2] detailed documentation: https://reactivex.io/documentation/operators/sequenceequal.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                      evtIn;
    gr::PortIn<T>                        in;
    gr::PortIn<T>                        reference;
    gr::EventPortOut                     evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortOut<std::uint8_t, gr::Async> out;

    A<std::pmr::string, "segment filter", Doc<"trigger that ends a segment and publishes the verdict">> segment_filter;
    A<T, "tolerance", Doc<"largest difference still counted as equal">>                                 tolerance = T{};
    A<gr::Size_t, "n samples", Doc<"samples per segment, 0 = only on a trigger">>                       n_samples = 1024U;

    A<gr::Size_t, "n verdicts">                                                      n_verdicts   = 0U;
    A<gr::Size_t, "n mismatches", Doc<"samples that differed, across all segments">> n_mismatches = 0U;
    A<gr::Size_t, "n unjudged", Doc<"segments whose end the block never saw">>       n_unjudged   = 0U;

    GR_MAKE_REFLECTABLE(SequenceEqual, evtIn, in, reference, evtOut, out, segment_filter, tolerance, n_samples, n_verdicts, n_mismatches, n_unjudged);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState               _ends{};
    bool                     _segmented  = false;
    bool                     _equalSoFar = true;
    std::size_t              _compared   = 0UZ;
    std::deque<std::uint8_t> _verdicts;
    detail::PendingEvents    _reports;

    void start() { restart(); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _segmented = detail::compileOptionalFilter(segment_filter.value, std::string_view{}, _ends, "SequenceEqual");
        restart();
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& refSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (_segmented && !event.empty() && gr::trigger::BasicTriggerNameCtxMatcher::match(_ends, event) == gr::trigger::MatchResult::Matching) {
                close();
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        const std::size_t nSamples = std::min(inSpan.size(), refSpan.size()); // the rings are the skew queue
        detail::forEachSegment(
            inSpan, nSamples, [&](std::size_t from, std::size_t until) { processRun(inSpan, refSpan, from, until); },
            [&](const gr::property_map_view& tag, std::size_t) {
                if (_segmented && detail::matches(_ends, tag)) {
                    close();
                }
            });

        const std::size_t emitted  = detail::drainInto(_verdicts, outSpan);
        const std::size_t reported = _reports.drainInto(evtOutSpan, 0UZ, "SequenceEqual", this->unique_name.value());
        evtOutSpan.publish(reported);
        outSpan.publish(emitted);
        if (!inSpan.consume(nSamples) || !refSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void processRun(const gr::InputSpanLike auto& inSpan, const gr::InputSpanLike auto& refSpan, std::size_t from, std::size_t until) {
        for (std::size_t i = from; i < until; ++i) {
            const T difference = inSpan[i] > refSpan[i] ? static_cast<T>(inSpan[i] - refSpan[i]) : static_cast<T>(refSpan[i] - inSpan[i]);
            if (difference > tolerance) {
                _equalSoFar  = false;
                n_mismatches = n_mismatches + 1U;
            }
            ++_compared;
            if (n_samples > 0U && _compared >= static_cast<std::size_t>(n_samples.value)) {
                close();
            }
        }
    }

    void close() {
        if (_compared == 0UZ) {
            n_unjudged = n_unjudged + 1U;
            _reports.push(detail::makeErrorEvent("a segment ended with no samples compared; no verdict is published for it", this->unique_name.value()));
            restart();
            return;
        }
        _verdicts.push_back(static_cast<std::uint8_t>(_equalSoFar ? 1U : 0U));
        n_verdicts = n_verdicts + 1U;
        restart();
    }

    void restart() {
        _equalSoFar = true;
        _compared   = 0UZ;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::Throttle, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct Throttle : gr::Block<Throttle<T>> {
    using Description = Doc<R"(@brief report the first trigger of a burst and inhibit the rest for a period [throttle]

    in      ─T─T─T─────────T──▶    throttle
    evtOut  ─T─────────────T───▶   inhibit = 3 samples

This is a dead time: it reports at once and then shuts the door, which keeps a display or a recorder from being swamped by a
source faster than anything downstream.

 [1] example: https://rxmarbles.com/#throttle
 [2] detailed documentation: https://reactivex.io/documentation/operators/throttle.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::PortIn<T>    in;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortOut<T>   out;

    A<std::pmr::string, "filter", Doc<"trigger filter; empty = every trigger">>      filter;
    A<gr::Size_t, "n samples", Doc<"inhibit, samples">>                              n_samples   = 0U;
    A<float, "timeout", Doc<"inhibit, seconds">>                                     timeout     = 0.f;
    A<float, "sample rate", Doc<"Hz, converts 'timeout' into samples, 0 = unknown">> sample_rate = 0.f;

    A<gr::Size_t, "n items", Doc<"triggers seen">>                                 n_items      = 0U;
    A<gr::Size_t, "n emitted", Doc<"triggers that passed the inhibit">>            n_emitted    = 0U;
    A<gr::Size_t, "n suppressed", Doc<"triggers that arrived inside the inhibit">> n_suppressed = 0U;

    GR_MAKE_REFLECTABLE(Throttle, evtIn, in, evtOut, out, filter, n_samples, timeout, sample_rate, n_items, n_emitted, n_suppressed);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState                     _accept{};
    bool                           _filtered = false;
    TimeBase                       _time;
    std::size_t                    _streamIndex = 0UZ;
    std::optional<std::size_t>     _lastEmittedAt;
    detail::PendingEvents          _reports;
    std::vector<detail::TriggerAt> _items;

    void start() {
        _streamIndex = 0UZ;
        _lastEmittedAt.reset();
        _reports.clear();
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples = std::min(inSpan.size(), outSpan.size());

        detail::collectTriggers(
            evtSpan, inSpan, _streamIndex, _items, //
            [this](const gr::property_map_view& candidate) { return detail::acceptsTrigger(_accept, _filtered, candidate); }, [this](const gr::property_map_view& carried, std::size_t at) { _time.adopt(carried, at); });
        for (auto& [at, item] : _items) {
            hold(std::move(item), at);
        }

        std::ranges::copy(inSpan | std::views::take(nSamples), outSpan.begin());
        _streamIndex += nSamples;

        const std::size_t emitted = _reports.drainInto(evtOutSpan, 0UZ, "Throttle", this->unique_name.value());
        evtOutSpan.publish(emitted);
        outSpan.publish(nSamples);
        if (!inSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered = detail::compileOptionalFilter(filter.value, std::string_view{}, _accept, "Throttle");
        _time.setRate(static_cast<double>(sample_rate));
    }

private:
    void hold(gr::property_map&& item, std::size_t at) {
        n_items                 = n_items + 1U;
        const auto inhibit      = resolveDuration(_time, n_samples, timeout);
        const bool insideWindow = _lastEmittedAt.has_value() && inhibit.has_value() && at < *_lastEmittedAt + *inhibit;
        if (insideWindow) {
            n_suppressed = n_suppressed + 1U;
            return;
        }
        _reports.push(std::move(item));
        _lastEmittedAt = at;
        n_emitted      = n_emitted + 1U;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_EVENTSHAPING_HPP
