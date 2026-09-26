#ifndef GNURADIO_TRIGGER_EVENTREDUCE_HPP
#define GNURADIO_TRIGGER_EVENTREDUCE_HPP

#include <algorithm>
#include <cstdint>
#include <deque>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/MonotonicClock.hpp>
#include <gnuradio-4.0/trigger/SampleTest.hpp>
#include <gnuradio-4.0/trigger/Segments.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <memory_resource>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace gr::blocks::trigger {

enum class Accumulation : std::uint8_t { AUTO, last, sum, product, minimum, maximum, mean, count, all, any };

[[nodiscard]] inline Accumulation foldNamed(std::string_view text) noexcept {
    const auto named        = gr::meta::parseEnum<Accumulation>(text);
    const bool foldsSamples = named && *named != Accumulation::AUTO && *named != Accumulation::all && *named != Accumulation::any;
    return foldsSamples ? *named : Accumulation::sum;
}

GR_REGISTER_BLOCK("gr::blocks::trigger::Last", gr::blocks::trigger::Accumulate, ([T], gr::blocks::trigger::Accumulation::last), [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::Reduce", gr::blocks::trigger::Accumulate, ([T], gr::blocks::trigger::Accumulation::AUTO), [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::Every", gr::blocks::trigger::Accumulate, ([T], gr::blocks::trigger::Accumulation::all), [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::Any", gr::blocks::trigger::Accumulate, ([T], gr::blocks::trigger::Accumulation::any), [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])

template<typename T, Accumulation accumulation = Accumulation::AUTO>
requires(std::is_arithmetic_v<T>)
struct Accumulate : gr::Block<Accumulate<T, accumulation>, gr::NoTagPropagation> {
    constexpr static bool kIsTruth = accumulation == Accumulation::all || accumulation == Accumulation::any;
    using TOut                     = std::conditional_t<kIsTruth, std::uint8_t, T>;
    using Description              = Doc<R"(@brief one value per segment: the last item, a fold of them, or whether they all compared true [last, reduce, every]

    in   ─1──2──3──T──4──5─▶       last · reduce · scan · every
    out  ────────6─────9───▶       Reduce(sum)
                                    Last
                                    Every / Any

A segment ends on whichever comes first: a trigger matching `segment_filter`, `n_samples` samples, or `timeout`
seconds -- the rule `Histogram` publishes by. Rx emits at completion; a continuous stream never completes, so the
cadence is what keeps the block from being silent. `Every`/`Any` publish `uint8_t`: GR4 carries no stream of `bool`.

@code
auto& amplitude   = graph.emplaceBlock<Accumulate<float, Accumulation::maximum>>({{"segment_filter", "CMD_BP_START"}});
auto& withinLimit = graph.emplaceBlock<Accumulate<float, Accumulation::all>>({{"predicate", "less"}, {"threshold", 5.f}});
@endcode
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn              evtIn;
    gr::PortIn<T>                in;
    gr::PortOut<TOut, gr::Async> out;

    A<std::pmr::string, "segment filter", Doc<"trigger filter ending a segment, empty = never">>   segment_filter;
    A<std::pmr::string, "operation", Doc<"last|sum|product|minimum|maximum|mean|count">>           operation   = std::pmr::string("sum");
    A<std::pmr::string, "predicate", Doc<"greater|greater_equal|less|less_equal|equal|not_equal">> predicate   = std::pmr::string("greater");
    A<T, "threshold", Doc<"what each sample is compared against">>                                 threshold   = T{};
    A<gr::Size_t, "n samples", Doc<"samples per segment, 0 = trigger ends it">>                    n_samples   = 1024U;
    A<float, "timeout", Doc<"s, segment duration; <=0 = unset">>                                   timeout     = 0.f;
    A<float, "sample rate", Doc<"Hz, converts 'timeout' into samples, 0 = unknown">>               sample_rate = 0.f;

    A<gr::Size_t, "n segments">                                                       n_segments = 0U;
    A<gr::Size_t, "n pending", Doc<"samples in the segment still being accumulated">> n_pending  = 0U;

    GR_MAKE_REFLECTABLE(Accumulate, evtIn, in, out, segment_filter, operation, predicate, threshold, n_samples, timeout, sample_rate, n_segments, n_pending);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState       _ends{};
    bool             _filtered = false;
    TimeBase         _time;
    Accumulation     _mode        = accumulation;
    Comparison       _compare     = Comparison::greater;
    std::size_t      _nFolded     = 0UZ;
    std::size_t      _streamIndex = 0UZ;
    double           _sum         = 0.;
    T                _folded      = T{};
    bool             _truth       = true;
    std::deque<TOut> _results;

    void start() {
        _streamIndex = 0UZ;
        restart();
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered = detail::compileOptionalFilter(segment_filter.value, std::string_view{}, _ends, "Accumulate");
        if constexpr (accumulation == Accumulation::AUTO) {
            _mode = foldNamed(operation.value);
        }
        _compare = parseComparison(predicate.value);
        _time.setRate(static_cast<double>(sample_rate));
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (!event.empty() && accepts(event)) {
                close();
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        detail::forEachSegment(
            inSpan, inSpan.size(), [&](std::size_t from, std::size_t until) { processRun(inSpan, from, until); },
            [&](const gr::property_map_view& tag, std::size_t) {
                _time.adopt(tag, _streamIndex);
                if (accepts(tag)) {
                    close();
                }
            });
        n_pending = static_cast<gr::Size_t>(_nFolded);

        const std::size_t emitted = detail::drainInto(_results, outSpan);
        outSpan.publish(emitted);
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] bool accepts(const gr::property_map_view& candidate) {
        if (!_filtered || !candidate.contains(std::string_view{gr::tag::TRIGGER_NAME.key()})) {
            return false;
        }
        return gr::trigger::BasicTriggerNameCtxMatcher::match(_ends, candidate) == gr::trigger::MatchResult::Matching;
    }

    void processRun(const gr::InputSpanLike auto& inSpan, std::size_t from, std::size_t until) {
        const std::size_t cadence = resolveDuration(_time, n_samples, timeout).value_or(0UZ);
        for (std::size_t i = from; i < until; ++i) {
            fold(inSpan[i]);
            ++_streamIndex;
            if (cadence > 0UZ && _nFolded >= cadence) {
                close();
            }
        }
    }

    void fold(const T& sample) {
        _sum += static_cast<double>(sample);
        if (_nFolded == 0UZ) {
            _folded = sample;
        } else {
            switch (_mode) {
            case Accumulation::product: _folded = static_cast<T>(_folded * sample); break;
            case Accumulation::minimum: _folded = std::min(_folded, sample); break;
            case Accumulation::maximum: _folded = std::max(_folded, sample); break;
            case Accumulation::sum: _folded = static_cast<T>(_folded + sample); break;
            default: _folded = sample; break;
            }
        }
        if constexpr (kIsTruth) {
            const bool holds = compare(sample);
            _truth           = accumulation == Accumulation::all ? (_truth && holds) : (_truth || holds);
        }
        ++_nFolded;
    }

    [[nodiscard]] bool compare(const T& sample) const { return holds(_compare, sample, threshold.value); }

    void close() {
        if constexpr (kIsTruth) {
            _results.push_back(static_cast<std::uint8_t>(_truth ? 1U : 0U));
            n_segments = n_segments + 1U;
        } else {
            if (_nFolded == 0UZ) {
                restart();
                return;
            }
            _results.push_back(resultOf());
            n_segments = n_segments + 1U;
        }
        restart();
    }

    [[nodiscard]] TOut resultOf() const
    requires(!kIsTruth)
    {
        switch (_mode) {
        case Accumulation::last: return _folded;
        case Accumulation::mean: return static_cast<T>(_sum / static_cast<double>(_nFolded));
        case Accumulation::count: return static_cast<T>(_nFolded);
        default: return _folded; // sum, product, minimum, maximum
        }
    }

    void restart() {
        _nFolded = 0UZ;
        _sum     = 0.;
        _folded  = T{};
        _truth   = accumulation == Accumulation::all;
    }
};

enum class CountMode : std::uint8_t { AUTO, every, nth };

GR_REGISTER_BLOCK(gr::blocks::trigger::Count)

struct Count : gr::Block<Count, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief count matching events, report every nth or once at the nth, and the rate [count]

    evtIn   ─A──A──A──A──A──A─▶    count
    evtOut  ───────3────────6─▶    mode = every, n = 3

`rate` is the arrival rate in Hz, taken from the times the events themselves carry, so no clock is needed here.

 [1] example: https://rxmarbles.com/#count
 [2] detailed documentation: https://reactivex.io/documentation/operators/count.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::string, "filter", Doc<"filter naming the counted trigger, empty = all">> filter;
    A<std::pmr::string, "mode", Doc<"every|nth">>                                        mode         = std::pmr::string("every");
    A<gr::Size_t, "n", Doc<"prescale for 'every', occurrence for 'nth'; >=1">>           n            = 1U;
    A<std::pmr::string, "trigger name", Doc<"name of the event emitted">>                trigger_name = std::pmr::string("count");

    A<gr::Size_t, "n matches", Doc<"matching events seen">>                             n_matches        = 0U;
    A<gr::Size_t, "n emitted">                                                          n_emitted        = 0U;
    A<double, "rate", Doc<"Hz, first-to-latest dated match; 0 until then">>             rate             = 0.;
    A<gr::Size_t, "n undated", Doc<"matches with no trigger_time, excluded from rate">> n_undated        = 0U;
    A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">>          n_events_dropped = 0U;

    GR_MAKE_REFLECTABLE(Count, evtIn, evtOut, filter, mode, n, trigger_name, n_matches, n_emitted, rate, n_undated, n_events_dropped);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState                   _accept{};
    bool                         _filtered = false;
    CountMode                    _mode     = CountMode::every;
    std::uint64_t                _count    = 0U;
    std::optional<std::uint64_t> _firstAt;
    std::optional<std::uint64_t> _latestAt;
    EventStore                   _events;

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& /*newSettings*/) {
        if (n == 0U) {
            const gr::Size_t previous = oldSettings.value_or<gr::Size_t>(std::string("n"), 1U);
            gr::log::warning("Count: 'n' = 0 refused (a prescale of no events has no meaning); keeping {}", previous == 0U ? 1U : previous);
            n = previous == 0U ? 1U : previous;
        }
        _mode     = mode.value == "nth" ? CountMode::nth : CountMode::every;
        _filtered = detail::compileOptionalFilter(filter.value, std::string_view{}, _accept, "Count");
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
            if (_filtered && gr::trigger::BasicTriggerNameCtxMatcher::match(_accept, event) != gr::trigger::MatchResult::Matching) {
                continue;
            }
            ++_count;
            n_matches = static_cast<gr::Size_t>(_count);
            note(stored);
            if (!reports()) {
                continue;
            }
            if (gr::emitEvent(evtOutSpan, emitted, gr::property_map_view{report(stored)})) {
                ++emitted;
                n_emitted = n_emitted + 1U;
            }
        }
        _events.retire(consumed);

        evtOutSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    void note(const StoredEvent& stored) {
        if (!stored.dated()) {
            n_undated = n_undated + 1U;
            return;
        }
        if (!_firstAt.has_value()) {
            _firstAt = *stored.at;
        }
        _latestAt = *stored.at;
        if (_firstAt.has_value() && _latestAt > _firstAt && _count > 1U) {
            const double elapsed = static_cast<double>(*_latestAt - *_firstAt) * 1e-9;
            rate                 = elapsed > 0. ? static_cast<double>(_count - 1U) / elapsed : 0.;
        }
    }

    [[nodiscard]] bool reports() const noexcept {
        const std::uint64_t every = n.value == 0U ? 1U : n.value;
        return _mode == CountMode::nth ? _count == every : _count % every == 0U;
    }

    [[nodiscard]] gr::property_map report(const StoredEvent& stored) const {
        gr::property_map event = detail::makeEvent(trigger_name.value, this->unique_name.value(), gr::property_map_view{stored.event});
        gr::property_map details;
        details[std::string("count")]                        = static_cast<std::uint64_t>(_count);
        details[std::string("rate")]                         = rate.value;
        event[std::string(gr::tag::TRIGGER_META_INFO.key())] = std::move(details);
        return event;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::Sequence)

struct Sequence : gr::Block<Sequence, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief report a trigger only inside an armed window, and report a window that expired

    evtIn   ─arm──E────E───disarm──E─▶    (no RxMarbles equivalent: Rx has no armed window)
    evtOut  ──────S────S─────────────▶    rearm = auto
    evtOut  ──────S──────────────────▶    rearm = manual
                                           {"disarm_filter", "CMD_BP_END"}, {"timeout", 1.0}});

A trigger says what happened; an accelerator also needs *when it was allowed to matter*.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::string, "arm filter", Doc<"trigger filter that opens the window">>                   arm_filter;
    A<std::pmr::string, "trigger filter", Doc<"trigger filter reported while open">>                 trigger_filter;
    A<std::pmr::string, "disarm filter", Doc<"trigger filter that closes the window, empty = none">> disarm_filter;
    A<double, "timeout", Doc<"s, window duration; 0 = until disarmed">>                              timeout      = 0.;
    A<std::pmr::string, "rearm", Doc<"auto|manual">>                                                 rearm        = std::pmr::string("auto");
    A<std::pmr::string, "trigger name", Doc<"event name for a trigger inside the window">>           trigger_name = std::pmr::string("sequence");
    A<std::pmr::string, "timeout name", Doc<"event name for an expired window">>                     timeout_name = std::pmr::string("sequence_timeout");
    A<double, "flush after", Doc<"s, idle flush; 0 = off">>                                          flush_after  = 0.;

    A<gr::Size_t, "n sequences", Doc<"triggers reported inside a window">>            n_sequences      = 0U;
    A<gr::Size_t, "n timeouts", Doc<"windows that expired without a trigger">>        n_timeouts       = 0U;
    A<gr::Size_t, "n disarmed">                                                       n_disarmed       = 0U;
    A<gr::Size_t, "n ignored", Doc<"triggers that arrived while no window was open">> n_ignored        = 0U;
    A<gr::Size_t, "n undated", Doc<"events with no trigger_time">>                    n_undated        = 0U;
    A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">>        n_events_dropped = 0U;

    GR_MAKE_REFLECTABLE(Sequence, evtIn, evtOut, arm_filter, trigger_filter, disarm_filter, timeout, rearm, trigger_name, timeout_name, flush_after, n_sequences, n_timeouts, n_disarmed, n_ignored, n_undated, n_events_dropped);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState                   _arm{};
    MatchState                   _trigger{};
    MatchState                   _disarm{};
    bool                         _hasArm     = false;
    bool                         _hasTrigger = false;
    bool                         _hasDisarm  = false;
    bool                         _keepsOpen  = true; // rearm = auto
    bool                         _armed      = false;
    std::optional<std::uint64_t> _armedAt;
    std::uint64_t                _armedOnClock = kUnknownTime;
    EventStore                   _events;
    detail::PendingEvents        _pending;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _hasArm     = detail::compileOptionalFilter(arm_filter.value, std::string_view{}, _arm, "Sequence");
        _hasTrigger = detail::compileOptionalFilter(trigger_filter.value, std::string_view{}, _trigger, "Sequence");
        _hasDisarm  = detail::compileOptionalFilter(disarm_filter.value, std::string_view{}, _disarm, "Sequence");
        _keepsOpen  = rearm.value != "manual";
        close();
        _events.clear();
        _pending.clear();
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) {
        _events.drain(evtSpan);
        n_events_dropped = static_cast<gr::Size_t>(_events.dropped);

        for (const StoredEvent& stored : _events.ordered()) {
            judge(stored);
        }
        _events.retire(_events.size());
        expireOnOwnClock();

        const std::size_t emitted = _pending.drainInto(evtOutSpan, 0UZ, "Sequence", this->unique_name.value());
        evtOutSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    void judge(const StoredEvent& stored) {
        const gr::property_map_view event{stored.event};
        if (!stored.dated()) {
            n_undated = n_undated + 1U;
        }
        expireBefore(stored);

        if (_hasDisarm && matches(_disarm, event)) {
            if (_armed) {
                n_disarmed = n_disarmed + 1U;
            }
            close();
            return;
        }
        if (_hasArm && matches(_arm, event)) {
            open(stored);
            return;
        }
        if (_hasTrigger && matches(_trigger, event)) {
            if (!_armed) {
                n_ignored = n_ignored + 1U;
                return;
            }
            _pending.push(reportOf(trigger_name.value, stored.event, stored.at));
            n_sequences = n_sequences + 1U;
            if (!_keepsOpen) {
                close();
            }
        }
    }

    void expireBefore(const StoredEvent& stored) {
        if (!_armed || timeout <= 0. || !_armedAt.has_value() || !stored.dated()) {
            return;
        }
        const std::uint64_t deadline = *_armedAt + static_cast<std::uint64_t>(timeout * 1e9);
        if (*stored.at > deadline) {
            expire(deadline);
        }
    }

    void expireOnOwnClock() {
        if (!_armed || flush_after <= 0. || _armedOnClock == kUnknownTime) {
            return;
        }
        const std::uint64_t now = monotonicNowNs();
        if (now == kUnknownTime || now < _armedOnClock) {
            return;
        }
        if (static_cast<double>(now - _armedOnClock) * 1e-9 >= flush_after) {
            expire(_armedAt.has_value() ? *_armedAt + static_cast<std::uint64_t>(timeout * 1e9) : 0U);
        }
    }

    void expire(std::uint64_t deadline) {
        gr::property_map event = detail::makeEvent(timeout_name.value, this->unique_name.value());
        if (deadline > 0U) {
            event[std::string(gr::tag::TRIGGER_TIME.key())] = deadline;
        }
        gr::property_map details;
        details[std::string("armed_at")]                     = _armedAt.value_or(0U);
        event[std::string(gr::tag::TRIGGER_META_INFO.key())] = std::move(details);
        _pending.push(std::move(event));
        n_timeouts = n_timeouts + 1U;
        close();
    }

    void open(const StoredEvent& stored) {
        _armed        = true;
        _armedAt      = stored.at;
        _armedOnClock = stored.arrived == kUnknownTime ? monotonicNowNs() : stored.arrived;
    }

    void close() {
        _armed        = false;
        _armedAt      = std::nullopt;
        _armedOnClock = kUnknownTime;
    }

    [[nodiscard]] gr::property_map reportOf(std::string_view reportName, const gr::property_map& from, std::optional<std::uint64_t> at) const {
        gr::property_map event = detail::makeEvent(reportName, this->unique_name.value(), gr::property_map_view{from});
        gr::property_map details;
        details[std::string("armed_at")] = _armedAt.value_or(0U);
        if (at.has_value() && _armedAt.has_value() && *at >= *_armedAt) {
            details[std::string("elapsed")] = static_cast<double>(*at - *_armedAt) * 1e-9;
        }
        event[std::string(gr::tag::TRIGGER_META_INFO.key())] = std::move(details);
        return event;
    }

    [[nodiscard]] static bool matches(MatchState& state, const gr::property_map_view& event) { //
        return gr::trigger::BasicTriggerNameCtxMatcher::match(state, event) == gr::trigger::MatchResult::Matching;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::EventBuilder, [T], [ int16_t, int32_t, float, double ])

template<typename T>
struct EventBuilder : gr::Block<EventBuilder<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief assemble the fragments of one event, matched by timestamp or identifier, into one DataSet [zip, by carried time]

    in#0  ──F(t0)──────────F(t1)──▶   zip, but by carried time
    in#1  ────F(t0+20ns)─────F(t1)─▶  tolerance = 50 ns
    out   ──────E(t0)──────────E(t1)─▶

Rx `zip` pairs the nth of each input; this pairs what the fragments say about themselves.

 [1] example: https://rxmarbles.com/#zip
 [2] detailed documentation: https://reactivex.io/documentation/operators/zip.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    std::vector<gr::PortIn<gr::DataSet<T>>> in;
    gr::EventPortOut                        evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortOut<gr::DataSet<T>, gr::Async>  out;

    A<gr::Size_t, "n inputs", Doc<"fragments that make one event">, gr::Limits<1U, 64U>> n_inputs      = 0U;
    A<std::pmr::string, "key", Doc<"timestamp|event_id: what identifies the event">>     key           = std::pmr::string("timestamp");
    A<double, "tolerance", Doc<"s, max timestamp difference for the same event">>        tolerance     = 0.;
    A<double, "timeout", Doc<"s, lead time before judged incomplete">>                   timeout       = 1e-3;
    A<std::pmr::string, "on incomplete", Doc<"drop|emit_partial">>                       on_incomplete = std::pmr::string("drop");
    A<gr::Size_t, "max pending", Doc<"events assembling before the oldest is judged">>   max_pending   = 16U;
    A<double, "flush after", Doc<"s, idle flush; 0 = off">>                              flush_after   = 0.;

    A<gr::Size_t, "n events">                                                               n_events     = 0U;
    A<gr::Size_t, "n incomplete", Doc<"events judged with fragments missing">>              n_incomplete = 0U;
    A<gr::Size_t, "n dropped", Doc<"incomplete events not published">>                      n_dropped    = 0U;
    A<gr::Size_t, "n mismatched", Doc<"fragments refused because their extents disagreed">> n_mismatched = 0U;
    A<gr::Size_t, "n unkeyed", Doc<"fragments carrying nothing to match on">>               n_unkeyed    = 0U;
    A<gr::Size_t, "n pending", Doc<"events being assembled">>                               n_pending    = 0U;

    GR_MAKE_REFLECTABLE(EventBuilder, in, evtOut, out, n_inputs, key, tolerance, timeout, on_incomplete, max_pending, flush_after, n_events, n_incomplete, n_dropped, n_mismatched, n_unkeyed, n_pending);

    struct Assembling {
        std::int64_t                               at = 0;
        std::vector<std::optional<gr::DataSet<T>>> fragments; // one slot per input
        std::uint64_t                              arrived = kUnknownTime;

        [[nodiscard]] std::size_t held() const {
            return static_cast<std::size_t>(std::ranges::count_if(fragments, [](const auto& fragment) { return fragment.has_value(); }));
        }
    };

    std::deque<Assembling>     _pending;
    detail::PendingEvents      _reports;
    std::deque<gr::DataSet<T>> _assembled;

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("n_inputs") && oldSettings.find_value("n_inputs") != newSettings.find_value("n_inputs")) {
            in.resize(n_inputs);
            _pending.clear();
        }
    }

    template<gr::InputSpanLike TInput>
    gr::work::Status processBulk(const std::span<TInput>& ins, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) {
        for (std::size_t channel = 0UZ; channel < ins.size(); ++channel) {
            for (const gr::DataSet<T>& fragment : ins[channel]) {
                place(fragment, channel);
            }
            std::ignore = ins[channel].consume(ins[channel].size());
        }
        flushIdle();
        n_pending = static_cast<gr::Size_t>(_pending.size());

        const std::size_t emitted  = detail::drainInto(_assembled, outSpan);
        const std::size_t reported = _reports.drainInto(evtOutSpan, 0UZ, "EventBuilder", this->unique_name.value());
        evtOutSpan.publish(reported);
        outSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] std::optional<std::int64_t> keyOf(const gr::DataSet<T>& fragment) const {
        if (key.value == "event_id") {
            if (fragment.meta_information.empty()) {
                return std::nullopt;
            }
            const gr::property_map_view meta{fragment.meta_information[0]};
            if (const auto id = meta.template get_if<std::uint64_t>(std::string_view{"event_id"})) {
                return static_cast<std::int64_t>(*id);
            }
            return std::nullopt;
        }
        return fragment.timestamp == 0 ? std::nullopt : std::optional{fragment.timestamp};
    }

    void place(const gr::DataSet<T>& fragment, std::size_t channel) {
        const auto at = keyOf(fragment);
        if (!at.has_value()) {
            n_unkeyed = n_unkeyed + 1U;
            return;
        }
        judgeOlderThan(*at);

        auto event = find(*at);
        if (event == _pending.end()) {
            if (_pending.size() >= static_cast<std::size_t>(max_pending.value)) {
                judge(_pending.front());
                _pending.pop_front();
            }
            _pending.push_back(Assembling{.at = *at, .fragments = std::vector<std::optional<gr::DataSet<T>>>(in.size()), .arrived = monotonicNowNs()});
            event = std::prev(_pending.end());
        }
        if (channel >= event->fragments.size()) {
            return;
        }
        if (!fits(*event, fragment)) {
            n_mismatched = n_mismatched + 1U;
            _reports.push(detail::makeErrorEvent("a fragment's extents disagreed with the event's; a DataSet carries one extent per signal", this->unique_name.value()));
            return;
        }
        event->fragments[channel] = fragment;

        if (event->held() == in.size()) {
            publish(*event);
            erase(*at);
        }
    }

    [[nodiscard]] static bool fits(const Assembling& event, const gr::DataSet<T>& fragment) { //
        return std::ranges::none_of(event.fragments, [&](const auto& held) { return held && held->extents != fragment.extents; });
    }

    [[nodiscard]] std::deque<Assembling>::iterator find(std::int64_t at) {
        const std::int64_t window = static_cast<std::int64_t>(tolerance * 1e9);
        return std::ranges::find_if(_pending, [&](const Assembling& event) {
            const std::int64_t apart = event.at > at ? event.at - at : at - event.at;
            return key.value == "event_id" ? event.at == at : apart <= window;
        });
    }

    void erase(std::int64_t at) {
        const auto gone = std::ranges::find_if(_pending, [at](const Assembling& event) { return event.at == at; });
        if (gone != _pending.end()) {
            _pending.erase(gone);
        }
    }

    void judgeOlderThan(std::int64_t at) {
        if (timeout <= 0. || key.value == "event_id") {
            return;
        }
        const std::int64_t horizon = static_cast<std::int64_t>(timeout * 1e9);
        while (!_pending.empty() && at - _pending.front().at > horizon) {
            judge(_pending.front());
            _pending.pop_front();
        }
    }

    void flushIdle() {
        if (flush_after <= 0. || _pending.empty()) {
            return;
        }
        const std::uint64_t now = monotonicNowNs();
        if (now == kUnknownTime) {
            return;
        }
        while (!_pending.empty() && _pending.front().arrived != kUnknownTime && now > _pending.front().arrived && static_cast<double>(now - _pending.front().arrived) * 1e-9 >= flush_after) {
            judge(_pending.front());
            _pending.pop_front();
        }
    }

    void judge(Assembling& event) {
        n_incomplete = n_incomplete + 1U;
        if (on_incomplete.value != "emit_partial") {
            n_dropped = n_dropped + 1U;
            return;
        }
        publish(event);
    }

    void publish(Assembling& event) {
        gr::DataSet<T> whole;
        whole.timestamp = event.at;
        bool first      = true;
        for (const auto& held : event.fragments) {
            if (!held.has_value()) {
                continue;
            }
            if (first) {
                whole.axis_names  = held->axis_names;
                whole.axis_units  = held->axis_units;
                whole.axis_values = held->axis_values;
                whole.extents     = held->extents;
                first             = false;
            }
            whole.signal_names.insert(whole.signal_names.end(), held->signal_names.begin(), held->signal_names.end());
            whole.signal_quantities.insert(whole.signal_quantities.end(), held->signal_quantities.begin(), held->signal_quantities.end());
            whole.signal_units.insert(whole.signal_units.end(), held->signal_units.begin(), held->signal_units.end());
            whole.signal_values.insert(whole.signal_values.end(), held->signal_values.begin(), held->signal_values.end());
            whole.signal_ranges.insert(whole.signal_ranges.end(), held->signal_ranges.begin(), held->signal_ranges.end());
            whole.meta_information.insert(whole.meta_information.end(), held->meta_information.begin(), held->meta_information.end());
            whole.timing_events.insert(whole.timing_events.end(), held->timing_events.begin(), held->timing_events.end());
        }
        if (first) {
            return;
        }
        _assembled.push_back(std::move(whole));
        n_events = n_events + 1U;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_EVENTREDUCE_HPP
