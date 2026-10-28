#ifndef GNURADIO_TRIGGER_EVENTREDUCE_HPP
#define GNURADIO_TRIGGER_EVENTREDUCE_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/EventReference.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/SampleTest.hpp>
#include <gnuradio-4.0/trigger/Segments.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <limits>
#include <memory_resource>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace gr::blocks::trigger {

enum class Accumulation : std::uint8_t { automatic, last, sum, product, minimum, maximum, mean, count, all, any };

[[nodiscard]] inline Accumulation foldNamed(std::string_view text) noexcept {
    const auto named        = gr::meta::parseEnum<Accumulation>(text);
    const bool foldsSamples = named && *named != Accumulation::automatic && *named != Accumulation::all && *named != Accumulation::any;
    return foldsSamples ? *named : Accumulation::sum;
}

/** the running reduction of a segment by `Accumulate` and of a run by `Scan`; trivially copyable, so a kernel holds it */
template<typename T>
struct Fold {
    T           folded = T{};
    double      sum    = 0.;
    std::size_t n      = 0UZ;

    constexpr void add(const T& sample, Accumulation mode) noexcept {
        sum += static_cast<double>(sample);
        if (n == 0UZ) {
            folded = sample;
        } else {
            switch (mode) {
            case Accumulation::product: folded = static_cast<T>(folded * sample); break;
            case Accumulation::minimum: folded = std::min(folded, sample); break;
            case Accumulation::maximum: folded = std::max(folded, sample); break;
            case Accumulation::sum: folded = static_cast<T>(folded + sample); break;
            default: folded = sample; break;
            }
        }
        ++n;
    }

    [[nodiscard]] constexpr T value(Accumulation mode) const noexcept {
        switch (mode) {
        case Accumulation::mean: return static_cast<T>(sum / static_cast<double>(n));
        case Accumulation::count: return static_cast<T>(n);
        default: return folded;
        }
    }
};

GR_REGISTER_BLOCK("gr::blocks::trigger::Last", gr::blocks::trigger::Accumulate, ([T], gr::blocks::trigger::Accumulation::last), [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::Reduce", gr::blocks::trigger::Accumulate, ([T], gr::blocks::trigger::Accumulation::automatic), [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::Every", gr::blocks::trigger::Accumulate, ([T], gr::blocks::trigger::Accumulation::all), [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::Any", gr::blocks::trigger::Accumulate, ([T], gr::blocks::trigger::Accumulation::any), [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])

template<typename T, Accumulation accumulation = Accumulation::automatic>
requires(std::is_arithmetic_v<T>)
struct Accumulate : gr::Block<Accumulate<T, accumulation>, gr::NoTagPropagation> {
    constexpr static bool kIsTruth = accumulation == Accumulation::all || accumulation == Accumulation::any;
    using TOut                     = std::conditional_t<kIsTruth, std::uint8_t, T>;
    using Description              = Doc<R"(@brief one value per segment: the last item, a fold of them, or whether they all compared true [last, reduce, every]

    in   ─1──2──3──T──4──5─▶       last · reduce · scan · every
    out  ────────6─────9───▶       Reduce(sum)
                                    Last
                                    Every / Any
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn              evtIn;
    gr::PortIn<T>                in;
    gr::PortOut<TOut, gr::Async> out;

    A<std::pmr::string, "segment filter", Doc<"trigger filter ending a segment, empty = never; a segment ends on the first of it, n_samples or timeout">> segment_filter;
    A<std::pmr::string, "operation", Doc<"last|sum|product|minimum|maximum|mean|count">>                                                                  operation   = std::pmr::string("sum");
    A<std::pmr::string, "predicate", Doc<"greater|greaterEqual|less|lessEqual|equal|notEqual">>                                                           predicate   = std::pmr::string("greater");
    A<T, "threshold", Doc<"what each sample is compared against">>                                                                                        threshold   = T{};
    A<gr::Size_t, "n samples", Doc<"samples per segment, 0 = trigger ends it">>                                                                           n_samples   = 1024U;
    A<float, "timeout", Doc<"s, segment duration; <=0 = unset">>                                                                                          timeout     = 0.f;
    gr::SampleRate                                                                                                                                        sample_rate = 0.f;

    mutable A<gr::Size_t, "n segments", Doc<"segments reported">>                                               n_segments        = 0U;
    mutable A<gr::Size_t, "n pending", Doc<"samples in the segment still being accumulated">>                   n_pending         = 0U;
    mutable A<gr::Size_t, "n results dropped", Doc<"results lost, oldest first, while the output stayed full">> n_results_dropped = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState   _ends{};
    bool         _filtered       = false;
    Accumulation _mode           = accumulation;
    Comparison   _compare        = Comparison::greater;
    gr::Size_t   _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Accumulate, evtIn, in, out, segment_filter, operation, predicate, threshold, n_samples, timeout, sample_rate, n_segments, n_pending, n_results_dropped, _ends, _filtered, _mode, _compare, _settings_epoch);

    constexpr static std::size_t kMaxResults = 256UZ;

    mutable MatchState                    _ending{};
    mutable TimeBase                      _time;
    mutable Fold<T>                       _fold{};
    mutable std::size_t                   _streamIndex = 0UZ;
    mutable bool                          _truth       = true;
    mutable std::array<TOut, kMaxResults> _results{}; /// what the output had no room for, oldest at `_resultsFirst`
    mutable std::size_t                   _resultsFirst = 0UZ;
    mutable std::size_t                   _nResults     = 0UZ;
    mutable gr::Size_t                    _stateEpoch   = std::numeric_limits<gr::Size_t>::max();
    detail::ErrorMemory                   _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        _streamIndex = 0UZ;
        restart();
        _time.reset();
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered = detail::compileOptionalFilter(*this, segment_filter.value, std::string_view{}, _ends);
        if constexpr (accumulation == Accumulation::automatic) {
            _mode = foldNamed(operation.value);
        }
        if (const auto named = parseComparison(predicate.value)) {
            _compare = *named;
        } else {
            const std::string message = std::format("predicate '{}' refused (not a comparison name); keeping '{}'", predicate.value, gr::meta::enumName(_compare).value_or(""));
            detail::reportError(*this, "settingsChanged()", message);
            predicate = std::pmr::string(gr::meta::enumName(_compare).value_or(""));
        }
        _settings_epoch = _settings_epoch + 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) const {
        if (_stateEpoch != _settings_epoch) { // new settings, not a new run: the open segment carries on
            _ending = _ends;
            _time.setRate(static_cast<double>(sample_rate));
            _stateEpoch = _settings_epoch;
        }
        std::size_t emitted = 0UZ;
        for (; emitted < outSpan.size() && _nResults > 0UZ; ++emitted, _resultsFirst = (_resultsFirst + 1UZ) % kMaxResults, --_nResults) {
            outSpan[emitted] = _results[_resultsFirst];
        }

        for (const gr::property_map_view& event : evtSpan) {
            if (!event.empty() && accepts(event)) {
                close(outSpan, emitted);
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        detail::forEachSegment(
            inSpan, inSpan.size(), [&](std::size_t from, std::size_t until) { processRun(inSpan, from, until, outSpan, emitted); },
            [&](const gr::property_map_view& tag, std::size_t) {
                std::ignore = _time.adopt(tag, _streamIndex);
                if (accepts(tag)) {
                    close(outSpan, emitted);
                }
            });
        n_pending = static_cast<gr::Size_t>(_fold.n);

        outSpan.publish(emitted);
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] bool accepts(const gr::property_map_view& candidate) const { return _filtered && detail::acceptsTrigger(_ending, true, candidate); }

    void processRun(const gr::InputSpanLike auto& inSpan, std::size_t from, std::size_t until, auto& outSpan, std::size_t& emitted) const {
        const std::size_t cadence = resolveDuration(_time, n_samples, timeout).value_or(0UZ);
        for (std::size_t i = from; i < until; ++i) {
            fold(inSpan[i]);
            ++_streamIndex;
            if (cadence > 0UZ && _fold.n >= cadence) {
                close(outSpan, emitted);
            }
        }
    }

    void fold(const T& sample) const {
        _fold.add(sample, _mode);
        if constexpr (kIsTruth) {
            const bool holds = compare(sample);
            _truth           = accumulation == Accumulation::all ? (_truth && holds) : (_truth || holds);
        }
    }

    [[nodiscard]] bool compare(const T& sample) const { return holds(_compare, sample, threshold.value); }

    void close(auto& outSpan, std::size_t& emitted) const {
        if constexpr (kIsTruth) {
            deliver(static_cast<std::uint8_t>(_truth ? 1U : 0U), outSpan, emitted);
            n_segments = n_segments + 1U;
        } else {
            if (_fold.n == 0UZ) {
                restart();
                return;
            }
            deliver(_fold.value(_mode), outSpan, emitted);
            n_segments = n_segments + 1U;
        }
        restart();
    }

    /** straight to the output while it has room and nothing older waits, else behind what waits */
    void deliver(TOut result, auto& outSpan, std::size_t& emitted) const {
        if (_nResults == 0UZ && emitted < outSpan.size()) {
            outSpan[emitted++] = result;
            return;
        }
        if (_nResults == kMaxResults) {
            _resultsFirst = (_resultsFirst + 1UZ) % kMaxResults;
            --_nResults;
            n_results_dropped = n_results_dropped + 1U;
        }
        _results[(_resultsFirst + _nResults) % kMaxResults] = result;
        ++_nResults;
    }

    void restart() const {
        _fold  = Fold<T>{};
        _truth = accumulation == Accumulation::all;
    }
};

enum class CountMode : std::uint8_t { automatic, every, nth, within };
enum class CountWindow : std::uint8_t { tumbling, sliding };

GR_REGISTER_BLOCK(gr::blocks::trigger::Count)

struct Count : gr::Block<Count, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief count matching events, report every nth or once at the nth, and the rate [count]

    evtIn   ─A──A──A──A──A──A─▶    count
    evtOut  ───────3────────6─▶    mode = every, n = 3

 [1] example: https://rxmarbles.com/#count
 [2] detailed documentation: https://reactivex.io/documentation/operators/count.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::string, "filter", Doc<"filter naming the counted trigger, empty = all">>                                                                                      filter;
    A<std::pmr::string, "reset filter", Doc<"an event matching this starts the count over and is not itself counted; empty = never">>                                         reset_filter;
    A<std::pmr::string, "mode", Doc<"every|nth|within">>                                                                                                                      mode         = std::pmr::string("every");
    A<gr::Size_t, "n", Doc<"prescale for 'every', occurrence for 'nth', matches for 'within'; >=1">>                                                                          n            = 1U;
    A<double, "interval", Doc<"s, the time 'within' counts over">>                                                                                                            interval     = 0.;
    A<double, "max lateness", Doc<"s, how long after its trigger_time an event may be decided; earlier ones wait as long">>                                                   max_lateness = 0.;
    A<std::pmr::string, "window", Doc<"tumbling|sliding, for 'within': opens at the first match and restarts after interval, or holds the last n; consumed once it reports">> window       = std::pmr::string("tumbling");
    gr::TriggerName                                                                                                                                                           trigger_name;

    mutable A<gr::Size_t, "n matches", Doc<"matching events seen">>                                         n_matches        = 0U;
    mutable A<gr::Size_t, "n emitted", Doc<"counts reported">>                                              n_emitted        = 0U;
    mutable A<double, "rate", Doc<"Hz, from the matches' own trigger_time, first to latest; 0 until then">> rate             = 0.;
    mutable A<gr::Size_t, "n undated", Doc<"matches with no trigger_time, excluded from rate">>             n_undated        = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events the store, or a full output, had no room for">>   n_events_dropped = 0U;
    mutable A<gr::Size_t, "n pending", Doc<"events held back for max_lateness as of the last work call">>   n_pending        = 0U;
    mutable A<gr::Size_t, "n resets", Doc<"counts a reset event started over">>                             n_resets         = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState  _accept{};
    MatchState  _reset{};
    bool        _filtered       = false;
    bool        _resettable     = false;
    CountMode   _mode           = CountMode::every;
    CountWindow _window         = CountWindow::tumbling;
    gr::Size_t  _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Count, evtIn, evtOut, filter, reset_filter, mode, n, interval, max_lateness, window, trigger_name, n_matches, n_emitted, rate, n_undated, n_events_dropped, n_pending, n_resets, _accept, _reset, _filtered, _resettable, _mode, _window, _settings_epoch);

    constexpr static std::size_t kMaxWithin = 64UZ;

    mutable MatchState                            _accepting{};
    mutable MatchState                            _resetting{};
    mutable std::optional<std::uint64_t>          _firstAt;
    mutable std::uint64_t                         _nDated = 0U; /// matches that carried a time, the rate's count
    mutable std::array<std::uint64_t, kMaxWithin> _within{};    /// times of the matches the window holds, oldest first
    mutable std::size_t                           _nWithin    = 0UZ;
    mutable gr::Size_t                            _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable EventStore                            _events;

    detail::ErrorMemory _deferredError{};

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& /*newSettings*/) {
        if (n == 0U) {
            const gr::Size_t  previous = oldSettings.value_or<gr::Size_t>("n", 1U);
            const std::string message  = std::format("'n' = 0 refused (a prescale of no events has no meaning); keeping {}", previous == 0U ? 1U : previous);
            detail::reportError(*this, "settingsChanged()", message);
            n = previous == 0U ? 1U : previous;
        }
        _mode   = gr::meta::parseEnum<CountMode>(mode.value).value_or(CountMode::every);
        _window = gr::meta::parseEnum<CountWindow>(window.value).value_or(CountWindow::tumbling);
        if (_mode == CountMode::within && (n > kMaxWithin || interval <= 0.)) {
            const std::string message = std::format("'within' needs 1 <= n <= {} and an interval > 0 s (n = {}, interval = {} s); counting 'every' instead", kMaxWithin, n.value, interval.value);
            detail::reportError(*this, "settingsChanged()", message);
            _mode = CountMode::every;
        }
        _filtered       = detail::compileOptionalFilter(*this, filter.value, std::string_view{}, _accept);
        _resettable     = detail::compileOptionalFilter(*this, reset_filter.value, std::string_view{}, _reset);
        _settings_epoch = _settings_epoch + 1U;
        _events.clear();
        _events.lateness = detail::nanoseconds(max_lateness.value);
    }

    void start() {
        detail::sendDeferredError(*this);
        detail::nameTriggerAfterBlock(*this);
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(evtSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _accepting  = _accept;
            _resetting  = _reset;
            n_matches   = 0U;
            _firstAt    = std::nullopt;
            _nDated     = 0U;
            rate        = 0.;
            _nWithin    = 0UZ;
            _stateEpoch = _settings_epoch;
        }

        std::size_t emitted = 0UZ;
        std::ignore         = forEachEvent<kOnHost>(_events, evtSpan, kOnHost ? std::numeric_limits<std::size_t>::max() : evtOutSpan.size(), [&](const gr::property_map_view& event) {
            if (emitted >= evtOutSpan.size()) {
                return false;
            }
            if (_resettable && detail::matches(_resetting, event)) {
                startOver();
                return true;
            }
            if (_filtered && !detail::matches(_accepting, event)) {
                return true;
            }
            n_matches = n_matches + 1U;
            note(event);
            if (!reports(event)) {
                return true;
            }
            alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> blob{};
            gr::property_map_view                                                             report = gr::formatEvent(blob, detail::kEventKeys + 1U);
            if (writeReport(report, event) && gr::emit(evtOutSpan, emitted, report)) {
                ++emitted;
                n_emitted = n_emitted + 1U;
            } else {
                n_events_dropped = n_events_dropped + 1U;
            }
            return true;
        });
        if constexpr (kOnHost) {
            n_events_dropped = n_events_dropped + static_cast<gr::Size_t>(_events.dropped);
            _events.dropped  = 0U;
            n_pending        = static_cast<gr::Size_t>(_events.size());
        }
        evtOutSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    void startOver() const {
        n_matches = 0U;
        _firstAt  = std::nullopt;
        _nDated   = 0U;
        rate      = 0.;
        _nWithin  = 0UZ;
        n_resets  = n_resets + 1U;
    }

    void note(const gr::property_map_view& event) const {
        const auto at = event.template get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
        if (!at) {
            n_undated = n_undated + 1U;
            return;
        }
        ++_nDated;
        if (!_firstAt.has_value()) {
            _firstAt = *at;
        }
        if (*at > *_firstAt && _nDated > 1U) {
            const double elapsed = static_cast<double>(*at - *_firstAt) * 1e-9;
            rate                 = elapsed > 0. ? static_cast<double>(_nDated - 1U) / elapsed : 0.;
        }
    }

    [[nodiscard]] bool reports(const gr::property_map_view& event) const noexcept {
        const std::uint64_t every = n.value == 0U ? 1U : n.value;
        switch (_mode) {
        case CountMode::nth: return n_matches.value == every;
        case CountMode::within: return withinReached(event);
        case CountMode::automatic:
        case CountMode::every: return n_matches.value % every == 0U;
        }
        return false;
    }

    /** an undated match cannot be placed in an interval, so it does not count towards one */
    [[nodiscard]] bool withinReached(const gr::property_map_view& event) const noexcept {
        const auto at = event.template get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
        if (!at) {
            return false;
        }
        const auto span    = static_cast<std::int64_t>(detail::nanoseconds(interval.value));
        const auto spanned = [&](std::uint64_t held) { // signed: a late match may precede what is held; no llabs, which a kernel lacks
            const std::int64_t apart = static_cast<std::int64_t>(*at) - static_cast<std::int64_t>(held);
            return (apart < 0 ? -apart : apart) <= span;
        };
        if (_window == CountWindow::tumbling) {
            if (_nWithin > 0UZ && !spanned(_within[0])) {
                _nWithin = 0UZ;
            }
        } else {
            const auto stale = std::ranges::find_if(_within.begin(), _within.begin() + static_cast<std::ptrdiff_t>(_nWithin), spanned);
            const auto kept  = std::ranges::move(stale, _within.begin() + static_cast<std::ptrdiff_t>(_nWithin), _within.begin()).out;
            _nWithin         = static_cast<std::size_t>(std::ranges::distance(_within.begin(), kept));
        }
        _within[_nWithin++] = *at;
        if (_nWithin < static_cast<std::size_t>(n.value)) {
            return false;
        }
        _nWithin = 0UZ; // consumed once
        return true;
    }

    [[nodiscard]] bool writeReport(gr::property_map_view& report, const gr::property_map_view& from) const noexcept {
        const auto countAndRate = [&](gr::property_map_view& details) { return details.try_emplace(std::string_view{"count"}, static_cast<std::uint64_t>(n_matches.value)) && details.try_emplace(std::string_view{"rate"}, rate.value); };
        return detail::writeEvent(report, trigger_name.value, from, 2U, countAndRate);
    }
};

enum class DelayBounds : std::uint8_t { closed, open, minOpen, maxOpen };
enum class Report : std::uint8_t { sequence, timeout, both };

GR_REGISTER_BLOCK(gr::blocks::trigger::Then)

struct Then : gr::Block<Then, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief report B when it follows A within [min_delay, max_delay], and report an A that waited in vain

    evtIn   ─A──B─────A────────────▶    (no RxMarbles equivalent: Rx has no armed window)
    evtOut  ────S─────────timeout──▶    min_delay = 0, max_delay = 1 s
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::string, "arm filter", Doc<"trigger filter naming A; empty = every B arms the wait for the next, a gap longer than max_delay is one timeout">> arm_filter;
    A<std::pmr::string, "trigger filter", Doc<"trigger filter naming B">>                                                                                     trigger_filter;
    A<std::pmr::string, "disarm filter", Doc<"trigger filter that cancels a wait, empty = none">>                                                             disarm_filter;
    A<double, "min delay", Doc<"s, earliest B after A">>                                                                                                      min_delay    = 0.;
    A<double, "max delay", Doc<"s, latest B after A; 0 = no deadline">>                                                                                       max_delay    = 0.;
    A<std::pmr::string, "bounds", Doc<"closed|open|minOpen|maxOpen: which ends of the delay range are open">>                                                 bounds       = std::pmr::string("maxOpen");
    A<std::pmr::string, "report", Doc<"sequence|timeout|both: which events go out">>                                                                          report       = std::pmr::string("both");
    A<double, "max lateness", Doc<"s, how long after its trigger_time an event may be decided; earlier ones wait as long">>                                   max_lateness = 0.;
    A<std::pmr::string, "on repeat", Doc<"ignore|restart: an A while waiting">>                                                                               on_repeat    = std::pmr::string("ignore");
    A<std::pmr::string, "reference", Doc<"decision|start: the time A and B are dated by, the start tuple of a delayed decision or its trigger_time">>         reference    = std::pmr::string("decision");
    gr::TriggerName                                                                                                                                           trigger_name;
    A<std::pmr::string, "timeout name", Doc<"event name for an A no B followed, dated at its deadline">>                                                      timeout_name = std::pmr::string("then_timeout");

    mutable A<gr::Size_t, "n sequences", Doc<"B reported after an A">>                                                             n_sequences      = 0U;
    mutable A<gr::Size_t, "n timeouts", Doc<"A that waited past max_delay">>                                                       n_timeouts       = 0U;
    mutable A<gr::Size_t, "n disarmed", Doc<"A that a disarm_filter event cancelled">>                                             n_disarmed       = 0U;
    mutable A<gr::Size_t, "n ignored", Doc<"B with no A waiting, or earlier than min_delay">>                                      n_ignored        = 0U;
    mutable A<gr::Size_t, "n undated", Doc<"events with no trigger_time">>                                                         n_undated        = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">>                                             n_events_dropped = 0U;
    mutable A<gr::Size_t, "n pending", Doc<"an A still waiting, and events held back for max_lateness, as of the last work call">> n_pending        = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState     _arm{};
    MatchState     _trigger{};
    MatchState     _disarm{};
    bool           _hasArm         = false;
    bool           _hasTrigger     = false;
    bool           _hasDisarm      = false;
    bool           _restarts       = false;
    bool           _watches        = false; /// no arm_filter, a trigger_filter and a deadline: each B opens the wait for the next
    DelayBounds    _bounds         = DelayBounds::maxOpen;
    Report         _report         = Report::both;
    EventReference _reference      = EventReference::decision;
    gr::Size_t     _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Then, evtIn, evtOut, arm_filter, trigger_filter, disarm_filter, min_delay, max_delay, bounds, report, max_lateness, on_repeat, reference, trigger_name, timeout_name, n_sequences, n_timeouts, n_disarmed, n_ignored, n_undated, n_events_dropped, n_pending, //
        _arm, _trigger, _disarm, _hasArm, _hasTrigger, _hasDisarm, _restarts, _watches, _bounds, _report, _reference, _settings_epoch);

    mutable MatchState        _arming{};
    mutable MatchState        _triggering{};
    mutable MatchState        _disarming{};
    mutable detail::HeldEvent _armedBy;            /// the name, time and offset of the A a B would follow; empty: none waits
    mutable std::uint64_t     _armedAt    = 0U;    /// [ns] the A's reference time
    mutable bool              _timedOut   = false; /// watching: a timeout was reported and no B has ended it yet
    mutable gr::Size_t        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable EventStore        _events;
    detail::ErrorMemory       _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        detail::nameTriggerAfterBlock(*this);
    }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        _hasArm     = detail::compileOptionalFilter(*this, arm_filter.value, std::string_view{}, _arm);
        _hasTrigger = detail::compileOptionalFilter(*this, trigger_filter.value, std::string_view{}, _trigger);
        _hasDisarm  = detail::compileOptionalFilter(*this, disarm_filter.value, std::string_view{}, _disarm);
        _restarts   = on_repeat.value == "restart";
        _bounds     = gr::meta::parseEnum<DelayBounds>(bounds.value).value_or(DelayBounds::maxOpen);
        _report     = gr::meta::parseEnum<Report>(report.value).value_or(Report::both);
        _reference  = gr::meta::parseEnum<EventReference>(reference.value).value_or(EventReference::decision);
        if (max_delay > 0. && min_delay > max_delay) {
            const double      previousMin = oldSettings.value_or<double>("min_delay", 0.);
            const double      previousMax = oldSettings.value_or<double>("max_delay", 0.);
            const std::string message     = std::format("delay range [{}, {}] s refused (inverted); keeping [{}, {}] s", min_delay.value, max_delay.value, previousMin, previousMax);
            detail::reportError(*this, "settingsChanged()", message);
            min_delay = previousMin;
            max_delay = previousMax;
        }
        if (!_hasArm && _hasTrigger && newSettings.contains("max_delay") && !(max_delay > 0.)) {
            const double      previousMax = oldSettings.value_or<double>("max_delay", 0.);
            const std::string message     = std::format("without an arm_filter the block watches the gaps between trigger_filter events, which needs max_delay > 0; keeping {} s", previousMax);
            detail::reportError(*this, "settingsChanged()", message);
            max_delay = previousMax;
        }
        _watches = !_hasArm && _hasTrigger && max_delay > 0.;
        _events.clear();
        _events.lateness = detail::nanoseconds(max_lateness.value);
        _settings_epoch  = _settings_epoch + 1U;
    }

    void reset() {
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
        _events.clear();
        n_sequences      = 0U;
        n_timeouts       = 0U;
        n_disarmed       = 0U;
        n_ignored        = 0U;
        n_undated        = 0U;
        n_events_dropped = 0U;
        n_pending        = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(evtSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _arming     = _arm;
            _triggering = _trigger;
            _disarming  = _disarm;
            _armedBy.clear();
            _timedOut   = false;
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
            // the deadline has passed once nothing dated up to it can still arrive; held-back events must be judged first
            if (!_armedBy.empty() && _events.watermark && _events.ready() == 0UZ && emitted < evtOutSpan.size()) {
                const std::uint64_t passed = *_events.watermark > _events.lateness ? *_events.watermark - _events.lateness : 0U;
                if (tooLate(since(passed, _armedAt))) {
                    expire(evtOutSpan, emitted);
                }
            }
            n_pending = static_cast<gr::Size_t>(_events.size() + (_armedBy.empty() ? 0UZ : 1UZ));
        }
        evtOutSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    /** signed: an event can arrive later than one dated after it */
    [[nodiscard]] static std::int64_t since(std::uint64_t at, std::uint64_t armedAt) noexcept { return static_cast<std::int64_t>(at) - static_cast<std::int64_t>(armedAt); }

    [[nodiscard]] bool tooEarly(std::int64_t elapsedNs) const noexcept {
        const auto minimum = static_cast<std::int64_t>(detail::nanoseconds(min_delay));
        const bool open    = _bounds == DelayBounds::open || _bounds == DelayBounds::minOpen;
        return open ? elapsedNs <= minimum : elapsedNs < minimum;
    }

    [[nodiscard]] bool tooLate(std::int64_t elapsedNs) const noexcept {
        if (max_delay <= 0.) {
            return false;
        }
        const auto maximum = static_cast<std::int64_t>(detail::nanoseconds(max_delay));
        const bool open    = _bounds == DelayBounds::open || _bounds == DelayBounds::maxOpen;
        return open ? elapsedNs >= maximum : elapsedNs > maximum;
    }

    /** keeps of the A only what a report refers to, formatted to exactly one held slot, so any A arms */
    void arm(const gr::property_map_view& event, std::uint64_t at) const {
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> blob{};
        gr::property_map_view                                                             armedTuple = gr::formatEvent(blob, detail::kReferenceKeys);
        std::ignore                                                                                  = detail::writeReference(armedTuple, event);
        std::ignore                                                                                  = _armedBy.copy(armedTuple.blob());
        _armedAt                                                                                     = at;
    }

    /** at most one report per event: an expiry leaves no A for a B to follow */
    void judge(const gr::property_map_view& event, auto& evtOutSpan, std::size_t& emitted) const {
        const auto stamp = event.template get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
        if (!stamp) {
            n_undated = n_undated + 1U;
            return; // an undated event can neither arm, nor follow, nor show that time has passed
        }
        const std::uint64_t at = referenceTimeOf(event, *stamp, _reference);
        if (!_armedBy.empty() && tooLate(since(at, _armedAt))) {
            expire(evtOutSpan, emitted);
        }

        if (_hasDisarm && detail::matches(_disarming, event)) {
            if (!_armedBy.empty()) {
                n_disarmed = n_disarmed + 1U;
            }
            _armedBy.clear();
            return;
        }
        if (_hasArm && detail::matches(_arming, event)) {
            if (_armedBy.empty() || _restarts) {
                arm(event, at);
            }
            return;
        }
        if (_hasTrigger && detail::matches(_triggering, event)) {
            if (!_armedBy.empty() && tooEarly(since(at, _armedAt))) { // a B dated before its A is early too
                n_ignored = n_ignored + 1U;
                return;
            }
            if (!_armedBy.empty()) {
                if (_report != Report::timeout) {
                    const auto withA = [&](gr::property_map_view& details) { return detail::writeReference(details, _armedBy.view()); };
                    if (gr::emit(evtOutSpan, emitted, detail::kEventKeys, [&](gr::property_map_view& sequence) { return detail::writeEvent(sequence, trigger_name.value, event, detail::kReferenceKeys, withA); }).has_value()) {
                        ++emitted;
                    }
                }
                n_sequences = n_sequences + 1U;
            } else if (!_watches) { // watching: a B after a timeout is the recovery, not an ignored B
                n_ignored = n_ignored + 1U;
            }
            _armedBy.clear();
            _timedOut = false;
            if (_watches) { // the next window opens at this B
                arm(event, at);
            }
            return;
        }
        if (_watches && _armedBy.empty() && !_timedOut) { // the first dated event, of any name, opens the first window
            arm(event, at);
        }
    }

    void expire(auto& evtOutSpan, std::size_t& emitted) const {
        if (_report != Report::sequence) {
            const std::uint64_t deadline = _armedAt + detail::nanoseconds(max_delay);
            const auto          withA    = [&](gr::property_map_view& details) { return detail::writeReference(details, _armedBy.view()); };
            const auto          dated    = [&](gr::property_map_view& timeout) { return detail::writeEvent(timeout, timeout_name.value, gr::property_map_view{}, detail::kReferenceKeys, withA) && timeout.try_emplace(gr::tag::kTriggerTime.key(), deadline); };
            if (gr::emit(evtOutSpan, emitted, detail::kEventKeys, dated).has_value()) {
                ++emitted;
            }
        }
        n_timeouts = n_timeouts + 1U;
        _armedBy.clear();
        _timedOut = true;
    }
};

enum class BuilderKey : std::uint8_t { timestamp, eventId };
enum class IncompleteEvent : std::uint8_t { drop, emitPartial };

GR_REGISTER_BLOCK(gr::blocks::trigger::EventBuilder, [T], [ int16_t, int32_t, float, double ])

template<typename T>
struct EventBuilder : gr::Block<EventBuilder<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief assemble the fragments of one event, matched by timestamp or identifier, into one DataSet [zip, by carried time]

    in#0  ──F(t0)──────────F(t1)──▶   zip, but by carried time
    in#1  ────F(t0+20ns)─────F(t1)─▶  tolerance = 50 ns
    out   ──────E(t0)──────────E(t1)─▶

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

    A<gr::Size_t, "n events">                                                               n_events     = 0U;
    A<gr::Size_t, "n incomplete", Doc<"events judged with fragments missing">>              n_incomplete = 0U;
    A<gr::Size_t, "n dropped", Doc<"incomplete events not published">>                      n_dropped    = 0U;
    A<gr::Size_t, "n mismatched", Doc<"fragments refused because their extents disagreed">> n_mismatched = 0U;
    A<gr::Size_t, "n unkeyed", Doc<"fragments carrying nothing to match on">>               n_unkeyed    = 0U;
    A<gr::Size_t, "n pending", Doc<"events being assembled">>                               n_pending    = 0U;

    GR_MAKE_REFLECTABLE(EventBuilder, in, evtOut, out, n_inputs, key, tolerance, timeout, on_incomplete, max_pending, n_events, n_incomplete, n_dropped, n_mismatched, n_unkeyed, n_pending);

    struct Assembling {
        std::int64_t                               at = 0;
        std::vector<std::optional<gr::DataSet<T>>> fragments;

        [[nodiscard]] std::size_t held() const {
            return static_cast<std::size_t>(std::ranges::count_if(fragments, [](const auto& fragment) { return fragment.has_value(); }));
        }
    };

    std::deque<Assembling>     _pending;
    detail::PendingEvents      _reports;
    std::deque<gr::DataSet<T>> _assembled;
    BuilderKey                 _key          = BuilderKey::timestamp;
    IncompleteEvent            _onIncomplete = IncompleteEvent::drop;

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        _key          = key.value == "event_id" ? BuilderKey::eventId : BuilderKey::timestamp;
        _onIncomplete = on_incomplete.value == "emit_partial" ? IncompleteEvent::emitPartial : IncompleteEvent::drop;
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
        n_pending = static_cast<gr::Size_t>(_pending.size());

        const std::size_t emitted  = detail::drainInto(_assembled, outSpan);
        const std::size_t reported = _reports.drainInto(evtOutSpan, 0UZ, *this);
        evtOutSpan.publish(reported);
        outSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] std::optional<std::int64_t> keyOf(const gr::DataSet<T>& fragment) const {
        if (_key == BuilderKey::eventId) {
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
            _pending.push_back(Assembling{.at = *at, .fragments = std::vector<std::optional<gr::DataSet<T>>>(in.size())});
            event = std::prev(_pending.end());
        }
        if (channel >= event->fragments.size()) {
            return;
        }
        if (!fits(*event, fragment)) {
            n_mismatched = n_mismatched + 1U;
            detail::reportError(*this, "processBulk()", "a fragment's extents disagreed with the event's; a DataSet carries one extent per signal");
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
        const std::int64_t window = static_cast<std::int64_t>(detail::nanoseconds(tolerance));
        return std::ranges::find_if(_pending, [&](const Assembling& event) {
            const std::int64_t apart = event.at > at ? event.at - at : at - event.at;
            return _key == BuilderKey::eventId ? event.at == at : apart <= window;
        });
    }

    void erase(std::int64_t at) {
        const auto gone = std::ranges::find_if(_pending, [at](const Assembling& event) { return event.at == at; });
        if (gone != _pending.end()) {
            _pending.erase(gone);
        }
    }

    void judgeOlderThan(std::int64_t at) {
        if (timeout <= 0. || _key == BuilderKey::eventId) {
            return;
        }
        const std::int64_t horizon = static_cast<std::int64_t>(detail::nanoseconds(timeout));
        while (!_pending.empty() && at - _pending.front().at > horizon) {
            judge(_pending.front());
            _pending.pop_front();
        }
    }

    void judge(Assembling& event) {
        n_incomplete = n_incomplete + 1U;
        if (_onIncomplete != IncompleteEvent::emitPartial) {
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
