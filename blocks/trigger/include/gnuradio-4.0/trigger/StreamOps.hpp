#ifndef GNURADIO_TRIGGER_STREAMOPS_HPP
#define GNURADIO_TRIGGER_STREAMOPS_HPP

#include <algorithm>
#include <deque>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/ConditionSource.hpp>
#include <gnuradio-4.0/trigger/EventReduce.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/SamplePredicate.hpp>
#include <gnuradio-4.0/trigger/Segments.hpp>
#include <limits>
#include <memory_resource>
#include <optional>
#include <ranges>
#include <string>
#include <vector>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::Mux, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct Mux : gr::Block<Mux<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief forward the input its context names, and put the states back into one stream [merge, by context]

    in#0    ─a─a─a────────────▶    merge, but by context
    in#1    ────────b─b─b─────▶    contexts = RAMP, FLATTOP
    evtIn   ─RAMP───FLATTOP───▶
    out     ─a─a─a──b─b─b─────▶

Rx `merge` interleaves by arrival; this selects, so exactly one input is live at a time and the rest are refused.

 [1] example: https://rxmarbles.com/#merge
 [2] detailed documentation: https://reactivex.io/documentation/operators/merge.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                       evtIn;
    std::vector<gr::PortIn<T, gr::Async>> in;
    gr::PortOut<T>                        out;

    A<std::vector<std::string>, "contexts", Doc<"one context per input, in input order">> contexts;
    A<bool, "back pressure", Doc<"true: do not consume from un-selected inputs">>         back_pressure = false;

    A<gr::Size_t, "n switches", Doc<"context changes acted on">>              n_switches  = 0U;
    A<gr::Size_t, "n dropped", Doc<"samples read from an un-selected input">> n_dropped   = 0U;
    A<gr::Size_t, "n unmatched", Doc<"contexts naming no input">>             n_unmatched = 0U;

    GR_MAKE_REFLECTABLE(Mux, evtIn, in, out, contexts, back_pressure, n_switches, n_dropped, n_unmatched);

    constexpr static std::size_t kNothingSelected = std::numeric_limits<std::size_t>::max();

    std::size_t _selected = kNothingSelected;

    void start() { _selected = kNothingSelected; }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("contexts") && oldSettings.find_value("contexts") != newSettings.find_value("contexts")) {
            in.resize(contexts.value.size());
            _selected = kNothingSelected;
        }
    }

    template<gr::InputSpanLike TInput>
    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, std::span<TInput>& ins, gr::OutputSpanLike auto& outSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (!event.empty()) {
                select(event);
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        const std::size_t forwarded = _selected < ins.size() ? forward(ins[_selected], outSpan) : 0UZ;
        for (std::size_t channel = 0UZ; channel < ins.size(); ++channel) {
            if (channel != _selected) {
                discard(ins[channel]);
            }
        }

        outSpan.publish(forwarded);
        return gr::work::Status::OK;
    }

private:
    void select(const gr::property_map_view& event) {
        const auto named = event.template get_if<std::string_view>(std::string_view{gr::tag::CONTEXT.key()});
        if (!named) {
            return;
        }
        const auto found = std::ranges::find(contexts.value, *named);
        if (found == contexts.value.end()) {
            n_unmatched = n_unmatched + 1U;
            return;
        }
        const std::size_t wanted = static_cast<std::size_t>(std::ranges::distance(contexts.value.begin(), found));
        if (wanted != _selected) {
            _selected  = wanted;
            n_switches = n_switches + 1U;
        }
    }

    [[nodiscard]] std::size_t forward(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples = std::min(inSpan.size(), outSpan.size());
        std::ranges::copy(inSpan | std::views::take(nSamples), outSpan.begin());
        for (const auto& tag : inSpan.rawTags()) {
            if (tag.index >= inSpan.streamIndex && tag.index - inSpan.streamIndex < nSamples) {
                outSpan.publishTag(tag.map, tag.index - inSpan.streamIndex);
            }
        }
        std::ignore = inSpan.consume(nSamples);
        return nSamples;
    }

    void discard(gr::InputSpanLike auto& inSpan) {
        if (back_pressure) {
            std::ignore = inSpan.consume(0UZ);
            return;
        }
        n_dropped   = n_dropped + static_cast<gr::Size_t>(inSpan.size());
        std::ignore = inSpan.consume(inSpan.size());
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::Pairwise, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct Pairwise : gr::Block<Pairwise<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief every sample beside the one before it, on two aligned outputs [pairwise]

    in        ─a──b──c──d─▶       pairwise
    previous  ────a──b──c─▶       the first sample has no predecessor, so no pair starts there
    current   ────b──c──d─▶

Rx pairs the two into one value; a GR4 stream carries one sample, so the pair travels as two outputs a downstream block reads
together.

 [1] example: https://rxmarbles.com/#pairwise
 [2] detailed documentation: https://reactivex.io/documentation/operators/pairwise.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>  in;
    gr::PortOut<T> previous;
    gr::PortOut<T> current;

    A<gr::Size_t, "n pairs", Doc<"pairs published">> n_pairs = 0U;

    GR_MAKE_REFLECTABLE(Pairwise, in, previous, current, n_pairs);

    T    _previous = T{};
    bool _started  = false;

    void start() {
        _previous = T{};
        _started  = false;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& previousSpan, gr::OutputSpanLike auto& currentSpan) {
        const std::size_t room     = std::min(previousSpan.size(), currentSpan.size());
        const std::size_t nSamples = std::min(inSpan.size(), _started ? room : room + 1UZ);
        if (nSamples == 0UZ) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }
        const auto conditions = detail::collectConditions(inSpan, nSamples);

        std::size_t emitted = 0UZ;
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            const T sample = inSpan[i];
            if (_started) {
                previousSpan[emitted] = _previous;
                currentSpan[emitted]  = sample;
                if (const gr::property_map* tag = conditions.tagAt(i)) {
                    currentSpan.publishTag(*tag, emitted);
                }
                ++emitted;
            }
            _previous = sample;
            _started  = true;
        }

        n_pairs = n_pairs + static_cast<gr::Size_t>(emitted);
        previousSpan.publish(emitted);
        currentSpan.publish(emitted);
        if (!inSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::SampleFilter, [T], [ int16_t, int32_t, float, double ])

template<typename T>
struct SampleFilter : gr::Block<SampleFilter<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief pass the samples that satisfy a test, drop the rest [filter]

    in   ─2──30──22──5──60──1─▶    filter
    out  ────30──22────60─────▶    predicate = greater, threshold = 10

`EventFilter` does this to events; this does it to samples, and both spell the comparison the same way.

 [1] example: https://rxmarbles.com/#filter
 [2] detailed documentation: https://reactivex.io/documentation/operators/filter.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    A<std::pmr::string, "predicate", Doc<"greater|greater_equal|less|less_equal|equal|not_equal">> predicate = std::pmr::string("greater");
    A<double, "threshold", Doc<"the value a sample is compared against">>                          threshold = 0.;
    A<std::pmr::string, "expression", Doc<"ExprTk in x and threshold, empty = use 'predicate'">>   expression;

    A<gr::Size_t, "n passed", Doc<"samples forwarded">>             n_passed  = 0U;
    A<gr::Size_t, "n dropped", Doc<"samples that failed the test">> n_dropped = 0U;

    GR_MAKE_REFLECTABLE(SampleFilter, in, out, predicate, threshold, expression, n_passed, n_dropped);

    detail::SamplePredicate<T> _test;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if (const auto refused = _test.configure(predicate.value, expression.value, static_cast<T>(threshold))) {
            gr::log::warning("SampleFilter: {}", *refused);
        }
        n_passed  = 0U;
        n_dropped = 0U;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples   = std::min(inSpan.size(), outSpan.size());
        const auto        conditions = detail::collectConditions(inSpan, nSamples);

        std::size_t emitted = 0UZ;
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            if (!_test(inSpan[i])) {
                n_dropped = n_dropped + 1U;
                continue;
            }
            if (const gr::property_map* tag = conditions.tagAt(i)) {
                outSpan.publishTag(*tag, emitted);
            }
            outSpan[emitted++] = inSpan[i];
            n_passed           = n_passed + 1U;
        }

        outSpan.publish(emitted);
        if (!inSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::Scan, [T], [ int16_t, int32_t, int64_t, float, double ])

template<typename T>
struct Scan : gr::Block<Scan<T>> {
    using Description = Doc<R"(@brief accumulate as the samples arrive, publishing the running value for each [scan]

    in   ─1──2──3───R:1──2─▶       scan
    out  ─1──3──6───1────3─▶       operation = sum, reset_filter = "R"

`Accumulate` publishes once per segment, this publishes on every sample, and the two spell the operation the same way.

 [1] example: https://rxmarbles.com/#scan
 [2] detailed documentation: https://reactivex.io/documentation/operators/scan.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    A<std::pmr::string, "operation", Doc<"last|sum|product|minimum|maximum|mean|count">>    operation = std::pmr::string("sum");
    A<std::pmr::string, "reset filter", Doc<"trigger filter restarting it, empty = never">> reset_filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                         match_mode;

    A<gr::Size_t, "n resets", Doc<"restarts acted on">> n_resets = 0U;

    GR_MAKE_REFLECTABLE(Scan, in, out, operation, reset_filter, match_mode, n_resets);

    Accumulation       _mode = Accumulation::sum;
    detail::MatchState _reset{};
    bool               _resettable = false;
    T                  _folded     = T{};
    double             _sum        = 0.;
    std::size_t        _nFolded    = 0UZ;

    void start() { restart(); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _mode       = foldNamed(operation.value);
        _resettable = detail::compileOptionalFilter(reset_filter.value, match_mode.value, _reset, "Scan");
        restart();
        n_resets = 0U;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples = std::min(inSpan.size(), outSpan.size());

        auto foldRun = [&](std::size_t from, std::size_t until) {
            for (std::size_t i = from; i < until; ++i) {
                fold(inSpan[i]);
                outSpan[i] = runningValue();
            }
        };
        auto restartOn = [&](const gr::property_map_view& tag, std::size_t /*at*/) {
            if (_resettable && detail::matches(_reset, tag)) {
                restart();
                n_resets = n_resets + 1U;
            }
        };
        detail::forEachSegment(inSpan, nSamples, foldRun, restartOn);

        outSpan.publish(nSamples);
        if (!inSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
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
        ++_nFolded;
    }

    [[nodiscard]] T runningValue() const {
        switch (_mode) {
        case Accumulation::mean: return static_cast<T>(_sum / static_cast<double>(_nFolded));
        case Accumulation::count: return static_cast<T>(_nFolded);
        default: return _folded; // last, sum, product, minimum, maximum
        }
    }

    void restart() {
        _folded  = T{};
        _sum     = 0.;
        _nFolded = 0UZ;
    }
};

GR_REGISTER_BLOCK("gr::blocks::trigger::TakeLast", gr::blocks::trigger::Tail, ([T], true), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SkipLast", gr::blocks::trigger::Tail, ([T], false), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T, bool keepsTheTail>
struct Tail : gr::Block<Tail<T, keepsTheTail>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief keep the last n samples of a segment, or everything but them [takeLast, skipLast]

    in        ─a──b──c──d──E─▶
    TakeLast  ───────────c─d─▶     n = 2, segment ends at E
    SkipLast  ─a──b──────────▶

The tail of a run is only known once the run has ended, so `TakeLast` holds n samples back and publishes them where the trigger
that ended the segment sits; `SkipLast` releases a sample as soon as n newer ones exist and drops what is still held when the
segment ends.

 [1] example: https://rxmarbles.com/#takeLast, https://rxmarbles.com/#skipLast
 [2] detailed documentation: https://reactivex.io/documentation/operators/takelast.html, https://reactivex.io/documentation/operators/skiplast.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>             in;
    gr::PortOut<T, gr::Async> out;

    A<std::pmr::string, "filter", Doc<"trigger filter ending a segment, empty = at the stream's end">> filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                                    match_mode;
    A<gr::Size_t, "n", Doc<"length of the tail, in samples">>                                          n = 1U;

    A<gr::Size_t, "n passed", Doc<"samples forwarded">>           n_passed   = 0U;
    A<gr::Size_t, "n dropped", Doc<"samples the tail rule lost">> n_dropped  = 0U;
    A<gr::Size_t, "n segments", Doc<"segments ended">>            n_segments = 0U;

    GR_MAKE_REFLECTABLE(Tail, in, out, filter, match_mode, n, n_passed, n_dropped, n_segments);

    detail::MatchState   _ends{};
    bool                 _filtered = false;
    gr::HistoryBuffer<T> _held{1UZ};
    std::deque<T>        _ready;

    void start() {
        _held.reset();
        _ready.clear();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered = detail::compileOptionalFilter(filter.value, match_mode.value, _ends, keepsTheTail ? "TakeLast" : "SkipLast");
        if (n == 0U) {
            n = 1U;
        }
        _held      = gr::HistoryBuffer<T>(static_cast<std::size_t>(n));
        n_passed   = 0U;
        n_dropped  = 0U;
        n_segments = 0U;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        if (!_ready.empty() && outSpan.size() == 0UZ) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        const std::optional<std::size_t> untilEoS = samples_to_eos_tag(in);
        const std::size_t                nSamples = admissible(inSpan.size(), outSpan.size());

        auto holdRun = [&](std::size_t from, std::size_t until) {
            for (std::size_t i = from; i < until; ++i) {
                hold(inSpan[i]);
            }
        };
        auto endOn = [&](const gr::property_map_view& tag, std::size_t /*at*/) {
            if (_filtered && detail::matches(_ends, tag)) {
                closeSegment();
            }
        };
        detail::forEachSegment(inSpan, nSamples, holdRun, endOn);

        const bool streamEndsHere = untilEoS.has_value() && *untilEoS <= nSamples;
        if (streamEndsHere) {
            closeSegment();
        }

        const std::size_t emitted = detail::drainInto(_ready, outSpan);
        n_passed                  = n_passed + static_cast<gr::Size_t>(emitted);

        outSpan.publish(emitted);
        if (!inSpan.consume(nSamples)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

    [[nodiscard]] std::size_t admissible(std::size_t available, std::size_t room) const {
        if constexpr (keepsTheTail) {
            return available;
        } else {
            const std::size_t stillFilling = static_cast<std::size_t>(n) - std::min(_held.size(), static_cast<std::size_t>(n));
            return std::min(available, room + stillFilling);
        }
    }

    void hold(const T& sample) {
        if (_held.size() == static_cast<std::size_t>(n)) {
            if constexpr (keepsTheTail) {
                n_dropped = n_dropped + 1U;
            } else {
                _ready.push_back(_held[0UZ]);
            }
        }
        _held.push_back(sample);
    }

    void closeSegment() {
        if constexpr (keepsTheTail) {
            _ready.insert(_ready.end(), _held.begin(), _held.end());
        } else {
            n_dropped = n_dropped + static_cast<gr::Size_t>(_held.size());
        }
        _held.reset();
        n_segments = n_segments + 1U;
    }
};

template<typename T>
using TakeLast = Tail<T, true>;

template<typename T>
using SkipLast = Tail<T, false>;

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_STREAMOPS_HPP
