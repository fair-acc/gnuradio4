#ifndef GNURADIO_TRIGGER_STREAMOPS_HPP
#define GNURADIO_TRIGGER_STREAMOPS_HPP

#include <algorithm>
#include <array>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
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

GR_REGISTER_BLOCK(gr::blocks::trigger::Pairwise, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct Pairwise : gr::Block<Pairwise<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief every sample beside the one before it, on two aligned outputs [pairwise]

    in        ─a──b──c──d─▶       pairwise
    previous  ────a──b──c─▶       the first sample has no predecessor, so no pair starts there
    current   ────b──c──d─▶

 [1] example: https://rxmarbles.com/#pairwise
 [2] detailed documentation: https://reactivex.io/documentation/operators/pairwise.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>  in;
    gr::PortOut<T> previous;
    gr::PortOut<T> current;

    mutable A<gr::Size_t, "n pairs", Doc<"pairs published">> n_pairs = 0U;

    GR_MAKE_REFLECTABLE(Pairwise, in, previous, current, n_pairs);

    mutable T    _previous = T{};
    mutable bool _started  = false;

    void start() {
        _previous = T{};
        _started  = false;
    }

    void reset() {
        _previous = T{};
        _started  = false;
        n_pairs   = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& previousSpan, gr::OutputSpanLike auto& currentSpan) const {
        const std::size_t room     = std::min(previousSpan.size(), currentSpan.size());
        const std::size_t nSamples = std::min(inSpan.size(), _started ? room : room + 1UZ);
        if (nSamples == 0UZ) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        auto        tags    = inSpan.rawTags();
        auto        next    = std::ranges::begin(tags);
        std::size_t emitted = 0UZ;
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            const T sample = inSpan[i];
            for (; next != std::ranges::end(tags) && (*next).index <= inSpan.streamIndex + i; ++next) {
                if (_started && (*next).index == inSpan.streamIndex + i) {
                    currentSpan.publishTag(gr::property_map_view{(*next).map}, emitted);
                }
            }
            if (_started) {
                previousSpan[emitted] = _previous;
                currentSpan[emitted]  = sample;
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

 [1] example: https://rxmarbles.com/#filter
 [2] detailed documentation: https://reactivex.io/documentation/operators/filter.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    A<std::pmr::string, "predicate", Doc<"greater|greaterEqual|less|lessEqual|equal|notEqual">>  predicate = std::pmr::string("greater");
    A<double, "threshold", Doc<"the value a sample is compared against">>                        threshold = 0.;
    A<std::pmr::string, "expression", Doc<"ExprTk in x and threshold, empty = use 'predicate'">> expression;

    mutable A<gr::Size_t, "n passed", Doc<"samples forwarded">>             n_passed  = 0U;
    mutable A<gr::Size_t, "n dropped", Doc<"samples that failed the test">> n_dropped = 0U;

    Comparison _comparison = Comparison::greater;

    GR_MAKE_REFLECTABLE(SampleFilter, in, out, predicate, threshold, expression, n_passed, n_dropped, _comparison);

    mutable detail::SamplePredicate<T> _test;
    detail::ErrorMemory                _deferredError{};

    void start() { detail::sendDeferredError(*this); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if (const auto refused = _test.configure(predicate.value, expression.value, static_cast<T>(threshold))) {
            detail::reportError(*this, "settingsChanged()", *refused);
        }
        _comparison = _test.comparison;
    }

    void reset() {
        n_passed  = 0U;
        n_dropped = 0U;
    }

    [[nodiscard]] std::string_view hostOnlyReason() const noexcept { return _test.hostOnlyReason(); }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool    kOnHost  = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        const std::size_t nSamples = std::min(inSpan.size(), outSpan.size());
        auto              tags     = inSpan.rawTags();
        auto              next     = std::ranges::begin(tags);
        std::size_t       emitted  = 0UZ;
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            const auto here = next;
            while (next != std::ranges::end(tags) && (*next).index <= inSpan.streamIndex + i) {
                ++next;
            }
            const bool passes = detail::passes<kOnHost>(_test, _comparison, inSpan[i], static_cast<T>(threshold));
            if (!passes) {
                n_dropped = n_dropped + 1U;
                continue;
            }
            for (auto tag = here; tag != next; ++tag) {
                if ((*tag).index == inSpan.streamIndex + i) {
                    outSpan.publishTag(gr::property_map_view{(*tag).map}, emitted);
                }
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

    mutable A<gr::Size_t, "n resets", Doc<"restarts acted on">> n_resets = 0U;

    Accumulation       _mode = Accumulation::sum;
    detail::MatchState _reset{};
    bool               _resettable     = false;
    gr::Size_t         _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Scan, in, out, operation, reset_filter, match_mode, n_resets, _mode, _reset, _resettable, _settings_epoch);

    mutable Fold<T>            _fold{};
    mutable detail::MatchState _resetMatching{};
    mutable gr::Size_t         _stateEpoch = std::numeric_limits<gr::Size_t>::max();

    void start() { _stateEpoch = std::numeric_limits<gr::Size_t>::max(); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _mode           = foldNamed(operation.value);
        _resettable     = detail::compileOptionalFilter(*this, reset_filter.value, match_mode.value, _reset);
        _settings_epoch = _settings_epoch + 1U;
    }

    void reset() {
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
        n_resets    = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) const {
        if (_stateEpoch != _settings_epoch) {
            _fold          = Fold<T>{};
            _resetMatching = _reset;
            _stateEpoch    = _settings_epoch;
        }
        const std::size_t nSamples = std::min(inSpan.size(), outSpan.size());

        auto foldRun = [&](std::size_t from, std::size_t until) {
            for (std::size_t i = from; i < until; ++i) {
                _fold.add(inSpan[i], _mode);
                outSpan[i] = _fold.value(_mode);
            }
        };
        auto restartOn = [&](const gr::property_map_view& tag, std::size_t /*at*/) {
            if (_resettable && detail::matches(_resetMatching, tag)) {
                _fold    = Fold<T>{};
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
};

GR_REGISTER_BLOCK("gr::blocks::trigger::TakeLast", gr::blocks::trigger::Tail, ([T], true), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::SkipLast", gr::blocks::trigger::Tail, ([T], false), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T, bool keepsTheTail>
struct Tail : gr::Block<Tail<T, keepsTheTail>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief keep the last n samples of a segment, or everything but them [takeLast, skipLast]

    in        ─a──b──c──d──E─▶
    TakeLast  ───────────c─d─▶     n = 2, segment ends at E
    SkipLast  ─a──b──────────▶

 [1] example: https://rxmarbles.com/#takeLast, https://rxmarbles.com/#skipLast
 [2] detailed documentation: https://reactivex.io/documentation/operators/takelast.html, https://reactivex.io/documentation/operators/skiplast.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>             in;
    gr::PortOut<T, gr::Async> out;

    A<std::pmr::string, "filter", Doc<"trigger filter ending a segment, empty = at the stream's end">>                                         filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                                                                            match_mode;
    A<gr::Size_t, "n", Doc<"length of the tail, in samples; 1 to 64: TakeLast publishes it where the ending trigger sits, SkipLast drops it">> n = 1U;

    mutable A<gr::Size_t, "n passed", Doc<"samples forwarded">>           n_passed   = 0U;
    mutable A<gr::Size_t, "n dropped", Doc<"samples the tail rule lost">> n_dropped  = 0U;
    mutable A<gr::Size_t, "n segments", Doc<"segments ended">>            n_segments = 0U;

    constexpr static std::size_t kMaxTail  = 64UZ;
    constexpr static std::size_t kMaxReady = 2UZ * kMaxTail;

    detail::MatchState _ends{};
    bool               _filtered       = false;
    gr::Size_t         _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Tail, in, out, filter, match_mode, n, n_passed, n_dropped, n_segments, _ends, _filtered, _settings_epoch);

    /** the kernel's own between work calls: two rings, oldest at `first` */
    mutable std::array<T, kMaxTail>  _held{};
    mutable std::size_t              _heldFirst = 0UZ;
    mutable std::size_t              _nHeld     = 0UZ;
    mutable std::array<T, kMaxReady> _ready{};
    mutable std::size_t              _readyFirst = 0UZ;
    mutable std::size_t              _nReady     = 0UZ;
    mutable detail::MatchState       _endsMatching{};
    mutable gr::Size_t               _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    detail::ErrorMemory              _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        _nHeld      = 0UZ;
        _nReady     = 0UZ;
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered = detail::compileOptionalFilter(*this, filter.value, match_mode.value, _ends);
        if (n == 0U || n > kMaxTail) {
            const std::string message = std::format("{}: n {} is outside 1 to {} and is clamped", keepsTheTail ? "TakeLast" : "SkipLast", n.value, kMaxTail);
            detail::reportError(*this, "settingsChanged()", message);
            n = std::clamp(n.value, gr::Size_t{1U}, static_cast<gr::Size_t>(kMaxTail));
        }
        _settings_epoch = _settings_epoch + 1U;
    }

    void reset() {
        _nHeld      = 0UZ;
        _nReady     = 0UZ;
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
        n_passed    = 0U;
        n_dropped   = 0U;
        n_segments  = 0U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) const {
        if (_stateEpoch != _settings_epoch) {
            _nHeld        = 0UZ;
            _endsMatching = _ends;
            _stateEpoch   = _settings_epoch;
        }
        if (_nReady > 0UZ && outSpan.size() == 0UZ) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        const std::size_t nSpan = admissible(inSpan.size(), outSpan.size());
        auto              tags  = inSpan.rawTags();
        auto              next  = std::ranges::begin(tags);
        std::size_t       nSeen = 0UZ;
        for (; nSeen < nSpan && _nReady + kMaxTail <= kMaxReady; ++nSeen) {
            const std::size_t at = inSpan.streamIndex + nSeen;
            for (; next != std::ranges::end(tags) && (*next).index <= at; ++next) {
                if ((*next).index == at && _filtered && detail::matches(_endsMatching, gr::property_map_view{(*next).map})) {
                    closeSegment();
                }
            }
            hold(inSpan[nSeen]);
        }
        const bool streamEnds = [&] {
            if constexpr (gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>) {
                return inSpan.streamEnds();
            } else {
                /** a read-only query the port API offers only non-const */
                return gr::endsStreamAt(const_cast<gr::PortIn<T>&>(in), inSpan.streamIndex + nSeen);
            }
        }();
        if (nSeen == inSpan.size() && streamEnds && _nReady + _nHeld <= kMaxReady) {
            closeSegment();
        }

        std::size_t emitted = 0UZ;
        for (; emitted < outSpan.size() && _nReady > 0UZ; ++emitted, _readyFirst = (_readyFirst + 1UZ) % kMaxReady, --_nReady) {
            outSpan[emitted] = _ready[_readyFirst];
        }
        n_passed = n_passed + static_cast<gr::Size_t>(emitted);

        outSpan.publish(emitted);
        if (!inSpan.consume(nSeen)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

    [[nodiscard]] std::size_t admissible(std::size_t available, std::size_t room) const {
        if constexpr (keepsTheTail) {
            return available;
        } else {
            const std::size_t stillFilling = static_cast<std::size_t>(n) - std::min(_nHeld, static_cast<std::size_t>(n));
            return std::min(available, room + stillFilling);
        }
    }

    void release(const T& sample) const {
        _ready[(_readyFirst + _nReady) % kMaxReady] = sample;
        ++_nReady;
    }

    void hold(const T& sample) const {
        if (_nHeld == static_cast<std::size_t>(n)) {
            if constexpr (keepsTheTail) {
                n_dropped = n_dropped + 1U;
            } else {
                release(_held[_heldFirst]);
            }
            _heldFirst = (_heldFirst + 1UZ) % kMaxTail;
            --_nHeld;
        }
        _held[(_heldFirst + _nHeld) % kMaxTail] = sample;
        ++_nHeld;
    }

    void closeSegment() const {
        if constexpr (keepsTheTail) {
            for (std::size_t k = 0UZ; k < _nHeld; ++k) {
                release(_held[(_heldFirst + k) % kMaxTail]);
            }
        } else {
            n_dropped = n_dropped + static_cast<gr::Size_t>(_nHeld);
        }
        _nHeld     = 0UZ;
        n_segments = n_segments + 1U;
    }
};

template<typename T>
using TakeLast = Tail<T, true>;

template<typename T>
using SkipLast = Tail<T, false>;

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_STREAMOPS_HPP
