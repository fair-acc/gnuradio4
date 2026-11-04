#ifndef GNURADIO_TRIGGER_EVENTSHAPING_HPP
#define GNURADIO_TRIGGER_EVENTSHAPING_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/Segments.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <limits>
#include <memory_resource>
#include <optional>
#include <ranges>
#include <span>
#include <string>

#include <utility>
#include <variant>
#include <vector>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK("gr::blocks::trigger::Debounce", gr::blocks::trigger::Burst, ([T], true), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::trigger::Throttle", gr::blocks::trigger::Burst, ([T], false), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T, bool keepsLast>
struct Burst : gr::Block<Burst<T, keepsLast>> {
    using Description = std::conditional_t<keepsLast, Doc<R"(@brief report one trigger per burst, the last one, after the burst has gone quiet [debounce]

    in      ─T─T─T─────────T──▶    debounce
    evtOut  ───────T─────────T─▶   quiet = 3 samples

 [1] example: https://rxmarbles.com/#debounce
 [2] detailed documentation: https://reactivex.io/documentation/operators/debounce.html
)">,
        Doc<R"(@brief report the first trigger of a burst and inhibit the rest for a period [throttle]

    in      ─T─T─T─────────T──▶    throttle
    evtOut  ─T─────────────T───▶   inhibit = 3 samples

 [1] example: https://rxmarbles.com/#throttle
 [2] detailed documentation: https://reactivex.io/documentation/operators/throttle.html
)">>;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::PortIn<T>    in;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortOut<T>   out;

    A<std::pmr::string, "filter", Doc<"trigger filter; empty = every trigger">>           filter;
    A<gr::Size_t, "n samples", Doc<"Debounce: quiet period, Throttle: inhibit, samples">> n_samples   = 0U;
    A<float, "timeout", Doc<"Debounce: quiet period, Throttle: inhibit, seconds">>        timeout     = 0.f;
    gr::SampleRate                                                                        sample_rate = 0.f;

    mutable A<gr::Size_t, "n items", Doc<"triggers seen">>                                               n_items      = 0U;
    mutable A<gr::Size_t, "n emitted", Doc<"triggers reported">>                                         n_emitted    = 0U;
    mutable A<gr::Size_t, "n suppressed", Doc<"triggers a later one replaced or the inhibit swallowed">> n_suppressed = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState _accept{};
    bool       _filtered       = false;
    gr::Size_t _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Burst, evtIn, in, evtOut, out, filter, n_samples, timeout, sample_rate, n_items, n_emitted, n_suppressed, _accept, _filtered, _settings_epoch);

    mutable MatchState                                                       _accepting{};
    mutable TimeBase                                                         _time;
    mutable std::conditional_t<keepsLast, detail::HeldEvent, std::monostate> _held{};
    mutable std::size_t                                                      _heldAt = 0UZ;
    mutable std::optional<std::size_t>                                       _lastEmittedAt;
    mutable gr::Size_t                                                       _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable EventStore                                                       _events;

    void start() { _stateEpoch = std::numeric_limits<gr::Size_t>::max(); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered = detail::compileOptionalFilter(*this, filter.value, std::string_view{}, _accept);
        _events.clear();
        _settings_epoch = _settings_epoch + 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtOutSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _accepting = _accept;
            _time.reset();
            _time.setRate(static_cast<double>(sample_rate));
            if constexpr (keepsLast) {
                _held.clear();
            }
            _lastEmittedAt.reset();
            _stateEpoch = _settings_epoch;
        }
        const std::size_t nSamples  = std::min(inSpan.size(), outSpan.size());
        std::size_t       nEmitted  = 0UZ;
        const auto        onTrigger = [&](const gr::property_map_view& trigger, std::size_t at) {
            if (!detail::acceptsTrigger(_accepting, _filtered, trigger)) {
                return true;
            }
            if constexpr (keepsLast) {
                return hold<kOnHost>(trigger, at, evtOutSpan, nEmitted);
            } else {
                const auto inhibit = resolveDuration(_time, n_samples, timeout);
                if (_lastEmittedAt.has_value() && inhibit.has_value() && at < *_lastEmittedAt + *inhibit) {
                    n_items      = n_items + 1U;
                    n_suppressed = n_suppressed + 1U;
                    return true;
                }
                if (!report(trigger, evtOutSpan, nEmitted)) {
                    return false;
                }
                n_items        = n_items + 1U;
                _lastEmittedAt = at;
                return true;
            }
        };

        const std::size_t walked = forEachTrigger<kOnHost>(_events, evtSpan, inSpan, nSamples, kOnHost ? std::numeric_limits<std::size_t>::max() : evtOutSpan.size(), onTrigger, [&](const gr::property_map_view& tags, std::size_t at) { std::ignore = _time.adopt(tags, at); });
        if constexpr (keepsLast) {
            std::ignore = releaseBefore(inSpan.streamIndex + walked, evtOutSpan, nEmitted);
        }

        for (std::size_t i = 0UZ; i < walked; ++i) {
            outSpan[i] = inSpan[i];
        }
        evtOutSpan.publish(nEmitted);
        outSpan.publish(walked);
        if (!inSpan.consume(walked)) {
            return gr::work::Status::ERROR;
        }
        return walked == 0UZ && nSamples > 0UZ ? gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS : gr::work::Status::OK;
    }

private:
    [[nodiscard]] bool report(const gr::property_map_view& event, auto& evtOutSpan, std::size_t& nEmitted) const {
        if (evtOutSpan.isConnected) {
            if (!gr::emit(evtOutSpan, nEmitted, event)) {
                return false;
            }
            ++nEmitted;
        }
        n_emitted = n_emitted + 1U;
        return true;
    }

    [[nodiscard]] bool releaseBefore(std::size_t position, auto& evtOutSpan, std::size_t& nEmitted) const
    requires keepsLast
    {
        if (_held.empty()) {
            return true;
        }
        const auto quiet = resolveDuration(_time, n_samples, timeout);
        if (quiet.has_value() && position < _heldAt + *quiet) {
            return true;
        }
        if (!report(_held.view(), evtOutSpan, nEmitted)) {
            return false;
        }
        _held.clear();
        return true;
    }

    template<bool onHost>
    [[nodiscard]] bool hold(const gr::property_map_view& trigger, std::size_t at, auto& evtOutSpan, std::size_t& nEmitted) const
    requires keepsLast
    {
        if (!releaseBefore(at, evtOutSpan, nEmitted)) {
            return false;
        }
        n_items = n_items + 1U;
        if (!_held.empty()) {
            n_suppressed = n_suppressed + 1U;
        }
        if (!_held.template assign<onHost>(trigger)) {
            _held.clear();
            if constexpr (onHost) {
                detail::reportError(*this, "processBulk()", detail::kEventTooLarge);
            }
            return true;
        }
        _heldAt = at;
        return true;
    }
};

template<typename T>
using Debounce = Burst<T, true>;

template<typename T>
using Throttle = Burst<T, false>;

GR_REGISTER_BLOCK(gr::blocks::trigger::DelayWhen, [T], [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])

template<typename T>
struct DelayWhen : gr::Block<DelayWhen<T>> {
    using Description = Doc<R"(@brief delay each item by its own delay, from a parallel input, and emit them as they fall due [delayWhen]

    in        ─1──2──3─▶           delayWhen
    delayIn  ─4──2──0─▶           (samples)
    out       ─3──2──1─▶           due at 2, 3, 4

 [1] example: https://rxmarbles.com/#delayWhen
 [2] detailed documentation: https://reactivex.io/documentation/operators/delaywhen.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>             in;
    gr::PortIn<gr::Size_t>    delayIn;
    gr::PortOut<T, gr::Async> out;

    constexpr static std::size_t                                                           kMaxPending = 1024UZ;
    A<gr::Size_t, "max pending", Doc<"samples held pending">, gr::Limits<0U, kMaxPending>> max_pending = 1024U;

    mutable A<gr::Size_t, "n delayed", Doc<"samples that waited">>                              n_delayed  = 0U;
    mutable A<gr::Size_t, "n overflow", Doc<"samples forwarded in arrival order, buffer full">> n_overflow = 0U;
    mutable A<gr::Size_t, "n pending", Doc<"samples waiting to come due">>                      n_pending  = 0U;

    constexpr static std::string_view kPendingFull = "the samples waiting are at 'max_pending'; the order is arrival order from here";

    gr::Size_t _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(DelayWhen, in, delayIn, out, max_pending, n_delayed, n_overflow, n_pending, _settings_epoch);

    struct Waiting {
        T           sample;
        std::size_t dueAt;
    };

    /** the first `_nWaiting`, latest due first, so the next due is popped from the back */
    mutable std::array<Waiting, kMaxPending> _waiting{};
    mutable std::size_t                      _nWaiting   = 0UZ;
    mutable bool                             _full       = false;
    mutable gr::Size_t                       _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    detail::ErrorMemory                      _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) { _settings_epoch = _settings_epoch + 1U; }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& delaySpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _nWaiting   = 0UZ;
            _full       = false;
            _stateEpoch = _settings_epoch;
        }
        const std::size_t nSamples = std::min(inSpan.size(), delaySpan.size());
        std::size_t       emitted  = 0UZ;
        const auto        emitDue  = [&](std::size_t now) {
            while (emitted < outSpan.size() && _nWaiting > 0UZ && _waiting[_nWaiting - 1UZ].dueAt <= now) {
                outSpan[emitted++] = _waiting[--_nWaiting].sample;
            }
        };

        std::size_t consumed = 0UZ;
        for (; consumed < nSamples; ++consumed) {
            const std::size_t now = inSpan.streamIndex + consumed;
            emitDue(now);
            if (_nWaiting >= static_cast<std::size_t>(max_pending.value)) {
                if (emitted >= outSpan.size()) {
                    break;
                }
                if (!_full) {
                    _full = true;
                    if constexpr (kOnHost) {
                        detail::reportError(*this, "processBulk()", kPendingFull);
                    }
                }
                outSpan[emitted++] = inSpan[consumed];
                n_overflow         = n_overflow + 1U;
                continue;
            }
            _full                   = false;
            const std::size_t delay = static_cast<std::size_t>(delaySpan[consumed]);
            hold(inSpan[consumed], now + delay);
            if (delay > 0UZ) {
                n_delayed = n_delayed + 1U;
            }
        }
        emitDue(inSpan.streamIndex + consumed);
        n_pending = static_cast<gr::Size_t>(_nWaiting);

        outSpan.publish(emitted);
        if (!inSpan.consume(consumed) || !delaySpan.consume(consumed)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void hold(const T& sample, std::size_t dueAt) const {
        const auto waiting = std::span(_waiting.data(), _nWaiting);
        const auto at      = static_cast<std::size_t>(std::ranges::partition_point(waiting, [dueAt](const Waiting& held) { return held.dueAt > dueAt; }) - waiting.begin());
        std::ranges::move_backward(_waiting.begin() + static_cast<std::ptrdiff_t>(at), _waiting.begin() + static_cast<std::ptrdiff_t>(_nWaiting), _waiting.begin() + static_cast<std::ptrdiff_t>(_nWaiting + 1UZ));
        _waiting[at] = Waiting{.sample = sample, .dueAt = dueAt};
        ++_nWaiting;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::Distinct, [T], [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])
GR_REGISTER_BLOCK(gr::blocks::trigger::DistinctUntilChanged, [T], [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])

template<typename T>
struct Distinct : gr::Block<Distinct<T>> {
    using Description = Doc<R"(@brief emit each value the first time it is seen, within a bounded memory [distinct]

    in   ─1──2──1──3──2──4─▶       distinct
    out  ─1──2─────3─────4─▶       max_values bounds what is remembered

 [1] example: https://rxmarbles.com/#distinct
 [2] detailed documentation: https://reactivex.io/documentation/operators/distinct.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn           evtIn;
    gr::PortIn<T>             in;
    gr::PortOut<T, gr::Async> out;

    constexpr static std::size_t                                                                                                              kMaxValues = 1024UZ;
    A<gr::Size_t, "max values", Doc<"values remembered; past it everything is forwarded, counted in n_overflow">, gr::Limits<0U, kMaxValues>> max_values = 1024U;
    A<std::pmr::string, "segment filter", Doc<"clears memory on a trigger, empty = never">>                                                   segment_filter;

    mutable A<gr::Size_t, "n passed", Doc<"values seen for the first time">>                  n_passed     = 0U;
    mutable A<gr::Size_t, "n suppressed", Doc<"repeats of a value already seen">>             n_suppressed = 0U;
    mutable A<gr::Size_t, "n overflow", Doc<"samples forwarded because the memory was full">> n_overflow   = 0U;

    using MatchState                              = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;
    constexpr static std::string_view kMemoryFull = "the value memory is full; every sample is forwarded from here";

    MatchState _clears{};
    bool       _segmented      = false;
    gr::Size_t _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Distinct, evtIn, in, out, max_values, segment_filter, n_passed, n_suppressed, n_overflow, _clears, _segmented, _settings_epoch);

    /** sorted, the first `_nSeen` entries */
    mutable std::array<T, kMaxValues> _seen{};
    mutable std::size_t               _nSeen = 0UZ;
    mutable bool                      _full  = false;
    mutable MatchState                _clearing{};
    mutable gr::Size_t                _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    detail::ErrorMemory               _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _segmented      = detail::compileOptionalFilter(*this, segment_filter.value, std::string_view{}, _clears);
        _settings_epoch = _settings_epoch + 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _clearing   = _clears;
            _nSeen      = 0UZ;
            _full       = false;
            _stateEpoch = _settings_epoch;
        }
        for (const gr::property_map_view& event : evtSpan) {
            if (_segmented && !event.empty() && detail::matches(_clearing, event)) {
                forget();
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        std::size_t passed = 0UZ;
        std::size_t seen   = 0UZ;
        auto        tags   = inSpan.rawTags();
        auto        next   = std::ranges::begin(tags);
        for (; seen < inSpan.size() && passed < outSpan.size(); ++seen) {
            const auto first = detail::tagsOn(next, std::ranges::end(tags), inSpan.streamIndex + seen);
            for (auto tag = first; tag != next; ++tag) {
                if (_segmented && detail::matches(_clearing, gr::property_map_view{(*tag).map})) {
                    forget();
                }
            }
            if (admits<kOnHost>(inSpan[seen])) {
                outSpan[passed++] = inSpan[seen];
            }
        }

        outSpan.publish(passed);
        if (!inSpan.consume(seen)) {
            return gr::work::Status::ERROR;
        }
        return seen == 0UZ && inSpan.size() > 0UZ ? gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS : gr::work::Status::OK;
    }

private:
    void forget() const {
        _nSeen = 0UZ;
        _full  = false;
    }

    template<bool onHost>
    [[nodiscard]] bool admits(const T& value) const {
        if (_full) {
            n_overflow = n_overflow + 1U;
            return true;
        }
        const auto seen  = std::span<const T>(_seen.data(), _nSeen);
        const auto where = std::ranges::lower_bound(seen, value);
        if (where != seen.end() && *where == value) {
            n_suppressed = n_suppressed + 1U;
            return false;
        }
        if (_nSeen >= static_cast<std::size_t>(max_values.value)) {
            _full = true;
            if constexpr (onHost) {
                detail::reportError(*this, "processBulk()", kMemoryFull);
            }
            n_overflow = n_overflow + 1U;
            return true;
        }
        const auto at = static_cast<std::size_t>(where - seen.begin());
        std::ranges::move_backward(_seen.begin() + static_cast<std::ptrdiff_t>(at), _seen.begin() + static_cast<std::ptrdiff_t>(_nSeen), _seen.begin() + static_cast<std::ptrdiff_t>(_nSeen + 1UZ));
        _seen[at] = value;
        ++_nSeen;
        n_passed = n_passed + 1U;
        return true;
    }
};

template<typename T>
struct DistinctUntilChanged : gr::Block<DistinctUntilChanged<T>> {
    using Description = Doc<R"(@brief emit an item only where it differs from the one before it [distinctUntilChanged]

    in   ─1──1──2──2──1──3─▶       distinctUntilChanged
    out  ─1─────2─────1──3─▶

 [1] example: https://rxmarbles.com/#distinctUntilChanged
 [2] detailed documentation: https://reactivex.io/documentation/operators/distinctuntilchanged.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>             in;
    gr::PortOut<T, gr::Async> out;

    mutable A<gr::Size_t, "n passed", Doc<"samples forwarded">>                      n_passed     = 0U;
    mutable A<gr::Size_t, "n suppressed", Doc<"samples equal to their predecessor">> n_suppressed = 0U;

    GR_MAKE_REFLECTABLE(DistinctUntilChanged, in, out, n_passed, n_suppressed);

    mutable T    _previous = T{};
    mutable bool _started  = false;

    void start() { _started = false; }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) const {
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

 [1] example: https://rxmarbles.com/#repeat
 [2] detailed documentation: https://reactivex.io/documentation/operators/repeat.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn           evtIn;
    gr::PortIn<T>             in;
    gr::PortOut<T, gr::Async> out;

    A<gr::Size_t, "n repeats", Doc<"plays per segment">>                                n_repeats = 1U;
    constexpr static std::size_t                                                        kCapacity = 4096UZ;
    A<gr::Size_t, "capacity", Doc<"replay store, samples">, gr::Limits<0U, kCapacity>>  capacity  = 4096U;
    A<std::pmr::string, "segment filter", Doc<"ends the segment, empty = by capacity">> segment_filter;

    mutable A<gr::Size_t, "n segments", Doc<"segments repeated">>                                     n_segments = 0U;
    mutable A<gr::Size_t, "n overflow", Doc<"segments passed through once because they did not fit">> n_overflow = 0U;
    mutable A<gr::Size_t, "n captured", Doc<"samples in the segment being captured">>                 n_captured = 0U;

    using MatchState                            = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;
    constexpr static std::string_view kOutgrown = "a segment outgrew the replay store; it is passed through once instead";

    MatchState _ends{};
    bool       _segmented      = false;
    gr::Size_t _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(Repeat, evtIn, in, out, n_repeats, capacity, segment_filter, n_segments, n_overflow, n_captured, _ends, _segmented, _settings_epoch);

    mutable MatchState               _ending{};
    mutable std::array<T, kCapacity> _store{};
    mutable std::size_t              _nStored = 0UZ;
    mutable bool                     _tooLong = false;
    mutable std::array<T, kCapacity> _playing{};
    mutable std::size_t              _nPlaying   = 0UZ;
    mutable std::size_t              _position   = 0UZ;
    mutable gr::Size_t               _playsLeft  = 0U;
    mutable gr::Size_t               _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable EventStore               _events;
    detail::ErrorMemory              _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _segmented = detail::compileOptionalFilter(*this, segment_filter.value, std::string_view{}, _ends);
        _events.clear();
        _settings_epoch = _settings_epoch + 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _ending     = _ends;
            _nStored    = 0UZ;
            _tooLong    = false;
            _nPlaying   = 0UZ;
            _position   = 0UZ;
            _playsLeft  = 0U;
            _stateEpoch = _settings_epoch;
        }
        std::size_t emitted = 0UZ;
        const auto  play    = [&] {
            while (emitted < outSpan.size() && _playsLeft > 0U) {
                if (_position >= _nPlaying) {
                    _playsLeft = _playsLeft - 1U;
                    _position  = 0UZ;
                    if (_nPlaying == 0UZ) {
                        break;
                    }
                    continue;
                }
                outSpan[emitted++] = _playing[_position++];
            }
        };
        const auto endsSegment = [&](const gr::property_map_view& trigger) { return _segmented && detail::matches(_ending, trigger); };

        std::ignore = forEachEvent<kOnHost>(_events, evtSpan, std::numeric_limits<std::size_t>::max(), [&](const gr::property_map_view& event) {
            if (endsSegment(event)) {
                close();
            }
            return true;
        });

        play();
        const std::size_t                                                                 room = capacity == 0U ? 1UZ : static_cast<std::size_t>(capacity.value);
        auto                                                                              tags = inSpan.rawTags();
        auto                                                                              next = std::ranges::begin(tags);
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> scratch{};
        std::size_t                                                                       consumed = 0UZ;
        for (; consumed < inSpan.size(); ++consumed) {
            const auto first = detail::tagsOn(next, std::ranges::end(tags), inSpan.streamIndex + consumed);
            if (first != next && endsSegment(detail::mergedAt(first, next, scratch))) {
                close();
            }
            if (!_tooLong && _nStored >= room) {
                if (!_segmented) {
                    close();
                } else if (_playsLeft > 0U) {
                    break;
                } else {
                    _tooLong   = true;
                    n_overflow = n_overflow + 1U;
                    if constexpr (kOnHost) {
                        detail::reportError(*this, "processBulk()", kOutgrown);
                    }
                    replay(1U);
                    play();
                }
            }
            if (_tooLong) {
                if (_playsLeft > 0U || emitted >= outSpan.size()) {
                    break;
                }
                outSpan[emitted++] = inSpan[consumed];
                continue;
            }
            _store[_nStored++] = inSpan[consumed];
        }
        play();
        n_captured = static_cast<gr::Size_t>(_nStored);

        outSpan.publish(emitted);
        if (!inSpan.consume(consumed)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void close() const {
        if (_tooLong) {
            _nStored = 0UZ;
            _tooLong = false;
            return;
        }
        if (_nStored == 0UZ) {
            return;
        }
        replay(n_repeats == 0U ? 1U : n_repeats.value);
        n_segments = n_segments + 1U;
    }

    void replay(gr::Size_t times) const {
        std::ranges::copy(std::span<const T>(_store.data(), _nStored), _playing.begin());
        _nPlaying  = _nStored;
        _nStored   = 0UZ;
        _position  = 0UZ;
        _playsLeft = times;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::SequenceEqual, [T], [ int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double ])

template<typename T>
struct SequenceEqual : gr::Block<SequenceEqual<T>> {
    using Description = Doc<R"(@brief emit one truth value per segment: whether two streams carried the same items [sequenceEqual]

    in         ─1──2──9──4─▶       sequenceEqual
    reference  ─1──2──3──4─▶
    out        ──────────0─▶       0 = differ, 1 = equal (uint8_t: GR4 carries no bool stream)

 [1] example: https://rxmarbles.com/#sequenceEqual
 [2] detailed documentation: https://reactivex.io/documentation/operators/sequenceequal.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                      evtIn;
    gr::PortIn<T>                        in;
    gr::PortIn<T>                        reference;
    gr::PortOut<std::uint8_t, gr::Async> out;

    A<std::pmr::string, "segment filter", Doc<"trigger that ends a segment and publishes the verdict">> segment_filter;
    A<T, "tolerance", Doc<"largest difference still counted as equal">>                                 tolerance = T{};
    A<gr::Size_t, "n samples", Doc<"samples per segment, 0 = only on a trigger">>                       n_samples = 1024U;

    mutable A<gr::Size_t, "n verdicts", Doc<"truth values emitted">>                         n_verdicts   = 0U;
    mutable A<gr::Size_t, "n mismatches", Doc<"samples that differed, across all segments">> n_mismatches = 0U;
    mutable A<gr::Size_t, "n unjudged", Doc<"segments whose end the block never saw">>       n_unjudged   = 0U;

    using MatchState                                   = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;
    constexpr static std::string_view kNothingCompared = "a segment ended with no samples compared; no verdict is published for it";

    MatchState _ends{};
    bool       _segmented      = false;
    gr::Size_t _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(SequenceEqual, evtIn, in, reference, out, segment_filter, tolerance, n_samples, n_verdicts, n_mismatches, n_unjudged, _ends, _segmented, _settings_epoch);

    mutable MatchState  _ending{};
    mutable bool        _equalSoFar = true;
    mutable std::size_t _compared   = 0UZ;
    mutable gr::Size_t  _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable EventStore  _events;

    void start() { _stateEpoch = std::numeric_limits<gr::Size_t>::max(); }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _segmented = detail::compileOptionalFilter(*this, segment_filter.value, std::string_view{}, _ends);
        _events.clear();
        _settings_epoch = _settings_epoch + 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& refSpan, gr::OutputSpanLike auto& outSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(inSpan)>>;
        if (_stateEpoch != _settings_epoch) {
            _ending     = _ends;
            _equalSoFar = true;
            _compared   = 0UZ;
            _stateEpoch = _settings_epoch;
        }
        std::size_t nVerdicts = 0UZ;
        const auto  close     = [&] {
            if (_compared == 0UZ) {
                n_unjudged = n_unjudged + 1U;
                if constexpr (kOnHost) {
                    detail::reportError(*this, "processBulk()", kNothingCompared);
                }
                return true;
            }
            if (nVerdicts >= outSpan.size()) {
                return false;
            }
            outSpan[nVerdicts++] = static_cast<std::uint8_t>(_equalSoFar ? 1U : 0U);
            n_verdicts           = n_verdicts + 1U;
            _equalSoFar          = true;
            _compared            = 0UZ;
            return true;
        };
        const auto endsSegment = [&](const gr::property_map_view& trigger) { return _segmented && detail::matches(_ending, trigger); };

        bool stopped = false;
        std::ignore  = forEachEvent<kOnHost>(_events, evtSpan, kOnHost ? std::numeric_limits<std::size_t>::max() : outSpan.size(), [&](const gr::property_map_view& event) {
            stopped = endsSegment(event) && !close();
            return !stopped;
        });

        const std::size_t                                                                 nSamples   = stopped ? 0UZ : std::min(inSpan.size(), refSpan.size());
        const std::size_t                                                                 perSegment = static_cast<std::size_t>(n_samples.value);
        auto                                                                              tags       = inSpan.rawTags();
        auto                                                                              next       = std::ranges::begin(tags);
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> scratch{};
        std::size_t                                                                       consumed = 0UZ;
        for (; consumed < nSamples; ++consumed) {
            const auto first = detail::tagsOn(next, std::ranges::end(tags), inSpan.streamIndex + consumed);
            if (perSegment > 0UZ && _compared >= perSegment && !close()) {
                break;
            }
            if (first != next && endsSegment(detail::mergedAt(first, next, scratch)) && !close()) {
                break;
            }
            const T difference = inSpan[consumed] > refSpan[consumed] ? static_cast<T>(inSpan[consumed] - refSpan[consumed]) : static_cast<T>(refSpan[consumed] - inSpan[consumed]);
            if (difference > tolerance) {
                _equalSoFar  = false;
                n_mismatches = n_mismatches + 1U;
            }
            ++_compared;
        }
        if (perSegment > 0UZ && _compared >= perSegment) {
            std::ignore = close();
        }

        outSpan.publish(nVerdicts);
        if (!inSpan.consume(consumed) || !refSpan.consume(consumed)) {
            return gr::work::Status::ERROR;
        }
        return stopped || (consumed == 0UZ && nSamples > 0UZ) ? gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS : gr::work::Status::OK;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_EVENTSHAPING_HPP
