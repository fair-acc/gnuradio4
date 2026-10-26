#ifndef GNURADIO_MEASUREMENT_MEASURE_HPP
#define GNURADIO_MEASUREMENT_MEASURE_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/Measurement.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <gnuradio-4.0/trigger/WindowCollector.hpp>

namespace gr::blocks::measurement {

enum class Quantity : std::uint8_t {
    minimum,
    maximum,
    peakToPeak,
    mean,
    rms,
    acRms,
    median,
    top,
    base,
    amplitude,
    riseTime,
    fallTime,
    positiveWidth,
    negativeWidth,
    dutyCycle,
    overshoot,
    undershoot,
    area,
    sampleStdDev,
    populationStdDev,
    period,
    frequency,
    delay,
    phase,
};

GR_REGISTER_BLOCK("gr::blocks::measurement::Minimum", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::minimum), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Maximum", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::maximum), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::PeakToPeak", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::peakToPeak), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Mean", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::mean), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Rms", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::rms), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::AcRms", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::acRms), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Median", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::median), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Top", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::top), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Base", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::base), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Amplitude", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::amplitude), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::RiseTime", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::riseTime), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::FallTime", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::fallTime), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::PositiveWidth", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::positiveWidth), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::NegativeWidth", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::negativeWidth), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::DutyCycle", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::dutyCycle), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Overshoot", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::overshoot), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Undershoot", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::undershoot), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Area", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::area), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::SampleStdDev", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::sampleStdDev), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::PopulationStdDev", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::populationStdDev), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Period", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::period), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Frequency", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::frequency), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Delay", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::delay), [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::measurement::Phase", gr::blocks::measurement::Measure, ([T], gr::blocks::measurement::Quantity::phase), [ float, double ])

template<typename T, Quantity quantity>
requires std::is_floating_point_v<T>
struct Measure : gr::Block<Measure<T, quantity>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief measure one quantity of each channel over a gate, and report it as an event and a tag (IEEE 181-2011)

    in#0    ──▁▁▁╱▔▔▔╲▁▁▁╱▔▔▔╲▁▁▁──▶   passes through unchanged, as out#0
    evtIn   ───O──────────────C───▶   open_filter, close_filter
    evtOut  ──────────────────R───▶   value and unit, or a reason; the gate it covers

Times are in seconds where `sample_rate` is known, else in samples.

 [1] specification: IEEE Std 181-2011, transition and pulse parameters
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    struct Gate {
        std::uint64_t    startNs = 0U;
        std::pmr::string opener;           /// the trigger_name of the event that opened it
        bool             awaiting = false; /// opened to wait for a close event, capped at max_length
        bool             closed   = false; /// and one came
    };
    using Collector = gr::blocks::trigger::WindowCollector<T, Gate>;

    constexpr static bool kPair = quantity == Quantity::delay || quantity == Quantity::phase;

    gr::EventPortIn                           evtIn;
    std::vector<gr::PortIn<T>>                in;
    std::vector<gr::PortOut<T, gr::Optional>> out;
    gr::EventPortOut                          evtOut{{.streamSlotsPerPublish = 32UZ}}; /// one result per channel and gate

    A<gr::Size_t, "n inputs", Doc<"channels measured together; a delay or phase compares channel 1 against channel 0 at their first rising mid-level crossings, phase in degrees in (-180, 180]">, gr::Limits<kPair ? 2U : 1U, kPair ? 2U : 32U>> n_inputs = kPair ? 2U : 1U;
    A<std::pmr::string, "open filter", Doc<"trigger filter of the events that open a gate; empty = back-to-back gates">>                                                                                                                          open_filter;
    A<std::pmr::string, "close filter", Doc<"trigger filter of the events that close one; empty = n_post after the opening event">>                                                                                                               close_filter;
    A<gr::Size_t, "n pre", Doc<"samples a gate starts before its opening event">>                                                                                                                                                                 n_pre          = 0U;
    A<gr::Size_t, "n post", Doc<"samples a gate ends after its opening event, end exclusive">>                                                                                                                                                    n_post         = 0U;
    A<gr::Size_t, "length", Doc<"samples of a back-to-back gate">>                                                                                                                                                                                length         = 1024U;
    A<gr::Size_t, "max length", Doc<"longest gate, and how far back one may reach">>                                                                                                                                                              max_length     = 1U << 16U;
    A<double, "max lateness", Doc<"s, how long after its trigger_time an event may arrive; earlier ones wait as long">>                                                                                                                           max_lateness   = 0.;
    A<double, "reference low", Doc<"fraction of the amplitude a transition starts at">>                                                                                                                                                           reference_low  = 0.1;
    A<double, "reference mid", Doc<"fraction of the amplitude widths are measured at; period and frequency use its first two rising crossings">>                                                                                                  reference_mid  = 0.5;
    A<double, "reference high", Doc<"fraction of the amplitude a transition ends at">>                                                                                                                                                            reference_high = 0.9;
    gr::SampleRate                                                                                                                                                                                                                                sample_rate    = 0.f;
    gr::TriggerName                                                                                                                                                                                                                               trigger_name;
    A<gr::Size_t, "n measured", Doc<"valid results reported">>                                                                                                                                                                                    n_measured    = 0U;
    A<gr::Size_t, "n invalid", Doc<"results reported as invalid, with their reason">>                                                                                                                                                             n_invalid     = 0U;
    A<gr::Size_t, "n unplaceable", Doc<"opening or closing events the stream could not date">>                                                                                                                                                    n_unplaceable = 0U;
    A<gr::Size_t, "n gaps", Doc<"n_dropped_samples tags seen; a gate holding one is invalid, reason censored, never interpolated across">>                                                                                                        n_gaps        = 0U;

    GR_MAKE_REFLECTABLE(Measure, evtIn, in, out, evtOut, n_inputs, open_filter, close_filter, n_pre, n_post, length, max_length, max_lateness, reference_low, reference_mid, reference_high, sample_rate, trigger_name, n_measured, n_invalid, n_unplaceable, n_gaps);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState                                                         _opens{};
    MatchState                                                         _closes{};
    bool                                                               _gated         = false;
    bool                                                               _closeFiltered = false;
    std::vector<Collector>                                             _collectors;
    std::vector<std::pmr::string>                                      _units;
    gr::blocks::trigger::TimeBase                                      _time;
    gr::blocks::trigger::EventStore                                    _events;
    gr::blocks::trigger::detail::PendingEvents                         _results;
    std::vector<std::vector<std::pair<std::size_t, gr::property_map>>> _tags; /// per channel, results to tag and the sample each was decided at
    std::vector<std::size_t>                                           _gaps; /// stream positions a drop of samples ended at
    std::size_t                                                        _streamIndex = 0UZ;
    std::size_t                                                        _nextUngated = 0UZ;

    gr::blocks::trigger::detail::ErrorMemory _deferredError{};

    void start() {
        gr::blocks::trigger::detail::sendDeferredError(*this);
        gr::blocks::trigger::detail::nameTriggerAfterBlock(*this);
    }

    void reset() {
        for (Collector& collector : _collectors) {
            collector.reset();
        }
        for (auto& waiting : _tags) {
            waiting.clear();
        }
        _events.clear();
        _results.clear();
        _gaps.clear();
        _streamIndex  = 0UZ;
        _nextUngated  = 0UZ;
        n_measured    = 0U;
        n_invalid     = 0U;
        n_unplaceable = 0U;
        n_gaps        = 0U;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if (in.size() != n_inputs) {
            in.resize(n_inputs);
            out.resize(n_inputs);
        }
        _gated         = gr::blocks::trigger::detail::compileOptionalFilter(*this, open_filter.value, std::string_view{}, _opens);
        _closeFiltered = gr::blocks::trigger::detail::compileOptionalFilter(*this, close_filter.value, std::string_view{}, _closes);
        _time.setRate(static_cast<double>(sample_rate));
        _events.lateness = gr::blocks::trigger::detail::nanoseconds(max_lateness.value);
        if (_collectors.size() != in.size()) {
            _collectors.assign(in.size(), Collector{});
            _units.assign(in.size(), std::pmr::string{});
            _tags.assign(in.size(), {});
            _streamIndex = 0UZ;
            _nextUngated = 0UZ;
        }
    }

    template<gr::InputSpanLike TInput, gr::OutputSpanLike TOutput>
    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, std::span<TInput>& ins, std::span<TOutput>& outs, gr::OutputSpanLike auto& evtOutSpan) {
        _events.drain(evtSpan);
        const std::size_t available = std::ranges::min(ins | std::views::transform([](const auto& span) { return span.size(); }));
        const std::size_t room      = std::ranges::min(outs | std::views::transform([](const auto& span) { return span.size(); }));
        const std::size_t n         = std::min(available, room);

        noteTags(ins, n);
        for (std::size_t channel = 0UZ; channel < ins.size(); ++channel) {
            _collectors[channel].reachBack(static_cast<std::size_t>(max_length + n_pre) + 1UZ + n); // a gate anywhere in this call is placeable
            _collectors[channel].push(std::span<const T>{ins[channel].begin(), n});
        }
        _streamIndex += n;

        if (_gated) {
            openGates();
        } else {
            openUngated();
        }
        const std::size_t callStart = _streamIndex - n;
        report(callStart);

        for (std::size_t channel = 0UZ; channel < ins.size(); ++channel) {
            std::ranges::copy(ins[channel] | std::views::take(n), outs[channel].begin());
            if (n > 0UZ) {
                for (const auto& [decided, result] : _tags[channel]) {
                    outs[channel].publishTag(result, decided - callStart);
                }
                _tags[channel].clear();
            }
            outs[channel].publish(n);
        }

        evtOutSpan.publish(_results.drainInto(evtOutSpan, 0UZ, *this));
        for (auto& span : ins) {
            if (!span.consume(n)) {
                return gr::work::Status::ERROR;
            }
        }
        return gr::work::Status::OK;
    }

private:
    void noteTags(const auto& ins, std::size_t n) {
        for (std::size_t channel = 0UZ; channel < ins.size(); ++channel) {
            for (const auto& tag : ins[channel].rawTags()) {
                if (tag.index < ins[channel].streamIndex || tag.index - ins[channel].streamIndex >= n) {
                    continue;
                }
                const gr::property_map_view tagMap{tag.map};
                const std::size_t           at = _streamIndex + (tag.index - ins[channel].streamIndex);
                if (channel == 0UZ) {
                    std::ignore = _time.adopt(tagMap, at);
                }
                if (trigger::detail::isGap(tagMap) && (_gaps.empty() || _gaps.back() != at)) {
                    _gaps.push_back(at);
                    n_gaps = n_gaps + 1U;
                }
                if (const auto unit = tagMap.get_if<std::string_view>(gr::tag::SIGNAL_UNIT.key())) {
                    _units[channel] = std::pmr::string(*unit);
                }
            }
        }
    }

    void openUngated() {
        const auto span = static_cast<std::size_t>(std::max<gr::Size_t>(length, 1U));
        while (_nextUngated + span <= _streamIndex) {
            open(_nextUngated, _nextUngated + span);
            _nextUngated += span;
        }
    }

    void openGates() {
        const std::size_t ready   = _events.ready();
        std::size_t       handled = 0UZ;
        for (const gr::blocks::trigger::StoredEvent& stored : _events.ordered()) {
            if (handled >= ready) {
                break;
            }
            const gr::property_map_view event{stored.event};
            const auto                  start = stored.at ? _time.indexAt(*stored.at) : std::nullopt;
            if (!start) {
                n_unplaceable = n_unplaceable + 1U;
                ++handled;
                continue;
            }
            const bool closes = _closeFiltered && gr::blocks::trigger::detail::acceptsTrigger(_closes, true, event);
            const bool opens  = gr::blocks::trigger::detail::acceptsTrigger(_opens, true, event);
            if (!closes && *start + static_cast<std::size_t>(n_post.value) > _streamIndex) {
                break; // its gate is not yet in the stream: wait for the samples, not for the next event
            }
            if (closes) {
                for (Collector& collector : _collectors) {
                    if (!collector.openWindows.empty()) {
                        collector.openWindows.front().closesAt    = std::max(collector.openWindows.front().startsAt, *start);
                        collector.openWindows.front().info.closed = true;
                    }
                }
            } else if (opens) {
                const bool        censored = *start < n_pre; // the stream began inside the gate
                const std::size_t first    = censored ? 0UZ : *start - static_cast<std::size_t>(n_pre.value);
                const std::size_t closesAt = _closeFiltered ? first + static_cast<std::size_t>(max_length.value) : *start + static_cast<std::size_t>(n_post.value);
                open(first, closesAt, censored, _closeFiltered, event.template get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()).value_or(std::string_view{}));
            }
            ++handled;
        }
        _events.retire(handled);
    }

    void open(std::size_t first, std::size_t closesAt, bool censored = false, bool awaiting = false, std::string_view opener = {}) {
        const Gate gate{.startNs = _time.at(first).value_or(0U), .opener = std::pmr::string(opener), .awaiting = awaiting, .closed = false};
        for (Collector& collector : _collectors) {
            std::ignore = collector.open(first, closesAt, gate, censored);
        }
    }

    /** a result is decided at its gate's last sample, or at this call's first when a late close moved the gate's end before it */
    void report(std::size_t callStart) {
        const std::size_t nReporting = kPair ? 1UZ : _collectors.size();
        for (std::size_t channel = 0UZ; channel < nReporting; ++channel) {
            while (auto window = _collectors[channel].take()) {
                const std::optional<typename Collector::Window> partner = kPair ? _collectors[1UZ].take() : std::nullopt;
                const std::size_t                               decided = std::max(window->closesAt > 0UZ ? window->closesAt - 1UZ : 0UZ, callStart);
                const bool                                      gapped  = std::ranges::any_of(_gaps, [&](std::size_t at) { return window->startsAt < at && at < window->closesAt; });
                gr::property_map                                result  = resultOf(*window, partner ? &*partner : nullptr, channel, decided, gapped);
                _tags[channel].emplace_back(decided, result);
                _results.push(std::move(result));
            }
        }
        const std::size_t reach = static_cast<std::size_t>(std::max(max_length.value, length.value)) + n_pre + 1UZ;
        std::erase_if(_gaps, [&](std::size_t at) { return at + reach < _streamIndex; });
    }

    [[nodiscard]] gr::algorithm::measurement::Measured measured(std::span<const T> samples, std::span<const T> second) const {
        namespace m = gr::algorithm::measurement;
        const m::ReferenceLevels levels{.low = reference_low, .mid = reference_mid, .high = reference_high};
        switch (quantity) {
        case Quantity::minimum: return m::minimum(samples);
        case Quantity::maximum: return m::maximum(samples);
        case Quantity::peakToPeak: return m::peakToPeak(samples);
        case Quantity::mean: return m::mean(samples);
        case Quantity::rms: return m::rms(samples);
        case Quantity::acRms: return m::populationStdDev(samples);
        case Quantity::median: return m::median(samples);
        case Quantity::top: return m::top(samples);
        case Quantity::base: return m::base(samples);
        case Quantity::amplitude: return m::amplitude(samples);
        case Quantity::riseTime: return m::transitionTime(samples, true, levels);
        case Quantity::fallTime: return m::transitionTime(samples, false, levels);
        case Quantity::positiveWidth: return m::width(samples, true, levels);
        case Quantity::negativeWidth: return m::width(samples, false, levels);
        case Quantity::dutyCycle: return m::dutyCycle(samples, levels);
        case Quantity::overshoot: return m::overshoot(samples);
        case Quantity::undershoot: return m::undershoot(samples);
        case Quantity::area: return m::area(samples);
        case Quantity::sampleStdDev: return m::sampleStdDev(samples);
        case Quantity::populationStdDev: return m::populationStdDev(samples);
        case Quantity::period: return m::period(samples, levels);
        case Quantity::frequency: return m::frequency(samples, levels);
        case Quantity::delay: return m::delay(samples, second, levels);
        case Quantity::phase: return m::phase(samples, second, levels);
        }
        std::unreachable();
    }

    /** times come out in samples; a known rate turns them, and an area's sample axis, into seconds */
    [[nodiscard]] std::pair<double, std::pmr::string> scaled(double value, std::size_t channel) const {
        const bool              edgeTimed = quantity == Quantity::riseTime || quantity == Quantity::fallTime || quantity == Quantity::positiveWidth || quantity == Quantity::negativeWidth;
        const bool              timed     = edgeTimed || quantity == Quantity::period || quantity == Quantity::delay;
        const bool              ratio     = quantity == Quantity::dutyCycle || quantity == Quantity::overshoot || quantity == Quantity::undershoot;
        const bool              rated     = _time.measuresDuration();
        const std::pmr::string& unit      = _units[channel];
        if (ratio) {
            return {value, std::pmr::string("%")};
        }
        if (quantity == Quantity::phase) {
            return {value, std::pmr::string("deg")};
        }
        if (quantity == Quantity::frequency) {
            return rated ? std::pair{value * _time.rate(), std::pmr::string("Hz")} : std::pair{value, std::pmr::string("1/samples")};
        }
        if (timed) {
            return rated ? std::pair{value / _time.rate(), std::pmr::string("s")} : std::pair{value, std::pmr::string("samples")};
        }
        if (quantity == Quantity::area) {
            const std::string axis = rated ? "s" : "samples";
            const std::string per  = unit.empty() ? axis : std::string(unit) + "·" + axis;
            return {rated ? value / _time.rate() : value, std::pmr::string(per)};
        }
        return {value, unit};
    }

    [[nodiscard]] gr::property_map resultOf(const typename Collector::Window& window, const typename Collector::Window* partner, std::size_t channel, std::size_t decided, bool gapped) {
        const auto       found = measured(std::span<const T>{window.samples}, partner != nullptr ? std::span<const T>{partner->samples} : std::span<const T>{});
        gr::property_map details;
        const bool       truncated = window.info.awaiting && !window.info.closed; // no close came within max_length
        const bool       censored  = window.partial || (partner != nullptr && partner->partial) || gapped;
        const bool       complete  = !censored && !truncated;
        details["channel"]         = static_cast<gr::Size_t>(channel);
        details["n_samples"]       = static_cast<gr::Size_t>(window.samples.size());
        if (found && complete) {
            const auto [value, unit] = scaled(*found, channel);
            details["value"]         = value;
            details["unit"]          = unit;
            n_measured               = n_measured + 1U;
        } else {
            const std::string_view reason = [&]() -> std::string_view {
                if (censored) {
                    return "censored";
                }
                return truncated ? "truncated" : gr::meta::enumName(found.error()).value_or("invalid");
            }();
            details["reason"] = std::pmr::string(reason);
            n_invalid         = n_invalid + 1U;
        }
        if (window.info.startNs != 0U) { // decided at the gate's end, about the gate from its start
            details[gr::tag::TRIGGER_NAME.key()]   = window.info.opener;
            details[gr::tag::TRIGGER_TIME.key()]   = window.info.startNs;
            details[gr::tag::TRIGGER_OFFSET.key()] = 0.f; /// the gate starts on a sample of the grid
        }
        gr::property_map result;
        result[gr::tag::TRIGGER_NAME.key()] = trigger_name.value;
        if (const auto decidedNs = _time.at(decided)) {
            result[gr::tag::TRIGGER_TIME.key()]   = *decidedNs;
            result[gr::tag::TRIGGER_OFFSET.key()] = 0.f;
        }
        result[gr::tag::TRIGGER_META_INFO.key()] = std::move(details);
        return result;
    }
};

} // namespace gr::blocks::measurement

#endif // GNURADIO_MEASUREMENT_MEASURE_HPP
