#ifndef GNURADIO_TRIGGER_PATTERNTRIGGER_HPP
#define GNURADIO_TRIGGER_PATTERNTRIGGER_HPP

#include <cstdint>
#include <limits>
#include <memory_resource>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/DigitalLine.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

namespace gr::blocks::trigger {

enum class PatternWhen : std::uint8_t { enters, leaves, holds };

GR_REGISTER_BLOCK("gr::blocks::trigger::PatternTrigger", gr::blocks::trigger::PatternTrigger, [T], [ int16_t, int32_t, float, double ])

template<typename T>
requires(std::is_arithmetic_v<T>)
struct PatternTrigger : gr::Block<PatternTrigger<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief report when N channels stand in a pattern of 1/0/X at one instant

    in#0   ─1─1─0─0─1─▶              (no RxMarbles equivalent)
    in#1   ─0─1─1─0─0─▶              pattern = "10", when = enters
    evtOut ───────────T─▶
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut           evtOut{{.streamSlotsPerPublish = 8UZ}};
    std::vector<gr::PortIn<T>> in;

    A<gr::Size_t, "n inputs", Doc<"channels the pattern spans">, gr::Limits<1U, 32U>>                                                                                           n_inputs = 0U;
    A<std::pmr::string, "pattern", Doc<"one character per channel: 1 high, 0 low, X any (unknown included)">>                                                                   pattern;
    A<std::pmr::vector<T>, "thresholds", Doc<"per channel; a single value applies to all">>                                                                                     thresholds;
    A<T, "hysteresis", Doc<"full width of every comparator's band, centred on its threshold">>                                                                                  hysteresis{};
    A<std::pmr::string, "when", Doc<"enters|leaves|holds">>                                                                                                                     when                 = std::pmr::string("enters");
    A<std::pmr::string, "logic", Doc<"all|any: every channel's term, or at least one (X ignored)">>                                                                             logic                = std::pmr::string("all");
    A<gr::Size_t, "duration min samples", Doc<"'holds': samples to stand, 0 = unset">>                                                                                          duration_min_samples = 0U;
    A<float, "duration min seconds", Doc<"seconds, 0 = unset">>                                                                                                                 duration_min_seconds = 0.f;
    A<gr::Size_t, "skew samples", Doc<"samples a term stays satisfied after its channel left the state, so channels that differ in delay still coincide; 0 = the same sample">> skew_samples         = 0U;
    gr::SampleRate                                                                                                                                                              sample_rate          = 0.f;
    gr::TriggerName                                                                                                                                                             trigger_name;
    gr::Context                                                                                                                                                                 context;
    A<gr::Size_t, "n triggers">                                                                                                                                                 n_triggers       = 0U;
    A<gr::Size_t, "n rejected", Doc<"patterns that did not stand long enough">>                                                                                                 n_rejected       = 0U;
    A<gr::Size_t, "n gaps", Doc<"gap tags seen; each abandons a pattern in progress and puts every line back to unknown">>                                                      n_gaps           = 0U;
    A<gr::Size_t, "n abandoned", Doc<"patterns in progress that a gap interrupted">>                                                                                            n_abandoned      = 0U;
    A<gr::Size_t, "n events dropped", Doc<"reports the event output had no room for">>                                                                                          n_events_dropped = 0U;

    GR_MAKE_REFLECTABLE(PatternTrigger, evtOut, in, n_inputs, pattern, thresholds, hysteresis, when, logic, duration_min_samples, duration_min_seconds, skew_samples, sample_rate, trigger_name, context, n_triggers, n_rejected, n_gaps, n_abandoned, n_events_dropped);

    PatternWhen                 _when = PatternWhen::enters;
    std::vector<DigitalLine<T>> _lines;
    /** '1', '0' or 'X' per channel */
    std::vector<char> _required;
    /** samples since each channel's term last held; saturates */
    std::vector<std::size_t> _sinceHeld;
    std::size_t              _matchedFor = 0UZ;
    bool                     _reported   = false;
    TimeBase                 _time;
    std::size_t              _streamIndex = 0UZ;
    detail::PendingEvents    _pending;
    detail::ErrorMemory      _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        detail::nameTriggerAfterBlock(*this);
    }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("n_inputs") && oldSettings.find_value("n_inputs") != newSettings.find_value("n_inputs")) {
            in.resize(n_inputs);
        }
        _when = gr::meta::parseEnum<PatternWhen>(when.value).value_or(PatternWhen::enters);

        const std::size_t channels = in.size();
        _lines.assign(channels, DigitalLine<T>{});
        _required.assign(channels, 'X');
        if (pattern.value.size() == channels) {
            std::ranges::transform(pattern.value, _required.begin(), [](char c) { return (c == '1' || c == '0') ? c : 'X'; });
        } else if (!pattern.value.empty()) {
            const std::string message = std::format("'pattern' = '{}' has {} characters for {} channel(s); every channel is treated as 'X'", pattern.value, pattern.value.size(), channels);
            detail::reportError(*this, "settingsChanged()", message);
        }

        restartRun();
        _pending.queue.clear(); // the loss count n_events_dropped mirrors stays
    }

    void reset() {
        restartRun();
        _pending.clear();
        n_triggers       = 0U;
        n_rejected       = 0U;
        n_gaps           = 0U;
        n_abandoned      = 0U;
        n_events_dropped = 0U;
    }

    template<gr::InputSpanLike TInput>
    gr::work::Status processBulk(const std::span<TInput>& ins, gr::OutputSpanLike auto& evtOutSpan) {
        if (ins.empty()) {
            return gr::work::Status::OK;
        }
        const std::size_t n = std::ranges::min(ins | std::views::transform([](const auto& s) { return s.size(); }));

        for (std::size_t i = 0UZ; i < n; ++i) {
            adoptAnchors(ins, i);
            judge(ins, i);
        }

        const std::size_t published = _pending.drainInto(evtOutSpan, 0UZ, *this);
        n_events_dropped            = static_cast<gr::Size_t>(_pending.dropped);
        _streamIndex += n;

        evtOutSpan.publish(published);
        for (auto& channel : ins) {
            if (!channel.consume(n)) {
                return gr::work::Status::ERROR;
            }
        }
        return gr::work::Status::OK;
    }

private:
    void restartRun() {
        for (std::size_t channel = 0UZ; channel < _lines.size(); ++channel) {
            _lines[channel].configure({.level = thresholdFor(channel), .hysteresis = hysteresis.value});
        }
        _sinceHeld.assign(_lines.size(), std::numeric_limits<std::size_t>::max());
        _matchedFor  = 0UZ;
        _reported    = false;
        _streamIndex = 0UZ;
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
    }

    [[nodiscard]] T thresholdFor(std::size_t channel) const noexcept {
        if (thresholds.value.empty()) {
            return T{};
        }
        return channel < thresholds.value.size() ? thresholds.value[channel] : thresholds.value.back();
    }

    void adoptAnchors(const auto& ins, std::size_t offset) {
        for (const auto& channel : ins) {
            for (const auto& tag : channel.rawTags()) {
                if (tag.index >= channel.streamIndex && tag.index - channel.streamIndex == offset) {
                    std::ignore = _time.adopt(tag.map, _streamIndex + offset);
                    if (detail::isGap(gr::property_map_view{tag.map})) {
                        n_gaps = n_gaps + 1U;
                        forgetPattern();
                    }
                }
            }
        }
    }

    void forgetPattern() {
        if (_matchedFor > 0UZ && !_reported) {
            n_abandoned = n_abandoned + 1U;
        }
        for (DigitalLine<T>& line : _lines) {
            line.comparator.reset();
        }
        std::ranges::fill(_sinceHeld, std::numeric_limits<std::size_t>::max());
        _matchedFor = 0UZ;
        _reported   = false;
    }

    void judge(const auto& ins, std::size_t offset) {
        for (std::size_t channel = 0UZ; channel < ins.size() && channel < _lines.size(); ++channel) {
            _lines[channel].observe(ins[channel][offset]);
        }
        updateTermAges();

        const std::optional<bool> verdict = evaluate();
        if (!verdict) { // an unknown line leaves the pattern undecided: neither entered nor left
            _matchedFor = 0UZ;
            _reported   = false;
            return;
        }
        if (*verdict) {
            ++_matchedFor;
            if (_when == PatternWhen::enters && !_reported) {
                report(offset);
            } else if (_when == PatternWhen::holds && !_reported && stoodLongEnough()) {
                report(offset);
            }
            return;
        }

        if (_matchedFor > 0UZ) {
            if (_when == PatternWhen::leaves) {
                report(offset);
            } else if (_when == PatternWhen::holds && !_reported) {
                n_rejected = n_rejected + 1U;
            }
        }
        _matchedFor = 0UZ;
        _reported   = false;
    }

    [[nodiscard]] bool termHolds(std::size_t channel) const {
        const gr::trigger::LineState required = _required[channel] == '1' ? gr::trigger::LineState::high : gr::trigger::LineState::low;
        return _lines[channel].state() == required;
    }

    void updateTermAges() {
        for (std::size_t channel = 0UZ; channel < _required.size(); ++channel) {
            if (_required[channel] == 'X') {
                continue;
            }
            const std::size_t age            = _sinceHeld[channel];
            const std::size_t agedSaturating = age == std::numeric_limits<std::size_t>::max() ? age : age + 1UZ;
            _sinceHeld[channel]              = termHolds(channel) ? 0UZ : agedSaturating;
        }
    }

    /** three-valued: false AND unknown is false, true OR unknown is true, otherwise unknown stays unknown */
    [[nodiscard]] std::optional<bool> evaluate() const {
        const bool any       = logic.value == "any";
        bool       undecided = false;
        for (std::size_t channel = 0UZ; channel < _required.size(); ++channel) {
            if (_required[channel] == 'X') {
                continue;
            }
            const gr::trigger::LineState state = _lines[channel].state();
            if (state == gr::trigger::LineState::unknown) {
                undecided = true;
                continue;
            }
            const bool term = termHolds(channel) || _sinceHeld[channel] <= static_cast<std::size_t>(skew_samples);
            if (term == any) { // a true term decides 'any', a false one decides 'all'
                return any;
            }
        }
        return undecided ? std::nullopt : std::optional{!any};
    }

    [[nodiscard]] bool stoodLongEnough() const {
        const auto required = resolveDuration(_time, duration_min_samples, duration_min_seconds);
        return !required || _matchedFor >= *required;
    }

    void report(std::size_t offset) {
        _reported               = true;
        n_triggers              = n_triggers + 1U;
        const std::size_t at    = _streamIndex + offset;
        gr::property_map  found = datedTrigger(_time, at, 0., 0.f, trigger_name.value, context.value);
        if (_when == PatternWhen::holds) { // decided once the pattern had stood: the reference is its first matching sample
            gr::property_map details;
            details.reserve(detail::kReferenceKeys, static_cast<std::uint32_t>(128UZ + trigger_name.value.size()));
            details[gr::tag::kTriggerName.key()]   = trigger_name.value;
            std::ignore                            = writeTriggerTime(details, _time, at, -static_cast<double>(_matchedFor - 1UZ));
            found[gr::tag::kTriggerMetaInfo.key()] = std::move(details);
        }
        _pending.push(std::move(found));
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_PATTERNTRIGGER_HPP
