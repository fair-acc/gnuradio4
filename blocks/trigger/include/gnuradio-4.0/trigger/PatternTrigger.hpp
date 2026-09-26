#ifndef GNURADIO_TRIGGER_PATTERNTRIGGER_HPP
#define GNURADIO_TRIGGER_PATTERNTRIGGER_HPP

#include <cstdint>
#include <memory_resource>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/trigger/ConditionSource.hpp>
#include <gnuradio-4.0/trigger/DigitalLine.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

namespace gr::blocks::trigger {

enum class PatternWhen : std::uint8_t { AUTO, enters, leaves, holds };

GR_REGISTER_BLOCK("gr::blocks::trigger::PatternTrigger", gr::blocks::trigger::PatternTrigger, [T], [ int16_t, int32_t, float, double ])

template<typename T>
requires(std::is_arithmetic_v<T>)
struct PatternTrigger : gr::Block<PatternTrigger<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief report when N channels stand in a pattern of 1/0/X at one instant

    in#0   ─1─1─0─0─1─▶              (no RxMarbles equivalent)
    in#1   ─0─1─1─0─0─▶              pattern = "10", when = enters
    evtOut ───────────T─▶

The scope's logic trigger: `1`, `0` and `X` per channel, with `when` deciding whether the report is on entering the pattern,
leaving it, or holding it for `width_min_samples`.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut           evtOut{{.streamSlotsPerPublish = 8UZ}};
    std::vector<gr::PortIn<T>> in;

    A<gr::Size_t, "n inputs", Doc<"channels the pattern spans">, gr::Limits<1U, 32U>>         n_inputs = 0U;
    A<std::pmr::string, "pattern", Doc<"one character per channel: 1 high, 0 low, X either">> pattern;
    A<std::pmr::vector<T>, "thresholds", Doc<"per channel; a single value applies to all">>   thresholds;
    A<T, "hysteresis", Doc<"widens every comparator against chatter">>                        hysteresis{};
    A<std::pmr::string, "when", Doc<"enters|leaves|holds">>                                   when              = std::pmr::string("enters");
    A<gr::Size_t, "width min samples", Doc<"'holds': samples to stand, 0 = unset">>           width_min_samples = 0U;
    A<float, "width min seconds", Doc<"seconds, 0 = unset">>                                  width_min_seconds = 0.f;
    A<float, "sample rate", Doc<"Hz, dates the trigger, 0 = unknown">>                        sample_rate       = 0.f;
    A<std::pmr::string, "trigger name", Doc<"the name the emitted trigger carries">>          trigger_name      = std::pmr::string("pattern");
    A<std::pmr::string, "context", Doc<"context the emitted trigger carries">>                context;
    A<gr::Size_t, "n triggers">                                                               n_triggers       = 0U;
    A<gr::Size_t, "n rejected", Doc<"patterns that did not stand long enough">>               n_rejected       = 0U;
    A<gr::Size_t, "n events dropped", Doc<"reports the event output had no room for">>        n_events_dropped = 0U;

    GR_MAKE_REFLECTABLE(PatternTrigger, evtOut, in, n_inputs, pattern, thresholds, hysteresis, when, width_min_samples, width_min_seconds, sample_rate, trigger_name, context, n_triggers, n_rejected, n_events_dropped);

    PatternWhen                 _when = PatternWhen::enters;
    std::vector<DigitalLine<T>> _lines;
    std::vector<char>           _required; // '1', '0' or 'X' per channel
    std::size_t                 _matchedFor = 0UZ;
    bool                        _reported   = false;
    TimeBase                    _time;
    std::size_t                 _streamIndex = 0UZ;
    detail::PendingEvents       _pending;

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("n_inputs") && oldSettings.find_value("n_inputs") != newSettings.find_value("n_inputs")) {
            in.resize(n_inputs);
        }
        _when = parseWhen(when.value);
        _time.setRate(static_cast<double>(sample_rate));

        const std::size_t channels = in.size();
        _lines.assign(channels, DigitalLine<T>{});
        for (std::size_t channel = 0UZ; channel < channels; ++channel) {
            _lines[channel].configure(guard(), thresholdFor(channel));
        }
        _required.assign(channels, 'X');
        if (pattern.value.size() == channels) {
            std::ranges::transform(pattern.value, _required.begin(), [](char c) { return (c == '1' || c == '0') ? c : 'X'; });
        } else if (!pattern.value.empty()) {
            gr::log::warning("PatternTrigger: 'pattern' = '{}' has {} characters for {} channel(s); every channel is treated as 'X'", pattern.value, pattern.value.size(), channels);
        }

        _matchedFor  = 0UZ;
        _reported    = false;
        _streamIndex = 0UZ;
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
        _pending.clear();
        n_triggers       = 0U;
        n_rejected       = 0U;
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

        const std::size_t published = _pending.drainInto(evtOutSpan, 0UZ, this->name, this->unique_name);
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
    [[nodiscard]] static PatternWhen parseWhen(std::string_view text) noexcept {
        const auto named = gr::meta::parseEnum<PatternWhen>(text);
        return named && *named != PatternWhen::AUTO ? *named : PatternWhen::enters;
    }

    [[nodiscard]] T guard() const noexcept { return hysteresis.value == T{} ? T(1) : hysteresis.value; }

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
                }
            }
        }
    }

    void judge(const auto& ins, std::size_t offset) {
        for (std::size_t channel = 0UZ; channel < ins.size() && channel < _lines.size(); ++channel) {
            _lines[channel].observe(ins[channel][offset]);
        }

        const bool matches = std::ranges::all_of(std::views::iota(0UZ, _required.size()), [this](std::size_t channel) { return _required[channel] == 'X' || (_required[channel] == '1') == _lines[channel].high; });

        if (matches) {
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

    [[nodiscard]] bool stoodLongEnough() const {
        const auto required = resolveDuration(_time, width_min_samples, width_min_seconds);
        return !required || _matchedFor >= *required;
    }

    void report(std::size_t offset) {
        _reported                    = true;
        n_triggers                   = n_triggers + 1U;
        gr::property_map found       = datedTrigger(_time, _streamIndex + offset, gr::UncertainValue<float>{0.f, 0.f}, trigger_name.value, context.value);
        found[std::string("source")] = std::pmr::string(this->unique_name.value());
        _pending.push(std::move(found));
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_PATTERNTRIGGER_HPP
