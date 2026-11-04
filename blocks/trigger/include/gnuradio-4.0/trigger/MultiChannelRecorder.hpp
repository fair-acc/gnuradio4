#ifndef GNURADIO_TRIGGER_MULTICHANNELRECORDER_HPP
#define GNURADIO_TRIGGER_MULTICHANNELRECORDER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory_resource>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/trigger/EventReference.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <gnuradio-4.0/trigger/WindowCollector.hpp>

namespace gr::blocks::trigger {

enum class RecordAlign : std::uint8_t { composite, channel };
enum class RecordRetrigger : std::uint8_t { ignore, restart, extend, parallel };

/** what a recorded window remembers of the decision that opened it */
struct RecordedSpan {
    std::size_t   startIndex = 0UZ;
    std::uint64_t startNs    = 0U;
    std::uint64_t stopNs     = 0U; /// 0 = a single trigger
    std::uint64_t preNs      = 0U;
    std::uint64_t postNs     = 0U;
    /** ended by max_duration or a rate change; an unclosed window carries it until closed */
    bool truncated   = false;
    bool rateChanged = false;
    bool forced      = false; /// opened by `force_filter`
    bool automatic   = false; /// opened by `auto_timeout`, not by a decision
};

enum class RecordMode : std::uint8_t { normal, single, automatic };
enum class Opened : std::uint8_t { opened, merged, ignored, lost, unplaceable }; /// what a decision did to the windows

GR_REGISTER_BLOCK(gr::blocks::trigger::MultiChannelRecorder, [T], [ int16_t, int32_t, float, double ])

template<typename T>
struct MultiChannelRecorder : gr::Block<MultiChannelRecorder<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief extract the same window from every channel on one decision, as one DataSet

    evtIn  ────────X────────▶        (no RxMarbles equivalent)
    in#0   ─a─a─a─a─a─a─a─a─▶        n_pre = 2, n_post = 2
    in#1   ─b─b─b─b─b─b─b─b─▶
    out    ────────D────────▶        D = one DataSet, one signal per channel

A window is [anchor - pre, anchor + post).
Each signal is named `<signal_name><channel index>`.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                        evtIn;
    std::vector<gr::PortIn<T>>             in;
    gr::PortOut<gr::DataSet<T>, gr::Async> out;
    gr::EventPortOut                       evtOut;

    A<gr::Size_t, "n inputs", Doc<"channels recorded together">, gr::Limits<1U, 32U>>                                                                   n_inputs     = 0U;
    A<gr::Size_t, "n pre", Doc<"samples before the anchor, if no event or setting says">>                                                               n_pre        = 0U;
    A<gr::Size_t, "n post", Doc<"samples from the anchor on, if no event or setting says; a carried gr:trigger_stop_time ends the window there">>       n_post       = 0U;
    gr::TriggerPre                                                                                                                                      trigger_pre  = 0U;
    gr::TriggerPost                                                                                                                                     trigger_post = 0U;
    A<float, "history", Doc<"how far back a window may reach; 0 = twice the longest">, gr::Unit<"s">>                                                   history      = 0.f;
    A<bool, "emit partial", Doc<"a window cut short: emitted, flagged in gr:acquisition_flags, else dropped and reported">>                             emit_partial = true;
    A<std::pmr::string, "open filter", Doc<"trigger filter of the events that open a window; empty = all">>                                             open_filter;
    A<std::pmr::string, "close filter", Doc<"trigger filter of the events that close the oldest open window; empty = windows close themselves">>        close_filter;
    A<std::pmr::string, "retrigger", Doc<"a decision while a window is open: ignore|restart|extend|parallel">>                                          retrigger     = std::pmr::string("parallel");
    A<gr::Size_t, "max open", Doc<"windows open at once with retrigger = parallel">>                                                                    max_open      = 16U;
    A<float, "max duration", Doc<"how long an unclosed window may stay open; 0 = history">, gr::Unit<"s">>                                              max_duration  = 0.f;
    A<std::pmr::string, "align", Doc<"composite|channel">>                                                                                              align         = std::pmr::string("composite");
    A<gr::Size_t, "align channel", Doc<"'channel' only: whose trigger lines the window up">>                                                            align_channel = 0U;
    gr::SampleRate                                                                                                                                      sample_rate   = 0.f;
    gr::SignalName                                                                                                                                      signal_name   = std::pmr::string("channel");
    A<gr::Size_t, "n segments", Doc<"stop after this many, 0 = without end">>                                                                           n_segments    = 0U;
    A<std::pmr::string, "mode", Doc<"normal|single|auto: single records one window; auto forces one after auto_timeout without">>                       mode          = std::pmr::string("normal");
    A<float, "auto timeout", Doc<"'auto': how long without an accepted decision before a window is forced">, gr::Unit<"s">>                             auto_timeout  = 0.f;
    A<std::pmr::string, "force filter", Doc<"decisions that open a window whatever open_filter and holdoff say">>                                       force_filter  = std::pmr::string("forced");
    A<float, "holdoff", Doc<"a decision referring to less than this after the last accepted one is suppressed">, gr::Unit<"s">>                         holdoff       = 0.f;
    A<gr::Size_t, "holdoff count", Doc<"decisions suppressed after each accepted one">>                                                                 holdoff_count = 0U;
    A<float, "max lateness", Doc<"how long after its trigger_time a decision may arrive; earlier ones wait as long">, gr::Unit<"s">>                    max_lateness  = 0.f;
    A<std::pmr::string, "arm filter", Doc<"trigger filter that must be seen before a window may open, and opens nothing itself; empty = always armed">> arm_filter;
    A<std::pmr::string, "rearm", Doc<"auto = every decision may open a window, manual = one per arming (and re-opens a stopped single)">>               rearm         = std::pmr::string("auto");
    A<std::pmr::string, "reference", Doc<"decision|start: which time anchors the window, orders holdoff and picks the single window">>                  reference     = std::pmr::string("decision");
    A<std::pmr::string, "stage", Doc<"priming|unarmed|armed|awaiting_samples|holdoff|stopped, as the recorder last saw itself; read-only">>             stage         = std::pmr::string("priming");
    A<gr::Size_t, "n suppressed", Doc<"decisions holdoff or a single mode's stop turned down">>                                                         n_suppressed  = 0U;
    A<gr::Size_t, "n recorded">                                                                                                                         n_recorded    = 0U;
    A<gr::Size_t, "n undated", Doc<"decisions carrying no time, which cannot be placed">>                                                               n_undated     = 0U;
    A<gr::Size_t, "n unplaceable", Doc<"decisions whose sample could not be found">>                                                                    n_unplaceable = 0U;
    A<gr::Size_t, "n lost", Doc<"windows that could not be completed">>                                                                                 n_lost        = 0U;
    A<gr::Size_t, "n unmatched", Doc<"closes that found no open window">>                                                                               n_unmatched   = 0U;

    GR_MAKE_REFLECTABLE(MultiChannelRecorder, evtIn, in, out, evtOut, n_inputs, n_pre, n_post, trigger_pre, trigger_post, history, emit_partial, open_filter, close_filter, retrigger, max_open, max_duration, align, align_channel, sample_rate, signal_name, n_segments, mode, auto_timeout, force_filter, holdoff, holdoff_count, max_lateness, arm_filter, rearm, reference, stage, n_suppressed, n_recorded, n_undated, n_unplaceable, n_lost, n_unmatched);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    RecordAlign                                   _align     = RecordAlign::composite;
    RecordRetrigger                               _retrigger = RecordRetrigger::parallel;
    MatchState                                    _opens{};
    MatchState                                    _closes{};
    MatchState                                    _forces{};
    MatchState                                    _arms{};
    bool                                          _armFiltered   = false;
    bool                                          _armed         = true;
    bool                                          _manualRearm   = false;
    EventReference                                _reference     = EventReference::decision;
    bool                                          _openFiltered  = false;
    bool                                          _forceFiltered = false;
    RecordMode                                    _mode          = RecordMode::normal;
    std::optional<std::uint64_t>                  _lastAccepted;
    std::size_t                                   _lastAcceptedSample = 0UZ;
    gr::Size_t                                    _holdoffLeft        = 0U;
    bool                                          _stopped            = false;
    bool                                          _closeFiltered      = false;
    std::size_t                                   _reach              = 0UZ;
    std::vector<WindowCollector<T, RecordedSpan>> _collectors;
    TimeBase                                      _time;
    std::size_t                                   _streamIndex = 0UZ;
    EventStore                                    _events;
    std::optional<std::size_t>                    _lastAlignTag;
    std::size_t                                   _unreported = 0UZ;
    std::size_t                                   _chunk      = 0UZ; /// samples the current work call brought

    detail::ErrorMemory _deferredError{};

    void start() { detail::sendDeferredError(*this); }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("n_inputs") && oldSettings.find_value("n_inputs") != newSettings.find_value("n_inputs")) {
            in.resize(n_inputs);
        }
        refuseUnknown(rearm, "rearm", {"auto", "manual"}, oldSettings, newSettings);
        refuseUnknown(reference, "reference", {"decision", "start"}, oldSettings, newSettings);
        _align           = align == "channel" ? RecordAlign::channel : RecordAlign::composite;
        _retrigger       = gr::meta::parseEnum<RecordRetrigger>(retrigger.value).value_or(RecordRetrigger::parallel);
        _openFiltered    = detail::compileOptionalFilter(*this, open_filter.value, std::string_view{}, _opens);
        _closeFiltered   = detail::compileOptionalFilter(*this, close_filter.value, std::string_view{}, _closes);
        _forceFiltered   = detail::compileOptionalFilter(*this, force_filter.value, std::string_view{}, _forces);
        _armFiltered     = detail::compileOptionalFilter(*this, arm_filter.value, std::string_view{}, _arms);
        _armed           = !_armFiltered || (_armed && !newSettings.contains("arm_filter"));
        _manualRearm     = rearm == "manual";
        _reference       = reference == "start" ? EventReference::start : EventReference::decision;
        _mode            = parseMode(mode.value);
        _events.lateness = detail::nanoseconds(static_cast<double>(max_lateness.value));
        _time.setRate(static_cast<double>(sample_rate));

        if (_collectors.size() != in.size()) {
            reset();
        }
        widenReach();
    }

    /** closed windows the output had no room for when the stream ended are lost, and counted */
    void stop() {
        const std::size_t pending = _collectors.empty() ? 0UZ : _collectors[0].nClosed();
        if (pending > 0UZ) {
            n_lost                    = n_lost + static_cast<gr::Size_t>(pending);
            const std::string message = std::format("{} closed window(s) discarded at the end of the stream: the output had no room", pending);
            detail::reportError(*this, "stop()", message);
        }
        for (WindowCollector<T, RecordedSpan>& collector : _collectors) {
            collector.closedWindows.clear();
        }
    }

    /** also re-arms a single-mode recorder */
    void reset() {
        _collectors.assign(in.size(), WindowCollector<T, RecordedSpan>{});
        _streamIndex  = 0UZ;
        _lastAlignTag = std::nullopt;
        _events.clear();
        n_recorded          = 0U;
        n_undated           = 0U;
        n_unplaceable       = 0U;
        n_lost              = 0U;
        n_unmatched         = 0U;
        n_suppressed        = 0U;
        _lastAccepted       = std::nullopt;
        _lastAcceptedSample = 0UZ;
        _holdoffLeft        = 0U;
        _stopped            = false;
        _armed              = !_armFiltered;
        stage               = std::pmr::string("priming");
        widenReach();
    }

    template<gr::InputSpanLike TInput>
    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, const std::span<TInput>& ins, gr::OutputSpanLike auto& outSpan, gr::OutputSpanLike auto& /*evtOutSpan*/) {
        _events.drain(evtSpan);
        if (ins.empty()) {
            reportRefusals();
            return gr::work::Status::OK;
        }

        const std::size_t n = std::ranges::min(ins | std::views::transform([](const auto& s) { return s.size(); }));

        noteTagsAndTime(ins, n);
        _chunk = n;
        for (std::size_t channel = 0UZ; channel < ins.size() && channel < _collectors.size(); ++channel) {
            _collectors[channel].reachBack(_reach + n); // a decision anywhere in this call is placeable, however long the call
            _collectors[channel].push(std::span<const T>{ins[channel].begin(), n});
        }
        _streamIndex += n;

        openWindows();
        forceDeadlinesBefore(_streamIndex);
        const std::size_t emitted = emitReady(outSpan);
        noteStage();

        outSpan.publish(emitted);
        reportRefusals();
        for (auto& channel : ins) {
            if (!channel.consume(n)) {
                return gr::work::Status::ERROR;
            }
        }
        return gr::work::Status::OK;
    }

private:
    void noteTagsAndTime(const auto& ins, std::size_t n) {
        for (std::size_t channel = 0UZ; channel < ins.size(); ++channel) {
            for (const auto& tag : ins[channel].rawTags()) {
                if (tag.index < ins[channel].streamIndex) {
                    continue;
                }
                const std::size_t offset = tag.index - ins[channel].streamIndex;
                if (offset >= n) {
                    continue;
                }
                const gr::property_map_view tagMap{tag.map};
                const float* const          wireRate = tagMap.get_if<float>(gr::tag::kSampleRate.key());
                const float* const          rate     = wireRate != nullptr ? wireRate : tagMap.get_if<float>(gr::tag::kSampleRate.shortKey());
                /** the nominal rate, not one measured between anchors */
                if (rate && _time.nominalRateHz > 0. && static_cast<double>(*rate) != _time.nominalRateHz) {
                    truncateOpenWindows(_streamIndex + offset);
                }
                std::ignore = _time.adopt(tag.map, _streamIndex + offset);
                if (rate) {
                    widenReach();
                }
                if (channel == static_cast<std::size_t>(align_channel) && tag.map.contains(gr::tag::kTriggerName.key())) {
                    _lastAlignTag = _streamIndex + offset;
                }
            }
        }
    }

    void openWindows() {
        const std::size_t ready   = _events.ready();
        std::size_t       handled = 0UZ;
        for (const StoredEvent& decision : _events.ordered()) {
            if (handled >= ready) {
                break;
            }
            if (!decision.dated()) {
                if (arm(gr::property_map_view{decision.event})) {
                    ++handled;
                    continue;
                }
                n_undated = n_undated + 1U;
                ++_unreported;
                ++handled;
                continue;
            }
            const gr::property_map_view event{decision.event};
            const std::uint64_t         at    = referenceTimeOf(event, *decision.at, _reference);
            const auto                  start = placeAt(at);
            if (!start) {
                if (!dateable()) {
                    break;
                }
                n_unplaceable = n_unplaceable + 1U;
                ++_unreported;
                ++handled;
                continue;
            }
            if (*start >= _streamIndex) {
                break;
            }
            forceDeadlinesBefore(*start); // auto's deadlines and real decisions in the order of their samples
            if (arm(event)) {
                ++handled;
                continue;
            }
            if (_closeFiltered && detail::acceptsTrigger(_closes, true, event)) {
                close(*start, at, event);
                ++handled;
                continue;
            }
            const bool forced = _forceFiltered && detail::acceptsTrigger(_forces, true, event);
            if (!forced && _openFiltered && !detail::acceptsTrigger(_opens, true, event)) {
                ++handled;
                continue;
            }
            if (!accepts(forced, at)) {
                n_suppressed = n_suppressed + 1U;
                ++handled;
                continue;
            }
            settle(open(event, *start, at, {.forced = forced, .automatic = false}), at, *start);
            ++handled;
        }
        _events.retire(handled);
    }

    /** an event matching `arm_filter` arms and opens nothing; with `rearm` = manual it also re-opens a stopped single recorder */
    [[nodiscard]] bool arm(const gr::property_map_view& event) {
        if (!_armFiltered || !detail::acceptsTrigger(_arms, true, event)) {
            return false;
        }
        _armed   = true;
        _stopped = _stopped && !_manualRearm;
        return true;
    }

    void refuseUnknown(auto& setting, std::string_view key, std::initializer_list<std::string_view> allowed, const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (!newSettings.contains(key) || std::ranges::contains(allowed, std::string_view{setting.value})) {
            return;
        }
        const std::string previous = oldSettings.value_or<std::string>(std::string(key), std::string(*allowed.begin()));
        detail::reportError(*this, "settingsChanged()", std::format("{} '{}' is not one of the known values: keeps '{}'", key, setting.value, previous));
        setting = std::pmr::string(previous);
    }

    /** holdoff refers to the last accepted trigger's `trigger_time`, equality accepted; it is not extended by what it suppresses */
    [[nodiscard]] bool accepts(bool forced, std::uint64_t at) {
        if (_stopped) {
            return false;
        }
        if (forced) {
            return true;
        }
        if (!_armed) {
            return false;
        }
        const auto holdoffNs = detail::nanoseconds(static_cast<double>(holdoff.value));
        if (holdoffNs > 0U && _lastAccepted && at < *_lastAccepted + holdoffNs) {
            return false;
        }
        if (_holdoffLeft > 0U) {
            _holdoffLeft = _holdoffLeft - 1U;
            return false;
        }
        return true;
    }

    /** only a decision that opened or moved a window is accepted; one ignored by `retrigger` is suppressed, a lost or unplaceable one is counted as such */
    void settle(Opened result, std::uint64_t at, std::size_t start) {
        switch (result) {
        case Opened::opened:
        case Opened::merged: accepted(at, start); return;
        case Opened::ignored: n_suppressed = n_suppressed + 1U; return;
        case Opened::lost: return; // n_lost counted where the window was refused
        case Opened::unplaceable:
            n_unplaceable = n_unplaceable + 1U;
            ++_unreported;
            return;
        }
    }

    void accepted(std::uint64_t at, std::size_t start) {
        _lastAccepted       = at;
        _lastAcceptedSample = start;
        _holdoffLeft        = holdoff_count;
        _stopped            = _mode == RecordMode::single;
        _armed              = _armed && !(_manualRearm && _armFiltered);
    }

    /**
     * Auto: no accepted decision for auto_timeout forces a window at that deadline's sample, flagged so it is
     * not taken for a match; deadlines before `limit` are taken in sample order with the real decisions, so neither the
     * stream's cutting nor a refused window changes which come after
     */
    void forceDeadlinesBefore(std::size_t limit) {
        if (_mode != RecordMode::automatic || auto_timeout <= 0.f) {
            return;
        }
        const auto wait = toSamples(detail::nanoseconds(static_cast<double>(auto_timeout)));
        while (wait && *wait > 0UZ && !_stopped && _lastAcceptedSample + *wait < limit) {
            const std::size_t deadline = _lastAcceptedSample + *wait;
            const auto        startNs  = _time.at(deadline);
            if (!startNs) {
                return; // undatable yet: the deadline waits for an anchor
            }
            const Opened result = open(gr::property_map_view{}, deadline, *startNs, {.forced = false, .automatic = true});
            settle(result, *startNs, deadline);
            _lastAcceptedSample = deadline; // a refused deadline is passed, not retried
        }
    }

    [[nodiscard]] RecordMode parseMode(std::string_view text) const {
        if (text == "auto") {
            return RecordMode::automatic;
        }
        const auto named = gr::meta::parseEnum<RecordMode>(text);
        if (!named || *named == RecordMode::automatic) {
            detail::reportError(*this, "settingsChanged()", std::format("mode '{}' is not normal, single or auto; recording normally", text));
            return RecordMode::normal;
        }
        return *named;
    }

    void noteStage() {
        const bool       awaitsSamples = !_collectors.empty() && _collectors[0].nOpen() > 0UZ;
        std::string_view now           = "armed";
        if (awaitsSamples) {
            now = "awaiting_samples";
        } else if (_stopped) {
            now = "stopped";
        } else if (!_armed) {
            now = "unarmed";
        } else if (_holdoffLeft > 0U) {
            now = "holdoff";
        } else if (_streamIndex < n_pre) {
            now = "priming";
        }
        if (stage.value != now) {
            stage = std::pmr::string(now);
        }
    }

    struct Origin {
        bool forced    = false;
        bool automatic = false;
    };

    [[nodiscard]] Opened open(const gr::property_map_view& event, std::size_t start, std::uint64_t startNs, Origin origin) {
        const auto keyed = [&event](const auto& key) -> std::optional<std::uint64_t> {
            const auto* found = event.get_if<std::uint64_t>(std::string_view{key.key()});
            return found ? std::optional{*found} : std::nullopt;
        };
        const auto lengthOf = [this](std::optional<std::uint64_t> fromEvent, std::uint64_t fromSetting, std::size_t fallbackSamples) -> std::optional<std::pair<std::size_t, std::uint64_t>> {
            const std::optional<std::uint64_t> ns = fromEvent ? fromEvent : (fromSetting > 0U ? std::optional{fromSetting} : std::nullopt);
            if (!ns) {
                return std::pair{fallbackSamples, toNs(fallbackSamples)};
            }
            return toSamples(*ns).transform([&ns](std::size_t samples) { return std::pair{samples, *ns}; });
        };
        const auto pre  = lengthOf(keyed(gr::tag::kTriggerPre), trigger_pre, n_pre);
        const auto post = lengthOf(keyed(gr::tag::kTriggerPost), trigger_post, n_post);
        if (!pre || !post) {
            return Opened::unplaceable;
        }
        const std::optional<std::uint64_t> stopNs    = keyed(gr::tag::kTriggerStopTime);
        const std::size_t                  stop      = stopNs ? placeAt(*stopNs).value_or(start) : start;
        const bool                         openEnded = _closeFiltered;
        const std::size_t                  maxLength = max_duration > 0.f ? toSamples(detail::nanoseconds(static_cast<double>(max_duration))).value_or(_reach) : _reach;
        const RecordedSpan                 span{.startIndex = start, .startNs = startNs, .stopNs = stopNs.value_or(0U), .preNs = pre->second, .postNs = post->second, .truncated = openEnded, .rateChanged = false, .forced = origin.forced, .automatic = origin.automatic};
        const std::size_t                  first    = start >= pre->first ? start - pre->first : 0UZ;
        const std::size_t                  closesAt = openEnded ? start + maxLength : std::max(start, stop) + post->first;
        const bool                         partial  = start < pre->first;

        if (const auto current = newestWindow(); current && current->closesAt > start && _retrigger != RecordRetrigger::parallel) {
            if (_retrigger == RecordRetrigger::ignore) {
                return Opened::ignored;
            }
            const std::size_t  startsAt = _retrigger == RecordRetrigger::restart ? first : current->startsAt;
            const std::size_t  closes   = _retrigger == RecordRetrigger::restart ? closesAt : std::max(current->closesAt, closesAt);
            const RecordedSpan kept     = _retrigger == RecordRetrigger::restart ? span : current->info;
            const bool         cut      = _retrigger == RecordRetrigger::restart ? partial : current->partial;
            for (WindowCollector<T, RecordedSpan>& collector : _collectors) {
                eraseNewest(collector);
                std::ignore = collector.open(startsAt, closes, kept, cut);
            }
            return Opened::merged;
        }
        if (!_collectors.empty() && _collectors[0].nOpen() >= static_cast<std::size_t>(max_open)) {
            n_lost = n_lost + 1U;
            ++_unreported;
            return Opened::lost;
        }
        if (first + _reach + _chunk < _streamIndex) {
            return Opened::unplaceable;
        }
        for (WindowCollector<T, RecordedSpan>& collector : _collectors) {
            std::ignore = collector.open(first, closesAt, span, partial);
        }
        return Opened::opened;
    }

    /**
     * the history a window may reach back into; it widens with a longer window or a newly known rate and never
     * shrinks, so what is already recorded and open survives a setting a stream tag updates
     */
    void widenReach() {
        const std::size_t longest = std::max({static_cast<std::size_t>(n_pre + n_post), toSamples(trigger_pre).value_or(0UZ) + toSamples(trigger_post).value_or(0UZ), toSamples(detail::nanoseconds(static_cast<double>(max_duration))).value_or(0UZ)});
        const double      rate    = nominalRate();
        const std::size_t late    = toSamples(detail::nanoseconds(static_cast<double>(max_lateness))).value_or(0UZ); // a decision may arrive that much later
        const std::size_t asked   = history > 0.f && rate > 0. ? static_cast<std::size_t>(std::ceil(static_cast<double>(history) * rate)) : 2UZ * (longest + 1UZ) + 1UZ + late;
        _reach                    = std::max({_reach, asked, static_cast<std::size_t>(n_pre + n_post) + 2UZ + late});
        for (WindowCollector<T, RecordedSpan>& collector : _collectors) {
            collector.maxOpen = std::max<std::size_t>(max_open, 1UZ);
            collector.reachBack(_reach);
        }
    }

    /** the window of the latest decision not yet emitted, open or closed; the same in every channel's collector */
    template<typename TCollector>
    [[nodiscard]] static auto newestIn(TCollector& collector) {
        decltype(&collector.openWindows.front()) newest = nullptr;
        for (auto* windows : {&collector.openWindows, &collector.closedWindows}) {
            for (auto& window : *windows) {
                if (newest == nullptr || window.info.startIndex >= newest->info.startIndex) {
                    newest = &window;
                }
            }
        }
        return newest;
    }

    [[nodiscard]] const typename WindowCollector<T, RecordedSpan>::Window* newestWindow() const { return _collectors.empty() ? nullptr : newestIn(_collectors[0]); }

    static void eraseNewest(WindowCollector<T, RecordedSpan>& collector) {
        const auto* newest = newestIn(collector);
        for (auto* windows : {&collector.openWindows, &collector.closedWindows}) {
            std::erase_if(*windows, [newest](const auto& window) { return &window == newest; });
        }
    }

    void close(std::size_t at, std::uint64_t atNs, const gr::property_map_view& event) {
        if (_collectors.empty()) {
            return;
        }
        const auto& windows = _collectors[0].openWindows;
        const auto  oldest  = std::ranges::find_if(windows, [](const auto& window) { return window.info.truncated && !window.info.rateChanged; });
        if (oldest == windows.end()) {
            n_unmatched = n_unmatched + 1U;
            return;
        }
        const auto          position = static_cast<std::size_t>(std::ranges::distance(windows.begin(), oldest));
        const auto*         ownPost  = event.get_if<std::uint64_t>(std::string_view{gr::tag::kTriggerPost.key()});
        const std::uint64_t postNs   = ownPost ? *ownPost : oldest->info.postNs;
        const std::size_t   post     = postNs > 0U ? toSamples(postNs).value_or(static_cast<std::size_t>(n_post)) : static_cast<std::size_t>(n_post);
        for (WindowCollector<T, RecordedSpan>& collector : _collectors) {
            auto& window          = collector.openWindows[position];
            window.closesAt       = std::max(window.startsAt, std::min(window.closesAt, at + post));
            window.info.stopNs    = atNs;
            window.info.truncated = false;
        }
    }

    void truncateOpenWindows(std::size_t at) {
        for (WindowCollector<T, RecordedSpan>& collector : _collectors) {
            for (auto& window : collector.openWindows) {
                if (window.closesAt > at) {
                    window.closesAt         = std::max(window.startsAt + 1UZ, at);
                    window.info.truncated   = true;
                    window.info.rateChanged = true;
                }
            }
        }
    }

    [[nodiscard]] double nominalRate() const noexcept { return _time.nominalRateHz > 0. ? _time.nominalRateHz : static_cast<double>(sample_rate); }

    [[nodiscard]] std::optional<std::size_t> toSamples(std::uint64_t ns) const {
        const double rate = nominalRate();
        if (ns == 0U) {
            return 0UZ;
        }
        if (rate <= 0.) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(std::llround(static_cast<double>(ns) * rate * 1e-9));
    }

    [[nodiscard]] std::uint64_t toNs(std::size_t samples) const {
        const double rate = nominalRate();
        return rate > 0. ? static_cast<std::uint64_t>(std::llround(static_cast<double>(samples) * 1e9 / rate)) : 0U;
    }

    void reportRefusals() {
        if (_unreported == 0UZ) {
            return;
        }
        detail::reportError(*this, "processBulk()", std::format("{} window(s) refused or dropped: undated, unplaceable, or cut short", _unreported));
        _unreported = 0UZ;
    }

    [[nodiscard]] bool dateable() const noexcept { return _align == RecordAlign::channel ? _lastAlignTag.has_value() : _time.datesSamples(); }

    [[nodiscard]] std::optional<std::size_t> placeAt(std::uint64_t at) const {
        if (_align == RecordAlign::channel) {
            return _lastAlignTag;
        }
        return _time.indexAt(at);
    }

    [[nodiscard]] std::size_t emitReady(auto& outSpan) {
        std::size_t emitted = 0UZ;
        while (emitted < outSpan.size() && !_collectors.empty() && (n_segments == 0U || n_recorded < n_segments)) {
            std::vector<std::vector<T>>                                      channels;
            std::optional<typename WindowCollector<T, RecordedSpan>::Window> window;
            channels.reserve(_collectors.size());
            for (WindowCollector<T, RecordedSpan>& collector : _collectors) {
                auto taken = collector.take();
                if (!taken) {
                    break;
                }
                channels.push_back(std::move(taken->samples));
                window = std::move(taken);
            }
            if (channels.empty()) {
                return emitted;
            }
            const bool cutShort = window && (window->partial || window->info.truncated || window->info.rateChanged);
            if (!window || channels.size() != _collectors.size() || (cutShort && !emit_partial)) {
                n_lost = n_lost + 1U;
                ++_unreported;
                continue;
            }
            outSpan[emitted++] = assemble(channels, *window);
            n_recorded         = n_recorded + 1U;
        }
        return emitted;
    }

    [[nodiscard]] gr::DataSet<T> assemble(const std::vector<std::vector<T>>& channels, const typename WindowCollector<T, RecordedSpan>::Window& window) const {
        gr::DataSet<T>      set;
        const RecordedSpan& span   = window.info;
        const std::size_t   length = channels.empty() ? 0UZ : channels[0].size();
        const std::size_t   before = span.startIndex - window.startsAt;

        set.axis_names.emplace_back("time");
        set.axis_units.emplace_back("s");
        set.axis_values.resize(1UZ);
        const double period = nominalRate() > 0. ? 1. / nominalRate() : 1.;
        for (std::size_t i = 0UZ; i < length; ++i) {
            set.axis_values[0].emplace_back(static_cast<float>((static_cast<double>(i) - static_cast<double>(before)) * period));
        }

        set.extents.emplace_back(static_cast<std::int32_t>(length));
        set.timestamp = static_cast<std::int64_t>(span.startNs);
        for (std::size_t channel = 0UZ; channel < channels.size(); ++channel) {
            set.signal_names.emplace_back(std::format("{}{}", signal_name.value, channel));
            set.signal_quantities.emplace_back("");
            set.signal_units.emplace_back("");
            set.signal_values.insert(set.signal_values.end(), channels[channel].begin(), channels[channel].end());
        }
        set.signal_ranges.resize(channels.size());
        set.meta_information.resize(channels.size());
        set.timing_events.resize(channels.size());
        for (auto& info : set.meta_information) {
            info.insert_or_assign(std::string_view{"n_pre"}, static_cast<gr::Size_t>(before));
            info.insert_or_assign(std::string_view{"n_post"}, static_cast<gr::Size_t>(window.closesAt - window.startsAt - before));
            info.insert_or_assign(std::string_view{gr::tag::kTriggerTime.key()}, span.startNs);
            if (span.stopNs != 0U) {
                info.insert_or_assign(std::string_view{gr::tag::kTriggerStopTime.key()}, span.stopNs);
            }
            info.insert_or_assign(std::string_view{gr::tag::kTriggerPre.key()}, span.preNs);
            info.insert_or_assign(std::string_view{gr::tag::kTriggerPost.key()}, span.postNs);
            if (window.partial || span.truncated || span.rateChanged || span.forced || span.automatic) {
                gr::property_map flags;
                for (const auto& [flag, raised] : {std::pair{"incomplete", window.partial}, std::pair{"truncated", span.truncated}, std::pair{"rate_changed", span.rateChanged}, std::pair{"forced", span.forced}, std::pair{"auto_timeout", span.automatic}}) {
                    if (raised) {
                        flags[flag] = true;
                    }
                }
                info.insert_or_assign(std::string_view{gr::tag::kTriggerMetaInfo.key()}, gr::property_map{{std::string(gr::tag::kAcquisitionFlags.key()), std::move(flags)}});
            }
        }
        return set;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_MULTICHANNELRECORDER_HPP
