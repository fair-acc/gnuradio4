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
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <gnuradio-4.0/trigger/WindowCollector.hpp>

namespace gr::blocks::trigger {

enum class RecordAlign : std::uint8_t { composite, channel };
enum class RecordRetrigger : std::uint8_t { ignore, restart, extend, parallel };

/** what a recorded window remembers of the decision that opened it */
struct RecordedSpan {
    /** stream index of the window's start */
    std::size_t decision = 0UZ;
    /** ns, UTC */
    std::uint64_t start = 0U;
    /** ns, UTC; 0 = a single trigger */
    std::uint64_t stop = 0U;
    /** ns */
    std::uint64_t pre = 0U;
    /** ns */
    std::uint64_t post = 0U;
    /** ended by max_duration or a rate change; an unclosed window carries it until closed */
    bool truncated   = false;
    bool rateChanged = false;
};

GR_REGISTER_BLOCK(gr::blocks::trigger::MultiChannelRecorder, [T], [ int16_t, int32_t, float, double ])

template<typename T>
struct MultiChannelRecorder : gr::Block<MultiChannelRecorder<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief extract the same window from every channel on one decision, as one DataSet

    evtIn  ────────X────────▶        (no RxMarbles equivalent)
    in#0   ─a─a─a─a─a─a─a─a─▶        n_pre = 2, n_post = 2
    in#1   ─b─b─b─b─b─b─b─b─▶
    out    ────────D────────▶        D = one DataSet, one signal per channel

The transient recorder: N streams (one is fine), one shared decision, one object. A decision opens the window from `pre`
before its `trigger_time` to `post` after its `gr:trigger_stop_time` (else its `trigger_time`); pre and post come from the
event's `gr:trigger_pre`/`gr:trigger_post`, else from the `trigger_pre`/`trigger_post` settings, 0 taking the sample
counts `n_pre`/`n_post`. With `close_filter`, an event matching `open_filter` (any dated event if empty) opens a window
that the next event matching `close_filter` closes, oldest first; one never closed ends after `max_duration`. Each
signal's meta information carries the window's keys and, when it was cut short, `gr:acquisition_flags`. A `sample_rate`
of 0 means not yet known; `signal_name` is a prefix, each signal named `<signal_name><channel index>`.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                        evtIn;
    std::vector<gr::PortIn<T>>             in;
    gr::PortOut<gr::DataSet<T>, gr::Async> out;
    gr::EventPortOut                       evtOut;

    A<gr::Size_t, "n inputs", Doc<"channels recorded together">, gr::Limits<1U, 32U>>                                         n_inputs     = 0U;
    A<gr::Size_t, "n pre", Doc<"samples before the window, if no event or setting says">>                                     n_pre        = 0U;
    A<gr::Size_t, "n post", Doc<"samples after it, if no event or setting says">>                                             n_post       = 0U;
    gr::TriggerPre                                                                                                            trigger_pre  = 0U;
    gr::TriggerPost                                                                                                           trigger_post = 0U;
    A<float, "history", Doc<"how far back a window may reach; 0 = twice the longest">, gr::Unit<"s">>                         history      = 0.f;
    A<bool, "emit partial", Doc<"a window cut short: emitted flagged, else dropped and reported">>                            emit_partial = true;
    A<std::pmr::string, "open filter", Doc<"trigger filter of the events that open a window; empty = all">>                   open_filter;
    A<std::pmr::string, "close filter", Doc<"trigger filter of the events that close one; empty = windows close themselves">> close_filter;
    A<std::pmr::string, "retrigger", Doc<"a decision while a window is open: ignore|restart|extend|parallel">>                retrigger     = std::pmr::string("parallel");
    A<gr::Size_t, "max open", Doc<"windows open at once with retrigger = parallel">>                                          max_open      = 16U;
    A<float, "max duration", Doc<"how long an unclosed window may stay open; 0 = history">, gr::Unit<"s">>                    max_duration  = 0.f;
    A<std::pmr::string, "align", Doc<"composite|channel">>                                                                    align         = std::pmr::string("composite");
    A<gr::Size_t, "align channel", Doc<"'channel' only: whose trigger lines the window up">>                                  align_channel = 0U;
    gr::SampleRate                                                                                                            sample_rate   = 0.f;
    gr::SignalName                                                                                                            signal_name   = std::pmr::string("channel");
    A<gr::Size_t, "n segments", Doc<"stop after this many, 0 = without end">>                                                 n_segments    = 0U;
    A<gr::Size_t, "n recorded">                                                                                               n_recorded    = 0U;
    A<gr::Size_t, "n undated", Doc<"decisions carrying no time, which cannot be placed">>                                     n_undated     = 0U;
    A<gr::Size_t, "n unplaceable", Doc<"decisions whose sample could not be found">>                                          n_unplaceable = 0U;
    A<gr::Size_t, "n lost", Doc<"windows that could not be completed">>                                                       n_lost        = 0U;
    A<gr::Size_t, "n unmatched", Doc<"closes that found no open window">>                                                     n_unmatched   = 0U;

    GR_MAKE_REFLECTABLE(MultiChannelRecorder, evtIn, in, out, evtOut, n_inputs, n_pre, n_post, trigger_pre, trigger_post, history, emit_partial, open_filter, close_filter, retrigger, max_open, max_duration, align, align_channel, sample_rate, signal_name, n_segments, n_recorded, n_undated, n_unplaceable, n_lost, n_unmatched);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    RecordAlign                                   _align     = RecordAlign::composite;
    RecordRetrigger                               _retrigger = RecordRetrigger::parallel;
    MatchState                                    _opens{};
    MatchState                                    _closes{};
    bool                                          _openFiltered  = false;
    bool                                          _closeFiltered = false;
    std::size_t                                   _reach         = 0UZ;
    std::vector<WindowCollector<T, RecordedSpan>> _collectors;
    TimeBase                                      _time;
    std::size_t                                   _streamIndex = 0UZ;
    EventStore                                    _events;
    std::optional<std::size_t>                    _lastAlignTag;
    std::size_t                                   _unreported = 0UZ;

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("n_inputs") && oldSettings.find_value("n_inputs") != newSettings.find_value("n_inputs")) {
            in.resize(n_inputs);
        }
        _align         = align == "channel" ? RecordAlign::channel : RecordAlign::composite;
        _retrigger     = retrigger == "ignore" ? RecordRetrigger::ignore : retrigger == "restart" ? RecordRetrigger::restart : retrigger == "extend" ? RecordRetrigger::extend : RecordRetrigger::parallel;
        _openFiltered  = detail::compileOptionalFilter(open_filter.value, std::string_view{}, _opens, "MultiChannelRecorder");
        _closeFiltered = detail::compileOptionalFilter(close_filter.value, std::string_view{}, _closes, "MultiChannelRecorder");
        _time.setRate(static_cast<double>(sample_rate));

        if (_collectors.size() != in.size()) {
            _collectors.assign(in.size(), WindowCollector<T, RecordedSpan>{});
            _streamIndex  = 0UZ;
            _lastAlignTag = std::nullopt;
            _events.clear();
            n_recorded    = 0U;
            n_undated     = 0U;
            n_unplaceable = 0U;
            n_lost        = 0U;
            n_unmatched   = 0U;
        }
        widenReach();
    }

    template<gr::InputSpanLike TInput>
    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, const std::span<TInput>& ins, gr::OutputSpanLike auto& outSpan, gr::OutputSpanLike auto& evtOutSpan) {
        _events.drain(evtSpan);
        if (ins.empty()) {
            evtOutSpan.publish(reportRefusals(evtOutSpan));
            return gr::work::Status::OK;
        }

        const std::size_t n = std::ranges::min(ins | std::views::transform([](const auto& s) { return s.size(); }));

        noteTagsAndTime(ins, n);
        for (std::size_t channel = 0UZ; channel < ins.size() && channel < _collectors.size(); ++channel) {
            _collectors[channel].push(std::span<const T>{ins[channel].begin(), n});
        }
        _streamIndex += n;

        openWindows();
        const std::size_t emitted = emitReady(outSpan);

        outSpan.publish(emitted);
        evtOutSpan.publish(reportRefusals(evtOutSpan));
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
                const float* const          wireRate = tagMap.get_if<float>(gr::tag::SAMPLE_RATE.key());
                const float* const          rate     = wireRate != nullptr ? wireRate : tagMap.get_if<float>(gr::tag::SAMPLE_RATE.shortKey());
                /** the nominal rate, not one measured between anchors */
                if (rate && _time.nominalRateHz > 0. && static_cast<double>(*rate) != _time.nominalRateHz) {
                    truncateOpenWindows(_streamIndex + offset);
                }
                std::ignore = _time.adopt(tag.map, _streamIndex + offset);
                if (rate) {
                    widenReach();
                }
                if (channel == static_cast<std::size_t>(align_channel) && tag.map.contains(gr::tag::TRIGGER_NAME.key())) {
                    _lastAlignTag = _streamIndex + offset;
                }
            }
        }
    }

    void openWindows() {
        std::size_t handled = 0UZ;
        for (const StoredEvent& decision : _events.ordered()) {
            if (!decision.dated()) {
                n_undated = n_undated + 1U;
                ++_unreported;
                ++handled;
                continue;
            }
            const auto start = placeAt(*decision.at);
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
            const gr::property_map_view event{decision.event};
            if (_closeFiltered && detail::acceptsTrigger(_closes, true, event)) {
                close(*start, *decision.at, event);
                ++handled;
                continue;
            }
            if (_openFiltered && !detail::acceptsTrigger(_opens, true, event)) {
                ++handled;
                continue;
            }
            if (!open(event, *start, *decision.at)) {
                n_unplaceable = n_unplaceable + 1U;
                ++_unreported;
            }
            ++handled;
        }
        _events.retire(handled);
    }

    [[nodiscard]] bool open(const gr::property_map_view& event, std::size_t start, std::uint64_t startNs) {
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
        const auto pre  = lengthOf(keyed(gr::tag::TRIGGER_PRE), trigger_pre, n_pre);
        const auto post = lengthOf(keyed(gr::tag::TRIGGER_POST), trigger_post, n_post);
        if (!pre || !post) {
            return false;
        }
        const std::optional<std::uint64_t> stopNs    = keyed(gr::tag::TRIGGER_STOP_TIME);
        const std::size_t                  stop      = stopNs ? placeAt(*stopNs).value_or(start) : start;
        const bool                         openEnded = _closeFiltered;
        const std::size_t                  maxLength = max_duration > 0.f ? toSamples(static_cast<std::uint64_t>(static_cast<double>(max_duration) * 1e9)).value_or(_reach) : _reach;
        const RecordedSpan                 span{.decision = start, .start = startNs, .stop = stopNs.value_or(0U), .pre = pre->second, .post = post->second, .truncated = openEnded};
        const std::size_t                  first    = start >= pre->first ? start - pre->first : 0UZ;
        const std::size_t                  closesAt = 1UZ + (openEnded ? start + maxLength : std::max(start, stop) + post->first);
        const bool                         partial  = start < pre->first;

        if (const auto current = newestWindow(); current && current->closesAt > start && _retrigger != RecordRetrigger::parallel) {
            if (_retrigger == RecordRetrigger::ignore) {
                return true;
            }
            const std::size_t  startsAt = _retrigger == RecordRetrigger::restart ? first : current->startsAt;
            const std::size_t  closes   = _retrigger == RecordRetrigger::restart ? closesAt : std::max(current->closesAt, closesAt);
            const RecordedSpan kept     = _retrigger == RecordRetrigger::restart ? span : current->info;
            const bool         cut      = _retrigger == RecordRetrigger::restart ? partial : current->partial;
            for (WindowCollector<T, RecordedSpan>& collector : _collectors) {
                eraseNewest(collector);
                std::ignore = collector.open(startsAt, closes, kept, cut);
            }
            return true;
        }
        if (!_collectors.empty() && _collectors[0].nOpen() >= static_cast<std::size_t>(max_open)) {
            n_lost = n_lost + 1U;
            ++_unreported;
            return true;
        }
        if (first + _reach < _streamIndex) {
            return false;
        }
        for (WindowCollector<T, RecordedSpan>& collector : _collectors) {
            std::ignore = collector.open(first, closesAt, span, partial);
        }
        return true;
    }

    /**
     * the history a window may reach back into; it widens with a longer window or a newly known rate and never
     * shrinks, so what is already recorded and open survives a setting a stream tag updates
     */
    void widenReach() {
        const std::size_t longest = std::max({static_cast<std::size_t>(n_pre + n_post), toSamples(trigger_pre).value_or(0UZ) + toSamples(trigger_post).value_or(0UZ), toSamples(static_cast<std::uint64_t>(static_cast<double>(max_duration) * 1e9)).value_or(0UZ)});
        const double      rate    = nominalRate();
        const std::size_t asked   = history > 0.f && rate > 0. ? static_cast<std::size_t>(std::ceil(static_cast<double>(history) * rate)) : 2UZ * (longest + 1UZ) + 1UZ;
        _reach                    = std::max({_reach, asked, static_cast<std::size_t>(n_pre + n_post) + 2UZ});
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
                if (newest == nullptr || window.info.decision >= newest->info.decision) {
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
        const auto*         ownPost  = event.get_if<std::uint64_t>(std::string_view{gr::tag::TRIGGER_POST.key()});
        const std::uint64_t postNs   = ownPost ? *ownPost : oldest->info.post;
        const std::size_t   post     = postNs > 0U ? toSamples(postNs).value_or(static_cast<std::size_t>(n_post)) : static_cast<std::size_t>(n_post);
        for (WindowCollector<T, RecordedSpan>& collector : _collectors) {
            auto& window          = collector.openWindows[position];
            window.closesAt       = std::max(window.startsAt + 1UZ, std::min(window.closesAt, at + post + 1UZ));
            window.info.stop      = atNs;
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

    [[nodiscard]] std::size_t reportRefusals(auto& evtOutSpan) {
        if (_unreported == 0UZ) {
            return 0UZ;
        }
        const std::string reason = std::format("{} window(s) refused or dropped: undated, unplaceable, or cut short", _unreported);
        gr::log::warning("{}: {}", this->name, reason);
        if (!gr::emit(evtOutSpan, 0UZ, detail::makeErrorEvent(reason, this->unique_name))) {
            return 0UZ;
        }
        _unreported = 0UZ;
        return 1UZ;
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
        const std::size_t   before = span.decision - window.startsAt;

        set.axis_names.emplace_back("time");
        set.axis_units.emplace_back("s");
        set.axis_values.resize(1UZ);
        const double period = nominalRate() > 0. ? 1. / nominalRate() : 1.;
        for (std::size_t i = 0UZ; i < length; ++i) {
            set.axis_values[0].emplace_back(static_cast<float>((static_cast<double>(i) - static_cast<double>(before)) * period));
        }

        set.extents.emplace_back(static_cast<std::int32_t>(length));
        set.timestamp = static_cast<std::int64_t>(span.start);
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
            info.insert_or_assign(std::string_view{"n_post"}, static_cast<gr::Size_t>(window.closesAt - window.startsAt - before - 1UZ));
            info.insert_or_assign(std::string_view{gr::tag::TRIGGER_TIME.key()}, span.start);
            if (span.stop != 0U) {
                info.insert_or_assign(std::string_view{gr::tag::TRIGGER_STOP_TIME.key()}, span.stop);
            }
            info.insert_or_assign(std::string_view{gr::tag::TRIGGER_PRE.key()}, span.pre);
            info.insert_or_assign(std::string_view{gr::tag::TRIGGER_POST.key()}, span.post);
            if (window.partial || span.truncated || span.rateChanged) {
                gr::property_map flags;
                for (const auto& [flag, raised] : {std::pair{"incomplete", window.partial}, std::pair{"truncated", span.truncated}, std::pair{"rate_changed", span.rateChanged}}) {
                    if (raised) {
                        flags[std::string(flag)] = true;
                    }
                }
                info.insert_or_assign(std::string_view{gr::tag::TRIGGER_META_INFO.key()}, gr::property_map{{std::string(gr::tag::ACQUISITION_FLAGS.key()), std::move(flags)}});
            }
        }
        return set;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_MULTICHANNELRECORDER_HPP
