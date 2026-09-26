#ifndef GNURADIO_TRIGGER_MULTICHANNELRECORDER_HPP
#define GNURADIO_TRIGGER_MULTICHANNELRECORDER_HPP

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
#include <gnuradio-4.0/trigger/SegmentCollector.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

namespace gr::blocks::trigger {

enum class RecordAlign : std::uint8_t { composite, channel };

GR_REGISTER_BLOCK(gr::blocks::trigger::MultiChannelRecorder, [T], [ int16_t, int32_t, float, double ])

template<typename T>
struct MultiChannelRecorder : gr::Block<MultiChannelRecorder<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief extract the same window from every channel on one decision, as one DataSet

    evtIn  ────────X────────▶        (no RxMarbles equivalent)
    in#0   ─a─a─a─a─a─a─a─a─▶        n_pre = 2, n_post = 2
    in#1   ─b─b─b─b─b─b─b─b─▶
    out    ────────D────────▶        D = one DataSet, one signal per channel

The transient recorder: N streams, one shared decision, one object.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                        evtIn;
    std::vector<gr::PortIn<T>>             in;
    gr::PortOut<gr::DataSet<T>, gr::Async> out;

    A<gr::Size_t, "n inputs", Doc<"channels recorded together">, gr::Limits<1U, 32U>>        n_inputs       = 0U;
    A<gr::Size_t, "n pre", Doc<"samples kept from before the decision">>                     n_pre          = 0U;
    A<gr::Size_t, "n post", Doc<"samples kept from after it">>                               n_post         = 0U;
    A<gr::Size_t, "history margin", Doc<"extra history, samples">>                           history_margin = 0U;
    A<std::pmr::string, "align", Doc<"composite|channel">>                                   align          = std::pmr::string("composite");
    A<gr::Size_t, "align channel", Doc<"'channel' only: whose trigger lines the window up">> align_channel  = 0U;
    A<float, "sample rate", Doc<"Hz, 0 = from the tags">>                                    sample_rate    = 0.f;
    A<std::pmr::string, "signal name", Doc<"prefix for the recorded signals">>               signal_name    = std::pmr::string("channel");
    A<gr::Size_t, "n segments", Doc<"stop after this many, 0 = without end">>                n_segments     = 0U;
    A<gr::Size_t, "n recorded">                                                              n_recorded     = 0U;
    A<gr::Size_t, "n undated", Doc<"decisions carrying no time, which cannot be placed">>    n_undated      = 0U;
    A<gr::Size_t, "n unplaceable", Doc<"decisions whose sample could not be found">>         n_unplaceable  = 0U;
    A<gr::Size_t, "n lost", Doc<"samples gone before being read">>                           n_lost         = 0U;

    GR_MAKE_REFLECTABLE(MultiChannelRecorder, evtIn, in, out, n_inputs, n_pre, n_post, history_margin, align, align_channel, sample_rate, signal_name, n_segments, n_recorded, n_undated, n_unplaceable, n_lost);

    RecordAlign                      _align = RecordAlign::composite;
    std::vector<SegmentCollector<T>> _collectors;
    TimeBase                         _time;
    std::size_t                      _streamIndex = 0UZ;
    EventStore                       _events;
    std::optional<std::size_t>       _lastAlignTag; // the reference channel's most recent trigger

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("n_inputs") && oldSettings.find_value("n_inputs") != newSettings.find_value("n_inputs")) {
            in.resize(n_inputs);
        }
        _align = align == "channel" ? RecordAlign::channel : RecordAlign::composite;
        _time.setRate(static_cast<double>(sample_rate));

        const bool windowChanged = _collectors.size() != in.size() || newSettings.contains("n_pre") || newSettings.contains("n_post") || newSettings.contains("history_margin");
        if (windowChanged) {
            _collectors.assign(in.size(), SegmentCollector<T>{});
            for (SegmentCollector<T>& collector : _collectors) {
                collector.setWindow(n_pre, n_post, history_margin == 0U ? std::dynamic_extent : static_cast<std::size_t>(history_margin));
            }
            _streamIndex  = 0UZ;
            _lastAlignTag = std::nullopt;
            _events.clear();
            n_recorded    = 0U;
            n_undated     = 0U;
            n_unplaceable = 0U;
            n_lost        = 0U;
        }
    }

    template<gr::InputSpanLike TInput>
    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, const std::span<TInput>& ins, gr::OutputSpanLike auto& outSpan) {
        _events.drain(evtSpan);
        if (ins.empty()) {
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
                std::ignore = _time.adopt(tag.map, _streamIndex + offset);
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
                ++handled;
                continue;
            }
            const auto index = placeAt(*decision.at);
            if (!index) {
                if (!dateable()) {
                    break;
                }
                n_unplaceable = n_unplaceable + 1U;
                ++handled;
                continue;
            }
            for (SegmentCollector<T>& collector : _collectors) {
                std::ignore = collector.open(*index, *decision.at);
            }
            ++handled;
        }
        _events.retire(handled);
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
        while (emitted < outSpan.size() && !_collectors.empty() && _collectors[0].ready()) {
            if (n_segments != 0U && n_recorded >= n_segments) {
                return emitted;
            }
            std::vector<std::vector<T>> channels;
            std::uint64_t               at = 0U;
            channels.reserve(_collectors.size());
            bool complete = true;
            for (SegmentCollector<T>& collector : _collectors) {
                auto segment = collector.take();
                if (!segment) {
                    complete = false;
                    break;
                }
                at = segment->second;
                channels.push_back(std::move(segment->first));
            }
            if (!complete) {
                n_lost = n_lost + 1U;
                continue;
            }
            outSpan[emitted++] = assemble(channels, at);
            n_recorded         = n_recorded + 1U;
        }
        return emitted;
    }

    [[nodiscard]] gr::DataSet<T> assemble(const std::vector<std::vector<T>>& channels, std::uint64_t at) const {
        gr::DataSet<T>    set;
        const std::size_t length = channels.empty() ? 0UZ : channels[0].size();

        set.axis_names.emplace_back("time");
        set.axis_units.emplace_back("s");
        set.axis_values.resize(1UZ);
        const double period = _time.rate() > 0. ? 1. / _time.rate() : 1.;
        for (std::size_t i = 0UZ; i < length; ++i) {
            set.axis_values[0].emplace_back(static_cast<float>((static_cast<double>(i) - static_cast<double>(n_pre)) * period));
        }

        set.extents.emplace_back(static_cast<std::int32_t>(length));
        set.timestamp = static_cast<std::int64_t>(at);
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
            info.insert_or_assign(std::string_view{"n_pre"}, n_pre.value);
            info.insert_or_assign(std::string_view{"n_post"}, n_post.value);
            info.insert_or_assign(std::string_view{gr::tag::TRIGGER_TIME.shortKey()}, at);
        }
        return set;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_MULTICHANNELRECORDER_HPP
