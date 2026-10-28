#ifndef GNURADIO_TRIGGER_TAGBRIDGE_HPP
#define GNURADIO_TRIGGER_TAGBRIDGE_HPP

#include <cmath>
#include <cstdint>
#include <deque>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <optional>
#include <string>
#include <string_view>

namespace gr::blocks::trigger {

inline constexpr std::string_view kInjectionTag  = "tag";
inline constexpr std::string_view kInjectionTime = "at";
inline constexpr std::string_view kInjectionLate = "late";

GR_REGISTER_BLOCK(gr::blocks::trigger::EventToTag, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct EventToTag : gr::Block<EventToTag<T>> {
    using Description = Doc<R"(@brief publish a stream tag for each event that asks for one, in arrival order

    evtIn   ─E────E──▶               (no RxMarbles equivalent)
    in      ─a─a─a─a─a─▶
    out     ─T:a─a─T:a─a─▶           placed at the sample the event's time names
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn evtIn;
    gr::PortIn<T>   in;
    gr::PortOut<T>  out;

    gr::SampleRate                                                                                                                               sample_rate        = 1.0f;
    A<double, "delay", Doc<"s, added to the time a request asks for and to the trigger_time of its tag, trigger_offset kept; 0 = as requested">> delay              = 0.;
    A<gr::Size_t, "max pending events", Doc<"requests held; a further one drops the oldest">>                                                    max_pending_events = static_cast<gr::Size_t>(detail::kMaxQueuedEvents);
    A<gr::Size_t, "n events dropped", Doc<"requests dropped because the queue was full">>                                                        n_events_dropped   = 0U;
    A<gr::Size_t, "late", Doc<"requests published after the time they asked for">>                                                               n_late             = 0U;

    GR_MAKE_REFLECTABLE(EventToTag, evtIn, in, out, sample_rate, delay, max_pending_events, n_events_dropped, n_late);

    detail::PendingEvents _pending; /// requests as {kInjectionTag: tag, kInjectionTime: at (optional)}, both already re-dated by `delay`
    TimeBase              _time;
    std::size_t           _streamIndex = 0UZ;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) { _pending.capacity = static_cast<std::size_t>(max_pending_events); }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        acceptRequests(evtSpan);

        const std::size_t n = std::min(inSpan.size(), outSpan.size());
        for (const auto& tag : inSpan.rawTags()) {
            adoptAnchor(tag.map, tag.index - inSpan.streamIndex);
        }

        for (std::size_t i = 0UZ; i < n; ++i) {
            gr::property_map merged;
            while (!_pending.queue.empty()) {
                const auto overdue = lateness(gr::property_map_view{_pending.queue.front()}, _streamIndex + i);
                if (!overdue) {
                    break;
                }
                if (*overdue) {
                    merged[kInjectionLate] = true;
                    ++n_late;
                }
                if (const auto tag = gr::property_map_view{_pending.queue.front()}.template get_if<gr::property_map>(kInjectionTag)) {
                    for (const auto& [key, value] : *tag) {
                        merged.insert_or_assign(key, value);
                    }
                }
                _pending.queue.pop_front();
            }
            if (!merged.empty()) {
                outSpan.publishTag(merged, i);
            }
        }
        _pending.reportLoss(*this);
        n_events_dropped = static_cast<gr::Size_t>(_pending.dropped);

        std::ranges::copy(inSpan | std::views::take(n), outSpan.begin());
        _streamIndex += n;
        outSpan.publish(n);
        if (!evtSpan.consume(evtSpan.size()) || !inSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void acceptRequests(gr::InputSpanLike auto& evtSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (event.empty()) {
                continue;
            }
            gr::property_map tag;
            if (const auto carried = event.template get_if<gr::property_map>(kInjectionTag)) {
                tag = *carried;
            }
            if (tag.empty()) {
                detail::reportError(*this, "processBulk()", std::format("an event carried no '{}' map", kInjectionTag));
                continue;
            }
            const auto shifted = [&](std::uint64_t ns) {
                const auto delayNs = static_cast<std::int64_t>(std::llround(delay.value * 1e9));
                return static_cast<std::uint64_t>(static_cast<std::int64_t>(ns) + delayNs);
            };
            if (const auto* stamp = gr::property_map_view{tag}.template get_if<std::uint64_t>(gr::tag::kTriggerTime.key()); stamp != nullptr) {
                tag[gr::tag::kTriggerTime.key()] = shifted(*stamp);
            }
            gr::property_map request{{std::string(kInjectionTag), std::move(tag)}};
            if (const auto at = event.template get_if<std::uint64_t>(kInjectionTime)) {
                request[kInjectionTime] = shifted(*at);
            }
            _pending.push(std::move(request));
        }
    }

    void adoptAnchor(const gr::property_map_view& tagMap, std::size_t offset) {
        _time.setRate(static_cast<double>(sample_rate));
        if (_time.adopt(tagMap, _streamIndex + offset) == AnchorChange::backward) {
            detail::reportError(*this, "processBulk()", "the new anchor precedes the one before it; pending requests are kept");
        }
    }

    /** nullopt: not due at `streamIndex`; true: due and later than it asked (or it named no time); false: due on time */
    [[nodiscard]] std::optional<bool> lateness(const gr::property_map_view& request, std::size_t streamIndex) const noexcept {
        const auto at = request.template get_if<std::uint64_t>(kInjectionTime);
        if (!at) {
            return true;
        }
        const auto now = _time.at(streamIndex);
        if (!now || *now < *at) {
            return std::nullopt;
        }
        return *at < *now;
    }
};

GR_REGISTER_BLOCK(gr::blocks::trigger::EventToMessage)

struct EventToMessage : gr::Block<EventToMessage, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief send a Set message to `target` for each event the filter names, its payload the event's `fields`

    evtIn   ─E──F──E─▶                 filter = "E", fields = {value: threshold}
    msgOut  ─M─────M─▶                 M = Set target/endpoint {threshold: E's value}

A value is read from the event's trigger_meta_info, else from the event itself; an event carrying none of the keys is skipped.
The scheduler forwards a block's messages to its own clients only, so `target` receives it through whoever routes them.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn evtIn;

    A<std::pmr::string, "filter", Doc<"trigger filter, as EventFilter's; empty: every event">>                         filter;
    A<std::string, "target", Doc<"service name of the receiving block: its unique or its given name">>                 target;
    A<std::string, "endpoint", Doc<"property the message sets; 'StagedSettings' applies at the next sample boundary">> endpoint = std::string(gr::block::property::kStagedSetting);
    A<gr::property_map, "fields", Doc<"event key to message key, e.g. {value: threshold}">>                            fields;
    A<gr::Size_t, "n sent", Doc<"messages sent">>                                                                      n_sent    = 0U;
    A<gr::Size_t, "n skipped", Doc<"accepted events carrying no key of fields">>                                       n_skipped = 0U;

    GR_MAKE_REFLECTABLE(EventToMessage, evtIn, filter, target, endpoint, fields, n_sent, n_skipped);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState _accept{};
    bool       _filtered = false;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) { _filtered = detail::compileOptionalFilter(*this, filter.value, std::string_view{}, _accept); }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (event.empty() || (_filtered && gr::trigger::BasicTriggerNameCtxMatcher::match(_accept, event) == gr::trigger::MatchResult::Ignore)) {
                continue;
            }
            gr::property_map payload;
            const auto       copyFields = [&](const gr::property_map_view& from) {
                for (const auto& [key, value] : from) {
                    if (const auto messageKey = fields.value.get_if<std::string_view>(std::string_view(key))) {
                        std::ignore = payload.insert_or_assign(*messageKey, value);
                    }
                }
            };
            copyFields(event);
            if (const auto meta = event.template get_if<gr::property_map_view>(gr::tag::kTriggerMetaInfo.key())) {
                copyFields(*meta);
            }
            if (payload.empty()) {
                n_skipped = n_skipped + 1U;
                continue;
            }
            gr::emit<gr::message::Command::Set>(this->msgOut, std::move(payload), {.endpoint = endpoint.value, .serviceName = target.value});
            n_sent = n_sent + 1U;
        }
        return evtSpan.consume(evtSpan.size()) ? gr::work::Status::OK : gr::work::Status::ERROR;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TAGBRIDGE_HPP
