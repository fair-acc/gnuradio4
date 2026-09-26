#ifndef GNURADIO_TRIGGER_TAGBRIDGE_HPP
#define GNURADIO_TRIGGER_TAGBRIDGE_HPP

#include <cstdint>
#include <deque>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <memory_resource>
#include <optional>
#include <string>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::TagToMessage, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct TagToMessage : gr::Block<TagToMessage<T>> {
    using Description = Doc<R"(@brief turn the tags a filter accepts into events, leaving the stream untouched

    in      ─a──T:b──c──U:d─▶        (no RxMarbles equivalent: a change of carrier, not of content)
    out     ─a──T:b──c──U:d─▶
    evtOut  ─────E──────────▶        filter = the tag's name

The bridge out of the stream: a condition found in the samples becomes an event several blocks can read, without the stream
itself changing.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "filter", Doc<"trigger filter, e.g. 'start/ctx' or 'start^/stop'">>   filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                           match_mode;
    A<std::pmr::string, "trigger name", Doc<"name carried by the event, default: the tag's">> trigger_name;
    A<bool, "include tag", Doc<"shallow-merge the original tag fields into the event">>       include_tag        = false;
    A<gr::Size_t, "max pending events", Doc<"events held while evtOut is full">>              max_pending_events = 64U;
    A<gr::Size_t, "n events dropped", Doc<"events the queue lost, oldest first">>             n_events_dropped   = 0U;
    A<gr::Size_t, "n invalid triggers", Doc<"undated triggers">>                              n_invalid_triggers = 0U;

    GR_MAKE_REFLECTABLE(TagToMessage, evtOut, in, out, filter, match_mode, trigger_name, include_tag, max_pending_events, n_events_dropped, n_invalid_triggers);

    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _matcher{};
    std::size_t                                         _streamIndex = 0UZ;
    detail::PendingEvents                               _pending;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        detail::compileFilterInto(filter.value, match_mode.value, _matcher, "TagToMessage");
        _pending.capacity = static_cast<std::size_t>(max_pending_events);
        _pending.clear();
        n_events_dropped   = 0U;
        n_invalid_triggers = 0U;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(inSpan.size(), outSpan.size());

        for (const auto& tag : inSpan.rawTags()) {
            const std::size_t offset = tag.index - inSpan.streamIndex;
            if (offset >= n) {
                continue;
            }
            const auto result = gr::trigger::BasicTriggerNameCtxMatcher::match(_matcher, tag.map);
            if (result == gr::trigger::MatchResult::Ignore) {
                continue;
            }
            if (detail::isIncompleteTrigger(tag.map)) {
                n_invalid_triggers = n_invalid_triggers + 1U;
                gr::log::warning("TagToMessage: a trigger lacking '{}' or '{}' was reported", gr::tag::TRIGGER_TIME.key(), gr::tag::TRIGGER_OFFSET.key());
                _pending.push(detail::makeErrorEvent(std::format("a trigger lacking '{}' or '{}' was reported", gr::tag::TRIGGER_TIME.key(), gr::tag::TRIGGER_OFFSET.key()), this->unique_name.value()));
            }
            _pending.push(makeEvent(tag.map, result));
        }
        const std::size_t published = _pending.drainInto(evtSpan, 0UZ, this->name, this->unique_name);
        n_events_dropped            = static_cast<gr::Size_t>(_pending.dropped);

        std::ranges::copy(inSpan | std::views::take(n), outSpan.begin());
        _streamIndex += n;
        outSpan.publish(n);
        evtSpan.publish(published);
        if (!inSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] gr::property_map makeEvent(const gr::property_map_view& tagMap, gr::trigger::MatchResult result) const {
        gr::property_map event;
        if (include_tag) {
            for (const auto& key : tagMap.keys()) {
                if (auto value = tagMap.find_value(key)) {
                    event[std::string(key)] = gr::pmt::Value(value.value());
                }
            }
        }

        std::string name(trigger_name.value);
        if (name.empty()) {
            if (const auto fromTag = tagMap.template get_if<std::string_view>(gr::tag::TRIGGER_NAME.key())) {
                name = std::pmr::string(*fromTag);
            }
        }
        event[std::string(gr::tag::TRIGGER_NAME.key())] = std::move(name);
        event[std::string("source")]                    = std::pmr::string(this->unique_name.value());
        event[std::string("state")]                     = std::pmr::string(result == gr::trigger::MatchResult::Matching ? "active" : "inactive");
        for (const auto& key : {gr::tag::TRIGGER_TIME.key(), gr::tag::TRIGGER_OFFSET.key(), gr::tag::TRIGGER_TIME_ERROR.key()}) {
            if (const auto value = tagMap.find_value(key)) {
                event[std::string(key)] = gr::pmt::Value(value.value());
            }
        }
        return event;
    }
};

inline constexpr std::string_view kInjectionTag  = "tag";
inline constexpr std::string_view kInjectionTime = "at";
inline constexpr std::string_view kInjectionLate = "late";

GR_REGISTER_BLOCK(gr::blocks::trigger::MessageToTag, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct MessageToTag : gr::Block<MessageToTag<T>> {
    using Description = Doc<R"(@brief publish a stream tag for each event that asks for one, in arrival order

    evtIn   ─E────E──▶               (no RxMarbles equivalent)
    in      ─a─a─a─a─a─▶
    out     ─T:a─a─T:a─a─▶           placed at the sample the event's time names

The bridge into the stream, and the inverse of `TagToMessage`: a decision made elsewhere becomes in-band, sample-exact where the
event carries a time the stream can be dated against.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn evtIn;
    gr::PortIn<T>   in;
    gr::PortOut<T>  out;

    A<float, "sample rate", Doc<"Hz, used to advance stream time between anchors">>         sample_rate = 1.0f;
    A<gr::Size_t, "queue depth", Doc<"pending requests held before the newest is refused">> queue_depth = 64U;
    A<gr::Size_t, "refused", Doc<"requests dropped because the queue was full">>            n_refused   = 0U;
    A<gr::Size_t, "late", Doc<"requests published after the time they asked for">>          n_late      = 0U;

    GR_MAKE_REFLECTABLE(MessageToTag, evtIn, in, out, sample_rate, queue_depth, n_refused, n_late);

    struct Request {
        gr::property_map             tag;
        std::optional<std::uint64_t> at;
        bool                         overdue = false;
    };

    std::deque<Request> _pending;
    TimeBase            _time;
    std::size_t         _streamIndex = 0UZ;

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        acceptRequests(evtSpan);

        const std::size_t n = std::min(inSpan.size(), outSpan.size());
        for (const auto& tag : inSpan.rawTags()) {
            adoptAnchor(tag.map, tag.index - inSpan.streamIndex);
        }

        for (std::size_t i = 0UZ; i < n; ++i) {
            gr::property_map merged;
            while (!_pending.empty() && dueAt(_pending.front(), _streamIndex + i)) {
                Request request = std::move(_pending.front());
                _pending.pop_front();
                if (request.overdue) {
                    merged[std::string(kInjectionLate)] = true;
                    ++n_late;
                }
                for (const auto& key : request.tag.keys()) { // later request wins on a shared key
                    if (auto value = request.tag.find_value(key)) {
                        merged.insert_or_assign(key, gr::pmt::Value(value.value()));
                    }
                }
            }
            if (!merged.empty()) {
                outSpan.publishTag(merged, i);
            }
        }

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
            if (_pending.size() >= static_cast<std::size_t>(queue_depth)) {
                ++n_refused;
                gr::log::warning("MessageToTag: {} pending requests already, refusing the newest", _pending.size());
                continue;
            }
            Request request;
            if (const auto tag = event.template get_if<gr::property_map>(kInjectionTag)) {
                request.tag = *tag;
            }
            if (const auto at = event.template get_if<std::uint64_t>(kInjectionTime)) {
                request.at = *at;
            }
            if (request.tag.empty()) {
                gr::log::warning("MessageToTag: an event carried no '{}' map", kInjectionTag);
                continue;
            }
            _pending.push_back(std::move(request));
        }
    }

    void adoptAnchor(const gr::property_map_view& tagMap, std::size_t offset) {
        _time.setRate(static_cast<double>(sample_rate));
        if (_time.adopt(tagMap, _streamIndex + offset) == AnchorChange::backward) {
            gr::log::warning("MessageToTag: the new anchor precedes the one before it; pending requests are kept");
        }
    }

    [[nodiscard]] std::optional<std::uint64_t> timeAt(std::size_t streamIndex) const noexcept { return _time.at(streamIndex); }

    [[nodiscard]] bool dueAt(Request& request, std::size_t streamIndex) noexcept {
        if (!request.at) {
            request.overdue = true;
            return true;
        }
        const auto now = timeAt(streamIndex);
        if (!now) {
            return false;
        }
        if (*now < *request.at) {
            return false;
        }
        request.overdue = request.overdue || *request.at < *now;
        return true;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TAGBRIDGE_HPP
