#ifndef GNURADIO_TRIGGER_TAGBRIDGE_HPP
#define GNURADIO_TRIGGER_TAGBRIDGE_HPP

#include <cstdint>
#include <deque>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <optional>
#include <string>

namespace gr::blocks::trigger {

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

The bridge into the stream, and the inverse of `TagToEvent`: a decision made elsewhere becomes in-band, sample-exact where the
event carries a time the stream can be dated against.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn evtIn;
    gr::PortIn<T>   in;
    gr::PortOut<T>  out;

    gr::SampleRate                                                                          sample_rate = 1.0f;
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
                for (const auto& key : request.tag.keys()) {
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
