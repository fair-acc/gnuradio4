#ifndef GNURADIO_TRIGGER_EVENTREFERENCE_HPP
#define GNURADIO_TRIGGER_EVENTREFERENCE_HPP

#include <cstdint>
#include <string_view>

#include <gnuradio-4.0/Tag.hpp>

namespace gr::blocks::trigger {

enum class EventReference : std::uint8_t { decision, start };

/** the time an operator dates an event by: its `trigger_time`, or with `start` the start tuple in its `trigger_meta_info`, the decision time when it carries none */
[[nodiscard]] inline std::uint64_t referenceTimeOf(const gr::property_map_view& event, std::uint64_t decisionAt, EventReference reference) noexcept {
    if (reference != EventReference::start) {
        return decisionAt;
    }
    const auto details = event.template get_if<gr::property_map_view>(gr::tag::kTriggerMetaInfo.key());
    const auto started = details ? details->template get_if<std::uint64_t>(gr::tag::kTriggerTime.key()) : nullptr;
    return started ? *started : decisionAt;
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_EVENTREFERENCE_HPP
