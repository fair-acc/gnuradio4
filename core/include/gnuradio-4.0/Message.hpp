#ifndef GNURADIO_MESSAGE_HPP
#define GNURADIO_MESSAGE_HPP

#include "gnuradio-4.0/Tag.hpp"
#include <string_view>

#include <pmtv/pmt.hpp>

namespace gr {

namespace message::key {
const std::string Sender    = "SENDER_KEY";
const std::string Target    = "TARGET_KEY";
const std::string Kind      = "KIND_KEY";
const std::string What      = "WHAT_KEY";
const std::string Data      = "DATA_KEY";
const std::string Location  = "LOCATION_KEY";
const std::string ErrorInfo = "ERROR_INFO_KEY"; // optional: if a message has an additional error information
} // namespace message::key

namespace message::kind {
const std::string Error                       = "ERROR_KIND";
const std::string Graph_update                = "GRAPH_UPDATE_KIND";
const std::string UpdateSettings              = "UPDATE_SETTINGS_KIND";
const std::string SettingsChanged             = "SETTINGS_CHANGED_KIND";
const std::string SettingsChangeRequested     = "REQUESTED_SETTINGS_CHANGE_KIND";
const std::string SchedulerStateUpdate        = "SCHEDULER_UPDATE_KIND";
const std::string SchedulerStateChangeRequest = "SCHEDULER_COMMAND_KIND";
} // namespace message::kind

using Message = property_map;

template<typename T>
std::optional<T>
messageField(const Message &message, const std::string &key) {
    auto it = message.find(key);
    if (it == message.end()) {
        return {};
    }

    return std::get<T>(it->second);
}

} // namespace gr

#endif // include guard
