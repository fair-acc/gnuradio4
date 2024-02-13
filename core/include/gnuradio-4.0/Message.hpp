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
const std::string Error           = "ERROR_KIND";
const std::string Graph_update    = "GRAPH_UPDATE_KIND";
const std::string UpdateSettings  = "UPDATE_SETTINGS_KIND";
const std::string SettingsChanged = "SETTINGS_CHANGED_KIND";
const std::string SchedulerUpdate = "SCHEDULER_UPDATE_KIND";
} // namespace message::kind

namespace message::scheduler::command {
const std::string Start  = "gr::scheduler::command::START";
const std::string Stop   = "gr::scheduler::command::STOP";
const std::string Pause  = "gr::scheduler::command::PAUSE";
const std::string Resume = "gr::scheduler::command::RESUME";
} // namespace message::scheduler::command

namespace message::scheduler::update {
const std::string Started  = "gr::scheduler::update::STARTED";
const std::string Stopping = "gr::scheduler::update::STOPPING";
const std::string Stopped  = "gr::scheduler::update::STOPPED";
const std::string Pausing  = "gr::scheduler::update::PAUSING";
const std::string Paused   = "gr::scheduler::update::PAUSED";
} // namespace message::scheduler::update

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
