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
const std::string Type      = "TYPE_KEY";
const std::string What      = "WHAT_KEY";
const std::string Data      = "DATA_KEY";
const std::string Location  = "LOCATION_KEY";
const std::string ErrorInfo = "ERROR_INFO_KEY"; // optional: if a message has an additional error information
} // namespace message::key

namespace message::kind {
const std::string Error           = "ERROR_KIND";
const std::string GraphUpdate     = "GRAPH_UPDATE_KIND";
const std::string SchedulerUpdate = "SCHEDULER_UPDATE_KIND";
const std::string UpdateSettings  = "UPDATE_SETTINGS_KIND";
const std::string SettingsChanged = "SETTINGS_CHANGED_KIND";
} // namespace message::kind

namespace message::graph_update {
namespace what {
const std::string BlockAdded         = "ADDED_BLOCK_GRAPH_UPDATE";
const std::string BlockRemoved       = "REMOVED_BLOCK_GRAPH_UPDATE";
const std::string BlockExchanged     = "EXCHANGED_BLOCK_GRAPH_UPDATE";
const std::string BlocksConnected    = "CONNECTED_BLOCKS_GRAPH_UPDATE";
const std::string BlocksDisconnected = "DISCONNECTED_BLOCKS_GRAPH_UPDATE";
} // namespace what

namespace data {
const std::string AddedBlockId   = "ADDED_BLOCK_ID";  // for BlockAdded, BlockExchanged
const std::string RemovedBlockId = "REMOVE_BLOCK_ID"; // for BlockRemoved, BlockExchanged

const std::string SourceBlock      = "SOURCE_BLOCK";
const std::string SourcePort       = "SOURCE_PORT";
const std::string DestinationBlock = "DESTINATION_BLOCK";
const std::string DestinationPort  = "DESTINATION_PORT";
} // namespace data
} // namespace message::graph_update

namespace message::scheduler::command {
const std::string StartScheduler  = "START_SCHEDULER";
const std::string StopScheduler   = "STOP_SCHEDULER";
const std::string PauseScheduler  = "PAUSE_SCHEDULER";
const std::string ResumeScheduler = "RESUME_SCHEDULER";
} // namespace message::scheduler::command

namespace message::scheduler::update {
const std::string SchedulerStarted  = "STARTED_SCHEDULER_UPDATE";
const std::string SchedulerPaused   = "PAUSED_SCHEDULER_UPDATE";
const std::string SchedulerPausing  = "PAUSING_SCHEDULER_UPDATE";
const std::string SchedulerStopped  = "STOPPED_SCHEDULER_UPDATE";
const std::string SchedulerStopping = "STOPPING_SCHEDULER_UPDATE";
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
