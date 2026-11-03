#ifndef GNURADIO_WORK_STATUS_HPP
#define GNURADIO_WORK_STATUS_HPP

#include <array>
#include <cstddef>
#include <limits>
#include <string_view>
#include <utility>

namespace gr::work {

enum class Status {
    ERROR                     = -100, /// error occurred in the work function
    INSUFFICIENT_OUTPUT_ITEMS = -3,   /// work requires a larger output buffer to produce output
    INSUFFICIENT_INPUT_ITEMS  = -2,   /// work requires a larger input buffer to produce output
    DONE                      = -1,   /// this block has completed its processing and the flowgraph should be done
    OK                        = 0,    /// work call was successful and return values in i/o structs are valid
};

struct Result {
    std::size_t requested_work = std::numeric_limits<std::size_t>::max();
    std::size_t performed_work = 0;
    Status      status         = Status::OK;
};

/// if the block reported an error or insufficient buffers, zero the counts
inline void sanitiseProcessStatus(Status status, std::size_t& processedIn, std::size_t& processedOut) noexcept {
    if (status == Status::INSUFFICIENT_OUTPUT_ITEMS || status == Status::INSUFFICIENT_INPUT_ITEMS || status == Status::ERROR) {
        processedIn  = 0UZ;
        processedOut = 0UZ;
    }
}

/// compute how much work was performed — used by the scheduler for block prioritisation
inline std::size_t computePerformedWork(Status status, std::size_t processedIn, std::size_t processedOut, bool isSource) noexcept {
    if (status != Status::OK) {
        return 0UZ;
    }
    return isSource ? processedOut : processedIn;
}

} // namespace gr::work

// Compile-time performance override; phased out with C++26 reflection. Co-located with the enum (not in
// reflection.hpp) so any TU that formats work::Status sees the specialisation, never the primary — and so
// this MCU-path header stays reflection-free. The primary is forward-declared rather than included.
namespace gr::meta::detail {
template<typename E>
struct EnumTraits;

template<>
struct EnumTraits<gr::work::Status> {
    static constexpr std::array<std::pair<gr::work::Status, std::string_view>, 5> entries = {{
        {gr::work::Status::ERROR, "ERROR"},
        {gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS, "INSUFFICIENT_OUTPUT_ITEMS"},
        {gr::work::Status::INSUFFICIENT_INPUT_ITEMS, "INSUFFICIENT_INPUT_ITEMS"},
        {gr::work::Status::DONE, "DONE"},
        {gr::work::Status::OK, "OK"},
    }};
};
} // namespace gr::meta::detail

#endif // GNURADIO_WORK_STATUS_HPP
