#ifndef GNURADIO_MEMORYMONITOR_HPP
#define GNURADIO_MEMORYMONITOR_HPP

#include <filesystem>
#include <stdexcept>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <psapi.h>
#include <windows.h>
#elif defined(__linux__)
#include <fstream>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace gr {
namespace memory {

/*
For Linux `/proc/pid/statm` is used. It provides information about memory usage, measured in pages.
The following information is available:
size       (1) total program size (same as VmSize in /proc/pid/status)
resident   (2) resident set size
shared     (3) number of resident shared pages
text       (4) text (code)
lib        (5) library (unused since Linux 2.6; always 0)
data       (6) data + stack
dt         (7) dirty pages (unused since Linux 2.6; always 0)

For the moment only resident memory is provided. But it can be extended further with the information available from `proc/self/statm` or `/proc/self/stat`.
*/
struct Stat {
    std::size_t residentSize; // Resident Set Size: number of bytes the process has in real memory
};

inline Stat getUsage() {
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return {.residentSize = 0UZ};
    }
    return {.residentSize = pmc.WorkingSetSize};

#elif defined(__EMSCRIPTEN__)
    return {.residentSize = 0UZ};

#elif defined(__linux__)
    if (!std::filesystem::exists("/proc/self/statm")) {
        return {.residentSize = 0UZ};
    }
    std::ifstream stat("/proc/self/statm");
    if (!stat.is_open()) {
        return {.residentSize = 0UZ};
    }
    std::string ignore;
    std::size_t rss;
    stat >> ignore >> rss;
    rss = rss * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));

    return {.residentSize = rss};

#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t      infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) != KERN_SUCCESS) {
        return {.residentSize = 0UZ};
    }
    return {.residentSize = info.resident_size};

#else
    return {.residentSize = 0UZ};
#endif
}

inline std::string getPlatformName() {
#if defined(_WIN32) || defined(_WIN64)
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "";
#endif
}

} // namespace memory

} // namespace gr

#endif // GNURADIO_MEMORYMONITOR_HPP
