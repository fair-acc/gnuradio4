#ifndef GNURADIO_MEMORYMONITOR_HPP
#define GNURADIO_MEMORYMONITOR_HPP

#include <filesystem>
#include <stdexcept>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#elif defined(__linux__)
#include <fstream>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif
#if defined(__EMSCRIPTEN__)
#include <emscripten/heap.h>
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

Resident and virtual (total program) size are provided; the rest of `/proc/self/statm` or `/proc/self/stat` could be added the same way.
*/
struct Stat {
    std::size_t residentSize = 0UZ; // Resident Set Size: number of bytes the process has in real memory; 0 where the platform does not tell
    std::size_t virtualSize  = 0UZ; // bytes: VmSize on Linux, virtual_size on macOS, the commit charge on Windows, the wasm memory under Emscripten
};

inline Stat getUsage() {
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return {};
    }
    return {.residentSize = pmc.WorkingSetSize, .virtualSize = pmc.PagefileUsage};

#elif defined(__EMSCRIPTEN__)
    return {.residentSize = 0UZ, .virtualSize = emscripten_get_heap_size()};

#elif defined(__linux__)
    if (!std::filesystem::exists("/proc/self/statm")) {
        return {};
    }
    std::ifstream stat("/proc/self/statm");
    if (!stat.is_open()) {
        return {};
    }
    std::size_t sizePages     = 0UZ;
    std::size_t residentPages = 0UZ;
    stat >> sizePages >> residentPages;
    const auto pageSize = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    return {.residentSize = residentPages * pageSize, .virtualSize = sizePages * pageSize};

#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t      infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) != KERN_SUCCESS) {
        return {};
    }
    return {.residentSize = info.resident_size, .virtualSize = info.virtual_size};

#else
    return {};
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
