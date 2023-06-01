#ifndef THREADAFFINITY_HPP
#define THREADAFFINITY_HPP

#include <algorithm>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <system_error>
#include <thread>
#include <vector>

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__))) // UNIX-style OS
#include <unistd.h>
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
#include <pthread.h>
#include <sched.h>
#endif
#endif

namespace fair::thread_pool::thread {

constexpr size_t THREAD_MAX_NAME_LENGTH  = 16;
constexpr int    THREAD_UNINITIALISED    = 1;
constexpr int    THREAD_ERROR_UNKNOWN    = 2;
constexpr int    THREAD_VALUE_RANGE      = 3;
constexpr int    THREAD_INVALID_ARGUMENT = 22;
constexpr int    THREAD_ERANGE           = 34;

class thread_exception : public std::error_category {
    using std::error_category::error_category;

public:
    constexpr thread_exception()
        : std::error_category(){};

    const char *name() const noexcept override { return "thread_exception"; };
    std::string message(int errorCode) const override {
        switch (errorCode) {
        case THREAD_UNINITIALISED:
            return "thread uninitialised or user does not have the appropriate rights (ie. CAP_SYS_NICE capability)";
        case THREAD_ERROR_UNKNOWN:
            return "thread error code 2";
        case THREAD_INVALID_ARGUMENT:
            return "invalid argument";
        case THREAD_ERANGE:
            return fmt::format("length of the string specified pointed to by name exceeds the allowed limit THREAD_MAX_NAME_LENGTH = '{}'", THREAD_MAX_NAME_LENGTH);
        case THREAD_VALUE_RANGE:
            return fmt::format("priority out of valid range for scheduling policy", THREAD_MAX_NAME_LENGTH);
        default:
            return fmt::format("unknown threading error code {}", errorCode);
        }
    };
};

template<class type>
#ifdef __EMSCRIPTEN__
    concept thread_type = std::is_same_v<type, std::thread>;
#else
    concept thread_type = std::is_same_v<type, std::thread> || std::is_same_v<type, std::jthread>;
#endif

namespace detail {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    template<typename Tp, typename... Us>
    constexpr decltype(auto) firstElement(Tp && t, Us && ...) noexcept {
        return std::forward<Tp>(t);
    }

    inline constexpr pthread_t getPosixHandler(thread_type auto &...t) noexcept {
        if constexpr (sizeof...(t) > 0) {
            return firstElement(t...).native_handle();
        } else {
            return pthread_self();
        }
    }

    inline std::string getThreadName(const pthread_t &handle) {
        if (handle == 0U) {
            return "uninitialised thread";
        }
        char threadName[THREAD_MAX_NAME_LENGTH];
        if (int rc = pthread_getname_np(handle, threadName, THREAD_MAX_NAME_LENGTH); rc != 0) {
            throw std::system_error(rc, thread_exception(), "getThreadName(thread_type)");
        }
        return std::string{ threadName, std::min(strlen(threadName), THREAD_MAX_NAME_LENGTH) };
    }

    inline int getPid() { return getpid(); }
#else
    int getPid() {
        return 0;
    }
#endif
} // namespace detail

inline std::string getProcessName(const int pid = detail::getPid()) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    if (std::ifstream in(fmt::format("/proc/{}/comm", pid), std::ios::in); in.is_open()) {
        std::string fileContent;
        std::getline(in, fileContent, '\n');
        return fileContent;
    }
#endif
    return "unknown_process";
} // namespace detail

inline std::string getThreadName(thread_type auto &...thread) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    const pthread_t handle = detail::getPosixHandler(thread...);
    if (handle == 0U) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), "getThreadName(thread_type)");
    }
    return detail::getThreadName(handle);
#else
    return "unknown thread name";
#endif
}

inline void setProcessName(const std::string_view &processName, int pid = detail::getPid()) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    std::ofstream out(fmt::format("/proc/{}/comm", pid), std::ios::out);
    if (!out.is_open()) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("setProcessName({},{})", processName, pid));
    }
    out << std::string{ processName.cbegin(), std::min(15LU, processName.size()) };
    out.close();
#endif
}

inline void setThreadName(const std::string_view &threadName, thread_type auto &...thread) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    const pthread_t handle = detail::getPosixHandler(thread...);
    if (handle == 0U) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("setThreadName({}, thread_type)", threadName, detail::getThreadName(handle)));
    }
    if (int rc = pthread_setname_np(handle, threadName.data()); rc < 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("setThreadName({},{}) - error code '{}'", threadName, detail::getThreadName(handle), rc));
    }
#endif
}

namespace detail {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
inline std::vector<bool> getAffinityMask(const cpu_set_t &cpuSet) {
    std::vector<bool> bitMask(std::min(sizeof(cpu_set_t), static_cast<size_t>(std::thread::hardware_concurrency())));
    for (size_t i = 0; i < bitMask.size(); i++) {
        bitMask[i] = CPU_ISSET(i, &cpuSet);
    }
    return bitMask;
}

template<class T>
requires requires(T value) { value[0]; }
inline constexpr cpu_set_t getAffinityMask(const T &threadMap) {
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    size_t nMax = std::min(threadMap.size(), static_cast<size_t>(std::thread::hardware_concurrency()));
    for (size_t i = 0; i < nMax; i++) {
        if (threadMap[i]) {
            CPU_SET(i, &cpuSet);
        } else {
            CPU_CLR(i, &cpuSet);
        }
    }
    return cpuSet;
}
#endif
} // namespace detail

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
inline std::vector<bool> getThreadAffinity(thread_type auto &...thread) {
    const pthread_t handle = detail::getPosixHandler(thread...);
    if (handle == 0U) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("getThreadAffinity(thread_type)"));
    }
    cpu_set_t cpuSet;
    if (int rc = pthread_getaffinity_np(handle, sizeof(cpu_set_t), &cpuSet); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("getThreadAffinity({})", detail::getThreadName(handle)));
    }
    return detail::getAffinityMask(cpuSet);
}
#else
std::vector<bool> getThreadAffinity(thread_type auto &...) {
    return std::vector<bool>(std::thread::hardware_concurrency()); // cannot set affinity for non-posix threads
}
#endif

template<class T>
requires requires(T value) { value[0]; }
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
inline constexpr void setThreadAffinity(const T &threadMap, thread_type auto &...thread) {
    const pthread_t handle = detail::getPosixHandler(thread...);
    if (handle == 0U) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("setThreadAffinity(std::vector<bool, {}> = {{{}}}, thread_type)", threadMap.size(), fmt::join(threadMap.begin(), threadMap.end(), ", ")));
    }
    cpu_set_t cpuSet = detail::getAffinityMask(threadMap);
    if (int rc = pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuSet); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("setThreadAffinity(std::vector<bool, {}> = {{{}}}, {})", threadMap.size(), fmt::join(threadMap.begin(), threadMap.end(), ", "), detail::getThreadName(handle)));
    }
}
#else
constexpr bool setThreadAffinity(const T &threadMap, thread_type auto &...) {
    return false; // cannot set affinity for non-posix threads
}
#endif

inline std::vector<bool> getProcessAffinity(const int pid = detail::getPid()) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    if (pid <= 0) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("getProcessAffinity({}) -- invalid pid", pid));
    }
    cpu_set_t cpuSet;
    if (int rc = sched_getaffinity(pid, sizeof(cpu_set_t), &cpuSet); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("getProcessAffinity(std::bitset<{{}}> = {{}}, thread_type)")); // todo: fix format string
    }
    return detail::getAffinityMask(cpuSet);
#else
    return std::vector<bool>(std::thread::hardware_concurrency()); // cannot set affinity for non-posix threads
#endif
}

template<class T>
requires requires(T value) { std::get<0>(value); }
inline constexpr bool setProcessAffinity(const T &threadMap, const int pid = detail::getPid()) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    if (pid <= 0) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("setProcessAffinity(std::vector<bool, {}> = {{{}}}, {})", threadMap.size(), fmt::join(threadMap.begin(), threadMap.end(), ", "), pid));
    }
    cpu_set_t cpuSet = detail::getAffinityMask(threadMap);
    if (int rc = sched_setaffinity(pid, sizeof(cpu_set_t), &cpuSet); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("setProcessAffinity(std::vector<bool, {}> = {{{}}}, {})", threadMap.size(), fmt::join(threadMap.begin(), threadMap.end(), ", "), pid));
    }

    return true;
#else
    return false;                                                  // cannot set affinity for non-posix threads
#endif
}
enum Policy {
    UNKNOWN     = -1,
    OTHER       = 0,
    FIFO        = 1,
    ROUND_ROBIN = 2
};

struct SchedulingParameter {
    Policy policy; // e.g. SCHED_OTHER, SCHED_RR, FSCHED_FIFO
    int    priority;
};

namespace detail {
inline Policy getEnumPolicy(const int policy) {
    switch (policy) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    case SCHED_FIFO: return Policy::FIFO;
    case SCHED_RR: return Policy::ROUND_ROBIN;
    case SCHED_OTHER: return Policy::OTHER;
#endif
    default:
        return Policy::UNKNOWN;
    }
}
} // namespace detail

inline struct SchedulingParameter getProcessSchedulingParameter(const int pid = detail::getPid()) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    if (pid <= 0) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("getProcessSchedulingParameter({}) -- invalid pid", pid));
    }
    struct sched_param param;
    const int          policy = sched_getscheduler(pid);
    if (int rc = sched_getparam(pid, &param); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("getProcessSchedulingParameter({}) - sched_getparam error", pid));
    }
    return SchedulingParameter{ .policy = detail::getEnumPolicy(policy), .priority = param.sched_priority };
#else
    return {};
#endif
}

inline void setProcessSchedulingParameter(Policy scheduler, int priority, const int pid = detail::getPid()) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    if (pid <= 0) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("setProcessSchedulingParameter({}, {}, {}) -- invalid pid", scheduler, priority, pid));
    }
    const int minPriority = sched_get_priority_min(scheduler);
    const int maxPriority = sched_get_priority_max(scheduler);
    if (priority < minPriority || priority > maxPriority) {
        throw std::system_error(THREAD_VALUE_RANGE, thread_exception(), fmt::format("setProcessSchedulingParameter({}, {}, {}) -- requested priority out-of-range [{}, {}]", scheduler, priority, pid, minPriority, maxPriority));
    }
    struct sched_param param {
        .sched_priority = priority
    };
    if (int rc = sched_setscheduler(pid, scheduler, &param); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("setProcessSchedulingParameter({}, {}, {}) - sched_setscheduler return code: {}", scheduler, priority, pid, rc));
    }
#endif
}

inline struct SchedulingParameter getThreadSchedulingParameter(thread_type auto &...thread) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    const pthread_t handle = detail::getPosixHandler(thread...);
    if (handle == 0U) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("getThreadSchedulingParameter(thread_type) -- invalid thread"));
    }
    struct sched_param param;
    int                policy;
    if (int rc = pthread_getschedparam(handle, &policy, &param); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("getThreadSchedulingParameter({}) - sched_getparam error", detail::getThreadName(handle)));
    }
    return { .policy = detail::getEnumPolicy(policy), .priority = param.sched_priority };
#else
    return {};
#endif
}

inline void setThreadSchedulingParameter(Policy scheduler, int priority, thread_type auto &...thread) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__)
    const pthread_t handle = detail::getPosixHandler(thread...);
    if (handle == 0U) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("setThreadSchedulingParameter({}, {}, thread_type) -- invalid thread", scheduler, priority));
    }
    const int minPriority = sched_get_priority_min(scheduler);
    const int maxPriority = sched_get_priority_max(scheduler);
    if (priority < minPriority || priority > maxPriority) {
        throw std::system_error(THREAD_VALUE_RANGE, thread_exception(), fmt::format("setThreadSchedulingParameter({}, {}, {}) -- requested priority out-of-range [{}, {}]", scheduler, priority, detail::getThreadName(handle), minPriority, maxPriority));
    }
    struct sched_param param {
        .sched_priority = priority
    };
    if (int rc = pthread_setschedparam(handle, scheduler, &param); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("setThreadSchedulingParameter({}, {}, {}) - pthread_setschedparam return code: {}", scheduler, priority, detail::getThreadName(handle), rc));
    }
#endif
}

} // namespace opencmw::thread

#endif // THREADAFFINITY_HPP
