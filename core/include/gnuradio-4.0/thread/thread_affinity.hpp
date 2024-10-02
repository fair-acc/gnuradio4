#ifndef THREADAFFINITY_HPP
#define THREADAFFINITY_HPP

#include <algorithm>
#include <fmt/core.h>
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

namespace gr::thread_pool::thread {

constexpr size_t THREAD_MAX_NAME_LENGTH  = 16;
constexpr int    THREAD_UNINITIALISED    = 1;
constexpr int    THREAD_ERROR_UNKNOWN    = 2;
constexpr int    THREAD_VALUE_RANGE      = 3;
constexpr int    THREAD_INVALID_ARGUMENT = 22;
constexpr int    THREAD_ERANGE           = 34;

class thread_exception : public std::error_category {
    using std::error_category::error_category;

public:
    constexpr thread_exception() : std::error_category(){};

    const char* name() const noexcept override { return "thread_exception"; };

    std::string message(int errorCode) const override {
        switch (errorCode) {
        case THREAD_UNINITIALISED: return "thread uninitialised or user does not have the appropriate rights (ie. CAP_SYS_NICE capability)";
        case THREAD_ERROR_UNKNOWN: return "thread error code 2";
        case THREAD_INVALID_ARGUMENT: return "invalid argument";
        case THREAD_ERANGE: return fmt::format("length of the string specified pointed to by name exceeds the allowed limit THREAD_MAX_NAME_LENGTH = '{}'", THREAD_MAX_NAME_LENGTH);
        case THREAD_VALUE_RANGE: return fmt::format("priority out of valid range for scheduling policy", THREAD_MAX_NAME_LENGTH);
        default: return fmt::format("unknown threading error code {}", errorCode);
        }
    };
};

template<class type>
#if __cpp_lib_jthread >= 201911L
concept thread_type = std::is_same_v<type, std::thread> || std::is_same_v<type, std::jthread>;
#else
concept thread_type = std::is_same_v<type, std::thread>;
#endif

namespace detail {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
template<typename Tp, typename... Us>
constexpr decltype(auto) firstElement(Tp&& t, Us&&...) noexcept {
    return std::forward<Tp>(t);
}

inline constexpr pthread_t getPosixHandler(thread_type auto&... t) noexcept {
    if constexpr (sizeof...(t) > 0) {
        return firstElement(t...).native_handle();
    } else {
        return pthread_self();
    }
}

inline std::string getThreadName(const pthread_t& handle) {
    if (handle == 0U) {
        return "uninitialised thread";
    }
    char threadName[THREAD_MAX_NAME_LENGTH];
    if (int rc = pthread_getname_np(handle, threadName, THREAD_MAX_NAME_LENGTH); rc != 0) {
        throw std::system_error(rc, thread_exception(), "getThreadName(thread_type)");
    }
    return std::string{threadName, strnlen(threadName, THREAD_MAX_NAME_LENGTH)};
}

inline int getPid() { return getpid(); }
#else
inline int getPid() { return 0; }
#endif
} // namespace detail

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline std::string getProcessName(const int pid = detail::getPid()) {
    if (std::ifstream in(fmt::format("/proc/{}/comm", pid), std::ios::in); in.is_open()) {
        std::string fileContent;
        std::getline(in, fileContent, '\n');
        return fileContent;
    }
    return "unknown_process";
}
#else
inline std::string getProcessName(const int /*pid*/ = -1) { return "unknown_process"; }
#endif

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline std::string getThreadName(thread_type auto&... thread) {
    const pthread_t handle = detail::getPosixHandler(thread...);
    if (handle == 0U) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), "getThreadName(thread_type)");
    }
    return detail::getThreadName(handle);
}
#else
inline std::string getThreadName(thread_type auto&... /*thread*/) { return "unknown thread name"; }
#endif

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline void setProcessName(const std::string_view& processName, int pid = detail::getPid()) {
    std::ofstream out(fmt::format("/proc/{}/comm", pid), std::ios::out);
    if (!out.is_open()) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("setProcessName({},{})", processName, pid));
    }
    out << std::string{processName.cbegin(), std::min(15LU, processName.size())};
    out.close();
}
#else
inline void setProcessName(const std::string_view& /*processName*/, int /*pid*/ = -1) {}
#endif

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline void setThreadName(const std::string_view& threadName, thread_type auto&... thread) {
    const pthread_t handle = detail::getPosixHandler(thread...);
    if (handle == 0U) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("setThreadName({}, thread_type)", threadName, detail::getThreadName(handle)));
    }
    if (int rc = pthread_setname_np(handle, threadName.data()); rc < 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("setThreadName({},{}) - error code '{}'", threadName, detail::getThreadName(handle), rc));
    }
}
#else
inline void setThreadName(const std::string_view& /*threadName*/, thread_type auto&... /*thread*/) {}
#endif

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
namespace detail {
inline std::vector<bool> getAffinityMask(const cpu_set_t& cpuSet) {
    std::vector<bool> bitMask(std::min(sizeof(cpu_set_t), static_cast<size_t>(std::thread::hardware_concurrency())));
    for (size_t i = 0; i < bitMask.size(); i++) {
        bitMask[i] = CPU_ISSET(i, &cpuSet);
    }
    return bitMask;
}

template<class T>
requires requires(T value) { value[0]; }
inline constexpr cpu_set_t getAffinityMask(const T& threadMap) {
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
} // namespace detail
#endif

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline std::vector<bool> getThreadAffinity(thread_type auto&... thread) {
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
std::vector<bool> getThreadAffinity(thread_type auto&...) {
    return std::vector<bool>(std::thread::hardware_concurrency()); // cannot set affinity for non-posix threads
}
#endif

template<class T>
requires requires(T value) { value[0]; }
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline constexpr void setThreadAffinity(const T& threadMap, thread_type auto&... thread) {
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
constexpr bool setThreadAffinity(const T& /*threadMap*/, thread_type auto&...) {
    return false; // cannot set affinity for non-posix threads
}
#endif

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline std::vector<bool> getProcessAffinity(const int pid = detail::getPid()) {
    if (pid <= 0) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("getProcessAffinity({}) -- invalid pid", pid));
    }
    cpu_set_t cpuSet;
    if (int rc = sched_getaffinity(pid, sizeof(cpu_set_t), &cpuSet); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("getProcessAffinity(std::bitset<{{}}> = {{}}, thread_type)")); // todo: fix format string
    }
    return detail::getAffinityMask(cpuSet);
}
#else
inline std::vector<bool> getProcessAffinity(const int /*pid*/ = -1) {
    return std::vector<bool>(std::thread::hardware_concurrency()); // cannot set affinity for non-posix threads
}
#endif

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
template<class T>
requires requires(T value) { std::get<0>(value); }
inline constexpr bool setProcessAffinity(const T& threadMap, const int pid = detail::getPid()) {
    if (pid <= 0) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("setProcessAffinity(std::vector<bool, {}> = {{{}}}, {})", threadMap.size(), fmt::join(threadMap.begin(), threadMap.end(), ", "), pid));
    }
    cpu_set_t cpuSet = detail::getAffinityMask(threadMap);
    if (int rc = sched_setaffinity(pid, sizeof(cpu_set_t), &cpuSet); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("setProcessAffinity(std::vector<bool, {}> = {{{}}}, {})", threadMap.size(), fmt::join(threadMap.begin(), threadMap.end(), ", "), pid));
    }

    return true;
}
#else
template<class T>
requires requires(T value) { std::get<0>(value); }
inline constexpr bool setProcessAffinity(const T& /*threadMap*/, const int /*pid*/ = -1) {
    return false; // cannot set affinity for non-posix threads
}
#endif
enum Policy { UNKNOWN = -1, OTHER = 0, FIFO = 1, ROUND_ROBIN = 2 };
} // namespace gr::thread_pool::thread

template<>
struct fmt::formatter<gr::thread_pool::thread::Policy> {
    using Policy = gr::thread_pool::thread::Policy;

    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(Policy policy, FormatContext& ctx) const {
        std::string policy_name;
        switch (policy) {
        case Policy::UNKNOWN: policy_name = "UNKNOWN"; break;
        case Policy::OTHER: policy_name = "OTHER"; break;
        case Policy::FIFO: policy_name = "FIFO"; break;
        case Policy::ROUND_ROBIN: policy_name = "ROUND_ROBIN"; break;
        default: policy_name = "INVALID_POLICY"; break;
        }
        return fmt::format_to(ctx.out(), "{}", policy_name);
    }
};

namespace gr::thread_pool::thread {

struct SchedulingParameter {
    Policy policy; // e.g. SCHED_OTHER, SCHED_RR, FSCHED_FIFO
    int    priority;
};

namespace detail {
inline Policy getEnumPolicy(const int policy) {
    switch (policy) {
#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
    case SCHED_FIFO: return Policy::FIFO;
    case SCHED_RR: return Policy::ROUND_ROBIN;
    case SCHED_OTHER: return Policy::OTHER;
#endif
    default: return Policy::UNKNOWN;
    }
}
} // namespace detail

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline struct SchedulingParameter getProcessSchedulingParameter(const int pid = detail::getPid()) {
    if (pid <= 0) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("getProcessSchedulingParameter({}) -- invalid pid", pid));
    }
    struct sched_param param;
    const int          policy = sched_getscheduler(pid);
    if (int rc = sched_getparam(pid, &param); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("getProcessSchedulingParameter({}) - sched_getparam error", pid));
    }
    return SchedulingParameter{.policy = detail::getEnumPolicy(policy), .priority = param.sched_priority};
}
#else
inline struct SchedulingParameter getProcessSchedulingParameter(const int /*pid*/ = -1) { return {}; }
#endif

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline void setProcessSchedulingParameter(Policy scheduler, int priority, const int pid = detail::getPid()) {
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
}
#else
inline void setProcessSchedulingParameter(Policy /*scheduler*/, int /*priority*/, const int /*pid*/ = -1) {}
#endif

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline struct SchedulingParameter getThreadSchedulingParameter(thread_type auto&... thread) {
    const pthread_t handle = detail::getPosixHandler(thread...);
    if (handle == 0U) {
        throw std::system_error(THREAD_UNINITIALISED, thread_exception(), fmt::format("getThreadSchedulingParameter(thread_type) -- invalid thread"));
    }
    struct sched_param param;
    int                policy;
    if (int rc = pthread_getschedparam(handle, &policy, &param); rc != 0) {
        throw std::system_error(rc, thread_exception(), fmt::format("getThreadSchedulingParameter({}) - sched_getparam error", detail::getThreadName(handle)));
    }
    return {.policy = detail::getEnumPolicy(policy), .priority = param.sched_priority};
}
#else
inline struct SchedulingParameter getThreadSchedulingParameter(thread_type auto&... /*thread*/) { return {}; }
#endif

#if defined(_POSIX_VERSION) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
inline void setThreadSchedulingParameter(Policy scheduler, int priority, thread_type auto&... thread) {
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
}
#else
inline void setThreadSchedulingParameter(Policy /*scheduler*/, int /*priority*/, thread_type auto&... /*thread*/) {}
#endif

} // namespace gr::thread_pool::thread

#endif // THREADAFFINITY_HPP
