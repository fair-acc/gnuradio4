#ifndef GNURADIO_SCHEDULER_REGISTRY_HPP
#define GNURADIO_SCHEDULER_REGISTRY_HPP

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/execution/gpu_scheduler.hpp>

namespace gr::device {

/**
 * @brief Singleton registry mapping `compute_domain` strings to DeviceScheduler instances.
 *
 * Resolves domain strings like `"gpu:sycl:0"` to the corresponding DeviceScheduler by exact
 * match, then progressively shorter prefixes (`"gpu:sycl"`, `"gpu"`). Falls back to a
 * CPU-only scheduler when no match is found.
 *
 * Usage:
 * @code
 * auto ctx = std::make_unique<gr::device::DeviceContext>(syclQueue);
 * gr::device::SchedulerRegistry::instance().registerContext("gpu:sycl", std::move(ctx));
 * auto& sched = gr::device::SchedulerRegistry::instance().resolve("gpu:sycl:0");
 * @endcode
 */
class SchedulerRegistry {
    mutable std::mutex _mtx;

    struct Hash {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
    };
    struct Eq {
        using is_transparent = void;
        bool operator()(std::string_view a, std::string_view b) const noexcept { return a == b; }
    };

    std::unordered_map<std::string, std::unique_ptr<DeviceContext>, Hash, Eq>              _contexts;
    std::unordered_map<std::string, std::unique_ptr<execution::DeviceScheduler>, Hash, Eq> _schedulers;

    DeviceContextCpu           _cpuFallbackCtx;
    execution::DeviceScheduler _cpuFallbackSched{_cpuFallbackCtx};

    // exact match, then progressively shorter `:`-separated prefixes ("gpu:sycl:0" -> "gpu:sycl" -> "gpu")
    [[nodiscard]] execution::DeviceScheduler* lookupLocked(std::string_view computeDomain) {
        if (auto it = _schedulers.find(computeDomain); it != _schedulers.end()) {
            return it->second.get();
        }

        auto domain = std::string(computeDomain);
        while (!domain.empty()) {
            const auto pos = domain.rfind(':');
            if (pos == std::string::npos) {
                const auto it = _schedulers.find(domain);
                return it != _schedulers.end() ? it->second.get() : nullptr;
            }
            domain.resize(pos);
            if (const auto it = _schedulers.find(domain); it != _schedulers.end()) {
                return it->second.get();
            }
        }
        return nullptr;
    }

public:
    static SchedulerRegistry& instance() {
        static SchedulerRegistry r;
        return r;
    }

    void registerContext(std::string_view name, std::unique_ptr<DeviceContext> ctx) {
        std::scoped_lock lk(_mtx);
        auto             key   = std::string(name);
        auto             sched = std::make_unique<execution::DeviceScheduler>(*ctx);
        _contexts[key]         = std::move(ctx);
        _schedulers[key]       = std::move(sched);
    }

    /// nullptr when no context is registered for the domain — absence is explicit, never a silent CPU fallback
    [[nodiscard]] execution::DeviceScheduler* tryResolve(std::string_view computeDomain) {
        std::scoped_lock lk(_mtx);
        return lookupLocked(computeDomain);
    }

    [[nodiscard]] execution::DeviceScheduler& resolve(std::string_view computeDomain) {
        if (computeDomain.empty() || computeDomain.starts_with("default") || computeDomain == "host") {
            return _cpuFallbackSched;
        }

        std::scoped_lock            lk(_mtx);
        execution::DeviceScheduler* found = lookupLocked(computeDomain);
        return found != nullptr ? *found : _cpuFallbackSched;
    }
};

} // namespace gr::device

#endif // GNURADIO_SCHEDULER_REGISTRY_HPP
