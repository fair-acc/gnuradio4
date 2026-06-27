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

    [[nodiscard]] execution::DeviceScheduler& resolve(std::string_view computeDomain) {
        if (computeDomain.empty() || computeDomain.starts_with("default") || computeDomain == "host") {
            return _cpuFallbackSched;
        }

        std::scoped_lock lk(_mtx);
        auto             it = _schedulers.find(computeDomain);
        if (it != _schedulers.end()) {
            return *it->second;
        }

        auto domain = std::string(computeDomain);
        while (!domain.empty()) {
            auto pos = domain.rfind(':');
            if (pos == std::string::npos) {
                it = _schedulers.find(domain);
                if (it != _schedulers.end()) {
                    return *it->second;
                }
                break;
            }
            domain.resize(pos);
            it = _schedulers.find(domain);
            if (it != _schedulers.end()) {
                return *it->second;
            }
        }

        return _cpuFallbackSched;
    }
};

} // namespace gr::device

#endif // GNURADIO_SCHEDULER_REGISTRY_HPP
