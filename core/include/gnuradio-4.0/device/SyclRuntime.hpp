#ifndef GNURADIO_DEVICE_SYCL_RUNTIME_HPP
#define GNURADIO_DEVICE_SYCL_RUNTIME_HPP

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/SchedulerRegistry.hpp>
#include <gnuradio-4.0/device/UsmMemoryResource.hpp>

namespace gr::device {

#if GR_DEVICE_HAS_SYCL_IMPL
namespace detail {

inline sycl::queue& defaultSyclQueue() {
    static sycl::queue queue{sycl::property::queue::in_order{}};
    return queue;
}

inline UsmMemoryResource& defaultSyclUsmResource() {
    static UsmMemoryResource resource{defaultSyclQueue()};
    return resource;
}

inline std::pmr::memory_resource* defaultSyclUsmProvider(const ComputeDomain&, void* ctx) {
    if (ctx != nullptr) {
        return static_cast<std::pmr::memory_resource*>(ctx);
    }
    return &defaultSyclUsmResource();
}

// one owning queue per enumerated device; `DeviceContextSycl` only borrows it
inline std::vector<std::unique_ptr<sycl::queue>>& enumeratedSyclQueues() {
    static std::vector<std::unique_ptr<sycl::queue>> queues;
    return queues;
}

} // namespace detail
#endif

/**
 * @brief discover the SYCL devices and publish one `DeviceContext` per device.
 *
 * Registers the canonical domains `cpu:sycl` / `gpu:sycl` (first device of each kind) and the indexed
 * domains `cpu:sycl:<i>` / `gpu:sycl:<i>`, matching the `ComputeDomain` grammar. A kind that has no
 * device stays unregistered, so `SchedulerRegistry::tryResolve` reports its absence rather than
 * silently handing back the CPU fallback. Returns false when the build has no SYCL backend.
 */
[[nodiscard]] inline bool registerSyclRuntime() {
#if GR_DEVICE_HAS_SYCL_IMPL
    static std::once_flag once;
    std::call_once(once, [] {
        ComputeRegistry::instance().register_provider("sycl", &detail::defaultSyclUsmProvider);

        SchedulerRegistry& registry = SchedulerRegistry::instance();
        const auto         publish  = [&registry](std::string_view domain, sycl::queue& queue) { registry.registerContext(domain, std::make_unique<DeviceContextSycl>(queue)); };

        // the default queue backs the USM provider, so it claims the canonical domain of its own kind
        // and no second queue is opened for its device
        sycl::queue&       defaultQueue  = detail::defaultSyclQueue();
        const sycl::device defaultDevice = defaultQueue.get_device();

        bool cpuCanonical = false;
        bool gpuCanonical = false;
        if (defaultDevice.is_cpu() || defaultDevice.is_gpu()) {
            const bool gpu = defaultDevice.is_gpu();
            publish(gpu ? "gpu:sycl" : "cpu:sycl", defaultQueue);
            (gpu ? gpuCanonical : cpuCanonical) = true;
        }

        std::size_t cpuIndex = 0UZ;
        std::size_t gpuIndex = 0UZ;
        for (const sycl::device& device : sycl::device::get_devices()) {
            const bool gpu = device.is_gpu();
            if (!gpu && !device.is_cpu()) {
                continue; // accelerators/FPGAs need their own domain kind
            }

            sycl::queue* queue = &defaultQueue;
            if (device != defaultDevice) {
                queue = detail::enumeratedSyclQueues().emplace_back(std::make_unique<sycl::queue>(device, sycl::property::queue::in_order{})).get();
            }

            const std::string kind      = gpu ? "gpu" : "cpu";
            std::size_t&      index     = gpu ? gpuIndex : cpuIndex;
            bool&             canonical = gpu ? gpuCanonical : cpuCanonical;

            publish(kind + ":sycl:" + std::to_string(index), *queue);
            if (!canonical) {
                publish(kind + ":sycl", *queue);
                canonical = true;
            }
            ++index;
        }
    });
    return true;
#else
    registerUsmProvider();
    return false;
#endif
}

} // namespace gr::device

#endif // GNURADIO_DEVICE_SYCL_RUNTIME_HPP
