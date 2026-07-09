#ifndef GNURADIO_DEVICE_SYCL_RUNTIME_HPP
#define GNURADIO_DEVICE_SYCL_RUNTIME_HPP

#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/device/UsmMemoryResource.hpp>

namespace gr::device {

#if GR_DEVICE_HAS_SYCL_IMPL
namespace detail {

// the process-global latch behind every queue's async handler; see AsyncErrorState
inline std::shared_ptr<AsyncErrorState> syclErrorState() {
    static std::shared_ptr<AsyncErrorState> s = std::make_shared<AsyncErrorState>();
    return s;
}

inline sycl::async_handler makeSyclAsyncHandler() {
    auto state = syclErrorState();
    return [state](sycl::exception_list errs) {
        for (const std::exception_ptr& e : errs) {
            try {
                std::rethrow_exception(e);
            } catch (const std::exception& ex) {
                state->record(ex.what());
            } catch (...) {
                state->record("unknown device async error");
            }
        }
    };
}

inline sycl::queue& defaultSyclQueue() {
    static sycl::queue queue{makeSyclAsyncHandler(), sycl::property_list{sycl::property::queue::in_order{}}};
    return queue;
}

inline UsmMemoryResource& defaultSyclUsmResource() {
    static UsmMemoryResource resource{defaultSyclQueue()};
    return resource;
}

// one resource per queue: a USM pointer is only dereferenceable from its queue's context
inline UsmMemoryResource& usmResourceFor(sycl::queue& queue) {
    static std::map<sycl::queue*, std::unique_ptr<UsmMemoryResource>> byQueue; // filled during registration only
    if (&queue == &defaultSyclQueue()) {
        return defaultSyclUsmResource();
    }
    auto [entry, inserted] = byQueue.try_emplace(&queue);
    if (inserted) {
        entry->second = std::make_unique<UsmMemoryResource>(queue);
    }
    return *entry->second;
}

// (kind, deviceIndex) -> the resource of the queue that serves that domain; deviceIndex -1 = the kind's canonical device
inline std::map<std::pair<std::string, int>, UsmMemoryResource*>& syclUsmResourcesByDomain() {
    static std::map<std::pair<std::string, int>, UsmMemoryResource*> resources; // immutable after registration
    return resources;
}

inline std::pmr::memory_resource* defaultSyclUsmProvider(const ComputeDomain& domain, void* ctx) {
    if (ctx != nullptr) {
        return static_cast<std::pmr::memory_resource*>(ctx);
    }
    // `Access::DeviceOnly` records that both endpoints sit on one device; it resolves to shared USM, which is
    // device-resident all the same. A ring in non-host-addressable memory would additionally have to be
    // double-mapped, and no portable SYCL API reserves and maps physical pages twice.
    const auto& resources = syclUsmResourcesByDomain();
    if (const auto exact = resources.find({std::string(domain.kind), domain.deviceIndex}); exact != resources.end()) {
        return exact->second;
    }
    if (const auto canonical = resources.find({std::string(domain.kind), -1}); canonical != resources.end()) {
        return canonical->second;
    }
    return &defaultSyclUsmResource();
}

// `DeviceContextSycl` only borrows these
inline std::vector<std::unique_ptr<sycl::queue>>& enumeratedSyclQueues() {
    static std::vector<std::unique_ptr<sycl::queue>> queues;
    return queues;
}

} // namespace detail
#endif

/**
 * @brief discover the SYCL devices and publish one `DeviceContext` per device.
 *
 * Registers the canonical `host:sycl` / `gpu:sycl` (first device of each kind) and the indexed
 * `host:sycl:<i>` / `gpu:sycl:<i>`. A kind with no device stays unregistered, so `tryResolve` reports its absence
 * rather than silently handing back the CPU fallback. False when the build has no SYCL backend.
 */
[[nodiscard]] inline bool registerSyclRuntime() {
#if GR_DEVICE_HAS_SYCL_IMPL
    static std::once_flag once;
    std::call_once(once, [] {
        ComputeRegistry::instance().register_provider("sycl", &detail::defaultSyclUsmProvider);

        DeviceContextRegistry& registry = DeviceContextRegistry::instance();
        const auto publish = [&registry](const std::string& kind, int deviceIndex, sycl::queue& queue) {
            const std::string domain = deviceIndex < 0 ? kind + ":sycl" : kind + ":sycl:" + std::to_string(deviceIndex);
            registry.registerContext(domain, std::make_unique<DeviceContextSycl>(queue, detail::syclErrorState()));
            detail::syclUsmResourcesByDomain()[{kind, deviceIndex}] = &detail::usmResourceFor(queue);
        };

        // the default queue backs the USM provider, so it claims the canonical domain of its own kind
        sycl::queue&       defaultQueue  = detail::defaultSyclQueue();
        const sycl::device defaultDevice = defaultQueue.get_device();

        bool cpuCanonical = false;
        bool gpuCanonical = false;
        if (defaultDevice.is_cpu() || defaultDevice.is_gpu()) {
            const bool gpu = defaultDevice.is_gpu();
            publish(gpu ? "gpu" : "host", -1, defaultQueue);
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
                queue = detail::enumeratedSyclQueues().emplace_back(std::make_unique<sycl::queue>(device, detail::makeSyclAsyncHandler(), sycl::property_list{sycl::property::queue::in_order{}})).get();
            }

            const std::string kind      = gpu ? "gpu" : "host"; // a SYCL CPU device is host-resident memory, SYCL execution
            std::size_t&      index     = gpu ? gpuIndex : cpuIndex;
            bool&             canonical = gpu ? gpuCanonical : cpuCanonical;

            publish(kind, static_cast<int>(index), *queue);
            if (!canonical) {
                publish(kind, -1, *queue);
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
