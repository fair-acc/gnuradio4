#ifndef GNURADIO_DEVICE_SYCL_RUNTIME_HPP
#define GNURADIO_DEVICE_SYCL_RUNTIME_HPP

#include <array>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Export.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/UsmMemoryResource.hpp>

namespace gr::device {

#if GR_DEVICE_HAS_SYCL
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

GNURADIO_EXPORT inline sycl::queue& defaultSyclQueue() {
    static sycl::queue queue{makeSyclAsyncHandler(), sycl::property_list{sycl::property::queue::in_order{}}};
    return queue;
}

GNURADIO_EXPORT inline UsmMemoryResource& defaultSyclUsmResource() {
    static UsmMemoryResource resource{defaultSyclQueue()};
    return resource;
}

// (kind, deviceIndex) -> the resource of the queue that serves that domain; deviceIndex -1 = the kind's canonical device
GNURADIO_EXPORT inline std::map<std::pair<std::string, int>, UsmMemoryResource*>& syclUsmResourcesByDomain() {
    static auto& resources = *new std::map<std::pair<std::string, int>, UsmMemoryResource*>(); // never destroyed, as `syclQueues()`; immutable after registration
    return resources;
}

inline std::pmr::memory_resource* defaultSyclUsmProvider(const ComputeDomain& domain, void* ctx) {
    if (ctx != nullptr) {
        return static_cast<std::pmr::memory_resource*>(ctx);
    }
    // an edge interior to one device never crosses to the host, so it can hold memory the host cannot address
    if (domain.access == Access::DeviceOnly) {
        const auto& byDomain = syclUsmResourcesByDomain();
        auto        entry    = byDomain.find({std::string(domain.kind), domain.deviceIndex});
        if (entry == byDomain.end()) {
            entry = byDomain.find({std::string(domain.kind), -1});
        }
        if (entry != byDomain.end() && entry->second->queue() != nullptr) {
            return &deviceOnlyResourceFor(*entry->second->queue());
        }
    }
    // a device edge crossing back to the host: filled by one bulk copy, then READ by the host
    if (domain.access == Access::HostOnly) {
        const auto& byDomain = syclUsmResourcesByDomain();
        auto        entry    = byDomain.find({std::string(domain.kind), domain.deviceIndex});
        if (entry == byDomain.end()) {
            entry = byDomain.find({std::string(domain.kind), -1});
        }
        if (entry != byDomain.end() && entry->second->queue() != nullptr) {
            return &pinnedHostResourceFor(*entry->second->queue());
        }
    }
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
    // Deliberately never destroyed. These queues outlive `main`, and AdaptiveCpp's own globals tear down in an
    // order no translation unit controls: destroying them at exit races the CUDA driver's deinitialisation and
    // reports `CUDA_ERROR_DEINITIALIZED` (error code 4) from `~cuda_queue`. A process-lifetime runtime handle is
    // the textbook case for a leak on purpose -- the OS reclaims it, and nothing observes the difference.
    static auto& queues = *new std::vector<std::unique_ptr<sycl::queue>>();
    return queues;
}

} // namespace detail
#endif

/// A build with a SYCL backend always enumerates a CPU device, so `host:sycl` is served unless the runtime is
/// broken. A device test that finds it absent has quietly asserted nothing, which is the failure mode this exists
/// to make loud; a GPU, by contrast, may legitimately be missing.
[[nodiscard]] inline bool hostSyclIsServed() { return DeviceContextRegistry::instance().tryResolve("host:sycl") != nullptr; }

/**
 * @brief discover the SYCL devices and publish one `DeviceContext` per device.
 *
 * Publishes one context per device as `host:sycl:<i>` / `gpu:sycl:<i>`, and points the un-indexed
 * `host:sycl` / `gpu:sycl` at one of them — the default queue's device where it has one — so both spellings name
 * the same context. A kind with no device stays unregistered, so `tryResolve` reports its absence rather than
 * silently handing back the CPU fallback. False when the build has no SYCL backend.
 */
[[nodiscard]] GNURADIO_EXPORT inline bool registerSyclRuntime() {
#if GR_DEVICE_HAS_SYCL
    static std::once_flag once;
    std::call_once(once, [] {
        ComputeRegistry::instance().registerProvider("sycl", &detail::defaultSyclUsmProvider);
        ComputeRegistry::instance().registerDomainResolver(+[](std::string_view declaredDomain) { return DeviceContextRegistry::instance().resolve(declaredDomain).resolved; });

        DeviceContextRegistry& registry = DeviceContextRegistry::instance();
        const auto             publish  = [&registry](const std::string& kind, int deviceIndex, sycl::queue& queue) {
            auto context            = std::make_unique<DeviceContextSycl>(queue, detail::syclErrorState());
            context->sharedResource = &usmResourceFor(queue); // the same object the edge allocator uses, not a second one
            registry.registerContext(kind + ":sycl:" + std::to_string(deviceIndex), std::move(context));
            detail::syclUsmResourcesByDomain()[{kind, deviceIndex}] = &usmResourceFor(queue);
        };
        const auto claimUnindexedSpelling = [&registry](const std::string& kind, int deviceIndex) {
            registry.registerAlias(kind + ":sycl", kind + ":sycl:" + std::to_string(deviceIndex));
            detail::syclUsmResourcesByDomain()[{kind, -1}] = detail::syclUsmResourcesByDomain()[{kind, deviceIndex}];
        };

        // the default queue backs the USM provider, so its device claims the un-indexed spelling of its own kind
        sycl::queue&       defaultQueue  = detail::defaultSyclQueue();
        const sycl::device defaultDevice = defaultQueue.get_device();

        bool cpuCanonical = false;
        bool gpuCanonical = false;

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
            if (!canonical || device == defaultDevice) {
                claimUnindexedSpelling(kind, static_cast<int>(index));
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
