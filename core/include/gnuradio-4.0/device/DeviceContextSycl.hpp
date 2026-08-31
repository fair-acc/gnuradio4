#ifndef GNURADIO_DEVICE_CONTEXT_SYCL_HPP
#define GNURADIO_DEVICE_CONTEXT_SYCL_HPP

#include <unordered_map>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::device {

/**
 * @brief latch for AdaptiveCpp's process-global asynchronous error list.
 *
 * acpp delivers a faulting kernel or copy only through `queue::throw_asynchronous()`, never a plain `wait()`, and
 * a CUDA error invalidates the device's sticky primary context that every queue on it shares — so one instance is
 * shared by all of them and a fault poisons the process. Latches permanently: acpp exposes no reset.
 */
struct AsyncErrorState {
    std::atomic<bool> poisoned{false};
    std::mutex        mtx;
    std::string       message;

    void record(std::string msg) {
        std::scoped_lock lock(mtx);
        if (!poisoned.load(std::memory_order_relaxed)) {
            message = std::move(msg);
        }
        poisoned.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool isPoisoned() const noexcept { return poisoned.load(std::memory_order_acquire); }

    [[nodiscard]] std::string messageCopy() {
        std::scoped_lock lock(mtx);
        return message;
    }
};

/**
 * @brief SYCL backend for DeviceContext: USM allocation, queue-based transfer, parallel dispatch.
 *
 * Borrows the queue, which must outlive the context. Without SYCL, `SyclQueue` is a null host-copy queue so block
 * specialisations keep one signature.
 */
struct DeviceContextSycl final : DeviceContext {
    /// a sycl::queue is a reference-counted handle, so the context keeps the referent alive rather than pointing at
    /// a queue the caller may destroy first -- scratch is released from destructors that run long after the call
    SyclQueue                        ownedQueue;
    SyclQueue*                       queue = nullptr;
    std::shared_ptr<AsyncErrorState> errorState;

    explicit DeviceContextSycl(SyclQueue& q, std::shared_ptr<AsyncErrorState> state = nullptr) : ownedQueue(q), queue(&ownedQueue), errorState(std::move(state)) {}

    DeviceContextSycl(const DeviceContextSycl&)            = delete; // `queue` points into this object
    DeviceContextSycl& operator=(const DeviceContextSycl&) = delete;
    DeviceContextSycl(DeviceContextSycl&&)                 = delete;
    DeviceContextSycl& operator=(DeviceContextSycl&&)      = delete;

    [[nodiscard]] bool isDeviceAccessible([[maybe_unused]] const void* ptr) const noexcept override {
#if GR_DEVICE_HAS_SYCL_IMPL
        if (ptr == nullptr || queue == nullptr) {
            return false;
        }
        const sycl::usm::alloc kind = sycl::get_pointer_type(ptr, queue->get_context());
        return kind == sycl::usm::alloc::shared || kind == sycl::usm::alloc::device;
#else
        return false;
#endif
    }

    [[nodiscard]] DeviceBackend backend() const noexcept override {
#if GR_DEVICE_HAS_SYCL_IMPL
        return DeviceBackend::SYCL;
#else
        return DeviceBackend::CPU_Fallback;
#endif
    }
    [[nodiscard]] DeviceType deviceType() const noexcept override {
#if GR_DEVICE_HAS_SYCL_IMPL
        if (queue->get_device().is_gpu()) {
            return DeviceType::GPU;
        }
        if (queue->get_device().is_cpu()) {
            return DeviceType::CPU;
        }
        return DeviceType::Accelerator;
#else
        return DeviceType::CPU;
#endif
    }
    [[nodiscard]] std::string shortName() const override {
#if GR_DEVICE_HAS_SYCL_IMPL
        auto dev = queue->get_device();
        if (dev.is_cpu()) {
            return "SYCL:CPU";
        }
        // GPU: keep vendor + model, trim bus/interface suffixes
        auto devName = dev.get_info<sycl::info::device::name>();
        if (auto pos = devName.find("/PCIe"); pos != std::string::npos) {
            devName.resize(pos);
        }
        if (auto pos = devName.find("/SSE"); pos != std::string::npos) {
            devName.resize(pos);
        }
        return "SYCL:" + devName;
#else
        return "SYCL:null";
#endif
    }
#if GR_DEVICE_HAS_SYCL_IMPL
    [[nodiscard]] std::string name() const override { return queue->get_device().get_info<sycl::info::device::name>(); }
    [[nodiscard]] std::string version() const override { return queue->get_device().get_info<sycl::info::device::driver_version>(); }
#else
    [[nodiscard]] std::string name() const override { return "null SYCL queue"; }
    [[nodiscard]] std::string version() const override { return "none"; }
#endif

    void upload(const void* host, DeviceBuffer dst, std::size_t bytes) override {
        queue->memcpy(reinterpret_cast<void*>(dst.token), host, bytes).wait();
    }
    void download(DeviceBuffer src, void* host, std::size_t bytes) override {
        queue->memcpy(host, reinterpret_cast<void*>(src.token), bytes).wait();
    }
    void wait() override { queue->wait(); }

    [[nodiscard]] std::optional<std::string> peekDeviceError() noexcept override {
#if GR_DEVICE_HAS_SYCL_IMPL
        // waitless: drain acpp's global list for ops that have ALREADY completed, never block on work in flight,
        // never touch a poisoned context again
        if (queue && !(errorState && errorState->isPoisoned())) {
            queue->throw_asynchronous();
        }
#endif
        if (errorState && errorState->isPoisoned()) {
            return errorState->messageCopy();
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::string> pollDeviceError() noexcept override {
#if GR_DEVICE_HAS_SYCL_IMPL
        // sync so a pending fault registers into acpp's global list, then drain it; a top-level queue::wait()
        // registers even when the block synced individual events. Skip a poisoned queue: each re-issue re-faults.
        if (queue && !(errorState && errorState->isPoisoned())) {
            queue->wait();
            queue->throw_asynchronous();
        }
#endif
        if (errorState && errorState->isPoisoned()) {
            return errorState->messageCopy();
        }
        return std::nullopt;
    }

    template<typename F>
    void parallelFor(std::size_t count, F&& f) {
#if GR_DEVICE_HAS_SYCL_IMPL
        queue->submit([count, f = std::forward<F>(f)](sycl::handler& h) { h.parallel_for(sycl::range<1>{count}, [f](sycl::id<1> idx) { f(idx[0]); }); }).wait();
#else
        for (std::size_t i = 0; i < count; ++i) {
            f(i);
        }
#endif
    }

#if GR_DEVICE_HAS_SYCL_IMPL
    [[nodiscard]] DeviceBuffer allocate(std::size_t bytes, std::size_t /*align*/, Residency wanted) override {
        void* p = nullptr;
        if (wanted == Residency::host) {
            p = sycl::malloc_host(bytes, *queue);
        } else if (wanted == Residency::shared) {
            p = sycl::malloc_shared(bytes, *queue);
        } else if (wanted == Residency::devicePtr) {
            p = sycl::malloc_device(bytes, *queue);
        } else {
            return {}; // "invalid" is never a request to honour
        }
        return p == nullptr ? DeviceBuffer{} : DeviceBuffer{.token = reinterpret_cast<std::uintptr_t>(p), .bytes = bytes, .residency = wanted};
    }

    void deallocate(DeviceBuffer buf) override {
        if (!buf) {
            return;
        }
        sycl::free(reinterpret_cast<void*>(buf.token), *queue);
    }
#else
    static constexpr std::size_t kAlign = alignof(std::max_align_t);

    [[nodiscard]] DeviceBuffer allocate(std::size_t bytes, std::size_t /*align*/, Residency wanted) override {
        if (wanted == Residency::invalid) {
            return {};
        }
        void* p = ::operator new(bytes, std::align_val_t{kAlign}, std::nothrow);
        return p == nullptr ? DeviceBuffer{} : DeviceBuffer{.token = reinterpret_cast<std::uintptr_t>(p), .bytes = bytes, .residency = wanted};
    }

    void deallocate(DeviceBuffer buf) override {
        if (!buf) {
            return;
        }
        ::operator delete(reinterpret_cast<void*>(buf.token), std::align_val_t{kAlign});
    }
#endif
};

/**
 * @brief The context belonging to a queue, owned for the run rather than by whoever asked first.
 *
 * A block handed a queue through an expert hatch must not build and keep its own context: the queue may be a
 * local that dies before the block, and a second call with a different queue would otherwise be served by the
 * first one's context, quietly using the wrong scratch. Keyed on the queue itself -- a SYCL queue is a
 * reference-counted handle, so a copy of a queue resolves to the same context as the original.
 */
[[nodiscard]] inline DeviceContextSycl& syclContextFor(SyclQueue& queue) {
    static std::mutex                                                       mutex;
    static std::unordered_map<SyclQueue, std::unique_ptr<DeviceContextSycl>> contexts;

    std::scoped_lock lock(mutex);
    auto&            slot = contexts[queue];
    if (!slot) {
        slot = std::make_unique<DeviceContextSycl>(queue);
    }
    return *slot;
}

} // namespace gr::device

#endif // GNURADIO_DEVICE_CONTEXT_SYCL_HPP
