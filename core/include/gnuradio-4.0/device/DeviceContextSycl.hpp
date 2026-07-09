#ifndef GNURADIO_DEVICE_CONTEXT_SYCL_HPP
#define GNURADIO_DEVICE_CONTEXT_SYCL_HPP

#include <atomic>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <gnuradio-4.0/Logger.hpp>
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
    static constexpr DeviceBackend kBackend = DeviceBackend::SYCL; // lets DeviceContext::as<> check without RTTI

    /// a sycl::queue is a reference-counted handle, so the context keeps the referent alive rather than pointing at
    /// a queue the caller may destroy first -- scratch is released from destructors that run long after the call
    SyclQueue                        ownedQueue;
    SyclQueue*                       queue = nullptr;
    std::shared_ptr<AsyncErrorState> errorState;

    /// Deferred completion rests on this queue being IN-ORDER: a dispatch that returns before its kernel has run is
    /// safe only because every later access to the same device memory is enqueued behind it. An out-of-order queue
    /// would pass every residency test and silently break that, so the assumption is checked once, here.
    explicit DeviceContextSycl(SyclQueue& q, std::shared_ptr<AsyncErrorState> state = nullptr) : ownedQueue(q), queue(&ownedQueue), errorState(std::move(state)) {
#if GR_DEVICE_HAS_SYCL
        if (!queue->is_in_order()) {
            gr::log::error("device context: the SYCL queue is out-of-order; deferred completion assumes in-order submission and will be unsound on it");
        }
#endif
    }

    DeviceContextSycl(const DeviceContextSycl&)            = delete; // `queue` points into this object
    DeviceContextSycl& operator=(const DeviceContextSycl&) = delete;
    DeviceContextSycl(DeviceContextSycl&&)                 = delete;
    DeviceContextSycl& operator=(DeviceContextSycl&&)      = delete;

    [[nodiscard]] bool isDeviceAccessible([[maybe_unused]] const void* ptr) const noexcept override {
#if GR_DEVICE_HAS_SYCL
        if (ptr == nullptr || queue == nullptr) {
            return false;
        }
        const sycl::usm::alloc kind = sycl::get_pointer_type(ptr, queue->get_context());
        return kind == sycl::usm::alloc::shared || kind == sycl::usm::alloc::device;
#else
        return false;
#endif
    }

    [[nodiscard]] bool isDeviceOnly([[maybe_unused]] const void* ptr) const noexcept override {
#if GR_DEVICE_HAS_SYCL
        if (ptr == nullptr || queue == nullptr) {
            return false;
        }
        return sycl::get_pointer_type(ptr, queue->get_context()) == sycl::usm::alloc::device;
#else
        return false;
#endif
    }

    void copy(void* destination, const void* source, std::size_t bytes, bool await = true) override {
        auto event = queue->memcpy(destination, source, bytes);
        if (await) {
            event.wait();
            _syncCount.fetch_add(1UZ, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] DeviceBackend backend() const noexcept override {
#if GR_DEVICE_HAS_SYCL
        return DeviceBackend::SYCL;
#else
        return DeviceBackend::CPU_Fallback;
#endif
    }
    [[nodiscard]] DeviceType deviceType() const noexcept override {
#if GR_DEVICE_HAS_SYCL
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
#if GR_DEVICE_HAS_SYCL
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
#if GR_DEVICE_HAS_SYCL
    [[nodiscard]] std::string name() const override { return queue->get_device().get_info<sycl::info::device::name>(); }
    [[nodiscard]] std::string version() const override { return queue->get_device().get_info<sycl::info::device::driver_version>(); }
#else
    [[nodiscard]] std::string name() const override { return "null SYCL queue"; }
    [[nodiscard]] std::string version() const override { return "none"; }
#endif

    void upload(const void* host, DeviceBuffer dst, std::size_t bytes) override {
        queue->memcpy(reinterpret_cast<void*>(dst.token), host, bytes).wait();
        _syncCount.fetch_add(1UZ, std::memory_order_relaxed);
    }
    void download(DeviceBuffer src, void* host, std::size_t bytes) override {
        queue->memcpy(host, reinterpret_cast<void*>(src.token), bytes).wait();
        _syncCount.fetch_add(1UZ, std::memory_order_relaxed);
    }
    void wait() override {
        queue->wait();
        _syncCount.fetch_add(1UZ, std::memory_order_relaxed);
    }

    [[nodiscard]] std::optional<std::string> peekDeviceError() noexcept override {
#if GR_DEVICE_HAS_SYCL
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
#if GR_DEVICE_HAS_SYCL
        // sync so a pending fault registers into acpp's global list, then drain it; a top-level queue::wait()
        // registers even when the block synced individual events. Skip a poisoned queue: each re-issue re-faults.
        if (queue && !(errorState && errorState->isPoisoned())) {
            queue->wait();
            queue->throw_asynchronous();
            _syncCount.fetch_add(1UZ, std::memory_order_relaxed);
        }
#endif
        if (errorState && errorState->isPoisoned()) {
            return errorState->messageCopy();
        }
        return std::nullopt;
    }

    void recordKernelFailure(std::string_view cause) {
        const std::string message = std::format("kernel submission failed: {}", cause);
        if (errorState) {
            errorState->record(message);
        } else {
            gr::log::error("DeviceContextSycl::parallelFor: {}", message);
        }
    }

    /// `await = false` returns once the kernel is enqueued. Correct only on an in-order queue and only when every
    /// buffer it touches is device-only, so nothing on this thread can observe the half-finished result.
    template<typename F>
    void parallelFor(std::size_t count, F&& f, bool await = true) {
#if GR_DEVICE_HAS_SYCL
        // a submit or a wait can throw synchronously, and this runs on a scheduler thread: an escaping exception
        // would terminate the process instead of being reported, so it poisons the context and surfaces as a
        // refusal through the error poll that every dispatch path already makes
        try {
            auto event = queue->submit([count, f = std::forward<F>(f)](sycl::handler& h) { h.parallel_for(sycl::range<1>{count}, [f](sycl::id<1> idx) { f(idx[0]); }); });
            if (await) {
                event.wait();
                _syncCount.fetch_add(1UZ, std::memory_order_relaxed);
            }
        } catch (const std::exception& e) {
            recordKernelFailure(e.what());
        } catch (...) {
            recordKernelFailure("unknown exception");
        }
#else
        std::ignore = await;
        for (std::size_t i = 0; i < count; ++i) {
            f(i);
        }
#endif
    }

#if GR_DEVICE_HAS_SYCL
    /// AdaptiveCpp's host backend returns the same pointer for `aligned_alloc_*` as for `malloc_*`, so an
    /// over-aligned request is honoured here by over-allocating and aligning up rather than by trusting it.
    [[nodiscard]] DeviceBuffer allocate(std::size_t bytes, std::size_t align, Residency wanted) override {
        const std::size_t effective = std::max(align, alignof(std::max_align_t));
        const std::size_t request   = bytes + effective - 1UZ;
        void*             p         = nullptr;
        if (wanted == Residency::host) {
            p = sycl::aligned_alloc_host(effective, request, *queue);
        } else if (wanted == Residency::shared) {
            p = sycl::aligned_alloc_shared(effective, request, *queue);
        } else if (wanted == Residency::devicePtr) {
            p = sycl::aligned_alloc_device(effective, request, *queue);
        } else {
            return {}; // "invalid" is never a request to honour
        }
        if (p == nullptr) {
            return {};
        }
        const std::uintptr_t raw     = reinterpret_cast<std::uintptr_t>(p);
        const std::uintptr_t aligned = (raw + effective - 1UZ) & ~static_cast<std::uintptr_t>(effective - 1UZ);
        return DeviceBuffer{.token = aligned, .bytes = bytes, .residency = wanted, .align = effective, .allocation = raw};
    }

    void deallocate(DeviceBuffer buf) override {
        if (!buf) {
            return;
        }
        sycl::free(reinterpret_cast<void*>(buf.allocation != 0 ? buf.allocation : buf.token), *queue);
    }
#else
    static constexpr std::size_t kAlign = alignof(std::max_align_t);

    [[nodiscard]] DeviceBuffer allocate(std::size_t bytes, std::size_t align, Residency wanted) override {
        if (wanted == Residency::invalid) {
            return {};
        }
        const std::size_t effective = std::max(align, kAlign);
        void*             p         = ::operator new(bytes, std::align_val_t{effective}, std::nothrow);
        return p == nullptr ? DeviceBuffer{} : DeviceBuffer{.token = reinterpret_cast<std::uintptr_t>(p), .bytes = bytes, .residency = wanted, .align = effective, .allocation = reinterpret_cast<std::uintptr_t>(p)};
    }

    void deallocate(DeviceBuffer buf) override {
        if (!buf) {
            return;
        }
        ::operator delete(reinterpret_cast<void*>(buf.allocation != 0 ? buf.allocation : buf.token), std::align_val_t{buf.align});
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
    static std::mutex                                                        mutex;
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
