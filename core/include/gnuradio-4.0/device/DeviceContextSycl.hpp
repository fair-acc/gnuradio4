#ifndef GNURADIO_DEVICE_CONTEXT_SYCL_HPP
#define GNURADIO_DEVICE_CONTEXT_SYCL_HPP

#include <gnuradio-4.0/device/BackendCompat.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::device {

/**
 * @brief SYCL backend for DeviceContext: USM allocation, queue-based transfer, parallel dispatch.
 *
 * Owns a non-owning pointer to a SyclQueue. The queue must outlive this context. In non-SYCL
 * builds, SyclQueue is a null host-copy queue so block specialisations can keep one signature.
 *
 * @example
 * gr::device::SyclQueue q;
 * gr::device::DeviceContextSycl ctx(q);
 * auto* buf = ctx.allocateDevice<float>(1024);
 * ctx.parallelFor(1024, [buf](std::size_t i) { buf[i] *= 2.f; });
 * ctx.wait();
 */
struct DeviceContextSycl final : DeviceContext {
    using DeviceContext::copyDeviceToHost;
    using DeviceContext::copyHostToDevice;

    SyclQueue* queue = nullptr;

    explicit DeviceContextSycl(SyclQueue& q) : queue(&q) {}

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

    void copyHostToDevice(const void* host, void* device, std::size_t bytes) override { queue->memcpy(device, host, bytes).wait(); }
    void copyDeviceToHost(const void* device, void* host, std::size_t bytes) override { queue->memcpy(host, device, bytes).wait(); }
    void wait() override { queue->wait(); }

    // SYCL-specific: parallel dispatch with synchronous wait
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
    [[nodiscard]] void* allocateDeviceRaw(std::size_t bytes, std::size_t /*alignment*/) override { return sycl::malloc_device(bytes, *queue); }
    [[nodiscard]] void* allocateHostRaw(std::size_t bytes, std::size_t /*alignment*/) override { return sycl::malloc_host(bytes, *queue); }
    [[nodiscard]] void* allocateSharedRaw(std::size_t bytes, std::size_t /*alignment*/) override { return sycl::malloc_shared(bytes, *queue); }
    void                deallocateRaw(void* ptr) override { sycl::free(ptr, *queue); }
#else
    static constexpr std::size_t kAlign = alignof(std::max_align_t);
    [[nodiscard]] void*          allocateDeviceRaw(std::size_t bytes, std::size_t /*alignment*/) override { return ::operator new(bytes, std::align_val_t{kAlign}, std::nothrow); }
    [[nodiscard]] void*          allocateHostRaw(std::size_t bytes, std::size_t /*alignment*/) override { return ::operator new(bytes, std::align_val_t{kAlign}, std::nothrow); }
    [[nodiscard]] void*          allocateSharedRaw(std::size_t bytes, std::size_t /*alignment*/) override { return ::operator new(bytes, std::align_val_t{kAlign}, std::nothrow); }
    void                         deallocateRaw(void* ptr) override { ::operator delete(ptr, std::align_val_t{kAlign}); }
#endif
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_CONTEXT_SYCL_HPP
