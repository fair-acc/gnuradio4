#ifndef GNURADIO_DEVICE_CONTEXT_SYCL_HPP
#define GNURADIO_DEVICE_CONTEXT_SYCL_HPP

#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::device {

#if GR_DEVICE_HAS_SYCL_IMPL

/**
 * @brief SYCL backend for DeviceContext: USM allocation, queue-based transfer, parallel dispatch.
 *
 * Owns a non-owning pointer to a sycl::queue. The queue must outlive this context.
 * Provides parallelFor() for SYCL kernel dispatch (not part of the virtual base — SYCL-specific).
 *
 * @example
 * sycl::queue q(sycl::gpu_selector_v);
 * gr::device::DeviceContextSycl ctx(q);
 * auto* buf = ctx.allocateDevice<float>(1024);
 * ctx.parallelFor(1024, [buf](std::size_t i) { buf[i] *= 2.f; });
 * ctx.wait();
 */
struct DeviceContextSycl final : DeviceContext {
    using DeviceContext::copyDeviceToHost;
    using DeviceContext::copyHostToDevice;

    sycl::queue* queue = nullptr;

    explicit DeviceContextSycl(sycl::queue& q) : queue(&q) {}

    [[nodiscard]] DeviceBackend backend() const noexcept override { return DeviceBackend::SYCL; }
    [[nodiscard]] DeviceType    deviceType() const noexcept override {
        if (queue->get_device().is_gpu()) {
            return DeviceType::GPU;
        }
        if (queue->get_device().is_cpu()) {
            return DeviceType::CPU;
        }
        return DeviceType::Accelerator;
    }
    [[nodiscard]] std::string shortName() const override {
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
    }
    [[nodiscard]] std::string name() const override { return queue->get_device().get_info<sycl::info::device::name>(); }
    [[nodiscard]] std::string version() const override { return queue->get_device().get_info<sycl::info::device::driver_version>(); }

    void copyHostToDevice(const void* host, void* device, std::size_t bytes) override { queue->memcpy(device, host, bytes).wait(); }
    void copyDeviceToHost(const void* device, void* host, std::size_t bytes) override { queue->memcpy(host, device, bytes).wait(); }
    void wait() override { queue->wait(); }

    // SYCL-specific: parallel dispatch with synchronous wait
    template<typename F>
    void parallelFor(std::size_t count, F&& f) {
        queue->submit([count, f = std::forward<F>(f)](sycl::handler& h) { h.parallel_for(sycl::range<1>{count}, [f](sycl::id<1> idx) { f(idx[0]); }); }).wait();
    }

    [[nodiscard]] void* allocateDeviceRaw(std::size_t bytes, std::size_t /*alignment*/) override { return sycl::malloc_device(bytes, *queue); }
    [[nodiscard]] void* allocateHostRaw(std::size_t bytes, std::size_t /*alignment*/) override { return sycl::malloc_host(bytes, *queue); }
    [[nodiscard]] void* allocateSharedRaw(std::size_t bytes, std::size_t /*alignment*/) override { return sycl::malloc_shared(bytes, *queue); }
    void                deallocateRaw(void* ptr) override { sycl::free(ptr, *queue); }
};

#endif // GR_DEVICE_HAS_SYCL_IMPL

} // namespace gr::device

#endif // GNURADIO_DEVICE_CONTEXT_SYCL_HPP
