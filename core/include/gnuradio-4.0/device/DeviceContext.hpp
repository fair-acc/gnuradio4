#ifndef GNURADIO_DEVICE_CONTEXT_HPP
#define GNURADIO_DEVICE_CONTEXT_HPP

#include <cstddef>
#include <cstring>
#include <new>
#include <string>

#include <gnuradio-4.0/device/BackendDetect.hpp>

namespace gr::device {

/**
 * @brief Abstract base for backend-agnostic device memory management and data transfer.
 *
 * Subclasses implement allocation, transfer, and synchronisation for a specific backend.
 * Users can implement their own backends (CUDA, ROCm, ...) by subclassing DeviceContext.
 *
 * @example
 * std::unique_ptr<DeviceContext> ctx = std::make_unique<DeviceContextCpu>();
 * auto* buf = ctx->allocateDevice<float>(1024);
 * ctx->copyHostToDevice(hostData, buf, 1024);
 * ctx->deallocate(buf);
 */
struct DeviceContext {
    virtual ~DeviceContext() = default;

    [[nodiscard]] virtual DeviceBackend backend() const noexcept    = 0;
    [[nodiscard]] virtual DeviceType    deviceType() const noexcept = 0;
    [[nodiscard]] virtual std::string   shortName() const           = 0; // "CPU", "SYCL:RTX 3070", "GLSL:RTX 3070"
    [[nodiscard]] virtual std::string   name() const                = 0; // "NVIDIA GeForce RTX 3070"
    [[nodiscard]] virtual std::string   version() const             = 0; // "OpenGL 4.3 NVIDIA 595.45.04"
    [[nodiscard]] bool                  isGpu() const noexcept { return deviceType() == DeviceType::GPU; }

    // backward compat
    [[nodiscard]] std::string deviceName() const { return name(); }

    virtual void copyHostToDevice(const void* host, void* device, std::size_t bytes) = 0;
    virtual void copyDeviceToHost(const void* device, void* host, std::size_t bytes) = 0;
    virtual void wait()                                                              = 0;

    template<typename T>
    [[nodiscard]] T* allocateDevice(std::size_t count) {
        return static_cast<T*>(allocateDeviceRaw(count * sizeof(T), alignof(T)));
    }
    template<typename T>
    [[nodiscard]] T* allocateHost(std::size_t count) {
        return static_cast<T*>(allocateHostRaw(count * sizeof(T), alignof(T)));
    }
    template<typename T>
    [[nodiscard]] T* allocateShared(std::size_t count) {
        return static_cast<T*>(allocateSharedRaw(count * sizeof(T), alignof(T)));
    }
    template<typename T>
    void deallocate(T* ptr) {
        deallocateRaw(static_cast<void*>(ptr));
    }

    template<typename T>
    void copyHostToDevice(const T* host, T* device, std::size_t count) {
        copyHostToDevice(static_cast<const void*>(host), static_cast<void*>(device), count * sizeof(T));
    }
    template<typename T>
    void copyDeviceToHost(const T* device, T* host, std::size_t count) {
        copyDeviceToHost(static_cast<const void*>(device), static_cast<void*>(host), count * sizeof(T));
    }

    [[nodiscard]] virtual void* allocateDeviceRaw(std::size_t bytes, std::size_t alignment) = 0;
    [[nodiscard]] virtual void* allocateHostRaw(std::size_t bytes, std::size_t alignment)   = 0;
    [[nodiscard]] virtual void* allocateSharedRaw(std::size_t bytes, std::size_t alignment) = 0;
    virtual void                deallocateRaw(void* ptr)                                    = 0;
};

/// @brief CPU-only DeviceContext: heap allocation, memcpy transfers, no GPU.
struct DeviceContextCpu final : DeviceContext {
    using DeviceContext::copyDeviceToHost;
    using DeviceContext::copyHostToDevice;

    [[nodiscard]] DeviceBackend backend() const noexcept override { return DeviceBackend::CPU_Fallback; }
    [[nodiscard]] DeviceType    deviceType() const noexcept override { return DeviceType::CPU; }
    [[nodiscard]] std::string   shortName() const override { return "CPU"; }
    [[nodiscard]] std::string   name() const override { return "CPU fallback"; }
    [[nodiscard]] std::string   version() const override { return "host"; }

    void copyHostToDevice(const void* host, void* device, std::size_t bytes) override { std::memcpy(device, host, bytes); }
    void copyDeviceToHost(const void* device, void* host, std::size_t bytes) override { std::memcpy(host, device, bytes); }
    void wait() override {}

    static constexpr std::size_t kAlign = alignof(std::max_align_t);
    [[nodiscard]] void*          allocateDeviceRaw(std::size_t bytes, std::size_t) override { return ::operator new(bytes, std::align_val_t{kAlign}); }
    [[nodiscard]] void*          allocateHostRaw(std::size_t bytes, std::size_t) override { return ::operator new(bytes, std::align_val_t{kAlign}); }
    [[nodiscard]] void*          allocateSharedRaw(std::size_t bytes, std::size_t) override { return ::operator new(bytes, std::align_val_t{kAlign}); }
    void                deallocateRaw(void* ptr) override { ::operator delete(ptr, std::align_val_t{kAlign}); }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_CONTEXT_HPP
