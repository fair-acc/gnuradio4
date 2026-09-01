#ifndef GNURADIO_DEVICE_CONTEXT_HPP
#define GNURADIO_DEVICE_CONTEXT_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <string>
#include <type_traits>

#include <gnuradio-4.0/device/BackendDetect.hpp>

namespace gr::device {

/// which side may dereference an allocation
enum class Residency : std::uint8_t { invalid, host, shared, devicePtr };

/**
 * @brief A backend allocation plus the one thing a `void*` cannot carry: who may dereference it.
 *
 * `devicePointer<T>()` yields nullptr for residencies a kernel cannot index, rather than misleading.
 */
struct DeviceBuffer {
    std::uintptr_t token     = 0;
    std::size_t    bytes     = 0;
    Residency      residency = Residency::invalid;

    [[nodiscard]] constexpr explicit operator bool() const noexcept { return residency != Residency::invalid; }

    template<typename T>
    [[nodiscard]] T* devicePointer() const noexcept {
        return (residency == Residency::shared || residency == Residency::devicePtr) ? reinterpret_cast<T*>(token) : nullptr;
    }
};
static_assert(std::is_trivially_copyable_v<DeviceBuffer>);

/**
 * @brief Abstract base for backend-agnostic device memory management and data transfer.
 *
 * Subclass it for a backend (CUDA, ROCm, ...). Handles stop at the kernel-launch boundary: dispatch resolves
 * `devicePointer<T>()` once on the host and captures the raw pointer.
 */
struct DeviceContext {
    virtual ~DeviceContext() = default;

    // false once DeviceContextRegistry::withdraw() marks the published domain gone. The registry never destroys a
    // published context — a block's cached context pointer may outlive the withdrawal — so this flag, checked on
    // every dispatch, is what stops it being used.
    [[nodiscard]] bool served() const noexcept { return _served.load(std::memory_order_acquire); }
    void               withdraw() noexcept { _served.store(false, std::memory_order_release); }

    [[nodiscard]] virtual DeviceBackend backend() const noexcept    = 0;
    [[nodiscard]] virtual DeviceType    deviceType() const noexcept = 0;
    [[nodiscard]] virtual std::string   shortName() const           = 0; // "CPU", "SYCL:CPU", "SYCL:NVIDIA GeForce RTX 3070"
    [[nodiscard]] virtual std::string   name() const                = 0; // "NVIDIA GeForce RTX 3070"
    [[nodiscard]] virtual std::string   version() const             = 0; // the backend's own driver-version string, e.g. "13030"
    [[nodiscard]] bool                  isGpu() const noexcept { return deviceType() == DeviceType::GPU; }

    virtual void wait() = 0;

    // true when `ptr` is memory this context's device can dereference directly (USM); conservatively false
    [[nodiscard]] virtual bool isDeviceAccessible(const void* /*ptr*/) const noexcept { return false; }

    // latch read, no device synchronisation, so a healthy steady state pays nothing. Code submitting device work
    // outside ExecutionStrategy::dispatch must do its own trailing `pollDeviceError`, which does synchronise.
    [[nodiscard]] virtual std::optional<std::string> peekDeviceError() noexcept { return std::nullopt; }

    // non-blocking poll of a backend's async error channel; nullopt means clean, as for a backend without one
    [[nodiscard]] virtual std::optional<std::string> pollDeviceError() noexcept { return std::nullopt; }

    template<typename T>
    [[nodiscard]] DeviceBuffer allocateDevice(std::size_t count) {
        return allocate(count * sizeof(T), alignof(T), Residency::devicePtr);
    }
    template<typename T>
    [[nodiscard]] DeviceBuffer allocateHost(std::size_t count) {
        return allocate(count * sizeof(T), alignof(T), Residency::host);
    }
    template<typename T>
    [[nodiscard]] DeviceBuffer allocateShared(std::size_t count) {
        return allocate(count * sizeof(T), alignof(T), Residency::shared);
    }

    template<typename T>
    void copyHostToDevice(const T* host, DeviceBuffer dst, std::size_t count) {
        upload(static_cast<const void*>(host), dst, count * sizeof(T));
    }
    template<typename T>
    void copyDeviceToHost(DeviceBuffer src, T* host, std::size_t count) {
        download(src, static_cast<void*>(host), count * sizeof(T));
    }

    [[nodiscard]] virtual DeviceBuffer allocate(std::size_t bytes, std::size_t align, Residency wanted) = 0; // invalid = "cannot serve"; NEVER a lying token
    virtual void                       deallocate(DeviceBuffer buf)                                     = 0;
    virtual void                       upload(const void* host, DeviceBuffer dst, std::size_t bytes)    = 0;
    virtual void                       download(DeviceBuffer src, void* host, std::size_t bytes)        = 0;

private:
    std::atomic<bool> _served{true};
};

/// @brief CPU-only DeviceContext: heap allocation, memcpy transfers, no GPU.
struct DeviceContextCpu final : DeviceContext {
    [[nodiscard]] DeviceBackend backend() const noexcept override { return DeviceBackend::CPU_Fallback; }
    [[nodiscard]] DeviceType    deviceType() const noexcept override { return DeviceType::CPU; }
    [[nodiscard]] std::string   shortName() const override { return "CPU"; }
    [[nodiscard]] std::string   name() const override { return "CPU fallback"; }
    [[nodiscard]] std::string   version() const override { return "host"; }

    void wait() override {}

    static constexpr std::size_t kAlign = alignof(std::max_align_t);

    // invalid-on-failure rather than null-on-failure, so every backend shares one contract
    [[nodiscard]] DeviceBuffer allocate(std::size_t bytes, std::size_t align, Residency wanted) override {
        if (wanted != Residency::host && wanted != Residency::shared) {
            return {}; // a CPU-only context has no true device memory
        }
        if (align > kAlign) {
            return {}; // DeviceBuffer has no align field to hand back to operator delete: cannot over-align
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

    void upload(const void* host, DeviceBuffer dst, std::size_t bytes) override { std::memcpy(reinterpret_cast<void*>(dst.token), host, bytes); }
    void download(DeviceBuffer src, void* host, std::size_t bytes) override { std::memcpy(host, reinterpret_cast<void*>(src.token), bytes); }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_CONTEXT_HPP
