#ifndef GNURADIO_DEVICE_CONTEXT_HPP
#define GNURADIO_DEVICE_CONTEXT_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <string>

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
    std::uintptr_t token      = 0;
    std::size_t    bytes      = 0;
    Residency      residency  = Residency::invalid;
    std::size_t    align      = alignof(std::max_align_t); // as allocated: deallocation needs it back
    std::uintptr_t allocation = 0;                         // what the allocator returned; `token` may be aligned up inside it

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

    [[nodiscard]] virtual DeviceBackend backend() const noexcept    = 0;
    [[nodiscard]] virtual DeviceType    deviceType() const noexcept = 0;
    [[nodiscard]] virtual std::string   shortName() const           = 0; // "CPU", "SYCL:CPU", "SYCL:NVIDIA GeForce RTX 3070"
    [[nodiscard]] virtual std::string   name() const                = 0; // "NVIDIA GeForce RTX 3070"
    [[nodiscard]] virtual std::string   version() const             = 0; // the backend's own driver-version string, e.g. "13030"
    [[nodiscard]] bool                  isGpu() const noexcept { return deviceType() == DeviceType::GPU; }

    /**
     * @brief Typed escape for a body that genuinely needs one backend's own handle.
     *
     * The ordinary body never names a backend. A body that must -- to submit a hand-written kernel, say -- says so
     * here, which makes the dependency greppable instead of hiding it in a signature. Returns nullptr when this
     * context is served by a different backend, which the caller must handle rather than assume.
     */
    template<typename TContext>
    [[nodiscard]] TContext* as() noexcept {
        return backend() == TContext::kBackend ? static_cast<TContext*>(this) : nullptr;
    }

    /// Bulk copy between two regions this context can address. The host does a plain memcpy; a device backend
    /// issues it on its own queue, so device-only memory is reachable without the caller naming a backend.
    /// `await = false` returns as soon as the copy is enqueued: correct only on an in-order queue, and only when a
    /// later `wait()` on the same queue stands between the copy and whoever reads the destination.
    virtual void copy(void* destination, const void* source, std::size_t bytes, bool /*await*/ = true) { std::memcpy(destination, source, bytes); }

    virtual void wait() = 0;

    /// how many times this context has actually synchronised with its device. The cascade's headline property is a
    /// count, not a duration -- "one barrier per group `work()`, not one per block" -- so it is observable rather
    /// than inferred from a stopwatch. Implementations bump it wherever they block. Atomic because a context is
    /// process-global and every scheduler thread dispatching through it bumps the same counter.
    [[nodiscard]] std::uint64_t syncCount() const noexcept { return _syncCount.load(std::memory_order_relaxed); }

protected:
    std::atomic<std::uint64_t> _syncCount{0UZ};

public:
    // true when `ptr` is memory this context's device can dereference directly (USM); conservatively false
    [[nodiscard]] virtual bool isDeviceAccessible(const void* /*ptr*/) const noexcept { return false; }

    /// Stricter than `isDeviceAccessible`, which also accepts shared USM: true only for memory the HOST may not
    /// dereference. That is the question deferred completion turns on -- a dispatch may return before its kernel has
    /// finished only if nothing it touched can be read on this thread in the meantime.
    [[nodiscard]] virtual bool isDeviceOnly(const void* /*ptr*/) const noexcept { return false; }

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
};

/**
 * @brief CPU-only DeviceContext: heap allocation, memcpy transfers, no GPU.
 *
 * The native-host backing, beside `host:sycl` and `gpu:sycl`: what a gcc, clang or emscripten build has, and what
 * code taking a `DeviceContext&` can construct directly when no backend was compiled in. Deliberately NOT registered
 * under `host` -- `tryResolve` must keep reporting that a domain is unserved rather than handing back a CPU context,
 * or a block whose `gpu:sycl` downgraded to `host` would silently dispatch here instead of refusing (`qa_DeviceContext`
 * asserts exactly that).
 */
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

    void upload(const void* host, DeviceBuffer dst, std::size_t bytes) override { std::memcpy(reinterpret_cast<void*>(dst.token), host, bytes); }
    void download(DeviceBuffer src, void* host, std::size_t bytes) override { std::memcpy(host, reinterpret_cast<void*>(src.token), bytes); }
};

/**
 * @brief The CPU context a block is given when the scheduler chose no device for it.
 *
 * Holding one is not a decision to dispatch: the block still gates that on its own compute domain, so a `gpu:sycl`
 * that fell back to the host refuses rather than quietly running its kernel here. That is the same reason
 * `DeviceContextCpu` is kept out of the registry under `host` (`qa_DeviceContext` asserts it), and the reason this
 * is a reference rather than a null pointer: every `work()` call has a context to allocate and copy through, and
 * none of them can mistake having one for being on a device.
 */
[[nodiscard]] inline DeviceContext& hostBackend() noexcept {
    static DeviceContextCpu instance;
    return instance;
}

} // namespace gr::device

#endif // GNURADIO_DEVICE_CONTEXT_HPP
