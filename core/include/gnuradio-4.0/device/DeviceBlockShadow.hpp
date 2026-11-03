#ifndef GNURADIO_DEVICE_BLOCK_SHADOW_HPP
#define GNURADIO_DEVICE_BLOCK_SHADOW_HPP

#include <cstddef>
#include <cstdint>

#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::device {

/**
 * The device-resident copy of a block, kept across work() calls.
 *
 * A kernel body is a member function, so `this` must live in device-visible memory. The shadow outlives a single
 * dispatch and is refreshed when the settings epoch moves on, or rebuilt for a different context.
 */
struct DeviceBlockShadow {
    DeviceBuffer                   mirror{};
    DeviceBuffer                   control{}; // see controlArea()
    std::size_t                    controlBytes    = 0UZ;
    DeviceContext*                 context         = nullptr;
    std::uint64_t                  epoch           = kNeverRefreshed;
    bool                           workInFlight    = false; // a dispatch returned before its kernel finished reading `mirror`
    static constexpr std::uint64_t kNeverRefreshed = static_cast<std::uint64_t>(-1);

    DeviceBlockShadow()                                    = default;
    DeviceBlockShadow(const DeviceBlockShadow&)            = delete;
    DeviceBlockShadow& operator=(const DeviceBlockShadow&) = delete;
    DeviceBlockShadow(DeviceBlockShadow&& other) noexcept : mirror(other.mirror), control(other.control), controlBytes(other.controlBytes), context(other.context), epoch(other.epoch), workInFlight(other.workInFlight) {
        other.mirror       = {};
        other.control      = {};
        other.controlBytes = 0UZ;
    }
    DeviceBlockShadow& operator=(DeviceBlockShadow&&) = delete;
    ~DeviceBlockShadow() { release(); }

    /// the kernel reads `mirror` directly, so anything that frees or rewrites it must first let that kernel finish
    void awaitWorkInFlight() noexcept {
        if (workInFlight && context != nullptr) {
            context->wait();
            workInFlight = false;
        }
    }

    void release() noexcept {
        awaitWorkInFlight();
        if (context != nullptr) {
            if (mirror) {
                context->deallocate(mirror);
            }
            if (control) {
                context->deallocate(control);
            }
        }
        mirror       = {};
        control      = {};
        controlBytes = 0UZ;
        context      = nullptr;
        epoch        = kNeverRefreshed;
    }

    /// The status word, per-port accounting and tag slots a framework dispatch hands its kernel. Allocating them per
    /// `work()` call costs a `sycl::free` each time, which is an implicit device synchronisation on CUDA and so
    /// undoes a deferred launch; and a deferred kernel is still writing them when the dispatch returns, so they
    /// cannot be freed there anyway. Grown, never shrunk, released with the shadow. Every region is padded to
    /// `max_align_t` so the caller can carve it without per-region alignment arithmetic.
    [[nodiscard]] std::byte* controlArea(DeviceContext& target, std::size_t bytes) noexcept {
        if (context != &target) { // every buffer is freed through the context that allocated it
            release();
            context = &target;
        }
        if (controlBytes < bytes) {
            awaitWorkInFlight();
            if (control) {
                context->deallocate(control);
            }
            control      = target.allocate(bytes, alignof(std::max_align_t), Residency::shared);
            controlBytes = control ? bytes : 0UZ;
        }
        return control.devicePointer<std::byte>();
    }

    /// (re)acquire device memory when the block is dispatched to a different context; invalid on allocation failure
    [[nodiscard]] DeviceBuffer acquire(DeviceContext& target, std::size_t bytes, std::size_t alignment) noexcept {
        if (context != &target) {
            release();
            context = &target;
        }
        if (!mirror) { // the context alone does not say the memory is still there: controlArea() may have let it go
            mirror = target.allocate(bytes, alignment, Residency::shared);
        }
        return mirror;
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_BLOCK_SHADOW_HPP
