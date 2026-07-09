#ifndef GNURADIO_DEVICE_BLOCK_SHADOW_HPP
#define GNURADIO_DEVICE_BLOCK_SHADOW_HPP

#include <cstddef>
#include <cstdint>

#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::device {

/**
 * @brief The device-resident copy of a block, kept across work() calls.
 *
 * A kernel body is a member function, so `this` must live in device-visible memory. The shadow outlives a single
 * dispatch and is refreshed when the settings epoch moves on, or rebuilt for a different context.
 */
struct DeviceBlockShadow {
    DeviceBuffer                   mirror{};
    DeviceContext*                 context         = nullptr;
    std::uint64_t                  epoch           = kNeverRefreshed;
    static constexpr std::uint64_t kNeverRefreshed = static_cast<std::uint64_t>(-1);

    DeviceBlockShadow()                                    = default;
    DeviceBlockShadow(const DeviceBlockShadow&)            = delete;
    DeviceBlockShadow& operator=(const DeviceBlockShadow&) = delete;
    DeviceBlockShadow(DeviceBlockShadow&& other) noexcept : mirror(other.mirror), context(other.context), epoch(other.epoch) { other.mirror = {}; }
    DeviceBlockShadow& operator=(DeviceBlockShadow&&) = delete;
    ~DeviceBlockShadow() { release(); }

    void release() noexcept {
        if (mirror && context != nullptr) {
            context->deallocate(mirror);
        }
        mirror  = {};
        context = nullptr;
        epoch   = kNeverRefreshed;
    }

    /// (re)acquire device memory when the block is dispatched to a different context; invalid on allocation failure
    [[nodiscard]] DeviceBuffer acquire(DeviceContext& target, std::size_t bytes, std::size_t alignment) noexcept {
        if (context != &target) {
            release();
            context = &target;
            mirror  = target.allocate(bytes, alignment, Residency::shared);
        }
        return mirror;
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_BLOCK_SHADOW_HPP
