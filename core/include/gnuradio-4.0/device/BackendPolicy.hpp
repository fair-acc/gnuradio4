#ifndef GNURADIO_DEVICE_BACKEND_POLICY_HPP
#define GNURADIO_DEVICE_BACKEND_POLICY_HPP

#include <cstddef>

#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr {

/**
 * @brief How one backend launches a data-parallel kernel.
 *
 * A launch cannot be a virtual on `DeviceContext`: SYCL forbids function pointers in device code, so the kernel
 * has to stay a compile-time type. This trait is where that knowledge lives instead, one specialisation per
 * backend, so adding a backend means writing a specialisation rather than editing every dispatch site.
 *
 * The primary template means "this context has no device launch"; a caller runs the kernel as a host loop.
 */
template<typename TContext>
struct backend_policy {
    static constexpr bool has_parallel_for = false;

    static constexpr device::DeviceBackend backend() noexcept { return device::DeviceBackend::CPU_Fallback; }
};

/// the concrete context behind a type-erased one, or nullptr when this backend is not the one running it.
/// `backend()` is the RTTI-free discriminator the contexts already carry.
template<typename TContext>
[[nodiscard]] TContext* backendCast(device::DeviceContext& context) noexcept {
    return context.backend() == backend_policy<TContext>::backend() ? static_cast<TContext*>(&context) : nullptr;
}

} // namespace gr

#endif // GNURADIO_DEVICE_BACKEND_POLICY_HPP
