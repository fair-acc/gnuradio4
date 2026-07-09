#ifndef GNURADIO_DEVICE_PARALLEL_FOR_HPP
#define GNURADIO_DEVICE_PARALLEL_FOR_HPP

#include <cstddef>

#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>

namespace gr::device {

/**
 * @brief backend-neutral data-parallel launch over a `DeviceContext`.
 *
 * The kernel stays a compile-time type on purpose: SYCL forbids function pointers in device code, so
 * a launch cannot be a virtual on `DeviceContext`. Backend selection is therefore the RTTI-free
 * `backend()` enum check plus a `static_cast`, and contexts without a device path run the kernel as a
 * host loop rather than silently doing nothing.
 */
template<typename TKernel>
void parallelFor(DeviceContext& context, std::size_t count, TKernel kernel) {
    if (context.backend() == DeviceBackend::SYCL) {
        static_cast<DeviceContextSycl&>(context).parallelFor(count, kernel); // backend() pre-checked; no RTTI
        return;
    }

    for (std::size_t i = 0UZ; i < count; ++i) {
        kernel(i);
    }
    context.wait();
}

} // namespace gr::device

#endif // GNURADIO_DEVICE_PARALLEL_FOR_HPP
