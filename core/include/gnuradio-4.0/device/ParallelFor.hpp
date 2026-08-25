#ifndef GNURADIO_DEVICE_PARALLEL_FOR_HPP
#define GNURADIO_DEVICE_PARALLEL_FOR_HPP

#include <cstddef>

#include <gnuradio-4.0/device/BackendPolicy.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>

namespace gr::device {

/**
 * backend-neutral data-parallel launch over a `DeviceContext`.
 *
 * `gr::backend_policy` says whether a backend can launch and how; the chain of backends to try is enumerated
 * below, so adding one means specialising that trait AND adding a branch here. A context no policy claims
 * runs the kernel as a host loop rather than silently doing nothing.
 */
template<typename TKernel>
void parallelFor(DeviceContext& context, std::size_t count, TKernel kernel, bool await = true) {
    if constexpr (backend_policy<DeviceContextSycl>::has_parallel_for) {
        if (auto* sycl = backendCast<DeviceContextSycl>(context); sycl != nullptr) {
            backend_policy<DeviceContextSycl>::parallelFor(*sycl, count, kernel, await);
            return;
        }
    }

    for (std::size_t i = 0UZ; i < count; ++i) { // a host loop has already finished by the time it returns
        kernel(i);
    }
    if (await) {
        context.wait();
    }
}

} // namespace gr::device

#endif // GNURADIO_DEVICE_PARALLEL_FOR_HPP
