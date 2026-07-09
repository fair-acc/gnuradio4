#ifndef GNURADIO_DEVICE_BACKEND_COMPAT_HPP
#define GNURADIO_DEVICE_BACKEND_COMPAT_HPP

#include <cstddef>
#include <cstring>

#include <gnuradio-4.0/device/BackendDetect.hpp>

namespace gr::device {

#if GR_DEVICE_HAS_SYCL_IMPL
using SyclQueue = sycl::queue;
#else
struct NullSyclEvent {
    void wait() const noexcept {}
};

struct SyclQueue {
    void wait() const noexcept {}

    NullSyclEvent memcpy(void* dst, const void* src, std::size_t bytes) const {
        std::memcpy(dst, src, bytes);
        return {};
    }
};
#endif

using GlslProgram                                = unsigned int;
inline constexpr GlslProgram kInvalidGlslProgram = 0U;

} // namespace gr::device

#endif // GNURADIO_DEVICE_BACKEND_COMPAT_HPP
