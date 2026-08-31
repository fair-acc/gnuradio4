#ifndef GNURADIO_BACKEND_DETECT_HPP
#define GNURADIO_BACKEND_DETECT_HPP

#if __has_include(<sycl/sycl.hpp>) && defined(__ACPP__)
#include <sycl/sycl.hpp>
#define GR_DEVICE_HAS_SYCL_IMPL 1
#else
#define GR_DEVICE_HAS_SYCL_IMPL 0
#endif

#include <cstddef>
#include <functional>

#include <cstring>

#define GR_DEVICE_HAS_ANY_BACKEND GR_DEVICE_HAS_SYCL_IMPL

namespace gr::device {

inline constexpr bool kHasSycl = GR_DEVICE_HAS_SYCL_IMPL;
inline constexpr bool kHasCuda = false;
inline constexpr bool kHasRocm = false;

// CUDA and ROCm are declared but unserved: both are pointer-based like SYCL, so they reuse the residency model
// rather than needing a separate one.
enum class DeviceBackend { SYCL, CUDA, ROCm, CPU_Fallback };
enum class DeviceType { CPU, GPU, FPGA, Accelerator };

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

    // a SYCL queue is a reference-counted handle and hashes as one; without SYCL there is a single host queue,
    // so every copy names it and syclContextFor() keys them all to one context
    bool operator==(const SyclQueue&) const noexcept { return true; }
};
#endif

} // namespace gr::device

#if !GR_DEVICE_HAS_SYCL_IMPL
template<>
struct std::hash<gr::device::SyclQueue> {
    std::size_t operator()(const gr::device::SyclQueue&) const noexcept { return 0UZ; }
};
#endif

#endif // GNURADIO_BACKEND_DETECT_HPP
