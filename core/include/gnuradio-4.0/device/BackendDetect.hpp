#ifndef GNURADIO_BACKEND_DETECT_HPP
#define GNURADIO_BACKEND_DETECT_HPP

#if __has_include(<sycl/sycl.hpp>) && (defined(__ACPP__) || defined(SYCL_LANGUAGE_VERSION))
#include <sycl/sycl.hpp>
#define GR_DEVICE_HAS_SYCL 1
#else
#define GR_DEVICE_HAS_SYCL 0
#endif

// The CUDA RUNTIME api is plain C, so a `DeviceContext` over it -- allocation, transfer, synchronisation --
// compiles with the ordinary C++ compiler and needs neither `nvcc` nor `enable_language(CUDA)`. Gated on the
// build asking for it as well as the header being present, because the `libcudart` link dependency is opt-in.
#if defined(GR_ENABLE_CUDA) && __has_include(<cuda_runtime.h>)
#define GR_DEVICE_HAS_CUDA 1
#else
#define GR_DEVICE_HAS_CUDA 0
#endif

// ROCm through HIP, whose runtime API mirrors CUDA's name for name; verified on ROCm 6.4 (RX 7700S, gfx1102).
#if defined(GR_ENABLE_ROCM) && __has_include(<hip/hip_runtime.h>)
#define GR_DEVICE_HAS_ROCM 1
#else
#define GR_DEVICE_HAS_ROCM 0
#endif

#include <cstddef>
#include <functional>

#include <cstring>

namespace gr::device {

/// `GR_DEVICE_HAS_SYCL` is the ONE preprocessor conditional this codebase needs: only the preprocessor can answer
/// `__has_include`, and only a macro can keep `sycl::`-typed code out of a translation unit that has no SYCL. Use it
/// nowhere else -- here in the backend abstraction, and in user code or a test that writes a SYCL-specific body.
/// Everywhere else ask `kHasDeviceBackend` in an `if constexpr`: the device stack is written to compile whether or
/// not a backend is present, so the question is one of behaviour, not of what will parse.
inline constexpr bool kHasSycl = GR_DEVICE_HAS_SYCL;

/// whether a CUDA `DeviceContext` can be constructed: memory and transfers only, NOT kernel launch
inline constexpr bool kHasCuda = GR_DEVICE_HAS_CUDA;

/// the same for ROCm/HIP
inline constexpr bool kHasRocm = GR_DEVICE_HAS_ROCM;

/// whether any device backend is compiled in -- what a block, a graph or a scheduler actually wants to know.
///
/// Deliberately NOT `kHasSycl || kHasCuda`: this answers "can a block's kernel be dispatched", and the CUDA
/// context carries memory and transfers but has no launch path, so a graph that believed otherwise would hand
/// kernels to a context that cannot run them. CUDA joins this only when `parallelFor` can reach it.
inline constexpr bool kHasDeviceBackend = kHasSycl;

// CUDA and ROCm are declared but unserved: both are pointer-based like SYCL, so they reuse the residency model
// rather than needing a separate one.
enum class DeviceBackend { SYCL, CUDA, ROCm, CPU_Fallback };
enum class DeviceType { CPU, GPU, Accelerator };

#if GR_DEVICE_HAS_SYCL
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

#if !GR_DEVICE_HAS_SYCL
template<>
struct std::hash<gr::device::SyclQueue> {
    std::size_t operator()(const gr::device::SyclQueue&) const noexcept { return 0UZ; }
};
#endif

#endif // GNURADIO_BACKEND_DETECT_HPP
