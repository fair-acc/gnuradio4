#ifndef GNURADIO_BACKEND_DETECT_HPP
#define GNURADIO_BACKEND_DETECT_HPP

#if __has_include(<sycl/sycl.hpp>) && defined(__ACPP__)
#include <sycl/sycl.hpp>
#define GR_DEVICE_HAS_SYCL_IMPL 1
#else
#define GR_DEVICE_HAS_SYCL_IMPL 0
#endif

namespace gr::device {

inline constexpr bool kHasSycl   = GR_DEVICE_HAS_SYCL_IMPL;
inline constexpr bool kHasCuda   = false;
inline constexpr bool kHasRocm   = false;
inline constexpr bool kHasWebGpu = false;

enum class DeviceBackend { SYCL, GLSL, CUDA, ROCm, WebGPU, CPU_Fallback };
enum class DeviceType { CPU, GPU, FPGA, Accelerator };

} // namespace gr::device

#endif // GNURADIO_BACKEND_DETECT_HPP
