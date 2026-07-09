#ifndef GNURADIO_BACKEND_DETECT_HPP
#define GNURADIO_BACKEND_DETECT_HPP

#if __has_include(<sycl/sycl.hpp>) && defined(__ACPP__)
#include <sycl/sycl.hpp>
#define GR_DEVICE_HAS_SYCL_IMPL 1
#else
#define GR_DEVICE_HAS_SYCL_IMPL 0
#endif

// detected here rather than in GlComputeContext.hpp so the Block seam can test for a GLSL backend
// without pulling the EGL/GL headers into every translation unit that logs or processes samples
#if __has_include(<EGL/egl.h>) && __has_include(<GL/gl.h>)
#define GR_DEVICE_HAS_GL_COMPUTE 1
#else
#define GR_DEVICE_HAS_GL_COMPUTE 0
#endif

#define GR_DEVICE_HAS_ANY_BACKEND (GR_DEVICE_HAS_SYCL_IMPL || GR_DEVICE_HAS_GL_COMPUTE)

namespace gr::device {

inline constexpr bool kHasSycl   = GR_DEVICE_HAS_SYCL_IMPL;
inline constexpr bool kHasGlsl   = GR_DEVICE_HAS_GL_COMPUTE;
inline constexpr bool kHasCuda   = false;
inline constexpr bool kHasRocm   = false;
inline constexpr bool kHasWebGpu = false;

enum class DeviceBackend { SYCL, GLSL, CUDA, ROCm, WebGPU, CPU_Fallback };
enum class DeviceType { CPU, GPU, FPGA, Accelerator };

} // namespace gr::device

#endif // GNURADIO_BACKEND_DETECT_HPP
