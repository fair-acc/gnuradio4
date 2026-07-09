#ifndef GNURADIO_META_DEVICE_ANNOTATIONS_HPP
#define GNURADIO_META_DEVICE_ANNOTATIONS_HPP

/**
 * @brief How a function or lambda says it may run on a device.
 *
 * SYCL single-source emits device code for anything a kernel reaches, so under AdaptiveCpp both macros are
 * empty and cost nothing. CUDA and HIP do not: they require every device-reachable function to carry
 * `__host__ __device__`, and every kernel lambda `__device__`. Spelling that through these macros now means a
 * second backend is a build change at the call sites rather than an edit of each one.
 *
 * Lives in `meta` rather than `device` so the algorithm layer can annotate its per-item helpers without
 * depending on core.
 *
 * ```cpp
 * GR_DEVICE_FN constexpr T scaleOne(T v, T g) noexcept { return v * g; }
 *
 * gr::device::parallelFor(ctx, n, [d, g] GR_DEVICE_LAMBDA(std::size_t i) { d[i] = scaleOne(d[i], g); });
 * ```
 */

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GR_DEVICE_FN     __host__ __device__
#define GR_DEVICE_LAMBDA __device__
#else
#define GR_DEVICE_FN
#define GR_DEVICE_LAMBDA
#endif

#endif // GNURADIO_META_DEVICE_ANNOTATIONS_HPP
