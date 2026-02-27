#ifndef GNURADIO_CACHE_LINE_SIZE_HPP
#define GNURADIO_CACHE_LINE_SIZE_HPP

#include <cstddef>
#include <new>

namespace gr {

/**
 * Cache line size constant for `alignas` on performance-critical types (false sharing
 * padding, true sharing packing).
 *
 * The priority chain below is needed because the C++ standard constant
 * `std::hardware_destructive_interference_size` is unreliable on several targets:
 *
 *  1. A compile-time `-DGR_CACHE_LINE_SIZE=<N>` override takes precedence, allowing
 *     cross-compilation or non-standard platforms to set the correct value.
 *
 *  2. Apple ARM64 (M1–M4) uses 128-byte L2 cache lines, but Clang 19+ reports 64
 *     via `std::hardware_destructive_interference_size`. This is a known compiler bug:
 *       - https://github.com/llvm/llvm-project/issues/182951  (Apple ARM64 value wrong)
 *       - https://github.com/llvm/llvm-project/issues/60174   (original feature request)
 *       - https://github.com/llvm/llvm-project/pull/89446     (added __GCC_*_SIZE macros)
 *     The Apple check must therefore come before the `__cpp_lib_hardware_interference_size`
 *     test, because that macro IS defined on Clang 19+ but yields the wrong value.
 *
 *  3. When the standard constant is available and not affected by the bugs above, use it.
 *
 *  4. Fall back to 64 bytes (correct for x86-64, conservative for most other targets,
 *     covers Emscripten which does not define the standard constant).
 */
#if defined(GR_CACHE_LINE_SIZE)
inline constexpr std::size_t kCacheLine = GR_CACHE_LINE_SIZE;
#elif defined(__APPLE__) && defined(__aarch64__)
inline constexpr std::size_t kCacheLine = 128UZ;
#elif defined(__cpp_lib_hardware_interference_size)
inline constexpr std::size_t kCacheLine = std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t kCacheLine = 64UZ;
#endif

} // namespace gr

#endif // GNURADIO_CACHE_LINE_SIZE_HPP
