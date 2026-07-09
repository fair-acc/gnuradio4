#ifndef GNURADIO_BM_FFT_BACKENDS_HELPERS_HPP
#define GNURADIO_BM_FFT_BACKENDS_HELPERS_HPP
// separate TU + SHARED library under acpp — see G10 in claude_wip.md (AdaptiveCpp#2042)
#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gr::benchmark_fft {

using C = std::complex<float>;

struct FFTBackend {
    std::string shortName;
    // init for a given FFT size (called once before timed section, triggers JIT warmup)
    std::function<void(std::size_t N, std::size_t maxBatches)> init;
    // H2D + compute + D2H for nBatches FFTs of size N; returns first output element
    std::function<float(const C* input, std::size_t N, std::size_t nBatches)> compute;
};

// returns all available device-backed FFT backends (CPU SYCL, GPU SYCL, GPU GLSL, ...)
// each backend manages its own DeviceContext, SyclFFT/GlslFFT, and device buffers
std::vector<FFTBackend> availableBackends();

// device info string for printing (one line per device)
std::string deviceInfo();

} // namespace gr::benchmark_fft

#endif // GNURADIO_BM_FFT_BACKENDS_HELPERS_HPP
