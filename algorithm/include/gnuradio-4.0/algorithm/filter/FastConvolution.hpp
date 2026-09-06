#ifndef GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP
#define GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP

#include <bit>
#include <cstddef>
#include <span>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>

namespace gr::algorithm::filter {

/**
 * @brief Overlap-save convolution: transform once, multiply, transform back.
 *
 * A direct filter costs one multiply-add per tap per sample; this costs a transform per frame regardless of
 * how many taps there are, so it overtakes the direct form once the filter is long. The frame carries the
 * previous frame's tail, and the first `nTaps - 1` outputs of each frame are the wrap-around the tail exists
 * to make correct -- they are discarded, which is what makes it overlap-*save*.
 *
 * The transform is supplied by the caller, so the same routine serves a host FFT and a device one.
 */
template<typename T>
requires std::floating_point<T>
struct FastConvolution {
    using Complex = std::complex<T>;

    /// smallest power of two that leaves at least one useful output per frame
    [[nodiscard]] static constexpr std::size_t frameSizeFor(std::size_t nTaps, std::size_t wantedOutputs) noexcept { return std::bit_ceil(nTaps + wantedOutputs - 1UZ); }

    [[nodiscard]] static constexpr std::size_t outputsPerFrame(std::size_t frameSize, std::size_t nTaps) noexcept { return frameSize + 1UZ - nTaps; }

    /// the taps, transformed once and reused for every frame
    [[nodiscard]] static std::vector<Complex> transformTaps(std::span<const T> taps, std::size_t frameSize) {
        std::vector<Complex> spectrum(frameSize, Complex{});
        for (std::size_t k = 0UZ; k < taps.size(); ++k) {
            spectrum[k] = Complex{taps[k], T{}};
        }
        gr::algorithm::FFT<Complex, Complex> transform;
        std::vector<Complex>                 out(frameSize);
        transform.compute(std::span<const Complex>(spectrum), out);
        return out;
    }

    /// one frame in, `outputsPerFrame` samples out; `frame` spans `frameSize` inputs ending at the newest
    static void convolveFrame(std::span<const T> frame, std::span<const Complex> tapSpectrum, std::span<T> output) {
        const std::size_t    frameSize = tapSpectrum.size();
        std::vector<Complex> work(frameSize, Complex{});
        for (std::size_t i = 0UZ; i < frameSize && i < frame.size(); ++i) {
            work[i] = Complex{frame[i], T{}};
        }

        gr::algorithm::FFT<Complex, Complex> transform;
        std::vector<Complex>                 spectrum(frameSize);
        transform.compute(std::span<const Complex>(work), spectrum);
        for (std::size_t k = 0UZ; k < frameSize; ++k) {
            spectrum[k] *= tapSpectrum[k];
        }

        // inverse by conjugation, so one forward transform serves both directions
        for (Complex& value : spectrum) {
            value = std::conj(value);
        }
        std::vector<Complex> cyclic(frameSize);
        transform.compute(std::span<const Complex>(spectrum), cyclic);
        const T scale = T{1} / static_cast<T>(frameSize);

        const std::size_t discard = frameSize - output.size(); // the wrap-around the saved tail exists to make right
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            output[n] = std::conj(cyclic[discard + n]).real() * scale;
        }
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP
