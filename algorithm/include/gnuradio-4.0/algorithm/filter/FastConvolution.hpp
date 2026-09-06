#ifndef GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP
#define GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP

#include <bit>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>

namespace gr::algorithm::filter {

/**
 * Overlap-save convolution: transform once, multiply, transform back.
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

    /// where a caller choosing its own domain should switch, in taps. A default rather than a claim -- a machine
    /// or a backend moves it, so name the domain explicitly to override it.
    static constexpr std::size_t kFrequencyDomainFromTaps = 128UZ;

    /// smallest power of two that fits `nTaps - 1` of overlap plus `wantedOutputs` outputs
    [[nodiscard]] static constexpr std::size_t frameSizeFor(std::size_t nTaps, std::size_t wantedOutputs) noexcept {
        assert(nTaps >= 1UZ && wantedOutputs >= 1UZ);
        return std::bit_ceil(nTaps + wantedOutputs - 1UZ);
    }

    [[nodiscard]] static constexpr std::size_t outputsPerFrame(std::size_t frameSize, std::size_t nTaps) noexcept {
        assert(nTaps >= 1UZ && nTaps <= frameSize && "a frame shorter than the filter yields no output, and the count would wrap");
        return frameSize + 1UZ - nTaps;
    }

    /// the taps, transformed once and reused for every frame
    [[nodiscard]] static std::vector<Complex> transformTaps(std::span<const T> taps, std::size_t frameSize) {
        assert(!taps.empty() && taps.size() <= frameSize && "the taps must fit the frame they are transformed into");
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
        const std::size_t frameSize = tapSpectrum.size();
        // a short frame or an oversized output is a caller error, not something to paper over: zero-padding the
        // one gives a wrong answer silently, and the other makes `discard` below wrap
        assert(frame.size() >= frameSize && "the frame must carry a whole transform's worth of input");
        assert(output.size() <= frameSize && "a frame cannot yield more outputs than it has points");

        std::vector<Complex> work(frameSize, Complex{});
        for (std::size_t i = 0UZ; i < frameSize; ++i) {
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

        const std::size_t discard = frameSize - output.size();
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            output[n] = std::conj(cyclic[discard + n]).real() * scale;
        }
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP
