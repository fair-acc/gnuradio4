#ifndef GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP
#define GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP

#include <algorithm>
#include <bit>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/fourier/SimdFFT.hpp>
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
 *
 * T. G. Stockham, "High-speed convolution and correlation", in Proc. AFIPS Spring Joint Computer Conf., vol. 28,
 * 1966, pp. 229-233, for computing convolution through the transform. The sectioning below is standard practice
 * and is described in the prose above rather than attributed.
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

    /// real-valued transform: `SimdFFT` packs `N` reals, bins 0 and N/2 in the first two slots, the rest interleaved
    using RealFft = gr::algorithm::SimdFFT<T, gr::algorithm::Transform::Real>;
    using Aligned = std::vector<T, gr::allocator::Aligned<T, 64UZ>>;

    std::optional<RealFft> _transform;
    Aligned                _frameSamples;
    Aligned                _spectrum;
    Aligned                _cyclic;

    /// one frame in, `outputsPerFrame` samples out; `frame` spans `frameSize` inputs ending at the newest.
    /// `tapSpectrum` is the complex spectrum `transformTaps` produced -- only its first `frameSize / 2 + 1` bins are
    /// read, the rest being the conjugate half the real transform never computes.
    void convolveFrame(std::span<const T> frame, std::span<const Complex> tapSpectrum, std::span<T> output) {
        const std::size_t frameSize = tapSpectrum.size();
        if (!RealFft::canProcessSize(frameSize, gr::algorithm::Order::Ordered)) {
            convolveFrameAsComplex(frame, tapSpectrum, output); // a frame this transform cannot factor still has to filter
            return;
        }
        if (_frameSamples.size() != frameSize) {
            _transform.emplace(frameSize);
            _frameSamples.assign(frameSize, T{});
            _spectrum.assign(frameSize, T{});
            _cyclic.assign(frameSize, T{});
        }

        const std::size_t nGiven = std::min(frameSize, frame.size());
        std::copy_n(frame.begin(), nGiven, _frameSamples.begin());
        std::fill(_frameSamples.begin() + static_cast<std::ptrdiff_t>(nGiven), _frameSamples.end(), T{}); // a short frame pads with silence

        _transform->template transform<gr::algorithm::Direction::Forward, gr::algorithm::Order::Ordered>(_frameSamples, _spectrum);

        const std::size_t nyquist = frameSize / 2UZ;
        _spectrum[0] *= tapSpectrum[0].real();       // bin 0 is real, and so is the tap weighting it
        _spectrum[1] *= tapSpectrum[nyquist].real(); // bin N/2 likewise, which is why it shares the front of the packing
        for (std::size_t k = 1UZ; k < nyquist; ++k) {
            const T        re        = _spectrum[2UZ * k];
            const T        im        = _spectrum[2UZ * k + 1UZ];
            const Complex& tap       = tapSpectrum[k];
            _spectrum[2UZ * k]       = re * tap.real() - im * tap.imag();
            _spectrum[2UZ * k + 1UZ] = re * tap.imag() + im * tap.real();
        }

        _transform->template transform<gr::algorithm::Direction::Backward, gr::algorithm::Order::Ordered>(_spectrum, _cyclic);

        const T           scale   = T{1} / static_cast<T>(frameSize);
        const std::size_t discard = frameSize - output.size(); // the wrap-around the saved tail exists to make right
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            output[n] = _cyclic[discard + n] * scale;
        }
    }

private:
    /// the complex route, for a frame size the real transform cannot factor. Same result, twice the transform work.
    void convolveFrameAsComplex(std::span<const T> frame, std::span<const Complex> tapSpectrum, std::span<T> output) {
        const std::size_t frameSize = tapSpectrum.size();
        if (_work.size() != frameSize) {
            _work.resize(frameSize);
            _product.resize(frameSize);
            _cyclicComplex.resize(frameSize);
        }
        const std::size_t nGiven = std::min(frameSize, frame.size());
        for (std::size_t i = 0UZ; i < nGiven; ++i) {
            _work[i] = Complex{frame[i], T{}};
        }
        std::fill(_work.begin() + static_cast<std::ptrdiff_t>(nGiven), _work.end(), Complex{});

        _complexTransform.compute(std::span<const Complex>(_work), _product);
        for (std::size_t k = 0UZ; k < frameSize; ++k) {
            _product[k] *= tapSpectrum[k];
        }
        for (Complex& value : _product) { // inverse by conjugation, so one forward transform serves both directions
            value = std::conj(value);
        }
        _complexTransform.compute(std::span<const Complex>(_product), _cyclicComplex);

        const T           scale   = T{1} / static_cast<T>(frameSize);
        const std::size_t discard = frameSize - output.size();
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            output[n] = std::conj(_cyclicComplex[discard + n]).real() * scale;
        }
    }

    gr::algorithm::FFT<Complex, Complex> _complexTransform;
    std::vector<Complex>                 _work;
    std::vector<Complex>                 _product;
    std::vector<Complex>                 _cyclicComplex;
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP
