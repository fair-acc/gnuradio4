#ifndef GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP
#define GNURADIO_ALGORITHM_FAST_CONVOLUTION_HPP

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/fourier/SimdFFT.hpp>
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
 *
 * T. G. Stockham, "High-speed convolution and correlation", in Proc. AFIPS Spring Joint Computer Conf., vol. 28,
 * 1966, pp. 229-233, for computing convolution through the transform. The sectioning below is standard practice
 * and is described in the prose above rather than attributed.
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

    /// real-valued transform: `SimdFFT` packs `N` reals, bins 0 and N/2 in the first two slots, the rest interleaved
    using RealFft = gr::algorithm::SimdFFT<T, gr::algorithm::Transform::Real>;
    using Aligned = std::vector<T, gr::allocator::Aligned<T, 64UZ>>;

    std::optional<RealFft> _transform;
    std::vector<Complex>   _tapNatural;
    Aligned                _tapPacked;
    Aligned                _frameSamples;
    Aligned                _spectrum;
    Aligned                _cyclic;

    /// the taps in the layout `convolveFrame` multiplies in, transformed once per design.
    ///
    /// Unordered is faster and admits more frame sizes -- the ordered real transform also demands a multiple of
    /// `2 * L * L` -- but it leaves the bins permuted, and a pointwise product survives that only if both sides
    /// carry the SAME permutation. Hence the taps go through this transform rather than arriving natural-order.
    void prepareTaps(std::span<const T> taps, std::size_t frameSize) {
        assert(!taps.empty() && taps.size() <= frameSize && "the taps must fit the frame they are transformed into");
        _tapNatural = transformTaps(taps, frameSize); // the complex fallback below multiplies in natural order
        if (!RealFft::canProcessSize(frameSize, gr::algorithm::Order::Unordered)) {
            _tapPacked.clear();
            return;
        }
        resizeFor(frameSize);
        std::fill(_frameSamples.begin(), _frameSamples.end(), T{});
        std::copy_n(taps.begin(), taps.size(), _frameSamples.begin());
        _tapPacked.assign(frameSize, T{});
        _transform->template transform<gr::algorithm::Direction::Forward, gr::algorithm::Order::Unordered>(_frameSamples, _tapPacked);
    }

    /// one frame in, `outputsPerFrame` samples out; `frame` spans `frameSize` inputs ending at the newest.
    /// `prepareTaps` must have run for this frame size.
    void convolveFrame(std::span<const T> frame, std::span<T> output) {
        const std::size_t frameSize = _tapNatural.size();
        // a short frame or an oversized output is a caller error, not something to paper over: zero-padding the
        // one gives a wrong answer silently, and the other makes `discard` below wrap
        assert(frameSize > 0UZ && "prepareTaps must run before a frame is convolved");
        assert(frame.size() >= frameSize && "the frame must carry a whole transform's worth of input");
        assert(output.size() <= frameSize && "a frame cannot yield more outputs than it has points");

        if (_tapPacked.size() != frameSize) {
            convolveFrameAsComplex(frame, _tapNatural, output); // a frame this transform cannot factor still has to filter
            return;
        }
        resizeFor(frameSize);

        const std::size_t nGiven = std::min(frameSize, frame.size());
        std::copy_n(frame.begin(), nGiven, _frameSamples.begin());
        std::fill(_frameSamples.begin() + static_cast<std::ptrdiff_t>(nGiven), _frameSamples.end(), T{}); // a short frame pads with silence

        _transform->template transform<gr::algorithm::Direction::Forward, gr::algorithm::Order::Unordered>(_frameSamples, _spectrum);

        // the unordered layout runs in blocks of 2L floats holding [Re(k..k+L-1), Im(k..k+L-1)], so a bin's halves
        // sit L apart, not side by side. Which bin lands where does not matter -- both sides share the permutation.
        // Bins 0 and N/2 are real and pair at the front: DC at 0, Nyquist where Im(DC) would be.
        constexpr std::size_t kLanes = RealFft::simdSize();
        _spectrum[0] *= _tapPacked[0];
        _spectrum[kLanes] *= _tapPacked[kLanes];
        const auto multiplyBin = [this](std::size_t re, std::size_t im) {
            const T signalRe = _spectrum[re];
            const T signalIm = _spectrum[im];
            const T tapRe    = _tapPacked[re];
            const T tapIm    = _tapPacked[im];
            _spectrum[re]    = signalRe * tapRe - signalIm * tapIm;
            _spectrum[im]    = signalRe * tapIm + signalIm * tapRe;
        };
        for (std::size_t lane = 1UZ; lane < kLanes; ++lane) { // the first block, less the pair handled above
            multiplyBin(lane, kLanes + lane);
        }
        for (std::size_t block = 2UZ * kLanes; block < frameSize; block += 2UZ * kLanes) {
            for (std::size_t lane = 0UZ; lane < kLanes; ++lane) {
                multiplyBin(block + lane, block + kLanes + lane);
            }
        }

        _transform->template transform<gr::algorithm::Direction::Backward, gr::algorithm::Order::Unordered>(_spectrum, _cyclic);

        const T           scale   = T{1} / static_cast<T>(frameSize);
        const std::size_t discard = frameSize - output.size();
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            output[n] = _cyclic[discard + n] * scale;
        }
    }

private:
    void resizeFor(std::size_t frameSize) {
        if (_frameSamples.size() == frameSize) {
            return;
        }
        _transform.emplace(frameSize);
        _frameSamples.assign(frameSize, T{});
        _spectrum.assign(frameSize, T{});
        _cyclic.assign(frameSize, T{});
    }

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
