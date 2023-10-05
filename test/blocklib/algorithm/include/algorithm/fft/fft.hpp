#ifndef GRAPH_PROTOTYPE_ALGORITHM_FFT_HPP
#define GRAPH_PROTOTYPE_ALGORITHM_FFT_HPP

#include "block.hpp"
#include "fft_types.hpp"
#include "window.hpp"
#include <ranges>

namespace gr::algorithm {

template<typename T>
    requires(ComplexType<T> || std::floating_point<T>)
struct FFT {
    using PrecisionType = FFTAlgoPrecision<T>::type;
    using OutDataType   = std::conditional_t<ComplexType<T>, T, std::complex<T>>;

    std::vector<OutDataType> twiddleFactors{};
    std::size_t              fftSize{ 0 };

    FFT()                   = default;
    FFT(const FFT &rhs)     = delete;
    FFT(FFT &&rhs) noexcept = delete;
    FFT &
    operator=(const FFT &rhs)
            = delete;
    FFT &
    operator=(FFT &&rhs) noexcept
            = delete;

    ~FFT() = default;

    void
    initAll() {
        precomputeTwiddleFactors();
    }

    std::vector<OutDataType>
    computeFFT(const std::vector<T> &in) {
        std::vector<OutDataType> out(in.size());
        computeFFT(in, out);
        return out;
    }

    void
    computeFFT(const std::vector<T> &in, std::vector<OutDataType> &out) {
        if (!std::has_single_bit(in.size())) {
            throw std::invalid_argument(fmt::format("Input data must have 2^N samples, input size: ", in.size()));
        }
        if (fftSize != in.size()) {
            fftSize = in.size();
            initAll();
        }

        // For the moment no optimization for real data inputs, just create complex with zero imaginary value.
        if constexpr (!ComplexType<T>) {
            std::ranges::transform(in.begin(), in.end(), out.begin(), [](const auto c) { return OutDataType(c, 0.); });
        } else {
            std::ranges::copy(in.begin(), in.end(), out.begin());
        }

        /**
         * Real-valued fast fourier transform algorithms
         * H.V. Sorensen, D.L. Jones, M.T. Heideman, C.S. Burrus (1987),
         * in: IEEE Trans on Acoustics, Speech, & Signal Processing, 35
         */
        bitReversalPermutation(out);

        std::size_t omega_kCounter = 0;
        for (std::size_t s = 2; s <= fftSize; s *= 2) {
            const auto half_s = s / 2;
            for (std::size_t k = 0; k < fftSize; k += s) {
                for (std::size_t j = 0; j < half_s; j++) {
                    const auto t{ twiddleFactors[omega_kCounter++] * out[k + j + half_s] };
                    const auto u{ out[k + j] };
                    out[k + j]          = u + t;
                    out[k + j + half_s] = u - t;
                }
            }
        }
    }

private:
    void
    bitReversalPermutation(std::vector<OutDataType> &vec) const noexcept {
        for (std::size_t j = 0, rev = 0; j < fftSize; j++) {
            if (j < rev) std::swap(vec[j], vec[rev]);
            auto maskLen = static_cast<std::size_t>(std::countr_zero(j + 1) + 1);
            rev ^= fftSize - (fftSize >> maskLen);
        }
    }

    void
    precomputeTwiddleFactors() {
        twiddleFactors.clear();
        const auto minus2Pi = PrecisionType(-2. * std::numbers::pi);
        for (std::size_t s = 2; s <= fftSize; s *= 2) {
            const std::size_t m{ s / 2 };
            const OutDataType w{ std::exp(OutDataType(0., minus2Pi / static_cast<PrecisionType>(s))) };
            for (std::size_t k = 0; k < fftSize; k += s) {
                OutDataType wk{ 1., 0. };
                for (std::size_t j = 0; j < m; j++) {
                    twiddleFactors.push_back(wk);
                    wk *= w;
                }
            }
        }
    }
};

} // namespace gr::algorithm

#endif // GRAPH_PROTOTYPE_ALGORITHM_FFT_HPP
