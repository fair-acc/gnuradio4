#ifndef GNURADIO_ALGORITHM_FFT_HPP
#define GNURADIO_ALGORITHM_FFT_HPP

#include <ranges>

#include "window.hpp"

namespace gr::algorithm {

template<typename TInput, typename TOutput = std::conditional<gr::meta::complex_like<TInput>, TInput, std::complex<typename TInput::value_type>>>
    requires((gr::meta::complex_like<TInput> || std::floating_point<TInput>) && (gr::meta::complex_like<TOutput>) )
struct FFT {
    using Precision = TOutput::value_type;

    std::vector<TOutput> twiddleFactors{};
    std::size_t          fftSize{ 0 };

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

    auto
    compute(const std::ranges::input_range auto &in, std::ranges::output_range<TOutput> auto &&out) {
        if constexpr (requires(std::size_t n) { out.resize(n); }) {
            if (out.size() != in.size()) {
                out.resize(in.size());
            }
        } else {
            static_assert(std::tuple_size_v<decltype(in)> == std::tuple_size_v<decltype(out)>, "Size mismatch for fixed-size container.");
        }

        if (!std::has_single_bit(in.size())) {
            throw std::invalid_argument(fmt::format("Input data must have 2^N samples, input size: ", in.size()));
        }
        if (fftSize != in.size()) {
            fftSize = in.size();
            initAll();
        }

        // For the moment no optimization for real data inputs, just create complex with zero imaginary value.
        if constexpr (!gr::meta::complex_like<TInput>) {
            std::ranges::transform(in.begin(), in.end(), out.begin(), [](const auto c) { return TOutput(c, 0.); });
        } else {
            // precision is defined by output type, if cast is needed, let `std::copy` do the job
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

        return out;
    }

    auto
    compute(const std::ranges::input_range auto &in) {
        return compute(in, std::vector<TOutput>(in.size()));
    }

private:
    void
    bitReversalPermutation(std::vector<TOutput> &vec) const noexcept {
        for (std::size_t j = 0, rev = 0; j < fftSize; j++) {
            if (j < rev) std::swap(vec[j], vec[rev]);
            auto maskLen = static_cast<std::size_t>(std::countr_zero(j + 1) + 1);
            rev ^= fftSize - (fftSize >> maskLen);
        }
    }

    void
    precomputeTwiddleFactors() {
        twiddleFactors.clear();
        const auto minus2Pi = Precision(-2. * std::numbers::pi);
        for (std::size_t s = 2; s <= fftSize; s *= 2) {
            const std::size_t m{ s / 2 };
            const TOutput     w{ std::exp(TOutput(0., minus2Pi / static_cast<Precision>(s))) };
            for (std::size_t k = 0; k < fftSize; k += s) {
                TOutput wk{ 1., 0. };
                for (std::size_t j = 0; j < m; j++) {
                    twiddleFactors.push_back(wk);
                    wk *= w;
                }
            }
        }
    }
};

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_FFT_HPP
