#ifndef GNURADIO_ALGORITHM_FFTPF_HPP
#define GNURADIO_ALGORITHM_FFTPF_HPP

#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#if defined(__GNUC__) && !defined(__clang__) && !defined(__EMSCRIPTEN__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <pffft.hpp>
#if defined(__GNUC__) && !defined(__clang__) && !defined(__EMSCRIPTEN__)
#pragma GCC diagnostic pop
#endif
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <complex>
#include <format>
#include <ranges>
#include <stdexcept>

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::algorithm {

template<typename TInput, typename TOutput = std::conditional<gr::meta::complex_like<TInput>, TInput, std::complex<typename TInput::value_type>>>
requires((gr::meta::complex_like<TInput> || std::floating_point<TInput>) && (gr::meta::complex_like<TOutput>))
struct FFTpf {
    using Precision = typename TOutput::value_type;
    using Complex   = std::complex<Precision>;

    std::size_t                   fftSize{32UZ};
    pffft::Fft<TInput>            fftImpl{static_cast<int>(fftSize), 8192};
    pffft::AlignedVector<TInput>  alignedInput;
    pffft::AlignedVector<Complex> alignedOutput;

    FFTpf()                            = default;
    FFTpf(const FFTpf&)                = delete;
    FFTpf(FFTpf&&) noexcept            = delete;
    FFTpf& operator=(const FFTpf&)     = delete;
    FFTpf& operator=(FFTpf&&) noexcept = delete;
    ~FFTpf()                           = default;

    void initAll() {
        fftImpl.prepareLength(static_cast<int>(fftSize));
        if (!fftImpl.isValid()) {
            throw std::domain_error(std::format("fft size {} not supported by pffft.", fftSize));
        }
        alignedInput.resize(static_cast<std::size_t>(fftImpl.getLength()));
        alignedOutput.resize(static_cast<std::size_t>(fftImpl.getSpectrumSize()));
    }

    auto compute(const std::ranges::input_range auto& in, std::ranges::output_range<TOutput> auto&& out) {
        if (fftSize != in.size()) {
            fftSize = in.size();
            initAll();
        }
        if (in.size() != alignedInput.size()) {
            alignedInput.resize(in.size());
            alignedOutput.resize(static_cast<std::size_t>(fftImpl.getSpectrumSize()));
        }
        if (out.size() != alignedOutput.size()) {
            if constexpr (requires { out.resize(fftSize); }) {
                out.resize(alignedOutput.size());
            } else {
                throw std::out_of_range(std::format("Output vector size ({}) is not enough, at least {} needed. ", out.size(), fftSize));
            }
        }

        // copy input data to aligned input vector
        if constexpr (gr::meta::complex_like<TInput>) {
            std::ranges::transform(in, alignedInput.begin(), [](const auto& c) { return Complex(static_cast<Precision>(c.real()), static_cast<Precision>(c.imag())); });
        } else {
            std::ranges::transform(in, alignedInput.begin(), [](const auto& c) { return Precision(c); });
        }

        fftImpl.forward(alignedInput, alignedOutput);

        assert(out.size() >= alignedOutput.size());
        std::ranges::transform(alignedOutput, out.begin(), [](const Complex& c) { //
            return TOutput{static_cast<typename TOutput::value_type>(c.real()), static_cast<typename TOutput::value_type>(c.imag())};
        });

        return out;
    }

    auto compute(const std::ranges::input_range auto& in) { return compute(in, std::vector<TOutput>(in.size())); }
};

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_FFTPF_HPP
