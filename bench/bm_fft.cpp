#include "benchmark.hpp"

#include "../test/blocklib/core/fft/fft.hpp"

#include <fmt/format.h>
#include <numbers>

/// This custom implementation of FFT ("compute_fft_v1" and "computeFftV1") is done only for performance comparison with default FFTW implementation.

/**
 * Real-valued fast fourier transform algorithms
 * H.V. Sorensen, D.L. Jones, M.T. Heideman, C.S. Burrus (1987),
 * in: IEEE Trans on Acoustics, Speech, & Signal Processing, 35
 */

template<typename T>
    requires gr::blocks::fft::ComplexType<T>
void
computeFftV1(std::vector<T> &signal) {
    const std::size_t N{ signal.size() };
    for (std::size_t j = 0, rev = 0; j < N; j++) {
        if (j < rev) std::swap(signal[j], signal[rev]);
        auto maskLen = static_cast<std::size_t>(std::countr_zero(j + 1) + 1);
        rev ^= N - (N >> maskLen);
    }

    for (std::size_t s = 2; s <= N; s *= 2) {
        const std::size_t m{ s / 2 };
        const T           w{ exp(T(0., static_cast<typename T::value_type>(-2. * std::numbers::pi) / static_cast<typename T::value_type>(s))) };
        for (std::size_t k = 0; k < N; k += s) {
            T wk{ 1., 0. };
            for (std::size_t j = 0; j < m; j++) {
                const T t{ wk * signal[k + j + m] };
                const T u{ signal[k + j] };
                signal[k + j]     = u + t;
                signal[k + j + m] = u - t;
                wk *= w;
            }
        }
    }
}

/**
 * Fast Fourier-Transform according to Cooley–Tukey
 * reference: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Pseudocode
 */
template<typename T>
    requires gr::blocks::fft::ComplexType<T>
void
computeFftV2(std::vector<T> &signal) {
    const std::size_t N{ signal.size() };

    if (N == 1) return;

    std::vector<T> even(N / 2);
    std::vector<T> odd(N / 2);
    for (std::size_t i = 0; i < N / 2; i++) {
        even[i] = signal[2 * i];
        odd[i]  = signal[2 * i + 1];
    }

    computeFftV2(even);
    computeFftV2(odd);

    const typename T::value_type wn{ static_cast<typename T::value_type>(2. * std::numbers::pi) / static_cast<typename T::value_type>(N) };
    for (std::size_t i = 0; i < N / 2; i++) {
        const T wkn(std::cos(wn * static_cast<typename T::value_type>(i)), std::sin(wn * static_cast<typename T::value_type>(i)));
        signal[i]         = even[i] + wkn * odd[i];
        signal[i + N / 2] = even[i] - wkn * odd[i];
    }
}

template<typename T>
    requires gr::blocks::fft::ComplexType<T>
std::vector<typename T::value_type>
computeMagnitudeSpectrum(std::vector<T> &fftSignal) {
    const std::size_t                   N{ fftSignal.size() };
    std::vector<typename T::value_type> magnitudeSpectrum(N / 2);
    for (std::size_t i = 0; i < N / 2; i++) {
        magnitudeSpectrum[i] = std::hypot(fftSignal[i].real(), fftSignal[i].imag()) * static_cast<typename T::value_type>(2.) / static_cast<typename T::value_type>(N);
    }
    return magnitudeSpectrum;
}

template<typename T>
std::vector<T>
generateSinSample(std::size_t N, double sampleRate, double frequency, double amplitude) {
    std::vector<T> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        if constexpr (gr::blocks::fft::ComplexType<T>) {
            signal[i] = { static_cast<typename T::value_type>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sampleRate)), 0. };
        } else {
            signal[i] = static_cast<T>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sampleRate));
        }
    }
    return signal;
}

template<typename T>
void
testFft(bool withMagSpectrum) {
    using namespace benchmark;
    using namespace boost::ut::reflection;

    constexpr std::size_t N{ 1024 }; // must be power of 2
    constexpr double      sampleRate{ 256. };
    constexpr double      frequency{ 100. };
    constexpr double      amplitude{ 1. };
    constexpr int         nRepetitions{ 100 };

    static_assert(std::has_single_bit(N));

    std::vector<T> signal  = generateSinSample<T>(N, sampleRate, frequency, amplitude);
    std::string    nameOpt = withMagSpectrum ? "fft+mag" : "fft";

    {
        gr::blocks::fft::fft<T> fft1{};
        std::ignore = fft1.settings().set({ { "fftSize", N } });
        std::ignore = fft1.settings().apply_staged_parameters();
        fft1.inputHistory.push_back_bulk(signal.begin(), signal.end());

        ::benchmark::benchmark<nRepetitions>(fmt::format("{} - {} fftw", nameOpt, type_name<T>())) = [&fft1, &withMagSpectrum] {
            fft1.prepareInput();
            fft1.computeFft();
            if (withMagSpectrum) fft1.computeMagnitudeSpectrum();
        };
    }

    if constexpr (gr::blocks::fft::ComplexType<T>) {
        ::benchmark::benchmark<nRepetitions>(fmt::format("{} - {} fft_v1", nameOpt, type_name<T>())) = [&signal, &withMagSpectrum] {
            auto signalCopy = signal;
            computeFftV1<T>(signalCopy);
            if (withMagSpectrum) [[maybe_unused]]
                auto magnitudeSpectrum = computeMagnitudeSpectrum<T>(signalCopy);
        };

        ::benchmark::benchmark<nRepetitions>(fmt::format("{} - {} fft_v2", nameOpt, type_name<T>())) = [&signal, &withMagSpectrum] {
            auto signalCopy = signal;
            computeFftV2<T>(signalCopy);
            if (withMagSpectrum) [[maybe_unused]]
                auto magnitudeSpectrum = computeMagnitudeSpectrum<T>(signalCopy);
        };
    }
}

inline const boost::ut::suite _fft_bm_tests = [] {
    std::tuple<std::complex<float>, std::complex<double>> complexTypesToTest{};
    std::tuple<float, double>                             realTypesToTest{};

    std::apply([]<class... TArgs>(TArgs... /*args*/) { (testFft<TArgs>(false), ...); }, complexTypesToTest);
    benchmark::results::add_separator();
    std::apply([]<class... TArgs>(TArgs... /*args*/) { (testFft<TArgs>(true), ...); }, complexTypesToTest);
    benchmark::results::add_separator();
    std::apply([]<class... TArgs>(TArgs... /*args*/) { (testFft<TArgs>(false), ...); }, realTypesToTest);
    benchmark::results::add_separator();
    std::apply([]<class... TArgs>(TArgs... /*args*/) { (testFft<TArgs>(true), ...); }, realTypesToTest);
};

int
main() { /* not needed by the UT framework */
}