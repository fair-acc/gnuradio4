#include "benchmark.hpp"

#include "../test/blocklib/core/fft/fft.hpp"

#include <fmt/format.h>
#include <numbers>

/// This custom implementation of FFT ("compute_fft_v1" and "compute_fft_v1") is done only for performance comparison with default FFTW implementation.

/**
 * Real-valued fast fourier transform algorithms
 * H.V. Sorensen, D.L. Jones, M.T. Heideman, C.S. Burrus (1987),
 * in: IEEE Trans on Acoustics, Speech, & Signal Processing, 35
 */

template<typename T>
    requires gr::blocks::fft::ComplexType<T>
void
compute_fft_v1(std::vector<T> &signal) {
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
compute_fft_v2(std::vector<T> &signal) {
    const std::size_t N{ signal.size() };

    if (N == 1) return;

    std::vector<T> even(N / 2);
    std::vector<T> odd(N / 2);
    for (std::size_t i = 0; i < N / 2; i++) {
        even[i] = signal[2 * i];
        odd[i]  = signal[2 * i + 1];
    }

    compute_fft_v2(even);
    compute_fft_v2(odd);

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
compute_magnitude_spectrum(std::vector<T> &fft_signal) {
    const std::size_t                   N{ fft_signal.size() };
    std::vector<typename T::value_type> magnitude_spectrum(N / 2);
    for (std::size_t i = 0; i < N / 2; i++) {
        magnitude_spectrum[i] = std::hypot(fft_signal[i].real(), fft_signal[i].imag()) * static_cast<typename T::value_type>(2.) / static_cast<typename T::value_type>(N);
    }
    return magnitude_spectrum;
}

template<typename T>
std::vector<T>
generate_sin_sample(std::size_t N, double sample_rate, double frequency, double amplitude) {
    std::vector<T> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        if constexpr (gr::blocks::fft::ComplexType<T>) {
            signal[i] = { static_cast<typename T::value_type>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate)), 0. };
        } else {
            signal[i] = static_cast<T>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate));
        }
    }
    return signal;
}

template<typename T>
void
test_fft(bool with_mag_spectrum) {
    using namespace benchmark;
    using namespace boost::ut::reflection;

    constexpr std::size_t N{ 1024 }; // must be power of 2
    constexpr double      sample_rate{ 256. };
    constexpr double      frequency{ 100. };
    constexpr double      amplitude{ 1. };
    constexpr int         n_repetitions{ 100 };

    static_assert(std::has_single_bit(N));

    std::vector<T> signal   = generate_sin_sample<T>(N, sample_rate, frequency, amplitude);
    std::string    name_opt = with_mag_spectrum ? "fft+mag" : "fft";

    {
        gr::blocks::fft::fft<T> fft1{};
        std::ignore = fft1.settings().set({ { "fft_size", N } });
        std::ignore = fft1.settings().apply_staged_parameters();
        fft1.inputHistory.push_back_bulk(signal.begin(), signal.end());

        ::benchmark::benchmark<n_repetitions>(fmt::format("{} - {} fftw", name_opt, type_name<T>())) = [&fft1, &with_mag_spectrum] {
            fft1.prepare_input();
            fft1.compute_fft();
            if (with_mag_spectrum) fft1.compute_magnitude_spectrum();
        };
    }

    if constexpr (gr::blocks::fft::ComplexType<T>) {
        ::benchmark::benchmark<n_repetitions>(fmt::format("{} - {} fft_v1", name_opt, type_name<T>())) = [&signal, &with_mag_spectrum] {
            auto signal_copy = signal;
            compute_fft_v1<T>(signal_copy);
            if (with_mag_spectrum) [[maybe_unused]]
                auto magnitude_spectrum = compute_magnitude_spectrum<T>(signal_copy);
        };

        ::benchmark::benchmark<n_repetitions>(fmt::format("{} - {} fft_v2", name_opt, type_name<T>())) = [&signal, &with_mag_spectrum] {
            auto signal_copy = signal;
            compute_fft_v2<T>(signal_copy);
            if (with_mag_spectrum) [[maybe_unused]]
                auto magnitude_spectrum = compute_magnitude_spectrum<T>(signal_copy);
        };
    }
}

inline const boost::ut::suite _fft_bm_tests = [] {
    std::tuple<std::complex<float>, std::complex<double>> complex_types_to_test{};
    std::tuple<float, double>                             real_types_to_test{};

    std::apply([]<class... TArgs>(TArgs... /*args*/) { (test_fft<TArgs>(false), ...); }, complex_types_to_test);
    benchmark::results::add_separator();
    std::apply([]<class... TArgs>(TArgs... /*args*/) { (test_fft<TArgs>(true), ...); }, complex_types_to_test);
    benchmark::results::add_separator();
    std::apply([]<class... TArgs>(TArgs... /*args*/) { (test_fft<TArgs>(false), ...); }, real_types_to_test);
    benchmark::results::add_separator();
    std::apply([]<class... TArgs>(TArgs... /*args*/) { (test_fft<TArgs>(true), ...); }, real_types_to_test);
};

int
main() { /* not needed by the UT framework */
}