#include "../test/blocklib/core/fft/fft.hpp"
#include "benchmark.hpp"
#include "dataset.hpp"
#include <algorithm/fft/fft.hpp>
#include <algorithm/fft/fftw.hpp>
#include <fmt/format.h>
#include <numbers>

/// This custom implementation of FFT is done only for performance comparison with default FFTW implementation.
/**
 * Fast Fourier-Transform according to Cooley–Tukey
 * reference: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Pseudocode
 */
template<typename T>
    requires gr::algorithm::ComplexType<T>
void
computeFFFTCooleyTukey(std::vector<T> &signal) {
    const std::size_t N{ signal.size() };

    if (N == 1) return;

    std::vector<T> even(N / 2);
    std::vector<T> odd(N / 2);
    for (std::size_t i = 0; i < N / 2; i++) {
        even[i] = signal[2 * i];
        odd[i]  = signal[2 * i + 1];
    }

    computeFFFTCooleyTukey(even);
    computeFFFTCooleyTukey(odd);

    const typename T::value_type wn{ static_cast<typename T::value_type>(2. * std::numbers::pi_v<double>) / static_cast<typename T::value_type>(N) };
    for (std::size_t i = 0; i < N / 2; i++) {
        const T wkn(std::cos(wn * static_cast<typename T::value_type>(i)), std::sin(wn * static_cast<typename T::value_type>(i)));
        signal[i]         = even[i] + wkn * odd[i];
        signal[i + N / 2] = even[i] - wkn * odd[i];
    }
}

template<typename T>
std::vector<T>
generateSinSample(std::size_t N, double sampleRate, double frequency, double amplitude) {
    std::vector<T> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        if constexpr (gr::algorithm::ComplexType<T>) {
            signal[i] = { static_cast<typename T::value_type>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sampleRate)), 0. };
        } else {
            signal[i] = static_cast<T>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sampleRate));
        }
    }
    return signal;
}

template<typename T>
void
testFFT() {
    using namespace benchmark;
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using namespace fair::graph;
    using namespace gr::algorithm;

    constexpr std::uint32_t N{ 1024U }; // must be power of 2
    constexpr double      sampleRate{ 256. };
    constexpr double      frequency{ 100. };
    constexpr double      amplitude{ 1. };
    constexpr int         nRepetitions{ 100 };

    using PrecisionType = FFTAlgoPrecision<T>::type;

    static_assert(std::has_single_bit(N));

    std::vector<T> signal = generateSinSample<T>(N, sampleRate, frequency, amplitude);

    {
        gr::blocks::fft::FFT<T, DataSet<PrecisionType>, FFTw<FFTInDataType<T, PrecisionType>>> fft1({ { "fftSize", N } });
        std::ignore                                                                    = fft1.settings().apply_staged_parameters();

        std::vector<DataSet<PrecisionType>> resultingDataSets(1);
        ::benchmark::benchmark<nRepetitions>(fmt::format("{} - fftw", type_name<T>())) = [&fft1, &signal, &resultingDataSets] {
            expect(fair::graph::work_return_status_t::OK == fft1.process_bulk(signal, resultingDataSets));
        };
    }
    {
        gr::blocks::fft::FFT<T, DataSet<PrecisionType>, FFT<FFTInDataType<T, PrecisionType>>> fft1({ { "fftSize", N } });
        std::ignore                                                                   = fft1.settings().apply_staged_parameters();

        std::vector<DataSet<PrecisionType>> resultingDataSets(1);
        ::benchmark::benchmark<nRepetitions>(fmt::format("{} - fft", type_name<T>())) = [&fft1, &signal, &resultingDataSets] {
            expect(fair::graph::work_return_status_t::OK == fft1.process_bulk(signal, resultingDataSets));
        };
    }

    if constexpr (gr::algorithm::ComplexType<T>) {
        ::benchmark::benchmark<nRepetitions>(fmt::format("{} - fftCT", type_name<T>())) = [&signal] {
            auto signalCopy = signal;
            computeFFFTCooleyTukey<T>(signalCopy);
        };
    }

    ::benchmark::results::add_separator();
}

inline const boost::ut::suite _fft_bm_tests = [] {
    std::tuple<std::complex<float>, std::complex<double>> complexTypesToTest{};
    std::tuple<float, double>                             realTypesToTest{};

    std::apply([]<class... TArgs>(TArgs... /*args*/) { (testFFT<TArgs>(), ...); }, complexTypesToTest);
    std::apply([]<class... TArgs>(TArgs... /*args*/) { (testFFT<TArgs>(), ...); }, realTypesToTest);
};

int
main() { /* not needed by the UT framework */
}