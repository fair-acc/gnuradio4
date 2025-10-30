#include "gnuradio-4.0/meta/UnitTestHelper.hpp"

#include <benchmark.hpp>

#include <numbers>

#include <format>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#ifndef __EMSCRIPTEN__
#include <gnuradio-4.0/algorithm/fourier/fftpf.hpp>
#endif
#include <gnuradio-4.0/algorithm/fourier/fftw.hpp>

template<typename T>
std::vector<T, gr::allocator::Aligned<T>> generateSinSample(std::size_t N, double sampleRate, double frequency, double amplitude) {
    std::vector<T, gr::allocator::Aligned<T>> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        if constexpr (gr::meta::complex_like<T>) {
            signal[i] = {static_cast<typename T::value_type>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sampleRate)), 0.};
        } else {
            signal[i] = static_cast<T>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sampleRate));
        }
    }
    return signal;
}

template<typename T>
struct FFTAlgoPrecision {
    using type = T;
};

template<gr::meta::complex_like T>
struct FFTAlgoPrecision<T> {
    using type = typename T::value_type;
};

template<typename T, template<typename...> typename Algorithm>
void testFFT() {
    using namespace benchmark;
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using namespace gr;
    using namespace gr::algorithm;

    using PrecisionType = std::remove_cv_t<typename FFTAlgoPrecision<T>::type>;
    using OutputType    = std::complex<PrecisionType>;
    using FFTAlgo       = Algorithm<T, OutputType>;

    for (std::size_t N : {512UZ, 1024UZ, 8192UZ, 65536UZ, 1009UZ /* prime */}) {
        constexpr int         nRepetitions{1000};
        constexpr std::size_t binnedFrequency{5UZ};

        auto              templateName = [](std::string_view typeName) { return std::string(typeName.substr(0, typeName.find('<'))); };
        const std::string algoName     = templateName(gr::meta::type_name<FFTAlgo>());
        const std::string bmName       = std::format("{:21} - {:22} N = {:5}", algoName, type_name<T>(), N);
        const std::size_t scaling      = static_cast<std::size_t>(static_cast<double>(N) * std::log(static_cast<double>(N)));

        try {
            std::vector<T, gr::allocator::Aligned<T>> signal = generateSinSample<T>(N, 1., static_cast<double>(binnedFrequency) / static_cast<double>(N), 1.);
            FFTAlgo                                   fft;
            auto                                      output = fft.compute(signal); // warm up cache/twiddlefactors

            ::benchmark::benchmark<nRepetitions>(std::string_view(bmName), scaling) = [&fft, &signal, &output] { output = fft.compute(signal); };

            using namespace boost::ut;
            PrecisionType expectedMag = -static_cast<PrecisionType>(N) / static_cast<PrecisionType>(2); // N.B. un-normalised FFT
            expect(approx(output[binnedFrequency].imag(), expectedMag, static_cast<PrecisionType>(.1f))) << bmName << fatal;
        } catch (const std::exception& e) {
            std::println(stderr, "{} : {}", bmName, e.what());
        } catch (...) {
            std::println(stderr, "{} : unknown exception", bmName);
        }
    }

    ::benchmark::results::add_separator();
}

inline const boost::ut::suite<"FFT forward tests"> _fft_bm_tests = [] {
    using namespace gr::algorithm;

    auto testAll = [&]<typename... Ts>(auto /*types*/) { //
        auto testForType = []<typename T>() {
            ([]<template<typename, typename> typename Algo>() { testFFT<T, Algo>(); }.template operator()<FFTw>(), //
#ifndef __EMSCRIPTEN__
                []<template<typename, typename> typename Algo>() { testFFT<T, Algo>(); }.template operator()<FFTpf>(), //
#endif
                []<template<typename, typename> typename Algo>() { testFFT<T, Algo>(); }.template operator()<FFT>()); //
        };
        (testForType.template operator()<Ts>(), ...);
    };
    testAll.operator()<std::complex<float>, std::complex<double>, float, double>(0);
    std::println("N.B. ops/s values are scaled with N*log(N).");
};

int main() { /* not needed by the UT framework */ }
