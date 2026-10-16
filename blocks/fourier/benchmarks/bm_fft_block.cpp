#include <benchmark.hpp>

#include <algorithm>
#include <numbers>

#include <format>

#include <gnuradio-4.0/DataSet.hpp>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/fourier/fft.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

/// This custom implementation of FFT is done only for performance comparison with default FFTW implementation.
/**
 * Fast Fourier-Transform according to Cooley–Tukey
 * reference: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Pseudocode
 */
template<typename T>
requires gr::meta::complex_like<T>
void computeFFTCooleyTukey(std::vector<T>& signal) {
    const std::size_t N{signal.size()};

    if (N == 1) {
        return;
    }

    std::vector<T> even(N / 2);
    std::vector<T> odd(N / 2);
    for (std::size_t i = 0; i < N / 2; i++) {
        even[i] = signal[2 * i];
        odd[i]  = signal[2 * i + 1];
    }

    computeFFTCooleyTukey(even);
    computeFFTCooleyTukey(odd);

    const typename T::value_type wn{static_cast<typename T::value_type>(2. * std::numbers::pi_v<double>) / static_cast<typename T::value_type>(N)};
    for (std::size_t i = 0; i < N / 2; i++) {
        const T wkn(std::cos(wn * static_cast<typename T::value_type>(i)), std::sin(wn * static_cast<typename T::value_type>(i)));
        signal[i]         = even[i] + wkn * odd[i];
        signal[i + N / 2] = even[i] - wkn * odd[i];
    }
}

template<typename T>
std::vector<T> generateSinSample(std::size_t N, double sampleRate, double frequency, double amplitude) {
    std::vector<T> signal(N);
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
    using type = T::value_type;
};

// one input/output port pair, wired once and reused: re-wiring per repetition would benchmark port setup rather
// than the transform
template<typename TInput, typename TOutput>
struct FftHarness {
    gr::PortOut<TInput>  srcOut;
    gr::PortIn<TInput>   fftIn{};
    gr::PortOut<TOutput> fftOutPort;
    gr::PortIn<TOutput>  sinkIn{};

    FftHarness() {
        boost::ut::expect(srcOut.connect(fftIn).has_value());
        boost::ut::expect(fftOutPort.connect(sinkIn).has_value());
    }

    void run(gr::blocks::fft::FFT<TInput, TOutput>& fftBlock, const std::vector<TInput>& signal) {
        {
            auto wspan = srcOut.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(signal.size());
            std::ranges::copy(signal, wspan.begin());
            wspan.publish(signal.size());
        }
        auto inSpan = fftIn.template get<gr::SpanReleasePolicy::ProcessAll>(signal.size());
        {
            auto outSpan = fftOutPort.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            boost::ut::expect(gr::work::Status::OK == fftBlock.processBulk(inSpan, outSpan));
        } // scope so the WriterSpan's destructor publishes before the read below
        auto readBack = sinkIn.template get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
        boost::ut::expect(boost::ut::eq(readBack.size(), 1UZ));
    }
};

template<typename T>
void testFFT() {
    using namespace benchmark;
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using namespace gr;
    using namespace gr::algorithm;

    constexpr gr::Size_t N{1024U}; // must be power of 2
    constexpr double     sampleRate{256.};
    constexpr double     frequency{100.};
    constexpr double     amplitude{1.};
    constexpr int        nRepetitions{100};

    using PrecisionType = FFTAlgoPrecision<T>::type;

    static_assert(std::has_single_bit(N));

    std::vector<T> signal = generateSinSample<T>(N, sampleRate, frequency, amplitude);

    {
        gr::blocks::fft::FFT<T, DataSet<PrecisionType>> fft1({{"fft_size", N}});
        fft1.init(fft1.progress);

        FftHarness<T, DataSet<PrecisionType>> harness;
        ::benchmark::benchmark<nRepetitions>(std::format("{} - fft", type_name<T>())) = [&fft1, &harness, &signal] { harness.run(fft1, signal); };
    }

    if constexpr (gr::meta::complex_like<T>) {
        ::benchmark::benchmark<nRepetitions>(std::format("{} - fftCT", type_name<T>())) = [&signal] {
            auto signalCopy = signal;
            computeFFTCooleyTukey<T>(signalCopy);
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

inline const boost::ut::suite _raw_window_bm_tests = [] {
    using namespace benchmark;
    static constexpr gr::Size_t kSamples = 1U << 22U;
    const auto                  runWith  = [](std::string window) {
        gr::Graph graph;
        auto&     source = graph.emplaceBlock<gr::testing::CountingSource<std::complex<float>>>({{"n_samples_max", kSamples}});
        auto&     fft    = graph.emplaceBlock<gr::blocks::fft::FFT<float>>({{"fft_size", gr::Size_t(4096)}, {"window", std::move(window)}});
        auto&     sink   = graph.emplaceBlock<gr::testing::CountingSink<std::complex<float>>>();
        std::ignore      = graph.connect<"out", "in">(source, fft);
        std::ignore      = graph.connect<"out", "in">(fft, sink);
        gr::scheduler::Simple<> scheduler;
        std::ignore = scheduler.exchange(std::move(graph));
        std::ignore = scheduler.runAndWait();
    };
    "raw FFT<float> 4096, no window"_benchmark.repeat<5>(kSamples)   = [&] { runWith("None"); };
    "raw FFT<float> 4096, Hann window"_benchmark.repeat<5>(kSamples) = [&] { runWith("Hann"); };

    static std::vector<std::complex<float>> frames(kSamples, {1.f, -1.f});
    static std::vector<float>               hann(4096UZ);
    gr::algorithm::window::create(std::span<float>(hann), gr::algorithm::window::Type::Hann);
    "control: Hann window multiply alone, 4096-sample frames"_benchmark.repeat<5>(kSamples) = [] {
        for (std::size_t frame = 0UZ; frame < frames.size(); frame += hann.size()) {
            gr::algorithm::fft::applyWindow(std::span(frames).subspan(frame, hann.size()), std::span<const float>(hann));
        }
        force_to_memory(frames);
    };
};

int main() { /* not needed by the UT framework */ }
