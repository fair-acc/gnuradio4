#include <boost/ut.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

#include "algorithm/fft/fft.hpp"
#include "algorithm/fft/fftw.hpp"
#include "blocklib/core/fft/fft.hpp"
#include <fmt/format.h>
#include <graph.hpp>
#include <node.hpp>
#include <numbers>
#include <scheduler.hpp>

namespace fg = fair::graph;

template<typename T>
struct CountSource : public fg::node<CountSource<T>> {
    fg::PortOut<T> out{};
    int            count{ 0 };
    int            nSamples{ 1024 };

    constexpr std::make_signed_t<std::size_t>
    available_samples(const CountSource & /*d*/) noexcept {
        const auto ret = nSamples - count;
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    constexpr T
    process_one() {
        return static_cast<T>(count++);
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (CountSource<T>), out, count, nSamples);

template<typename T>
std::vector<T>
generateSinSample(std::size_t N, double sample_rate, double frequency, double amplitude) {
    std::vector<T> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        if constexpr (gr::algorithm::ComplexType<T>) {
            signal[i] = { static_cast<typename T::value_type>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate)), 0. };
        } else {
            signal[i] = static_cast<T>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate));
        }
    }
    return signal;
}

template<typename T, typename U = T>
bool
equalVectors(const std::vector<T> &v1, const std::vector<U> &v2, double tolerance = std::is_same_v<T, double> ? 1.e-5 : 1e-4) {
    if (v1.size() != v2.size()) {
        return false;
    }
    if constexpr (gr::algorithm::ComplexType<T>) {
        return std::equal(v1.begin(), v1.end(), v2.begin(), [&tolerance](const auto &l, const auto &r) {
            return std::abs(l.real() - r.real()) < static_cast<typename T::value_type>(tolerance) && std::abs(l.imag() - r.imag()) < static_cast<typename T::value_type>(tolerance);
        });
    } else {
        return std::equal(v1.begin(), v1.end(), v2.begin(), [&tolerance](const auto &l, const auto &r) { return std::abs(static_cast<double>(l) - static_cast<double>(r)) < tolerance; });
    }
}

template<typename T, typename inT, typename outT, typename pT>
void
testFFTwTypes() {
    using namespace boost::ut;
    gr::algorithm::FFTw<T> fftBlock;
    expect(std::is_same_v<typename std::remove_pointer_t<decltype(fftBlock.fftwIn.get())>, inT>) << "";
    expect(std::is_same_v<typename std::remove_pointer_t<decltype(fftBlock.fftwOut.get())>, outT>) << "";
    expect(std::is_same_v<decltype(fftBlock.fftwPlan.get()), pT>) << "";
}

template<typename T, typename U, typename A>
void
equalDataset(const gr::blocks::fft::FFT<T, fg::DataSet<U>, A> &fftBlock, const fg::DataSet<U> &ds1, float sample_rate) {
    using namespace boost::ut;
    using namespace boost::ut::reflection;

    const U    tolerance = U(0.0001);

    const auto N         = fftBlock._magnitudeSpectrum.size();
    auto const freq      = static_cast<U>(sample_rate) / static_cast<U>(fftBlock.fftSize);
    expect(ge(ds1.signal_values.size(), N)) << fmt::format("<{}> DataSet signal length {} vs. magnitude size {}", type_name<T>(), ds1.signal_values.size(), N);
    if (N == fftBlock.fftSize) { // complex input
        expect(approx(ds1.signal_values[0], -(static_cast<U>(N) / U(2.f)) * freq, tolerance)) << fmt::format("<{}> equal DataSet frequency[0]", type_name<T>());
        expect(approx(ds1.signal_values[N - 1], (static_cast<U>(N) / U(2.f) - U(1.f)) * freq, tolerance)) << fmt::format("<{}> equal DataSet frequency[0]", type_name<T>());
    } else { // real input
        expect(approx(ds1.signal_values[0], 0 * freq, tolerance)) << fmt::format("<{}> equal DataSet frequency[0]", type_name<T>());
        expect(approx(ds1.signal_values[N - 1], (static_cast<U>(N) - U(1.f)) * freq, tolerance)) << fmt::format("<{}> equal DataSet frequency[0]", type_name<T>());
    };
    bool       isEqualFFTOut = true;
    const auto NSize         = static_cast<std::size_t>(N);
    for (std::size_t i = 0U; i < NSize; i++) {
        if (std::abs(ds1.signal_values[i + NSize] - static_cast<U>(fftBlock._outData[i].real())) > tolerance
            || std::abs(ds1.signal_values[i + 2U * NSize] - static_cast<U>(fftBlock._outData[i].imag())) > tolerance) {
            isEqualFFTOut = false;
            break;
        }
    }
    expect(eq(isEqualFFTOut, true)) << fmt::format("<{}> equal DataSet FFT output", type_name<T>());
    expect(equalVectors<U>(std::vector(ds1.signal_values.begin() + static_cast<std::ptrdiff_t>(3U * N), ds1.signal_values.begin() + static_cast<std::ptrdiff_t>(4U * N)), fftBlock._magnitudeSpectrum))
            << fmt::format("<{}> equal DataSet magnitude", type_name<T>());
    expect(equalVectors<U>(std::vector(ds1.signal_values.begin() + static_cast<std::ptrdiff_t>(4U * N), ds1.signal_values.begin() + static_cast<std::ptrdiff_t>(5U * N)), fftBlock._phaseSpectrum))
            << fmt::format("<{}> equal DataSet phase", type_name<T>());

    for (std::size_t i = 0U; i < 5; i++) {
        const auto mm = std::minmax_element(std::next(ds1.signal_values.begin(), static_cast<std::ptrdiff_t>(i * N)), std::next(ds1.signal_values.begin(), static_cast<std::ptrdiff_t>((i + 1U) * N)));
        expect(approx(*mm.first, ds1.signal_ranges[i][0], tolerance));
        expect(approx(*mm.second, ds1.signal_ranges[i][1], tolerance));
    }
}

template<typename T>
using FFTwAlgo = gr::algorithm::FFTw<T>;

template<typename T>
using FFTAlgo = gr::algorithm::FFT<T>;

template<typename T, typename U, template<typename> typename A>
struct TypePair {
    using InType   = T;
    using OutType  = U;
    using AlgoType = A<gr::algorithm::FFTInDataType<T, typename U::value_type>>;
};

const boost::ut::suite fftTests = [] {
    using namespace boost::ut;
    using namespace gr::blocks::fft;
    using namespace boost::ut::reflection;

    std::tuple<TypePair<std::complex<float>, DataSet<float>, FFTwAlgo>, TypePair<std::complex<float>, DataSet<float>, FFTAlgo>, TypePair<std::complex<double>, DataSet<float>, FFTwAlgo>,
               TypePair<std::complex<double>, DataSet<double>, FFTAlgo>>
            complexTypesWithAlgoToTest{};

    std::tuple<TypePair<std::complex<float>, DataSet<float>, FFTwAlgo>, TypePair<std::complex<float>, DataSet<float>, FFTAlgo>, TypePair<std::complex<double>, DataSet<double>, FFTwAlgo>,
               TypePair<std::complex<double>, DataSet<float>, FFTAlgo>, TypePair<float, DataSet<float>, FFTwAlgo>, TypePair<float, DataSet<float>, FFTAlgo>, TypePair<double, DataSet<float>, FFTwAlgo>,
               TypePair<double, DataSet<float>, FFTAlgo>>
                              typesWithAlgoToTest{};

    std::tuple<float, double> floatingTypesToTest{};

    "FFT sin tests"_test = []<typename T>() {
        using InType   = T::InType;
        using OutType  = T::OutType;
        using AlgoType = T::AlgoType;
        FFT<InType, OutType, AlgoType> fftBlock{};
        constexpr double               tolerance{ 1.e-5 };
        struct TestParams {
            std::uint32_t N{ 1024 };           // must be power of 2
            double        sample_rate{ 128. }; // must be power of 2 (only for the unit test for easy comparison with true result)
            double        frequency{ 1. };
            double        amplitude{ 1. };
            bool          outputInDb{ false };
        };

        std::vector<TestParams> testCases = { { 256, 128., 10., 5., false }, { 512, 4., 1., 1., false }, { 512, 32., 1., 0.1, false }, { 256, 128., 10., 5., true } };
        for (const auto &t : testCases) {
            assert(std::has_single_bit(t.N));
            assert(std::has_single_bit(static_cast<std::size_t>(t.sample_rate)));

            std::ignore = fftBlock.settings().set({ { "fftSize", t.N } });
            std::ignore = fftBlock.settings().set({ { "outputInDb", t.outputInDb } });
            std::ignore = fftBlock.settings().set({ { "window", static_cast<int>(gr::algorithm::window::Type::None) } });
            std::ignore = fftBlock.settings().apply_staged_parameters();
            const auto           signal{ generateSinSample<InType>(t.N, t.sample_rate, t.frequency, t.amplitude) };
            std::vector<OutType> resultingDataSets(1);
            expect(fair::graph::work_return_status_t::OK == fftBlock.process_bulk(signal, resultingDataSets));

            const auto peakIndex{
                static_cast<std::size_t>(std::distance(fftBlock._magnitudeSpectrum.begin(),
                                                       std::max_element(fftBlock._magnitudeSpectrum.begin(), std::next(fftBlock._magnitudeSpectrum.begin(), static_cast<std::ptrdiff_t>(t.N / 2u)))))
            }; // only positive frequencies from FFT
            const auto peakAmplitude = fftBlock._magnitudeSpectrum[peakIndex];
            const auto peakFrequency{ static_cast<double>(peakIndex) * t.sample_rate / static_cast<double>(t.N) };

            const auto expectedAmplitude = t.outputInDb ? 20. * log10(std::abs(t.amplitude)) : t.amplitude;
            expect(approx(static_cast<double>(peakAmplitude), expectedAmplitude, tolerance)) << fmt::format("{} equal amplitude", type_name<T>());
            expect(approx(peakFrequency, t.frequency, tolerance)) << fmt::format("{} equal frequency", type_name<T>());
        }
    } | typesWithAlgoToTest;

    "FFT pattern tests"_test = []<typename T>() {
        using InType   = T::InType;
        using OutType  = T::OutType;
        using AlgoType = T::AlgoType;
        constexpr double               tolerance{ 1.e-5 };
        constexpr std::uint32_t        N{ 16 };
        FFT<InType, OutType, AlgoType> fftBlock({ { "fftSize", N }, { "window", static_cast<int>(gr::algorithm::window::Type::None) } });
        std::ignore = fftBlock.settings().apply_staged_parameters();

        std::vector<InType> signal(N);

        static_assert(N == 16, "expected values are calculated for N == 16");
        std::size_t expectedPeakIndex{ 0 };
        InType      expectedFft0{ 0., 0. };
        double      expectedPeakAmplitude{ 0. };
        for (std::size_t iT = 0; iT < 5; iT++) {
            if (iT == 0) {
                std::ranges::fill(signal.begin(), signal.end(), InType(0., 0.));
                expectedFft0          = { 0., 0. };
                expectedPeakAmplitude = 0.;
            } else if (iT == 1) {
                std::ranges::fill(signal.begin(), signal.end(), InType(1., 0.));
                expectedFft0          = { 16., 0. };
                expectedPeakAmplitude = 2.;
            } else if (iT == 2) {
                std::ranges::fill(signal.begin(), signal.end(), InType(1., 1.));
                expectedFft0          = { 16., 16. };
                expectedPeakAmplitude = std::sqrt(8.);
            } else if (iT == 3) {
                std::iota(signal.begin(), signal.end(), 1);
                expectedFft0          = { 136., 0. };
                expectedPeakAmplitude = 17.;
            } else if (iT == 4) {
                int i = 0;
                std::ranges::generate(signal.begin(), signal.end(), [&i] { return InType(static_cast<typename InType::value_type>(i++ % 2), 0.); });
                expectedFft0          = { 8., 0. };
                expectedPeakAmplitude = 1.;
            }
            std::vector<OutType> resultingDataSets(1);
            expect(fair::graph::work_return_status_t::OK == fftBlock.process_bulk(signal, resultingDataSets));

            const auto peakIndex{ static_cast<std::size_t>(std::distance(fftBlock._magnitudeSpectrum.begin(), std::ranges::max_element(fftBlock._magnitudeSpectrum))) };
            const auto peakAmplitude{ fftBlock._magnitudeSpectrum[peakIndex] };

            expect(eq(peakIndex, expectedPeakIndex)) << fmt::format("<{}> equal peak index", type_name<T>());
            expect(approx(static_cast<double>(peakAmplitude), expectedPeakAmplitude, tolerance)) << fmt::format("<{}> equal amplitude", type_name<T>());
            expect(approx(static_cast<double>(fftBlock._outData[0].real()), static_cast<double>(expectedFft0.real()), tolerance)) << fmt::format("<{}> equal fft[0].real()", type_name<T>());
            expect(approx(static_cast<double>(fftBlock._outData[0].imag()), static_cast<double>(expectedFft0.imag()), tolerance)) << fmt::format("<{}> equal fft[0].imag()", type_name<T>());
        }
    } | complexTypesWithAlgoToTest;

    "FFT process_bulk tests"_test = []<typename T>() {
        using InType   = T::InType;
        using OutType  = T::OutType;
        using AlgoType = T::AlgoType;

        constexpr std::uint32_t        N{ 16 };
        constexpr float                sample_rate{ 1.f };
        FFT<InType, OutType, AlgoType> fftBlock({ { "fftSize", N }, { "sample_rate", sample_rate } });
        std::ignore = fftBlock.settings().apply_staged_parameters();
        expect(eq(fftBlock.algorithm, fair::meta::type_name<AlgoType>()));

        std::vector<InType> signal(N);
        std::iota(signal.begin(), signal.end(), 1);
        std::vector<OutType> v{ OutType() };
        std::span<OutType>   outSpan(v);

        expect(fair::graph::work_return_status_t::OK == fftBlock.process_bulk(signal, outSpan));
        equalDataset(fftBlock, v[0], sample_rate);
    } | typesWithAlgoToTest;

    "FFT types tests"_test = [] {
        expect(std::is_same_v<FFT<std::complex<float>>::value_type, float>) << "output type must be float";
        expect(std::is_same_v<FFT<std::complex<double>>::value_type, float>) << "output type must be float";
        expect(std::is_same_v<FFT<float>::value_type, float>) << "output type must be float";
        expect(std::is_same_v<FFT<double>::value_type, float>) << "output type must be float";
        expect(std::is_same_v<FFT<int>::value_type, float>) << "output type must be float";
        expect(std::is_same_v<FFT<std::complex<float>, fg::DataSet<double>>::value_type, double>) << "output type must be double";
        expect(std::is_same_v<FFT<float, fg::DataSet<double>>::value_type, double>) << "output type must be double";
    };

    "FFT fftw types tests"_test = [] {
        testFFTwTypes<std::complex<float>, fftwf_complex, fftwf_complex, fftwf_plan>();
        testFFTwTypes<std::complex<double>, fftw_complex, fftw_complex, fftw_plan>();
        testFFTwTypes<float, float, fftwf_complex, fftwf_plan>();
        testFFTwTypes<double, double, fftw_complex, fftw_plan>();
    };

    "FFT flow graph example"_test = [] {
        // This test checks how fftw works if one creates and destroys several fft blocks in different graph flows
        using namespace boost::ut;
        using Scheduler      = fair::graph::scheduler::simple<>;
        auto      threadPool = std::make_shared<fair::thread_pool::BasicThreadPool>("custom pool", fair::thread_pool::CPU_BOUND, 2, 2);
        fg::graph flow1;
        auto     &source1  = flow1.make_node<CountSource<double>>();
        auto     &fftBlock = flow1.make_node<FFT<double>>({ { "fftSize", static_cast<std::uint32_t>(16) } });
        std::ignore        = flow1.connect<"out">(source1).to<"in">(fftBlock);
        auto sched1        = Scheduler(std::move(flow1), threadPool);

        // run 2 times to check potential memory problems
        for (int i = 0; i < 2; i++) {
            fg::graph flow2;
            auto     &source2 = flow2.make_node<CountSource<double>>();
            auto     &fft2    = flow2.make_node<FFT<double>>({ { "fftSize", static_cast<std::uint32_t>(16) } });
            std::ignore       = flow2.connect<"out">(source2).to<"in">(fft2);
            auto sched2       = Scheduler(std::move(flow2), threadPool);
            sched2.run_and_wait();
            expect(approx(source2.count, source2.nSamples, 1e-4));
        }
        sched1.run_and_wait();
        expect(approx(source1.count, source1.nSamples, 1e-4));
    };

    "FFT window function tests"_test = []<typename T>() {
        using InType   = T::InType;
        using OutType  = T::OutType;
        using AlgoType = T::AlgoType;

        FFT<InType, OutType, AlgoType> fftBlock{};

        using value_type = OutType::value_type;
        constexpr value_type    tolerance{ value_type(0.00001) };

        constexpr std::uint32_t N{ 8 };
        using enum gr::algorithm::window::Type;
        std::vector<gr::algorithm::window::Type> testCases = { None, Rectangular, Hamming, Hann, HannExp, Blackman, Nuttall, BlackmanHarris, BlackmanNuttall, FlatTop, Exponential };

        for (const auto &t : testCases) {
            std::ignore = fftBlock.settings().set({ { "fftSize", N } });
            std::ignore = fftBlock.settings().set({ { "window", static_cast<int>(t) } });
            std::ignore = fftBlock.settings().apply_staged_parameters();

            std::vector<InType> signal(N);
            if constexpr (gr::algorithm::ComplexType<InType>) {
                typename InType::value_type i = 0.;
                std::ranges::generate(signal.begin(), signal.end(), [&i] {
                    i = i + static_cast<typename InType::value_type>(1.);
                    return InType(i, i);
                });
            } else {
                std::iota(signal.begin(), signal.end(), 1.);
            }
            std::vector<OutType> resultingDataSets(1);
            expect(fair::graph::work_return_status_t::OK == fftBlock.process_bulk(signal, resultingDataSets));

            expect(eq(fftBlock.fftSize, N)) << fmt::format("<{}> equal fft size", type_name<T>());
            expect(eq(fftBlock._window.size(), N)) << fmt::format("<{}> equal window vector size", type_name<T>());
            expect(eq(fftBlock.window, static_cast<int>(t))) << fmt::format("<{}> equal window function", type_name<T>());

            std::vector<value_type> windowFunc = gr::algorithm::window::create<value_type>(t, N);
            for (std::size_t i = 0; i < N; i++) {
                if constexpr (gr::algorithm::ComplexType<InType>) {
                    const auto expValue = static_cast<value_type>(signal[i].real()) * windowFunc[i];
                    expect(approx(fftBlock._inData[i].real(), expValue, tolerance)) << fmt::format("<{}> equal fftwIn complex.real", type_name<T>());
                    expect(approx(fftBlock._inData[i].imag(), expValue, tolerance)) << fmt::format("<{}> equal fftwIn complex.imag", type_name<T>());
                } else {
                    const value_type expValue = static_cast<value_type>(signal[i]) * static_cast<value_type>(windowFunc[i]);
                    expect(approx(fftBlock._inData[i], expValue, tolerance)) << fmt::format("<{}> equal fftwIn", type_name<T>());
                }
            }
        }
    } | typesWithAlgoToTest;

    "FFT window tests"_test = []<typename T>() {
        // Expected value for size 8
        std::vector<T> Rectangular8{ 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
        std::vector<T> Hamming8{ 0.07672f, 0.2119312255f, 0.53836f, 0.8647887745f, 1.0f, 0.8647887745f, 0.53836f, 0.2119312255f };
        std::vector<T> Hann8{ 0.f, 0.1882550991f, 0.611260467f, 0.950484434f, 0.950484434f, 0.611260467f, 0.1882550991f, 0.f };
        std::vector<T> Blackman8{ 0.f, 0.09045342435f, 0.4591829575f, 0.9203636181f, 0.9203636181f, 0.4591829575f, 0.09045342435f, 0.f };
        std::vector<T> BlackmanHarris8{ 0.00006f, 0.03339172348f, 0.3328335043f, 0.8893697722f, 0.8893697722f, 0.3328335043f, 0.03339172348f, 0.00006f };
        std::vector<T> BlackmanNuttall8{ 0.0003628f, 0.03777576895f, 0.34272762f, 0.8918518611f, 0.8918518611f, 0.34272762f, 0.03777576895f, 0.0003628f };
        std::vector<T> Exponential8{ 1.f, 1.042546905f, 1.08690405f, 1.133148453f, 1.181360413f, 1.231623642f, 1.284025417f, 1.338656724f };
        std::vector<T> FlatTop8{ 0.004f, -0.1696424054f, 0.04525319348f, 3.622389212f, 3.622389212f, 0.04525319348f, -0.1696424054f, 0.004f };
        std::vector<T> HannExp8{ 0.f, 0.611260467f, 0.950484434f, 0.1882550991f, 0.1882550991f, 0.950484434f, 0.611260467f, 0.f };
        std::vector<T> Nuttall8{ 0.f, 0.0311427368f, 0.3264168059f, 0.8876284573f, 0.8876284573f, 0.3264168059f, 0.0311427368f, 0.f };
        std::vector<T> Kaiser8{ 0.5714348848f, 0.7650986027f, 0.9113132365f, 0.9899091685f, 0.9899091685f, 0.9113132365f, 0.7650986027f, 0.5714348848f };

        // check all windows for unwanted changes
        using gr::algorithm::window::create;
        using enum gr::algorithm::window::Type;
        expect(equalVectors<T>(create<T>(None, 8), Rectangular8)) << fmt::format("<{}> equal Rectangular8[8] vector {} vs. ref: {}", type_name<T>(), create<T>(None, 8), Rectangular8);
        expect(equalVectors<T>(create<T>(Rectangular, 8), Rectangular8)) << fmt::format("<{}> equal Rectangular[8]vector {} vs. ref: {}", type_name<T>(), create<T>(Rectangular, 8), Rectangular8);
        expect(equalVectors<T>(create<T>(Hamming, 8), Hamming8)) << fmt::format("<{}> equal Hamming[8] vector {} vs. ref: {}", type_name<T>(), create<T>(Hamming, 8), Hamming8);
        expect(equalVectors<T>(create<T>(Hann, 8), Hann8)) << fmt::format("<{}> equal Hann[8] vector {} vs. ref: {}", type_name<T>(), create<T>(Hann, 8), Hann8);
        expect(equalVectors<T>(create<T>(Blackman, 8), Blackman8)) << fmt::format("<{}> equal Blackman[8] vvector {} vs. ref: {}", type_name<T>(), create<T>(Blackman, 8), Blackman8);
        expect(equalVectors<T>(create<T>(BlackmanHarris, 8), BlackmanHarris8)) << fmt::format("<{}> equal BlackmanHarris[8] vector {} vs. ref: {}", type_name<T>(), create<T>(BlackmanHarris, 8), BlackmanHarris8);
        expect(equalVectors<T>(create<T>(BlackmanNuttall, 8), BlackmanNuttall8)) << fmt::format("<{}> equal BlackmanNuttall[8] vector {} vs. ref: {}", type_name<T>(), create<T>(BlackmanNuttall, 8), BlackmanNuttall8);
        expect(equalVectors<T>(create<T>(Exponential, 8), Exponential8)) << fmt::format("<{}> equal Exponential[8] vector {} vs. ref: {}", type_name<T>(), create<T>(Exponential, 8), Exponential8);
        expect(equalVectors<T>(create<T>(FlatTop, 8), FlatTop8)) << fmt::format("<{}> equal FlatTop[8] vector {} vs. ref: {}", type_name<T>(), create<T>(FlatTop, 8), FlatTop8);
        expect(equalVectors<T>(create<T>(HannExp, 8), HannExp8)) << fmt::format("<{}> equal HannExp[8] vector {} vs. ref: {}", type_name<T>(), create<T>(HannExp, 8), HannExp8);
        expect(equalVectors<T>(create<T>(Nuttall, 8), Nuttall8)) << fmt::format("<{}> equal Nuttall[8] vector {} vs. ref: {}", type_name<T>(), create<T>(Nuttall, 8), Nuttall8);
        expect(equalVectors<T>(create<T>(Kaiser, 8), Kaiser8)) << fmt::format("<{}> equal Kaiser[8] vector {} vs. ref: {}", type_name<T>(), create<T>(Kaiser, 8), Kaiser8);

        // test zero length
        expect(eq(create<T>(None, 0).size(), 0u)) << fmt::format("<{}> zero size None[8] vectors", type_name<T>());
        expect(eq(create<T>(Rectangular, 0).size(), 0u)) << fmt::format("<{}> zero size Rectangular[8] vectors", type_name<T>());
        expect(eq(create<T>(Hamming, 0).size(), 0u)) << fmt::format("<{}> zero size Hamming[8] vectors", type_name<T>());
        expect(eq(create<T>(Hann, 0).size(), 0u)) << fmt::format("<{}> zero size Hann[8] vectors", type_name<T>());
        expect(eq(create<T>(Blackman, 0).size(), 0u)) << fmt::format("<{}> zero size Blackman[8] vectors", type_name<T>());
        expect(eq(create<T>(BlackmanHarris, 0).size(), 0u)) << fmt::format("<{}> zero size BlackmanHarris[8] vectors", type_name<T>());
        expect(eq(create<T>(BlackmanNuttall, 0).size(), 0u)) << fmt::format("<{}> zero size BlackmanNuttall[8] vectors", type_name<T>());
        expect(eq(create<T>(Exponential, 0).size(), 0u)) << fmt::format("<{}> zero size Exponential[8] vectors", type_name<T>());
        expect(eq(create<T>(FlatTop, 0).size(), 0u)) << fmt::format("<{}> zero size FlatTop[8] vectors", type_name<T>());
        expect(eq(create<T>(HannExp, 0).size(), 0u)) << fmt::format("<{}> zero size HannExp[8] vectors", type_name<T>());
        expect(eq(create<T>(Nuttall, 0).size(), 0u)) << fmt::format("<{}> zero size Nuttall[8] vectors", type_name<T>());
        expect(eq(create<T>(Kaiser, 0).size(), 0u)) << fmt::format("<{}> zero size Kaiser[8] vectors", type_name<T>());
    } | floatingTypesToTest;

    "FFTW wisdom import/export tests"_test = []() {
        gr::algorithm::FFTw<double> fftw1{};

        std::string                 wisdomString1 = fftw1.exportWisdomToString();
        fftw1.forgetWisdom();
        int importOk = fftw1.importWisdomFromString(wisdomString1);
        expect(eq(importOk, 1)) << "Wisdom import from string.";
        std::string wisdomString2 = fftw1.exportWisdomToString();

        // lines are not always at the same order thus it is hard to compare
        // expect(eq(wisdomString1, wisdomString2)) << "Wisdom strings are the same.";
    };
};

int
main() { /* not needed for UT */
}
