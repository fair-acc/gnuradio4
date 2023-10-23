#include <array>
#include <cassert>
#include <numbers>
#include <numeric>

#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/algorithm/fourier/fftw.hpp>

template<typename T>
std::vector<T>
generateSinSample(std::size_t N, double sample_rate, double frequency, double amplitude) {
    std::vector<T> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        if constexpr (gr::meta::complex_like<T>) {
            signal[i] = { static_cast<typename T::value_type>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate)), 0. };
        } else {
            signal[i] = static_cast<T>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate));
        }
    }
    return signal;
}

template<gr::meta::array_or_vector_type T, gr::meta::array_or_vector_type U = T>
bool
equalVectors(const T &v1, const U &v2, double tolerance = std::is_same_v<typename T::value_type, double> ? 1.e-5 : 1e-4) {
    if (v1.size() != v2.size()) {
        return false;
    }
    if constexpr (gr::meta::complex_like<typename T::value_type>) {
        return std::ranges::equal(v1, v2, [&tolerance](const auto &l, const auto &r) {
            return std::abs(l.real() - r.real()) < static_cast<typename T::value_type>(tolerance) && std::abs(l.imag() - r.imag()) < static_cast<typename T::value_type>(tolerance);
        });
    } else {
        return std::ranges::equal(v1, v2, [&tolerance](const auto &l, const auto &r) { return std::abs(static_cast<double>(l) - static_cast<double>(r)) < tolerance; });
    }
}

template<typename TInput, typename TOutput, typename TExpInput, typename TExpOutput, typename TExpPlan>
void
testFFTwTypes() {
    using namespace boost::ut;
    gr::algorithm::FFTw<TInput, TOutput> fftBlock;
    expect(std::is_same_v<typename std::remove_pointer_t<decltype(fftBlock.fftwIn.get())>, TExpInput>) << "";
    expect(std::is_same_v<typename std::remove_pointer_t<decltype(fftBlock.fftwOut.get())>, TExpOutput>) << "";
    expect(std::is_same_v<decltype(fftBlock.fftwPlan.get()), TExpPlan>) << "";
}

template<typename TInput, typename TOutput, template<typename, typename> typename TAlgo>
struct TestTypes {
    using InType   = TInput;
    using OutType  = TOutput;
    using AlgoType = TAlgo<TInput, TOutput>;
};

const boost::ut::suite<"FFT algorithms and window functions"> windowTests = [] {
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using gr::algorithm::window::create;
    using gr::algorithm::FFT;
    using gr::algorithm::FFTw;

    using ComplexTypesToTest = std::tuple<
            // complex input, same in-out precision
            TestTypes<std::complex<float>, std::complex<float>, FFT>, TestTypes<std::complex<float>, std::complex<float>, FFTw>, TestTypes<std::complex<double>, std::complex<double>, FFT>,
            TestTypes<std::complex<double>, std::complex<double>, FFTw>,
            // complex input, different in-out precision
            TestTypes<std::complex<float>, std::complex<double>, FFT>, TestTypes<std::complex<float>, std::complex<double>, FFTw>, TestTypes<std::complex<double>, std::complex<float>, FFT>,
            TestTypes<std::complex<double>, std::complex<float>, FFTw>>;

    using RealTypesToTest = std::tuple<
            // real input, same in-out precision
            TestTypes<float, std::complex<float>, FFT>, TestTypes<float, std::complex<float>, FFTw>, TestTypes<double, std::complex<double>, FFT>, TestTypes<double, std::complex<double>, FFTw>,
            // real input, different in-out precision
            TestTypes<double, std::complex<float>, FFT>, TestTypes<double, std::complex<float>, FFTw>, TestTypes<double, std::complex<float>, FFT>, TestTypes<double, std::complex<float>, FFTw>>;

    using AllTypesToTest = decltype(std::tuple_cat(std::declval<ComplexTypesToTest>(), std::declval<RealTypesToTest>()));

    "FFT algo sin tests"_test = []<typename T>() {
        typename T::AlgoType fftAlgo{};
        constexpr double     tolerance{ 1.e-5 };
        struct TestParams {
            std::uint32_t N{ 1024 };           // must be power of 2
            double        sample_rate{ 128. }; // must be power of 2 (only for the unit test for easy comparison with true result)
            double        frequency{ 1. };
            double        amplitude{ 1. };
            bool          outputInDb{ false };
        };

        std::vector<TestParams> testCases = { { 256, 128., 10., 5., false }, { 512, 4., 1., 1., false }, { 512, 32., 1., 0.1, false }, { 256, 128., 10., 5., false } };
        for (const auto &t : testCases) {
            assert(std::has_single_bit(t.N));
            assert(std::has_single_bit(static_cast<std::size_t>(t.sample_rate)));

            const auto signal{ generateSinSample<typename T::InType>(t.N, t.sample_rate, t.frequency, t.amplitude) };
            auto       fftResult         = fftAlgo.compute(signal);
            auto       magnitudeSpectrum = gr::algorithm::fft::computeMagnitudeSpectrum(fftResult);
            auto       phase             = gr::algorithm::fft::computePhaseSpectrum(fftResult, { .outputInDeg = true, .unwrapPhase = true });
            const auto peakIndex{ static_cast<std::size_t>(
                    std::distance(magnitudeSpectrum.begin(),
                                  std::max_element(magnitudeSpectrum.begin(), std::next(magnitudeSpectrum.begin(), static_cast<std::ptrdiff_t>(t.N / 2u))))) }; // only positive frequencies from FFT
            const auto peakAmplitude = magnitudeSpectrum[peakIndex];
            const auto peakFrequency{ static_cast<double>(peakIndex) * t.sample_rate / static_cast<double>(t.N) };

            const auto expectedAmplitude = t.outputInDb ? 20. * log10(std::abs(t.amplitude)) : t.amplitude;
            expect(approx(static_cast<double>(peakAmplitude), expectedAmplitude, tolerance)) << fmt::format("{} equal amplitude", type_name<T>());
            expect(approx(peakFrequency, t.frequency, tolerance)) << fmt::format("{} equal frequency", type_name<T>());
        }
    } | AllTypesToTest{};

    "FFT algo pattern tests"_test = []<typename T>() {
        using InType = T::InType;
        typename T::AlgoType    fftAlgo{};
        constexpr double        tolerance{ 1.e-5 };
        constexpr std::uint32_t N{ 16 };
        static_assert(N == 16, "expected values are calculated for N == 16");

        std::vector<InType> signal(N);
        std::size_t         expectedPeakIndex{ 0 };
        InType              expectedFft0{ 0., 0. };
        double              expectedPeakAmplitude{ 0. };
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

            auto fftResult         = fftAlgo.compute(signal);
            auto magnitudeSpectrum = gr::algorithm::fft::computeMagnitudeSpectrum(fftResult);

            const auto peakIndex{ static_cast<std::size_t>(std::distance(magnitudeSpectrum.begin(), std::ranges::max_element(magnitudeSpectrum))) };
            const auto peakAmplitude{ magnitudeSpectrum[peakIndex] };

            expect(eq(peakIndex, expectedPeakIndex)) << fmt::format("<{}> equal peak index", type_name<T>());
            expect(approx(static_cast<double>(peakAmplitude), expectedPeakAmplitude, tolerance)) << fmt::format("<{}> equal amplitude", type_name<T>());
            expect(approx(static_cast<double>(fftResult[0].real()), static_cast<double>(expectedFft0.real()), tolerance)) << fmt::format("<{}> equal fft[0].real()", type_name<T>());
            expect(approx(static_cast<double>(fftResult[0].imag()), static_cast<double>(expectedFft0.imag()), tolerance)) << fmt::format("<{}> equal fft[0].imag()", type_name<T>());
        }
    } | ComplexTypesToTest{};

    "Unwrap Phase tests"_test = [] {
        std::vector<double> phase = { 0.2, -1., 2.5, -3.1, 0.9, -0.5, 1.2, 0.8, 1.5, -1.2, -2.7, 0.9, -0.8, -1.4, 0.6, 1.1, -1.9, 0.4, 1.3, -0.7 };
        // Output generated with python numpy.unwrap(phase)
        std::vector<double> expOut = { 0.2,         -1.,          -3.78318531,  -3.1,         -5.38318531,  -6.78318531,  -5.08318531,  -5.48318531,  -4.78318531,  -7.48318531,
                                       -8.98318531, -11.66637061, -13.36637061, -13.96637061, -11.96637061, -11.46637061, -14.46637061, -12.16637061, -11.26637061, -13.26637061 };
        gr::algorithm::fft::unwrapPhase(phase);
        expect(equalVectors(phase, expOut)) << "unwrapped phases are equal";
    };

    "FFTw types tests"_test = [] {
        testFFTwTypes<std::complex<float>, std::complex<float>, fftwf_complex, fftwf_complex, fftwf_plan>();
        testFFTwTypes<std::complex<double>, std::complex<double>, fftw_complex, fftw_complex, fftw_plan>();
        testFFTwTypes<float, std::complex<float>, float, fftwf_complex, fftwf_plan>();
        testFFTwTypes<double, std::complex<double>, double, fftw_complex, fftw_plan>();
    };

    "FFTW wisdom import/export tests"_test = []() {
        gr::algorithm::FFTw<double, std::complex<double>> fftw1{};

        std::string wisdomString1 = fftw1.exportWisdomToString();
        fftw1.forgetWisdom();
        int importOk = fftw1.importWisdomFromString(wisdomString1);
        expect(eq(importOk, 1)) << "Wisdom import from string.";
        std::string wisdomString2 = fftw1.exportWisdomToString();

        // lines are not always at the same order thus it is hard to compare
        // expect(eq(wisdomString1, wisdomString2)) << "Wisdom strings are the same.";
    };

    "window pre-computed array tests"_test = []<typename T>() { // this tests regression w.r.t. changed implementations
        // Expected value for size 8
        std::array RectangularRef{ 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
        std::array HammingRef{ 0.07672f, 0.25053218f, 0.64108455f, 0.9542833f, 0.95428324f, 0.6410846f, 0.25053206f, 0.07672f };
        std::array HannRef{ 0.f, 0.1882550991f, 0.611260467f, 0.950484434f, 0.950484434f, 0.611260467f, 0.1882550991f, 0.f };
        std::array BlackmanRef{ 0.f, 0.09045342435f, 0.4591829575f, 0.9203636181f, 0.9203636181f, 0.4591829575f, 0.09045342435f, 0.f };
        std::array BlackmanHarrisRef{ 0.00006f, 0.03339172348f, 0.3328335043f, 0.8893697722f, 0.8893697722f, 0.3328335043f, 0.03339172348f, 0.00006f };
        std::array BlackmanNuttallRef{ 0.0003628f, 0.03777576895f, 0.34272762f, 0.8918518611f, 0.8918518611f, 0.34272762f, 0.03777576895f, 0.0003628f };
        std::array ExponentialRef{ 1.f, 1.042546905f, 1.08690405f, 1.133148453f, 1.181360413f, 1.231623642f, 1.284025417f, 1.338656724f };
        std::array FlatTopRef{ 0.004f, -0.1696424054f, 0.04525319348f, 3.622389212f, 3.622389212f, 0.04525319348f, -0.1696424054f, 0.004f };
        std::array HannExpRef{ 0.f, 0.611260467f, 0.950484434f, 0.1882550991f, 0.1882550991f, 0.950484434f, 0.611260467f, 0.f };
        std::array NuttallRef{ 0.f, 0.0311427368f, 0.3264168059f, 0.8876284573f, 0.8876284573f, 0.3264168059f, 0.0311427368f, 0.f };
        std::array KaiserRef{ 0.5714348848f, 0.7650986027f, 0.9113132365f, 0.9899091685f, 0.9899091685f, 0.9113132365f, 0.7650986027f, 0.5714348848f };

        // check all windows for unwanted changes
        using enum gr::algorithm::window::Type;
        expect(equalVectors(create<T>(None, 8), RectangularRef)) << fmt::format("<{}> equal Rectangular vector {} vs. ref: {}", type_name<T>(), create<T>(None, 8), RectangularRef);
        expect(equalVectors(create<T>(Rectangular, 8), RectangularRef)) << fmt::format("<{}> equal Rectangular vector {} vs. ref: {}", type_name<T>(), create<T>(Rectangular, 8), RectangularRef);
        expect(equalVectors(create<T>(Hamming, 8), HammingRef)) << fmt::format("<{}> equal Hamming vector {} vs. ref: {}", type_name<T>(), create<T>(Hamming, 8), HammingRef);
        expect(equalVectors(create<T>(Hann, 8), HannRef)) << fmt::format("<{}> equal Hann vector {} vs. ref: {}", type_name<T>(), create<T>(Hann, 8), HannRef);
        expect(equalVectors(create<T>(Blackman, 8), BlackmanRef)) << fmt::format("<{}> equal Blackman vvector {} vs. ref: {}", type_name<T>(), create<T>(Blackman, 8), BlackmanRef);
        expect(equalVectors(create<T>(BlackmanHarris, 8), BlackmanHarrisRef))
                << fmt::format("<{}> equal BlackmanHarris vector {} vs. ref: {}", type_name<T>(), create<T>(BlackmanHarris, 8), BlackmanHarrisRef);
        expect(equalVectors(create<T>(BlackmanNuttall, 8), BlackmanNuttallRef))
                << fmt::format("<{}> equal BlackmanNuttall vector {} vs. ref: {}", type_name<T>(), create<T>(BlackmanNuttall, 8), BlackmanNuttallRef);
        expect(equalVectors(create<T>(Exponential, 8), ExponentialRef)) << fmt::format("<{}> equal Exponential vector {} vs. ref: {}", type_name<T>(), create<T>(Exponential, 8), ExponentialRef);
        expect(equalVectors(create<T>(FlatTop, 8), FlatTopRef)) << fmt::format("<{}> equal FlatTop vector {} vs. ref: {}", type_name<T>(), create<T>(FlatTop, 8), FlatTopRef);
        expect(equalVectors(create<T>(HannExp, 8), HannExpRef)) << fmt::format("<{}> equal HannExp vector {} vs. ref: {}", type_name<T>(), create<T>(HannExp, 8), HannExpRef);
        expect(equalVectors(create<T>(Nuttall, 8), NuttallRef)) << fmt::format("<{}> equal Nuttall vector {} vs. ref: {}", type_name<T>(), create<T>(Nuttall, 8), NuttallRef);
        expect(equalVectors(create<T>(Kaiser, 8), KaiserRef)) << fmt::format("<{}> equal Kaiser vector {} vs. ref: {}", type_name<T>(), create<T>(Kaiser, 8), KaiserRef);

        // test zero length
        expect(eq(create<T>(None, 0).size(), 0u)) << fmt::format("<{}> zero size None vectors", type_name<T>());
        expect(eq(create<T>(Rectangular, 0).size(), 0u)) << fmt::format("<{}> zero size Rectangular vectors", type_name<T>());
        expect(eq(create<T>(Hamming, 0).size(), 0u)) << fmt::format("<{}> zero size Hamming vectors", type_name<T>());
        expect(eq(create<T>(Hann, 0).size(), 0u)) << fmt::format("<{}> zero size Hann vectors", type_name<T>());
        expect(eq(create<T>(Blackman, 0).size(), 0u)) << fmt::format("<{}> zero size Blackman vectors", type_name<T>());
        expect(eq(create<T>(BlackmanHarris, 0).size(), 0u)) << fmt::format("<{}> zero size BlackmanHarris vectors", type_name<T>());
        expect(eq(create<T>(BlackmanNuttall, 0).size(), 0u)) << fmt::format("<{}> zero size BlackmanNuttall vectors", type_name<T>());
        expect(eq(create<T>(Exponential, 0).size(), 0u)) << fmt::format("<{}> zero size Exponential vectors", type_name<T>());
        expect(eq(create<T>(FlatTop, 0).size(), 0u)) << fmt::format("<{}> zero size FlatTop vectors", type_name<T>());
        expect(eq(create<T>(HannExp, 0).size(), 0u)) << fmt::format("<{}> zero size HannExp vectors", type_name<T>());
        expect(eq(create<T>(Nuttall, 0).size(), 0u)) << fmt::format("<{}> zero size Nuttall vectors", type_name<T>());
        expect(eq(create<T>(Kaiser, 0).size(), 0u)) << fmt::format("<{}> zero size Kaiser vectors", type_name<T>());
    } | std::tuple<float, double>();

    "basic window tests"_test = [](auto &val) {
        const auto &[window, windowName] = val;
        using enum gr::algorithm::window::Type;

        const auto w = create(window, 1024U);
        expect(eq(w.size(), 1024U));

        if (window == Exponential || window == FlatTop || window == Blackman || window == Nuttall) {
            return; // min max out of [0, 1] by design and/or numerical corner cases
        }
        const auto [min, max] = std::ranges::minmax_element(w);
        expect(ge(*min, 0.f)) << fmt::format("window {} min value\n", windowName);
        expect(le(*max, 1.f)) << fmt::format("window {} max value\n", windowName);
    } | magic_enum::enum_entries<gr::algorithm::window::Type>();

    "window corner cases"_test = []<typename T>() {
        static_assert(not magic_enum::enum_cast<gr::algorithm::window::Type>("UnknownWindow", magic_enum::case_insensitive).has_value());
        expect(throws<std::invalid_argument>([] { std::ignore = create(gr::algorithm::window::Type::Kaiser, 1); })) << "invalid Kaiser window size";
        expect(throws<std::invalid_argument>([] { std::ignore = create(gr::algorithm::window::Type::Kaiser, 2, -1.f); })) << "invalid Kaiser window beta";
    } | std::tuple<float, double>();
};

int
main() { /* not needed for UT */
}
