#include <array>
#include <cassert>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>

#include <boost/ut.hpp>

#include <gnuradio-4.0/meta/formatter.hpp>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>

template<typename T>
std::vector<T> generateSinSample(std::size_t N, double sample_rate, double frequency, double amplitude) {
    std::vector<T> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        if constexpr (gr::meta::complex_like<T>) {
            signal[i] = {static_cast<typename T::value_type>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate)), 0.};
        } else {
            signal[i] = static_cast<T>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate));
        }
    }
    return signal;
}

template<gr::meta::array_or_vector_type T, gr::meta::array_or_vector_type U = T>
bool equalVectors(const T& v1, const U& v2, double tolerance = std::is_same_v<typename T::value_type, double> ? 1.e-5 : 1e-4) {
    if (v1.size() != v2.size()) {
        return false;
    }
    if constexpr (gr::meta::complex_like<typename T::value_type>) {
        return std::ranges::equal(v1, v2, [&tolerance](const auto& l, const auto& r) { return std::abs(l.real() - r.real()) < static_cast<typename T::value_type>(tolerance) && std::abs(l.imag() - r.imag()) < static_cast<typename T::value_type>(tolerance); });
    } else {
        return std::ranges::equal(v1, v2, [&tolerance](const auto& l, const auto& r) { return std::abs(static_cast<double>(l) - static_cast<double>(r)) < tolerance; });
    }
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

    using ComplexTypesToTest = std::tuple<
        // complex input, same in-out precision
        TestTypes<std::complex<float>, std::complex<float>, FFT>, TestTypes<std::complex<double>, std::complex<double>, FFT>,
        // complex input, different in-out precision
        TestTypes<std::complex<float>, std::complex<double>, FFT>, TestTypes<std::complex<double>, std::complex<float>, FFT>>;

    using RealTypesToTest = std::tuple<
        // real input, same in-out precision
        TestTypes<float, std::complex<float>, FFT>, TestTypes<double, std::complex<double>, FFT>,
        // real input, different in-out precision
        TestTypes<double, std::complex<float>, FFT>, TestTypes<double, std::complex<float>, FFT>>;

    using AllTypesToTest = decltype(std::tuple_cat(std::declval<ComplexTypesToTest>(), std::declval<RealTypesToTest>()));

    "FFT algo sin tests"_test = []<typename T>() {
        typename T::AlgoType fftAlgo{};
        constexpr double     tolerance{1.e-5};
        struct TestParams {
            gr::Size_t N{1024};           // must be power of 2
            double     sample_rate{128.}; // must be power of 2 (only for the unit test for easy comparison with true result)
            double     frequency{1.};
            double     amplitude{1.};
            bool       outputInDb{false};
        };

        std::vector<TestParams> testCases = {{256, 128., 10., 5., false}, {512, 4., 1., 1., false}, {512, 32., 1., 0.1, false}, {256, 128., 10., 5., false}};
        for (const auto& t : testCases) {
            assert(std::has_single_bit(t.N));
            assert(std::has_single_bit(static_cast<std::size_t>(t.sample_rate)));

            const auto signal{generateSinSample<typename T::InType>(t.N, t.sample_rate, t.frequency, t.amplitude)};
            auto       fftResult         = fftAlgo.compute(signal);
            auto       magnitudeSpectrum = gr::algorithm::fft::computeMagnitudeSpectrum(fftResult);
            auto       phase             = gr::algorithm::fft::computePhaseSpectrum(fftResult, {.outputInDeg = true, .unwrapPhase = true});
            const auto peakIndex{static_cast<std::size_t>(std::distance(magnitudeSpectrum.begin(), std::max_element(magnitudeSpectrum.begin(), std::next(magnitudeSpectrum.begin(), static_cast<std::ptrdiff_t>(t.N / 2u)))))}; // only positive frequencies from FFT
            const auto peakAmplitude = magnitudeSpectrum[peakIndex];
            const auto peakFrequency{static_cast<double>(peakIndex) * t.sample_rate / static_cast<double>(t.N)};

            const auto expectedAmplitude = t.outputInDb ? 20. * log10(std::abs(t.amplitude)) : t.amplitude;
            expect(approx(static_cast<double>(peakAmplitude), expectedAmplitude, tolerance)) << std::format("{} equal amplitude", type_name<T>());
            expect(approx(peakFrequency, t.frequency, tolerance)) << std::format("{} equal frequency", type_name<T>());
        }
    } | AllTypesToTest{};

    "FFT algo pattern tests"_test = []<typename T>() {
        using InType = T::InType;
        typename T::AlgoType fftAlgo{};
        constexpr double     tolerance{1.e-5};
        constexpr gr::Size_t N{16};
        static_assert(N == 16, "expected values are calculated for N == 16");

        std::vector<InType> signal(N);
        std::size_t         expectedPeakIndex{0};
        InType              expectedFft0{0., 0.};
        double              expectedPeakAmplitude{0.};
        for (std::size_t iT = 0; iT < 5; iT++) {
            if (iT == 0) {
                std::ranges::fill(signal.begin(), signal.end(), InType(0., 0.));
                expectedFft0          = {0., 0.};
                expectedPeakAmplitude = 0.;
            } else if (iT == 1) {
                std::ranges::fill(signal.begin(), signal.end(), InType(1., 0.));
                expectedFft0          = {16., 0.};
                expectedPeakAmplitude = 2.;
            } else if (iT == 2) {
                std::ranges::fill(signal.begin(), signal.end(), InType(1., 1.));
                expectedFft0          = {16., 16.};
                expectedPeakAmplitude = std::sqrt(8.);
            } else if (iT == 3) {
                std::iota(signal.begin(), signal.end(), 1);
                expectedFft0          = {136., 0.};
                expectedPeakAmplitude = 17.;
            } else if (iT == 4) {
                int i = 0;
                std::ranges::generate(signal.begin(), signal.end(), [&i] { return InType(static_cast<typename InType::value_type>(i++ % 2), 0.); });
                expectedFft0          = {8., 0.};
                expectedPeakAmplitude = 1.;
            }

            auto fftResult         = fftAlgo.compute(signal);
            auto magnitudeSpectrum = gr::algorithm::fft::computeMagnitudeSpectrum(fftResult);

            const auto peakIndex{static_cast<std::size_t>(std::distance(magnitudeSpectrum.begin(), std::ranges::max_element(magnitudeSpectrum)))};
            const auto peakAmplitude{magnitudeSpectrum[peakIndex]};

            expect(eq(peakIndex, expectedPeakIndex)) << std::format("<{}> equal peak index", type_name<T>());
            expect(approx(static_cast<double>(peakAmplitude), expectedPeakAmplitude, tolerance)) << std::format("<{}> equal amplitude", type_name<T>());
            expect(approx(static_cast<double>(fftResult[0].real()), static_cast<double>(expectedFft0.real()), tolerance)) << std::format("<{}> equal fft[0].real()", type_name<T>());
            expect(approx(static_cast<double>(fftResult[0].imag()), static_cast<double>(expectedFft0.imag()), tolerance)) << std::format("<{}> equal fft[0].imag()", type_name<T>());
        }
    } | ComplexTypesToTest{};

    "Unwrap Phase tests"_test = [] {
        std::vector<double> phase = {0.2, -1., 2.5, -3.1, 0.9, -0.5, 1.2, 0.8, 1.5, -1.2, -2.7, 0.9, -0.8, -1.4, 0.6, 1.1, -1.9, 0.4, 1.3, -0.7};
        // Output generated with python numpy.unwrap(phase)
        std::vector<double> expOut = {0.2, -1., -3.78318531, -3.1, -5.38318531, -6.78318531, -5.08318531, -5.48318531, -4.78318531, -7.48318531, -8.98318531, -11.66637061, -13.36637061, -13.96637061, -11.96637061, -11.46637061, -14.46637061, -12.16637061, -11.26637061, -13.26637061};
        gr::algorithm::fft::unwrapPhase(phase);
        expect(equalVectors(phase, expOut)) << "unwrapped phases are equal";
    };

    "window pre-computed array tests"_test = []<typename T>() { // this tests regression w.r.t. changed implementations
        // Expected value for size 8
        std::array RectangularRef{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
        std::array HammingRef{0.07672f, 0.25053218f, 0.64108455f, 0.9542833f, 0.95428324f, 0.6410846f, 0.25053206f, 0.07672f};
        std::array HannRef{0.f, 0.1882550991f, 0.611260467f, 0.950484434f, 0.950484434f, 0.611260467f, 0.1882550991f, 0.f};
        std::array BlackmanRef{0.f, 0.09045342435f, 0.4591829575f, 0.9203636181f, 0.9203636181f, 0.4591829575f, 0.09045342435f, 0.f};
        std::array BlackmanHarrisRef{0.00006f, 0.03339172348f, 0.3328335043f, 0.8893697722f, 0.8893697722f, 0.3328335043f, 0.03339172348f, 0.00006f};
        std::array BlackmanNuttallRef{0.0003628f, 0.03777576895f, 0.34272762f, 0.8918518611f, 0.8918518611f, 0.34272762f, 0.03777576895f, 0.0003628f};
        std::array ExponentialRef{1.f, 1.042546905f, 1.08690405f, 1.133148453f, 1.181360413f, 1.231623642f, 1.284025417f, 1.338656724f};
        std::array FlatTopRef{0.004f, -0.1696424054f, 0.04525319348f, 3.622389212f, 3.622389212f, 0.04525319348f, -0.1696424054f, 0.004f};
        std::array HannExpRef{0.f, 0.611260467f, 0.950484434f, 0.1882550991f, 0.1882550991f, 0.950484434f, 0.611260467f, 0.f};
        std::array NuttallRef{0.f, 0.0311427368f, 0.3264168059f, 0.8876284573f, 0.8876284573f, 0.3264168059f, 0.0311427368f, 0.f};
        std::array KaiserRef{0.5714348848f, 0.7650986027f, 0.9113132365f, 0.9899091685f, 0.9899091685f, 0.9113132365f, 0.7650986027f, 0.5714348848f};

        // check all windows for unwanted changes
        using enum gr::algorithm::window::Type;
        expect(equalVectors(create<T>(None, 8), RectangularRef)) << std::format("<{}> equal Rectangular vector {} vs. ref: {}", type_name<T>(), create<T>(None, 8), RectangularRef);
        expect(equalVectors(create<T>(Rectangular, 8), RectangularRef)) << std::format("<{}> equal Rectangular vector {} vs. ref: {}", type_name<T>(), create<T>(Rectangular, 8), RectangularRef);
        expect(equalVectors(create<T>(Hamming, 8), HammingRef)) << std::format("<{}> equal Hamming vector {} vs. ref: {}", type_name<T>(), create<T>(Hamming, 8), HammingRef);
        expect(equalVectors(create<T>(Hann, 8), HannRef)) << std::format("<{}> equal Hann vector {} vs. ref: {}", type_name<T>(), create<T>(Hann, 8), HannRef);
        expect(equalVectors(create<T>(Blackman, 8), BlackmanRef)) << std::format("<{}> equal Blackman vvector {} vs. ref: {}", type_name<T>(), create<T>(Blackman, 8), BlackmanRef);
        expect(equalVectors(create<T>(BlackmanHarris, 8), BlackmanHarrisRef)) << std::format("<{}> equal BlackmanHarris vector {} vs. ref: {}", type_name<T>(), create<T>(BlackmanHarris, 8), BlackmanHarrisRef);
        expect(equalVectors(create<T>(BlackmanNuttall, 8), BlackmanNuttallRef)) << std::format("<{}> equal BlackmanNuttall vector {} vs. ref: {}", type_name<T>(), create<T>(BlackmanNuttall, 8), BlackmanNuttallRef);
        expect(equalVectors(create<T>(Exponential, 8), ExponentialRef)) << std::format("<{}> equal Exponential vector {} vs. ref: {}", type_name<T>(), create<T>(Exponential, 8), ExponentialRef);
        expect(equalVectors(create<T>(FlatTop, 8), FlatTopRef)) << std::format("<{}> equal FlatTop vector {} vs. ref: {}", type_name<T>(), create<T>(FlatTop, 8), FlatTopRef);
        expect(equalVectors(create<T>(HannExp, 8), HannExpRef)) << std::format("<{}> equal HannExp vector {} vs. ref: {}", type_name<T>(), create<T>(HannExp, 8), HannExpRef);
        expect(equalVectors(create<T>(Nuttall, 8), NuttallRef)) << std::format("<{}> equal Nuttall vector {} vs. ref: {}", type_name<T>(), create<T>(Nuttall, 8), NuttallRef);
        expect(equalVectors(create<T>(Kaiser, 8), KaiserRef)) << std::format("<{}> equal Kaiser vector {} vs. ref: {}", type_name<T>(), create<T>(Kaiser, 8), KaiserRef);

        // test zero length
        expect(eq(create<T>(None, 0).size(), 0u)) << std::format("<{}> zero size None vectors", type_name<T>());
        expect(eq(create<T>(Rectangular, 0).size(), 0u)) << std::format("<{}> zero size Rectangular vectors", type_name<T>());
        expect(eq(create<T>(Hamming, 0).size(), 0u)) << std::format("<{}> zero size Hamming vectors", type_name<T>());
        expect(eq(create<T>(Hann, 0).size(), 0u)) << std::format("<{}> zero size Hann vectors", type_name<T>());
        expect(eq(create<T>(Blackman, 0).size(), 0u)) << std::format("<{}> zero size Blackman vectors", type_name<T>());
        expect(eq(create<T>(BlackmanHarris, 0).size(), 0u)) << std::format("<{}> zero size BlackmanHarris vectors", type_name<T>());
        expect(eq(create<T>(BlackmanNuttall, 0).size(), 0u)) << std::format("<{}> zero size BlackmanNuttall vectors", type_name<T>());
        expect(eq(create<T>(Exponential, 0).size(), 0u)) << std::format("<{}> zero size Exponential vectors", type_name<T>());
        expect(eq(create<T>(FlatTop, 0).size(), 0u)) << std::format("<{}> zero size FlatTop vectors", type_name<T>());
        expect(eq(create<T>(HannExp, 0).size(), 0u)) << std::format("<{}> zero size HannExp vectors", type_name<T>());
        expect(eq(create<T>(Nuttall, 0).size(), 0u)) << std::format("<{}> zero size Nuttall vectors", type_name<T>());
        expect(eq(create<T>(Kaiser, 0).size(), 0u)) << std::format("<{}> zero size Kaiser vectors", type_name<T>());
    } | std::tuple<float, double>();

    "basic window tests"_test = [](auto& val) {
        const auto& [window, windowName] = val;
        using enum gr::algorithm::window::Type;

        const auto w = create(window, 1024U);
        expect(eq(w.size(), 1024U));

        if (window == Exponential || window == FlatTop || window == Blackman || window == Nuttall) {
            return; // min max out of [0, 1] by design and/or numerical corner cases
        }
        const auto [min, max] = std::ranges::minmax_element(w);
        expect(ge(*min, 0.f)) << std::format("window {} min value\n", windowName);
        expect(le(*max, 1.f)) << std::format("window {} max value\n", windowName);
    } | magic_enum::enum_entries<gr::algorithm::window::Type>();

    "window corner cases"_test = []<typename T>() {
        static_assert(not magic_enum::enum_cast<gr::algorithm::window::Type>("UnknownWindow", magic_enum::case_insensitive).has_value());
        expect(throws<std::invalid_argument>([] { std::ignore = create(gr::algorithm::window::Type::Kaiser, 1); })) << "invalid Kaiser window size";
        expect(throws<std::invalid_argument>([] { std::ignore = create(gr::algorithm::window::Type::Kaiser, 2, -1.f); })) << "invalid Kaiser window beta";
    } | std::tuple<float, double>();
};

const boost::ut::suite<"FFT common kernel-callable cores"> fftCommonCoreTests = [] {
    using namespace boost::ut;
    namespace fft = gr::algorithm::fft;

    "applyWindow span core matches the per-element core"_test = [] {
        std::vector<float> samples{1.f, 2.f, 3.f, 4.f, 5.f};
        std::vector<float> window{0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

        std::vector<float> viaElementCore(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            viaElementCore[i] = fft::applyWindowOne(samples[i], window[i]);
        }

        std::vector<float> viaSpanCore = samples;
        fft::applyWindow(std::span<float>{viaSpanCore}, std::span<const float>{window});

        expect(std::ranges::equal(viaElementCore, viaSpanCore)) << "per-element core and span core agree";
    };

    "magnitude core matches computeMagnitudeSpectrum wrapper"_test = [] {
        const std::vector<std::complex<double>> fftIn{{3., 4.}, {0., 0.}, {-1., 1.}, {2., -2.}};
        for (const bool outputInDb : {false, true}) {
            const auto wrapperOut = fft::computeMagnitudeSpectrum(fftIn, {.outputInDb = outputInDb});

            std::vector<double> coreOut(fftIn.size());
            for (std::size_t i = 0; i < fftIn.size(); ++i) {
                coreOut[i] = fft::computeMagnitudeOne(fftIn[i], fftIn.size(), outputInDb);
            }
            expect(std::ranges::equal(wrapperOut, coreOut)) << std::format("magnitude core vs. wrapper (outputInDb={})", outputInDb);
        }
    };

    "magnitude core matches computeMagnitudeSpectrum wrapper for a half spectrum"_test = [] {
        // pins that the wrapper still normalises by the full fftIn.size() even though only the
        // first half is written -- a truncated span passed to the core would silently normalise
        // by the half size instead
        const std::vector<std::complex<double>> fftIn{{3., 4.}, {0., 0.}, {-1., 1.}, {2., -2.}, {5., -1.}, {-2., 3.}};
        const auto                              wrapperOut = fft::computeMagnitudeSpectrum(fftIn, {.computeHalfSpectrum = true});

        std::vector<double> coreOut(fftIn.size() / 2UZ);
        for (std::size_t i = 0; i < coreOut.size(); ++i) {
            coreOut[i] = fft::computeMagnitudeOne(fftIn[i], fftIn.size(), false);
        }
        expect(std::ranges::equal(wrapperOut, coreOut)) << "half-spectrum magnitude core vs. wrapper";
    };

    "includeNyquist extends the half spectrum by one bin (rfft convention: DC..Nyquist inclusive)"_test = [] {
        const std::vector<std::complex<double>> fftIn{{3., 4.}, {0., 0.}, {-1., 1.}, {2., -2.}, {5., -1.}, {-2., 3.}};

        const auto magWithoutNyquist = fft::computeMagnitudeSpectrum(fftIn, {.computeHalfSpectrum = true, .includeNyquist = false});
        const auto magWithNyquist    = fft::computeMagnitudeSpectrum(fftIn, {.computeHalfSpectrum = true, .includeNyquist = true});
        expect(eq(magWithoutNyquist.size(), fftIn.size() / 2UZ));
        expect(eq(magWithNyquist.size(), fftIn.size() / 2UZ + 1UZ));
        expect(std::ranges::equal(magWithoutNyquist, std::span{magWithNyquist}.first(magWithoutNyquist.size()))) << "the extra bin must not perturb the bins already present";
        expect(approx(magWithNyquist.back(), fft::computeMagnitudeOne(fftIn[fftIn.size() / 2UZ], fftIn.size(), false), 1e-12)) << "the extra bin is exactly the Nyquist bin";

        const auto phaseWithoutNyquist = fft::computePhaseSpectrum(fftIn, {.computeHalfSpectrum = true, .includeNyquist = false});
        const auto phaseWithNyquist    = fft::computePhaseSpectrum(fftIn, {.computeHalfSpectrum = true, .includeNyquist = true});
        expect(eq(phaseWithoutNyquist.size(), fftIn.size() / 2UZ));
        expect(eq(phaseWithNyquist.size(), fftIn.size() / 2UZ + 1UZ));
        expect(std::ranges::equal(phaseWithoutNyquist, std::span{phaseWithNyquist}.first(phaseWithoutNyquist.size()))) << "the extra bin must not perturb the bins already present";
        expect(approx(phaseWithNyquist.back(), fft::computePhaseOne(fftIn[fftIn.size() / 2UZ]), 1e-12)) << "the extra bin is exactly the Nyquist bin";
    };

    "phase core matches computePhaseSpectrum wrapper (no unwrap/deg/shift)"_test = [] {
        const std::vector<std::complex<double>> fftIn{{3., 4.}, {0., 0.}, {-1., 1.}, {2., -2.}};
        const auto                              wrapperOut = fft::computePhaseSpectrum(fftIn);

        std::vector<double> coreOut(fftIn.size());
        for (std::size_t i = 0; i < fftIn.size(); ++i) {
            coreOut[i] = fft::computePhaseOne(fftIn[i]);
        }
        expect(std::ranges::equal(wrapperOut, coreOut)) << "phase core vs. wrapper";
    };

    "fftShiftIndex reproduces the rotate-based spectrum shift"_test = [] {
        const std::vector<std::complex<double>> fftIn{{0., 0.}, {1., 0.}, {2., 0.}, {3., 0.}, {4., 0.}, {5., 0.}, {6., 0.}, {7., 0.}};
        const auto                              shifted = fft::computeMagnitudeSpectrum(fftIn, {.shiftSpectrum = true});

        std::vector<double> viaIndex(fftIn.size());
        for (std::size_t i = 0; i < fftIn.size(); ++i) {
            viaIndex[i] = fft::computeMagnitudeOne(fftIn[fft::fftShiftIndex(i, fftIn.size())], fftIn.size(), false);
        }
        expect(std::ranges::equal(shifted, viaIndex)) << "index-mapped shift matches the rotate-based shift";
    };

    "unwrapPhase span overload matches the container overload it now backs"_test = [] {
        std::vector<double> viaContainer = {0.2, -1., 2.5, -3.1, 0.9, -0.5, 1.2, 0.8, 1.5, -1.2, -2.7, 0.9, -0.8, -1.4, 0.6, 1.1, -1.9, 0.4, 1.3, -0.7};
        std::vector<double> viaSpan      = viaContainer;
        fft::unwrapPhase(viaContainer);
        fft::unwrapPhase(std::span<double>{viaSpan});
        expect(std::ranges::equal(viaContainer, viaSpan)) << "container overload delegates exactly to the span overload";
    };

    "unwrapPhase hand-computed short case"_test = [] {
        // raw diff[2] = -3.0 - 3.0 = -6.0 < -pi -> one +2*pi correction
        std::vector<double> phase = {0.0, 3.0, -3.0};
        fft::unwrapPhase(phase);
        expect(approx(phase[0], 0.0, 1e-9));
        expect(approx(phase[1], 3.0, 1e-9));
        expect(approx(phase[2], -3.0 + 2. * std::numbers::pi_v<double>, 1e-9));
    };

    "unwrapPhase never produces a consecutive jump larger than pi across a multiply-wrapping signal"_test = [] {
        constexpr std::size_t N = 500;
        std::vector<double>   phase(N);
        for (std::size_t i = 0; i < N; ++i) {
            const double ramp = 1.3 * static_cast<double>(i); // steep enough to wrap several times over N samples
            phase[i]          = std::atan2(std::sin(ramp), std::cos(ramp));
        }
        fft::unwrapPhase(phase);

        const double pi = std::numbers::pi_v<double>;
        for (std::size_t i = 1; i < N; ++i) {
            expect(le(std::abs(phase[i] - phase[i - 1]), pi + 1e-9)) << std::format("consecutive unwrapped diff exceeds pi at i={}", i);
        }
    };

    "unwrapPhase is a no-op on empty and single-sample spans"_test = [] {
        std::vector<double> empty{};
        fft::unwrapPhase(empty);
        expect(empty.empty());

        std::vector<double> single{1.23};
        fft::unwrapPhase(single);
        expect(approx(single[0], 1.23, 1e-9));
    };

    "unwrapPhase leaves an exact +pi difference uncorrected (tie convention)"_test = [] {
        const double        pi = std::numbers::pi_v<double>;
        std::vector<double> phase{0.0, pi};
        fft::unwrapPhase(phase);
        expect(approx(phase[0], 0.0, 1e-12));
        expect(approx(phase[1], pi, 1e-12)) << "a +pi tie must not trigger the -2*pi correction";
    };

    "unwrapPhase leaves an exact -pi difference uncorrected (tie convention)"_test = [] {
        const double        pi = std::numbers::pi_v<double>;
        std::vector<double> phase{0.0, -pi};
        fft::unwrapPhase(phase);
        expect(approx(phase[0], 0.0, 1e-12));
        expect(approx(phase[1], -pi, 1e-12)) << "a -pi tie must not trigger the +2*pi correction";
    };

    "unwrapPhase leaves an exact +pi/-pi difference uncorrected (tie convention, float)"_test = [] {
        const float        pi = std::numbers::pi_v<float>;
        std::vector<float> plusTie{0.0f, pi};
        fft::unwrapPhase(plusTie);
        expect(approx(plusTie[1], pi, 1e-6f));

        std::vector<float> minusTie{0.0f, -pi};
        fft::unwrapPhase(minusTie);
        expect(approx(minusTie[1], -pi, 1e-6f));
    };

    "unwrapPhase does not let a NaN bin poison the running correction count"_test = [] {
        const double        pi  = std::numbers::pi_v<double>;
        const double        nan = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> phase{0.0, 3.0, -3.0, nan, 0.5, 3.0};
        fft::unwrapPhase(phase);

        expect(approx(phase[0], 0.0, 1e-9));
        expect(approx(phase[1], 3.0, 1e-9));            // diff 3.0 < pi -> no correction, k = 0
        expect(approx(phase[2], -3.0 + 2. * pi, 1e-9)); // diff -6.0 < -pi -> k = 1
        expect(std::isnan(phase[3]));                   // NaN bin stays NaN
        expect(approx(phase[4], 0.5 + 2. * pi, 1e-9)) << "the pre-NaN correction count (k=1) must still apply several samples later";
        expect(approx(phase[5], 3.0 + 2. * pi, 1e-9)) << "corrections resume normally once finite differences return";
    };

    "unwrapPhase tracks a long run of same-sign wraps without drift"_test = [] {
        constexpr std::size_t N  = 2000;
        const double          pi = std::numbers::pi_v<double>;
        std::vector<double>   phase(N);
        for (std::size_t i = 0; i < N; ++i) {
            const double ramp = 0.9 * static_cast<double>(i); // monotonic ramp -> every wrap has the same sign
            phase[i]          = std::atan2(std::sin(ramp), std::cos(ramp));
        }
        fft::unwrapPhase(phase);

        for (std::size_t i = 1; i < N; ++i) {
            expect(le(std::abs(phase[i] - phase[i - 1]), pi + 1e-9)) << std::format("consecutive unwrapped diff exceeds pi at i={}", i);
        }
        expect(gt(phase.back() - phase.front(), 0.0)) << "a monotonically increasing ramp must unwrap to a monotonically increasing result";
    };
};

namespace {
constexpr float kWindowedThirdSample = [] {
    std::array<float, 4>           samples{1.f, 2.f, 3.f, 4.f};
    constexpr std::array<float, 4> coefficients{0.5f, 0.5f, 0.5f, 0.5f};
    gr::algorithm::fft::applyWindow(std::span<float>{samples}, std::span<const float>{coefficients});
    return samples[2];
}();
static_assert(kWindowedThirdSample == 1.5f, "applyWindow span core must stay allocation-free and usable in a constexpr context");
static_assert(gr::algorithm::fft::fftShiftIndex(3UZ, 8UZ) == 7UZ, "fftShiftIndex is pure index arithmetic, usable in a constexpr context");
} // namespace

int main() { /* not needed for UT */ }
