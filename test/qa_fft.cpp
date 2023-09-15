#include <boost/ut.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

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

template<typename T, typename U>
bool
equalVectors(const std::vector<T> &v1, const std::vector<U> &v2, double tolerance = 1.e-6) {
    if constexpr (gr::blocks::fft::ComplexType<T>) {
        return std::equal(v1.begin(), v1.end(), v2.begin(), [&tolerance](const auto &l, const auto &r) {
            return std::abs(l.real() - r.real()) < static_cast<typename T::value_type>(tolerance) && std::abs(l.imag() - r.imag()) < static_cast<typename T::value_type>(tolerance);
        });
    } else {
        return std::equal(v1.begin(), v1.end(), v2.begin(), [&tolerance](const auto &l, const auto &r) { return std::abs(static_cast<double>(l) - static_cast<double>(r)) < tolerance; });
    }
}

template<typename T, typename inT, typename outT, typename pT>
void
testFftwTypes() {
    using namespace boost::ut;
    gr::blocks::fft::fft<T> fft1;
    expect(std::is_same_v<typename std::remove_pointer_t<decltype(fft1.fftwIn.get())>, inT>) << "";
    expect(std::is_same_v<typename std::remove_pointer_t<decltype(fft1.fftwOut.get())>, outT>) << "";
    expect(std::is_same_v<decltype(fft1.fftwPlan.get()), pT>) << "";
}

const boost::ut::suite fftTests = [] {
    using namespace boost::ut;
    using namespace gr::blocks::fft;
    using namespace boost::ut::reflection;
    std::tuple<std::complex<float>, std::complex<double>>                     complexTypesToTest{};
    std::tuple<std::complex<float>, std::complex<double>, float, double, int> typesToTest{};
    std::tuple<float, double>                                                 floatingTypesToTest{};

    "FFT sin tests"_test = []<typename T>() {
        fft<T>           fft1{};
        constexpr double tolerance{ 1.e-6 };
        struct TestParams {
            std::size_t N{ 1024 };          // must be power of 2
            double      sampleRate{ 128. }; // must be power of 2 (only for the unit test for easy comparison with true result)
            double      frequency{ 1. };
            double      amplitude{ 1. };
            bool        outputInDb{ false };
        };

        std::vector<TestParams> testCases = { { 256, 128., 10., 5., false }, { 512, 4., 1., 1., false }, { 512, 32., 1., 0.1, false }, { 256, 128., 10., 5., true } };
        for (const auto &t : testCases) {
            assert(std::has_single_bit(t.N));
            assert(std::has_single_bit(static_cast<std::size_t>(t.sampleRate)));

            std::ignore = fft1.settings().set({ { "fftSize", t.N } });
            std::ignore = fft1.settings().set({ { "outputInDb", t.outputInDb } });
            std::ignore = fft1.settings().apply_staged_parameters();
            const auto signal{ generateSinSample<T>(t.N, t.sampleRate, t.frequency, t.amplitude) };
            for (const auto s : signal) {
                fft1.inputHistory.push_back(static_cast<typename fft<T>::InHistoryType>(s));
            }
            fft1.prepareInput();
            fft1.computeFft();
            fft1.computeMagnitudeSpectrum();

            const auto peakIndex{
                static_cast<std::size_t>(std::distance(fft1.magnitudeSpectrum.begin(),
                                                       std::max_element(fft1.magnitudeSpectrum.begin(), std::next(fft1.magnitudeSpectrum.begin(), static_cast<std::ptrdiff_t>(t.N / 2u)))))
            }; // only positive frequencies from FFT
            const auto peakAmplitude = fft1.magnitudeSpectrum[peakIndex];
            const auto peakFrequency{ static_cast<double>(peakIndex) * t.sampleRate / static_cast<double>(t.N) };

            const auto expectedAmplitude = t.outputInDb ? 20. * log10(std::abs(t.amplitude)) : t.amplitude;
            if constexpr (!std::is_same_v<int, T>) {
                expect(approx(static_cast<double>(peakAmplitude), expectedAmplitude, tolerance)) << fmt::format("<{}> equal amplitude", type_name<T>());
                expect(approx(peakFrequency, t.frequency, tolerance)) << fmt::format("<{}> equal frequency", type_name<T>());
            }
        }
    } | typesToTest;

    "FFT pattern tests"_test = []<typename T>() {
        fft<T>                fft1{};
        constexpr double      tolerance{ 1.e-6 };
        constexpr std::size_t N{ 16 };
        std::ignore = fft1.settings().set({ { "fftSize", N } });
        std::ignore = fft1.settings().apply_staged_parameters();

        std::vector<T> signal(N);

        static_assert(N == 16, "expected values are calculated for N == 16");
        std::size_t expectedPeakIndex{ 0 };
        T           expectedFft0{ 0., 0. };
        double      expectedPeakAmplitude{ 0. };
        for (std::size_t iT = 0; iT < 5; iT++) {
            if (iT == 0) {
                std::fill(signal.begin(), signal.end(), T(0., 0.));
                expectedFft0          = { 0., 0. };
                expectedPeakAmplitude = 0.;
            } else if (iT == 1) {
                std::fill(signal.begin(), signal.end(), T(1., 0.));
                expectedFft0          = { 16., 0. };
                expectedPeakAmplitude = 2.;
            } else if (iT == 2) {
                std::fill(signal.begin(), signal.end(), T(1., 1.));
                expectedFft0          = { 16., 16. };
                expectedPeakAmplitude = std::sqrt(8.);
            } else if (iT == 3) {
                std::iota(signal.begin(), signal.end(), 1);
                expectedFft0          = { 136., 0. };
                expectedPeakAmplitude = 17.;
            } else if (iT == 4) {
                int i = 0;
                std::generate(signal.begin(), signal.end(), [&i] { return T(static_cast<typename T::value_type>(i++ % 2), 0.); });
                expectedFft0          = { 8., 0. };
                expectedPeakAmplitude = 1.;
            }

            fft1.inputHistory.push_back_bulk(signal.begin(), signal.end());
            fft1.prepareInput();
            fft1.computeFft();
            fft1.computeMagnitudeSpectrum();

            const auto peakIndex{ static_cast<std::size_t>(std::distance(fft1.magnitudeSpectrum.begin(), std::max_element(fft1.magnitudeSpectrum.begin(), fft1.magnitudeSpectrum.end()))) };
            const auto peakAmplitude{ fft1.magnitudeSpectrum[peakIndex] };

            expect(eq(peakIndex, expectedPeakIndex)) << fmt::format("<{}> equal peak index", type_name<T>());
            expect(approx(static_cast<double>(peakAmplitude), expectedPeakAmplitude, tolerance)) << fmt::format("<{}> equal amplitude", type_name<T>());
            expect(approx(static_cast<double>(fft1.fftwOut[0][0]), static_cast<double>(expectedFft0.real()), tolerance)) << fmt::format("<{}> equal fft[0].real()", type_name<T>());
            expect(approx(static_cast<double>(fft1.fftwOut[0][1]), static_cast<double>(expectedFft0.imag()), tolerance)) << fmt::format("<{}> equal fft[0].imag()", type_name<T>());
        }
    } | complexTypesToTest;

    "FFT process_one tests"_test = []<typename T>() {
        fft<T>                fft1{};
        constexpr std::size_t N{ 16 };
        std::ignore         = fft1.settings().set({ { "fftSize", N } });
        std::ignore         = fft1.settings().apply_staged_parameters();
        using DatasetType   = typename fft<T>::U;
        using InHistoryType = typename fft<T>::InHistoryType;

        std::vector<T> signal(N);
        std::iota(signal.begin(), signal.end(), 1);
        DataSet<DatasetType> ds1{};
        for (std::size_t i = 0; i < N; i++) ds1 = fft1.process_one(signal[i]);
        expect(equalVectors<InHistoryType, T>(std::vector(fft1.inputHistory.begin(), fft1.inputHistory.end()), signal)) << fmt::format("<{}> equal history buffer", type_name<T>());
        const auto N2 = fft1.magnitudeSpectrum.size();
        expect(equalVectors<DatasetType, DatasetType>(std::vector(ds1.signal_values.begin() + static_cast<std::ptrdiff_t>(2U * N2), ds1.signal_values.begin() + static_cast<std::ptrdiff_t>(3U * N2)),
                                                      fft1.magnitudeSpectrum))
                << fmt::format("<{}> equal DataSet magnitude", type_name<T>());

        for (std::size_t i = 0; i < 4; i++) {
            const auto mm = std::minmax_element(std::next(ds1.signal_values.begin(), static_cast<std::ptrdiff_t>(i * N2)),
                                                std::next(ds1.signal_values.begin(), static_cast<std::ptrdiff_t>((i + 1U) * N2)));
            if constexpr (std::is_integral_v<DatasetType>) {
                expect(eq(*mm.first, ds1.signal_ranges[i][0]));
                expect(eq(*mm.second, ds1.signal_ranges[i][1]));
            } else {
                constexpr DatasetType tolerance{ static_cast<DatasetType>(0.000001) };
                expect(approx(*mm.first, ds1.signal_ranges[i][0], tolerance));
                expect(approx(*mm.second, ds1.signal_ranges[i][1], tolerance));
            }
        }

        std::iota(signal.begin(), signal.end(), N + 1);
        for (std::size_t i = 0; i < N; i++) ds1 = fft1.process_one(signal[i]);
        expect(equalVectors<InHistoryType, T>(std::vector(fft1.inputHistory.begin(), fft1.inputHistory.end()), signal)) << fmt::format("<{}> equal history buffer", type_name<T>());
        expect(equalVectors<DatasetType, DatasetType>(std::vector(ds1.signal_values.begin() + static_cast<std::ptrdiff_t>(2U * N2), ds1.signal_values.begin() + static_cast<std::ptrdiff_t>(3U * N2)),
                                                      fft1.magnitudeSpectrum))
                << fmt::format("<{}> equal DataSet magnitude", type_name<T>());
    } | typesToTest;

    "FFT types tests"_test = [] {
        expect(std::is_same_v<fft<std::complex<float>>::U, float>) << "output type must be float";
        expect(std::is_same_v<fft<std::complex<double>>::U, double>) << "output type must be double";
        expect(std::is_same_v<fft<float>::U, float>) << "output type must be float";
        expect(std::is_same_v<fft<double>::U, double>) << "output type must be double";
        expect(std::is_same_v<fft<int>::U, int>) << "output type must be int";
        expect(std::is_same_v<fft<unsigned int>::U, int>) << "output type must be int";
        expect(std::is_same_v<fft<int64_t>::U, int64_t>) << "output type must be int64_t";
        expect(std::is_same_v<fft<uint64_t>::U, int64_t>) << "output type must be int64_t";
    };

    "FFT fftw types tests"_test = [] {
        testFftwTypes<std::complex<float>, fftwf_complex, fftwf_complex, fftwf_plan>();
        testFftwTypes<std::complex<double>, fftw_complex, fftw_complex, fftw_plan>();
        testFftwTypes<float, float, fftwf_complex, fftwf_plan>();
        testFftwTypes<double, double, fftw_complex, fftw_plan>();
        testFftwTypes<int, float, fftwf_complex, fftwf_plan>();
        testFftwTypes<unsigned int, float, fftwf_complex, fftwf_plan>();
    };

    "FFT flow graph example"_test = [] {
        // This test checks how fftw works if one creates and destroys several fft blocks in different graph flows
        using namespace boost::ut;
        using Scheduler      = fair::graph::scheduler::simple<>;
        auto      threadPool = std::make_shared<fair::thread_pool::BasicThreadPool>("custom pool", fair::thread_pool::CPU_BOUND, 2, 2);

        fg::graph flow1;
        auto     &source1 = flow1.make_node<CountSource<double>>();
        auto     &fft1    = flow1.make_node<fft<double>>({ { "fftSize", 16 } });
        std::ignore       = flow1.connect<"out">(source1).to<"in">(fft1);
        auto sched1       = Scheduler(std::move(flow1), threadPool);

        // run 2 times to check potential memory problems
        for (int i = 0; i < 2; i++) {
            fg::graph flow2;
            auto     &source2 = flow2.make_node<CountSource<double>>();
            auto     &fft2    = flow2.make_node<fft<double>>({ { "fftSize", 16 } });
            std::ignore       = flow2.connect<"out">(source2).to<"in">(fft2);
            auto sched2       = Scheduler(std::move(flow2), threadPool);
            sched2.run_and_wait();
            expect(approx(source2.count, source2.nSamples, 1e-4));
        }

        sched1.run_and_wait();
        expect(approx(source1.count, source1.nSamples, 1e-4));
    };

    "FFT window function tests"_test = []<typename T>() {
        fft<T> fft1{};
        using PrecisionType = typename fft<T>::PrecisionType;
        using InHistoryType = typename fft<T>::InHistoryType;
        constexpr PrecisionType     tolerance{ PrecisionType(0.00001) };
        constexpr std::size_t       N{ 8 };

        std::vector<WindowFunction> testCases = { WindowFunction::None,    WindowFunction::Rectangular,    WindowFunction::Hamming,
                                                  WindowFunction::Hann,    WindowFunction::HannExp,        WindowFunction::Blackman,
                                                  WindowFunction::Nuttall, WindowFunction::BlackmanHarris, WindowFunction::BlackmanNuttall,
                                                  WindowFunction::FlatTop, WindowFunction::Exponential };

        for (const auto &t : testCases) {
            std::ignore = fft1.settings().set({ { "fftSize", N } });
            std::ignore = fft1.settings().set({ { "window", static_cast<int>(t) } });
            std::ignore = fft1.settings().apply_staged_parameters();

            std::vector<T> signal(N);
            if constexpr (ComplexType<T>) {
                typename T::value_type i = 0.;
                std::generate(signal.begin(), signal.end(), [&i] {
                    i = i + static_cast<typename T::value_type>(1.);
                    return T(i, i);
                });
            } else {
                std::iota(signal.begin(), signal.end(), 1.);
            }
            for (const auto s : signal) fft1.inputHistory.push_back(static_cast<InHistoryType>(s));
            fft1.prepareInput();

            expect(eq(fft1.fftSize, N)) << fmt::format("<{}> equal fft size", type_name<T>());
            expect(eq(fft1.windowVector.size(), (t == WindowFunction::None) ? 0 : N)) << fmt::format("<{}> equal window vector size", type_name<T>());
            expect(eq(fft1.window, static_cast<int>(t))) << fmt::format("<{}> equal window function", type_name<T>());

            std::vector<double> windowFunc = createWindowFunction<double>(t, N);
            for (std::size_t i = 0; i < N; i++) {
                if constexpr (ComplexType<T>) {
                    const typename T::value_type expValue = (t == WindowFunction::None) ? signal[i].real() : signal[i].real() * static_cast<typename T::value_type>(windowFunc[i]);
                    expect(approx(fft1.fftwIn.get()[i][0], expValue, tolerance)) << fmt::format("<{}> equal fftwIn complex.real", type_name<T>());
                    expect(approx(fft1.fftwIn.get()[i][1], expValue, tolerance)) << fmt::format("<{}> equal fftwIn complex.imag", type_name<T>());
                } else {
                    const PrecisionType expValue = (t == WindowFunction::None) ? static_cast<PrecisionType>(signal[i])
                                                                               : static_cast<PrecisionType>(signal[i]) * static_cast<PrecisionType>(windowFunc[i]);
                    expect(approx(fft1.fftwIn.get()[i], expValue, tolerance)) << fmt::format("<{}> equal fftwIn", type_name<T>());
                }
            }
        }
    } | typesToTest;

    "FFT window tests"_test = []<typename T>() {
        // Expected value for size 8
        std::vector<double> None8{};
        std::vector<double> Rectangular8{ 1., 1., 1., 1., 1., 1., 1., 1. };
        std::vector<double> Hamming8{ 0.07671999999999995, 0.2119312255330421, 0.53836, 0.8647887744669578, 1.0, 0.8647887744669578, 0.5383600000000001, 0.21193122553304222 };
        std::vector<double> Hann8{ 0., 0.1882550990706332, 0.6112604669781572, 0.9504844339512095, 0.9504844339512095, 0.6112604669781573, 0.1882550990706333, 0. };
        std::vector<double> Blackman8{ -1.3877787807814457E-17, 0.09045342435412804, 0.45918295754596355, 0.9203636180999081,
                                       0.9203636180999083,      0.45918295754596383, 0.09045342435412812, -1.3877787807814457E-17 };
        std::vector<double> BlackmanHarris8{ 6.0000000000001025E-5, 0.03339172347815117, 0.332833504298565,   0.8893697722232837,
                                             0.8893697722232838,    0.3328335042985652,  0.03339172347815122, 6.0000000000001025E-5 };
        std::vector<double> BlackmanNuttall8{ 3.628000000000381E-4, 0.03777576895352025, 0.34272761996881945, 0.8918518610776603,
                                              0.8918518610776603,   0.3427276199688196,  0.0377757689535203,  3.628000000000381E-4 };
        std::vector<double> Exponential8{ 1., 1.0425469051899914, 1.086904049521229, 1.1331484530668263, 1.1813604128656459, 1.2316236423470497, 1.2840254166877414, 1.338656724353094 };
        std::vector<double> FlatTop8{ 0.004000000000000087, -0.16964240541774014, 0.04525319347985671,  3.622389211937882,
                                      3.6223892119378833,   0.04525319347985735,  -0.16964240541774012, 0.004000000000000087 };
        std::vector<double> HannExp8{ 0., 0.6112604669781572, 0.9504844339512096, 0.18825509907063334, 0.18825509907063315, 0.9504844339512096, 0.6112604669781574, 5.99903913064743E-32 };
        std::vector<double> Nuttall8{ -2.42861286636753E-17, 0.031142736797915613, 0.3264168059086425,   0.8876284572934416,
                                      0.8876284572934416,    0.32641680590864275,  0.031142736797915654, -2.42861286636753E-17 };

        // check all windows for unwanted changes
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::None, 8), None8)) << fmt::format("<{}> equal None[8] vectors", type_name<T>());
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::Rectangular, 8), Rectangular8)) << fmt::format("<{}> equal Rectangular[8] vectors", type_name<T>());
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::Hamming, 8), Hamming8)) << fmt::format("<{}> equal Hamming[8] vectors", type_name<T>());
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::Hann, 8), Hann8)) << fmt::format("<{}> equal Hann[8] vectors", type_name<T>());
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::Blackman, 8), Blackman8)) << fmt::format("<{}> equal Blackman[8] vectors", type_name<T>());
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::BlackmanHarris, 8), BlackmanHarris8)) << fmt::format("<{}> equal BlackmanHarris[8] vectors", type_name<T>());
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::BlackmanNuttall, 8), BlackmanNuttall8)) << fmt::format("<{}> equal BlackmanNuttall[8] vectors", type_name<T>());
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::Exponential, 8), Exponential8)) << fmt::format("<{}> equal Exponential[8] vectors", type_name<T>());
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::FlatTop, 8), FlatTop8)) << fmt::format("<{}> equal FlatTop[8] vectors", type_name<T>());
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::HannExp, 8), HannExp8)) << fmt::format("<{}> equal HannExp[8] vectors", type_name<T>());
        expect(equalVectors<T, double>(createWindowFunction<T>(WindowFunction::Nuttall, 8), Nuttall8)) << fmt::format("<{}> equal Nuttall[8] vectors", type_name<T>());

        // test zero length
        expect(eq(createWindowFunction<T>(WindowFunction::None, 0).size(), 0u)) << fmt::format("<{}> zero size None[8] vectors", type_name<T>());
        expect(eq(createWindowFunction<T>(WindowFunction::Rectangular, 0).size(), 0u)) << fmt::format("<{}> zero size Rectangular[8] vectors", type_name<T>());
        expect(eq(createWindowFunction<T>(WindowFunction::Hamming, 0).size(), 0u)) << fmt::format("<{}> zero size Hamming[8] vectors", type_name<T>());
        expect(eq(createWindowFunction<T>(WindowFunction::Hann, 0).size(), 0u)) << fmt::format("<{}> zero size Hann[8] vectors", type_name<T>());
        expect(eq(createWindowFunction<T>(WindowFunction::Blackman, 0).size(), 0u)) << fmt::format("<{}> zero size Blackman[8] vectors", type_name<T>());
        expect(eq(createWindowFunction<T>(WindowFunction::BlackmanHarris, 0).size(), 0u)) << fmt::format("<{}> zero size BlackmanHarris[8] vectors", type_name<T>());
        expect(eq(createWindowFunction<T>(WindowFunction::BlackmanNuttall, 0).size(), 0u)) << fmt::format("<{}> zero size BlackmanNuttall[8] vectors", type_name<T>());
        expect(eq(createWindowFunction<T>(WindowFunction::Exponential, 0).size(), 0u)) << fmt::format("<{}> zero size Exponential[8] vectors", type_name<T>());
        expect(eq(createWindowFunction<T>(WindowFunction::FlatTop, 0).size(), 0u)) << fmt::format("<{}> zero size FlatTop[8] vectors", type_name<T>());
        expect(eq(createWindowFunction<T>(WindowFunction::HannExp, 0).size(), 0u)) << fmt::format("<{}> zero size HannExp[8] vectors", type_name<T>());
        expect(eq(createWindowFunction<T>(WindowFunction::Nuttall, 0).size(), 0u)) << fmt::format("<{}> zero size Nuttall[8] vectors", type_name<T>());
    } | floatingTypesToTest;

    "FFT wisdom import/export tests"_test = []() {
        fft<double> fft1{};

        std::string wisdomString1 = fft1.exportWisdomToString();
        fft1.forgetWisdom();
        int importOk = fft1.importWisdomFromString(wisdomString1);
        expect(eq(importOk, 1)) << "Wisdom import from string.";
        std::string wisdomString2 = fft1.exportWisdomToString();

        // lines are not always at the same order thus it is hard to compare
        // expect(eq(wisdomString1, wisdomString2)) << "Wisdom strings are the same.";
    };
};

int
main() { /* not needed for UT */
}