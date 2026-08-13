#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <numeric>
#include <span>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetEstimators.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetHelper.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetMath.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>

#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/fourier/fft.hpp>

template<typename T>
std::vector<T> generateSineSample(std::size_t N, float sample_rate, float frequency, float amplitude) {
    std::vector<T> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        if constexpr (gr::meta::complex_like<T>) { // generate complex-valued sine wave -> should appear above 0 Hz (no negative component)
            float phase = 2.f * std::numbers::pi_v<float> * frequency * static_cast<float>(i) / sample_rate;
            signal[i]   = {static_cast<typename T::value_type>(amplitude * std::sin(phase)), static_cast<typename T::value_type>(-amplitude * std::cos(phase))};
        } else { // generate real-valued sine wave -> should appear above 0 Hz
            signal[i] = static_cast<T>(amplitude * std::sin(2.f * std::numbers::pi_v<float> * frequency * static_cast<float>(i) / sample_rate));
        }
    }
    return signal;
}

template<typename TInput, typename TOutput>
struct TestTypes {
    using InType  = TInput;
    using OutType = TOutput;
};

// drives the block's processBulk directly via manually wired ports (no scheduler): the unified FFT block
// consumes/produces InputSpanLike/OutputSpanLike spans, not plain std::span, so a bare processBulk(signal,
// outSpan) call is no longer well-formed
template<typename TInput, typename TOutput>
TOutput runFftBlock(gr::blocks::fft::FFT<TInput, TOutput>& fftBlock, const std::vector<TInput>& signal) {
    using namespace boost::ut;

    gr::PortOut<TInput> srcOut;
    gr::PortIn<TInput>  fftIn;
    expect(srcOut.connect(fftIn).has_value());
    {
        auto wspan = srcOut.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(signal.size());
        expect(eq(wspan.size(), signal.size()));
        std::ranges::copy(signal, wspan.begin());
        wspan.publish(signal.size());
    }
    auto inSpan = fftIn.template get<gr::SpanReleasePolicy::ProcessAll>(signal.size());

    gr::PortOut<TOutput> fftOutPort;
    gr::PortIn<TOutput>  sinkIn;
    expect(fftOutPort.connect(sinkIn).has_value());
    {
        auto outSpan = fftOutPort.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
        expect(gr::work::Status::OK == fftBlock.processBulk(inSpan, outSpan));
    } // scope so the WriterSpan's destructor publishes before the read below

    auto readBack = sinkIn.template get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
    expect(eq(readBack.size(), 1UZ));
    return readBack[0];
}

// D1's negative case: whether FFT<T, U> even names a type is only SFINAE-detectable inside a template's own
// substitution (a bare `requires { typename FFT<ConcreteT, ConcreteU>; }` with no dependent parameters is a
// hard constraint-failure error, not a "not satisfied" result) -- so the check needs this dependent wrapper
template<typename T, typename U>
constexpr bool isWellFormedFFT = requires { typename gr::blocks::fft::FFT<T, U>; };

const boost::ut::suite<"Fourier Transforms"> fftTests = [] {
    using namespace boost::ut;
    using namespace gr::blocks::fft;
    using namespace boost::ut::reflection;
    using gr::DataSet;

    // matched T/DataSet<P> precision pairs only -- D1 (fft.hpp) makes a precision mismatch (e.g.
    // complex<double> -> DataSet<float>) a hard compile error, so the old mixed-precision cases no
    // longer form a type at all; see "FFT block types tests" below for that guard
    using AllTypesToTest = std::tuple<
        // complex input, matching in/out precision
        TestTypes<std::complex<float>, DataSet<float>>, TestTypes<std::complex<double>, DataSet<double>>,
        // real input, matching in/out precision
        TestTypes<float, DataSet<float>>, TestTypes<double, DataSet<double>>>;

    "FFT processBulk tests"_test = []<typename T>() {
        using InType    = T::InType;
        using OutType   = T::OutType;
        using ValueType = typename OutType::value_type;

        constexpr gr::Size_t N{256};
        constexpr float      sample_rate{1.f};
        constexpr float      testFrequency{0.1f * sample_rate};
        FFT<InType, OutType> fftBlock({{"fft_size", N}, {"sample_rate", sample_rate}, {"output_in_db", true}});
        fftBlock.init(fftBlock.progress);

        std::vector<InType> signal  = generateSineSample<InType>(N, sample_rate, testFrequency, 1.f);
        const OutType       dataSet = runFftBlock(fftBlock, signal);

        std::expected<void, gr::Error> dsCheck = gr::dataset::checkConsistency(dataSet, std::format("TestDataSet({} -> {})", gr::meta::type_name<InType>(), gr::meta::type_name<OutType>()));
        expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;

        const ValueType tolerance = ValueType(0.0001);

        const auto N_mag = fftBlock._magnitudeSpectrum.size();
        auto const freq  = static_cast<ValueType>(sample_rate) / static_cast<ValueType>(fftBlock.fft_size);
        expect(ge(dataSet.axisValues(0UZ).size(), dataSet.signalValues(0UZ).size())) << std::format("<{}> DataSet axis size {} vs. signal size {}", type_name<T>(), dataSet.axisValues(0UZ).size(), dataSet.signalValues(0UZ).size());
        expect(ge(dataSet.signalValues(0UZ).size(), N_mag)) << std::format("<{}> DataSet signal length {} vs. magnitude size {}", type_name<T>(), dataSet.signalValues(0UZ).size(), N_mag);
        if (N_mag == fftBlock.fft_size) { // complex input
            expect(approx(dataSet.axisValues(0UZ).front(), -(static_cast<ValueType>(N_mag) / ValueType(2.f)) * freq, tolerance)) << std::format("<{}> equal DataSet frequency[0]", type_name<T>());
            expect(approx(dataSet.axisValues(0UZ).back(), (static_cast<ValueType>(N_mag) / ValueType(2.f) - ValueType(1.f)) * freq, tolerance)) << std::format("<{}> equal DataSet frequency[0]", type_name<T>());
        } else { // real input
            expect(approx(dataSet.axisValues(0UZ).front(), 0 * freq, tolerance)) << std::format("<{}> equal DataSet frequency[0]", type_name<T>());
            expect(approx(dataSet.axisValues(0UZ).back(), (static_cast<ValueType>(N_mag) - ValueType(1.f)) * freq, tolerance)) << std::format("<{}> equal DataSet frequency[0]", type_name<T>());
        };

        expect(gr::test::eq_collections(dataSet.signalValues(0UZ), fftBlock._magnitudeSpectrum)) << std::format("<{}> equal DataSet magnitude", type_name<T>());
        expect(gr::test::eq_collections(dataSet.signalValues(1UZ), fftBlock._phaseSpectrum)) << std::format("<{}> equal DataSet phase", type_name<T>());
        if (N_mag == fftBlock.fft_size) { // complex input: full spectrum, fftshifted like magnitude/phase
            auto shiftedRe = std::views::iota(0UZ, N_mag) | std::views::transform([&fftBlock, N_mag](std::size_t i) { return fftBlock._outData[gr::algorithm::fft::fftShiftIndex(i, N_mag)].real(); });
            auto shiftedIm = std::views::iota(0UZ, N_mag) | std::views::transform([&fftBlock, N_mag](std::size_t i) { return fftBlock._outData[gr::algorithm::fft::fftShiftIndex(i, N_mag)].imag(); });
            expect(gr::test::approx_collections(dataSet.signalValues(2UZ), shiftedRe, tolerance)) << std::format("<{}> equal DataSet FFT real output", type_name<T>());
            expect(gr::test::approx_collections(dataSet.signalValues(3UZ), shiftedIm, tolerance)) << std::format("<{}> equal DataSet FFT imaginary output", type_name<T>());
        } else { // real input: DC..Nyquist inclusive, same natural bin order as magnitude/phase
            expect(gr::test::approx_collections(dataSet.signalValues(2UZ), std::span{fftBlock._outData}.first(N_mag) | std::views::transform([](const auto& c) { return c.real(); }), tolerance)) << std::format("<{}> equal DataSet FFT real output", type_name<T>());
            expect(gr::test::approx_collections(dataSet.signalValues(3UZ), std::span{fftBlock._outData}.first(N_mag) | std::views::transform([](const auto& c) { return c.imag(); }), tolerance)) << std::format("<{}> equal DataSet FFT imaginary output", type_name<T>());
        }

        // convention-independent coherence check: reconstructing magnitude/phase from the DataSet's own Re/Im at
        // the same bin must reproduce them, whichever layout convention is in play
        for (std::size_t i = 0UZ; i < N_mag; ++i) {
            const ValueType re        = dataSet.signalValues(2UZ)[i];
            const ValueType im        = dataSet.signalValues(3UZ)[i];
            const ValueType linearMag = std::hypot(re, im) * ValueType(2) / static_cast<ValueType>(fftBlock.fft_size);
            if (linearMag > ValueType(0)) {
                const ValueType expectedMagDb = ValueType(20) * std::log10(linearMag);
                expect(approx(dataSet.signalValues(0UZ)[i], expectedMagDb, ValueType(1e-2))) << std::format("<{}> magnitude vs. Re/Im mismatch at bin {}", type_name<T>(), i);
            }
            expect(approx(dataSet.signalValues(1UZ)[i], std::atan2(im, re), tolerance)) << std::format("<{}> phase vs. Re/Im mismatch at bin {}", type_name<T>(), i);
        }

        for (std::size_t i = 0UZ; i < dataSet.size(); i++) {
            const auto [min, max] = std::ranges::minmax_element(dataSet.signalValues(i));
            expect(approx(*min, dataSet.signalRange(i).min, tolerance)) << std::format("signal '{}' min mismatch: LHS={} vs RHS={}", dataSet.signalName(i), *min, dataSet.signalRange(i).min);
            expect(approx(*max, dataSet.signalRange(i).max, tolerance)) << std::format("signal '{}' max mismatch: LHS={} vs RHS={}", dataSet.signalName(i), *max, dataSet.signalRange(i).max);
        }

        // check for matching test frequency peak
        ValueType peak = gr::dataset::estimators::getLocationMaximumGaussInterpolated(dataSet);
        expect(approx(peak, ValueType(testFrequency), ValueType(1) / ValueType(N_mag))) << "detected test frequency mismatch";

        std::println("\nplot magnitude spectrum for case: {}->{}", gr::meta::type_name<InType>(), gr::meta::type_name<OutType>());
        gr::dataset::draw(dataSet, {.chart_width = 130UZ, .chart_height = 28UZ}, 0UZ);
    } | AllTypesToTest{};

    "FFT block types tests"_test = [] {
        static_assert(std::is_same_v<FFT<std::complex<float>, gr::DataSet<float>>::value_type, float>, "output type must be float");
        static_assert(std::is_same_v<FFT<std::complex<double>, gr::DataSet<double>>::value_type, double>, "output type must be double");
        static_assert(std::is_same_v<FFT<float, gr::DataSet<float>>::value_type, float>, "output type must be float");
        static_assert(std::is_same_v<FFT<double, gr::DataSet<double>>::value_type, double>, "output type must be double");

        // D1: a T/DataSet<P> precision mismatch is a hard compile error, not a silently-wrong block
        static_assert(!isWellFormedFFT<float, DataSet<double>>, "mismatched real precision must not compile");
        static_assert(!isWellFormedFFT<double, DataSet<float>>, "mismatched real precision must not compile");
        static_assert(!isWellFormedFFT<std::complex<float>, DataSet<double>>, "mismatched complex precision must not compile");
        static_assert(!isWellFormedFFT<std::complex<double>, DataSet<float>>, "mismatched complex precision must not compile");
        // the unconstrained stream-mode default (U = complex<complex<float>>) this closes off entirely
        static_assert(!isWellFormedFFT<std::complex<float>, std::complex<std::complex<float>>>, "stream mode requires T to be floating-point");
    };

    "FFT flow graph example"_test = [] {
        // This test checks how the FFT block works if one creates and destroys several fft blocks in different graph flows
        using namespace boost::ut;
        using Scheduler = gr::scheduler::Simple<>;
        gr::Graph flow1;
        auto&     source1  = flow1.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(1024)}, {"mark_tag", false}});
        auto&     fftBlock = flow1.emplaceBlock<FFT<float, DataSet<float>>>({{"fft_size", static_cast<gr::Size_t>(16)}});
        expect(flow1.connect<"out", "in">(source1, fftBlock).has_value());
        Scheduler sched1;
        ;
        if (auto ret = sched1.exchange(std::move(flow1)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }

        // run 2 times to check potential memory problems
        for (int i = 0; i < 2; i++) {
            gr::Graph flow2;
            auto&     source2 = flow2.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(1024)}, {"mark_tag", false}});
            auto&     fft2    = flow2.emplaceBlock<FFT<float, DataSet<float>>>({{"fft_size", static_cast<gr::Size_t>(16)}});
            expect(flow2.connect<"out", "in">(source2, fft2).has_value());
            Scheduler sched2;
            ;
            if (auto ret = sched2.exchange(std::move(flow2)); !ret) {
                throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
            }
            expect(sched2.runAndWait().has_value());
            expect(eq(source2._nSamplesProduced, source2.n_samples_max));
        }
        expect(sched1.runAndWait().has_value());
        expect(eq(source1._nSamplesProduced, source1.n_samples_max));
    };

    "window function tests"_test = []<typename T>() {
        using InType  = T::InType;
        using OutType = T::OutType;

        FFT<InType, OutType> fftBlock{};

        using value_type = OutType::value_type;
        constexpr value_type tolerance{value_type(0.00001)};

        constexpr gr::Size_t N{8};
        constexpr auto       kAllWindows = gr::meta::enumValues<gr::algorithm::window::Type>();
        for (const auto& window : kAllWindows) {
            const std::string windowName{gr::meta::enumName(window).value_or("")};
            expect(fftBlock.settings().set({{"fft_size", N}, {"window", windowName}}).empty());
            expect(fftBlock.settings().activateContext() != std::nullopt);
            std::ignore = fftBlock.settings().applyStagedParameters();

            std::vector<InType> signal(N);
            if constexpr (gr::meta::complex_like<InType>) {
                typename InType::value_type i = 0.;
                std::ranges::generate(signal.begin(), signal.end(), [&i] {
                    i = i + static_cast<typename InType::value_type>(1.);
                    return InType(i, i);
                });
            } else {
                std::iota(signal.begin(), signal.end(), 1.);
            }
            std::ignore = runFftBlock(fftBlock, signal);

            expect(eq(fftBlock.fft_size, N)) << std::format("<{}> equal fft size", type_name<T>());
            expect(eq(fftBlock.window_coefficients.size(), N)) << std::format("<{}> equal window vector size", type_name<T>());
            expect(eq(fftBlock.window.value, windowName)) << std::format("<{}> equal window function", type_name<T>());

            std::vector<value_type> windowFunc = gr::algorithm::window::create<value_type>(window, N);
            for (std::size_t i = 0; i < N; i++) {
                if constexpr (gr::meta::complex_like<InType>) {
                    const auto expValue = static_cast<value_type>(signal[i].real()) * windowFunc[i];
                    expect(approx(fftBlock._inData[i].real(), expValue, tolerance)) << std::format("<{}> equal complex.real", type_name<T>());
                    expect(approx(fftBlock._inData[i].imag(), expValue, tolerance)) << std::format("<{}> equal complex.imag", type_name<T>());
                } else {
                    const value_type expValue = static_cast<value_type>(signal[i]) * static_cast<value_type>(windowFunc[i]);
                    expect(approx(fftBlock._inData[i], expValue, tolerance)) << std::format("<{}> equal fft", type_name<T>());
                }
            }
        }
    } | AllTypesToTest{};
};

namespace {

template<typename T>
struct CollectorSink : gr::Block<CollectorSink<T>> {
    gr::PortIn<T>  in;
    std::vector<T> received;

    GR_MAKE_REFLECTABLE(CollectorSink, in);

    void processOne(const T& value) { received.push_back(value); }
};

// exact bin-centred tones: real sine sin(2*pi*bin*n/N), or complex exp(2i*pi*bin*n/N) (bin may be negative)
std::vector<float> generateRealSine(std::size_t nSamples, std::size_t bin) {
    std::vector<float> signal(nSamples);
    for (std::size_t n = 0; n < nSamples; ++n) {
        signal[n] = std::sin(2.f * std::numbers::pi_v<float> * static_cast<float>(bin) * static_cast<float>(n) / static_cast<float>(nSamples));
    }
    return signal;
}

std::vector<std::complex<float>> generateComplexTone(std::size_t nSamples, std::ptrdiff_t bin) {
    std::vector<std::complex<float>> signal(nSamples);
    for (std::size_t n = 0; n < nSamples; ++n) {
        const float phase = 2.f * std::numbers::pi_v<float> * static_cast<float>(bin) * static_cast<float>(n) / static_cast<float>(nSamples);
        signal[n]         = {std::cos(phase), std::sin(phase)};
    }
    return signal;
}

template<typename T>
std::vector<T> generateDelayedImpulse(std::size_t nSamples, std::size_t n0) {
    std::vector<T> signal(nSamples, T{});
    signal[n0] = T{1};
    return signal;
}

std::size_t peakBin(std::span<const float> magnitude) { return static_cast<std::size_t>(std::ranges::distance(magnitude.begin(), std::ranges::max_element(magnitude))); }

float coherentGain(gr::algorithm::window::Type type, std::size_t nSamples) {
    const std::vector<float> w = gr::algorithm::window::create<float>(type, nSamples);
    return std::accumulate(w.begin(), w.end(), 0.f) / static_cast<float>(nSamples);
}

bool approxRel(float actual, float expected, float relTolerance, float absFloor = 1e-4f) { return std::abs(actual - expected) <= std::max(relTolerance * std::abs(expected), absFloor); }

template<typename TInput>
gr::DataSet<float> runSpectrum(std::vector<TInput> signal, gr::property_map extraSettings = {}) {
    gr::property_map settings{{"fft_size", static_cast<gr::Size_t>(signal.size())}};
    for (const auto& [key, value] : extraSettings) {
        settings[key] = value;
    }
    gr::blocks::fft::FFT<TInput, gr::DataSet<float>> fftBlock(std::move(settings));
    fftBlock.init(fftBlock.progress);
    return runFftBlock(fftBlock, signal);
}

template<typename TInput>
gr::DataSet<float> runSpectrumGraph(std::vector<TInput> signal, gr::property_map extraSettings = {}) {
    using namespace boost::ut;
    gr::property_map settings{{"fft_size", static_cast<gr::Size_t>(signal.size())}};
    for (const auto& [key, value] : extraSettings) {
        settings[key] = value;
    }
    gr::Graph flow;
    auto&     src  = flow.emplaceBlock<gr::testing::TagSource<TInput>>({{"values", signal}, {"n_samples_max", static_cast<gr::Size_t>(signal.size())}});
    auto&     fft  = flow.emplaceBlock<gr::blocks::fft::FFT<TInput, gr::DataSet<float>>>(std::move(settings));
    auto&     sink = flow.emplaceBlock<CollectorSink<gr::DataSet<float>>>();
    expect(flow.connect<"out", "in">(src, fft).has_value());
    expect(flow.connect<"out", "in">(fft, sink).has_value());
    gr::scheduler::Simple<> sched;
    expect(sched.exchange(std::move(flow)).has_value());
    expect(sched.runAndWait().has_value());
    expect(eq(sink.received.size(), 1UZ)) << fatal;
    return sink.received.front();
}

} // namespace

// analytic replacement of the retired golden-vector parity sweep: every expectation below is a closed-form
// physical property of the DFT (peak location/amplitude, linear phase ramp of a delayed impulse, Parseval,
// Hermitian symmetry, axis frequencies), not a captured output of any implementation
const boost::ut::suite<"FFT spectrum physics"> fftPhysicsTests = [] {
    using namespace boost::ut;
    using C = std::complex<float>;
    using gr::algorithm::window::Type;

    constexpr float kRelTol = 1e-5f; // relative float tolerance for analytically exact expectations

    "bin-centred complex tone: peak bin and amplitude 2*A*coherentGain, for every window type"_test = [] {
        constexpr std::size_t N           = 256;
        constexpr std::size_t bin         = 17;
        constexpr auto        kAllWindows = gr::meta::enumValues<Type>();
        for (const Type windowType : kAllWindows) {
            const std::string        name{gr::meta::enumName(windowType).value_or("")};
            const gr::DataSet<float> ds = runSpectrum<C>(generateComplexTone(N, bin), {{"window", name}});

            // X[k] = sum w[n] * exp(2i*pi*bin*n/N) * exp(-2i*pi*k*n/N) collapses to sum(w) at k == bin exactly,
            // so with the 2/N normalisation the peak magnitude is 2*coherentGain, independent of window shape
            const auto magnitude = ds.signalValues(0UZ);
            expect(eq(magnitude.size(), N)) << name;
            expect(eq(peakBin(magnitude), N / 2 + bin)) << std::format("window {}: +bin tone must peak at N/2+bin after fftshift", name);
            expect(approxRel(magnitude[N / 2 + bin], 2.f * coherentGain(windowType, N), kRelTol)) << std::format("window {}: peak amplitude vs. 2*coherentGain", name);
        }
    };

    "negative-frequency complex tone peaks below DC (fftshift direction)"_test = [] {
        constexpr std::size_t    N   = 256;
        constexpr std::ptrdiff_t bin = -17;
        const gr::DataSet<float> ds  = runSpectrum<C>(generateComplexTone(N, bin), {{"window", std::string("Rectangular")}});
        expect(eq(peakBin(ds.signalValues(0UZ)), N / 2 - 17UZ)) << "-bin tone must land at N/2-bin: negative frequencies come first";
        expect(approxRel(ds.signalValues(0UZ)[N / 2 - 17UZ], 2.f, kRelTol));
    };

    "bin-centred real sine: peak bin exact, amplitude A*coherentGain, for every window type"_test = [] {
        constexpr std::size_t N           = 256;
        constexpr std::size_t bin         = 17;
        constexpr auto        kAllWindows = gr::meta::enumValues<Type>();
        for (const Type windowType : kAllWindows) {
            const std::string        name{gr::meta::enumName(windowType).value_or("")};
            const gr::DataSet<float> ds        = runSpectrum<float>(generateRealSine(N, bin), {{"window", name}});
            const auto               magnitude = ds.signalValues(0UZ);
            expect(eq(magnitude.size(), N / 2 + 1)) << name;
            expect(eq(peakBin(magnitude), bin)) << std::format("window {}: real sine must peak at its bin (rfft layout, DC..Nyquist)", name);
            // the negative-frequency image contributes W(2*bin)/W(0) leakage to the peak (up to ~1% for
            // windows with flat sidelobes, e.g. Hamming), so this bound is 1e-2 rather than kRelTol
            expect(approxRel(magnitude[bin], coherentGain(windowType, N), 1e-2f)) << std::format("window {}: peak amplitude vs. coherentGain", name);
        }
    };

    "delayed impulse: unwrapped phase is the exact ramp -2*pi*n0*k/N (real input)"_test = [] {
        constexpr std::size_t N  = 64;
        constexpr std::size_t n0 = 17;                                 // off any clean N/4 fraction so consecutive diffs avoid the +/-pi boundary
        for (const std::string windowName : {"Rectangular", "Hann"}) { // windowing a single-sample impulse only rescales it
            const gr::DataSet<float> ds    = runSpectrum<float>(generateDelayedImpulse<float>(N, n0), {{"window", windowName}, {"unwrap_phase", true}});
            const auto               phase = ds.signalValues(1UZ);
            expect(eq(phase.size(), N / 2 + 1)) << windowName;
            for (std::size_t k = 0; k < phase.size(); ++k) {
                const float expected = -2.f * std::numbers::pi_v<float> * static_cast<float>(n0) * static_cast<float>(k) / static_cast<float>(N);
                expect(approxRel(phase[k], expected, kRelTol)) << std::format("window {}: unwrapped phase at bin {}: {} vs. {}", windowName, k, phase[k], expected);
            }
            // several genuine wraps must have occurred, otherwise this test would not exercise unwrapPhase
            expect(lt(phase.back(), -2.f * 2.f * std::numbers::pi_v<float>)) << "stimulus did not wrap phase multiple times";
        }
    };

    "delayed impulse: unwrap runs in natural bin order, then fftshift (complex input)"_test = [] {
        constexpr std::size_t    N     = 64;
        constexpr std::size_t    n0    = 17;
        const gr::DataSet<float> ds    = runSpectrum<C>(generateDelayedImpulse<C>(N, n0), {{"window", std::string("Rectangular")}, {"unwrap_phase", true}});
        const auto               phase = ds.signalValues(1UZ);
        expect(eq(phase.size(), N));
        for (std::size_t j = 0; j < N; ++j) {
            const std::size_t k        = (j + N / 2) % N; // natural-order bin displayed at shifted position j
            const float       expected = -2.f * std::numbers::pi_v<float> * static_cast<float>(n0) * static_cast<float>(k) / static_cast<float>(N);
            expect(approxRel(phase[j], expected, kRelTol)) << std::format("shifted bin {}: {} vs. {} -- unwrap-then-shift order violated?", j, phase[j], expected);
        }

        // |X_k| = w[n0] for every bin: a delayed impulse has a flat magnitude spectrum
        const auto magnitude = ds.signalValues(0UZ);
        for (std::size_t j = 0; j < N; ++j) {
            expect(approxRel(magnitude[j], 2.f / static_cast<float>(N), kRelTol)) << std::format("impulse magnitude must be flat, bin {}", j);
        }
    };

    "without unwrap_phase the phase stays within (-pi, pi]"_test = [] {
        constexpr std::size_t    N     = 64;
        const gr::DataSet<float> ds    = runSpectrum<float>(generateDelayedImpulse<float>(N, 17), {{"window", std::string("Rectangular")}});
        const auto               phase = ds.signalValues(1UZ);
        for (std::size_t k = 0; k < phase.size(); ++k) {
            expect(le(std::abs(phase[k]), std::numbers::pi_v<float> + 1e-5f)) << std::format("raw atan2 phase out of principal range at bin {}", k);
        }
    };

    "dB output is 20*log10 of the linear magnitude; degrees are radians * 180/pi"_test = [] {
        constexpr std::size_t N      = 256;
        const auto            signal = generateRealSine(N, 17);

        const gr::DataSet<float> base      = runSpectrum<float>(signal);
        const gr::DataSet<float> inDb      = runSpectrum<float>(signal, {{"output_in_db", true}});
        const gr::DataSet<float> inDeg     = runSpectrum<float>(signal, {{"output_in_deg", true}});
        const gr::DataSet<float> unwrapRad = runSpectrum<float>(signal, {{"unwrap_phase", true}});
        const gr::DataSet<float> unwrapDeg = runSpectrum<float>(signal, {{"unwrap_phase", true}, {"output_in_deg", true}});

        for (std::size_t k = 0; k < base.signalValues(0UZ).size(); ++k) {
            const float linear = base.signalValues(0UZ)[k];
            if (linear > 1e-10f) {
                expect(approxRel(inDb.signalValues(0UZ)[k], 20.f * std::log10(linear), kRelTol, 1e-3f)) << std::format("dB relation at bin {}", k);
            }
            expect(approxRel(inDeg.signalValues(1UZ)[k], base.signalValues(1UZ)[k] * 180.f * std::numbers::inv_pi_v<float>, kRelTol)) << std::format("degree relation at bin {}", k);
            expect(approxRel(unwrapDeg.signalValues(1UZ)[k], unwrapRad.signalValues(1UZ)[k] * 180.f * std::numbers::inv_pi_v<float>, kRelTol)) << std::format("degree relation after unwrap at bin {}", k);
        }
        expect(gr::test::eq_collections(inDb.signalValues(1UZ), base.signalValues(1UZ))) << "output_in_db must not touch the phase";
        expect(gr::test::eq_collections(inDeg.signalValues(0UZ), base.signalValues(0UZ))) << "output_in_deg must not touch the magnitude";
    };

    "real input: DC and Nyquist bins are purely real (Hermitian symmetry endpoints)"_test = [] {
        constexpr std::size_t    N  = 64;
        const gr::DataSet<float> ds = runSpectrum<float>(generateRealSine(N, 17), {{"window", std::string("Hann")}});
        expect(eq(ds.signalValues(2UZ).size(), N / 2 + 1));
        expect(le(std::abs(ds.signalValues(3UZ).front()), 1e-3f)) << "Im(DC) must vanish for real input";
        expect(le(std::abs(ds.signalValues(3UZ).back()), 1e-3f)) << "Im(Nyquist) must vanish for real input";
    };

    "Parseval: spectral energy equals N times the windowed signal energy"_test = [] {
        constexpr std::size_t N = 64;

        // complex tone, rectangular window: sum |X_k|^2 == N * sum |x_n|^2 == N * N
        const gr::DataSet<float> full       = runSpectrum<C>(generateComplexTone(N, 5), {{"window", std::string("Rectangular")}});
        float                    fullEnergy = 0.f;
        for (std::size_t k = 0; k < N; ++k) {
            const float re = full.signalValues(2UZ)[k];
            const float im = full.signalValues(3UZ)[k];
            fullEnergy += re * re + im * im;
        }
        expect(approxRel(fullEnergy, static_cast<float>(N) * static_cast<float>(N), kRelTol)) << "Parseval, complex input";

        // real bin-centred sine, rectangular window: signal energy N/2, reconstructed via Hermitian symmetry
        const gr::DataSet<float> half     = runSpectrum<float>(generateRealSine(N, 5), {{"window", std::string("Rectangular")}});
        const auto               energyAt = [&half](std::size_t k) {
            const float re = half.signalValues(2UZ)[k];
            const float im = half.signalValues(3UZ)[k];
            return re * re + im * im;
        };
        float halfEnergy = energyAt(0UZ) + energyAt(N / 2);
        for (std::size_t k = 1; k < N / 2; ++k) {
            halfEnergy += 2.f * energyAt(k);
        }
        expect(approxRel(halfEnergy, static_cast<float>(N) * static_cast<float>(N) / 2.f, kRelTol)) << "Parseval, real input";
    };

    "frequency axis is physical: DC, fs/N spacing, Nyquist at fs/2 -- graph-driven"_test = [] {
        // graph-driven on purpose: staging sample_rate together with fft_size used to trigger the framework's
        // resampling rescale and compress the axis by fft_size; the block now opts out (see settingsChanged)
        constexpr std::size_t N  = 64;
        constexpr float       fs = 48000.f;

        const gr::DataSet<float> real     = runSpectrumGraph<float>(generateRealSine(N, 5), {{"sample_rate", fs}});
        const auto               realAxis = real.axisValues(0UZ);
        expect(eq(realAxis.size(), N / 2 + 1));
        expect(approxRel(realAxis.front(), 0.f, kRelTol));
        expect(approxRel(realAxis.back(), fs / 2.f, kRelTol)) << "Nyquist must sit at fs/2, not fs/(2*N)";
        expect(approxRel(realAxis[1] - realAxis[0], fs / static_cast<float>(N), kRelTol)) << "bin spacing must be fs/N";
        if (const float* metaRate = real.meta_information[0].get_if<float>("sample_rate")) {
            expect(approxRel(*metaRate, fs, kRelTol)) << "meta_information must carry the input sample rate";
        } else {
            expect(false) << "meta_information lacks sample_rate";
        }

        const gr::DataSet<float> cplx     = runSpectrumGraph<std::complex<float>>(generateComplexTone(N, 5), {{"sample_rate", fs}});
        const auto               cplxAxis = cplx.axisValues(0UZ);
        expect(eq(cplxAxis.size(), N));
        expect(approxRel(cplxAxis.front(), -fs / 2.f, kRelTol)) << "full spectrum starts at -fs/2";
        expect(approxRel(cplxAxis[N / 2], 0.f, kRelTol)) << "DC must sit at index N/2";
        expect(approxRel(cplxAxis.back(), fs / 2.f - fs / static_cast<float>(N), kRelTol)) << "full spectrum ends at fs/2 - fs/N";
    };
};

int main() { /* not needed for UT */ }
