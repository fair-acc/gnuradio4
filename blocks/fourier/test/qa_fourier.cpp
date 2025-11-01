#include <numbers>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetEstimators.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetHelper.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetMath.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>

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

const boost::ut::suite<"Fourier Transforms"> fftTests = [] {
    using namespace boost::ut;
    using namespace gr::blocks::fft;
    using namespace boost::ut::reflection;

    using AllTypesToTest = std::tuple<
        // complex input, same in-out precision
        TestTypes<std::complex<float>, DataSet<float>>, TestTypes<std::complex<double>, DataSet<double>>,
        // complex input, different in-out precision
        TestTypes<std::complex<double>, DataSet<float>>, TestTypes<std::complex<float>, DataSet<double>>,
        // real input, same in-out precision
        TestTypes<float, DataSet<float>>, TestTypes<double, DataSet<double>>,
        // real input, different in-out precision
        TestTypes<float, DataSet<double>>, TestTypes<double, DataSet<float>>>;

    "FFT processBulk tests"_test = []<typename T>() {
        using InType    = T::InType;
        using OutType   = T::OutType;
        using ValueType = typename OutType::value_type;

        constexpr gr::Size_t N{256};
        constexpr float      sample_rate{1.f};
        constexpr float      testFrequency{0.1f * sample_rate};
        FFT<InType, OutType> fftBlock({{"fftSize", N}, {"sample_rate", sample_rate}, {"outputInDb", true}});
        fftBlock.init(fftBlock.progress);

        expect(eq(fftBlock.algorithm, gr::meta::type_name<algorithm::FFT<InType, std::complex<typename OutType::value_type>>>()));

        std::vector<InType>  signal = generateSineSample<InType>(N, sample_rate, testFrequency, 1.f);
        std::vector<OutType> v{OutType()};
        std::span<OutType>   outSpan(v);

        expect(gr::work::Status::OK == fftBlock.processBulk(signal, outSpan));
        std::expected<void, gr::Error> dsCheck = dataset::checkConsistency(v[0], std::format("TestDataSet({} -> {})", gr::meta::type_name<InType>(), gr::meta::type_name<OutType>()));
        expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;

        // check for DataSet equality
        const auto& dataSet = v[0];

        const ValueType tolerance = ValueType(0.0001);

        const auto N_mag = fftBlock._magnitudeSpectrum.size();
        auto const freq  = static_cast<ValueType>(sample_rate) / static_cast<ValueType>(fftBlock.fftSize);
        expect(ge(dataSet.axisValues(0UZ).size(), dataSet.signalValues(0UZ).size())) << std::format("<{}> DataSet axis size {} vs. signal size {}", type_name<T>(), dataSet.axisValues(0UZ).size(), dataSet.signalValues(0UZ).size());
        expect(ge(dataSet.signalValues(0UZ).size(), N_mag)) << std::format("<{}> DataSet signal length {} vs. magnitude size {}", type_name<T>(), dataSet.signalValues(0UZ).size(), N_mag);
        if (N_mag == fftBlock.fftSize) { // complex input
            expect(approx(dataSet.axisValues(0UZ).front(), -(static_cast<ValueType>(N_mag) / ValueType(2.f)) * freq, tolerance)) << std::format("<{}> equal DataSet frequency[0]", type_name<T>());
            expect(approx(dataSet.axisValues(0UZ).back(), (static_cast<ValueType>(N_mag) / ValueType(2.f) - ValueType(1.f)) * freq, tolerance)) << std::format("<{}> equal DataSet frequency[0]", type_name<T>());
        } else { // real input
            expect(approx(dataSet.axisValues(0UZ).front(), 0 * freq, tolerance)) << std::format("<{}> equal DataSet frequency[0]", type_name<T>());
            expect(approx(dataSet.axisValues(0UZ).back(), (static_cast<ValueType>(N_mag) - ValueType(1.f)) * freq, tolerance)) << std::format("<{}> equal DataSet frequency[0]", type_name<T>());
        };

        expect(gr::test::eq_collections(dataSet.signalValues(0UZ), fftBlock._magnitudeSpectrum)) << std::format("<{}> equal DataSet magnitude", type_name<T>());
        expect(gr::test::eq_collections(dataSet.signalValues(1UZ), fftBlock._phaseSpectrum)) << std::format("<{}> equal DataSet phase", type_name<T>());
        expect(gr::test::approx_collections(dataSet.signalValues(2UZ), std::span{fftBlock._outData}.last(N_mag) | std::views::transform([](const auto& c) { return c.real(); }), tolerance)) << std::format("<{}> equal DataSet FFT real output", type_name<T>());
        expect(gr::test::approx_collections(dataSet.signalValues(3UZ), std::span{fftBlock._outData}.last(N_mag) | std::views::transform([](const auto& c) { return c.imag(); }), tolerance)) << std::format("<{}> equal DataSet FFT imaginary output", type_name<T>());

        for (std::size_t i = 0UZ; i < dataSet.size(); i++) {
            const auto [min, max] = std::ranges::minmax_element(dataSet.signalValues(i));
            expect(approx(*min, dataSet.signalRange(i).min, tolerance)) << std::format("signal '{}' min mismatch: LHS={} vs RHS={}", dataSet.signalName(i), *min, dataSet.signalRange(i).min);
            expect(approx(*max, dataSet.signalRange(i).max, tolerance)) << std::format("signal '{}' max mismatch: LHS={} vs RHS={}", dataSet.signalName(i), *max, dataSet.signalRange(i).max);
        }

        // check for matching test frequency peak
        ValueType peak = gr::dataset::estimators::getLocationMaximumGaussInterpolated(dataSet);
        expect(approx(peak, ValueType(testFrequency), ValueType(1) / ValueType(N_mag))) << "detected test frequency mismatch";

        // plot magnitude spectrum
        std::println("\nplot magnitude spectrum for case: {}->{}", gr::meta::type_name<InType>(), gr::meta::type_name<OutType>());
        gr::dataset::draw(dataSet, {.chart_width = 130UZ, .chart_height = 28UZ}, 0UZ);
    } | AllTypesToTest{};

    "FFT block types tests"_test = [] {
        static_assert(std::is_same_v<FFT<std::complex<float>>::value_type, float>, "output type must be float");
        static_assert(std::is_same_v<FFT<std::complex<double>>::value_type, double>, "output type must be float");
        static_assert(std::is_same_v<FFT<float>::value_type, float>, "output type must be float");
        static_assert(std::is_same_v<FFT<double>::value_type, double>, "output type must be float");
        static_assert(std::is_same_v<FFT<std::complex<float>, gr::DataSet<double>>::value_type, double>, "output type must be double");
        static_assert(std::is_same_v<FFT<float, gr::DataSet<double>>::value_type, double>, "output type must be double");
    };

    "FFT flow graph example"_test = [] {
        // This test checks how the FFT block works if one creates and destroys several fft blocks in different graph flows
        using namespace boost::ut;
        using Scheduler = gr::scheduler::Simple<>;
        gr::Graph flow1;
        auto&     source1  = flow1.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(1024)}, {"mark_tag", false}});
        auto&     fftBlock = flow1.emplaceBlock<FFT<float>>({{"fftSize", static_cast<gr::Size_t>(16)}});
        expect(eq(gr::ConnectionResult::SUCCESS, flow1.connect<"out">(source1).to<"in">(fftBlock)));
        Scheduler sched1;
        ;
        if (auto ret = sched1.exchange(std::move(flow1)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }

        // run 2 times to check potential memory problems
        for (int i = 0; i < 2; i++) {
            gr::Graph flow2;
            auto&     source2 = flow2.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(1024)}, {"mark_tag", false}});
            auto&     fft2    = flow2.emplaceBlock<FFT<float>>({{"fftSize", static_cast<gr::Size_t>(16)}});
            expect(eq(gr::ConnectionResult::SUCCESS, flow2.connect<"out">(source2).to<"in">(fft2)));
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
        for (const auto& [window, windowName] : magic_enum::enum_entries<gr::algorithm::window::Type>()) {
            expect(fftBlock.settings().set({{"fftSize", N}, {"window", std::string(windowName)}}).empty());
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
            std::vector<OutType> resultingDataSets(1);
            expect(gr::work::Status::OK == fftBlock.processBulk(signal, resultingDataSets));

            expect(eq(fftBlock.fftSize, N)) << std::format("<{}> equal fft size", type_name<T>());
            expect(eq(fftBlock._window.size(), N)) << std::format("<{}> equal window vector size", type_name<T>());
            expect(eq(fftBlock.window.value, magic_enum::enum_name(window))) << std::format("<{}> equal window function", type_name<T>());

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

int main() { /* not needed for UT */ }
