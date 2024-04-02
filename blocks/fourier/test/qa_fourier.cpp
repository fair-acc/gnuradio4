#include <numbers>

#include <boost/ut.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fftw.hpp>

#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/fourier/fft.hpp>

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

template<typename T, typename U = T>
bool
equalVectors(const std::vector<T> &v1, const std::vector<U> &v2, double tolerance = std::is_same_v<T, double> ? 1.e-5 : 1e-4) {
    if (v1.size() != v2.size()) {
        return false;
    }
    if constexpr (gr::meta::complex_like<T>) {
        return std::equal(v1.begin(), v1.end(), v2.begin(), [&tolerance](const auto &l, const auto &r) {
            return std::abs(l.real() - r.real()) < static_cast<typename T::value_type>(tolerance) && std::abs(l.imag() - r.imag()) < static_cast<typename T::value_type>(tolerance);
        });
    } else {
        return std::equal(v1.begin(), v1.end(), v2.begin(), [&tolerance](const auto &l, const auto &r) { return std::abs(static_cast<double>(l) - static_cast<double>(r)) < tolerance; });
    }
}

template<typename T, typename U>
void
equalDataset(const gr::blocks::fft::FFT<T, gr::DataSet<U>> &fftBlock, const gr::DataSet<U> &ds1, float sample_rate) {
    using namespace boost::ut;
    using namespace boost::ut::reflection;

    const U tolerance = U(0.0001);

    const auto N    = fftBlock._magnitudeSpectrum.size();
    auto const freq = static_cast<U>(sample_rate) / static_cast<U>(fftBlock.fftSize);
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
        using InType  = T::InType;
        using OutType = T::OutType;

        constexpr gr::Size_t N{ 16 };
        constexpr float      sample_rate{ 1.f };
        FFT<InType, OutType> fftBlock({ { "fftSize", N }, { "sample_rate", sample_rate } });
        std::ignore = fftBlock.settings().applyStagedParameters();

        expect(eq(fftBlock.algorithm, gr::meta::type_name<algorithm::FFT<InType, std::complex<typename OutType::value_type>>>()));

        std::vector<InType> signal(N);
        std::iota(signal.begin(), signal.end(), 1);
        std::vector<OutType> v{ OutType() };
        std::span<OutType>   outSpan(v);

        expect(gr::work::Status::OK == fftBlock.processBulk(signal, outSpan));
        equalDataset(fftBlock, v[0], sample_rate);
    } | AllTypesToTest{};

    "FFT block types tests"_test = [] {
        expect(std::is_same_v<FFT<std::complex<float>>::value_type, float>) << "output type must be float";
        expect(std::is_same_v<FFT<std::complex<double>>::value_type, double>) << "output type must be float";
        expect(std::is_same_v<FFT<float>::value_type, float>) << "output type must be float";
        expect(std::is_same_v<FFT<double>::value_type, double>) << "output type must be float";
        expect(std::is_same_v<FFT<std::complex<float>, gr::DataSet<double>>::value_type, double>) << "output type must be double";
        expect(std::is_same_v<FFT<float, gr::DataSet<double>>::value_type, double>) << "output type must be double";
    };

    "FFT flow graph example"_test = [] {
        // This test checks how fftw works if one creates and destroys several fft blocks in different graph flows
        using namespace boost::ut;
        using Scheduler      = gr::scheduler::Simple<>;
        auto      threadPool = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 2);
        gr::Graph flow1;
        auto     &source1 = flow1.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({ { "n_samples_max", static_cast<gr::Size_t>(1024) }, { "mark_tag", false } });
        auto     &fftBlock = flow1.emplaceBlock<FFT<float>>({ { "fftSize", static_cast<gr::Size_t>(16) } });
        expect(eq(gr::ConnectionResult::SUCCESS, flow1.connect<"out">(source1).to<"in">(fftBlock)));
        auto sched1 = Scheduler(std::move(flow1), threadPool);

        // run 2 times to check potential memory problems
        for (int i = 0; i < 2; i++) {
            gr::Graph flow2;
            auto     &source2 = flow2.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({ { "n_samples_max", static_cast<gr::Size_t>(1024) }, { "mark_tag", false } });
            auto     &fft2    = flow2.emplaceBlock<FFT<float>>({ { "fftSize", static_cast<gr::Size_t>(16) } });
            expect(eq(gr::ConnectionResult::SUCCESS, flow2.connect<"out">(source2).to<"in">(fft2)));
            auto sched2 = Scheduler(std::move(flow2), threadPool);
            sched2.runAndWait();
            expect(eq(source2.n_samples_produced, source2.n_samples_max));
        }
        sched1.runAndWait();
        expect(eq(source1.n_samples_produced, source1.n_samples_max));
    };

    "window function tests"_test = []<typename T>() {
        using InType  = T::InType;
        using OutType = T::OutType;

        FFT<InType, OutType> fftBlock{};

        using value_type = OutType::value_type;
        constexpr value_type tolerance{ value_type(0.00001) };

        constexpr gr::Size_t N{ 8 };
        for (const auto &[window, windowName] : magic_enum::enum_entries<gr::algorithm::window::Type>()) {
            std::ignore = fftBlock.settings().set({ { "fftSize", N } });
            std::ignore = fftBlock.settings().set({ { "window", std::string(windowName) } });
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

            expect(eq(fftBlock.fftSize, N)) << fmt::format("<{}> equal fft size", type_name<T>());
            expect(eq(fftBlock._window.size(), N)) << fmt::format("<{}> equal window vector size", type_name<T>());
            expect(eq(fftBlock.window.value, magic_enum::enum_name(window))) << fmt::format("<{}> equal window function", type_name<T>());

            std::vector<value_type> windowFunc = gr::algorithm::window::create<value_type>(window, N);
            for (std::size_t i = 0; i < N; i++) {
                if constexpr (gr::meta::complex_like<InType>) {
                    const auto expValue = static_cast<value_type>(signal[i].real()) * windowFunc[i];
                    expect(approx(fftBlock._inData[i].real(), expValue, tolerance)) << fmt::format("<{}> equal fftwIn complex.real", type_name<T>());
                    expect(approx(fftBlock._inData[i].imag(), expValue, tolerance)) << fmt::format("<{}> equal fftwIn complex.imag", type_name<T>());
                } else {
                    const value_type expValue = static_cast<value_type>(signal[i]) * static_cast<value_type>(windowFunc[i]);
                    expect(approx(fftBlock._inData[i], expValue, tolerance)) << fmt::format("<{}> equal fftwIn", type_name<T>());
                }
            }
        }
    } | AllTypesToTest{};
};

int
main() { /* not needed for UT */
}
