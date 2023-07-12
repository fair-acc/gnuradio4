#include <boost/ut.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

#include "blocklib/core/fft/fft.hpp"
#include <fmt/format.h>
#include <node.hpp>
#include <numbers>

namespace fg = fair::graph;

template<typename T>
std::vector<T>
generate_sin_sample(std::size_t N, double sample_rate, double frequency, double amplitude) {
    std::vector<T> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        if constexpr (gr::blocks::fft::ComplexType<T>) {
            signal[i] = { static_cast<typename T::value_type>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate)), 0. };
        } else {
            signal[i] = static_cast<T>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate));
        }
    }
    return signal;
}

template<typename T>
bool
equal_vectors(const std::vector<T> &v1, const std::vector<T> &v2, double tolerance = 1.e-6) {
    if constexpr (gr::blocks::fft::ComplexType<T>) {
        return std::equal(v1.begin(), v1.end(), v2.begin(), [&tolerance](const auto &l, const auto &r) {
            return std::abs(l.real() - r.real()) < static_cast<typename T::value_type>(tolerance) && std::abs(l.imag() - r.imag()) < static_cast<typename T::value_type>(tolerance);
        });
    } else {
        return std::equal(v1.begin(), v1.end(), v2.begin(), [&tolerance](const auto &l, const auto &r) { return std::abs(l - r) < static_cast<T>(tolerance); });
    }
}

template<typename T, typename inT, typename outT, typename pT>
void
test_fftw_types() {
    using namespace boost::ut;
    gr::blocks::fft::fft<T> fft1;
    expect(std::is_same_v<typename std::remove_pointer_t<decltype(fft1.fftw_in)>, inT>) << "";
    expect(std::is_same_v<typename std::remove_pointer_t<decltype(fft1.fftw_out)>, outT>) << "";
    expect(std::is_same_v<decltype(fft1.fftw_p), pT>) << "";
}

const boost::ut::suite _fft_tests = [] {
    using namespace boost::ut;
    using namespace gr::blocks::fft;
    using namespace boost::ut::reflection;
    std::tuple<std::complex<float>, std::complex<double>>                complex_types_to_test{};
    std::tuple<std::complex<float>, std::complex<double>, float, double> types_to_test{};

    "FFT sin tests"_test = []<typename T>() {
        fft<T>           fft1{};
        constexpr double tolerance{ 1.e-6 };
        struct TestParams {
            std::size_t N{ 1024 };           // must be power of 2
            double      sample_rate{ 128. }; // must be power of 2 (only for the unit test for easy comparison with true result)
            double      frequency{ 1. };
            double      amplitude{ 1. };
            bool        output_in_dB{ false };
        };

        std::vector<TestParams> testCases = { { 256, 128., 10., 5., false }, { 512, 4., 1., 1., false }, { 512, 32., 1., 0.1, false }, { 256, 128., 10., 5., true } };
        for (const auto &t : testCases) {
            assert(std::has_single_bit(t.N));
            assert(std::has_single_bit(static_cast<std::size_t>(t.sample_rate)));

            std::ignore = fft1.settings().set({ { "fft_size", t.N } });
            std::ignore = fft1.settings().set({ { "output_in_dB", t.output_in_dB } });
            std::ignore = fft1.settings().apply_staged_parameters();
            const auto signal{ generate_sin_sample<T>(t.N, t.sample_rate, t.frequency, t.amplitude) };

            fft1.inputHistory.push_back_bulk(signal.begin(), signal.end());
            fft1.prepare_input();
            fft1.compute_fft();
            fft1.compute_magnitude_spectrum();

            const auto peak_index{
                static_cast<std::size_t>(std::distance(fft1.magnitude_spectrum.begin(),
                                                       std::max_element(fft1.magnitude_spectrum.begin(), std::next(fft1.magnitude_spectrum.begin(), static_cast<std::ptrdiff_t>(t.N / 2u)))))
            }; // only positive frequencies from FFT
            const auto peak_amplitude = fft1.magnitude_spectrum[peak_index];
            const auto peak_frequency{ static_cast<double>(peak_index) * t.sample_rate / static_cast<double>(t.N) };

            const auto expected_amplitude = t.output_in_dB ? 20. * log10(std::abs(t.amplitude)) : t.amplitude;
            expect(approx(peak_amplitude, expected_amplitude, tolerance)) << fmt::format("<{}> equal amplitude", type_name<T>());
            expect(approx(peak_frequency, t.frequency, tolerance)) << fmt::format("<{}> equal frequency", type_name<T>());
        }
    } | types_to_test;

    "FFT pattern tests"_test = []<typename T>() {
        fft<T>                fft1{};
        constexpr double      tolerance{ 1.e-6 };
        constexpr std::size_t N = 16;
        std::ignore             = fft1.settings().set({ { "fft_size", N } });
        std::ignore             = fft1.settings().apply_staged_parameters();

        std::vector<T> signal(N);

        static_assert(N == 16, "expected values are calculated for N == 16");
        int    expected_peak_index{ 0 };
        T      expected_fft0{ 0., 0. };
        double expected_peak_amplitude{ 0. };
        for (std::size_t iT = 0; iT < 5; iT++) {
            if (iT == 0) {
                std::fill(signal.begin(), signal.end(), T(0., 0.));
                expected_fft0           = { 0., 0. };
                expected_peak_amplitude = 0.;
            } else if (iT == 1) {
                std::fill(signal.begin(), signal.end(), T(1., 0.));
                expected_fft0           = { 16., 0. };
                expected_peak_amplitude = 2.;
            } else if (iT == 2) {
                std::fill(signal.begin(), signal.end(), T(1., 1.));
                expected_fft0           = { 16., 16. };
                expected_peak_amplitude = std::sqrt(8.);
            } else if (iT == 3) {
                std::iota(signal.begin(), signal.end(), 1);
                expected_fft0           = { 136., 0. };
                expected_peak_amplitude = 17.;
            } else if (iT == 4) {
                int i = 0;
                std::generate(signal.begin(), signal.end(), [&i] { return T(static_cast<typename T::value_type>(i++ % 2), 0.); });
                expected_fft0           = { 8., 0. };
                expected_peak_amplitude = 1.;
            }

            fft1.inputHistory.push_back_bulk(signal.begin(), signal.end());
            fft1.prepare_input();
            fft1.compute_fft();
            fft1.compute_magnitude_spectrum();

            const auto peak_index{ static_cast<std::size_t>(std::distance(fft1.magnitude_spectrum.begin(), std::max_element(fft1.magnitude_spectrum.begin(), fft1.magnitude_spectrum.end()))) };
            const auto peak_amplitude{ fft1.magnitude_spectrum[peak_index] };

            expect(eq(peak_index, expected_peak_index)) << fmt::format("<{}> equal peak index", type_name<T>());
            expect(approx(peak_amplitude, expected_peak_amplitude, tolerance)) << fmt::format("<{}> equal amplitude", type_name<T>());
            expect(approx(fft1.fftw_out[0][0], expected_fft0.real(), tolerance)) << fmt::format("<{}> equal fft[0].real()", type_name<T>());
            expect(approx(fft1.fftw_out[0][1], expected_fft0.imag(), tolerance)) << fmt::format("<{}> equal fft[0].imag()", type_name<T>());
        }
    } | complex_types_to_test;

    "FFT process_one tests"_test = []<typename T>() {
        fft<T>                fft1{};
        constexpr std::size_t N = 16;
        std::ignore             = fft1.settings().set({ { "fft_size", N } });
        std::ignore             = fft1.settings().apply_staged_parameters();
        using dataset_type      = typename fft<T>::U;

        std::vector<T> signal(N);
        std::iota(signal.begin(), signal.end(), 1);
        DataSet<dataset_type> ds1{};
        for (std::size_t i = 0; i < N; i++) ds1 = fft1.process_one(signal[i]);
        expect(equal_vectors<T>(std::vector(fft1.inputHistory.begin(), fft1.inputHistory.end()), signal)) << fmt::format("<{}> equal history buffer", type_name<T>());
        const auto N2 = static_cast<int>(fft1.magnitude_spectrum.size());
        expect(equal_vectors<dataset_type>(std::vector(ds1.signal_values.begin() + 2 * N2, ds1.signal_values.begin() + 3 * N2), fft1.magnitude_spectrum))
                << fmt::format("<{}> equal DataSet magnitude", type_name<T>());

        std::iota(signal.begin(), signal.end(), N + 1);
        for (std::size_t i = 0; i < N; i++) ds1 = fft1.process_one(signal[i]);
        expect(equal_vectors<T>(std::vector(fft1.inputHistory.begin(), fft1.inputHistory.end()), signal)) << fmt::format("<{}> equal history buffer", type_name<T>());
        expect(equal_vectors<dataset_type>(std::vector(ds1.signal_values.begin() + 2 * N2, ds1.signal_values.begin() + 3 * N2), fft1.magnitude_spectrum))
                << fmt::format("<{}> equal DataSet magnitude", type_name<T>());
    } | types_to_test;

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
        test_fftw_types<std::complex<float>, fftwf_complex, fftwf_complex, fftwf_plan>();
        test_fftw_types<std::complex<double>, fftw_complex, fftw_complex, fftw_plan>();
        test_fftw_types<float, float, fftwf_complex, fftwf_plan>();
        test_fftw_types<double, double, fftw_complex, fftw_plan>();
        test_fftw_types<int, float, fftwf_complex, fftwf_plan>();
        test_fftw_types<unsigned int, float, fftwf_complex, fftwf_plan>();
    };
};

int
main() { /* not needed for UT */
}