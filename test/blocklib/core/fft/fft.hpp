#ifndef GRAPH_PROTOTYPE_FFT_HPP
#define GRAPH_PROTOTYPE_FFT_HPP

#include <dataset.hpp>
#include <fftw3.h>
#include <history_buffer.hpp>
#include <node.hpp>

namespace gr::blocks::fft {

using namespace fair::graph;

template<typename T>
concept ComplexType = std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

template<typename T>
concept DoubleType = std::is_same_v<T, std::complex<double>> || std::is_same_v<T, double>;

template<typename T>
concept LongDoubleType = std::is_same_v<T, std::complex<long double>> || std::is_same_v<T, long double>;

template<typename T>
concept FFTSupportedTypes = ComplexType<T> || std::integral<T> || std::floating_point<T>;

template<FFTSupportedTypes T>
struct fft : node<fft<T>> {
public:
    static_assert(not LongDoubleType<T>, "long double is not supported");

    struct fftw_base {
    private:
        static inline int        counter = 0;
        static inline std::mutex mutex;

    protected:
        template<auto F>
        static void
        cleanup_impl() {
            std::lock_guard lock(mutex);
            if (--counter == 0) {
                F();
            }
        }

    public:
        static void
        ref() {
            std::lock_guard lock(mutex);
            counter++;
        }
    };

    // clang-format off
    template<typename FftwT>
    struct fftw : fftw_base {
        using plan_type     = fftwf_plan;
        using in_data_type  = std::conditional_t<ComplexType<FftwT>, fftwf_complex, float>;
        using out_data_type = fftwf_complex;
        using precision_type = float;
        using in_history_type  = std::conditional_t<ComplexType<FftwT>, std::complex<float>, float>; // history_buffer can not store c-arrays -> use std::complex

        static void execute(const plan_type p) { fftwf_execute(p); }
        static void destroy_plan(plan_type p) { fftwf_destroy_plan(p); }
        static void free(void *p) { fftwf_free(p); }
        static void cleanup() { fftw_base::template cleanup_impl<fftwf_cleanup>(); }
        static void * malloc(std::size_t n) { return fftwf_malloc(n);}
        static plan_type plan(int p_n, in_data_type *p_in, out_data_type *p_out, int p_sign, unsigned int p_flags) {
            if constexpr (std::is_same_v<in_data_type, float>) {
                return fftwf_plan_dft_r2c_1d(p_n, p_in, p_out, p_flags);
            } else {
                return fftwf_plan_dft_1d(p_n, p_in, p_out, p_sign, p_flags);
            }
        }
        static int import_wisdom_from_filename(const std::string& path) {return fftwf_import_wisdom_from_filename(path.c_str());}
        static int export_wisdom_to_filename(const std::string& path) {return fftwf_export_wisdom_to_filename(path.c_str());}
    };

    template<DoubleType FftwDoubleT>
    struct fftw<FftwDoubleT> : fftw_base {
        using plan_type     = fftw_plan;
        using in_data_type  = std::conditional_t<ComplexType<FftwDoubleT>, fftw_complex, double>;
        using out_data_type = fftw_complex;
        using precision_type = double;
        using in_history_type  = std::conditional_t<ComplexType<FftwDoubleT>, std::complex<double>, double>; // history_buffer can not store c-arrays -> use std::complex

        static void execute(const plan_type p) { fftw_execute(p); }
        static void destroy_plan(plan_type p) { fftw_destroy_plan(p); }
        static void free(void *p) { fftw_free(p); }
        static void cleanup() { fftw_base::template cleanup_impl<fftw_cleanup>(); }
        static void * malloc(std::size_t n) { return fftw_malloc(n);}
        static plan_type plan(int p_n, in_data_type *p_in, out_data_type *p_out, int p_sign, unsigned int p_flags) {
            if constexpr (std::is_same_v<in_data_type, double>) {
                return fftw_plan_dft_r2c_1d(p_n, p_in, p_out, p_flags);
            } else {
                return fftw_plan_dft_1d(p_n, p_in, p_out, p_sign, p_flags);
            }
        }
        static int import_wisdom_from_filename(const std::string& path) {return fftw_import_wisdom_from_filename(path.c_str());}
        static int export_wisdom_to_filename(const std::string& path) {return fftw_export_wisdom_to_filename(path.c_str());}
    };

    template <typename signedT> struct make_signed { using type = signedT;};
    template <std::integral signedT> struct make_signed<signedT> { using type = std::make_signed_t<signedT>; };
    template<typename evalU> struct eval_output_type { using type1 = evalU; using type2 = typename make_signed<T>::type;};
    template<ComplexType evalU> struct eval_output_type<evalU> { using type1 = typename evalU::value_type; using type2 = evalU;};
    using U = std::conditional_t<ComplexType<T>, typename eval_output_type<T>::type1, typename eval_output_type<T>::type2>;
    // clang-format on

    using plan_type       = typename fftw<T>::plan_type;
    using in_data_type    = typename fftw<T>::in_data_type;
    using out_data_type   = typename fftw<T>::out_data_type;
    using precision_type  = typename fftw<T>::precision_type;
    using in_history_type = typename fftw<T>::in_history_type;

    IN<T>                           in;
    OUT<DataSet<U>>                 out;

    std::size_t                     fft_size{ 1024 };
    bool                            output_in_dB{ false };
    std::string                     wisdom_path{ ".gr_fftw_wisdom" };
    int                             sign{ FFTW_FORWARD };
    unsigned int                    flags{ FFTW_ESTIMATE }; // FFTW_EXHAUSTIVE, FFTW_MEASURE, FFTW_ESTIMATE
    history_buffer<in_history_type> inputHistory{ fft_size };
    in_data_type                   *fftw_in{ nullptr };
    out_data_type                  *fftw_out{ nullptr };
    plan_type                       fftw_p{ nullptr };
    std::vector<U>                  magnitude_spectrum{};
    std::vector<U>                  phase_spectrum{};

    fft() { init_all(); }

    fft(const fft &rhs)     = delete;
    fft(fft &&rhs) noexcept = delete;
    fft &
    operator=(const fft &rhs)
            = delete;
    fft &
    operator=(fft &&rhs) noexcept
            = delete;

    ~fft() { clear_fftw(); }

    void
    settings_changed(const property_map & /*old_settings*/, const property_map &new_settings) noexcept {
        if (new_settings.contains("fft_size") && fft_size != inputHistory.capacity()) {
            inputHistory = history_buffer<in_history_type>(fft_size);
            init_all();
        }
    }

    constexpr DataSet<U>
    process_one(T input) noexcept {
        if constexpr (std::is_same_v<T, in_history_type>) {
            inputHistory.push_back(input);
        } else {
            inputHistory.push_back(static_cast<in_history_type>(input));
        }

        if (inputHistory.size() >= fft_size) {
            prepare_input();
            compute_fft();
            compute_magnitude_spectrum();
            return create_dataset();
        } else {
            return DataSet<U>();
        }
    }

    constexpr DataSet<U>
    create_dataset() {
        DataSet<U> ds{};
        ds.timestamp = 0;
        const std::size_t N{ magnitude_spectrum.size() };

        ds.axis_names   = { "time", "fft.real", "fft.imag", "magnitude", "phase" };
        ds.axis_units   = { "u.a.", "u.a.", "u.a.", "u.a.", "u.a." };
        ds.extents      = { 4, static_cast<int32_t>(N) };
        ds.layout       = fair::graph::layout_right{};
        ds.signal_names = { "fft.real", "fft.imag", "magnitude", "phase" };
        ds.signal_units = { "u.a.", "u.a.", "u.a.", "u.a." };

        ds.signal_values.resize(4 * N);
        for (std::size_t i = 0; i < N; i++) {
            if constexpr (std::is_same_v<U, precision_type>) {
                ds.signal_values[i]     = fftw_out[i][0];
                ds.signal_values[i + N] = fftw_out[i][1];
            } else {
                ds.signal_values[i]     = static_cast<U>(fftw_out[i][0]);
                ds.signal_values[i + N] = static_cast<U>(fftw_out[i][1]);
            }
            ds.signal_values[i + 2 * N] = magnitude_spectrum[i];
            ds.signal_values[i + 3 * N] = phase_spectrum[i];
        }

        ds.signal_errors = {};

        for (std::size_t i = 0; i < 4; i++) {
            const auto mm = std::minmax_element(std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(i * N)), std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>((i + 1) * N)));
            ds.signal_ranges.push_back({ *mm.first, *mm.second });
        }

        return ds;
    }

    void
    prepare_input() {
        static_assert(sizeof(in_data_type) == sizeof(in_history_type), "std::complex<T> and T[2] must have the same size");
        std::memcpy(fftw_in, &(*inputHistory.begin()), sizeof(in_data_type) * fft_size);
    }

    void
    compute_fft() {
        fftw<T>::execute(fftw_p);
    }

    void
    compute_magnitude_spectrum() {
        for (std::size_t i = 0; i < magnitude_spectrum.size(); i++) {
            const precision_type mag{ std::hypot(fftw_out[i][0], fftw_out[i][1]) * static_cast<precision_type>(2.0) / static_cast<precision_type>(fft_size) };
            magnitude_spectrum[i] = static_cast<U>(output_in_dB ? static_cast<precision_type>(20.) * std::log10(std::abs(mag)) : mag);
        }
    }

    void
    compute_phase_spectrum() {
        for (std::size_t i = 0; i < phase_spectrum.size(); i++) {
            const auto phase{ std::atan2(fftw_out[i][1], fftw_out[i][0]) };
            phase_spectrum[i] = static_cast<U>(output_in_dB ? 20. * log10(abs(phase)) : phase);
        }
    }

    void
    init_all() {
        fftw<T>::ref();
        clear_fftw();
        fftw<T>::ref();
        fftw_in = static_cast<in_data_type *>(fftw<T>::malloc(sizeof(in_data_type) * fft_size));
        if constexpr (ComplexType<T>) {
            fftw_out = static_cast<out_data_type *>(fftw<T>::malloc(sizeof(out_data_type) * fft_size));
            magnitude_spectrum.resize(fft_size);
            phase_spectrum.resize(fft_size);
        } else {
            fftw_out = static_cast<out_data_type *>(fftw<T>::malloc(sizeof(out_data_type) * (1 + fft_size / 2)));
            magnitude_spectrum.resize(fft_size / 2);
            phase_spectrum.resize(fft_size / 2);
        }
        import_wisdom();
        fftw_p = fftw<T>::plan(static_cast<int>(fft_size), fftw_in, fftw_out, sign, flags);
        export_wisdom();
    }

    void
    clear_fftw() {
        if (fftw_p != nullptr) {
            fftw<T>::destroy_plan(fftw_p);
            fftw_p = nullptr;
        }
        if (fftw_in != nullptr) {
            fftw<T>::free(fftw_in);
            fftw_in = nullptr;
        }
        if (fftw_out != nullptr) {
            fftw<T>::free(fftw_out);
            fftw_out = nullptr;
        }
        fftw<T>::cleanup();
        magnitude_spectrum.clear();
        phase_spectrum.clear();
    }

    void
    import_wisdom() {
        fftw<T>::import_wisdom_from_filename(wisdom_path);
    }

    void
    export_wisdom() {
        fftw<T>::export_wisdom_to_filename(wisdom_path);
    }
};

} // namespace gr::blocks::fft

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (gr::blocks::fft::fft<T>), in, out, fft_size, output_in_dB);

#endif // GRAPH_PROTOTYPE_FFT_HPP
