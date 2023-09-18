#ifndef GRAPH_PROTOTYPE_FFT_HPP
#define GRAPH_PROTOTYPE_FFT_HPP

#include "window.hpp"
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

inline static std::mutex fftw_plan_mutex;

template<FFTSupportedTypes T>
struct fft : node<fft<T>> {
public:
    static_assert(not LongDoubleType<T>, "long double is not supported");

    // clang-format off
    template<typename FftwT>
    struct fftw {
        using PlanType       = fftwf_plan;
        using InDataType    = std::conditional_t<ComplexType<FftwT>, fftwf_complex, float>;
        using OutDataType   = fftwf_complex;
        using PrecisionType  = float;
        using InHistoryType = std::conditional_t<ComplexType<FftwT>, std::complex<float>, float>; // history_buffer can not store c-arrays -> use std::complex
        using InUniquePtr   = std::unique_ptr<InDataType [], decltype([](InDataType *ptr) { fftwf_free(ptr); })>;
        using OutUniquePtr  = std::unique_ptr<OutDataType [], decltype([](OutDataType *ptr) { fftwf_free(ptr); })>;
        using PlanUniquePtr = std::unique_ptr<std::remove_pointer_t<PlanType>, decltype([](PlanType ptr) { fftwf_destroy_plan(ptr); })>;

        static void execute(const PlanType p) { fftwf_execute(p); }
        static void cleanup() { fftwf_cleanup(); }
        static void * malloc(std::size_t n) { return fftwf_malloc(n);}
        static PlanType plan(int p_n, InDataType *p_in, OutDataType *p_out, int p_sign, unsigned int p_flags) {
            if constexpr (std::is_same_v<InDataType, float>) {
                return fftwf_plan_dft_r2c_1d(p_n, p_in, p_out, p_flags);
            } else {
                return fftwf_plan_dft_1d(p_n, p_in, p_out, p_sign, p_flags);
            }
        }
        static int importWisdomFromFilename(const std::string& path) {return fftwf_import_wisdom_from_filename(path.c_str());}
        static int exportWisdomToFilename(const std::string& path) {return fftwf_export_wisdom_to_filename(path.c_str());}
        static int importWisdomFromString(const std::string& str) {return fftwf_import_wisdom_from_string(str.c_str());}
        static std::string exportWisdomToString() {
            char* cstr = fftwf_export_wisdom_to_string();
            if (cstr == nullptr) return "";
            std::string str(cstr);
            fftwf_free(cstr);
            return str;
        }
        static void forgetWisdom() {fftwf_forget_wisdom();}
    };

    template<DoubleType FftwDoubleT>
    struct fftw<FftwDoubleT> {
        using PlanType       = fftw_plan;
        using InDataType    = std::conditional_t<ComplexType<FftwDoubleT>, fftw_complex, double>;
        using OutDataType   = fftw_complex;
        using PrecisionType  = double;
        using InHistoryType = std::conditional_t<ComplexType<FftwDoubleT>, std::complex<double>, double>; // history_buffer can not store c-arrays -> use std::complex
        using InUniquePtr   = std::unique_ptr<InDataType [], decltype([](InDataType *ptr) { fftwf_free(ptr); })>;
        using OutUniquePtr  = std::unique_ptr<OutDataType [], decltype([](OutDataType *ptr) { fftwf_free(ptr); })>;
        using PlanUniquePtr = std::unique_ptr<std::remove_pointer_t<PlanType>, decltype([](PlanType ptr) { fftw_destroy_plan(ptr); })>;

        static void execute(const PlanType p) { fftw_execute(p); }
        static void cleanup() { fftw_cleanup(); }
        static void * malloc(std::size_t n) { return fftw_malloc(n);}
        static PlanType plan(int p_n, InDataType *p_in, OutDataType *p_out, int p_sign, unsigned int p_flags) {
            if constexpr (std::is_same_v<InDataType, double>) {
                return fftw_plan_dft_r2c_1d(p_n, p_in, p_out, p_flags);
            } else {
                return fftw_plan_dft_1d(p_n, p_in, p_out, p_sign, p_flags);
            }
        }
        static int importWisdomFromFilename(const std::string& path) {return fftw_import_wisdom_from_filename(path.c_str());}
        static int exportWisdomToFilename(const std::string& path) {return fftw_export_wisdom_to_filename(path.c_str());}
        static int importWisdomFromString(const std::string& str) {return fftw_import_wisdom_from_string(str.c_str());}
        static std::string exportWisdomToString() {
            char* cstr = fftw_export_wisdom_to_string();
            if (cstr == nullptr) return "";
            std::string str(cstr);
            fftw_free(cstr);
            return str;
        }
        static void forgetWisdom() {fftw_forget_wisdom();}
    };

    template <typename signedT> struct MakeSigned { using type = signedT;};
    template <std::integral signedT> struct MakeSigned<signedT> { using type = std::make_signed_t<signedT>; };
    template<typename evalU> struct EvalOutputType { using type1 = evalU; using type2 = typename MakeSigned<T>::type;};
    template<ComplexType evalU> struct EvalOutputType<evalU> { using type1 = typename evalU::value_type; using type2 = evalU;};
    using U = std::conditional_t<ComplexType<T>, typename EvalOutputType<T>::type1, typename EvalOutputType<T>::type2>;
    // clang-format on

    using InDataType    = typename fftw<T>::InDataType;
    using OutDataType   = typename fftw<T>::OutDataType;
    using PrecisionType = typename fftw<T>::PrecisionType;
    using InHistoryType = typename fftw<T>::InHistoryType;
    using InUniquePtr   = typename fftw<T>::InUniquePtr;
    using OutUniquePtr  = typename fftw<T>::OutUniquePtr;
    using PlanUniquePtr = typename fftw<T>::PlanUniquePtr;

    IN<T>                         in;
    OUT<DataSet<U>>               out;

    std::size_t                   fftSize{ 1024 };
    int                           window{ static_cast<int>(WindowFunction::None) };
    std::vector<PrecisionType>    windowVector;
    bool                          outputInDb{ false };
    std::string                   wisdomPath{ ".gr_fftw_wisdom" };
    int                           sign{ FFTW_FORWARD };
    unsigned int                  flags{ FFTW_ESTIMATE }; // FFTW_EXHAUSTIVE, FFTW_MEASURE, FFTW_ESTIMATE
    history_buffer<InHistoryType> inputHistory{ fftSize };
    InUniquePtr                   fftwIn{};
    OutUniquePtr                  fftwOut{};
    PlanUniquePtr                 fftwPlan{};
    std::vector<U>                magnitudeSpectrum{};
    std::vector<U>                phaseSpectrum{};

    fft() { initAll(); }

    fft(const fft &rhs)     = delete;
    fft(fft &&rhs) noexcept = delete;
    fft &
    operator=(const fft &rhs)
            = delete;
    fft &
    operator=(fft &&rhs) noexcept
            = delete;

    ~fft() { clearFftw(); }

    void
    settings_changed(const property_map & /*old_settings*/, const property_map &newSettings) noexcept {
        if (newSettings.contains("fftSize") && fftSize != inputHistory.capacity()) {
            inputHistory = history_buffer<InHistoryType>(fftSize);
            initAll();
        } else if (newSettings.contains("window")) { // no need to create window twice if fftSize was changed
            initWindowFunction();
        }
    }

    constexpr DataSet<U>
    process_one(T input) noexcept {
        if constexpr (std::is_same_v<T, InHistoryType>) {
            inputHistory.push_back(input);
        } else {
            inputHistory.push_back(static_cast<InHistoryType>(input));
        }

        if (inputHistory.size() >= fftSize) {
            prepareInput();
            computeFft();
            computeMagnitudeSpectrum();
            computePhaseSpectrum();
            return createDataset();
        } else {
            return DataSet<U>();
        }
    }

    constexpr DataSet<U>
    createDataset() {
        DataSet<U> ds{};
        ds.timestamp = 0;
        const std::size_t N{ magnitudeSpectrum.size() };

        ds.axis_names   = { "time", "fft.real", "fft.imag", "magnitude", "phase" };
        ds.axis_units   = { "u.a.", "u.a.", "u.a.", "u.a.", "rad" };
        ds.extents      = { 4, static_cast<int32_t>(N) };
        ds.layout       = fair::graph::layout_right{};
        ds.signal_names = { "fft.real", "fft.imag", "magnitude", "phase" };
        ds.signal_units = { "u.a.", "u.a.", "u.a.", "rad" };

        ds.signal_values.resize(4 * N);
        ds.signal_ranges = { { std::numeric_limits<U>::max(), std::numeric_limits<U>::lowest() },
                             { std::numeric_limits<U>::max(), std::numeric_limits<U>::lowest() },
                             { std::numeric_limits<U>::max(), std::numeric_limits<U>::lowest() },
                             { std::numeric_limits<U>::max(), std::numeric_limits<U>::lowest() } };
        for (std::size_t i = 0; i < N; i++) {
            if constexpr (std::is_same_v<U, PrecisionType>) {
                ds.signal_values[i]     = fftwOut[i][0];
                ds.signal_values[i + N] = fftwOut[i][1];
            } else {
                ds.signal_values[i]     = static_cast<U>(fftwOut[i][0]);
                ds.signal_values[i + N] = static_cast<U>(fftwOut[i][1]);
            }
            ds.signal_ranges[0][0]      = std::min(ds.signal_ranges[0][0], ds.signal_values[i]);
            ds.signal_ranges[0][1]      = std::max(ds.signal_ranges[0][1], ds.signal_values[i]);
            ds.signal_ranges[1][0]      = std::min(ds.signal_ranges[1][0], ds.signal_values[i + N]);
            ds.signal_ranges[1][1]      = std::max(ds.signal_ranges[1][1], ds.signal_values[i + N]);

            ds.signal_values[i + 2 * N] = magnitudeSpectrum[i];
            ds.signal_ranges[2][0]      = std::min(ds.signal_ranges[2][0], magnitudeSpectrum[i]);
            ds.signal_ranges[2][1]      = std::max(ds.signal_ranges[2][1], magnitudeSpectrum[i]);

            ds.signal_values[i + 3 * N] = phaseSpectrum[i];
            ds.signal_ranges[3][0]      = std::min(ds.signal_ranges[3][0], phaseSpectrum[i]);
            ds.signal_ranges[3][1]      = std::max(ds.signal_ranges[3][1], phaseSpectrum[i]);
        }

        ds.signal_errors    = {};
        ds.meta_information = {
            { { "fftSize", fftSize }, { "window", window }, { "outputInDb", outputInDb }, { "numerator", this->numerator }, { "denominator", this->denominator }, { "stride", this->stride } }
        };

        return ds;
    }

    void
    prepareInput() {
        static_assert(sizeof(InDataType) == sizeof(InHistoryType), "std::complex<T> and T[2] must have the same size");
        std::memcpy(fftwIn.get(), &(*inputHistory.begin()), sizeof(InDataType) * fftSize);
        // apply window function if needed
        if (window != static_cast<int>(WindowFunction::None)) {
            if (fftSize != windowVector.size()) throw std::runtime_error(fmt::format("fftSize({}) and windowVector.size({}) are not equal.", fftSize, windowVector.size()));
            for (std::size_t i = 0; i < fftSize; i++) {
                if constexpr (ComplexType<T>) {
                    fftwIn[i][0] *= windowVector[i];
                    fftwIn[i][1] *= windowVector[i];
                } else {
                    fftwIn[i] *= windowVector[i];
                }
            }
        }
    }

    void
    computeFft() {
        fftw<T>::execute(fftwPlan.get());
    }

    void
    computeMagnitudeSpectrum() {
        for (std::size_t i = 0; i < magnitudeSpectrum.size(); i++) {
            const PrecisionType mag{ std::hypot(fftwOut[i][0], fftwOut[i][1]) * static_cast<PrecisionType>(2.0) / static_cast<PrecisionType>(fftSize) };
            magnitudeSpectrum[i] = static_cast<U>(outputInDb ? static_cast<PrecisionType>(20.) * std::log10(std::abs(mag)) : mag);
        }
    }

    void
    computePhaseSpectrum() {
        for (std::size_t i = 0; i < phaseSpectrum.size(); i++) {
            const auto phase{ std::atan2(fftwOut[i][1], fftwOut[i][0]) };
            phaseSpectrum[i] = outputInDb ? static_cast<U>(20.) * static_cast<U>(std::log10(std::abs(phase))) : static_cast<U>(phase);
        }
    }

    void
    initAll() {
        clearFftw();
        fftwIn = InUniquePtr(static_cast<InDataType *>(fftw<T>::malloc(sizeof(InDataType) * fftSize)));
        if constexpr (ComplexType<T>) {
            fftwOut = OutUniquePtr(static_cast<OutDataType *>(fftw<T>::malloc(sizeof(OutDataType) * fftSize)));
            magnitudeSpectrum.resize(fftSize);
            phaseSpectrum.resize(fftSize);
        } else {
            fftwOut = OutUniquePtr(static_cast<OutDataType *>(fftw<T>::malloc(sizeof(OutDataType) * (1 + fftSize / 2))));
            magnitudeSpectrum.resize(fftSize / 2);
            phaseSpectrum.resize(fftSize / 2);
        }
        {
            std::lock_guard lg{ fftw_plan_mutex };
            importWisdom();
            fftwPlan = PlanUniquePtr(fftw<T>::plan(static_cast<int>(fftSize), fftwIn.get(), fftwOut.get(), sign, flags));
            exportWisdom();
        }

        initWindowFunction();
    }

    void
    initWindowFunction() {
        windowVector = createWindowFunction<PrecisionType>(static_cast<WindowFunction>(window), fftSize);
    }

    void
    clearFftw() {
        {
            std::lock_guard lg{ fftw_plan_mutex };
            fftwPlan.reset();
        }

        fftwIn.reset();
        fftwOut.reset();

        // fftw<T>::cleanup(); // No need for fftw_cleanup -> After calling it, all existing plans become undefined
        magnitudeSpectrum.clear();
        phaseSpectrum.clear();
    }

    int
    importWisdom() {
        // TODO: lock file while importing wisdom?
        return fftw<T>::importWisdomFromFilename(wisdomPath);
    }

    int
    exportWisdom() {
        // TODO: lock file while exporting wisdom?
        return fftw<T>::exportWisdomToFilename(wisdomPath);
    }

    int
    importWisdomFromString(const std::string wisdomString) {
        return fftw<T>::importWisdomFromString(wisdomString);
    }

    std::string
    exportWisdomToString() {
        return fftw<T>::exportWisdomToString();
    }

    void
    forgetWisdom() {
        return fftw<T>::forgetWisdom();
    }
};

} // namespace gr::blocks::fft

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (gr::blocks::fft::fft<T>), in, out, fftSize, outputInDb, window);

#endif // GRAPH_PROTOTYPE_FFT_HPP
