#ifndef GRAPH_PROTOTYPE_ALGORITHM_FFTW_HPP
#define GRAPH_PROTOTYPE_ALGORITHM_FFTW_HPP

#include <fftw3.h>

#include <gnuradio-4.0/dataset.hpp>
#include <gnuradio-4.0/history_buffer.hpp>
#include <gnuradio-4.0/node.hpp>

#include "fft_types.hpp"
#include "window.hpp"

namespace gr::algorithm {

using namespace fair::graph;

template<typename T>
concept FFTwDoubleType = std::is_same_v<T, std::complex<double>> || std::is_same_v<T, double>;

template<typename T>
    requires(ComplexType<T> || std::floating_point<T>)
struct FFTw {
private:
    inline static std::mutex fftw_plan_mutex;

public:
    // clang-format off
    template<typename FftwT>
    struct fftwImpl {
        using PlanType        = fftwf_plan;
        using InAlgoDataType  = std::conditional_t<ComplexType<FftwT>, fftwf_complex, float>;
        using OutAlgoDataType = fftwf_complex;
        using InUniquePtr     = std::unique_ptr<InAlgoDataType [], decltype([](InAlgoDataType *ptr) { fftwf_free(ptr); })>;
        using OutUniquePtr    = std::unique_ptr<OutAlgoDataType [], decltype([](OutAlgoDataType *ptr) { fftwf_free(ptr); })>;
        using PlanUniquePtr   = std::unique_ptr<std::remove_pointer_t<PlanType>, decltype([](PlanType ptr) { fftwf_destroy_plan(ptr); })>;

        static void execute(const PlanType p) { fftwf_execute(p); }
        static void cleanup() { fftwf_cleanup(); }
        static void * malloc(std::size_t n) { return fftwf_malloc(n);}
        static PlanType plan(int p_n, InAlgoDataType *p_in, OutAlgoDataType *p_out, int p_sign, unsigned int p_flags) {
            if constexpr (std::is_same_v<InAlgoDataType, float>) {
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

    template<FFTwDoubleType FftwDoubleT>
    struct fftwImpl<FftwDoubleT> {
        using PlanType        = fftw_plan;
        using InAlgoDataType  = std::conditional_t<ComplexType<FftwDoubleT>, fftw_complex, double>;
        using OutAlgoDataType = fftw_complex;
        using InUniquePtr     = std::unique_ptr<InAlgoDataType [], decltype([](InAlgoDataType *ptr) { fftwf_free(ptr); })>;
        using OutUniquePtr    = std::unique_ptr<OutAlgoDataType [], decltype([](OutAlgoDataType *ptr) { fftwf_free(ptr); })>;
        using PlanUniquePtr   = std::unique_ptr<std::remove_pointer_t<PlanType>, decltype([](PlanType ptr) { fftw_destroy_plan(ptr); })>;

        static void execute(const PlanType p) { fftw_execute(p); }
        static void cleanup() { fftw_cleanup(); }
        static void * malloc(std::size_t n) { return fftw_malloc(n);}
        static PlanType plan(int p_n, InAlgoDataType *p_in, OutAlgoDataType *p_out, int p_sign, unsigned int p_flags) {
            if constexpr (std::is_same_v<InAlgoDataType, double>) {
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

    // clang-format on
    using OutDataType     = std::conditional_t<ComplexType<T>, T, std::complex<T>>;
    using InAlgoDataType  = typename fftwImpl<T>::InAlgoDataType;
    using OutAlgoDataType = typename fftwImpl<T>::OutAlgoDataType;
    using InUniquePtr     = typename fftwImpl<T>::InUniquePtr;
    using OutUniquePtr    = typename fftwImpl<T>::OutUniquePtr;
    using PlanUniquePtr   = typename fftwImpl<T>::PlanUniquePtr;

    std::size_t   fftSize{ 0 };
    std::string   wisdomPath{ ".gr_fftw_wisdom" };
    int           sign{ FFTW_FORWARD };
    unsigned int  flags{ FFTW_ESTIMATE }; // FFTW_EXHAUSTIVE, FFTW_MEASURE, FFTW_ESTIMATE
    InUniquePtr   fftwIn{};
    OutUniquePtr  fftwOut{};
    PlanUniquePtr fftwPlan{};

    FFTw()                    = default;
    FFTw(const FFTw &rhs)     = delete;
    FFTw(FFTw &&rhs) noexcept = delete;
    FFTw &
    operator=(const FFTw &rhs)
            = delete;
    FFTw &
    operator=(FFTw &&rhs) noexcept
            = delete;

    ~FFTw() { clearFftw(); }

    std::vector<OutDataType>
    computeFFT(const std::vector<T> &in) {
        if (fftSize != in.size()) {
            fftSize = in.size();
            initAll();
        }
        std::vector<OutDataType> out(getOutputSize());
        computeFFT(in, out);
        return out;
    }

    void
    computeFFT(const std::vector<T> &in, std::vector<OutDataType> &out) {
        if (!std::has_single_bit(in.size())) {
            throw std::invalid_argument(fmt::format("Input data must have 2^N samples, input size: ", in.size()));
        }

        if (fftSize != in.size()) {
            fftSize = in.size();
            initAll();
        }

        if (out.size() < getOutputSize()) {
            throw std::out_of_range(fmt::format("Output vector size ({}) is not enough, at least {} needed. ", out.size(), getOutputSize()));
        }

        static_assert(sizeof(InAlgoDataType) == sizeof(T), "Input std::complex<T> and T[2] must have the same size.");
        std::memcpy(fftwIn.get(), &(*in.begin()), sizeof(InAlgoDataType) * fftSize);

        fftwImpl<T>::execute(fftwPlan.get());

        static_assert(sizeof(OutDataType) == sizeof(OutAlgoDataType), "Output std::complex<T> and T[2] must have the same size.");
        std::memcpy(out.data(), fftwOut.get(), sizeof(OutDataType) * getOutputSize());
    }

    [[nodiscard]] inline int
    importWisdom() const {
        // lock file while importing wisdom?
        return fftwImpl<T>::importWisdomFromFilename(wisdomPath);
    }

    [[nodiscard]] inline int
    exportWisdom() const {
        // lock file while exporting wisdom?
        return fftwImpl<T>::exportWisdomToFilename(wisdomPath);
    }

    [[nodiscard]] inline int
    importWisdomFromString(const std::string &wisdomString) const {
        return fftwImpl<T>::importWisdomFromString(wisdomString);
    }

    [[nodiscard]] std::string
    exportWisdomToString() const {
        return fftwImpl<T>::exportWisdomToString();
    }

    inline void
    forgetWisdom() const {
        return fftwImpl<T>::forgetWisdom();
    }

private:
    [[nodiscard]] constexpr std::size_t
    getOutputSize() const {
        if constexpr (ComplexType<T>) {
            return fftSize;
        } else {
            return 1 + fftSize / 2;
        }
    }

    void
    initAll() {
        clearFftw();
        fftwIn  = InUniquePtr(static_cast<InAlgoDataType *>(fftwImpl<T>::malloc(sizeof(InAlgoDataType) * fftSize)));
        fftwOut = OutUniquePtr(static_cast<OutAlgoDataType *>(fftwImpl<T>::malloc(sizeof(OutAlgoDataType) * getOutputSize())));

        {
            std::lock_guard lg{ fftw_plan_mutex };
            // what to do if error is returned
            std::ignore = importWisdom();
            fftwPlan    = PlanUniquePtr(fftwImpl<T>::plan(static_cast<int>(fftSize), fftwIn.get(), fftwOut.get(), sign, flags));
            std::ignore = exportWisdom();
        }
    }

    void
    clearFftw() {
        {
            std::lock_guard lg{ fftw_plan_mutex };
            fftwPlan.reset();
        }
        fftwIn.reset();
        fftwOut.reset();
    }
};

} // namespace gr::algorithm

#endif // GRAPH_PROTOTYPE_ALGORITHM_FFTW_HPP
