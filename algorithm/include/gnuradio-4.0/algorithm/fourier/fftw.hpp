#ifndef GNURADIO_ALGORITHM_FFTW_HPP
#define GNURADIO_ALGORITHM_FFTW_HPP

#include <fftw3.h>

#include "window.hpp"

namespace gr::algorithm {

template<typename TInput, typename TOutput = std::conditional<gr::meta::complex_like<TInput>, TInput, std::complex<typename TInput::value_type>>>
    requires((gr::meta::complex_like<TInput> || std::floating_point<TInput>) && (gr::meta::complex_like<TOutput>) )
struct FFTw {
private:
    inline static std::mutex fftw_plan_mutex;

public:
    // clang-format off
    template<typename TData>
    struct FFTwImpl {
        using PlanType        = fftwf_plan;
        using InAlgoDataType  = std::conditional_t<gr::meta::complex_like<TData>, fftwf_complex, float>;
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

    template<typename TData>
       requires (std::is_same_v<TData, std::complex<double>> || std::is_same_v<TData, double>)
    struct FFTwImpl<TData> {
        using PlanType        = fftw_plan;
        using InAlgoDataType  = std::conditional_t<gr::meta::complex_like<TData>, fftw_complex, double>;
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
    // Precision of the algorithm is defined by the output type `TOutput:value_type`
    using AlgoDataType    = std::conditional_t<gr::meta::complex_like<TInput>, TOutput, typename TOutput::value_type>;
    using InAlgoDataType  = typename FFTwImpl<AlgoDataType>::InAlgoDataType;
    using OutAlgoDataType = typename FFTwImpl<AlgoDataType>::OutAlgoDataType;
    using InUniquePtr     = typename FFTwImpl<AlgoDataType>::InUniquePtr;
    using OutUniquePtr    = typename FFTwImpl<AlgoDataType>::OutUniquePtr;
    using PlanUniquePtr   = typename FFTwImpl<AlgoDataType>::PlanUniquePtr;

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

    auto
    compute(const std::ranges::input_range auto &in, std::ranges::output_range<TOutput> auto &&out) {
        if constexpr (requires(std::size_t n) { out.resize(n); }) {
            if (out.size() != in.size()) {
                out.resize(in.size());
            }
        } else {
            static_assert(std::tuple_size_v<decltype(in)> == std::tuple_size_v<decltype(out)>, "Size mismatch for fixed-size container.");
        }

        if (!std::has_single_bit(in.size())) {
            throw std::invalid_argument(fmt::format("Input data must have 2^N samples, input size: ", in.size()));
        }

        if (fftSize != in.size()) {
            fftSize = in.size();
            initAll();
        }

        if (out.size() < fftSize) {
            throw std::out_of_range(fmt::format("Output vector size ({}) is not enough, at least {} needed. ", out.size(), fftSize));
        }

        // precision is defined by output type, if needed cast input
        if constexpr (!std::is_same_v<TInput, AlgoDataType>) {
            std::span<AlgoDataType> inSpan(reinterpret_cast<AlgoDataType *>(fftwIn.get()), in.size());
            std::ranges::transform(in.begin(), in.end(), inSpan.begin(), [](const auto c) { return static_cast<AlgoDataType>(c); });
        } else {
            std::memcpy(fftwIn.get(), &(*in.begin()), sizeof(InAlgoDataType) * fftSize);
        }

        FFTwImpl<AlgoDataType>::execute(fftwPlan.get());

        static_assert(sizeof(TOutput) == sizeof(OutAlgoDataType), "Sizes of TOutput type and OutAlgoDataType are not equal.");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
        // Switch off warning: ‘void* memcpy(void*, const void*, size_t)’ copying an object of non-trivial type ‘class std::complex<float>’ from an array of ‘float [2]’
        std::memcpy(out.data(), fftwOut.get(), sizeof(TOutput) * getOutputSize());
#pragma GCC diagnostic pop
        // for the real input to complex a Hermitian output is produced by fftw, perform mirroring and conjugation fftw spectra to the second half
        if (!gr::meta::complex_like<TInput>) {
            const auto halfIt = std::next(out.begin(), static_cast<std::ptrdiff_t>(fftSize / 2));
            std::ranges::transform(out.begin(), halfIt, halfIt, [](auto c) { return std::conj(c); });
            std::reverse(halfIt, out.end());
        }

        return out;
    }

    auto
    compute(const std::ranges::input_range auto &in) {
        return compute(in, std::vector<TOutput>());
    }

    [[nodiscard]] inline int
    importWisdom() const {
        // lock file while importing wisdom?
        return FFTwImpl<AlgoDataType>::importWisdomFromFilename(wisdomPath);
    }

    [[nodiscard]] inline int
    exportWisdom() const {
        // lock file while exporting wisdom?
        return FFTwImpl<AlgoDataType>::exportWisdomToFilename(wisdomPath);
    }

    [[nodiscard]] inline int
    importWisdomFromString(const std::string &wisdomString) const {
        return FFTwImpl<AlgoDataType>::importWisdomFromString(wisdomString);
    }

    [[nodiscard]] std::string
    exportWisdomToString() const {
        return FFTwImpl<AlgoDataType>::exportWisdomToString();
    }

    inline void
    forgetWisdom() const {
        return FFTwImpl<AlgoDataType>::forgetWisdom();
    }

private:
    [[nodiscard]] constexpr std::size_t
    getOutputSize() const {
        if constexpr (gr::meta::complex_like<TInput>) {
            return fftSize;
        } else {
            return 1 + fftSize / 2;
        }
    }

    void
    initAll() {
        clearFftw();
        fftwIn  = InUniquePtr(static_cast<InAlgoDataType *>(FFTwImpl<AlgoDataType>::malloc(sizeof(InAlgoDataType) * fftSize)));
        fftwOut = OutUniquePtr(static_cast<OutAlgoDataType *>(FFTwImpl<AlgoDataType>::malloc(sizeof(OutAlgoDataType) * getOutputSize())));

        {
            std::lock_guard lg{ fftw_plan_mutex };
            // what to do if error is returned
            std::ignore = importWisdom();
            fftwPlan    = PlanUniquePtr(FFTwImpl<AlgoDataType>::plan(static_cast<int>(fftSize), fftwIn.get(), fftwOut.get(), sign, flags));
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

#endif // GNURADIO_ALGORITHM_FFTW_HPP
