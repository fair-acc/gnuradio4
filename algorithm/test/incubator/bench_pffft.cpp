#include "gnuradio-4.0/meta/utils.hpp"
#include <boost/ut.hpp>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
// GCC has no -Weverything; ignore the big sets you enable globally:
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wvla"
#endif

// … noisy code …

#include "SimdFFT.cpp"
#include "SimdFFT.hpp"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <print>

#include <assert.h>
#include <bit>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_FFT_ALGOS 2
enum {
    ALGO_PFFFT_U, /* = 8 */
    ALGO_PFFFT_O  /* = 9 */
};

#define NUM_TYPES 7
enum {
    TYPE_PREP        = 0, /* time for preparation in ms */
    TYPE_DUR_NS      = 1, /* time per fft in ns */
    TYPE_DUR_FASTEST = 2, /* relative time to fastest */
    TYPE_REL_PFFFT   = 3, /* relative time to ALGO_PFFFT */
    TYPE_ITER        = 4, /* # of iterations in measurement */
    TYPE_MFLOPS      = 5, /* MFlops/sec */
    TYPE_DUR_TOT     = 6  /* test duration in sec */
};
/* double tmeas[NUM_TYPES][NUM_FFT_ALGOS]; */

const char* algoName[NUM_FFT_ALGOS] = {
    "PFFFT-U(simd)", /* unordered */
    "PFFFT (simd) "  /* ordered */
};

int compiledInAlgo[NUM_FFT_ALGOS] = {
    1, /* "PFFFT_U    " */
    1  /* "PFFFT_O    " */
};

const char* algoTableHeader[NUM_FFT_ALGOS][2] = {{"| real PFFFT-U ", "| cplx PFFFT-U "}, {"|  real  PFFFT ", "|  cplx  PFFFT "}};

const char* typeText[NUM_TYPES] = {"preparation in ms", "time per fft in ns", "relative to fastest", "relative to pffft", "measured_num_iters", "mflops", "test duration in sec"};

const char* typeFilenamePart[NUM_TYPES] = {"1-preparation-in-ms", "2-timePerFft-in-ns", "3-rel-fastest", "4-rel-pffft", "5-num-iter", "6-mflops", "7-duration-in-sec"};

#define SAVE_ALL_TYPES 0

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

unsigned Log2(std::size_t v) {
    /* we don't need speed records .. obvious way is good enough */
    /* https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogObvious */
    /* Find the log base 2 of an integer with the MSB N set in O(N) operations (the obvious way):
     * unsigned v: 32-bit word to find the log base 2 of */
    unsigned r = 0; /* r will be lg(v) */
    while (v >>= 1) {
        r++;
    }
    return r;
}

double frand() { return rand() / static_cast<double>(RAND_MAX); }

double uclock_sec(void) { return static_cast<double>(clock()) / static_cast<double>(CLOCKS_PER_SEC); }

/* compare results with the regular fftpack */
std::size_t pffft_validate_N(std::size_t /*N*/, std::size_t /*cplx*/) { return 2UZ; }

std::size_t pffft_validate(std::size_t cplx) {
    static std::size_t Ntest[] = {16, 32, 64, 96, 128, 160, 192, 256, 288, 384, 5 * 96, 512, 576, 5 * 128, 800, 864, 1024, 2048, 2592, 4000, 4096, 12000, 36864, 0};
    std::size_t        k, r;
    for (k = 0; Ntest[k]; ++k) {
        std::size_t N = Ntest[k];
        if (N == 16 && !cplx) {
            continue;
        }
        r = pffft_validate_N(N, cplx);
        if (r) {
            return r;
        }
    }
    return 0UZ;
}

int array_output_format = 1;

void print_table(const char* txt, FILE* tableFile) {
    fprintf(stdout, "%s", txt);
    if (tableFile && tableFile != stdout) {
        fprintf(tableFile, "%s", txt);
    }
}

void print_table_flops(double mflops, FILE* tableFile) {
    fprintf(stdout, "|%11.0f   ", static_cast<double>(mflops));
    if (tableFile && tableFile != stdout) {
        fprintf(tableFile, "|%11.0f   ", static_cast<double>(mflops));
    }
}

void print_table_fftsize(std::size_t N, FILE* tableFile) {
    std::print("|{:9}", N);
    if (tableFile && tableFile != stdout) {
        std::print(tableFile, "|{:9}", N);
    }
}

double show_output(std::string_view name, std::size_t N, bool cplx, double flops, /* pass -1 for “n/a” */
    double t0,                                                                    /* same units for t0/t1 (e.g. seconds) */
    double t1, std::size_t max_iter, std::FILE* tableFile) {
    constexpr double eps     = 1e-16; // guard against div-by-zero
    const double     total   = t1 - t0;
    const double     T_ns    = total / 2.0 / static_cast<double>(max_iter) * 1e9;
    const bool       haveOps = (flops >= 0.0);
    const double     mflops  = haveOps ? (flops / 1e6) / (total + eps) : 0.0;

    if (array_output_format) {
        if (haveOps) {
            print_table_flops(mflops, tableFile);
        } else {
            print_table("|        n/a   ", tableFile);
        }
    } else if (haveOps) {
        // Example: N=  512, CPLX          FFT :   1234 MFlops [t=   800 ns, 100 runs]
        using namespace std::literals;
        std::println("N={:5}, {:4} {:>16} : {:6.0f} MFlops [t={:6.0f} ns, {} runs]", N, cplx ? "CPLX" : "REAL", name, mflops, T_ns, max_iter);
    }

    std::fflush(stdout);
    return T_ns;
}

template<std::floating_point T, Transform tramsform>
double cal_benchmark(std::size_t N) {
    const std::size_t                         log2N  = Log2(N);
    std::size_t                               Nfloat = (tramsform == Transform::Complex ? N * 2UZ : N);
    std::vector<T, gr::allocator::Aligned<T>> input(Nfloat);
    std::vector<T, gr::allocator::Aligned<T>> output(Nfloat);
    double                                    t0, t1, tstop, timeDiff, nI;

    assert(std::has_single_bit(N));
    for (std::size_t k = 0UZ; k < Nfloat; ++k) {
        input[k] = std::sqrt(static_cast<T>(k + 1UZ));
    }

    /* PFFFT-U (unordered) benchmark */
    auto        setup = SimdFFT<T, tramsform>(N);
    std::size_t iter  = 0;
    t0                = uclock_sec();
    tstop             = t0 + 0.25; /* benchmark duration: 250 ms */
    do {
        for (std::size_t k = 0; k < 512; ++k) {
            setup.transform(forward, unordered, input, output);
            setup.transform(backward, unordered, input, output);
            ++iter;
        }
        t1 = uclock_sec();
    } while (t1 < tstop);

    timeDiff = (t1 - t0);                               /* duration per fft() */
    nI       = static_cast<double>(iter * (log2N * N)); /* number of iterations "normalized" to O(N) = N*log2(N) */
    return (nI / timeDiff);                             /* normalized iterations per second */
}

template<std::floating_point T, Transform tramsform>
void benchmark_ffts(std::size_t N, int /*withFFTWfullMeas*/, double iterCal, double tmeas[NUM_TYPES][NUM_FFT_ALGOS], int haveAlgo[NUM_FFT_ALGOS], FILE* tableFile) {
    constexpr bool               isComplex  = (tramsform == Transform::Complex);
    const std::size_t            log2N      = Log2(N);
    std::size_t                  nextPow2N  = std::bit_ceil(N);
    [[maybe_unused]] std::size_t log2NextN  = Log2(nextPow2N);
    std::size_t                  pffftPow2N = nextPow2N;

    std::size_t                               Nfloat = (isComplex ? MAX(nextPow2N, pffftPow2N) * 2UZ : MAX(nextPow2N, pffftPow2N));
    std::vector<T, gr::allocator::Aligned<T>> input(Nfloat);
    std::vector<T, gr::allocator::Aligned<T>> output(Nfloat);
    double                                    te, t0, t1, tstop, flops, Tfastest;

    const double      max_test_duration = 0.150;                                                                /* test duration 150 ms */
    double            numIter           = max_test_duration * iterCal / static_cast<double>(log2N * N);         /* number of iteration for max_test_duration */
    const std::size_t step_iter         = std::max<std::size_t>(1UZ, static_cast<std::size_t>(0.01 * numIter)); /* one hundredth */
    std::size_t       max_iter          = std::max<std::size_t>(1UZ, static_cast<std::size_t>(numIter));        /* minimum 1 iteration */
    max_iter                            = std::max(1UZ, static_cast<std::size_t>(numIter));

    memset(input.data(), 0UZ, Nfloat * sizeof(T));
    if (Nfloat < 32UZ) {
        for (std::size_t k = 0UZ; k < Nfloat; k += 4) {
            input[k] = sqrtf(static_cast<T>(k + 1UZ));
        }
    } else {
        for (std::size_t k = 0UZ; k < Nfloat; k += (Nfloat / 16)) {
            input[k] = std::sqrt(static_cast<T>(k + 1UZ));
        }
    }

    for (std::size_t k = 0UZ; k < NUM_TYPES; ++k) {
        for (std::size_t iter = 0UZ; iter < NUM_FFT_ALGOS; ++iter) {
            tmeas[k][iter] = 0.0;
        }
    }

    if (pffftPow2N >= SimdFFT<T, tramsform>::minSize()) {
        te         = uclock_sec();
        auto setup = SimdFFT<T, tramsform>(pffftPow2N);
        t0         = uclock_sec();
        tstop      = t0 + max_test_duration;
        max_iter   = 0;
        do {
            for (std::size_t k = 0UZ; k < step_iter; ++k) {
                setup.template transform<Direction::Forward, Order::Unordered>(input, output);
                setup.template transform<Direction::Backward, Order::Unordered>(input, output);
                ++max_iter;
            }
            t1 = uclock_sec();
        } while (t1 < tstop);

        flops                             = static_cast<double>(max_iter * 2) * ((isComplex ? 5 : 2.5) * static_cast<double>(N) * log(static_cast<double>(N)) / M_LN2); /* see http://www.fftw.org/speed/method.html */
        tmeas[TYPE_ITER][ALGO_PFFFT_U]    = static_cast<double>(max_iter);
        tmeas[TYPE_MFLOPS][ALGO_PFFFT_U]  = flops / 1e6 / (t1 - t0 + 1e-16);
        tmeas[TYPE_DUR_TOT][ALGO_PFFFT_U] = t1 - t0;
        tmeas[TYPE_DUR_NS][ALGO_PFFFT_U]  = show_output("PFFFT-U", N, isComplex, flops, t0, t1, max_iter, tableFile);
        tmeas[TYPE_PREP][ALGO_PFFFT_U]    = (t0 - te) * 1e3;
        haveAlgo[ALGO_PFFFT_U]            = 1;
    } else {
        show_output("PFFFT-U", N, isComplex, -1, -1, -1, gr::meta::invalid_index, tableFile);
    }

    if (pffftPow2N >= SimdFFT<T, tramsform>::minSize()) {
        te         = uclock_sec();
        auto setup = SimdFFT<T, tramsform>(pffftPow2N);
        t0         = uclock_sec();
        tstop      = t0 + max_test_duration;
        max_iter   = 0;
        do {
            for (std::size_t k = 0; k < step_iter; ++k) {
                setup.template transform<Direction::Forward, Order::Ordered>(input, output);
                setup.template transform<Direction::Backward, Order::Ordered>(input, output);
                ++max_iter;
            }
            t1 = uclock_sec();
        } while (t1 < tstop);

        flops                             = static_cast<double>(max_iter * 2) * ((isComplex ? 5 : 2.5) * static_cast<double>(N) * log(static_cast<double>(N)) / M_LN2); /* see http://www.fftw.org/speed/method.html */
        tmeas[TYPE_ITER][ALGO_PFFFT_O]    = static_cast<double>(max_iter);
        tmeas[TYPE_MFLOPS][ALGO_PFFFT_O]  = flops / 1e6 / (t1 - t0 + 1e-16);
        tmeas[TYPE_DUR_TOT][ALGO_PFFFT_O] = t1 - t0;
        tmeas[TYPE_DUR_NS][ALGO_PFFFT_O]  = show_output("PFFFT", N, isComplex, flops, t0, t1, max_iter, tableFile);
        tmeas[TYPE_PREP][ALGO_PFFFT_O]    = (t0 - te) * 1e3;
        haveAlgo[ALGO_PFFFT_O]            = 1;
    } else {
        show_output("PFFFT", N, isComplex, -1, -1, -1, gr::meta::invalid_index, tableFile);
    }

    if (!array_output_format) {
        printf("prepare/ms:     ");
        for (std::size_t iter = 0UZ; iter < NUM_FFT_ALGOS; ++iter) {
            if (haveAlgo[iter] && tmeas[TYPE_DUR_NS][iter] > 0.0) {
                printf("%s %.3f    ", algoName[iter], tmeas[TYPE_PREP][iter]);
            }
        }
        printf("\n");
    }
    Tfastest = 0.0;
    for (std::size_t iter = 0UZ; iter < NUM_FFT_ALGOS; ++iter) {
        if (Tfastest == 0.0 || (tmeas[TYPE_DUR_NS][iter] != 0.0 && tmeas[TYPE_DUR_NS][iter] < Tfastest)) {
            Tfastest = tmeas[TYPE_DUR_NS][iter];
        }
    }
    if (Tfastest > 0.0) {
        if (!array_output_format) {
            printf("relative fast:  ");
        }
        for (std::size_t iter = 0; iter < NUM_FFT_ALGOS; ++iter) {
            if (haveAlgo[iter] && tmeas[TYPE_DUR_NS][iter] > 0.0) {
                tmeas[TYPE_DUR_FASTEST][iter] = tmeas[TYPE_DUR_NS][iter] / Tfastest;
                if (!array_output_format) {
                    printf("%s %.3f    ", algoName[iter], tmeas[TYPE_DUR_FASTEST][iter]);
                }
            }
        }
        if (!array_output_format) {
            printf("\n");
        }
    }

    {
        if (!array_output_format) {
            printf("relative pffft: ");
        }
        for (std::size_t iter = 0; iter < NUM_FFT_ALGOS; ++iter) {
            if (haveAlgo[iter] && tmeas[TYPE_DUR_NS][iter] > 0.0) {
                tmeas[TYPE_REL_PFFFT][iter] = tmeas[TYPE_DUR_NS][iter] / tmeas[TYPE_DUR_NS][ALGO_PFFFT_O];
                if (!array_output_format) {
                    printf("%s %.3f    ", algoName[iter], tmeas[TYPE_REL_PFFFT][iter]);
                }
            }
        }
        if (!array_output_format) {
            printf("\n");
        }
    }

    if (!array_output_format) {
        printf("--\n");
    }
}

int main(int, char**) {
    /* unfortunately, the fft size must be a multiple of 16 for complex FFTs
       and 32 for real FFTs -- a lot of stuff would need to be rewritten to
       handle other cases (or maybe just switch to a scalar fft, I don't know..) */

#if 0 /* include powers of 2 ? */
constexpr std::size_t NUMNONPOW2LENS = 23UZ;
  std::size_t  NnonPow2[NUMNONPOW2LENS] = {
    64, 96, 128, 160, 192,   256, 384, 5*96, 512, 5*128,
    3*256, 800, 1024, 2048, 2400,   4096, 8192, 9*1024, 16384, 32768,
    256*1024, 1024*1024, -1 };
#else
    constexpr std::size_t NUMNONPOW2LENS = 11UZ;
    // std::size_t           NnonPow2[NUMNONPOW2LENS] = {96, 160, 192, 384, 5 * 96, 5 * 128, 3 * 256, 800, 2400, 9 * 1024, gr::meta::invalid_index};
#endif

#ifdef DEBUG
    constexpr std::size_t NUMPOW2FFTLENS = 11UZ; // shortend debug benchmark tests.
#else
    constexpr std::size_t NUMPOW2FFTLENS = 22UZ;
#endif

    constexpr std::size_t MAXNUMFFTLENS = std::max(NUMPOW2FFTLENS, NUMNONPOW2LENS);
    std::size_t           Npow2[NUMPOW2FFTLENS]; /* exp = 1 .. 21, -1 */
    const std::size_t*    Nvalues = nullptr;
    double                tmeas[2UZ][MAXNUMFFTLENS][NUM_TYPES][NUM_FFT_ALGOS];
    double                iterCalReal = 0.0, iterCalCplx = 0.0;

    int         benchReal = 1, benchCplx = 1, withFFTWfullMeas = 0, outputTable2File = 1, usePow2 = 0;
    std::size_t max_N = 1024 * 1024 * 2;
    int         realCplxIdx;
    FILE*       tableFile = nullptr;

    int haveAlgo[NUM_FFT_ALGOS];

    for (std::size_t k = 1; k <= NUMPOW2FFTLENS; ++k) {
        Npow2[k - 1] = (k == NUMPOW2FFTLENS) ? gr::meta::invalid_index : (1 << k);
    }
    Nvalues = Npow2; /* set default .. for comparisons .. */

    for (std::size_t i = 0; i < NUM_FFT_ALGOS; ++i) {
        haveAlgo[i] = 0;
    }

    using T = float;
    printf("SimdFFT min real fft:    %lu\n", SimdFFT<T, Transform::Real>::minSize());
    printf("SimdFFT min complex fft: %lu\n", SimdFFT<T, Transform::Complex>::minSize());
    printf("\n");

    clock();
    /* double TClockDur = 1.0 / CLOCKS_PER_SEC;
    printf("clock() duration for CLOCKS_PER_SEC = %f sec = %f ms\n", TClockDur, 1000.0 * TClockDur );
    */

    /* calibrate test duration */
    int quicktest = 0;
    if (!quicktest) {
        double t0, t1, dur;
        printf("calibrating fft benchmark duration at size N = 512 ..\n");
        t0 = uclock_sec();
        if (benchReal) {
            iterCalReal = cal_benchmark<T, Transform::Real>(512UZ);
            printf("real fft iterCal = %f\n", iterCalReal);
        }
        if (benchCplx) {
            iterCalCplx = cal_benchmark<T, Transform::Complex>(512UZ);
            printf("cplx fft iterCal = %f\n", iterCalCplx);
        }
        t1  = uclock_sec();
        dur = t1 - t0;
        printf("calibration done in %f sec.\n\n", dur);
    }

    if (!array_output_format) {
        if (benchReal) {
            for (std::size_t i = 0UZ; Nvalues[i] > 0 && Nvalues[i] <= max_N; ++i) {
                benchmark_ffts<T, Transform::Real>(Nvalues[i], withFFTWfullMeas, iterCalReal, tmeas[0][i], haveAlgo, nullptr);
            }
        }
        if (benchCplx) {
            for (std::size_t i = 0UZ; Nvalues[i] > 0 && Nvalues[i] <= max_N; ++i) {
                benchmark_ffts<T, Transform::Complex>(Nvalues[i], withFFTWfullMeas, iterCalCplx, tmeas[1][i], haveAlgo, nullptr);
            }
        }

    } else {
        if (outputTable2File) {
            tableFile = fopen(usePow2 ? "bench-fft-table-pow2.txt" : "bench-fft-table-non2.txt", "w");
        }
        /* print table headers */
        printf("table shows MFlops; higher values indicate faster computation\n\n");

        {
            print_table("| input len ", tableFile);
            for (realCplxIdx = 0; realCplxIdx < 2; ++realCplxIdx) {
                if ((realCplxIdx == 0 && !benchReal) || (realCplxIdx == 1 && !benchCplx)) {
                    continue;
                }
                for (std::size_t k = 0; k < NUM_FFT_ALGOS; ++k) {
                    if (compiledInAlgo[k]) {
                        print_table(algoTableHeader[k][realCplxIdx], tableFile);
                    }
                }
            }
            print_table("|\n", tableFile);
        }
        /* print table value seperators */
        {
            print_table("|----------", tableFile);
            for (realCplxIdx = 0; realCplxIdx < 2; ++realCplxIdx) {
                if ((realCplxIdx == 0 && !benchReal) || (realCplxIdx == 1 && !benchCplx)) {
                    continue;
                }
                for (std::size_t k = 0; k < NUM_FFT_ALGOS; ++k) {
                    if (compiledInAlgo[k]) {
                        print_table(":|-------------", tableFile);
                    }
                }
            }
            print_table(":|\n", tableFile);
        }

        for (std::size_t i = 0; Nvalues[i] > 0 && Nvalues[i] <= max_N; ++i) {
            double t0;
            double t1;
            print_table_fftsize(Nvalues[i], tableFile);
            t0 = uclock_sec();
            if (benchReal) {
                benchmark_ffts<T, Transform::Real>(Nvalues[i], withFFTWfullMeas, iterCalReal, tmeas[0][i], haveAlgo, tableFile);
            }
            if (benchCplx) {
                benchmark_ffts<T, Transform::Complex>(Nvalues[i], withFFTWfullMeas, iterCalCplx, tmeas[1][i], haveAlgo, tableFile);
            }
            t1 = uclock_sec();
            print_table("|\n", tableFile);
            /* printf("all ffts for size %d took %f sec\n", Nvalues[i], t1-t0); */
            (void)t0;
            (void)t1;
        }
        fprintf(stdout, " (numbers are given in MFlops)\n");
        if (outputTable2File) {
            fclose(tableFile);
        }
    }

    printf("\n");

    return 0;
}

#include <cmath>
#include <complex>
#include <numbers>
#include <print>
#include <random>
#include <vector>

template<Transform transform_, Order ordering_, std::size_t N>
struct Flags {
    constexpr static Transform   transform = transform_;
    constexpr static Order       ordering  = ordering_;
    constexpr static std::size_t size      = N;
};

template<std::floating_point T>
auto generate_chirp(std::size_t N, T f_start, T f_end) {
    std::vector<T> signal(N);
    for (std::size_t n = 0; n < N; ++n) {
        T t       = T(n) / T(N);
        T freq    = f_start + (f_end - f_start) * t;
        signal[n] = std::sin(T{2} * std::numbers::pi_v<T> * freq * T(n));
    }
    return signal;
}

boost::ut::suite<"SimdFFT Comprehensive"> _ = [] {
    using namespace boost::ut;
    "SimdFFT<T> tests"_test = []<typename T> {
        "invalid transform size detection"_test = [](std::size_t invalidN) {
            expect(!SimdFFT<T, Transform::Real>::canProcessSize(invalidN, Order::Ordered));
            expect(!SimdFFT<T, Transform::Complex>::canProcessSize(invalidN, Order::Ordered) || invalidN == 48); // 48 is OK for complex

            // verify exception thrown
            expect(throws<gr::exception>([&invalidN] {
                SimdFFT<T, Transform::Real>               setup(invalidN);
                std::vector<T, gr::allocator::Aligned<T>> in(invalidN);
                std::vector<T, gr::allocator::Aligned<T>> out(invalidN);
                setup.transform(forward, ordered, in, out);
            })) << std::format("N={} should throw", invalidN);
        } | std::array{7UZ, 11UZ, 14UZ, 17UZ, 31UZ, 35UZ, 48UZ, 49UZ}; // primes and non-factorable

        using enum Transform;
        using enum Order;
        "sine-wave detection"_test =
            []<typename Args>() {
                constexpr std::size_t N = std::remove_reference_t<Args>::size;
                using Flag              = std::remove_reference_t<Args>;
                if (!SimdFFT<T, Flag::transform>::canProcessSize(N, Flag::ordering)) {
                    skip / test(std::format("unsupported - N={} {} {} sine-wave FFT test", N, Flag::transform, Flag::ordering)) = [] {};
                    return; // skip
                }

                constexpr auto valid_frequencies = []<std::size_t N>() {
                    std::array<T, 4> freqs{};
                    std::size_t      idx = 0;
                    // choose k/N where k creates integer periods
                    for (auto k : {N / 8, N / 4, 3 * N / 8, N / 2 - 1}) {
                        if (k > 0 && k < N / 2) {
                            freqs[idx++] = static_cast<T>(k) / T(N);
                        }
                    }
                    return std::pair{freqs, idx};
                };

                constexpr bool                                       is_real = (Flag::transform == Real);
                auto [freqs, num_freqs]                                      = valid_frequencies.template operator()<N>();

                for (std::size_t f_idx = 0; f_idx < num_freqs; ++f_idx) {
                    T           freq         = T(freqs[f_idx]);
                    std::size_t expected_bin = static_cast<std::size_t>(freq * N);

                    std::vector<T, gr::allocator::Aligned<T>> input(is_real ? N : 2 * N);
                    std::vector<T, gr::allocator::Aligned<T>> output(is_real ? N : 2 * N);

                    // generate sine wave
                    for (std::size_t n = 0; n < N; ++n) {
                        T phase = T{2} * std::numbers::pi_v<T> * freq * T(n);
                        if constexpr (is_real) {
                            input[n] = std::sin(phase);
                        } else {
                            input[2 * n]     = std::cos(phase);
                            input[2 * n + 1] = std::sin(phase);
                        }
                    }

                    SimdFFT<T, Flag::transform> setup(N);
                    setup.template transform<Direction::Forward, Flag::ordering>(input, output);

                    // compute magnitudes
                    const std::size_t num_bins = is_real ? (N / 2 + 1) : N;
                    std::vector<T>    magnitudes(num_bins);

                    if constexpr (is_real) {
                        magnitudes[0]     = std::abs(output[0]) / T(N);
                        magnitudes[N / 2] = std::abs(output[1]) / T(N);
                        for (std::size_t k = 1; k < N / 2; ++k) {
                            magnitudes[k] = std::hypot(output[2 * k], output[2 * k + 1]) / T(N);
                        }
                    } else {
                        for (std::size_t k = 0; k < N; ++k) {
                            magnitudes[k] = std::hypot(output[2 * k], output[2 * k + 1]) / T(N);
                        }
                    }

                    const auto        peak_it  = std::max_element(magnitudes.begin(), magnitudes.end());
                    const std::size_t peak_bin = static_cast<std::size_t>(std::distance(magnitudes.begin(), peak_it));
                    expect(eq(peak_bin, expected_bin)) << std::format("N={}, f={:.3f}, bin {}, expected {}", N, freq, peak_bin, expected_bin);
                }
            } |
            std::tuple{Flags<Real, Ordered, 32UZ>{}, Flags<Real, Ordered, 48UZ>{} /* should be impossible for real-valued FFT */, Flags<Real, Ordered, 64UZ>{}, Flags<Real, Ordered, 128UZ>{}, // real valued tests
                Flags<Real, Ordered, 160UZ>{}, Flags<Real, Ordered, 512UZ>{}, Flags<Real, Ordered, 1024UZ>{}, Flags<Complex, Ordered, 16UZ>{},                                                 //
                Flags<Complex, Ordered, 32UZ>{}, Flags<Complex, Ordered, 48UZ>{}, Flags<Complex, Ordered, 64UZ>{}, Flags<Complex, Ordered, 128UZ>{},                                           //
                Flags<Complex, Ordered, 140UZ>{}, Flags<Complex, Ordered, 1024UZ>{}};                                                                                                          // complex tests

        "sine round-trip identity"_test =
            []<typename Args>() {
                constexpr std::size_t N = std::remove_reference_t<Args>::size;
                using Flag              = std::remove_reference_t<Args>;
                if (!SimdFFT<T, Flag::transform>::canProcessSize(N, Flag::ordering)) {
                    skip / test(std::format("unsupported - N={} {} {} sine round-trip identity FFT test", N, Flag::transform, Flag::ordering)) = [] {};
                    return; // skip
                }

                const std::size_t nSamples  = Flag::transform == Real ? N : 2 * N;
                const T           tolerance = T(1e-5f) * T(N);

                std::vector<T, gr::allocator::Aligned<T>> input(nSamples);
                std::vector<T, gr::allocator::Aligned<T>> spectrum(nSamples);
                std::vector<T, gr::allocator::Aligned<T>> reconstructed(nSamples);

                // multi-tone signal
                for (std::size_t i = 0; i < N; ++i) {
                    T val = std::sin(T{2} * std::numbers::pi_v<T> * T(i) / T(8)) + T{0.5} * std::cos(T{2} * std::numbers::pi_v<T> * T(i) / T(4));
                    if constexpr (Flag::transform == Real) {
                        input[i] = val;
                    } else {
                        input[2 * i]     = val;
                        input[2 * i + 1] = std::sin(T{2} * std::numbers::pi_v<T> * T(i) / T(6));
                    }
                }

                SimdFFT<T, Flag::transform> setup(N);
                setup.template transform<Direction::Forward, Flag::ordering>(input, spectrum);
                T spectrum_energy = T{0};
                for (std::size_t i = 0; i < nSamples; ++i) {
                    spectrum_energy += std::abs(spectrum[i]);
                }

                std::string spectrumE = std::format("spectrum energy: {}, first few values: {} {} {} {}", //
                    spectrum_energy, spectrum[0], spectrum[1], spectrum[2], spectrum[3]);
                setup.template transform<Direction::Backward, Flag::ordering>(spectrum, reconstructed);

                for (auto& val : reconstructed) {
                    val /= N;
                }

                T max_error = T{0};
                for (std::size_t i = 0; i < nSamples; ++i) {
                    max_error = std::max(max_error, std::abs(input[i] - reconstructed[i]));
                    expect(approx(input[i], reconstructed[i], tolerance)) << std::format("index: {}, N={}, {}, {}, error={} - energy={}", i, N, Flag::transform, Flag::ordering, max_error, spectrumE);
                    if (std::abs(input[i] - reconstructed[i]) > tolerance) {
                        return;
                    }
                }

                expect(le(max_error, tolerance)) << std::format("N={}, error={} - energy={}", N, max_error, spectrumE);
            } |
            std::tuple{
                Flags<Real, Ordered, 32UZ>{}, Flags<Real, Ordered, 64UZ>{}, Flags<Real, Ordered, 128UZ>{}, Flags<Real, Ordered, 256UZ>{}, Flags<Real, Ordered, 512UZ>{}, Flags<Real, Ordered, 1024UZ>{}, Flags<Real, Ordered, 2048UZ>{},                                                                                   // real-valued, ordered
                Flags<Real, Unordered, 32UZ>{}, Flags<Real, Unordered, 48UZ>{}, Flags<Real, Unordered, 64UZ>{}, Flags<Real, Unordered, 128UZ>{}, Flags<Real, Unordered, 256UZ>{}, Flags<Real, Unordered, 512UZ>{}, Flags<Real, Unordered, 1024UZ>{}, Flags<Real, Unordered, 2048UZ>{},                                     // real-valued, unordered
                Flags<Complex, Ordered, 32UZ>{}, Flags<Complex, Ordered, 64UZ>{}, Flags<Complex, Ordered, 48UZ>{}, Flags<Complex, Ordered, 128UZ>{}, Flags<Complex, Ordered, 1024UZ>{}, Flags<Complex, Ordered, 2048UZ>{},                                                                                                 // complex ordered
                Flags<Complex, Unordered, 32UZ>{}, Flags<Complex, Unordered, 64UZ>{}, Flags<Complex, Unordered, 48UZ>{}, Flags<Complex, Unordered, 128UZ>{}, Flags<Complex, Unordered, 1024UZ>{}, Flags<Complex, Unordered, 2048UZ>{},                                                                                     // complex unordered
                Flags<Real, Unordered, 48UZ>{} /* 2^4 × 3 */, Flags<Real, Unordered, 60UZ>{} /* 2^4 × 3 * 5 */, Flags<Real, Unordered, 80UZ>{} /* 2^4 × 5 */, Flags<Real, Unordered, 96UZ>{} /* 2^5 × 3 */, Flags<Real, Unordered, 96UZ>{} /* 2^5 × 3 */, Flags<Real, Unordered, 160UZ>{}, /* 2^5 × 5 */                   // real-valued, radix 3 & 6
                Flags<Complex, Unordered, 48UZ>{} /* 2^4 × 3 */, Flags<Complex, Unordered, 60UZ>{} /* 2^4 × 3 * 5 */, Flags<Complex, Unordered, 80UZ>{} /* 2^4 × 5 */, Flags<Complex, Unordered, 96UZ>{} /* 2^5 × 3 */, Flags<Complex, Unordered, 96UZ>{} /* 2^5 × 3 */, Flags<Complex, Unordered, 160UZ>{}, /* 2^5 × 5 */ // complex unordered, radix 3 & 6
                Flags<Complex, Ordered, 48UZ>{} /* 2^4 × 3 */, Flags<Complex, Ordered, 60UZ>{} /* 2^4 × 3 * 5 */, Flags<Complex, Ordered, 80UZ>{} /* 2^4 × 5 */, Flags<Complex, Ordered, 96UZ>{} /* 2^5 × 3 */, Flags<Complex, Ordered, 96UZ>{} /* 2^5 × 3 */, Flags<Complex, Ordered, 160UZ>{} /* 2^5 × 5 */              // complex ordered, radix 3 & 6
            };

        // , 192UZ /* 2^6 × 3 */, 240UZ /* 2^4 × 3 × 5 */, 256UZ /* 2^8 */, 320UZ /* 2^6 × 5 */,     //
        // 384UZ /* 2^7 × 3 */, 480UZ /* 2^5 × 3 × 5 */, 512UZ /* 2^9 */, 1024UZ /* 2^10 */, 2048UZ /* 2^11 */          //

        "chirp round-trip identity"_test =
            []<typename Args>() {
                constexpr std::size_t N = std::remove_reference_t<Args>::size;
                using Flag              = std::remove_reference_t<Args>;

                if (!SimdFFT<T, Flag::transform>::canProcessSize(N, Flag::ordering)) {
                    skip / test(std::format("unsupported - N={} {} {} FFT chirp round-trip identity test", N, Flag::transform, Flag::ordering)) = [] {};
                    return;
                }

                const std::size_t nSamples  = Flag::transform == Real ? N : 2 * N;
                const T           tolerance = T(1e-5f) * N;

                std::vector<T, gr::allocator::Aligned<T>> input(nSamples);
                std::vector<T, gr::allocator::Aligned<T>> spectrum(nSamples);
                std::vector<T, gr::allocator::Aligned<T>> reconstructed(nSamples);

                // chirp from f=0.05 to f=0.45
                auto chirp = generate_chirp<T>(N, T(0.05f), T(0.45f));

                for (std::size_t n = 0; n < N; ++n) {
                    if constexpr (Flag::transform == Real) {
                        input[n] = chirp[n];
                    } else {
                        input[2 * n]     = chirp[n];
                        input[2 * n + 1] = generate_chirp<T>(N, T(0.1f), T(0.4))[n];
                    }
                }

                SimdFFT<T, Flag::transform> setup(N);
                setup.template transform<Direction::Forward, Flag::ordering>(input, spectrum);
                setup.template transform<Direction::Backward, Flag::ordering>(spectrum, reconstructed);

                for (auto& val : reconstructed) {
                    val /= N;
                }

                T max_error = T{0};
                for (std::size_t i = 0; i < nSamples; ++i) {
                    max_error = std::max(max_error, std::abs(input[i] - reconstructed[i]));
                }

                expect(max_error < tolerance) << std::format("chirp N={}, error={}", N, max_error);
            } |
            std::tuple{Flags<Real, Ordered, 32>{}, Flags<Real, Ordered, 64>{}, Flags<Real, Unordered, 32>{}, Flags<Real, Unordered, 64>{},           //
                Flags<Complex, Unordered, 32>{}, Flags<Complex, Unordered, 64>{}, Flags<Complex, Unordered, 96>{}, Flags<Complex, Unordered, 512>{}, // complex unordered
                Flags<Complex, Ordered, 32>{}, Flags<Complex, Ordered, 64>{}, Flags<Complex, Unordered, 96>{}, Flags<Complex, Ordered, 512>{}};      // complex ordered

        "radix"_test = [](Transform transform) {
            constexpr static std::size_t L = vec<T, 4>::size();

            "radix"_test = [&transform]<typename Args>() {
                constexpr std::size_t Radix = 2UZ;
                test(std::format("{} radix-{}", transform, Radix)) =
                    [&transform, &Radix](std::size_t N) {
                        if (N % Radix != 0) {
                            return; // skip if N is not divisible by radix
                        }

                        // Calculate parameters based on radix and transform type
                        const std::size_t stride = [&]() {
                            if (transform == Complex) {
                                return Radix;
                            }
                            // Real transforms: stride is 2*radix (Hermitian symmetry)
                            return 2 * Radix;
                        }();

                        const std::size_t nGroups = N / stride;

                        // Radix-r butterfly: needs (radix+1)×logical_size for safety margin
                        const std::size_t bufSize = (Radix + 1) * nGroups * stride * L;

                        std::vector<T, gr::allocator::Aligned<T>> input(bufSize);
                        std::iota(input.begin(), input.end(), T(0));

                        // Twiddle factors: (radix-1) complex values = 2*(radix-1) floats
                        std::vector<T> twiddles(2 * (Radix - 1));
                        for (std::size_t i = 0; i < Radix - 1; ++i) {
                            const T angle       = -2 * std::numbers::pi_v<T> * T(i + 1) / T(Radix);
                            twiddles[2 * i]     = std::cos(angle);
                            twiddles[2 * i + 1] = std::sin(angle);
                        }

                        std::vector<T, gr::allocator::Aligned<T>> fwdOutput(bufSize);
                        std::vector<T, gr::allocator::Aligned<T>> bwdOutput(bufSize);

                        if constexpr (Radix == 2UZ) {
                            if (transform == Complex) {
                                complexRadix2<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            } else {
                                realRadix2<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 3UZ) {
                            if (transform == Complex) {
                                complexRadix3<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            } else {
                                realRadix3<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 4UZ) {
                            if (transform == Complex) {
                                complexRadix4<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            } else {
                                realRadix4<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 5UZ) {
                            if (transform == Complex) {
                                complexRadix5<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            } else {
                                realRadix5<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            }
                        }

                        expect(std::all_of(fwdOutput.begin(), fwdOutput.end(), [](T x) { return std::isfinite(x); })) << std::format("forward radix-{}: invalid values detected", Radix);

                        if constexpr (Radix == 2) {
                            if (transform == Complex) {
                                complexRadix2<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            } else {
                                realRadix2<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 3) {
                            if (transform == Complex) {
                                complexRadix3<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            } else {
                                realRadix3<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 4) {
                            if (transform == Complex) {
                                complexRadix4<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            } else {
                                realRadix4<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 5) {
                            if (transform == Complex) {
                                complexRadix5<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            } else {
                                realRadix5<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            }
                        }

                        expect(std::all_of(bwdOutput.begin(), bwdOutput.end(), [](T x) { return std::isfinite(x); })) << std::format("backward radix-{}: invalid values detected", Radix);
                    } |
                    std::array{
                        32UZ /* 2^5 */, 48UZ /* 2^4 × 3 */, 64UZ /* 2^6 */, 80UZ /* 2^4 × 5 */, 96UZ /* 2^5 × 3 */, 128UZ /* 2^7 */, //
                        160UZ /* 2^5 × 5 */, 192UZ /* 2^6 × 3 */, 240UZ /* 2^4 × 3 × 5 */, 256UZ /* 2^8 */, 320UZ /* 2^6 × 5 */,     //
                        384UZ /* 2^7 × 3 */, 480UZ /* 2^5 × 3 × 5 */, 512UZ /* 2^9 */, 1024UZ /* 2^10 */, 2048UZ /* 2^11 */          //
                    };
            } | std::tuple{std::integral_constant<std::size_t, 2UZ>{}, std::integral_constant<std::size_t, 4UZ>{}, std::integral_constant<std::size_t, 3UZ>{}, std::integral_constant<std::size_t, 5UZ>{}};
        } | std::array{Complex, Real};

        /*
        skip / "hermitian symmetry for real signals"_test = [](std::size_t N) {
            std::vector<T, gr::allocator::Aligned<T>> input(N);
            std::vector<T, gr::allocator::Aligned<T>> output(N);

            std::mt19937                      rng(42);
            std::uniform_real_distribution<T> dist(T{-1}, T{1});
            for (auto& val : input) {
                val = dist(rng);
            }

            PFFFT_Setup<T, Real> setup(N);
            pffft_transform<Direction::Forward, Ordered>(setup, input, output);

            // check X[k] = conj(X[N-k])
            const T tolerance = T(1e-5f);
            for (std::size_t k = 1; k < N / 2; ++k) {
                T re_k  = output[2 * k];
                T im_k  = output[2 * k + 1];
                T re_nk = output[2 * (N - k)];
                T im_nk = output[2 * (N - k) + 1];

                expect(std::abs(re_k - re_nk) < tolerance) << std::format("Re symmetry at k={}", k);
                expect(std::abs(im_k + im_nk) < tolerance) << std::format("Im symmetry at k={}", k);
            }
        } | std::array{32UZ, 128UZ, 1024UZ};
        */

        "DC edge cases"_test = [](std::size_t N) {
            std::vector<T, gr::allocator::Aligned<T>> input(N, T(1.5f)); // constant DC
            std::vector<T, gr::allocator::Aligned<T>> output(N);

            SimdFFT<T, Real> setup(N);
            setup.template transform<Direction::Forward, Ordered>(input, output);

            // all energy should be at DC
            expect(le(std::abs(output[0] - T(1.5) * T(N)), T(1e-4f))) << "DC component";

            for (std::size_t k = 1; k < N / 2; ++k) {
                const T mag = std::hypot(output[2 * k], output[2 * k + 1]);
                expect(le(mag, T(1e-4))) << std::format("bin {} should be zero", k);
            }
        } | std::array{32UZ, 64UZ, 256UZ};

        "Nyquist signal edge case"_test = [](std::size_t N) {
            std::vector<T, gr::allocator::Aligned<T>> input(N);
            for (std::size_t n = 0; n < N; ++n) {
                input[n] = (n % 2 == 0) ? T{1} : T{-1}; // alternating +/- at Nyquist
            }

            std::vector<T, gr::allocator::Aligned<T>> output(N);
            SimdFFT<T, Real>                          setup(N);
            setup.template transform<Direction::Forward, Ordered>(input, output);

            // all energy at Nyquist (bin N/2)
            expect(gt(std::abs(output[1]), T(0.9) * T(N))) << "Nyquist magnitude";
        } | std::array{32UZ, 64UZ, 128UZ, 256UZ};

        "linearity"_test = [](std::size_t N) {
            std::vector<T, gr::allocator::Aligned<T>> x(2 * N), y(2 * N);
            std::vector<T, gr::allocator::Aligned<T>> fx(2 * N), fy(2 * N), fsum(2 * N);
            std::vector<T, gr::allocator::Aligned<T>> sum(2 * N);

            std::mt19937                      rng(123);
            std::uniform_real_distribution<T> dist(T{-1}, T{1});

            for (std::size_t i = 0; i < 2 * N; ++i) {
                x[i]   = dist(rng);
                y[i]   = dist(rng);
                sum[i] = x[i] + y[i];
            }

            SimdFFT<T, Complex> setup(N);
            setup.transform(forward, ordered, x, fx);
            setup.transform(forward, ordered, y, fy);
            setup.transform(forward, ordered, sum, fsum);

            // check FFT(x+y) = FFT(x) + FFT(y)
            T max_error = T{0};
            for (std::size_t i = 0UZ; i < 2UZ * N; ++i) {
                max_error = std::max(max_error, std::abs(fsum[i] - (fx[i] + fy[i])));
            }

            expect(lt(max_error, T(1e-4) * T(N))) << std::format("linearity error={}", max_error);
        } | std::array{32UZ, 64UZ, 128UZ, 256UZ};
    } | std::tuple{float{} /*, double{}*/};
};
