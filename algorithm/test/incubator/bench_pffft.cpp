
#include "gnuradio-4.0/meta/utils.hpp"

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

#include "pffft.cpp"
#include "pffft.h"

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

const int saveType[NUM_TYPES] = {
    1, /* "1-preparation-in-ms" */
    0, /* "2-timePerFft-in-ns"  */
    0, /* "3-rel-fastest"       */
    1, /* "4-rel-pffft"         */
    1, /* "5-num-iter"          */
    1, /* "6-mflops"            */
    1, /* "7-duration-in-sec"   */
};

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

template<std::floating_point T, fft_transform_t tramsform>
double cal_benchmark(std::size_t N) {
    const std::size_t log2N  = Log2(N);
    std::size_t       Nfloat = (tramsform == fft_transform_t::Complex ? N * 2UZ : N);
    std::size_t       Nbytes = Nfloat * sizeof(T);
    T *               X = pffft_aligned_malloc<T>(Nbytes + sizeof(T)), *Y = pffft_aligned_malloc<T>(Nbytes + 2 * sizeof(T)), *Z = pffft_aligned_malloc<T>(Nbytes);
    double            t0, t1, tstop, timeDiff, nI;

    assert(std::has_single_bit(N));
    for (std::size_t k = 0UZ; k < Nfloat; ++k) {
        X[k] = std::sqrt(static_cast<T>(k + 1UZ));
    }

    /* PFFFT-U (unordered) benchmark */
    auto        s    = PFFFT_Setup<T, tramsform>(N);
    std::size_t iter = 0;
    t0               = uclock_sec();
    tstop            = t0 + 0.25; /* benchmark duration: 250 ms */
    do {
        for (std::size_t k = 0; k < 512; ++k) {
            pffft_transform<fft_direction_t::Forward>(s, X, Z, Y);
            pffft_transform<fft_direction_t::Backward>(s, X, Z, Y);
            ++iter;
        }
        t1 = uclock_sec();
    } while (t1 < tstop);
    pffft_aligned_free(X);
    pffft_aligned_free(Y);
    pffft_aligned_free(Z);

    timeDiff = (t1 - t0);                               /* duration per fft() */
    nI       = static_cast<double>(iter * (log2N * N)); /* number of iterations "normalized" to O(N) = N*log2(N) */
    return (nI / timeDiff);                             /* normalized iterations per second */
}

template<std::floating_point T, fft_transform_t tramsform>
void benchmark_ffts(std::size_t N, int /*withFFTWfullMeas*/, double iterCal, double tmeas[NUM_TYPES][NUM_FFT_ALGOS], int haveAlgo[NUM_FFT_ALGOS], FILE* tableFile) {
    constexpr bool               isComplex  = (tramsform == fft_transform_t::Complex);
    const std::size_t            log2N      = Log2(N);
    std::size_t                  nextPow2N  = std::bit_ceil(N);
    [[maybe_unused]] std::size_t log2NextN  = Log2(nextPow2N);
    std::size_t                  pffftPow2N = nextPow2N;

    std::size_t Nfloat = (isComplex ? MAX(nextPow2N, pffftPow2N) * 2UZ : MAX(nextPow2N, pffftPow2N));
    std::size_t Nmax;
    std::size_t Nbytes = Nfloat * sizeof(T);

    T *    X = pffft_aligned_malloc<T>(Nbytes + sizeof(T)), *Y = pffft_aligned_malloc<T>(Nbytes + 2 * sizeof(T)), *Z = pffft_aligned_malloc<T>(Nbytes);
    double te, t0, t1, tstop, flops, Tfastest;

    const double      max_test_duration = 0.150;                                                                /* test duration 150 ms */
    double            numIter           = max_test_duration * iterCal / static_cast<double>(log2N * N);         /* number of iteration for max_test_duration */
    const std::size_t step_iter         = std::max<std::size_t>(1UZ, static_cast<std::size_t>(0.01 * numIter)); /* one hundredth */
    std::size_t       max_iter          = std::max<std::size_t>(1UZ, static_cast<std::size_t>(numIter));        /* minimum 1 iteration */
    max_iter                            = std::max(1UZ, static_cast<std::size_t>(numIter));

    constexpr float checkVal = 12345.0F;

    /* printf("benchmark_ffts(N = %d, cplx = %d): Nfloat = %d, X_mem = 0x%p, X = %p\n", N, cplx, Nfloat, X_mem, X); */

    memset(X, 0UZ, Nfloat * sizeof(T));
    if (Nfloat < 32UZ) {
        for (std::size_t k = 0UZ; k < Nfloat; k += 4) {
            X[k] = sqrtf(static_cast<T>(k + 1UZ));
        }
    } else {
        for (std::size_t k = 0UZ; k < Nfloat; k += (Nfloat / 16)) {
            X[k] = std::sqrt(static_cast<T>(k + 1UZ));
        }
    }

    for (std::size_t k = 0UZ; k < NUM_TYPES; ++k) {
        for (std::size_t iter = 0UZ; iter < NUM_FFT_ALGOS; ++iter) {
            tmeas[k][iter] = 0.0;
        }
    }

    /* FFTPack benchmark */
    Nmax    = (isComplex ? N * 2 : N);
    X[Nmax] = checkVal;

    /* PFFFT-U (unordered) benchmark */
    Nmax    = (isComplex ? pffftPow2N * 2 : pffftPow2N);
    X[Nmax] = checkVal;
    if (pffftPow2N >= pffft_min_fft_size<T>(isComplex ? fft_transform_t::Complex : fft_transform_t::Real)) {
        te       = uclock_sec();
        auto s   = PFFFT_Setup<T, tramsform>(pffftPow2N);
        t0       = uclock_sec();
        tstop    = t0 + max_test_duration;
        max_iter = 0;
        do {
            for (std::size_t k = 0UZ; k < step_iter; ++k) {
                assert(X[Nmax] == checkVal);
                pffft_transform<fft_direction_t::Forward>(s, X, Z, Y);
                assert(X[Nmax] == checkVal);
                pffft_transform<fft_direction_t::Backward>(s, X, Z, Y);
                assert(X[Nmax] == checkVal);
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

    if (pffftPow2N >= pffft_min_fft_size<T>(isComplex ? fft_transform_t::Complex : fft_transform_t::Real)) {
        te       = uclock_sec();
        auto s   = PFFFT_Setup<T, tramsform>(pffftPow2N);
        t0       = uclock_sec();
        tstop    = t0 + max_test_duration;
        max_iter = 0;
        do {
            for (std::size_t k = 0; k < step_iter; ++k) {
                assert(X[Nmax] == checkVal);
                pffft_transform_ordered<fft_direction_t::Forward>(s, X, Z, Y);
                assert(X[Nmax] == checkVal);
                pffft_transform_ordered<fft_direction_t::Backward>(s, X, Z, Y);
                assert(X[Nmax] == checkVal);
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

    pffft_aligned_free(X);
    pffft_aligned_free(Y);
    pffft_aligned_free(Z);
}

template<std::floating_point T>
bool test_fft_sine_wave(); // just as a sanity check

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

    constexpr std::size_t NUMPOW2FFTLENS = 22UZ;
    constexpr std::size_t MAXNUMFFTLENS  = std::max(NUMPOW2FFTLENS, NUMNONPOW2LENS);
    std::size_t           Npow2[NUMPOW2FFTLENS]; /* exp = 1 .. 21, -1 */
    const std::size_t*    Nvalues = nullptr;
    double                tmeas[2UZ][MAXNUMFFTLENS][NUM_TYPES][NUM_FFT_ALGOS];
    double                iterCalReal = 0.0, iterCalCplx = 0.0;

    int         benchReal = 1, benchCplx = 1, withFFTWfullMeas = 0, outputTable2File = 1, usePow2 = 1;
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
    printf("pffft architecture:    '%s'\n", "stdx::simd");
    printf("pffft SIMD size:       %lu\n", pffft_simd_size<T>());
    printf("pffft min real fft:    %lu\n", pffft_min_fft_size<T>(fft_transform_t::Real));
    printf("pffft min complex fft: %lu\n", pffft_min_fft_size<T>(fft_transform_t::Complex));
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
            iterCalReal = cal_benchmark<T, fft_transform_t::Real>(512UZ);
            printf("real fft iterCal = %f\n", iterCalReal);
        }
        if (benchCplx) {
            iterCalCplx = cal_benchmark<T, fft_transform_t::Complex>(512UZ);
            printf("cplx fft iterCal = %f\n", iterCalCplx);
        }
        t1  = uclock_sec();
        dur = t1 - t0;
        printf("calibration done in %f sec.\n\n", dur);
    }

    if (!array_output_format) {
        if (benchReal) {
            for (std::size_t i = 0UZ; Nvalues[i] > 0 && Nvalues[i] <= max_N; ++i) {
                benchmark_ffts<T, fft_transform_t::Real>(Nvalues[i], withFFTWfullMeas, iterCalReal, tmeas[0][i], haveAlgo, nullptr);
            }
        }
        if (benchCplx) {
            for (std::size_t i = 0UZ; Nvalues[i] > 0 && Nvalues[i] <= max_N; ++i) {
                benchmark_ffts<T, fft_transform_t::Complex>(Nvalues[i], withFFTWfullMeas, iterCalCplx, tmeas[1][i], haveAlgo, nullptr);
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
                benchmark_ffts<T, fft_transform_t::Real>(Nvalues[i], withFFTWfullMeas, iterCalReal, tmeas[0][i], haveAlgo, tableFile);
            }
            if (benchCplx) {
                benchmark_ffts<T, fft_transform_t::Complex>(Nvalues[i], withFFTWfullMeas, iterCalCplx, tmeas[1][i], haveAlgo, tableFile);
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

    bool result = test_fft_sine_wave<float>();
    result &= test_fft_sine_wave<double>(); // TODO: enable this

    return result ? 0 : 1;
}

#include <cmath>
#include <complex>
#include <iomanip>
#include <print>
#include <string>
#include <string_view>
#include <vector>

template<std::floating_point T>
bool test_fft_sine_wave() {
    constexpr std::size_t N = 128UZ;
    constexpr T           freq_normalized{static_cast<T>(0.25)}; // f/fs = 0.25
    constexpr std::size_t expected_bin = static_cast<std::size_t>(freq_normalized * N);
    constexpr T           tolerance{static_cast<T>(1e-7f) * N};

    constexpr std::string_view kRed             = "\033[31m";
    constexpr std::string_view kGreen           = "\033[32m";
    constexpr std::string_view kReset           = "\033[0m";
    bool                       all_tests_passed = true;

    // Test 1: Real-valued FFT with pure sine wave
    {
        std::println("Testing REAL FFT with pure sine wave at f={}*fs (bin: {})...", freq_normalized, expected_bin);

        T* input  = pffft_aligned_malloc<T>(N * sizeof(T));
        T* output = pffft_aligned_malloc<T>(N * sizeof(T));
        T* work   = pffft_aligned_malloc<T>(N * sizeof(T));

        // generate pure sine wave: sin(2*pi*f*n/N)
        for (std::size_t n = 0; n < N; ++n) {
            input[n] = std::sin(T{2} * std::numbers::pi_v<T> * freq_normalized * static_cast<T>(n));
        }

        PFFFT_Setup<T, fft_transform_t::Real> setup(N);
        pffft_transform_ordered<fft_direction_t::Forward>(setup, input, output, work);

        // For real FFT, output format is: [DC, bin1_real, bin1_imag, ..., binN/2-1_real, binN/2-1_imag, Nyquist]
        // Calculate magnitude for each bin
        std::vector<T> magnitudes(N / 2 + 1, T(0));
        magnitudes[0]     = std::abs(output[0]); // DC component
        magnitudes[N / 2] = std::abs(output[1]); // Nyquist

        for (std::size_t k = 1; k < N / 2; ++k) {
            T real        = output[2 * k];
            T imag        = output[2 * k + 1];
            magnitudes[k] = std::sqrt(real * real + imag * imag);
        }

        // find peak bin
        std::size_t peak_bin = 0UZ;
        T           peak_mag{0};
        for (std::size_t k = 0UZ; k <= N / 2UZ; ++k) {
            if (magnitudes[k] > peak_mag) {
                peak_mag = magnitudes[k];
                peak_bin = k;
            }
        }

        // Normalize peak magnitude (should be approximately N/2 for a unit sine wave)
        peak_mag /= (N / 2);

        std::println("  Peak found at bin {} with normalized magnitude {}", peak_bin, peak_mag);

        // check if peak is at expected bin with correct magnitude
        if (peak_bin != expected_bin) {
            std::println("{}  ERROR: Peak at wrong bin! Expected {}, got {}{}", kRed, expected_bin, peak_bin, kReset);
            all_tests_passed = false;
        } else if (std::abs(peak_mag - T(1)) > tolerance) {
            std::println("{}  ERROR: Peak magnitude incorrect! Expected ~1.0, got {}{}", kRed, peak_mag, kReset);
            all_tests_passed = false;
        } else {
            // check that other bins are near zero
            T noise_floor = T(0);
            for (std::size_t k = 0; k <= N / 2; ++k) {
                if (k != peak_bin) {
                    noise_floor = std::max(noise_floor, magnitudes[k] / (N / 2));
                }
            }
            if (noise_floor > tolerance) {
                std::println("  WARNING: High noise floor: {}", noise_floor);
                ;
            } else {
                std::println("  PASSED! Noise floor: {}", noise_floor);
            }
        }

        pffft_aligned_free(input);
        pffft_aligned_free(output);
        pffft_aligned_free(work);
    }

    // Test 2: Complex-valued FFT with complex exponential
    {
        std::println("\nTesting COMPLEX FFT with complex exponential at f=0.25*fs (bin {})...", expected_bin);

        // allocate aligned memory (2x size for complex)
        T* input  = pffft_aligned_malloc<T>(2 * N * sizeof(T));
        T* output = pffft_aligned_malloc<T>(2 * N * sizeof(T));
        T* work   = pffft_aligned_malloc<T>(2 * N * sizeof(T));

        // generate complex exponential: exp(j*2*pi*f*n/N)
        for (std::size_t n = 0; n < N; ++n) {
            T phase          = T(2) * std::numbers::pi_v<T> * freq_normalized * static_cast<T>(n);
            input[2 * n]     = std::cos(phase); // real part
            input[2 * n + 1] = std::sin(phase); // imaginary part
        }

        PFFFT_Setup<T, fft_transform_t::Complex> setup(N);
        pffft_transform_ordered<fft_direction_t::Forward>(setup, input, output, work);

        // Calculate magnitude for each bin
        std::vector<T> magnitudes(N, T(0));
        for (std::size_t k = 0; k < N; ++k) {
            T real        = output[2 * k];
            T imag        = output[2 * k + 1];
            magnitudes[k] = std::sqrt(real * real + imag * imag);
        }

        // Find peak bin
        std::size_t peak_bin = 0;
        T           peak_mag{0};
        for (std::size_t k = 0; k < N; ++k) {
            if (magnitudes[k] > peak_mag) {
                peak_mag = magnitudes[k];
                peak_bin = k;
            }
        }

        // Normalize peak magnitude (should be approximately N for a unit complex exponential)
        peak_mag /= N;

        std::println("  Peak found at bin {} with normalized magnitude {}", peak_bin, peak_mag);

        // Check if peak is at expected bin with correct magnitude
        if (peak_bin != expected_bin) {
            std::println("{}  ERROR: Peak at wrong bin! Expected {}, got {}{}", kRed, expected_bin, peak_bin, kReset);
            all_tests_passed = false;
        } else if (std::abs(peak_mag - T(1)) > tolerance) {
            std::println("{}  ERROR: Peak magnitude incorrect! Expected ~1.0, got {}{}", kRed, peak_mag, kReset);
            all_tests_passed = false;
        } else {
            // check that other bins are near zero
            T noise_floor{0};
            for (std::size_t k = 0UZ; k < N; ++k) {
                if (k != peak_bin) {
                    noise_floor = std::max(noise_floor, magnitudes[k] / N);
                }
            }
            if (noise_floor > tolerance) {
                std::println("  WARNING: High noise floor: {}", noise_floor);
            } else {
                std::println("  PASSED! Noise floor: {}", noise_floor);
            }
        }

        pffft_aligned_free(input);
        pffft_aligned_free(output);
        pffft_aligned_free(work);
    }

    // Test 3: Inverse FFT reconstruction test (complex)
    {
        std::println("\nTesting COMPLEX FFT forward-inverse reconstruction...");

        T* input         = pffft_aligned_malloc<T>(2 * N * sizeof(T));
        T* spectrum      = pffft_aligned_malloc<T>(2 * N * sizeof(T));
        T* reconstructed = pffft_aligned_malloc<T>(2 * N * sizeof(T));
        T* work          = pffft_aligned_malloc<T>(2 * N * sizeof(T));

        // Generate complex test signal
        for (std::size_t n = 0; n < N; ++n) {
            T phase          = T{2} * std::numbers::pi_v<T> * freq_normalized * static_cast<T>(n);
            input[2 * n]     = std::cos(phase);
            input[2 * n + 1] = std::sin(phase);
        }

        PFFFT_Setup<T, fft_transform_t::Complex> setup(N);

        pffft_transform_ordered<fft_direction_t::Forward>(setup, input, spectrum, work);
        pffft_transform_ordered<fft_direction_t::Backward>(setup, spectrum, reconstructed, work);

        // scale by 1/N (this FFT doesn't normalize the inverse)
        for (std::size_t i = 0; i < 2 * N; ++i) {
            reconstructed[i] /= N;
        }

        // check reconstruction error
        T max_error{0};
        for (std::size_t i = 0; i < 2 * N; ++i) {
            T error   = std::abs(input[i] - reconstructed[i]);
            max_error = std::max(max_error, error);
        }

        std::println("  Max reconstruction error: {}", max_error);
        if (max_error > tolerance) {
            std::println("{}  ERROR: Reconstruction error too large!{}", kRed, kReset);
            all_tests_passed = false;
        } else {
            std::println("  PASSED!\n");
        }

        pffft_aligned_free(input);
        pffft_aligned_free(spectrum);
        pffft_aligned_free(reconstructed);
        pffft_aligned_free(work);
    }

    using namespace std::literals;
    if (all_tests_passed) {
        std::println("{}  ALL {} TESTS PASSED!{}", kGreen, gr::meta::type_name<T>(), kReset);
    } else {
        std::println("{}  SOME  {} TESTS FAILED!{}", kRed, gr::meta::type_name<T>(), kReset);
    }
    return all_tests_passed;
}
