#include "gnuradio-4.0/meta/utils.hpp"
#include <gnuradio-4.0/algorithm/fourier/SimdFFT.hpp>

#include <assert.h>
#include <bit>
#include <math.h>
#include <print>
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

using namespace gr::algorithm;

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
