// Cross-N correctness gate for the block-decomposed radix-2 Stockham GPU FFT (SyclFFT).
//
// Sweeps N = 8..65536 x wgSize {256,512,1024} x nBatches {1,3}, checking each GPU forward transform against the host
// Stockham oracle, an independent direct DFT for small N, and an inverse round-trip. A SYCL CPU device runs the same
// kernel, so the gate keeps its teeth where only host:sycl resolves.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/fourier/SyclFFT.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>

namespace {

using GC = gr::complex<float>;

constexpr double      kTol     = 1e-4;
constexpr std::size_t kDftMaxN = 256; // direct DFT is O(N^2) — an independent anchor only for the small sizes

[[gnu::constructor]] void configureAdaptiveCppStorage() {
    setenv("ACPP_APPDB_DIR", "/tmp/gnuradio4-acpp-appdb", 0);
    setenv("XDG_DATA_HOME", "/tmp/gnuradio4-acpp-data", 0);
}

std::vector<GC> seededInput(std::size_t total, std::uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<GC>                       data(total);
    for (GC& z : data) {
        z = {dist(rng), dist(rng)};
    }
    return data;
}

// direct O(N^2) forward DFT in double precision — independent of the Stockham index algebra
std::vector<GC> directDft(std::span<const GC> in) {
    const std::size_t N     = in.size();
    const double      theta = -2.0 * std::numbers::pi_v<double> / static_cast<double>(N);
    std::vector<GC>   out(N);
    for (std::size_t k = 0; k < N; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (std::size_t n = 0; n < N; ++n) {
            const double ang = theta * static_cast<double>(k) * static_cast<double>(n);
            const double c   = std::cos(ang);
            const double s   = std::sin(ang);
            re += static_cast<double>(in[n].re) * c - static_cast<double>(in[n].im) * s;
            im += static_cast<double>(in[n].re) * s + static_cast<double>(in[n].im) * c;
        }
        out[k] = {static_cast<float>(re), static_cast<float>(im)};
    }
    return out;
}

struct ErrMetrics {
    double relL2   = 0.0;
    double relLinf = 0.0;
};

ErrMetrics compare(std::span<const GC> ref, std::span<const GC> got) {
    double l2Num   = 0.0;
    double l2Den   = 0.0;
    double linfNum = 0.0;
    double linfDen = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = std::hypot(static_cast<double>(got[i].re) - static_cast<double>(ref[i].re), static_cast<double>(got[i].im) - static_cast<double>(ref[i].im));
        const double r = std::hypot(static_cast<double>(ref[i].re), static_cast<double>(ref[i].im));
        l2Num += d * d;
        l2Den += r * r;
        linfNum = std::max(linfNum, d);
        linfDen = std::max(linfDen, r);
    }
    return {l2Den > 0.0 ? std::sqrt(l2Num / l2Den) : std::sqrt(l2Num), linfDen > 0.0 ? linfNum / linfDen : linfNum};
}

bool runCell(gr::device::DeviceContext& ctx, std::size_t N, std::size_t wgSize, std::size_t nBatches) {
    const std::size_t total = N * nBatches;

    gr::device::SyclFFT fft;
    fft.init(ctx, N, {.workgroupSize = wgSize});

    const std::vector<GC> input = seededInput(total, static_cast<std::uint32_t>(N * 131u + wgSize * 7u + nBatches));

    // host reference: forwardStockhamCpu is in-place and single-transform — run it per batch slice
    std::vector<GC> reference = input;
    for (std::size_t b = 0; b < nBatches; ++b) {
        fft.forwardStockhamCpu(std::span<GC>{reference.data() + b * N, N});
    }

    gr::device::DeviceBuffer dData = ctx.allocateDevice<GC>(total);
    if (!dData) {
        std::printf("  N=%6zu wg=%4zu x%zu | device allocation returned null   FAIL\n", N, wgSize, nBatches);
        return false;
    }

    ctx.copyHostToDevice(input.data(), dData, total);
    fft.forwardBatch(ctx, std::span<GC>{dData.devicePointer<GC>(), total}, N);
    std::vector<GC> gpuForward(total);
    ctx.copyDeviceToHost(dData, gpuForward.data(), total);
    const ErrMetrics fwd = compare(reference, gpuForward);

    // independent direct-DFT anchor for the small sizes (does not trust the Stockham oracle)
    double dftRelL2 = 0.0;
    bool   dftOk    = true;
    if (N <= kDftMaxN) {
        for (std::size_t b = 0; b < nBatches; ++b) {
            const std::vector<GC> dft = directDft(std::span<const GC>{input.data() + b * N, N});
            const ErrMetrics      m   = compare(dft, std::span<const GC>{gpuForward.data() + b * N, N});
            dftRelL2                  = std::max({dftRelL2, m.relL2, m.relLinf});
        }
        dftOk = dftRelL2 < kTol;
    }

    // inverse round-trip: inverseBatch already applies 1/N, so the result is compared to the input directly
    ctx.copyHostToDevice(input.data(), dData, total);
    fft.forwardBatch(ctx, std::span<GC>{dData.devicePointer<GC>(), total}, N);
    fft.inverseBatch(ctx, std::span<GC>{dData.devicePointer<GC>(), total}, N);
    std::vector<GC> roundTrip(total);
    ctx.copyDeviceToHost(dData, roundTrip.data(), total);
    ctx.deallocate(dData);
    const ErrMetrics rt = compare(input, roundTrip);

    const bool ok = fwd.relL2 < kTol && fwd.relLinf < kTol && rt.relL2 < kTol && rt.relLinf < kTol && dftOk;

    char dftField[24];
    if (N <= kDftMaxN) {
        std::snprintf(dftField, sizeof(dftField), "dft=%.1e", dftRelL2);
    } else {
        std::snprintf(dftField, sizeof(dftField), "%-8s", "");
    }
    std::printf("  N=%6zu wg=%4zu x%zu | fwd L2=%.1e Linf=%.1e | %s | rt L2=%.1e%s\n", N, wgSize, nBatches, fwd.relL2, fwd.relLinf, dftField, rt.relL2, ok ? "" : "   FAIL");
    return ok;
}

} // namespace

// plain main (not a Boost.UT suite): kernels launch while the AdaptiveCpp runtime is alive, and the 77 skip is honoured
int main() {
    if (!gr::device::registerSyclRuntime()) {
        std::printf("SKIP: built without a SYCL backend\n");
        return 77;
    }

    gr::device::DeviceContextRegistry& registry = gr::device::DeviceContextRegistry::instance();

    // prefer a real GPU, else a SYCL CPU device running the very same kernel, so the gate keeps its teeth
    gr::device::DeviceContext* ctx = nullptr;
    std::string_view           domain;
    for (const std::string_view candidate : {std::string_view{"gpu:sycl"}, std::string_view{"host:sycl"}}) {
        gr::device::DeviceContext* sched = registry.tryResolve(candidate);
        if (sched != nullptr && sched->backend() == gr::device::DeviceBackend::SYCL) {
            ctx    = sched;
            domain = candidate;
            break;
        }
    }
    if (ctx == nullptr) {
        std::printf("SKIP: neither gpu:sycl nor host:sycl resolves to a SYCL device\n");
        return 77;
    }

    try {
        std::printf("SyclFFT cross-N Stockham gate on %.*s (%s)\n", static_cast<int>(domain.size()), domain.data(), ctx->shortName().c_str());
        int failures = 0;
        int cells    = 0;
        for (std::size_t N = 8; N <= 65536; N <<= 1) {
            for (std::size_t wgSize : {256UZ, 512UZ, 1024UZ}) {
                for (std::size_t nBatches : {1UZ, 3UZ}) {
                    ++cells;
                    if (!runCell(*ctx, N, wgSize, nBatches)) {
                        ++failures;
                    }
                }
            }
        }
        std::printf("\n%d/%d cells passed (tol %.0e)\n", cells - failures, cells, kTol);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: exception: %s\n", e.what());
        return 1;
    }
}
