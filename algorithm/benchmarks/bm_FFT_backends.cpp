#include <benchmark.hpp>

#include <bit>
#include <complex>
#include <numbers>
#include <print>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>

#include "bm_FFT_backends_helpers.hpp"

using C = std::complex<float>;

namespace {

std::vector<C> generateTone(std::size_t N, std::size_t bin) {
    std::vector<C> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        float phase = 2.f * std::numbers::pi_v<float> * static_cast<float>(bin) * static_cast<float>(i) / static_cast<float>(N);
        data[i]     = {std::cos(phase), std::sin(phase)};
    }
    return data;
}

constexpr std::array kSizes   = {1024UZ, 2048UZ, 4096UZ, 8192UZ, 16384UZ, 32768UZ, 65536UZ};
constexpr std::array kBatches = {1UZ, 16UZ, 128UZ};

// FFTW convention: 5·N·log₂(N) floating point operations per complex FFT
constexpr std::size_t fftFlops(std::size_t N, std::size_t nBatches = 1) { return nBatches * 5 * N * static_cast<std::size_t>(std::countr_zero(N)); }

} // namespace

const boost::ut::suite<"FFT backend benchmarks"> benchmarks = [] {
    using namespace benchmark;

    // discover available device backends
    auto backends = gr::benchmark_fft::availableBackends();

    std::println("Backends:");
    std::println("  SimdFFT (CPU, reference)");
    for (const auto& b : backends) {
        std::println("  {}", b.shortName);
    }
    std::println("  ops/s normalised to 5·N·log₂(N) FLOP/s (FFTW convention)");
    std::println("");

    for (std::size_t nBatches : kBatches) {
        if (nBatches > 1) {
            ::benchmark::results::add_separator();
        }

        // ── CPU SimdFFT (always available, reference baseline) ──
        for (std::size_t N : kSizes) {
            auto batchInput = std::make_shared<std::vector<C>>(N * nBatches);
            for (std::size_t b = 0; b < nBatches; ++b) {
                auto tone = generateTone(N, (nBatches == 1) ? 5 : b % 32 + 1);
                std::ranges::copy(tone, batchInput->begin() + static_cast<std::ptrdiff_t>(b * N));
            }
            auto cpuFft = std::make_shared<gr::algorithm::FFT<C, C>>();
            auto output = std::make_shared<std::vector<C>>(N);
            cpuFft->compute(std::span<const C>(batchInput->data(), N), *output);

            ::benchmark::benchmark<10>(std::format("x{:<3}| {:<30} | N={:>5}", nBatches, "SimdFFT", N), fftFlops(N, nBatches)) = [batchInput, cpuFft, output, N, nBatches] {
                for (std::size_t b = 0; b < nBatches; ++b) {
                    cpuFft->compute(std::span<const C>(batchInput->data() + b * N, N), *output);
                }
                return (*output)[0];
            };
        }

        // ── device backends (discovered at runtime) ──
        for (auto& backend : backends) {
            ::benchmark::results::add_separator();

            for (std::size_t N : kSizes) {
                auto batchInput = std::make_shared<std::vector<C>>(N * nBatches);
                for (std::size_t b = 0; b < nBatches; ++b) {
                    auto tone = generateTone(N, (nBatches == 1) ? 5 : b % 32 + 1);
                    std::ranges::copy(tone, batchInput->begin() + static_cast<std::ptrdiff_t>(b * N));
                }

                // init + warmup (JIT, buffer allocation — outside timed section)
                backend.init(N, nBatches);
                ::benchmark::benchmark<10>(std::format("x{:<3}| {:<30} | N={:>5}", nBatches, backend.shortName, N), fftFlops(N, nBatches)) = [&backend, batchInput, N, nBatches] { return backend.compute(batchInput->data(), N, nBatches); };
            }
        }
    }
};

int main() { /* not needed for UT */ }
