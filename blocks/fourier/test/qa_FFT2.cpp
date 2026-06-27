#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <numeric>
#include <span>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/algorithm/fourier/GlslFFT.hpp>
#include <gnuradio-4.0/algorithm/fourier/SyclFFT.hpp>
#include <gnuradio-4.0/basic/TransferBlocks.hpp>
#include <gnuradio-4.0/device/DeviceContextGLSL.hpp>
#include <gnuradio-4.0/fourier/FFT2.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

using namespace boost::ut;
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

std::size_t findPeakBin(std::span<const C> spectrum) {
    std::size_t peak   = 0;
    float       maxMag = 0.f;
    for (std::size_t i = 0; i < spectrum.size(); ++i) {
        float mag = std::abs(spectrum[i]);
        if (mag > maxMag) {
            maxMag = mag;
            peak   = i;
        }
    }
    return peak;
}

float maxError(const std::vector<C>& a, const std::vector<C>& b) {
    float err = 0.f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(a[i] - b[i]));
    }
    return err;
}

// SimdFFT reference
void simdForward(const std::vector<C>& input, std::vector<C>& output) {
    static gr::algorithm::FFT<C, C> fft;
    fft.compute(input, output);
}

using GC = gr::complex<float>;

#if GR_DEVICE_HAS_GL_COMPUTE
// singleton GL context — leaked intentionally to outlive all test suites and avoid
// eglTerminate running before GlslFFT destructors (causes segfault on process exit)
gr::device::DeviceContextGLSL* glCtx() {
    struct State {
        gr::device::GlComputeContext  gl;
        gr::device::DeviceContextGLSL ctx{gl};
        bool                          ok;
        State() : ok(gl.isAvailable()) {}
    };
    static auto& s = *new State();
    return s.ok ? &s.ctx : nullptr;
}
#endif

} // namespace

const suite<"FFT CPU SimdFFT"> cpuSimdTests = [] {
    "forward matches known DFT"_test = [] {
        auto           input = generateTone(4096, 5);
        std::vector<C> output(4096);
        simdForward(input, output);
        expect(eq(findPeakBin(output), 5UZ));
        expect(gt(std::abs(output[5]), 4096.f * 0.9f));
    };

    "forward+inverse round-trip"_test = [] {
        auto           input = generateTone(4096, 7);
        std::vector<C> spectrum(4096), conjugated(4096), recovered(4096);
        simdForward(input, spectrum);
        std::ranges::transform(spectrum, conjugated.begin(), [](C z) { return std::conj(z); });
        gr::algorithm::FFT<C, C> fftInv;
        fftInv.compute(conjugated, recovered);
        float invN = 1.f / 4096.f;
        std::ranges::transform(recovered, recovered.begin(), [invN](C z) { return std::conj(z) * invN; });
        expect(lt(maxError(input, recovered), 1e-4f));
    };

    "power-of-2 sizes"_test = [] {
        for (std::size_t N : {1024UZ, 2048UZ, 4096UZ, 8192UZ}) {
            auto           input = generateTone(N, 3);
            std::vector<C> output(N);
            simdForward(input, output);
            expect(eq(findPeakBin(output), 3UZ)) << "N=" << N;
        }
    };
};

const suite<"FFT SyclFFT:CPU"> syclCpuTests = [] {
    "forward matches known DFT"_test = [] {
        gr::device::DeviceContextCpu ctx;
        gr::device::SyclFFT          fft;
        fft.init(ctx, 4096);
        auto  input = generateTone(4096, 5);
        auto* d     = ctx.allocateShared<GC>(4096);
        ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, 4096);
        fft.forward(ctx, std::span{d, 4096});
        std::vector<C> output(4096);
        ctx.copyDeviceToHost(d, reinterpret_cast<GC*>(output.data()), 4096);
        ctx.deallocate(d);
        expect(eq(findPeakBin(output), 5UZ));
        expect(gt(std::abs(output[5]), 4096.f * 0.9f));
    };

    "forward+inverse round-trip"_test = [] {
        gr::device::DeviceContextCpu ctx;
        gr::device::SyclFFT          fft;
        fft.init(ctx, 4096);
        auto  input = generateTone(4096, 11);
        auto* d     = ctx.allocateShared<GC>(4096);
        ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, 4096);
        fft.forward(ctx, std::span{d, 4096});
        fft.inverse(ctx, std::span{d, 4096});
        std::vector<C> recovered(4096);
        ctx.copyDeviceToHost(d, reinterpret_cast<GC*>(recovered.data()), 4096);
        ctx.deallocate(d);
        expect(lt(maxError(input, recovered), 1e-4f));
    };

    "batched 4x1024"_test = [] {
        gr::device::DeviceContextCpu ctx;
        gr::device::SyclFFT          fft;
        fft.init(ctx, 1024);
        auto* d = ctx.allocateShared<GC>(1024 * 4);
        for (std::size_t b = 0; b < 4; ++b) {
            auto tone = generateTone(1024, b + 1);
            ctx.copyHostToDevice(reinterpret_cast<const GC*>(tone.data()), d + b * 1024, 1024);
        }
        fft.forwardBatch(ctx, std::span{d, 4096}, 1024);
        for (std::size_t b = 0; b < 4; ++b) {
            std::vector<C> output(1024);
            ctx.copyDeviceToHost(d + b * 1024, reinterpret_cast<GC*>(output.data()), 1024);
            expect(eq(findPeakBin(output), b + 1)) << "batch " << b;
        }
        ctx.deallocate(d);
    };
};

const suite<"FFT SyclFFT:CPU vs SimdFFT"> syclCpuCrossTests = [] {
    for (std::size_t N : {256UZ, 1024UZ, 4096UZ}) {
        for (std::size_t bin : {1UZ, 5UZ, N / 4}) {
            boost::ut::test(std::format("N={} bin={}", N, bin)) = [=] {
                auto           input = generateTone(N, bin);
                std::vector<C> expected(N);
                simdForward(input, expected);

                gr::device::DeviceContextCpu ctx;
                gr::device::SyclFFT          fft;
                fft.init(ctx, N);
                auto* d = ctx.allocateShared<GC>(N);
                ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, N);
                fft.forward(ctx, std::span{d, N});
                std::vector<C> actual(N);
                ctx.copyDeviceToHost(d, reinterpret_cast<GC*>(actual.data()), N);
                ctx.deallocate(d);
                expect(lt(maxError(expected, actual), 1e-3f));
            };
        }
    }

    "batched 4x1024"_test = [] {
        constexpr std::size_t N = 1024, B = 4;
        std::vector<C>        input(N * B), expected(N * B);
        for (std::size_t b = 0; b < B; ++b) {
            auto tone = generateTone(N, b + 1);
            std::ranges::copy(tone, input.begin() + static_cast<std::ptrdiff_t>(b * N));
            std::vector<C> out(N);
            simdForward(tone, out);
            std::ranges::copy(out, expected.begin() + static_cast<std::ptrdiff_t>(b * N));
        }
        gr::device::DeviceContextCpu ctx;
        gr::device::SyclFFT          fft;
        fft.init(ctx, N);
        auto* d = ctx.allocateShared<GC>(N * B);
        ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, N * B);
        fft.forwardBatch(ctx, std::span{d, N * B}, N);
        std::vector<C> actual(N * B);
        ctx.copyDeviceToHost(d, reinterpret_cast<GC*>(actual.data()), N * B);
        ctx.deallocate(d);
        expect(lt(maxError(expected, actual), 1e-3f));
    };

    "forward+inverse round-trip"_test = [] {
        auto                         input = generateTone(2048, 11);
        gr::device::DeviceContextCpu ctx;
        gr::device::SyclFFT          fft;
        fft.init(ctx, 2048);
        auto* d = ctx.allocateShared<GC>(2048);
        ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, 2048);
        fft.forward(ctx, std::span{d, 2048});
        fft.inverse(ctx, std::span{d, 2048});
        std::vector<C> recovered(2048);
        ctx.copyDeviceToHost(d, reinterpret_cast<GC*>(recovered.data()), 2048);
        ctx.deallocate(d);
        expect(lt(maxError(input, recovered), 1e-4f));
    };
};

const suite<"FFT Stockham vs SimdFFT"> stockhamTests = [] {
    for (std::size_t N : {8UZ, 16UZ, 64UZ, 256UZ, 1024UZ, 4096UZ}) {
        for (std::size_t bin : {1UZ, 5UZ, N / 4}) {
            boost::ut::test(std::format("N={} bin={}", N, bin)) = [=] {
                auto           input = generateTone(N, bin);
                std::vector<C> expected(N);
                simdForward(input, expected);

                gr::device::DeviceContextCpu ctx;
                gr::device::SyclFFT          fft;
                fft.init(ctx, N);

                // run Stockham on CPU (same algorithm as GPU, just sequential)
                std::vector<GC> gcData(N);
                for (std::size_t i = 0; i < N; ++i) {
                    gcData[i] = {input[i].real(), input[i].imag()};
                }
                fft.forwardStockhamCpu(std::span{gcData.data(), N});

                std::vector<C> actual(N);
                for (std::size_t i = 0; i < N; ++i) {
                    actual[i] = {gcData[i].re, gcData[i].im};
                }
                expect(lt(maxError(expected, actual), 1e-3f)) << std::format("N={} bin={}", N, bin);
            };
        }
    }

    "batched via loop"_test = [] {
        constexpr std::size_t        N = 512, B = 4;
        gr::device::DeviceContextCpu ctx;
        gr::device::SyclFFT          fft;
        fft.init(ctx, N);

        for (std::size_t b = 0; b < B; ++b) {
            auto           input = generateTone(N, b + 1);
            std::vector<C> expected(N);
            simdForward(input, expected);

            std::vector<GC> gcData(N);
            for (std::size_t i = 0; i < N; ++i) {
                gcData[i] = {input[i].real(), input[i].imag()};
            }
            fft.forwardStockhamCpu(std::span{gcData.data(), N});

            std::vector<C> actual(N);
            for (std::size_t i = 0; i < N; ++i) {
                actual[i] = {gcData[i].re, gcData[i].im};
            }
            expect(lt(maxError(expected, actual), 1e-3f)) << "batch " << b;
        }
    };
};

#if GR_DEVICE_HAS_GL_COMPUTE
const suite<"FFT GlslFFT"> glslTests = [] {
    auto* ctx = glCtx();
    if (!ctx) {
        return;
    }

    "forward matches known DFT"_test = [ctx] {
        gr::device::GlslFFT fft;
        fft.init(*ctx, 4096);
        auto  input = generateTone(4096, 5);
        auto* d     = ctx->allocateDevice<GC>(4096);
        ctx->copyHostToDevice(reinterpret_cast<const GC*>(input.data()), static_cast<GC*>(d), 4096);
        fft.forward(*ctx, d, 4096);
        std::vector<C> output(4096);
        ctx->copyDeviceToHost(static_cast<GC*>(d), reinterpret_cast<GC*>(output.data()), 4096);
        ctx->deallocateRaw(d);
        expect(eq(findPeakBin(output), 5UZ));
        expect(gt(std::abs(output[5]), 4096.f * 0.9f));
    };

    "batched 4x1024"_test = [ctx] {
        constexpr std::size_t N = 1024, B = 4;
        gr::device::GlslFFT   fft;
        fft.init(*ctx, N);
        std::vector<C> input(N * B);
        for (std::size_t b = 0; b < B; ++b) {
            auto tone = generateTone(N, b + 1);
            std::ranges::copy(tone, input.begin() + static_cast<std::ptrdiff_t>(b * N));
        }
        auto* d = ctx->allocateDevice<GC>(N * B);
        ctx->copyHostToDevice(reinterpret_cast<const GC*>(input.data()), static_cast<GC*>(d), N * B);
        fft.forward(*ctx, d, N, B);
        std::vector<C> output(N * B);
        ctx->copyDeviceToHost(static_cast<GC*>(d), reinterpret_cast<GC*>(output.data()), N * B);
        ctx->deallocateRaw(d);
        for (std::size_t b = 0; b < B; ++b) {
            expect(eq(findPeakBin(std::span<C>(output.data() + b * N, N)), b + 1)) << "batch " << b;
        }
    };
};
#endif

const suite<"FFT2 graph integration"> graphTests = [] {
    "Source -> FFT2 -> Sink (CPU)"_test = [] {
        constexpr gr::Size_t N = 4096;
        gr::Graph            flow;
        auto&                src  = flow.emplaceBlock<gr::testing::CountingSource<C>>({{"n_samples_max", N}});
        auto&                fft  = flow.emplaceBlock<gr::blocks::fourier::FFT2<float>>({{"fft_size", N}});
        auto&                sink = flow.emplaceBlock<gr::testing::CountingSink<C>>({{"n_samples_max", N}});
        expect(flow.connect<"out", "in">(src, fft).has_value());
        expect(flow.connect<"out", "in">(fft, sink).has_value());
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());
        expect(eq(sink.count.value, N));
    };

    "Source -> H2D -> FFT2 -> D2H -> Sink (explicit transfer)"_test = [] {
        constexpr gr::Size_t N = 4096;
        gr::Graph            flow;
        auto&                src  = flow.emplaceBlock<gr::testing::CountingSource<C>>({{"n_samples_max", N}});
        auto&                h2d  = flow.emplaceBlock<gr::basic::HostToDevice<C>>({{"chunk_size", N}});
        auto&                fft  = flow.emplaceBlock<gr::blocks::fourier::FFT2<float>>({{"fft_size", N}});
        auto&                d2h  = flow.emplaceBlock<gr::basic::DeviceToHost<C>>();
        auto&                sink = flow.emplaceBlock<gr::testing::CountingSink<C>>({{"n_samples_max", N}});
        expect(flow.connect<"out", "in">(src, h2d).has_value());
        expect(flow.connect<"out", "in">(h2d, fft).has_value());
        expect(flow.connect<"out", "in">(fft, d2h).has_value());
        expect(flow.connect<"out", "in">(d2h, sink).has_value());
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());
        expect(eq(sink.count.value, N));
    };
};

int main() { /* not needed for UT */ }
