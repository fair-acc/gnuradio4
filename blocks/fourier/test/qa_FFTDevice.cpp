#ifdef GR_FFT_DEVICE_TEST

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numbers>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include <gnuradio-4.0/algorithm/fourier/SyclFFT.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>

namespace {

using C  = std::complex<float>;
using GC = gr::complex<float>;

[[gnu::constructor]] void configureAdaptiveCppStorage() {
    setenv("ACPP_APPDB_DIR", "/tmp/gnuradio4-acpp-appdb", 0);
    setenv("XDG_DATA_HOME", "/tmp/gnuradio4-acpp-data", 0);
}

std::vector<C> generateTone(std::size_t n, std::size_t bin) {
    std::vector<C> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float phase = 2.f * std::numbers::pi_v<float> * static_cast<float>(bin) * static_cast<float>(i) / static_cast<float>(n);
        data[i]           = {std::cos(phase), std::sin(phase)};
    }
    return data;
}

void simdForward(const std::vector<C>& input, std::vector<C>& output) {
    gr::algorithm::FFT<C, C> fft;
    fft.compute(input, output);
}

float maxError(std::span<const C> a, std::span<const C> b) {
    float err = 0.f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(a[i] - b[i]));
    }
    return err;
}

bool runOnDevice(const sycl::device& dev) {
    constexpr std::size_t n = 256UZ;
    constexpr std::size_t b = 2UZ;

    const std::string name = dev.get_info<sycl::info::device::name>();
    std::printf("FFT SYCL device: %s (cpu=%d gpu=%d)\n", name.c_str(), static_cast<int>(dev.is_cpu()), static_cast<int>(dev.is_gpu()));

    sycl::queue                   queue{dev};
    gr::device::DeviceContextSycl ctx{queue};
    gr::device::SyclFFT           fft;
    fft.init(ctx, n);

    std::vector<C> expected(n * b);
    std::vector<C> spectrum(n * b);
    for (std::size_t batch = 0; batch < b; ++batch) {
        auto tone = generateTone(n, batch + 3UZ);
        std::ranges::copy(tone, expected.begin() + static_cast<std::ptrdiff_t>(batch * n));

        std::vector<C> oneSpectrum(n);
        simdForward(tone, oneSpectrum);
        std::ranges::copy(oneSpectrum, spectrum.begin() + static_cast<std::ptrdiff_t>(batch * n));
    }

    gr::device::DeviceBuffer deviceData = ctx.allocateDevice<GC>(n * b);
    if (!deviceData) {
        std::printf("FAIL: device allocation returned null\n");
        return false;
    }

    ctx.copyHostToDevice(reinterpret_cast<const GC*>(spectrum.data()), deviceData, n * b);
    fft.inverseBatch(ctx, std::span<GC>{deviceData.devicePointer<GC>(), n * b}, n);

    std::vector<C> actual(n * b);
    ctx.copyDeviceToHost(deviceData, reinterpret_cast<GC*>(actual.data()), n * b);
    ctx.deallocate(deviceData);

    const float err = maxError(expected, actual);
    std::printf("FFT SYCL inverseBatch error: %.8g\n", static_cast<double>(err));
    return err < 1e-3f;
}

} // namespace

int main() {
    try {
        bool sawDevice = false;
        bool ok        = true;
        for (const sycl::device& dev : sycl::device::get_devices()) {
            if (!dev.is_cpu() && !dev.is_gpu()) {
                continue;
            }
            sawDevice = true;
            ok        = runOnDevice(dev) && ok;
        }
        if (!sawDevice) {
            std::printf("SKIP: no SYCL CPU/GPU devices available\n");
            return 77;
        }
        return ok ? 0 : 1;
    } catch (const sycl::exception& e) {
        std::printf("FAIL: SYCL exception: %s\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: exception: %s\n", e.what());
        return 1;
    }
}

#else

#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/algorithm/fourier/SyclFFT.hpp>
#include <gnuradio-4.0/basic/TransferBlocks.hpp>
#include <gnuradio-4.0/fourier/fft.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using namespace boost::ut;
using C = std::complex<float>;

static_assert(requires(gr::device::SyclQueue& queue, gr::blocks::fft::FFT<float>& fft, gr::traits::block::detail::DummyInputSpan<const C>& input, gr::traits::block::detail::DummyOutputSpan<C>& output) {
    { fft.processBulk_sycl(queue, input, output) } -> std::same_as<gr::work::Status>;
});

// spectrum mode device dispatch exists only for T == std::complex<float> (gr::device::SyclFFT is a float-only,
// complex-only tier); real input and double precision have no device transform to reuse and stay host-only
static_assert(requires(gr::device::SyclQueue& queue, gr::blocks::fft::FFT<C, gr::DataSet<float>>& fft, gr::traits::block::detail::DummyInputSpan<const C>& input, gr::traits::block::detail::DummyOutputSpan<gr::DataSet<float>>& output) {
    { fft.processBulk_sycl(queue, input, output) } -> std::same_as<gr::work::Status>;
});

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

} // namespace

const auto cpuSimdTests = [] {
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

const auto syclCpuTests = [] {
    "forward matches known DFT"_test = [] {
        gr::device::DeviceContextCpu ctx;
        gr::device::SyclFFT          fft;
        fft.init(ctx, 4096);
        auto input = generateTone(4096, 5);
        auto d     = ctx.allocateShared<GC>(4096);
        ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, 4096);
        fft.forward(ctx, std::span{d.devicePointer<GC>(), 4096});
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
        auto input = generateTone(4096, 11);
        auto d     = ctx.allocateShared<GC>(4096);
        ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, 4096);
        fft.forward(ctx, std::span{d.devicePointer<GC>(), 4096});
        fft.inverse(ctx, std::span{d.devicePointer<GC>(), 4096});
        std::vector<C> recovered(4096);
        ctx.copyDeviceToHost(d, reinterpret_cast<GC*>(recovered.data()), 4096);
        ctx.deallocate(d);
        expect(lt(maxError(input, recovered), 1e-4f));
    };

    "batched 4x1024"_test = [] {
        gr::device::DeviceContextCpu ctx;
        gr::device::SyclFFT          fft;
        fft.init(ctx, 1024);

        std::vector<C> input(4096);
        for (std::size_t b = 0; b < 4; ++b) {
            auto tone = generateTone(1024, b + 1);
            std::ranges::copy(tone, input.begin() + static_cast<std::ptrdiff_t>(b * 1024));
        }

        auto d = ctx.allocateShared<GC>(4096);
        ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, 4096);
        fft.forwardBatch(ctx, std::span{d.devicePointer<GC>(), 4096}, 1024);

        std::vector<C> output(4096);
        ctx.copyDeviceToHost(d, reinterpret_cast<GC*>(output.data()), 4096);
        ctx.deallocate(d);

        for (std::size_t b = 0; b < 4; ++b) {
            expect(eq(findPeakBin(std::span<const C>(output.data() + b * 1024, 1024)), b + 1)) << "batch " << b;
        }
    };

    "inverseBatch handles batches independently"_test = [] {
        constexpr std::size_t N = 256UZ;
        constexpr std::size_t B = 2UZ;

        std::vector<C> input(N * B), expected(N * B);
        for (std::size_t b = 0; b < B; ++b) {
            auto tone = generateTone(N, b + 3UZ);
            std::ranges::copy(tone, expected.begin() + static_cast<std::ptrdiff_t>(b * N));

            std::vector<C> spectrum(N);
            simdForward(tone, spectrum);
            std::ranges::copy(spectrum, input.begin() + static_cast<std::ptrdiff_t>(b * N));
        }

        gr::device::DeviceContextCpu ctx;
        gr::device::SyclFFT          fft;
        fft.init(ctx, N);
        fft.inverseBatch(ctx, std::span<GC>{reinterpret_cast<GC*>(input.data()), input.size()}, N);

        expect(lt(maxError(expected, input), 1e-4f));
    };
};

const auto syclCpuCrossTests = [] {
    for (std::size_t N : {256UZ, 1024UZ, 4096UZ}) {
        for (std::size_t bin : {1UZ, 5UZ, N / 4}) {
            boost::ut::test(std::format("N={} bin={}", N, bin)) = [=] {
                auto           input = generateTone(N, bin);
                std::vector<C> expected(N);
                simdForward(input, expected);

                gr::device::DeviceContextCpu ctx;
                gr::device::SyclFFT          fft;
                fft.init(ctx, N);
                auto d = ctx.allocateShared<GC>(N);
                ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, N);
                fft.forward(ctx, std::span{d.devicePointer<GC>(), N});
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
        auto d = ctx.allocateShared<GC>(N * B);
        ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, N * B);
        fft.forwardBatch(ctx, std::span{d.devicePointer<GC>(), N * B}, N);
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
        auto d = ctx.allocateShared<GC>(2048);
        ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), d, 2048);
        fft.forward(ctx, std::span{d.devicePointer<GC>(), 2048});
        fft.inverse(ctx, std::span{d.devicePointer<GC>(), 2048});
        std::vector<C> recovered(2048);
        ctx.copyDeviceToHost(d, reinterpret_cast<GC*>(recovered.data()), 2048);
        ctx.deallocate(d);
        expect(lt(maxError(input, recovered), 1e-4f));
    };
};

const auto stockhamTests = [] {
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

const auto graphTests = [] {
    "Source -> FFT -> Sink (CPU)"_test = [] {
        constexpr gr::Size_t N = 4096;
        gr::Graph            flow;
        auto&                src  = flow.emplaceBlock<gr::testing::CountingSource<C>>({{"n_samples_max", N}});
        auto&                fft  = flow.emplaceBlock<gr::blocks::fft::FFT<float>>({{"fft_size", N}});
        auto&                sink = flow.emplaceBlock<gr::testing::CountingSink<C>>({{"n_samples_max", N}});
        expect(flow.connect<"out", "in">(src, fft).has_value());
        expect(flow.connect<"out", "in">(fft, sink).has_value());
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());
        expect(eq(sink.count.value, N));
    };

    "Source -> H2D -> FFT -> D2H -> Sink (explicit transfer)"_test = [] {
        constexpr gr::Size_t N = 4096;
        gr::Graph            flow;
        auto&                src  = flow.emplaceBlock<gr::testing::CountingSource<C>>({{"n_samples_max", N}});
        auto&                h2d  = flow.emplaceBlock<gr::basic::HostToDevice<C>>({{"chunk_size", N}});
        auto&                fft  = flow.emplaceBlock<gr::blocks::fft::FFT<float>>({{"fft_size", N}});
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

namespace {

template<typename T>
struct CollectorSink : gr::Block<CollectorSink<T>> {
    gr::PortIn<T>  in;
    std::vector<T> received;

    GR_MAKE_REFLECTABLE(CollectorSink, in);

    void processOne(const T& value) { received.push_back(value); }
};

std::vector<float> generateRealSine(std::size_t N, std::size_t bin) {
    std::vector<float> signal(N);
    for (std::size_t i = 0; i < N; ++i) {
        signal[i] = std::sin(2.f * std::numbers::pi_v<float> * static_cast<float>(bin) * static_cast<float>(i) / static_cast<float>(N));
    }
    return signal;
}

std::size_t findPeakBinF(std::span<const float> magnitude) {
    std::size_t peak   = 0;
    float       maxMag = 0.f;
    for (std::size_t i = 0; i < magnitude.size(); ++i) {
        if (magnitude[i] > maxMag) {
            maxMag = magnitude[i];
            peak   = i;
        }
    }
    return peak;
}

template<typename TInput>
gr::DataSet<float> runSpectrumGraph(std::vector<TInput> signal, const gr::property_map& extraSettings = {}) {
    gr::Graph        flow;
    gr::property_map fftSettings{{"fft_size", static_cast<gr::Size_t>(signal.size())}};
    for (const auto& [key, value] : extraSettings) {
        fftSettings[key] = value;
    }
    auto& src  = flow.emplaceBlock<gr::testing::TagSource<TInput>>({{"values", signal}, {"n_samples_max", static_cast<gr::Size_t>(signal.size())}});
    auto& fft  = flow.emplaceBlock<gr::blocks::fft::FFT<TInput, gr::DataSet<float>>>(fftSettings);
    auto& sink = flow.emplaceBlock<CollectorSink<gr::DataSet<float>>>();
    expect(flow.connect<"out", "in">(src, fft).has_value());
    expect(flow.connect<"out", "in">(fft, sink).has_value());
    gr::scheduler::Simple<> sched;
    expect(sched.exchange(std::move(flow)).has_value());
    expect(sched.runAndWait().has_value());
    expect(eq(sink.received.size(), 1UZ));
    return sink.received.empty() ? gr::DataSet<float>{} : sink.received.front();
}

// same block as runSpectrumGraph, invoked directly (no scheduler, no Resampling/tag machinery) via
// manually wired ports: a graph-driven vs. direct-processBulk consistency check on the same FFT block
template<typename TInput>
gr::DataSet<float> runReferenceSpectrum(const std::vector<TInput>& signal, const gr::property_map& extraSettings = {}) {
    gr::property_map settings{{"fft_size", static_cast<gr::Size_t>(signal.size())}};
    for (const auto& [key, value] : extraSettings) {
        settings[key] = value;
    }
    gr::blocks::fft::FFT<TInput, gr::DataSet<float>> reference(settings);
    reference.init(reference.progress);

    gr::PortOut<TInput> srcOut;
    gr::PortIn<TInput>  fftIn;
    expect(srcOut.connect(fftIn).has_value());
    {
        auto wspan = srcOut.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(signal.size());
        expect(eq(wspan.size(), signal.size()));
        std::ranges::copy(signal, wspan.begin());
        wspan.publish(signal.size());
    }
    auto inSpan = fftIn.template get<gr::SpanReleasePolicy::ProcessAll>(signal.size());

    gr::PortOut<gr::DataSet<float>> fftOutPort;
    gr::PortIn<gr::DataSet<float>>  sinkIn;
    expect(fftOutPort.connect(sinkIn).has_value());
    {
        auto outSpan = fftOutPort.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
        expect(gr::work::Status::OK == reference.processBulk(inSpan, outSpan));
    } // scope so the WriterSpan's destructor publishes before the read below

    auto readBack = sinkIn.template get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
    expect(eq(readBack.size(), 1UZ));
    return readBack[0];
}

void expectSpectraMatch(const gr::DataSet<float>& actual, const gr::DataSet<float>& expected) {
    expect(eq(actual.axisValues(0UZ).size(), expected.axisValues(0UZ).size()));
    for (std::size_t i = 0; i < std::min(actual.axisValues(0UZ).size(), expected.axisValues(0UZ).size()); ++i) {
        expect(approx(actual.axisValues(0UZ)[i], expected.axisValues(0UZ)[i], 1e-3f)) << std::format("axis[{}]", i);
    }
    expect(eq(actual.signalValues(0UZ).size(), expected.signalValues(0UZ).size()));
    const auto& magnitudes   = expected.signalValues(0UZ);
    const float phaseIsCarriedAbove = 1e-2f * (magnitudes.empty() ? 0.f : std::ranges::max(magnitudes));
    for (std::size_t i = 0; i < actual.signalValues(0UZ).size(); ++i) {
        expect(approx(actual.signalValues(0UZ)[i], expected.signalValues(0UZ)[i], 1e-3f)) << std::format("magnitude[{}]", i);
        if (magnitudes[i] > phaseIsCarriedAbove) { // atan2 of two vanishing components follows the rounding, not the signal
            expect(approx(actual.signalValues(1UZ)[i], expected.signalValues(1UZ)[i], 1e-3f)) << std::format("phase[{}]", i);
        }
        expect(approx(actual.signalValues(2UZ)[i], expected.signalValues(2UZ)[i], 1e-3f)) << std::format("re[{}]", i);
        expect(approx(actual.signalValues(3UZ)[i], expected.signalValues(3UZ)[i], 1e-3f)) << std::format("im[{}]", i);
    }
}

// device kernels do not compute atan2/hypot bit-identically to host libm, so device-vs-host comparisons
// need a relative tolerance -- 1e-5 is what float precision supports; absFloor covers exact-zero bins
bool approxRel(float actual, float expected, float relTolerance = 1e-5f, float absFloor = 1e-4f) { return std::abs(actual - expected) <= std::max(relTolerance * std::abs(expected), absFloor); }

} // namespace

const auto spectrumTests = [] {
    "processBulk_sycl executes (host-fallback queue) and matches the host path"_test = [] {
        // real ports, no scheduler: exercises the actual processBulk_sycl body (memcpy, parallelFor, forwardBatch,
        // copyDeviceToHost, unwrap) instead of only checking that the call is declared well-formed
        constexpr std::size_t N      = 64;
        constexpr std::size_t bin    = 6;
        auto                  signal = generateTone(N, bin);

        gr::blocks::fft::FFT<C, gr::DataSet<float>> fft({{"fft_size", static_cast<gr::Size_t>(N)}, {"sample_rate", 1.f}});
        fft.init(fft.progress);

        gr::PortOut<C> srcOut;
        gr::PortIn<C>  fftIn;
        expect(srcOut.connect(fftIn).has_value());
        {
            auto wspan = srcOut.tryReserve<gr::SpanReleasePolicy::ProcessAll>(N);
            expect(eq(wspan.size(), N));
            std::ranges::copy(signal, wspan.begin());
            wspan.publish(N);
        }
        auto inSpan = fftIn.get<gr::SpanReleasePolicy::ProcessAll>(N);

        gr::PortOut<gr::DataSet<float>> fftOutPort;
        gr::PortIn<gr::DataSet<float>>  sinkIn;
        expect(fftOutPort.connect(sinkIn).has_value());
        {
            auto outSpan = fftOutPort.tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            gr::device::SyclQueue queue{};
            expect(gr::work::Status::OK == fft.processBulk_sycl(queue, inSpan, outSpan));
        } // scope so the WriterSpan's destructor publishes before the read below

        auto readBack = sinkIn.get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
        expect(eq(readBack.size(), 1UZ));

        gr::DataSet<float> expected = runReferenceSpectrum<C>(signal, {{"sample_rate", 1.f}});
        expectSpectraMatch(readBack[0], expected);
    };

    "real input: graph-driven matches direct processBulk"_test = [] {
        constexpr std::size_t N      = 256;
        constexpr std::size_t bin    = 17;
        auto                  signal = generateRealSine(N, bin);

        gr::DataSet<float> actual   = runSpectrumGraph<float>(signal, {{"sample_rate", 1.f}});
        gr::DataSet<float> expected = runReferenceSpectrum<float>(signal, {{"sample_rate", 1.f}});

        expectSpectraMatch(actual, expected);
        expect(eq(findPeakBinF(actual.signalValues(0UZ)), bin));
    };

    "complex input: graph-driven matches direct processBulk"_test = [] {
        constexpr std::size_t N      = 256;
        constexpr std::size_t bin    = 12;
        auto                  signal = generateTone(N, bin);

        gr::DataSet<float> actual   = runSpectrumGraph<C>(signal, {{"sample_rate", 1.f}});
        gr::DataSet<float> expected = runReferenceSpectrum<C>(signal, {{"sample_rate", 1.f}});

        expectSpectraMatch(actual, expected);
    };

    "output_in_db, output_in_deg and unwrap_phase: graph-driven matches direct processBulk"_test = [] {
        constexpr std::size_t N      = 256;
        constexpr std::size_t bin    = 33; // away from DC/Nyquist so unwrap and deg conversion both exercise real bins
        auto                  signal = generateRealSine(N, bin);

        gr::DataSet<float> actual   = runSpectrumGraph<float>(signal, {{"sample_rate", 1.f}, {"output_in_db", true}, {"output_in_deg", true}, {"unwrap_phase", true}});
        gr::DataSet<float> expected = runReferenceSpectrum<float>(signal, {{"sample_rate", 1.f}, {"output_in_db", true}, {"output_in_deg", true}, {"unwrap_phase", true}});

        expectSpectraMatch(actual, expected);
    };

    "output_in_db scales the magnitude"_test = [] {
        constexpr std::size_t N      = 256;
        constexpr std::size_t bin    = 9;
        auto                  signal = generateRealSine(N, bin);

        gr::DataSet<float> linear = runSpectrumGraph<float>(signal, {{"sample_rate", 1.f}, {"output_in_db", false}});
        gr::DataSet<float> db     = runSpectrumGraph<float>(signal, {{"sample_rate", 1.f}, {"output_in_db", true}});

        const auto peak = findPeakBinF(linear.signalValues(0UZ));
        expect(gt(linear.signalValues(0UZ)[peak], 0.1f)); // Hann window reduces coherent gain to ~0.5 of the raw amplitude
        expect(approx(db.signalValues(0UZ)[peak], 20.f * std::log10(linear.signalValues(0UZ)[peak]), 1e-2f));
    };

    "window coefficients regenerate for the configured size and shape"_test = [] {
        gr::blocks::fft::FFT<float, gr::DataSet<float>> hann({{"fft_size", 64U}, {"window", std::string("Hann")}});
        hann.init(hann.progress);
        gr::blocks::fft::FFT<float, gr::DataSet<float>> rect({{"fft_size", 64U}, {"window", std::string("Rectangular")}});
        rect.init(rect.progress);

        expect(eq(hann.window_coefficients.size(), 64UZ));
        expect(eq(rect.window_coefficients.size(), 64UZ));
        expect(approx(rect.window_coefficients[10], 1.f, 1e-6f));
        expect(neq(hann.window_coefficients[10], rect.window_coefficients[10]));
    };

    "batches multiple DataSets per graph run"_test = [] {
        constexpr std::size_t N       = 128;
        constexpr std::size_t nChunks = 3;
        auto                  signal  = generateRealSine(N, 5);
        std::vector<float>    repeated;
        for (std::size_t c = 0; c < nChunks; ++c) {
            repeated.insert(repeated.end(), signal.begin(), signal.end());
        }

        gr::Graph flow;
        auto&     src  = flow.emplaceBlock<gr::testing::TagSource<float>>({{"values", repeated}, {"n_samples_max", static_cast<gr::Size_t>(repeated.size())}});
        auto&     fft  = flow.emplaceBlock<gr::blocks::fft::FFT<float, gr::DataSet<float>>>({{"fft_size", static_cast<gr::Size_t>(N)}});
        auto&     sink = flow.emplaceBlock<CollectorSink<gr::DataSet<float>>>();
        expect(flow.connect<"out", "in">(src, fft).has_value());
        expect(flow.connect<"out", "in">(fft, sink).has_value());
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());
        expect(eq(sink.received.size(), nChunks));
        for (const auto& ds : sink.received) {
            expect(eq(findPeakBinF(ds.signalValues(0UZ)), 5UZ));
        }
    };
};

const auto deviceUnwrapOrder = [] {
    "unwrap-then-shift order matches the host reference (complex input, delayed impulse)"_test = [] {
        // a delayed impulse gives an exactly linear (mod 2*pi) phase ramp across bins, guaranteeing several
        // genuine unwrap corrections regardless of window; n0 is detuned off a clean N/4 fraction
        constexpr std::size_t nFft = 64UZ;
        constexpr std::size_t n0   = 17UZ;
        std::vector<C>        signal(nFft, C{0.f, 0.f});
        signal[n0] = C{1.f, 0.f};

        gr::blocks::fft::FFT<C, gr::DataSet<float>> fft({{"fft_size", static_cast<gr::Size_t>(nFft)}, {"sample_rate", 1.f}, {"unwrap_phase", true}, {"window", std::string("Rectangular")}});
        fft.init(fft.progress);

        gr::PortOut<C> srcOut;
        gr::PortIn<C>  fftIn;
        expect(srcOut.connect(fftIn).has_value());
        {
            auto wspan = srcOut.tryReserve<gr::SpanReleasePolicy::ProcessAll>(nFft);
            expect(eq(wspan.size(), nFft));
            std::ranges::copy(signal, wspan.begin());
            wspan.publish(nFft);
        }
        auto inSpan = fftIn.get<gr::SpanReleasePolicy::ProcessAll>(nFft);

        gr::PortOut<gr::DataSet<float>> fftOutPort;
        gr::PortIn<gr::DataSet<float>>  sinkIn;
        expect(fftOutPort.connect(sinkIn).has_value());
        {
            auto outSpan = fftOutPort.tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            gr::device::SyclQueue queue{};
            expect(gr::work::Status::OK == fft.processBulk_sycl(queue, inSpan, outSpan));
        } // scope so the WriterSpan's destructor publishes before the read below

        auto readBack = sinkIn.get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
        expect(eq(readBack.size(), 1UZ));

        // ground truth: recompute magnitude and phase from the exact same natural-order complex bins the device
        // path produced (fft._outData), through the canonical host algorithm (unwrap in natural order, then
        // fftshift) -- this isolates the shift/unwrap ordering from host-vs-device transform differences
        const auto groundTruthMag   = gr::algorithm::fft::computeMagnitudeSpectrum(fft._outData, gr::algorithm::fft::ConfigMagnitude{.computeHalfSpectrum = false, .outputInDb = false, .shiftSpectrum = true});
        const auto groundTruthPhase = gr::algorithm::fft::computePhaseSpectrum(fft._outData, gr::algorithm::fft::ConfigPhase{.computeHalfSpectrum = false, .outputInDeg = false, .unwrapPhase = true, .shiftSpectrum = true});

        float maxAbsPhase = 0.f;
        for (const float p : groundTruthPhase) {
            maxAbsPhase = std::max(maxAbsPhase, std::abs(p));
        }
        expect(gt(maxAbsPhase, 2.f * std::numbers::pi_v<float>)) << "stimulus did not genuinely wrap phase -- test is not exercising unwrap";

        const auto deviceMag   = readBack[0].signalValues(0UZ);
        const auto devicePhase = readBack[0].signalValues(1UZ);
        expect(eq(deviceMag.size(), groundTruthMag.size()));
        expect(eq(devicePhase.size(), groundTruthPhase.size()));
        for (std::size_t i = 0; i < std::min(deviceMag.size(), groundTruthMag.size()); ++i) {
            expect(approxRel(deviceMag[i], groundTruthMag[i])) << std::format("bin {}: magnitude device={} host-order-groundtruth={}", i, deviceMag[i], groundTruthMag[i]);
        }
        for (std::size_t i = 0; i < std::min(devicePhase.size(), groundTruthPhase.size()); ++i) {
            // a wrong 2*pi multiple from unwrapping across the shift seam is ~6.28 -- far outside this tolerance
            expect(approxRel(devicePhase[i], groundTruthPhase[i])) << std::format("bin {}: phase device={} host-order-groundtruth={}", i, devicePhase[i], groundTruthPhase[i]);
        }
    };
};

int main() { // the suites run here, not at static destruction: a kernel launched from `~runner` outlives AdaptiveCpp's kernel registry
    "FFT CPU SimdFFT"_test                            = cpuSimdTests;
    "FFT SyclFFT:CPU"_test                            = syclCpuTests;
    "FFT SyclFFT:CPU vs SimdFFT"_test                 = syclCpuCrossTests;
    "FFT Stockham vs SimdFFT"_test                    = stockhamTests;
    "FFT graph integration"_test                      = graphTests;
    "FFT spectrum mode"_test                          = spectrumTests;
    "FFT device dispatch: phase-unwrap ordering"_test = deviceUnwrapOrder;
    return 0;
}

#endif
