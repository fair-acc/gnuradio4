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

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/DeviceFFT.hpp>

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
    gr::device::DeviceFFT         fft;
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

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <format>
#include <numbers>
#include <numeric>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/TransferBlocks.hpp>
#include <gnuradio-4.0/device/DeviceFFT.hpp>
#include <gnuradio-4.0/fourier/fft.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using namespace boost::ut;
using C = std::complex<float>;

static_assert(requires(gr::device::DeviceContext& ctx, gr::blocks::fft::FFT<float>& fft, gr::traits::block::detail::DummyInputSpan<const C>& input, gr::traits::block::detail::DummyOutputSpan<C>& output) {
    { fft.processBulk(ctx, input, output) } -> std::same_as<gr::work::Status>;
});

static_assert(requires(gr::device::DeviceContext& ctx, gr::blocks::fft::FFT<C, gr::DataSet<float>>& fft, gr::traits::block::detail::DummyInputSpan<const C>& input, gr::traits::block::detail::DummyOutputSpan<gr::DataSet<float>>& output) {
    { fft.processBulk(ctx, input, output) } -> std::same_as<gr::work::Status>;
});
static_assert(requires(gr::device::DeviceContext& ctx, gr::blocks::fft::FFT<float, gr::DataSet<float>>& fft, gr::traits::block::detail::DummyInputSpan<const float>& input, gr::traits::block::detail::DummyOutputSpan<gr::DataSet<float>>& output) {
    { fft.processBulk(ctx, input, output) } -> std::same_as<gr::work::Status>;
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

const auto cpuContextTests = [] {
    "forward matches known DFT"_test = [] {
        gr::device::DeviceContextCpu ctx;
        gr::device::DeviceFFT        fft;
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
        gr::device::DeviceFFT        fft;
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
        gr::device::DeviceFFT        fft;
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
        gr::device::DeviceFFT        fft;
        fft.init(ctx, N);
        fft.inverseBatch(ctx, std::span<GC>{reinterpret_cast<GC*>(input.data()), input.size()}, N);

        expect(lt(maxError(expected, input), 1e-4f));
    };
};

const auto cpuContextCrossTests = [] {
    for (std::size_t N : {256UZ, 1024UZ, 4096UZ}) {
        for (std::size_t bin : {1UZ, 5UZ, N / 4}) {
            boost::ut::test(std::format("N={} bin={}", N, bin)) = [=] {
                auto           input = generateTone(N, bin);
                std::vector<C> expected(N);
                simdForward(input, expected);

                gr::device::DeviceContextCpu ctx;
                gr::device::DeviceFFT        fft;
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
        gr::device::DeviceFFT        fft;
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
        gr::device::DeviceFFT        fft;
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
                gr::device::DeviceFFT        fft;
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
        gr::device::DeviceFFT        fft;
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

/// as every registry context: `processBulk(ctx, ...)` orders its copies and kernels by submission
[[nodiscard]] gr::device::SyclQueue inOrderQueue() {
#if GR_DEVICE_HAS_SYCL
    return gr::device::SyclQueue{sycl::property::queue::in_order{}};
#else
    return {};
#endif
}

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
    const auto& magnitudes          = expected.signalValues(0UZ);
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
    "processBulk(ctx, ...) executes (host-fallback queue) and matches the host path"_test = [] {
        // real ports, no scheduler: exercises the actual processBulk(ctx, ...) body (memcpy, parallelFor, forwardBatch,
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
            auto                  outSpan = fftOutPort.tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            gr::device::SyclQueue queue   = inOrderQueue();
            expect(gr::work::Status::OK == fft.processBulk(gr::device::syclContextFor(queue), inSpan, outSpan));
        } // scope so the WriterSpan's destructor publishes before the read below

        auto readBack = sinkIn.get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
        expect(eq(readBack.size(), 1UZ));

        gr::DataSet<float> expected = runReferenceSpectrum<C>(signal, {{"sample_rate", 1.f}});
        expectSpectraMatch(readBack[0], expected);
    };

    "real input: processBulk(ctx, ...) folds the half spectrum as the host path does"_test = [] {
        constexpr std::size_t N      = 128;
        constexpr std::size_t kBin   = 9;
        const auto            signal = generateRealSine(N, kBin);

        gr::blocks::fft::FFT<float, gr::DataSet<float>> fft({{"fft_size", static_cast<gr::Size_t>(N)}, {"sample_rate", 1.f}});
        fft.init(fft.progress);

        gr::PortOut<float> srcOut;
        gr::PortIn<float>  fftIn;
        expect(srcOut.connect(fftIn).has_value());
        {
            auto wspan = srcOut.tryReserve<gr::SpanReleasePolicy::ProcessAll>(N);
            std::ranges::copy(signal, wspan.begin());
            wspan.publish(N);
        }
        auto inSpan = fftIn.get<gr::SpanReleasePolicy::ProcessAll>(N);

        gr::PortOut<gr::DataSet<float>> fftOutPort;
        gr::PortIn<gr::DataSet<float>>  sinkIn;
        expect(fftOutPort.connect(sinkIn).has_value());
        {
            auto                  outSpan = fftOutPort.tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            gr::device::SyclQueue queue   = inOrderQueue();
            expect(gr::work::Status::OK == fft.processBulk(gr::device::syclContextFor(queue), inSpan, outSpan));
        }

        auto readBack = sinkIn.get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
        expect(fatal(eq(readBack.size(), 1UZ)));
        expect(eq(readBack[0].signalValues(0UZ).size(), N / 2UZ + 1UZ)) << "DC to Nyquist";
        expectSpectraMatch(readBack[0], runReferenceSpectrum<float>(signal, {{"sample_rate", 1.f}}));
        expect(eq(findPeakBinF(readBack[0].signalValues(0UZ)), kBin));
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

        expect(eq(hann._windowCoefficients.size(), 64UZ));
        expect(eq(rect._windowCoefficients.size(), 64UZ));
        expect(approx(rect._windowCoefficients[10], 1.f, 1e-6f));
        expect(neq(hann._windowCoefficients[10], rect._windowCoefficients[10]));
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
            auto                  outSpan = fftOutPort.tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            gr::device::SyclQueue queue   = inOrderQueue();
            expect(gr::work::Status::OK == fft.processBulk(gr::device::syclContextFor(queue), inSpan, outSpan));
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

template<typename TInput>
void expectSpectraOnDomain(auto& fixture, std::string_view domain, const std::vector<TInput>& frame, std::size_t bin) {
    constexpr std::size_t kFrames = 3UZ;
    const std::size_t     N       = frame.size();
    std::vector<TInput>   signal;
    for (std::size_t f = 0UZ; f < kFrames; ++f) {
        signal.insert(signal.end(), frame.begin(), frame.end());
    }
    auto& source = fixture.template emplace<gr::testing::TagSource<TInput, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"values", signal}, {"n_samples_max", static_cast<gr::Size_t>(signal.size())}, {"mark_tag", false}});
    auto& fft    = fixture.template emplace<gr::blocks::fft::FFT<TInput, gr::DataSet<float>>>({{"fft_size", static_cast<gr::Size_t>(N)}, {"sample_rate", 1.f}, {"gr:compute_domain", std::string(domain) + "!"}});
    auto& sink   = fixture.template emplace<CollectorSink<gr::DataSet<float>>>();
    expect(fixture.template connect<"out", "in">(source, fft).has_value());
    expect(fixture.template connect<"out", "in">(fft, sink).has_value());
    const auto ran = gr::testing::expectRanOnAdapter(domain, [&] { return fixture.run(); });
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);

    expect(fatal(eq(sink.received.size(), kFrames))) << domain;
    const gr::DataSet<float> expected = runReferenceSpectrum<TInput>(frame, {{"sample_rate", 1.f}});
    for (const gr::DataSet<float>& spectrum : sink.received) {
        expectSpectraMatch(spectrum, expected);
        const std::size_t peak = findPeakBinF(spectrum.signalValues(0UZ));
        expect(eq(gr::meta::complex_like<TInput> ? peak - N / 2UZ : peak, bin)) << domain << ": the full spectrum is fftshifted, DC in its middle";
    }
}

const auto spectrumDomainTests = [] {
    using gr::testing::operator""_domain_test;
    "a real input's half spectrum is the host's on every served domain"_domain_test    = [](auto& fixture, std::string_view domain) { expectSpectraOnDomain<float>(fixture, domain, generateRealSine(256UZ, 17UZ), 17UZ); } | gr::testing::overGraphs(gr::testing::kAllDomains);
    "a complex input's full spectrum is the host's on every served domain"_domain_test = [](auto& fixture, std::string_view domain) { expectSpectraOnDomain<C>(fixture, domain, generateTone(256UZ, 12UZ), 12UZ); } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

const auto rawWindowTests = [] {
    using gr::testing::operator""_domain_test;
    "a windowed raw FFT transforms the windowed frames, one frame per call, on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        constexpr std::size_t kFftSize = 64UZ;
        constexpr std::size_t kFrames  = 4UZ; // from the second call on, a WebGPU window launch is queued, not awaited

        std::vector<C> signal(kFftSize * kFrames);
        for (std::size_t i = 0UZ; i < signal.size(); ++i) {
            const float phase = 2.f * std::numbers::pi_v<float> * 5.37f * static_cast<float>(i) / static_cast<float>(kFftSize);
            signal[i]         = {std::cos(phase), 0.5f * std::sin(phase)};
        }
        std::vector<float> hann(kFftSize);
        gr::algorithm::window::create(std::span<float>(hann), gr::algorithm::window::Type::Hann);
        std::vector<C> expected;
        for (std::size_t f = 0UZ; f < kFrames; ++f) {
            std::vector<C> frame(kFftSize);
            std::vector<C> spectrum(kFftSize);
            for (std::size_t i = 0UZ; i < kFftSize; ++i) {
                frame[i] = signal[f * kFftSize + i] * hann[i];
            }
            simdForward(frame, spectrum);
            expected.insert(expected.end(), spectrum.begin(), spectrum.end());
        }

        auto& source = fixture.template emplace<gr::testing::TagSource<C, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"values", signal}, {"n_samples_max", static_cast<gr::Size_t>(signal.size())}, {"mark_tag", false}});
        auto& fft    = fixture.template emplace<gr::blocks::fft::FFT<float>>({{"fft_size", static_cast<gr::Size_t>(kFftSize)}, {"window", std::string("Hann")}, {"gr:compute_domain", std::string(domain) + "!"}});
        auto& sink   = fixture.template emplace<gr::testing::TagSink<C, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"log_samples", true}});
        for (std::size_t f = 1UZ; f < kFrames; ++f) { // input is split at tags: one frame per call
            source._tags.push_back({f * kFftSize, {{"frame", static_cast<gr::Size_t>(f)}}});
        }
        expect(fixture.template connect<"out", "in">(source, fft).has_value());
        expect(fixture.template connect<"out", "in">(fft, sink).has_value());
        const std::uint64_t dispatchedBefore = gr::testing::gpuDispatchesOn(domain);
        const auto          ran              = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
        if (std::ranges::contains(gr::testing::kWebGpuDomainNames, domain)) {
            expect(ge(gr::testing::gpuDispatchesOn(domain) - dispatchedBefore, static_cast<std::uint64_t>(kFrames))) << domain << ": one window launch per call";
        }

        const std::vector<C> actual(sink._samples.begin(), sink._samples.end());
        expect(fatal(eq(actual.size(), expected.size()))) << domain;
        expect(lt(maxError(expected, actual), 1e-3f)) << domain;
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a window on the inverse is reported once, when it is set, not on every later settings change"_test = [] {
        gr::blocks::fft::FFT<float> fft({{"fft_size", gr::Size_t(64)}, {"window", std::string("Hann")}});
        gr::MsgPortIn               fromBlock;
        expect(fft.msgOut.connect(fromBlock).has_value());
        fft.init(fft.progress);
        const auto errorsAfter = [&fft, &fromBlock](const gr::property_map& changes) {
            std::ignore                 = fft.settings().setStaged(changes);
            std::ignore                 = fft.settings().applyStagedParameters();
            const std::size_t available = fromBlock.streamReader().available();
            auto              messages  = fromBlock.streamReader().get(available);
            const auto        errors    = std::ranges::count_if(messages, [](const gr::Message& message) { return !message.data.has_value(); });
            std::ignore                 = messages.consume(available);
            return static_cast<std::size_t>(errors);
        };
        expect(eq(errorsAfter({{"inverse", true}}), 1UZ)) << "the inverse runs unwindowed, and says so";
        expect(eq(errorsAfter({{"sample_rate", 48000.f}}), 0UZ)) << "a forwarded sample rate repeats nothing";
    };
};

const auto stagedTests = [] {
    "on a WebGPU domain the staged transform of device-only data matches the SIMD FFT, forward, inverse and batched"_test = [] {
        gr::testing::registerDeviceRuntimes();
        constexpr std::size_t B = 3UZ;
        const C               kSentinel{7.f, -7.f};
        for (const std::string_view domain : gr::testing::kWebGpuDomainNames) {
            if (!gr::testing::isServedForTest(domain)) {
                continue;
            }
            gr::device::DeviceContext& ctx = *gr::device::DeviceContextRegistry::instance().tryResolve(domain);
            for (const std::size_t N : {2UZ, 512UZ, 1024UZ}) { // one and nine stages (odd: the result ends in the ping-pong buffer), and ten
                const std::size_t partial = N - 1UZ;           // values after the last whole batch, one short of another: the transform leaves them alone
                std::vector<C>    input(N * B + partial, kSentinel);
                std::vector<C>    expected(N * B);
                for (std::size_t b = 0UZ; b < B; ++b) {
                    const std::vector<C> tone = generateTone(N, (2UZ * b + 1UZ) % N);
                    std::vector<C>       spectrum(N);
                    simdForward(tone, spectrum);
                    std::ranges::copy(tone, input.begin() + static_cast<std::ptrdiff_t>(b * N));
                    std::ranges::copy(spectrum, expected.begin() + static_cast<std::ptrdiff_t>(b * N));
                }
                const std::uint64_t   dispatchedBefore = gr::testing::gpuDispatchesOn(domain);
                gr::device::DeviceFFT fft;
                fft.init(ctx, N);
                gr::device::DeviceBuffer device = ctx.allocateDevice<GC>(input.size()); // GPU-only: the host transform could not read it
                ctx.copyHostToDevice(reinterpret_cast<const GC*>(input.data()), device, input.size());
                const std::uint64_t syncsBefore = ctx.syncCount();
                fft.forwardBatch(ctx, std::span{device.devicePointer<GC>(), input.size()}, N);
                expect(eq(ctx.syncCount(), syncsBefore)) << domain << " N=" << N << ": no stage waits for the device; the copy back runs after them";
                std::vector<C> spectra(input.size());
                ctx.copyDeviceToHost(device, reinterpret_cast<GC*>(spectra.data()), input.size());
                const std::vector<C> wholeBatches(spectra.begin(), spectra.begin() + static_cast<std::ptrdiff_t>(N * B));
                expect(lt(maxError(expected, wholeBatches), 1e-3f)) << domain << " N=" << N << ": the staged forward transform against the SIMD FFT";
                expect(std::ranges::all_of(std::span<const C>(spectra).last(partial), [&](const C& value) { return value == kSentinel; })) << domain << " N=" << N << ": the partial batch is left alone";
                expect(ge(gr::testing::gpuDispatchesOn(domain), dispatchedBefore + static_cast<std::uint64_t>((std::countr_zero(N) + 2) / 3))) << domain << " N=" << N << ": one dispatch per pass of up to three stages";

                fft.inverseBatch(ctx, std::span{device.devicePointer<GC>(), input.size()}, N);
                std::vector<C> recovered(input.size());
                ctx.copyDeviceToHost(device, reinterpret_cast<GC*>(recovered.data()), input.size());
                ctx.deallocate(device);
                expect(lt(maxError(input, recovered), 1e-4f)) << domain << " N=" << N << ": the inverse recovers the tones and the partial batch";
            }
        }
    };
};

int main() {
    "FFT CPU SimdFFT"_test                            = cpuSimdTests;
    "FFT DeviceFFT:CPU"_test                          = cpuContextTests;
    "FFT DeviceFFT:CPU vs SimdFFT"_test               = cpuContextCrossTests;
    "FFT Stockham vs SimdFFT"_test                    = stockhamTests;
    "FFT graph integration"_test                      = graphTests;
    "FFT spectrum mode"_test                          = spectrumTests;
    "FFT device dispatch: phase-unwrap ordering"_test = deviceUnwrapOrder;
    "FFT raw window"_test                             = rawWindowTests;
    "FFT spectrum mode on every served domain"_test   = spectrumDomainTests;
    "FFT staged on WebGPU"_test                       = stagedTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}

#endif
