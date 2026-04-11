#ifndef GNURADIO_FFT2_HPP
#define GNURADIO_FFT2_HPP

#include <format>
#include <span>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/fourier/SyclFFT.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/ShaderFragment.hpp>

namespace gr::blocks::fourier {

GR_REGISTER_BLOCK("gr::blocks::fourier::FFT2", gr::blocks::fourier::FFT2, [T], [ float, double ])

template<typename T>
struct FFT2 : gr::Block<FFT2<T>> {
    using Description = Doc<R""(raw forward/inverse FFT, dispatches to CPU SIMD or device.

Outputs std::complex<T>, not DataSet. Composable for FFT-based FIR, spectrum chains.
Supports single and batched operation (batch count auto-detected from input size).
Dispatches to CPU SimdFFT, GPU SYCL (Stockham), or GPU GLSL (Stockham) depending
on the compute_domain setting.)"">;

    using ComplexType = std::complex<T>;

    PortIn<ComplexType>  in;
    PortOut<ComplexType> out;

    Annotated<gr::Size_t, "fft size", Limits<8UZ, 1048576UZ>> fft_size = 4096UZ;
    Annotated<bool, "inverse">                                inverse  = false;

    GR_MAKE_REFLECTABLE(FFT2, in, out, fft_size, inverse);

    gr::algorithm::FFT<ComplexType, ComplexType> _cpuFft;
    gr::device::SyclFFT                          _syclFft;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("fft_size")) {
            _cpuFft.fftSize = 0;
        }
    }

    gr::work::Status processBulk(InputSpanLike auto& inSpan, OutputSpanLike auto& outSpan) {
        const auto available = std::min(inSpan.size(), outSpan.size());
        const auto N         = static_cast<std::size_t>(fft_size);
        if (available < N) {
            std::ignore = inSpan.consume(0);
            outSpan.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        const auto nBatches = available / N;
        const auto total    = nBatches * N;

        for (std::size_t b = 0; b < nBatches; ++b) {
            auto inSlice  = std::span<const ComplexType>(inSpan.data() + b * N, N);
            auto outSlice = std::span<ComplexType>(outSpan.data() + b * N, N);

            if (inverse) {
                std::ranges::transform(inSlice, outSlice.begin(), [](auto z) { return std::conj(z); });
                _cpuFft.compute(outSlice, outSlice);
                T invN = T(1) / static_cast<T>(N);
                std::ranges::transform(outSlice, outSlice.begin(), [invN](auto z) { return std::conj(z) * invN; });
            } else {
                _cpuFft.compute(inSlice, outSlice);
            }
        }

        std::ignore = inSpan.consume(total);
        outSpan.publish(total);
        return work::Status::OK;
    }

#if GR_DEVICE_HAS_SYCL_IMPL
    std::unique_ptr<gr::device::DeviceContextSycl> _syclCtxPtr;

    gr::work::Status processBulk_sycl(sycl::queue& q, InputSpanLike auto& inSpan, OutputSpanLike auto& outSpan) {
        const auto N         = static_cast<std::size_t>(fft_size);
        const auto available = std::min(inSpan.size(), outSpan.size());
        if (available < N) {
            std::ignore = inSpan.consume(0);
            outSpan.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        if (!_syclCtxPtr) {
            _syclCtxPtr = std::make_unique<gr::device::DeviceContextSycl>(q);
        }
        auto& ctx = *_syclCtxPtr;
        _syclFft.init(ctx, N);

        const auto nBatches = available / N;
        const auto total    = nBatches * N;

        q.memcpy(outSpan.data(), inSpan.data(), total * sizeof(ComplexType)).wait();

        auto outData = std::span<gr::complex<T>>{reinterpret_cast<gr::complex<T>*>(outSpan.data()), total};
        if (inverse) {
            _syclFft.inverse(ctx, outData);
        } else {
            _syclFft.forwardBatch(ctx, outData, N);
        }

        std::ignore = inSpan.consume(total);
        outSpan.publish(total);
        return gr::work::Status::OK;
    }
#endif

    [[nodiscard]] gr::device::ShaderFragment shaderFragment() const {
        return {.glslFunction = "float process(float x) { return x; }", // identity stub; FFT dispatch via GLSL goes through GlslFFT directly
            .constants        = {},
            .inputChunkSize   = static_cast<std::size_t>(fft_size),
            .outputChunkSize  = static_cast<std::size_t>(fft_size),
            .workgroupSize    = 256
        };
    }
};

} // namespace gr::blocks::fourier

#endif // GNURADIO_FFT2_HPP
