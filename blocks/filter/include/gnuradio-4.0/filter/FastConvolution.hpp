#ifndef GNURADIO_BLOCKS_FAST_CONVOLUTION_HPP
#define GNURADIO_BLOCKS_FAST_CONVOLUTION_HPP

#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/WindowGeometry.hpp>
#include <gnuradio-4.0/algorithm/filter/FastConvolution.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/device/SyclFFT.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::FastConvolutionFilter, [T], [ float, double ])

/**
 * @brief FIR filtering by overlap-save, for filters long enough that a transform per frame beats a tap per sample.
 *
 * Declares the same window shape as the direct filter it replaces -- a frame in, the frame's useful samples out,
 * advancing by those -- so the two are interchangeable in a graph and can be compared against each other.
 *
 * On a device every frame in the span is transformed in one batch, so the cost that matters is the batch, not
 * the frame. The tap spectrum is the same numbers on both sides; only who evaluates the transform differs.
 */
template<typename T>
requires std::floating_point<T>
struct FastConvolutionFilter : Block<FastConvolutionFilter<T>, Resampling<>, Stride<>> {
    using Algorithm   = gr::algorithm::filter::FastConvolution<T>;
    using Description = Doc<R""(FIR filter evaluated by overlap-save in the frequency domain, the frame stated directly through `outputs_per_frame`.

`fir_filter` reaches the same evaluation through `filter_domain` and picks it by tap count; this block exists to pin
the domain and the frame size, which is what a device batch and a like-for-like comparison against the tap form both
need.

 * T. G. Stockham, "High-speed convolution and correlation", in Proc. AFIPS Spring Joint Computer Conf., vol. 28,
   1966, pp. 229-233.)"">;

    PortIn<T>  in;
    PortOut<T> out;

    std::pmr::vector<T>                                                taps{T{1}}; // pmr so the taps can be re-seated onto device memory, as USER_API_GPU_Blocks.md requires
    Annotated<gr::Size_t, "outputs_per_frame", Limits<1UZ, 1048576UZ>> outputs_per_frame = 256U;

    GR_MAKE_REFLECTABLE(FastConvolutionFilter, in, out, taps, outputs_per_frame);

    std::vector<typename Algorithm::Complex> _tapSpectrum; // device only: the host keeps its own inside `_convolution`
    Algorithm                                _convolution; // holds the transform's plan and buffers across frames
    gr::device::SyclFFT                      _syclFft;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        if (taps.empty()) {
            // a filter with no taps is not a pass-through here: the frame would carry more outputs than points and
            // the discard below would wrap, so the design is refused while it can still be named
            this->emitErrorMessage("settingsChanged()", "a transform-domain filter needs at least one tap");
            return;
        }
        const std::size_t frameSize = Algorithm::frameSizeFor(taps.size(), static_cast<std::size_t>(outputs_per_frame));
        const std::size_t nOutputs  = Algorithm::outputsPerFrame(frameSize, taps.size());

        _tapSpectrum = Algorithm::transformTaps(taps, frameSize);
        _convolution.prepareTaps(taps, frameSize);

        this->input_chunk_size  = static_cast<gr::Size_t>(frameSize);
        this->output_chunk_size = static_cast<gr::Size_t>(nOutputs);
        this->stride            = static_cast<gr::Size_t>(nOutputs);
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const gr::WindowGeometry frames = gr::windowGeometry(*this, input.size(), output.size());
        for (std::size_t frame = 0UZ; frame < frames.nWindows; ++frame) {
            _convolution.convolveFrame(std::span<const T>{input.data() + frame * frames.hop, frames.inChunk}, std::span<T>{output.data() + frame * frames.outChunk, frames.outChunk});
        }
        return gr::work::Status::OK;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output)
    requires std::same_as<T, float> // gr::device::SyclFFT is a float-only tier; double precision stays on the host
    {
        using Complex = gr::device::SyclFFT::C;

        const gr::WindowGeometry frames = gr::windowGeometry(*this, input.size(), output.size());
        if (frames.nWindows == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const std::size_t frameSize = frames.inChunk;
        const std::size_t nOutputs  = frames.outChunk;
        const std::size_t nFrames   = frames.nWindows;
        const std::size_t nBins     = nFrames * frameSize;
        const std::size_t nResults  = nFrames * nOutputs;

        _syclFft.init(ctx, frameSize);

        gr::device::DeviceBuffer frameSpectra = ctx.allocateShared<Complex>(nBins);
        gr::device::DeviceBuffer tapSpectrum  = ctx.allocateShared<Complex>(frameSize);
        Complex*                 spectra      = frameSpectra.devicePointer<Complex>();
        const Complex*           filterBins   = tapSpectrum.devicePointer<Complex>();
        if (spectra == nullptr || filterBins == nullptr) {
            ctx.deallocate(frameSpectra);
            ctx.deallocate(tapSpectrum);
            if (!this->deviceFallbackIsAllowed("could not obtain device memory for the frame spectra")) {
                return gr::work::Status::ERROR;
            }
            return processBulk(input, output);
        }
        ctx.copyHostToDevice(reinterpret_cast<const Complex*>(_tapSpectrum.data()), tapSpectrum, frameSize);

        const T* samples = input.data();
        gr::device::parallelFor(ctx, nBins, [spectra, samples, frameSize, nOutputs] GR_DEVICE_LAMBDA(std::size_t i) { spectra[i] = Complex{samples[(i / frameSize) * nOutputs + i % frameSize], 0.f}; });
        _syclFft.forwardBatch(ctx, std::span<Complex>{spectra, nBins}, frameSize);
        gr::device::parallelFor(ctx, nBins, [spectra, filterBins, frameSize] GR_DEVICE_LAMBDA(std::size_t i) {
            const Complex signal = spectra[i];
            const Complex filter = filterBins[i % frameSize];
            spectra[i]           = Complex{signal.re * filter.re - signal.im * filter.im, signal.re * filter.im + signal.im * filter.re};
        });
        _syclFft.inverseBatch(ctx, std::span<Complex>{spectra, nBins}, frameSize);

        T*                results = output.data();
        const std::size_t discard = frameSize - nOutputs; // the wrap-around the saved tail exists to make right
        gr::device::parallelFor(ctx, nResults, [results, spectra, frameSize, nOutputs, discard] GR_DEVICE_LAMBDA(std::size_t i) { results[i] = spectra[(i / nOutputs) * frameSize + discard + i % nOutputs].re; });
        ctx.wait();

        ctx.deallocate(frameSpectra);
        ctx.deallocate(tapSpectrum);

        std::ignore = input.consume(nResults);
        output.publish(nResults);
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_BLOCKS_FAST_CONVOLUTION_HPP
