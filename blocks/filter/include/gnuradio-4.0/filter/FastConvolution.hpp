#ifndef GNURADIO_BLOCKS_FAST_CONVOLUTION_HPP
#define GNURADIO_BLOCKS_FAST_CONVOLUTION_HPP

#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/filter/FastConvolution.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::FastConvolutionFilter, [T], [ float, double ])

/**
 * @brief FIR filtering by overlap-save, for filters long enough that a transform per frame beats a tap per sample.
 *
 * Declares the same window shape as the direct filter it replaces -- a frame in, the frame's useful samples out,
 * advancing by those -- so the two are interchangeable in a graph and can be compared against each other. The
 * transform is a host one; reaching a device means handing `gr::device::SyclFFT` a queue, which is what the FFT
 * block does, and is a separate step from this.
 */
template<typename T>
requires std::floating_point<T>
struct FastConvolutionFilter : Block<FastConvolutionFilter<T>, Resampling<>, Stride<>> {
    using Algorithm   = gr::algorithm::filter::FastConvolution<T>;
    using Description = Doc<"FIR filter evaluated by overlap-save in the frequency domain">;

    PortIn<T>  in;
    PortOut<T> out;

    std::vector<T>                                                     taps{T{1}};
    Annotated<gr::Size_t, "outputs_per_frame", Limits<1UZ, 1048576UZ>> outputs_per_frame = 256U;

    GR_MAKE_REFLECTABLE(FastConvolutionFilter, in, out, taps, outputs_per_frame);

    std::vector<typename Algorithm::Complex> _tapSpectrum;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const std::size_t frameSize = Algorithm::frameSizeFor(taps.size(), static_cast<std::size_t>(outputs_per_frame));
        const std::size_t nOutputs  = Algorithm::outputsPerFrame(frameSize, taps.size());

        _tapSpectrum = Algorithm::transformTaps(taps, frameSize);

        this->input_chunk_size  = static_cast<gr::Size_t>(frameSize);
        this->output_chunk_size = static_cast<gr::Size_t>(nOutputs);
        this->stride            = static_cast<gr::Size_t>(nOutputs);
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t frameSize = _tapSpectrum.size();
        const std::size_t nOutputs  = static_cast<std::size_t>(this->output_chunk_size);
        for (std::size_t frame = 0UZ; (frame + 1UZ) * nOutputs <= output.size() && frame * nOutputs + frameSize <= input.size(); ++frame) {
            Algorithm::convolveFrame(std::span<const T>{input.data() + frame * nOutputs, frameSize}, _tapSpectrum, std::span<T>{output.data() + frame * nOutputs, nOutputs});
        }
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_BLOCKS_FAST_CONVOLUTION_HPP
