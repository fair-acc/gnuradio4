#ifndef GNURADIO_BLOCKS_FRAME_STATISTICS_HPP
#define GNURADIO_BLOCKS_FRAME_STATISTICS_HPP

#include <cmath>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::RootMeanSquare, [T], [ float, double ])

/**
 * @brief One root-mean-square figure per frame of input.
 *
 * The shape shared by every per-frame statistic -- RMS, peak, an AGC's gain estimate -- is many samples in,
 * one figure out, repeated. Stated as a window, the framework reduces each frame in its own work item, so a
 * stream with many frames in flight parallelises without the block knowing where it runs.
 *
 * Reducing a *single* large frame cooperatively is a different problem; `gr::algorithm::Reduce` does that.
 */
template<typename T>
requires std::floating_point<T>
struct RootMeanSquare : Block<RootMeanSquare<T>, Resampling<>, Stride<>> {
    using Description = Doc<"root-mean-square of each frame of `frame_size` input samples">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "frame_size", Limits<1UZ, 1048576UZ>> frame_size = 1024U;

    GR_MAKE_REFLECTABLE(RootMeanSquare, in, out, frame_size);

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        this->input_chunk_size  = frame_size;
        this->output_chunk_size = 1U;
        this->stride            = frame_size;
    }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) const noexcept {
        const std::size_t frame = static_cast<std::size_t>(frame_size);
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            T sumOfSquares{};
            for (std::size_t i = 0UZ; i < frame; ++i) {
                const T sample = input[n * frame + i];
                sumOfSquares += sample * sample;
            }
            output[n] = static_cast<T>(std::sqrt(static_cast<double>(sumOfSquares) / static_cast<double>(frame)));
        }
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_BLOCKS_FRAME_STATISTICS_HPP
