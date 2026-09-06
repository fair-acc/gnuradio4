#ifndef GNURADIO_INTERLEAVED_STREAM_DECIMATOR_HPP
#define GNURADIO_INTERLEAVED_STREAM_DECIMATOR_HPP

#include <complex>
#include <cstdint>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::InterleavedStreamDecimator, [T], [ float, double, std::int8_t, std::int16_t, std::int32_t, std::complex<float> ])

/**
 * @brief Keeps one group of `interleave` samples out of every `decimation` groups.
 *
 * For an interleaved multi-channel stream — I,Q,I,Q… being the common case — the group is the unit that must
 * stay together, so this drops whole groups rather than samples. There is NO anti-aliasing filter: everything
 * above the new Nyquist folds back. Use `RationalResampler` where that matters; this is the cheap path for a
 * stream that is already band-limited, or where the caller filters elsewhere.
 */
template<typename T>
struct InterleavedStreamDecimator : Block<InterleavedStreamDecimator<T>, Resampling<>> {
    using Description = Doc<R""(keeps one group of `interleave` samples out of every `decimation` groups, with no anti-aliasing filter.

Deliberately unfiltered, for a stream already band-limited or one where the aliases do not matter; use
`RationalResampler` otherwise.

 * R. E. Crochiere and L. R. Rabiner, "Interpolation and decimation of digital signals - a tutorial review",
   Proc. IEEE, vol. 69, no. 3, pp. 300-331, 1981.)"">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "decimation", Doc<"groups consumed per group produced">, Visible, Limits<1U, 65536U>>                 decimation       = 1U;
    Annotated<gr::Size_t, "interleave", Doc<"samples per group; 2 for an interleaved I/Q stream">, Visible, Limits<1U, 65536U>> interleave       = 1U;
    Annotated<gr::Size_t, "groups_per_frame", Doc<"groups produced per work call">, Limits<1U, 65536U>>                         groups_per_frame = 1024U;

    GR_MAKE_REFLECTABLE(InterleavedStreamDecimator, in, out, decimation, interleave, groups_per_frame);

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const std::size_t width  = static_cast<std::size_t>(interleave);
        const std::size_t factor = static_cast<std::size_t>(decimation);
        const std::size_t groups = static_cast<std::size_t>(groups_per_frame);

        this->input_chunk_size  = static_cast<gr::Size_t>(groups * factor * width);
        this->output_chunk_size = static_cast<gr::Size_t>(groups * width);
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) const noexcept {
        const std::size_t width  = static_cast<std::size_t>(interleave);
        const std::size_t factor = static_cast<std::size_t>(decimation);
        const std::size_t groups = output.size() / std::max(std::size_t{1}, width);

        for (std::size_t g = 0UZ; g < groups; ++g) {
            const std::size_t from = g * factor * width;
            for (std::size_t i = 0UZ; i < width; ++i) {
                if (from + i < input.size()) {
                    output[g * width + i] = input[from + i];
                }
            }
        }
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_INTERLEAVED_STREAM_DECIMATOR_HPP
