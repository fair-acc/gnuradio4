#ifndef GNURADIO_BLOCKS_CORRELATOR_HPP
#define GNURADIO_BLOCKS_CORRELATOR_HPP

#include <memory_resource>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::Correlator, [T], [ float, double ])

/**
 * @brief Cross-correlates the stream against a stored reference, one lag per output sample.
 *
 * Each lag is independent of every other, so the block states its window -- as many inputs as it needs to
 * cover the lags plus the reference, that many lags out -- and the framework runs those lags concurrently on
 * whichever domain the block is placed. Nothing in the body is device-specific.
 *
 * Locating the peak of the result is a reduction over one span, which is a different shape; use
 * `gr::algorithm::Reduce` on the output rather than asking this block for it.
 */
template<typename T>
requires std::floating_point<T>
struct Correlator : Block<Correlator<T>, Resampling<>, Stride<>> {
    using Description = Doc<"cross-correlation against a stored reference, one lag per output sample">;

    PortIn<T>  in;
    PortOut<T> out;

    std::pmr::vector<T>                                   reference{};
    Annotated<gr::Size_t, "lags", Limits<1UZ, 1048576UZ>> lags = 64U;

    GR_MAKE_REFLECTABLE(Correlator, in, out, reference, lags);

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const std::size_t length = std::max(std::size_t{1}, reference.size());
        this->output_chunk_size  = lags;
        this->input_chunk_size   = static_cast<gr::Size_t>(static_cast<std::size_t>(lags) + length - 1UZ);
        this->stride             = lags;
    }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) const noexcept {
        const std::size_t length = reference.size();
        for (std::size_t lag = 0UZ; lag < output.size(); ++lag) {
            T sum{};
            for (std::size_t k = 0UZ; k < length; ++k) {
                sum += reference[k] * input[lag + k];
            }
            output[lag] = sum;
        }
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_BLOCKS_CORRELATOR_HPP
