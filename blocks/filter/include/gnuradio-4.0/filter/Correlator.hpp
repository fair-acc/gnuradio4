#ifndef GNURADIO_BLOCKS_CORRELATOR_HPP
#define GNURADIO_BLOCKS_CORRELATOR_HPP

#include <memory_resource>

#include <complex>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::Correlator, [T], [ float, double, std::complex<float>, std::complex<double> ])

/// cross-correlates the stream against a stored reference, one lag per output sample. Each lag is independent
/// of every other, so the block states its window -- as many inputs as it needs to cover the lags plus the
/// reference, that many lags out -- and the framework runs those lags concurrently on whichever domain the
/// block is placed. Locating the peak of the result is a reduction over one span, a different shape; use
/// `gr::algorithm::Reduce` on the output rather than asking this block for it.
template<typename T>
requires(std::floating_point<T> || gr::meta::complex_like<T>)
struct Correlator : Block<Correlator<T>, Resampling<>, Stride<>> {
    using Description = Doc<"cross-correlation against a stored reference, one lag per output sample; the reference is conjugated, so a complex signal correlated against itself peaks real and positive">;

    PortIn<T>  in;
    PortOut<T> out;

    std::pmr::vector<T>                                   reference{};
    Annotated<gr::Size_t, "lags", Limits<1UZ, 1048576UZ>> lags = 64U;

    GR_MAKE_REFLECTABLE(Correlator, in, out, reference, lags);

    /// R_xy(l) = Σ conj(y[k]) x[l+k) -- the conjugate is on the reference, which is the convention that makes a
    /// signal's autocorrelation peak on the real axis instead of rotating with the signal's own phase
    [[nodiscard]] static constexpr T conjugated(T value) noexcept {
        if constexpr (gr::meta::complex_like<T>) {
            return std::conj(value);
        } else {
            return value;
        }
    }

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
                sum += conjugated(reference[k]) * input[lag + k];
            }
            output[lag] = sum;
        }
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_BLOCKS_CORRELATOR_HPP
