#ifndef GNURADIO_RATIONAL_RESAMPLER_HPP
#define GNURADIO_RATIONAL_RESAMPLER_HPP

#include <memory_resource>
#include <span>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/filter/PolyphaseResampler.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::RationalResampler, [T], [ float, double ])

/**
 * @brief Rational L/M sample-rate conversion.
 *
 * The block states its window through `Resampling<>`/`Stride<>` — one hop of `decimation` input samples yields
 * `interpolation` outputs — so the framework may run any number of those windows at once, on the host or on a
 * device, without the body knowing which.
 */
template<typename T>
requires std::floating_point<T>
struct RationalResampler : Block<RationalResampler<T>, Resampling<>, Stride<>> {
    using Algorithm   = gr::algorithm::filter::PolyphaseResampler<T>;
    using Description = Doc<"resamples by a rational factor interpolation/decimation using a polyphase low-pass">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "interpolation", Limits<1UZ, 1024UZ>> interpolation = 1U;
    Annotated<gr::Size_t, "decimation", Limits<1UZ, 1024UZ>>    decimation    = 1U;
    Annotated<gr::Size_t, "n_taps", Limits<1UZ, 8192UZ>>        n_taps        = 32U;

    std::pmr::vector<T> phases{}; // phase-major prototype, re-seated onto device memory when the block runs there

    GR_MAKE_REFLECTABLE(RationalResampler, in, out, interpolation, decimation, n_taps, phases);

    gr::Size_t _phaseLength = 1U;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const std::size_t L          = std::max(std::size_t{1}, static_cast<std::size_t>(interpolation));
        const std::size_t M          = std::max(std::size_t{1}, static_cast<std::size_t>(decimation));
        const T           cutoff     = T{0.5} / static_cast<T>(std::max(L, M)); // the narrower of the two Nyquist limits
        const auto        prototype  = Algorithm::designPrototype(std::max(static_cast<std::size_t>(n_taps), L), cutoff);
        const auto        decomposed = Algorithm::decompose(prototype, L);

        phases.assign(decomposed.begin(), decomposed.end());
        _phaseLength = static_cast<gr::Size_t>(Algorithm::phaseLength(prototype.size(), L));

        this->output_chunk_size = static_cast<gr::Size_t>(L);
        this->input_chunk_size  = static_cast<gr::Size_t>(Algorithm::windowLength(L, L, M, _phaseLength));
        this->stride            = static_cast<gr::Size_t>(M);
    }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) const noexcept {
        const std::span<const T> window{input.data(), input.size()};
        const std::span<const T> taps{phases.data(), phases.size()};
        for (std::size_t n = 0UZ; n < output.size(); ++n) { // n is global to the span, and the phase formula is too
            output[n] = Algorithm::sampleAt(window, taps, static_cast<std::size_t>(_phaseLength), static_cast<std::size_t>(interpolation), static_cast<std::size_t>(decimation), n);
        }
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_RATIONAL_RESAMPLER_HPP
