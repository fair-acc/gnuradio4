#ifndef GNURADIO_POLYPHASE_ARBITRARY_RESAMPLER_HPP
#define GNURADIO_POLYPHASE_ARBITRARY_RESAMPLER_HPP

#include <complex>
#include <memory_resource>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <gnuradio-4.0/algorithm/filter/PolyphaseArbitraryResampler.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::PolyphaseArbitraryResampler, [T], [ float, double, std::complex<float>, std::complex<double> ])

/**
 * Resamples at an arbitrary ratio through a fixed polyphase bank read between its arms.
 *
 * This is the shape GR 3.10 calls a Polyphase Arbitrary Resampler, and it also covers the Fractional
 * Resampler: both are a bank of sub-sample delays interpolated between, differing only in arm count.
 * `DriftResampler` solves the same problem by cubic Hermite instead — cheaper, and without a bank to design.
 */
template<typename T>
requires(std::floating_point<T> || gr::meta::complex_like<T>)
struct PolyphaseArbitraryResampler : Block<PolyphaseArbitraryResampler<T>> {
    using Real        = gr::meta::fundamental_base_value_type_t<T>;
    using Algorithm   = gr::algorithm::filter::PolyphaseArbitraryResampler<Real>;
    using Description = Doc<R""(resamples at an arbitrary but fixed ratio, interpolating between the arms of a polyphase bank of sub-sample delays.

`RationalResampler` is exact where the ratio is rational; `DriftResampler` carries a ratio that changes while the
stream runs. `n_phases` sets how finely the delay is quantised and `n_taps` is the length of the whole prototype, so
the floor follows `n_taps / n_phases`.

 * f. j. harris, "Multirate Signal Processing for Communication Systems". Upper Saddle River, NJ: Prentice Hall, 2004, ch. 7.)"">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<float, "resample_ratio", Doc<"output samples per input sample; above one interpolates">, Visible, Limits<0.f, 1024.f>> resample_ratio    = 1.f;
    Annotated<gr::Size_t, "n_phases", Doc<"bank arms; more arms means less interpolation error between them">, Limits<2U, 1024U>>    n_phases          = 32U;
    Annotated<gr::Size_t, "n_taps", Doc<"prototype taps per arm">, Limits<1U, 1024U>>                                                n_taps            = 16U;
    Annotated<gr::Size_t, "outputs_per_frame", Doc<"output samples produced per work call">, Limits<1U, 65536U>>                     outputs_per_frame = 256U;

    double _positionOffset = 0.0; // fractional input position the next frame starts at

    std::pmr::vector<Real> phases{}; // phase-major bank, re-seated onto device memory when the block runs there

    GR_MAKE_REFLECTABLE(PolyphaseArbitraryResampler, in, out, resample_ratio, n_phases, n_taps, phases);

    [[nodiscard]] constexpr std::size_t phaseLength() const noexcept { return phases.size() / std::max(std::size_t{1}, static_cast<std::size_t>(n_phases)); }

    PolyphaseArbitraryResampler(gr::property_map init = {}) : Block<PolyphaseArbitraryResampler<T>>(std::move(init)) {}

    void reset() { _positionOffset = 0.0; }

    /// An arbitrary ratio cannot be stated as a chunk pair, so the forwarded rate is scaled here -- without it
    /// a graph downstream of a 1.37x resampler would still read the source's rate.
    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/, property_map& forwardSettings) {
        const std::size_t nPhases    = static_cast<std::size_t>(n_phases);
        const std::size_t nTaps      = std::max(static_cast<std::size_t>(n_taps) * nPhases, 2UZ * nPhases);
        const auto        prototype  = Algorithm::designPrototype(nTaps, nPhases);
        const auto        decomposed = Algorithm::decompose(prototype, nPhases);
        phases.assign(decomposed.begin(), decomposed.end());
        this->in.min_samples = static_cast<gr::Size_t>(Algorithm::phaseLength(prototype.size(), nPhases) + 1UZ);

        if (const auto it = forwardSettings.find(gr::tag::SAMPLE_RATE.shortKey()); it != forwardSettings.end()) {
            if (const float* inputRate = (*it).second.template get_if<float>(); inputRate != nullptr && resample_ratio > 0.f) {
                forwardSettings.insert_or_assign(gr::tag::SAMPLE_RATE.shortKey(), resample_ratio * (*inputRate));
            }
        }
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const auto progress = Algorithm::template resample<T>(std::span<const T>{input.data(), input.size()}, std::span<T>{output.data(), output.size()}, //
            std::span<const Real>{phases.data(), phases.size()}, phaseLength(), static_cast<std::size_t>(n_phases), static_cast<double>(resample_ratio), _positionOffset);

        if (progress.produced == 0UZ) {
            // the span is shorter than one window and will not grow: draining it is what ends the stream
            std::ignore = input.consume(input.size());
            output.publish(0UZ);
            return gr::work::Status::OK;
        }
        std::ignore = input.consume(progress.consumed);
        output.publish(progress.produced);
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_POLYPHASE_ARBITRARY_RESAMPLER_HPP
