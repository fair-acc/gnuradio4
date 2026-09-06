#ifndef GNURADIO_BLOCKS_DRIFT_RESAMPLER_HPP
#define GNURADIO_BLOCKS_DRIFT_RESAMPLER_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory_resource>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

#include <gnuradio-4.0/algorithm/filter/HermiteResampler.hpp>
#include <gnuradio-4.0/algorithm/filter/PolyphaseArbitraryResampler.hpp>

namespace gr::filter {

namespace detail {
constexpr bool isPositiveRatio(float r) noexcept { return r > 0.f; }
} // namespace detail

GR_REGISTER_BLOCK(gr::filter::DriftResampler, [T], [ float, double, std::complex<float>, std::complex<double> ])

/// resamples at an arbitrary, possibly drifting ratio. Where `RationalResampler` states a fixed window and lets
/// the framework run those windows concurrently, this one cannot: the number of outputs per input is not fixed,
/// so it owns its own accounting -- the trade for a ratio that may move while the stream runs. The accounting is
/// what is sequential, not the arithmetic: within one call the ratio is fixed, so output `m` reads
/// `phase + m / ratio` and the outputs are independent. The device hatch below settles the count first and then
/// evaluates them as one flat parallel loop.
template<typename T>
requires(std::floating_point<T> || gr::meta::complex_like<T>)
struct DriftResampler : Block<DriftResampler<T>> {
    using Real        = gr::meta::fundamental_base_value_type_t<T>; // the taps are real; the samples need not be
    using Hermite     = gr::algorithm::filter::HermiteResampler<T>;
    using Polyphase   = gr::algorithm::filter::PolyphaseArbitraryResampler<Real>;
    using Description = Doc<R""(resamples at an arbitrary ratio that may drift while the stream runs; cubic Hermite or a polyphase bank by setting.

The only one of the three that cannot state a fixed rate, so it owns its own sample accounting and tag mapping. Where
the ratio is constant prefer `RationalResampler` (exact, rational) or `PolyphaseArbitraryResampler` (any fixed ratio).
Hermite is cheap and right when the ratio moves every sample; the bank is dearer and what a wide band needs to stay
clean.

 * E. Catmull and R. Rom, "A class of local interpolating splines", in Computer Aided Geometric Design,
   R. E. Barnhill and R. F. Riesenfeld, Eds. New York: Academic Press, 1974, pp. 317-326.
 * f. j. harris, "Multirate Signal Processing for Communication Systems". Upper Saddle River, NJ: Prentice Hall, 2004, ch. 7.)"">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<float, "ratio", Doc<"output samples per input sample; may be changed while running">, Limits<0.f, 1024.f, &detail::isPositiveRatio>>                                                                                               ratio         = 1.f;
    Annotated<gr::algorithm::filter::InterpolationKernel, "interpolation", Doc<"'Hermite' (cheap, 4-point cubic) or 'Polyphase' (a windowed-sinc bank)">, Visible>                                                                               interpolation = gr::algorithm::filter::InterpolationKernel::Hermite;
    Annotated<gr::Size_t, "n_phases", Doc<"polyphase arms: how finely the fractional delay is quantised; ignored by the Hermite kernel">, Limits<2U, 1024U>>                                                                                     n_phases      = 32U;
    Annotated<gr::Size_t, "n_taps", Doc<"length of the WHOLE prototype, so each arm gets n_taps/n_phases of it -- raise both together, or finer delay steps are paid for with a shorter arm; ignored by the Hermite kernel">, Limits<4U, 8192U>> n_taps        = 1024U;

    std::pmr::vector<Real> _phases{}; // the polyphase bank, empty while the Hermite kernel is selected

    GR_MAKE_REFLECTABLE(DriftResampler, in, out, ratio, interpolation, n_phases, n_taps, _phases);

    static constexpr gr::Size_t kMinimumWindow = 4U; // the four points the interpolant is built from

    double      _phase    = 0.0;
    std::size_t _phaseLen = 0UZ;

    DriftResampler(gr::property_map init = {}) : Block<DriftResampler<T>>(std::move(init)) { in.min_samples = kMinimumWindow; }

    void reset() { _phase = 0.0; }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        // the phase carries across a kernel change: both count in input samples, so a switch mid-stream shifts
        // the interpolation quality and nothing else
        if (interpolation != gr::algorithm::filter::InterpolationKernel::Polyphase) {
            _phases.clear();
            _phaseLen      = 0UZ;
            in.min_samples = kMinimumWindow;
            return;
        }
        const std::size_t nPhases   = static_cast<std::size_t>(n_phases);
        const auto        prototype = Polyphase::designPrototype(std::max(static_cast<std::size_t>(n_taps), 4UZ * nPhases), nPhases);
        const auto        bank      = Polyphase::decompose(prototype, nPhases);
        _phases.assign(bank.begin(), bank.end());
        _phaseLen      = Polyphase::phaseLength(prototype.size(), nPhases);
        in.min_samples = static_cast<gr::Size_t>(std::max(_phaseLen, static_cast<std::size_t>(kMinimumWindow)));
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::span<const T> window{input.data(), input.size()};
        const std::span<T>       published{output.data(), output.size()};

        const double basePhase = _phase; // the kernel moves `_phase`, and the tag mapping below needs its start
        const double stepSize  = ratio > 0.f ? 1.0 / static_cast<double>(ratio) : 0.0;

        // the two kernels report the same pair but each as its own type, so the counts are taken rather than
        // the struct: a ternary over them does not compile and a shared base would be a type for its own sake
        struct {
            std::size_t consumed = 0UZ;
            std::size_t produced = 0UZ;
        } progress;
        if (interpolation == gr::algorithm::filter::InterpolationKernel::Polyphase && !_phases.empty()) {
            const auto step = Polyphase::template resample<T>(window, published, _phases, _phaseLen, static_cast<std::size_t>(n_phases), static_cast<double>(ratio), _phase);
            progress        = {step.consumed, step.produced};
        } else {
            const auto step = Hermite::resample(window, published, static_cast<double>(ratio), _phase);
            progress        = {step.consumed, step.produced};
        }
        if (progress.produced == 0UZ) {
            // the span is shorter than the window the interpolant needs and will not grow: draining it is what
            // ends the stream, where holding on to it would spin forever asking for samples nobody will send
            std::ignore = input.consume(input.size());
            output.publish(0UZ);
            return gr::work::Status::OK;
        }
        forwardTagsOnto(input, output, progress.produced, basePhase, stepSize);
        std::ignore = input.consume(progress.consumed);
        output.publish(progress.produced);
        return gr::work::Status::OK;
    }

    /// how many outputs this span can yield, and where the read position ends up.
    ///
    /// The host loop discovers this by running until the window no longer fits; a parallel one has to know it
    /// up front. `bound` is the first position whose window runs past the span, so the closed form is
    /// `(bound - phase) * ratio` -- then corrected by at most a step, because a rounding either way would
    /// disagree with the loop's own test and the two paths must publish the same count.
    [[nodiscard]] constexpr std::size_t outputsFor(std::size_t nIn, std::size_t nOutRoom, std::size_t window, double step) const noexcept {
        if (nIn < window || nOutRoom == 0UZ || step <= 0.0) {
            return 0UZ;
        }
        const double bound = static_cast<double>(nIn - window + 1UZ);
        std::size_t  nOut  = nOutRoom;
        if (const double reach = (bound - _phase) / step; reach < static_cast<double>(nOutRoom)) {
            nOut = reach <= 0.0 ? 0UZ : static_cast<std::size_t>(std::ceil(reach));
        }
        while (nOut > 0UZ && _phase + static_cast<double>(nOut - 1UZ) * step >= bound) {
            --nOut;
        }
        return nOut;
    }

    /// The same interpolation, on a device: one work item per output, each reading its own position.
    ///
    /// The hatch rather than a declared window, because the output count per input is not a chunk pair and the
    /// accounting has to stay with the block. The host path keeps its own loop unchanged -- at the rates a
    /// Hermite frame sustains it is already near the framework's ceiling, and a launch would only cost.
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) noexcept {
        const bool byBank = interpolation == gr::algorithm::filter::InterpolationKernel::Polyphase && !_phases.empty();
        // inputs the newest output reads from its base, matching each kernel's guard: the bank reaches one sample
        // past its window to interpolate past its last arm, the interpolant reads two ahead of its index
        const std::size_t window = byBank ? _phaseLen + 1UZ : 3UZ;
        const std::size_t nIn    = input.size();
        const double      step   = ratio > 0.f ? 1.0 / static_cast<double>(ratio) : 0.0;
        const std::size_t nOut   = outputsFor(nIn, output.size(), window, step);
        if (nOut == 0UZ) {
            // the span is shorter than the window the interpolant needs; draining it is what ends the stream
            std::ignore = input.consume(nIn >= window ? 0UZ : nIn);
            output.publish(0UZ);
            return gr::work::Status::OK;
        }

        const T* const    samples   = input.data();
        T* const          published = output.data();
        const double      basePhase = _phase;
        const Real* const taps      = _phases.data();
        const std::size_t nTaps     = _phases.size();
        const std::size_t phaseLen  = _phaseLen;
        const std::size_t nPhases   = static_cast<std::size_t>(n_phases);

        if (byBank) {
            gr::device::parallelFor(ctx, nOut, [published, samples, taps, nIn, nTaps, phaseLen, nPhases, basePhase, step] GR_DEVICE_LAMBDA(std::size_t m) { published[m] = Polyphase::template sampleAt<T>(std::span<const T>{samples, nIn}, std::span<const Real>{taps, nTaps}, phaseLen, nPhases, basePhase + static_cast<double>(m) * step); });
        } else {
            gr::device::parallelFor(ctx, nOut, [published, samples, nIn, basePhase, step] GR_DEVICE_LAMBDA(std::size_t m) {
                const double      position = basePhase + static_cast<double>(m) * step;
                const std::size_t index    = static_cast<std::size_t>(position);
                published[m]               = Hermite::sampleAt(std::span<const T>{samples, nIn}, index, static_cast<Real>(position - static_cast<double>(index)));
            });
        }

        // the read position may run past the span when one output consumes many inputs; releasing more than was
        // given would be a lie, so the remainder rides in the phase instead. The interpolant also keeps a lead-in
        // for its left neighbour, which the bank, reading forward from its base, does not -- as each host kernel does
        const double      finalPosition = basePhase + static_cast<double>(nOut) * step;
        const std::size_t reached       = std::min(static_cast<std::size_t>(finalPosition), nIn);
        const std::size_t consumed      = byBank || reached == 0UZ ? reached : reached - 1UZ;
        _phase                          = finalPosition - static_cast<double>(consumed);
        forwardTagsOnto(input, output, nOut, basePhase, step);
        std::ignore = input.consume(consumed);
        output.publish(nOut);
        return gr::work::Status::OK;
    }

    /// carry the input's tags onto the outputs they belong to.
    ///
    /// A block declaring `Resampling<>` has the framework map tag indices for it; this one cannot declare a ratio
    /// at all, so it maps them itself or the stream arrives downstream stripped of its metadata. Output `m` reads
    /// `basePhase + m * step`, so input `i`'s tag belongs to the first output reaching it.
    static void forwardTagsOnto(InputSpanLike auto& input, OutputSpanLike auto& output, std::size_t nOut, double basePhase, double step) {
        if (nOut == 0UZ || step <= 0.0) {
            return;
        }
        for (const auto& [relativeIndex, tagMap] : input.tags()) {
            const double reach = (static_cast<double>(relativeIndex) - basePhase) / step;
            if (reach >= static_cast<double>(nOut)) {
                continue; // its output is not produced yet; the tag rides on to the next call
            }
            output.publishTag(tagMap, reach <= 0.0 ? 0UZ : static_cast<std::size_t>(std::ceil(reach)));
        }
    }

    /// A drifting ratio cannot be stated as a chunk pair, so the framework's static rescale cannot express it and
    /// the block scales the forwarded rate itself. Without this a graph downstream of a 1.5x resampler still reads
    /// the source's rate.
    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/, property_map& forwardSettings) {
        if (const auto it = forwardSettings.find(gr::tag::SAMPLE_RATE.shortKey()); it != forwardSettings.end()) {
            if (const float* inputRate = (*it).second.template get_if<float>(); inputRate != nullptr && ratio > 0.f) {
                forwardSettings.insert_or_assign(gr::tag::SAMPLE_RATE.shortKey(), ratio * (*inputRate));
            }
        }
    }
};

} // namespace gr::filter

#endif // GNURADIO_BLOCKS_DRIFT_RESAMPLER_HPP
