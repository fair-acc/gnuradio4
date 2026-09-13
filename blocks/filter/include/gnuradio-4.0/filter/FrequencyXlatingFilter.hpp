#ifndef GNURADIO_FREQUENCY_XLATING_FILTER_HPP
#define GNURADIO_FREQUENCY_XLATING_FILTER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <format>
#include <memory_resource>
#include <numbers>
#include <span>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <gnuradio-4.0/WindowGeometry.hpp>
#include <gnuradio-4.0/algorithm/filter/DifferenceEquation.hpp>
#include <gnuradio-4.0/algorithm/filter/FastConvolution.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterForms.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::FrequencyXlatingFilter, [T], [ std::complex<float>, std::complex<double> ])

/**
 * @brief Translates a band to baseband, filters it, and decimates — the wideband-receive workhorse.
 *
 * One block covers what GR 3.10 split across Frequency Xlating FIR Filter, Frequency Xlating FFT Filter and
 * Xlating Low Pass Filter: the mixer is the same in all three and the rest is a filter whose evaluation is a
 * setting, not a separate block.
 *
 * The taps are real, so the complex stream filters as two real passes over one set of coefficients.
 *
 * `filter_type` picks how those passes are evaluated, and the two differ in more than their coefficients:
 *
 *   - FIR is a plain dot product over a window, so it is stateless and every output is independent. The block
 *     therefore declares an `input_chunk_size` larger than its `stride`, and the framework re-presents the
 *     lead-in each call — the same overlap `PolyphaseChannelizer` uses. Decimation is free: only the outputs
 *     that are published are evaluated, rather than filtering everything and throwing most of it away.
 *   - IIR is a recursion whose state depends on every input it was ever given, so its windows must not
 *     overlap and it runs sequentially. `stride` equals `input_chunk_size` in that case.
 *
 * `filter_domain` then says how an FIR is evaluated: a tap per sample, or one transform per frame by
 * overlap-save -- the same filter at a cost that stops growing with the tap count. The transform's tap spectrum
 * and plan are host objects, so it is a host form only; a device runs the tap form, already parallel over outputs.
 */
template<typename T>
requires gr::meta::complex_like<T>
struct FrequencyXlatingFilter : Block<FrequencyXlatingFilter<T>, Resampling<>, Stride<>> {
    using Real        = gr::meta::fundamental_base_value_type_t<T>;
    using Cascade     = gr::algorithm::filter::Cascade<Real>;
    using Fir         = gr::algorithm::filter::Fir<Real>;
    using Convolution = gr::algorithm::filter::FastConvolution<Real>;
    using Description = Doc<R""(translates one band to baseband, filters it and decimates; FIR or IIR, time or transform domain, by setting.

One band at a time: for many adjacent bands at once a polyphase channelizer costs far less per channel.

 * f. j. harris, "Multirate Signal Processing for Communication Systems". Upper Saddle River, NJ: Prentice Hall, 2004, ch. 6.)"">;

    PortIn<T>                                                                                                                                                                                                                                              in;
    PortOut<T>                                                                                                                                                                                                                                             out;
    Annotated<double, "frequency", Doc<"band centre in Hz, translated to DC before filtering; retunable at run time by a `gr:frequency` tag">, Visible>                                                                                                    frequency         = 0.0;
    Annotated<float, "sample_rate", Doc<"input sample rate in Hz">, Visible>                                                                                                                                                                               sample_rate       = 1'000'000.f;
    Annotated<float, "cutoff", Doc<"low-pass cutoff in Hz, applied after translation">, Visible>                                                                                                                                                           cutoff            = 100'000.f;
    Annotated<gr::algorithm::filter::FilterType, "filter_type", Doc<"'FIR' or 'IIR'">, Visible>                                                                                                                                                            filter_type       = gr::algorithm::filter::FilterType::FIR;
    Annotated<gr::Size_t, "filter_order", Doc<"design order: sets an IIR's order directly; for an FIR it sets the transition width (0.1/order) and the tap count follows from the Kaiser estimate, so a HIGHER order means MORE taps">, Limits<1U, 8192U>> filter_order      = 32U;
    Annotated<gr::algorithm::filter::ConvolutionDomain, "filter_domain", Doc<"FIR only: where the convolution is evaluated -- 'Time' a tap per sample, 'Frequency' one transform per frame, 'Auto' by tap count">, Visible>                                filter_domain     = gr::algorithm::filter::ConvolutionDomain::Auto;
    Annotated<gr::Size_t, "decimation", Doc<"output samples produced per this many filtered samples">, Visible, Limits<1U, 65536U>>                                                                                                                        decimation        = 1U;
    Annotated<gr::Size_t, "samples_per_frame", Doc<"output samples produced per work call; the transform domain rounds the frame up to a power of two, so it is a lower bound there">, Limits<1U, 65536U>>                                                 samples_per_frame = 1024U;

    // every buffer the work call touches is a reflected pmr vector, so it re-seats onto device memory when the
    // block runs there; a kernel can neither allocate nor reach a plain std::vector left on the host
    std::pmr::vector<Real> _mixedReal{};
    std::pmr::vector<Real> _mixedImag{};
    std::pmr::vector<Real> _outReal{};
    std::pmr::vector<Real> _outImag{};
    std::pmr::vector<Real> _b{}; // sections row-major, zero-padded to a common width
    std::pmr::vector<Real> _a{};
    std::pmr::vector<Real> _bReversed{}; // the FIR taps newest-weight-last, so both dot-product walks run forward
    std::pmr::vector<Real> _stateReal{}; // the recursion's accumulators; unused by the FIR path
    std::pmr::vector<Real> _stateImag{};

    GR_MAKE_REFLECTABLE(FrequencyXlatingFilter, in, out, frequency, sample_rate, cutoff, filter_type, filter_order, filter_domain, decimation, samples_per_frame, _mixedReal, _mixedImag, _outReal, _outImag, _b, _a, _bReversed, _stateReal, _stateImag);

    std::size_t _nCoefficients{};     // per section
    std::size_t _nLead{};             // samples of lead-in the window carries; 0 unless the FIR path is running
    bool        _isFir       = true;  // settled with the design, never inferred from the coefficients per call
    bool        _isTransform = false; // ditto for the domain, so no work call re-derives the choice
    double      _windowPhase = 0.0;   // the NCO phase at the window's first sample; carries, or the band jumps

    std::vector<typename Convolution::Complex> _tapSpectrum; // host only: the taps transformed once per design
    Convolution                                _convolution; // holds the transform's plan and buffers across frames

    /// the transform is a host form; on a device the tap form is already parallel over outputs
    [[nodiscard]] bool runsByTransform() const {
        using gr::algorithm::filter::ConvolutionDomain;
        if (!_isFir || _nCoefficients == 0UZ || gr::ComputeDomain::parse(this->compute_domain.value).isDevice()) {
            return false;
        }
        switch (filter_domain.value) {
        case ConvolutionDomain::Time: return false;
        case ConvolutionDomain::Frequency: return true;
        case ConvolutionDomain::Auto: break;
        }
        return _nCoefficients >= Convolution::kFrequencyDomainFromTaps;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        // a bare retune changes only where the NCO sits: the taps are unchanged and the cascade state must
        // survive, or every retune would ring. `_windowPhase` itself always carries, so the translation stays
        // phase-continuous across the change.
        const bool designUnaffected = !_b.empty() && std::ranges::none_of(std::array{"cutoff", "filter_order", "filter_type", "filter_domain", "sample_rate"}, [&newSettings](std::string_view key) { return newSettings.contains(std::string(key)); });
        if (designUnaffected) {
            declareChunks();
            return;
        }

        FilterParameters params;
        params.order = static_cast<std::size_t>(filter_order);
        params.fLow  = static_cast<double>(cutoff);
        params.fs    = static_cast<double>(sample_rate);

        _isFir = filter_type == gr::algorithm::filter::FilterType::FIR;

        if (filter_domain == gr::algorithm::filter::ConvolutionDomain::Frequency) {
            if (!_isFir) {
                this->emitErrorMessage("FrequencyXlatingFilter::settingsChanged()", gr::Error("an overlap-save frame has nowhere to put feedback: filter_domain 'Frequency' is FIR only"));
            } else if (gr::ComputeDomain::parse(this->compute_domain.value).isDevice()) {
                this->emitErrorMessage("FrequencyXlatingFilter::settingsChanged()", gr::Error(std::format("filter_domain 'Frequency' keeps its tap spectrum on the host, so it cannot run on '{}'", this->compute_domain.value)));
            }
        }

        std::vector<std::vector<Real>> sectionsB;
        std::vector<std::vector<Real>> sectionsA;
        if (_isFir) {
            const auto taps = fir::designFilter<Real>(Type::LOWPASS, params);
            sectionsB.emplace_back(taps.b.begin(), taps.b.end());
            sectionsA.emplace_back(taps.a.begin(), taps.a.end());
        } else {
            for (const auto& section : iir::designFilter<Real>(Type::LOWPASS, params)) {
                sectionsB.emplace_back(section.b.begin(), section.b.end());
                sectionsA.emplace_back(section.a.begin(), section.a.end());
            }
        }

        _nCoefficients = 0UZ;
        for (const auto& section : sectionsB) {
            _nCoefficients = std::max(_nCoefficients, section.size());
        }
        for (const auto& section : sectionsA) {
            _nCoefficients = std::max(_nCoefficients, section.size());
        }

        const std::size_t nSections = sectionsB.size();
        _b.assign(nSections * _nCoefficients, Real{});
        _a.assign(nSections * _nCoefficients, Real{});
        for (std::size_t s = 0UZ; s < nSections; ++s) {
            std::copy(sectionsB[s].begin(), sectionsB[s].end(), _b.begin() + static_cast<std::ptrdiff_t>(s * _nCoefficients));
            std::copy(sectionsA[s].begin(), sectionsA[s].end(), _a.begin() + static_cast<std::ptrdiff_t>(s * _nCoefficients));
        }

        // the FIR path reads its taps reversed; doing it once here is what lets the dot product vectorise
        _bReversed.assign(_isFir ? _b.rbegin() : _b.rend(), _b.rend());
        _stateReal.assign(_isFir ? 0UZ : Cascade::stateSize(nSections, _nCoefficients), Real{});
        _stateImag.assign(_stateReal.size(), Real{});
        _tapSpectrum.clear();
        declareChunks();
    }

    /// an FIR window carries its own lead-in and the windows overlap; a recursion's must not, or the cascade
    /// would be handed the same samples twice
    void declareChunks() {
        _isTransform            = runsByTransform();
        const std::size_t decim = std::max(std::size_t{1}, static_cast<std::size_t>(decimation));
        const std::size_t nOut  = static_cast<std::size_t>(samples_per_frame);
        if (_isTransform) {
            declareTransformChunks(nOut * decim, decim);
            return;
        }
        _nLead                  = _isFir && _nCoefficients > 0UZ ? _nCoefficients - 1UZ : 0UZ;
        const std::size_t nStep = nOut * decim;
        this->output_chunk_size = static_cast<gr::Size_t>(nOut);
        this->input_chunk_size  = static_cast<gr::Size_t>(_nLead + nStep);
        this->stride            = static_cast<gr::Size_t>(nStep);
        resizeScratch(_nLead + nStep, nOut);
    }

    /// overlap-save geometry: the frame is the power of two that holds the wanted step, the lead-in it re-presents
    /// is exactly the wrap-around it discards, and the step is trimmed to whole decimated outputs so every frame
    /// starts on the same decimation phase. `frameSizeFor` bounds the step from below, so a frame always yields one.
    void declareTransformChunks(std::size_t wantedStep, std::size_t decim) {
        const std::size_t frameSize = Convolution::frameSizeFor(_nCoefficients, std::max(wantedStep, decim));
        const std::size_t nStep     = (Convolution::outputsPerFrame(frameSize, _nCoefficients) / decim) * decim;

        _nLead                  = frameSize - nStep;
        this->output_chunk_size = static_cast<gr::Size_t>(nStep / decim);
        this->input_chunk_size  = static_cast<gr::Size_t>(frameSize);
        this->stride            = static_cast<gr::Size_t>(nStep);
        resizeScratch(frameSize, nStep);
        if (_tapSpectrum.size() != frameSize) {
            _tapSpectrum = Convolution::transformTaps(std::span<const Real>{_b.data(), _nCoefficients}, frameSize);
            _convolution.prepareTaps(std::span<const Real>{_b.data(), _nCoefficients}, frameSize);
        }
    }

    void resizeScratch(std::size_t nIn, std::size_t nOut) {
        _mixedReal.assign(nIn, Real{});
        _mixedImag.assign(nIn, Real{});
        _outReal.assign(nOut, Real{});
        _outImag.assign(nOut, Real{});
    }

    [[nodiscard]] constexpr bool scratchFits(std::size_t nWindow, std::size_t nOut) const noexcept { return _mixedReal.size() >= nWindow && _outReal.size() >= nOut; }

    /// the band-to-baseband multiply for one sample of the window, `basePhase` being the window's own start
    /// the phase is a function of the index rather than an accumulator, so the loop carries no dependency and
    /// the same expression serves the host and a kernel
    /// the same translation as `mixInto`, walked rather than indexed.
    ///
    /// A work item must compute its own angle; a host loop need not, and a sine and cosine per sample was this
    /// block's largest single cost. Rotating a phasor costs one complex multiply instead. Re-seeded from
    /// `basePhase` each window, so the drift bounds by the window rather than compounding along the stream.
    static void mixWindow(const T* input, Real* mixedReal, Real* mixedImag, std::size_t n, double basePhase, double step) noexcept {
        double       cosine  = std::cos(basePhase);
        double       sine    = std::sin(basePhase);
        const double cosStep = std::cos(step);
        const double sinStep = std::sin(step);
        for (std::size_t k = 0UZ; k < n; ++k) {
            const Real real = static_cast<Real>(input[k].real());
            const Real imag = static_cast<Real>(input[k].imag());
            mixedReal[k]    = real * static_cast<Real>(cosine) - imag * static_cast<Real>(sine);
            mixedImag[k]    = real * static_cast<Real>(sine) + imag * static_cast<Real>(cosine);

            const double rotated = cosine * cosStep - sine * sinStep;
            sine                 = cosine * sinStep + sine * cosStep;
            cosine               = rotated;
        }
    }

    GR_DEVICE_FN static constexpr void mixInto(const T* input, Real* mixedReal, Real* mixedImag, std::size_t n, double basePhase, double step) noexcept {
        const double angle  = basePhase + step * static_cast<double>(n);
        const Real   cosine = static_cast<Real>(std::cos(angle));
        const Real   sine   = static_cast<Real>(std::sin(angle));
        const Real   real   = static_cast<Real>(input[n].real());
        const Real   imag   = static_cast<Real>(input[n].imag());
        mixedReal[n]        = real * cosine - imag * sine;
        mixedImag[n]        = real * sine + imag * cosine;
    }

    /// output `m` of the FIR path, scalar and device-callable
    /// with `nLead == nTaps - 1` the window for output `m` starts exactly at `m * decim`, so both the taps and
    /// the samples are walked forward from there
    GR_DEVICE_FN static constexpr T firOutputAt(const Real* mixedReal, const Real* mixedImag, const Real* tapsReversed, std::size_t nTaps, std::size_t decim, std::size_t m) noexcept {
        const std::size_t oldest = m * decim;
        return T{Fir::template dotReversed<Real>(mixedReal + oldest, tapsReversed, nTaps), //
            Fir::template dotReversed<Real>(mixedImag + oldest, tapsReversed, nTaps)};
    }

    /// how many outputs this call may publish, and how many inputs that consumes
    [[nodiscard]] constexpr std::size_t outputsFor(std::size_t nIn, std::size_t nOutRoom) const noexcept {
        const std::size_t decim = std::max(std::size_t{1}, static_cast<std::size_t>(decimation));
        return nIn > _nLead ? std::min(nOutRoom, (nIn - _nLead) / decim) : 0UZ;
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        if (_isTransform) {
            return convolveFrames(input, output);
        }
        const std::size_t decim = std::max(std::size_t{1}, static_cast<std::size_t>(decimation));
        const std::size_t nIn   = input.size();
        const std::size_t nOut  = outputsFor(nIn, output.size());
        if (nOut == 0UZ || _nCoefficients == 0UZ) {
            return gr::work::Status::OK;
        }
        const std::size_t nWindow = _nLead + nOut * decim;
        if (!scratchFits(nWindow, nOut)) {
            resizeScratch(nWindow, nOut); // a span beyond the declared chunk: grow once rather than per call
        }

        const double step = -2.0 * std::numbers::pi * frequency.value / static_cast<double>(sample_rate);
        mixWindow(input.data(), _mixedReal.data(), _mixedImag.data(), nWindow, _windowPhase, step);

        if (_isFir) {
            const Real* const tapsReversed = _bReversed.data();
            for (std::size_t m = 0UZ; m < nOut; ++m) { // the host takes the vector path; a kernel cannot
                const std::size_t oldest = m * decim;
                output[m]                = T{Fir::dotReversedSimd(_mixedReal.data() + oldest, tapsReversed, _nCoefficients), //
                    Fir::dotReversedSimd(_mixedImag.data() + oldest, tapsReversed, _nCoefficients)};
            }
        } else {
            Cascade::filter(std::span<const Real>{_mixedReal.data(), nWindow}, _b, _a, _nCoefficients, _stateReal, std::span<Real>{_outReal.data(), nOut}, decim);
            Cascade::filter(std::span<const Real>{_mixedImag.data(), nWindow}, _b, _a, _nCoefficients, _stateImag, std::span<Real>{_outImag.data(), nOut}, decim);
            for (std::size_t m = 0UZ; m < nOut; ++m) {
                output[m] = T{_outReal[m], _outImag[m]};
            }
        }

        advancePhase(step, nOut * decim);
        return gr::work::Status::OK;
    }

    /// mix the frame, convolve each real plane against the one tap spectrum, then keep every `decimation`-th of
    /// the samples the frame is responsible for
    [[nodiscard]] gr::work::Status convolveFrames(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const gr::WindowGeometry frames = gr::windowGeometry(*this, input.size(), output.size());
        if (frames.nWindows == 0UZ || _nCoefficients == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const std::size_t decim = std::max(std::size_t{1}, static_cast<std::size_t>(decimation));
        const double      step  = -2.0 * std::numbers::pi * frequency.value / static_cast<double>(sample_rate);

        for (std::size_t frame = 0UZ; frame < frames.nWindows; ++frame) {
            const T* const samples = input.data() + frame * frames.hop;
            mixWindow(samples, _mixedReal.data(), _mixedImag.data(), frames.inChunk, _windowPhase, step);
            _convolution.convolveFrame(std::span<const Real>{_mixedReal.data(), frames.inChunk}, std::span<Real>{_outReal.data(), frames.hop});
            _convolution.convolveFrame(std::span<const Real>{_mixedImag.data(), frames.inChunk}, std::span<Real>{_outImag.data(), frames.hop});

            T* const published = output.data() + frame * frames.outChunk;
            for (std::size_t m = 0UZ; m < frames.outChunk; ++m) {
                published[m] = T{_outReal[m * decim], _outImag[m * decim]};
            }
            advancePhase(step, frames.hop);
        }

        std::ignore = input.consume(frames.nWindows * frames.hop);
        output.publish(frames.nWindows * frames.outChunk);
        return gr::work::Status::OK;
    }

    /**
     * @brief The same two passes, on a device.
     *
     * The mix and every FIR output are index-independent, so both run as flat parallel loops.
     *
     * A hatch is handed the spans as they come rather than the declared chunk, so the frame can be the whole
     * stream and the mixing scratch is taken from the context per call. The coefficients and the recursion's
     * accumulators come from the relocated members instead: those must persist between calls.
     *
     * The IIR path is here for RESIDENCY, not speed — a recursion has no parallelism to offer, so it runs in a
     * single work item and reads worse than the host. It earns its place by keeping the block on the device
     * between two device blocks. Choose FIR where the figure matters.
     */
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, auto& output) noexcept {
        const std::size_t decim = std::max(std::size_t{1}, static_cast<std::size_t>(decimation));
        const std::size_t nOut  = outputsFor(input.size(), output.size());
        if (nOut == 0UZ || _nCoefficients == 0UZ) {
            return gr::work::Status::OK;
        }
        const std::size_t nWindow = _nLead + nOut * decim;

        gr::device::DeviceBuffer realBuffer = ctx.allocateShared<Real>(nWindow);
        gr::device::DeviceBuffer imagBuffer = ctx.allocateShared<Real>(nWindow);
        Real* const              mixedReal  = realBuffer.devicePointer<Real>();
        Real* const              mixedImag  = imagBuffer.devicePointer<Real>();
        const auto               release    = [&] {
            ctx.deallocate(realBuffer);
            ctx.deallocate(imagBuffer);
        };
        if (mixedReal == nullptr || mixedImag == nullptr) {
            release();
            return gr::work::Status::ERROR;
        }

        const double      step      = -2.0 * std::numbers::pi * frequency.value / static_cast<double>(sample_rate);
        const double      basePhase = _windowPhase;
        const T* const    samples   = input.data();
        T* const          published = output.data();
        const Real* const taps      = _bReversed.data();
        const Real* const forward   = _b.data();
        const Real* const feedback  = _a.data();
        Real* const       stateReal = _stateReal.data();
        Real* const       stateImag = _stateImag.data();
        const std::size_t nCoeff    = _nCoefficients;
        const std::size_t nSections = _b.size() / std::max(std::size_t{1}, nCoeff);

        gr::device::parallelFor(ctx, nWindow, [samples, mixedReal, mixedImag, basePhase, step] GR_DEVICE_LAMBDA(std::size_t n) { //
            mixInto(samples, mixedReal, mixedImag, n, basePhase, step);
        });

        if (_isFir) {
            gr::device::parallelFor(ctx, nOut, [published, mixedReal, mixedImag, taps, nCoeff, decim] GR_DEVICE_LAMBDA(std::size_t m) { //
                published[m] = firOutputAt(mixedReal, mixedImag, taps, nCoeff, decim, m);
            });
        } else {
            gr::device::parallelFor(ctx, 1UZ, [published, mixedReal, mixedImag, forward, feedback, stateReal, stateImag, nCoeff, nSections, nOut, decim] GR_DEVICE_LAMBDA(std::size_t) {
                const std::span<const Real> bRows{forward, nSections * nCoeff};
                const std::span<const Real> aRows{feedback, nSections * nCoeff};
                const std::span<Real>       real{stateReal, Cascade::stateSize(nSections, nCoeff)};
                const std::span<Real>       imag{stateImag, Cascade::stateSize(nSections, nCoeff)};
                for (std::size_t i = 0UZ; i < nOut * decim; ++i) {
                    const Real filteredReal = Cascade::step(mixedReal[i], bRows, aRows, nCoeff, real);
                    const Real filteredImag = Cascade::step(mixedImag[i], bRows, aRows, nCoeff, imag);
                    if (i % decim == 0UZ) {
                        published[i / decim] = T{filteredReal, filteredImag};
                    }
                }
            });
        }

        release();
        advancePhase(step, nOut * decim);
        return gr::work::Status::OK;
    }

    /// the window advances by what was consumed, and the phase is kept bounded or the angle loses precision
    constexpr void advancePhase(double step, std::size_t nConsumed) noexcept { _windowPhase = std::remainder(_windowPhase + step * static_cast<double>(nConsumed), 2.0 * std::numbers::pi); }
};

} // namespace gr::filter

#endif // GNURADIO_FREQUENCY_XLATING_FILTER_HPP
