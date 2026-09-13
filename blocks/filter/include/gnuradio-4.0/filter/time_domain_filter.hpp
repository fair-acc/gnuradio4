#ifndef GNURADIO_TIME_DOMAIN_FILTER_HPP
#define GNURADIO_TIME_DOMAIN_FILTER_HPP
#include <algorithm>
#include <complex>
#include <execution>
#include <functional>
#include <iterator>
#include <numeric>
#include <variant>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/WindowGeometry.hpp>
#include <gnuradio-4.0/algorithm/filter/DifferenceEquation.hpp>
#include <gnuradio-4.0/algorithm/filter/FastConvolution.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterForms.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/device/SyclFFT.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include <magic_enum.hpp>

namespace gr::filter {

using namespace gr;

/// the algorithm layer owns both; these are the spellings the block layer and its registrations use
using ConvolutionDomain = gr::algorithm::filter::ConvolutionDomain;
using IIRForm           = gr::algorithm::filter::IIRForm;

template<typename T, ConvolutionDomain form = ConvolutionDomain::Auto>
requires std::floating_point<T>
struct fir_filter : Block<fir_filter<T, form>, Resampling<>, Stride<>> {
    using TParent     = Block<fir_filter<T, form>, Resampling<>, Stride<>>;
    using Convolution = gr::algorithm::filter::FastConvolution<T>;
    using Convolve    = gr::algorithm::filter::Fir<T>;
    using Description = Doc<R""(@brief Finite Impulse Response (FIR) filter class

The transfer function of an FIR filter is given by:
H(z) = b[0] + b[1]*z^-1 + b[2]*z^-2 + ... + b[N]*z^-N

There are two ways to evaluate it and they compute the same samples to within the rounding of a different summation
order: a multiply-add per tap per sample, or one transform per frame by overlap-save. Which is cheaper depends on
the tap count; `bm_Filter` sweeps the crossover on whatever machine it is run.

'ConvolutionDomain' says which one to use. TIME_DOMAIN and FREQUENCY_DOMAIN compile only their own path, for a target where
code size is the constraint. AUTO carries both and picks: the 'mode' setting when it names a form, otherwise the
tap count. On a device AUTO always takes the transform -- that is what a device is worth using for, and the tap
form there would need a kernel of its own -- so a graph wanting the tap form on a device instantiates TIME_DOMAIN,
which keeps the framework's window dispatch and is the faster of the two at short lengths.

Both forms are sample-count preserving and `y[0]` belongs to `x[0]`: the window preceding the stream is primed with
zeros and its tail carried across spans, which is also exactly the overlap the transform form discards. They are
therefore interchangeable sample for sample, which is what lets the Auto form choose between them by tap count.
)"">;

    /// where `Auto` switches on the host, in taps. A default rather than a claim -- a machine or a backend moves it,
    /// so name the domain explicitly to override it. On a device `Auto` takes the transform whatever the tap count.
    static constexpr std::size_t kFrequencyDomainFromTaps = 128UZ;

    PortIn<T>  in;
    PortOut<T> out;
    Tensor<T>  b{T{1}}; // feed-forward coefficients

    Annotated<ConvolutionDomain, "mode", Doc<"AUTO only: which form to run, or AUTO again to choose by tap count; ignored when the form is fixed at compile time">, Visible> mode              = ConvolutionDomain::Auto;
    Annotated<gr::Size_t, "outputs_per_frame", Doc<"frequency domain only: samples produced per transform; ignored in the time domain">, Limits<1UZ, 1048576UZ>>             outputs_per_frame = 256U;

    /// the `b.size() - 1` samples preceding this span, and the frame the transform form builds from them. Reflected
    /// so the pmr-field migration re-seats them onto device memory, underscored so they stay off the settings
    /// surface. The lead-in is what makes the first window full, so neither form loses the head of the stream.
    mutable Tensor<T> _leadIn{};
    mutable Tensor<T> _frame{};

    GR_MAKE_REFLECTABLE(fir_filter, in, out, b, mode, outputs_per_frame, _leadIn, _frame);

    /// the transform's state, and nothing at all for a time-domain-only instantiation to carry
    using TapSpectrum     = std::conditional_t<form == ConvolutionDomain::Time, std::monostate, std::vector<typename Convolution::Complex>>;
    using DeviceFft       = std::conditional_t<form == ConvolutionDomain::Time, std::monostate, gr::device::SyclFFT>;
    using HostConvolution = std::conditional_t<form == ConvolutionDomain::Time, std::monostate, Convolution>;
    TapSpectrum             _tapSpectrum;
    DeviceFft               _syclFft;
    mutable HostConvolution _convolution; // holds the transform's plan and buffers across frames

    [[nodiscard]] bool runsByTransform() const {
        if constexpr (form == ConvolutionDomain::Frequency) {
            return true;
        } else if constexpr (form == ConvolutionDomain::Time) {
            return false;
        } else {
            if (gr::ComputeDomain::parse(this->compute_domain.value).isDevice()) {
                return true; // only the transform has a kernel on this instantiation
            }
            switch (mode.value) {
            case ConvolutionDomain::Time: return false;
            case ConvolutionDomain::Frequency: return true;
            case ConvolutionDomain::Auto: break;
            }
            return b.size() >= kFrequencyDomainFromTaps;
        }
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const std::size_t nTaps = std::max(std::size_t{1}, b.size());

        if constexpr (form == ConvolutionDomain::Auto) {
            if (mode.value == ConvolutionDomain::Time && gr::ComputeDomain::parse(this->compute_domain.value).isDevice()) {
                this->emitErrorMessage("fir_filter::settingsChanged()", //
                    gr::Error(std::format("mode 'TIME_DOMAIN' is not available from the AUTO form on compute_domain '{}': instantiate fir_filter<T, ConvolutionDomain::Time>, which keeps the framework's window dispatch", this->compute_domain.value)));
            }
        }

        _leadIn = Tensor<T>(std::vector<T>(nTaps - 1UZ, T{0}));

        if (!runsByTransform()) {
            this->input_chunk_size  = 1U; // 1:1 over the stream; the carried lead-in is what completes the first window
            this->output_chunk_size = 1U;
            this->stride            = 0U;
            return;
        }
        if constexpr (form != ConvolutionDomain::Time) {
            const std::size_t frameSize = Convolution::frameSizeFor(nTaps, static_cast<std::size_t>(outputs_per_frame));
            const std::size_t nOutputs  = Convolution::outputsPerFrame(frameSize, nTaps);

            _tapSpectrum = Convolution::transformTaps(std::span<const T>{b.data(), b.size()}, frameSize);
            _frame       = Tensor<T>(std::vector<T>(frameSize, T{0}));

            // 1:1 as well: a frame is built from the carried lead-in plus this chunk, which is precisely the
            // overlap overlap-save discards, so the block consumes and publishes the same count
            this->input_chunk_size  = static_cast<gr::Size_t>(nOutputs);
            this->output_chunk_size = static_cast<gr::Size_t>(nOutputs);
            this->stride            = 0U;
        }
    }

    /// the tap form, and the only body a TIME_DOMAIN instantiation has: plain spans, so the framework runs the
    /// declared window per work item and a device spreads them without this block writing a kernel
    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<T> output) const noexcept
    requires(form == ConvolutionDomain::Time)
    {
        convolveTaps(input, output);
        return gr::work::Status::OK;
    }

    /// AUTO and FREQUENCY_DOMAIN: one body over the span, taking whichever route settingsChanged declared
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output)
    requires(form != ConvolutionDomain::Time)
    {
        if (!runsByTransform()) {
            convolveTaps(std::span<const T>{input.data(), input.size()}, std::span<T>{output.data(), output.size()});
            return gr::work::Status::OK;
        }
        if constexpr (form != ConvolutionDomain::Time) {
            // each frame is `[carried lead-in ++ this chunk]`, which is exactly the overlap overlap-save discards,
            // so the block consumes and publishes the same count and agrees with the tap form sample for sample
            const std::size_t lead      = std::max(std::size_t{1}, b.size()) - 1UZ;
            const std::size_t frameSize = _frame.size();
            if (frameSize > lead) {
                const std::size_t perFrame = frameSize - lead;
                for (std::size_t frame = 0UZ; frame < std::min(input.size(), output.size()) / perFrame; ++frame) {
                    std::ranges::copy(std::span<const T>{_leadIn.data(), lead}, _frame.data());
                    std::ranges::copy(std::span<const T>{input.data() + frame * perFrame, perFrame}, _frame.data() + lead);
                    _convolution.convolveFrame(std::span<const T>{_frame.data(), frameSize}, std::span<T>{output.data() + frame * perFrame, perFrame});
                    std::ranges::copy(std::span<const T>{_frame.data() + perFrame, lead}, _leadIn.data());
                }
            }
        }
        return gr::work::Status::OK;
    }

    /// the convolution, as the algorithm layer computes it: `convolveStreaming` carries the lead-in this block
    /// keeps, so the block is 1:1 over the stream and needs to declare no window of its own.
    void convolveTaps(std::span<const T> input, std::span<T> output) const noexcept { Convolve::convolveStreaming(input, std::span<const T>{b.data(), b.size()}, std::span<T>{_leadIn.data(), _leadIn.size()}, output); }

    /// the transform on a device: every frame in the span is transformed in one batch, so what costs is the batch
    /// and not the frame. The tap spectrum is the same numbers on both sides; only who evaluates it differs.
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output)
    requires(form != ConvolutionDomain::Time and std::same_as<T, float>) // gr::device::SyclFFT is a float-only tier
    {
        using Complex = gr::device::SyclFFT::C;

        const gr::WindowGeometry frames = gr::windowGeometry(*this, input.size(), output.size());
        if (frames.nWindows == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        // the transform length is the frame the taps were transformed into, not the chunk the framework hands over:
        // a frame is `[carried lead-in ++ this chunk]`, so the block stays 1:1 with the tap form
        const std::size_t frameSize = _frame.size();
        const std::size_t lead      = std::max(std::size_t{1}, b.size()) - 1UZ;
        if (frameSize <= lead || _leadIn.size() != lead) {
            return processBulk(input, output);
        }
        const std::size_t nOutputs = frameSize - lead;
        const std::size_t nFrames  = std::min(input.size(), output.size()) / nOutputs;
        if (nFrames == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const std::size_t nBins    = nFrames * frameSize;
        const std::size_t nResults = nFrames * nOutputs;

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

        const T* samples     = input.data();
        const T* leadInRead  = _leadIn.data();
        T* const leadInWrite = _leadIn.data();
        gr::device::parallelFor(ctx, nBins, [spectra, samples, leadInRead, frameSize, nOutputs, lead](std::size_t i) {
            const std::size_t position = (i / frameSize) * nOutputs + i % frameSize; // measured from the lead-in's start
            spectra[i]                 = Complex{position < lead ? leadInRead[position] : samples[position - lead], 0.f};
        });
        _syclFft.forwardBatch(ctx, std::span<Complex>{spectra, nBins}, frameSize);
        gr::device::parallelFor(ctx, nBins, [spectra, filterBins, frameSize](std::size_t i) {
            const Complex signal = spectra[i];
            const Complex filter = filterBins[i % frameSize];
            spectra[i]           = Complex{signal.re * filter.re - signal.im * filter.im, signal.re * filter.im + signal.im * filter.re};
        });
        _syclFft.inverseBatch(ctx, std::span<Complex>{spectra, nBins}, frameSize);

        T*                results = output.data();
        const std::size_t discard = frameSize - nOutputs; // the wrap-around the saved tail exists to make right
        gr::device::parallelFor(ctx, nResults, [results, spectra, frameSize, nOutputs, discard](std::size_t i) { results[i] = spectra[(i / nOutputs) * frameSize + discard + i % nOutputs].re; });
        // carry this span's tail for the next call's first frame. nResults >= lead holds whenever a whole frame ran,
        // so every work item reads `samples` only and none races another's write.
        gr::device::parallelFor(ctx, lead, [leadInWrite, samples, lead, nResults](std::size_t k) { leadInWrite[k] = samples[nResults + k - lead]; });
        ctx.wait();

        ctx.deallocate(frameSpectra);
        ctx.deallocate(tapSpectrum);

        std::ignore = input.consume(nResults);
        output.publish(nResults);
        return gr::work::Status::OK;
    }
};

namespace detail {
using firAutoF32 = fir_filter<float, ConvolutionDomain::Auto>;
using firAutoF64 = fir_filter<double, ConvolutionDomain::Auto>;
} // namespace detail

GR_REGISTER_BLOCK("gr::filter::fir_filter<float32>", gr::filter::detail::firAutoF32)
GR_REGISTER_BLOCK("gr::filter::fir_filter<float64>", gr::filter::detail::firAutoF64)
GR_REGISTER_BLOCK(gr::filter::fir_filter, ([T], gr::filter::ConvolutionDomain::Time), [ float, double ])
GR_REGISTER_BLOCK(gr::filter::fir_filter, ([T], gr::filter::ConvolutionDomain::Frequency), [ float, double ])

GR_REGISTER_BLOCK(gr::filter::iir_filter, ([T], gr::filter::IIRForm::DF_I), [ float, double ])
GR_REGISTER_BLOCK(gr::filter::iir_filter, ([T], gr::filter::IIRForm::DF_II), [ float, double ])
GR_REGISTER_BLOCK(gr::filter::iir_filter, ([T], gr::filter::IIRForm::DF_I_TRANSPOSED), [ float, double ])
GR_REGISTER_BLOCK(gr::filter::iir_filter, ([T], gr::filter::IIRForm::DF_II_TRANSPOSED), [ float, double ])

template<typename T, IIRForm form = std::is_floating_point_v<T> ? IIRForm::DF_II : IIRForm::DF_I>
requires std::floating_point<T>
struct iir_filter : Block<iir_filter<T, form>> {
    using Description = Doc<R""(
@brief Infinite Impulse Response (IIR) filter class

b are the feed-forward coefficients (N.B. b[0] denoting the newest and b[-1] the previous sample)
a are the feedback coefficients

The recursion makes this one work item over the whole span rather than one per sample, so it is not faster on a
device than on the host. It runs there so that a cascade does not have to leave the device around it: a low-pass
with a very low cut-off costs thousands of FIR taps and fewer than eight IIR coefficients, and paying a
device-to-host-to-device round trip at that hop costs far more than the filter itself.
)"">;
    using Recursion   = gr::algorithm::filter::Iir<T, form>;

    PortIn<T>  in;
    PortOut<T> out;
    Tensor<T>  b{1}; // feed-forward coefficients
    Tensor<T>  a{1}; // feedback coefficients

    /// the recursion's accumulators, as long as the coefficients require. Reflected so that the framework re-seats it
    /// onto the device resource -- host and mirror then address one buffer, so a dispatch continues where the last one
    /// left off. The leading underscore keeps it off the settings surface.
    mutable Tensor<T> _state{};

    GR_MAKE_REFLECTABLE(iir_filter, in, out, b, a, _state);

    /// a kernel cannot allocate, so the accumulators must exist before the first dispatch -- whichever route the
    /// coefficients took to get here
    void start() { ensureState(); }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) const {
        // the accumulators are sized before the first dispatch and never here: reassigning a reflected pmr member
        // during processing frees the storage a device mirror is already pointing at, without bumping the epoch that
        // would have told the mirror to follow
        const std::size_t nSamples = std::min(input.size(), output.size());
        Recursion::filter(std::span<const T>{input.data(), nSamples}, std::span<const T>{b.data(), b.size()}, std::span<const T>{a.data(), a.size()}, //
            std::span<T>{_state.data(), _state.size()}, std::span<T>{output.data(), nSamples});
        std::ignore = input.consume(nSamples);
        output.publish(nSamples);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (!newSettings.contains("b") && !newSettings.contains("a")) {
            return;
        }
        ensureState();
    }

    /// the accumulators follow the coefficients, and carrying old ones over would filter with a history whose samples
    /// were weighted by a response no longer in effect. Sizing them here rather than only when the settings change
    /// keeps a block whose coefficients were assigned directly -- as a test or a benchmark does -- filtering correctly.
    void ensureState() const {
        const std::size_t required = Recursion::stateSize(b.size(), a.size() > 0UZ ? a.size() - 1UZ : 0UZ);
        if (_state.size() != required) {
            _state = Tensor<T>(std::vector<T>(required, T{0}));
        }
    }

    /// one sample of the recursion, for a caller holding a sample rather than a span
    [[nodiscard]] T filterOne(T input) const {
        ensureState();
        const std::size_t  nState    = _state.size() / Recursion::kHistories;
        const std::span<T> primary   = std::span<T>{_state.data(), nState};
        const std::span<T> secondary = Recursion::kHistories == 2UZ ? std::span<T>{_state.data() + nState, nState} : primary;
        return Recursion::step(input, std::span<const T>{b.data(), b.size()}, std::span<const T>{a.data(), a.size()}, primary, secondary);
    }
};

GR_REGISTER_BLOCK(gr::filter::BasicFilter, ([T]), [ double, float, gr::UncertainValue<float>, gr::UncertainValue<double> ])
GR_REGISTER_BLOCK(gr::filter::BasicFilterProto, ([T], gr::Resampling<1UZ, 1UZ, false>), [ double, float, gr::UncertainValue<float>, gr::UncertainValue<double> ])

using FilterType = gr::algorithm::filter::FilterType;
enum class CoefficientSource { Designed, Manual };

template<typename T, typename... Args>
requires(std::floating_point<T> or std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>)
struct BasicFilterProto : Block<BasicFilterProto<T, Args...>, Args...> {
    using TParent     = Block<BasicFilterProto<T, Args...>, Args...>;
    using Description = Doc<R""(@brief Basic Digital Filter class supporting FIR and IIR filters

This block implements a digital filter which can be configured as either FIR or IIR,
with selectable filter type (low-pass, high-pass, band-pass, band-stop), and supports resampling.

Two further axes: the coefficients are either designed from the response above or given directly as 'b' and 'a',
and the convolution is evaluated either a tap at a time or by transform. The transform is FIR only -- an
overlap-save frame has nowhere to put feedback -- and it pays once the filter is long enough that a transform per
frame beats a tap per sample.

A designed IIR arrives as a cascade of sections, and it is carried as one: 'b' and 'a' hold the sections one after
another, 'outputs_per_frame' apart. Designing writes both, so switching to 'Manual' means stating both -- a
feed-forward filter whose 'a' still holds a designed denominator is an IIR, not the FIR that was meant.
Multiplying the sections into a single direct form would be exact in exact arithmetic and unusable in floating
point -- the designer refuses to emit more than a biquad for 'float' precisely because a high-order direct form is
ill-conditioned, and folding them back together would undo that. The rows are what runs, on the host and on a
device alike.

A device runs the tap-a-time evaluation correctly but not quickly: a cascade carries state from one sample to the
next, so the span is one work item and an accelerator has nothing to spread across its lanes. For a tap-domain FIR
on a device name 'fir_filter<T, ConvolutionDomain::Time>', which declares one output per window and so hands the
framework as many independent items as the span allows. The transform domain is the one worth running here on a
device.
)"">;
    using ValueType   = meta::fundamental_base_value_type_t<T>;
    /// only the uncertainty path needs a section object; the plain path evaluates the rows below, so it must not
    /// carry the three host vectors a cascade brings with it
    using FilterImpl = std::conditional_t<UncertainValueLike<T>, filter::ErrorPropagatingFilter<T>, std::monostate>;

    /// the transform carries one sample type, so an uncertainty-propagating value cannot take that route
    static constexpr bool kCanTransform = std::floating_point<ValueType> and std::same_as<T, ValueType> and not TParent::StrideControl::kIsConst;
    /// the cascade the non-uncertainty path runs; an uncertainty-propagating value goes through `_filter` instead
    using Cascade = gr::algorithm::filter::Cascade<ValueType>;
    using Fir     = gr::algorithm::filter::Fir<ValueType>;

    PortIn<T>  in;
    PortOut<T> out;

    // Public settings
    Annotated<FilterType, "filter_type", Doc<"Filter type ('FIR' or 'IIR')">, Visible>                                                         filter_type     = FilterType::IIR;
    Annotated<filter::Type, "filter_response", Doc<"Filter response ('LOWPASS', 'HIGHPASS', 'BANDPASS', 'BANDSTOP')">, Visible>                filter_response = filter::Type::LOWPASS;
    Annotated<gr::Size_t, "filter_order", Doc<"Filter order">>                                                                                 filter_order{3};
    Annotated<float, "f_low", Doc<"Low cutoff frequency in Hz">, Visible>                                                                      f_low{0.1f};
    Annotated<float, "f_high", Doc<"High cutoff frequency in Hz (only for BANDPASS/BANDSTOP)">, Visible>                                       f_high{0.2f};
    Annotated<float, "sample rate", Doc<"Sample rate in Hz">, Visible>                                                                         sample_rate{1.0f};
    Annotated<gr::Size_t, "decimation factor", Doc<"1: none, i.e. preserving the relationship: N_out = N_in/decimate">>                        decimate{1U};
    Annotated<filter::iir::Design, "iir_design_method", Doc<"IIR Filter design method ('BUTTERWORTH', 'BESSEL', 'CHEBYSHEV1', 'CHEBYSHEV2')">> iir_design_method = filter::iir::Design::BUTTERWORTH;
    Annotated<algorithm::window::Type, "fir_design_method", Doc<"FIR Filter design method ('None', 'Rectangular', 'Hamming', 'Hann', 'HannExp', 'Blackman', 'Nuttall', 'BlackmanHarris', 'BlackmanNuttall', 'FlatTop', 'Exponential', 'Kaiser')">> //
                                                                                                                                                                               fir_design_method  = algorithm::window::Type::Kaiser;
    Annotated<ConvolutionDomain, "filter_domain", Doc<"where the convolution is evaluated: 'Time' a tap per sample, 'Frequency' one transform per frame (FIR only)">, Visible> filter_domain      = ConvolutionDomain::Time;
    Annotated<CoefficientSource, "coefficient_source", Doc<"where the coefficients come from: 'Designed' from the response above, 'Manual' from 'b' and 'a'">, Visible>        coefficient_source = CoefficientSource::Designed;
    Tensor<ValueType>                                                                                                                                                          b{ValueType{1}}; // feed-forward coefficients, one section per row once designed
    Tensor<ValueType>                                                                                                                                                          a{ValueType{1}}; // feedback coefficients, ditto; a[.,0] normalises its section
    Annotated<gr::Size_t, "outputs_per_frame", Doc<"frequency domain: samples produced per transform">, Limits<1UZ, 1048576UZ>>                                                outputs_per_frame = 256U;
    /// bumped whenever the rows change: a device mirror copies reflected members only, so this is what tells a
    /// kernel that the state it is holding belongs to coefficients that are no longer in effect. The leading
    /// underscore keeps it off the settings surface.
    gr::Size_t _design_epoch = 0U;
    /// how many coefficients one section occupies in 'b' and 'a': the sections are stored one after another, so a
    /// rank-1 tensor carries a cascade without the kernel needing to reason about a shape
    gr::Size_t _section_stride = 1U;
    /// one transposed-direct-form-II accumulator per state of each section, as long as the design requires. Reflected
    /// so that the framework re-seats it onto the device resource -- a kernel then reads the same buffer the host wrote
    /// and a dispatch continues where the last one left off.
    mutable Tensor<ValueType> _state{};

    GR_MAKE_REFLECTABLE(BasicFilterProto, in, out, filter_type, filter_response, filter_order, f_low, f_high, sample_rate, decimate, iir_design_method, fir_design_method, filter_domain, coefficient_source, b, a, outputs_per_frame, _design_epoch, _section_stride, _state);

    FilterImpl                           _filter;      // uncertainty path only
    std::vector<std::complex<ValueType>> _tapSpectrum; // frequency domain, host only: the taps transformed once
    /// frequency domain, host only: the transform's plan and buffers, held across frames rather than rebuilt per frame
    mutable gr::algorithm::filter::FastConvolution<ValueType> _convolution;
    mutable gr::Size_t                                        _stateEpoch = 0U;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) { designFilter(); }

    void designFilter() {
        using namespace gr::filter;

        FilterParameters params;
        params.order = filter_order;
        params.fLow  = static_cast<double>(f_low);
        params.fHigh = static_cast<double>(f_high);
        params.fs    = static_cast<double>(sample_rate);

        if (filter_domain == ConvolutionDomain::Frequency) {
            designTransformedFilter(params);
            return;
        }

        if constexpr (not TParent::ResamplingControl::kIsConst) {
            this->input_chunk_size  = std::max(gr::Size_t{1}, decimate.value);
            this->output_chunk_size = 1U;
        }
        if constexpr (not TParent::StrideControl::kIsConst) {
            this->stride = 0U; // back-to-back
        }

        if constexpr (UncertainValueLike<T>) { // the uncertainty path keeps the cascade object it propagates through
            if (coefficient_source == CoefficientSource::Manual) {
                _filter = FilterImpl(FilterCoefficients<ValueType>{.b = {b.begin(), b.end()}, .a = {a.begin(), a.end()}});
            } else if (filter_type == FilterType::FIR) {
                _filter = FilterImpl(fir::designFilter<ValueType>(filter_response, params, fir_design_method));
            } else {
                _filter = FilterImpl(iir::designFilter<ValueType>(filter_response, params, iir_design_method));
            }
            return;
        } else {
            if (coefficient_source == CoefficientSource::Manual) {
                adoptRows(); // 'b' and 'a' are the sections the user gave
            } else if (filter_type == FilterType::FIR) {
                packSections(std::array{fir::designFilter<ValueType>(filter_response, params, fir_design_method)});
            } else {
                packSections(iir::designFilter<ValueType>(filter_response, params, iir_design_method));
            }
        }
    }

    /// one row per section, shorter sections zero-padded: a trailing zero coefficient multiplies a state slot that
    /// then contributes nothing, so the padding is exact rather than merely harmless
    void packSections(const std::ranges::range auto& sections) {
        const std::size_t nSections = std::ranges::size(sections);
        std::size_t       nCoeffs   = 1UZ;
        for (const auto& section : sections) {
            nCoeffs = std::max(nCoeffs, std::max(section.b.size(), section.a.size()));
        }
        resetRows(nSections, nCoeffs);
        std::size_t row = 0UZ;
        for (const auto& section : sections) {
            ValueType* bRow = b.data() + row * nCoeffs;
            ValueType* aRow = a.data() + row * nCoeffs;
            std::ranges::copy(section.b, bRow);
            std::ranges::copy(section.a, aRow);
            if (section.a.empty()) {
                aRow[0] = ValueType{1};
            }
            ++row;
        }
        ++_design_epoch;
    }

    /// the user supplied one section: bring 'b' and 'a' to a common length before anything reads them, because a
    /// feed-forward-only filter is given a single 'a' and the recursion still indexes one slot per state
    void adoptRows() {
        // coefficients given by hand are one section: a cascade is what the designer produces, and a plain
        // 'b'/'a' pair has no way to say where one section ends and the next begins
        const std::size_t nSections = 1UZ;
        const std::size_t bCoeffs   = b.size();
        const std::size_t aCoeffs   = a.size();
        const std::size_t nCoeffs   = std::max({bCoeffs, aCoeffs, std::size_t{1}});

        const std::vector<ValueType> givenB(b.begin(), b.end());
        const std::vector<ValueType> givenA(a.begin(), a.end());
        resetRows(nSections, nCoeffs);
        for (std::size_t row = 0UZ; row < nSections; ++row) {
            for (std::size_t j = 0UZ; j < bCoeffs && row * bCoeffs + j < givenB.size(); ++j) {
                b.data()[row * nCoeffs + j] = givenB[row * bCoeffs + j];
            }
            for (std::size_t j = 0UZ; j < aCoeffs && row * aCoeffs + j < givenA.size(); ++j) {
                a.data()[row * nCoeffs + j] = givenA[row * aCoeffs + j];
            }
            if (a.data()[row * nCoeffs] == ValueType{}) { // a section without a stated denominator is feed-forward only
                a.data()[row * nCoeffs] = ValueType{1};
            }
        }
        ++_design_epoch;
    }

    /// every row the same length, zeroed: a padded coefficient multiplies a state slot that then contributes
    /// nothing, which is what makes ragged sections representable as a rectangle
    void resetRows(std::size_t nSections, std::size_t nCoeffs) {
        b.resize({nSections * nCoeffs}, ValueType{});
        a.resize({nSections * nCoeffs}, ValueType{});
        _state.resize({std::max<std::size_t>(1UZ, nSections * (nCoeffs > 0UZ ? nCoeffs - 1UZ : 0UZ))}, ValueType{});
        std::ranges::fill(std::span<ValueType>{_state.data(), _state.size()}, ValueType{});
        std::ranges::fill(std::span<ValueType>{b.data(), nSections * nCoeffs}, ValueType{});
        std::ranges::fill(std::span<ValueType>{a.data(), nSections * nCoeffs}, ValueType{});
        _section_stride = static_cast<gr::Size_t>(nCoeffs);
    }

    /// one section whose feedback is a bare normalisation: what a designed FIR is, and what a convolution computes
    /// directly instead of recursively
    [[nodiscard]] bool runsAsConvolution() const noexcept {
        if (sectionCount() != 1UZ) {
            return false;
        }
        const std::size_t nCoeffs = coefficientsPerSection();
        for (std::size_t k = 1UZ; k < nCoeffs; ++k) {
            if (a.data()[k] != ValueType{0}) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::size_t coefficientsPerSection() const noexcept { return std::max<std::size_t>(1UZ, static_cast<std::size_t>(_section_stride)); }
    [[nodiscard]] std::size_t sectionCount() const noexcept { return std::max<std::size_t>(1UZ, b.size() / coefficientsPerSection()); }

    void designTransformedFilter(const filter::FilterParameters& params) {
        if constexpr (kCanTransform) {
            using Convolution = gr::algorithm::filter::FastConvolution<ValueType>;

            if (filter_type == FilterType::IIR) {
                this->emitErrorMessage("BasicFilter::settingsChanged()", gr::Error("an overlap-save frame has nowhere to put feedback: filter_domain 'Frequency' is FIR only"));
                return;
            }
            // the tap spectrum is a host vector, and a kernel that reached it would read a host pointer: valid on
            // 'host:sycl', a fault or a hang on a real device. Refuse the domain rather than find out there.
            if (gr::ComputeDomain::parse(this->compute_domain.value).isDevice()) {
                this->emitErrorMessage("BasicFilter::settingsChanged()", gr::Error(std::format("filter_domain 'Frequency' keeps its tap spectrum on the host, so it cannot run on '{}'", this->compute_domain.value)));
                return;
            }

            const std::vector<ValueType> taps = (coefficient_source == CoefficientSource::Manual) //
                                                    ? std::vector<ValueType>{b.begin(), b.end()}
                                                    : gr::filter::fir::designFilter<ValueType>(filter_response, params, fir_design_method).b;

            const std::size_t frameSize = Convolution::frameSizeFor(taps.size(), static_cast<std::size_t>(outputs_per_frame));
            const std::size_t nOutputs  = Convolution::outputsPerFrame(frameSize, taps.size());

            _tapSpectrum = Convolution::transformTaps(taps, frameSize);

            this->input_chunk_size  = static_cast<gr::Size_t>(frameSize);
            this->output_chunk_size = static_cast<gr::Size_t>(nOutputs);
            this->stride            = static_cast<gr::Size_t>(nOutputs);
        } else {
            std::ignore = params;
            this->emitErrorMessage("BasicFilter::settingsChanged()", gr::Error("filter_domain 'Frequency' needs a plain floating-point sample type and a settable stride"));
        }
    }

    /// one sample through the cascade, for a caller holding a sample rather than a span
    [[nodiscard]] T filterOne(T input) const noexcept
    requires(not UncertainValueLike<T>)
    {
        return Cascade::step(input, std::span<const T>{b.data(), b.size()}, std::span<const T>{a.data(), a.size()}, coefficientsPerSection(), std::span<ValueType>{_state.data(), _state.size()});
    }

    [[nodiscard]] T filterOne(T input) noexcept
    requires(UncertainValueLike<T>)
    {
        return _filter.processOne(input);
    }

    /// the plain path: one work item over the span, so it runs wherever the block is sent. The view form is not
    /// offered on purpose -- a declared window would have the framework run the recursion as parallel windows over
    /// one shared state, which is not what a cascade computes.
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) const
    requires(not UncertainValueLike<T>)
    {
        // the transform allocates and it throws, so it must not be compiled into a kernel at all: a device
        // instantiation of this body leaves it out rather than trusting the setting to have been refused. Relying
        // on the refusal alone pulls operator delete, __cxa_throw and the FFT's heap into the device module.
        if constexpr (not std::same_as<std::remove_cvref_t<decltype(input)>, gr::device::DeviceInputSpan<T>>) {
            if (filter_domain == ConvolutionDomain::Frequency) {
                return convolveFrames(input, output);
            }
        }
        return filterSamples(input, output);
    }

    /// the cascade, one work item over the span
    [[nodiscard]] gr::work::Status filterSamples(InputSpanLike auto& input, OutputSpanLike auto& output) const {
        if (_stateEpoch != _design_epoch) { // the coefficients changed under the state a previous dispatch left
            std::ranges::fill(std::span<ValueType>{_state.data(), _state.size()}, ValueType{});
            _stateEpoch = _design_epoch;
        }

        const std::size_t decim = std::max(std::size_t{1}, static_cast<std::size_t>(decimate));
        const std::size_t nOut  = std::min(input.size() / decim, output.size());
        const std::size_t nIn   = nOut * decim;

        // a section with no feedback is a convolution, and computing it as one costs a multiply per tap instead of the
        // two and a state write the recursion needs -- and it accumulates along the output axis, which the recursion
        // cannot. The slots then hold the input lead-in rather than accumulators; both are as long as the section and
        // both are zeroed when the design changes, which is what makes one storage serve either meaning.
        // the regime follows the DESIGN, never the span length: the two write `_state` with incompatible meanings --
        // raw input history here, transposed-DF-II accumulators below -- so switching between calls would read one
        // as the other and corrupt `b.size() - 1` samples at every crossing
        if (decim == 1UZ && runsAsConvolution()) {
            Fir::convolveStreaming(std::span<const ValueType>{input.data(), nIn}, std::span<const ValueType>{b.data(), b.size()}, //
                std::span<ValueType>{_state.data(), _state.size()}, std::span<ValueType>{output.data(), nOut});
            std::ignore = input.consume(nIn);
            output.publish(nOut);
            return nOut == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
        }

        Cascade::filter(std::span<const T>{input.data(), nIn}, std::span<const T>{b.data(), b.size()}, std::span<const T>{a.data(), a.size()}, coefficientsPerSection(), //
            std::span<ValueType>{_state.data(), _state.size()}, std::span<ValueType>{output.data(), nOut}, decim);
        std::ignore = input.consume(nIn);
        output.publish(nOut);
        return nOut == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
    }

    /// the uncertainty path stays on the host: it propagates through per-section auto-correlations, which is not a
    /// kernel's work. Non-const on purpose, so the dispatcher refuses it by name rather than mis-running it.
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output)
    requires(UncertainValueLike<T>)
    {
        const std::size_t decim    = std::max(std::size_t{1}, static_cast<std::size_t>(decimate));
        const std::size_t nOut     = std::min(input.size() / decim, output.size());
        const std::size_t nIn      = nOut * decim;
        std::size_t       outIndex = 0UZ;
        for (std::size_t i = 0UZ; i < nIn; ++i) {
            const T filtered = filterOne(input[i]);
            if (i % decim == 0UZ) {
                output[outIndex++] = filtered;
            }
        }
        std::ignore = input.consume(nIn);
        output.publish(nOut);
        return nOut == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
    }

    [[nodiscard]] gr::work::Status convolveFrames(InputSpanLike auto& input, OutputSpanLike auto& output) const {
        if constexpr (kCanTransform) {
            const gr::WindowGeometry frames = gr::windowGeometry(*this, input.size(), output.size());
            for (std::size_t frame = 0UZ; frame < frames.nWindows; ++frame) {
                _convolution.convolveFrame(std::span<const T>{input.data() + frame * frames.hop, frames.inChunk}, std::span<T>{output.data() + frame * frames.outChunk, frames.outChunk});
            }
            std::ignore = input.consume(frames.nWindows * frames.hop);
            output.publish(frames.nWindows * frames.outChunk);
            return frames.nWindows == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
        } else {
            std::ignore = input;
            std::ignore = output;
            return gr::work::Status::ERROR;
        }
    }
};

template<typename T>
using BasicFilter = BasicFilterProto<T, Resampling<>, Stride<>>;

template<typename T>
using BasicDecimatingFilter = BasicFilterProto<T, Resampling<1UZ, 1UZ, false>>;

GR_REGISTER_BLOCK(gr::filter::Decimator, [T], [ uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double, std::complex<float>, std::complex<double>, gr::UncertainValue<float>, gr::UncertainValue<double> ])

template<typename T>
struct Decimator : Block<Decimator<T>, Resampling<1UZ, 1UZ, false>> {
    using TParent     = Block<Decimator<T>, Resampling<1UZ, 1UZ, false>>;
    using Description = Doc<R""(@brief Basic Decimator Block

This block implements a decimator for downsampling (dropping) input data by a
configurable factor. Filtering is not included in this implementation so expect
aliasing and sub-sampling related effects.
)"">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "decimation factor", Doc<"Factor by which to downsample/drop input data">, Visible> decim{1};

    GR_MAKE_REFLECTABLE(Decimator, in, out, decim);

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) { this->input_chunk_size = decim; }

    [[nodiscard]] work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept {
        assert(output.size() >= input.size() / decim);

        std::size_t out_sample_idx = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (i % decim == 0) {
                output[out_sample_idx++] = input[i];
            }
        }
        return work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_TIME_DOMAIN_FILTER_HPP
