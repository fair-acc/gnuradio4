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
#include <gnuradio-4.0/algorithm/filter/FastConvolution.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/algorithm/fourier/SyclFFT.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include <magic_enum.hpp>

namespace gr::filter {

using namespace gr;

/// where the convolution is evaluated. The explicit forms compile only their own path and carry only their own
/// state, which is what a target counting code size wants; AUTO carries both and chooses.
enum class IRForm {
    AUTO,             /// choose at run time: the 'mode' setting, or failing that the tap count
    TIME_DOMAIN,      /// a multiply-add per tap per sample
    FREQUENCY_DOMAIN, /// one transform per frame, by overlap-save
};

template<typename T, IRForm form = IRForm::AUTO>
requires std::floating_point<T>
struct fir_filter : Block<fir_filter<T, form>, Resampling<>, Stride<>> {
    using TParent     = Block<fir_filter<T, form>, Resampling<>, Stride<>>;
    using Convolution = gr::algorithm::filter::FastConvolution<T>;
    using Description = Doc<R""(@brief Finite Impulse Response (FIR) filter class

The transfer function of an FIR filter is given by:
H(z) = b[0] + b[1]*z^-1 + b[2]*z^-2 + ... + b[N]*z^-N

There are two ways to evaluate it and they compute the same samples: a multiply-add per tap per sample, or one
transform per frame by overlap-save. Which is cheaper depends only on how many taps there are -- measured on this
machine the two cross around 256 taps on the host, and by a few thousand taps the tap form is two orders of
magnitude behind.

'IRForm' says which one to use. TIME_DOMAIN and FREQUENCY_DOMAIN compile only their own path, for a target where
code size is the constraint. AUTO carries both and picks: the 'mode' setting when it names a form, otherwise the
tap count. On a device AUTO always takes the transform -- that is what a device is worth using for, and the tap
form there would need a kernel of its own -- so a graph wanting the tap form on a device instantiates TIME_DOMAIN,
which keeps the framework's window dispatch and is the faster of the two at short lengths.

Stated as a sliding window rather than a delay line, a window cannot answer until it is full, so the first output
corresponds to the `b.size()`-th input and the stream is shorter by `b.size() - 1`. Both forms agree on this, so
they are interchangeable sample for sample.
)"">;

    /// the measured crossover on the host, in taps: below it a tap per sample wins, above it a transform per frame.
    /// It governs only the host path -- on a device AUTO takes the transform whatever the tap count -- and it is a
    /// default rather than a claim: a machine or a backend moves it, so name the form explicitly to override it.
    static constexpr std::size_t kFrequencyDomainFromTaps = 128UZ;

    PortIn<T>  in;
    PortOut<T> out;
    Tensor<T>  b{T{1}}; // feed-forward coefficients

    Annotated<IRForm, "mode", Doc<"AUTO only: which form to run, or AUTO again to choose by tap count; ignored when the form is fixed at compile time">, Visible> mode              = IRForm::AUTO;
    Annotated<gr::Size_t, "outputs_per_frame", Doc<"frequency domain only: samples produced per transform; ignored in the time domain">, Limits<1UZ, 1048576UZ>>  outputs_per_frame = 256U;

    GR_MAKE_REFLECTABLE(fir_filter, in, out, b, mode, outputs_per_frame);

    /// the transform's state, and nothing at all for a time-domain-only instantiation to carry
    using TapSpectrum     = std::conditional_t<form == IRForm::TIME_DOMAIN, std::monostate, std::vector<typename Convolution::Complex>>;
    using DeviceFft       = std::conditional_t<form == IRForm::TIME_DOMAIN, std::monostate, gr::device::SyclFFT>;
    using HostConvolution = std::conditional_t<form == IRForm::TIME_DOMAIN, std::monostate, Convolution>;
    TapSpectrum             _tapSpectrum;
    DeviceFft               _syclFft;
    mutable HostConvolution _convolution; // holds the transform's plan and buffers across frames

    [[nodiscard]] bool runsByTransform() const {
        if constexpr (form == IRForm::FREQUENCY_DOMAIN) {
            return true;
        } else if constexpr (form == IRForm::TIME_DOMAIN) {
            return false;
        } else {
            if (gr::ComputeDomain::parse(this->compute_domain.value).isDevice()) {
                return true; // only the transform has a kernel on this instantiation
            }
            switch (mode.value) {
            case IRForm::TIME_DOMAIN: return false;
            case IRForm::FREQUENCY_DOMAIN: return true;
            case IRForm::AUTO: break;
            }
            return b.size() >= kFrequencyDomainFromTaps;
        }
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const std::size_t nTaps = std::max(std::size_t{1}, b.size());

        if constexpr (form == IRForm::AUTO) {
            if (mode.value == IRForm::TIME_DOMAIN && gr::ComputeDomain::parse(this->compute_domain.value).isDevice()) {
                this->emitErrorMessage("fir_filter::settingsChanged()", //
                    gr::Error(std::format("mode 'TIME_DOMAIN' is not available from the AUTO form on compute_domain '{}': instantiate fir_filter<T, IRForm::TIME_DOMAIN>, which keeps the framework's window dispatch", this->compute_domain.value)));
            }
        }

        if (!runsByTransform()) {
            this->input_chunk_size  = static_cast<gr::Size_t>(nTaps);
            this->output_chunk_size = 1U;
            this->stride            = 1U;
            return;
        }
        if constexpr (form != IRForm::TIME_DOMAIN) {
            const std::size_t frameSize = Convolution::frameSizeFor(nTaps, static_cast<std::size_t>(outputs_per_frame));
            const std::size_t nOutputs  = Convolution::outputsPerFrame(frameSize, nTaps);

            _tapSpectrum = Convolution::transformTaps(std::span<const T>{b.data(), b.size()}, frameSize);

            this->input_chunk_size  = static_cast<gr::Size_t>(frameSize);
            this->output_chunk_size = static_cast<gr::Size_t>(nOutputs);
            this->stride            = static_cast<gr::Size_t>(nOutputs);
        }
    }

    /// the tap form, and the only body a TIME_DOMAIN instantiation has: plain spans, so the framework runs the
    /// declared window per work item and a device spreads them without this block writing a kernel
    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<T> output) const noexcept
    requires(form == IRForm::TIME_DOMAIN)
    {
        convolveTaps(input, output);
        return gr::work::Status::OK;
    }

    /// AUTO and FREQUENCY_DOMAIN: one body over the span, taking whichever route settingsChanged declared
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output)
    requires(form != IRForm::TIME_DOMAIN)
    {
        if (!runsByTransform()) {
            convolveTaps(std::span<const T>{input.data(), input.size()}, std::span<T>{output.data(), output.size()});
            return gr::work::Status::OK;
        }
        const gr::WindowGeometry frames = gr::windowGeometry(*this, input.size(), output.size());
        for (std::size_t frame = 0UZ; frame < frames.nWindows; ++frame) {
            _convolution.convolveFrame(std::span<const T>{input.data() + frame * frames.hop, frames.inChunk}, _tapSpectrum, std::span<T>{output.data() + frame * frames.outChunk, frames.outChunk});
        }
        return gr::work::Status::OK;
    }

    /// one inner product per output: the taps are walked backwards and the window forwards, which leaves the
    /// vectorising to the standard library rather than to a hand-rolled loop it may not reassociate
    void convolveTaps(std::span<const T> input, std::span<T> output) const noexcept {
        const std::size_t nTaps          = std::max(std::size_t{1}, b.size());
        const auto        newestTapFirst = std::make_reverse_iterator(b.cend());
        const auto        oldestTapLast  = std::make_reverse_iterator(b.cbegin());
        const std::size_t nOut           = input.size() + 1UZ >= nTaps ? std::min(output.size(), input.size() + 1UZ - nTaps) : 0UZ;
        for (std::size_t n = 0UZ; n < nOut; ++n) {
            output[n] = std::transform_reduce(std::execution::unseq, newestTapFirst, oldestTapLast, input.data() + n, T{0}, std::plus<>{}, std::multiplies<>{});
        }
    }

    /// the transform on a device: every frame in the span is transformed in one batch, so what costs is the batch
    /// and not the frame. The tap spectrum is the same numbers on both sides; only who evaluates it differs.
    [[nodiscard]] gr::work::Status processBulkDevice(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output)
    requires(form != IRForm::TIME_DOMAIN and std::same_as<T, float>) // gr::device::SyclFFT is a float-only tier
    {
        using Complex = gr::device::SyclFFT::C;

        const gr::WindowGeometry frames = gr::windowGeometry(*this, input.size(), output.size());
        if (frames.nWindows == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const std::size_t frameSize = frames.inChunk;
        const std::size_t nOutputs  = frames.outChunk;
        const std::size_t nFrames   = frames.nWindows;
        const std::size_t nBins     = nFrames * frameSize;
        const std::size_t nResults  = nFrames * nOutputs;

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

        const T* samples = input.data();
        gr::device::parallelFor(ctx, nBins, [spectra, samples, frameSize, nOutputs](std::size_t i) { spectra[i] = Complex{samples[(i / frameSize) * nOutputs + i % frameSize], 0.f}; });
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
        ctx.wait();

        ctx.deallocate(frameSpectra);
        ctx.deallocate(tapSpectrum);

        std::ignore = input.consume(nResults);
        output.publish(nResults);
        return gr::work::Status::OK;
    }
};

namespace detail {
using firAutoF32 = fir_filter<float, IRForm::AUTO>;
using firAutoF64 = fir_filter<double, IRForm::AUTO>;
} // namespace detail

GR_REGISTER_BLOCK("gr::filter::fir_filter<float32>", gr::filter::detail::firAutoF32)
GR_REGISTER_BLOCK("gr::filter::fir_filter<float64>", gr::filter::detail::firAutoF64)
GR_REGISTER_BLOCK(gr::filter::fir_filter, ([T], gr::filter::IRForm::TIME_DOMAIN), [ float, double ])
GR_REGISTER_BLOCK(gr::filter::fir_filter, ([T], gr::filter::IRForm::FREQUENCY_DOMAIN), [ float, double ])

enum class IIRForm {
    DF_I,  /// direct form I: preferred for fixed-point arithmetics (e.g. no overflow)
    DF_II, /// direct form II: preferred for floating-point arithmetics (less operations)
    DF_I_TRANSPOSED,
    DF_II_TRANSPOSED,
};

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
    /// the recursion's own memory, fixed so that it is `std::array`-backed and travels into the device mirror
    /// verbatim; a filter needing more than this is a cascade of biquads rather than one section
    static constexpr std::size_t kMaxCoefficients = 32UZ;

    PortIn<T>  in;
    PortOut<T> out;
    Tensor<T>  b{1}; // feed-forward coefficients
    Tensor<T>  a{1}; // feedback coefficients

    GR_MAKE_REFLECTABLE(iir_filter, in, out, b, a);

    // device-private: the recursion carries these between dispatches and nothing copies them back to the host
    mutable HistoryBuffer<T, kMaxCoefficients> inputHistory{};
    mutable HistoryBuffer<T, kMaxCoefficients> outputHistory{};

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) const {
        const std::size_t nSamples = std::min(input.size(), output.size());
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            output[i] = filterOne(input[i]);
        }
        std::ignore = input.consume(nSamples);
        output.publish(nSamples);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (!newSettings.contains("b") && !newSettings.contains("a")) {
            return;
        }
        if (const std::size_t required = std::max(a.size(), b.size()); required > kMaxCoefficients) {
            // an error message alone does not stop a graph, and the history would then wrap modulo its capacity and
            // filter with whatever that left behind: pass the signal through instead, so the refusal is visible
            this->emitErrorMessage("iir_filter::settingsChanged()", //
                gr::Error(std::format("{} coefficients exceed the {} this filter keeps; cascade biquad sections instead -- passing the signal through unfiltered", required, kMaxCoefficients)));
            b             = Tensor<T>{T{1}};
            a             = Tensor<T>{T{1}};
            inputHistory  = HistoryBuffer<T, kMaxCoefficients>{};
            outputHistory = HistoryBuffer<T, kMaxCoefficients>{};
        }
    }

    /// one sample of the recursion; the forms differ in where the state is kept, not in what they compute
    [[nodiscard]] T filterOne(T input) const noexcept {
        if constexpr (form == IIRForm::DF_I) {
            // y[n] = b[0] * x[n]   + b[1] * x[n-1] + ... + b[N] * x[n-N]
            //      - a[1] * y[n-1] - a[2] * y[n-2] - ... - a[M] * y[n-M]
            inputHistory.push_front(input);
            const T feedforward = std::transform_reduce(std::execution::unseq, b.cbegin(), b.cend(), inputHistory.cbegin(), T{0}, std::plus<>{}, std::multiplies<>{});
            const T feedback    = std::transform_reduce(std::execution::unseq, a.cbegin() + 1, a.cend(), outputHistory.cbegin(), T{0}, std::plus<>{}, std::multiplies<>{});
            const T output      = feedforward - feedback;
            outputHistory.push_front(output);
            return output;
        } else if constexpr (form == IIRForm::DF_II) {
            // w[n] = x[n] - a[1] * w[n-1] - a[2] * w[n-2] - ... - a[M] * w[n-M]
            // y[n] =        b[0] * w[n]   + b[1] * w[n-1] + ... + b[N] * w[n-N]
            const T w = input - std::transform_reduce(std::execution::unseq, a.cbegin() + 1, a.cend(), inputHistory.cbegin(), T{0}, std::plus<>{}, std::multiplies<>{});
            inputHistory.push_front(w);

            return std::transform_reduce(std::execution::unseq, b.cbegin(), b.cend(), inputHistory.cbegin(), T{0}, std::plus<>{}, std::multiplies<>{});
        } else if constexpr (form == IIRForm::DF_I_TRANSPOSED) {
            // w_1[n] = x[n] - a[1] * w_2[n-1] - a[2] * w_2[n-2] - ... - a[M] * w_2[n-M]
            // y[n]   = b[0] * w_2[n] + b[1] * w_2[n-1] + ... + b[N] * w_2[n-N]
            const T v0 = input - std::transform_reduce(std::execution::unseq, a.cbegin() + 1, a.cend(), outputHistory.cbegin(), T{0}, std::plus<>{}, std::multiplies<>{});
            outputHistory.push_front(v0);

            return std::transform_reduce(std::execution::unseq, b.cbegin(), b.cend(), outputHistory.cbegin(), T{0}, std::plus<>{}, std::multiplies<>{});
        } else if constexpr (form == IIRForm::DF_II_TRANSPOSED) {
            // y[n] = b_0 * f[n] + Σ (b_k * f[n−k] − a_k * y[n−k]) for k = 1 to N
            const T output = b[0] * input + std::transform_reduce(std::execution::unseq, b.cbegin() + 1, b.cend(), inputHistory.cbegin(), T{0}, std::plus<>{}, std::multiplies<>{}) - std::transform_reduce(std::execution::unseq, a.cbegin() + 1, a.cend(), outputHistory.cbegin(), T{0}, std::plus<>{}, std::multiplies<>{});

            inputHistory.push_front(input);
            outputHistory.push_front(output);
            return output;
        }
    }
};

GR_REGISTER_BLOCK(gr::filter::BasicFilter, ([T]), [ double, float, gr::UncertainValue<float>, gr::UncertainValue<double> ])
GR_REGISTER_BLOCK(gr::filter::BasicFilterProto, ([T], gr::Resampling<1UZ, 1UZ, false>), [ double, float, gr::UncertainValue<float>, gr::UncertainValue<double> ])

enum class FilterType { FIR, IIR };
enum class FilterDomain { Time, Frequency };
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
on a device name 'fir_filter<T, IRForm::TIME_DOMAIN>', which declares one output per window and so hands the
framework as many independent items as the span allows. The transform domain is the one worth running here on a
device.
)"">;
    using ValueType   = meta::fundamental_base_value_type_t<T>;
    /// only the uncertainty path needs a section object; the plain path evaluates the rows below, so it must not
    /// carry the three host vectors a cascade brings with it
    using FilterImpl = std::conditional_t<UncertainValueLike<T>, filter::ErrorPropagatingFilter<T>, std::monostate>;

    /// the transform carries one sample type, so an uncertainty-propagating value cannot take that route
    static constexpr bool kCanTransform = std::floating_point<ValueType> and std::same_as<T, ValueType> and not TParent::StrideControl::kIsConst;
    /// a cascade is biquads, so sixteen sections span an order-32 design
    static constexpr std::size_t kMaxSections = 16UZ;
    /// what a kernel carries between dispatches; a designed FIR of order 32 is 713 taps and is refused rather than
    /// silently wrapped, which is the failure an over-long coefficient set used to produce
    static constexpr std::size_t kMaxStates = 512UZ;

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
    Annotated<FilterDomain, "filter_domain", Doc<"where the convolution is evaluated: 'Time' a tap per sample, 'Frequency' one transform per frame (FIR only)">, Visible> filter_domain      = FilterDomain::Time;
    Annotated<CoefficientSource, "coefficient_source", Doc<"where the coefficients come from: 'Designed' from the response above, 'Manual' from 'b' and 'a'">, Visible>   coefficient_source = CoefficientSource::Designed;
    Tensor<ValueType>                                                                                                                                                     b{ValueType{1}}; // feed-forward coefficients, one section per row once designed
    Tensor<ValueType>                                                                                                                                                     a{ValueType{1}}; // feedback coefficients, ditto; a[.,0] normalises its section
    Annotated<gr::Size_t, "outputs_per_frame", Doc<"frequency domain: samples produced per transform">, Limits<1UZ, 1048576UZ>>                                           outputs_per_frame = 256U;
    /// bumped whenever the rows change: a device mirror copies reflected members only, so this is what tells a
    /// kernel that the state it is holding belongs to coefficients that are no longer in effect. The leading
    /// underscore keeps it off the settings surface.
    gr::Size_t _design_epoch = 0U;
    /// how many coefficients one section occupies in 'b' and 'a': the sections are stored one after another, so a
    /// rank-1 tensor carries a cascade without the kernel needing to reason about a shape
    gr::Size_t _section_stride = 1U;

    GR_MAKE_REFLECTABLE(BasicFilterProto, in, out, filter_type, filter_response, filter_order, f_low, f_high, sample_rate, decimate, iir_design_method, fir_design_method, filter_domain, coefficient_source, b, a, outputs_per_frame, _design_epoch, _section_stride);

    FilterImpl                           _filter;      // uncertainty path only
    std::vector<std::complex<ValueType>> _tapSpectrum; // frequency domain, host only: the taps transformed once
    /// frequency domain, host only: the transform's plan and buffers, held across frames rather than rebuilt per frame
    mutable gr::algorithm::filter::FastConvolution<ValueType> _convolution;
    /// device-private: one transposed-direct-form-II accumulator per state of each section, carried between
    /// dispatches and never copied back
    mutable std::array<T, kMaxStates> _state{};
    mutable gr::Size_t                _stateEpoch = 0U;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) { designFilter(); }

    void designFilter() {
        using namespace gr::filter;

        FilterParameters params;
        params.order = filter_order;
        params.fLow  = static_cast<double>(f_low);
        params.fHigh = static_cast<double>(f_high);
        params.fs    = static_cast<double>(sample_rate);

        if (filter_domain == FilterDomain::Frequency) {
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
        if (!withinBounds(nSections, nCoeffs)) {
            return;
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
        if (!withinBounds(nSections, nCoeffs)) {
            return;
        }

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
        std::ranges::fill(std::span<ValueType>{b.data(), nSections * nCoeffs}, ValueType{});
        std::ranges::fill(std::span<ValueType>{a.data(), nSections * nCoeffs}, ValueType{});
        _section_stride = static_cast<gr::Size_t>(nCoeffs);
    }

    [[nodiscard]] bool withinBounds(std::size_t nSections, std::size_t nCoeffs) {
        const std::size_t states = nSections * (nCoeffs > 0UZ ? nCoeffs - 1UZ : 0UZ);
        if (nSections <= kMaxSections && states <= kMaxStates) {
            return true;
        }
        // a passed-through filter is a visible wrong answer; keeping the oversize design would wrap the state
        // silently, which is worse than being told
        this->emitErrorMessage("BasicFilter::settingsChanged()",
            gr::Error(std::format("{} sections of {} coefficients need {} state slots, more than the {} sections or {} slots this filter carries; passing the signal through unfiltered instead", //
                nSections, nCoeffs, states, kMaxSections, kMaxStates)));
        resetRows(1UZ, 1UZ);
        b.data()[0] = ValueType{1};
        a.data()[0] = ValueType{1};
        ++_design_epoch;
        return false;
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

    /// one sample through the cascade, transposed direct form II: each section keeps one accumulator per state and
    /// the sections chain, which is what a designed cascade means
    [[nodiscard]] T filterOne(T input) const noexcept
    requires(not UncertainValueLike<T>)
    {
        const std::size_t nSections = sectionCount();
        const std::size_t nCoeffs   = coefficientsPerSection();
        const std::size_t nStates   = nCoeffs > 0UZ ? nCoeffs - 1UZ : 0UZ;

        T sample = input;
        for (std::size_t section = 0UZ; section < nSections; ++section) {
            const ValueType* bRow  = b.data() + section * nCoeffs;
            const ValueType* aRow  = a.data() + section * nCoeffs;
            T*               state = _state.data() + section * nStates;

            if (nStates == 0UZ) { // a bare gain has nothing to remember
                sample = static_cast<T>(bRow[0]) * sample;
                continue;
            }
            const T output = static_cast<T>(bRow[0]) * sample + state[0];
            for (std::size_t j = 0UZ; j + 1UZ < nStates; ++j) {
                state[j] = static_cast<T>(bRow[j + 1UZ]) * sample - static_cast<T>(aRow[j + 1UZ]) * output + state[j + 1UZ];
            }
            state[nStates - 1UZ] = static_cast<T>(bRow[nStates]) * sample - static_cast<T>(aRow[nStates]) * output;
            sample               = output;
        }
        return sample;
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
            if (filter_domain == FilterDomain::Frequency) {
                return convolveFrames(input, output);
            }
        }
        return filterSamples(input, output);
    }

    /// the cascade, one work item over the span
    [[nodiscard]] gr::work::Status filterSamples(InputSpanLike auto& input, OutputSpanLike auto& output) const {
        if (_stateEpoch != _design_epoch) { // the coefficients changed under the state a previous dispatch left
            _state.fill(T{});
            _stateEpoch = _design_epoch;
        }

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
                _convolution.convolveFrame(std::span<const T>{input.data() + frame * frames.hop, frames.inChunk}, _tapSpectrum, std::span<T>{output.data() + frame * frames.outChunk, frames.outChunk});
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
