#ifndef GNURADIO_TIME_DOMAIN_FILTER_HPP
#define GNURADIO_TIME_DOMAIN_FILTER_HPP
#include <algorithm>
#include <execution>
#include <functional>
#include <iterator>
#include <numeric>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include <magic_enum.hpp>

namespace gr::filter {

using namespace gr;

GR_REGISTER_BLOCK(gr::filter::fir_filter, [T], [ float, double ])

template<typename T>
requires std::floating_point<T>
struct fir_filter : Block<fir_filter<T>, Resampling<>, Stride<>> {
    using Description = Doc<R""(@brief Finite Impulse Response (FIR) filter class

The transfer function of an FIR filter is given by:
H(z) = b[0] + b[1]*z^-1 + b[2]*z^-2 + ... + b[N]*z^-N

Stated as a sliding window rather than a delay line: one inner product per output, over `b.size()` samples ending at
the newest. The window that precedes the stream is primed with zeros, so the block is sample-count preserving and
`y[0]` corresponds to `x[0]` -- the same alignment a delay line gives, without carrying one.
)"">;
    PortIn<T>  in;
    PortOut<T> out;
    Tensor<T>  b{T{1}}; // feed-forward coefficients

    /// the `b.size() - 1` samples preceding this span; zeros until the stream has produced that many. Reflected so
    /// that the pmr-field migration re-seats it onto device memory -- the kernel reads it -- and underscored so it
    /// stays off the settings surface.
    mutable Tensor<T> _leadIn{};

    GR_MAKE_REFLECTABLE(fir_filter, in, out, b, _leadIn);

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        this->input_chunk_size  = 1U; // 1:1 over the stream: the lead-in below is what makes the first window full
        this->output_chunk_size = 1U;
        this->stride            = 0U;
        _leadIn                 = Tensor<T>(std::vector<T>(std::max(std::size_t{1}, b.size()) - 1UZ, T{0}));
    }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) const noexcept {
        const std::size_t nTaps = std::max(std::size_t{1}, b.size());
        const std::size_t lead  = nTaps - 1UZ;
        const std::size_t nOut  = std::min(output.size(), input.size());
        // no allocation here: this body compiles into a device kernel, where operator new does not resolve. A
        // lead-in that settingsChanged has not sized yet reads as the zeros it would have been primed with.
        const bool carriesLeadIn = _leadIn.size() == lead;

        // the first `lead` outputs straddle the seam: their window begins before this span and is completed from
        // the carried tail. Spelled out because the two halves index different buffers.
        for (std::size_t n = 0UZ; n < std::min(lead, nOut); ++n) {
            T accumulator{0};
            for (std::size_t j = 0UZ; j <= n; ++j) {
                accumulator += b[j] * input[n - j];
            }
            if (carriesLeadIn) {
                for (std::size_t j = n + 1UZ; j <= lead; ++j) {
                    accumulator += b[j] * _leadIn[lead - (j - n)];
                }
            }
            output[n] = accumulator;
        }

        // the rest sit wholly inside the span: the taps are walked backwards and the window forwards, stated as one
        // inner product so the vectorising is the standard library's rather than a loop the compiler refuses to touch
        const auto newestTapFirst = std::make_reverse_iterator(b.cend());
        const auto oldestTapLast  = std::make_reverse_iterator(b.cbegin());
        for (std::size_t n = lead; n < nOut; ++n) {
            output[n] = std::transform_reduce(std::execution::unseq, newestTapFirst, oldestTapLast, input.data() + n - lead, T{0}, std::plus<>{}, std::multiplies<>{});
        }

        // carry the last `lead` samples of [_leadIn ++ input] for the next span's seam. Reading at nOut + k, which
        // is never below the k being written, so the update is safe in place.
        if (carriesLeadIn) {
            for (std::size_t k = 0UZ; k < lead; ++k) {
                _leadIn[k] = (nOut + k) < lead ? _leadIn[nOut + k] : input[nOut + k - lead];
            }
        }
        return gr::work::Status::OK;
    }
};

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

template<typename T, typename... Args>
requires(std::floating_point<T> or std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>)
struct BasicFilterProto : Block<BasicFilterProto<T, Args...>, Args...> {
    using TParent     = Block<BasicFilterProto<T, Args...>, Args...>;
    using Description = Doc<R""(@brief Basic Digital Filter class supporting FIR and IIR filters

This block implements a digital filter which can be configured as either FIR or IIR,
with selectable filter type (low-pass, high-pass, band-pass, band-stop), and supports resampling.
)"">;
    using ValueType   = meta::fundamental_base_value_type_t<T>;

    PortIn<T>  in;
    PortOut<T> out;

    using FilterImpl = std::conditional_t<UncertainValueLike<T>, filter::ErrorPropagatingFilter<T>, filter::Filter<T>>;

    FilterImpl _filter;

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
        fir_design_method = algorithm::window::Type::Kaiser;

    GR_MAKE_REFLECTABLE(BasicFilterProto, in, out, filter_type, filter_response, filter_order, f_low, f_high, sample_rate, decimate, iir_design_method, fir_design_method);

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) { designFilter(); }

    void designFilter() {
        using namespace gr::filter;

        if constexpr (not TParent::ResamplingControl::kIsConst) {
            this->input_chunk_size = decimate;
        }

        // Set up filter parameters
        FilterParameters params;
        params.order = filter_order;
        params.fLow  = static_cast<double>(f_low);
        params.fHigh = static_cast<double>(f_high);
        params.fs    = static_cast<double>(sample_rate);

        if (filter_type == FilterType::FIR) { // design FIR filter
            _filter = FilterImpl(fir::designFilter<ValueType>(filter_response, params, fir_design_method));
        } else if (filter_type == FilterType::IIR) { // design IIR filter
            _filter = FilterImpl(iir::designFilter<ValueType>(filter_response, params, iir_design_method));
        }
    }

    [[nodiscard]] T processOne(T input) noexcept
    requires(TParent::ResamplingControl::kIsConst)
    {
        return _filter.processOne(input);
    }

    [[nodiscard]] work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept
    requires(not TParent::ResamplingControl::kIsConst)
    {
        assert(output.size() >= input.size() / decimate);

        std::size_t out_sample_idx = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            T output_sample = _filter.processOne(input[i]);

            if (i % decimate == 0) {
                output[out_sample_idx++] = output_sample;
            }
        }
        return work::Status::OK;
    }
};

template<typename T>
using BasicFilter = BasicFilterProto<T>;

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
