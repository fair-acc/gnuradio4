#ifndef GNURADIO_LOCALISATION_PAIR_CROSS_SPECTRA_HPP
#define GNURADIO_LOCALISATION_PAIR_CROSS_SPECTRA_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory_resource>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/algorithm/peak/Peak.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/device/SyclFFT.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::localisation {

enum class Weighting : std::uint8_t { none, phat, scot, roth };

namespace detail {

using gr::algorithm::ElementPair;
using gr::algorithm::pairOf;
using gr::algorithm::pairsOf;

/** the frequencies kept, in Hz: for real samples the band holds |f|, for complex ones the signed baseband offset */
template<std::floating_point Real>
struct Band {
    Real        binHz;
    std::size_t fftSize;
    bool        realSamples;
    Real        fMin;
    Real        fMax;

    [[nodiscard]] GR_DEVICE_FN constexpr bool holds(std::size_t k) const noexcept {
        const Real signedHz = gr::algorithm::fft::binOffsetHz(k, fftSize, binHz);
        const Real f        = realSamples && signedHz < Real{0} ? -signedHz : signedHz;
        return f >= fMin && f <= fMax;
    }
};

template<std::floating_point Real, typename TSample>
[[nodiscard]] Band<Real> bandOf(float sampleRate, std::size_t N, float fMin, float fMax) noexcept {
    return {.binHz = static_cast<Real>(sampleRate) / static_cast<Real>(N), .fftSize = N, .realSamples = !gr::meta::complex_like<TSample>, .fMin = static_cast<Real>(fMin), .fMax = static_cast<Real>(fMax)};
}

inline constexpr std::size_t kPeakColumns = 3UZ; /**< {delay, peak, confidence} */

/** the `gr:axis_*` of a [P, n] tensor: dimension 0 the element pairs, dimension 1 `name` in `unit` from `origin` by `step` */
[[nodiscard]] inline property_map pairAxes(std::string_view name, std::string_view unit, double origin, double step) {
    const gr::Tensor<double>     origins{0.0, origin};
    const gr::Tensor<double>     steps{1.0, step};
    const gr::Tensor<bool>       periodic(gr::data_from, {false, false});
    const gr::Tensor<pmt::Value> units{pmt::Value{""}, pmt::Value{unit}};
    const gr::Tensor<pmt::Value> names{pmt::Value{"pair"}, pmt::Value{name}};
    return {{std::string(gr::tag::kAxisOrigin.key()), origins}, {std::string(gr::tag::kAxisStep.key()), steps}, {std::string(gr::tag::kAxisPeriodic.key()), periodic}, //
        {std::string(gr::tag::kAxisUnit.key()), units}, {std::string(gr::tag::kAxisName.key()), names}};
}

/** element n of the cross-spectra [P,F] of the spectra X [M,F]: pair n / F at bin n % F, X_i conj(X_j) */
template<std::floating_point Real>
[[nodiscard]] GR_DEVICE_FN gr::complex<Real> crossSpectrum(const gr::complex<Real>* X, std::size_t n, std::size_t M, std::size_t F) noexcept {
    const ElementPair pair = pairOf(n / F, M);
    return X[pair.i * F + n % F] * conj(X[pair.j * F + n % F]);
}

/**
 * one pair's cross-spectrum G over F bins into `out` (in place allowed), divided by a weight plus eps times the band's largest weight, zero outside
 * the band: PHAT |G|, SCOT sqrt(G_ii G_jj), Roth G_ii (element i of the pair i < j as the reference), none no division (Knapp and Carter 1976);
 * `powerI`, `powerJ`: the auto-spectra G_ii, G_jj, read by SCOT and Roth only
 */
template<std::floating_point Real>
GR_DEVICE_FN void weightPair(const gr::complex<Real>* G, gr::complex<Real>* out, std::size_t F, Weighting weighting, Real eps, Band<Real> band, const Real* powerI = nullptr, const Real* powerJ = nullptr) noexcept {
    const auto weightOf = [&](std::size_t k) {
        switch (weighting) {
        case Weighting::phat: return abs(G[k]);
        case Weighting::scot: return std::sqrt(powerI[k] * powerJ[k]);
        case Weighting::roth: return powerI[k];
        case Weighting::none: break;
        }
        return Real{1};
    };
    Real largest = Real{0};
    for (std::size_t k = 0UZ; k < F; ++k) {
        if (band.holds(k)) {
            largest = std::max(largest, weightOf(k));
        }
    }
    for (std::size_t k = 0UZ; k < F; ++k) {
        const Real divisor = weighting == Weighting::none ? Real{1} : weightOf(k) + eps * largest;
        out[k]             = band.holds(k) && divisor > Real{0} ? G[k] / divisor : gr::complex<Real>{};
    }
}

/** row {delay, peak, confidence} of one lag curve of L = 2 c + 1 samples (lag 0 at c), searched within c +- maxLag */
template<std::floating_point T>
GR_DEVICE_FN void findPairDelay(const T* curve, std::size_t L, std::size_t maxLag, T secondsPerLag, T* row) noexcept {
    const std::size_t centre = L / 2UZ;
    const std::size_t first  = centre - std::min(maxLag, centre);
    const std::size_t last   = centre + std::min(maxLag, centre);
    std::size_t       best   = first;
    for (std::size_t l = first + 1UZ; l <= last; ++l) {
        best = curve[l] > curve[best] ? l : best;
    }
    T second = -std::numeric_limits<T>::infinity();
    for (std::size_t l = first; l <= last; ++l) {
        second = l != best && gr::algorithm::peak::isLocalMaximum(curve, l, L) && curve[l] > second ? curve[l] : second;
    }
    const bool apex = gr::algorithm::peak::isLocalMaximum(curve, best, L);
    using Peak      = gr::algorithm::peak::ParabolicPeak<T>;
    const Peak peak = apex ? gr::algorithm::peak::parabolicPeak(curve[best - 1UZ], curve[best], curve[best + 1UZ]) : Peak{.offsetSamples = T{0}, .value = curve[best]};
    row[0]          = (static_cast<T>(best) - static_cast<T>(centre) + peak.offsetSamples) * secondsPerLag;
    row[1]          = peak.value;
    row[2]          = second > T{0} ? peak.value / second : std::numeric_limits<T>::infinity();
}

} // namespace detail

GR_REGISTER_BLOCK(gr::localisation::PairCrossSpectra, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::floating_point<gr::meta::fundamental_base_value_type_t<T>>
struct PairCrossSpectra : Block<PairCrossSpectra<T>, Reshaping> {
    using Description = Doc<R""(@brief the cross-spectra G_ij = X_i conj(X_j) of every element pair i < j of spectra [M,F]: [P,F], P = M(M-1)/2 pairs in
lexicographic order (0,1), (0,2), ..., (1,2), ...

`T` is the sample type the spectra came from, as for `STFT`; it does not change the arithmetic.
A positive lag in the pair's correlation means element i hears the signal later.
Reference: C. H. Knapp and G. C. Carter, "The generalized correlation method for estimation of time delay," IEEE Trans. Acoust., Speech,
Signal Process., vol. 24, no. 4, pp. 320-327, 1976.)"">;

    using Real    = gr::meta::fundamental_base_value_type_t<T>;
    using Complex = std::complex<Real>;

    TensorPortIn<Complex>  in;
    TensorPortOut<Complex> out;

    GR_MAKE_REFLECTABLE(PairCrossSpectra, in, out);

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return crossSpectra(gr::device::hostBackend(), input, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) { return crossSpectra(ctx, input, output); }

    [[nodiscard]] gr::work::Status crossSpectra(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        using DeviceComplex        = gr::complex<Real>;
        const std::size_t nOffered = std::min(input.size(), output.size());
        std::size_t       k        = 0UZ;
        for (; k < nOffered; ++k) {
            const TensorView<Complex>& spectra = input[k];
            if (spectra.rank() != 2UZ || spectra.extent(0UZ) < 2UZ) [[unlikely]] {
                this->emitErrorMessage("processBulk()", std::format("frame {} has shape {}, expects [M,F] with M >= 2", input.streamIndex + k, spectra.extents()));
                this->requestStop();
                break;
            }
            const std::size_t                M = spectra.extent(0UZ);
            const std::size_t                F = spectra.extent(1UZ);
            const std::array<std::size_t, 2> extents{detail::pairsOf(M), F};
            auto                             cross = gr::emit(*this, output, k, extents);
            if (!cross) [[unlikely]] {
                break;
            }
            const auto* const X = reinterpret_cast<const DeviceComplex*>(spectra.data());
            auto* const       G = reinterpret_cast<DeviceComplex*>(cross->data());
            gr::device::parallelFor(ctx, extents[0] * F, [X, G, M, F] GR_DEVICE_LAMBDA(std::size_t n) { G[n] = detail::crossSpectrum(X, n, M, F); });
        }
        std::ignore = input.consume(k);
        output.publish(k);
        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::localisation::PairCrossCorrelation, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::floating_point<gr::meta::fundamental_base_value_type_t<T>>
struct PairCrossCorrelation : Block<PairCrossCorrelation<T>, Resampling<>, Framing, Reshaping> {
    using Description = Doc<R""(@brief the generalised cross-correlation of every element pair of frames [M,N]: [P, 2 max_lag + 1] over the lags
-max_lag..+max_lag samples, a positive lag meaning element i (of pair i < j) hears the signal later.

Each row is transformed by an N-point FFT (no window), the pairs' cross-spectra and the rows' auto-spectra summed over `averages` frames,
weighted (PHAT by default, SCOT, Roth or none) and transformed back, divided by N (unweighted also by `averages`): one curve per `averages` frames.
The correlation is circular, so `max_lag` should stay well below N/2 (0: N/2 - 1; N/2 or more stops).
Real samples give the correlation, complex ones its magnitude (the envelope). A lag in seconds is the lag over `gr:sample_rate`.
The axes ("pair"; "lag" in s from -max_lag / sample_rate by 1 / sample_rate) go out as `gr:axis_*` with the first curve and after max_lag or the rate changes.
On a device N must be a power of two and the samples `float` or `std::complex<float>`.
References: C. H. Knapp and G. C. Carter, "The generalized correlation method for estimation of time delay," IEEE Trans. Acoust., Speech,
Signal Process., vol. 24, no. 4, pp. 320-327, 1976; J. Chen, J. Benesty, and Y. Huang, "Time delay estimation in room acoustic
environments: an overview," EURASIP J. Adv. Signal Process., vol. 2006, art. 026503, 2006.)"">;

    using Real    = gr::meta::fundamental_base_value_type_t<T>;
    using Complex = std::complex<Real>;

    enum class Direction : std::uint8_t { forward, inverse };

    TensorPortIn<T>     in;
    TensorPortOut<Real> out;

    Annotated<gr::Size_t, "max_lag", Doc<"largest lag in samples, below N/2; 0: N/2 - 1">>                                                                                              max_lag     = 0U;
    Annotated<Weighting, "weighting", Doc<"none, phat, scot or roth; scot and roth need averages of 2 or more to differ from phat">>                                                    weighting   = Weighting::phat;
    Annotated<gr::Size_t, "averages", Doc<"frames whose spectra are averaged into one curve">, Limits<1U, 4096U>>                                                                       averages    = 1U;
    Annotated<float, "eps", Doc<"regularisation relative to the band's largest magnitude">, Limits<0.f, std::numeric_limits<float>::max()>>                                             eps         = 1e-6f;
    Annotated<float, "f_min", Unit<"Hz">, Doc<"lowest frequency kept">>                                                                                                                 f_min       = std::numeric_limits<float>::lowest();
    Annotated<float, "f_max", Unit<"Hz">, Doc<"highest frequency kept">>                                                                                                                f_max       = std::numeric_limits<float>::max();
    Annotated<float, "sample_rate", Unit<"Hz">, Doc<"places the band; forwarded from upstream">, Limits<0.f, std::numeric_limits<float>::max(), [](float rate) { return 0.f < rate; }>> sample_rate = 1.f;

    /** [M,N]; reflected, as the one below, so a kernel works on it in place */
    std::pmr::vector<Complex> _spectra;
    /** [P,N]: the summed, then weighted cross-spectra, then the correlations */
    std::pmr::vector<Complex> _cross;
    /** [M,N]: the summed auto-spectra |X|^2 */
    std::pmr::vector<Real> _power;

    GR_MAKE_REFLECTABLE(PairCrossCorrelation, in, out, max_lag, weighting, averages, eps, f_min, f_max, sample_rate, _spectra, _cross, _power);

    std::vector<Complex>                         _transformed;
    gr::algorithm::FFT<Complex, Complex>         _fft;
    gr::device::SyclFFT                          _syclFft;
    std::optional<std::pair<std::size_t, float>> _axesPublishedFor; /// {max_lag in effect, sample_rate} of the last axes tag

    void start() { _axesPublishedFor.reset(); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        this->input_chunk_size  = averages.value;
        this->output_chunk_size = 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return correlate(gr::device::hostBackend(), input, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output)
    requires std::same_as<Real, float>
    {
        return correlate(ctx, input, output);
    }

    [[nodiscard]] gr::work::Status correlate(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        using DeviceComplex        = gr::complex<Real>;
        using DeviceSample         = std::conditional_t<gr::meta::complex_like<T>, DeviceComplex, Real>;
        const bool        onDevice = &ctx != &gr::device::hostBackend();
        const std::size_t A        = averages.value;
        const std::size_t nOffered = std::min(input.size() / A, output.size());
        std::size_t       k        = 0UZ;
        for (; k < nOffered; ++k) {
            const TensorView<T>& first         = input[k * A];
            const std::size_t    M             = first.rank() == 2UZ ? first.extent(0UZ) : 0UZ;
            const std::size_t    N             = first.rank() == 2UZ ? first.extent(1UZ) : 0UZ;
            const std::size_t    L             = max_lag.value == 0U ? N / 2UZ - std::min(N / 2UZ, 1UZ) : static_cast<std::size_t>(max_lag.value);
            const auto           shapedAsFirst = [&](std::size_t f) { return std::ranges::equal(input[f].extents(), first.extents()); };
            const bool           alike         = std::ranges::all_of(std::views::iota(k * A, k * A + A), shapedAsFirst);
            if (M < 2UZ || N < 2UZ || 2UZ * L >= N || !alike || (onDevice && !std::has_single_bit(N))) [[unlikely]] {
                const std::string message = std::format("frame {} has shape {}, expects [M,N] with M >= 2, max_lag {} < N/2, the same over the {} frames averaged, on a device N a power of two", input.streamIndex + k * A, first.extents(), L, A);
                this->emitErrorMessage("processBulk()", message);
                this->requestStop();
                break;
            }
            const std::size_t                P = detail::pairsOf(M);
            const std::array<std::size_t, 2> extents{P, 2UZ * L + 1UZ};
            auto                             lags = gr::emit(*this, output, k, extents);
            if (!lags) [[unlikely]] {
                break;
            }
            if (const std::pair axesFor{L, sample_rate.value}; _axesPublishedFor != axesFor) {
                const double secondsPerLag = 1.0 / static_cast<double>(sample_rate.value);
                output.publishTag(detail::pairAxes("lag", "s", -static_cast<double>(L) * secondsPerLag, secondsPerLag), k);
                _axesPublishedFor = axesFor;
            }
            _spectra.resize(M * N);
            _cross.resize(P * N);
            const bool needsPower = weighting.value == Weighting::scot || weighting.value == Weighting::roth;
            _power.resize(needsPower ? M * N : 0UZ);
            auto* const X     = reinterpret_cast<DeviceComplex*>(_spectra.data());
            auto* const G     = reinterpret_cast<DeviceComplex*>(_cross.data());
            Real* const power = _power.data();
            for (std::size_t f = 0UZ; f < A; ++f) {
                const bool        firstFrame = f == 0UZ;
                const auto* const samples    = reinterpret_cast<const DeviceSample*>(input[k * A + f].data());
                gr::device::parallelFor(ctx, M * N, [X, samples] GR_DEVICE_LAMBDA(std::size_t n) { X[n] = DeviceComplex(samples[n]); });
                transformRows(ctx, _spectra, N, Direction::forward);
                gr::device::parallelFor(ctx, P * N, [X, G, M, N, firstFrame] GR_DEVICE_LAMBDA(std::size_t n) {
                    const DeviceComplex cross = detail::crossSpectrum(X, n, M, N);
                    G[n]                      = firstFrame ? cross : G[n] + cross;
                });
                if (needsPower) {
                    gr::device::parallelFor(ctx, M * N, [X, power, firstFrame] GR_DEVICE_LAMBDA(std::size_t n) {
                        const Real squared = X[n].real() * X[n].real() + X[n].imag() * X[n].imag();
                        power[n]           = firstFrame ? squared : power[n] + squared;
                    });
                }
            }

            const detail::Band<Real> band     = detail::bandOf<Real, T>(sample_rate, N, f_min, f_max);
            const Weighting          mode     = weighting;
            const Real               relative = static_cast<Real>(eps);
            gr::device::parallelFor(ctx, P, [G, power, M, N, mode, relative, band] GR_DEVICE_LAMBDA(std::size_t p) {
                const detail::ElementPair pair   = detail::pairOf(p, M);
                const Real* const         powerI = power == nullptr ? nullptr : power + pair.i * N;
                const Real* const         powerJ = power == nullptr ? nullptr : power + pair.j * N;
                detail::weightPair(G + p * N, G + p * N, N, mode, relative, band, powerI, powerJ);
            });
            transformRows(ctx, _cross, N, Direction::inverse);

            auto* const correlations = lags->data();
            const auto  width        = extents[1];
            const Real  perAverage   = mode == Weighting::none ? Real{1} / static_cast<Real>(A) : Real{1}; // a weighting already removes the sum's scale
            gr::device::parallelFor(ctx, P * width, [G, correlations, N, L, width, perAverage] GR_DEVICE_LAMBDA(std::size_t n) {
                const DeviceComplex r = G[n / width * N + (n % width + N - L) % N];
                if constexpr (gr::meta::complex_like<T>) {
                    correlations[n] = abs(r) * perAverage;
                } else {
                    correlations[n] = r.real() * perAverage;
                }
            });
        }
        std::ignore = input.consume(k * A);
        output.publish(k);
        return gr::work::Status::OK;
    }

    /** every N-row of `rows` transformed in place, the inverse divided by N */
    void transformRows(gr::device::DeviceContext& ctx, std::pmr::vector<Complex>& rows, std::size_t N, Direction direction) {
        if constexpr (std::same_as<Real, float>) {
            if (&ctx != &gr::device::hostBackend()) {
                const std::span<gr::complex<float>> batch(reinterpret_cast<gr::complex<float>*>(rows.data()), rows.size());
                _syclFft.init(ctx, N);
                if (direction == Direction::forward) {
                    _syclFft.forwardBatch(ctx, batch, N);
                } else {
                    _syclFft.inverseBatch(ctx, batch, N);
                }
                return;
            }
        }
        _transformed.resize(N);
        for (std::size_t r = 0UZ; r < rows.size() / N; ++r) {
            const std::span<Complex> row(rows.data() + r * N, N);
            if (direction == Direction::inverse) {
                _fft.computeInverse(row, row);
            } else {
                _fft.compute(row, _transformed);
                std::ranges::copy(_transformed, row.begin());
            }
        }
    }
};

GR_REGISTER_BLOCK(gr::localisation::PairDelay, [T], [ float, double ])

template<std::floating_point T>
struct PairDelay : Block<PairDelay<T>, Reshaping> {
    using Description = Doc<R""(@brief the peak of each lag curve [P,L] (lag 0 at the middle, L odd, as from `PairCrossCorrelation`): [P,3] = {delay in s,
peak value, confidence}, the peak placed between samples by a parabola through it and its neighbours; its axes ("pair"; "column") go out with
the first frame and after the number of pairs changes.

The delay is the lag over `sample_rate`; confidence is the peak over the second-highest local maximum within the search (the curve's end
samples are none), +inf without one or when that is not positive. A peak at the search's edge that is not a local maximum is not interpolated.
`max_lag` narrows the search around lag 0; 0 searches the whole curve.
References: J. O. Smith III and X. Serra, "PARSHL: An analysis/synthesis program for non-harmonic sounds based on a sinusoidal
representation," in Proc. ICMC, 1987, pp. 290-297; E. Jacobsen and P. Kootsookos, "Fast, accurate frequency estimators," IEEE Signal
Process. Mag., vol. 24, no. 3, pp. 123-125, 2007.)"">;

    TensorPortIn<T>  in;
    TensorPortOut<T> out;

    Annotated<gr::Size_t, "max_lag", Doc<"search within +- this many lags; 0: the whole curve">>                                                                                        max_lag     = 0U;
    Annotated<float, "sample_rate", Unit<"Hz">, Doc<"lags per second; forwarded from upstream">, Limits<0.f, std::numeric_limits<float>::max(), [](float rate) { return 0.f < rate; }>> sample_rate = 1.f;

    GR_MAKE_REFLECTABLE(PairDelay, in, out, max_lag, sample_rate);

    std::size_t _axesPublishedForPairs = 0UZ; /// 0: none yet

    void start() { _axesPublishedForPairs = 0UZ; }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return findPeaks(gr::device::hostBackend(), input, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) { return findPeaks(ctx, input, output); }

    [[nodiscard]] gr::work::Status findPeaks(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t nOffered = std::min(input.size(), output.size());
        std::size_t       k        = 0UZ;
        for (; k < nOffered; ++k) {
            const TensorView<T>& curves = input[k];
            const std::size_t    L      = curves.rank() == 2UZ ? curves.extent(1UZ) : 0UZ;
            if (L % 2UZ == 0UZ) [[unlikely]] {
                this->emitErrorMessage("processBulk()", std::format("frame {} has shape {}, expects [P,L] with L odd", input.streamIndex + k, curves.extents()));
                this->requestStop();
                break;
            }
            const std::size_t                P = curves.extent(0UZ);
            const std::array<std::size_t, 2> extents{P, detail::kPeakColumns};
            auto                             peaks = gr::emit(*this, output, k, extents);
            if (peaks && _axesPublishedForPairs != P) {
                output.publishTag(detail::pairAxes("column", "", 0.0, 1.0), k);
                _axesPublishedForPairs = P;
            }
            if (!peaks) [[unlikely]] {
                break;
            }
            const T* const    lags          = curves.data();
            T* const          rows          = peaks->data();
            const std::size_t maxLag        = max_lag.value == 0U ? L : static_cast<std::size_t>(max_lag.value);
            const T           secondsPerLag = T{1} / static_cast<T>(sample_rate);
            gr::device::parallelFor(ctx, P, [lags, rows, L, maxLag, secondsPerLag] GR_DEVICE_LAMBDA(std::size_t p) { detail::findPairDelay(lags + p * L, L, maxLag, secondsPerLag, rows + p * detail::kPeakColumns); });
        }
        std::ignore = input.consume(k);
        output.publish(k);
        return gr::work::Status::OK;
    }
};

} // namespace gr::localisation

#endif // GNURADIO_LOCALISATION_PAIR_CROSS_SPECTRA_HPP
