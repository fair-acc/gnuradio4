#ifndef GNURADIO_BEAMFORMING_BEAMFORMER_HPP
#define GNURADIO_BEAMFORMING_BEAMFORMER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <expected>
#include <format>
#include <limits>
#include <memory_resource>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/SVD.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::beamforming {

/**
 * bin k's frequency in Hz of F bins at `sampleRate`: one-sided k fs/N (N = 2(F-1)) for real samples, the signed offset for complex ones plus
 * the carrier; a single bin of a snapshot [M] is the carrier
 */
template<typename TSample, std::floating_point Real>
[[nodiscard]] GR_DEVICE_FN Real binFrequency(std::size_t k, std::size_t F, bool snapshot, Real sampleRate, Real carrier) noexcept {
    if (snapshot) {
        return carrier;
    }
    const std::size_t N = gr::algorithm::fft::samplesOf<TSample>(F);
    if (N == 0UZ) {
        return Real{0};
    }
    const Real offset = gr::algorithm::fft::binOffsetHz(k, N, sampleRate / static_cast<Real>(N));
    return gr::meta::complex_like<TSample> ? carrier + offset : std::abs(offset);
}

namespace detail {

/** [M,F] spectra or an [M] snapshot as (M, F); (0, 0) for any other shape */
template<typename TView>
[[nodiscard]] std::array<std::size_t, 2> elementsAndBins(const TView& spectra) noexcept {
    if (spectra.rank() == 1UZ) {
        return {spectra.extent(0UZ), 1UZ};
    }
    return spectra.rank() == 2UZ ? std::array{spectra.extent(0UZ), spectra.extent(1UZ)} : std::array{0UZ, 0UZ};
}

template<typename TComplex>
[[nodiscard]] GR_DEVICE_FN TComplex conjDot(const TComplex* a, std::size_t strideA, const TComplex* x, std::size_t strideX, std::size_t n) noexcept {
    TComplex sum{};
    for (std::size_t i = 0UZ; i < n; ++i) {
        sum += conj(a[i * strideA]) * x[i * strideX];
    }
    return sum;
}

/** a^H R^-1 a that a distortionless constraint can divide by */
template<std::floating_point Real>
[[nodiscard]] GR_DEVICE_FN constexpr bool usableGain(Real gain) noexcept {
    return gain > Real{0} && gain <= std::numeric_limits<Real>::max();
}

} // namespace detail

GR_REGISTER_BLOCK(gr::beamforming::SteeringVectors, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::floating_point<gr::meta::fundamental_base_value_type_t<T>>
struct SteeringVectors : Block<SteeringVectors<T>> {
    using Description = Doc<R""(@brief the steering vectors of the selected peaks, per frame of spectra [M,F] or snapshot [M]: [K,F,M] with
a_m(f) = exp(-j 2 pi f tau_m), tau_m element m's delay for the peak's source.

The latest tensor on `peaks` [K, D+...] selects the sources (as from `TensorPeakFinder` or `SourceFromDelays`); its leading `coordinates`
columns name each in map order: far (azimuth) or (elevation, azimuth) in rad, near (y, x) or (z, y, x) in m. Until a selection arrives the
output is [0,F,M]. Bin frequencies follow `T`: real samples, one-sided bins at k fs/N; complex samples, the signed offset plus the carrier
`frequency` (`gr:frequency`); a snapshot [M] is one bin at the carrier.
References: H. Krim and M. Viberg, "Two decades of array signal processing research: The parametric approach," IEEE Signal Process. Mag.,
vol. 13, no. 4, pp. 67-94, 1996.)"">;

    using Real    = gr::meta::fundamental_base_value_type_t<T>;
    using Complex = std::complex<Real>;

    TensorPortIn<Complex>               in;
    TensorPortIn<Real, Async, Optional> peaks;
    TensorPortOut<Complex>              out;

    gr::algorithm::ElementPositions                                                                                                                                                     element_positions;
    gr::algorithm::PropagationSpeed                                                                                                                                                     propagation_speed = 0.0;
    gr::algorithm::FieldSetting                                                                                                                                                         field             = gr::algorithm::Field::far;
    Annotated<gr::Size_t, "coordinates", Doc<"D, the peaks' leading coordinate columns: far 1 or 2, near 2 or 3">, Limits<1U, 3U>>                                                      coordinates       = 1U;
    Annotated<float, "sample_rate", Unit<"Hz">, Doc<"places the bins; forwarded from upstream">, Limits<0.f, std::numeric_limits<float>::max(), [](float rate) { return 0.f < rate; }>> sample_rate       = 1.f;
    gr::Frequency                                                                                                                                                                       frequency         = 0.0;

    /** [K, M]: each selected source's element delays in s; reflected, so a kernel reads it in place */
    std::pmr::vector<Real> _delays;

    GR_MAKE_REFLECTABLE(SteeringVectors, in, peaks, out, element_positions, propagation_speed, field, coordinates, sample_rate, frequency, _delays);

    std::size_t       _sources = 0UZ;
    std::vector<Real> _selection;
    std::size_t       _selectionColumns = 0UZ;
    std::size_t       _inputElements    = 0UZ;

    void start() {
        _delays.clear();
        _selection.clear();
        _sources       = 0UZ;
        _inputElements = 0UZ;
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, InputSpanLike auto& selection, OutputSpanLike auto& output) { return steer(gr::device::hostBackend(), input, selection, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, InputSpanLike auto& selection, OutputSpanLike auto& output) { return steer(ctx, input, selection, output); }

    [[nodiscard]] gr::work::Status steer(gr::device::DeviceContext& ctx, InputSpanLike auto& input, InputSpanLike auto& selection, OutputSpanLike auto& output) {
        using DeviceComplex = gr::complex<Real>;
        if (selection.isConnected && selection.size() > 0UZ) {
            const TensorView<Real>& rows   = selection[selection.size() - 1UZ];
            const bool              matrix = rows.rank() == 2UZ;
            _selection.resize(rows.size());
            ctx.copy(_selection.data(), rows.data(), _selection.size() * sizeof(Real));
            _selectionColumns   = matrix ? rows.extent(1UZ) : 0UZ;
            const auto selected = placeSources(_selection, matrix ? rows.extent(0UZ) : 0UZ, _selectionColumns);
            std::ignore         = selection.consume(selection.size());
            if (!selected) [[unlikely]] {
                this->emitErrorMessage("processBulk()", selected.error());
                this->requestStop();
                output.publish(0UZ);
                return gr::work::Status::ERROR;
            }
        }
        const std::size_t nOffered = std::min(input.size(), output.size());
        std::size_t       k        = 0UZ;
        for (; k < nOffered; ++k) {
            const TensorView<Complex>& spectra  = input[k];
            const bool                 snapshot = spectra.rank() == 1UZ;
            const auto [M, F]                   = detail::elementsAndBins(spectra);
            /** frames of M elements seen before fit the geometry: a change to another count is refused */
            const auto placed = [&]() -> std::expected<void, std::string> {
                if (M == _inputElements) {
                    return {};
                }
                const auto elements = gr::algorithm::placedElements(element_positions.value);
                if (elements && *elements != M) {
                    return std::unexpected(std::format("frame {} has shape {}, expects [M,F] or [M] for the {} elements", input.streamIndex + k, spectra.extents(), *elements));
                }
                return elements.transform([](std::size_t) {});
            }();
            if (F == 0UZ || !placed || _delays.size() != _sources * M) [[unlikely]] {
                this->emitErrorMessage("processBulk()", placed ? std::format("frame {} has shape {}, expects [M,F] or [M] with steering delays for its M", input.streamIndex + k, spectra.extents()) : placed.error());
                this->requestStop();
                break;
            }
            const std::array<std::size_t, 3> extents{_sources, F, M};
            auto                             vectors = gr::emit(*this, output, k, extents);
            if (!vectors) [[unlikely]] {
                break;
            }
            const Real* const delays  = _delays.data();
            auto* const       a       = reinterpret_cast<DeviceComplex*>(vectors->data());
            const Real        rate    = static_cast<Real>(sample_rate);
            const Real        carrier = static_cast<Real>(frequency.value);
            gr::device::parallelFor(ctx, _sources * F * M, [delays, a, F, M, snapshot, rate, carrier] GR_DEVICE_LAMBDA(std::size_t n) {
                const std::size_t source = n / (F * M);
                const std::size_t bin    = n / M % F;
                const Real        phase  = Real{-2} * std::numbers::pi_v<Real> * binFrequency<T>(bin, F, snapshot, rate, carrier) * delays[source * M + n % M];
                a[n]                     = DeviceComplex(std::cos(phase), std::sin(phase));
            });
            _inputElements = M;
        }
        std::ignore = input.consume(k);
        output.publish(k);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        gr::algorithm::refuseGeometry(*this, oldSettings, newSettings, _inputElements);
        if (_selection.empty() || _selectionColumns == 0UZ) {
            return;
        }
        if (const auto selected = placeSources(_selection, _selection.size() / _selectionColumns, _selectionColumns); !selected) {
            _sources = 0UZ;
            this->emitErrorMessage("settingsChanged()", selected.error());
            this->requestStop();
        }
    }

    /** the element delays [K, M] of the sources peak rows [K, columns] name; an error says what does not fit */
    [[nodiscard]] std::expected<void, std::string> placeSources(std::span<const Real> rows, std::size_t K, std::size_t columns) {
        const auto&       positions = element_positions.value;
        const auto        elements  = gr::algorithm::elementsOf(positions, propagation_speed.value);
        const std::size_t D         = static_cast<std::size_t>(coordinates.value);
        const bool        fits      = field.value == gr::algorithm::Field::far ? D <= 2UZ : D >= 2UZ;
        if (!elements) {
            return std::unexpected(elements.error());
        }
        if (!fits || columns < D || rows.size() != K * columns) {
            return std::unexpected(std::format("peaks [{}, {}] for {} coordinates of a {} field", K, columns, D, gr::meta::enumName(field.value).value_or("")));
        }
        const std::size_t M = *elements;
        _delays.resize(K * M);
        for (std::size_t source = 0UZ; source < K; ++source) {
            const std::array<double, 3> point = gr::algorithm::pointOfCoordinates(rows.data() + source * columns, D, field.value);
            for (std::size_t m = 0UZ; m < M; ++m) {
                _delays[source * M + m] = static_cast<Real>(gr::algorithm::delayOf(positions.data() + 3UZ * m, point, field.value, propagation_speed.value));
            }
        }
        _sources = K;
        return {};
    }
};

/** an adaptive beam's constraint: distortionless (MVDR) keeps unit gain towards its source, nulls (LCMV) also none towards the others */
enum class Constraint : std::uint8_t { conventional, distortionless, nulls };

/** what a bin's solve found, written by the kernel and read back on the host */
enum class BinStatus : std::uint8_t { solved, covarianceNotSolvable, sourceNotInvertible };

template<typename T, Constraint kConstraint>
requires std::floating_point<gr::meta::fundamental_base_value_type_t<T>>
struct Beamformer : Block<Beamformer<T, kConstraint>> {
    using Description = Doc<R""(@brief beams: per frame of spectra [M,F] (or a snapshot [M]), steering vectors [K,F,M] and, for the adaptive
constraints, the spatial covariance [F,M,M] (as from `CrossSpectralMatrix`), the beams [K,F] y_k(f) = w_k(f)^H x(f).

Conventional: w_k = a_k / (a_k^H a_k), unit gain towards source k, the elements' noise averaged (R = I; a_k / M for unit-modulus vectors).
MVDR: w_k = R^-1 a_k / (a_k^H R^-1 a_k), unit gain towards source k and the least power from elsewhere. LCMV: unit gain towards source k and
none towards the other K - 1; bins with numerically dependent constraints (including DC) output zero for every beam.
R is loaded by `diagonal_loading` times its mean diagonal
and solved by Cholesky factorisation per bin, on the host or a device.
References: J. Capon, "High-resolution frequency-wavenumber spectrum analysis," Proc. IEEE, vol. 57, no. 8, pp. 1408-1418, 1969;
O. L. Frost III, "An algorithm for linearly constrained adaptive array processing," Proc. IEEE, vol. 60, no. 8, pp. 926-935, 1972;
H. Krim and M. Viberg, "Two decades of array signal processing research: The parametric approach," IEEE Signal Process. Mag., vol. 13,
no. 4, pp. 67-94, 1996;
J. Li, P. Stoica, and Z. Wang, "On robust Capon beamforming and diagonal loading," IEEE Trans. Signal Process., vol. 51, no. 7,
pp. 1702-1715, 2003.)"">;

    using Real    = gr::meta::fundamental_base_value_type_t<T>;
    using Complex = std::complex<Real>;

    static constexpr bool kUsesCovariance = kConstraint != Constraint::conventional;

    TensorPortIn<Complex>           in;
    TensorPortIn<Complex>           steering;
    TensorPortIn<Complex, Optional> covariance;
    TensorPortOut<Complex>          out;

    Annotated<float, "diagonal_loading", Doc<"added to the covariance's diagonal, relative to its mean diagonal">, Limits<0.f, std::numeric_limits<float>::max()>> diagonal_loading = 0.01f;

    std::pmr::vector<Complex>      _work;
    std::pmr::vector<std::uint8_t> _status;

    GR_MAKE_REFLECTABLE(Beamformer, in, steering, covariance, out, diagonal_loading, _work, _status);

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, InputSpanLike auto& vectors, InputSpanLike auto& covariances, OutputSpanLike auto& output) { return beamform(gr::device::hostBackend(), input, vectors, covariances, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, InputSpanLike auto& vectors, InputSpanLike auto& covariances, OutputSpanLike auto& output) { return beamform(ctx, input, vectors, covariances, output); }

    [[nodiscard]] gr::work::Status beamform(gr::device::DeviceContext& ctx, InputSpanLike auto& input, InputSpanLike auto& vectors, InputSpanLike auto& covariances, OutputSpanLike auto& output) {
        using DeviceComplex = gr::complex<Real>;
        if (kUsesCovariance && !covariances.isConnected) [[unlikely]] {
            this->emitErrorMessage("processBulk()", "an adaptive beamformer needs the covariance [F,M,M] on `covariance`");
            this->requestStop();
            return gr::work::Status::ERROR;
        }
        const std::size_t nOffered = std::min({input.size(), vectors.size(), kUsesCovariance ? covariances.size() : input.size(), output.size()});
        std::size_t       k        = 0UZ;
        for (; k < nOffered; ++k) {
            const auto [M, F]      = detail::elementsAndBins(input[k]);
            const auto& a          = vectors[k];
            const bool  vectorsFit = M > 0UZ && a.rank() == 3UZ && a.extent(1UZ) == F && a.extent(2UZ) == M;
            const bool  fits       = vectorsFit && (!kUsesCovariance || (covariances[k].rank() == 3UZ && covariances[k].extent(0UZ) == F && covariances[k].extent(1UZ) == M && covariances[k].extent(2UZ) == M));
            if (!fits) [[unlikely]] {
                const std::string covarianceShape = kUsesCovariance ? std::format("{}", covariances[k].extents()) : std::string("unused");
                this->emitErrorMessage("processBulk()", std::format("frame {}: spectra {}, steering vectors {} and covariance {} do not match [M,F], [K,F,M] and [F,M,M]", input.streamIndex + k, input[k].extents(), a.extents(), covarianceShape));
                this->requestStop();
                break;
            }
            const std::size_t K = a.extent(0UZ);
            if (kConstraint == Constraint::nulls && K > M) [[unlikely]] {
                this->emitErrorMessage("processBulk()", std::format("{} sources for {} elements: no null for each", K, M));
                this->requestStop();
                break;
            }
            const std::array<std::size_t, 2> extents{K, F};
            auto                             beams = gr::emit(*this, output, k, extents);
            if (!beams) [[unlikely]] {
                break;
            }
            const std::size_t binStride = M * M + K * M + K * K + 2UZ * K;
            _work.resize(F * binStride);
            _status.resize(F);
            auto* const       work    = reinterpret_cast<DeviceComplex*>(_work.data());
            auto* const       status  = _status.data();
            const auto* const X       = reinterpret_cast<const DeviceComplex*>(input[k].data());
            const auto* const A       = reinterpret_cast<const DeviceComplex*>(a.data());
            const auto* const Rcov    = kUsesCovariance ? reinterpret_cast<const DeviceComplex*>(covariances[k].data()) : nullptr;
            auto* const       Y       = reinterpret_cast<DeviceComplex*>(beams->data());
            const Real        loading = static_cast<Real>(diagonal_loading);
            gr::device::parallelFor(ctx, F, [work, status, X, A, Rcov, Y, loading, binStride, M, K, F] GR_DEVICE_LAMBDA(std::size_t f) {
                DeviceComplex* const Rf = work + f * binStride;
                DeviceComplex* const V  = Rf + M * M;
                DeviceComplex* const G  = V + K * M;
                DeviceComplex* const z  = G + K * K;
                DeviceComplex* const c  = z + K;
                for (std::size_t source = 0UZ; source < K; ++source) {
                    for (std::size_t m = 0UZ; m < M; ++m) {
                        V[source * M + m] = A[(source * F + f) * M + m];
                    }
                }
                status[f] = static_cast<std::uint8_t>(BinStatus::solved);
                Real trace{};
                if constexpr (kUsesCovariance) {
                    for (std::size_t m = 0UZ; m < M; ++m) {
                        trace += Rcov[f * M * M + m * M + m].real();
                    }
                }
                if (trace > Real{0}) {
                    const DeviceComplex* const Rin = Rcov + f * M * M;
                    for (std::size_t n = 0UZ; n < M * M; ++n) {
                        Rf[n] = Rin[n];
                    }
                    for (std::size_t m = 0UZ; m < M; ++m) {
                        Rf[m * M + m] += loading * trace / static_cast<Real>(M);
                    }
                    if (!gr::math::cholesky_factor(Rf, M)) {
                        status[f] = static_cast<std::uint8_t>(BinStatus::covarianceNotSolvable);
                        return;
                    }
                    for (std::size_t source = 0UZ; source < K; ++source) {
                        gr::math::cholesky_solve_factored(Rf, M, V + source * M);
                    }
                }
                for (std::size_t j = 0UZ; j < K; ++j) {
                    for (std::size_t source = 0UZ; source < K; ++source) {
                        G[j * K + source] = detail::conjDot(A + (j * F + f) * M, 1UZ, V + source * M, 1UZ, M);
                    }
                }
                if constexpr (kConstraint != Constraint::nulls) {
                    for (std::size_t source = 0UZ; source < K; ++source) {
                        if (!detail::usableGain(G[source * K + source].real())) {
                            status[f] = static_cast<std::uint8_t>(BinStatus::sourceNotInvertible);
                            return;
                        }
                    }
                } else {
                    for (std::size_t source = 0UZ; source < K; ++source) {
                        z[source] = G[source * K + source];
                    }
                    const bool factored    = gr::math::cholesky_factor(G, K);
                    bool       independent = factored;
                    for (std::size_t source = 0UZ; source < K; ++source) {
                        const Real minPivot = std::sqrt(z[source].real() * std::numeric_limits<Real>::epsilon() * static_cast<Real>(K));
                        independent         = independent && G[source * K + source].real() > minPivot;
                    }
                    if (!independent) {
                        for (std::size_t source = 0UZ; source < K; ++source) {
                            Y[source * F + f] = DeviceComplex{};
                        }
                        return;
                    }
                }
                for (std::size_t source = 0UZ; source < K; ++source) {
                    z[source] = detail::conjDot(V + source * M, 1UZ, X + f, F, M);
                }
                for (std::size_t source = 0UZ; source < K; ++source) {
                    if constexpr (kConstraint != Constraint::nulls) {
                        Y[source * F + f] = z[source] / G[source * K + source].real();
                    } else {
                        for (std::size_t j = 0UZ; j < K; ++j) {
                            c[j] = j == source ? DeviceComplex{Real{1}} : DeviceComplex{};
                        }
                        gr::math::cholesky_solve_factored(static_cast<const DeviceComplex*>(G), K, c);
                        DeviceComplex sum{};
                        for (std::size_t j = 0UZ; j < K; ++j) {
                            sum += conj(c[j]) * z[j];
                        }
                        Y[source * F + f] = sum;
                    }
                }
            });
            const auto failed = std::ranges::find_if(_status, [](std::uint8_t code) { return code != static_cast<std::uint8_t>(BinStatus::solved); });
            if (failed != _status.end()) [[unlikely]] {
                const std::size_t f = static_cast<std::size_t>(failed - _status.begin());
                std::string       refusal;
                switch (static_cast<BinStatus>(*failed)) {
                case BinStatus::covarianceNotSolvable: refusal = std::format("bin {}: the loaded covariance cannot be solved", f); break;
                default: {
                    const DeviceComplex* const G      = reinterpret_cast<const DeviceComplex*>(_work.data()) + f * binStride + M * M + K * M;
                    std::size_t                source = 0UZ;
                    while (source + 1UZ < K && detail::usableGain(G[source * K + source].real())) {
                        ++source;
                    }
                    refusal = std::format("source {}: a^H R^-1 a is {}, the covariance cannot be inverted towards it", source, G[source * K + source].real());
                }
                }
                this->emitErrorMessage("processBulk()", refusal);
                this->requestStop();
                break;
            }
        }
        std::ignore = input.consume(k);
        std::ignore = vectors.consume(k);
        if constexpr (kUsesCovariance) {
            std::ignore = covariances.consume(k);
        }
        output.publish(k);
        return gr::work::Status::OK;
    }
};

template<typename T>
using ConventionalBeamformer = Beamformer<T, Constraint::conventional>;

template<typename T>
using MinimumVarianceBeamformer = Beamformer<T, Constraint::distortionless>;
template<typename T>
using NullingBeamformer = Beamformer<T, Constraint::nulls>;

GR_REGISTER_BLOCK("gr::beamforming::ConventionalBeamformer", gr::beamforming::Beamformer, ([T], gr::beamforming::Constraint::conventional), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::beamforming::MinimumVarianceBeamformer", gr::beamforming::Beamformer, ([T], gr::beamforming::Constraint::distortionless), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::beamforming::NullingBeamformer", gr::beamforming::Beamformer, ([T], gr::beamforming::Constraint::nulls), [ float, double, std::complex<float>, std::complex<double> ])

} // namespace gr::beamforming

#endif // GNURADIO_BEAMFORMING_BEAMFORMER_HPP
