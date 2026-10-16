#ifndef GNURADIO_FOURIER_STFT_HPP
#define GNURADIO_FOURIER_STFT_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <memory_resource>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/device/SyclFFT.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::blocks::fft {

namespace detail {
/** a device FFT takes powers of two only; false, with the frame refused and the graph stopped, otherwise */
[[nodiscard]] bool deviceTakesLength(auto& block, bool onDevice, std::size_t N, std::size_t frameIndex, const auto& frame) {
    if (!onDevice || std::has_single_bit(N)) {
        return true;
    }
    block.emitErrorMessage("processBulk()", std::format("frame {} has shape {}, expects N a power of two on a device", frameIndex, frame.extents()));
    block.requestStop();
    return false;
}
} // namespace detail

inline constexpr gr::algorithm::window::Type kDefaultStftWindow = gr::algorithm::window::Type::Hann;

GR_REGISTER_BLOCK("gr::blocks::fft::STFT", gr::blocks::fft::STFT, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::floating_point<gr::meta::fundamental_base_value_type_t<T>>
struct STFT : Block<STFT<T>> {
    using Description = Doc<R""(@brief the windowed spectrum of each row of a frame [M,N]: [M, N/2+1] for real samples (one-sided), [M,N] for complex ones;
DC first, no shift, no scaling.

The window (default Hann, in its periodic form) multiplies each row before its FFT; the frames' overlap comes from the framing stride (`StreamsToTensor`).
N must be at least 2, and even for real samples so that `InverseSTFT` can tell N from N/2 + 1 bins; on a device N must be a power of
two and the samples `float` or `std::complex<float>`.
References: J. B. Allen, "Short term spectral analysis, synthesis, and modification by discrete Fourier transform," IEEE Trans. Acoust., Speech,
Signal Process., vol. 25, no. 3, pp. 235-238, 1977; F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier
transform," Proc. IEEE, vol. 66, no. 1, pp. 51-83, 1978.)"">;

    using Real    = gr::meta::fundamental_base_value_type_t<T>;
    using Complex = std::complex<Real>;

    TensorPortIn<T>        in;
    TensorPortOut<Complex> out;

    Annotated<gr::algorithm::window::Type, "window", Doc<gr::algorithm::window::TypeNames>> window = kDefaultStftWindow;

    std::pmr::vector<Real> _windowCoefficients;
    /** [M,N] the rows a device transforms when real samples do not fit their one-sided output */
    std::pmr::vector<Complex> _batch;

    GR_MAKE_REFLECTABLE(STFT, in, out, window, _windowCoefficients, _batch);

    std::vector<Complex>                 _row;
    std::vector<Complex>                 _spectrum;
    gr::algorithm::FFT<Complex, Complex> _fft;
    gr::device::SyclFFT                  _syclFft;

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return transform(gr::device::hostBackend(), input, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output)
    requires std::same_as<Real, float>
    {
        return transform(ctx, input, output);
    }

    [[nodiscard]] gr::work::Status transform(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const bool        onDevice = &ctx != &gr::device::hostBackend();
        const std::size_t nOffered = std::min(input.size(), output.size());
        std::size_t       k        = 0UZ;
        for (; k < nOffered; ++k) {
            const TensorView<T>& frame = input[k];
            if (!transformable(frame, input.streamIndex + k)) [[unlikely]] {
                break;
            }
            const std::size_t M = frame.extent(0UZ);
            const std::size_t N = frame.extent(1UZ);
            const std::size_t F = gr::algorithm::fft::binsOf(N, !gr::meta::complex_like<T>);
            if (!detail::deviceTakesLength(*this, onDevice, N, input.streamIndex + k, frame)) [[unlikely]] {
                break;
            }
            const std::array<std::size_t, 2> extents{M, F};
            auto                             spectra = gr::emit(*this, output, k, extents);
            if (!spectra) [[unlikely]] {
                break;
            }
            prepare(N);
            if constexpr (std::same_as<Real, float>) {
                if (onDevice) {
                    using DeviceComplex = gr::complex<float>;
                    using DeviceSample  = std::conditional_t<gr::meta::complex_like<T>, DeviceComplex, float>;
                    if constexpr (!gr::meta::complex_like<T>) {
                        _batch.resize(M * N);
                    }
                    auto* const       bins         = reinterpret_cast<DeviceComplex*>(spectra->data());
                    auto* const       batch        = reinterpret_cast<DeviceComplex*>(_batch.data());
                    auto* const       transformed  = gr::meta::complex_like<T> ? bins : batch;
                    const auto* const samples      = reinterpret_cast<const DeviceSample*>(frame.data());
                    const Real* const coefficients = _windowCoefficients.data();
                    const auto        windowed     = [transformed, samples, coefficients, N] GR_DEVICE_LAMBDA(std::size_t i) { //
                        transformed[i] = gr::algorithm::fft::applyWindowOne(DeviceComplex(samples[i]), coefficients[i % N]);
                    };
                    gr::device::parallelFor(ctx, M * N, windowed);
                    _syclFft.init(ctx, N);
                    _syclFft.forwardBatch(ctx, std::span<DeviceComplex>(transformed, M * N), N);
                    if constexpr (!gr::meta::complex_like<T>) {
                        gr::device::parallelFor(ctx, M * F, [transformed, bins, N, F] GR_DEVICE_LAMBDA(std::size_t i) { //
                            bins[i] = transformed[(i / F) * N + i % F];
                        });
                    }
                    continue;
                }
            }
            for (std::size_t m = 0UZ; m < M; ++m) {
                const T* const samples = frame.data() + m * N;
                std::ranges::copy(std::span<const T>(samples, N), _row.begin());
                gr::algorithm::fft::applyWindow(std::span<Complex>(_row), std::span<const Real>(_windowCoefficients));
                _fft.compute(_row, _spectrum);
                std::copy_n(_spectrum.begin(), F, spectra->begin() + static_cast<std::ptrdiff_t>(m * F));
            }
        }
        std::ignore = input.consume(k);
        output.publish(k);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("window")) {
            _windowCoefficients.clear();
        }
    }

    /** false, with the error reported, for a frame other than [M,N] with N >= 2, even for real samples */
    [[nodiscard]] bool transformable(const TensorView<T>& frame, std::size_t frameIndex) {
        const std::size_t N = frame.rank() == 2UZ ? frame.extent(1UZ) : 0UZ;
        if (N >= 2UZ && (gr::meta::complex_like<T> || N % 2UZ == 0UZ)) [[likely]] {
            return true;
        }
        const std::string message = std::format("frame {} has shape {}, expects [M,N] with N >= 2, even for real samples", frameIndex, frame.extents());
        this->emitErrorMessage("processBulk()", message);
        this->requestStop();
        return false;
    }

    void prepare(std::size_t N) {
        if (_windowCoefficients.size() == N) {
            return;
        }
        _windowCoefficients.assign(N, Real{1});
        gr::algorithm::window::create(std::span<Real>(_windowCoefficients), window, gr::algorithm::window::kDefaultBeta<Real>, true);
        _row.resize(N);
        _spectrum.resize(N);
    }
};

GR_REGISTER_BLOCK("gr::blocks::fft::InverseSTFT", gr::blocks::fft::InverseSTFT, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::floating_point<gr::meta::fundamental_base_value_type_t<T>>
struct InverseSTFT : Block<InverseSTFT<T>> {
    using Description = Doc<R""(@brief samples back from the spectra [M,F] of `STFT` by weighted overlap-add: per frame a tensor [M, step] of the samples
the frame completes; `T` real takes one-sided spectra (F = N/2 + 1, N even), `T` complex full ones (F = N).

Each frame's inverse FFT (divided by N) is multiplied by the synthesis window -- the analysis window of `STFT` -- and added in;
the sum is divided by the squares of the windows that overlapped there so far, so any window and step whose squares overlap without a
gap reconstruct the input exactly wherever a window is non-zero; the last N - step samples stay in the overlap-add when the stream ends.
The hop is `gr:sample_rate` / `gr:frame_rate` of the frames' tag, N/2 before any rate arrived.
On a device N must be a power of two and the samples `float` or `std::complex<float>`.
Reference: D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform," IEEE Trans. Acoust., Speech, Signal
Process., vol. 32, no. 2, pp. 236-243, 1984.)"">;

    using Real    = gr::meta::fundamental_base_value_type_t<T>;
    using Complex = std::complex<Real>;

    TensorPortIn<Complex> in;
    TensorPortOut<T>      out;

    Annotated<gr::algorithm::window::Type, "window", Doc<gr::algorithm::window::TypeNames>> window = kDefaultStftWindow;

    std::pmr::vector<Real> _windowCoefficients;
    /** [ceil(N/hop), hop]: row c the squared windows of c + 1 frames summed per position of a hop */
    std::pmr::vector<Real> _overlapSquares;
    /** [M,N]: the overlap-add, each row a ring whose oldest sample is at `_head` */
    std::pmr::vector<T> _accumulated;
    /** [M,N] the full spectra a device transforms in place */
    std::pmr::vector<Complex> _batch;

    GR_MAKE_REFLECTABLE(InverseSTFT, in, out, window, _windowCoefficients, _overlapSquares, _accumulated, _batch);

    std::vector<Complex>                 _spectrum;
    std::vector<Complex>                 _samples;
    gr::algorithm::FFT<Complex, Complex> _fft;
    gr::device::SyclFFT                  _syclFft;
    std::size_t                          _M            = 0UZ;
    std::size_t                          _hop          = 0UZ;
    std::size_t                          _head         = 0UZ;
    std::size_t                          _nFrames      = 0UZ;
    std::size_t                          _hopFromRates = 0UZ;

    void start() {
        _windowCoefficients.clear();
        _hopFromRates = 0UZ;
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return synthesise(gr::device::hostBackend(), input, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output)
    requires std::same_as<Real, float>
    {
        return synthesise(ctx, input, output);
    }

    [[nodiscard]] gr::work::Status synthesise(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const bool        onDevice = &ctx != &gr::device::hostBackend();
        const std::size_t nOffered = std::min(input.size(), output.size());
        std::size_t       k        = 0UZ;
        for (; k < nOffered; ++k) {
            const TensorView<Complex>& frame = input[k];
            if (frame.rank() != 2UZ || frame.extent(1UZ) < 2UZ) [[unlikely]] {
                const std::size_t frameIndex = input.streamIndex + k;
                this->emitErrorMessage("processBulk()", std::format("frame {} has shape {}, expects [M,F] with F >= 2", frameIndex, frame.extents()));
                this->requestStop();
                break;
            }
            const std::size_t M = frame.extent(0UZ);
            const std::size_t F = frame.extent(1UZ);
            const std::size_t N = gr::algorithm::fft::samplesOf(F, !gr::meta::complex_like<T>);
            if (!detail::deviceTakesLength(*this, onDevice, N, input.streamIndex + k, frame)) [[unlikely]] {
                break;
            }
            const std::size_t hop = hopSamplesOf(input, k, N);
            if (hop == 0UZ || !prepare(M, N, hop)) {
                break;
            }
            const std::array<std::size_t, 2> extents{M, hop};
            auto                             completed = gr::emit(*this, output, k, extents);
            if (!completed) [[unlikely]] {
                break;
            }
            if constexpr (std::same_as<Real, float>) {
                if (onDevice) {
                    using DeviceComplex = gr::complex<float>;
                    using DeviceSample  = std::conditional_t<gr::meta::complex_like<T>, DeviceComplex, float>;
                    _batch.resize(M * N);
                    auto* const       spectrum = reinterpret_cast<DeviceComplex*>(_batch.data());
                    const auto* const bins     = reinterpret_cast<const DeviceComplex*>(frame.data());
                    gr::device::parallelFor(ctx, M * N, [spectrum, bins, N, F] GR_DEVICE_LAMBDA(std::size_t i) {
                        const std::size_t m   = i / N;
                        const std::size_t bin = i % N;
                        spectrum[i]           = bin < F ? bins[m * F + bin] : conj(bins[m * F + N - bin]);
                    });
                    _syclFft.init(ctx, N);
                    _syclFft.inverseBatch(ctx, std::span<DeviceComplex>(spectrum, M * N), N);
                    auto* const       accumulated  = reinterpret_cast<DeviceSample*>(_accumulated.data());
                    const Real* const coefficients = _windowCoefficients.data();
                    gr::device::parallelFor(ctx, M * N, [accumulated, spectrum, coefficients, N, head = _head] GR_DEVICE_LAMBDA(std::size_t i) {
                        const std::size_t   j      = i % N;
                        const DeviceComplex sample = spectrum[i] * coefficients[j];
                        if constexpr (gr::meta::complex_like<T>) {
                            accumulated[i - j + (head + j) % N] += sample;
                        } else {
                            accumulated[i - j + (head + j) % N] += sample.real();
                        }
                    });
                    auto* const done = reinterpret_cast<DeviceSample*>(completed->data());
                    gr::device::parallelFor(ctx, M * hop, [accumulated, done, squares = startUpSquares(), N, hop, head = _head] GR_DEVICE_LAMBDA(std::size_t i) {
                        const std::size_t j    = i % hop;
                        DeviceSample&     slot = accumulated[i / hop * N + (head + j) % N];
                        done[i]                = squares[j] > 0.f ? slot / squares[j] : DeviceSample{};
                        slot                   = DeviceSample{};
                    });
                    _head = (_head + hop) % N;
                    ++_nFrames;
                    continue;
                }
            }
            const Real* const squares = startUpSquares();
            for (std::size_t m = 0UZ; m < M; ++m) {
                const Complex* const bins = frame.data() + m * F;
                std::copy_n(bins, F, _spectrum.begin());
                if constexpr (!gr::meta::complex_like<T>) {
                    for (std::size_t bin = F; bin < N; ++bin) {
                        _spectrum[bin] = std::conj(bins[N - bin]);
                    }
                }
                _fft.computeInverse(_spectrum, _samples);
                T* const row = _accumulated.data() + m * N;
                for (std::size_t j = 0UZ; j < N; ++j) {
                    const Complex sample = _samples[j] * _windowCoefficients[j];
                    if constexpr (gr::meta::complex_like<T>) {
                        row[(_head + j) % N] += sample;
                    } else {
                        row[(_head + j) % N] += sample.real();
                    }
                }
                T* const done = completed->data() + m * hop;
                for (std::size_t j = 0UZ; j < hop; ++j) {
                    T& slot = row[(_head + j) % N];
                    done[j] = squares[j] > Real{0} ? slot / squares[j] : T{};
                    slot    = T{};
                }
            }
            _head = (_head + hop) % N;
            ++_nFrames;
        }
        std::ignore = input.consume(k);
        output.publish(k);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("window")) {
            _windowCoefficients.clear();
        }
    }

    /** the hop of frame k of N samples: `gr:sample_rate` / `gr:frame_rate` of the frame's tag, else N/2 */
    [[nodiscard]] std::size_t hopSamplesOf(InputSpanLike auto& input, std::size_t k, std::size_t N) {
        const std::string_view sampleRateKey{gr::tag::SAMPLE_RATE.key()};
        const std::string_view frameRateKey{gr::tag::FRAME_RATE.key()};
        for (const auto& [relativeIndex, tagMap] : input.tags()) {
            const auto& tag = tagMap.get();
            if (std::cmp_equal(relativeIndex, k) && tag.contains(sampleRateKey)) {
                const float sampleRate = tag.template value_or<float>(sampleRateKey, 0.f);
                const float frameRate  = tag.template value_or<float>(frameRateKey, sampleRate);
                _hopFromRates          = frameRate > 0.f ? static_cast<std::size_t>(std::lround(sampleRate / frameRate)) : 0UZ;
            }
        }
        return _hopFromRates != 0UZ ? _hopFromRates : N / 2UZ;
    }

    /** window, overlap normalisation and accumulator for M rows of N samples `hop` apart; false, with the error reported, when they cannot reconstruct */
    [[nodiscard]] bool prepare(std::size_t M, std::size_t N, std::size_t hop) {
        if (_windowCoefficients.size() == N && _M == M && _hop == hop) {
            return true;
        }
        if (hop > N) [[unlikely]] {
            this->emitErrorMessage("processBulk()", std::format("hop {} for frames of {} samples", hop, N));
            this->requestStop();
            return false;
        }
        _windowCoefficients.assign(N, Real{1});
        gr::algorithm::window::create(std::span<Real>(_windowCoefficients), window, gr::algorithm::window::kDefaultBeta<Real>, true);
        const std::size_t nRows = (N + hop - 1UZ) / hop;
        _overlapSquares.assign(nRows * hop, Real{0});
        for (std::size_t n = 0UZ; n < N; ++n) {
            for (std::size_t c = n / hop; c < nRows; ++c) {
                _overlapSquares[c * hop + n % hop] += _windowCoefficients[n] * _windowCoefficients[n];
            }
        }
        const std::span<const Real> steadyState(_overlapSquares.data() + (nRows - 1UZ) * hop, hop);
        if (std::ranges::any_of(steadyState, [](Real sum) { return sum <= Real{0}; })) [[unlikely]] {
            _windowCoefficients.clear();
            this->emitErrorMessage("processBulk()", std::format("the window's squares leave a gap at hop {}: no reconstruction", hop));
            this->requestStop();
            return false;
        }
        _accumulated.assign(M * N, T{});
        _spectrum.resize(N);
        _samples.resize(N);
        _M       = M;
        _hop     = hop;
        _head    = 0UZ;
        _nFrames = 0UZ;
        return true;
    }

    /** the normalisation for the hop the current frame completes: fewer windows overlap it during start-up */
    [[nodiscard]] const Real* startUpSquares() const noexcept {
        const std::size_t nRows = _overlapSquares.size() / _hop;
        return _overlapSquares.data() + std::min(_nFrames, nRows - 1UZ) * _hop;
    }
};

} // namespace gr::blocks::fft

#endif // GNURADIO_FOURIER_STFT_HPP
