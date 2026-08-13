#ifndef GNURADIO_FFT_HPP
#define GNURADIO_FFT_HPP

#include <algorithm>
#include <bit>
#include <format>
#include <memory>
#include <span>
#include <utility>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/WindowGeometry.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/device/SyclFFT.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::blocks::fft {

namespace detail {
template<typename T>
concept FftPrecision = std::same_as<T, float> || std::same_as<T, double>;

template<typename U>
concept FftSpectrumOutput = requires { typename U::value_type; } && FftPrecision<typename U::value_type> && std::same_as<U, DataSet<typename U::value_type>>;

template<typename T, typename U>
concept FftStreamPair = FftPrecision<T> && std::same_as<U, std::complex<T>>;

template<typename T, typename U>
concept FftSpectrumPair = FftSpectrumOutput<U> && (std::same_as<T, typename U::value_type> || std::same_as<T, std::complex<typename U::value_type>>);

/// both specialisations derive their chunk sizes from `fft_size`, so a settings change that also names one is
/// contradicting itself. Saying so beats silently discarding the caller's value or slicing by a size the
/// framework does not plan for.
template<typename TBlock>
void refuseChunkSizeOverride(TBlock& block, const property_map& newSettings, gr::Size_t derived) {
    for (const auto& key : {"input_chunk_size", "output_chunk_size"}) {
        const auto it = newSettings.find(std::string_view{key});
        if (it == newSettings.end()) {
            continue;
        }
        if (const auto* requested = (*it).second.template get_if<gr::Size_t>(); requested != nullptr && *requested != derived) {
            block.emitErrorMessage("settingsChanged()", std::format("{} is derived from fft_size ({}) and cannot be set to {}", key, derived, *requested));
        }
    }
}
constexpr bool isPowerOfTwo(std::size_t n) noexcept { return std::has_single_bit(n); }

} // namespace detail
using detail::refuseChunkSizeOverride;

GR_REGISTER_BLOCK("gr::blocks::fft::FFT", gr::blocks::fft::FFT, [T], [ float, double ])
GR_REGISTER_BLOCK("gr::blocks::fft::FFT", gr::blocks::fft::FFT, ([T], [U]), [float], [gr::DataSet<float>])
GR_REGISTER_BLOCK("gr::blocks::fft::FFT", gr::blocks::fft::FFT, ([T], [U]), [double], [gr::DataSet<double>])
GR_REGISTER_BLOCK("gr::blocks::fft::FFT", gr::blocks::fft::FFT, ([T], [U]), [std::complex<float>], [gr::DataSet<float>])
GR_REGISTER_BLOCK("gr::blocks::fft::FFT", gr::blocks::fft::FFT, ([T], [U]), [std::complex<double>], [gr::DataSet<double>])

template<typename T, typename U = std::complex<T>>
requires(detail::FftStreamPair<T, U> || detail::FftSpectrumPair<T, U>)
struct FFT : gr::Block<FFT<T, U>, gr::Resampling<1UZ, 1UZ, false>, gr::Stride<>> { // 1:1, runtime chunk = fft_size (set in settingsChanged)
    using Description = Doc<R""(raw forward/inverse FFT, emitting `std::complex<T>` rather than a `DataSet`.

Batch count is auto-detected from the input size, and the transform dispatches to the CPU `SimdFFT` or to SYCL
(Stockham) per `compute_domain`. The `FFT<T, DataSet<P>>` partial specialisation below is the spectrum mode, which
emits one windowed magnitude/phase `DataSet` per transform instead of the raw bins.

 * J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series",
   Math. Comput., vol. 19, no. 90, pp. 297-301, 1965.)"">;

    using ComplexType = std::complex<T>;

    PortIn<ComplexType>  in;
    PortOut<ComplexType> out;

    Annotated<gr::Size_t, "fft size", Limits<8UZ, 1048576UZ, &detail::isPowerOfTwo>> fft_size = 4096UZ;
    Annotated<bool, "inverse">                                                       inverse  = false;

    GR_MAKE_REFLECTABLE(FFT, in, out, fft_size, inverse);

    gr::algorithm::FFT<ComplexType, ComplexType> _cpuFft;
    gr::device::SyclFFT                          _syclFft;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        refuseChunkSizeOverride(*this, newSettings, static_cast<gr::Size_t>(fft_size));
        // whole transforms only: the seam consumes exactly `count`, so a partial batch drifts the windows
        this->input_chunk_size  = fft_size;
        this->output_chunk_size = fft_size;
    }

    gr::work::Status processBulk(InputSpanLike auto& inSpan, OutputSpanLike auto& outSpan) {
        const auto N      = static_cast<std::size_t>(fft_size);
        const auto frames = gr::windowGeometry(*this, inSpan.size(), outSpan.size());
        if (frames.nWindows == 0UZ) {
            std::ignore = inSpan.consume(0);
            outSpan.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        const auto hop      = frames.hop;
        const auto nBatches = frames.nWindows;
        const auto total    = nBatches * N;

        for (std::size_t b = 0; b < nBatches; ++b) {
            auto inSlice  = std::span<const ComplexType>(inSpan.data() + b * hop, N);
            auto outSlice = std::span<ComplexType>(outSpan.data() + b * N, N);

            if (inverse) {
                std::ranges::transform(inSlice, outSlice.begin(), [](auto z) { return std::conj(z); });
                _cpuFft.compute(outSlice, outSlice);
                T invN = T(1) / static_cast<T>(N);
                std::ranges::transform(outSlice, outSlice.begin(), [invN](auto z) { return std::conj(z) * invN; });
            } else {
                _cpuFft.compute(inSlice, outSlice);
            }
        }

        std::ignore = inSpan.consume(total);
        outSpan.publish(total);
        return work::Status::OK;
    }

    gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& inSpan, OutputSpanLike auto& outSpan)
    requires std::same_as<T, float> // gr::device::SyclFFT is a float-only tier; double precision stays on the host
    {
        const auto N      = static_cast<std::size_t>(fft_size);
        const auto frames = gr::windowGeometry(*this, inSpan.size(), outSpan.size());
        if (frames.nWindows == 0UZ) {
            std::ignore = inSpan.consume(0);
            outSpan.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        _syclFft.init(ctx, N);

        const auto hop      = frames.hop;
        const auto nBatches = frames.nWindows;
        const auto total    = nBatches * N;

        // no wait: the queue is in-order and the transform's own final wait covers these copies
        if (hop == N) {
            ctx.copy(outSpan.data(), inSpan.data(), total * sizeof(ComplexType), false);
        } else { // overlapping frames are not contiguous in the input, so each is gathered on its own
            for (std::size_t b = 0; b < nBatches; ++b) {
                ctx.copy(outSpan.data() + b * N, inSpan.data() + b * hop, N * sizeof(ComplexType), false);
            }
        }

        auto outData = std::span<gr::complex<T>>{reinterpret_cast<gr::complex<T>*>(outSpan.data()), total};
        if (inverse) {
            _syclFft.inverseBatch(ctx, outData, N);
        } else {
            _syclFft.forwardBatch(ctx, outData, N);
        }

        std::ignore = inSpan.consume(total);
        outSpan.publish(total);
        return gr::work::Status::OK;
    }
};

// N:1 resampling, one DataSet<P> per fft_size samples; a floating-point T selects the half spectrum
template<typename T, typename P>
requires detail::FftSpectrumPair<T, DataSet<P>>
struct FFT<T, DataSet<P>> : gr::Block<FFT<T, DataSet<P>>, gr::Resampling<1UZ, 1UZ, false>> {
    using Description = Doc<R""(FFT spectrum mode: `N:1` resampling to one windowed magnitude/phase `DataSet<P>` per `fft_size` input samples.

Real-to-complex `T` gives the half spectrum `[DC, +fs/2]`, complex-like `T` the full fftshifted spectrum
`[-fs/2, +fs/2)`. Device dispatch covers only `T == std::complex<float>`. The primary template above is the raw
streaming form, which emits the bins themselves and applies no window.

 * J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series",
   Math. Comput., vol. 19, no. 90, pp. 297-301, 1965.
 * f. j. harris, "On the use of windows for harmonic analysis with the discrete Fourier transform",
   Proc. IEEE, vol. 66, no. 1, pp. 51-83, 1978.)"">;

    using U           = DataSet<P>;
    using value_type  = P;
    using InDataType  = std::conditional_t<gr::meta::complex_like<T>, std::complex<value_type>, value_type>;
    using OutDataType = std::complex<value_type>;

    constexpr static bool computeFullSpectrum = gr::meta::complex_like<T>;

    PortIn<T>  in;
    PortOut<U> out;

    Annotated<gr::Size_t, "fft size", Limits<8UZ, 1048576UZ, &detail::isPowerOfTwo>>        fft_size = 4096UZ;
    Annotated<gr::algorithm::window::Type, "window", Doc<gr::algorithm::window::TypeNames>> window   = gr::algorithm::window::Type::Hann;
    Annotated<bool, "output in dB", Doc<"calculate output in decibels">>                    output_in_db{false};
    Annotated<bool, "output in deg", Doc<"calculate phase in degrees">>                     output_in_deg{false};
    Annotated<bool, "unwrap phase", Doc<"calculate unwrapped phase">>                       unwrap_phase{false};
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>                  sample_rate = 1.f;
    Annotated<std::string, "signal name", Visible>                                          signal_name = "unknown signal";
    Annotated<std::string, "signal unit", Visible, Doc<"signal's physical SI unit">>        signal_unit = "a.u.";
    Annotated<float, "signal min", Doc<"signal physical min. (e.g. DAQ) limit">>            signal_min  = -std::numeric_limits<float>::max();
    Annotated<float, "signal max", Doc<"signal physical max. (e.g. DAQ) limit">>            signal_max  = +std::numeric_limits<float>::max();

    // reflected so the pmr-field migration re-seats it onto device memory, giving the kernel a visible window
    std::pmr::vector<value_type> window_coefficients;

    GR_MAKE_REFLECTABLE(FFT, in, out, fft_size, window, output_in_db, output_in_deg, unwrap_phase, sample_rate, signal_name, signal_unit, signal_min, signal_max, window_coefficients);

    gr::algorithm::FFT<T, OutDataType> _fftImpl{};
    std::vector<InDataType>            _inData{};
    std::vector<OutDataType>           _outData{};
    std::vector<value_type>            _magnitudeSpectrum{};
    std::vector<value_type>            _phaseSpectrum{};
    gr::device::DeviceContext*         _deviceCtx = nullptr; // owned by the registry, outlives the block
    gr::device::SyclFFT                _syclFft{};           // float-only tier; only ever used when T == std::complex<float>
    gr::device::DeviceBuffer           _deviceComplex{};
    gr::device::DeviceBuffer           _deviceWindow{};
    std::size_t                        _deviceWindowCapacity{0UZ};
    gr::device::DeviceBuffer           _deviceMagnitude{};
    gr::device::DeviceBuffer           _devicePhase{};
    std::size_t                        _deviceCapacity = 0;

    ~FFT() { freeDeviceScratch(); }

    gr::work::Status processBulk(InputSpanLike auto& inSpan, OutputSpanLike auto& outSpan) {
        const auto N        = static_cast<std::size_t>(fft_size);
        const auto nBatches = std::min(inSpan.size() / N, outSpan.size());
        if (nBatches == 0) {
            std::ignore = inSpan.consume(0);
            outSpan.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        for (std::size_t b = 0; b < nBatches; ++b) {
            computeSpectrum(std::span<const T>(inSpan.data() + b * N, N));
            outSpan[b] = assembleDataSet();
        }

        std::ignore = inSpan.consume(nBatches * N);
        outSpan.publish(nBatches);
        return work::Status::OK;
    }

    gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& inSpan, OutputSpanLike auto& outSpan)
    requires std::same_as<T, std::complex<float>> // gr::device::SyclFFT is a float-only, complex-only tier
    {
        const auto N        = static_cast<std::size_t>(fft_size);
        const auto nBatches = std::min(inSpan.size() / N, outSpan.size());
        if (nBatches == 0) {
            std::ignore = inSpan.consume(0);
            outSpan.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const std::size_t total = nBatches * N;

        _deviceCtx = &ctx;
        _syclFft.init(ctx, N);
        ensureDeviceScratch(ctx, total);

        auto* const complexDev = _deviceComplex.devicePointer<gr::complex<value_type>>();
        auto* const magDev     = _deviceMagnitude.devicePointer<value_type>();
        auto* const phaseDev   = _devicePhase.devicePointer<value_type>();
        auto* const windowDev  = deviceWindow(ctx, N);
        if (complexDev == nullptr || magDev == nullptr || phaseDev == nullptr || windowDev == nullptr) {
            this->emitErrorMessage("processBulk()", "the device could not serve the scratch this transform needs");
            return work::Status::ERROR;
        }

        ctx.copy(complexDev, inSpan.data(), total * sizeof(T), false); // in-order queue: the windowing kernel below is ordered after it

        gr::device::parallelFor(ctx, total, [complexDev, windowDev, N] GR_DEVICE_LAMBDA(std::size_t idx) { complexDev[idx] = gr::algorithm::fft::applyWindowOne(complexDev[idx], windowDev[idx % N]); });

        _syclFft.forwardBatch(ctx, std::span<gr::complex<value_type>>{complexDev, total}, N);

        // unwrap must precede the fftshift and any degree conversion, matching computePhaseSpectrum's order
        auto* const fftOut     = reinterpret_cast<std::complex<value_type>*>(complexDev);
        const bool  outputInDb = output_in_db;
        gr::device::parallelFor(ctx, total, [fftOut, magDev, phaseDev, N, outputInDb] GR_DEVICE_LAMBDA(std::size_t idx) {
            magDev[idx]   = gr::algorithm::fft::computeMagnitudeOne(fftOut[idx], N, outputInDb); // complex input: a two-sided spectrum, nothing folds
            phaseDev[idx] = gr::algorithm::fft::computePhaseOne(fftOut[idx]);
        });

        for (std::size_t b = 0; b < nBatches; ++b) {
            // one wait for the three: the queue is in-order, so awaiting the last awaits all of them
            ctx.copy(_outData.data(), complexDev + b * N, N * sizeof(OutDataType), false);
            ctx.copy(_magnitudeSpectrum.data(), magDev + b * N, N * sizeof(value_type), false);
            ctx.copy(_phaseSpectrum.data(), phaseDev + b * N, N * sizeof(value_type));

            if (unwrap_phase) {
                gr::algorithm::fft::unwrapPhase(std::span<value_type>(_phaseSpectrum));
            }
            if (output_in_deg) {
                std::ranges::transform(_phaseSpectrum, _phaseSpectrum.begin(), [](value_type phase) { return gr::algorithm::fft::radToDeg(phase); });
            }

            const auto halfN = std::ssize(_magnitudeSpectrum) / 2;
            std::ranges::rotate(_magnitudeSpectrum, _magnitudeSpectrum.begin() + halfN);
            std::ranges::rotate(_phaseSpectrum, _phaseSpectrum.begin() + halfN);

            outSpan[b] = assembleDataSet();
        }

        std::ignore = inSpan.consume(total);
        outSpan.publish(nBatches);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, property_map& newSettings, property_map& forwardSettings) {
        // dropping sample_rate from the forwarded settings opts out of the N:1 rescale, which would corrupt
        // the axis; the emitted DataSet carries its own
        forwardSettings.erase(gr::tag::SAMPLE_RATE.shortKey());

        refuseChunkSizeOverride(*this, newSettings, static_cast<gr::Size_t>(fft_size));
        this->input_chunk_size = static_cast<gr::Size_t>(fft_size);

        if (!newSettings.contains("fft_size") && !newSettings.contains("window")) {
            return;
        }
        _inData.clear(); // force the rebuild below: `window` may have changed at an unchanged size
        ensureHostScratch(static_cast<std::size_t>(fft_size));
    }

    /// the scratch one transform needs. Called where it is USED as well as from `settingsChanged`, because the
    /// framework skips that callback entirely for a block emplaced with no settings -- which would otherwise
    /// leave every buffer empty and transform into it.
    void ensureHostScratch(std::size_t n) {
        if (window_coefficients.size() != n) {
            window_coefficients.resize(n); // reflected, so a caller can set it at any length; the kernels index it by fft_size
            gr::algorithm::window::create(std::span<value_type>(window_coefficients), window);
        }
        if (_inData.size() == n) {
            return;
        }
        window_coefficients.resize(n);
        gr::algorithm::window::create(std::span<value_type>(window_coefficients), window);

        _inData.resize(n);
        _outData.resize(n); // _fftImpl.compute() always returns the full (Hermitian-symmetric for real input) spectrum
        _magnitudeSpectrum.resize(computeFullSpectrum ? n : (n / 2UZ + 1UZ));
        _phaseSpectrum.resize(computeFullSpectrum ? n : (n / 2UZ + 1UZ));
    }

    void computeSpectrum(std::span<const T> inputChunk) {
        ensureHostScratch(static_cast<std::size_t>(fft_size));
        if constexpr (std::is_same_v<T, InDataType>) {
            std::copy_n(inputChunk.begin(), fft_size, _inData.begin());
        } else {
            std::ranges::transform(inputChunk, _inData.begin(), [](const T c) { return static_cast<InDataType>(c); });
        }

        gr::algorithm::fft::applyWindow(std::span<InDataType>(_inData), std::span<const value_type>(window_coefficients));

        _outData           = _fftImpl.compute(_inData);
        _magnitudeSpectrum = gr::algorithm::fft::computeMagnitudeSpectrum(_outData, _magnitudeSpectrum, gr::algorithm::fft::ConfigMagnitude{.computeHalfSpectrum = !computeFullSpectrum, .includeNyquist = true, .outputInDb = output_in_db, .shiftSpectrum = true});
        _phaseSpectrum     = gr::algorithm::fft::computePhaseSpectrum(_outData, _phaseSpectrum, gr::algorithm::fft::ConfigPhase{.computeHalfSpectrum = !computeFullSpectrum, .includeNyquist = true, .outputInDeg = output_in_deg, .unwrapPhase = unwrap_phase, .shiftSpectrum = true});
    }

    constexpr U assembleDataSet() {
        U ds{};
        ds.timestamp = 0;
        const std::size_t     N{_magnitudeSpectrum.size()};
        constexpr std::size_t nSignals = 4;

        ds.extents = {static_cast<int32_t>(N)};
        ds.layout  = gr::LayoutRight{};

        ds.axis_names = {"Frequency"};
        ds.axis_units = {"Hz"};
        ds.axis_values.resize(ds.nDimensions());
        ds.axis_values[0UZ].resize(N);

        auto const freqWidth = static_cast<value_type>(sample_rate) / static_cast<value_type>(fft_size);
        if constexpr (computeFullSpectrum) { // complex-valued FFT output: symmetric spectrum [-fs/2, +fs/2]
            auto const freqOffset = static_cast<value_type>(N / 2) * freqWidth;
            std::ranges::transform(std::views::iota(0UZ, N), std::ranges::begin(ds.axisValues(0UZ)), [freqWidth, freqOffset](const auto i) { return static_cast<value_type>(i) * freqWidth - freqOffset; });
        } else { // real-valued FFT output: [DC (0), +fs/2] (only upper half, negative is a point-symmetric copy)
            std::ranges::transform(std::views::iota(0UZ, N), std::ranges::begin(ds.axisValues(0UZ)), [freqWidth](const auto i) { return static_cast<value_type>(i) * freqWidth; });
        }

        ds.signal_names      = {std::format("Magnitude({})", signal_name), std::format("Phase({})", signal_name), std::format("Re(FFT({}))", signal_name), std::format("Im(FFT({}))", signal_name)};
        ds.signal_quantities = {"Magnitude(FFT)", "Phase(FFT)", "Re(FFT)", "Im(FFT)"};
        ds.signal_units      = {std::format("{}/√Hz", signal_unit), "rad", std::format("Re{}", signal_unit), std::format("Im{}", signal_unit)};
        assert(ds.signal_names.size() == nSignals);

        ds.signal_values.resize(nSignals * N);
        ds.signal_ranges.resize(nSignals);

        assert(_magnitudeSpectrum.size() == ds.signalValues(0UZ).size());
        std::copy_n(_magnitudeSpectrum.begin(), N, ds.signalValues(0UZ).begin());
        assert(_phaseSpectrum.size() == ds.signalValues(1UZ).size());
        std::copy_n(_phaseSpectrum.begin(), N, ds.signalValues(1UZ).begin());

        if constexpr (computeFullSpectrum) { // complex in -> complex-out FFT: full spectrum, fftshifted to align with magnitude/phase
            std::ranges::transform(std::views::iota(0UZ, N), ds.signalValues(2UZ).begin(), [this, N](std::size_t i) { return std::real(_outData[gr::algorithm::fft::fftShiftIndex(i, N)]); });
            std::ranges::transform(std::views::iota(0UZ, N), ds.signalValues(3UZ).begin(), [this, N](std::size_t i) { return std::imag(_outData[gr::algorithm::fft::fftShiftIndex(i, N)]); });
        } else { // real-valued FFT -- DC..Nyquist inclusive, same natural bin order as magnitude/phase
            auto fftHalfSpectrum = std::span{_outData}.first(N);
            std::ranges::transform(fftHalfSpectrum, ds.signalValues(2UZ).begin(), [](const auto& c) { return std::real(c); });
            std::ranges::transform(fftHalfSpectrum, ds.signalValues(3UZ).begin(), [](const auto& c) { return std::imag(c); });
        }

        for (std::size_t i = 0; i < nSignals; i++) {
            const auto mm       = std::minmax_element(std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(i * N)), std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>((i + 1U) * N)));
            ds.signal_ranges[i] = {*mm.first, *mm.second};
        }

        const auto& meta_info = property_map{                                                                                                                              //
            {std::pmr::string("sample_rate"), Value(sample_rate)},                                                                                                         //
            {std::pmr::string("window"), Value(std::pmr::string(gr::meta::enumName(window.value).value_or("")))}, {std::pmr::string("output_in_db"), Value(output_in_db)}, //
            {std::pmr::string("output_in_deg"), Value(output_in_deg)},                                                                                                     //
            {std::pmr::string("unwrap_phase"), Value(unwrap_phase)},                                                                                                       //
            {std::pmr::string("input_chunk_size"), Value(this->input_chunk_size)},                                                                                         //
            {std::pmr::string("output_chunk_size"), gr::Value(this->output_chunk_size)},                                                                                   //
            {std::pmr::string("stride"), gr::Value(this->stride)}};

        ds.meta_information.resize(nSignals);
        for (std::size_t i = 0UZ; i < nSignals; i++) {
            ds.meta_information[i] = meta_info;
        }

        ds.timing_events.resize(nSignals);

        return ds;
    }

    /// driven straight from `processBulk(ctx, ...)` the coefficients are still host-side, where a kernel cannot follow
    [[nodiscard]] const value_type* deviceWindow(gr::device::DeviceContext& ctx, std::size_t n) {
        const value_type* host = window_coefficients.data();
        if (ctx.isDeviceAccessible(host)) {
            return host;
        }
        if (_deviceWindowCapacity < n) {
            ctx.deallocate(_deviceWindow);
            _deviceWindow         = ctx.allocateDevice<value_type>(n);
            _deviceWindowCapacity = _deviceWindow ? n : 0UZ;
        }
        auto* const staged = _deviceWindow.devicePointer<value_type>();
        if (staged == nullptr) {
            return nullptr; // the caller fails the work call: a kernel cannot follow the host pointer
        }
        ctx.copyHostToDevice(host, _deviceWindow, n);
        return staged;
    }

    void ensureDeviceScratch(gr::device::DeviceContext& ctx, std::size_t total) {
        if (_deviceCapacity >= total) {
            return;
        }
        freeDeviceScratch();
        _deviceComplex   = ctx.allocateDevice<gr::complex<value_type>>(total);
        _deviceMagnitude = ctx.allocateDevice<value_type>(total);
        _devicePhase     = ctx.allocateDevice<value_type>(total);
        // allocateDevice answers with an invalid buffer when it cannot serve; latching the capacity anyway would
        // short-circuit every later call and leave the kernels writing through null
        _deviceCapacity = (_deviceComplex && _deviceMagnitude && _devicePhase) ? total : 0UZ;
    }

    void freeDeviceScratch() {
        if (_deviceCtx == nullptr) {
            return;
        }
        _deviceCtx->deallocate(_deviceWindow);
        _deviceWindow         = gr::device::DeviceBuffer{};
        _deviceWindowCapacity = 0UZ;
        _deviceCtx->deallocate(_deviceComplex);
        _deviceCtx->deallocate(_deviceMagnitude);
        _deviceCtx->deallocate(_devicePhase);
        _deviceComplex   = gr::device::DeviceBuffer{};
        _deviceMagnitude = gr::device::DeviceBuffer{};
        _devicePhase     = gr::device::DeviceBuffer{};
        _deviceCapacity  = 0;
    }
};

} // namespace gr::blocks::fft

#endif // GNURADIO_FFT_HPP
