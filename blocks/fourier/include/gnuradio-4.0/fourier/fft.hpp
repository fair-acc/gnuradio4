#ifndef GNURADIO_FFT_HPP
#define GNURADIO_FFT_HPP

#include <execution>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/algorithm/fourier/fftw.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>

namespace gr::blocks::fft {

using namespace gr;

template<typename T>
struct OutputDataSet {
    using type = DataSet<T>;
};

template<gr::meta::complex_like T>
struct OutputDataSet<T> {
    using type = DataSet<typename T::value_type>;
};

template<typename T, typename U = OutputDataSet<T>::type, template<typename, typename> typename FourierAlgorithm = gr::algorithm::FFT>
requires((gr::meta::complex_like<T> || std::floating_point<T>) && (std::is_same_v<U, DataSet<float>> || std::is_same_v<U, DataSet<double>>))
struct FFT : public Block<FFT<T, U, FourierAlgorithm>, Resampling<1024LU, 1LU>> {
    using Description = Doc<R""(
@brief Performs a (Fast) Fourier Transform (FFT) on the given input data.

The FFT block is capable of performing Fourier Transform computations on real or complex data,
and populates a DataSet with the results, including real, imaginary, magnitude, and phase
spectrum of the signal. For details see:
 * https://en.wikipedia.org/wiki/Fourier_transform
 * https://en.wikipedia.org/wiki/Discrete-time_Fourier_transform
 * https://en.wikipedia.org/wiki/Fast_Fourier_transform

On the choice of window (mathematically aka. apodisation) functions
(SA = Side-lobe Attenuation (near ... far), FR = Frequency Resolution, MR = Magnitude Response):
 * None (0):
   - SA: ~13 ... 40 dB | FR: Narrow (finest distinction between frequencies) | MR: Large ripple.
   - No window applied, same as Rectangular
 * Rectangular (1):
   - SA: ~13 ... 40 dB | FR: Narrow | MR: Large ripple.
   - No window applied, same as None
 * Hamming (2):
   - SA: ~41 ... 60 dB | FR: Moderate | MR: ~0.019% ripple.
   - Balanced between frequency resolution and side-lobe attenuation.
   - Best for: General purpose.
 * Hann (3, default):
   - SA: ~31 ... 105 dB | FR: Narrower than Hamming | MR: ~0.036% ripple.
   - Good frequency resolution, relatively low side-lobe.
   - Best for: Spectral analysis, especially when resolving closely spaced frequencies or for
     ensuring minimal leakage when multiple signals are present. This makes it an ideal default
     choice for most applications.
 * HannExp (4):
   - SA: ~50 ... 90 dB (estimate) | FR: Moderate | MR: Variable.
 * Blackman (5):
   - SA: ~58 ... 80 dB | FR: Wider than Hamming | MR: ~0.002% ripple.
   - Reduced leakage at the expense of a wider main-lobe.
 * Nuttall (6):
   - SA: ~64 ... 90 dB | FR: Wider than Blackman | MR: ~0.001% ripple.
   - Very low side-lobe but reduced frequency resolution.
   - Best for: Spectral purity.
 * BlackmanHarris (7):
   - SA: ~67 ... 92 dB | FR: Similar to Blackman | MR: ~0.0002% ripple.
   - High side-lobe attenuation, lesser frequency resolution than Hamming.
 * BlackmanNuttall (8):
   - SA: ~65 ... 88 dB | FR: Similar to Blackman | MR: ~0.0001% ripple.
   - Blend of Blackman & Nuttall properties.
 * FlatTop (9):
   - SA: ~44 ... 70 dB | FR: Widest among all | MR: Very precise (minimal ripple).
   - Precision amplitude measurements but poor frequency resolution.
   - Best for: Precise amplitude measurements.
 * Exponential (10):
   - SA: Variable | FR: Moderate | MR: Variable.
   - Best for: signals with decaying amplitudes.
 * Kaiser (11):
   - SA: Adjustable | FR: Adjustable | MR: Variable.
   - Customizable side-lobe attenuation and frequency resolution trade-off.
   - Best for: Custom trade-offs between SA and FR.

@tparam T type of the input signal.
@tparam U type of the output data (presently limited to DataSet<float> and DataSet<double>)
@tparam FourierAlgorithm the specific algorithm used to perform the Fourier Transform (can be DFT, FFT, FFTW).
)"">;
    using value_type  = U::value_type;
    using InDataType  = std::conditional_t<gr::meta::complex_like<T>, std::complex<value_type>, value_type>;
    using OutDataType = std::complex<value_type>;

    PortIn<T>                         in{};
    PortOut<U, RequiredSamples<1, 1>> out{};

    FourierAlgorithm<T, std::complex<typename U::value_type>> _fftImpl{};
    gr::algorithm::window::Type                               _windowType = gr::algorithm::window::Type::Hann;
    std::vector<value_type>                                   _window     = gr::algorithm::window::create<value_type>(_windowType, 1024U);

    // settings
    const std::string                                                                algorithm = gr::meta::type_name<decltype(_fftImpl)>();
    Annotated<gr::Size_t, "FFT size", Doc<"FFT size">>                               fftSize{1024U};
    Annotated<std::string, "window type", Doc<gr::algorithm::window::TypeNames>>     window = std::string(magic_enum::enum_name(_windowType));
    Annotated<bool, "output in dB", Doc<"calculate output in decibels">>             outputInDb{false};
    Annotated<bool, "output in deg", Doc<"calculate phase in degrees">>              outputInDeg{false};
    Annotated<bool, "unwrap phase", Doc<"calculate unwrapped phase">>                unwrapPhase{false};
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>           sample_rate = 1.f;
    Annotated<std::string, "signal name", Visible>                                   signal_name = "unknown signal";
    Annotated<std::string, "signal unit", Visible, Doc<"signal's physical SI unit">> signal_unit = "a.u.";
    Annotated<float, "signal min", Doc<"signal physical min. (e.g. DAQ) limit">>     signal_min  = -std::numeric_limits<float>::max();
    Annotated<float, "signal max", Doc<"signal physical max. (e.g. DAQ) limit">>     signal_max  = +std::numeric_limits<float>::max();

    GR_MAKE_REFLECTABLE(FFT, in, out, algorithm, fftSize, window, outputInDb, outputInDeg, unwrapPhase, sample_rate, signal_name, signal_unit, signal_min, signal_max);

    // semi-private caching vectors (need to be public for unit-test) -> TODO: move to FFT implementations, casting from T -> U::value_type should be done there
    std::vector<InDataType>  _inData             = std::vector<InDataType>(fftSize, 0);
    std::vector<OutDataType> _outData            = std::vector<OutDataType>(gr::meta::complex_like<T> ? fftSize.value : (1U + fftSize.value / 2U), 0);
    std::vector<value_type>  _magnitudeSpectrum  = std::vector<value_type>(gr::meta::complex_like<T> ? fftSize.value : (1U + fftSize.value / 2U), 0);
    std::vector<value_type>  _phaseSpectrum      = std::vector<value_type>(gr::meta::complex_like<T> ? fftSize.value : (1U + fftSize.value / 2U), 0);
    constexpr static bool    computeFullSpectrum = gr::meta::complex_like<T>;

    void settingsChanged(const property_map& /*old_settings*/, const property_map& newSettings) noexcept {
        if (!newSettings.contains("fftSize") && !newSettings.contains("window")) {
            // do need to only handle interdependent settings -> can early return
            return;
        }

        const std::size_t newSize = fftSize;
        in.max_samples            = newSize;
        in.min_samples            = newSize;
        this->input_chunk_size    = newSize;
        _window.resize(newSize, 0);

        _windowType = magic_enum::enum_cast<gr::algorithm::window::Type>(window, magic_enum::case_insensitive).value_or(_windowType);
        gr::algorithm::window::create(_window, _windowType);

        // N.B. this should become part of the Fourier transform implementation
        _inData.resize(fftSize, 0);
        _outData.resize(computeFullSpectrum ? newSize : (1U + newSize / 2), 0);
        _magnitudeSpectrum.resize(computeFullSpectrum ? newSize : (newSize / 2), 0);
        _phaseSpectrum.resize(computeFullSpectrum ? newSize : (newSize / 2), 0);
    }

    [[nodiscard]] constexpr work::Status processBulk(std::span<const T> input, std::span<U> output) {
        if constexpr (std::is_same_v<T, InDataType>) {
            std::copy_n(input.begin(), fftSize, _inData.begin());
        } else {
            std::ranges::transform(input.begin(), input.end(), _inData.begin(), [](const T c) { return static_cast<InDataType>(c); });
        }

        // apply window function
        for (std::size_t i = 0U; i < fftSize; i++) {
            if constexpr (gr::meta::complex_like<T>) {
                _inData[i].real(_inData[i].real() * _window[i]);
                _inData[i].imag(_inData[i].imag() * _window[i]);
            } else {
                _inData[i] *= _window[i];
            }
        }

        _outData           = _fftImpl.compute(_inData);
        _magnitudeSpectrum = gr::algorithm::fft::computeMagnitudeSpectrum(_outData, _magnitudeSpectrum, algorithm::fft::ConfigMagnitude{.computeHalfSpectrum = !computeFullSpectrum, .outputInDb = outputInDb});
        _phaseSpectrum     = gr::algorithm::fft::computePhaseSpectrum(_outData, _phaseSpectrum, algorithm::fft::ConfigPhase{.computeHalfSpectrum = !computeFullSpectrum, .outputInDeg = outputInDeg, .unwrapPhase = unwrapPhase});

        output[0] = createDataset();

        return work::Status::OK;
    }

    constexpr U createDataset() {
        U ds{};
        ds.timestamp = 0;
        const std::size_t N{_magnitudeSpectrum.size()};
        const std::size_t dim = 5;

        ds.axis_names   = {"Frequency", "Re(FFT)", "Im(FFT)", "Magnitude", "Phase"};
        ds.axis_units   = {"Hz", signal_unit, fmt::format("i{}", signal_unit), fmt::format("{}/√Hz", signal_unit), "rad"};
        ds.extents      = {dim, static_cast<int32_t>(N)};
        ds.layout       = gr::LayoutRight{};
        ds.signal_names = {signal_name, fmt::format("Re(FFT({}))", signal_name), fmt::format("Im(FFT({}))", signal_name), fmt::format("Magnitude({})", signal_name), fmt::format("Phase({})", signal_name)};
        ds.signal_units = {"Hz", signal_unit, fmt::format("i{}", signal_unit), fmt::format("{}/√Hz", signal_unit), "rad"};

        ds.signal_values.resize(dim * N);
        auto const freqWidth = static_cast<value_type>(sample_rate) / static_cast<value_type>(fftSize);
        if constexpr (gr::meta::complex_like<T>) {
            auto const freqOffset = static_cast<value_type>(N / 2) * freqWidth;
            std::ranges::transform(std::views::iota(0UL, N), std::ranges::begin(ds.signal_values), [freqWidth, freqOffset](const auto i) { return static_cast<value_type>(i) * freqWidth - freqOffset; });
        } else {
            std::ranges::transform(std::views::iota(0UL, N), std::ranges::begin(ds.signal_values), [freqWidth](const auto i) { return static_cast<value_type>(i) * freqWidth; });
        }
        std::ranges::transform(_outData.begin(), _outData.end(), std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(N)), [](const auto& c) { return c.real(); });
        std::ranges::transform(_outData.begin(), _outData.end(), std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(2U * N)), [](const auto& c) { return c.imag(); });
        std::copy_n(_magnitudeSpectrum.begin(), N, std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(3U * N)));
        std::copy_n(_phaseSpectrum.begin(), N, std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(4U * N)));

        ds.signal_ranges.resize(dim);
        for (std::size_t i = 0; i < dim; i++) {
            const auto mm       = std::minmax_element(std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(i * N)), std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>((i + 1U) * N)));
            ds.signal_ranges[i] = {*mm.first, *mm.second};
        }

        ds.meta_information = {{{"sample_rate", sample_rate}, {"signal_name", signal_name}, {"signal_unit", signal_unit}, {"signal_min", signal_min}, {"signal_max", signal_max}, //
            {"fft_size", fftSize}, {"window", window}, {"output_in_db", outputInDb}, {"output_in_deg", outputInDeg}, {"unwrap_phase", unwrapPhase},                               //
            {"input_chunk_size", this->input_chunk_size}, {"output_chunk_size", this->output_chunk_size}, {"stride", this->stride}}};

        return ds;
    }
};

template<typename T>
using DefaultFFT = FFT<T, typename OutputDataSet<T>::type, gr::algorithm::FFT>;

} // namespace gr::blocks::fft

auto registerFFT = gr::registerBlock<"gr::blocks::fft::DefaultFFT", gr::blocks::fft::DefaultFFT, float, double>(gr::globalBlockRegistry());

#endif // GNURADIO_FFT_HPP
