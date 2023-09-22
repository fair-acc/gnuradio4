#ifndef GRAPH_PROTOTYPE_FFT_HPP
#define GRAPH_PROTOTYPE_FFT_HPP

#include <algorithm/fft/fft.hpp>
#include <algorithm/fft/fft_common.hpp>
#include <algorithm/fft/fft_types.hpp>
#include <algorithm/fft/fftw.hpp>
#include <algorithm/fft/window.hpp>
#include <dataset.hpp>
#include <execution>
#include <history_buffer.hpp>
#include <node.hpp>

namespace gr::blocks::fft {

using namespace fair::graph;
using gr::algorithm::ComplexType;
using gr::algorithm::FFTInDataType;
using gr::algorithm::FFTOutDataType;

template<typename T, typename U = DataSet<float>, typename FourierAlgoImpl = gr::algorithm::FFT<FFTInDataType<T, typename U::value_type>>>
    requires(ComplexType<T> || std::floating_point<T> || std::is_same_v<U, DataSet<float>> || std::is_same_v<U, DataSet<double>>)
struct FFT : public node<FFT<T, U, FourierAlgoImpl>, PerformDecimationInterpolation> {
public:
    using PrecisionType = U::value_type;
    using InDataType    = FFTInDataType<T, PrecisionType>;
    using OutDataType   = FFTOutDataType<PrecisionType>;

    PortIn<T>                                                                        in;
    PortOut<U, RequiredSamples<1, 1>>                                                out;

    FourierAlgoImpl                                                                  fftImpl;
    Annotated<std::size_t, "FFT size", Doc<"FFT size">>                              fftSize{ 1024 };
    Annotated<int, "window function", Doc<"window (apodization) function">>          window{ static_cast<int>(gr::algorithm::WindowFunction::Hann) };
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>           sampleRate = 1.f;
    Annotated<std::string, "signal name", Visible>                                   signalName = std::string("unknown signal");
    Annotated<std::string, "signal unit", Visible, Doc<"signal's physical SI unit">> signalUnit = std::string("a.u.");
    Annotated<T, "signal min", Doc<"signal physical min. (e.g. DAQ) limit">>         signalMin  = std::numeric_limits<T>::lowest();
    Annotated<T, "signal max", Doc<"signal physical max. (e.g. DAQ) limit">>         signalMax  = std::numeric_limits<T>::max();
    Annotated<bool, "output in dB", Doc<"calculate output in decibels">>             outputInDb{ false };
    std::vector<PrecisionType>                                                       windowVector;
    std::vector<InDataType>                                                          inData{};
    std::vector<OutDataType>                                                         outData{};
    std::vector<PrecisionType>                                                       magnitudeSpectrum{};
    std::vector<PrecisionType>                                                       phaseSpectrum{};

    FFT() {
        initAll();
        initWindowFunction();
    };

    explicit FFT(std::initializer_list<std::pair<const std::string, pmtv::pmt>> init_parameter) noexcept : node<FFT<T, U, FourierAlgoImpl>, PerformDecimationInterpolation>(init_parameter) {}

    FFT(const FFT &rhs)     = delete;
    FFT(FFT &&rhs) noexcept = delete;
    FFT &
    operator=(const FFT &rhs)
            = delete;
    FFT &
    operator=(FFT &&rhs) noexcept
            = delete;

    ~FFT() = default;

    void
    settings_changed(const property_map & /*old_settings*/, const property_map &newSettings) noexcept {
        if (newSettings.contains("fftSize") && fftSize != inData.size()) {
            initAll();
            initWindowFunction();
        } else if (newSettings.contains("window")) {
            initWindowFunction();
        }
    }

    [[nodiscard]] constexpr work_return_status_t
    process_bulk(std::span<const T> input, std::span<U> output) {
        if (input.size() != fftSize) {
            throw std::out_of_range(fmt::format("Input span size ({}) is not equal to FFT size ({}).", input.size(), fftSize));
        }

        if (output.size() != 1) {
            throw std::out_of_range(fmt::format("Output span size ({}) must be 1.", output.size()));
        }

        prepareInput(input);
        computeFFT();
        computeMagnitudeSpectrum();
        computePhaseSpectrum();
        output[0] = createDataset();

        return work_return_status_t::OK;
    }

    constexpr U
    createDataset() {
        U ds{};
        ds.timestamp = 0;
        const std::size_t N{ magnitudeSpectrum.size() };
        const std::size_t dim = 5;

        ds.axis_names         = { "Frequency", "Re(FFT)", "Im(FFT)", "Magnitude", "Phase" };
        ds.axis_units         = { "Hz", signalUnit, fmt::format("i{}", signalUnit), fmt::format("{}/√Hz", signalUnit), "rad" };
        ds.extents            = { dim, static_cast<int32_t>(N) };
        ds.layout             = fair::graph::layout_right{};
        ds.signal_names = { signalName, fmt::format("Re(FFT({}))", signalName), fmt::format("Im(FFT({}))", signalName), fmt::format("Magnitude({})", signalName), fmt::format("Phase({})", signalName) };
        ds.signal_units = { "Hz", signalUnit, fmt::format("i{}", signalUnit), fmt::format("{}/√Hz", signalUnit), "rad" };

        ds.signal_values.resize(dim * N);
        auto const freqWidth = static_cast<PrecisionType>(sampleRate) / static_cast<PrecisionType>(fftSize);
        if constexpr (ComplexType<T>) {
            auto const freqOffset = static_cast<PrecisionType>(N / 2) * freqWidth;
            std::ranges::transform(std::views::iota(std::size_t(0), N), std::ranges::begin(ds.signal_values),
                                   [freqWidth, freqOffset](const auto i) { return static_cast<PrecisionType>(i) * freqWidth - freqOffset; });
        } else {
            std::ranges::transform(std::views::iota(std::size_t(0), N), std::ranges::begin(ds.signal_values), [freqWidth](const auto i) { return static_cast<PrecisionType>(i) * freqWidth; });
        }
        std::ranges::transform(outData.begin(), outData.end(), std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(N)), [](const auto &c) { return c.real(); });
        std::ranges::transform(outData.begin(), outData.end(), std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(2U * N)), [](const auto &c) { return c.imag(); });
        std::copy_n(magnitudeSpectrum.begin(), N, std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(3U * N)));
        std::copy_n(phaseSpectrum.begin(), N, std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(4U * N)));

        ds.signal_ranges.resize(dim);
        for (std::size_t i = 0; i < dim; i++) {
            const auto mm = std::minmax_element(std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>(i * N)), std::next(ds.signal_values.begin(), static_cast<std::ptrdiff_t>((i + 1U) * N)));
            ds.signal_ranges[i] = { *mm.first, *mm.second };
        }

        ds.signal_errors    = {};
        ds.meta_information = { { { "sampleRate", sampleRate },
                                  { "signalName", signalName },
                                  { "signalUnit", signalUnit },
                                  { "signalMin", signalMin },
                                  { "signalMax", signalMax },
                                  { "fftSize", fftSize },
                                  { "window", window },
                                  { "outputInDb", outputInDb },
                                  { "numerator", this->numerator },
                                  { "denominator", this->denominator },
                                  { "stride", this->stride } } };

        return ds;
    }

    void
    prepareInput(std::span<const T> input) {
        if (std::is_same_v<T, InDataType>) {
            std::copy_n(input.begin(), fftSize, inData.begin());
        } else {
            std::ranges::transform(input.begin(), input.end(), inData.begin(), [](const T c) { return static_cast<InDataType>(c); });
        }

        // apply window function if needed
        if (window != static_cast<int>(gr::algorithm::WindowFunction::None)) {
            if (fftSize != windowVector.size()) {
                throw std::invalid_argument(fmt::format("fftSize({}) and windowVector.size({}) are not equal.", fftSize, windowVector.size()));
            }
            for (std::size_t i = 0; i < fftSize; i++) {
                if constexpr (ComplexType<T>) {
                    inData[i].real(inData[i].real() * windowVector[i]);
                    inData[i].imag(inData[i].imag() * windowVector[i]);
                } else {
                    inData[i] *= windowVector[i];
                }
            }
        }
    }

    inline void
    computeFFT() {
        outData = fftImpl.computeFFT(inData);
    }

    inline void
    computeMagnitudeSpectrum() {
        gr::algorithm::computeMagnitudeSpectrum(outData, magnitudeSpectrum, fftSize, outputInDb);
    }

    inline void
    computePhaseSpectrum() {
        gr::algorithm::computePhaseSpectrum(outData, phaseSpectrum);
    }

    void
    initAll() {
        in.max_samples    = fftSize;
        in.min_samples    = fftSize;
        this->denominator = fftSize;
        clear();
        inData.resize(fftSize);
        if constexpr (ComplexType<T>) {
            outData.resize(fftSize);
            magnitudeSpectrum.resize(fftSize);
            phaseSpectrum.resize(fftSize);
        } else {
            outData.resize(1 + fftSize / 2);
            magnitudeSpectrum.resize(fftSize / 2);
            phaseSpectrum.resize(fftSize / 2);
        }
    }

    inline void
    initWindowFunction() {
        windowVector = createWindowFunction<PrecisionType>(static_cast<gr::algorithm::WindowFunction>(window.value), fftSize);
    }

    void
    clear() {
        inData.clear();
        outData.clear();
        magnitudeSpectrum.clear();
        phaseSpectrum.clear();
    }
};

} // namespace gr::blocks::fft

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, typename U, typename FourierAlgoImpl), (gr::blocks::fft::FFT<T, U, FourierAlgoImpl>), in, out, fftSize, sampleRate, signalName, signalUnit, signalMin,
                                    signalMax, outputInDb, window, fftImpl);

#endif // GRAPH_PROTOTYPE_FFT_HPP
