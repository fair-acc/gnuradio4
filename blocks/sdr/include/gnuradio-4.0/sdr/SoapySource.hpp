#ifndef GNURADIO_SOAPY_SOURCE_HPP
#define GNURADIO_SOAPY_SOURCE_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <gnuradio-4.0/sdr/SoapyRaiiWrapper.hpp>

namespace gr::blocks::sdr {

namespace detail {
inline bool equalWithinOnePercent(const auto& a, const auto& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](double x, double y) { return std::abs(x - y) <= 0.01 * std::max(std::abs(x), std::abs(y)); });
}
} // namespace detail

GR_REGISTER_BLOCK("gr::blocks::sdr::SoapySource", gr::blocks::sdr::SoapySource, ([T], 1UZ), [ uint8_t, int16_t, std::complex<float> ])
GR_REGISTER_BLOCK("gr::blocks::sdr::SoapyDualSource", gr::blocks::sdr::SoapySource, ([T], 2UZ), [ uint8_t, int16_t, std::complex<float> ])

template<typename T, std::size_t nPorts = std::dynamic_extent>
struct SoapySource : Block<SoapySource<T, nPorts>> {
    using Description = Doc<R"(SoapySDR source block for SDR hardware.
Supports single and multi-channel RX via SoapySDR's device-agnostic API.
Tested with RTL-SDR and LimeSDR drivers.)">;

    using TSizeChecker = Limits<0U, std::numeric_limits<std::uint32_t>::max(), [](std::uint32_t x) { return std::has_single_bit(x); }>;
    using TBasePort    = PortOut<T>;
    using TPortType    = std::conditional_t<nPorts == 1U, TBasePort, std::conditional_t<nPorts == std::dynamic_extent, std::vector<TBasePort>, std::array<TBasePort, nPorts>>>;

    TPortType out;

    // SigMF-aligned tags: core:sample_rate (global), core:frequency (captures), core:num_channels (global)
    Annotated<std::string, "device", Visible, Doc<"SoapySDR driver name">>                                                       device;
    Annotated<std::string, "device_parameter", Visible, Doc<"additional driver parameters">>                                     device_parameter;
    Annotated<float, "sample_rate", Unit<"Hz">, Visible, Doc<"ADC sample rate (SigMF core:sample_rate)">>                        sample_rate  = 1'000'000.f;
    Annotated<gr::Size_t, "num_channels", Visible, Doc<"number of RX channels (SigMF core:num_channels)">>                       num_channels = 1U;
    Annotated<std::vector<std::string>, "rx_antennae", Visible, Doc<"per-channel RX antenna selection">>                         rx_antennae;
    Annotated<std::vector<double>, "frequency", Unit<"Hz">, Visible, Doc<"per-channel center frequency (SigMF core:frequency)">> frequency     = initDefaultValues(107'000'000.);
    Annotated<std::vector<double>, "rx_bandwidths", Unit<"Hz">, Visible, Doc<"per-channel RX RF bandwidth">>                     rx_bandwidths = initDefaultValues(500'000.);
    Annotated<std::vector<double>, "rx_gains", Unit<"dB">, Visible, Doc<"per-channel RX tuner gain">>                            rx_gains      = initDefaultValues(10.);

    Annotated<std::uint32_t, "max_chunk_size", Doc<"max samples per read (ideally N x 512)">, Visible, TSizeChecker> max_chunk_size     = 512U << 4U;
    Annotated<std::uint32_t, "max_time_out_us", Unit<"us">, Doc<"SoapySDR polling timeout">>                         max_time_out_us    = 1'000;
    Annotated<gr::Size_t, "max_overflow_count", Doc<"max consecutive overflows before stop (0 = disable)">>          max_overflow_count = 10U;
    Annotated<gr::Size_t, "max_fragment_count", Doc<"max consecutive fragments before stop (0 = disable)">>          max_fragment_count = 100U;

    GR_MAKE_REFLECTABLE(SoapySource, out, device, device_parameter, sample_rate, num_channels, rx_antennae, frequency, rx_bandwidths, rx_gains, max_chunk_size, max_time_out_us, max_overflow_count, max_fragment_count);

    soapy::Device                          _device{};
    soapy::Device::Stream<T, SOAPY_SDR_RX> _rxStream{};
    gr::Size_t                             _fragmentCount = 0U;
    gr::Size_t                             _overFlowCount = 0U;

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (!_device.get()) {
            return;
        }

        bool needReinit = false;
        if ((newSettings.contains("device") && (oldSettings.at("device") != newSettings.at("device"))) || (newSettings.contains("num_channels") && (oldSettings.at("num_channels") != newSettings.at("num_channels"))) || (newSettings.contains("sample_rate"))) {
            needReinit = true;
        }

        if (newSettings.contains("rx_antennae")) {
            applyAntenna();
        }
        if (newSettings.contains("frequency") || newSettings.contains("sample_rate")) {
            applyFrequency();
        }
        if (newSettings.contains("rx_gains")) {
            applyGain();
        }

        if (needReinit && lifecycle::isActive(this->state())) {
            reinitDevice();
        }
    }

    void start() { reinitDevice(); }

    void pause() {
        if (auto r = _rxStream.deactivate(); !r) {
            this->emitErrorMessage("pause()", r.error());
        }
    }

    void resume() {
        if (auto r = _rxStream.activate(); !r) {
            this->emitErrorMessage("resume()", r.error());
        }
    }

    void stop() { _device.reset(); }

    constexpr work::Status processBulk(OutputSpanLike auto& output)
    requires(nPorts == 1)
    {
        const auto maxSamples = std::min(std::size_t{max_chunk_size.value}, output.size());

        int       flags   = 0;
        long long time_ns = 0;
        int       ret     = _rxStream.readStream(flags, time_ns, max_time_out_us, std::span<T>(output).subspan(0, maxSamples));

        auto status = handleStreamErrors(ret, flags);
        if (ret >= 0 && status == work::Status::OK) {
            output.publish(static_cast<std::size_t>(ret));
            return work::Status::OK;
        }
        output.publish(0UZ);
        return status;
    }

    template<OutputSpanLike TOutputBuffer>
    constexpr work::Status processBulk(std::span<TOutputBuffer>& outputs)
    requires(nPorts > 1)
    {
        auto maxSamples = std::min(static_cast<std::uint32_t>(outputs[0].size()), max_chunk_size.value);

        int       flags   = 0;
        int       ret     = SOAPY_SDR_TIMEOUT;
        long long time_ns = 0;
        {
            std::vector<std::span<T>> output(num_channels);
            for (std::size_t i = 0UZ; i < static_cast<std::size_t>(num_channels); ++i) {
                output[i] = std::span<T>(outputs[i]).subspan(0, maxSamples);
            }
            ret = _rxStream.readStreamIntoBufferList(flags, time_ns, static_cast<long int>(max_time_out_us), output);
        }

        auto status = handleStreamErrors(ret, flags);
        if (ret >= 0 && status == work::Status::OK) {
            std::ranges::for_each(outputs, [ret](auto& output) { output.publish(static_cast<std::size_t>(ret)); });
            return work::Status::OK;
        }
        std::ranges::for_each(outputs, [](auto& output) { output.publish(0UZ); });
        return status;
    }

    void reinitDevice() {
        _rxStream.reset();
        auto devResult = soapy::Device::make(soapy::Kwargs{{"driver", device.value}});
        if (!devResult) {
            this->emitErrorMessage("reinitDevice()", devResult.error());
            this->requestStop();
            return;
        }
        _device = std::move(*devResult);

        std::size_t nChannelMax    = _device.getNumChannels(SOAPY_SDR_RX);
        std::size_t nChannelNeeded = (nPorts != std::dynamic_extent) ? nPorts : static_cast<std::size_t>(num_channels.value);
        if (nChannelMax < nChannelNeeded) {
            this->emitErrorMessage("reinitDevice()", std::format("channel mismatch: need {} but device has {}", nChannelNeeded, nChannelMax));
            this->requestStop();
            return;
        }

        applySampleRate();
        applyAntenna();
        applyFrequency();
        applyBandwidth();
        applyGain();

        std::vector<gr::Size_t> channelIndices(num_channels);
        std::iota(channelIndices.begin(), channelIndices.end(), gr::Size_t{0});
        auto streamResult = _device.setupStream<T, SOAPY_SDR_RX>(channelIndices);
        if (!streamResult) {
            this->emitErrorMessage("reinitDevice()", streamResult.error());
            this->requestStop();
            return;
        }
        _rxStream = std::move(*streamResult);
        if (auto r = _rxStream.activate(); !r) {
            this->emitErrorMessage("reinitDevice()", r.error());
            this->requestStop();
        }
    }

    void applySampleRate() {
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (auto r = _device.setSampleRate(SOAPY_SDR_RX, i, static_cast<double>(sample_rate)); !r) {
                this->emitErrorMessage("applySampleRate()", r.error());
            }
        }
        std::vector<double> actual;
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            actual.push_back(_device.getSampleRate(SOAPY_SDR_RX, i));
        }
        if (!detail::equalWithinOnePercent(actual, std::vector<double>(num_channels, static_cast<double>(sample_rate)))) {
            this->emitErrorMessage("applySampleRate()", std::format("mismatch: set {} vs actual {}", sample_rate, gr::join(actual, ", ")));
        }
    }

    void applyAntenna() {
        if (rx_antennae->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            const auto& ant = rx_antennae->at(std::min(static_cast<std::size_t>(i), rx_antennae->size() - 1UZ));
            if (!ant.empty()) {
                if (auto r = _device.setAntenna(SOAPY_SDR_RX, i, ant); !r) {
                    this->emitErrorMessage("applyAntenna()", r.error());
                }
            }
        }
    }

    void applyFrequency() {
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            double freq = frequency->at(std::min(static_cast<std::size_t>(i), frequency->size() - 1UZ));
            if (auto r = _device.setCenterFrequency(SOAPY_SDR_RX, i, freq); !r) {
                this->emitErrorMessage("applyFrequency()", r.error());
            }
        }
        std::vector<double> actual;
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            actual.push_back(_device.getCenterFrequency(SOAPY_SDR_RX, i));
        }
        if (!detail::equalWithinOnePercent(actual, frequency.value)) {
            this->emitErrorMessage("applyFrequency()", std::format("mismatch: set {} vs actual {}", gr::join(frequency.value, ", "), gr::join(actual, ", ")));
        }
    }

    void applyBandwidth() {
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            double bw = rx_bandwidths->at(std::min(static_cast<std::size_t>(i), rx_bandwidths->size() - 1UZ));
            if (auto r = _device.setBandwidth(SOAPY_SDR_RX, i, bw); !r) {
                this->emitErrorMessage("applyBandwidth()", r.error());
            }
        }
    }

    void applyGain() {
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            double g = rx_gains->at(std::min(static_cast<std::size_t>(i), rx_gains->size() - 1UZ));
            if (auto r = _device.setGain(SOAPY_SDR_RX, i, g); !r) {
                this->emitErrorMessage("applyGain()", r.error());
            }
        }
    }

    work::Status handleStreamErrors(int ret, int flags) {
        if (ret >= 0) {
            if (max_fragment_count > 0 && (flags & SOAPY_SDR_MORE_FRAGMENTS)) {
                _fragmentCount++;
            } else {
                _fragmentCount = 0U;
            }
            _overFlowCount = 0U;
            if (max_fragment_count > 0 && _fragmentCount > max_fragment_count) {
                this->emitErrorMessage("handleStreamErrors()", std::format("MORE_FRAGMENTS: {} of max {}", _fragmentCount, max_fragment_count));
                this->requestStop();
                return work::Status::ERROR;
            }
        } else {
            switch (ret) {
            case SOAPY_SDR_TIMEOUT: return work::Status::OK;
            case SOAPY_SDR_OVERFLOW:
                _overFlowCount++;
                if (max_overflow_count > 0 && _overFlowCount > max_overflow_count) {
                    this->emitErrorMessage("handleStreamErrors()", std::format("OVERFLOW: {} of max {}", _overFlowCount, max_overflow_count));
                    this->requestStop();
                    return work::Status::ERROR;
                }
                return work::Status::OK;
            case SOAPY_SDR_CORRUPTION:
                this->emitErrorMessage("handleStreamErrors()", "CORRUPTION");
                this->requestStop();
                return work::Status::ERROR;
            default:
                this->emitErrorMessage("handleStreamErrors()", std::format("unknown error: {}", ret));
                this->requestStop();
                return work::Status::ERROR;
            }
        }
        return work::Status::OK;
    }

    template<typename U>
    static std::vector<U> initDefaultValues(U initialValue) {
        if constexpr (nPorts != std::dynamic_extent) {
            return std::vector<U>(nPorts, initialValue);
        } else {
            return std::vector<U>(1, initialValue);
        }
    }
};

template<typename T>
using SoapySimpleSource = SoapySource<T, 1UZ>;
template<typename T>
using SoapyDualSource = SoapySource<T, 2UZ>;

} // namespace gr::blocks::sdr

#endif // GNURADIO_SOAPY_SOURCE_HPP
