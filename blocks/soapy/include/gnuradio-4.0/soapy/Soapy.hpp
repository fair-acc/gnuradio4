#ifndef SOAPY_HPP
#define SOAPY_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#endif
#include <SoapySDR/Device.hpp> // needed for existing C++ return types that are ABI stable (i.e. header-only defined), to note: not compatible with GCC15
#include <SoapySDR/Formats.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <map>
#include <string>
#include <vector>

#include <gnuradio-4.0/meta/formatter.hpp>

#include "SoapyRaiiWrapper.hpp" // using SoapySDR's C-API as intermediate interface to mitigate ABI-issues between stdlibc++ and libc++

namespace gr::blocks::soapy {

namespace detail {
inline bool equalWithinOnePercent(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](double x, double y) { return std::abs(x - y) <= 0.01 * std::max(std::abs(x), std::abs(y)); });
}
} // namespace detail

GR_REGISTER_BLOCK("gr::blocks::soapy::SoapySimpleSource", gr::blocks::soapy::SoapySimpleSource, ([T], 1UZ), [ uint8_t, int16_t, std::complex<float> ])
GR_REGISTER_BLOCK("gr::blocks::soapy::SoapyDualSimpleSource", gr::blocks::soapy::SoapySimpleSource, ([T], 2UZ), [ uint8_t, int16_t, std::complex<float> ])

template<typename T, std::size_t nPorts = std::dynamic_extent>
struct SoapyBlock : public Block<SoapyBlock<T, nPorts> /*, BlockingIO<false>*/> {
    using Description = Doc<R""(A Soapy source block that interfaces with SDR hardware using the SoapySDR library.
This block supports multiple output ports and was tested against the 'rtlsdr' and 'lime' device driver.
)"">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = Annotated<U, description, Arguments...>; // optional shortening

    using TimePoint    = std::chrono::time_point<std::chrono::system_clock>;
    using TSizeChecker = Limits<0U, std::numeric_limits<std::uint32_t>::max(), [](std::uint32_t x) { return std::has_single_bit(x); }>;
    using TBasePort    = PortOut<T>;
    using TPortType    = std::conditional_t<nPorts == 1U, TBasePort, std::conditional_t<nPorts == std::dynamic_extent, std::vector<TBasePort>, std::array<TBasePort, nPorts>>>;

    TPortType out;

    A<std::string, "device driver name", Visible>                                                        device;
    A<std::string, "add. device parameter", Visible>                                                     device_parameter;
    A<float, "sample rate", Unit<"samples/s">, Doc<"sampling rate in samples per second (Hz)">, Visible> sample_rate = 1'000'000.f;
    A<std::vector<gr::Size_t>, "RX channel ID mapping vector", Visible>                                  rx_channels = initDefaultValues<true>(gr::Size_t(0U));
    A<std::vector<std::string>, "RX channel antenna mapping", Visible>                                   rx_antennae;
    A<std::vector<double>, "RX center frequency", Unit<"Hz">, Doc<"RX-RF center frequency">, Visible>    rx_center_frequency = initDefaultValues(107'000'000.);
    A<std::vector<double>, "RX bandwidth", Unit<"Hz">, Doc<"RX-RF bandwidth">, Visible>                  rx_bandwdith        = initDefaultValues(double(sample_rate / 2));
    A<std::vector<double>, "Rx gain", Unit<"dB">, Doc<"RX channel gain">, Visible>                       rx_gains            = initDefaultValues(10.);

    // low-level ABI
    A<std::uint32_t, "max polling chunk size", Doc<"ideally N x 512">, Visible, TSizeChecker> max_chunck_size    = 512U << 4U;
    A<std::uint32_t, "polling time out", Unit<"us">, Doc<"soapy polling time-out">>           max_time_out_us    = 1'000;
    A<gr::Size_t, "max overflow count", Doc<"0: disable">>                                    max_overflow_count = 10U;
    A<gr::Size_t, "max fragment count", Doc<"0: disable">>                                    max_fragment_count = 100U;

    GR_MAKE_REFLECTABLE(SoapyBlock, out, device, device_parameter, sample_rate, rx_channels, rx_antennae, rx_center_frequency, rx_bandwdith, rx_gains, max_chunck_size, max_time_out_us, max_overflow_count);

    Device                          _device{};
    Device::Stream<T, SOAPY_SDR_RX> _rxStream{};
    gr::Size_t                      _fragmentCount = 0U;
    gr::Size_t                      _overFlowCount = 0U;

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        bool needReinit = false;
        if ((newSettings.contains("device") && (oldSettings.at("device") != newSettings.at("device")))                   //
            || (newSettings.contains("rx_channels") && (oldSettings.at("rx_channels") != newSettings.at("rx_channels"))) //
            || (newSettings.contains("sample_rate"))) {
            needReinit = true;
        }

        if (newSettings.contains("rx_antennae")) {
            setAntennae();
        }
        if (newSettings.contains("rx_center_frequency") || newSettings.contains("sample_rate")) {
            setCenterFrequency();
        }
        if (newSettings.contains("rx_gains")) {
            setGains();
        }

        if (needReinit && lifecycle::isActive(this->state())) {
            // only force re-init if running and/or paused
            reinitDevice();
        }
    }

    void start() { reinitDevice(); }
    void pause() { _rxStream.deactivate(); }
    void resume() { _rxStream.activate(); }
    void stop() { _device.reset(); }

    constexpr work::Status processBulk(OutputSpanLike auto& output)
    requires(nPorts == 1)
    {
        // special case single ouput -> simplifies connect API because this doesn't require sub-indices
        const auto maxSamples = std::min(std::size_t{max_chunck_size.value}, output.size());

        int       flags   = 0;
        long long time_ns = 0; // driver specifc

        // non-blocking/blocking depending on the value of max_time_out_us (0...)
        int ret = _rxStream.readStream(flags, time_ns, max_time_out_us, std::span<T>(output).subspan(0, maxSamples));
        // for detailed debugging: detail::printSoapyReturnDebugInfo(ret, flags, time_ns);

        auto status = handleDeviceStreamingErrors(ret, flags);
        if (ret >= 0 && status == work::Status::OK) {
            output.publish(static_cast<std::size_t>(ret));
            return work::Status::OK;
        }

        // no data or some failure occured
        output.publish(0UZ);
        return status;
    }

    template<OutputSpanLike TOutputBuffer>
    constexpr work::Status processBulk(std::span<TOutputBuffer>& outputs)
    requires(nPorts > 1)
    {
        // general case multiple ouputs
        auto maxSamples = static_cast<std::uint32_t>(outputs[0].size()); // max available samples
        maxSamples      = std::min(maxSamples, max_chunck_size.value);

        int       flags   = 0;
        int       ret     = SOAPY_SDR_TIMEOUT;
        long long time_ns = 0; // driver specifc
        {
            std::vector<std::span<T>> output(rx_channels->size());
            for (std::size_t i = 0UZ; i < rx_channels->size(); ++i) {
                output[i] = std::span<T>(outputs[i]).subspan(0, maxSamples);
            }
            ret = _rxStream.readStreamIntoBufferList(flags, time_ns, static_cast<long int>(max_time_out_us), output);
        }
        // for detailed debugging: detail::printSoapyReturnDebugInfo(ret, flags, time_ns);

        auto status = handleDeviceStreamingErrors(ret, flags);
        if (ret >= 0 && status == work::Status::OK) {
            std::ranges::for_each(outputs, [ret](auto& output) { output.publish(static_cast<std::size_t>(ret)); });
            return work::Status::OK;
        }

        // no data or some failure occured
        std::ranges::for_each(outputs, [](auto& output) { output.publish(0UZ); });
        return status;
    }

    void reinitDevice() {
        _rxStream.reset();
        _device = Device(SoapySDR::Kwargs{{"driver", device.value}});

        std::size_t nChannelMax = _device.getNumChannels(SOAPY_SDR_RX);
        if (nChannelMax < rx_channels->size() || (nPorts != std::dynamic_extent && nChannelMax != rx_channels->size())) {
            throw gr::exception(std::format("device {} max channel mismatch: specified: {} vs max. {}", device.value, rx_channels->size(), nChannelMax));
        }

        setSampleRate();
        setAntennae();
        setCenterFrequency();
        setGains();
        _rxStream = _device.setupStream<T, SOAPY_SDR_RX>(rx_channels.value);
        _rxStream.activate();
    }

    void setSampleRate() {
        std::size_t nChannels = rx_channels->size();
        for (std::size_t i = 0UZ; i < nChannels; i++) {
            _device.setSampleRate(SOAPY_SDR_RX, rx_channels->at(i), static_cast<double>(sample_rate));
        }

        std::vector<double> actualSampleRates;
        for (std::size_t i = 0UZ; i < nChannels; i++) {
            actualSampleRates.push_back(_device.getSampleRate(SOAPY_SDR_RX, rx_channels->at(i)));
        }

        if (!detail::equalWithinOnePercent(actualSampleRates, std::vector<double>(nChannels, static_cast<double>(sample_rate)))) {
            throw gr::exception(std::format("sample rate mismatch:\nSet: {} vs. Actual: {}\n", sample_rate, gr::join(actualSampleRates, ", ")));
        }
    }

    void setAntennae() {
        if (rx_antennae->empty()) {
            return;
        }
        std::size_t nChannels  = rx_channels->size();
        std::size_t nAntaennae = rx_antennae->size();
        for (std::size_t i = 0UZ; i < nChannels; i++) {
            std::string antenna = rx_antennae->at(std::min(i, nAntaennae - 1UZ));
            if (!antenna.empty()) {
                _device.setAntenna(SOAPY_SDR_RX, rx_channels->at(i), antenna);
            }
        }
    }

    void setCenterFrequency() {
        std::size_t nChannels  = rx_channels->size();
        std::size_t nFrequency = rx_center_frequency->size();
        for (std::size_t i = 0UZ; i < nChannels; i++) {
            _device.setCenterFrequency(SOAPY_SDR_RX, rx_channels->at(i), rx_center_frequency->at(std::min(i, nFrequency - 1UZ)));
        }
        std::vector<double> rx_center_frequency_actual;
        for (std::size_t i = 0UZ; i < nChannels; i++) {
            rx_center_frequency_actual.push_back(_device.getCenterFrequency(SOAPY_SDR_RX, rx_channels->at(i)));
        }

        if (!detail::equalWithinOnePercent(rx_center_frequency_actual, rx_center_frequency.value)) {
            throw gr::exception(std::format("center frequency mismatch:\nSet: {} vs. Actual: {}\n", gr::join(rx_center_frequency.value, ", "), gr::join(rx_center_frequency_actual, ", ")));
        }
    }

    void setBandwidth() {
        std::size_t nChannels  = rx_channels->size();
        std::size_t nFrequency = rx_bandwdith->size();
        for (std::size_t i = 0UZ; i < nChannels; i++) {
            _device.setBandwidth(SOAPY_SDR_RX, rx_channels->at(i), rx_bandwdith->at(std::min(i, nFrequency - 1UZ)));
        }
    }

    void setGains() {
        std::size_t nChannels = rx_channels->size();
        std::size_t nGains    = rx_gains->size();
        for (std::size_t i = 0UZ; i < nChannels; i++) {
            _device.setGain(SOAPY_SDR_RX, rx_channels->at(i), rx_gains->at(std::min(i, nGains - 1UZ)));
        }
    }
    work::Status handleDeviceStreamingErrors(int ret, int flags) {
        if (ret >= 0) {
            if (max_fragment_count > 0 && flags & SOAPY_SDR_MORE_FRAGMENTS) {
                _fragmentCount++;
            } else {
                _fragmentCount = 0U;
            }
            _overFlowCount = 0U;
            if (max_fragment_count > 0 && _fragmentCount > max_fragment_count) {
                throw gr::exception(std::format("SOAPY_SDR_MORE_FRAGMENTS for Block {} Device {}: reached {} of requested max {} -> DAQ read-out/Scheduler/graph is too slow", this->name, device, _fragmentCount, max_fragment_count));
            }
        } else {
            switch (ret) {
            case SOAPY_SDR_TIMEOUT: return work::Status::OK;
            case SOAPY_SDR_OVERFLOW: {
                _overFlowCount++;
                if (max_overflow_count > 0 && _overFlowCount > max_overflow_count) {
                    throw gr::exception(std::format("SOAPY_SDR_OVERFLOW for Block {} Device {}: reached {} of requested max {}", this->name, device, _overFlowCount, max_overflow_count));
                }
                return work::Status::OK;
            }
            case SOAPY_SDR_CORRUPTION: {
                throw gr::exception(std::format("SOAPY_SDR_CORRUPTION for Block {} Device {}", this->name, device));
                return work::Status::ERROR;
            }
            default: throw gr::exception(std::format("unknown SoapySDR return type: {}", ret));
            }
        }
        return work::Status::OK;
    }

    template<bool increment = false, typename U>
    static std::vector<U> initDefaultValues(U initialValue) {
        std::vector<U> values;
        values.resize(nPorts);
        if constexpr (increment) {
            std::ranges::generate(values, [i = U(0), initialValue]() mutable { return initialValue + i++; });
        } else {
            std::fill(values.begin(), values.end(), initialValue);
        }
        return values;
    }
};
template<typename T>
using SoapySimpleSource = SoapyBlock<T, 1UZ>;
template<typename T>
using SoapyDualSimpleSource = SoapyBlock<T, 2UZ>;

} // namespace gr::blocks::soapy

#endif // SOAPY_HPP
