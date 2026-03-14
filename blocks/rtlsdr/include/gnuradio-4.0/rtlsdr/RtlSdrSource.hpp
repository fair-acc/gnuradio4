#ifndef GNURADIO_RTLSDR_SOURCE_HPP
#define GNURADIO_RTLSDR_SOURCE_HPP

#include <complex>
#include <cstdint>
#include <cstring>
#include <print>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <gnuradio-4.0/rtlsdr/RtlSdrDevice.hpp>

namespace gr::blocks::rtlsdr {

namespace detail {

inline constexpr std::size_t kReadBufferSize = 16'384;

inline void convertToComplex(const std::uint8_t* raw, std::complex<float>* out, std::size_t nIqPairs) {
    constexpr float kScale = 1.f / 127.5f;
    for (std::size_t i = 0; i < nIqPairs; ++i) {
        out[i] = {(static_cast<float>(raw[2 * i]) - 127.5f) * kScale, (static_cast<float>(raw[2 * i + 1]) - 127.5f) * kScale};
    }
}

} // namespace detail

GR_REGISTER_BLOCK("gr::blocks::rtlsdr::RtlSdrSource", gr::blocks::rtlsdr::RtlSdrSource, [T], [ uint8_t, std::complex<float> ])

template<typename T>
struct RtlSdrSource : gr::Block<RtlSdrSource<T>> {
    using Description = Doc<R"(RTL-SDR source using RTL2832U USB dongles.
Native: libusb-1.0 direct. WASM: WebUSB via thin JS shims.)">;

    gr::PortOut<T> out;

    Annotated<double, "center_frequency", Unit<"Hz">, Visible>                          center_frequency = 100.0e6;
    Annotated<double, "sample_rate", Unit<"Hz">, Visible>                               sample_rate      = 2.048e6;
    Annotated<float, "gain", Unit<"dB">, Visible>                                       gain             = 40.f;
    Annotated<bool, "auto_gain", Visible>                                               auto_gain        = true;
    Annotated<std::uint32_t, "device_index">                                            device_index     = 0U;
    Annotated<std::string, "device_name", Visible>                                      device_name;
    Annotated<std::int32_t, "ppm_correction">                                           ppm_correction = 0;
    Annotated<std::uint32_t, "polling period ms", Unit<"ms">, Doc<"IO polling period">> polling_period = 10U;

    GR_MAKE_REFLECTABLE(RtlSdrSource, out, center_frequency, sample_rate, gain, auto_gain, device_index, device_name, ppm_correction, polling_period);

    RtlSdrDevice _device;
    bool         _ioThreadDone = true;

    struct IoThreadGuard {
        bool& done;
        ~IoThreadGuard() { gr::atomic_ref(done).wait(false); }
    };
    IoThreadGuard _ioGuard{_ioThreadDone};

    void start() {
        gr::atomic_ref(_ioThreadDone).store_release(false);
        thread_pool::Manager::defaultIoPool()->execute([this]() { ioReadLoop(); });
    }

    void stop() {
        gr::atomic_ref(_ioThreadDone).wait(false);
        _device.close();
    }

    work::Result work(std::size_t requestedWork = std::numeric_limits<std::size_t>::max()) noexcept {
        if (!lifecycle::isActive(this->state())) {
            return {requestedWork, 0UZ, work::Status::DONE};
        }
        return {requestedWork, 1UZ, work::Status::OK};
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
#if !defined(__EMSCRIPTEN__)
        if (!_device.isOpen()) {
            return;
        }
        if (newSettings.contains("center_frequency")) {
            _device.setCenterFrequency(center_frequency);
        }
        if (newSettings.contains("gain") || newSettings.contains("auto_gain")) {
            if (auto_gain) {
                _device.setGainMode(true);
            } else {
                _device.setGainMode(false);
                _device.setTunerGain(gain);
            }
        }
        if (newSettings.contains("sample_rate")) {
            _device.setSampleRate(sample_rate);
        }
        if (newSettings.contains("ppm_correction")) {
            _device.setFreqCorrection(ppm_correction);
        }
#else
        (void)newSettings; // WASM: device configured by main-thread button handler (TODO: proxy)
#endif
    }

    void ioReadLoop() {
        thread_pool::thread::setThreadName(std::format("rtlsdr:{}", this->name.value));

        std::array<std::uint8_t, detail::kReadBufferSize> readBuf{};

        while (lifecycle::isActive(this->state())) {
            this->applyChangedSettings();

            if (!_device.isOpen()) {
                if (!_device.open(device_index)) {
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    continue;
                }
                device_name = _device._deviceName;
#if !defined(__EMSCRIPTEN__)
                _device.setSampleRate(sample_rate);
                _device.setCenterFrequency(center_frequency);
                if (auto_gain) {
                    _device.setGainMode(true);
                    _device.setAgcMode(true);
                } else {
                    _device.setGainMode(false);
                    _device.setAgcMode(false);
                    _device.setTunerGain(gain);
                }
                _device.setFreqCorrection(ppm_correction);
                _device.resetBuffer();
#endif
                std::println("[RTL-SDR] streaming: {}", device_name.value);
            }

            auto nRead = _device.readBulk(readBuf.data(), readBuf.size());
            if (nRead == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(polling_period));
                continue;
            }
            publishSamples(readBuf.data(), nRead);
        }

        gr::atomic_ref(_ioThreadDone).store_release(true);
        gr::atomic_ref(_ioThreadDone).notify_all();
    }

    void publishSamples(const std::uint8_t* data, std::size_t nBytes) {
        if constexpr (std::is_same_v<T, std::uint8_t>) {
            auto span = out.streamWriter().template tryReserve<SpanReleasePolicy::ProcessNone>(nBytes);
            if (span.empty()) {
                return;
            }
            std::memcpy(span.data(), data, nBytes);
            span.publish(nBytes);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            auto nSamples = nBytes / 2UZ;
            auto span     = out.streamWriter().template tryReserve<SpanReleasePolicy::ProcessNone>(nSamples);
            if (span.empty()) {
                return;
            }
            detail::convertToComplex(data, span.data(), nSamples);
            span.publish(nSamples);
        }
        this->progress->incrementAndGet();
        this->progress->notify_all();
    }
};

} // namespace gr::blocks::rtlsdr

#endif // GNURADIO_RTLSDR_SOURCE_HPP
