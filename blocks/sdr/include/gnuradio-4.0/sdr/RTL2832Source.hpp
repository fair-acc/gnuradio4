#ifndef GNURADIO_RTL2832_SOURCE_HPP
#define GNURADIO_RTL2832_SOURCE_HPP

#include <chrono>
#include <complex>
#include <cstdint>
#include <cstring>
#include <print>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <gnuradio-4.0/sdr/RTL2832Device.hpp>

namespace gr::blocks::sdr {

namespace detail {

inline void convertToComplex(const std::uint8_t* raw, std::complex<float>* out, std::size_t nIqPairs) {
    constexpr float kScale = 1.f / 127.5f;
    for (std::size_t i = 0; i < nIqPairs; ++i) {
        out[i] = {(static_cast<float>(raw[2 * i]) - 127.5f) * kScale, (static_cast<float>(raw[2 * i + 1]) - 127.5f) * kScale};
    }
}

inline std::uint64_t wallClockNs() { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()); }

} // namespace detail

GR_REGISTER_BLOCK("gr::blocks::sdr::RTL2832Source", gr::blocks::sdr::RTL2832Source, [T], [ uint8_t, std::complex<float> ])

template<typename T>
struct RTL2832Source : gr::Block<RTL2832Source<T>> {
    using Description = Doc<R"(RTL2832U SDR source using USB dongles (RTL2832U + R820T/R828D/E4000).
Native: Linux USB ioctl (zero-dependency). WASM: WebUSB via thin JS shims.

Operating modes:
  clk_in connected: forwards external timing tags (GPS/PPS) with clock-offset interpolation
  clk_in disconnected: emits best-effort wall-clock timestamps on every chunk)">;

    gr::PortIn<std::uint8_t, Optional> clk_in;
    gr::PortOut<T>                     out;

    Annotated<double, "center_frequency", Unit<"Hz">, Visible>                          center_frequency = 100.0e6;
    Annotated<float, "sample_rate", Unit<"Hz">, Visible, Limits<225e3f, 3.2e6f>>        sample_rate      = 2.048e6f;
    Annotated<float, "gain", Unit<"dB">, Visible>                                       gain             = 40.f;
    Annotated<bool, "auto_gain", Visible>                                               auto_gain        = true;
    Annotated<std::uint32_t, "device_index">                                            device_index     = 0U;
    Annotated<std::string, "device_name", Visible>                                      device_name;
    Annotated<std::int32_t, "ppm_correction">                                           ppm_correction   = 0;
    Annotated<std::uint32_t, "polling period ms", Unit<"ms">, Doc<"IO polling period">> polling_period   = 10U;
    Annotated<std::string, "trigger_name", Doc<"tag name for free-running mode">>       trigger_name     = std::string("SDR_WALLCLOCK");
    Annotated<bool, "emit_timing_tags", Doc<"emit timing tags on every chunk">>         emit_timing_tags = true;
    Annotated<bool, "emit_meta_info", Doc<"include device/clock metadata in tags">>     emit_meta_info   = true;
    Annotated<float, "tag_interval", Unit<"s">, Doc<"minimum interval between wallclock tags (0 = every chunk)">> tag_interval     = 1.0f;

    GR_MAKE_REFLECTABLE(RTL2832Source, clk_in, out, center_frequency, sample_rate, gain, auto_gain, device_index, device_name, ppm_correction, polling_period, trigger_name, emit_timing_tags, emit_meta_info, tag_interval);

    RTL2832Device _device;
    bool          _ioThreadDone     = true;
    std::int64_t  _clockOffsetNs    = 0;
    bool          _clockOffsetValid = false;
    std::string   _clockTriggerName;
    double        _prevCenterFreq = 0.0;
    float         _prevSampleRate = 0.f;
    float         _prevGain       = 0.f;
    bool          _prevAutoGain   = false;
    std::string   _prevDeviceName;
    bool          _firstEmission = true;
    std::uint64_t _lastTagTimeNs = 0UL;

    struct IoThreadGuard {
        bool& done;
        ~IoThreadGuard() { gr::atomic_ref(done).wait(false); }
    };
    IoThreadGuard _ioGuard{_ioThreadDone};

    void start() {
        _clockOffsetNs    = 0;
        _clockOffsetValid = false;
        _clockTriggerName.clear();
        _firstEmission = true;
        _lastTagTimeNs = 0UL;
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
    }

    void ioReadLoop() {
        thread_pool::thread::setThreadName(std::format("rtl2832:{}", this->name.value));

        // cache reader/writer references for the entire IO loop lifetime (start → stop)
        auto& outWriter = out.streamWriter();
        auto& clkReader = clk_in.streamReader();
        auto& clkTagRdr = clk_in.tagReader();

        constexpr std::size_t                     kReadBufferSize = 64UZ * 1024UZ;
        std::array<std::uint8_t, kReadBufferSize> readBuf{};
        const auto                                minDelay = std::chrono::milliseconds(polling_period);

        while (lifecycle::isActive(this->state())) {
            this->applyChangedSettings();

            if (!_device.isOpen()) {
                auto result = _device.open(device_index);
                if (!result) {
                    this->emitErrorMessage("ioReadLoop()", std::format("open failed: {}", result.error()));
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    continue;
                }
                device_name = _device._deviceName;
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
                _firstEmission = true;
                _lastTagTimeNs = 0UL;
                this->emitMessage("ioReadLoop()", {{"state", "streaming"}, {"device", device_name.value}});
                std::println("[RTL2832] streaming: {}", device_name.value);
            }

            auto result = _device.readBulk(readBuf.data(), readBuf.size());
            if (!result) {
                _device.close();
                this->emitErrorMessage("ioReadLoop()", std::format("device error: {}", result.error()));
                std::println(stderr, "[RTL2832] device error: {} — will retry", result.error());
                std::this_thread::sleep_for(minDelay);
                continue;
            }
            if (*result == 0) {
                std::this_thread::sleep_for(minDelay);
                continue;
            }

            auto tWallNs = detail::wallClockNs();
            drainClockInput(clkReader, clkTagRdr);
            publishSamples(outWriter, readBuf.data(), *result, tWallNs);
        }

        gr::atomic_ref(_ioThreadDone).store_release(true);
        gr::atomic_ref(_ioThreadDone).notify_all();
    }

    void drainClockInput(auto& clkReader, auto& clkTagRdr) {
        if (!clk_in.isConnected()) {
            return;
        }
        auto nAvailable = clkReader.available();
        if (nAvailable == 0) {
            return;
        }

        auto        tagData       = clkTagRdr.get(clkTagRdr.available());
        std::size_t nTagsConsumed = 0;

        for (const auto& clkTag : tagData) {
            ++nTagsConsumed;

            if (auto it = clkTag.map.find(std::pmr::string(tag::TRIGGER_TIME.shortKey())); it != clkTag.map.end()) {
                if (auto* timePtr = it->second.template get_if<std::uint64_t>()) {
                    auto triggerUtcNs = static_cast<std::int64_t>(*timePtr);

                    // prefer local_time from trigger_meta_info for accurate offset
                    std::int64_t localNs      = 0;
                    bool         hasLocalTime = false;

                    if (auto metaIt = clkTag.map.find(std::pmr::string(tag::TRIGGER_META_INFO.shortKey())); metaIt != clkTag.map.end()) {
                        if (auto* metaMap = metaIt->second.template get_if<property_map>()) {
                            if (auto ltIt = metaMap->find(std::pmr::string("local_time")); ltIt != metaMap->end()) {
                                if (auto* ltPtr = ltIt->second.template get_if<std::uint64_t>()) {
                                    localNs      = static_cast<std::int64_t>(*ltPtr);
                                    hasLocalTime = true;
                                }
                            }
                        }
                    }

                    _clockOffsetNs    = hasLocalTime ? (triggerUtcNs - localNs) : (triggerUtcNs - static_cast<std::int64_t>(detail::wallClockNs()));
                    _clockOffsetValid = true;
                }
            }

            if (auto it = clkTag.map.find(std::pmr::string(tag::TRIGGER_NAME.shortKey())); it != clkTag.map.end()) {
                if (auto* namePtr = it->second.template get_if<std::pmr::string>()) {
                    if (!namePtr->empty()) {
                        _clockTriggerName = std::string(*namePtr);
                    }
                }
            }
        }

        std::ignore = tagData.consume(nTagsConsumed);

        // consume data samples (we only care about tags, not clock sample values)
        auto clkSpan = clkReader.get(nAvailable);
        std::ignore  = clkSpan.consume(nAvailable);
    }

    void publishSamples(auto& writer, const std::uint8_t* data, std::size_t nBytes, std::uint64_t tWallNs) {
        std::size_t nOutputSamples = std::is_same_v<T, std::uint8_t> ? nBytes : nBytes / 2UZ;

        auto span = writer.template tryReserve<SpanReleasePolicy::ProcessNone>(nOutputSamples);
        if (span.empty()) {
            return;
        }

        if constexpr (std::is_same_v<T, std::uint8_t>) {
            std::memcpy(span.data(), data, nBytes);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            detail::convertToComplex(data, span.data(), nOutputSamples);
        }

        if (emit_timing_tags) {
            auto intervalNs = static_cast<std::uint64_t>(tag_interval.value * 1e9f);
            if (intervalNs == 0UL || _lastTagTimeNs == 0UL || (tWallNs - _lastTagTimeNs) >= intervalNs) {
                emitTimingTag(nOutputSamples, tWallNs);
                _lastTagTimeNs = tWallNs;
            }
        }

        span.publish(nOutputSamples);
        this->progress->incrementAndGet();
        this->progress->notify_all();
    }

    void emitTimingTag(std::size_t nSamples, std::uint64_t tWallNs) {
        // tWallNs ≈ time of last sample; compute UTC time of first sample
        auto tUtcLastNs      = static_cast<std::int64_t>(tWallNs) + _clockOffsetNs;
        auto chunkDurationNs = static_cast<std::int64_t>(static_cast<double>(nSamples - 1) / static_cast<double>(sample_rate.value) * 1e9);
        auto tUtcFirstNs     = static_cast<std::uint64_t>(std::max(std::int64_t{0}, tUtcLastNs - chunkDurationNs));

        bool hasClock = clk_in.isConnected() && !_clockTriggerName.empty();

        auto tagMap = out.makeTagMap();
        tag::put(tagMap, tag::TRIGGER_NAME, hasClock ? _clockTriggerName : trigger_name.value);
        tag::put(tagMap, tag::TRIGGER_TIME, tUtcFirstNs);
        tag::put(tagMap, tag::TRIGGER_OFFSET, 0.f);

        if (emit_meta_info) {
            auto metaInfo = out.makeTagMap();
            tag::put(metaInfo, "trigger_source", std::string("RTL2832"));

            std::string clockSource = (_clockOffsetValid && !_clockTriggerName.empty()) ? _clockTriggerName : std::string("wallclock");
            tag::put(metaInfo, "clock_source", std::move(clockSource));

            if (_clockOffsetValid) {
                tag::put(metaInfo, "clock_offset_ns", _clockOffsetNs);
            }

            emitChangedParams(metaInfo);
            tag::put(tagMap, tag::TRIGGER_META_INFO, std::move(metaInfo));
        }

        out.publishTag(std::move(tagMap), 0UZ);
    }

    void emitChangedParams(property_map& metaInfo) {
        if (_firstEmission || _prevDeviceName != device_name.value) {
            tag::put(metaInfo, "device_name", device_name.value);
            _prevDeviceName = device_name.value;
        }
        if (_firstEmission || _prevSampleRate != sample_rate.value) {
            tag::put(metaInfo, "sample_rate", sample_rate.value);
            _prevSampleRate = sample_rate.value;
        }
        if (_firstEmission || _prevCenterFreq != center_frequency.value) {
            tag::put(metaInfo, "center_frequency", center_frequency.value);
            _prevCenterFreq = center_frequency.value;
        }
        if (_firstEmission || _prevGain != gain.value) {
            tag::put(metaInfo, "gain", gain.value);
            _prevGain = gain.value;
        }
        if (_firstEmission || _prevAutoGain != static_cast<bool>(auto_gain.value)) {
            tag::put(metaInfo, "auto_gain", static_cast<bool>(auto_gain.value));
            _prevAutoGain = auto_gain.value;
        }
        _firstEmission = false;
    }
};

} // namespace gr::blocks::sdr

#endif // GNURADIO_RTL2832_SOURCE_HPP
