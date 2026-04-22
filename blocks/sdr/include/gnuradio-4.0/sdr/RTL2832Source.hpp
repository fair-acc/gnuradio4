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

#include <gnuradio-4.0/algorithm/SampleRateEstimator.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>

#include <gnuradio-4.0/sdr/RTL2832Device.hpp>

namespace gr::blocks::sdr {

namespace detail {

inline void convertToComplex(const std::uint8_t* raw, std::complex<float>* out, std::size_t nIqPairs) {
    constexpr float kScale = 1.f / 127.5f;
    for (std::size_t i = 0; i < nIqPairs; ++i) {
        out[i] = {(static_cast<float>(raw[2 * i]) - 127.5f) * kScale, (static_cast<float>(raw[2 * i + 1]) - 127.5f) * kScale};
    }
}

} // namespace detail

GR_REGISTER_BLOCK("gr::blocks::sdr::RTL2832Source", gr::blocks::sdr::RTL2832Source, [T], [ uint8_t, std::complex<float> ])

template<typename T>
struct RTL2832Source : gr::Block<RTL2832Source<T>> {
    using Description = Doc<R"(RTL2832U SDR source for USB dongles with the R820T/R820T2/R860, R828D, and E4000 tuners.
Native: Linux USB ioctl (zero-dependency). WASM: WebUSB via thin JS shims.

Operating modes:
  clk_in connected: forwards external timing tags (GPS/PPS) with clock-offset interpolation
  clk_in disconnected: emits best-effort wall-clock timestamps on every chunk)">;

    gr::PortIn<std::uint8_t, Optional> clk_in;
    gr::PortOut<T>                     out;

    Annotated<double, "frequency", Unit<"Hz">, Visible, Doc<"tuner LO frequency">>                             frequency    = 100.0e6;
    Annotated<float, "sample_rate", Unit<"Hz">, Visible, Limits<225e3f, 3.2e6f>, Doc<"ADC sample rate">>       sample_rate  = 2.048e6f;
    Annotated<float, "gain", Unit<"dB">, Visible, Doc<"tuner gain (manual mode)">>                             gain         = 40.f;
    Annotated<bool, "auto_gain", Visible, Doc<"enable hardware AGC">>                                          auto_gain    = true;
    Annotated<std::uint32_t, "device_index", Doc<"USB device index (0 = first dongle)">>                       device_index = 0U;
    Annotated<std::string, "device_name", Visible, Doc<"detected USB product name (read-only)">>               device_name;
    Annotated<std::int32_t, "ppm_correction", Doc<"crystal ppm correction applied to hardware PLL">>           ppm_correction     = 0;
    Annotated<std::uint32_t, "polling_period", Unit<"ms">, Doc<"IO thread sleep between USB reads">>           polling_period     = 10U;
    Annotated<std::string, "trigger_name", Doc<"tag trigger_name for free-running wallclock mode">>            trigger_name       = std::string("SDR_WALLCLOCK");
    Annotated<bool, "emit_timing_tags", Doc<"emit timing + ppm tags on every chunk">>                          emit_timing_tags   = true;
    Annotated<bool, "emit_meta_info", Doc<"include device/clock metadata in timing tags">>                     emit_meta_info     = true;
    Annotated<float, "tag_interval", Unit<"s">, Doc<"minimum interval between timing tags (0 = every chunk)">> tag_interval       = 1.0f;
    Annotated<bool, "dc_blocker_enabled", Doc<"IIR high-pass to remove DC offset (complex<float> only)">>      dc_blocker_enabled = true;
    Annotated<float, "dc_blocker_cutoff", Unit<"Hz">, Doc<"DC blocker high-pass cutoff frequency">>            dc_blocker_cutoff  = 10.f;
    Annotated<float, "ppm_estimator_cutoff", Unit<"Hz">, Doc<"LP cutoff for sample-rate estimator">>           ppm_estimator_cutoff =
#if defined(__EMSCRIPTEN__)
        0.01f; // WASM: performance.now() has ~1 ms jitter, needs heavier smoothing
#else
        0.1f; // native: USB transfer timestamps have ~100 us jitter
#endif
    Annotated<float, "ppm_tag_threshold", Doc<"emit corrected frequency/rate when ppm drift exceeds this">> ppm_tag_threshold = 0.1f;

    GR_MAKE_REFLECTABLE(RTL2832Source, clk_in, out, frequency, sample_rate, gain, auto_gain, device_index, device_name, ppm_correction, polling_period, trigger_name, emit_timing_tags, emit_meta_info, tag_interval, dc_blocker_enabled, dc_blocker_cutoff, ppm_estimator_cutoff, ppm_tag_threshold);

    RTL2832Device                  _device;
    bool                           _ioThreadDone     = true;
    std::int64_t                   _clockOffsetNs    = 0;
    bool                           _clockOffsetValid = false;
    std::string                    _clockTriggerName;
    double                         _prevFrequency  = 0.0;
    float                          _prevSampleRate = 0.f;
    float                          _prevGain       = 0.f;
    bool                           _prevAutoGain   = false;
    std::string                    _prevDeviceName;
    bool                           _firstEmission          = true;
    std::uint64_t                  _lastTagTimeNs          = 0UL;
    bool                           _retuneRequested        = false;
    std::uint8_t                   _postRetuneDiscardCount = 0;
    filter::Filter<float>          _dcFilterI;
    filter::Filter<float>          _dcFilterQ;
    algorithm::SampleRateEstimator _rateEstimator;
    float                          _ppmLastEmitted = 0.0f;

    struct IoThreadGuard {
        bool& done;
        explicit IoThreadGuard(bool& d) : done(d) {}
        IoThreadGuard(const IoThreadGuard&)            = delete;
        IoThreadGuard& operator=(const IoThreadGuard&) = delete;
        IoThreadGuard(IoThreadGuard&&)                 = delete;
        IoThreadGuard& operator=(IoThreadGuard&&)      = delete;
        ~IoThreadGuard() { gr::atomic_ref(done).wait(false); }
    };
    IoThreadGuard _ioGuard{_ioThreadDone};

    void start() {
        _clockOffsetNs    = 0;
        _clockOffsetValid = false;
        _clockTriggerName.clear();
        _firstEmission          = true;
        _lastTagTimeNs          = 0UL;
        _retuneRequested        = false;
        _postRetuneDiscardCount = 0;
        _ppmLastEmitted         = 0.0f;
        rebuildDcFilter();
        rebuildRateEstimator();
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
        if (this->disconnect_on_done && this->hasNoDownStreamConnectedChildren()) {
            this->requestStop();
            return {requestedWork, 0UZ, work::Status::DONE};
        }
        if (gr::atomic_ref(_ioThreadDone).load_acquire()) {
            this->requestStop();
            return {requestedWork, 0UZ, work::Status::DONE};
        }
        return {requestedWork, 1UZ, work::Status::OK};
    }

    void settingsChanged(const property_map& /*oldSettings*/, property_map& newSettings, property_map& forwardSettings) {
        if (!_device.isOpen()) {
            return;
        }
        if (newSettings.contains("frequency")) {
            _device.setCenterFrequency(frequency);
            _retuneRequested = true;
            forwardSettings.insert_or_assign(std::pmr::string("frequency"), frequency.value);
            forwardSettings.insert_or_assign(std::pmr::string("retune"), true);
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
            rebuildDcFilter();
            rebuildRateEstimator();
            forwardSettings.insert_or_assign(std::pmr::string("sample_rate"), sample_rate.value);
        }
        if (newSettings.contains("ppm_correction")) {
            _device.setFreqCorrection(ppm_correction);
        }
        if (newSettings.contains("dc_blocker_cutoff") || newSettings.contains("dc_blocker_enabled")) {
            rebuildDcFilter();
        }
        if (newSettings.contains("ppm_estimator_cutoff")) {
            _rateEstimator.filter_cutoff_hz = ppm_estimator_cutoff;
            _rateEstimator.rebuildFilter();
        }
    }

    void ioReadLoop() {
        thread_pool::thread::setThreadName(std::format("rtl2832:{}", this->name.value));

        auto& outWriter = out.streamWriter();
        auto& clkReader = clk_in.streamReader();
        auto& clkTagRdr = clk_in.tagReader();

        constexpr std::size_t                     kReadBufferSize = 64UZ * 1024UZ;
        std::array<std::uint8_t, kReadBufferSize> readBuf{};
        const auto                                minDelay = std::chrono::milliseconds(polling_period);

        while (lifecycle::isActive(this->state())) {
            this->applyChangedSettings();

            if (_retuneRequested) {
                _retuneRequested        = false;
                _postRetuneDiscardCount = 3;
                _dcFilterI.reset();
                _dcFilterQ.reset();
                _rateEstimator.resetPhase();
            }

            if (!_device.isOpen()) {
                auto result = _device.open(device_index);
                if (!result) {
                    this->emitErrorMessage("ioReadLoop()", std::format("open failed: {}", result.error()));
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    continue;
                }
                device_name = _device._deviceName;
                _device.setSampleRate(sample_rate);
                _device.setCenterFrequency(frequency);
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
                _firstEmission  = true;
                _lastTagTimeNs  = 0UL;
                _ppmLastEmitted = 0.0f;
                rebuildRateEstimator();
                this->emitMessage("ioReadLoop()", {{"state", "streaming"}, {"device", device_name.value}});
            }

            auto result = _device.readBulk(readBuf.data(), readBuf.size());
            if (!result) {
                _device.close();
                this->emitErrorMessage("ioReadLoop()", std::format("device error: {}", result.error()));
                std::this_thread::sleep_for(minDelay);
                continue;
            }
            if (*result == 0) {
                std::this_thread::sleep_for(minDelay);
                continue;
            }

            if (_postRetuneDiscardCount > 0) {
                --_postRetuneDiscardCount;
                continue;
            }

            auto        tWallNs        = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
            std::size_t nOutputSamples = std::is_same_v<T, std::uint8_t> ? *result : *result / 2UZ;
            double      tObsSeconds    = static_cast<double>(tWallNs) * 1e-9;
            _rateEstimator.update(tObsSeconds, nOutputSamples);

            drainClockInput(clkReader, clkTagRdr);
            publishSamples(outWriter, readBuf.data(), *result, tWallNs);
        }

        this->publishEoS();
        if (clk_in.isConnected()) {
            std::ignore = clk_in.disconnect();
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

        static constexpr std::string_view kTriggerTimeKey = tag::TRIGGER_TIME.shortKey();
        static constexpr std::string_view kMetaInfoKey    = tag::TRIGGER_META_INFO.shortKey();
        static constexpr std::string_view kTriggerNameKey = tag::TRIGGER_NAME.shortKey();
        static constexpr std::string_view kLocalTimeKey   = "local_time";

        auto        tagData       = clkTagRdr.get(clkTagRdr.available());
        std::size_t nTagsConsumed = 0;

        for (const auto& clkTag : tagData) {
            ++nTagsConsumed;

            if (auto it = clkTag.map.find(kTriggerTimeKey); it != clkTag.map.end()) {
                if (auto* timePtr = it->second.template get_if<std::uint64_t>()) {
                    auto triggerUtcNs = static_cast<std::int64_t>(*timePtr);

                    std::int64_t localNs      = 0;
                    bool         hasLocalTime = false;

                    if (auto metaIt = clkTag.map.find(kMetaInfoKey); metaIt != clkTag.map.end()) {
                        if (auto* metaMap = metaIt->second.template get_if<property_map>()) {
                            if (auto ltIt = metaMap->find(kLocalTimeKey); ltIt != metaMap->end()) {
                                if (auto* ltPtr = ltIt->second.template get_if<std::uint64_t>()) {
                                    localNs      = static_cast<std::int64_t>(*ltPtr);
                                    hasLocalTime = true;
                                }
                            }
                        }
                    }

                    auto nowNs        = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
                    _clockOffsetNs    = hasLocalTime ? (triggerUtcNs - localNs) : (triggerUtcNs - static_cast<std::int64_t>(nowNs));
                    _clockOffsetValid = true;
                }
            }

            if (auto it = clkTag.map.find(kTriggerNameKey); it != clkTag.map.end()) {
                if (auto nameView = it->second.template get_if<std::string_view>()) {
                    if (!nameView->empty()) {
                        _clockTriggerName = std::string(*nameView);
                    }
                }
            }
        }

        std::ignore = tagData.consume(nTagsConsumed);

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
            if (dc_blocker_enabled) {
                applyDcBlocker(span.data(), nOutputSamples);
            }
        }

        if (emit_timing_tags) {
            auto intervalNs = static_cast<std::uint64_t>(tag_interval.value * 1e9f);
            if (intervalNs == 0UL || _lastTagTimeNs == 0UL || (tWallNs - _lastTagTimeNs) >= intervalNs) {
                emitTimingTag(nOutputSamples, tWallNs);
                _lastTagTimeNs = tWallNs;
            } else {
                emitPpmTagIfNeeded();
            }
        }

        span.publish(nOutputSamples);
        this->progress->incrementAndGet();
        this->progress->notify_all();
    }

    void emitTimingTag(std::size_t nSamples, std::uint64_t tWallNs) {
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

        if (_rateEstimator._initialised) {
            float ppmNow = _rateEstimator.estimatedPpm();
            tag::put(tagMap, "sample_rate", static_cast<float>(_rateEstimator.estimatedRate()));
            tag::put(tagMap, "frequency", frequency.value * (1.0 + static_cast<double>(ppmNow) * 1e-6));
            tag::put(tagMap, "ppm_error", ppmNow);
            _ppmLastEmitted = ppmNow;
        }

        out.publishTag(std::move(tagMap), 0UZ);
    }

    void emitChangedParams(property_map& metaInfo) {
        if (_firstEmission || _prevDeviceName != device_name.value) {
            tag::put(metaInfo, "device_name", device_name.value);
            _prevDeviceName = device_name.value;
        }
        if (_firstEmission || _prevSampleRate != sample_rate.value) {
            tag::put(metaInfo, "sample_rate", static_cast<double>(sample_rate.value));
            _prevSampleRate = sample_rate.value;
        }
        if (_firstEmission || _prevFrequency != frequency.value) {
            tag::put(metaInfo, "frequency", frequency.value);
            _prevFrequency = frequency.value;
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

    void emitPpmTagIfNeeded() {
        if (!_rateEstimator._initialised) {
            return;
        }
        float ppmNow = _rateEstimator.estimatedPpm();
        if (std::abs(ppmNow) > 1000.f) {
            return; // estimator still in warm-up — real crystal drift is < ~100 ppm
        }
        if (std::abs(ppmNow - _ppmLastEmitted) < ppm_tag_threshold) {
            return;
        }

        float  correctedRate = static_cast<float>(_rateEstimator.estimatedRate());
        double correctedFreq = frequency.value * (1.0 + static_cast<double>(ppmNow) * 1e-6);

        auto tagMap = out.makeTagMap();
        tag::put(tagMap, "sample_rate", correctedRate);
        tag::put(tagMap, "frequency", correctedFreq);
        tag::put(tagMap, "ppm_error", ppmNow);
        out.publishTag(std::move(tagMap), 0UZ);

        _ppmLastEmitted = ppmNow;
    }

    void rebuildRateEstimator() {
        double nomRate                  = static_cast<double>(sample_rate.value);
        double updateHz                 = (nomRate > 0.0) ? (nomRate / (64.0 * 1024.0 / (std::is_same_v<T, std::uint8_t> ? 1.0 : 2.0))) : 250.0;
        _rateEstimator.filter_cutoff_hz = ppm_estimator_cutoff;
        _rateEstimator.reset(nomRate, updateHz);
    }

    void rebuildDcFilter() {
        if constexpr (std::is_same_v<T, std::complex<float>>) {
            if (dc_blocker_enabled && dc_blocker_cutoff > 0.f && sample_rate > 0.f) {
                auto coeffs = filter::iir::designFilter<float>(filter::Type::HIGHPASS, filter::FilterParameters{.order = 2UZ, .fHigh = static_cast<double>(dc_blocker_cutoff), .fs = static_cast<double>(sample_rate)}, filter::iir::Design::BUTTERWORTH);
                _dcFilterI  = filter::Filter<float>(coeffs);
                _dcFilterQ  = filter::Filter<float>(coeffs);
            }
        }
    }

    void applyDcBlocker(std::complex<float>* samples, std::size_t nSamples) {
        for (std::size_t i = 0; i < nSamples; ++i) {
            float filteredI = _dcFilterI.processOne(samples[i].real());
            float filteredQ = _dcFilterQ.processOne(samples[i].imag());
            samples[i]      = {filteredI, filteredQ};
        }
    }
};

} // namespace gr::blocks::sdr

#endif // GNURADIO_RTL2832_SOURCE_HPP
