#ifndef GNURADIO_AUDIO_BLOCKS_HPP
#define GNURADIO_AUDIO_BLOCKS_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/SampleRateEstimator.hpp>
#include <gnuradio-4.0/audio/AudioBackends.hpp>
#include <gnuradio-4.0/audio/EmscriptenAudioBackend.hpp>
#include <gnuradio-4.0/audio/SoundIoBackend.hpp>
#include <gnuradio-4.0/audio/WavSource.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <expected>
#include <format>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace gr::audio {

GR_REGISTER_BLOCK(gr::audio::AudioSource, [T], [ float, int16_t ])

template<detail::AudioSample T>
struct AudioSource : gr::Block<AudioSource<T>> {
    using Description = Doc<R""(Capture interleaved PCM samples from the platform audio input device.

`AudioSource` is intentionally PCM-only. It captures microphone or default input-device samples and
publishes them downstream as interleaved float or signed 16-bit PCM.
Publishes timing tags with estimated sample rate and optional GPS/PPS clock discipline.)"">;

    gr::PortIn<std::uint8_t, gr::Optional> clk_in;
    gr::PortOut<T>                         out;

    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"Requested capture sample rate. Updated to the active stream rate after start.">>               sample_rate   = 48000.f;
    gr::Annotated<gr::Size_t, "num_channels", gr::Visible, gr::Doc<"Requested interleaved channel count. Updated to the active stream channel count after start.">>          num_channels  = 1U;
    gr::Annotated<gr::Size_t, "buffer_frames", gr::Visible, gr::Doc<"Software queue depth in PCM frames">>                                                                   buffer_frames = 4096U;
    gr::Annotated<std::string, "device", gr::Visible, gr::Doc<"Device selector: empty or 'default' for system default, substring match on name, or '@id:...' for exact ID">> device;
    gr::Annotated<std::vector<std::string>, "available_devices", gr::Doc<"Detected audio input devices in 'name [id]' format">>                                              available_devices;
    gr::Annotated<bool, "emit_timing_tags", gr::Doc<"Emit timing tags with timestamps and rate estimates">>                                                                  emit_timing_tags = true;
    gr::Annotated<bool, "emit_meta_info", gr::Doc<"Include metadata in timing tags">>                                                                                        emit_meta_info   = true;
    gr::Annotated<float, "tag_interval", gr::Unit<"s">, gr::Doc<"Minimum interval between timing tags">>                                                                     tag_interval     = 1.0f;
    gr::Annotated<std::string, "trigger_name", gr::Doc<"Trigger name for free-running (no external clock) mode">>                                                            trigger_name     = std::string("AUDIO_WALLCLOCK");
    gr::Annotated<float, "ppm_estimator_cutoff", gr::Unit<"Hz">, gr::Doc<"Low-pass cutoff for sample rate estimator">>                                                       ppm_estimator_cutoff =
#if defined(__EMSCRIPTEN__)
        0.01f;
#else
        0.1f;
#endif
    gr::Annotated<bool, "permission", gr::Doc<"Read-only: whether microphone/input device permission has been granted">> permission = false;
    gr::Annotated<float, "level", gr::Doc<"Read-only: peak input signal level (0..1)">>                                  level      = 0.f;
    bool                                                                                                                 _useDummyBackendForTests{false};

    GR_MAKE_REFLECTABLE(AudioSource, clk_in, out, sample_rate, num_channels, buffer_frames, device, available_devices, emit_timing_tags, emit_meta_info, tag_interval, trigger_name, ppm_estimator_cutoff, permission, level);

    using gr::Block<AudioSource<T>>::Block;
#if defined(__EMSCRIPTEN__)
    using BackendImpl = detail::EmscriptenAudioWorkletSourceBackend<T>;
#else
    using BackendImpl = detail::SoundIoSourceBackend<T>;
#endif

    BackendImpl                    _backendImpl{};
    bool                           _failed{false};
    bool                           _formatTagPending{true};
    detail::AudioDeviceConfig      _activeConfig{};
    std::mutex                     _deviceMutex;
    algorithm::SampleRateEstimator _rateEstimator;
    algorithm::DriftCompensator<T> _driftCompensator;
    std::uint64_t                  _lastTagTimeNs{0U};
    std::int64_t                   _clockOffsetNs{0};
    bool                           _clockOffsetValid{false};
    std::string                    _clockTriggerName;

    void start() {
        std::lock_guard deviceLock(_deviceMutex);
        if (auto result = initialiseBackendUnlocked(); !result) {
            failUnlocked("AudioSource::start()", result.error());
        }
    }

    void stop() { shutdownDevice(); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        if (_activeConfig.sampleRate == 0U) {
            return;
        }
        const detail::AudioDeviceConfig requested{.sampleRate = currentSampleRate(), .numChannels = currentChannelCount(), .bufferFrames = buffer_frames.value, .device = device.value};
        if (requested.sampleRate == _activeConfig.sampleRate && requested.numChannels == _activeConfig.numChannels && requested.bufferFrames == _activeConfig.bufferFrames && requested.device == _activeConfig.device) {
            return;
        }
        std::lock_guard deviceLock(_deviceMutex);
        if (auto result = initialiseBackendUnlocked(); !result) {
            failUnlocked("AudioSource::settingsChanged()", result.error());
        }
    }

    gr::work::Result work(std::size_t requestedWork = std::numeric_limits<std::size_t>::max()) noexcept {
        if (!gr::lifecycle::isActive(this->state())) {
            return {requestedWork, 0UZ, gr::work::Status::DONE};
        }

        if (auto pollResult = _backendImpl.poll(); !pollResult) {
            this->emitErrorMessage("AudioSource::work()", pollResult.error());
            _backendImpl.requestStop();
            _failed = true;
            return {requestedWork, 0UZ, gr::work::Status::ERROR};
        }

        if (_failed) {
            return {requestedWork, 0UZ, gr::work::Status::ERROR};
        }

        drainClockInput();

        const std::size_t channelCount = std::max<std::size_t>(1U, static_cast<std::size_t>(num_channels.value));
        const std::size_t available    = _backendImpl._state.reader.available();
        if (available == 0U) {
            return {requestedWork, 0UZ, gr::work::Status::OK};
        }

        const std::size_t nFrameAligned = detail::wholeFrameSamples(available, channelCount);
        if (nFrameAligned == 0U) {
            return {requestedWork, 0UZ, gr::work::Status::OK};
        }

        // reserve extra space for potential drift insertion
        auto& outWriter = out.streamWriter();
        auto  outSpan   = outWriter.template tryReserve<gr::SpanReleasePolicy::ProcessNone>(nFrameAligned + channelCount);
        if (outSpan.empty()) {
            return {requestedWork, 0UZ, gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS};
        }

        std::size_t nProduced = _backendImpl.readToOutput(std::span<T>(outSpan.data(), nFrameAligned), channelCount);

        const bool streamActive = _backendImpl.isStreamActive();
        if (static_cast<bool>(permission.value) != streamActive) {
            permission = streamActive;
            this->settings().updateActiveParameters();
        }

        if (nProduced > 0U) {
            float peakLevel = 0.f;
            for (std::size_t i = 0U; i < nProduced; ++i) {
                if constexpr (std::same_as<T, float>) {
                    peakLevel = std::max(peakLevel, std::abs(outSpan[i]));
                } else {
                    peakLevel = std::max(peakLevel, std::abs(static_cast<float>(outSpan[i]) / 32768.f));
                }
            }
            level                      = peakLevel;
            const std::uint64_t tNowNs = detail::wallClockNs();
            _rateEstimator.update(static_cast<double>(tNowNs) * 1e-9, nProduced / channelCount);

            if (_rateEstimator.estimatedRate() > 0.0) {
                nProduced = _driftCompensator.compensateSource(std::span<T>(outSpan.data(), outSpan.size()), nProduced, _rateEstimator.estimatedRate(), static_cast<double>(sample_rate.value), channelCount);
            }

            if (_formatTagPending) {
                auto tagMap = property_map{};
                gr::tag::put(tagMap, gr::tag::SAMPLE_RATE, sample_rate.value);
                gr::tag::put(tagMap, gr::tag::NUM_CHANNELS, num_channels.value);
                gr::tag::put(tagMap, gr::tag::SIGNAL_NAME, std::string("audio_capture"));
                out.publishTag(std::move(tagMap), 0UZ);
                _formatTagPending = false;
            }

            if (emit_timing_tags.value) {
                maybeEmitTimingTag(tNowNs, nProduced, channelCount);
            }
        }

        outSpan.publish(nProduced);
        return {requestedWork, nProduced, gr::work::Status::OK};
    }

private:
    [[nodiscard]] std::uint32_t currentSampleRate() const {
        const auto rounded = std::lround(static_cast<double>(sample_rate.value));
        return rounded > 0L ? static_cast<std::uint32_t>(rounded) : 0U;
    }

    [[nodiscard]] std::uint32_t currentChannelCount() const { return num_channels.value > 0U ? static_cast<std::uint32_t>(num_channels.value) : 0U; }

    void maybeEmitTimingTag(std::uint64_t tNowNs, std::size_t /*nSamples*/, std::size_t /*channelCount*/) {
        const auto intervalNs = static_cast<std::uint64_t>(tag_interval.value * 1e9f);
        if (tNowNs - _lastTagTimeNs < intervalNs) {
            return;
        }
        _lastTagTimeNs = tNowNs;

        const auto tUtcNs   = static_cast<std::uint64_t>(static_cast<std::int64_t>(tNowNs) + _clockOffsetNs);
        const bool hasClock = clk_in.isConnected() && !_clockTriggerName.empty();

        auto tagMap = out.makeTagMap();
        gr::tag::put(tagMap, gr::tag::TRIGGER_NAME, hasClock ? _clockTriggerName : trigger_name.value);
        gr::tag::put(tagMap, gr::tag::TRIGGER_TIME, tUtcNs);
        gr::tag::put(tagMap, gr::tag::TRIGGER_OFFSET, 0.0f);

        if (emit_meta_info.value) {
            auto metaInfo = out.makeTagMap();
            gr::tag::put(metaInfo, "trigger_source", std::string("AudioSource"));
            gr::tag::put(metaInfo, "clock_source", hasClock ? _clockTriggerName : std::string("wallclock"));
            gr::tag::put(metaInfo, "software_latency", static_cast<float>(_backendImpl.softwareLatency()));
            if (_rateEstimator.estimatedRate() > 0.0) {
                gr::tag::put(metaInfo, "sample_rate", static_cast<float>(_rateEstimator.estimatedRate()));
                gr::tag::put(metaInfo, "ppm_error", _rateEstimator.estimatedPpm());
            }
            if (_clockOffsetValid) {
                gr::tag::put(metaInfo, "clock_offset_ns", _clockOffsetNs);
            }
            gr::tag::put(tagMap, gr::tag::TRIGGER_META_INFO, std::move(metaInfo));
        }

        if (_rateEstimator.estimatedRate() > 0.0) {
            gr::tag::put(tagMap, gr::tag::SAMPLE_RATE, static_cast<float>(_rateEstimator.estimatedRate()));
        }

        out.publishTag(std::move(tagMap), 0UZ);
    }

    void drainClockInput() {
        if (!clk_in.isConnected()) {
            return;
        }

        auto& clkReader  = clk_in.streamReader();
        auto& clkTagRdr  = clk_in.tagReader();
        auto  nAvailable = clkReader.available();
        if (nAvailable == 0) {
            return;
        }

        auto        tagData       = clkTagRdr.get(clkTagRdr.available());
        std::size_t nTagsConsumed = 0;

        for (const auto& clkTag : tagData) {
            ++nTagsConsumed;

            if (auto it = clkTag.map.find(std::pmr::string(gr::tag::TRIGGER_TIME.shortKey())); it != clkTag.map.end()) {
                if (auto* timePtr = it->second.template get_if<std::uint64_t>()) {
                    auto triggerUtcNs = static_cast<std::int64_t>(*timePtr);

                    std::int64_t localNs      = 0;
                    bool         hasLocalTime = false;

                    if (auto metaIt = clkTag.map.find(std::pmr::string(gr::tag::TRIGGER_META_INFO.shortKey())); metaIt != clkTag.map.end()) {
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

            if (auto it = clkTag.map.find(std::pmr::string(gr::tag::TRIGGER_NAME.shortKey())); it != clkTag.map.end()) {
                if (auto* namePtr = it->second.template get_if<std::pmr::string>()) {
                    if (!namePtr->empty()) {
                        _clockTriggerName = std::string(*namePtr);
                    }
                }
            }
        }

        std::ignore  = tagData.consume(nTagsConsumed);
        auto clkSpan = clkReader.get(nAvailable);
        std::ignore  = clkSpan.consume(nAvailable);
    }

    void failUnlocked(std::string_view endpoint, gr::Error error) {
        this->emitErrorMessage(endpoint, error);
        _backendImpl.shutdown();
        _failed = true;
    }

    [[nodiscard]] std::expected<void, gr::Error> initialiseBackendUnlocked() {
        const detail::AudioDeviceConfig config{.sampleRate = currentSampleRate(), .numChannels = currentChannelCount(), .bufferFrames = buffer_frames.value, .device = device.value, .useDummyBackendForTests = _useDummyBackendForTests};
        auto                            result = _backendImpl.start(config);
        if (!result) {
            return std::unexpected(result.error());
        }

        if (result->sampleRate != config.sampleRate && config.sampleRate != 0U) {
            this->emitErrorMessage("AudioSource::start()", gr::Error(std::format("requested sample rate {} Hz, device negotiated {} Hz", config.sampleRate, result->sampleRate)));
        }

        available_devices = _backendImpl._availableDevices;
        sample_rate       = static_cast<float>(result->sampleRate);
        num_channels      = static_cast<gr::Size_t>(result->numChannels);
        _activeConfig     = {.sampleRate = result->sampleRate, .numChannels = result->numChannels, .bufferFrames = buffer_frames.value, .device = device.value};
        _formatTagPending = true;
        _failed           = false;
        _lastTagTimeNs    = 0U;
        _clockOffsetNs    = 0;
        _clockOffsetValid = false;
        _clockTriggerName.clear();
        _driftCompensator.reset();

        const double expectedChunkRate  = static_cast<double>(result->sampleRate) / static_cast<double>(std::max<std::size_t>(1U, buffer_frames.value));
        _rateEstimator.filter_cutoff_hz = ppm_estimator_cutoff.value;
        _rateEstimator.reset(static_cast<double>(result->sampleRate), expectedChunkRate);

        return {};
    }

    void shutdownDevice() {
        std::lock_guard deviceLock(_deviceMutex);
        _backendImpl.shutdown();
    }
};

static_assert(gr::BlockLike<AudioSource<float>>);

GR_REGISTER_BLOCK(gr::audio::AudioSink, [T], [ float, int16_t ])

template<detail::AudioSample T>
struct AudioSink : gr::Block<AudioSink<T>> {
    using Description = Doc<R""(Play interleaved PCM samples on the platform audio output device.

`AudioSink` is intentionally PCM-only. It does not parse container formats or fetch files.
Pair it with `WavSource` or any other block that already produces decoded PCM.
Publishes timing tags with estimated consumption rate and software latency.)"">;

    gr::PortIn<T> in;

    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"PCM sample rate. Updated automatically; not intended to be set by the user.">>                 sample_rate   = 48000.f;
    gr::Annotated<gr::Size_t, "num_channels", gr::Visible, gr::Doc<"PCM interleaved channel count. Updated automatically; not intended to be set by the user.">>             num_channels  = 1U;
    gr::Annotated<gr::Size_t, "buffer_frames", gr::Visible, gr::Doc<"Software queue depth in PCM frames">>                                                                   buffer_frames = 4096U;
    gr::Annotated<std::string, "device", gr::Visible, gr::Doc<"Device selector: empty or 'default' for system default, substring match on name, or '@id:...' for exact ID">> device;
    gr::Annotated<std::vector<std::string>, "available_devices", gr::Doc<"Detected audio output devices in 'name [id]' format">>                                             available_devices;
    gr::Annotated<float, "ppm_estimator_cutoff", gr::Unit<"Hz">, gr::Doc<"Low-pass cutoff for sample rate estimator">>                                                       ppm_estimator_cutoff =
#if defined(__EMSCRIPTEN__)
        0.01f;
#else
        0.1f;
#endif
    gr::Annotated<bool, "debug_console", gr::Doc<"Log diagnostic info to stderr">>                                         debug_console = false;
    gr::Annotated<bool, "permission", gr::Doc<"Read-only: whether audio output device/context is active (not suspended)">> permission    = false;
    gr::Annotated<float, "level", gr::Doc<"Read-only: peak output signal level (0..1)">>                                   level         = 0.f;
    bool                                                                                                                   _useDummyBackendForTests{false};

    GR_MAKE_REFLECTABLE(AudioSink, in, sample_rate, num_channels, buffer_frames, device, available_devices, ppm_estimator_cutoff, debug_console, permission, level);

    using gr::Block<AudioSink<T>>::Block;
#if defined(__EMSCRIPTEN__)
    using BackendImpl = detail::EmscriptenAudioWorkletSinkBackend<T>;
#else
    using BackendImpl = detail::SoundIoSinkBackend<T>;
#endif

    BackendImpl                    _backendImpl{};
    bool                           _failed{false};
    detail::AudioDeviceConfig      _activeConfig{};
    std::mutex                     _deviceMutex;
    algorithm::SampleRateEstimator _rateEstimator;
    std::size_t                    _lastReportedUnderruns{0U};

    void start() {
        std::lock_guard deviceLock(_deviceMutex);
        if (auto result = initialiseBackendUnlocked(); !result) {
            failUnlocked("AudioSink::start()", result.error());
        }
    }

    void stop() { shutdownDevice(); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        if (_activeConfig.sampleRate == 0U) {
            return;
        }
        const detail::AudioDeviceConfig requested{.sampleRate = currentSampleRate(), .numChannels = currentChannelCount(), .bufferFrames = buffer_frames.value, .device = device.value};
        if (requested.sampleRate == _activeConfig.sampleRate && requested.numChannels == _activeConfig.numChannels && requested.bufferFrames == _activeConfig.bufferFrames && requested.device == _activeConfig.device) {
            return;
        }
        std::lock_guard deviceLock(_deviceMutex);
        if (auto result = initialiseBackendUnlocked(); !result) {
            failUnlocked("AudioSink::settingsChanged()", result.error());
        }
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan) {
        if (inSpan.empty()) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        if (auto pollResult = _backendImpl.poll(); !pollResult) {
            this->emitErrorMessage("AudioSink::processBulk()", pollResult.error());
            _backendImpl.requestStop();
            _failed     = true;
            std::ignore = inSpan.consume(0U);
            return gr::work::Status::ERROR;
        }

        if (_failed) {
            std::ignore = inSpan.consume(0U);
            return gr::work::Status::ERROR;
        }

        const std::size_t channelCount  = std::max<std::size_t>(1U, static_cast<std::size_t>(num_channels.value));
        const std::size_t nFrameSamples = inSpan.size() - (inSpan.size() % channelCount);
        if (nFrameSamples == 0U) {
            std::ignore = inSpan.consume(0U);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        const bool streamActive = _backendImpl.isStreamActive();
        if (static_cast<bool>(permission.value) != streamActive) {
            permission = streamActive;
            this->settings().updateActiveParameters();
        }

        const std::size_t nWritten = _backendImpl.writeFromInput(inSpan, channelCount);

        if (nWritten > 0U) {
            float peakLevel = 0.f;
            for (std::size_t i = 0U; i < nWritten; ++i) {
                if constexpr (std::same_as<T, float>) {
                    peakLevel = std::max(peakLevel, std::abs(inSpan[i]));
                } else {
                    peakLevel = std::max(peakLevel, std::abs(static_cast<float>(inSpan[i]) / 32768.f));
                }
            }
            level = peakLevel;

            const std::uint64_t tNowNs = detail::wallClockNs();
            _rateEstimator.update(static_cast<double>(tNowNs) * 1e-9, nWritten / channelCount);
        }

        std::ignore = inSpan.consume(nWritten);
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] std::uint32_t currentSampleRate() const {
        const auto rounded = std::lround(static_cast<double>(sample_rate.value));
        return rounded > 0L ? static_cast<std::uint32_t>(rounded) : 0U;
    }

    [[nodiscard]] std::uint32_t currentChannelCount() const { return num_channels.value > 0U ? static_cast<std::uint32_t>(num_channels.value) : 0U; }

    void failUnlocked(std::string_view endpoint, gr::Error error) {
        this->emitErrorMessage(endpoint, error);
        _backendImpl.shutdown();
        _failed = true;
    }

    [[nodiscard]] std::expected<void, gr::Error> initialiseBackendUnlocked() {
        const detail::AudioDeviceConfig config{.sampleRate = currentSampleRate(), .numChannels = currentChannelCount(), .bufferFrames = buffer_frames.value, .device = device.value, .useDummyBackendForTests = _useDummyBackendForTests};
        auto                            result = _backendImpl.start(config);
        if (!result) {
            return std::unexpected(result.error());
        }

        const auto& actual = *result;
        if (actual.sampleRate != config.sampleRate && config.sampleRate != 0U) {
            this->emitErrorMessage("AudioSink::start()", gr::Error(std::format("requested sample rate {} Hz, device negotiated {} Hz", config.sampleRate, actual.sampleRate)));
            sample_rate = static_cast<float>(actual.sampleRate);
        }
        if (actual.numChannels != config.numChannels && config.numChannels != 0U) {
            num_channels = static_cast<gr::Size_t>(actual.numChannels);
        }

        available_devices     = _backendImpl._availableDevices;
        _activeConfig         = {.sampleRate = actual.sampleRate, .numChannels = actual.numChannels, .bufferFrames = buffer_frames.value, .device = device.value};
        _failed                = false;
        _lastReportedUnderruns = 0U;

        const double expectedChunkRate  = static_cast<double>(actual.sampleRate) / static_cast<double>(std::max<std::size_t>(1U, buffer_frames.value));
        _rateEstimator.filter_cutoff_hz = ppm_estimator_cutoff.value;
        _rateEstimator.reset(static_cast<double>(actual.sampleRate), expectedChunkRate);

        return {};
    }

    void shutdownDevice() {
        std::lock_guard deviceLock(_deviceMutex);
        _backendImpl.shutdown();
    }
};

static_assert(gr::BlockLike<AudioSink<float>>);

} // namespace gr::audio

#endif // GNURADIO_AUDIO_BLOCKS_HPP
