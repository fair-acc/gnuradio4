#ifndef GNURADIO_AUDIO_BLOCKS_HPP
#define GNURADIO_AUDIO_BLOCKS_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/SampleRateEstimator.hpp>
#include <gnuradio-4.0/audio/AudioBackends.hpp>
#include <gnuradio-4.0/audio/EmscriptenAudioBackend.hpp>
#include <gnuradio-4.0/audio/SoundIoBackend.hpp>
#include <gnuradio-4.0/fileio/WavBlocks.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <expected>
#include <format>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace gr::audio {

namespace detail {

[[nodiscard]] inline std::optional<gr::Value> findFormatValue(const auto& map, const auto& defaultTag) {
    if (auto value = map.find_value(defaultTag)) {
        return gr::Value(*value);
    }
    if (auto value = map.find_value(defaultTag.shortKey())) {
        return gr::Value(*value);
    }
    return std::nullopt;
}

#if defined(__EMSCRIPTEN__)
inline constexpr float kDefaultPpmEstimatorCutoff = 0.01f;
#else
inline constexpr float kDefaultPpmEstimatorCutoff = 0.1f;
#endif

} // namespace detail

GR_REGISTER_BLOCK("gr::audio::AudioSource", gr::audio::AudioSource, [T], [ float, int16_t ])

template<detail::AudioSample T, AudioPortMode portMode = AudioPortMode::interleaved>
struct AudioSource : gr::Block<AudioSource<T, portMode>> {
    using Description = Doc<"Captures float or int16 PCM as interleaved frames or as synchronised mono ports at the negotiated device rate.">;
    using AudioPort   = gr::PortOut<T>;
    using PublishSpan = decltype(std::declval<AudioPort&>().streamWriter().template tryReserve<gr::SpanReleasePolicy::ProcessNone>(0UZ));

    gr::PortIn<std::uint8_t, gr::Optional>                                                        clk_in;
    std::conditional_t<portMode == AudioPortMode::interleaved, AudioPort, std::vector<AudioPort>> out;

    gr::Annotated<float, "req_sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"preferred capture rate">>                                                                  req_sample_rate = 48000.f;
    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"negotiated capture rate">>                                                                     sample_rate     = 0.f;
    gr::Annotated<gr::Size_t, "n_outputs", gr::Visible, gr::Limits<1U, 32U>, gr::Doc<"logical channels: one interleaved port, or one mono port each">>                       n_outputs       = 1U;
    gr::Annotated<gr::Size_t, "num_channels", gr::Doc<"legacy alias of n_outputs">>                                                                                          num_channels    = 1U;
    gr::Annotated<ChannelPadding, "channel_padding", gr::Doc<"fill for channels the device lacks">>                                                                          channel_padding = ChannelPadding::zero;
    gr::Annotated<float, "io_buffer_size", gr::Unit<"s">, gr::Limits<0.1f, 10.f>, gr::Doc<"I/O buffer size in seconds">>                                                     io_buffer_size  = 5.0f;
    gr::Annotated<std::string, "device", gr::Visible, gr::Doc<"Device selector: empty or 'default' for system default, substring match on name, or '@id:...' for exact ID">> device;
    gr::Annotated<std::vector<std::string>, "available_devices", gr::Doc<"Detected audio input devices in 'name [id]' format">>                                              available_devices;
    gr::Annotated<bool, "emit_timing_tags", gr::Doc<"Emit timing tags with timestamps and rate estimates">>                                                                  emit_timing_tags     = true;
    gr::Annotated<bool, "emit_meta_info", gr::Doc<"Include metadata in timing tags">>                                                                                        emit_meta_info       = true;
    gr::Annotated<float, "tag_interval", gr::Unit<"s">, gr::Doc<"Minimum interval between timing tags">>                                                                     tag_interval         = 1.0f;
    gr::Annotated<std::string, "trigger_name", gr::Doc<"Trigger name for free-running (no external clock) mode">>                                                            trigger_name         = std::string("AUDIO_WALLCLOCK");
    gr::Annotated<float, "ppm_estimator_cutoff", gr::Unit<"Hz">, gr::Doc<"Low-pass cutoff for sample rate estimator">>                                                       ppm_estimator_cutoff = detail::kDefaultPpmEstimatorCutoff;
    gr::Annotated<algorithm::DriftCorrection, "drift_correction", gr::Doc<"Drift compensation mode: None, Linear, Cubic, or AdaptiveResampling">>                            drift_correction     = algorithm::DriftCorrection::Linear;
    gr::Annotated<bool, "permission", gr::Doc<"Read-only: whether microphone/input device permission has been granted">>                                                     permission           = false;
    bool                                                                                                                                                                     _useDummyBackendForTests{false};

    GR_MAKE_REFLECTABLE(AudioSource, clk_in, out, req_sample_rate, sample_rate, n_outputs, num_channels, channel_padding, io_buffer_size, device, available_devices, emit_timing_tags, emit_meta_info, tag_interval, trigger_name, ppm_estimator_cutoff, drift_correction, permission);
#if defined(__EMSCRIPTEN__)
    using BackendImpl = detail::EmscriptenAudioWorkletSourceBackend<T>;
#else
    using BackendImpl = detail::SoundIoSourceBackend<T>;
#endif

    BackendImpl                              _backendImpl{};
    std::atomic<bool>                        _failed{false};
    bool                                     _formatTagPending{true};
    detail::AudioStreamFormat                _activeFormat{};
    algorithm::SampleRateEstimator           _rateEstimator;
    algorithm::DriftCompensator<T>           _driftCompensator;
    std::uint64_t                            _lastTagTimeNs{0U};
    std::int64_t                             _clockOffsetNs{0};
    bool                                     _clockOffsetValid{false};
    std::string                              _clockTriggerName;
    std::size_t                              _logicalChannels{1UZ};
    std::vector<T>                           _captureSamples;
    std::vector<PublishSpan>                 _publishSpans;
    detail::AudioDiagnostics                 _diagnostics;
    std::optional<detail::AudioBackendState> _diagnosticState;
    bool                                     _restartPending{false};
    std::string                              _lastError;

    gr::thread_pool::PooledIoTask _ioTask;

    explicit AudioSource(property_map parameters = {}) : gr::Block<AudioSource>(detail::normaliseAudioSettings(parameters, "n_outputs")), _logicalChannels(detail::configuredChannelCount(parameters, "n_outputs")) {
        if constexpr (portMode == AudioPortMode::multiplexed) {
            out.resize(std::clamp(_logicalChannels, 1UZ, detail::kMaxAudioChannels));
        }
        this->propertyCallbacks["audio_backend"] = static_cast<gr::BlockBase::PropertyCallback>(&AudioSource::propertyCallbackDiagnostics);
    }

    void start() {
        _restartPending = false;
        reconfigure();
        if (_failed) {
            _backendImpl.shutdown();
            return;
        }
        _ioTask.start([this]() { ioReadLoop(); });
    }

    void reconfigure() {
        const detail::AudioDeviceConfig config{.sampleRate = detail::validSampleRate(req_sample_rate.value), .numChannels = detail::validChannelCount(_logicalChannels), .bufferSeconds = io_buffer_size.value, .device = device.value, .useDummyBackendForTests = _useDummyBackendForTests, .channelPadding = channel_padding.value};
        if (config.sampleRate == 0U || config.numChannels == 0U) {
            fail("AudioSource::reconfigure", gr::Error(std::format("n_outputs must be within [1, {}] and req_sample_rate positive", detail::kMaxAudioChannels)));
            return;
        }
        auto result = _backendImpl.start(config);
        if (!result) {
            fail("AudioSource::reconfigure", result.error());
            return;
        }
        if (result->sampleRate != 0U && result->sampleRate != config.sampleRate) {
            gr::log::warning("{}: requested sample rate {} Hz, device negotiated {} Hz", this->name.value, config.sampleRate, result->sampleRate);
        }
        available_devices = _backendImpl.availableDevices();
        _captureSamples.resize((detail::kMaxBatchFrames + 1UZ) * _logicalChannels);
        _publishSpans.reserve(outputPorts().size());
        _diagnosticState.reset();
        _lastError.clear();
        _failed           = false;
        _lastTagTimeNs    = 0U;
        _clockOffsetNs    = 0;
        _clockOffsetValid = false;
        _clockTriggerName.clear();
        _driftCompensator.mode = drift_correction.value;
        _driftCompensator.reset();
        _rateEstimator.filter_cutoff_hz = ppm_estimator_cutoff.value;
        applyNegotiatedFormat(*result);
        refreshDiagnostics();
        this->settings().updateActiveParameters();
    }

    void stop() {
        if (!_ioTask.stopAndJoin()) {
            return;
        }
        _backendImpl.shutdown();
        _diagnostics.markStopped();
    }

    gr::work::Result work(std::size_t requestedWork = std::numeric_limits<std::size_t>::max(), [[maybe_unused]] gr::device::DeviceContext& computeBackend = gr::device::hostBackend()) noexcept {
        if (!gr::lifecycle::isActive(this->state())) {
            return {requestedWork, 0UZ, gr::work::Status::DONE};
        }
        if (_ioTask.hasFinished()) {
            this->requestStop();
            return {requestedWork, 0UZ, gr::work::Status::DONE};
        }
        return {requestedWork, 1UZ, gr::work::Status::OK};
    }

    void ioReadLoop() {
        gr::thread_pool::thread::setThreadName(std::format("audio-src:{}", this->name.value));

        auto& clkReader = clk_in.streamReader();
        auto& clkTagRdr = clk_in.tagReader();

        const std::size_t channelCount = _logicalChannels;

        while (!_ioTask.stopRequested() && gr::lifecycle::isActive(this->state())) {
            this->applyChangedSettings(false);
            if (std::exchange(_restartPending, false)) {
                reconfigure();
            }
            if (_failed) {
                break;
            }
            if (auto pollResult = _backendImpl.poll(); !pollResult) {
                fail("AudioSource::ioReadLoop()", pollResult.error());
                continue;
            }

            const auto format = _backendImpl.streamFormat();
            if (format.sampleRate != 0U && (format.sampleRate != _activeFormat.sampleRate || format.numChannels != _activeFormat.numChannels)) {
                applyNegotiatedFormat(format);
                this->settings().updateActiveParameters();
                _diagnosticState.reset();
            }
            if (permission.value != _backendImpl.permissionGranted()) {
                permission = _backendImpl.permissionGranted();
                this->settings().updateActiveParameters();
            }
            if (_diagnosticState != _backendImpl.backendState()) {
                refreshDiagnostics();
            }
            _diagnostics.refreshCounters(_backendImpl.state());

            const std::size_t nFrameAligned = detail::wholeFrameSamples(_backendImpl.state().reader.available(), channelCount);
            if (nFrameAligned == 0U) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            drainClockInput(clkReader, clkTagRdr);
            publishSamples(nFrameAligned, channelCount);
        }

        this->publishEoS();
    }

    void settingsChanged(const property_map&, const property_map& newSettings) {
        const auto validWidth = static_cast<gr::Size_t>(std::clamp(_logicalChannels, 1UZ, detail::kMaxAudioChannels));
        if (n_outputs.value != validWidth) {
            this->emitErrorMessage("AudioSource::settingsChanged", gr::Error("n_outputs cannot change after construction"));
            n_outputs = validWidth;
        }
        num_channels = validWidth;
        _backendImpl.state().channelPadding.store(channel_padding.value, std::memory_order_release);
        _driftCompensator.mode = drift_correction.value;
        if (_ioTask.isRunning() && (newSettings.contains("req_sample_rate") || newSettings.contains("device") || newSettings.contains("io_buffer_size"))) {
            _restartPending = true;
        }
        sample_rate = static_cast<float>(_activeFormat.sampleRate);
        _diagnostics.update([this](property_map& snapshot) { snapshot.insert_or_assign("req_sample_rate", req_sample_rate.value); });
    }

    void applyNegotiatedFormat(detail::AudioStreamFormat format) {
        sample_rate       = static_cast<float>(format.sampleRate);
        _activeFormat     = format;
        _formatTagPending = true;
        _rateEstimator.reset(static_cast<double>(format.sampleRate), static_cast<double>(format.sampleRate) / static_cast<double>(detail::bufferFramesFor(io_buffer_size.value, format.sampleRate)));
    }

    void refreshDiagnostics() {
        property_map result = _backendImpl.diagnostics();
        const auto   state  = result.value_or<std::string>("state", "stopped");
        if (_failed && state != "denied" && state != "unavailable" && state != "ended") {
            result.insert_or_assign("state", std::string("failed"));
        }
        result.insert_or_assign("req_sample_rate", req_sample_rate.value);
        result.insert_or_assign("sample_rate", sample_rate.value);
        result.insert_or_assign("logical_channels", static_cast<gr::Size_t>(_logicalChannels));
        result.insert_or_assign("device_channels", _activeFormat.numChannels);
        result.insert_or_assign("permission_granted", permission.value);
        result.insert_or_assign("error_message", _lastError);
        _diagnosticState = _backendImpl.backendState();
        _diagnostics.update([&](property_map& snapshot) { snapshot = std::move(result); });
        _diagnostics.refreshCounters(_backendImpl.state());
    }

    std::optional<gr::Message> propertyCallbackDiagnostics(std::string_view, gr::Message message) { return _diagnostics.reply(std::move(message)); }

    void publishSamples(std::size_t nFrameAligned, std::size_t channelCount) {
        const std::size_t portChannels = portMode == AudioPortMode::interleaved ? channelCount : 1UZ;
        const auto        ports        = outputPorts();
        std::size_t       nFrames      = std::min(nFrameAligned / channelCount, _captureSamples.size() / channelCount - 1UZ);
        bool              connected    = false;
        for (auto& port : ports) {
            if (port.isConnected()) {
                connected               = true;
                const std::size_t space = port.streamWriter().available() / portChannels;
                nFrames                 = std::min(nFrames, space > 1UZ ? space - 1UZ : 0UZ);
            }
        }
        if (!connected || nFrames == 0UZ) {
            auto discard = _backendImpl.state().reader.get(nFrameAligned);
            std::ignore  = discard.consume(discard.size());
            if (connected) {
                _backendImpl.state().overflowCount.fetch_add(1UZ, std::memory_order_relaxed);
            }
            return;
        }
        std::size_t nProduced = _backendImpl.readToOutput(std::span<T>(_captureSamples.data(), nFrames * channelCount), channelCount);
        if (nProduced == 0UZ) {
            return;
        }
        const std::uint64_t tNowNs = detail::wallClockNs();
        _rateEstimator.update(static_cast<double>(tNowNs) * 1e-9, nProduced / channelCount);
        const double nomRate       = static_cast<double>(sample_rate.value);
        const double estimatedRate = std::clamp(_rateEstimator.estimatedRate(), nomRate * 0.9, nomRate * 1.1);
        if (estimatedRate > 0.0) {
            nProduced = _driftCompensator.compensateSource(std::span<T>(_captureSamples.data(), (nFrames + 1UZ) * channelCount), nProduced, estimatedRate, nomRate, channelCount);
        }
        _publishSpans.clear();
        const std::size_t outputFrames = nProduced / channelCount;
        for (auto& port : ports) {
            if (!port.isConnected()) {
                continue;
            }
            auto span = port.streamWriter().template tryReserve<gr::SpanReleasePolicy::ProcessNone>(outputFrames * portChannels);
            if (span.empty()) {
                _publishSpans.clear();
                _backendImpl.state().overflowCount.fetch_add(1UZ, std::memory_order_relaxed);
                return;
            }
            _publishSpans.push_back(std::move(span));
        }
        std::size_t spanIndex = 0UZ;
        for (std::size_t channel = 0UZ; channel < ports.size(); ++channel) {
            if (!ports[channel].isConnected()) {
                continue;
            }
            auto& span = _publishSpans[spanIndex++];
            if constexpr (portMode == AudioPortMode::interleaved) {
                std::copy_n(_captureSamples.begin(), nProduced, span.begin());
            } else {
                for (std::size_t frame = 0UZ; frame < outputFrames; ++frame) {
                    span[frame] = _captureSamples[frame * channelCount + channel];
                }
            }
        }
        if (std::exchange(_formatTagPending, false)) {
            property_map tagMap;
            gr::tag::put(tagMap, gr::tag::SAMPLE_RATE, sample_rate.value);
            gr::tag::put(tagMap, gr::tag::NUM_CHANNELS, static_cast<gr::Size_t>(portChannels));
            gr::tag::put(tagMap, gr::tag::SIGNAL_NAME, std::string("audio_capture"));
            for (auto& port : ports) {
                port.publishTag(tagMap, 0UZ);
            }
        }
        if (emit_timing_tags.value) {
            maybeEmitTimingTag(tNowNs);
        }
        for (auto& span : _publishSpans) {
            span.publish(outputFrames * portChannels);
        }
        _publishSpans.clear();
        if (this->progress) {
            this->progress->incrementAndGet();
            this->progress->notify_all();
        }
    }

private:
    [[nodiscard]] std::span<AudioPort> outputPorts() {
        if constexpr (portMode == AudioPortMode::interleaved) {
            return {&out, 1UZ};
        } else {
            return out;
        }
    }

    void fail(std::string_view endpoint, const gr::Error& error) {
        _lastError = error.message;
        this->emitErrorMessage(endpoint, error);
        _failed = true;
        refreshDiagnostics();
    }

    void maybeEmitTimingTag(std::uint64_t tNowNs) {
        const auto intervalNs = static_cast<std::uint64_t>(tag_interval.value * 1e9f);
        if (tNowNs - _lastTagTimeNs < intervalNs) {
            return;
        }
        _lastTagTimeNs = tNowNs;

        const auto tUtcNs   = static_cast<std::uint64_t>(static_cast<std::int64_t>(tNowNs) + _clockOffsetNs);
        const bool hasClock = clk_in.isConnected() && !_clockTriggerName.empty();

        auto tagMap = property_map{};
        gr::tag::put(tagMap, gr::tag::TRIGGER_NAME, hasClock ? _clockTriggerName : trigger_name.value);
        gr::tag::put(tagMap, gr::tag::TRIGGER_TIME, tUtcNs);
        gr::tag::put(tagMap, gr::tag::TRIGGER_OFFSET, 0.0f);
        if (_rateEstimator.estimatedRate() > 0.0) {
            gr::tag::put(tagMap, gr::tag::EST_SAMPLE_RATE, static_cast<float>(_rateEstimator.estimatedRate()));
        }

        if (emit_meta_info.value) {
            auto metaInfo = property_map{};
            gr::tag::put(metaInfo, "trigger_source", std::string("AudioSource"));
            gr::tag::put(metaInfo, "clock_source", hasClock ? _clockTriggerName : std::string("wallclock"));
            gr::tag::put(metaInfo, "software_latency", static_cast<float>(_backendImpl.softwareLatency()));
            if (_rateEstimator.estimatedRate() > 0.0) {
                gr::tag::put(metaInfo, "ppm_error", _rateEstimator.estimatedPpm());
            }
            if (_clockOffsetValid) {
                gr::tag::put(metaInfo, "clock_offset_ns", _clockOffsetNs);
            }
            gr::tag::put(tagMap, gr::tag::TRIGGER_META_INFO, std::move(metaInfo));
        }

        for (auto& port : outputPorts()) {
            port.publishTag(tagMap, 0UZ);
        }
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

            if (auto it = clkTag.map.find(std::pmr::string(gr::tag::TRIGGER_TIME)); it != clkTag.map.end()) {
                const Value ownedTimeEntry = (*it).second;
                if (const auto* timePtr = ownedTimeEntry.template get_if<std::uint64_t>()) {
                    auto triggerUtcNs = static_cast<std::int64_t>(*timePtr);

                    std::int64_t localNs      = 0;
                    bool         hasLocalTime = false;

                    if (auto metaIt = clkTag.map.find(std::pmr::string(gr::tag::TRIGGER_META_INFO)); metaIt != clkTag.map.end()) {
                        const Value ownedMetaEntry = (*metaIt).second;
                        if (auto metaMap = ownedMetaEntry.template get_if<property_map>()) {
                            if (auto ltIt = metaMap->find(std::pmr::string("local_time")); ltIt != metaMap->end()) {
                                const Value ownedLocalTimeEntry = (*ltIt).second;
                                if (auto* ltPtr = ownedLocalTimeEntry.template get_if<std::uint64_t>()) {
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

            if (auto it = clkTag.map.find(std::pmr::string(gr::tag::TRIGGER_NAME)); it != clkTag.map.end()) {
                const Value ownedNameEntry = (*it).second;
                if (auto nameView = ownedNameEntry.template get_if<std::string_view>()) {
                    if (!nameView->empty()) {
                        _clockTriggerName = std::string(*nameView);
                    }
                }
            }
        }

        std::ignore  = tagData.consume(nTagsConsumed);
        auto clkSpan = clkReader.get(nAvailable);
        std::ignore  = clkSpan.consume(nAvailable);
    }
};

static_assert(gr::BlockLike<AudioSource<float>>);

GR_REGISTER_BLOCK("gr::audio::AudioSourceMultiplexed", gr::audio::AudioSourceMultiplexed, [T], [ float, int16_t ])

template<detail::AudioSample T>
using AudioSourceMultiplexed = AudioSource<T, AudioPortMode::multiplexed>;

static_assert(gr::BlockLike<AudioSourceMultiplexed<float>>);

GR_REGISTER_BLOCK("gr::audio::AudioSink", gr::audio::AudioSink, [T], [ float, int16_t ])

template<detail::AudioSample T, AudioPortMode portMode = AudioPortMode::interleaved>
struct AudioSink : gr::Block<AudioSink<T, portMode>> {
    using Description = Doc<"Plays interleaved or synchronised mono PCM at the format of the first tagged samples of a run.">;

    std::conditional_t<portMode == AudioPortMode::interleaved, gr::PortIn<T>, std::vector<gr::PortIn<T, gr::Optional>>> in;

    gr::Annotated<float, "req_sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"preferred rate for untagged input">>                                                       req_sample_rate = 48000.f;
    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"negotiated playback rate">>                                                                    sample_rate     = 0.f;
    gr::Annotated<float, "est_sample_rate", gr::Unit<"Hz">, gr::Doc<"read-only: measured playback rate">>                                                                    est_sample_rate = 0.f;
    gr::Annotated<gr::Size_t, "n_inputs", gr::Visible, gr::Limits<1U, 32U>, gr::Doc<"configured channels: one interleaved port, or one mono port each">>                     n_inputs        = 1U;
    gr::Annotated<gr::Size_t, "num_channels", gr::Doc<"active logical channels">>                                                                                            num_channels    = 1U;
    gr::Annotated<ChannelPadding, "channel_padding", gr::Doc<"fill for extra device channels">>                                                                              channel_padding = ChannelPadding::zero;
    gr::Annotated<float, "io_buffer_size", gr::Unit<"s">, gr::Limits<0.1f, 10.f>, gr::Doc<"I/O staging buffer size in seconds">>                                             io_buffer_size  = 5.0f;
    gr::Annotated<std::string, "device", gr::Visible, gr::Doc<"Device selector: empty or 'default' for system default, substring match on name, or '@id:...' for exact ID">> device;
    gr::Annotated<std::vector<std::string>, "available_devices", gr::Doc<"Detected audio output devices in 'name [id]' format">>                                             available_devices;
    gr::Annotated<float, "ppm_estimator_cutoff", gr::Unit<"Hz">, gr::Doc<"Low-pass cutoff for sample rate estimator">>                                                       ppm_estimator_cutoff = detail::kDefaultPpmEstimatorCutoff;
    gr::Annotated<algorithm::DriftCorrection, "drift_correction", gr::Doc<"Drift compensation mode: None, Linear, Cubic, or AdaptiveResampling">>                            drift_correction     = algorithm::DriftCorrection::Linear;
    gr::Annotated<bool, "permission", gr::Doc<"Read-only: whether audio output device/context is active (not suspended)">>                                                   permission           = false;
    bool                                                                                                                                                                     _useDummyBackendForTests{false};

    GR_MAKE_REFLECTABLE(AudioSink, in, req_sample_rate, sample_rate, est_sample_rate, n_inputs, num_channels, channel_padding, io_buffer_size, device, available_devices, ppm_estimator_cutoff, drift_correction, permission);
#if defined(__EMSCRIPTEN__)
    using BackendImpl = detail::EmscriptenAudioWorkletSinkBackend<T>;
#else
    using BackendImpl = detail::SoundIoSinkBackend<T>;
#endif

    BackendImpl                              _backendImpl{};
    std::atomic<bool>                        _failed{false};
    std::atomic<bool>                        _discardStaged{false};
    std::atomic<float>                       _measuredSampleRate{0.f};
    detail::AudioDeviceConfig                _activeConfig{};
    std::mutex                               _deviceMutex;
    detail::AudioDiagnostics                 _diagnostics;
    std::optional<detail::AudioBackendState> _diagnosticState;
    bool                                     _restartPending{false};
    algorithm::SampleRateEstimator           _rateEstimator;
    algorithm::DriftCompensator<T>           _driftCompensator;
    double                                   _smoothedFillLevel{0.5};
    std::size_t                              _bufferCapacity{0U};
    detail::AudioStateBase<T>                _staging;
    std::size_t                              _totalStagedSamples{0U};
    std::size_t                              _configuredChannels{1UZ};
    std::size_t                              _logicalChannels{1UZ};
    std::uint32_t                            _taggedSampleRate{0U};
    std::size_t                              _taggedChannels{0UZ};
    bool                                     _stopped{false};
    std::vector<bool>                        _connectedInputs;
    std::vector<T>                           _adjustedSamples;
    std::size_t                              _inputChannels{0UZ};
    std::vector<std::uint32_t>               _inputRates;
    std::chrono::steady_clock::time_point    _nextSettingsPublication{};
    std::string                              _lastError;

    gr::thread_pool::PooledIoTask _ioTask;

    explicit AudioSink(property_map parameters = {}) : gr::Block<AudioSink>(detail::normaliseAudioSettings(parameters, "n_inputs")), _configuredChannels(detail::configuredChannelCount(parameters, "n_inputs")), _logicalChannels(_configuredChannels) {
        if constexpr (portMode == AudioPortMode::multiplexed) {
            in.resize(std::clamp(_configuredChannels, 1UZ, detail::kMaxAudioChannels));
        }
        _inputRates.resize(portMode == AudioPortMode::interleaved ? 1UZ : std::clamp(_configuredChannels, 1UZ, detail::kMaxAudioChannels));
        this->propertyCallbacks["audio_backend"] = static_cast<gr::BlockBase::PropertyCallback>(&AudioSink::propertyCallbackDiagnostics);
    }

    void start() {
        std::lock_guard deviceLock(_deviceMutex);
        std::ranges::fill(_inputRates, 0U);
        _inputChannels      = 0UZ;
        _taggedSampleRate   = 0U;
        _taggedChannels     = 0UZ;
        _totalStagedSamples = 0UZ;
        _restartPending     = false;
        _stopped            = false;
        _logicalChannels    = _configuredChannels;
        num_channels        = static_cast<gr::Size_t>(_logicalChannels);
        if constexpr (portMode == AudioPortMode::multiplexed) {
            _connectedInputs = in | std::views::transform([](const auto& port) { return port.isConnected(); }) | std::ranges::to<std::vector<bool>>();
        }
        if (auto result = initialiseBackendUnlocked(); !result) {
            fail("AudioSink::start()", result.error());
            _backendImpl.shutdown();
            return;
        }
        refreshDiagnostics();
        publishActiveSettings();
        _ioTask.start([this]() { ioWriteLoop(); });
    }

    void stop() {
        std::lock_guard deviceLock(_deviceMutex);
        _stopped = true;
        if (!_ioTask.stopAndJoin(drainTimeout() + std::chrono::seconds(1))) {
            return;
        }
        _backendImpl.shutdown();
        _diagnostics.markStopped();
    }

    void settingsChanged(const property_map&, const property_map& newSettings) {
        const auto validWidth = static_cast<gr::Size_t>(std::clamp(_configuredChannels, 1UZ, detail::kMaxAudioChannels));
        if (n_inputs.value != validWidth) {
            this->emitErrorMessage("AudioSink::settingsChanged", gr::Error("n_inputs cannot change after construction"));
            n_inputs = validWidth;
        }
        num_channels    = static_cast<gr::Size_t>(_logicalChannels);
        sample_rate     = static_cast<float>(_activeConfig.sampleRate);
        est_sample_rate = _measuredSampleRate.load(std::memory_order_relaxed);
        if (_ioTask.isRunning() && (newSettings.contains("req_sample_rate") || newSettings.contains("device") || newSettings.contains("io_buffer_size"))) {
            _restartPending = true;
        }
        _diagnostics.update([this](property_map& snapshot) { snapshot.insert_or_assign("req_sample_rate", req_sample_rate.value); });
    }

    void refreshDiagnostics() {
        property_map result = _backendImpl.diagnostics();
        if (_failed) {
            result.insert_or_assign("state", std::string("failed"));
        }
        result.insert_or_assign("sample_rate", static_cast<float>(_activeConfig.sampleRate));
        result.insert_or_assign("logical_channels", static_cast<gr::Size_t>(_logicalChannels));
        result.insert_or_assign("device_channels", _activeConfig.numChannels);
        result.insert_or_assign("discarded_channels", static_cast<gr::Size_t>(_logicalChannels > _activeConfig.numChannels ? _logicalChannels - _activeConfig.numChannels : 0UZ));
        _diagnosticState = _backendImpl.backendState();
        _diagnostics.update([&](property_map& snapshot) {
            result.insert_or_assign("req_sample_rate", snapshot.value_or<float>("req_sample_rate", 48000.f));
            result.insert_or_assign("error_message", _lastError);
            snapshot = std::move(result);
        });
        _diagnostics.refreshCounters(_backendImpl.state());
    }

    std::optional<gr::Message> propertyCallbackDiagnostics(std::string_view, gr::Message message) { return _diagnostics.reply(std::move(message)); }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan)
    requires(portMode == AudioPortMode::interleaved)
    {
        const auto reject = [&](gr::work::Status status) {
            std::ignore = inSpan.consume(0UZ);
            return status;
        };
        if (std::exchange(_restartPending, false)) {
            restartBackend();
        }
        if (inSpan.empty()) {
            return reject(gr::work::Status::INSUFFICIENT_INPUT_ITEMS);
        }
        if (_failed || !validateInputTags(inSpan, 1UZ)) {
            return reject(gr::work::Status::ERROR);
        }
        const std::size_t inputChannels = _inputChannels == 0UZ ? _logicalChannels : _inputChannels;
        const std::size_t boundary      = formatBoundary(inSpan);
        if (boundary < inputChannels) {
            if (boundary < inSpan.size()) {
                fail("AudioSink::processBulk", gr::Error("a format tag splits an interleaved frame"));
                return reject(gr::work::Status::ERROR);
            }
            return reject(gr::work::Status::INSUFFICIENT_INPUT_ITEMS);
        }
        const std::size_t frames = stagingFrames(boundary / inputChannels);
        if (frames == 0UZ) {
            return reject(gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS);
        }
        std::array<std::optional<std::size_t>, detail::kMaxAudioChannels> inputChannelOfLogical{};
        for (std::size_t logical = 0UZ; logical < _logicalChannels; ++logical) {
            inputChannelOfLogical[logical] = detail::paddingSourceChannel(logical, inputChannels, channel_padding.value);
        }
        const std::size_t staged = stageFrames(frames, [&](std::size_t frame, std::size_t logical) { return inputChannelOfLogical[logical] ? inSpan[frame * inputChannels + *inputChannelOfLogical[logical]] : T{}; });
        std::ignore              = inSpan.consume(staged * inputChannels);
        refreshActiveSettings();
        return staged > 0UZ ? gr::work::Status::OK : gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
    }

    template<gr::InputSpanLike TSpan>
    [[nodiscard]] gr::work::Status processBulk(std::span<TSpan>& inputs)
    requires(portMode == AudioPortMode::multiplexed)
    {
        const auto reject = [&](gr::work::Status status) {
            for (auto& input : inputs) {
                std::ignore = input.consume(0UZ);
            }
            return status;
        };
        if (std::exchange(_restartPending, false)) {
            restartBackend();
        }
        if (_failed || inputs.size() != _connectedInputs.size()) {
            return reject(gr::work::Status::ERROR);
        }
        std::size_t frames = detail::kMaxBatchFrames;
        for (std::size_t channel = 0UZ; channel < inputs.size(); ++channel) {
            if (inputs[channel].isConnected != _connectedInputs[channel]) {
                this->emitErrorMessage("AudioSink::processBulk", gr::Error("input connectivity cannot change during a run"));
                return reject(gr::work::Status::ERROR);
            }
            if (_connectedInputs[channel]) {
                frames = std::min(frames, formatBoundary(inputs[channel]));
            }
        }
        if (!std::ranges::contains(_connectedInputs, true) || frames == 0UZ) {
            return reject(gr::work::Status::INSUFFICIENT_INPUT_ITEMS);
        }
        for (std::size_t channel = 0UZ; channel < inputs.size(); ++channel) {
            if (_connectedInputs[channel] && !validateInputTags(inputs[channel], frames, channel)) {
                return reject(gr::work::Status::ERROR);
            }
        }
        frames = stagingFrames(frames);
        if (frames == 0UZ) {
            return reject(gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS);
        }
        const std::size_t staged = stageFrames(frames, [&](std::size_t frame, std::size_t logical) { return _connectedInputs[logical] ? inputs[logical][frame] : T{}; });
        for (std::size_t channel = 0UZ; channel < inputs.size(); ++channel) {
            std::ignore = inputs[channel].consume(_connectedInputs[channel] ? staged : 0UZ);
        }
        refreshActiveSettings();
        return staged > 0UZ ? gr::work::Status::OK : gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
    }

private:
    [[nodiscard]] std::chrono::milliseconds drainTimeout() const { return std::chrono::milliseconds(static_cast<int>(_activeConfig.bufferSeconds * 1000.f) + 2000); }

    [[nodiscard]] std::size_t stagingFrames(std::size_t available) const { return _activeConfig.numChannels == 0U ? 0UZ : std::min({available, detail::kMaxBatchFrames, _staging.writer.available() / _activeConfig.numChannels}); }

    [[nodiscard]] std::size_t formatBoundary(const gr::InputSpanLike auto& input) const {
        for (const auto& tag : input.rawTags()) {
            if (tag.index > input.streamIndex && (detail::findFormatValue(tag.map, gr::tag::SAMPLE_RATE) || detail::findFormatValue(tag.map, gr::tag::NUM_CHANNELS))) {
                return std::min(input.size(), tag.index - input.streamIndex);
            }
        }
        return input.size();
    }

    [[nodiscard]] std::size_t stageFrames(std::size_t frames, auto&& logicalSample) {
        const std::size_t deviceChannels = _activeConfig.numChannels;
        auto              span           = _staging.writer.tryReserve(frames * deviceChannels);
        if (span.empty()) {
            return 0UZ;
        }
        detail::adaptChannels<T>(std::span<T>(span.data(), frames * deviceChannels), frames, _logicalChannels, deviceChannels, channel_padding.value, logicalSample);
        span.publish(frames * deviceChannels);
        _totalStagedSamples += frames * deviceChannels;
        return frames;
    }

    [[nodiscard]] bool validateInputTags(gr::InputSpanLike auto& input, std::size_t prefix, std::size_t channel = 0UZ) {
        const bool    firstFormatOfRun = _totalStagedSamples == 0UZ;
        std::uint32_t inputRate        = _inputRates[channel];
        for (const auto& tag : input.rawTags()) {
            if (tag.index >= input.streamIndex + prefix) {
                break;
            }
            bool restart = false;
            if (const auto channels = detail::findFormatValue(tag.map, gr::tag::NUM_CHANNELS)) {
                const std::size_t count = gr::pmt::convert_safely<gr::Size_t>(*channels).value_or(0U);
                if constexpr (portMode == AudioPortMode::interleaved) {
                    if (detail::validChannelCount(count) == 0U) {
                        return fail("AudioSink::processBulk", gr::Error("invalid interleaved input channel count"));
                    }
                    _inputChannels = count;
                    if (firstFormatOfRun && _logicalChannels != count) {
                        _taggedChannels = count;
                        restart         = true;
                    }
                } else if (count != 1UZ) {
                    return fail("AudioSink::processBulk", gr::Error("each multiplexed input must contain mono PCM"));
                }
            }
            if (const auto rate = detail::findFormatValue(tag.map, gr::tag::SAMPLE_RATE)) {
                const std::uint32_t taggedRate = detail::validSampleRate(gr::pmt::convert_safely<float>(*rate).value_or(0.f));
                if (taggedRate == 0U) {
                    return fail("AudioSink::processBulk", gr::Error("invalid PCM sample_rate tag"));
                }
                const bool otherInputsAgree = std::ranges::all_of(_inputRates, [taggedRate](std::uint32_t other) { return other == 0U || other == taggedRate; });
                if (firstFormatOfRun && otherInputsAgree && taggedRate != _activeConfig.sampleRate) {
                    _taggedSampleRate = taggedRate;
                    restart           = true;
                }
                inputRate = taggedRate;
            }
            if (restart) {
                restartBackend();
            }
            if (_failed) {
                return false;
            }
        }
        if (inputRate != 0U && inputRate != _activeConfig.sampleRate) {
            return fail("AudioSink::processBulk", gr::Error(std::format("PCM sample_rate {} Hz does not match playback {} Hz; use an explicit resampler", inputRate, _activeConfig.sampleRate)));
        }
        _inputRates[channel] = inputRate;
        return true;
    }

    bool fail(std::string_view endpoint, const gr::Error& error) {
        _diagnostics.update([&](property_map& snapshot) {
            _lastError = error.message;
            snapshot.insert_or_assign("state", std::string("failed"));
            snapshot.insert_or_assign("error_message", _lastError);
        });
        _failed = true;
        this->emitErrorMessage(endpoint, error);
        return false;
    }

    void publishActiveSettings() {
        constexpr std::chrono::seconds kPublicationInterval{1};
        permission               = _backendImpl.isStreamActive();
        est_sample_rate          = _measuredSampleRate.load(std::memory_order_relaxed);
        _nextSettingsPublication = std::chrono::steady_clock::now() + kPublicationInterval;
        this->settings().updateActiveParameters();
    }

    void refreshActiveSettings() {
        if (permission.value != _backendImpl.isStreamActive() || std::chrono::steady_clock::now() >= _nextSettingsPublication) {
            publishActiveSettings();
        }
    }

    void restartBackend() {
        std::lock_guard deviceLock(_deviceMutex);
        if (_stopped) {
            return;
        }
        _discardStaged.store(true, std::memory_order_release);
        const bool joined = _ioTask.stopAndJoin();
        _discardStaged.store(false, std::memory_order_release);
        if (!joined) {
            _failed = true;
            return;
        }
        if (_taggedChannels != 0UZ) {
            _logicalChannels = std::exchange(_taggedChannels, 0UZ);
            num_channels     = static_cast<gr::Size_t>(_logicalChannels);
        }
        if (auto result = initialiseBackendUnlocked(); !result) {
            fail("AudioSink::restartBackend", result.error());
            _backendImpl.shutdown();
            return;
        }
        refreshDiagnostics();
        publishActiveSettings();
        _ioTask.start([this]() { ioWriteLoop(); });
    }

    void ioWriteLoop() {
        gr::thread_pool::thread::setThreadName(std::format("audio-sink:{}", this->name.value));
        const std::size_t deviceChannels = std::max<std::size_t>(1U, _activeConfig.numChannels);
        const double      nominalRate    = static_cast<double>(_activeConfig.sampleRate);
        auto&             backendState   = _backendImpl.state();

        const auto pollBackend = [&] {
            if (auto pollResult = _backendImpl.poll(); !pollResult) {
                fail("AudioSink::ioWriteLoop", pollResult.error());
                return false;
            }
            if (_diagnosticState != _backendImpl.backendState()) {
                refreshDiagnostics();
            }
            _diagnostics.refreshCounters(backendState);
            return true;
        };

        const auto transferStagedToBackend = [&] {
            const std::size_t backendSpace = backendState.writer.available();
            const std::size_t nToTransfer  = detail::playbackTransferSamples(_staging.reader.available(), backendSpace, _adjustedSamples.size(), deviceChannels);
            if (!_backendImpl.isStreamActive() || nToTransfer == 0U) {
                return false;
            }
            constexpr double kEmaAlpha  = 0.01;
            constexpr double kServoGain = 0.001;
            const double     fillRatio  = _bufferCapacity > 0U ? static_cast<double>(backendState.reader.available()) / static_cast<double>(_bufferCapacity) : 0.5;
            _smoothedFillLevel          = _smoothedFillLevel * (1.0 - kEmaAlpha) + fillRatio * kEmaAlpha;
            const double servoRatio     = std::clamp(1.0 + (_smoothedFillLevel - 0.5) * kServoGain, 1.0 - detail::kMaxServoRatioDeviation, 1.0 + detail::kMaxServoRatioDeviation);

            auto              readSpan      = _staging.reader.get(nToTransfer);
            const std::size_t nFrameAligned = detail::wholeFrameSamples(readSpan.size(), deviceChannels);
            const std::size_t nAdjusted     = _driftCompensator.compensateSink(std::span<const T>(readSpan.begin(), nFrameAligned), std::span<T>(_adjustedSamples.data(), std::min(_adjustedSamples.size(), backendSpace)), nFrameAligned, nominalRate * servoRatio, nominalRate, deviceChannels);
            std::ignore                     = backendState.writeFromInput(std::span<const T>(_adjustedSamples.data(), nAdjusted), deviceChannels);
            std::ignore                     = readSpan.consume(nFrameAligned);
            _rateEstimator.update(static_cast<double>(detail::wallClockNs()) * 1e-9, nFrameAligned / deviceChannels);
            _measuredSampleRate.store(static_cast<float>(_rateEstimator.estimatedRate()), std::memory_order_relaxed);
            return true;
        };

        constexpr double  kPrefillSeconds = 0.05;
        const std::size_t prefillSamples  = static_cast<std::size_t>(nominalRate * kPrefillSeconds) * deviceChannels;
        const auto        prefillDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (!_ioTask.stopRequested() && _staging.reader.available() < prefillSamples && std::chrono::steady_clock::now() < prefillDeadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        while (!_ioTask.stopRequested() && pollBackend()) {
            if (!transferStagedToBackend()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (_discardStaged.load(std::memory_order_acquire)) {
            return;
        }

        const auto drainDeadline = std::chrono::steady_clock::now() + drainTimeout();
        while (detail::wholeFrameSamples(_staging.reader.available(), deviceChannels) > 0U && std::chrono::steady_clock::now() < drainDeadline && _backendImpl.poll()) {
            const std::size_t toWrite = _backendImpl.isStreamActive() ? detail::wholeFrameSamples(std::min(_staging.reader.available(), backendState.writer.available()), deviceChannels) : 0UZ;
            if (toWrite == 0U) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            auto readSpan = _staging.reader.get(toWrite);
            std::ignore   = readSpan.consume(backendState.writeFromInput(std::span<const T>(readSpan.begin(), readSpan.size()), deviceChannels));
        }
        while (backendState.reader.available() > deviceChannels && _backendImpl.isStreamActive() && std::chrono::steady_clock::now() < drainDeadline && _backendImpl.poll()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        gr::log::debug("{}: playback stopped after {} staged samples, {} underruns, {} overflows", this->name.value, _totalStagedSamples, backendState.underrunCount.load(std::memory_order_relaxed), backendState.overflowCount.load(std::memory_order_relaxed));
    }

    [[nodiscard]] std::expected<void, gr::Error> initialiseBackendUnlocked() {
        const detail::AudioDeviceConfig config{.sampleRate = _taggedSampleRate != 0U ? _taggedSampleRate : detail::validSampleRate(req_sample_rate.value), .numChannels = detail::validChannelCount(_logicalChannels), .bufferSeconds = io_buffer_size.value, .device = device.value, .useDummyBackendForTests = _useDummyBackendForTests, .channelPadding = channel_padding.value, .intentionalSilence = portMode == AudioPortMode::multiplexed && !std::ranges::contains(_connectedInputs, true)};
        if (config.sampleRate == 0U || config.numChannels == 0U) {
            return std::unexpected(gr::Error(std::format("n_inputs must be within [1, {}] and req_sample_rate positive", detail::kMaxAudioChannels)));
        }
        auto result = _backendImpl.start(config);
        if (!result) {
            return std::unexpected(result.error());
        }
        if (result->sampleRate != config.sampleRate) {
            gr::log::warning("{}: requested sample rate {} Hz, device negotiated {} Hz", this->name.value, config.sampleRate, result->sampleRate);
        }
        const std::size_t bufferFrames = detail::bufferFramesFor(config.bufferSeconds, result->sampleRate);
        sample_rate                    = static_cast<float>(result->sampleRate);
        available_devices              = _backendImpl.availableDevices();
        _activeConfig                  = {.sampleRate = result->sampleRate, .numChannels = result->numChannels, .bufferSeconds = config.bufferSeconds};
        _failed                        = false;
        _smoothedFillLevel             = 0.5;
        _bufferCapacity                = detail::AudioStateBase<T>::bufferCapacitySamples(result->numChannels, bufferFrames);
        _driftCompensator.mode         = drift_correction.value;
        _driftCompensator.reset();
        _diagnostics.update([&](property_map&) { _lastError.clear(); });
        _staging.recreateBuffer(_bufferCapacity);
        _adjustedSamples.resize(_bufferCapacity + result->numChannels);
        _rateEstimator.filter_cutoff_hz = ppm_estimator_cutoff.value;
        _rateEstimator.reset(static_cast<double>(result->sampleRate), static_cast<double>(result->sampleRate) / static_cast<double>(bufferFrames));
        _measuredSampleRate.store(0.f, std::memory_order_relaxed);
        return {};
    }
};

static_assert(gr::BlockLike<AudioSink<float>>);

GR_REGISTER_BLOCK("gr::audio::AudioSinkMultiplexed", gr::audio::AudioSinkMultiplexed, [T], [ float, int16_t ])

template<detail::AudioSample T>
using AudioSinkMultiplexed = AudioSink<T, AudioPortMode::multiplexed>;

static_assert(gr::BlockLike<AudioSinkMultiplexed<float>>);

} // namespace gr::audio

#endif // GNURADIO_AUDIO_BLOCKS_HPP
