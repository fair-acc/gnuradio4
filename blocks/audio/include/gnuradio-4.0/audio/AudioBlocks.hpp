#ifndef GNURADIO_AUDIO_BLOCKS_HPP
#define GNURADIO_AUDIO_BLOCKS_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/audio/AudioBackends.hpp>
#include <gnuradio-4.0/audio/EmscriptenAudioWorkletBackend.hpp>
#include <gnuradio-4.0/audio/SoundIoBackend.hpp>
#include <gnuradio-4.0/audio/WavSource.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <algorithm>
#include <cmath>
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
publishes them downstream as interleaved float or signed 16-bit PCM.)"">;

    gr::PortOut<T> out;

    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"Requested capture sample rate. Updated to the active stream rate after start.">>               sample_rate   = 48000.f;
    gr::Annotated<gr::Size_t, "num_channels", gr::Visible, gr::Doc<"Requested interleaved channel count. Updated to the active stream channel count after start.">>          num_channels  = 1U;
    gr::Annotated<gr::Size_t, "buffer_frames", gr::Visible, gr::Doc<"Software queue depth in PCM frames">>                                                                   buffer_frames = 4096U;
    gr::Annotated<std::string, "device", gr::Visible, gr::Doc<"Device selector: empty or 'default' for system default, substring match on name, or '@id:...' for exact ID">> device;
    gr::Annotated<std::vector<std::string>, "available_devices", gr::Doc<"Detected audio input devices in 'name [id]' format">>                                              available_devices;
    bool                                                                                                                                                                     _useDummyBackendForTests{false};

    GR_MAKE_REFLECTABLE(AudioSource, out, sample_rate, num_channels, buffer_frames, device, available_devices);

    using gr::Block<AudioSource<T>>::Block;
#if defined(__EMSCRIPTEN__)
    using BackendImpl = detail::EmscriptenAudioWorkletSourceBackend<T>;
#else
    using BackendImpl = detail::SoundIoSourceBackend<T>;
#endif

    BackendImpl               _backendImpl{};
    bool                      _failed{false};
    bool                      _formatTagPending{true};
    detail::AudioDeviceConfig _activeConfig{};
    std::mutex                _deviceMutex;

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

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        if (outSpan.empty()) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        if (auto pollResult = _backendImpl.poll(); !pollResult) {
            this->emitErrorMessage("AudioSource::processBulk()", pollResult.error());
            _backendImpl.requestStop();
            _failed = true;
            outSpan.publish(0U);
            return gr::work::Status::ERROR;
        }

        if (_failed) {
            outSpan.publish(0U);
            return gr::work::Status::ERROR;
        }

        const std::size_t channelCount = std::max<std::size_t>(1U, static_cast<std::size_t>(num_channels.value));
        const std::size_t nPublished   = _backendImpl.readToOutput(std::span<T>(outSpan), channelCount);
        if (nPublished > 0U && _formatTagPending) {
            publishFormatTag(outSpan, 0U);
            _formatTagPending = false;
        }

        outSpan.publish(nPublished);
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] std::uint32_t currentSampleRate() const {
        const auto rounded = std::lround(static_cast<double>(sample_rate.value));
        return rounded > 0L ? static_cast<std::uint32_t>(rounded) : 0U;
    }

    [[nodiscard]] std::uint32_t currentChannelCount() const { return num_channels.value > 0U ? static_cast<std::uint32_t>(num_channels.value) : 0U; }

    void publishFormatTag(gr::OutputSpanLike auto& outSpan, std::size_t offset) const {
        auto tagMap = property_map{};
        gr::tag::put(tagMap, gr::tag::SAMPLE_RATE, sample_rate.value);
        gr::tag::put(tagMap, gr::tag::NUM_CHANNELS, num_channels.value);
        gr::tag::put(tagMap, gr::tag::SIGNAL_NAME, std::string("audio_capture"));
        outSpan.publishTag(std::move(tagMap), offset);
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
Pair it with `WavSource` or any other block that already produces decoded PCM.)"">;

    gr::PortIn<T> in;

    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"PCM sample rate. Updated automatically; not intended to be set by the user.">>                 sample_rate   = 48000.f;
    gr::Annotated<gr::Size_t, "num_channels", gr::Visible, gr::Doc<"PCM interleaved channel count. Updated automatically; not intended to be set by the user.">>             num_channels  = 1U;
    gr::Annotated<gr::Size_t, "buffer_frames", gr::Visible, gr::Doc<"Software queue depth in PCM frames">>                                                                   buffer_frames = 4096U;
    gr::Annotated<std::string, "device", gr::Visible, gr::Doc<"Device selector: empty or 'default' for system default, substring match on name, or '@id:...' for exact ID">> device;
    gr::Annotated<std::vector<std::string>, "available_devices", gr::Doc<"Detected audio output devices in 'name [id]' format">>                                             available_devices;
    bool                                                                                                                                                                     _useDummyBackendForTests{false};

    GR_MAKE_REFLECTABLE(AudioSink, in, sample_rate, num_channels, buffer_frames, device, available_devices);

    using gr::Block<AudioSink<T>>::Block;
#if defined(__EMSCRIPTEN__)
    using BackendImpl = detail::EmscriptenAudioWorkletSinkBackend<T>;
#else
    using BackendImpl = detail::SoundIoSinkBackend<T>;
#endif

    BackendImpl               _backendImpl{};
    bool                      _failed{false};
    detail::AudioDeviceConfig _activeConfig{};
    std::mutex                _deviceMutex;

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

        const std::size_t nWritten = _backendImpl.writeFromInput(inSpan, channelCount);
        std::ignore                = inSpan.consume(nWritten);
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

        available_devices = _backendImpl._availableDevices;
        _activeConfig     = {.sampleRate = actual.sampleRate, .numChannels = actual.numChannels, .bufferFrames = buffer_frames.value, .device = device.value};
        _failed           = false;
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
