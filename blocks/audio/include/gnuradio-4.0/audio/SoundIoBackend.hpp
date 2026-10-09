#ifndef GNURADIO_AUDIO_SOUNDIO_BACKEND_HPP
#define GNURADIO_AUDIO_SOUNDIO_BACKEND_HPP

#include <gnuradio-4.0/audio/AudioBackends.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#if !defined(__EMSCRIPTEN__)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
#include <soundio/soundio.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <ranges>
#include <source_location>
#include <string_view>
#include <utility>
#include <vector>

namespace gr::audio::detail {

#if !defined(__EMSCRIPTEN__)

template<AudioSample T>
[[nodiscard]] constexpr SoundIoFormat soundIoFormatFor();

template<>
[[nodiscard]] constexpr SoundIoFormat soundIoFormatFor<float>() {
    return SoundIoFormatFloat32NE;
}

template<>
[[nodiscard]] constexpr SoundIoFormat soundIoFormatFor<std::int16_t>() {
    return SoundIoFormatS16NE;
}

inline gr::Error makeSoundIoError(std::string_view operation, int error, std::source_location location = std::source_location::current()) { return gr::Error(std::format("{}: {}", operation, soundio_strerror(error)), location); }

[[nodiscard]] inline const SoundIoChannelLayout* selectSoundIoLayout(SoundIoDevice* device, std::uint32_t requestedChannels) {
    const auto* preferred = soundio_channel_layout_get_default(static_cast<int>(requestedChannels));
    if (preferred != nullptr && soundio_device_supports_layout(device, preferred)) {
        return preferred;
    }
    const int  requested    = static_cast<int>(requestedChannels);
    auto       layouts      = std::span(device->layouts, static_cast<std::size_t>(std::max(0, device->layout_count))) | std::views::filter([](const SoundIoChannelLayout& layout) { return layout.channel_count > 0; });
    const auto fallbackRank = [requested](const SoundIoChannelLayout& layout) { return std::pair{layout.channel_count < requested, std::abs(layout.channel_count - requested)}; };
    const auto selected     = std::ranges::min_element(layouts, std::less{}, fallbackRank);
    return selected == layouts.end() ? nullptr : &*selected;
}

[[nodiscard]] inline std::expected<int, gr::Error> selectSoundIoSampleRate(SoundIoDevice* device, std::uint32_t requestedRate) {
    const int rate = soundio_device_nearest_sample_rate(device, static_cast<int>(requestedRate));
    if (rate <= 0) {
        return std::unexpected(gr::Error(std::format("audio device '{}' reports no supported sample rate", device->name != nullptr ? device->name : "")));
    }
    return rate;
}

[[nodiscard]] inline std::vector<AudioDeviceInfo> enumerateSoundIoDevices(SoundIo* sio, bool isInput) {
    const int                    count = isInput ? soundio_input_device_count(sio) : soundio_output_device_count(sio);
    std::vector<AudioDeviceInfo> result;
    result.reserve(static_cast<std::size_t>(std::max(0, count)));
    for (int i = 0; i < count; ++i) {
        SoundIoDevice* dev = isInput ? soundio_get_input_device(sio, i) : soundio_get_output_device(sio, i);
        if (dev != nullptr) {
            result.push_back({.name = dev->name != nullptr ? dev->name : "", .id = dev->id != nullptr ? dev->id : ""});
            soundio_device_unref(dev);
        }
    }
    return result;
}

[[nodiscard]] inline std::expected<SoundIoDevice*, gr::Error> resolveSoundIoDevice(SoundIo* sio, std::string_view deviceSpec, bool isInput, std::span<const AudioDeviceInfo> deviceInfos) {
    auto resolved = resolveDeviceIndex(deviceSpec, deviceInfos);
    if (resolved.has_value()) {
        SoundIoDevice* dev = isInput ? soundio_get_input_device(sio, static_cast<int>(*resolved)) : soundio_get_output_device(sio, static_cast<int>(*resolved));
        if (dev == nullptr) {
            return std::unexpected(gr::Error(std::format("failed to acquire {} device at index {}", isInput ? "input" : "output", *resolved)));
        }
        return dev;
    }

    if (!isDefaultDevice(deviceSpec)) {
        return std::unexpected(gr::Error(std::format("no {} device matching '{}' found", isInput ? "input" : "output", deviceSpec)));
    }

    const int defaultIndex = isInput ? soundio_default_input_device_index(sio) : soundio_default_output_device_index(sio);
    if (defaultIndex < 0) {
        return std::unexpected(gr::Error(std::format("no default {} device found", isInput ? "input" : "output")));
    }
    SoundIoDevice* dev = isInput ? soundio_get_input_device(sio, defaultIndex) : soundio_get_output_device(sio, defaultIndex);
    if (dev == nullptr) {
        return std::unexpected(gr::Error(std::format("failed to acquire default {} device", isInput ? "input" : "output")));
    }
    return dev;
}

struct SoundIoSession {
    SoundIo*                 soundio{nullptr};
    SoundIoDevice*           device{nullptr};
    std::atomic<int>         pendingError{SoundIoErrorNone};
    std::vector<std::string> availableDevices;

    [[nodiscard]] std::expected<void, gr::Error> open(const AudioDeviceConfig& config, bool isInput) {
        soundio = soundio_create();
        if (soundio == nullptr) {
            return std::unexpected(gr::Error("soundio_create(): out of memory"));
        }
        if (const int error = config.useDummyBackendForTests ? soundio_connect_backend(soundio, SoundIoBackendDummy) : soundio_connect(soundio); error != SoundIoErrorNone) {
            return std::unexpected(makeSoundIoError("soundio_connect()", error));
        }
        soundio_flush_events(soundio);
        const auto deviceInfos = enumerateSoundIoDevices(soundio, isInput);
        availableDevices       = formatDeviceList(deviceInfos);
        auto resolved          = resolveSoundIoDevice(soundio, config.device, isInput, deviceInfos);
        if (!resolved) {
            return std::unexpected(resolved.error());
        }
        device = *resolved;
        return {};
    }

    [[nodiscard]] std::expected<std::pair<SoundIoChannelLayout, int>, gr::Error> negotiate(const AudioDeviceConfig& config) const {
        const SoundIoChannelLayout* layout = selectSoundIoLayout(device, config.numChannels);
        if (layout == nullptr) {
            return std::unexpected(gr::Error(std::format("audio device '{}' has no supported channel layout", device->name != nullptr ? device->name : "")));
        }
        return selectSoundIoSampleRate(device, config.sampleRate).transform([layout](int rate) { return std::pair{*layout, rate}; });
    }

    void close() {
        if (device != nullptr) {
            soundio_device_unref(device);
            device = nullptr;
        }
        if (soundio != nullptr) {
            soundio_destroy(soundio);
            soundio = nullptr;
        }
        pendingError.store(SoundIoErrorNone, std::memory_order_release);
    }

    void storeError(int error) {
        int expected = SoundIoErrorNone;
        std::ignore  = pendingError.compare_exchange_strong(expected, error, std::memory_order_acq_rel);
    }

    [[nodiscard]] std::expected<void, gr::Error> takeError() {
        if (const int error = pendingError.exchange(SoundIoErrorNone, std::memory_order_acq_rel); error != SoundIoErrorNone) {
            return std::unexpected(makeSoundIoError("libsoundio stream error", error));
        }
        return {};
    }
};

template<AudioSample T>
struct SoundIoSinkBackend {
    AudioSinkState<T> _state{};
    SoundIoSession    _session;
    SoundIoOutStream* _outstream{nullptr};
    AudioStreamFormat _format{};

    [[nodiscard]] AudioSinkState<T>&       state() { return _state; }
    [[nodiscard]] const AudioSinkState<T>& state() const { return _state; }

    [[nodiscard]] std::expected<AudioStreamFormat, gr::Error> start(const AudioDeviceConfig& config) {
        shutdown();
        _state.intentionalSilence.store(config.intentionalSilence, std::memory_order_release);
        const auto fail = [this](gr::Error error) {
            shutdown();
            return std::unexpected(std::move(error));
        };
        if (auto opened = _session.open(config, false); !opened) {
            return fail(opened.error());
        }
        _outstream = soundio_outstream_create(_session.device);
        if (_outstream == nullptr) {
            return fail(gr::Error("soundio_outstream_create(): out of memory"));
        }
        const auto negotiated = _session.negotiate(config);
        if (!negotiated) {
            return fail(negotiated.error());
        }
        _outstream->userdata           = this;
        _outstream->format             = soundIoFormatFor<T>();
        _outstream->layout             = negotiated->first;
        _outstream->sample_rate        = negotiated->second;
        _outstream->software_latency   = static_cast<double>(config.bufferSeconds);
        _outstream->write_callback     = &SoundIoSinkBackend::writeCallback;
        _outstream->underflow_callback = &SoundIoSinkBackend::underflowCallback;
        _outstream->error_callback     = [](SoundIoOutStream* outstream, int error) { static_cast<SoundIoSinkBackend*>(outstream->userdata)->_session.storeError(error); };
        _outstream->name               = "GNU Radio AudioSink";
        if (const int error = soundio_outstream_open(_outstream); error != SoundIoErrorNone) {
            return fail(makeSoundIoError("soundio_outstream_open()", error));
        }
        if (const int error = _outstream->layout_error; error != SoundIoErrorNone) {
            return fail(makeSoundIoError("soundio_outstream_open(): layout", error));
        }
        _format = {.sampleRate = static_cast<std::uint32_t>(_outstream->sample_rate), .numChannels = static_cast<std::uint32_t>(_outstream->layout.channel_count)};
        _state.recreateBuffer(AudioSinkState<T>::bufferCapacitySamples(_format.numChannels, bufferFramesFor(config.bufferSeconds, _format.sampleRate)));
        _state.stopRequested.store(false, std::memory_order_release);
        if (const int error = soundio_outstream_start(_outstream); error != SoundIoErrorNone) {
            return fail(makeSoundIoError("soundio_outstream_start()", error));
        }
        return _format;
    }

    void shutdown() {
        _state.stopRequested.store(true, std::memory_order_release);
        if (_outstream != nullptr) {
            soundio_outstream_destroy(_outstream);
            _outstream = nullptr;
        }
        _session.close();
        _state.recreateBuffer(1U);
        _format = {};
    }

    [[nodiscard]] std::expected<void, gr::Error> poll() { return _session.takeError(); }

    [[nodiscard]] bool                     isStreamActive() const { return _outstream != nullptr; }
    [[nodiscard]] bool                     permissionGranted() const { return isStreamActive(); }
    [[nodiscard]] AudioBackendState        backendState() const { return isStreamActive() ? AudioBackendState::running : AudioBackendState::stopped; }
    [[nodiscard]] gr::property_map         diagnostics() const { return {{"state", std::string(gr::meta::enumName(backendState()).value_or(""))}}; }
    [[nodiscard]] double                   softwareLatency() const { return _outstream != nullptr ? _outstream->software_latency : 0.0; }
    [[nodiscard]] AudioStreamFormat        streamFormat() const { return _format; }
    [[nodiscard]] std::vector<std::string> availableDevices() const { return _session.availableDevices; }

private:
    static void underflowCallback(SoundIoOutStream* outstream) {
        auto* self = static_cast<SoundIoSinkBackend*>(outstream->userdata);
        if (!self->_state.intentionalSilence.load(std::memory_order_relaxed)) {
            self->_state.underrunCount.fetch_add(1UZ, std::memory_order_relaxed);
        }
    }

    static void writeCallback(SoundIoOutStream* outstream, int /*frameCountMin*/, int frameCountMax) {
        auto*             self         = static_cast<SoundIoSinkBackend*>(outstream->userdata);
        const std::size_t channelCount = std::max<std::size_t>(1U, static_cast<std::size_t>(outstream->layout.channel_count));
        for (int framesLeft = frameCountMax; framesLeft > 0;) {
            SoundIoChannelArea* areas      = nullptr;
            int                 frameCount = framesLeft;
            if (const int error = soundio_outstream_begin_write(outstream, &areas, &frameCount); error != SoundIoErrorNone) {
                if (error != SoundIoErrorUnderflow) {
                    self->_session.storeError(error);
                }
                return;
            }
            if (frameCount <= 0) {
                return;
            }
            if (areas != nullptr) {
                const std::size_t requestedFrames = static_cast<std::size_t>(frameCount);
                const std::size_t copiedFrames    = std::min(requestedFrames, self->_state.reader.available() / channelCount);
                if (copiedFrames < requestedFrames && !self->_state.intentionalSilence.load(std::memory_order_relaxed)) {
                    self->_state.underrunCount.fetch_add(1UZ, std::memory_order_relaxed);
                }
                auto readSpan = self->_state.reader.get(copiedFrames * channelCount);
                for (std::size_t frame = 0U; frame < requestedFrames; ++frame) {
                    for (std::size_t channel = 0U; channel < channelCount; ++channel) {
                        const T value = frame < copiedFrames ? readSpan[frame * channelCount + channel] : T{};
                        std::memcpy(areas[channel].ptr + areas[channel].step * static_cast<int>(frame), &value, sizeof(T));
                    }
                }
                std::ignore = readSpan.consume(copiedFrames * channelCount);
            }
            if (const int error = soundio_outstream_end_write(outstream); error != SoundIoErrorNone && error != SoundIoErrorUnderflow) {
                self->_session.storeError(error);
                return;
            }
            framesLeft -= frameCount;
        }
    }
};

template<AudioSample T>
struct SoundIoSourceBackend {
    AudioSourceState<T> _state{};
    SoundIoSession      _session;
    SoundIoInStream*    _instream{nullptr};
    AudioStreamFormat   _format{};

    [[nodiscard]] AudioSourceState<T>&       state() { return _state; }
    [[nodiscard]] const AudioSourceState<T>& state() const { return _state; }

    [[nodiscard]] std::expected<AudioStreamFormat, gr::Error> start(const AudioDeviceConfig& config) {
        shutdown();
        const auto fail = [this](gr::Error error) {
            shutdown();
            return std::unexpected(std::move(error));
        };
        if (auto opened = _session.open(config, true); !opened) {
            return fail(opened.error());
        }
        _instream = soundio_instream_create(_session.device);
        if (_instream == nullptr) {
            return fail(gr::Error("soundio_instream_create(): out of memory"));
        }
        const auto negotiated = _session.negotiate(config);
        if (!negotiated) {
            return fail(negotiated.error());
        }
        constexpr double kMaxCaptureCallbackSeconds = 0.05;
        _instream->userdata                         = this;
        _instream->format                           = soundIoFormatFor<T>();
        _instream->layout                           = negotiated->first;
        _instream->sample_rate                      = negotiated->second;
        _instream->software_latency                 = std::min(kMaxCaptureCallbackSeconds, static_cast<double>(config.bufferSeconds));
        _instream->read_callback                    = &SoundIoSourceBackend::readCallback;
        _instream->overflow_callback                = [](SoundIoInStream* instream) { static_cast<SoundIoSourceBackend*>(instream->userdata)->_state.overflowCount.fetch_add(1UZ, std::memory_order_relaxed); };
        _instream->error_callback                   = [](SoundIoInStream* instream, int error) { static_cast<SoundIoSourceBackend*>(instream->userdata)->_session.storeError(error); };
        _instream->name                             = "GNU Radio AudioSource";
        _instream->non_terminal_hint                = true;
        if (const int error = soundio_instream_open(_instream); error != SoundIoErrorNone) {
            return fail(makeSoundIoError("soundio_instream_open()", error));
        }
        if (const int error = _instream->layout_error; error != SoundIoErrorNone) {
            return fail(makeSoundIoError("soundio_instream_open(): layout", error));
        }
        _format            = {.sampleRate = static_cast<std::uint32_t>(_instream->sample_rate), .numChannels = static_cast<std::uint32_t>(_instream->layout.channel_count)};
        _state.numChannels = config.numChannels;
        _state.channelPadding.store(config.channelPadding, std::memory_order_release);
        _state.recreateBuffer(AudioSourceState<T>::bufferCapacitySamples(config.numChannels, bufferFramesFor(config.bufferSeconds, _format.sampleRate)));
        _state.stopRequested.store(false, std::memory_order_release);
        if (const int error = soundio_instream_start(_instream); error != SoundIoErrorNone) {
            return fail(makeSoundIoError("soundio_instream_start()", error));
        }
        return _format;
    }

    void shutdown() {
        _state.stopRequested.store(true, std::memory_order_release);
        if (_instream != nullptr) {
            soundio_instream_destroy(_instream);
            _instream = nullptr;
        }
        _session.close();
        _state.recreateBuffer(1U);
        _format = {};
    }

    [[nodiscard]] std::expected<void, gr::Error> poll() { return _session.takeError(); }

    [[nodiscard]] bool                     isStreamActive() const { return _instream != nullptr; }
    [[nodiscard]] bool                     permissionGranted() const { return isStreamActive(); }
    [[nodiscard]] AudioBackendState        backendState() const { return isStreamActive() ? AudioBackendState::running : AudioBackendState::stopped; }
    [[nodiscard]] gr::property_map         diagnostics() const { return {{"state", std::string(gr::meta::enumName(backendState()).value_or(""))}}; }
    [[nodiscard]] double                   softwareLatency() const { return _instream != nullptr ? _instream->software_latency : 0.0; }
    [[nodiscard]] AudioStreamFormat        streamFormat() const { return _format; }
    [[nodiscard]] std::vector<std::string> availableDevices() const { return _session.availableDevices; }
    [[nodiscard]] std::size_t              readToOutput(std::span<T> output, std::size_t channelCount) { return _state.readToOutput(output, channelCount); }

private:
    static void readCallback(SoundIoInStream* instream, int /*frameCountMin*/, int frameCountMax) {
        auto*             self         = static_cast<SoundIoSourceBackend*>(instream->userdata);
        const std::size_t channelCount = std::max<std::size_t>(1U, static_cast<std::size_t>(instream->layout.channel_count));
        for (int framesLeft = frameCountMax; framesLeft > 0;) {
            SoundIoChannelArea* areas      = nullptr;
            int                 frameCount = framesLeft;
            if (const int error = soundio_instream_begin_read(instream, &areas, &frameCount); error != SoundIoErrorNone) {
                self->_session.storeError(error);
                return;
            }
            if (frameCount <= 0) {
                return;
            }
            std::ignore = self->_state.writeChannels(static_cast<std::size_t>(frameCount), channelCount, self->_state.numChannels, [areas](std::size_t frame, std::size_t channel) {
                T value{};
                if (areas != nullptr) {
                    std::memcpy(&value, areas[channel].ptr + areas[channel].step * static_cast<int>(frame), sizeof(T));
                }
                return value;
            });
            if (const int error = soundio_instream_end_read(instream); error != SoundIoErrorNone) {
                self->_session.storeError(error);
                return;
            }
            framesLeft -= frameCount;
        }
    }
};

#endif

} // namespace gr::audio::detail

#endif // GNURADIO_AUDIO_SOUNDIO_BACKEND_HPP
