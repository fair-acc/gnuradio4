#ifndef GNURADIO_AUDIO_SOUNDIO_BACKEND_HPP
#define GNURADIO_AUDIO_SOUNDIO_BACKEND_HPP

#include <gnuradio-4.0/audio/AudioBackends.hpp>

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
#include <cstring>
#include <expected>
#include <format>
#include <source_location>
#include <string_view>

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

template<AudioSample T>
struct SoundIoSinkBackend {
    AudioSinkState<T> _state{};
    SoundIo*          _soundio{nullptr};
    SoundIoDevice*    _device{nullptr};
    SoundIoOutStream* _outstream{nullptr};
    std::atomic<int>  _pendingError{SoundIoErrorNone};

    [[nodiscard]] std::expected<void, gr::Error> start(const AudioDeviceConfig& config) {
        shutdown();

        if (config.sampleRate == 0U || config.numChannels == 0U) {
            return std::unexpected(gr::Error("AudioSink requires sample_rate > 0 and num_channels > 0"));
        }

        _soundio = soundio_create();
        if (_soundio == nullptr) {
            return std::unexpected(gr::Error("soundio_create(): out of memory"));
        }

        const int connectError = config.useDummyBackendForTests ? soundio_connect_backend(_soundio, SoundIoBackendDummy) : soundio_connect(_soundio);
        if (connectError != SoundIoErrorNone) {
            shutdown();
            return std::unexpected(makeSoundIoError("soundio_connect()", connectError));
        }

        soundio_flush_events(_soundio);

        const int defaultDeviceIndex = soundio_default_output_device_index(_soundio);
        if (defaultDeviceIndex < 0) {
            shutdown();
            return std::unexpected(gr::Error("soundio_default_output_device_index(): no output device found"));
        }

        _device = soundio_get_output_device(_soundio, defaultDeviceIndex);
        if (_device == nullptr) {
            shutdown();
            return std::unexpected(gr::Error("soundio_get_output_device(): failed to acquire output device"));
        }

        _outstream = soundio_outstream_create(_device);
        if (_outstream == nullptr) {
            shutdown();
            return std::unexpected(gr::Error("soundio_outstream_create(): out of memory"));
        }

        const int                   channelCount = static_cast<int>(config.numChannels);
        const SoundIoChannelLayout* layout       = soundio_channel_layout_get_default(channelCount);
        if (layout == nullptr) {
            shutdown();
            return std::unexpected(gr::Error(std::format("libsoundio does not provide a default layout for {} channels", channelCount)));
        }

        _outstream->userdata           = this;
        _outstream->format             = soundIoFormatFor<T>();
        _outstream->sample_rate        = static_cast<int>(config.sampleRate);
        _outstream->layout             = *layout;
        _outstream->software_latency   = static_cast<double>(std::max<std::size_t>(1U, config.bufferFrames)) / static_cast<double>(config.sampleRate);
        _outstream->write_callback     = &SoundIoSinkBackend::writeCallback;
        _outstream->underflow_callback = &SoundIoSinkBackend::underflowCallback;
        _outstream->error_callback     = &SoundIoSinkBackend::errorCallback;
        _outstream->name               = "GNU Radio AudioSink";

        const int openError = soundio_outstream_open(_outstream);
        if (openError != SoundIoErrorNone) {
            shutdown();
            return std::unexpected(makeSoundIoError("soundio_outstream_open()", openError));
        }

        if (_outstream->layout_error != SoundIoErrorNone) {
            shutdown();
            return std::unexpected(makeSoundIoError("soundio_outstream_open(): layout", _outstream->layout_error));
        }

        _state.recreateBuffer(AudioSinkState<T>::bufferCapacitySamples(config.numChannels, config.bufferFrames));
        _state.stopRequested.store(false, std::memory_order_release);
        _pendingError.store(SoundIoErrorNone, std::memory_order_release);

        const int startError = soundio_outstream_start(_outstream);
        if (startError != SoundIoErrorNone) {
            shutdown();
            return std::unexpected(makeSoundIoError("soundio_outstream_start()", startError));
        }

        return {};
    }

    void shutdown() {
        _state.stopRequested.store(true, std::memory_order_release);

        if (_outstream != nullptr) {
            soundio_outstream_destroy(_outstream);
            _outstream = nullptr;
        }
        if (_device != nullptr) {
            soundio_device_unref(_device);
            _device = nullptr;
        }
        if (_soundio != nullptr) {
            soundio_destroy(_soundio);
            _soundio = nullptr;
        }

        _pendingError.store(SoundIoErrorNone, std::memory_order_release);
        _state.recreateBuffer(1U);
    }

    [[nodiscard]] std::expected<void, gr::Error> poll() {
        const int error = _pendingError.exchange(SoundIoErrorNone, std::memory_order_acq_rel);
        if (error != SoundIoErrorNone) {
            return std::unexpected(makeSoundIoError("libsoundio stream error", error));
        }
        return {};
    }

    void requestStop() { _state.stopRequested.store(true, std::memory_order_release); }

    template<typename InputSpan>
    [[nodiscard]] std::size_t writeFromInput(const InputSpan& inSpan, std::size_t channelCount) {
        return _state.writeFromInput(inSpan, channelCount);
    }

private:
    static void underflowCallback(SoundIoOutStream* /*outstream*/) {}

    static void errorCallback(SoundIoOutStream* outstream, int error) {
        auto* self = static_cast<SoundIoSinkBackend*>(outstream->userdata);
        if (self != nullptr) {
            self->storePendingError(error);
        }
    }

    void storePendingError(int error) {
        int expected = SoundIoErrorNone;
        std::ignore  = _pendingError.compare_exchange_strong(expected, error, std::memory_order_acq_rel);
    }

    static void writeCallback(SoundIoOutStream* outstream, int /*frameCountMin*/, int frameCountMax) {
        auto* self = static_cast<SoundIoSinkBackend*>(outstream->userdata);
        if (self == nullptr || frameCountMax <= 0) {
            return;
        }

        const std::size_t channelCount = std::max<std::size_t>(1U, static_cast<std::size_t>(outstream->layout.channel_count));
        int               framesLeft   = frameCountMax;

        while (framesLeft > 0) {
            SoundIoChannelArea* areas      = nullptr;
            int                 frameCount = framesLeft;
            const int           beginError = soundio_outstream_begin_write(outstream, &areas, &frameCount);
            if (beginError != SoundIoErrorNone) {
                if (beginError != SoundIoErrorUnderflow) {
                    self->storePendingError(beginError);
                }
                return;
            }

            if (frameCount <= 0) {
                break;
            }

            if (areas == nullptr) {
                const int endError = soundio_outstream_end_write(outstream);
                if (endError != SoundIoErrorNone && endError != SoundIoErrorUnderflow) {
                    self->storePendingError(endError);
                    return;
                }
                framesLeft -= frameCount;
                continue;
            }

            const std::size_t requestedFrames = static_cast<std::size_t>(frameCount);
            const std::size_t availableFrames = channelCount > 0U ? self->_state.reader.available() / channelCount : 0U;
            const std::size_t copiedFrames    = std::min(requestedFrames, availableFrames);

            if (copiedFrames > 0U) {
                auto readSpan = self->_state.reader.get(copiedFrames * channelCount);
                for (std::size_t frame = 0U; frame < requestedFrames; ++frame) {
                    for (std::size_t channel = 0U; channel < channelCount; ++channel) {
                        const T value = frame < copiedFrames ? readSpan[frame * channelCount + channel] : T{};
                        std::memcpy(areas[channel].ptr + areas[channel].step * static_cast<int>(frame), &value, sizeof(T));
                    }
                }
                std::ignore = readSpan.consume(copiedFrames * channelCount);
            } else {
                for (std::size_t frame = 0U; frame < requestedFrames; ++frame) {
                    for (std::size_t channel = 0U; channel < channelCount; ++channel) {
                        const T value{};
                        std::memcpy(areas[channel].ptr + areas[channel].step * static_cast<int>(frame), &value, sizeof(T));
                    }
                }
            }

            const int endError = soundio_outstream_end_write(outstream);
            if (endError != SoundIoErrorNone && endError != SoundIoErrorUnderflow) {
                self->storePendingError(endError);
                return;
            }

            framesLeft -= frameCount;
        }
    }
};

template<AudioSample T>
struct SoundIoSourceBackend {
    AudioSourceState<T> _state{};
    SoundIo*            _soundio{nullptr};
    SoundIoDevice*      _device{nullptr};
    SoundIoInStream*    _instream{nullptr};
    std::atomic<int>    _pendingError{SoundIoErrorNone};

    [[nodiscard]] std::expected<AudioSourceFormat, gr::Error> start(const AudioDeviceConfig& config) {
        shutdown();

        if (config.sampleRate == 0U || config.numChannels == 0U) {
            return std::unexpected(gr::Error("AudioSource requires sample_rate > 0 and num_channels > 0"));
        }

        _soundio = soundio_create();
        if (_soundio == nullptr) {
            return std::unexpected(gr::Error("soundio_create(): out of memory"));
        }

        const int connectError = config.useDummyBackendForTests ? soundio_connect_backend(_soundio, SoundIoBackendDummy) : soundio_connect(_soundio);
        if (connectError != SoundIoErrorNone) {
            shutdown();
            return std::unexpected(makeSoundIoError("soundio_connect()", connectError));
        }

        soundio_flush_events(_soundio);

        const int defaultDeviceIndex = soundio_default_input_device_index(_soundio);
        if (defaultDeviceIndex < 0) {
            shutdown();
            return std::unexpected(gr::Error("soundio_default_input_device_index(): no input device found"));
        }

        _device = soundio_get_input_device(_soundio, defaultDeviceIndex);
        if (_device == nullptr) {
            shutdown();
            return std::unexpected(gr::Error("soundio_get_input_device(): failed to acquire input device"));
        }

        _instream = soundio_instream_create(_device);
        if (_instream == nullptr) {
            shutdown();
            return std::unexpected(gr::Error("soundio_instream_create(): out of memory"));
        }

        const int                   channelCount = static_cast<int>(config.numChannels);
        const SoundIoChannelLayout* layout       = soundio_channel_layout_get_default(channelCount);
        if (layout == nullptr) {
            shutdown();
            return std::unexpected(gr::Error(std::format("libsoundio does not provide a default input layout for {} channels", channelCount)));
        }

        _instream->userdata          = this;
        _instream->format            = soundIoFormatFor<T>();
        _instream->sample_rate       = static_cast<int>(config.sampleRate);
        _instream->layout            = *layout;
        _instream->software_latency  = static_cast<double>(std::max<std::size_t>(1U, config.bufferFrames)) / static_cast<double>(config.sampleRate);
        _instream->read_callback     = &SoundIoSourceBackend::readCallback;
        _instream->overflow_callback = &SoundIoSourceBackend::overflowCallback;
        _instream->error_callback    = &SoundIoSourceBackend::errorCallback;
        _instream->name              = "GNU Radio AudioSource";
        _instream->non_terminal_hint = true;

        const int openError = soundio_instream_open(_instream);
        if (openError != SoundIoErrorNone) {
            shutdown();
            return std::unexpected(makeSoundIoError("soundio_instream_open()", openError));
        }

        if (_instream->layout_error != SoundIoErrorNone) {
            shutdown();
            return std::unexpected(makeSoundIoError("soundio_instream_open(): layout", _instream->layout_error));
        }

        const auto activeChannelCount = static_cast<std::uint32_t>(std::max(1, _instream->layout.channel_count));
        const auto activeSampleRate   = static_cast<std::uint32_t>(std::max(1, _instream->sample_rate));
        _state.recreateBuffer(AudioSourceState<T>::bufferCapacitySamples(activeChannelCount, config.bufferFrames));
        _state.stopRequested.store(false, std::memory_order_release);
        _pendingError.store(SoundIoErrorNone, std::memory_order_release);

        const int startError = soundio_instream_start(_instream);
        if (startError != SoundIoErrorNone) {
            shutdown();
            return std::unexpected(makeSoundIoError("soundio_instream_start()", startError));
        }

        return AudioSourceFormat{
            .sampleRate  = activeSampleRate,
            .numChannels = activeChannelCount,
        };
    }

    void shutdown() {
        _state.stopRequested.store(true, std::memory_order_release);

        if (_instream != nullptr) {
            soundio_instream_destroy(_instream);
            _instream = nullptr;
        }
        if (_device != nullptr) {
            soundio_device_unref(_device);
            _device = nullptr;
        }
        if (_soundio != nullptr) {
            soundio_destroy(_soundio);
            _soundio = nullptr;
        }

        _pendingError.store(SoundIoErrorNone, std::memory_order_release);
        _state.recreateBuffer(1U);
    }

    [[nodiscard]] std::expected<void, gr::Error> poll() {
        const int error = _pendingError.exchange(SoundIoErrorNone, std::memory_order_acq_rel);
        if (error != SoundIoErrorNone) {
            return std::unexpected(makeSoundIoError("libsoundio capture error", error));
        }
        return {};
    }

    void requestStop() { _state.stopRequested.store(true, std::memory_order_release); }

    [[nodiscard]] std::size_t readToOutput(std::span<T> output, std::size_t channelCount) { return _state.readToOutput(output, channelCount); }

private:
    static void overflowCallback(SoundIoInStream* /*instream*/) {}

    static void errorCallback(SoundIoInStream* instream, int error) {
        auto* self = static_cast<SoundIoSourceBackend*>(instream->userdata);
        if (self != nullptr) {
            self->storePendingError(error);
        }
    }

    void storePendingError(int error) {
        int expected = SoundIoErrorNone;
        std::ignore  = _pendingError.compare_exchange_strong(expected, error, std::memory_order_acq_rel);
    }

    void writeSilenceFrames(std::size_t frameCount, std::size_t channelCount) {
        if (frameCount == 0U || channelCount == 0U) {
            return;
        }

        const std::size_t nSamplesToWrite = std::min(frameCount * channelCount, wholeFrameSamples(_state.writer.available(), channelCount));
        if (nSamplesToWrite == 0U) {
            return;
        }

        auto writeSpan = _state.writer.tryReserve(nSamplesToWrite);
        if (writeSpan.empty()) {
            return;
        }

        const std::size_t published = wholeFrameSamples(writeSpan.size(), channelCount);
        if (published == 0U) {
            return;
        }

        std::fill_n(writeSpan.begin(), static_cast<std::ptrdiff_t>(published), T{});
        writeSpan.publish(published);
    }

    void writeFramesFromAreas(SoundIoChannelArea* areas, std::size_t frameCount, std::size_t channelCount) {
        if (areas == nullptr || frameCount == 0U || channelCount == 0U) {
            return;
        }

        const std::size_t nSamplesToWrite = std::min(frameCount * channelCount, wholeFrameSamples(_state.writer.available(), channelCount));
        if (nSamplesToWrite == 0U) {
            return;
        }

        auto writeSpan = _state.writer.tryReserve(nSamplesToWrite);
        if (writeSpan.empty()) {
            return;
        }

        const std::size_t published = wholeFrameSamples(writeSpan.size(), channelCount);
        if (published == 0U) {
            return;
        }

        const std::size_t chunkFrames = published / channelCount;
        for (std::size_t frame = 0U; frame < chunkFrames; ++frame) {
            for (std::size_t channel = 0U; channel < channelCount; ++channel) {
                T value{};
                std::memcpy(&value, areas[channel].ptr, sizeof(T));
                writeSpan[frame * channelCount + channel] = value;
                areas[channel].ptr += areas[channel].step;
            }
        }

        writeSpan.publish(published);
    }

    static void readCallback(SoundIoInStream* instream, int /*frameCountMin*/, int frameCountMax) {
        auto* self = static_cast<SoundIoSourceBackend*>(instream->userdata);
        if (self == nullptr || frameCountMax <= 0) {
            return;
        }

        const std::size_t channelCount = std::max<std::size_t>(1U, static_cast<std::size_t>(instream->layout.channel_count));
        int               framesLeft   = frameCountMax;

        while (framesLeft > 0) {
            SoundIoChannelArea* areas      = nullptr;
            int                 frameCount = framesLeft;
            const int           beginError = soundio_instream_begin_read(instream, &areas, &frameCount);
            if (beginError != SoundIoErrorNone) {
                self->storePendingError(beginError);
                return;
            }

            if (frameCount <= 0) {
                break;
            }

            const std::size_t frames = static_cast<std::size_t>(frameCount);
            if (areas == nullptr) {
                self->writeSilenceFrames(frames, channelCount);
            } else {
                self->writeFramesFromAreas(areas, frames, channelCount);
            }

            const int endError = soundio_instream_end_read(instream);
            if (endError != SoundIoErrorNone) {
                self->storePendingError(endError);
                return;
            }

            framesLeft -= frameCount;
        }
    }
};

#endif

} // namespace gr::audio::detail

#endif // GNURADIO_AUDIO_SOUNDIO_BACKEND_HPP
