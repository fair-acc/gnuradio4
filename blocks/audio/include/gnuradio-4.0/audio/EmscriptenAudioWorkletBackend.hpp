#ifndef GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_WORKLET_BACKEND_HPP
#define GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_WORKLET_BACKEND_HPP

#if defined(__EMSCRIPTEN__)

#include <gnuradio-4.0/audio/AudioBackends.hpp>
#include <gnuradio-4.0/audio/EmscriptenAudioWorkletHelper.hpp>

#include <algorithm>
#include <cstddef>
#include <expected>
#include <format>

namespace gr::audio::detail {

template<typename TState>
void shutdownWorkletBackend(TState& state, WebAudioPendingWorkletInit& pendingInit, WebAudioWorkletRuntime& runtime, std::uintptr_t opaque) {
    state.stopRequested.store(true, std::memory_order_release);

    gr_webaudio_cancel_create_worklet_node(pendingInit);
    if (runtime.audioContext != 0) {
        gr_webaudio_unregister(opaque);
        gr_webaudio_destroy_worklet_runtime(runtime);
    }

    state.recreateBuffer(1U);
}

[[nodiscard]] inline std::expected<bool, gr::Error> pollPendingWorkletInit(std::uintptr_t opaque, WebAudioPendingWorkletInit& pendingInit, WebAudioWorkletRuntime& runtime) {
    if (pendingInit.state == nullptr) {
        return false;
    }

    WebAudioWorkletRuntime readyRuntime{};
    auto                   result = gr_webaudio_poll_create_worklet_node(pendingInit, readyRuntime);
    if (!result) {
        return std::unexpected(result.error());
    }
    if (*result) {
        runtime = readyRuntime;
        gr_webaudio_register_context(opaque, runtime.audioContext);
    }
    return *result;
}

template<AudioSample T>
struct EmscriptenAudioWorkletSinkBackend {
    AudioSinkState<T>          _state{};
    WebAudioPendingWorkletInit _pendingInit{};
    WebAudioWorkletRuntime     _runtime{};
    std::size_t                _channelCount{0U};
    std::vector<std::string>   _availableDevices;

    [[nodiscard]] std::expected<AudioStreamFormat, gr::Error> start(const AudioDeviceConfig& config) {
        shutdown();

        if (config.sampleRate == 0U || config.numChannels == 0U) {
            return std::unexpected(gr::Error("AudioSink requires sample_rate > 0 and num_channels > 0"));
        }

        _state.recreateBuffer(AudioSinkState<T>::bufferCapacitySamples(config.numChannels, config.bufferFrames));
        _state.stopRequested.store(false, std::memory_order_release);

        WebAudioWorkletNodeConfig workletConfig{};
        workletConfig.requestedSampleRate = static_cast<int>(config.sampleRate);
        workletConfig.numberOfInputs      = 0;
        workletConfig.outputChannelCount  = static_cast<int>(config.numChannels);

        auto pendingInit = gr_webaudio_begin_create_worklet_node(workletConfig, &EmscriptenAudioWorkletSinkBackend::processAudio, this);
        if (!pendingInit) {
            shutdown();
            return std::unexpected(pendingInit.error());
        }

        _pendingInit      = *pendingInit;
        _channelCount     = static_cast<std::size_t>(config.numChannels);
        _availableDevices = {"default [default]"};
        return AudioStreamFormat{
            .sampleRate  = _pendingInit.sampleRate,
            .numChannels = config.numChannels,
        };
    }

    void shutdown() {
        shutdownWorkletBackend(_state, _pendingInit, _runtime, reinterpret_cast<std::uintptr_t>(this));
        _channelCount = 0U;
    }

    [[nodiscard]] std::expected<void, gr::Error> poll() {
        if (auto result = pollPendingWorkletInit(reinterpret_cast<std::uintptr_t>(this), _pendingInit, _runtime); !result) {
            return std::unexpected(result.error());
        }
        return {};
    }

    void requestStop() { _state.stopRequested.store(true, std::memory_order_release); }

    template<typename InputSpan>
    [[nodiscard]] std::size_t writeFromInput(const InputSpan& inSpan, std::size_t channelCount) {
        return _state.writeFromInput(inSpan, channelCount);
    }

private:
    static bool processAudio(int /*numInputs*/, const AudioSampleFrame* /*inputs*/, int numOutputs, AudioSampleFrame* outputs, int /*numParams*/, const AudioParamFrame* /*params*/, void* userData) {
        auto* self = static_cast<EmscriptenAudioWorkletSinkBackend*>(userData);
        if (numOutputs <= 0 || outputs == nullptr) {
            return true;
        }

        auto& firstOutput = outputs[0];
        if (firstOutput.data == nullptr || firstOutput.samplesPerChannel <= 0 || firstOutput.numberOfChannels <= 0) {
            return true;
        }

        const std::size_t frameCount   = static_cast<std::size_t>(firstOutput.samplesPerChannel);
        const std::size_t channelCount = static_cast<std::size_t>(firstOutput.numberOfChannels);
        const std::size_t sampleCount  = frameCount * channelCount;

        if (self == nullptr || self->_channelCount == 0U || self->_channelCount != channelCount) {
            std::fill_n(firstOutput.data, static_cast<std::ptrdiff_t>(sampleCount), 0.0f);
        } else {
            // WebAudio uses planar channel buffers; the sink queue stores interleaved PCM frames.
            self->_state.readPlanarFloat(firstOutput.data, frameCount, channelCount);
        }

        for (int outputIndex = 1; outputIndex < numOutputs; ++outputIndex) {
            if (outputs[outputIndex].data != nullptr && outputs[outputIndex].samplesPerChannel > 0 && outputs[outputIndex].numberOfChannels > 0) {
                const std::size_t samples = static_cast<std::size_t>(outputs[outputIndex].samplesPerChannel) * static_cast<std::size_t>(outputs[outputIndex].numberOfChannels);
                std::fill_n(outputs[outputIndex].data, static_cast<std::ptrdiff_t>(samples), 0.0f);
            }
        }

        return true;
    }
};

template<AudioSample T>
struct EmscriptenAudioWorkletSourceBackend {
    AudioSourceState<T>        _state{};
    WebAudioPendingWorkletInit _pendingInit{};
    WebAudioWorkletRuntime     _runtime{};
    std::size_t                _channelCount{0U};
    std::vector<std::string>   _availableDevices;

    [[nodiscard]] std::expected<AudioStreamFormat, gr::Error> start(const AudioDeviceConfig& config) {
        shutdown();

        if (config.sampleRate == 0U || config.numChannels == 0U) {
            return std::unexpected(gr::Error("AudioSource requires sample_rate > 0 and num_channels > 0"));
        }

        _channelCount = std::max<std::size_t>(1U, static_cast<std::size_t>(config.numChannels));
        _state.recreateBuffer(AudioSourceState<T>::bufferCapacitySamples(_channelCount, config.bufferFrames));
        _state.stopRequested.store(false, std::memory_order_release);

        WebAudioWorkletNodeConfig workletConfig{};
        workletConfig.requestedSampleRate = 0;
        workletConfig.numberOfInputs      = 1;
        workletConfig.outputChannelCount  = static_cast<int>(_channelCount);

        auto pendingInit = gr_webaudio_begin_create_worklet_node(workletConfig, &EmscriptenAudioWorkletSourceBackend::processAudio, this);
        if (!pendingInit) {
            shutdown();
            return std::unexpected(pendingInit.error());
        }
        _pendingInit      = *pendingInit;
        _availableDevices = {"default [default]"};

        return AudioStreamFormat{
            .sampleRate  = _pendingInit.sampleRate,
            .numChannels = static_cast<std::uint32_t>(_channelCount),
        };
    }

    void shutdown() {
        shutdownWorkletBackend(_state, _pendingInit, _runtime, reinterpret_cast<std::uintptr_t>(this));
        _channelCount = 0U;
    }

    [[nodiscard]] std::expected<void, gr::Error> poll() {
        if (auto result = pollPendingWorkletInit(reinterpret_cast<std::uintptr_t>(this), _pendingInit, _runtime); !result) {
            return std::unexpected(result.error());
        } else if (*result) {
            if (gr_webaudio_capture_attach(reinterpret_cast<std::uintptr_t>(this), _runtime.node, static_cast<int>(_channelCount)) == 0) {
                gr_webaudio_unregister(reinterpret_cast<std::uintptr_t>(this));
                gr_webaudio_destroy_worklet_runtime(_runtime);
                _runtime = {};
                return std::unexpected(gr::Error("WebAudio microphone initialisation failed"));
            }
        }

        if (_runtime.audioContext != 0 && gr_webaudio_capture_failed(reinterpret_cast<std::uintptr_t>(this)) != 0) {
            return std::unexpected(gr::Error("WebAudio microphone capture failed"));
        }
        return {};
    }

    void requestStop() { _state.stopRequested.store(true, std::memory_order_release); }

    [[nodiscard]] std::size_t readToOutput(std::span<T> output, std::size_t channelCount) { return _state.readToOutput(output, channelCount); }

private:
    static bool processAudio(int numInputs, const AudioSampleFrame* inputs, int numOutputs, AudioSampleFrame* outputs, int /*numParams*/, const AudioParamFrame* /*params*/, void* userData) {
        auto* self = static_cast<EmscriptenAudioWorkletSourceBackend*>(userData);

        for (int outputIndex = 0; outputIndex < numOutputs; ++outputIndex) {
            if (outputs != nullptr && outputs[outputIndex].data != nullptr && outputs[outputIndex].samplesPerChannel > 0 && outputs[outputIndex].numberOfChannels > 0) {
                const std::size_t samples = static_cast<std::size_t>(outputs[outputIndex].samplesPerChannel) * static_cast<std::size_t>(outputs[outputIndex].numberOfChannels);
                std::fill_n(outputs[outputIndex].data, static_cast<std::ptrdiff_t>(samples), 0.0f);
            }
        }

        if (self == nullptr || numInputs <= 0 || inputs == nullptr || inputs[0].data == nullptr || inputs[0].samplesPerChannel <= 0 || self->_channelCount == 0U) {
            return true;
        }

        const auto&       firstInput    = inputs[0];
        const std::size_t frameCount    = static_cast<std::size_t>(firstInput.samplesPerChannel);
        const std::size_t inputChannels = static_cast<std::size_t>(std::max(0, firstInput.numberOfChannels));
        const std::size_t channelCount  = self->_channelCount;
        // WebAudio provides planar channel buffers; the source queue stores interleaved PCM frames.
        static_cast<void>(self->_state.writePlanarFloat(firstInput.data, frameCount, inputChannels, channelCount));
        return true;
    }
};

} // namespace gr::audio::detail

#endif

#endif // GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_WORKLET_BACKEND_HPP
