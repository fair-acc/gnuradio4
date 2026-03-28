#ifndef GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_BACKEND_HPP
#define GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_BACKEND_HPP

#if defined(__EMSCRIPTEN__)

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/audio/AudioBackends.hpp>
#include <gnuradio-4.0/common/DeviceRegistry.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <string_view>

#include <emscripten.h>
#include <emscripten/threading.h>
#include <emscripten/webaudio.h>
#include <malloc.h>

namespace gr::audio::detail {

// -- worklet runtime types --

struct WebAudioWorkletRuntime {
    EMSCRIPTEN_WEBAUDIO_T           audioContext{0};
    EMSCRIPTEN_AUDIO_WORKLET_NODE_T node{0};
    void*                           workletStack{nullptr};
    std::uint32_t                   sampleRate{0U};
};

struct WebAudioWorkletNodeConfig {
    int requestedSampleRate{0};
    int numberOfInputs{0};
    int outputChannelCount{1};
};

enum class InitStatus : int {
    pending,
    succeeded,
    failed,
    cancelled,
};

struct PendingWorkletInitState {
    std::atomic<unsigned int>            refCount{2U};
    WebAudioWorkletNodeConfig            config{};
    EmscriptenWorkletNodeProcessCallback processCallback{nullptr};
    void*                                userData{nullptr};
    WebAudioWorkletRuntime               runtime{};
    std::string                          errorMessage{};
    std::atomic<InitStatus>              status{InitStatus::pending};
    std::atomic<bool>                    cancelRequested{false};
};

struct WebAudioPendingWorkletInit {
    std::uint32_t            sampleRate{0U};
    PendingWorkletInitState* state{nullptr};
};

constexpr std::size_t kAudioWorkletStackSize = 128U * 1024U;

struct MainThreadJsTask {
    std::uintptr_t                  opaque{0U};
    EMSCRIPTEN_WEBAUDIO_T           audioContext{0};
    EMSCRIPTEN_AUDIO_WORKLET_NODE_T workletNode{0};
    int                             channelCount{0};
    int                             result{0};
    const char*                     deviceId{nullptr};
};

// -- main-thread dispatch --

inline void runOnMainThread(void (*fn)(void*), void* opaque) {
    if (emscripten_is_main_runtime_thread()) {
        fn(opaque);
    } else {
        emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VI, fn, opaque);
    }
}

// -- worklet lifecycle helpers --

inline void cleanupRuntime(WebAudioWorkletRuntime& runtime) {
    if (runtime.node != 0) {
        emscripten_destroy_web_audio_node(runtime.node);
        runtime.node = 0;
    }
    if (runtime.audioContext != 0) {
        emscripten_destroy_audio_context(runtime.audioContext);
        runtime.audioContext = 0;
    }
    if (runtime.workletStack != nullptr) {
        std::free(runtime.workletStack);
        runtime.workletStack = nullptr;
    }
    runtime.sampleRate = 0U;
}

inline void releasePendingInit(PendingWorkletInitState* state) {
    if (state != nullptr && state->refCount.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
        delete state;
    }
}

inline void finishPendingInit(PendingWorkletInitState* state, InitStatus status, std::string_view errorMessage = {}) {
    if (state == nullptr) {
        return;
    }

    if (state->cancelRequested.load(std::memory_order_acquire)) {
        cleanupRuntime(state->runtime);
        state->errorMessage.clear();
        state->status.store(InitStatus::cancelled, std::memory_order_release);
        releasePendingInit(state);
        return;
    }

    if (status == InitStatus::failed) {
        cleanupRuntime(state->runtime);
        state->errorMessage = errorMessage;
    } else {
        state->errorMessage.clear();
    }
    state->status.store(status, std::memory_order_release);
    releasePendingInit(state);
}

inline void processorCreated(EMSCRIPTEN_WEBAUDIO_T audioContext, bool success, void* userData) {
    auto* state = static_cast<PendingWorkletInitState*>(userData);
    if (state == nullptr) {
        return;
    }
    if (!success) {
        finishPendingInit(state, InitStatus::failed, "WebAudio AudioWorklet processor creation failed");
        return;
    }

    const int outputChannelCount     = std::max(1, state->config.outputChannelCount);
    int       outputChannelCounts[1] = {outputChannelCount};

    EmscriptenAudioWorkletNodeCreateOptions nodeOptions{};
    nodeOptions.numberOfInputs        = std::max(0, state->config.numberOfInputs);
    nodeOptions.numberOfOutputs       = 1;
    nodeOptions.outputChannelCounts   = outputChannelCounts;
    nodeOptions.channelCount          = static_cast<unsigned long>(outputChannelCount);
    nodeOptions.channelCountMode      = WEBAUDIO_CHANNEL_COUNT_MODE_EXPLICIT;
    nodeOptions.channelInterpretation = WEBAUDIO_CHANNEL_INTERPRETATION_DISCRETE;

    state->runtime.node = emscripten_create_wasm_audio_worklet_node(audioContext, "gr-audio-worklet", &nodeOptions, state->processCallback, state->userData);
    if (state->runtime.node == 0) {
        finishPendingInit(state, InitStatus::failed, "WebAudio AudioWorklet node creation failed");
        return;
    }

    emscripten_audio_node_connect(state->runtime.node, audioContext, 0, 0);
    finishPendingInit(state, InitStatus::succeeded);
}

inline void workletThreadStarted(EMSCRIPTEN_WEBAUDIO_T audioContext, bool success, void* userData) {
    auto* state = static_cast<PendingWorkletInitState*>(userData);
    if (state == nullptr) {
        return;
    }
    if (!success) {
        finishPendingInit(state, InitStatus::failed, "WebAudio AudioWorklet thread initialisation failed");
        return;
    }

    WebAudioWorkletProcessorCreateOptions processorOptions{};
    processorOptions.name = "gr-audio-worklet";
    emscripten_create_wasm_audio_worklet_processor_async(audioContext, &processorOptions, &processorCreated, state);
}

inline void startCreateWorkletNodeOnMainThread(void* opaque) {
    auto* state = static_cast<PendingWorkletInitState*>(opaque);
    if (state == nullptr) {
        return;
    }
    if (state->processCallback == nullptr) {
        finishPendingInit(state, InitStatus::failed, "WebAudio AudioWorklet callback is null");
        return;
    }

    EmscriptenWebAudioCreateAttributes attributes{};
    attributes.latencyHint    = "interactive";
    attributes.sampleRate     = static_cast<std::uint32_t>(std::max(0, state->config.requestedSampleRate));
    attributes.renderSizeHint = AUDIO_CONTEXT_RENDER_SIZE_DEFAULT;

    state->runtime.audioContext = emscripten_create_audio_context(&attributes);
    if (state->runtime.audioContext == 0) {
        finishPendingInit(state, InitStatus::failed, "WebAudio AudioContext creation failed");
        return;
    }
    state->runtime.sampleRate = static_cast<std::uint32_t>(std::max(1, emscripten_audio_context_sample_rate(state->runtime.audioContext)));

    state->runtime.workletStack = memalign(16, kAudioWorkletStackSize);
    if (state->runtime.workletStack == nullptr) {
        finishPendingInit(state, InitStatus::failed, "WebAudio AudioWorklet stack allocation failed");
        return;
    }

    emscripten_start_wasm_audio_worklet_thread_async(state->runtime.audioContext, state->runtime.workletStack, static_cast<std::uint32_t>(kAudioWorkletStackSize), &workletThreadStarted, state);
}

// -- worklet node creation/polling/cancellation --

[[nodiscard]] inline std::expected<WebAudioPendingWorkletInit, gr::Error> gr_webaudio_begin_create_worklet_node(const WebAudioWorkletNodeConfig& config, EmscriptenWorkletNodeProcessCallback processCallback, void* userData) {
    auto* state            = new PendingWorkletInitState{};
    state->config          = config;
    state->processCallback = processCallback;
    state->userData        = userData;

    runOnMainThread(&startCreateWorkletNodeOnMainThread, state);

    const auto status = state->status.load(std::memory_order_acquire);
    if (status == InitStatus::failed) {
        const gr::Error error(state->errorMessage);
        releasePendingInit(state);
        return std::unexpected(error);
    }

    if (status == InitStatus::cancelled) {
        releasePendingInit(state);
        return std::unexpected(gr::Error("WebAudio AudioWorklet initialisation was cancelled"));
    }

    return WebAudioPendingWorkletInit{
        .sampleRate = state->runtime.sampleRate,
        .state      = state,
    };
}

[[nodiscard]] inline std::expected<bool, gr::Error> gr_webaudio_poll_create_worklet_node(WebAudioPendingWorkletInit& pendingInit, WebAudioWorkletRuntime& runtime) {
    auto* state = pendingInit.state;
    if (state == nullptr) {
        return false;
    }

    const auto finish = [&pendingInit, state]() {
        pendingInit = {};
        releasePendingInit(state);
    };
    const auto status = state->status.load(std::memory_order_acquire);

    if (status == InitStatus::pending) {
        return false;
    }

    if (status == InitStatus::succeeded) {
        runtime        = state->runtime;
        state->runtime = {};
        finish();
        return true;
    }

    if (status == InitStatus::failed) {
        const gr::Error error(state->errorMessage);
        finish();
        return std::unexpected(error);
    }

    if (status == InitStatus::cancelled) {
        finish();
        return std::unexpected(gr::Error("WebAudio AudioWorklet initialisation was cancelled"));
    }

    finish();
    return std::unexpected(gr::Error("WebAudio AudioWorklet initialisation reached an invalid state"));
}

inline void gr_webaudio_cancel_create_worklet_node(WebAudioPendingWorkletInit& pendingInit) {
    auto* state = pendingInit.state;
    if (state == nullptr) {
        return;
    }

    pendingInit = {};
    state->cancelRequested.store(true, std::memory_order_release);
    const auto status = state->status.load(std::memory_order_acquire);
    if (status == InitStatus::succeeded) {
        runOnMainThread(
            [](void* opaque) {
                auto* s = static_cast<PendingWorkletInitState*>(opaque);
                if (s != nullptr) {
                    cleanupRuntime(s->runtime);
                }
            },
            state);
    }
    releasePendingInit(state);
}

inline void gr_webaudio_destroy_worklet_runtime(WebAudioWorkletRuntime& runtime) {
    if (runtime.audioContext == 0 && runtime.node == 0 && runtime.workletStack == nullptr) {
        return;
    }
    runOnMainThread(
        [](void* opaque) {
            auto* rt = static_cast<WebAudioWorkletRuntime*>(opaque);
            if (rt == nullptr) {
                return;
            }
            cleanupRuntime(*rt);
        },
        &runtime);
}

// -- JavaScript bridge functions (main-thread callbacks) --

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdollar-in-identifier-extension"
#endif

// clang-format off
inline void registerContextOnMainThread(void* opaqueTask) {
    auto* task = static_cast<MainThreadJsTask*>(opaqueTask);
    if (task == nullptr) {
        return;
    }

    EM_ASM({
        const opaque = $0;
        const contextHandle = $1;
        const context = emscriptenGetAudioObject(contextHandle);
        if (!context) {
            return;
        }

        if (!globalThis.__grAudioWeb) {
            const audio = {};
            audio.devices = {};
            audio.unlock = function() {
                const devices = Object.values(audio.devices);
                for (let i = 0; i < devices.length; ++i) {
                    const state = devices[i];
                    if (state && !state.destroyed && state.context && state.context.resume) {
                        state.context.resume().catch(function(error) {
                            console.error('[Audio] Failed to resume WebAudio context', error);
                        });
                    }
                }

                document.removeEventListener('touchend', audio.unlock, true);
                document.removeEventListener('click', audio.unlock, true);
                document.removeEventListener('keydown', audio.unlock, true);
            };

            globalThis.__grAudioWeb = audio;
            document.addEventListener('touchend', audio.unlock, true);
            document.addEventListener('click', audio.unlock, true);
            document.addEventListener('keydown', audio.unlock, true);
        }

        const state = {};
        state.context = context;
        state.destroyed = false;
        state.failed = false;
        state.stream = null;
        state.streamNode = null;
        globalThis.__grAudioWeb.devices[opaque] = state;
        if (state.context.resume) {
            state.context.resume().catch(function(error) {
                console.error('[Audio] Failed to resume WebAudio context', error);
            });
        }
    }, task->opaque, task->audioContext);
}

inline void captureAttachOnMainThread(void* opaqueTask) {
    auto* task = static_cast<MainThreadJsTask*>(opaqueTask);
    if (task == nullptr) {
        return;
    }

    task->result = EM_ASM_INT({
        const opaque = $0;
        const nodeHandle = $1;
        const channelCount = $2;
        const deviceIdPtr = $3;

        if (!globalThis.__grAudioWeb) {
            return 0;
        }

        const state = globalThis.__grAudioWeb.devices[opaque];
        if (!state) {
            return 0;
        }

        const context = state.context;
        const node = emscriptenGetAudioObject(nodeHandle);
        if (!context || !node) {
            return 0;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('[AudioSource] navigator.mediaDevices.getUserMedia is not available');
            state.failed = true;
            return 0;
        }

        const audioConstraint = { channelCount: channelCount };
        if (deviceIdPtr !== 0) {
            const deviceId = UTF8ToString(deviceIdPtr);
            if (deviceId.length > 0) {
                audioConstraint.deviceId = { exact: deviceId };
            }
        }
        state.failed = false;
        navigator.mediaDevices.getUserMedia({ audio: audioConstraint, video: false })
            .then(function(stream) {
                if (state.destroyed || !globalThis.__grAudioWeb || globalThis.__grAudioWeb.devices[opaque] !== state) {
                    stream.getTracks().forEach(function(track) { track.stop(); });
                    return;
                }

                state.stream = stream;
                state.streamNode = context.createMediaStreamSource(stream);
                state.streamNode.connect(node);
            })
            .catch(function(error) {
                if (state.destroyed) {
                    return;
                }
                console.error('[AudioSource] Failed to get user media', error);
                state.failed = true;
            });

        return 1;
    }, task->opaque, task->workletNode, task->channelCount, task->deviceId);
}

inline void captureFailedOnMainThread(void* opaqueTask) {
    auto* task = static_cast<MainThreadJsTask*>(opaqueTask);
    if (task == nullptr) {
        return;
    }

    task->result = EM_ASM_INT({
        const opaque = $0;
        if (!globalThis.__grAudioWeb) {
            return 0;
        }

        const state = globalThis.__grAudioWeb.devices[opaque];
        if (!state) {
            return 0;
        }
        return state.failed ? 1 : 0;
    }, task->opaque);
}

inline void unregisterOnMainThread(void* opaqueTask) {
    auto* task = static_cast<MainThreadJsTask*>(opaqueTask);
    if (task == nullptr) {
        return;
    }

    EM_ASM({
        const opaque = $0;
        if (!globalThis.__grAudioWeb) {
            return;
        }

        const state = globalThis.__grAudioWeb.devices[opaque];
        if (!state) {
            return;
        }

        try {
            state.destroyed = true;
            if (state.streamNode) {
                try {
                    state.streamNode.disconnect();
                } catch (error) {
                }
            }
            if (state.stream) {
                try {
                    state.stream.getTracks().forEach(function(track) { track.stop(); });
                } catch (error) {
                }
            }
        } finally {
            delete globalThis.__grAudioWeb.devices[opaque];
            if (Object.keys(globalThis.__grAudioWeb.devices).length === 0) {
                document.removeEventListener('touchend', globalThis.__grAudioWeb.unlock, true);
                document.removeEventListener('click', globalThis.__grAudioWeb.unlock, true);
                document.removeEventListener('keydown', globalThis.__grAudioWeb.unlock, true);
                delete globalThis.__grAudioWeb;
            }
        }
    }, task->opaque);
}

inline int checkMicrophonePermissionOnMainThread_impl() {
    return EM_ASM_INT({
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            return -1;
        }
        return 0;
    });
}

inline int requestMicrophonePermissionOnMainThread_impl() {
    return EM_ASM_INT({
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            return -1;
        }
        // getUserMedia is async — we trigger it and track the result via globalThis
        if (!globalThis.__grAudioMicGranted) {
            globalThis.__grAudioMicGranted = 0;
        }
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then(function(stream) {
                stream.getTracks().forEach(function(track) { track.stop(); });
                globalThis.__grAudioMicGranted = 1;
            })
            .catch(function(error) {
                console.error('[Audio] Microphone permission denied', error);
                globalThis.__grAudioMicGranted = -1;
            });
        return 0;
    });
}

inline int getMicrophonePermissionState_impl() {
    return EM_ASM_INT({
        if (typeof globalThis.__grAudioMicGranted === 'undefined') {
            return 0;
        }
        return globalThis.__grAudioMicGranted;
    });
}
// clang-format on

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// -- C++ wrappers for JS bridge --

inline void gr_webaudio_register_context(std::uintptr_t opaque, EMSCRIPTEN_WEBAUDIO_T audioContext) {
    MainThreadJsTask task{.opaque = opaque, .audioContext = audioContext};
    runOnMainThread(&registerContextOnMainThread, &task);
}

inline int gr_webaudio_capture_attach(std::uintptr_t opaque, EMSCRIPTEN_AUDIO_WORKLET_NODE_T workletNode, int channelCount, const char* deviceId = nullptr) {
    MainThreadJsTask task{
        .opaque       = opaque,
        .workletNode  = workletNode,
        .channelCount = channelCount,
        .deviceId     = deviceId,
    };
    runOnMainThread(&captureAttachOnMainThread, &task);
    return task.result;
}

inline int gr_webaudio_capture_failed(std::uintptr_t opaque) {
    MainThreadJsTask task{.opaque = opaque};
    runOnMainThread(&captureFailedOnMainThread, &task);
    return task.result;
}

inline void gr_webaudio_unregister(std::uintptr_t opaque) {
    MainThreadJsTask task{.opaque = opaque};
    runOnMainThread(&unregisterOnMainThread, &task);
}

// -- backend helpers --

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

// -- sink backend --

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

// -- source backend --

template<AudioSample T>
struct EmscriptenAudioWorkletSourceBackend {
    AudioSourceState<T>        _state{};
    WebAudioPendingWorkletInit _pendingInit{};
    WebAudioWorkletRuntime     _runtime{};
    std::size_t                _channelCount{0U};
    std::string                _deviceId;
    std::vector<std::string>   _availableDevices;

    [[nodiscard]] std::expected<AudioStreamFormat, gr::Error> start(const AudioDeviceConfig& config) {
        shutdown();

        if (config.sampleRate == 0U || config.numChannels == 0U) {
            return std::unexpected(gr::Error("AudioSource requires sample_rate > 0 and num_channels > 0"));
        }

        _channelCount = std::max<std::size_t>(1U, static_cast<std::size_t>(config.numChannels));
        _deviceId     = config.device;
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
        _deviceId.clear();
    }

    [[nodiscard]] std::expected<void, gr::Error> poll() {
        if (auto result = pollPendingWorkletInit(reinterpret_cast<std::uintptr_t>(this), _pendingInit, _runtime); !result) {
            return std::unexpected(result.error());
        } else if (*result) {
            const char* devId = _deviceId.empty() ? nullptr : _deviceId.c_str();
            if (gr_webaudio_capture_attach(reinterpret_cast<std::uintptr_t>(this), _runtime.node, static_cast<int>(_channelCount), devId) == 0) {
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
        static_cast<void>(self->_state.writePlanarFloat(firstInput.data, frameCount, inputChannels, channelCount));
        return true;
    }
};

// -- WebAudioDevice: DeviceRegistry integration --

struct WebAudioDevice : gr::blocks::common::DeviceBase {
    int _permissionState{0}; // 0=unknown, 1=granted, -1=denied

    [[nodiscard]] std::string_view id() const noexcept override { return "audio"; }
    [[nodiscard]] std::string_view displayName() const noexcept override { return "Microphone (WebAudio)"; }

    void init() override { _permissionState = checkMicrophonePermissionOnMainThread_impl() < 0 ? -1 : 0; }

    [[nodiscard]] bool isApiAvailable() const noexcept override { return _permissionState >= 0; }
    [[nodiscard]] int  grantedCount() const noexcept override { return getMicrophonePermissionState_impl() > 0 ? 1 : 0; }

    void requestPermission() override { requestMicrophonePermissionOnMainThread_impl(); }

    [[nodiscard]] std::expected<int, std::string> connect(int /*portIndex*/, int /*param*/) override { return 0; }
    void                                          disconnect(int /*handle*/) override {}

    [[nodiscard]] std::string lastError() const override { return ""; }
};

inline gr::blocks::common::AutoRegister autoRegWebAudio(std::make_shared<WebAudioDevice>());

} // namespace gr::audio::detail

#endif // __EMSCRIPTEN__

#endif // GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_BACKEND_HPP
