#ifndef GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_BACKEND_HPP
#define GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_BACKEND_HPP

#if defined(__EMSCRIPTEN__)

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/audio/AudioBackends.hpp>
#include <gnuradio-4.0/common/DeviceRegistry.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <emscripten.h>
#include <emscripten/threading.h>
#include <emscripten/webaudio.h>
#include <malloc.h>

extern "C" {
EMSCRIPTEN_KEEPALIVE inline char __em_lib_deps_gr_audio[] __attribute__((section("em_lib_deps"), aligned(1))) = "$emscriptenRegisterAudioObject,$emscriptenGetAudioObject,$lengthBytesUTF8,$stringToUTF8";
}

namespace gr::audio::detail {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"

struct WebAudioWorkletRuntime {
    EMSCRIPTEN_WEBAUDIO_T           audioContext{0};
    EMSCRIPTEN_AUDIO_WORKLET_NODE_T node{0};
    void*                           workletStack{nullptr};
    std::uint32_t                   sampleRate{0U};
    std::uint32_t                   numChannels{0U};
    std::shared_ptr<void>           callbackOwner;
};

[[nodiscard]] inline std::vector<std::unique_ptr<WebAudioWorkletRuntime>>& mainThreadRetiredRuntimes() {
    static std::vector<std::unique_ptr<WebAudioWorkletRuntime>> runtimes;
    return runtimes;
}

extern "C" EMSCRIPTEN_KEEPALIVE inline void gr_audio_release_runtime(std::uintptr_t retired) {
    std::erase_if(mainThreadRetiredRuntimes(), [retired](const std::unique_ptr<WebAudioWorkletRuntime>& runtime) {
        if (reinterpret_cast<std::uintptr_t>(runtime.get()) != retired) {
            return false;
        }
        std::free(runtime->workletStack);
        return true;
    });
}

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
    std::uint32_t            numChannels{0U};
    PendingWorkletInitState* state{nullptr};
};

constexpr std::size_t kAudioWorkletStackSize = 128U * 1024U;

struct MainThreadJsTask {
    std::uintptr_t                  opaque{0U};
    EMSCRIPTEN_WEBAUDIO_T           audioContext{0};
    EMSCRIPTEN_AUDIO_WORKLET_NODE_T workletNode{0};
    int                             channelCount{0};
    int                             sampleRate{0};
    int                             result{0};
    const char*                     deviceId{nullptr};
    const char*                     field{nullptr};
    std::string                     text{};
};

inline void runOnMainThread(void (*fn)(void*), void* opaque) {
    if (emscripten_is_main_runtime_thread()) {
        fn(opaque);
    } else {
        emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VI, fn, opaque);
    }
}

inline MainThreadJsTask runOnMainThread(void (*fn)(void*), MainThreadJsTask task) {
    runOnMainThread(fn, &task);
    return task;
}

inline void cleanupRuntime(WebAudioWorkletRuntime& runtime) {
    if (runtime.node != 0) {
        emscripten_destroy_web_audio_node(runtime.node);
        runtime.node = 0;
    }
    if (runtime.audioContext == 0) {
        std::free(runtime.workletStack);
        runtime = {};
        return;
    }
    const auto context = runtime.audioContext;
    auto&      retired = mainThreadRetiredRuntimes().emplace_back(std::make_unique<WebAudioWorkletRuntime>(std::move(runtime)));
    runtime            = {};
    EM_ASM(
        {
            const context = emscriptenGetAudioObject($0);
            const retired = $1;
            context.suspend().then(function() { return context.close(); }).then(function() { _gr_audio_release_runtime(retired); }).catch(function(error) {
                const cleanup = globalThis.__grAudioCleanup || (globalThis.__grAudioCleanup = {failures : 0, lastError : null});
                cleanup.failures += 1;
                cleanup.lastError = {name : error.name, message : error.message};
                console.error('[Audio] Context cleanup failed; retaining callback memory', error);
            });
        },
        context, reinterpret_cast<std::uintptr_t>(retired.get()));
    emscripten_destroy_audio_context(context);
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
    if (state->cancelRequested.load(std::memory_order_acquire)) {
        finishPendingInit(state, InitStatus::cancelled);
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
    nodeOptions.channelCountMode      = state->config.numberOfInputs > 0 ? WEBAUDIO_CHANNEL_COUNT_MODE_CLAMPED_MAX : WEBAUDIO_CHANNEL_COUNT_MODE_EXPLICIT;
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

    state->runtime.audioContext = EM_ASM_INT(
        {
            const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
            if (!Context) {
                return 0;
            }
            let context;
            try {
                context = new Context({latencyHint : 'interactive', sampleRate : $0 > 0 ? $0 : undefined});
            } catch (error) {
                if (error.name != 'NotSupportedError') {
                    console.error('[Audio] AudioContext creation failed', error);
                    return 0;
                }
                context = new Context({latencyHint : 'interactive'});
            }
            return emscriptenRegisterAudioObject(context);
        },
        state->config.requestedSampleRate);
    if (state->runtime.audioContext == 0) {
        finishPendingInit(state, InitStatus::failed, "WebAudio AudioContext creation failed");
        return;
    }
    state->runtime.sampleRate  = static_cast<std::uint32_t>(std::max(1, emscripten_audio_context_sample_rate(state->runtime.audioContext)));
    state->runtime.numChannels = static_cast<std::uint32_t>(std::max(1, state->config.outputChannelCount));

    if (state->config.numberOfInputs == 0) {
        state->config.outputChannelCount = EM_ASM_INT(
            {
                const context                     = emscriptenGetAudioObject($0);
                const destination                 = context.destination;
                const channels                    = Math.min($1, destination.maxChannelCount || destination.channelCount || 2);
                destination.channelInterpretation = 'discrete';
                destination.channelCount          = channels;
                return destination.channelCount;
            },
            state->runtime.audioContext, state->config.outputChannelCount);
        state->runtime.numChannels = static_cast<std::uint32_t>(state->config.outputChannelCount);
    }

    EM_ASM(
        {
            var ctx = emscriptenGetAudioObject($0);
            if (ctx && ctx.resume) {
                ctx.resume();
            }
        },
        state->runtime.audioContext);

    state->runtime.workletStack = memalign(16, kAudioWorkletStackSize);
    if (state->runtime.workletStack == nullptr) {
        finishPendingInit(state, InitStatus::failed, "WebAudio AudioWorklet stack allocation failed");
        return;
    }

    emscripten_start_wasm_audio_worklet_thread_async(state->runtime.audioContext, state->runtime.workletStack, static_cast<std::uint32_t>(kAudioWorkletStackSize), &workletThreadStarted, state);
}

[[nodiscard]] inline std::expected<WebAudioPendingWorkletInit, gr::Error> gr_webaudio_begin_create_worklet_node(const WebAudioWorkletNodeConfig& config, EmscriptenWorkletNodeProcessCallback processCallback, std::shared_ptr<void> callbackOwner) {
    auto* state                  = new PendingWorkletInitState{};
    state->config                = config;
    state->processCallback       = processCallback;
    state->userData              = callbackOwner.get();
    state->runtime.callbackOwner = std::move(callbackOwner);

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
        .sampleRate  = state->runtime.sampleRate,
        .numChannels = state->runtime.numChannels,
        .state       = state,
    };
}

inline void gr_webaudio_cancel_create_worklet_node(WebAudioPendingWorkletInit& pendingInit) {
    auto* state = pendingInit.state;
    if (state == nullptr) {
        return;
    }

    pendingInit = {};
    runOnMainThread(
        [](void* opaque) {
            auto* state = static_cast<PendingWorkletInitState*>(opaque);
            state->cancelRequested.store(true, std::memory_order_release);
            if (state->status.load(std::memory_order_acquire) == InitStatus::succeeded) {
                cleanupRuntime(state->runtime);
                state->status.store(InitStatus::cancelled, std::memory_order_release);
            }
        },
        state);
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

// clang-format off
inline void registerContextOnMainThread(void* opaqueTask) {
    auto* task = static_cast<MainThreadJsTask*>(opaqueTask);
    if (task == nullptr) {
        return;
    }

    EM_ASM({
        const opaque = $0;
        const contextHandle = $1;
        const context = contextHandle ? emscriptenGetAudioObject(contextHandle) : null;

        if (!globalThis.__grAudioWeb) {
            const audio = {};
            audio.devices = {};
            audio.unlock = function() {
                const devices = Object.values(audio.devices);
                for (let i = 0; i < devices.length; ++i) {
                    const state = devices[i];
                    if (state && !state.destroyed && state.context && state.context.resume && state.context.state === 'suspended') {
                        state.context.resume().catch(function(error) {
                            console.error('[Audio] Failed to resume WebAudio context', error);
                        });
                    }
                }
            };

            globalThis.__grAudioWeb = audio;
            document.addEventListener('touchend', audio.unlock, true);
            document.addEventListener('click', audio.unlock, true);
            document.addEventListener('keydown', audio.unlock, true);
        }

        const state = globalThis.__grAudioWeb.devices[opaque] || ({
            destroyed: false, failed: false, stream: null, streamNode: null,
            permission: 'pending', originalError: null, lastError: null,
            trackSampleRate: 0, trackChannelCount: 0, capture: false
        });
        if (context) {
            state.context = context;
        }
        globalThis.__grAudioWeb.devices[opaque] = state;
        if (state.context && state.context.resume) {
            state.context.resume().catch(function(error) {
                console.error('[Audio] Failed to resume WebAudio context', error);
            });
        }
    }, task->opaque, task->audioContext);
}

inline void captureBeginOnMainThread(void* opaqueTask) {
    auto* task = static_cast<MainThreadJsTask*>(opaqueTask);
    if (task == nullptr) {
        return;
    }

    task->result = EM_ASM_INT({
        const opaque = $0;
        const requestedSampleRate = $1;
        const channelCount = $2;
        const deviceIdPtr = $3;

        if (!globalThis.__grAudioWeb) {
            return 0;
        }

        const state = globalThis.__grAudioWeb.devices[opaque];
        if (!state) {
            return 0;
        }

        state.capture = true;
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            state.lastError = ({ name: 'NotFoundError', message: 'getUserMedia is unavailable', constraint: "" });
            state.failed = true;
            return 0;
        }

        const audioConstraint = ({ channelCount: { ideal: channelCount }, sampleRate: { ideal: requestedSampleRate } });
        audioConstraint.echoCancellation = false;
        audioConstraint.noiseSuppression = false;
        audioConstraint.autoGainControl = false;
        if (deviceIdPtr !== 0) {
            const deviceId = UTF8ToString(deviceIdPtr);
            if (deviceId.length > 0) {
                audioConstraint.deviceId = { exact: deviceId };
            }
        }
        state.failed = false;
        const recordError = function(error) {
            return ({ name: error.name || 'Error', message: error.message || String(error), constraint: error.constraint || "" });
        };
        const acquire = function(optional) {
            const constraints = Object.assign({}, audioConstraint);
            if (!optional) {
                delete constraints.channelCount;
                delete constraints.sampleRate;
            }
            return navigator.mediaDevices.getUserMedia({ audio: constraints, video: false });
        };
        acquire(true).catch(function(error) {
            state.originalError = recordError(error);
            if (state.destroyed || error.name !== 'OverconstrainedError') {
                throw error;
            }
            return acquire(false);
        })
            .then(function(stream) {
                if (state.destroyed || !globalThis.__grAudioWeb || globalThis.__grAudioWeb.devices[opaque] !== state) {
                    stream.getTracks().forEach(function(track) { track.stop(); });
                    return;
                }

                const track = stream.getAudioTracks()[0];
                if (!track) {
                    stream.getTracks().forEach(function(track) { track.stop(); });
                    throw ({ name: 'NotFoundError', message: 'capture has no audio track' });
                }
                const settings = track && track.getSettings ? track.getSettings() : {};
                state.trackSampleRate = settings.sampleRate || -1;
                state.trackChannelCount = settings.channelCount || -1;
                state.settings = settings;
                state.permission = 'granted';
                state.stream = stream;
            })
            .catch(function(error) {
                if (state.destroyed) {
                    return;
                }
                console.error('[AudioSource] Failed to get user media', error);
                state.lastError = recordError(error);
                state.permission = error.name === 'NotAllowedError' ? 'denied' : state.permission;
                state.failed = true;
            });

        return 1;
    }, task->opaque, task->sampleRate, task->channelCount, task->deviceId);
}

inline void captureAttachOnMainThread(void* opaqueTask) {
    auto* task = static_cast<MainThreadJsTask*>(opaqueTask);
    task->result = EM_ASM_INT({
        const state = globalThis.__grAudioWeb && globalThis.__grAudioWeb.devices[$0];
        const node = emscriptenGetAudioObject($1);
        if (!state || !state.stream || !state.context || !node) {
            return 0;
        }
        node.channelCount = Math.min($2, state.trackChannelCount > 0 ? state.trackChannelCount : $2);
        node.channelCountMode = 'clamped-max';
        node.channelInterpretation = 'discrete';
        state.streamNode = state.context.createMediaStreamSource(state.stream);
        state.streamNode.connect(node);
        state.attached = true;
        return 1;
    }, task->opaque, task->workletNode, task->channelCount);
}

inline void backendStateOnMainThread(void* opaqueTask) {
    auto* task = static_cast<MainThreadJsTask*>(opaqueTask);
    task->result = EM_ASM_INT({
        const state = globalThis.__grAudioWeb && globalThis.__grAudioWeb.devices[$0];
        if (!state || state.destroyed) { return 6; }
        if (state.failed) {
            const name = state.lastError && state.lastError.name;
            return name === 'NotAllowedError' ? 3 : name === 'NotFoundError' ? 4 : 5;
        }
        const track = state.stream && state.stream.getAudioTracks()[0];
        if (state.capture && (!track || !state.attached)) { return 0; }
        if (track && track.readyState !== 'live') { return 7; }
        if (track && track.muted) { return 8; }
        if (!state.context) { return 0; }
        if (state.context.state === 'closed') { return 6; }
        return state.context.state === 'running' ? 1 : 2;
    }, task->opaque);
}

inline void backendTextOnMainThread(void* opaqueTask) {
    auto* task = static_cast<MainThreadJsTask*>(opaqueTask);
    const auto readText = [&](char* buffer, std::size_t capacity) {
        return EM_ASM_INT({
        const state = globalThis.__grAudioWeb && globalThis.__grAudioWeb.devices[$0];
        const field = UTF8ToString($1);
        const parts = field.split('.');
        const nested = state && parts.length === 2 && state[parts[0]];
        const value = field === 'cleanup_failure_count' ? String(globalThis.__grAudioCleanup && globalThis.__grAudioCleanup.failures || 0) : !state ? "" : parts.length === 2 ? (nested && nested[parts[1]] || "") : field === 'last_error' ? JSON.stringify(state.lastError || {}) :
            field === 'original_error' ? JSON.stringify(state.originalError || {}) :
            field === 'context_state' ? (state.context && state.context.state || 'pending') :
            field === 'track_state' ? (state.stream && state.stream.getAudioTracks()[0].readyState || 'pending') : String(state[field] || "");
        if ($2) { stringToUTF8(value, $2, $3); }
        return lengthBytesUTF8(value);
        }, task->opaque, task->field, buffer, capacity);
    };
    const auto size = static_cast<std::size_t>(readText(nullptr, 0UZ));
    task->text.resize(size + 1UZ);
    readText(task->text.data(), task->text.size());
    task->text.resize(size);
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

inline void gr_webaudio_register_context(std::uintptr_t opaque, EMSCRIPTEN_WEBAUDIO_T audioContext) { std::ignore = runOnMainThread(&registerContextOnMainThread, {.opaque = opaque, .audioContext = audioContext}); }

// clang-format off
inline void gr_webaudio_resume_all_contexts() {
    runOnMainThread([](void*) {
        EM_ASM({
            if (globalThis.__grAudioWeb) {
                const devices = Object.values(globalThis.__grAudioWeb.devices);
                for (let i = 0; i < devices.length; ++i) {
                    const state = devices[i];
                    if (state && !state.destroyed && state.context && state.context.state !== 'running' && state.context.resume) {
                        state.context.resume().then(function() {
                            console.log('[Audio] AudioContext resumed via registry, state:', state.context.state);
                        }).catch(function(e) {
                            console.error('[Audio] resume failed:', e);
                        });
                    }
                }
            }
            if (typeof emscriptenGetAudioObject === 'function') {
                for (let handle = 1; handle < 100; ++handle) {
                    try {
                        var obj = emscriptenGetAudioObject(handle);
                        if (obj && obj.resume && obj.state !== 'running') {
                            obj.resume().then(function() {
                                console.log('[Audio] AudioContext handle', handle, 'resumed, state:', obj.state);
                            }).catch(function() {});
                        }
                    } catch(e) {}
                }
            }
        });
    }, nullptr);
}
// clang-format on

inline AudioBackendState gr_webaudio_backend_state(std::uintptr_t opaque) { return static_cast<AudioBackendState>(std::clamp(runOnMainThread(&backendStateOnMainThread, {.opaque = opaque}).result, 0, static_cast<int>(AudioBackendState::muted))); }

inline std::string gr_webaudio_backend_text(std::uintptr_t opaque, const char* field) { return runOnMainThread(&backendTextOnMainThread, {.opaque = opaque, .field = field}).text; }

[[nodiscard]] inline gr::property_map webAudioDiagnostics(std::uintptr_t opaque, std::size_t observedChannels) {
    gr::property_map result{{"state", std::string(gr::meta::enumName(gr_webaudio_backend_state(opaque)).value_or("unknown"))}, {"worklet_channels", static_cast<std::uint64_t>(observedChannels)}};
    result.insert_or_assign("cleanup_failure_count", static_cast<std::uint64_t>(std::strtoull(gr_webaudio_backend_text(opaque, "cleanup_failure_count").c_str(), nullptr, 10)));
    for (const char* field : {"permission", "context_state", "track_state"}) {
        result.insert_or_assign(field, gr_webaudio_backend_text(opaque, field));
    }
    for (const auto& [key, field] : {std::pair{"track_sample_rate", "trackSampleRate"}, std::pair{"track_channels", "trackChannelCount"}}) {
        const auto value = gr_webaudio_backend_text(opaque, field);
        if (!value.empty() && value != "-1") {
            result.insert_or_assign(key, static_cast<std::uint32_t>(std::strtoul(value.c_str(), nullptr, 10)));
        }
    }
    for (const auto& [key, field] : {std::pair{"original_error", "originalError"}, std::pair{"last_error", "lastError"}}) {
        gr::property_map error;
        for (const char* part : {"name", "message", "constraint"}) {
            const auto name = std::format("{}.{}", field, part);
            error.insert_or_assign(part, gr_webaudio_backend_text(opaque, name.c_str()));
        }
        result.insert_or_assign(key, std::move(error));
    }
    return result;
}

inline void gr_webaudio_unregister(std::uintptr_t opaque) { std::ignore = runOnMainThread(&unregisterOnMainThread, {.opaque = opaque}); }

[[nodiscard]] inline std::expected<bool, gr::Error> pollPendingWorkletInit(std::uintptr_t opaque, WebAudioPendingWorkletInit& pendingInit, WebAudioWorkletRuntime& runtime) {
    auto* state = pendingInit.state;
    if (state == nullptr) {
        return false;
    }
    const auto status = state->status.load(std::memory_order_acquire);
    if (status == InitStatus::pending) {
        return false;
    }
    pendingInit = {};
    const gr::Error error(status == InitStatus::failed ? state->errorMessage : std::string("WebAudio AudioWorklet initialisation was cancelled"));
    if (status == InitStatus::succeeded) {
        runtime = std::exchange(state->runtime, {});
    }
    releasePendingInit(state);
    if (status != InitStatus::succeeded) {
        return std::unexpected(error);
    }
    gr_webaudio_register_context(opaque, runtime.audioContext);
    return true;
}

template<typename TState>
struct WebAudioWorkletStream {
    std::shared_ptr<TState>               state{std::make_shared<TState>()};
    WebAudioPendingWorkletInit            pendingInit{};
    WebAudioWorkletRuntime                runtime{};
    AudioStreamFormat                     format{};
    std::atomic<AudioBackendState>        status{AudioBackendState::pending};
    std::chrono::steady_clock::time_point nextStateQuery{};

    void shutdown(std::uintptr_t opaque) {
        state->stopRequested.store(true, std::memory_order_release);
        gr_webaudio_cancel_create_worklet_node(pendingInit);
        gr_webaudio_unregister(opaque);
        gr_webaudio_destroy_worklet_runtime(runtime);
        state          = std::make_shared<TState>();
        format         = {};
        status         = AudioBackendState::stopped;
        nextStateQuery = {};
    }

    [[nodiscard]] bool isStateQueryDue() {
        constexpr std::chrono::milliseconds kStateQueryInterval{50};
        const auto                          now = std::chrono::steady_clock::now();
        if (now < nextStateQuery) {
            return false;
        }
        nextStateQuery = now + kStateQueryInterval;
        return true;
    }

    [[nodiscard]] static bool isTerminal(AudioBackendState backendState) {
        using enum AudioBackendState;
        return backendState == denied || backendState == unavailable || backendState == failed || backendState == stopped || backendState == ended;
    }
};

template<AudioSample T>
struct EmscriptenAudioWorkletSinkBackend {
    WebAudioWorkletStream<AudioSinkState<T>> _stream;

    [[nodiscard]] AudioSinkState<T>&       state() { return *_stream.state; }
    [[nodiscard]] const AudioSinkState<T>& state() const { return *_stream.state; }

    [[nodiscard]] std::expected<AudioStreamFormat, gr::Error> start(const AudioDeviceConfig& config) {
        shutdown();
        _stream.status   = AudioBackendState::pending;
        auto pendingInit = gr_webaudio_begin_create_worklet_node({.requestedSampleRate = static_cast<int>(config.sampleRate), .numberOfInputs = 0, .outputChannelCount = static_cast<int>(config.numChannels)}, &EmscriptenAudioWorkletSinkBackend::processAudio, _stream.state);
        if (!pendingInit) {
            shutdown();
            return std::unexpected(pendingInit.error());
        }
        AudioSinkState<T>& sinkState = *_stream.state;
        _stream.pendingInit          = *pendingInit;
        _stream.format               = {.sampleRate = pendingInit->sampleRate, .numChannels = pendingInit->numChannels};
        sinkState.intentionalSilence.store(config.intentionalSilence, std::memory_order_release);
        sinkState.channelPadding.store(config.channelPadding, std::memory_order_release);
        sinkState.recreateBuffer(AudioSinkState<T>::bufferCapacitySamples(_stream.format.numChannels, bufferFramesFor(config.bufferSeconds, _stream.format.sampleRate)));
        sinkState.numChannels = _stream.format.numChannels;
        sinkState.stopRequested.store(false, std::memory_order_release);
        return _stream.format;
    }

    void shutdown() { _stream.shutdown(reinterpret_cast<std::uintptr_t>(this)); }

    [[nodiscard]] std::expected<void, gr::Error> poll() {
        if (auto result = pollPendingWorkletInit(reinterpret_cast<std::uintptr_t>(this), _stream.pendingInit, _stream.runtime); !result) {
            return std::unexpected(result.error());
        }
        if (_stream.runtime.audioContext == 0) {
            _stream.status = AudioBackendState::pending;
        } else if (_stream.isStateQueryDue()) {
            _stream.status = gr_webaudio_backend_state(reinterpret_cast<std::uintptr_t>(this));
            if (_stream.isTerminal(_stream.status)) {
                return std::unexpected(gr::Error(std::format("WebAudio playback context is {}", gr::meta::enumName(_stream.status.load()).value_or("unknown"))));
            }
        }
        return {};
    }

    [[nodiscard]] AudioBackendState        backendState() const { return _stream.status; }
    [[nodiscard]] bool                     isStreamActive() const { return _stream.status == AudioBackendState::running; }
    [[nodiscard]] double                   softwareLatency() const { return 0.0; }
    [[nodiscard]] AudioStreamFormat        streamFormat() const { return _stream.format; }
    [[nodiscard]] std::vector<std::string> availableDevices() const { return {"default [default]"}; }
    [[nodiscard]] gr::property_map         diagnostics() const { return webAudioDiagnostics(reinterpret_cast<std::uintptr_t>(this), _stream.format.numChannels); }

private:
    static bool processAudio(int /*numInputs*/, const AudioSampleFrame* /*inputs*/, int numOutputs, AudioSampleFrame* outputs, int /*numParams*/, const AudioParamFrame* /*params*/, void* userData) {
        auto* self = static_cast<AudioSinkState<T>*>(userData);
        if (self->stopRequested.load(std::memory_order_acquire)) {
            return false;
        }
        if (numOutputs <= 0 || outputs == nullptr || outputs[0].data == nullptr || outputs[0].samplesPerChannel <= 0 || outputs[0].numberOfChannels <= 0) {
            return true;
        }
        const std::size_t frameCount      = static_cast<std::size_t>(outputs[0].samplesPerChannel);
        const std::size_t channelCount    = static_cast<std::size_t>(outputs[0].numberOfChannels);
        const std::size_t logicalChannels = self->numChannels.load(std::memory_order_acquire);
        if (logicalChannels == 0UZ) {
            std::fill_n(outputs[0].data, static_cast<std::ptrdiff_t>(frameCount * channelCount), 0.0f);
        } else {
            self->readPlanarFloat(outputs[0].data, frameCount, logicalChannels, channelCount);
        }
        return true;
    }
};

template<AudioSample T>
struct EmscriptenAudioWorkletSourceBackend {
    WebAudioWorkletStream<AudioSourceState<T>> _stream;
    float                                      _bufferSeconds{0.f};
    bool                                       _permissionGranted{false};

    [[nodiscard]] AudioSourceState<T>&       state() { return *_stream.state; }
    [[nodiscard]] const AudioSourceState<T>& state() const { return *_stream.state; }

    [[nodiscard]] std::expected<AudioStreamFormat, gr::Error> start(const AudioDeviceConfig& config) {
        shutdown();
        _bufferSeconds             = config.bufferSeconds;
        _permissionGranted         = false;
        _stream.status             = AudioBackendState::pending;
        _stream.state->numChannels = config.numChannels;
        _stream.state->channelPadding.store(config.channelPadding, std::memory_order_release);
        _stream.state->stopRequested.store(false, std::memory_order_release);
        const auto opaque = reinterpret_cast<std::uintptr_t>(this);
        gr_webaudio_register_context(opaque, 0);
        const std::string deviceId = config.device.starts_with("@id:") ? config.device.substr(4UZ) : isDefaultDevice(config.device) ? "" : config.device;
        if (runOnMainThread(&captureBeginOnMainThread, {.opaque = opaque, .channelCount = static_cast<int>(config.numChannels), .sampleRate = static_cast<int>(config.sampleRate), .deviceId = deviceId.empty() ? nullptr : deviceId.c_str()}).result == 0) {
            _stream.status = gr_webaudio_backend_state(opaque);
            return std::unexpected(gr::Error("WebAudio microphone acquisition could not start"));
        }
        return _stream.format;
    }

    void shutdown() { _stream.shutdown(reinterpret_cast<std::uintptr_t>(this)); }

    [[nodiscard]] std::expected<void, gr::Error> poll() {
        const auto        opaque       = reinterpret_cast<std::uintptr_t>(this);
        const std::size_t channelCount = _stream.state->numChannels;
        const bool        stateQueried = _stream.isStateQueryDue();
        if (stateQueried) {
            _stream.status = gr_webaudio_backend_state(opaque);
            if (_stream.isTerminal(_stream.status)) {
                return std::unexpected(gr::Error(gr_webaudio_backend_text(opaque, "last_error")));
            }
        }
        if (stateQueried && _stream.status == AudioBackendState::pending && _stream.pendingInit.state == nullptr && _stream.runtime.audioContext == 0) {
            const std::string trackRateText = gr_webaudio_backend_text(opaque, "trackSampleRate");
            if (trackRateText.empty()) {
                return {};
            }
            int trackRate      = 0;
            std::ignore        = std::from_chars(trackRateText.data(), trackRateText.data() + trackRateText.size(), trackRate);
            _permissionGranted = true;
            auto pending       = gr_webaudio_begin_create_worklet_node({.requestedSampleRate = std::max(0, trackRate), .numberOfInputs = 1, .outputChannelCount = static_cast<int>(channelCount)}, &EmscriptenAudioWorkletSourceBackend::processAudio, _stream.state);
            if (!pending) {
                return std::unexpected(pending.error());
            }
            _stream.pendingInit       = *pending;
            _stream.format.sampleRate = pending->sampleRate;
            _stream.state->recreateBuffer(AudioSourceState<T>::bufferCapacitySamples(channelCount, bufferFramesFor(_bufferSeconds, _stream.format.sampleRate)));
        }
        if (auto attached = pollPendingWorkletInit(opaque, _stream.pendingInit, _stream.runtime); !attached) {
            return std::unexpected(attached.error());
        } else if (*attached && runOnMainThread(&captureAttachOnMainThread, {.opaque = opaque, .workletNode = _stream.runtime.node, .channelCount = static_cast<int>(channelCount)}).result == 0) {
            gr_webaudio_unregister(opaque);
            gr_webaudio_destroy_worklet_runtime(_stream.runtime);
            return std::unexpected(gr::Error("WebAudio microphone initialisation failed"));
        }
        _stream.format.numChannels = static_cast<std::uint32_t>(_stream.state->observedChannels.load(std::memory_order_relaxed));
        return {};
    }

    [[nodiscard]] AudioBackendState        backendState() const { return _stream.status; }
    [[nodiscard]] bool                     isStreamActive() const { return _stream.status == AudioBackendState::running && _stream.state->observedChannels.load(std::memory_order_relaxed) != 0UZ; }
    [[nodiscard]] bool                     permissionGranted() const { return _permissionGranted; }
    [[nodiscard]] double                   softwareLatency() const { return 0.0; }
    [[nodiscard]] AudioStreamFormat        streamFormat() const { return _stream.format; }
    [[nodiscard]] std::vector<std::string> availableDevices() const { return {"default [default]"}; }
    [[nodiscard]] gr::property_map         diagnostics() const { return webAudioDiagnostics(reinterpret_cast<std::uintptr_t>(this), _stream.state->observedChannels.load(std::memory_order_relaxed)); }
    [[nodiscard]] std::size_t              readToOutput(std::span<T> output, std::size_t channelCount) { return _stream.state->readToOutput(output, channelCount); }

private:
    static bool processAudio(int numInputs, const AudioSampleFrame* inputs, int numOutputs, AudioSampleFrame* outputs, int /*numParams*/, const AudioParamFrame* /*params*/, void* userData) {
        auto* self = static_cast<AudioSourceState<T>*>(userData);
        if (self->stopRequested.load(std::memory_order_acquire)) {
            return false;
        }
        for (int outputIndex = 0; outputIndex < numOutputs; ++outputIndex) {
            if (outputs != nullptr && outputs[outputIndex].data != nullptr && outputs[outputIndex].samplesPerChannel > 0 && outputs[outputIndex].numberOfChannels > 0) {
                const std::size_t samples = static_cast<std::size_t>(outputs[outputIndex].samplesPerChannel) * static_cast<std::size_t>(outputs[outputIndex].numberOfChannels);
                std::fill_n(outputs[outputIndex].data, static_cast<std::ptrdiff_t>(samples), 0.0f);
            }
        }
        if (numInputs <= 0 || inputs == nullptr || inputs[0].data == nullptr || inputs[0].samplesPerChannel <= 0 || self->numChannels == 0UZ) {
            return true;
        }
        std::ignore = self->writePlanarFloat(inputs[0].data, static_cast<std::size_t>(inputs[0].samplesPerChannel), static_cast<std::size_t>(std::max(0, inputs[0].numberOfChannels)), self->numChannels);
        return true;
    }
};

struct WebAudioDevice : gr::blocks::common::DeviceBase {
    static constexpr std::string_view kId = "audio";

    bool _apiAvailable{true};

    [[nodiscard]] std::string_view id() const noexcept override { return kId; }
    [[nodiscard]] std::string_view displayName() const noexcept override { return "Microphone (WebAudio)"; }

    void init() override { _apiAvailable = checkMicrophonePermissionOnMainThread_impl() >= 0; }

    [[nodiscard]] bool isApiAvailable() const noexcept override { return _apiAvailable; }
    [[nodiscard]] int  grantedCount() const noexcept override { return getMicrophonePermissionState_impl() > 0 ? 1 : 0; }

    void requestPermission() override { requestMicrophonePermissionOnMainThread_impl(); }

    [[nodiscard]] std::expected<int, std::string> connect(int /*portIndex*/, int /*param*/) override { return 0; }
    void                                          disconnect(int /*handle*/) override {}

    [[nodiscard]] std::string lastError() const override { return ""; }
};

inline gr::blocks::common::AutoRegister autoRegWebAudio(std::make_shared<WebAudioDevice>());

#pragma clang diagnostic pop

} // namespace gr::audio::detail

#endif // __EMSCRIPTEN__

#endif // GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_BACKEND_HPP
