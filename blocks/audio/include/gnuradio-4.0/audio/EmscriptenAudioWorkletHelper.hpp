#ifndef GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_WORKLET_HELPER_HPP
#define GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_WORKLET_HELPER_HPP

#if !defined(__EMSCRIPTEN__)
#error "EmscriptenAudioWorkletHelper.hpp is only available on Emscripten"
#endif

#include <gnuradio-4.0/Message.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <string>
#include <string_view>

#include <emscripten.h>
#include <emscripten/threading.h>
#include <emscripten/webaudio.h>
#include <malloc.h>

namespace gr::audio::detail {

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
};

inline void runOnMainThread(void (*fn)(void*), void* opaque) {
    if (emscripten_is_main_runtime_thread()) {
        fn(opaque);
    } else {
        emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VI, fn, opaque);
    }
}

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
                auto* state = static_cast<PendingWorkletInitState*>(opaque);
                if (state != nullptr) {
                    cleanupRuntime(state->runtime);
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
            auto* runtime = static_cast<WebAudioWorkletRuntime*>(opaque);
            if (runtime == nullptr) {
                return;
            }
            cleanupRuntime(*runtime);
        },
        &runtime);
}

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

        state.failed = false;
        navigator.mediaDevices.getUserMedia({ audio: { channelCount: channelCount }, video: false })
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
    }, task->opaque, task->workletNode, task->channelCount);
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
// clang-format on

inline void gr_webaudio_register_context(std::uintptr_t opaque, EMSCRIPTEN_WEBAUDIO_T audioContext) {
    MainThreadJsTask task{.opaque = opaque, .audioContext = audioContext};
    runOnMainThread(&registerContextOnMainThread, &task);
}

inline int gr_webaudio_capture_attach(std::uintptr_t opaque, EMSCRIPTEN_AUDIO_WORKLET_NODE_T workletNode, int channelCount) {
    MainThreadJsTask task{
        .opaque       = opaque,
        .workletNode  = workletNode,
        .channelCount = channelCount,
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

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

} // namespace gr::audio::detail

#endif // GNURADIO_AUDIO_EMSCRIPTEN_AUDIO_WORKLET_HELPER_HPP
