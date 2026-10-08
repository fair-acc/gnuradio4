#include <gnuradio-4.0/audio/EmscriptenAudioBackend.hpp>

#include <array>
#include <format>
#include <string>

gr::audio::detail::EmscriptenAudioWorkletSourceBackend<float> capture;
std::uint64_t                                                 capturedFrames{0U};
double                                                        capturedEnergy{0.0};
std::size_t                                                   paddingErrors{0UZ};
gr::audio::ChannelPadding                                     padding{gr::audio::ChannelPadding::zero};
std::string                                                   diagnosticJson;

EM_JS(void, suspendProbeContext, (int context), { emscriptenGetAudioObject(context).suspend(); });

extern "C" {
EMSCRIPTEN_KEEPALIVE int probe_start(int cyclic) {
    capturedFrames = 0U;
    capturedEnergy = 0.0;
    paddingErrors  = 0UZ;
    padding        = cyclic != 0 ? gr::audio::ChannelPadding::cyclic : gr::audio::ChannelPadding::zero;
    return capture.start({.sampleRate = 96000U, .numChannels = 2U, .bufferSeconds = 0.1f, .channelPadding = padding}).has_value() ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int probe_poll() {
    const auto                result = capture.poll();
    std::array<float, 4096UZ> samples{};
    const std::size_t         count = capture.readToOutput(samples, 2UZ);
    capturedFrames += count / 2UZ;
    for (std::size_t index = 0UZ; index < count; index += 2UZ) {
        capturedEnergy += static_cast<double>(samples[index]) * static_cast<double>(samples[index]);
        paddingErrors += samples[index + 1UZ] != (padding == gr::audio::ChannelPadding::cyclic ? samples[index] : 0.f) ? 1UZ : 0UZ;
    }
    return result.has_value() ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE const char* probe_diagnostics() {
    const auto diagnostics = capture.diagnostics();
    diagnosticJson         = std::format(R"({{"state":{:?},"context_state":{:?},"track_state":{:?},"req_sample_rate":96000,"sample_rate":{},"track_sample_rate":{},"track_channels":{},"worklet_channels":{},"frames":{},"energy":{},"padding_errors":{},"cleanup_failure_count":{},"original_error":{:?},"last_error":{:?}}})", diagnostics.value_or<std::string>("state", "unknown"), diagnostics.value_or<std::string>("context_state", "unknown"), diagnostics.value_or<std::string>("track_state", "unknown"), capture.streamFormat().sampleRate, diagnostics.value_or<std::uint32_t>("track_sample_rate", 0U), diagnostics.value_or<std::uint32_t>("track_channels", 0U), capture.streamFormat().numChannels, capturedFrames, capturedEnergy, paddingErrors, diagnostics.value_or<std::uint64_t>("cleanup_failure_count", 0U), diagnostics.value_or<gr::property_map>("original_error", {}).value_or<std::string>("name", ""), diagnostics.value_or<gr::property_map>("last_error", {}).value_or<std::string>("name", ""));
    return diagnosticJson.c_str();
}

EMSCRIPTEN_KEEPALIVE void probe_suspend() { suspendProbeContext(capture._stream.runtime.audioContext); }
EMSCRIPTEN_KEEPALIVE void probe_resume() { emscripten_resume_audio_context_sync(capture._stream.runtime.audioContext); }
EMSCRIPTEN_KEEPALIVE void probe_stop() { capture.shutdown(); }
}

int main() { return 0; }
