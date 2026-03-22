#include <emscripten/emscripten.h>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIoEmscriptenHelper.hpp>
#include <gnuradio-4.0/audio/AudioBlocks.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <atomic>
#include <exception>
#include <print>
#include <string>

namespace fileio = gr::algorithm::fileio;

namespace {
std::atomic<bool> playbackRunning{false};
std::atomic<bool> playbackDone{true};

void runPlaybackGraph(std::string uri) {
    try {
        std::println("[AudioTest] play request: {}", uri);
        std::println("[AudioTest] main runtime thread: {}", fileio::isMainThread());

        gr::Graph graph;
        auto&     sink   = graph.emplaceBlock<gr::audio::AudioSink<float>>();
        auto&     source = graph.emplaceBlock<gr::audio::WavSource<float>>({{"uri", std::move(uri)}});

        if (!graph.connect<"out", "in">(source, sink).has_value()) {
            std::println("[AudioTest] failed to connect WavSource to AudioSink");
        } else {
            gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> scheduler;
            if (auto exchangeResult = scheduler.exchange(std::move(graph)); !exchangeResult.has_value()) {
                std::println("[AudioTest] scheduler init failed: {}", exchangeResult.error().message);
            } else {
                std::println("[AudioTest] graph started; audio is playing");
                if (auto runResult = scheduler.runAndWait(); !runResult.has_value()) {
                    std::println("[AudioTest] scheduler run failed: {}", runResult.error().message);
                }
            }
        }
    } catch (const std::exception& ex) {
        std::println("[AudioTest] worker exception: {}", ex.what());
    }

    playbackDone.store(true, std::memory_order_release);
    playbackRunning.store(false, std::memory_order_release);
    std::println("[AudioTest] playback worker finished");
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int play_audio_from_uri(const char* uri) {
    try {
        const std::string uriValue = uri != nullptr ? uri : "";
        if (uriValue.empty()) {
            std::println("[AudioTest] empty URI");
            return 0;
        }

        bool expected = false;
        if (!playbackRunning.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            std::println("[AudioTest] playback already in progress");
            return 0;
        }

        playbackDone.store(false, std::memory_order_release);
        gr::thread_pool::Manager::defaultIoPool()->execute([uriValue]() mutable { runPlaybackGraph(std::move(uriValue)); });
        return 1;
    } catch (const std::exception& ex) {
        std::println("[AudioTest] exception: {}", ex.what());
        playbackDone.store(true, std::memory_order_release);
        playbackRunning.store(false, std::memory_order_release);
        return 0;
    }
}

EMSCRIPTEN_KEEPALIVE int audio_playback_is_done() { return playbackDone.load(std::memory_order_acquire) ? 1 : 0; }

EMSCRIPTEN_KEEPALIVE void finish_audio_playback() { std::println("[AudioTest] playback cleanup finished"); }

} // extern "C"

int main() {
    std::println("[AudioTest] WASM ready. Click one of the Play buttons to start audio.");
    return 0;
}
