#include <emscripten/em_asm.h>
#include <emscripten/emscripten.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIoEmscriptenHelper.hpp>
#include <gnuradio-4.0/audio/AudioBlocks.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <print>
#include <string>
#include <string_view>

namespace fileio = gr::algorithm::fileio;

namespace audio_test_app_detail {

// clang-format off
// Append mic chart fragments on the main thread because stdout logging is printed line by line in the browser.
inline void appendLogFragment(const char* text) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdollar-in-identifier-extension"
    MAIN_THREAD_ASYNC_EM_ASM(
        {
            if (typeof window.appendAudioTestLogFragment === 'function') {
                window.appendAudioTestLogFragment(UTF8ToString($0));
            }
        },
        text);
#pragma GCC diagnostic pop
}
// clang-format on

struct LevelMonitor : gr::Block<LevelMonitor> {
    using Clock = std::chrono::steady_clock;

    gr::PortIn<float> in;

    GR_MAKE_REFLECTABLE(LevelMonitor, in);

    std::chrono::milliseconds printPeriod{20};
    Clock::time_point         lastPrintTime = Clock::now();
    std::string               linePrefix{"[AudioTest] mic "};
    float                     absSum       = 0.0f;
    std::size_t               sampleCount  = 0U;
    std::size_t               linePosition = 0U;

    void start() {
        lastPrintTime = Clock::now();
        absSum        = 0.0f;
        sampleCount   = 0U;
        linePosition  = 0U;
        appendLogFragment("\n");
    }

    void stop() {
        if (linePosition != 0U) {
            appendLogFragment("\n");
            linePosition = 0U;
        }
    }

    gr::work::Status processBulk(auto& input) {
        const std::size_t n = input.size();
        if (n == 0) {
            std::ignore = input.consume(0U);
            return gr::work::Status::OK;
        }

        for (std::size_t i = 0; i < n; i++) {
            absSum += std::abs(input[i]);
        }
        sampleCount += n;

        const auto now = Clock::now();
        if (now - lastPrintTime >= printPeriod && sampleCount > 0U) {
            const float meanAmplitude = absSum / static_cast<float>(sampleCount);
            if (linePosition == 0U) {
                appendLogFragment(linePrefix.c_str());
            }
            appendLogFragment(levelSymbol(meanAmplitude).data());
            ++linePosition;
            if (linePosition >= 100U) {
                appendLogFragment("\n");
                linePosition = 0U;
            }
            lastPrintTime = now;
            absSum        = 0.0f;
            sampleCount   = 0U;
        }

        std::ignore = input.consume(n);
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] static std::string_view levelSymbol(float meanAmplitude) {
        static constexpr std::array<std::string_view, 8> symbols  = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
        static constexpr float                           minLevel = 0.0f;
        static constexpr float                           maxLevel = 0.20f;

        const float clamped = meanAmplitude < minLevel ? minLevel : (meanAmplitude > maxLevel ? maxLevel : meanAmplitude);
        const float level   = (clamped - minLevel) / (maxLevel - minLevel);
        auto        index   = static_cast<std::size_t>(level * static_cast<float>(symbols.size() - 1U) + 0.5f);
        if (index >= symbols.size()) {
            index = symbols.size() - 1U;
        }
        return symbols[index];
    }
};

} // namespace audio_test_app_detail

namespace {
using Scheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;

std::atomic<bool>          playbackRunning{false};
std::atomic<bool>          micRunning{false};
std::shared_ptr<Scheduler> micScheduler;

void runGraph(Scheduler& scheduler, gr::Graph&& graph) {
    if (auto exchangeResult = scheduler.exchange(std::move(graph)); !exchangeResult.has_value()) {
        std::println("[AudioTest] graph init failed: {}", exchangeResult.error().message);
        return;
    }

    std::println("[AudioTest] graph started");
    if (auto runResult = scheduler.runAndWait(); !runResult.has_value()) {
        std::println("[AudioTest] graph run failed: {}", runResult.error().message);
    }
}

void runPlaybackGraph(std::string uri) {
    try {
        std::println("[AudioTest] play request: {}", uri);
        std::println("[AudioTest] main runtime thread: {}", fileio::isMainThread());

        gr::Graph graph;
        auto&     source  = graph.emplaceBlock<gr::audio::WavSource<float>>({{"uri", std::move(uri)}});
        auto&     monitor = graph.emplaceBlock<audio_test_app_detail::LevelMonitor>();
        auto&     sink    = graph.emplaceBlock<gr::audio::AudioSink<float>>();

        monitor.linePrefix = "[AudioTest] play ";
        graph.connect<"out", "in">(source, monitor).value();
        graph.connect<"out", "in">(source, sink).value();
        Scheduler scheduler;
        runGraph(scheduler, std::move(graph));
    } catch (const std::exception& ex) {
        std::println("[AudioTest] worker exception: {}", ex.what());
    }

    playbackRunning.store(false, std::memory_order_release);
    std::println("[AudioTest] playback worker finished");
}

void runMicGraph(std::shared_ptr<Scheduler> scheduler) {
    try {
        std::println("[AudioTest] starting microphone loopback");
        std::println("[AudioTest] main runtime thread: {}", fileio::isMainThread());

        gr::Graph graph;
        auto&     source  = graph.emplaceBlock<gr::audio::AudioSource<float>>();
        auto&     monitor = graph.emplaceBlock<audio_test_app_detail::LevelMonitor>();
        auto&     sink    = graph.emplaceBlock<gr::audio::AudioSink<float>>();

        monitor.linePrefix = "[AudioTest] mic ";
        graph.connect<"out", "in">(source, monitor).value();
        graph.connect<"out", "in">(source, sink).value();
        runGraph(*scheduler, std::move(graph));
    } catch (const std::exception& ex) {
        std::println("[AudioTest] mic worker exception: {}", ex.what());
    }

    micRunning.store(false, std::memory_order_release);
    std::println("[AudioTest] microphone loopback worker finished");
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

        gr::thread_pool::Manager::defaultIoPool()->execute([uriValue]() mutable { runPlaybackGraph(std::move(uriValue)); });
        return 1;
    } catch (const std::exception& ex) {
        std::println("[AudioTest] exception: {}", ex.what());
        playbackRunning.store(false, std::memory_order_release);
        return 0;
    }
}

EMSCRIPTEN_KEEPALIVE int audio_playback_is_running() { return playbackRunning.load(std::memory_order_acquire) ? 1 : 0; }

EMSCRIPTEN_KEEPALIVE int start_mic() {
    bool expected = false;
    if (!micRunning.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        std::println("[AudioTest] microphone loopback already in progress");
        return 0;
    }

    try {
        auto scheduler = std::make_shared<Scheduler>();
        micScheduler   = scheduler;
        gr::thread_pool::Manager::defaultIoPool()->execute([scheduler] { runMicGraph(scheduler); });
        return 1;
    } catch (const std::exception& ex) {
        std::println("[AudioTest] microphone loopback exception: {}", ex.what());
        micScheduler.reset();
        micRunning.store(false, std::memory_order_release);
        return 0;
    }
}

EMSCRIPTEN_KEEPALIVE void stop_mic() {
    if (micScheduler) {
        std::println("[AudioTest] stopping microphone loopback");
        micScheduler->requestStop();
    }
}

EMSCRIPTEN_KEEPALIVE int mic_is_running() {
    const bool running = micRunning.load(std::memory_order_acquire);
    if (!running) {
        micScheduler.reset();
    }
    return running ? 1 : 0;
}

} // extern "C"

int main() {
    std::println("[AudioTest] WASM ready. Click one of the audio buttons to start playback or microphone loopback.");
    return 0;
}
