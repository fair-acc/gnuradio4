// GR4 GPS receiver example — works on both native (POSIX/Windows) and WASM (WebSerial).
//
// Native:  cmake --build build --target ex1_gps_receiver_gr4 && ./build/.../ex1_gps_receiver_gr4 [/dev/ttyACM0]
// WASM:    em++ build → serve → open in Chrome/Edge → click "Connect Devices"

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/GpsSource.hpp>
#include <gnuradio-4.0/common/DeviceRegistry.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <atomic>
#include <print>
#include <thread>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

using namespace gr;
using namespace gr::timing;

namespace {

std::atomic<bool> stopRequested{false};

void runGraph(std::string_view devicePath) {
    std::println("[GPS-GR4] starting graph, device: '{}'", devicePath.empty() ? "(auto-detect)" : devicePath);

    Graph graph;
    auto& gps  = graph.emplaceBlock<GpsSource>({{"device_path", std::string(devicePath)}});
    auto& sink = graph.emplaceBlock<testing::TagSink<std::uint8_t, testing::ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});
    auto  conn = graph.connect<"out", "in">(gps, sink);
    if (!conn.has_value()) {
        std::println(stderr, "[GPS-GR4] connection failed");
        return;
    }

    scheduler::Simple sched;
    if (!sched.exchange(std::move(graph)).has_value()) {
        std::println(stderr, "[GPS-GR4] scheduler init failed");
        return;
    }

    auto schedThread = std::thread([&sched] { sched.runAndWait(); });

    while (!stopRequested.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    sched.requestStop();
    schedThread.join();

    std::println("[GPS-GR4] stopped — samples: {}  tags: {}", sink._nSamplesProduced, sink._tags.size());
}

} // namespace

#if !defined(__EMSCRIPTEN__)

#include <csignal>

namespace {
void sigHandler(int) { stopRequested.store(true, std::memory_order_relaxed); }
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    gr::blocks::common::DeviceRegistry::instance().init();
    std::signal(SIGINT, sigHandler);
    std::signal(SIGTERM, sigHandler);

    std::string_view devicePath = (argc > 1) ? argv[1] : "";
    runGraph(devicePath);
    return 0;
}

#else // __EMSCRIPTEN__

namespace {
std::atomic<bool> graphRunning{false};
std::atomic<bool> graphDone{true};
} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE void gr_requestAllPermissions() { gr::blocks::common::DeviceRegistry::instance().requestAllPermissions(); }

EMSCRIPTEN_KEEPALIVE void startGpsGraph() {
    if (graphRunning.load(std::memory_order_relaxed)) {
        std::println("[GPS-GR4] graph already running");
        return;
    }
    graphDone.wait(false); // wait for previous run to finish
    stopRequested.store(false, std::memory_order_relaxed);
    graphDone.store(false, std::memory_order_relaxed);
    graphRunning.store(true, std::memory_order_relaxed);
    thread_pool::Manager::defaultIoPool()->execute([&] {
        std::println("[GPS-GR4] starting graph — IO thread will connect once serial port is granted.");
        runGraph(""); // GpsSource::ioReadLoop polls for device availability
        graphRunning.store(false, std::memory_order_relaxed);
        graphDone.store(true, std::memory_order_release);
        graphDone.notify_all();
    });
}

EMSCRIPTEN_KEEPALIVE void stopGpsGraph() {
    stopRequested.store(true, std::memory_order_relaxed);
    graphDone.wait(false);
}

} // extern "C"

int main() {
    gr::blocks::common::DeviceRegistry::instance().init();
    std::println("[GPS-GR4] WASM loaded. Click 'Connect Devices' to grant permissions.");
    startGpsGraph();
    return 0;
}

#endif // __EMSCRIPTEN__
