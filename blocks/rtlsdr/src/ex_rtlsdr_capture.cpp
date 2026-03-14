// RTL-SDR capture example — works on both native (Linux) and WASM (WebUSB).
//
// Native:  cmake --build build --target ex_rtlsdr_capture && ./build/.../ex_rtlsdr_capture
// WASM:    emcmake cmake ... → cmake --build → serve → open in Chrome/Edge → click "Connect RTL-SDR"

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/rtlsdr/RtlSdrSource.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <atomic>
#include <chrono>
#include <print>
#include <thread>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

using namespace gr;

namespace {

std::atomic<bool> stopRequested{false};

void runCapture() {
    constexpr double kSampleRate      = 2.048e6;
    constexpr double kCenterFrequency = 100.0e6;

    std::println("[RTL-SDR] starting capture: {:.3f} MHz @ {:.3f} MS/s", kCenterFrequency / 1e6, kSampleRate / 1e6);

    Graph graph;
    auto& src  = graph.emplaceBlock<blocks::rtlsdr::RtlSdrSource<std::complex<float>>>({
        {"center_frequency", kCenterFrequency},
        {"sample_rate", kSampleRate},
        {"auto_gain", true},
    });
    auto& sink = graph.emplaceBlock<testing::TagSink<std::complex<float>, testing::ProcessFunction::USE_PROCESS_BULK>>({
        {"log_samples", false},
        {"log_tags", false},
        {"verbose_console", true},
    });
    auto  conn = graph.connect<"out", "in">(src, sink);
    if (!conn.has_value()) {
        std::println(stderr, "[RTL-SDR] connection failed");
        return;
    }

    scheduler::Simple sched;
    if (!sched.exchange(std::move(graph)).has_value()) {
        std::println(stderr, "[RTL-SDR] scheduler init failed");
        return;
    }

    auto schedThread = std::thread([&sched] { sched.runAndWait(); });

    auto           lastReport      = std::chrono::steady_clock::now();
    std::size_t    lastSamples     = 0UZ;
    constexpr auto kReportInterval = std::chrono::seconds(2);

    while (!stopRequested.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        auto now = std::chrono::steady_clock::now();
        if (now - lastReport >= kReportInterval) {
            std::size_t total   = sink._nSamplesProduced;
            std::size_t delta   = total - lastSamples;
            double      elapsed = std::chrono::duration<double>(now - lastReport).count();
            double      rate    = elapsed > 0.0 ? static_cast<double>(delta) / elapsed : 0.0;

            std::println("[RTL-SDR] samples: {}  rate: {:.3f} MS/s  tags: {}", total, rate / 1e6, sink._tags.size());

            lastSamples = total;
            lastReport  = now;
        }
    }

    sched.requestStop();
    schedThread.join();
    std::println("[RTL-SDR] stopped — total samples: {}  tags: {}", sink._nSamplesProduced, sink._tags.size());
}

} // namespace

#if !defined(__EMSCRIPTEN__)

#include <csignal>

namespace {
void sigHandler(int) { stopRequested.store(true, std::memory_order_relaxed); }
} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::signal(SIGINT, sigHandler);
    std::signal(SIGTERM, sigHandler);
    runCapture();
    return 0;
}

#else // __EMSCRIPTEN__

namespace {
std::atomic<bool> graphRunning{false};
} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE void rtlsdrRequestPermission() { gr::blocks::rtlsdr::js_rtl_request_device(); }

EMSCRIPTEN_KEEPALIVE int rtlsdrConnect() {
    // runs on main thread — Asyncify works here
    static gr::blocks::rtlsdr::RtlSdrDevice mainDevice;
    int                                     ret = gr::blocks::rtlsdr::js_rtl_open_device(0);
    if (ret < 0) {
        std::println(stderr, "[RTL-SDR] WebUSB open failed");
        return -1;
    }
    // init demod + tuner via the C++ register protocol (EM_ASYNC_JS control transfers)
    mainDevice._open.store(true, std::memory_order_release);
    auto initResult = mainDevice.initDevice();
    if (!initResult) {
        std::println(stderr, "[RTL-SDR] init failed: {}", initResult.error());
        return -1;
    }
    mainDevice.setSampleRate(2.048e6);
    mainDevice.setCenterFrequency(100.0e6);
    mainDevice.setGainMode(true);
    mainDevice.setAgcMode(true);
    mainDevice.resetBuffer();
    mainDevice.startBulkRead();
    // signal IO thread that device is ready
    gr::blocks::rtlsdr::detail::wasmDeviceReady().store(true, std::memory_order_release);
    std::println("[RTL-SDR] device ready — IO thread can start reading");
    return 0;
}

EMSCRIPTEN_KEEPALIVE void startRtlSdrCapture() {
    if (graphRunning.exchange(true, std::memory_order_acq_rel)) {
        std::println("[RTL-SDR] capture already running");
        return;
    }
    stopRequested.store(false, std::memory_order_relaxed);
    thread_pool::Manager::defaultIoPool()->execute([] {
        runCapture();
        graphRunning.store(false, std::memory_order_release);
        std::println("[RTL-SDR] capture stopped");
    });
}

EMSCRIPTEN_KEEPALIVE void stopRtlSdrCapture() { stopRequested.store(true, std::memory_order_relaxed); }

} // extern "C"

int main() {
    std::println("[RTL-SDR] WASM loaded. Click 'Connect RTL-SDR' to grant USB permission.");
    return 0;
}

#endif // __EMSCRIPTEN__
