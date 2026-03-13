// GPS/GNSS serial reader — standalone example using NMEADevice + NMEAParser.
// Tests the serial device abstraction and NMEA parsing without the GR4 block/graph framework.
//
// Native:  ./ex0_gps_receiver [/dev/ttyACM0]
// WASM:    serve with COOP/COEP headers → open in Chrome/Edge → click "Connect Devices"

#include <gnuradio-4.0/NMEADevice.hpp>
#include <gnuradio-4.0/NMEAParser.hpp>

#include <atomic>
#include <chrono>
#include <print>
#include <thread>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <gnuradio-4.0/common/DeviceRegistry.hpp>
#endif

using namespace gr::timing;

namespace {

std::atomic<bool> stopRequested{false};

constexpr std::string_view fixTypeName(FixType ft) {
    switch (ft) {
    case FixType::none: return "none";
    case FixType::fix2D: return "2D";
    case FixType::fix3D: return "3D";
    }
    return "?";
}

std::string formatUtcNs(std::uint64_t utcNs) {
    if (utcNs == 0) {
        return "----------T--:--:--.---Z";
    }
    using namespace std::chrono;
    auto           tp = sys_time<nanoseconds>(nanoseconds(static_cast<std::int64_t>(utcNs)));
    auto           dp = floor<days>(tp);
    year_month_day ymd{dp};
    hh_mm_ss       tod{floor<milliseconds>(tp - dp)};
    return std::format("{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:03d}Z", static_cast<int>(ymd.year()), static_cast<unsigned>(ymd.month()), static_cast<unsigned>(ymd.day()), tod.hours().count(), tod.minutes().count(), tod.seconds().count(), tod.subseconds().count());
}

void printFix(const GpsFix& fix) { std::println("{} | {:>11.7f}°{} {:>12.7f}°{} | alt:{:7.1f}m | sat:{:2d} | hdop:{:.1f} | spd:{:6.1f}km/h | hdg:{:5.1f}° | {}", formatUtcNs(fix.utcTimestampNs), std::abs(fix.latitude), (fix.latitude >= 0.f ? 'N' : 'S'), std::abs(fix.longitude), (fix.longitude >= 0.f ? 'E' : 'W'), fix.altitude, fix.satellites, fix.hdop, fix.speedKmh, fix.headingDeg, fixTypeName(fix.fixType)); }

void readLoop(SerialPort& port) {
    NMEAParser              parser;
    std::string             buffer;
    std::array<char, 512UZ> temp{};

    std::println("[GPS] reading NMEA data...");
    while (!stopRequested.load(std::memory_order_relaxed)) {
        auto n = port.read(std::span(temp), std::chrono::milliseconds{1000});
        if (n > 0) {
            auto localTime = detail::wallClockNs();
            buffer.append(temp.data(), n);

            std::size_t pos;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                auto lineView = std::string_view(buffer).substr(0, pos);
                if (!lineView.empty() && lineView.back() == '\r') {
                    lineView.remove_suffix(1);
                }
                if (auto fix = parser.parseLine(lineView, localTime)) {
                    printFix(*fix);
                }
                buffer.erase(0, pos + 1);
            }
        }
    }
    port.close();
    std::println("[GPS] stopped");
}

#if !defined(__EMSCRIPTEN__)
bool tryOpenAndRun(std::string_view devicePath) {
    auto deviceResult = selectNMEADevice(devicePath);
    if (!deviceResult) {
        std::println(stderr, "[ERROR] {}", deviceResult.error());
        return false;
    }
    auto& device = *deviceResult;
    std::println("[GPS] device: {} (VID:{:04x} PID:{:04x} {} - {})", device.devicePath, device.vendorId, device.productId, device.vendor, device.model);

    auto portResult = SerialPort::open(device.devicePath, BaudRate::Baud9600);
    if (!portResult) {
        std::println(stderr, "[ERROR] {}", portResult.error());
        return false;
    }
    readLoop(*portResult);
    return true;
}
#endif

} // namespace

#if !defined(__EMSCRIPTEN__)

#include <csignal>

namespace {
void sigHandler(int) { stopRequested.store(true, std::memory_order_relaxed); }
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::signal(SIGINT, sigHandler);
    std::signal(SIGTERM, sigHandler);

    std::string_view devicePath = (argc > 1) ? argv[1] : "";
    return tryOpenAndRun(devicePath) ? 0 : 1;
}

#else // __EMSCRIPTEN__

namespace {
std::atomic<bool> running{false};
std::atomic<bool> done{true};
} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE void gr_requestAllPermissions() { gr::blocks::common::DeviceRegistry::instance().requestAllPermissions(); }

EMSCRIPTEN_KEEPALIVE void startGpsGraph() { // named for shared shell HTML compatibility
    if (running.load(std::memory_order_relaxed)) {
        std::println("[GPS] already running");
        return;
    }
    done.wait(false);
    stopRequested.store(false, std::memory_order_relaxed);
    done.store(false, std::memory_order_relaxed);
    running.store(true, std::memory_order_relaxed);

    std::thread([] {
        std::println("[GPS] waiting for serial port...");

        SerialPort port;
        while (!stopRequested.load(std::memory_order_relaxed) && !port.isOpen()) {
            auto deviceResult = autoDetectNMEADevice();
            if (deviceResult) {
                std::println("[GPS] found: {}", deviceResult->devicePath);
                auto portResult = SerialPort::open(deviceResult->devicePath, BaudRate::Baud9600);
                if (portResult) {
                    port = std::move(*portResult);
                }
            }
            if (!port.isOpen()) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        if (port.isOpen()) {
            readLoop(port);
        }

        running.store(false, std::memory_order_relaxed);
        done.store(true, std::memory_order_release);
        done.notify_all();
    }).detach();
}

EMSCRIPTEN_KEEPALIVE void stopGpsGraph() { // named for shared shell HTML compatibility
    stopRequested.store(true, std::memory_order_relaxed);
    done.wait(false);
}

} // extern "C"

int main() {
    gr::blocks::common::DeviceRegistry::instance().init();
    std::println("[GPS] WASM loaded. Click 'Connect Devices' to start.");
    startGpsGraph();
    return 0;
}

#endif // __EMSCRIPTEN__
