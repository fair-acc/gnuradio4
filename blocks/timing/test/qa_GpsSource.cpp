#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/GpsSource.hpp>
#include <gnuradio-4.0/formatter/ValueFormatter.hpp>

#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)
#if defined(__APPLE__)
#include <util.h> // openpty() on macOS
#else
#include <pty.h>
#endif
#include <thread>
#include <unistd.h>
#endif

using namespace boost::ut;
using namespace gr;
using namespace gr::timing;
using namespace gr::testing;

namespace {

std::string nmea(std::string_view body) {
    auto cs = gr::timing::detail::computeNMEAChecksum(body.substr(1)); // skip '$'
    return std::format("{}*{:02X}", body, cs);
}

} // namespace

#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)

namespace {

struct PtyPair {
    int         master = -1;
    int         slave  = -1;
    std::string slaveName;

    PtyPair() = default;
    ~PtyPair() {
        if (master >= 0) {
            ::close(master);
        }
        if (slave >= 0) {
            ::close(slave);
        }
    }

    PtyPair(const PtyPair&)            = delete;
    PtyPair& operator=(const PtyPair&) = delete;
    PtyPair(PtyPair&& o) noexcept : master(std::exchange(o.master, -1)), slave(std::exchange(o.slave, -1)), slaveName(std::move(o.slaveName)) {}
    PtyPair& operator=(PtyPair&& o) noexcept {
        std::swap(master, o.master);
        std::swap(slave, o.slave);
        slaveName = std::move(o.slaveName);
        return *this;
    }

    static std::optional<PtyPair> create() {
        PtyPair p;
        if (::openpty(&p.master, &p.slave, nullptr, nullptr, nullptr) != 0) {
            return std::nullopt;
        }
        p.slaveName = ::ttyname(p.slave);
        return p;
    }

    void writeLine(std::string_view line) const {
        std::string           msg = std::string(line) + "\r\n";
        [[maybe_unused]] auto n   = ::write(master, msg.data(), msg.size());
    }
};

void sendNMEASequence(const PtyPair& pty, int startSecond, int count, int delayMs = 50) {
    for (int i = 0; i < count; ++i) {
        int sec = (startSecond + i) % 60;
        pty.writeLine(nmea(std::format("$GPRMC,1200{:02d}.00,A,5001.1900,N,00840.6570,E,0.5,45.0,110326,,,A", sec)));
        pty.writeLine(nmea(std::format("$GPGGA,1200{:02d}.00,5001.1900,N,00840.6570,E,1,10,0.8,136.0,M,47.0,M,,", sec)));
        pty.writeLine(nmea("$GPGSA,A,3,04,05,09,12,,,,,,,,,1.8,1.0,1.5"));
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    }
}

} // namespace

#endif // POSIX

const boost::ut::suite<"GpsSource"> gpsSourceTests = [] {
#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)
    "GpsSource emits PPS tags via PTY"_test = [] {
        auto pty = PtyPair::create();
        expect(pty.has_value()) << "PTY creation must succeed";
        if (!pty)
            return;

        Graph testGraph;
        auto& gps  = testGraph.emplaceBlock<GpsSource>({{"device_path", std::string(pty->slaveName)}});
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});
        expect(testGraph.connect<"out", "in">(gps, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // send 4 NMEA fixes → 3 PPS boundaries (sec 0→1, 1→2, 2→3)
        sendNMEASequence(*pty, 0, 4, 80);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        sched.requestStop();
        schedThread.join();

        expect(ge(sink._nSamplesProduced, 2UZ)) << "at least 2 PPS samples emitted";

        bool foundTriggerTag = false;
        for (const auto& tag : sink._tags) {
            if (auto it = tag.map.find(std::pmr::string("trigger_name")); it != tag.map.end()) {
                foundTriggerTag = true;
            }
            if (auto it = tag.map.find(std::pmr::string("trigger_time")); it != tag.map.end()) {
                auto time = it->second.value_or(std::uint64_t{0});
                expect(gt(time, 0ULL)) << "trigger_time > 0";
            }
            if (auto it = tag.map.find(std::pmr::string("trigger_meta_info")); it != tag.map.end()) {
                auto* meta = it->second.get_if<property_map>();
                expect(meta != nullptr) << "trigger_meta_info is a map";
                if (meta) {
                    expect(meta->contains(std::pmr::string("geolocation"))) << "geolocation present";
                    expect(meta->contains(std::pmr::string("local_time"))) << "local_time present";
                    expect(meta->contains(std::pmr::string("fix_type"))) << "fix_type present";
                    expect(meta->contains(std::pmr::string("device_info"))) << "device_info present";
                }
            }
        }
        expect(foundTriggerTag) << "at least one trigger tag found";
    };

    "GpsSource emit_meta_info=false omits trigger_meta_info"_test = [] {
        auto pty = PtyPair::create();
        expect(pty.has_value()) << "PTY creation must succeed";
        if (!pty)
            return;

        Graph testGraph;
        auto& gps  = testGraph.emplaceBlock<GpsSource>({{"device_path", std::string(pty->slaveName)}, {"emit_meta_info", false}});
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});
        expect(testGraph.connect<"out", "in">(gps, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        sendNMEASequence(*pty, 0, 3, 80);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        sched.requestStop();
        schedThread.join();

        expect(ge(sink._nSamplesProduced, 1UZ)) << "at least 1 PPS sample";

        for (const auto& tag : sink._tags) {
            expect(!tag.map.contains(std::pmr::string("trigger_meta_info"))) << "no meta_info when emit_meta_info=false";
            if (auto it = tag.map.find(std::pmr::string("trigger_name")); it != tag.map.end()) {
                expect(true) << "trigger_name still present";
            }
        }
    };

    "GpsSource clock mode emits continuous samples with PPS tags"_test = [] {
        auto pty = PtyPair::create();
        expect(pty.has_value()) << "PTY creation must succeed";
        if (!pty)
            return;

        Graph testGraph;
        auto& gps  = testGraph.emplaceBlock<GpsSource>({{"device_path", std::string(pty->slaveName)}, {"emit_mode", std::string("clock")}, {"sample_rate", 100.f}});
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});
        expect(testGraph.connect<"out", "in">(gps, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        sendNMEASequence(*pty, 0, 4, 80);
        std::this_thread::sleep_for(std::chrono::milliseconds(800));

        sched.requestStop();
        schedThread.join();

        // at 100 Hz for ~1s, expect many more samples than PPS events
        expect(ge(sink._nSamplesProduced, 50UZ)) << "clock mode produces continuous samples";

        bool foundPpsTag = false;
        for (const auto& tag : sink._tags) {
            if (tag.map.contains(std::pmr::string("trigger_name"))) {
                foundPpsTag = true;
            }
        }
        expect(foundPpsTag) << "PPS tags embedded in clock stream";
    };

    "GpsSource handles unlocked GPS (no fix)"_test = [] {
        auto pty = PtyPair::create();
        expect(pty.has_value()) << "PTY creation must succeed";
        if (!pty)
            return;

        Graph testGraph;
        auto& gps  = testGraph.emplaceBlock<GpsSource>({{"device_path", std::string(pty->slaveName)}});
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});
        expect(testGraph.connect<"out", "in">(gps, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // send invalid-fix RMC (status=V)
        pty->writeLine(nmea("$GPRMC,120000.00,V,,,,,,,110326,,,N"));
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        pty->writeLine(nmea("$GPRMC,120001.00,V,,,,,,,110326,,,N"));
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        sched.requestStop();
        schedThread.join();

        expect(ge(sink._nSamplesProduced, 1UZ)) << "PPS emitted even without fix";

        bool foundUnlocked = false;
        for (const auto& tag : sink._tags) {
            if (auto it = tag.map.find(std::pmr::string("trigger_name")); it != tag.map.end()) {
                auto name = std::string(it->second.value_or(std::string_view{}));
                if (name.find("unlocked") != std::string::npos) {
                    foundUnlocked = true;
                }
            }
        }
        expect(foundUnlocked) << "trigger_name contains 'unlocked' for no-fix data";
    };

    "GpsSource real device smoke test"_test = [] {
        auto device = selectNMEADevice({}, false);
        if (!device) {
            std::println("  [SKIP] no GPS device detected: {}", device.error());
            expect(true) << "skipped — no device";
            return;
        }

        Graph testGraph;
        auto& gps  = testGraph.emplaceBlock<GpsSource>();
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});
        expect(testGraph.connect<"out", "in">(gps, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(std::chrono::seconds(5));
        sched.requestStop();
        schedThread.join();

        std::println("  [INFO] samples: {}  tags: {}", sink._nSamplesProduced, sink._tags.size());
        expect(ge(sink._nSamplesProduced, 1UZ)) << "at least 1 PPS from real device in 5s";
        expect(!sink._tags.empty()) << "at least one tag received";
    };

#else
    "GpsSource PTY tests skipped on non-POSIX"_test = [] { expect(true) << "WASM/Windows: PTY tests not available"; };
#endif
};

int main() { return 0; }
