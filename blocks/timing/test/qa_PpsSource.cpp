#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/PpsSource.hpp>

using namespace boost::ut;
using namespace gr;
using namespace gr::timing;
using namespace gr::testing;

const boost::ut::suite<"PpsSource"> ppsSourceTests = [] {
    "PpsSource NTP mode emits PPS tags"_test = [] {
        Graph testGraph;
        auto& pps  = testGraph.emplaceBlock<PpsSource>({{"clock_mode", std::string("NTP")}});
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});
        expect(testGraph.connect<"out", "in">(pps, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));

        sched.requestStop();
        schedThread.join();

        expect(ge(sink._nSamplesProduced, 1UZ)) << "at least 1 PPS sample in ~2.5s";

        bool foundTriggerTag = false;
        for (const auto& tag : sink._tags) {
            if (auto it = tag.map.find(std::pmr::string("trigger_name")); it != tag.map.end()) {
                foundTriggerTag = true;
                auto name       = std::string(it->second.value_or(std::string_view{}));
                expect(name.find("PPS") != std::string::npos) << "trigger_name contains PPS";
                expect(name.find("NTP") != std::string::npos) << "trigger_name contains NTP";
            }
            if (auto it = tag.map.find(std::pmr::string("trigger_time")); it != tag.map.end()) {
                auto time = it->second.value_or(std::uint64_t{0});
                expect(gt(time, 0ULL)) << "trigger_time > 0";
            }
            if (auto it = tag.map.find(std::pmr::string("trigger_meta_info")); it != tag.map.end()) {
                auto* meta = it->second.get_if<property_map>();
                expect(meta != nullptr) << "trigger_meta_info is a map";
                if (meta) {
                    expect(meta->contains(std::pmr::string("clock_mode"))) << "clock_mode present";
                    expect(meta->contains(std::pmr::string("wakeup_offset_ns"))) << "wakeup_offset_ns present";
                    expect(meta->contains(std::pmr::string("synchronised"))) << "synchronised present";
                    expect(meta->contains(std::pmr::string("kernel_offset_ns"))) << "kernel_offset_ns present";
                    expect(meta->contains(std::pmr::string("est_error_ns"))) << "est_error_ns present";
                    expect(meta->contains(std::pmr::string("sequence"))) << "sequence present";
                    expect(meta->contains(std::pmr::string("leap_status"))) << "leap_status present";
                }
            }
        }
        expect(foundTriggerTag) << "at least one trigger tag found";
    };

    "PpsSource emit_meta_info=false omits trigger_meta_info"_test = [] {
        Graph testGraph;
        auto& pps  = testGraph.emplaceBlock<PpsSource>({{"clock_mode", std::string("NTP")}, {"emit_meta_info", false}});
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});
        expect(testGraph.connect<"out", "in">(pps, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));

        sched.requestStop();
        schedThread.join();

        expect(ge(sink._nSamplesProduced, 1UZ)) << "at least 1 PPS sample";

        for (const auto& tag : sink._tags) {
            expect(!tag.map.contains(std::pmr::string("trigger_meta_info"))) << "no meta_info when disabled";
        }
    };

    "PpsSource clock mode emits continuous samples"_test = [] {
        Graph testGraph;
        auto& pps  = testGraph.emplaceBlock<PpsSource>({{"clock_mode", std::string("NTP")}, {"emit_mode", std::string("clock")}, {"sample_rate", 100.f}});
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});
        expect(testGraph.connect<"out", "in">(pps, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));

        sched.requestStop();
        schedThread.join();

        expect(ge(sink._nSamplesProduced, 50UZ)) << "clock mode produces continuous samples";

        bool foundPpsTag = false;
        for (const auto& tag : sink._tags) {
            if (tag.map.contains(std::pmr::string("trigger_name"))) {
                foundPpsTag = true;
            }
        }
        expect(foundPpsTag) << "PPS tags embedded in clock stream";
    };

    "PpsSource Auto mode resolves to NTP or PTP"_test = [] {
        Graph testGraph;
        auto& pps  = testGraph.emplaceBlock<PpsSource>(); // default: Auto
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});
        expect(testGraph.connect<"out", "in">(pps, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));

        sched.requestStop();
        schedThread.join();

        expect(ge(sink._nSamplesProduced, 1UZ)) << "Auto mode produces at least 1 PPS";
    };

    "KernelDiscipline default values"_test = [] {
        KernelDiscipline d;
        expect(!d.synchronised);
        expect(eq(d.offsetNs, std::int64_t{0}));
        expect(eq(d.taiUtcOffsetS, std::int32_t{0}));
        expect(eq(d.leapStatus, std::uint8_t{0}));
    };

    "queryKernelDiscipline returns plausible values"_test = [] {
        auto d = gr::timing::detail::queryKernelDiscipline();
        // on a running Linux system, these should be finite values
        expect(d.maxErrorNs >= 0) << "max_error >= 0";
        expect(d.estErrorNs >= 0) << "est_error >= 0";
        // leap status must be one of the defined values
        expect(le(d.leapStatus, std::uint8_t{5})) << "leap_status in range";
    };

    "timespecToNs round-trips correctly"_test = [] {
        timespec ts{.tv_sec = 1000, .tv_nsec = 500};
        auto     ns = gr::timing::detail::timespecToNs(ts);
        expect(eq(ns, 1000ULL * 1'000'000'000ULL + 500ULL));
    };

    "clockModeName covers all modes"_test = [] {
        expect(eq(gr::timing::detail::clockModeName(ClockMode::NTP), std::string_view("NTP")));
        expect(eq(gr::timing::detail::clockModeName(ClockMode::PTP), std::string_view("PTP")));
        expect(eq(gr::timing::detail::clockModeName(ClockMode::TAI), std::string_view("TAI")));
        expect(eq(gr::timing::detail::clockModeName(ClockMode::HwPps), std::string_view("HwPps")));
        expect(eq(gr::timing::detail::clockModeName(ClockMode::Auto), std::string_view("Auto")));
    };

    "leapStatusName covers all states"_test = [] {
        expect(eq(gr::timing::detail::leapStatusName(TIME_OK), std::string_view("OK")));
        expect(eq(gr::timing::detail::leapStatusName(TIME_INS), std::string_view("insert_leap")));
        expect(eq(gr::timing::detail::leapStatusName(TIME_DEL), std::string_view("delete_leap")));
        expect(eq(gr::timing::detail::leapStatusName(TIME_OOP), std::string_view("leap_in_progress")));
        expect(eq(gr::timing::detail::leapStatusName(TIME_WAIT), std::string_view("leap_occurred")));
        expect(eq(gr::timing::detail::leapStatusName(TIME_ERROR), std::string_view("unsynchronised")));
    };

    "ClockMode enum values"_test = [] {
        expect(eq(static_cast<std::uint8_t>(ClockMode::NTP), std::uint8_t{0}));
        expect(eq(static_cast<std::uint8_t>(ClockMode::PTP), std::uint8_t{1}));
        expect(eq(static_cast<std::uint8_t>(ClockMode::TAI), std::uint8_t{2}));
        expect(eq(static_cast<std::uint8_t>(ClockMode::HwPps), std::uint8_t{3}));
        expect(eq(static_cast<std::uint8_t>(ClockMode::Auto), std::uint8_t{4}));
    };
};

int main() { return 0; }
