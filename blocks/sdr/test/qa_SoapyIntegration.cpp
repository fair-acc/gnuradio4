#include <boost/ut.hpp>

#include <complex>

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/sdr/SoapySource.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using namespace boost::ut;
using CF32 = std::complex<float>;

namespace {

auto runWithWatchdog(auto& sched, std::chrono::seconds timeout = std::chrono::seconds{6}) {
    auto watchdog = std::jthread([&sched, timeout](std::stop_token stoken) {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline && !stoken.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!stoken.stop_requested()) {
            sched.requestStop();
        }
    });
    auto ret      = sched.runAndWait();
    return ret;
}

} // namespace

const boost::ut::suite<"SoapySource + Loopback"> integrationTests = [] {
    using namespace gr;
    using namespace gr::blocks::sdr;
    using namespace gr::testing;
    using Sched = gr::scheduler::Simple<>;

    "single-channel rxOnly delivers samples"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 10000;

        auto& source = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "loopback"},
            {"device_parameter", std::string("device_mode=rx_only")},
            {"sample_rate", 1e6f},
            {"frequency", std::vector{100e3}},
            {"rx_gains", std::vector{0.}},
        });
        auto& sink   = flow.emplaceBlock<CountingSink<CF32>>({{"n_samples_max", nSamples}});
        expect(flow.connect<"out", "in">(source, sink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(runWithWatchdog(sched).has_value());
        expect(eq(sink.count.value, nSamples));
    };

    "2-channel rxOnly delivers to both sinks"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 5000;

        auto& source = flow.emplaceBlock<SoapySource<CF32, 2UZ>>({
            {"device", "loopback"},
            {"device_parameter", std::string("device_mode=rx_only,num_channels=2")},
            {"sample_rate", 1e6f},
            {"num_channels", gr::Size_t{2}},
            {"frequency", std::vector{100e3, 200e3}},
            {"rx_gains", std::vector{0., 0.}},
        });
        auto& sink1  = flow.emplaceBlock<CountingSink<CF32>>({{"n_samples_max", nSamples}});
        auto& sink2  = flow.emplaceBlock<CountingSink<CF32>>({{"n_samples_max", nSamples}});
        expect(flow.connect<"out#0", "in">(source, sink1).has_value());
        expect(flow.connect<"out#1", "in">(source, sink2).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(runWithWatchdog(sched).has_value());
        expect(eq(sink1.count.value, nSamples)) << "channel 0";
        expect(eq(sink2.count.value, nSamples)) << "channel 1";
    };

    "rxOnly propagates tags through the graph"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 10000;

        auto& source = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "loopback"},
            {"device_parameter", std::string("device_mode=rx_only")},
            {"sample_rate", 1e6f},
            {"frequency", std::vector{433.92e6}},
            {"rx_gains", std::vector{0.}},
            {"emit_timing_tags", true},
            {"emit_meta_info", true},
            {"tag_interval", 0.f}, // tag every chunk
        });
        auto& sink   = flow.emplaceBlock<TagSink<CF32, ProcessFunction::USE_PROCESS_BULK>>({
            {"n_samples_expected", nSamples},
            {"log_tags", true},
            {"log_samples", false},
        });
        expect(flow.connect<"out", "in">(source, sink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(runWithWatchdog(sched).has_value());
        expect(ge(sink._nSamplesProduced, nSamples));
        expect(!sink._tags.empty()) << "should receive at least one tag";
    };

    "clk_in forwards external timing tags"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 10000;
        constexpr float      rate     = 1e6f;

        auto& clock = flow.emplaceBlock<gr::basic::ClockSource<std::uint8_t>>({
            {"sample_rate", rate},
            {"n_samples_max", gr::Size_t{0}}, // unlimited — stopped by scheduler
            {"chunk_size", gr::Size_t{100}},
        });
        clock.tags  = {
            {0, {{std::pmr::string(gr::tag::TRIGGER_NAME.shortKey()), std::pmr::string("GPS_PPS")}, {std::pmr::string(gr::tag::TRIGGER_TIME.shortKey()), std::uint64_t{1'000'000'000ULL}}}},
        };

        auto& source = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "loopback"},
            {"device_parameter", std::string("device_mode=rx_only")},
            {"sample_rate", rate},
            {"frequency", std::vector{100e3}},
            {"rx_gains", std::vector{0.}},
            {"emit_timing_tags", true},
            {"emit_meta_info", true},
            {"tag_interval", 0.f},
        });

        auto& sink = flow.emplaceBlock<TagSink<CF32, ProcessFunction::USE_PROCESS_BULK>>({
            {"n_samples_expected", nSamples},
            {"log_tags", true},
            {"log_samples", false},
        });

        expect(flow.connect<"out", "clk_in">(clock, source).has_value());
        expect(flow.connect<"out", "in">(source, sink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(runWithWatchdog(sched).has_value());
        expect(ge(sink._nSamplesProduced, nSamples));
        expect(!sink._tags.empty()) << "should receive timing tags";

        // verify at least one tag carries the forwarded clock name
        bool foundClockTag = false;
        for (const auto& tag : sink._tags) {
            auto nameIt = tag.map.find(std::pmr::string(gr::tag::TRIGGER_NAME.shortKey()));
            if (nameIt != tag.map.end()) {
                if (auto* name = nameIt->second.get_if<std::pmr::string>()) {
                    if (*name == "GPS_PPS") {
                        foundClockTag = true;
                        break;
                    }
                }
            }
        }
        expect(foundClockTag) << "at least one timing tag should carry the clk_in trigger name";
    };

    "DC blocker removes DC offset from rxOnly tone"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 50000; // need enough for IIR to settle

        auto& source = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "loopback"},
            {"device_parameter", std::string("device_mode=rx_only")},
            {"sample_rate", 1e6f},
            {"frequency", std::vector{0.}}, // DC tone (frequency=0 → constant {1,0})
            {"rx_gains", std::vector{0.}},
            {"dc_blocker_enabled", true},
            {"dc_blocker_cutoff", 100.f}, // aggressive cutoff for fast settling
        });
        auto& sink   = flow.emplaceBlock<TagSink<CF32, ProcessFunction::USE_PROCESS_BULK>>({
            {"n_samples_expected", nSamples},
            {"log_tags", false},
            {"log_samples", true},
        });
        expect(flow.connect<"out", "in">(source, sink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(runWithWatchdog(sched).has_value());
        expect(ge(sink._nSamplesProduced, nSamples));

        // the last quarter of samples should have near-zero DC after the filter settles
        auto   totalSamples = sink._samples.size();
        double sumMag       = 0.0;
        auto   startIdx     = totalSamples * 3 / 4;
        for (std::size_t i = startIdx; i < totalSamples; ++i) {
            sumMag += static_cast<double>(std::abs(sink._samples[i]));
        }
        auto avgMag = static_cast<float>(sumMag / static_cast<double>(totalSamples - startIdx));
        expect(lt(avgMag, 0.1f)) << std::format("DC blocker should suppress DC tone, avg magnitude: {}", avgMag);
    };

    "timing tags emitted at configured interval"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 50000;
        constexpr float      rate     = 1e6f;

        auto& source = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "loopback"},
            {"device_parameter", std::string("device_mode=rx_only")},
            {"sample_rate", rate},
            {"frequency", std::vector{100e3}},
            {"rx_gains", std::vector{0.}},
            {"emit_timing_tags", true},
            {"tag_interval", 0.01f}, // 10 ms between tags → expect ~5 tags in 50 ms of data
        });
        auto& sink   = flow.emplaceBlock<TagSink<CF32, ProcessFunction::USE_PROCESS_BULK>>({
            {"n_samples_expected", nSamples},
            {"log_tags", true},
            {"log_samples", false},
        });
        expect(flow.connect<"out", "in">(source, sink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(runWithWatchdog(sched).has_value());
        expect(ge(sink._nSamplesProduced, nSamples));
        expect(ge(sink._tags.size(), 2UZ)) << std::format("expected multiple timing tags, got {}", sink._tags.size());
    };

    "loopback mode TX→RX through scheduler"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 10000;

        // in default loopback mode, rxOnly tone is NOT active — need TX data
        // without a SoapySink, the loopback buffer stays empty → expect TIMEOUT → scheduler stops via watchdog
        // this test verifies the scheduler handles the loopback-without-TX gracefully
        auto& source = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "loopback"},
            {"sample_rate", 1e6f},
            {"frequency", std::vector{100e3}},
            {"rx_gains", std::vector{0.}},
        });
        auto& sink   = flow.emplaceBlock<CountingSink<CF32>>({{"n_samples_max", nSamples}});
        expect(flow.connect<"out", "in">(source, sink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());

        // short watchdog — loopback without TX will not deliver samples
        auto ret = runWithWatchdog(sched, std::chrono::seconds{3});
        expect(ret.has_value());
        expect(lt(sink.count.value, nSamples)) << "loopback without TX should not deliver all samples";
    };
};

int main() { /* not needed for UT */ }
