#include <boost/ut.hpp>

#include <complex>
#include <cstdlib>
#include <filesystem>

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/common/USBDevice.hpp>
#include <gnuradio-4.0/sdr/LoopbackDevice.hpp>
#include <gnuradio-4.0/sdr/SoapySink.hpp>
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
            {"device", "loopback"}, {"device_parameter", std::string("device_mode=rx_only")}, {"sample_rate", 1e6f}, {"frequency", std::vector{433.92e6}}, {"rx_gains", std::vector{0.}}, {"emit_timing_tags", true}, {"emit_meta_info", true}, {"tag_interval", 0.f}, // tag every chunk
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
                if (auto name = nameIt->second.get_if<std::string_view>()) {
                    if (*name == "GPS_PPS") {
                        foundClockTag = true;
                        break;
                    }
                }
            }
        }
        expect(foundClockTag) << "at least one timing tag should carry the clk_in trigger name";
    };

    "clk_in EoS stops SoapySource"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nClockSamples = 500;
        constexpr float      rate          = 1e6f;

        auto& clock = flow.emplaceBlock<gr::basic::ClockSource<std::uint8_t>>({
            {"sample_rate", rate},
            {"n_samples_max", nClockSamples},
            {"chunk_size", gr::Size_t{100}},
        });

        auto& source = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "loopback"},
            {"device_parameter", std::string("device_mode=rx_only")},
            {"sample_rate", rate},
            {"frequency", std::vector{100e3}},
            {"rx_gains", std::vector{0.}},
        });
        auto& sink   = flow.emplaceBlock<TagSink<CF32, ProcessFunction::USE_PROCESS_BULK>>({
            {"n_samples_expected", gr::Size_t{0}}, // unlimited — stopped by EoS
            {"log_tags", true},
            {"log_samples", false},
        });

        expect(flow.connect<"out", "clk_in">(clock, source).has_value());
        expect(flow.connect<"out", "in">(source, sink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        auto ret = runWithWatchdog(sched, std::chrono::seconds{4});
        expect(ret.has_value());
        expect(gt(sink._nSamplesProduced, gr::Size_t{0})) << "should have received samples before clk_in EoS";
    };

    "DC blocker removes DC offset from rxOnly tone"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 50000; // need enough for IIR to settle

        auto& source = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "loopback"}, {"device_parameter", std::string("device_mode=rx_only")}, {"sample_rate", 1e6f}, {"frequency", std::vector{0.}}, // DC tone (frequency=0 → constant {1,0})
            {"rx_gains", std::vector{0.}}, {"dc_blocker_enabled", true}, {"dc_blocker_cutoff", 100.f},                                               // aggressive cutoff for fast settling
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
            {"device", "loopback"}, {"device_parameter", std::string("device_mode=rx_only")}, {"sample_rate", rate}, {"frequency", std::vector{100e3}}, {"rx_gains", std::vector{0.}}, {"emit_timing_tags", true}, {"tag_interval", 0.01f}, // 10 ms between tags → expect ~5 tags in 50 ms of data
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

const boost::ut::suite<"SoapySink + SoapySource shared device"> txRxTests = [] {
    using namespace gr;
    using namespace gr::blocks::sdr;
    using namespace gr::testing;
    using Sched = gr::scheduler::Simple<>;

    "TX→RX round-trip through shared loopback device"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 10000;

        auto& txSource = flow.emplaceBlock<gr::basic::ClockSource<CF32>>({
            {"n_samples_max", nSamples},
            {"sample_rate", 1e6f},
            {"chunk_size", gr::Size_t{1024}},
        });
        auto& sink     = flow.emplaceBlock<SoapySink<CF32, 1UZ>>({
            {"device", "loopback"},
            {"sample_rate", 1e6f},
            {"frequency", std::vector{100e3}},
            {"tx_gains", std::vector{0.}},
        });
        auto& source   = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "loopback"},
            {"sample_rate", 1e6f},
            {"frequency", std::vector{100e3}},
            {"rx_gains", std::vector{0.}},
        });
        auto& rxSink   = flow.emplaceBlock<CountingSink<CF32>>({{"n_samples_max", nSamples}});

        expect(flow.connect<"out", "in">(txSource, sink).has_value());
        expect(flow.connect<"out", "in">(source, rxSink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(runWithWatchdog(sched).has_value());
        expect(eq(rxSink.count.value, nSamples)) << std::format("expected {} samples, got {}", nSamples, rxSink.count.value);
    };

    "shared device handle verified via Device::make()"_test = [] {
        auto dev1 = soapy::Device::make({{"driver", "loopback"}});
        auto dev2 = soapy::Device::make({{"driver", "loopback"}});
        expect(dev1.has_value());
        expect(dev2.has_value());
        expect(eq(dev1->get(), dev2->get())) << "same kwargs should return same device handle";
    };

    "device handle survives after one user resets"_test = [] {
        auto dev1 = soapy::Device::make({{"driver", "loopback"}});
        auto dev2 = soapy::Device::make({{"driver", "loopback"}});
        expect(dev1.has_value());
        expect(dev2.has_value());
        auto* rawPtr = dev1->get();
        dev1->reset();
        expect(eq(dev1->get(), static_cast<SoapySDRDevice*>(nullptr)));
        expect(eq(dev2->get(), rawPtr)) << "second handle should still be valid";
    };

    "SoapySink standalone with txOnly loopback"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 5000;

        auto& txSource = flow.emplaceBlock<gr::basic::ClockSource<CF32>>({
            {"n_samples_max", nSamples},
            {"sample_rate", 1e6f},
            {"chunk_size", gr::Size_t{512}},
        });
        auto& sink     = flow.emplaceBlock<SoapySink<CF32, 1UZ>>({
            {"device", "loopback"},
            {"device_parameter", std::string("device_mode=tx_only")},
            {"sample_rate", 1e6f},
            {"frequency", std::vector{100e3}},
            {"tx_gains", std::vector{0.}},
        });
        expect(flow.connect<"out", "in">(txSource, sink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(runWithWatchdog(sched).has_value());
    };

    "2-channel TX→RX round-trip"_test = [] {
        gr::Graph            flow;
        constexpr gr::Size_t nSamples = 5000;

        auto& txSrc1  = flow.emplaceBlock<gr::basic::ClockSource<CF32>>({
            {"n_samples_max", nSamples},
            {"sample_rate", 1e6f},
            {"chunk_size", gr::Size_t{512}},
        });
        auto& txSrc2  = flow.emplaceBlock<gr::basic::ClockSource<CF32>>({
            {"n_samples_max", nSamples},
            {"sample_rate", 1e6f},
            {"chunk_size", gr::Size_t{512}},
        });
        auto& sink    = flow.emplaceBlock<SoapySink<CF32, 2UZ>>({
            {"device", "loopback"},
            {"device_parameter", std::string("num_channels=2")},
            {"sample_rate", 1e6f},
            {"num_channels", gr::Size_t{2}},
            {"frequency", std::vector{100e3, 200e3}},
            {"tx_gains", std::vector{0., 0.}},
        });
        auto& source  = flow.emplaceBlock<SoapySource<CF32, 2UZ>>({
            {"device", "loopback"},
            {"device_parameter", std::string("num_channels=2")},
            {"sample_rate", 1e6f},
            {"num_channels", gr::Size_t{2}},
            {"frequency", std::vector{100e3, 200e3}},
            {"rx_gains", std::vector{0., 0.}},
        });
        auto& rxSink1 = flow.emplaceBlock<CountingSink<CF32>>({{"n_samples_max", nSamples}});
        auto& rxSink2 = flow.emplaceBlock<CountingSink<CF32>>({{"n_samples_max", nSamples}});

        expect(flow.connect<"out", "in#0">(txSrc1, sink).has_value());
        expect(flow.connect<"out", "in#1">(txSrc2, sink).has_value());
        expect(flow.connect<"out#0", "in">(source, rxSink1).has_value());
        expect(flow.connect<"out#1", "in">(source, rxSink2).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(runWithWatchdog(sched, std::chrono::seconds{20}).has_value());
        constexpr auto kMinExpected = static_cast<gr::Size_t>(nSamples * 0.9);
        expect(ge(rxSink1.count.value, kMinExpected)) << std::format("channel 0: got {}, expected at least {}", rxSink1.count.value, kMinExpected);
        expect(ge(rxSink2.count.value, kMinExpected)) << std::format("channel 1: got {}, expected at least {}", rxSink2.count.value, kMinExpected);
    };
};

const boost::ut::suite<"LimeSDR hardware"> limeTests = [] {
    using namespace gr;
    using namespace gr::blocks::sdr;
    using namespace gr::testing;
    using Sched = gr::scheduler::Simple<>;

    auto limeAvailable = [] {
        auto devices = soapy::Device::enumerate({{"driver", "lime"}});
        return !devices.empty();
    };

    auto resetLimeUsb = [] {
#if defined(__linux__)
        auto devices = gr::blocks::common::enumerateUSBDevices(std::array{
            gr::blocks::common::USBDeviceId{0x1D50, 0x6108, "LimeSDR-USB"},
        });
        for (const auto& dev : devices) {
            gr::blocks::common::USBDevice usbDev;
            if (auto r = usbDev.open(dev); r) {
                std::ignore = usbDev.reset();
            }
        }
        if (!devices.empty()) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
#endif
    };

    "full-duplex TX+RX on shared LimeSDR device"_test = [&] {
        if (!limeAvailable()) {
            std::println(stderr, "[SKIP] no LimeSDR device found");
            return;
        }
        resetLimeUsb();

        gr::Graph            flow;
        constexpr float      rate     = 1e6f;
        constexpr gr::Size_t nSamples = static_cast<gr::Size_t>(rate * 10);

        auto& txSource = flow.emplaceBlock<gr::basic::ClockSource<CF32>>({
            {"n_samples_max", nSamples},
            {"sample_rate", rate},
            {"chunk_size", gr::Size_t{4096}},
        });
        auto& sink     = flow.emplaceBlock<SoapySink<CF32, 1UZ>>({
            {"device", "lime"},
            {"sample_rate", rate},
            {"frequency", std::vector{433.92e6}},
            {"tx_antennae", std::vector<std::string>{"BAND1"}},
            {"tx_bandwidths", std::vector{5e6}},
            {"tx_gains", std::vector{20.}},
        });
        auto& source   = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "lime"},
            {"sample_rate", rate},
            {"frequency", std::vector{433.92e6}},
            {"rx_antennae", std::vector<std::string>{"LNAW"}},
            {"rx_bandwidths", std::vector{5e6}},
            {"rx_gains", std::vector{20.}},
            {"max_overflow_count", gr::Size_t{0}},
            {"verbose_overflow", true},
        });
        auto& rxSink   = flow.emplaceBlock<CountingSink<CF32>>({{"n_samples_max", nSamples}});

        expect(flow.connect<"out", "in">(txSource, sink).has_value());
        expect(flow.connect<"out", "in">(source, rxSink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        auto ret = runWithWatchdog(sched, std::chrono::seconds{15});
        if (!ret.has_value()) {
            expect(false) << std::format("scheduler error: {}", ret.error());
        }

        std::println("LimeSDR full-duplex: RX received {} of {} samples", rxSink.count.value, nSamples);
        expect(eq(rxSink.count.value, nSamples));
    };

    "2-channel duplex on LimeSDR"_test = [&] {
        if (!limeAvailable()) {
            std::println(stderr, "[SKIP] no LimeSDR device found");
            return;
        }
        resetLimeUsb();

        gr::Graph            flow;
        constexpr float      rate     = 1e6f;
        constexpr gr::Size_t nSamples = static_cast<gr::Size_t>(rate * 10);

        auto& txSrc1  = flow.emplaceBlock<gr::basic::ClockSource<CF32>>({
            {"n_samples_max", nSamples},
            {"sample_rate", rate},
            {"chunk_size", gr::Size_t{4096}},
        });
        auto& txSrc2  = flow.emplaceBlock<gr::basic::ClockSource<CF32>>({
            {"n_samples_max", nSamples},
            {"sample_rate", rate},
            {"chunk_size", gr::Size_t{4096}},
        });
        auto& sink    = flow.emplaceBlock<SoapySink<CF32, 2UZ>>({
            {"device", "lime"},
            {"sample_rate", rate},
            {"num_channels", gr::Size_t{2}},
            {"frequency", std::vector{433.92e6, 433.92e6}},
            {"tx_antennae", std::vector<std::string>{"BAND1", "BAND1"}},
            {"tx_bandwidths", std::vector{5e6, 5e6}},
            {"tx_gains", std::vector{20., 20.}},
        });
        auto& source  = flow.emplaceBlock<SoapySource<CF32, 2UZ>>({
            {"device", "lime"},
            {"sample_rate", rate},
            {"num_channels", gr::Size_t{2}},
            {"frequency", std::vector{433.92e6, 433.92e6}},
            {"rx_antennae", std::vector<std::string>{"LNAW", "LNAW"}},
            {"rx_bandwidths", std::vector{5e6, 5e6}},
            {"rx_gains", std::vector{20., 20.}},
            {"max_overflow_count", gr::Size_t{0}},
            {"verbose_overflow", true},
        });
        auto& rxSink1 = flow.emplaceBlock<CountingSink<CF32>>({{"n_samples_max", nSamples}});
        auto& rxSink2 = flow.emplaceBlock<CountingSink<CF32>>({{"n_samples_max", nSamples}});

        expect(flow.connect<"out", "in#0">(txSrc1, sink).has_value());
        expect(flow.connect<"out", "in#1">(txSrc2, sink).has_value());
        expect(flow.connect<"out#0", "in">(source, rxSink1).has_value());
        expect(flow.connect<"out#1", "in">(source, rxSink2).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        auto ret = runWithWatchdog(sched, std::chrono::seconds{20});
        expect(ret.has_value());

        constexpr auto kMinExpected = static_cast<gr::Size_t>(nSamples * 0.9);
        std::println("LimeSDR 2-ch duplex: ch0={}, ch1={} of {} samples (min {})", rxSink1.count.value, rxSink2.count.value, nSamples, kMinExpected);
        expect(ge(rxSink1.count.value, kMinExpected)) << "channel 0";
        expect(ge(rxSink2.count.value, kMinExpected)) << "channel 1";
    };
};

int main() {
    if (!std::getenv("SOAPY_SDR_PLUGIN_PATH")) {
        std::error_code ec;
        auto            exePath = std::filesystem::read_symlink("/proc/self/exe", ec);
        if (ec) {
            std::println(stderr, "[qa_SoapyIntegration] SOAPY_SDR_PLUGIN_PATH not set and /proc/self/exe unreadable ({}) — loopback tests will fail", ec.message());
            return 0;
        }

        auto modulePath = exePath.parent_path() / "soapy_modules";
        if (std::filesystem::exists(modulePath)) {
            setenv("SOAPY_SDR_PLUGIN_PATH", modulePath.c_str(), 0);
        } else {
            std::println(stderr, "[qa_SoapyIntegration] SOAPY_SDR_PLUGIN_PATH not set and {} not found — loopback tests will fail", modulePath.string());
        }
    }
}

const boost::ut::suite<"SoapySink BurstTaper"> taperTests = [] {
    using namespace gr::blocks::sdr;
    namespace loopback = gr::blocks::sdr::loopback;

    "taper shapes TX envelope through loopback"_test = [] {
        constexpr float       kSampleRate = 100'000.f;
        constexpr float       kRampTime   = 0.01f; // 10 ms → 1000 ramp samples
        constexpr std::size_t kRampLen    = static_cast<std::size_t>(kRampTime * kSampleRate);
        constexpr std::size_t kTotalTx    = kRampLen * 3; // ramp-up + flat + margin

        auto device = loopback::DeviceRegistry::findOrCreate(80, {{"driver", "loopback#80"}, {"num_channels", "1"}});

        auto* txStream = device->setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, {0});
        auto* rxStream = device->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0});
        device->activateStream(txStream);
        device->activateStream(rxStream);

        gr::algorithm::BurstTaper<float> taper(gr::algorithm::TaperType::RaisedCosine, kRampTime, kSampleRate);
        std::ignore = taper.setTarget(true);

        std::vector<CF32> txBuf(kTotalTx, CF32{1.0f, 0.0f});
        taper.applyInPlace(std::span<float>(reinterpret_cast<float*>(txBuf.data()), kTotalTx));
        // applyInPlace operates on float; for CF32 we need per-sample application
        // redo with manual per-sample taper
        taper.reset();
        std::ignore = taper.setTarget(true);
        for (auto& s : txBuf) {
            s *= taper.processOne();
        }

        const void* txPtr  = txBuf.data();
        int         flags  = 0;
        long long   timeNs = 0;
        int         txRet  = device->writeStream(txStream, &txPtr, kTotalTx, flags, timeNs);
        expect(ge(txRet, 0)) << "writeStream failed";

        std::vector<CF32> rxBuf(kTotalTx);
        void*             rxPtr = rxBuf.data();
        int               rxRet = device->readStream(rxStream, &rxPtr, kTotalTx, flags, timeNs);
        expect(eq(static_cast<std::size_t>(rxRet), kTotalTx)) << "readStream short read";

        // verify ramp-up: first kRampLen samples have increasing magnitude
        float prevMag   = 0.f;
        bool  monotonic = true;
        for (std::size_t i = 1UZ; i < kRampLen; ++i) {
            float mag = std::abs(rxBuf[i]);
            if (mag < prevMag - 1e-6f) {
                monotonic = false;
            }
            prevMag = mag;
        }
        expect(monotonic) << "ramp-up should be monotonically increasing";
        expect(lt(std::abs(rxBuf[0]), 0.1f)) << "first sample should be near zero";
        expect(gt(std::abs(rxBuf[kRampLen - 1]), 0.9f)) << "last ramp sample should be near 1.0";

        // verify flat region: samples after ramp should be ~1.0
        for (std::size_t i = kRampLen; i < kRampLen * 2; ++i) {
            expect(gt(std::abs(rxBuf[i]), 0.99f)) << std::format("flat sample {} should be ~1.0", i);
        }

        device->deactivateStream(txStream);
        device->deactivateStream(rxStream);
    };

    "SoapySink with taper enabled passes samples through loopback"_test = [] {
        using Sched              = gr::scheduler::Simple<>;
        constexpr float kRate    = 1e6f;
        gr::Size_t      nSamples = 50'000;
        gr::Graph       flow;

        auto& clockSrc = flow.emplaceBlock<gr::basic::ClockSource<CF32>>({
            {"sample_rate", kRate},
            {"n_samples_max", nSamples},
        });
        auto& txSink   = flow.emplaceBlock<SoapySink<CF32, 1UZ>>({
            {"device", "loopback"},
            {"sample_rate", kRate},
            {"frequency", std::vector{107e6}},
            {"burst_taper_enabled", true},
            {"burst_ramp_time", 0.001f},
            {"burst_taper_type", std::string("Linear")},
        });
        auto& source   = flow.emplaceBlock<SoapySource<CF32, 1UZ>>({
            {"device", "loopback"},
            {"sample_rate", kRate},
            {"frequency", std::vector{107e6}},
        });
        auto& rxSink   = flow.emplaceBlock<gr::testing::CountingSink<CF32>>({{"n_samples_max", nSamples}});

        expect(flow.connect<"out", "in">(clockSrc, txSink).has_value());
        expect(flow.connect<"out", "in">(source, rxSink).has_value());

        Sched sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(runWithWatchdog(sched, std::chrono::seconds{10}).has_value());
        expect(ge(rxSink.count, nSamples * 0.8)) << std::format("expected ~{} RX samples, got {}", nSamples, rxSink.count);
    };

    "taper disabled passes unmodified signal"_test = [] {
        constexpr std::size_t kN = 64;

        auto  device   = loopback::DeviceRegistry::findOrCreate(82, {{"driver", "loopback#82"}, {"num_channels", "1"}});
        auto* txStream = device->setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, {0});
        auto* rxStream = device->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0});
        device->activateStream(txStream);
        device->activateStream(rxStream);

        std::vector<CF32> txBuf(kN, CF32{0.5f, -0.25f});
        const void*       txPtr  = txBuf.data();
        int               flags  = 0;
        long long         timeNs = 0;
        int               txRet  = device->writeStream(txStream, &txPtr, kN, flags, timeNs);
        expect(eq(txRet, static_cast<int>(kN)));

        std::vector<CF32> rxBuf(kN);
        void*             rxPtr = rxBuf.data();
        int               rxRet = device->readStream(rxStream, &rxPtr, kN, flags, timeNs);
        expect(eq(rxRet, static_cast<int>(kN)));

        for (std::size_t i = 0UZ; i < kN; ++i) {
            expect(approx(rxBuf[i].real(), 0.5f, 1e-5f)) << std::format("sample {} real", i);
            expect(approx(rxBuf[i].imag(), -0.25f, 1e-5f)) << std::format("sample {} imag", i);
        }

        device->deactivateStream(txStream);
        device->deactivateStream(rxStream);
    };
};
