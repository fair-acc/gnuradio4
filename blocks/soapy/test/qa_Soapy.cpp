#include <boost/ut.hpp>

#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/meta/formatter.hpp>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#endif
#include <SoapySDR/Device.h>
#include <SoapySDR/Device.hpp>
#include <SoapySDR/Modules.h>
#include <SoapySDR/Modules.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/soapy/Soapy.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

namespace gr::blocks::soapy {
static_assert(std::is_constructible_v<SoapyBlock<std::complex<float>>, gr::property_map>, "SoapyBlock not default constructible w/ property_map");
static_assert(std::is_constructible_v<SoapySimpleSource<std::complex<float>>, gr::property_map>, "SoapyBlock not default constructible w/ property_map");
} // namespace gr::blocks::soapy

const boost::ut::suite<"basic SoapySDR API "> basicSoapyAPI = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::soapy;
    using gr::blocks::soapy::Range;

    "helper functions"_test = [] { "range printer"_test = [] { expect(eq(std::format("{}", gr::blocks::soapy::Range{1.0, 10.0, 0.5}), std::string("Range{min: 1, max: 10, step: 0.5}"))); }; };

    "ModulesCheck"_test = [] {
        std::vector<std::string> modules = getSoapySDRModules();

        if (modules.empty()) {
            std::println("No SoapySDR modules found.");
        } else {
            std::println("Available SoapySDR modules:");
            for (const auto& module : modules) {
                std::print("  Module: {}\n", module);
            }
        }
    };

    std::set<std::string> availableDeviceDriver; // alt = {"rtlsdr"};
    "available devices"_test = [&availableDeviceDriver] {
        SoapySDR::KwargsList devices = Device::enumerate();

        std::println("Detected devices:");
        std::size_t count = 0UZ;
        for (const auto& device : devices) {
            std::println("   Found device #{}: [{}]", count++, gr::join(device, ", "));
            availableDeviceDriver.insert(device.at("driver"));
        }

        if (devices.empty()) {
            std::println(stderr, "no devices found");
            return;
        }
    };
    std::println("Detected available devices: [{}]", gr::join(availableDeviceDriver, ", "));

    "Basic API test"_test =
        [](std::string deviceDriver) {
            std::println("Basic API test - deviceDriver '{}'", deviceDriver);

            Device device(deviceDriver.empty() ? SoapySDR::Kwargs{} : SoapySDR::Kwargs{{"driver", deviceDriver}});

            "Device Setting Info"_test = [&device] {
                std::vector<ArgInfo> settingsInfo = device.getSettingInfo();
                std::println("available settings options:");
                for (const auto& info : settingsInfo) {
                    std::println("  {:15} = '{}' [{}]- {} {}   - options: {} - {}", info.key, info.value, info.units, info.description, info.range, //
                        gr::join(info.options, ", "), gr::join(info.optionNames, ", "));
                }
                // if (!settingsInfo.empty()) {
                //     device.writeSetting("test_setting", "test_value");
                //     auto value = device.readSetting("test_setting");
                //     expect(value == "test_value");
                // }
            };

            "Channel Setting"_test = [&device] {
                std::vector<ArgInfo> settingsInfo = device.getChannelSettingInfo(SOAPY_SDR_RX, 0);
                std::println("available channel settings options:");
                for (const auto& info : settingsInfo) {
                    std::print("  {:15} = '{}' ['{}'] {}   - options: {} - {}\n", info.key, info.value, info.units, info.description, gr::join(info.options, ", "), gr::join(info.optionNames, ", "));
                }

                if (!settingsInfo.empty()) {
                    const ArgInfo& testSetting = settingsInfo[0]; // using first setting for testing setting API identity
                    device.writeChannelSetting(SOAPY_SDR_RX, 0, testSetting.key, testSetting.value);
                    std::string value = device.readChannelSetting(SOAPY_SDR_RX, 0, testSetting.key);
                    expect(eq(value, testSetting.value));
                }

                const std::string testMapping = device.getFrontendMapping(SOAPY_SDR_RX);
                device.setFrontendMapping(SOAPY_SDR_RX, testMapping);
                std::string retrievedMapping = device.getFrontendMapping(SOAPY_SDR_RX);
                expect(eq(retrievedMapping, testMapping));

                size_t numChannels = device.getNumChannels(SOAPY_SDR_RX);
                std::println("Number of RX channels: {}", numChannels);
                expect(numChannels > 0U);

                SoapySDR::Kwargs channelInfo = device.getChannelInfo(SOAPY_SDR_RX, 0);
                std::println("Channel info for RX channel 0:");
                for (const auto& [key, value] : channelInfo) {
                    std::print("  {}: {}\n", key, value);
                }

                bool isFullDuplex = device.getFullDuplex(SOAPY_SDR_RX, 0);
                std::println("Is RX channel 0 full duplex? {}", isFullDuplex);
            };

            "antenna"_test = [&device] {
                std::vector<std::string> antennas      = device.listAvailableAntennas(SOAPY_SDR_RX, 0);
                std::string              activeAntenna = device.getAntenna(SOAPY_SDR_RX, 0);
                std::println("Rx antennas: [{}] - active: '{}'", gr::join(antennas, ", "), activeAntenna);
                // expect(std::ranges::find(antennas, activeAntenna) != antennas.cend());

                device.setAntenna(SOAPY_SDR_RX, 0, activeAntenna);
                expect(eq(device.getAntenna(SOAPY_SDR_RX, 0), activeAntenna));
            };

            "gain"_test = [&device, &deviceDriver] {
                std::vector<std::string> gains = device.listAvailableGainElements(SOAPY_SDR_RX, 0);
                std::println("Rx Gains: [{}]", gr::join(gains, ", "));

                bool hasAutoGainMode = device.hasAutomaticGainControl(SOAPY_SDR_RX, 0);
                bool autoGain        = device.isAutomaticGainControl(SOAPY_SDR_RX, 0);
                std::println("RX has auto-gain mode: {} - active: {}", hasAutoGainMode, autoGain);
                if (hasAutoGainMode) {
                    expect(!autoGain);

                    device.setAutomaticGainControl(SOAPY_SDR_RX, 0, !autoGain);
                    expect(eq(device.isAutomaticGainControl(SOAPY_SDR_RX, 0), !autoGain));

                    device.setAutomaticGainControl(SOAPY_SDR_RX, 0, autoGain);
                    expect(eq(device.isAutomaticGainControl(SOAPY_SDR_RX, 0), autoGain));
                }

                double activeGain = device.getGain(SOAPY_SDR_RX, 0, "TUNER");
                std::println("RX active gain: {}", activeGain);

                if (deviceDriver != "audio") { // audio does not implement gains
                    device.setGain(SOAPY_SDR_RX, 0, 10.);
                    expect(approx(device.getGain(SOAPY_SDR_RX, 0), 10.0, 1e-2));
                }
            };

            "bandwidth"_test = [&device] {
                std::vector<double> bandwidths = device.listAvailableBandwidths(SOAPY_SDR_RX, 0);
                std::println("Rx available bandwidths: [{}]", gr::join(bandwidths, ", "));
                if (!bandwidths.empty()) {
                    double activeBandwidth = device.getBandwidth(SOAPY_SDR_RX, 0);
                    std::println("active bandwidth: {}", activeBandwidth);
                    expect(std::ranges::find(bandwidths, activeBandwidth) != bandwidths.cend());

                    device.setBandwidth(SOAPY_SDR_RX, 0, activeBandwidth);
                    expect(eq(device.getBandwidth(SOAPY_SDR_RX, 0), activeBandwidth));
                }
            };

            "center RF frequency"_test = [&device] {
                std::vector<gr::blocks::soapy::Range> ranges          = device.getOverallFrequencyRange(SOAPY_SDR_RX, 0);
                double                                centerFrequency = device.getCenterFrequency(SOAPY_SDR_RX, 0);
                std::println("Rx freq ranges: [{}] - active: {} Hz", gr::join(ranges, ", "), centerFrequency);

                device.setCenterFrequency(SOAPY_SDR_RX, 0, 106e6);
                expect(approx(device.getCenterFrequency(SOAPY_SDR_RX, 0), 106e6, 1e4));

                // device.setCenterFrequency(SOAPY_SDR_RX, 0, 1e6, {{"RF", "OFFSET"}});
                // expect(approx(device.getCenterFrequency(SOAPY_SDR_RX, 0), 107e6, 1e4));
            };

            "sample rate"_test = [&device] {
                std::vector<double> ranges     = device.listSampleRates(SOAPY_SDR_RX, 0);
                double              sampleRate = device.getSampleRate(SOAPY_SDR_RX, 0);
                std::println("Rx base-band sample rates: [{}] - active: {}", gr::join(ranges, ", "), sampleRate);

                device.setSampleRate(SOAPY_SDR_RX, 0, 1e6);
                expect(approx(device.getSampleRate(SOAPY_SDR_RX, 0), 1e6, 1e2));
            };

            "time sources"_test = [&device] {
                std::vector<std::string> timeSources = device.listAvailableTimeSources();
                std::string              timeSource  = device.getTimeSource();
                std::println("time sources: [{}] - active: '{}'", gr::join(timeSources, ", "), timeSource);
                if (!timeSources.empty()) {
                    expect(std::ranges::find(timeSources, timeSource) != timeSources.cend());

                    // device.getHardwareTime(SOAPY_SDR_RX, 0, activeAntenna);
                    //  expect(eq(device.getAntenna(SOAPY_SDR_RX, 0), activeAntenna));
                }
                std::uint64_t hwTime = device.getHardwareTime();
                std::println("time source hw time: {}", hwTime);
            };

            "simple acquisition"_test = [&device, &deviceDriver] {
                using TValueType = std::complex<float>;
                std::vector<TValueType> externalBuffer;
                externalBuffer.resize(1UZ << 14UZ);

                if (deviceDriver != "audio") {
                    device.setSampleRate(SOAPY_SDR_RX, 0, 1'000'000.);
                } else { // audio cannot go > 1 MS/s and other SDRs not below 1 MS/s
                    device.setSampleRate(SOAPY_SDR_RX, 0, 44'100.);
                }
                device.setCenterFrequency(SOAPY_SDR_RX, 0, 107'000'000.);
                Device::Stream rxStream = device.setupStream<TValueType, SOAPY_SDR_RX>();
                rxStream.activate();

                constexpr int maxSamples      = 100'000;
                int           receivedSamples = 0UZ;
                for (int i = 0; (i < 1000) && (receivedSamples <= maxSamples); ++i) {
                    std::uint32_t pollingTimeOutUs = 10'000; // 10 ms
                    int           flags            = 0;
                    long long     time_ns          = 0;
                    int           ret              = rxStream.readStream(flags, time_ns, pollingTimeOutUs, externalBuffer);
                    if (ret > 0) {
                        receivedSamples += ret;
                    }
                    if (i < 10 || ret > 0 || ret < -1) {
                        std::print("{:3}: ", i);
                        gr::blocks::soapy::detail::printSoapyReturnDebugInfo(ret, flags, time_ns);
                    }
                }
                expect(ge(receivedSamples, 1000)) << std::format("did not received enough samples for deviceDriver = {}\n", deviceDriver);
                rxStream.deactivate();
            };

            std::println("Basic API test - deviceDriver '{}' -- DONE", deviceDriver);
        } //
        | availableDeviceDriver;
    //  |  std::array{"rtlsdr"s, "lime"s}; // alt: fixed tests

    "unload soapy modules"_test = [] { expect(nothrow([] { SoapySDR_unloadModules(); })) << "WARNING: unload SoapyModules - FAILED"; };
};

const boost::ut::suite<"Soapy Block API "> soapyBlockAPI = [] {
    using namespace boost::ut;

    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        // conditionally enable visual tests outside the CI
        boost::ext::ut::cfg<override> = {.tag = {"rtlsdr", "lime"}};
    }

    // create and return a watchdog thread and its control flag
    using TDuration = std::chrono::duration<std::chrono::steady_clock::rep, std::chrono::steady_clock::period>;
    using namespace std::chrono_literals;
    auto createWatchdog = [](auto& sched, TDuration timeOut = 2s, TDuration pollingPeriod = 40ms) {
        using namespace std::chrono_literals;
        auto externalInterventionNeeded = std::make_shared<std::atomic_bool>(false); // unique_ptr because you cannot move atomics

        // Create the watchdog thread
        std::thread watchdogThread([&sched, &externalInterventionNeeded, timeOut, pollingPeriod]() {
            auto timeout = std::chrono::steady_clock::now() + timeOut;
            while (std::chrono::steady_clock::now() < timeout) {
                if (sched.state() == gr::lifecycle::State::STOPPED) {
                    return;
                }
                std::this_thread::sleep_for(pollingPeriod);
            }
            // time-out reached, need to force termination of scheduler
            std::println("watchdog kicked in");
            externalInterventionNeeded->store(true, std::memory_order_relaxed);
            sched.requestStop();
            std::println("requested scheduler to stop");
        });

        return std::make_pair(std::move(watchdogThread), externalInterventionNeeded);
    };

    auto threadPool                                             = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 10UZ);
    tag("rtlsdr") / "basic RTL soapy data generation test"_test = [&threadPool, &createWatchdog] {
        using namespace gr;
        using namespace gr::blocks::soapy;
        using namespace gr::testing;
        using scheduler = gr::scheduler::Simple<>;
        gr::Graph flow;
        using ValueType = std::complex<float>;

        constexpr gr::Size_t nSamples = 1e5;

        auto& source  = flow.emplaceBlock<SoapyBlock<ValueType, 1UZ>>({
            //
            {"device", "rtlsdr"},                                //
            {"sample_rate", float(1e6)},                         //
            {"rx_center_frequency", std::vector<double>{107e6}}, //
            {"rx_gains", std::vector<double>{20.}},
        });
        auto& monitor = flow.emplaceBlock<Copy<ValueType>>();
        auto& sink    = flow.emplaceBlock<CountingSink<ValueType>>({{"n_samples_max", nSamples}});
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(monitor)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(monitor).to<"in">(sink)));

        auto sched                                        = scheduler{std::move(flow), threadPool};
        auto [watchdogThread, externalInterventionNeeded] = createWatchdog(sched, 6s);

        auto retVal = sched.runAndWait();

        expect(retVal.has_value()) << std::format("scheduler execution error: {}", retVal.error());

        if (watchdogThread.joinable()) {
            watchdogThread.join();
        }
        expect(!externalInterventionNeeded->load(std::memory_order_relaxed));
        expect(eq(sink.count, nSamples));

        std::println("N.B. 'basic RTL soapy data generation test' test finished");
    };

    tag("lime") / "basic Lime soapy data generation test"_test = [&threadPool, &createWatchdog] {
        using namespace gr;
        using namespace gr::blocks::soapy;
        using namespace gr::testing;
        using scheduler = gr::scheduler::Simple<>;
        gr::Graph flow;
        using ValueType = std::complex<float>;

        constexpr gr::Size_t nSamples = 1e5;

        auto& source = flow.emplaceBlock<SoapyBlock<ValueType, 2UZ>>({
            //
            {"device", "lime"},                                         //
            {"rx_channels", std::vector<gr::Size_t>{0U, 1U}},           //
            {"rx_antennae", std::vector<std::string>{"LNAW", "LNAW"}},  //
            {"sample_rate", float(1e6)},                                //
            {"rx_center_frequency", std::vector<double>{107e6, 107e6}}, //
            {"rx_bandwdith", std::vector<double>{0.5e6, 0.5e6}},        //
            {"rx_gains", std::vector<double>{10., 10.}},
        });
        auto& sink1  = flow.emplaceBlock<CountingSink<ValueType>>({{"n_samples_max", nSamples}});
        auto& sink2  = flow.emplaceBlock<CountingSink<ValueType>>({{"n_samples_max", nSamples}});
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out", 0>(source).to<"in">(sink1)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out", 1>(source).to<"in">(sink2)));

        auto sched = scheduler{std::move(flow), threadPool};

        auto [watchdogThread, externalInterventionNeeded] = createWatchdog(sched, 6s);

        auto retVal = sched.runAndWait();
        expect(retVal.has_value()) << std::format("scheduler execution error: {}", retVal.error());

        if (watchdogThread.joinable()) {
            watchdogThread.join();
        }
        expect(!externalInterventionNeeded->load(std::memory_order_relaxed));
        expect(eq(sink1.count, nSamples));
        expect(eq(sink2.count, nSamples));

        std::println("N.B. 'basic LimeSDR soapy data generation test' test finished");
    };

    "unload soapy modules"_test = [] { expect(nothrow([] { SoapySDR_unloadModules(); })) << "WARNING: unload SoapyModules - FAILED"; };
};

int main() { /* not needed for UT */ }
