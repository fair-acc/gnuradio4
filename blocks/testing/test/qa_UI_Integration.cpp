#include "boost/ut.hpp"

#include "gnuradio-4.0/Block.hpp"
#include "gnuradio-4.0/Graph.hpp"
#include "gnuradio-4.0/Scheduler.hpp"
#include "gnuradio-4.0/Tag.hpp"

#include "gnuradio-4.0/algorithm/ImChart.hpp"
#include "gnuradio-4.0/basic/ClockSource.hpp"
#include "gnuradio-4.0/basic/FunctionGenerator.hpp"
#include "gnuradio-4.0/meta/UnitTestHelper.hpp"
#include "gnuradio-4.0/testing/ImChartMonitor.hpp"
#include "gnuradio-4.0/testing/TagMonitors.hpp"

const boost::ut::suite TagTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    constexpr static auto initClockSource = [](auto& clockSrc) {
        auto addTimeTagEntry = []<typename T>(gr::basic::ClockSource<T>& clockSource, std::uint64_t timeInNanoseconds, const std::string& value) {
            clockSource.tag_times.value.push_back(timeInNanoseconds);
            clockSource.tag_values.value.push_back(value);
        };

        // all times are in nanoseconds
        constexpr std::uint64_t ms = 1'000'000; // ms -> ns conversion factor (wish we had a proper C++ units-lib integration)
        addTimeTagEntry(clockSrc, 10 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1");
        addTimeTagEntry(clockSrc, 100 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2");
        addTimeTagEntry(clockSrc, 300 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=3");
        addTimeTagEntry(clockSrc, 350 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4");
        addTimeTagEntry(clockSrc, 550 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=5");
        addTimeTagEntry(clockSrc, 650 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=6");
        addTimeTagEntry(clockSrc, 800 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=7");
        addTimeTagEntry(clockSrc, 850 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=8");
        clockSrc.repeat_period      = 1000 * ms;
        clockSrc.do_zero_order_hold = true;
    };

    constexpr static auto initFunctionGenerator = [](auto& funcGen) {
        using namespace function_generator;

        const auto now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 5.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=1"}).empty());
        expect(funcGen.settings().set(createLinearRampPropertyMap("CMD_BP_START", 5.f, 30.f, .2f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=2"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 30.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=3"}).empty());
        expect(funcGen.settings().set(createParabolicRampPropertyMap("CMD_BP_START", 30.f, 20.f, .1f, 0.02f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=4"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 20.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=5"}).empty());
        expect(funcGen.settings().set(createCubicSplinePropertyMap("CMD_BP_START", 20.f, 10.f, .1f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=6"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 10.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=7"}).empty());
        expect(funcGen.settings().set(createImpulseResponsePropertyMap("CMD_BP_START", 10.f, 20.f, .02f, .06f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=8"}).empty());
    };

    "FunctionGenerator + ClockSource FAIR test"_test = [] {
        using namespace function_generator;
        constexpr std::uint64_t lengthSeconds = 10;
        constexpr std::uint32_t N             = 1000 * lengthSeconds;
        constexpr float         sample_rate   = 1'000.f;

        Graph testGraph;
        auto& clockSrc = testGraph.emplaceBlock<gr::basic::ClockSource<std::uint8_t>>({{"sample_rate", sample_rate}, {"n_samples_max", N}, {"name", "ClockSource"}, {"verbose_console", false}});
        initClockSource(clockSrc);

        auto& funcGen = testGraph.emplaceBlock<FunctionGenerator<float>>({{"sample_rate", sample_rate}, {"name", "FunctionGenerator"}});
        initFunctionGenerator(funcGen);

        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "SampleGeneratorSink"}});

        // connect UI sink -- doesn't strictly need to be part of the graph due to BlockingIO<false> definition
        // but the present 'connect' API assumes it to be part of the Graph
        auto& uiSink = testGraph.emplaceBlock<testing::ImChartMonitor<float>>({{"name", "BasicImChartSink"}});
        expect(uiSink.meta_information.value.contains("Drawable")) << "drawable";

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect2(clockSrc, clockSrc.out, funcGen, funcGen.clk_in)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect2(funcGen, funcGen.out, sink, sink.in)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect2(funcGen, funcGen.out, uiSink, uiSink.in)));

        scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }

        std::thread uiLoop([&uiSink]() {
            std::println("start UI thread");
            while (uiSink.draw({{"reset_view", true}}) != work::Status::DONE) { // mocks UI update loop with 25 Hz repetition
                std::this_thread::sleep_for(std::chrono::milliseconds(40));     // 25 Hz <-> 40 ms period
            }
            std::println("asked to finish UI thread");
            std::this_thread::sleep_for(std::chrono::seconds(2)); // wait for another 2 seconds before closing down
            std::println("finished UI thread");
        });
        expect(sched.runAndWait().has_value());
        expect(eq(N, static_cast<std::uint32_t>(sink._samples.size()))) << "Number of samples does not match";
        uiLoop.join();
    };

    "FunctionGenerator + ClockSource FAIR test - Console Debug"_test = [] {
        using namespace function_generator;
        constexpr std::uint64_t lengthSeconds = 10;
        constexpr std::uint32_t N             = 1000 * lengthSeconds;
        constexpr float         sample_rate   = 1'000.f;

        Graph testGraph;
        auto& clockSrc = testGraph.emplaceBlock<gr::basic::ClockSource<std::uint8_t>>({{"sample_rate", sample_rate}, {"n_samples_max", N}, {"name", "ClockSource"}, {"verbose_console", false}});
        initClockSource(clockSrc);

        auto& funcGen = testGraph.emplaceBlock<FunctionGenerator<float>>({{"sample_rate", sample_rate}, {"name", "FunctionGenerator"}});
        initFunctionGenerator(funcGen);

        auto& uiSink = testGraph.emplaceBlock<testing::ImChartMonitor<float, false>>({{"reset_view", true}, {"plot_graph", true}, {"plot_timing", true}, {"timeout_ms", static_cast<gr::Size_t>(400)}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect2(clockSrc, clockSrc.out, funcGen, funcGen.clk_in)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect2(funcGen, funcGen.out, uiSink, uiSink.in)));

        scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
    };
};

int main() { /* not needed for UT */ }
