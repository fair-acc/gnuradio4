#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/basic/FunctionGenerator.hpp>
#include <gnuradio-4.0/basic/SignalGenerator.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

const boost::ut::suite TagTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    static const auto mismatchedKey = [](const property_map& map) {
        std::vector<std::string> keys;
        for (const auto& pair : map) {
            keys.push_back(pair.first);
        }
        return keys;
    };

    auto clockSourceTest = []<bool useIoThreadPool>(bool verbose = false) {
        // useIoThreadPool - true: scheduler/graph-provided thread, false: use user-provided call-back or thread
        if (verbose) {
            fmt::println("started ClockSource test w/ {}", useIoThreadPool ? "Graph/Block<T> provided-thread" : "user-provided thread");
        }
        constexpr gr::Size_t n_samples   = 1900;
        constexpr float      sample_rate = 2000.f;
        Graph                testGraph;
        auto&                src = testGraph.emplaceBlock<ClockSource<std::uint8_t, useIoThreadPool>>({{gr::tag::SAMPLE_RATE.shortKey(), sample_rate}, {"n_samples_max", n_samples}, {"name", "ClockSource"}, {"verbose_console", verbose}});
        src.tags                 = {
            {0, {{"key", "value@0"}}},       //
            {1, {{"key", "value@1"}}},       //
            {100, {{"key", "value@100"}}},   //
            {150, {{"key", "value@150"}}},   //
            {1000, {{"key", "value@1000"}}}, //
            {1001, {{"key", "value@1001"}}}, //
            {1002, {{"key", "value@1002"}}}, //
            {1023, {{"key", "value@1023"}}}  //
        };
        auto& sink1 = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink1"}});
        auto& sink2 = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSink2"}});
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).template to<"in">(sink1)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).template to<"in">(sink2)));

        scheduler::Simple sched{std::move(testGraph)};
        expect(sched.runAndWait().has_value());
        if (verbose) {
            fmt::println("finished ClockSource sched.runAndWait() w/ {}", useIoThreadPool ? "Graph/Block<T> provided-thread" : "user-provided thread");
        }

        expect(eq(src.n_samples_max, n_samples)) << "src did not accept require max input samples";
        expect(eq(src.n_samples_produced, n_samples)) << "src did not produce enough output samples";
        expect(eq(static_cast<gr::Size_t>(sink1._nSamplesProduced), n_samples)) << fmt::format("sink1 did not consume enough input samples ({} vs. {})", sink1._nSamplesProduced, n_samples);
        expect(eq(static_cast<gr::Size_t>(sink2._nSamplesProduced), n_samples)) << fmt::format("sink2 did not consume enough input samples ({} vs. {})", sink2._nSamplesProduced, n_samples);

        // TODO: last decimator/interpolator + stride addition seems to break the limiting the input samples to the min of available vs. n_samples-until next tags
        // expect(equal_tag_lists(src.tags, sink1._tags)) << "sink1 (USE_PROCESS_ONE) did not receive the required tags";
        // expect(equal_tag_lists(src.tags, sink2._tags)) << "sink2 (USE_PROCESS_BULK) did not receive the required tags";
        if (verbose) {
            fmt::println("finished ClockSource test w/ {}", useIoThreadPool ? "Graph/Block<T>-provided thread" : "user-provided thread");
        }
    };

    "ClockSource w/ Graph/Block<T> provided thread"_test = [&clockSourceTest] { clockSourceTest.template operator()<true>(false); };
    // "ClockSource w/ user-provided thread"_test           = [&clockSourceTest] { clockSourceTest.template operator()<false>(false); }; // TODO: check potential threading issue for user-provided threads

    "SignalGenerator test"_test = [] {
        const std::size_t        N      = 16; // test points
        const double             offset = 2.;
        std::vector<std::string> signals{"Const", "Sin", "Cos", "Square", "Saw", "Triangle"};

        for (const auto& sig : signals) {
            SignalGenerator<double> signalGen({{"signal_type", sig}, {gr::tag::SAMPLE_RATE.shortKey(), 2048.f}, {"frequency", 256.}, {"amplitude", 1.}, {"offset", offset}, {"phase", std::numbers::pi / 4}});
            signalGen.init(signalGen.progress, signalGen.ioThreadPool);

            // expected values corresponds to sample_rate = 1024., frequency = 128., amplitude = 1., offset = 0., phase = pi/4.
            std::map<std::string, std ::vector<double>> expResults = {{"Const", {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.}}, {"Sin", {0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0., 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0.}}, {"Cos", {0.707106, 0., -0.707106, -1., -0.7071067, 0., 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0., 0.707106, 1.}}, {"Square", {1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 1.}}, {"Saw", {0.25, 0.5, 0.75, -1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, -1., -0.75, -0.5, -0.25, 0.}}, {"Triangle", {0.5, 1., 0.5, 0., -0.5, -1., -0.5, 0., 0.5, 1., 0.5, 0., -0.5, -1., -0.5, 0.}}};

            for (std::size_t i = 0; i < N; i++) {
                const auto val = signalGen.processOne(0);
                const auto exp = expResults[sig][i] + offset;
                expect(approx(exp, val, 1e-5)) << fmt::format("SignalGenerator for signal: {} and i: {} does not match.", sig, i);
            }
        }
    };

    "SignalGenerator ImChart test"_test = [] {
        const std::size_t        N = 512; // test points
        std::vector<std::string> signals{"Const", "Sin", "Cos", "Square", "Saw", "Triangle"};
        for (const auto& sig : signals) {
            SignalGenerator<double> signalGen({{"signal_type", sig}, {gr::tag::SAMPLE_RATE.shortKey(), 8192.f}, {"frequency", 32.}, {"amplitude", 2.}, {"offset", 0.}, {"phase", std::numbers::pi / 4.}});
            signalGen.init(signalGen.progress, signalGen.ioThreadPool);

            std::vector<double> xValues(N), yValues(N);
            std::iota(xValues.begin(), xValues.end(), 0);
            std::ranges::generate(yValues, [&signalGen]() { return signalGen.processOne(0); });

            fmt::println("Chart {}\n\n", sig);
            auto chart = gr::graphs::ImChart<128, 16>({{0., static_cast<double>(N)}, {-2.6, 2.6}});
            chart.draw(xValues, yValues, sig);
            chart.draw();
        }
    };

    "SignalGenerator + ClockSource test"_test = [] {
        constexpr gr::Size_t n_samples   = 200;
        constexpr float      sample_rate = 1000.f;
        Graph                testGraph;
        auto&                clockSrc  = testGraph.emplaceBlock<ClockSource<std::uint8_t>>({{gr::tag::SAMPLE_RATE.shortKey(), sample_rate}, {"n_samples_max", n_samples}, {"name", "ClockSource"}});
        auto&                signalGen = testGraph.emplaceBlock<SignalGenerator<float>>({{gr::tag::SAMPLE_RATE.shortKey(), sample_rate}, {"name", "SignalGenerator"}});
        auto&                sink      = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink"}, {"verbose_console", true}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(clockSrc).to<"in">(signalGen)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(signalGen).to<"in">(sink)));

        scheduler::Simple sched{std::move(testGraph)};
        expect(sched.runAndWait().has_value());

        expect(eq(n_samples, static_cast<std::uint32_t>(sink._nSamplesProduced))) << "Number of samples does not match";
    };

    "FunctionGenerator ImChart test"_test = [] {
        using namespace function_generator;
        const std::size_t         N          = 128; // test points
        double                    startValue = 10.;
        double                    finalValue = 20.;
        std::vector<SignalType>   signals{Const, LinearRamp, ParabolicRamp, CubicSpline, ImpulseResponse};
        FunctionGenerator<double> funcGen;
        funcGen.init(funcGen.progress, funcGen.ioThreadPool);
        const auto now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

        for (const auto& sig : signals) {
            const property_map params{{createPropertyMapEntry(signal_type, sig), //
                {gr::tag::SAMPLE_RATE.shortKey(), 128.f},                        //
                createPropertyMapEntry(start_value, startValue),                 //
                createPropertyMapEntry(final_value, finalValue),                 //
                createPropertyMapEntry(duration, 1.),                            //
                createPropertyMapEntry(round_off_time, .15),                     //
                createPropertyMapEntry(impulse_time0, .2),                       //
                createPropertyMapEntry(impulse_time1, .15)}};

            expect(funcGen.settings().set(params, SettingsCtx{now, static_cast<int>(sig)}).empty());
        }
        expect(eq(funcGen.settings().getNStoredParameters(), 6UZ)); // +1 for default

        for (const auto& sig : signals) {
            expect(funcGen.settings().activateContext(SettingsCtx{now, static_cast<int>(sig)}) != std::nullopt);
            const auto applyResult = funcGen.settings().applyStagedParameters();
            expect(expect(eq(applyResult.forwardParameters.size(), 6UZ))) << fmt::format("incorrect number of to be forwarded settings. forward keys: {}\n", fmt::join(mismatchedKey(applyResult.forwardParameters), ", "));

            std::vector<double> xValues(N), yValues(N);
            std::iota(xValues.begin(), xValues.end(), 0);
            std::ranges::generate(yValues, [&funcGen]() { return funcGen.processOne(0.); });
            fmt::println("Chart {}\n\n", toString(sig));
            auto chart = gr::graphs::ImChart<128, 32>({{0., static_cast<double>(N)}, {7., 22.}});
            chart.draw(xValues, yValues, toString(sig));
            chart.draw();
        }
    };

    "FunctionGenerator + ClockSource test"_test = [] {
        using namespace std::string_literals;
        using namespace function_generator;
        constexpr std::uint32_t N           = 1000;
        constexpr float         sample_rate = 1000.f;
        Graph                   testGraph;
        auto&                   clockSrc = testGraph.emplaceBlock<ClockSource<std::uint8_t>>({{gr::tag::SAMPLE_RATE.shortKey(), sample_rate}, {"n_samples_max", N}, {"name", "ClockSource"}});
        const auto              now      = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

        clockSrc.tags = {Tag(0, {{tag::CONTEXT.shortKey(), "1"s}}), //
            Tag(100, {{tag::CONTEXT.shortKey(), "2"s}}),            //
            Tag(300, {{tag::CONTEXT.shortKey(), "3"s}}),            //
            Tag(350, {{tag::CONTEXT.shortKey(), "4"s}}),            //
            Tag(550, {{tag::CONTEXT.shortKey(), "5"s}}),            //
            Tag(650, {{tag::CONTEXT.shortKey(), "6"s}}),            //
            Tag(800, {{tag::CONTEXT.shortKey(), "7"s}}),            //
            Tag(850, {{tag::CONTEXT.shortKey(), "8"s}})};

        auto& funcGen = testGraph.emplaceBlock<FunctionGenerator<float>>({{gr::tag::SAMPLE_RATE.shortKey(), sample_rate}, {"name", "FunctionGenerator"}});
        expect(funcGen.settings().set(createConstPropertyMap("", 5.f), SettingsCtx{now, "1"}).empty());
        expect(funcGen.settings().set(createLinearRampPropertyMap("", 5.f, 30.f, .2f), SettingsCtx{now, "2"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("", 30.f), SettingsCtx{now, "3"}).empty());
        expect(funcGen.settings().set(createParabolicRampPropertyMap("", 30.f, 20.f, .1f, 0.02f), SettingsCtx{now, "4"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("", 20.f), SettingsCtx{now, "5"}).empty());
        expect(funcGen.settings().set(createCubicSplinePropertyMap("", 20.f, 10.f, .1f), SettingsCtx{now, "6"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("", 10.f), SettingsCtx{now, "7"}).empty());
        expect(funcGen.settings().set(createImpulseResponsePropertyMap("", 10.f, 20.f, .02f, .06f), SettingsCtx{now, "8"}).empty());

        expect(eq(funcGen.settings().getNStoredParameters(), 9UZ)); // +1 for default

        auto& sink = testGraph.emplaceBlock<gr::testing::TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink"}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(clockSrc).to<"in">(funcGen)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(funcGen).to<"in">(sink)));

        scheduler::Simple sched{std::move(testGraph)};
        expect(sched.runAndWait().has_value());
        expect(eq(N, static_cast<std::uint32_t>(sink._samples.size()))) << "Number of samples does not match";
        expect(eq(sink._tags.size(), clockSrc.tags.size())) << [&]() {
            std::string ret = fmt::format("DataSet nTags: {}\n", sink._tags.size());
            for (const auto& tag : sink._tags) {
                ret += fmt::format("tag.index: {} .map: {}\n", tag.index, tag.map);
            }
            return ret;
        }();

        fmt::println("\n\nChart FunctionGenerator + ClockSource test\n\n");
        std::vector<double> xValues(N);
        std::vector<double> yValues(sink._samples.begin(), sink._samples.end());
        std::iota(xValues.begin(), xValues.end(), 0);
        auto chart = gr::graphs::ImChart<128, 32>({{0., static_cast<double>(N)}, {0., 35.}});
        chart.draw(xValues, yValues, "Signal");
        chart.draw();
    };

    "FunctionGenerator + ClockSource FAIR test"_test = [] {
        using namespace function_generator;
        constexpr std::uint64_t lengthSeconds = 1;
        constexpr std::uint32_t N             = 1000 * lengthSeconds;
        constexpr float         sample_rate   = 1'000.f;

        Graph      testGraph;
        auto&      clockSrc = testGraph.emplaceBlock<ClockSource<std::uint8_t>>({{gr::tag::SAMPLE_RATE.shortKey(), sample_rate}, {"n_samples_max", N}, {"name", "ClockSource"}});
        const auto now      = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

        auto addTimeTagEntry = []<typename T>(ClockSource<T>& clockSource, std::uint64_t timeInNanoseconds, const std::string& value) {
            clockSource.tag_times.value.push_back(timeInNanoseconds);
            clockSource.tag_values.value.push_back(value);
        };

        // all times are in nanoseconds
        constexpr std::uint64_t ms = 1'000'000;                                       // ms -> ns
        addTimeTagEntry(clockSrc, 10 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1"); // <trigger_name>/<ctx>
        addTimeTagEntry(clockSrc, 100 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2");
        addTimeTagEntry(clockSrc, 300 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=3");
        addTimeTagEntry(clockSrc, 350 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4");
        addTimeTagEntry(clockSrc, 550 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=5");
        addTimeTagEntry(clockSrc, 650 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=6");
        addTimeTagEntry(clockSrc, 800 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=7");
        addTimeTagEntry(clockSrc, 850 * ms, "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=8");
        clockSrc.repeat_period      = 5'000 * ms;
        clockSrc.do_zero_order_hold = true;

        auto& funcGen = testGraph.emplaceBlock<FunctionGenerator<float>>({{gr::tag::SAMPLE_RATE.shortKey(), sample_rate}, {"name", "FunctionGenerator"}});
        // all times are in seconds
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 5.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=1"}).empty());
        expect(funcGen.settings().set(createLinearRampPropertyMap("CMD_BP_START", 5.f, 30.f, .2f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=2"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 30.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=3"}).empty());
        expect(funcGen.settings().set(createParabolicRampPropertyMap("CMD_BP_START", 30.f, 20.f, .1f, 0.02f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=4"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 20.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=5"}).empty());
        expect(funcGen.settings().set(createCubicSplinePropertyMap("CMD_BP_START", 20.f, 10.f, .1f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=6"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 10.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=7"}).empty());
        expect(funcGen.settings().set(createImpulseResponsePropertyMap("CMD_BP_START", 10.f, 20.f, .02f, .06f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=8"}).empty());

        expect(eq(funcGen.settings().getNStoredParameters(), 9UZ)); // +1 for default
        auto& sink = testGraph.emplaceBlock<gr::testing::TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink"}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(clockSrc).to<"in">(funcGen)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(funcGen).to<"in">(sink)));

        scheduler::Simple sched{std::move(testGraph)};
        expect(sched.runAndWait().has_value());
        expect(eq(N, static_cast<std::uint32_t>(sink._samples.size()))) << "Number of samples does not match";
        expect(eq(sink._tags.size(), 9UZ)) << [&]() {
            std::string ret = fmt::format("DataSet nTags: {}\n", sink._tags.size());
            for (const auto& tag : sink._tags) {
                ret += fmt::format("tag.index: {} .map: {}\n", tag.index, tag.map);
            }
            return ret;
        }();

        fmt::println("\n\nFunctionGenerator + ClockSource FAIR test\n\n");
        std::vector<double> xValues(N);
        std::vector<double> yValues(sink._samples.begin(), sink._samples.end());
        std::iota(xValues.begin(), xValues.end(), 0);
        auto chart = gr::graphs::ImChart<128, 32>({{0., static_cast<double>(N)}, {0., 35.}});
        chart.draw(xValues, yValues, "Signal");
        chart.draw();
    };
};

int main() { /* not needed for UT */ }
