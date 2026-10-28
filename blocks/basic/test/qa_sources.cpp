#include <array>
#include <boost/ut.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <gnuradio-4.0/basic/FunctionGeneratorCore.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/basic/FunctionGenerator.hpp>
#include <gnuradio-4.0/basic/SignalGenerator.hpp>
#include <gnuradio-4.0/basic/SpeechShapedNoise.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

const boost::ut::suite TagTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    static const auto mismatchedKey = [](const property_map& map) {
        Tensor<Value> keys;
        for (const auto& pair : map) {
            keys.push_back(pair.first);
        }
        return keys;
    };

    auto clockSourceTest = []<bool useIoThreadPool>(bool verbose = false) {
        // useIoThreadPool - true: scheduler/graph-provided thread, false: use user-provided call-back or thread
        if (verbose) {
            std::println("started ClockSource test w/ {}", useIoThreadPool ? "Graph/Block<T> provided-thread" : "user-provided thread");
        }
        constexpr gr::Size_t n_samples   = 1900;
        constexpr float      sample_rate = 2000.f;
        Graph                testGraph;
        auto&                src = testGraph.emplaceBlock<ClockSource<std::uint8_t>>({{gr::tag::kSampleRate, sample_rate}, {"n_samples_max", n_samples}, {"name", "ClockSource"}, {"verbose_console", verbose}});
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
        expect(testGraph.connect<"out", "in">(src, sink1).has_value());
        expect(testGraph.connect<"out", "in">(src, sink2).has_value());

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        if (verbose) {
            std::println("finished ClockSource sched.runAndWait() w/ {}", useIoThreadPool ? "Graph/Block<T> provided-thread" : "user-provided thread");
        }

        expect(eq(src.n_samples_max, n_samples)) << "src did not accept require max input samples";
        expect(eq(src.n_samples_produced, n_samples)) << "src did not produce enough output samples";
        expect(eq(static_cast<gr::Size_t>(sink1._nSamplesProduced), n_samples)) << std::format("sink1 did not consume enough input samples ({} vs. {})", sink1._nSamplesProduced, n_samples);
        expect(eq(static_cast<gr::Size_t>(sink2._nSamplesProduced), n_samples)) << std::format("sink2 did not consume enough input samples ({} vs. {})", sink2._nSamplesProduced, n_samples);

        // TODO: last decimator/interpolator + stride addition seems to break the limiting the input samples to the min of available vs. n_samples-until next tags
        // expect(equal_tag_lists(src.tags, sink1._tags)) << "sink1 (USE_PROCESS_ONE) did not receive the required tags";
        // expect(equal_tag_lists(src.tags, sink2._tags)) << "sink2 (USE_PROCESS_BULK) did not receive the required tags";
        if (verbose) {
            std::println("finished ClockSource test w/ {}", useIoThreadPool ? "Graph/Block<T>-provided thread" : "user-provided thread");
        }
    };

    "ClockSource w/ Graph/Block<T> provided thread"_test = [&clockSourceTest] { clockSourceTest.template operator()<true>(false); };
    // "ClockSource w/ user-provided thread"_test           = [&clockSourceTest] { clockSourceTest.template operator()<false>(false); }; // TODO: check potential threading issue for user-provided threads

    "SignalGenerator test"_test = [] {
        const std::size_t        N      = 16; // test points
        const double             offset = 2.;
        std::vector<std::string> signals{"Const", "Sin", "Cos", "Square", "Saw", "Triangle"};

        for (const auto& sig : signals) {
            SignalGenerator<double> signalGen({{"signal_type", sig}, {gr::tag::kSampleRate, 2048.f}, {"frequency", 256.}, {"amplitude", 1.}, {"offset", offset}, {"phase", std::numbers::pi / 4}});
            signalGen.init(signalGen.progress);

            // expected values corresponds to sample_rate = 1024., frequency = 128., amplitude = 1., offset = 0., phase = pi/4.
            std::map<std::string, std ::vector<double>> expResults = {{"Const", {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.}}, {"Sin", {0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0., 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0.}}, {"Cos", {0.707106, 0., -0.707106, -1., -0.7071067, 0., 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0., 0.707106, 1.}}, {"Square", {1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 1.}}, {"Saw", {0.25, 0.5, 0.75, -1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, -1., -0.75, -0.5, -0.25, 0.}}, {"Triangle", {0.5, 1., 0.5, 0., -0.5, -1., -0.5, 0., 0.5, 1., 0.5, 0., -0.5, -1., -0.5, 0.}}};

            for (std::size_t i = 0; i < N; i++) {
                const auto val = signalGen.generateSample();
                const auto exp = expResults[sig][i] + offset;
                expect(approx(exp, val, 1e-5)) << std::format("SignalGenerator for signal: {} and i: {} does not match.", sig, i);
            }
        }
    };

    "SignalGenerator FastSin/FastCos match Sin/Cos"_test = [] {
        constexpr std::size_t N = 200;
        for (const auto& [fast, precise] : std::vector<std::pair<std::string, std::string>>{{"FastSin", "Sin"}, {"FastCos", "Cos"}}) {
            SignalGenerator<double> genFast({{"signal_type", fast}, {gr::tag::kSampleRate, 1000.f}, {"frequency", 50.f}, {"amplitude", 1.f}, {"offset", 0.f}, {"phase", 0.f}});
            SignalGenerator<double> genPrecise({{"signal_type", precise}, {gr::tag::kSampleRate, 1000.f}, {"frequency", 50.f}, {"amplitude", 1.f}, {"offset", 0.f}, {"phase", 0.f}});
            genFast.init(genFast.progress);
            genPrecise.init(genPrecise.progress);
            for (std::size_t i = 0; i < N; ++i) {
                expect(approx(genPrecise.generateSample(), genFast.generateSample(), 1e-10)) << std::format("{} vs {} at sample {}", fast, precise, i);
            }
        }
    };

    "SignalGenerator noise types produce bounded output"_test = [] {
        constexpr std::size_t N = 10000;
        for (const auto& noiseType : {"UniformNoise", "TriangularNoise", "GaussianNoise"}) {
            SignalGenerator<double> gen({{"signal_type", std::string(noiseType)}, {gr::tag::kSampleRate, 1000.f}, {"amplitude", 2.f}, {"offset", 1.f}, {"seed", std::uint64_t(42)}});
            gen.init(gen.progress);
            double sum = 0.;
            for (std::size_t i = 0; i < N; ++i) {
                const double v = gen.generateSample();
                sum += v;
                if (std::string_view(noiseType) != "GaussianNoise") {
                    expect(ge(v, -2. + 1.)) << std::format("{} sample {} out of range: {}", noiseType, i, v);
                    expect(le(v, 2. + 1.)) << std::format("{} sample {} out of range: {}", noiseType, i, v);
                }
            }
            const double mean = sum / static_cast<double>(N);
            expect(approx(1., mean, 0.2)) << std::format("{} mean {} far from offset 1.0", noiseType, mean);
        }
    };

    "SignalGenerator integer output clamps correctly"_test = [] {
        SignalGenerator<std::int16_t> gen({{"signal_type", "Sin"}, {gr::tag::kSampleRate, 1000.f}, {"frequency", 50.f}, {"amplitude", 40000.f}, {"offset", 0.f}});
        gen.init(gen.progress);
        bool hitMax = false, hitMin = false;
        for (std::size_t i = 0; i < 100; ++i) {
            const auto v = gen.generateSample();
            expect(ge(v, std::numeric_limits<std::int16_t>::min()));
            expect(le(v, std::numeric_limits<std::int16_t>::max()));
            if (v == std::numeric_limits<std::int16_t>::max()) {
                hitMax = true;
            }
            if (v == std::numeric_limits<std::int16_t>::min()) {
                hitMin = true;
            }
        }
        expect(hitMax) << "amplitude 40000 should clamp to int16 max";
        expect(hitMin) << "amplitude 40000 should clamp to int16 min";
    };

    "SignalGenerator complex output produces analytic signal"_test = [] {
        SignalGenerator<std::complex<double>> gen({{"signal_type", "Sin"}, {gr::tag::kSampleRate, 1000.f}, {"frequency", 50.f}, {"amplitude", 1.f}, {"offset", 0.f}, {"phase", 0.f}});
        gen.init(gen.progress);
        for (std::size_t i = 0; i < 50; ++i) {
            const auto   sample = gen.generateSample();
            const double mag    = std::abs(sample);
            expect(approx(1., mag, 1e-10)) << std::format("analytic signal magnitude at sample {} should be ~1.0, got {}", i, mag);
        }
    };

    "SignalGenerator ImChart test"_test = [] {
        const std::size_t        N = 512; // test points
        std::vector<std::string> signals{"Const", "Sin", "Cos", "Square", "Saw", "Triangle"};
        for (const auto& sig : signals) {
            SignalGenerator<double> signalGen({{"signal_type", sig}, {gr::tag::kSampleRate, 8192.f}, {"frequency", 32.}, {"amplitude", 2.}, {"offset", 0.}, {"phase", std::numbers::pi / 4.}});
            signalGen.init(signalGen.progress);

            std::vector<double> xValues(N), yValues(N);
            std::iota(xValues.begin(), xValues.end(), 0);
            std::ranges::generate(yValues, [&signalGen]() { return signalGen.generateSample(); });

            std::println("Chart {}\n\n", sig);
            auto chart = gr::graphs::ImChart<128, 16>({{0., static_cast<double>(N)}, {-2.6, 2.6}});
            chart.draw(xValues, yValues, sig);
            chart.draw();
        }
    };

    "SignalGenerator + ClockSource test"_test = [] {
        constexpr gr::Size_t n_samples   = 200;
        constexpr float      sample_rate = 1000.f;
        Graph                testGraph;
        auto&                clockSrc  = testGraph.emplaceBlock<ClockSource<std::uint8_t>>({{gr::tag::kSampleRate, sample_rate}, {"n_samples_max", n_samples}, {"name", "ClockSource"}});
        auto&                signalGen = testGraph.emplaceBlock<SignalGenerator<float>>({{gr::tag::kSampleRate, sample_rate}, {"name", "SignalGenerator"}});
        auto&                sink      = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink"}, {"verbose_console", true}});

        expect(testGraph.connect<"out", "clk">(clockSrc, signalGen).has_value());
        expect(testGraph.connect<"out", "in">(signalGen, sink).has_value());

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
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
        funcGen.init(funcGen.progress);
        const auto now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

        for (const auto& sig : signals) {
            const property_map params{{createPropertyMapEntry(signal_type, sig), //
                {gr::tag::kSampleRate, 128.f},                                   //
                createPropertyMapEntry(start_value, startValue),                 //
                createPropertyMapEntry(final_value, finalValue),                 //
                createPropertyMapEntry(duration, 1.),                            //
                createPropertyMapEntry(round_off_time, .15),                     //
                createPropertyMapEntry(impulse_time0, .2),                       //
                createPropertyMapEntry(impulse_time1, .15)}};

            expect(funcGen.settings().set(params, SettingsCtx{now, std::format("{}", static_cast<int>(sig))}).empty());
        }
        expect(eq(funcGen.settings().getNStoredParameters(), 6UZ)); // +1 for default

        for (const auto& sig : signals) {
            expect(funcGen.settings().activateContext(SettingsCtx{now, std::format("{}", static_cast<int>(sig))}) != std::nullopt);
            const auto applyResult = funcGen.settings().applyStagedParameters();
            expect(expect(eq(applyResult.forwardParameters.size(), 7UZ))) << std::format("incorrect number of to be forwarded settings. forward keys: {}\n", gr::join(mismatchedKey(applyResult.forwardParameters), ", "));

            std::vector<double> xValues(N), yValues(N);
            std::iota(xValues.begin(), xValues.end(), 0);
            std::ranges::generate(yValues, [&funcGen]() { return funcGen.generateSample(); });
            std::println("Chart {}\n\n", std::string_view(toString(sig)));
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
        auto&                   clockSrc = testGraph.emplaceBlock<ClockSource<std::uint8_t>>({{gr::tag::kSampleRate, sample_rate}, {"n_samples_max", N}, {"name", "ClockSource"}});
        const auto              now      = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

        clockSrc.tags = {{0, tag::kContext("1"s)}, //
            {100, tag::kContext("2"s)},            //
            {300, tag::kContext("3"s)},            //
            {350, tag::kContext("4"s)},            //
            {550, tag::kContext("5"s)},            //
            {650, tag::kContext("6"s)},            //
            {800, tag::kContext("7"s)},            //
            {850, tag::kContext("8"s)}};

        auto& funcGen = testGraph.emplaceBlock<FunctionGenerator<float>>({{gr::tag::kSampleRate, sample_rate}, {"name", "FunctionGenerator"}});
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

        expect(testGraph.connect<"out", "clk">(clockSrc, funcGen).has_value());
        expect(testGraph.connect<"out", "in">(funcGen, sink).has_value());

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        expect(eq(N, static_cast<std::uint32_t>(sink._samples.size()))) << "Number of samples does not match";
        expect(eq(sink._tags.size(), clockSrc.tags.size())) << [&]() {
            std::string ret = std::format("DataSet nTags: {}\n", sink._tags.size());
            for (const auto& tag : sink._tags) {
                ret += std::format("tag.index: {} .map: {}\n", tag.index, tag.map);
            }
            return ret;
        }();
        expect(!sink._tags.empty()) << fatal;
        const auto& first       = sink._tags.front();
        const bool  bothOnFirst = first.map.contains(tag::kContext.key()) && first.map.contains(tag::kSampleRate.key());
        expect(eq(first.index, 0UZ) && bothOnFirst) << "context 1 and the forwarded sample_rate share sample 0 and arrive as one tag";

        std::println("\n\nChart FunctionGenerator + ClockSource test\n\n");
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
        auto&      clockSrc = testGraph.emplaceBlock<ClockSource<std::uint8_t>>({{gr::tag::kSampleRate, sample_rate}, {"n_samples_max", N}, {"name", "ClockSource"}});
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

        auto& funcGen = testGraph.emplaceBlock<FunctionGenerator<float>>({{gr::tag::kSampleRate, sample_rate}, {"name", "FunctionGenerator"}});
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

        expect(testGraph.connect<"out", "clk">(clockSrc, funcGen).has_value());
        expect(testGraph.connect<"out", "in">(funcGen, sink).has_value());

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        expect(eq(N, static_cast<std::uint32_t>(sink._samples.size()))) << "Number of samples does not match";
        expect(ge(sink._tags.size(), 8UZ)) << [&]() {
            std::string ret = std::format("DataSet nTags: {}\n", sink._tags.size());
            for (const auto& tag : sink._tags) {
                ret += std::format("tag.index: {} .map: {}\n", tag.index, tag.map);
            }
            return ret;
        }();

        std::println("\n\nFunctionGenerator + ClockSource FAIR test\n\n");
        std::vector<double> xValues(N);
        std::vector<double> yValues(sink._samples.begin(), sink._samples.end());
        std::iota(xValues.begin(), xValues.end(), 0);
        auto chart = gr::graphs::ImChart<128, 32>({{0., static_cast<double>(N)}, {0., 35.}});
        chart.draw(xValues, yValues, "Signal");
        chart.draw();
    };

    "ClockSource repeat_period emits tags across cycles"_test = [] {
        constexpr std::uint64_t ms          = 1'000'000; // ns
        constexpr float         sample_rate = 1'000.f;
        constexpr std::uint64_t repeatNs    = 500 * ms; // 500ms = 500 samples per cycle
        constexpr gr::Size_t    nCycles     = 3;
        constexpr gr::Size_t    N           = nCycles * 500; // 1500 samples

        Graph testGraph;
        auto& clockSrc = testGraph.emplaceBlock<ClockSource<std::uint8_t>>({{gr::tag::kSampleRate, sample_rate}, {"n_samples_max", N}, {"name", "ClockSource"}});

        auto addTimeTagEntry = []<typename T>(ClockSource<T>& clockSource, std::uint64_t timeInNanoseconds, const std::string& value) {
            clockSource.tag_times.value.push_back(timeInNanoseconds);
            clockSource.tag_values.value.push_back(value);
        };
        addTimeTagEntry(clockSrc, 10 * ms, "CMD_BP_START/ctx1");
        addTimeTagEntry(clockSrc, 100 * ms, "CMD_BP_START/ctx2");
        addTimeTagEntry(clockSrc, 200 * ms, "CMD_BP_START/ctx3");
        addTimeTagEntry(clockSrc, 400 * ms, "CMD_BP_START/ctx4");
        clockSrc.repeat_period = repeatNs;

        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink"}});
        expect(testGraph.connect<"out", "in">(clockSrc, sink).has_value());

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        expect(eq(N, static_cast<gr::Size_t>(sink._nSamplesProduced))) << "sample count mismatch";

        constexpr std::size_t tagsPerCycle = 4;
        expect(ge(sink._tags.size(), nCycles * tagsPerCycle)) << [&]() {
            std::string ret = std::format("expected >= {} tags ({}x{}), got {}:\n", nCycles * tagsPerCycle, nCycles, tagsPerCycle, sink._tags.size());
            for (const auto& tag : sink._tags) {
                ret += std::format("  tag.index: {} .map: {}\n", tag.index, tag.map);
            }
            return ret;
        }();

        // verify tags appear in each cycle (each cycle = 500 samples)
        for (gr::Size_t cycle = 0; cycle < nCycles; ++cycle) {
            const auto cycleStart   = cycle * 500UZ;
            const auto cycleEnd     = cycleStart + 500UZ;
            const auto nTagsInCycle = static_cast<std::size_t>(std::ranges::count_if(sink._tags, [&](const auto& tag) { return tag.index >= cycleStart && tag.index < cycleEnd; }));
            expect(ge(nTagsInCycle, tagsPerCycle)) << std::format("cycle {}: expected >= {} tags in [{}, {}), got {}", cycle, tagsPerCycle, cycleStart, cycleEnd, nTagsInCycle);
        }
    };

    "ClockSource stamps tags from start_time"_test = [] {
        constexpr std::uint64_t ms        = 1'000'000;                    // ns
        constexpr std::uint64_t startTime = 1'767'225'600'000'000'000ULL; // 2026-01-01T00:00:00Z

        Graph testGraph;
        auto& clockSrc = testGraph.emplaceBlock<ClockSource<std::uint8_t>>({{gr::tag::kSampleRate, 2'000.f}, {"n_samples_max", gr::Size_t{400}}, {"start_time", startTime}});
        for (std::uint64_t tagTime : {10 * ms, 50 * ms, 150 * ms}) {
            clockSrc.tag_times.value.push_back(tagTime);
        }
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink"}});
        expect(testGraph.connect<"out", "in">(clockSrc, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        std::vector<std::uint64_t> triggerTimes;
        std::vector<std::size_t>   triggerIndices;
        for (const auto& tag : sink._tags) {
            if (tag.map.contains(gr::tag::kTriggerTime)) {
                triggerTimes.push_back(tag.map.value_or<std::uint64_t>(gr::tag::kTriggerTime, 0U));
                triggerIndices.push_back(static_cast<std::size_t>(tag.index));
            }
        }
        expect(eq(triggerTimes, std::vector<std::uint64_t>{startTime, startTime + 10 * ms, startTime + 50 * ms, startTime + 150 * ms}));
        expect(eq(triggerIndices, std::vector<std::size_t>{0UZ, 20UZ, 100UZ, 300UZ})) << "a sample rate set through its tag key paces the clock at 2 kHz";
    };

    "FunctionGenerator tone types produce correct waveform"_test = [] {
        using namespace function_generator;
        constexpr float sampleRate = 1000.f;
        constexpr float freq       = 50.f;
        constexpr float amplitude  = 3.f;
        constexpr float offset     = 1.f;
        constexpr float phase      = 0.f;

        for (const auto& toneType : {Sin, Cos, FastSin, FastCos}) {
            FunctionGenerator<double> funcGen({tag::kSampleRate(sampleRate)});
            funcGen.init(funcGen.progress);
            const auto now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

            property_map params{createPropertyMapEntry(signal_type, toneType), createPropertyMapEntry(ParameterType::frequency, freq), createPropertyMapEntry(final_value, amplitude), createPropertyMapEntry(start_value, offset), createPropertyMapEntry(ParameterType::phase, phase), createPropertyMapEntry(duration, 0.f)};
            expect(funcGen.settings().set(params, SettingsCtx{now, "1"}).empty());
            expect(funcGen.settings().activateContext(SettingsCtx{now, "1"}) != std::nullopt);
            std::ignore = funcGen.settings().applyStagedParameters();

            constexpr std::size_t N     = 20; // one full period at 50 Hz / 1000 Hz
            double                sumSq = 0.;
            for (std::size_t i = 0; i < N; ++i) {
                const double v = funcGen.generateSample();
                sumSq += (v - static_cast<double>(offset)) * (v - static_cast<double>(offset));
            }
            const double rms = std::sqrt(sumSq / static_cast<double>(N));
            expect(approx(static_cast<double>(amplitude) / std::numbers::sqrt2, rms, 0.15)) << std::format("tone {} RMS mismatch", toString(toneType));
        }
    };

    "FunctionGenerator tone duration expiry holds at offset"_test = [] {
        using namespace function_generator;
        constexpr float sampleRate = 1000.f;

        FunctionGenerator<double> funcGen({tag::kSampleRate(sampleRate)});
        funcGen.init(funcGen.progress);
        const auto now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

        property_map params{createPropertyMapEntry(signal_type, Sin), createPropertyMapEntry(ParameterType::frequency, 100.f), createPropertyMapEntry(final_value, 5.f), // amplitude
            createPropertyMapEntry(start_value, 2.f),                                                                                                                    // offset
            createPropertyMapEntry(ParameterType::phase, 0.f), createPropertyMapEntry(duration, 0.01f)};                                                                 // 10 ms = 10 samples
        expect(funcGen.settings().set(params, SettingsCtx{now, "1"}).empty());
        expect(funcGen.settings().activateContext(SettingsCtx{now, "1"}) != std::nullopt);
        std::ignore = funcGen.settings().applyStagedParameters();

        // generate past the duration
        for (std::size_t i = 0; i < 15; ++i) {
            std::ignore = funcGen.generateSample();
        }
        // after expiry, output should hold at start_value (offset = 2.0)
        for (std::size_t i = 0; i < 10; ++i) {
            expect(approx(2., funcGen.generateSample(), 1e-10)) << std::format("sample {} after expiry should be offset", i);
        }
    };

    "FunctionGenerator noise types produce output"_test = [] {
        using namespace function_generator;
        constexpr float sampleRate = 1000.f;

        for (const auto& noiseType : {UniformNoise, TriangularNoise, GaussianNoise}) {
            FunctionGenerator<double> funcGen({tag::kSampleRate(sampleRate)});
            funcGen.init(funcGen.progress);
            const auto now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

            property_map params{createPropertyMapEntry(signal_type, noiseType), createPropertyMapEntry(start_value, 2.f)};
            expect(funcGen.settings().set(params, SettingsCtx{now, "1"}).empty());
            expect(funcGen.settings().activateContext(SettingsCtx{now, "1"}) != std::nullopt);
            std::ignore = funcGen.settings().applyStagedParameters();

            bool nonZero = false;
            for (std::size_t i = 0; i < 100; ++i) {
                if (funcGen.generateSample() != 0.) {
                    nonZero = true;
                }
            }
            expect(nonZero) << std::format("noise type {} should produce non-zero output", toString(noiseType));
        }
    };

    "FunctionGenerator complex output for tone type"_test = [] {
        using namespace function_generator;
        FunctionGenerator<std::complex<double>> funcGen({tag::kSampleRate(1000.f)});
        funcGen.init(funcGen.progress);
        const auto now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

        property_map params{createPropertyMapEntry(signal_type, Sin), createPropertyMapEntry(ParameterType::frequency, 50.f), createPropertyMapEntry(final_value, 1.f), createPropertyMapEntry(start_value, 0.f), createPropertyMapEntry(ParameterType::phase, 0.f), createPropertyMapEntry(duration, 0.f)};
        expect(funcGen.settings().set(params, SettingsCtx{now, "1"}).empty());
        expect(funcGen.settings().activateContext(SettingsCtx{now, "1"}) != std::nullopt);
        std::ignore = funcGen.settings().applyStagedParameters();

        for (std::size_t i = 0; i < 20; ++i) {
            const auto   sample = funcGen.generateSample();
            const double mag    = std::abs(sample);
            expect(approx(1., mag, 1e-10)) << std::format("analytic signal magnitude at sample {} should be ~1.0, got {}", i, mag);
        }
    };

    "FunctionGenerator integer output for tone type"_test = [] {
        using namespace function_generator;
        FunctionGenerator<std::int16_t> funcGen({tag::kSampleRate(1000.f)});
        funcGen.init(funcGen.progress);
        const auto now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

        property_map params{createPropertyMapEntry(signal_type, Sin), createPropertyMapEntry(ParameterType::frequency, 50.f), createPropertyMapEntry(final_value, 40000.f), createPropertyMapEntry(start_value, 0.f), createPropertyMapEntry(ParameterType::phase, 0.f), createPropertyMapEntry(duration, 0.f)};
        expect(funcGen.settings().set(params, SettingsCtx{now, "1"}).empty());
        expect(funcGen.settings().activateContext(SettingsCtx{now, "1"}) != std::nullopt);
        std::ignore = funcGen.settings().applyStagedParameters();

        bool hitMax = false, hitMin = false;
        for (std::size_t i = 0; i < 100; ++i) {
            const auto v = funcGen.generateSample();
            expect(ge(v, std::numeric_limits<std::int16_t>::min()));
            expect(le(v, std::numeric_limits<std::int16_t>::max()));
            if (v == std::numeric_limits<std::int16_t>::max()) {
                hitMax = true;
            }
            if (v == std::numeric_limits<std::int16_t>::min()) {
                hitMin = true;
            }
        }
        expect(hitMax) << "amplitude 40000 should clamp to int16 max";
        expect(hitMin) << "amplitude 40000 should clamp to int16 min";
    };

    "FunctionGenerator + ClockSource tone segment test"_test = [] {
        using namespace std::string_literals;
        using namespace function_generator;
        constexpr std::uint32_t N           = 500;
        constexpr float         sample_rate = 1000.f;
        Graph                   testGraph;
        auto&                   clockSrc = testGraph.emplaceBlock<ClockSource<std::uint8_t>>({{gr::tag::kSampleRate, sample_rate}, {"n_samples_max", N}, {"name", "ClockSource"}});
        const auto              now      = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

        clockSrc.tags = {{0, tag::kContext("1"s)}, {100, tag::kContext("2"s)}, {300, tag::kContext("3"s)}};

        auto& funcGen = testGraph.emplaceBlock<FunctionGenerator<float>>({{gr::tag::kSampleRate, sample_rate}, {"name", "FunctionGenerator"}});
        expect(funcGen.settings().set(createConstPropertyMap("", 0.f), SettingsCtx{now, "1"}).empty());
        expect(funcGen.settings().set(createSinPropertyMap("", 50.f, 5.f, 0.f, 1.f, 0.f), SettingsCtx{now, "2"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("", 1.f), SettingsCtx{now, "3"}).empty());

        auto& sink = testGraph.emplaceBlock<gr::testing::TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink"}});
        expect(testGraph.connect<"out", "clk">(clockSrc, funcGen).has_value());
        expect(testGraph.connect<"out", "in">(funcGen, sink).has_value());

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        expect(eq(N, static_cast<std::uint32_t>(sink._samples.size()))) << "number of samples does not match";

        // segment 1 (samples 0–99): Const = 0
        for (std::size_t i = 0; i < 100 && i < sink._samples.size(); ++i) {
            expect(approx(0.f, sink._samples[i], 1e-4f)) << std::format("const segment sample {}", i);
        }
        // segment 2 (samples 100–299): Sin with amplitude=5, offset=1 -> oscillating around 1.0
        double sumSq = 0.;
        for (std::size_t i = 100; i < 300 && i < sink._samples.size(); ++i) {
            sumSq += static_cast<double>((sink._samples[i] - 1.f) * (sink._samples[i] - 1.f));
        }
        const double rms = std::sqrt(sumSq / 200.);
        expect(approx(5. / std::numbers::sqrt2, rms, 0.3)) << std::format("sine segment RMS mismatch: {}", rms);
        // segment 3 (samples 300+): Const = 1
        for (std::size_t i = 300; i < sink._samples.size(); ++i) {
            expect(approx(1.f, sink._samples[i], 1e-4f)) << std::format("const-after-sine segment sample {}", i);
        }
    };
};

namespace {
struct SourceFaultContext : gr::device::DeviceContext {
    enum class Fault { STATE_ALLOCATION, OUTPUT_ALLOCATION, KERNEL, TRANSFER, POISONED, FALLBACK_TRANSFER };

    gr::device::DeviceContext& backing;
    Fault                      fault;
    std::size_t                allocations      = 0UZ;
    std::size_t                copies           = 0UZ;
    std::size_t                released         = 0UZ;
    std::size_t                poisonedReleases = 0UZ;
    bool                       poisoned         = false;

    std::vector<gr::device::DeviceBuffer> _owned;

    SourceFaultContext(gr::device::DeviceContext& context, Fault injected) : backing(context), fault(injected), poisoned(injected == Fault::POISONED) {}
    ~SourceFaultContext() override {
        for (const auto buffer : _owned) {
            backing.deallocate(buffer);
        }
    }

    gr::device::DeviceBackend  backend() const noexcept override { return gr::device::DeviceBackend::CPU_Fallback; }
    gr::device::DeviceType     deviceType() const noexcept override { return gr::device::DeviceType::CPU; }
    std::string                shortName() const override { return "source fault injection"; }
    std::string                name() const override { return shortName(); }
    std::string                version() const override { return "test"; }
    void                       wait() override { poisoned = poisoned || fault == Fault::KERNEL; }
    std::optional<std::string> peekDeviceError() noexcept override { return poisoned ? std::optional<std::string>{"injected source failure"} : backing.peekDeviceError(); }
    std::optional<std::string> pollDeviceError() noexcept override { return peekDeviceError(); }
    gr::device::DeviceBuffer   allocate(std::size_t bytes, std::size_t align, gr::device::Residency wanted) override {
        ++allocations;
        if (((fault == Fault::STATE_ALLOCATION || fault == Fault::FALLBACK_TRANSFER) && allocations == 1UZ) || (fault == Fault::OUTPUT_ALLOCATION && allocations == 2UZ)) {
            return {};
        }
        const auto buffer = backing.allocate(bytes, align, wanted);
        if (buffer) {
            _owned.push_back(buffer);
        }
        return buffer;
    }
    void deallocate(gr::device::DeviceBuffer buffer) override {
        if (poisoned) {
            ++poisonedReleases;
            return;
        }
        released += buffer ? 1UZ : 0UZ;
        backing.deallocate(buffer);
        std::erase_if(_owned, [buffer](const auto& owned) { return owned.token == buffer.token; });
    }
    void upload(const void* host, gr::device::DeviceBuffer destination, std::size_t bytes) override { backing.upload(host, destination, bytes); }
    void download(gr::device::DeviceBuffer source, void* host, std::size_t bytes) override { backing.download(source, host, bytes); }
    void copy(void* destination, const void* source, std::size_t bytes, bool await = true) override {
        ++copies;
        if (fault == Fault::TRANSFER || fault == Fault::FALLBACK_TRANSFER) {
            poisoned = true;
            return;
        }
        backing.copy(destination, source, bytes, await);
    }
};
} // namespace
const boost::ut::suite<"DeviceSignalSources"> deviceSignalSourceTests = [] {
    using namespace boost::ut;
    using gr::testing::operator""_domain_test;

    "signal waveforms retain their sequence across empty and odd device chunks"_domain_test = [](gr::testing::DomainContext& domain) {
        const auto checkSequence = [&]<typename T>() {
            gr::PortOut<std::uint8_t>     clock;
            gr::basic::SignalGenerator<T> source;
            gr::PortIn<T>                 sink;
            expect(clock.connect(source.clk).has_value());
            expect(source.out.connect(sink).has_value());
            source.init(source.progress);
            source.start();
            const gr::on_scope_exit stopSource{[&] { source.stop(); }};

            for (gr::signal::SignalType type : magic_enum::enum_values<gr::signal::SignalType>()) {
                expect(source.settings().set({{"signal_type", std::string(magic_enum::enum_name(type))}, {"frequency", 37.0}, {"sample_rate", 1000.f}, {"phase", 0.25f}, {"amplitude", 3.f}, {"offset", 1.f}, {"seed", std::uint64_t{42U}}}).empty());
                std::ignore = source.settings().activateContext();
                std::ignore = source.settings().applyStagedParameters();
                source._core.reset();
                gr::signal::SignalGeneratorCore<T> reference;
                reference.configure(type, 37.0, 1000.f, 0.25f, 3.f, 1.f, 42U);
                reference.reset();
                for (const std::size_t count : {0UZ, 1UZ, 7UZ, 0UZ, 31UZ, 1024UZ, 1UZ}) {
                    if (count != 0UZ) {
                        auto ticks = clock.tryReserve<gr::SpanReleasePolicy::ProcessAll>(count);
                        ticks.publish(count);
                    }
                    {
                        auto       input  = source.clk.template get<gr::SpanReleasePolicy::ProcessAll>(count);
                        auto       output = source.out.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(count);
                        const auto status = source.processBulk(domain.context(), input, output);
                        expect(status == (count == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK));
                    }
                    auto actual = sink.template get<gr::SpanReleasePolicy::ProcessAll>(count);
                    expect(eq(actual.size(), count));
                    for (const T sample : actual) {
                        const T expected = reference.generateSample();
                        if constexpr (std::integral<T>) {
                            expect(eq(sample, expected)) << magic_enum::enum_name(type);
                        } else {
                            expect(le(std::abs(sample - expected), 1.e-5)) << magic_enum::enum_name(type);
                        }
                    }
                }
            }
        };
        checkSequence.template operator()<std::uint8_t>();
        checkSequence.template operator()<std::uint16_t>();
        checkSequence.template operator()<std::uint32_t>();
        checkSequence.template operator()<std::uint64_t>();
        checkSequence.template operator()<std::int8_t>();
        checkSequence.template operator()<std::int16_t>();
        checkSequence.template operator()<std::int32_t>();
        checkSequence.template operator()<std::int64_t>();
        checkSequence.template operator()<float>();
        checkSequence.template operator()<double>();
        checkSequence.template operator()<std::complex<float>>();
        checkSequence.template operator()<std::complex<double>>();
    } | gr::testing::kAllDomains;

    "a clocked device signal follows the analytic sine and forwards clock tags"_domain_test = [](auto& fixture, std::string_view domain) {
        auto& clock            = fixture.template emplace<gr::testing::TagSource<std::uint8_t>>({{"n_samples_max", gr::Size_t{33U}}, {"sample_rate", 1024.f}});
        clock._tags            = {{7U, {{"marker", 42}}}};
        auto& source           = fixture.template emplace<gr::basic::SignalGenerator<double>>({{"gr:compute_domain", domain == "host" ? std::string(domain) : std::string(domain) + "!"}, {"sample_rate", 1024.f}, {"frequency", 64.f}, {"amplitude", 2.f}, {"offset", 1.f}});
        source.out.max_samples = 7UZ;
        auto& sink             = fixture.template emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.template connect<"out", "clk">(clock, source).has_value());
        expect(fixture.template connect<"out", "in">(source, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink._samples.size(), 33UZ));
        for (std::size_t i = 0UZ; i < sink._samples.size(); ++i) {
            const double expected = 2.0 * std::sin(std::numbers::pi * static_cast<double>(i) / 8.0) + 1.0;
            expect(approx(sink._samples[i], expected, 1.e-12)) << domain << i;
        }
        expect(std::ranges::any_of(sink._tags, [](const auto& recorded) { return recorded.index == 7U && recorded.map.template value_or<int>("marker", 0) == 42; }));
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a non-repeating linear chirp follows its analytic phase and then holds its offset"_domain_test = [](auto& fixture, std::string_view domain) {
        auto& clock  = fixture.template emplace<gr::testing::TagSource<std::uint8_t>>({{"n_samples_max", gr::Size_t{48U}}, {"sample_rate", 64.f}});
        auto& source = fixture.template emplace<gr::basic::SignalGenerator<double>>({{"gr:compute_domain", domain == "host" ? std::string(domain) : std::string(domain) + "!"}, {"sample_rate", 64.f}, {"signal_type", "LinearChirp"}, //
            {"f_start", 2.0}, {"f_stop", 6.0}, {"duration", 0.5}, {"repeat", false}, {"amplitude", 1.f}, {"offset", 0.5f}});
        auto& sink   = fixture.template emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.template connect<"out", "clk">(clock, source).has_value());
        expect(fixture.template connect<"out", "in">(source, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink._samples.size(), 48UZ));
        for (std::size_t i = 0UZ; i < sink._samples.size(); ++i) {
            const double t        = static_cast<double>(i) / 64.0;
            const double expected = i < 32UZ ? 0.5 + std::sin(2.0 * std::numbers::pi * (2.0 * t + 4.0 * t * t)) : 0.5; // f(t) = 2 Hz + 8 Hz/s · t
            expect(approx(sink._samples[i], expected, 1.e-12)) << domain << i;
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "allocation fallback copies the selected samples even into device-only output"_domain_test = [](gr::testing::DomainContext& domain) {
        constexpr std::size_t kCount = 7UZ;
        for (SourceFaultContext::Fault fault : {SourceFaultContext::Fault::STATE_ALLOCATION, SourceFaultContext::Fault::OUTPUT_ALLOCATION}) {
            SourceFaultContext                 context(domain.context(), fault);
            gr::PortOut<std::uint8_t>          clock;
            gr::basic::SignalGenerator<double> source;
            expect(clock.connect(source.clk).has_value());
            source._core.configure(gr::signal::SignalType::Sin, 125.0, 1000.f, 0.f, 2.f, 1.f, 0U);
            std::array<std::uint8_t, kCount>          ticks{};
            double*                                   samples = domain.allocDeviceOrShared<double>(kCount);
            gr::device::DeviceSpanAccounting          accounting;
            gr::device::DeviceInputSpan<std::uint8_t> input{._data = ticks.data(), ._size = kCount, ._acct = &accounting};
            gr::device::DeviceOutputSpan<double>      output{._data = samples, ._size = kCount, ._acct = &accounting};
            expect(source.processBulk(context, input, output) == gr::work::Status::OK);
            expect(eq(context.copies, 1UZ));
            expect(eq(accounting.consumed, kCount) && eq(accounting.published, kCount));
            std::array<double, kCount> actual{};
            domain.context().copy(actual.data(), samples, sizeof(actual));
            for (std::size_t i = 0UZ; i < actual.size(); ++i) {
                const double expected = 1.0 + 2.0 * std::sin(std::numbers::pi * static_cast<double>(i) / 4.0);
                expect(approx(actual[i], expected, 1.e-12));
            }
            expect(approx(source.generateSample(), 1.0 - std::numbers::sqrt2, 1.e-12));
        }
    } | gr::testing::kAllDomains;

    "a refused allocation or device fault publishes nothing and preserves generator state"_test = [] {
        using enum SourceFaultContext::Fault;
        const auto checkFaults = [](auto& source, auto& state, auto generate) {
            using T = typename std::remove_cvref_t<decltype(source.out)>::value_type;
            for (SourceFaultContext::Fault fault : {STATE_ALLOCATION, KERNEL, TRANSFER, POISONED, FALLBACK_TRANSFER}) {
                gr::device::DeviceContextCpu backing;
                SourceFaultContext           context(backing, fault);
                source.compute_domain                               = fault == FALLBACK_TRANSFER ? "host:sycl" : "host:sycl!";
                auto                                      reference = state;
                std::array<std::uint8_t, 7UZ>             ticks{};
                std::array<T, 7UZ>                        samples{};
                gr::device::DeviceSpanAccounting          accounting;
                gr::device::DeviceInputSpan<std::uint8_t> input{._data = ticks.data(), ._size = ticks.size(), ._acct = &accounting};
                gr::device::DeviceOutputSpan<T>           output{._data = samples.data(), ._size = samples.size(), ._acct = &accounting};
                expect(source.processBulk(context, input, output) == gr::work::Status::ERROR) << magic_enum::enum_name(fault);
                expect(accounting.consumeRequested && accounting.publishRequested);
                expect(eq(accounting.consumed, 0UZ) && eq(accounting.published, 0UZ));
                expect(eq(generate(state), generate(reference))) << magic_enum::enum_name(fault);
                expect(eq(context.poisonedReleases, 0UZ));
            }
        };

        std::array<gr::PortOut<std::uint8_t>, 3UZ> clocks;
        gr::basic::SignalGenerator<double>         signal;
        expect(clocks[0].connect(signal.clk).has_value());
        signal._core.configure(gr::signal::SignalType::GaussianNoise, 0.0, 1000.f, 0.f, 2.f, 1.f, 42U);
        checkFaults(signal, signal._core, [](auto& core) { return core.generateSample(); });

        gr::basic::FunctionGenerator<double> function;
        expect(clocks[1].connect(function.clk).has_value());
        function.signal_type               = gr::basic::function_generator::LinearRamp;
        function.final_value               = 1.f;
        function.duration                  = 1.f;
        function._core.samplePeriodSeconds = 0.125;
        const gr::basic::FunctionGeneratorCore<double>::Parameters ramp{.signalType = gr::basic::function_generator::LinearRamp, .finalValue = 1.0, .duration = 1.0};
        checkFaults(function, function._core, [&ramp](auto& core) { return core.generateSample(ramp); });

        gr::basic::SpeechShapedNoise noise;
        expect(clocks[2].connect(noise.clk).has_value());
        noise.start();
        checkFaults(noise, noise._state, [](auto& state) { return state.generateSample(); });
        noise.stop();
    };
};

const boost::ut::suite<"FunctionGeneratorCore"> functionGeneratorCoreTests = [] {
    using namespace boost::ut;
    using gr::basic::FunctionGeneratorCore;
    using enum gr::basic::function_generator::SignalType;

    "a linear ramp includes its final sample and then holds the final value"_test = [] {
        FunctionGeneratorCore<double> state;
        state.samplePeriodSeconds = 0.125;
        const FunctionGeneratorCore<double>::Parameters parameters{.signalType = LinearRamp, .startValue = 2.0, .finalValue = 6.0, .duration = 0.5};
        for (const double expected : {2.0, 3.0, 4.0, 5.0, 6.0, 6.0}) {
            expect(eq(state.generateSample(parameters), expected));
        }
    };

    "a parabolic ramp joins two quadratic ends to its linear middle"_test = [] {
        FunctionGeneratorCore<double> state;
        state.samplePeriodSeconds = 0.125;
        const FunctionGeneratorCore<double>::Parameters parameters{.signalType = ParabolicRamp, .startValue = 0.0, .finalValue = 1.0, .duration = 1.0, .roundOffTime = 0.25};
        constexpr std::array                            kExpected{0.0, 1.0 / 24.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 5.0 / 6.0, 23.0 / 24.0, 1.0, 1.0};
        for (const double expected : kExpected) {
            expect(approx(state.generateSample(parameters), expected, 1.e-14));
        }
    };

    "a cubic ramp has the Hermite value at one eighth of its duration"_test = [] {
        FunctionGeneratorCore<double> state;
        state.samplePeriodSeconds = 0.125;
        const FunctionGeneratorCore<double>::Parameters parameters{.signalType = CubicSpline, .startValue = 0.0, .finalValue = 1.0, .duration = 1.0};
        constexpr std::array                            kExpected{0.0, 11.0 / 256.0, 5.0 / 32.0, 81.0 / 256.0, 0.5, 175.0 / 256.0, 27.0 / 32.0, 245.0 / 256.0, 1.0, 1.0};
        for (const double expected : kExpected) {
            expect(eq(state.generateSample(parameters), expected));
        }
    };

    "an impulse response includes both endpoints of its active interval"_test = [] {
        FunctionGeneratorCore<double> state;
        state.samplePeriodSeconds = 0.125;
        const FunctionGeneratorCore<double>::Parameters parameters{.signalType = ImpulseResponse, .startValue = 2.0, .finalValue = 6.0, .impulseTime0 = 0.25, .impulseTime1 = 0.25};
        for (const double expected : {2.0, 2.0, 6.0, 6.0, 6.0, 2.0}) {
            expect(eq(state.generateSample(parameters), expected));
        }
    };

    "an analytic complex sine includes its duration then returns its real offset"_test = [] {
        FunctionGeneratorCore<std::complex<float>> state;
        state.samplePeriodSeconds = 0.125;
        state.tone.configure(gr::signal::ToneType::Sin, 1.0, 8.0, 0.0, 2.0, 1.0);
        const FunctionGeneratorCore<std::complex<float>>::Parameters parameters{.signalType = Sin, .startValue = 1.0, .finalValue = 2.0, .duration = 0.5};
        for (std::size_t i = 0UZ; i <= 4UZ; ++i) {
            const double              phase = std::numbers::pi * static_cast<double>(i) / 4.0;
            const std::complex<float> expected{static_cast<float>(1.0 + 2.0 * std::sin(phase)), static_cast<float>(-2.0 * std::cos(phase))};
            expect(le(std::abs(state.generateSample(parameters) - expected), 1.e-6f));
        }
        expect(eq(state.generateSample(parameters), std::complex<float>{1.f, 0.f}));
    };
};

const boost::ut::suite<"ConstantSourceAndClockedFunction"> constantSourceTests = [] {
    using namespace boost::ut;
    using gr::testing::operator""_domain_test;
    "a constant source terminates with its analytic value on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        auto& source = fixture.template emplace<gr::testing::ConstantSource<float>>({{"default_value", 3.f}, {"n_samples_max", gr::Size_t{17U}}, {"gr:compute_domain", domain == "host" ? std::string(domain) : std::string(domain) + "!"}});
        auto& sink   = fixture.template emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.template connect<"out", "in">(source, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._samples, std::vector<float>(17UZ, 3.f)));
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
    "a clocked constant source emits one value per clock sample on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        auto& clock  = fixture.template emplace<gr::testing::TagSource<std::uint8_t>>({{"n_samples_max", gr::Size_t{5U}}});
        auto& source = fixture.template emplace<gr::testing::ConstantSource<float>>({{"default_value", 3.f}, {"gr:compute_domain", domain == "host" ? std::string(domain) : std::string(domain) + "!"}});
        auto& sink   = fixture.template emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.template connect<"out", "clk">(clock, source).has_value());
        expect(fixture.template connect<"out", "in">(source, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._samples, std::vector<float>(5UZ, 3.f)));
        expect(eq(source.count.value, gr::Size_t{5U}));
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
    "a clocked function ramp follows its analytic samples on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        auto& clock    = fixture.template emplace<gr::testing::TagSource<std::uint8_t>>({{"n_samples_max", gr::Size_t{33U}}, {"sample_rate", 8.f}});
        auto& function = fixture.template emplace<gr::basic::FunctionGenerator<double>>({{"signal_type", "LinearRamp"}, {"start_value", 0.f}, {"final_value", 1.f}, {"duration", 1.f}, {"sample_rate", 8.f}, {"gr:compute_domain", domain == "host" ? std::string(domain) : std::string(domain) + "!"}});
        auto& sink     = fixture.template emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.template connect<"out", "clk">(clock, function).has_value());
        expect(fixture.template connect<"out", "in">(function, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._samples.size(), 33UZ));
        for (std::size_t i = 0UZ; i < sink._samples.size(); ++i) {
            expect(eq(sink._samples[i], std::min(1.0, static_cast<double>(i) / 8.0)));
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

int main() {
    using namespace boost::ut;
    using gr::testing::operator""_domain_test;

    constexpr std::uint64_t kT0 = 1'000'000'000U; /// ns

    "a clock dates its first sample and its time tag from start_time on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& source = fixture.template emplace<gr::basic::ClockSource<std::uint8_t>>({{"sample_rate", 1000.f}, {"n_samples_max", gr::Size_t{100U}}, {"chunk_size", gr::Size_t{10U}}, {"use_internal_thread", false}, {"start_time", kT0}, {"gr:compute_domain", std::string(domain)}, //
            {"tag_times", gr::Tensor<std::uint64_t>(gr::data_from, {20'000'000U})}, {"tag_values", std::vector<std::string>{"tick/"}}});
        auto& sink   = fixture.template emplace<gr::testing::TagSink<std::uint8_t, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.template connect<"out", "in">(source, sink).has_value());
        expect(fixture.run().has_value()) << domain;

        expect(eq(sink._samples.size(), 100UZ)) << domain;
        std::vector<gr::testing::OwningTag> dated;
        std::ranges::copy_if(sink._tags, std::back_inserter(dated), [](const auto& recorded) { return recorded.map.contains(gr::tag::kTriggerTime); });
        expect(eq(dated.size(), 2UZ)) << domain << ": one for the first sample, one for the tag time";
        if (dated.size() == 2UZ) {
            expect(eq(dated[0].index, 0UL)) << domain;
            expect(eq(dated[0].map.value_or<std::uint64_t>(gr::tag::kTriggerTime, 0U), kT0)) << domain << ": the first sample is start_time";
            expect(eq(dated[1].index, 20UL)) << domain << ": 20 ms at 1 kHz";
            expect(eq(dated[1].map.value_or<std::uint64_t>(gr::tag::kTriggerTime, 0U), kT0 + 20'000'000U)) << domain << ": start_time + 20 ms";
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "an undated clock publishes no tag of its own on its first sample"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<gr::basic::ClockSource<std::uint8_t>>({
            {"sample_rate", 1000.f},
            {"n_samples_max", gr::Size_t{20U}},
            {"use_internal_thread", false},
        });
        auto&                     sink   = fixture.emplace<gr::testing::TagSink<std::uint8_t, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.connect<"out", "in">(source, sink).has_value());
        expect(fixture.run().has_value());
        expect(std::ranges::none_of(sink._tags, [](const auto& recorded) { return recorded.map.contains(gr::tag::kTriggerTime); }));
    };

    "a descending tag_times is refused: the previous list stays and the error is reported"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     clock = fixture.emplace<gr::basic::ClockSource<std::uint8_t>>({{"tag_times", gr::Tensor<std::uint64_t>(gr::data_from, {10U, 20U})}, {"chunk_size", gr::Size_t{50U}}});
        gr::MsgPortIn             errors;
        expect(clock.msgOut.connect(errors).has_value());
        clock.settings().init();
        std::ignore = clock.settings().applyStagedParameters();
        expect(clock.settings().set({{"tag_times", gr::Tensor<std::uint64_t>(gr::data_from, {30U, 5U})}, {"chunk_size", gr::Size_t{0U}}}).empty());
        std::ignore        = clock.settings().activateContext();
        const auto applied = clock.settings().applyStagedParameters();
        expect(applied.refusedParameters.contains("chunk_size")) << "a chunk size of zero is outside its limits";
        expect(eq(errors.streamReader().available(), 1UZ)) << "the descending tag_times is reported as an error message";
        expect(eq(clock.tag_times.value.size(), 2UZ));
        expect(eq(clock.tag_times.value[0], std::uint64_t{10U})) << "kept";
        expect(eq(clock.chunk_size.value, gr::Size_t{50U})) << "kept";
        const auto stored = clock.settings().getStored("tag_times").value_or(gr::pmt::Value{}).value_or(gr::Tensor<std::uint64_t>{});
        expect(eq(stored.size(), 2UZ) && eq(stored[0], std::uint64_t{10U})) << "the stored set holds the list in use, so a later context switch does not refuse it again";
    };

    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
