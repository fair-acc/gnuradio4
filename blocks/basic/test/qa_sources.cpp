#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/basic/clock_source.hpp>
#include <gnuradio-4.0/basic/FunctionGenerator.hpp>
#include <gnuradio-4.0/basic/SignalGenerator.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#if defined(__clang__) && __clang_major__ >= 15
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

template<typename T>
constexpr void
addTimeTagEntry(gr::basic::ClockSource<T> &clockSource, std::uint64_t timeInNanoseconds, const std::string &value) {
    clockSource.tag_times.push_back(timeInNanoseconds);
    clockSource.tag_values.push_back(value);
}

const boost::ut::suite TagTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    static const auto mismatchedKey = [](const property_map &map) {
        std::vector<std::string> keys;
        for (const auto &pair : map) {
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
        auto                &src = testGraph.emplaceBlock<gr::basic::ClockSource<float, useIoThreadPool>>(
                { { "sample_rate", sample_rate }, { "n_samples_max", n_samples }, { "name", "ClockSource" }, { "verbose_console", verbose } });
        src.tags = {
            { 0, { { "key", "value@0" } } },       //
            { 1, { { "key", "value@1" } } },       //
            { 100, { { "key", "value@100" } } },   //
            { 150, { { "key", "value@150" } } },   //
            { 1000, { { "key", "value@1000" } } }, //
            { 1001, { { "key", "value@1001" } } }, //
            { 1002, { { "key", "value@1002" } } }, //
            { 1023, { { "key", "value@1023" } } }  //
        };
        auto &sink1 = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagSink1" } });
        auto &sink2 = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({ { "name", "TagSink2" } });
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).template to<"in">(sink1)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).template to<"in">(sink2)));

        scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();
        if (verbose) {
            fmt::println("finished ClockSource sched.runAndWait() w/ {}", useIoThreadPool ? "Graph/Block<T> provided-thread" : "user-provided thread");
        }

        expect(eq(src.n_samples_max, n_samples)) << "src did not accept require max input samples";
        expect(eq(src.n_samples_produced, n_samples)) << "src did not produce enough output samples";
        expect(eq(static_cast<gr::Size_t>(sink1.n_samples_produced), n_samples)) << fmt::format("sink1 did not consume enough input samples ({} vs. {})", sink1.n_samples_produced, n_samples);
        expect(eq(static_cast<gr::Size_t>(sink2.n_samples_produced), n_samples)) << fmt::format("sink2 did not consume enough input samples ({} vs. {})", sink2.n_samples_produced, n_samples);

        if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
            expect(approx(sink1.effective_sample_rate(), sample_rate, 500.f))
                    << fmt::format("sink1: effective sample rate {} vs {} +- {} does not match", sink1.effective_sample_rate(), sample_rate, 500.f);
            expect(approx(sink2.effective_sample_rate(), sample_rate, 500.f))
                    << fmt::format("sink2: effective sample rate {} vs {} +- {} does not match", sink1.effective_sample_rate(), sample_rate, 500.f);
        }

        fmt::print("sink1 (USE_PROCESS_ONE): effective {} vs. expected {} sample rate [Hz]\n", sink1.effective_sample_rate(), sample_rate);
        fmt::print("sink2 (USE_PROCESS_BULK): effective {} vs. expected {} sample rate [Hz]\n", sink2.effective_sample_rate(), sample_rate);

        // TODO: last decimator/interpolator + stride addition seems to break the limiting the input samples to the min of available vs. n_samples-until next tags
        // expect(equal_tag_lists(src.tags, sink1.tags)) << "sink1 (USE_PROCESS_ONE) did not receive the required tags";
        // expect(equal_tag_lists(src.tags, sink2.tags)) << "sink2 (USE_PROCESS_BULK) did not receive the required tags";
        if (verbose) {
            fmt::println("finished ClockSource test w/ {}", useIoThreadPool ? "Graph/Block<T>-provided thread" : "user-provided thread");
        }
    };

    "ClockSource w/ Graph/Block<T> provided thread"_test = [&clockSourceTest] { clockSourceTest.template operator()<true>(false); };
    // "ClockSource w/ user-provided thread"_test           = [&clockSourceTest] { clockSourceTest.template operator()<false>(false); }; // TODO: check potential threading issue for user-provided threads

    "SignalGenerator test"_test = [] {
        const std::size_t        N      = 16; // test points
        const double             offset = 2.;
        std::vector<std::string> signals{ "Const", "Sin", "Cos", "Square", "Saw", "Triangle" };

        for (const auto &sig : signals) {
            SignalGenerator<double> signalGen{};
            auto                    failed = signalGen.settings().set({ { "signal_type", sig }, //
                                                                        { "sample_rate", 2048.f },
                                                                        { "frequency", 256. },
                                                                        { "amplitude", 1. },
                                                                        { "offset", offset },
                                                                        { "phase", std::numbers::pi / 4 } });
            expect(failed.empty()) << fmt::format("settings have mismatching keys or value types. offending keys: {}\n", fmt::join(mismatchedKey(failed), ", "));
            const auto forwardSettings = signalGen.settings().applyStagedParameters();
            expect(eq(forwardSettings.size(), 1UZ)) << fmt::format("incorrect number of to be forwarded settings. forward keys: {}\n", fmt::join(mismatchedKey(forwardSettings), ", "));

            // expected values corresponds to sample_rate = 1024., frequency = 128., amplitude = 1., offset = 0., phase = pi/4.
            std::map<std::string, std ::vector<double>> expResults = {
                { "Const", { 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. } },
                { "Sin", { 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0., 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0. } },
                { "Cos", { 0.707106, 0., -0.707106, -1., -0.7071067, 0., 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0., 0.707106, 1. } },
                { "Square", { 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 1. } },
                { "Saw", { 0.25, 0.5, 0.75, -1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, -1., -0.75, -0.5, -0.25, 0. } },
                { "Triangle", { 0.5, 1., 0.5, 0., -0.5, -1., -0.5, 0., 0.5, 1., 0.5, 0., -0.5, -1., -0.5, 0. } }
            };

            for (std::size_t i = 0; i < N; i++) {
                const auto val = signalGen.processOne(0);
                const auto exp = expResults[sig][i] + offset;
                expect(approx(exp, val, 1e-5)) << fmt::format("SignalGenerator for signal: {} and i: {} does not match.", sig, i);
            }
        }
    };

    "SignalGenerator ImChart test"_test = [] {
        const std::size_t        N = 512; // test points
        std::vector<std::string> signals{ "Const", "Sin", "Cos", "Square", "Saw", "Triangle" };
        for (const auto &sig : signals) {
            SignalGenerator<double> signalGen{};
            const auto              failed = signalGen.settings().set({ { "signal_type", sig }, //
                                                                        { "sample_rate", 8192.f },
                                                                        { "frequency", 32. },
                                                                        { "amplitude", 2. },
                                                                        { "offset", 0. },
                                                                        { "phase", std::numbers::pi / 4. } });
            expect(failed.empty()) << fmt::format("settings have mismatching keys or value types. offending keys: {}\n", fmt::join(mismatchedKey(failed), ", "));
            const auto forwardSettings = signalGen.settings().applyStagedParameters();
            expect(eq(forwardSettings.size(), 1UZ)) << fmt::format("incorrect number of to be forwarded settings. forward keys: {}\n", fmt::join(mismatchedKey(forwardSettings), ", "));

            std::vector<double> xValues(N), yValues(N);
            std::iota(xValues.begin(), xValues.end(), 0);
            std::ranges::generate(yValues, [&signalGen]() { return signalGen.processOne(0); });

            fmt::println("Chart {}\n\n", sig);
            auto chart = gr::graphs::ImChart<128, 16>({ { 0., static_cast<double>(N) }, { -2.6, 2.6 } });
            chart.draw(xValues, yValues, sig);
            chart.draw();
        }
    };

    "SignalGenerator + ClockSource test"_test = [] {
        constexpr gr::Size_t n_samples   = 200;
        constexpr float      sample_rate = 1000.f;
        Graph                testGraph;
        auto                &clockSrc  = testGraph.emplaceBlock<gr::basic::ClockSource<float>>({ { "sample_rate", sample_rate }, { "n_samples_max", n_samples }, { "name", "ClockSource" } });
        auto                &signalGen = testGraph.emplaceBlock<SignalGenerator<float>>({ { "sample_rate", sample_rate }, { "name", "SignalGenerator" } });
        auto                &sink      = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagSink" }, { "verbose_console", true } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(clockSrc).to<"in">(signalGen)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(signalGen).to<"in">(sink)));

        scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        expect(eq(n_samples, static_cast<std::uint32_t>(sink.n_samples_produced))) << "Number of samples does not match";
    };

    "FunctionGenerator ImChart test"_test = [] {
        using namespace function_generator;
        const std::size_t       N          = 128; // test points
        double                  startValue = 10.;
        double                  finalValue = 20.;
        std::vector<SignalType> signals{ Const, LinearRamp, ParabolicRamp, CubicSpline, ImpulseResponse };
        for (const auto &sig : signals) {
            FunctionGenerator<double> funcGen{};
            std::ignore = funcGen.settings().set({ createPropertyMapEntry(signal_type, sig),
                                                   { "n_samples_max", static_cast<std::uint32_t>(N) },
                                                   { "sample_rate", 128.f },
                                                   createPropertyMapEntry(start_value, startValue),
                                                   createPropertyMapEntry(final_value, finalValue),
                                                   createPropertyMapEntry(duration, 1.),
                                                   createPropertyMapEntry(round_off_time, .15),
                                                   createPropertyMapEntry(impulse_time0, .2),
                                                   createPropertyMapEntry(impulse_time1, .15) });
            std::ignore = funcGen.settings().applyStagedParameters();

            std::vector<double> xValues(N), yValues(N);
            std::iota(xValues.begin(), xValues.end(), 0);
            std::ranges::generate(yValues, [&funcGen]() { return funcGen.processOne(0.); });
            fmt::println("Chart {}\n\n", toString(sig));
            auto chart = gr::graphs::ImChart<128, 32>({ { 0., static_cast<double>(N) }, { 7., 22. } });
            chart.draw(xValues, yValues, toString(sig));
            chart.draw();
        }
    };

    "FunctionGenerator + ClockSource test"_test = [] {
        using namespace function_generator;
        constexpr std::uint32_t N           = 1000;
        constexpr float         sample_rate = 1000.f;
        Graph                   testGraph;
        auto                   &clockSrc = testGraph.emplaceBlock<gr::basic::ClockSource<float>>({ { "sample_rate", sample_rate }, { "n_samples_max", N }, { "name", "ClockSource" } });

        clockSrc.tags = { Tag(0, createConstPropertyMap(5.f)),                              //
                          Tag(100, createLinearRampPropertyMap(5.f, 30.f, .2f)),            //
                          Tag(300, createConstPropertyMap(30.f)),                           //
                          Tag(350, createParabolicRampPropertyMap(30.f, 20.f, .1f, 0.02f)), //
                          Tag(550, createConstPropertyMap(20.f)),                           //
                          Tag(650, createCubicSplinePropertyMap(20.f, 10.f, .1f)),          //
                          Tag(800, createConstPropertyMap(10.f)),
                          Tag(850, createImpulseResponsePropertyMap(10.f, 20.f, .02f, .06f)) };

        auto &funcGen = testGraph.emplaceBlock<FunctionGenerator<float>>({ { "sample_rate", sample_rate }, { "name", "FunctionGenerator" } });
        auto &sink    = testGraph.emplaceBlock<gr::testing::TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagSink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(clockSrc).to<"in">(funcGen)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(funcGen).to<"in">(sink)));

        scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();
        expect(eq(N, static_cast<std::uint32_t>(sink.samples.size()))) << "Number of samples does not match";

        std::vector<double> xValues(N);
        std::vector<double> yValues(sink.samples.begin(), sink.samples.end());
        std::iota(xValues.begin(), xValues.end(), 0);
        auto chart = gr::graphs::ImChart<128, 32>({ { 0., static_cast<double>(N) }, { 0., 35. } });
        chart.draw(xValues, yValues, "Signal");
        chart.draw();
    };

    "FunctionGenerator + ClockSource FAIR test"_test = [] {
        using namespace function_generator;
        constexpr std::uint32_t N           = 1000;
        constexpr float         sample_rate = 10000.f;
        Graph                   testGraph;
        auto                   &clockSrc = testGraph.emplaceBlock<gr::basic::ClockSource<float>>({ { "sample_rate", sample_rate }, { "n_samples_max", N }, { "name", "ClockSource" } });

        // all times are in nanoseconds
        addTimeTagEntry(clockSrc, 1'000'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=1");
        addTimeTagEntry(clockSrc, 10'000'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=2");
        addTimeTagEntry(clockSrc, 30'000'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=3");
        addTimeTagEntry(clockSrc, 35'000'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=4");
        addTimeTagEntry(clockSrc, 55'000'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=5");
        addTimeTagEntry(clockSrc, 65'000'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=6");
        addTimeTagEntry(clockSrc, 80'000'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=7");
        addTimeTagEntry(clockSrc, 85'000'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=8");
        clockSrc.repeat_period      = 5'000'000;
        clockSrc.do_zero_order_hold = true;
        auto &funcGen               = testGraph.emplaceBlock<FunctionGenerator<float>>({ { "sample_rate", sample_rate }, { "name", "FunctionGenerator" } });
        funcGen.match_pred          = [](const auto &tableEntry, const auto &searchEntry, const auto attempt) -> std::optional<bool> {
            if (searchEntry.find("context") == searchEntry.end()) {
                return std::nullopt;
            }
            if (tableEntry.find("context") == tableEntry.end()) {
                return false;
            }

            const pmtv::pmt searchEntryContext = searchEntry.at("context");
            const pmtv::pmt tableEntryContext  = tableEntry.at("context");
            if (std::holds_alternative<std::string>(searchEntryContext) && std::holds_alternative<std::string>(tableEntryContext)) {
                const auto searchString = std::get<std::string>(searchEntryContext);
                const auto tableString  = std::get<std::string>(tableEntryContext);

                if (!searchString.starts_with("CMD_BP_START:")) {
                    return std::nullopt;
                }

                if (attempt >= searchString.length()) {
                    return std::nullopt;
                }
                auto [it1, it2] = std::ranges::mismatch(searchString, tableString);
                if (std::distance(searchString.begin(), it1) == static_cast<std::ptrdiff_t>(searchString.length() - attempt)) {
                    return true;
                }
            }
            return false;
        };

        // Time duration is in seconds
        funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=1" } }, createConstPropertyMap(5.f));
        funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=2" } }, createLinearRampPropertyMap(5.f, 30.f, 0.02f));
        funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=3" } }, createConstPropertyMap(30.f));
        funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=4" } }, createParabolicRampPropertyMap(30.f, 20.f, .01f, 0.002f));
        funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=5" } }, createConstPropertyMap(20.f));
        funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=6" } }, createCubicSplinePropertyMap(20.f, 10.f, 0.01f));
        funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=7" } }, createConstPropertyMap(10.f));
        funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=8" } }, createImpulseResponsePropertyMap(10.f, 20.f, 0.002f, 0.006f));

        auto &sink = testGraph.emplaceBlock<gr::testing::TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagSink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(clockSrc).to<"in">(funcGen)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(funcGen).to<"in">(sink)));

        scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();
        expect(eq(N, static_cast<std::uint32_t>(sink.samples.size()))) << "Number of samples does not match";

        std::vector<double> xValues(N);
        std::vector<double> yValues(sink.samples.begin(), sink.samples.end());
        std::iota(xValues.begin(), xValues.end(), 0);
        auto chart = gr::graphs::ImChart<128, 32>({ { 0., static_cast<double>(N) }, { 0., 35. } });
        chart.draw(xValues, yValues, "Signal");
        chart.draw();
    };
};

int
main() { /* not needed for UT */
}
