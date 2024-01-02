#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/basic/clock_source.hpp>
#include <gnuradio-4.0/basic/SignalGenerator.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#if defined(__clang__) && __clang_major__ >= 15
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

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
        constexpr std::uint32_t n_samples   = 1900;
        constexpr float         sample_rate = 2000.f;
        Graph                   testGraph;
        auto                   &src = testGraph.emplaceBlock<gr::basic::ClockSource<float, useIoThreadPool>>(
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
        expect(eq(static_cast<std::uint32_t>(sink1.n_samples_produced), n_samples)) << fmt::format("sink1 did not consume enough input samples ({} vs. {})", sink1.n_samples_produced, n_samples);
        expect(eq(static_cast<std::uint32_t>(sink2.n_samples_produced), n_samples)) << fmt::format("sink2 did not consume enough input samples ({} vs. {})", sink2.n_samples_produced, n_samples);

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
        constexpr std::uint32_t n_samples   = 200;
        constexpr float         sample_rate = 1000.f;
        Graph                   testGraph;
        auto                   &clockSrc  = testGraph.emplaceBlock<gr::basic::ClockSource<float>>({ { "sample_rate", sample_rate }, { "n_samples_max", n_samples }, { "name", "ClockSource" } });
        auto                   &signalGen = testGraph.emplaceBlock<SignalGenerator<float>>({ { "sample_rate", sample_rate }, { "name", "SignalGenerator" } });
        auto                   &sink      = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagSink" }, { "verbose_console", true } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(clockSrc).to<"in">(signalGen)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(signalGen).to<"in">(sink)));

        scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        expect(eq(n_samples, static_cast<std::uint32_t>(sink.n_samples_produced))) << "Number of samples does not match";
    };
};

int
main() { /* not needed for UT */
}
