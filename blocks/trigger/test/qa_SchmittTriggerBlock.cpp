#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/basic/FunctionGenerator.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/ImChartMonitor.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/SchmittTrigger.hpp>

using namespace boost::ut;

const suite<"SchmittTrigger Block"> triggerTests = [] {
    using namespace gr::basic;
    using namespace gr::testing;

    constexpr static float sample_rate       = 1000.f; // 100 Hz
    bool                   enableVisualTests = false;
    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        // conditionally enable visual tests outside the CI
        boost::ext::ut::cfg<override> = {.tag = std::vector<std::string_view>{"visual", "benchmarks"}};
        enableVisualTests             = true;
    }

    using enum gr::trigger::InterpolationMethod;

    "an interpolated edge lands on the same sample however the stream is cut"_test = [] {
        // 128 samples, low and high in runs of four: eight rising and eight falling edges, none near a span boundary by
        // luck alone
        const gr::property_map values{{"l", 0.f}, {"h", 5.f}};
        std::string            script;
        for (std::size_t run = 0UZ; run < 32UZ; ++run) {
            script += (run % 2UZ == 0UZ) ? "l l l l " : "h h h h ";
        }
        script += "|";

        // the small sizes are the point: the block must hold the scheduler to its window rather than accept a span too
        // short to place an edge in
        std::vector<std::string> answers;
        for (const std::size_t chunk : {0UZ, 1UZ, 2UZ, 3UZ, 32UZ, 40UZ, 64UZ}) {
            gr::testing::GraphFixture fixture;
            auto&                     source  = fixture.emplace<gr::blocks::trigger::MarbleSource<float>>({{"script", script}, {"sample_values", values}});
            auto&                     trigger = fixture.emplace<gr::blocks::trigger::SchmittTrigger<float, BASIC_LINEAR_INTERPOLATION>>({{"threshold", 2.f}, {"offset", 2.5f}});
            auto&                     sink    = fixture.emplace<gr::blocks::trigger::MarbleSink<float>>({{"sample_values", values}});
            if (chunk > 0UZ) {
                trigger.in.max_samples = chunk;
            }
            expect(fixture.connect<"out", "in">(source, trigger).has_value());
            expect(fixture.connect<"out", "in">(trigger, sink).has_value());
            expect(fixture.run().has_value());
            answers.push_back(sink.script());
        }

        expect(ge(answers.size(), 2UZ));
        for (std::size_t i = 1UZ; i < answers.size(); ++i) {
            expect(eq(answers[i], answers[0])) << "the interpolation window is declared through in.min_samples, so every admissible cut sees the same edges on the same samples";
        }
        expect(gt(std::ranges::count(answers[0], ':'), 8)) << "the scenario must produce edges, or it proves nothing";
    };

    "a trigger without interpolation passes every sample however closely the tags follow each other"_test = [] {
        for (const std::size_t tagSpacing : {1UZ, 16UZ, 32UZ}) {
            gr::blocks::trigger::SchmittTrigger<float, NO_INTERPOLATION> trigger({{"threshold", 2.f}, {"offset", 2.5f}});
            gr::PortOut<float>                                           source;
            gr::PortIn<float>                                            sink;
            expect(source.connect(trigger.in).has_value()) << fatal;
            expect(trigger.out.connect(sink).has_value()) << fatal;
            trigger.init(trigger.progress);
            expect(trigger.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << fatal;

            constexpr std::size_t kSamples = 128UZ;
            for (std::size_t at = 0UZ; at < kSamples; at += tagSpacing) {
                source.publishTag(gr::property_map_view{gr::property_map{{"spacing", static_cast<gr::Size_t>(tagSpacing)}}}, at);
            }
            {
                auto samples = source.streamWriter().reserve<gr::SpanReleasePolicy::ProcessAll>(kSamples);
                for (std::size_t i = 0UZ; i < kSamples; ++i) {
                    samples[i] = (i / 8UZ) % 2UZ == 0UZ ? 0.f : 5.f;
                }
                samples.publish(kSamples);
            }

            for (std::size_t call = 0UZ; call < 4UZ * kSamples && sink.streamReader().available() < kSamples; ++call) {
                std::ignore = trigger.work();
            }
            expect(eq(sink.streamReader().available(), kSamples)) << std::format("a tag every {} samples cuts each span short, which must not stall the block", tagSpacing);
        }
    };

    "an interpolating trigger keeps passing samples on a live stream however closely the tags follow each other"_test = []<typename TMethod> {
        using TTrigger                 = gr::blocks::trigger::SchmittTrigger<float, TMethod::value>;
        constexpr std::size_t kHistory = TTrigger::N_HISTORY;
        for (const std::size_t tagSpacing : {1UZ, kHistory - 1UZ, kHistory, kHistory + 1UZ, 2UZ * kHistory}) {
            TTrigger           trigger({{"threshold", 2.f}, {"offset", 2.5f}});
            gr::PortOut<float> source;
            gr::PortIn<float>  sink;
            expect(source.connect(trigger.in).has_value()) << fatal;
            expect(trigger.out.connect(sink).has_value()) << fatal;
            trigger.init(trigger.progress);
            expect(trigger.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << fatal;

            constexpr std::size_t kSamples = 8UZ * kHistory;
            for (std::size_t at = 0UZ; at < kSamples; at += tagSpacing) {
                source.publishTag(gr::property_map_view{gr::property_map{{"spacing", static_cast<gr::Size_t>(tagSpacing)}}}, at);
            }
            {
                auto samples = source.streamWriter().reserve<gr::SpanReleasePolicy::ProcessAll>(kSamples);
                for (std::size_t i = 0UZ; i < kSamples; ++i) {
                    samples[i] = (i / 8UZ) % 2UZ == 0UZ ? 0.f : 5.f;
                }
                samples.publish(kSamples);
            }

            for (std::size_t call = 0UZ; call < 4UZ * kSamples; ++call) {
                std::ignore = trigger.work();
            }
            expect(ge(sink.streamReader().available(), kSamples - kHistory)) << std::format("a tag every {} samples with no end of stream in sight: at most the interpolation window may be held back", tagSpacing);
        }
    } | std::tuple<std::integral_constant<gr::trigger::InterpolationMethod, BASIC_LINEAR_INTERPOLATION>, std::integral_constant<gr::trigger::InterpolationMethod, POLYNOMIAL_INTERPOLATION>>{};

    "a dead time suppresses every edge closer to the last emitted one, however the stream is cut"_test = [] {
        // a 0/5 square wave of period 8: rising edges at 4, 12, 20, 28 and falling edges at 8, 16, 24 in between
        const gr::property_map values{{"l", 0.f}, {"h", 5.f}};
        std::string            script;
        for (std::size_t run = 0UZ; run < 8UZ; ++run) {
            script += (run % 2UZ == 0UZ) ? "l l l l " : "h h h h ";
        }
        script += "|";

        const auto emittedEdges = [&](gr::Size_t deadTime, std::size_t chunk) {
            gr::testing::GraphFixture fixture;
            auto&                     source  = fixture.emplace<gr::blocks::trigger::MarbleSource<float>>({{"script", script}, {"sample_values", values}});
            auto&                     trigger = fixture.emplace<gr::blocks::trigger::SchmittTrigger<float, NO_INTERPOLATION>>({{"threshold", 2.f}, {"offset", 2.5f}, {"n_dead_time", deadTime}});
            auto&                     sink    = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"log_samples", false}});
            if (chunk > 0UZ) {
                trigger.in.max_samples = chunk;
            }
            expect(fixture.connect<"out", "in">(source, trigger).has_value());
            expect(fixture.connect<"out", "in">(trigger, sink).has_value());
            expect(fixture.run().has_value());
            std::vector<std::pair<std::size_t, std::string>> edges;
            for (const auto& tag : sink._tags) {
                if (const std::string name = tag.map.value_or<std::string>(std::string(gr::tag::TRIGGER_NAME.key()), std::string{}); !name.empty()) {
                    edges.emplace_back(tag.index, name);
                }
            }
            return edges;
        };

        const std::vector<std::pair<std::size_t, std::string>> risingOnly{{4UZ, "RISING"}, {12UZ, "RISING"}, {20UZ, "RISING"}, {28UZ, "RISING"}};
        for (const std::size_t chunk : {0UZ, 1UZ, 3UZ, 5UZ, 16UZ}) {
            expect(emittedEdges(6U, chunk) == risingOnly) << std::format("dead time 6 keeps only the rising edges (chunk {})", chunk);
        }
        expect(eq(emittedEdges(0U, 0UZ).size(), 7UZ)) << "without a dead time every rising and falling edge is emitted";
    };

    skip / "SchmittTrigger"_test =
        [&enableVisualTests]<class Method> {
            Graph graph;

            // create blocks
            auto& clockSrc = graph.emplaceBlock<gr::basic::ClockSource<std::uint8_t>>({//
                {"sample_rate", sample_rate}, {"n_samples_max", 1000U}, {"name", "ClockSource"},
                {"tag_times",
                    std::vector<std::uint64_t>{
                        0U,           // 0 ms - start - 50ms of bottom plateau
                        100'000'000U, // 100 ms - start - ramp-up
                        400'000'000U, // 300 ms - 50ms of top ceiling
                        500'000'000U, // 500 ms - start ramp-down
                        800'000'000U  // 700 ms - 100ms of bottom plateau
                    }},
                {"tag_values",
                    std::vector<std::string>{
                        "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=0", //
                        "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1", //
                        "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2", //
                        "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=3", //
                        "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4"  //
                    }}});

            auto& funcGen = graph.emplaceBlock<FunctionGenerator<float>>({{"sample_rate", sample_rate}, {"name", "FunctionGenerator"}, {"start_value", 0.1f}});
            using namespace function_generator;
            expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 0.1f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=0"}).empty());
            expect(funcGen.settings().set(createParabolicRampPropertyMap("CMD_BP_START", 0.1f, 1.1f, .3f, 0.02f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=1"}).empty());
            expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 1.1f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=2"}).empty());
            expect(funcGen.settings().set(createParabolicRampPropertyMap("CMD_BP_START", 1.1f, 0.1f, .3f, 0.02f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=3"}).empty());
            expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 0.1f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=4"}).empty());

            auto& schmittTrigger = graph.emplaceBlock<gr::blocks::trigger::SchmittTrigger<float, Method::value>>({
                {"name", "SchmittTrigger"},                      //
                {"threshold", .1f},                              //
                {"offset", .6f},                                 //
                {"trigger_name_rising_edge", "MY_RISING_EDGE"},  //
                {"trigger_name_falling_edge", "MY_FALLING_EDGE"} //
            });
            auto& tagSink        = graph.emplaceBlock<TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink"}, {"log_tags", true}, {"log_samples", false}, {"verbose_console", false}});

            // connect non-UI blocks
            expect(graph.connect<"out", "clk_in">(clockSrc, funcGen).has_value()) << "connect clockSrc->funcGen";
            expect(graph.connect<"out", "in">(funcGen, schmittTrigger).has_value()) << "connect funcGen->schmittTrigger";
            expect(graph.connect<"out", "in">(schmittTrigger, tagSink).has_value()) << "connect schmittTrigger->tagSink";
            std::thread uiLoop;
            if (enableVisualTests) {
                auto& uiSink1 = graph.emplaceBlock<ImChartMonitor<float>>({{"name", "ImChartSink1"}});
                auto& uiSink2 = graph.emplaceBlock<ImChartMonitor<float>>({{"name", "ImChartSink2"}});
                // connect UI blocks
                expect(graph.connect<"out", "in">(funcGen, uiSink1).has_value()) << "connect funcGen->uiSink1";
                expect(graph.connect<"out", "in">(schmittTrigger, uiSink2).has_value()) << "connect schmittTrigger->uiSink2";
                uiLoop = std::thread([&uiSink1, &uiSink2]() {
                    gr::thread_pool::thread::setThreadName("uiLoop");
                    bool drawUI = true;
                    while (drawUI) {
                        using enum gr::work::Status;
                        drawUI = false;
                        drawUI |= uiSink1.draw({{"reset_view", true}}) != DONE;
                        drawUI |= uiSink2.draw({}) != DONE;
                        std::this_thread::sleep_for(std::chrono::milliseconds(40));
                    }
                    std::this_thread::sleep_for(std::chrono::seconds(1)); // wait before shutting down
                });
            }

            gr::scheduler::Simple sched;
            if (auto ret = sched.exchange(std::move(graph)); !ret) {
                throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
            }
            expect(sched.runAndWait().has_value()) << "runAndWait";

            if (uiLoop.joinable()) {
                uiLoop.join();
            }
            enableVisualTests = false; // only for first test

            expect(eq(tagSink._tags.size(), 6UZ)) << std::format("test {} : expected total number of tags", magic_enum::enum_name(Method::value));

            // filter tags for those generated on rising and falling edges
            std::vector<std::size_t> rising_edge_indices;
            std::vector<std::size_t> falling_edge_indices;

            for (const auto& tag : tagSink._tags) {
                if (!tag.map.contains(gr::tag::TRIGGER_NAME)) {
                    continue;
                }
                std::string trigger_name = tag.map.find_value(gr::tag::TRIGGER_NAME).value().value_or(std::string());
                if (trigger_name == "MY_RISING_EDGE") {
                    rising_edge_indices.push_back(tag.index);
                } else if (trigger_name == "MY_FALLING_EDGE") {
                    falling_edge_indices.push_back(tag.index);
                }
            }
            expect(eq(rising_edge_indices.size(), 1UZ)) << std::format("test {} : expected one rising edge", magic_enum::enum_name(Method::value));
            expect(eq(falling_edge_indices.size(), 1UZ)) << std::format("test {} : expected one falling edge", magic_enum::enum_name(Method::value));

            { // where the interpolation put the two edges, on the stream's own index axis
                gr::testing::MarbleDiagram diagram{std::format("SchmittTrigger({}): the edges it found", magic_enum::enum_name(Method::value))};
                diagram.unit = "sample"; // the stream carries no anchor here, so the marks are placed by index
                auto& edges  = diagram.row("edges");
                for (const std::size_t at : rising_edge_indices) {
                    edges.at(static_cast<std::uint64_t>(at), "rising");
                }
                for (const std::size_t at : falling_edge_indices) {
                    edges.at(static_cast<std::uint64_t>(at), "falling");
                }
                diagram.condition(std::format("threshold 0.1 about an offset of 0.6, {}", magic_enum::enum_name(Method::value)));
                diagram.print();
            }

            if (Method::value == NO_INTERPOLATION) { // edge position once crossing the threshold
                expect(approx(rising_edge_indices[0], 278UZ, 2UZ)) << std::format("test {} : detected rising edge index", magic_enum::enum_name(Method::value));
                expect(approx(falling_edge_indices[0], 678UZ, 2UZ)) << std::format("test {} : detected falling edge index", magic_enum::enum_name(Method::value));
            } else { // exact edge position
                expect(approx(rising_edge_indices[0], 250UZ, 2UZ)) << std::format("test {} : detected rising edge index", magic_enum::enum_name(Method::value));
                expect(approx(falling_edge_indices[0], 650UZ, 2UZ)) << std::format("test {} : detected falling edge index", magic_enum::enum_name(Method::value));
            }
        } |
        std::tuple<std::integral_constant<gr::trigger::InterpolationMethod, LINEAR_INTERPOLATION>, //
            std::integral_constant<gr::trigger::InterpolationMethod, BASIC_LINEAR_INTERPOLATION>,  //
            std::integral_constant<gr::trigger::InterpolationMethod, POLYNOMIAL_INTERPOLATION>,    //
            std::integral_constant<gr::trigger::InterpolationMethod, NO_INTERPOLATION>>{};
};

int main() { /* not needed for UT */ }
