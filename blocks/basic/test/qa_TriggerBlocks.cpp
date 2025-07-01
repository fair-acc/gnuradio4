#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/basic/FunctionGenerator.hpp>
#include <gnuradio-4.0/basic/Trigger.hpp>
#include <gnuradio-4.0/testing/ImChartMonitor.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

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
    "SchmittTrigger"_test =
        [&enableVisualTests]<class Method> {
            Graph graph;

            // create blocks
            auto& clockSrc = graph.emplaceBlock<gr::basic::ClockSource<std::uint8_t>>({//
                {"sample_rate", sample_rate}, {"n_samples_max", 1000U}, {"name", "ClockSource"},
                {"tag_times",
                    std::vector<std::uint64_t>{
                        0U,           // 0 ms - start - 50ms of bottom plateau
                        100'000'000U, // 100 ms - start - ramp-up
                        400'000'000U, // 300 ms - 50ms of bottom plateau
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
                    }},
                {"do_zero_order_hold", true}});

            auto& funcGen = graph.emplaceBlock<FunctionGenerator<float>>({{"sample_rate", sample_rate}, {"name", "FunctionGenerator"}, {"start_value", 0.1f}});
            using namespace function_generator;
            expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 0.1f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=0"}).empty());
            expect(funcGen.settings().set(createParabolicRampPropertyMap("CMD_BP_START", 0.1f, 1.1f, .3f, 0.02f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=1"}).empty());
            expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 1.1f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=2"}).empty());
            expect(funcGen.settings().set(createParabolicRampPropertyMap("CMD_BP_START", 1.1f, 0.1f, .3f, 0.02f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=3"}).empty());
            expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 0.1f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=4"}).empty());

            auto& schmittTrigger = graph.emplaceBlock<gr::blocks::basic::SchmittTrigger<float, Method::value>>({
                {"name", "SchmittTrigger"},                      //
                {"threshold", .1f},                              //
                {"offset", .6f},                                 //
                {"trigger_name_rising_edge", "MY_RISING_EDGE"},  //
                {"trigger_name_falling_edge", "MY_FALLING_EDGE"} //
            });
            auto& tagSink        = graph.emplaceBlock<TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink"}, {"log_tags", true}, {"log_samples", false}, {"verbose_console", false}});

            // connect non-UI blocks
            expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(clockSrc).to<"in">(funcGen))) << "connect clockSrc->funcGen";
            expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(funcGen).to<"in">(schmittTrigger))) << "connect funcGen->schmittTrigger";
            expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(schmittTrigger).template to<"in">(tagSink))) << "connect schmittTrigger->tagSink";
            std::thread uiLoop;
            if (enableVisualTests) {
                auto& uiSink1 = graph.emplaceBlock<ImChartMonitor<float>>({{"name", "ImChartSink1"}});
                auto& uiSink2 = graph.emplaceBlock<ImChartMonitor<float>>({{"name", "ImChartSink2"}});
                // connect UI blocks
                expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(funcGen).to<"in">(uiSink1))) << "connect funcGen->uiSink1";
                expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(schmittTrigger).template to<"in">(uiSink2))) << "connect schmittTrigger->uiSink2";
                uiLoop = std::thread([&uiSink1, &uiSink2]() {
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

            gr::scheduler::Simple sched{std::move(graph)}; // declared here to ensure life-time of graph and blocks inside.
            expect(sched.runAndWait().has_value()) << "runAndWait";

            if (uiLoop.joinable()) {
                uiLoop.join();
            }
            enableVisualTests = false; // only for first test

            expect(eq(tagSink._tags.size(), 7UZ)) << std::format("test {} : expected total number of tags", magic_enum::enum_name(Method::value));

            // filter tags for those generated on rising and falling edges
            std::vector<std::size_t> rising_edge_indices;
            std::vector<std::size_t> falling_edge_indices;

            for (const auto& tag : tagSink._tags) {
                if (!tag.map.contains(std::string(gr::tag::TRIGGER_NAME.shortKey()))) {
                    continue;
                }
                std::string trigger_name = std::get<std::string>(tag.map.at(std::string(gr::tag::TRIGGER_NAME.shortKey())));
                if (trigger_name == "MY_RISING_EDGE") {
                    rising_edge_indices.push_back(tag.index);
                } else if (trigger_name == "MY_FALLING_EDGE") {
                    falling_edge_indices.push_back(tag.index);
                }
            }
            expect(eq(rising_edge_indices.size(), 1UZ)) << std::format("test {} : expected one rising edge", magic_enum::enum_name(Method::value));
            expect(eq(falling_edge_indices.size(), 1UZ)) << std::format("test {} : expected one falling edge", magic_enum::enum_name(Method::value));

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
            std::integral_constant<gr::trigger::InterpolationMethod, NO_INTERPOLATION>>{};
};

int main() { /* not needed for UT */ }
