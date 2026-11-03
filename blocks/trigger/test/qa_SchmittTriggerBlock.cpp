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
#include <gnuradio-4.0/trigger/SchmittTrigger.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

namespace qa_schmitt_trigger {
using namespace boost::ut;

const suite<"SchmittTrigger Block"> blockSuite = [] {
    using namespace gr::basic;
    using namespace gr::testing;

    constexpr static float kSampleRate       = 1000.f;
    bool                   enableVisualTests = false;
    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        boost::ext::ut::cfg<override> = {.tag = std::vector<std::string_view>{"visual", "benchmarks"}};
        enableVisualTests             = true;
    }

    using enum gr::trigger::InterpolationMethod;

    "an interpolated edge lands on the same sample however the stream is cut"_test = [] {
        const gr::property_map values{{"l", 0.f}, {"h", 5.f}};
        std::string            script;
        for (std::size_t run = 0UZ; run < 32UZ; ++run) {
            script += (run % 2UZ == 0UZ) ? "l l l l " : "h h h h ";
        }
        script += "|";

        /**
         * the small sizes are the point: the block must hold the scheduler to its window rather than accept a span too
         * short to place an edge in
         */
        std::vector<std::string> answers;
        for (const std::size_t chunk : {0UZ, 1UZ, 2UZ, 3UZ, 32UZ, 40UZ, 64UZ}) {
            gr::testing::GraphFixture fixture;
            auto&                     source  = fixture.emplace<gr::blocks::trigger::MarbleSource<float>>({{"script", script}, {"sample_values", values}});
            auto&                     trigger = fixture.emplace<gr::blocks::trigger::SchmittTrigger<float, BASIC_LINEAR_INTERPOLATION>>({{"threshold", 2.5f}, {"hysteresis", 4.f}});
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

    "an interpolating trigger releases its held-back window when the stream ends without an edge"_test = []<typename TMethod> {
        const gr::property_map values{{"l", 0.f}};
        std::string            script;
        for (std::size_t i = 0UZ; i < 64UZ; ++i) {
            script += "l ";
        }
        script += "|";
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<gr::blocks::trigger::MarbleSource<float>>({{"script", script}, {"sample_values", values}});
        auto&                     trigger = fixture.emplace<gr::blocks::trigger::SchmittTrigger<float, TMethod::value>>({{"threshold", 2.5f}, {"hysteresis", 4.f}});
        auto&                     sink    = fixture.emplace<gr::blocks::trigger::MarbleSink<float>>({{"sample_values", values}});
        expect(fixture.connect<"out", "in">(source, trigger).has_value());
        expect(fixture.connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._samples.size(), 64UZ));
    } | std::tuple<std::integral_constant<gr::trigger::InterpolationMethod, BASIC_LINEAR_INTERPOLATION>, std::integral_constant<gr::trigger::InterpolationMethod, POLYNOMIAL_INTERPOLATION>>{};

    "a trigger without interpolation passes every sample however closely the tags follow each other"_test = [] {
        for (const std::size_t tagSpacing : {1UZ, 16UZ, 32UZ}) {
            gr::blocks::trigger::SchmittTrigger<float, NO_INTERPOLATION> trigger({{"threshold", 2.5f}, {"hysteresis", 4.f}});
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
        constexpr std::size_t kHistory = TTrigger::kNHistory;
        for (const std::size_t tagSpacing : {1UZ, kHistory - 1UZ, kHistory, kHistory + 1UZ, 2UZ * kHistory}) {
            TTrigger           trigger({{"threshold", 2.5f}, {"hysteresis", 4.f}});
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
        const gr::property_map values{{"l", 0.f}, {"h", 5.f}};
        std::string            script;
        for (std::size_t run = 0UZ; run < 8UZ; ++run) {
            script += (run % 2UZ == 0UZ) ? "l l l l " : "h h h h ";
        }
        script += "|";

        const auto emittedEdges = [&](gr::Size_t deadTime, std::size_t chunk) {
            gr::testing::GraphFixture fixture;
            auto&                     source  = fixture.emplace<gr::blocks::trigger::MarbleSource<float>>({{"script", script}, {"sample_values", values}});
            auto&                     trigger = fixture.emplace<gr::blocks::trigger::SchmittTrigger<float, NO_INTERPOLATION>>({
                {"threshold", 2.5f},
                {"hysteresis", 4.f},
                {"dead_time", static_cast<double>(deadTime)},
                {"trigger_name_rising", std::string("RISING")},
                {"trigger_name_falling", std::string("FALLING")},
            });
            auto&                     sink    = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"log_samples", false}});
            if (chunk > 0UZ) {
                trigger.in.max_samples = chunk;
            }
            expect(fixture.connect<"out", "in">(source, trigger).has_value());
            expect(fixture.connect<"out", "in">(trigger, sink).has_value());
            expect(fixture.run().has_value());
            std::vector<std::pair<std::size_t, std::string>> edges;
            for (const auto& tag : sink._tags) {
                if (const std::string name = tag.map.value_or<std::string>(std::string(gr::tag::kTriggerName.key()), std::string{}); !name.empty()) {
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

            auto& clockSrc = graph.emplaceBlock<gr::basic::ClockSource<std::uint8_t>>({//
                {"sample_rate", kSampleRate}, {"n_samples_max", 1000U}, {"name", "ClockSource"}, {"tag_times", std::vector<std::uint64_t>{0U, 100'000'000U, 400'000'000U, 500'000'000U, 800'000'000U}},
                {"tag_values",
                    std::vector<std::string>{
                        "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=0", //
                        "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1", //
                        "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2", //
                        "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=3", //
                        "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4"  //
                    }}});

            auto& funcGen = graph.emplaceBlock<FunctionGenerator<float>>({{"sample_rate", kSampleRate}, {"name", "FunctionGenerator"}, {"start_value", 0.1f}});
            using namespace function_generator;
            expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 0.1f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=0"}).empty());
            expect(funcGen.settings().set(createParabolicRampPropertyMap("CMD_BP_START", 0.1f, 1.1f, .3f, 0.02f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=1"}).empty());
            expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 1.1f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=2"}).empty());
            expect(funcGen.settings().set(createParabolicRampPropertyMap("CMD_BP_START", 1.1f, 0.1f, .3f, 0.02f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=3"}).empty());
            expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 0.1f), SettingsCtx{.context = "FAIR.SELECTOR.C=1:S=1:P=4"}).empty());

            auto& schmittTrigger = graph.emplaceBlock<gr::blocks::trigger::SchmittTrigger<float, Method::value>>({
                {"name", "SchmittTrigger"},                 //
                {"threshold", .6f},                         //
                {"hysteresis", .2f},                        //
                {"trigger_name_rising", "MY_RISING_EDGE"},  //
                {"trigger_name_falling", "MY_FALLING_EDGE"} //
            });
            auto& tagSink        = graph.emplaceBlock<TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSink"}, {"log_tags", true}, {"log_samples", false}, {"verbose_console", false}});

            expect(graph.connect<"out", "clk">(clockSrc, funcGen).has_value()) << "connect clockSrc->funcGen";
            expect(graph.connect<"out", "in">(funcGen, schmittTrigger).has_value()) << "connect funcGen->schmittTrigger";
            expect(graph.connect<"out", "in">(schmittTrigger, tagSink).has_value()) << "connect schmittTrigger->tagSink";
            std::thread uiLoop;
            if (enableVisualTests) {
                auto& uiSink1 = graph.emplaceBlock<ImChartMonitor<float>>({{"name", "ImChartSink1"}});
                auto& uiSink2 = graph.emplaceBlock<ImChartMonitor<float>>({{"name", "ImChartSink2"}});
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
                    std::this_thread::sleep_for(std::chrono::seconds(1));
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
            enableVisualTests = false;

            expect(eq(tagSink._tags.size(), 7UZ)) << std::format("test {} : the five timing tags and one rising and one falling edge", magic_enum::enum_name(Method::value));
            const std::string firstContext = tagSink._tags.empty() ? std::string() : tagSink._tags.front().map.value_or<std::string>(gr::tag::kContext.key(), std::string());
            expect(eq(firstContext, std::string("FAIR.SELECTOR.C=1:S=1:P=0"))) << "the timing tag on sample 0, merged with the forwarded sample_rate, arrives";

            std::vector<std::size_t> risingEdgeIndices;
            std::vector<std::size_t> fallingEdgeIndices;

            for (const auto& tag : tagSink._tags) {
                if (!tag.map.contains(gr::tag::kTriggerName)) {
                    continue;
                }
                std::string triggerName = tag.map.find_value(gr::tag::kTriggerName).value().value_or(std::string());
                if (triggerName == "MY_RISING_EDGE") {
                    risingEdgeIndices.push_back(tag.index);
                } else if (triggerName == "MY_FALLING_EDGE") {
                    fallingEdgeIndices.push_back(tag.index);
                }
            }
            expect(eq(risingEdgeIndices.size(), 1UZ)) << std::format("test {} : expected one rising edge", magic_enum::enum_name(Method::value));
            expect(eq(fallingEdgeIndices.size(), 1UZ)) << std::format("test {} : expected one falling edge", magic_enum::enum_name(Method::value));

            {
                gr::testing::MarbleDiagram diagram{std::format("SchmittTrigger({}): the edges it found", magic_enum::enum_name(Method::value))};
                diagram.unit = "sample";
                auto& edges  = diagram.row("edges");
                for (const std::size_t at : risingEdgeIndices) {
                    edges.at(static_cast<std::uint64_t>(at), "rising");
                }
                for (const std::size_t at : fallingEdgeIndices) {
                    edges.at(static_cast<std::uint64_t>(at), "falling");
                }
                diagram.condition(std::format("a band of 0.2 about a threshold of 0.6, {}", magic_enum::enum_name(Method::value)));
                diagram.print();
            }

            if (Method::value == NO_INTERPOLATION || Method::value == BASIC_LINEAR_INTERPOLATION) { // the boundary 0.7 is crossed
                expect(approx(risingEdgeIndices[0], 278UZ, 2UZ)) << std::format("test {} : detected rising edge index", magic_enum::enum_name(Method::value));
                expect(approx(fallingEdgeIndices[0], 678UZ, 2UZ)) << std::format("test {} : detected falling edge index", magic_enum::enum_name(Method::value));
            } else { // the fits date the nominal level 0.6 retrospectively
                expect(approx(risingEdgeIndices[0], 250UZ, 2UZ)) << std::format("test {} : detected rising edge index", magic_enum::enum_name(Method::value));
                expect(approx(fallingEdgeIndices[0], 650UZ, 2UZ)) << std::format("test {} : detected falling edge index", magic_enum::enum_name(Method::value));
            }
        } |
        std::tuple<std::integral_constant<gr::trigger::InterpolationMethod, LINEAR_INTERPOLATION>, //
            std::integral_constant<gr::trigger::InterpolationMethod, BASIC_LINEAR_INTERPOLATION>,  //
            std::integral_constant<gr::trigger::InterpolationMethod, POLYNOMIAL_INTERPOLATION>,    //
            std::integral_constant<gr::trigger::InterpolationMethod, NO_INTERPOLATION>>{};
};

const suite<"SchmittTrigger events"> eventsSuite = [] {
    using enum gr::trigger::InterpolationMethod;
    using gr::blocks::trigger::MarbleSink;
    using gr::blocks::trigger::MarbleSource;

    const gr::property_map values{{"lo", 0.f}, {"hi", 5.f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("sync")}, //
                                          {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1'000'000U}},        //
                                          {std::string(gr::tag::kTriggerOffset.key()), 0.f}}}};

    "each edge also leaves on evtOut, dated as its tag"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:lo lo hi hi lo lo hi hi lo |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     trigger = fixture.emplace<gr::blocks::trigger::SchmittTrigger<float, NO_INTERPOLATION>>({
            {"threshold", 2.5f},
            {"hysteresis", 2.f},
            {"sample_rate", 1000.f},
            {"trigger_name_rising", std::string("RISING")},
            {"trigger_name_falling", std::string("FALLING")},
        });
        auto&                     sink    = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        auto&                     edges   = fixture.emplace<gr::trigger_test::EventTap>();
        expect(fixture.connect<"out", "in">(source, trigger).has_value());
        expect(fixture.connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(trigger, edges).has_value());
        expect(fixture.run().has_value());

        std::vector<std::pair<std::uint64_t, std::string>> tagged;
        for (const auto& recorded : sink._tags) {
            const gr::property_map_view edgeTag{recorded.map};
            const auto                  name = edgeTag.get_if<std::string_view>(std::string_view{gr::tag::kTriggerName.key()});
            const auto                  at   = edgeTag.get_if<std::uint64_t>(std::string_view{gr::tag::kTriggerTime.key()});
            if (name && at && (*name == "RISING" || *name == "FALLING")) {
                tagged.emplace_back(*at, std::string(*name));
            }
        }
        expect(eq(tagged.size(), 4UZ)) << "two rising and two falling edges";
        for (const auto& recorded : sink._tags) {
            expect(!gr::trigger_test::carriesSource(recorded.map)) << "the stream tag carries no source";
            expect(!gr::property_map_view{recorded.map}.contains(gr::tag::kTriggerMetaInfo.key())) << "nor trigger_meta_info";
        }
        expect(edges.dated() == tagged) << "the event carries the edge's time and name, as SchmittTrigger -> TagToEvent would";
        expect(eq(trigger.n_events_dropped.value, 0U));
        for (const gr::property_map& event : edges._events) {
            expect(!gr::trigger_test::carriesSource(event)) << "an event carries no source";
            expect(!gr::property_map_view{event}.contains(gr::tag::kTriggerMetaInfo.key())) << "and no trigger_meta_info: an edge has nothing to put in it";
        }
    };

    "an unconnected evtOut counts nothing"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:lo hi lo hi lo hi lo hi lo |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     trigger = fixture.emplace<gr::blocks::trigger::SchmittTrigger<float, NO_INTERPOLATION>>({{"threshold", 2.5f}, {"hysteresis", 2.f}, {"sample_rate", 1000.f}});
        auto&                     sink    = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.connect<"out", "in">(source, trigger).has_value());
        expect(fixture.connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(trigger.n_events_dropped.value, 0U));
    };
    "the edge setting picks the reported edges, each named by its polarity, and dead time in seconds equals the same time in samples"_test = [&] {
        const auto reported = [&](const gr::property_map& extra) {
            gr::property_map settings{
                {"threshold", 2.5f},
                {"hysteresis", 2.f},
                {"sample_rate", 1000.f},
                {"trigger_name", std::string("any")},
                {"trigger_name_rising", std::string("RISING")},
            };
            settings.insert(extra.begin(), extra.end());
            gr::testing::GraphFixture fixture;
            auto&                     source  = fixture.emplace<MarbleSource<float>>({
                {"script", std::string("T:lo lo hi hi lo lo hi hi lo |")},
                {"sample_values", values},
                {"sample_tags", tags},
            });
            auto&                     trigger = fixture.emplace<gr::blocks::trigger::SchmittTrigger<float, NO_INTERPOLATION>>(std::move(settings));
            auto&                     sink    = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            expect(fixture.connect<"out", "in">(source, trigger).has_value());
            expect(fixture.connect<"out", "in">(trigger, sink).has_value());
            expect(fixture.run().has_value());
            std::vector<std::pair<std::size_t, std::string>> found;
            for (const auto& recorded : sink._tags) {
                if (const auto name = recorded.map.value_or<std::string>(std::string(gr::tag::kTriggerName.key()), std::string{}); name == "RISING" || name == "any") {
                    found.emplace_back(recorded.index, name);
                }
            }
            return found;
        };
        const auto onlyNamed = [](const auto& edges, std::string_view name) { return std::ranges::count_if(edges, [&](const auto& edge) { return edge.second == name; }); };
        const auto both      = reported({{"edge", std::string("both")}});
        expect(eq(onlyNamed(both, "RISING"), 2) and eq(onlyNamed(both, "any"), 2)) << "a falling edge with no name of its own takes trigger_name";
        const auto risingOnly = reported({{"edge", std::string("rising")}});
        expect(eq(risingOnly.size(), 2UZ) and eq(onlyNamed(risingOnly, "RISING"), 2));
        const auto fallingOnly = reported({{"edge", std::string("falling")}});
        expect(eq(fallingOnly.size(), 2UZ) and eq(onlyNamed(fallingOnly, "any"), 2));
        const bool sameDeadTime = reported({{"dead_time", 0.005}, {"dead_time_unit", std::string("seconds")}}) == reported({{"dead_time", 5.}, {"dead_time_unit", std::string("samples")}});
        expect(sameDeadTime) << "5 ms at 1 kHz is 5 samples";
    };

    "a dead time in seconds with no sample_rate is counted and applied as none, never silently"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("lo lo hi hi lo lo hi hi lo |")}, {"sample_values", values}});
        auto&                     trigger = fixture.emplace<gr::blocks::trigger::SchmittTrigger<float, NO_INTERPOLATION>>({
            {"threshold", 2.5f},
            {"hysteresis", 2.f},
            {"dead_time", 0.005},
            {"dead_time_unit", std::string("seconds")},
        });
        auto&                     sink    = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.connect<"out", "in">(source, trigger).has_value());
        expect(fixture.connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(trigger.n_unmeasurable.value, 4U)) << "each of the four edges found its dead time unmeasurable";
    };
};

const suite<"SchmittTrigger gaps"> gapsSuite = [] {
    using enum gr::trigger::InterpolationMethod;
    using gr::blocks::trigger::MarbleSink;
    using gr::blocks::trigger::MarbleSource;

    const gr::property_map values{{"lo", 0.f}, {"hi", 5.f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("sync")}, //
                                          {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1'000'000U}},        //
                                          {std::string(gr::tag::kTriggerOffset.key()), 0.f}}},
        {"G", gr::property_map{{std::string(gr::tag::kNDroppedSamples.key()), gr::Size_t{100U}}}}};

    const auto edgesOf = [&]<gr::trigger::InterpolationMethod method>(std::string script) {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     trigger = fixture.emplace<gr::blocks::trigger::SchmittTrigger<float, method>>({
            {"threshold", 2.5f},
            {"hysteresis", 2.f},
            {"sample_rate", 1000.f},
            {"trigger_name_rising", std::string("RISING")},
            {"trigger_name_falling", std::string("FALLING")},
        });
        auto&                     sink    = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        auto&                     edges   = fixture.emplace<gr::trigger_test::EventTap>();
        expect(fixture.connect<"out", "in">(source, trigger).has_value());
        expect(fixture.connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(trigger, edges).has_value());
        expect(fixture.run().has_value());
        return std::pair{edges.names(), trigger.n_gaps.value};
    };

    std::string padded = "T:lo lo G:hi hi lo ";
    for (std::size_t i = 0UZ; i < 40UZ; ++i) {
        padded += "lo ";
    }
    padded += "|";

    "a gap between a low and a high sample is no edge: the detection restarts unknown and nothing is interpolated across it"_test = [&] {
        const auto control = edgesOf.template operator()<NO_INTERPOLATION>("T:lo lo hi hi lo |");
        expect(eq(control.first.size(), 2UZ)) << "without a gap: one rising, one falling";
        expect(eq(control.second, 0U));

        const auto plain = edgesOf.template operator()<NO_INTERPOLATION>("T:lo lo G:hi hi lo |");
        expect(plain.first == std::vector<std::string>{"FALLING"}) << "the rising edge would span the gap; the falling one is between two known samples";
        expect(eq(plain.second, 1U)) << "one counted reset";

        const auto interpolated = edgesOf.template operator()<BASIC_LINEAR_INTERPOLATION>(padded);
        expect(interpolated.first == std::vector<std::string>{"FALLING"}) << "got " << interpolated.first.size() << " edges";
        expect(eq(interpolated.second, 1U));
    };
};
} // namespace qa_schmitt_trigger

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
