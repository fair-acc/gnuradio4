#include <gnuradio-4.0/meta/utils.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/basic/FunctionGenerator.hpp>
#include <gnuradio-4.0/basic/StreamToDataSet.hpp>
#include <gnuradio-4.0/basic/clock_source.hpp>
#include <gnuradio-4.0/testing/ImChartMonitor.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <boost/ut.hpp>

const boost::ut::suite<"StreamToDataSet Block"> selectorTest = [] {
    using namespace boost::ut;

    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        // conditionally enable visual tests outside the CI
        boost::ext::ut::cfg<override> = {.tag = std::vector<std::string_view>{"visual", "benchmarks"}};
    }

    constexpr static auto funcMatcher = [](const gr::property_map& tableEntry, const gr::property_map& searchEntry,
                                            const std::size_t attempt) -> std::optional<bool> { // TODO: refactor FunctionGenerator to use TriggerMatcher
        using namespace gr;

        if (attempt > 0) {
            return std::nullopt;
        }
        if (!searchEntry.contains(tag::CONTEXT.shortKey())) {
            return std::nullopt;
        }
        if (!tableEntry.contains(tag::CONTEXT.shortKey())) {
            throw gr::exception(fmt::format("config error: tableEntry: {} does not contain {} key", tableEntry, tag::CONTEXT.shortKey()));
        }

        [[maybe_unused]] bool triggerNameEnds;
        [[maybe_unused]] bool ctxNameEnds;
        std::string           searchEvent;
        std::string           searchContext;
        const auto            searchEntryContext = std::get<std::string>(searchEntry.at(tag::CONTEXT.shortKey()));
        trigger::detail::parse(searchEntryContext, searchEvent, triggerNameEnds, searchContext, ctxNameEnds);

        const auto  tableEntryContext = std::get<std::string>(tableEntry.at(tag::CONTEXT.shortKey()));
        std::string triggerEvent;
        std::string triggerContext;
        trigger::detail::parse(tableEntryContext, triggerEvent, triggerNameEnds, triggerContext, ctxNameEnds);

        if (searchEvent != triggerEvent) {
            return false;
        }
        if (triggerContext.empty() || searchContext == triggerContext) {
            return true;
        }
        return false;
    };

    auto runUIExample = [](std::uint32_t lengthSeconds, std::string filter, gr::Size_t preSamples = 0U, gr::Size_t postSamples = 0U, gr::Size_t maxSamples = 10000U) {
        using namespace gr;
        using namespace gr::basic;
        using namespace gr::testing;

        using namespace function_generator;
        const std::uint32_t kN_SAMPLES_MAX = 1000U * lengthSeconds;
        constexpr float     sample_rate    = 1'000.f;

        Graph graph;

        // all times are in nanoseconds
        constexpr std::uint64_t ms       = 1'000'000; // ms -> ns conversion factor (wish we had a proper C++ units-lib integration)
        auto&                   clockSrc = graph.emplaceBlock<gr::basic::ClockSource<float>>({
            {"sample_rate", sample_rate},
            {"n_samples_max", kN_SAMPLES_MAX},
            {"name", "ClockSource"},                                                                                                           //
            {"tag_times", std::vector<std::uint64_t>{10 * ms, 90 * ms, 100 * ms, 300 * ms, 350 * ms, 550 * ms, 650 * ms, 800 * ms, 850 * ms}}, //
            {"tag_values", std::vector<std::string>{"CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1", "CMD_DIAG_TRIGGER1", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=3", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=5", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=6", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=7", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=8"}},
            {"repeat_period", std::uint64_t(1000 * ms)},
            {"do_zero_order_hold", true},
        });

        auto& funcGen      = graph.emplaceBlock<FunctionGenerator<float>>({{"sample_rate", sample_rate}, {"name", "FunctionGenerator"}});
        funcGen.match_pred = funcMatcher;

        using gr::tag::TRIGGER_NAME;
        using gr::tag::CONTEXT;

        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1"}}, createConstPropertyMap(5.f));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2"}}, createLinearRampPropertyMap(5.f, 30.f, 0.2f /* [s]*/));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=3"}}, createConstPropertyMap(30.f));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4"}}, createParabolicRampPropertyMap(30.f, 20.f, .1f, 0.02f /* [s]*/));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=5"}}, createConstPropertyMap(20.f));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=6"}}, createCubicSplinePropertyMap(20.f, 10.f, 0.1f /* [s]*/));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=7"}}, createConstPropertyMap(10.f));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=8"}}, createImpulseResponsePropertyMap(10.f, 20.f, 0.02f /* [s]*/, 0.06f /* [s]*/));

        auto& sink   = graph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "SampleGeneratorSink"}});
        auto& uiSink = graph.emplaceBlock<testing::ImChartMonitor<float>>({{"name", "ImChartSinkFull"}});
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(clockSrc).to<"in">(funcGen))) << "connect clockSrc->funcGen";
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(funcGen).to<"in">(sink))) << "connect funcGen->sink";
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(funcGen).to<"in">(uiSink))) << "connect funcGen->uiSink";

        const property_map blockSettings = {{"name", "StreamToDataSet"}, {"filter", filter}, {"n_pre", preSamples}, {"n_post", postSamples}, {"n_max", maxSamples}};
        //
        auto& streamFilter  = graph.emplaceBlock<StreamFilter<float>>(blockSettings);
        auto& dataSetFilter = graph.emplaceBlock<StreamToDataSet<float>>(blockSettings);
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(funcGen).template to<"in">(streamFilter))) << "connect funcGen->streamFilter";
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(funcGen).template to<"in">(dataSetFilter))) << "connect funcGen->dataSetFilter";

        auto& uiFilteredStreamSink = graph.emplaceBlock<testing::ImChartMonitor<float>>({{"name", "ImChartFilteredStream"}});
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(streamFilter).to<"in">(uiFilteredStreamSink))) << "connect funcGen->uiFilteredStreamSink";
        auto& uiDataSetSink = graph.emplaceBlock<testing::ImChartMonitor<DataSet<float>>>({{"name", "ImChartDataSet"}});
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(dataSetFilter).to<"in">(uiDataSetSink))) << "connect funcGen->uiDataSetSink";

        std::thread uiLoop([&uiSink, &uiFilteredStreamSink, &uiDataSetSink]() {
            bool drawUI = true;
            while (drawUI) { // mocks UI update loop
                using enum gr::work::Status;
                drawUI = false;
                drawUI |= uiSink.draw({{"reset_view", true}}) != DONE;
                drawUI |= uiFilteredStreamSink.draw() != DONE;
                drawUI |= uiDataSetSink.draw() != DONE;

                std::this_thread::sleep_for(std::chrono::milliseconds(40));
            }
            std::this_thread::sleep_for(std::chrono::seconds(1)); // wait for another second before closing down
        });

        gr::scheduler::Simple sched{std::move(graph)};
        expect(sched.runAndWait().has_value()) << fmt::format("runAndWait - filter {}", filter);

        expect(eq(clockSrc.sample_rate, sample_rate));
        expect(eq(funcGen.sample_rate, sample_rate));
        expect(eq(sink.sample_rate, sample_rate));
        expect(eq(uiSink.sample_rate, sample_rate));
        expect(eq(streamFilter.sample_rate, sample_rate));
        expect(eq(dataSetFilter.sample_rate, sample_rate));
        expect(eq(uiFilteredStreamSink.sample_rate, sample_rate));
        expect(eq(uiDataSetSink.sample_rate, sample_rate));

        uiLoop.join();
    };

    tag("visual") / "default example"_test             = [&runUIExample] { runUIExample(5, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4]", 0, 0); };
    tag("visual") / "default ^matcher"_test            = [&runUIExample] { runUIExample(5, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=3]", 0, 0); };
    tag("visual") / "default ^matcher + pre/post"_test = [&runUIExample] { runUIExample(5, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=3]", 30, 30); };
    tag("visual") / "single trigger"_test              = [&runUIExample] { runUIExample(10, "CMD_DIAG_TRIGGER1", 30, 30); };

    auto runTest = [](std::string filter, gr::Size_t preSamples, gr::Size_t postSamples, std::array<float, 2> expectedValues, std::size_t nTags, gr::Size_t maxSamples = 10000U) {
        using namespace gr;
        using namespace gr::basic;
        using namespace gr::testing;
        using gr::tag::TRIGGER_NAME;
        using gr::tag::CONTEXT;

        using namespace function_generator;
        constexpr gr::Size_t N_SAMPLES   = 402U;
        constexpr float      sample_rate = 1'000.f;

        Graph graph;

        // all times are in nanoseconds
        constexpr std::uint64_t ms       = 1'000'000;                                                                                             // ms -> ns conversion factor (wish we had a proper C++ units-lib integration)
        auto&                   clockSrc = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"sample_rate", sample_rate}, //
                              {"n_samples_max", N_SAMPLES},                                                                                       //
                              {"name", "TagSource"},                                                                                              //
                              {"verbose_console", false},                                                                                         //
                              {"repeat_tags", true}});

        auto genTrigger = [](Tag::signed_index_type index, std::string triggerName, std::string triggerCtx = {}) -> Tag {
            return {index, {{tag::TRIGGER_NAME.shortKey(), triggerName},         //
                               {tag::TRIGGER_TIME.shortKey(), std::uint64_t(0)}, //
                               {tag::TRIGGER_OFFSET.shortKey(), 0.f},            //
                               {tag::TRIGGER_META_INFO.shortKey(), property_map{{tag::CONTEXT.shortKey(), triggerCtx}}}}};
        };

        clockSrc._tags = { //
            // TODO: refactor triggerCtx do only contain the 'FAIR.SELECTOR...' info (-> needs changes in FunctionGenerator)
            genTrigger(10, "CMD_BP_START", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1"),  //
            genTrigger(50, "CMD_DIAG_TRIGGER1", "CMD_DIAG_TRIGGER1"),                  //
            genTrigger(100, "CMD_BP_START", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2"), //
            genTrigger(200, "CMD_BP_START", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=3"), //
            genTrigger(300, "CMD_BP_START", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4"), //
            genTrigger(400, "CMD_BP_START", "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=5")};

        auto& funcGen      = graph.emplaceBlock<FunctionGenerator<float>>({{"sample_rate", sample_rate}, {"name", "FunctionGenerator"}});
        funcGen.match_pred = funcMatcher;
        // TODO: add graphic
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1"}}, createConstPropertyMap(1.f));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2"}}, createConstPropertyMap(2.f));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=3"}}, createConstPropertyMap(3.f));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4"}}, createConstPropertyMap(4.f));
        funcGen.addFunctionTableEntry({{CONTEXT.shortKey(), "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=5"}}, createConstPropertyMap(5.f));
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(clockSrc).to<"in">(funcGen))) << "connect clockSrc->funcGen";

        const property_map blockSettings = {{"filter", filter}, {"n_pre", preSamples}, {"n_post", postSamples}, {"n_max", maxSamples}};
        // producing stream (filtered)
        auto& filterStreamToStream = graph.emplaceBlock<StreamFilter<float>>(blockSettings);
        auto& streamSink           = graph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "streamSink"}, {"log_tags", true}, {"log_samples", true}, {"n_samples_expected", N_SAMPLES}, {"verbose_console", false}});
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(funcGen).template to<"in">(filterStreamToStream))) << "connect funcGen->filterStreamToStream";
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(filterStreamToStream).template to<"in">(streamSink))) << "connect filterStreamToStream->streamSink";
        // producing DataSet (filtered)
        auto& filterStreamToDataSet = graph.emplaceBlock<StreamToDataSet<float>>(blockSettings);
        auto& dataSetSink           = graph.emplaceBlock<TagSink<DataSet<float>, ProcessFunction::USE_PROCESS_BULK>>({{"name", "dataSetSink"}, {"log_tags", true}, {"log_samples", true}, {"n_samples_expected", gr::Size_t(1)}, {"verbose_console", false}});
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(funcGen).template to<"in">(filterStreamToDataSet))) << "connect funcGen->filterStreamToDataSet";
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(filterStreamToDataSet).template to<"in">(dataSetSink))) << "connect filterStreamToDataSet->dataSetSink";

        gr::scheduler::Simple sched{std::move(graph)};
        fmt::println("start test with filter: {}", filter);
        expect(sched.runAndWait().has_value()) << fmt::format("runAndWait - filter {}", filter);
        fmt::println("start test with filter: {} -- DONE", filter);

        expect(eq(clockSrc.sample_rate, sample_rate)) << "clockSrc.sample_rate";
        expect(eq(funcGen.sample_rate, sample_rate)) << "funcGen.sample_rate";
        expect(eq(filterStreamToStream.sample_rate, sample_rate)) << "filterStreamToStream.sample_rate";
        expect(eq(filterStreamToDataSet.sample_rate, sample_rate)) << "filterStreamToDataSet.sample_rate";
        expect(eq(streamSink.sample_rate, sample_rate)) << "streamSink.sample_rate";
        expect(eq(dataSetSink.sample_rate, sample_rate)) << "dataSetSink.sample_rate";

        expect(!streamSink._samples.empty()) << "streamSink._samples.empty()";
        expect(eq(streamSink._samples.front(), expectedValues.front())) << fmt::format("streamSink._samples - first sample does not match({}): {}", streamSink._samples.size(), fmt::join(streamSink._samples, ", "));
        expect(eq(streamSink._samples.back(), expectedValues.back())) << fmt::format("streamSink._samples - last sample does not match({}): {}", streamSink._samples.size(), fmt::join(streamSink._samples, ", "));

        expect(eq(streamSink._tags.size(), nTags)) << [&]() {
            std::string ret = fmt::format("Stream nTags: {}\n", streamSink._tags.size());
            for (const auto& tag : streamSink._tags) {
                ret += fmt::format("tag.index: {} .map: {}\n", tag.index, tag.map);
            }
            return ret;
        }();

        expect(!dataSetSink._samples.empty()) << "dataSetSink did not receive the required minimum data";
        if (dataSetSink._samples.empty()) {
            return;
        }
        const DataSet<float>& dataSet = dataSetSink._samples.at(0UZ);
        expect(ge(streamSink._samples.size(), dataSet.signal_values.size())) << "streamSink needs to receive correct amount of samples";

        expect(fatal(!dataSet.signal_values.empty())) << "no samples in DataSet";
        expect(eq(dataSet.signal_values.front(), expectedValues.front())) << fmt::format("dataSet.signal_values - first sample does not match ({}): {}", dataSet.signal_values.size(), fmt::join(dataSet.signal_values, ", "));
        expect(eq(dataSet.signal_values.back(), expectedValues.back())) << fmt::format("dataSet.signal_values - last sample does not match({}): {}", dataSet.signal_values.size(), fmt::join(dataSet.signal_values, ", "));
        expect(fatal(eq(dataSet.timing_events.size(), 1UZ))) << "dataSetSink._samples[0] -> DataSet - timing_events match";
        const std::vector<Tag>& timingEvt0 = dataSet.timing_events[0];
        expect(eq(timingEvt0.size(), nTags)) << [&]() {
            std::string ret = fmt::format("DataSet nTags: {}\n", timingEvt0.size());
            for (const auto& tag : timingEvt0) {
                ret += fmt::format("tag.index: {} .map: {}\n", tag.index, tag.map);
            }
            return ret;
        }();
    };

    "start->stop matcher (excluding)"_test             = [&runTest] { runTest("[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4]", 0, 0, {1.f, 3.f}, 4UZ /* 3 BPs + custom diag event */); };
    "start->^stop matcher (including)"_test            = [&runTest] { runTest("[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=3]", 0, 0, {1.f, 3.f}, 4UZ /* 3 BPs + custom diag event */); };
    "start->^stop matcher (including. +pre/post)"_test = [&runTest] { runTest("[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=3]", 30, 30, {0.f, 4.f}, 5UZ /* 3+1 BPs (because of +30 post samples ranging into P=4) + custom diag event */); };
    "single trigger (+pre/post)"_test                  = [&runTest] { runTest("CMD_DIAG_TRIGGER1", 30, 30, {1.f, 1.f}, 1UZ); };
};

int main() { /* not needed for UT */ }
