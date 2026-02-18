#include <gnuradio-4.0/meta/utils.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/basic/FunctionGenerator.hpp>
#include <gnuradio-4.0/basic/StreamToDataSet.hpp>
#include <gnuradio-4.0/testing/ImChartMonitor.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <boost/ut.hpp>

const boost::ut::suite<"StreamToDataSet Block"> selectorTest = [] {
    using namespace boost::ut;

    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        // conditionally enable visual tests outside the CI
        boost::ext::ut::cfg<override> = {.tag = std::vector<std::string_view>{"visual", "benchmarks"}};
    }

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
        auto&                   clockSrc = graph.emplaceBlock<gr::basic::ClockSource<std::uint8_t>>({
            {"sample_rate", sample_rate},
            {"n_samples_max", kN_SAMPLES_MAX},
            {"name", "ClockSource"},                                                                                                                             //
            {"tag_times", Tensor<std::uint64_t>(data_from, {10 * ms, 90 * ms, 100 * ms, 300 * ms, 350 * ms, 400 * ms, 550 * ms, 650 * ms, 800 * ms, 850 * ms})}, //
            {"tag_values", Tensor<pmt::Value>{"CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1",                                                                          //
                                                 "CMD_DIAG_TRIGGER1",                                                                                                              //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2",                                                                                         //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=3",                                                                                         //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4",                                                                                         //
                                                 "CMD_DIAG_TRIGGER2",                                                                                                              //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=5",                                                                                         //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=6",                                                                                         //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=7",                                                                                         //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=8"}},
            {"repeat_period", 1000 * ms},
            {"do_zero_order_hold", true},
        });

        auto& funcGen = graph.emplaceBlock<FunctionGenerator<float>>({{"sample_rate", sample_rate}, {"signal_trigger", "CMD_BP_START"}, {"name", "FunctionGenerator"}});

        using gr::tag::TRIGGER_NAME;
        using gr::tag::CONTEXT;

        const auto now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 5.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=1"}).empty());
        expect(funcGen.settings().set(createLinearRampPropertyMap("CMD_BP_START", 5.f, 30.f, .2f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=2"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 30.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=3"}).empty());
        expect(funcGen.settings().set(createParabolicRampPropertyMap("CMD_BP_START", 30.f, 20.f, .1f, 0.02f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=4"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 20.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=5"}).empty());
        expect(funcGen.settings().set(createCubicSplinePropertyMap("CMD_BP_START", 20.f, 10.f, .1f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=6"}).empty());
        expect(funcGen.settings().set(createConstPropertyMap("CMD_BP_START", 10.f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=7"}).empty());
        expect(funcGen.settings().set(createImpulseResponsePropertyMap("CMD_BP_START", 10.f, 20.f, .02f, .06f), SettingsCtx{now, "FAIR.SELECTOR.C=1:S=1:P=8"}).empty());

        expect(eq(funcGen.settings().getNStoredParameters(), 9UZ));

        auto& sink   = graph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "SampleGeneratorSink"}});
        auto& uiSink = graph.emplaceBlock<testing::ImChartMonitor<float>>({{"name", "ImChartSinkFull"}});
        expect(eq(ConnectionResult::SUCCESS, graph.connect2(clockSrc, clockSrc.out, funcGen, funcGen.clk_in))) << "connect clockSrc->funcGen";
        expect(eq(ConnectionResult::SUCCESS, graph.connect2(funcGen, funcGen.out, sink, sink.in))) << "connect funcGen->sink";
        expect(eq(ConnectionResult::SUCCESS, graph.connect2(funcGen, funcGen.out, uiSink, uiSink.in))) << "connect funcGen->uiSink";

        const property_map blockSettings = {{"name", "StreamToDataSet"}, {"filter", filter}, {"n_pre", preSamples}, {"n_post", postSamples}, {"n_max", maxSamples}};
        //
        auto& streamFilter  = graph.emplaceBlock<StreamFilter<float>>(blockSettings);
        auto& dataSetFilter = graph.emplaceBlock<StreamToDataSet<float>>(blockSettings);
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect2(funcGen, funcGen.out, streamFilter, streamFilter.in))) << "connect funcGen->streamFilter";
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect2(funcGen, funcGen.out, dataSetFilter, dataSetFilter.in))) << "connect funcGen->dataSetFilter";

        auto& uiFilteredStreamSink = graph.emplaceBlock<testing::ImChartMonitor<float>>({{"name", "ImChartFilteredStream"}});
        expect(eq(ConnectionResult::SUCCESS, graph.connect2(streamFilter, streamFilter.out, uiFilteredStreamSink, uiFilteredStreamSink.in))) << "connect funcGen->uiFilteredStreamSink";
        auto& uiDataSetSink = graph.emplaceBlock<testing::ImChartMonitor<DataSet<float>>>({{"name", "ImChartDataSet"}});
        expect(eq(ConnectionResult::SUCCESS, graph.connect2(dataSetFilter, dataSetFilter.out, uiDataSetSink, uiDataSetSink.in))) << "connect funcGen->uiDataSetSink";

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

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value()) << std::format("runAndWait - filter {}", filter);

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
};

gr::Tag genTrigger(std::size_t index, std::string triggerName, std::string triggerCtx = {}) {
    return {index, {{gr::tag::TRIGGER_NAME.shortKey(), triggerName}, {gr::tag::TRIGGER_TIME.shortKey(), std::uint64_t(0)}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f}, //
                       {gr::tag::CONTEXT.shortKey(), triggerCtx},                                                                                                     //
                       {gr::tag::TRIGGER_META_INFO.shortKey(), gr::property_map{}}}};
};

const boost::ut::suite<"StreamToStream test"> streamToStreamTest = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    auto runTestStream = [](gr::Size_t nSamples, std::string filter, gr::Size_t preSamples, gr::Size_t postSamples, const std::vector<float>& expectedValues, std::size_t nTags) {
        constexpr float sample_rate = 1'000.f;
        Graph           graph;

        auto& tagSrc = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"sample_rate", sample_rate}, //
            {"n_samples_max", nSamples}, {"name", "TagSource"}, {"verbose_console", false}, {"repeat_tags", false}, {"mark_tag", false}});
        tagSrc._tags = {
            genTrigger(5, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=1"),  // start
            genTrigger(8, "CMD_DIAG_TRIGGER1", ""),                      // it is also used to split samples processing into 2 iterations
            genTrigger(10, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=2"), // stop
            genTrigger(12, "CMD_DIAG_TRIGGER1", ""),                     // it is also used as end trigger for "including" mode
            genTrigger(15, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=1"), // start
            genTrigger(20, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=2"), // stop
            genTrigger(22, "CMD_DIAG_TRIGGER1", "")                      // it is also used as end trigger for "including" mode
        };

        const property_map blockSettings        = {{"filter", filter}, {"n_pre", preSamples}, {"n_post", postSamples}};
        auto&              filterStreamToStream = graph.emplaceBlock<StreamFilter<float>>(blockSettings);
        auto&              streamSink           = graph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "streamSink"}, {"log_tags", true}, {"log_samples", true}, {"verbose_console", false}});
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect2(tagSrc, tagSrc.out, filterStreamToStream, filterStreamToStream.in)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect2(filterStreamToStream, filterStreamToStream.out, streamSink, streamSink.in)));

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        std::println("start -> Stream-to-Stream with filter: {} n_pre:{} n_post:{}", filter, preSamples, postSamples);
        expect(sched.runAndWait().has_value()) << std::format("runAndWait - filter {}", filter);
        std::println("done -> Stream-to-Stream with filter: {} n_pre:{} n_post:{}", filter, preSamples, postSamples);

        expect(eq(tagSrc.sample_rate, sample_rate));
        expect(eq(filterStreamToStream.sample_rate, sample_rate));
        expect(eq(streamSink.sample_rate, sample_rate));

        expect(eq(streamSink._samples.size(), expectedValues.size()));
        expect(std::ranges::equal(streamSink._samples, expectedValues));

        expect(eq(streamSink._tags.size(), nTags)) << [&]() {
            std::string ret = std::format("Stream nTags: {}\n", streamSink._tags.size());
            for (const auto& tag : streamSink._tags) {
                ret += std::format("tag.index: {} .map: {}\n", tag.index, tag.map);
            }
            return ret;
        }();
    };
    // We always test scenarios where no overlaps occur. If accumulation is currently active in the block, no new "Start" should happen.
    // Any new Start events are ignored, and this behavior is considered undefined for stream-to-stream mode
    std::vector<float> expectedValues                = {5, 6, 7, 8, 9, 15, 16, 17, 18, 19};
    "start->stop matcher (excluding)"_test           = [&runTestStream, &expectedValues] { runTestStream(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, 3UZ /* 2 BPs (2 starts) + custom diag trigger */); };
    expectedValues                                   = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    "start->stop matcher (excluding +pre/post)"_test = [&runTestStream, &expectedValues] { runTestStream(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 2, 2, expectedValues, 5UZ /* 4 BPs (2 starts + 2 stops) + custom diag trigger */); };

    expectedValues                                     = {5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21};
    "start->^stop matcher (including)"_test            = [&runTestStream, &expectedValues] { runTestStream(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, 5UZ /* 4 BPs (2 starts + 2 stops) + custom diag trigger */); };
    expectedValues                                     = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    "start->^stop matcher (including. +pre/post)"_test = [&runTestStream, &expectedValues] { runTestStream(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 2, 2, expectedValues, 7UZ /* 4 BPs (2 starts + 2 stops) + 3 custom diag event */); };

    expectedValues                    = {6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23};
    "single trigger (+pre/post)"_test = [&runTestStream, &expectedValues] { runTestStream(50U, "CMD_DIAG_TRIGGER1", 2, 2, expectedValues, 3UZ); };
};

const boost::ut::suite<"StreamToDataSet test"> streamToDataSetTest = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    auto runTestDataSet = [](gr::Size_t nSamples, std::string filter, gr::Size_t preSamples, gr::Size_t postSamples, const std::vector<std::vector<float>>& expectedValues, const std::vector<std::size_t>& nTags, gr::Size_t maxSamples = 100000U) {
        using namespace gr;
        using namespace gr::basic;
        using namespace gr::testing;

        constexpr float sample_rate = 1'000.f;
        Graph           graph;

        auto& tagSrc = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"sample_rate", sample_rate}, //
            {"n_samples_max", nSamples}, {"name", "TagSource"}, {"verbose_console", false}, {"repeat_tags", false}, {"mark_tag", false}});
        tagSrc._tags = {
            genTrigger(5, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=1"),  // start
            genTrigger(8, "CMD_DIAG_TRIGGER1", ""),                      // it is also used to split samples processing into 2 iterations
            genTrigger(10, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=2"), // stop
            genTrigger(12, "CMD_DIAG_TRIGGER1", ""),                     // it is also used as end trigger for "including" mode
            genTrigger(15, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=1"), // start
            genTrigger(20, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=1"), // start
            genTrigger(25, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=2"), // stop
            genTrigger(27, "CMD_DIAG_TRIGGER1", ""),                     // it is also used as end trigger for "including" mode
            genTrigger(30, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=2"), // stop
            genTrigger(32, "CMD_DIAG_TRIGGER1", "")                      // it is also used as end trigger for "including" mode
        };

        const property_map blockSettings         = {{"filter", filter}, {"n_pre", preSamples}, {"n_post", postSamples}, {"n_max", maxSamples}};
        auto&              filterStreamToDataSet = graph.emplaceBlock<StreamToDataSet<float>>(blockSettings);
        auto&              dataSetSink           = graph.emplaceBlock<TagSink<DataSet<float>, ProcessFunction::USE_PROCESS_BULK>>({{"name", "dataSetSink"}, {"log_tags", true}, {"log_samples", true}, {"verbose_console", false}});
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect2(tagSrc, tagSrc.out, filterStreamToDataSet, filterStreamToDataSet.in)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect2(filterStreamToDataSet, filterStreamToDataSet.out, dataSetSink, dataSetSink.in)));

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        std::println("start -> Stream-to-DataSet with filter: {} n_pre:{} n_post:{}", filter, preSamples, postSamples);
        expect(sched.runAndWait().has_value()) << std::format("runAndWait - filter {}", filter);
        std::println("done -> Stream-to-DataSet with filter: {} n_pre:{} n_post:{}", filter, preSamples, postSamples);

        expect(eq(tagSrc.sample_rate, sample_rate));
        expect(eq(filterStreamToDataSet.sample_rate, sample_rate));
        expect(eq(dataSetSink.sample_rate, sample_rate));

        expect(eq(dataSetSink._samples.size(), expectedValues.size()));
        for (std::size_t i = 0; i < dataSetSink._samples.size(); i++) {
            const DataSet<float>&          ds      = dataSetSink._samples.at(i);
            std::expected<void, gr::Error> dsCheck = dataset::checkConsistency(ds, "TestDataSet");
            expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
            expect(std::ranges::equal(ds.signal_values, expectedValues[i]));

            expect(fatal(eq(ds.timing_events.size(), 1UZ)));
            const auto& timingEvt0 = ds.timing_events[0];
            expect(eq(timingEvt0.size(), nTags[i])) << [&]() {
                std::string ret = std::format("DataSet nTags: {}\n", timingEvt0.size());
                for (const auto& tag : timingEvt0) {
                    ret += std::format("tag.index: {} .map: {}\n", tag.first, tag.second);
                }
                return ret;
            }();
        }
    };

    std::vector<std::vector<float>> expectedValues = {{5, 6, 7, 8, 9}, {15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}};
    "start->stop (excluding)"_test                 = [&runTestDataSet, &expectedValues] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, {3UZ, 3UZ, 4UZ}); };
    "start->stop (excluding) n_max=0"_test         = [&runTestDataSet, &expectedValues] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, {3UZ, 3UZ, 4UZ}, 0UZ); };

    expectedValues                           = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},                       //
                                  {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, //
                                  {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}};
    "start->stop (excluding +pre/post)"_test = [&runTestDataSet, &expectedValues] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 7, 7, expectedValues, {5UZ, 5UZ, 5UZ}); };

    expectedValues                          = {{5, 6, 7, 8, 9, 10, 11}, {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}, {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}};
    "start->^stop (including)"_test         = [&runTestDataSet, &expectedValues] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, {4UZ, 4UZ, 5UZ}); };
    "start->^stop (including) n_max=0"_test = [&runTestDataSet, &expectedValues] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, {4UZ, 4UZ, 5UZ}, 0UZ); };

    expectedValues                             = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},                       //
                                    {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}, //
                                    {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}};
    "start->^stop (including. +pre/post)"_test = [&runTestDataSet, &expectedValues] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 7, 7, expectedValues, {5UZ, 6UZ, 5UZ}); };

    expectedValues                    = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, //
                           {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},           //
                           {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33},      //
                           {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}};
    "single trigger (+pre/post)"_test = [&runTestDataSet, &expectedValues] { runTestDataSet(50U, "CMD_DIAG_TRIGGER1", 7, 7, expectedValues, {3UZ, 2UZ, 3UZ, 1UZ}); };

    // n_max test
    gr::Size_t nMaxSamples                = 6;
    expectedValues                        = {{5, 6, 7, 8, 9}, {15, 16, 17, 18, 19, 20}, {20, 21, 22, 23, 24, 25}};
    "start->stop (excluding, n_max)"_test = [&runTestDataSet, &expectedValues, &nMaxSamples] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, {3UZ, 2UZ, 2UZ}, nMaxSamples); };

    nMaxSamples                                     = 14;
    expectedValues                                  = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}, {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}};
    "start->stop (excluding +pre/post, n_max)"_test = [&runTestDataSet, &expectedValues, &nMaxSamples] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 7, 7, expectedValues, {4UZ, 2UZ, 2UZ}, nMaxSamples); };

    nMaxSamples                            = 6;
    expectedValues                         = {{5, 6, 7, 8, 9, 10}, {15, 16, 17, 18, 19, 20}, {20, 21, 22, 23, 24, 25}};
    "start->^stop (including, n_max)"_test = [&runTestDataSet, &expectedValues, &nMaxSamples] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, {3UZ, 2UZ, 2UZ}, nMaxSamples); };

    nMaxSamples                                       = 14;
    expectedValues                                    = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}, {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}};
    "start->^stop (including. +pre/post, n_max)"_test = [&runTestDataSet, &expectedValues, &nMaxSamples] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 7, 7, expectedValues, {4UZ, 2UZ, 2UZ}, nMaxSamples); };

    nMaxSamples                              = 14;
    expectedValues                           = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, //
                                  {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},           //
                                  {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33},      //
                                  {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}};
    "single trigger (+pre/post, n_max)"_test = [&runTestDataSet, &expectedValues, &nMaxSamples] { runTestDataSet(50U, "CMD_DIAG_TRIGGER1", 7, 7, expectedValues, {3UZ, 2UZ, 3UZ, 1UZ}, nMaxSamples); };
};

int main() { /* not needed for UT */ }
