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

#include <initializer_list>
#include <limits>
#include <source_location>
#include <string_view>

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
            {"name", "ClockSource"},                                                                                                                     //
            {"tag_times", std::vector<std::uint64_t>{10 * ms, 90 * ms, 100 * ms, 300 * ms, 350 * ms, 400 * ms, 550 * ms, 650 * ms, 800 * ms, 850 * ms}}, //
            {"tag_values", std::vector<std::string>{"CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1",                                                            //
                                                 "CMD_DIAG_TRIGGER1",                                                                                                      //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2",                                                                                 //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=3",                                                                                 //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=4",                                                                                 //
                                                 "CMD_DIAG_TRIGGER2",                                                                                                      //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=5",                                                                                 //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=6",                                                                                 //
                                                 "CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=7",                                                                                 //
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
        expect(graph.connect<"out", "clk_in">(clockSrc, funcGen).has_value()) << "connect clockSrc->funcGen";
        expect(graph.connect<"out", "in">(funcGen, sink).has_value()) << "connect funcGen->sink";
        expect(graph.connect<"out", "in">(funcGen, uiSink).has_value()) << "connect funcGen->uiSink";

        const property_map blockSettings = {{"name", "StreamToDataSet"}, {"filter", filter}, {"n_pre", preSamples}, {"n_post", postSamples}, {"n_max", maxSamples}};
        //
        auto& streamFilter  = graph.emplaceBlock<StreamFilter<float>>(blockSettings);
        auto& dataSetFilter = graph.emplaceBlock<StreamToDataSet<float>>(blockSettings);
        expect(graph.connect<"out", "in">(funcGen, streamFilter).has_value()) << "connect funcGen->streamFilter";
        expect(graph.connect<"out", "in">(funcGen, dataSetFilter).has_value()) << "connect funcGen->dataSetFilter";

        auto& uiFilteredStreamSink = graph.emplaceBlock<testing::ImChartMonitor<float>>({{"name", "ImChartFilteredStream"}});
        expect(graph.connect<"out", "in">(streamFilter, uiFilteredStreamSink).has_value()) << "connect funcGen->uiFilteredStreamSink";
        auto& uiDataSetSink = graph.emplaceBlock<testing::ImChartMonitor<DataSet<float>>>({{"name", "ImChartDataSet"}});
        expect(graph.connect<"out", "in">(dataSetFilter, uiDataSetSink).has_value()) << "connect funcGen->uiDataSetSink";

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

inline std::uint64_t gTriggerTimeCounter = 0UZ;
inline std::uint64_t nextTriggerTime() { return gTriggerTimeCounter++; }
inline void          resetTriggerTime(std::uint64_t v = 0UZ) { gTriggerTimeCounter = v; }

gr::Tag genTrigger(std::size_t index, std::string triggerName, std::string triggerCtx, std::uint64_t triggerTime) {
    return {index, {{gr::tag::TRIGGER_NAME.shortKey(), triggerName}, {gr::tag::TRIGGER_TIME.shortKey(), triggerTime}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f}, //
                       {gr::tag::CONTEXT.shortKey(), triggerCtx},                                                                                                //
                       {gr::tag::TRIGGER_META_INFO.shortKey(), gr::property_map{}}}};
};

gr::Tag genTrigger(std::size_t index, std::string triggerName, std::string triggerCtx = {}) { return genTrigger(index, std::move(triggerName), std::move(triggerCtx), nextTriggerTime()); }

gr::Tag genStart(std::size_t index, std::uint64_t triggerTime = std::numeric_limits<std::uint64_t>::max()) { return genTrigger(index, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=1", triggerTime == std::numeric_limits<std::uint64_t>::max() ? nextTriggerTime() : triggerTime); }

gr::Tag genStop(std::size_t index, std::uint64_t triggerTime = std::numeric_limits<std::uint64_t>::max()) { return genTrigger(index, "CMD_BP_START", "FAIR.SELECTOR.C=1:S=1:P=2", triggerTime == std::numeric_limits<std::uint64_t>::max() ? nextTriggerTime() : triggerTime); }

gr::Tag genSingle(std::size_t index, std::uint64_t triggerTime = std::numeric_limits<std::uint64_t>::max()) { return genTrigger(index, "CMD_DIAG_TRIGGER1", "", triggerTime == std::numeric_limits<std::uint64_t>::max() ? nextTriggerTime() : triggerTime); }

gr::Tag genNoTrigger(std::size_t index, std::uint64_t triggerTime = std::numeric_limits<std::uint64_t>::max()) { return genTrigger(index, "NO_TRIGGER", "", triggerTime == std::numeric_limits<std::uint64_t>::max() ? nextTriggerTime() : triggerTime); }

gr::Tag sampleRateTag(std::size_t index) { return {index, {{"sample_rate", 1000.f}}}; }

gr::Tag mergedAutoForwardTag(std::size_t index, std::initializer_list<gr::Tag> tags) {
    gr::property_map merged;
    for (const gr::Tag& tag : tags) {
        for (const auto& [key, value] : tag.map) {
            merged.insert_or_assign(key, value);
        }
    }
    return {index, std::move(merged)};
}

void printTagList(std::string_view label, const std::vector<gr::Tag>& tags) {
    std::println("{} tags ({}):", label, tags.size());
    for (const gr::Tag& tag : tags) {
        gr::testing::print_tag(tag, "  ");
    }
}

void runTestStream(gr::Size_t nSamples, std::string filter, gr::Size_t preSamples, gr::Size_t postSamples, const std::vector<float>& expectedValues, const std::vector<gr::Tag>& expectedTags, std::source_location srcLocation = std::source_location::current()) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    const auto locationStr = std::format("{}:{} ", srcLocation.file_name(), srcLocation.line());

    constexpr float sampleRate = 1'000.f;
    Graph           graph;

    auto& tagSrc = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"sample_rate", sampleRate}, //
        {"n_samples_max", nSamples}, {"name", "TagSource"}, {"verbose_console", false}, {"repeat_tags", false}, {"mark_tag", false}});

    resetTriggerTime();
    tagSrc._tags = {genNoTrigger(2), genSingle(4), genStart(5), genSingle(8), genStop(10), genSingle(12), genStart(15), genStop(20), genSingle(22)};

    const property_map blockSettings        = {{"filter", filter}, {"n_pre", preSamples}, {"n_post", postSamples}};
    auto&              filterStreamToStream = graph.emplaceBlock<StreamFilter<float>>(blockSettings);
    auto&              streamSink           = graph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "streamSink"}, {"log_tags", true}, {"log_samples", true}, {"verbose_console", false}});
    expect(graph.connect<"out", "in">(tagSrc, filterStreamToStream).has_value()) << locationStr;
    expect(graph.connect<"out", "in">(filterStreamToStream, streamSink).has_value()) << locationStr;

    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    std::println("start -> Stream-to-Stream with filter: {} n_pre:{} n_post:{}", filter, preSamples, postSamples);
    expect(sched.runAndWait().has_value()) << std::format("runAndWait - filter {}", filter) << locationStr;
    std::println("done -> Stream-to-Stream with filter: {} n_pre:{} n_post:{}", filter, preSamples, postSamples);

    expect(eq(tagSrc.sample_rate, sampleRate)) << locationStr;
    expect(eq(filterStreamToStream.sample_rate, sampleRate)) << locationStr;
    expect(eq(streamSink.sample_rate, sampleRate)) << locationStr;

    expect(eq(streamSink._samples.size(), expectedValues.size())) << locationStr;
    expect(std::ranges::equal(streamSink._samples, expectedValues)) << locationStr;
    const bool tagsMatch = equal_tag_lists(streamSink._tags, expectedTags);
    if (!tagsMatch) {
        printTagList("actual", streamSink._tags);
        printTagList("expected", expectedTags);
    }
    expect(tagsMatch) << locationStr;
};

const boost::ut::suite<"StreamToStream test"> streamToStreamTest = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    // We always test scenarios where no overlaps occur. If accumulation is currently active in the block, no new "Start" should happen.
    // Any new Start events are ignored, and this behavior is considered undefined for stream-to-stream mode

    "start->stop matcher (excluding)"_test = [] {
        const std::vector<float> expectedValues = {5, 6, 7, 8, 9, 15, 16, 17, 18, 19};
        resetTriggerTime();
        const std::vector<Tag> expectedTags = {
            mergedAutoForwardTag(0, {sampleRateTag(0), genNoTrigger(0), genSingle(0)}),
            genStart(0),
            genSingle(3),
            mergedAutoForwardTag(5, {genStop(5), genSingle(5)}),
            genStart(5),
        };
        runTestStream(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, expectedTags);
    };

    "start->stop matcher (excluding +pre/post)"_test = [] {
        const std::vector<float> expectedValues = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21};
        resetTriggerTime();
        const std::vector<Tag> expectedTags = {
            mergedAutoForwardTag(0, {sampleRateTag(0), genNoTrigger(0)}),
            genSingle(1),
            genStart(2),
            genSingle(5),
            genStop(7),
            genSingle(9),
            genStart(11),
            genStop(16),
        };
        runTestStream(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 2, 2, expectedValues, expectedTags);
    };

    "start->^stop matcher (including)"_test = [] {
        const std::vector<float> expectedValues = {5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21};
        resetTriggerTime();
        const std::vector<Tag> expectedTags = {
            mergedAutoForwardTag(0, {sampleRateTag(0), genNoTrigger(0), genSingle(0)}),
            genStart(0),
            genSingle(3),
            genStop(5),
            genSingle(7),
            genStart(7),
            genStop(12),
        };
        runTestStream(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, expectedTags);
    };

    "start->^stop matcher (including. +pre/post)"_test = [] {
        const std::vector<float> expectedValues = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
        resetTriggerTime();
        const std::vector<Tag> expectedTags = {
            mergedAutoForwardTag(0, {sampleRateTag(0), genNoTrigger(0)}),
            genSingle(1),
            genStart(2),
            genSingle(5),
            genStop(7),
            genSingle(9),
            genStart(13),
            genStop(18),
            genSingle(20),
        };
        runTestStream(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 2, 2, expectedValues, expectedTags);
    };

    "single trigger (+pre/post)"_test = [] {
        const std::vector<float> expectedValues = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23};
        resetTriggerTime();
        const std::vector<Tag> expectedTags = {
            mergedAutoForwardTag(0, {sampleRateTag(0)}),
            genNoTrigger(0),
            genSingle(2),
            genStart(3),
            genSingle(6),
            genStop(8),
            genSingle(10),
            mergedAutoForwardTag(12, {genStart(12)}),
            genStop(12),
            genSingle(14),
        };
        runTestStream(50U, "CMD_DIAG_TRIGGER1", 2, 2, expectedValues, expectedTags);
    };
};

void runTestDataSet(gr::Size_t nSamples, std::string filter, gr::Size_t preSamples, gr::Size_t postSamples, const std::vector<std::vector<float>>& expectedValues, const std::vector<std::vector<gr::Tag>>& expectedTags, gr::Size_t maxSamples = 100000U, std::source_location srcLocation = std::source_location::current()) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    const auto locationStr = std::format("{}:{} ", srcLocation.file_name(), srcLocation.line());

    constexpr float sampleRate = 1'000.f;
    Graph           graph;

    auto& tagSrc = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"sample_rate", sampleRate}, //
        {"n_samples_max", nSamples}, {"name", "TagSource"}, {"verbose_console", false}, {"repeat_tags", false}, {"mark_tag", false}});

    resetTriggerTime();
    tagSrc._tags = {genNoTrigger(2), genSingle(4), genNoTrigger(5), genStart(5), genSingle(8), genStop(10), genSingle(12), genStart(15), genStart(20), genStop(25), genSingle(27), genStop(30), genSingle(32)};

    const property_map blockSettings         = {{"filter", filter}, {"n_pre", preSamples}, {"n_post", postSamples}, {"n_max", maxSamples}};
    auto&              filterStreamToDataSet = graph.emplaceBlock<StreamToDataSet<float>>(blockSettings);
    auto&              dataSetSink           = graph.emplaceBlock<TagSink<DataSet<float>, ProcessFunction::USE_PROCESS_BULK>>({{"name", "dataSetSink"}, {"log_tags", true}, {"log_samples", true}, {"verbose_console", false}});
    expect(graph.connect<"out", "in">(tagSrc, filterStreamToDataSet).has_value()) << locationStr;
    expect(graph.connect<"out", "in">(filterStreamToDataSet, dataSetSink).has_value()) << locationStr;

    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    std::println("start -> Stream-to-DataSet with filter: {} n_pre:{} n_post:{}", filter, preSamples, postSamples);
    expect(sched.runAndWait().has_value()) << std::format("runAndWait - filter {}", filter) << locationStr;
    std::println("done -> Stream-to-DataSet with filter: {} n_pre:{} n_post:{}", filter, preSamples, postSamples);

    expect(eq(tagSrc.sample_rate, sampleRate)) << locationStr;
    expect(eq(filterStreamToDataSet.sample_rate, sampleRate)) << locationStr;
    expect(eq(dataSetSink.sample_rate, sampleRate)) << locationStr;

    expect(eq(dataSetSink._samples.size(), expectedValues.size())) << locationStr;
    for (std::size_t i = 0UZ; i < dataSetSink._samples.size(); i++) {
        const DataSet<float>&          ds      = dataSetSink._samples.at(i);
        std::expected<void, gr::Error> dsCheck = dataset::checkConsistency(ds, "TestDataSet");
        expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal << locationStr;
        expect(std::ranges::equal(ds.signal_values, expectedValues[i])) << locationStr;

        expect(fatal(eq(ds.timing_events.size(), 1UZ))) << locationStr;
        const auto& timingEvt0 = ds.timing_events[0];
        expect(eq(timingEvt0.size(), expectedTags[i].size())) << locationStr;
        auto             view = timingEvt0 | std::views::transform([](const auto& p) { return Tag(static_cast<std::size_t>(p.first), p.second); });
        std::vector<Tag> tags(std::ranges::begin(view), std::ranges::end(view));
        expect(equal_tag_lists(tags, expectedTags[i], std::vector<std::string>{gr::tag::TRIGGER_TIME.shortKey()})) << locationStr;
    }

    expect(le(dataSetSink._tags.size(), dataSetSink._samples.size())) << locationStr;
    expect(!dataSetSink._tags.empty()) << locationStr;
    const auto& autoForwardKeys = filterStreamToDataSet.settings().autoForwardParameters();
    for (const Tag& tag : dataSetSink._tags) {
        expect(le(tag.index, dataSetSink._samples.size() - 1UZ)) << locationStr;
        for (const auto& entry : tag.map) {
            expect(autoForwardKeys.contains(convert_string_domain(entry.first))) << locationStr;
        }
    }
    if (!dataSetSink._tags.empty()) {
        const bool hasSampleRate = dataSetSink._tags.front().map.contains("sample_rate");
        expect(hasSampleRate) << locationStr;
        if (hasSampleRate) {
            expect(eq(dataSetSink._tags.front().map.at("sample_rate").value_or(0.f), sampleRate)) << locationStr;
        }
    }
}

const boost::ut::suite<"StreamToDataSet test"> streamToDataSetTest = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    const std::vector<std::vector<float>> expectedValues1 = { //
        {5, 6, 7, 8, 9},                                      //
        {15, 16, 17, 18, 19, 20, 21, 22, 23, 24},             //
        {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}};
    const std::vector<std::vector<Tag>>   expectedTags1   = { //
        {genNoTrigger(0), genStart(0), genSingle(3)},     //
        {genStart(0), genStart(5)},                       //
        {genStart(0), genStop(5), genSingle(7)}};

    "start->stop (excluding)"_test         = [&expectedValues1, &expectedTags1] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues1, expectedTags1); };
    "start->stop (excluding) n_max=0"_test = [&expectedValues1, &expectedTags1] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues1, expectedTags1, 0UZ); };

    "start->stop (excluding +pre/post)"_test = [] {
        const std::vector<std::vector<float>> expectedValues = {                                            //
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},                                     //
            {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, //
            {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}};
        const std::vector<std::vector<Tag>>   expectedTags   = {                                                                                                          //
            {Tag{0, {{"sample_rate", 1000.f}}}, genNoTrigger(2), genSingle(4), genNoTrigger(5), genStart(5), genSingle(8), genStop(10), genSingle(12), genStart(15)}, //
            {genSingle(0), genStop(2), genSingle(4), genStart(7), genStart(12), genStop(17), genSingle(19), genStop(22)},                                             //
            {genStart(2), genStart(7), genStop(12), genSingle(14), genStop(17), genSingle(19)}};
        runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 7, 7, expectedValues, expectedTags);
    };

    const std::vector<std::vector<float>> expectedValues2 = { //
        {5, 6, 7, 8, 9, 10, 11},                              //
        {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26},     //
        {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}};
    const std::vector<std::vector<Tag>>   expectedTags2   = {         //
        {genNoTrigger(0), genStart(0), genSingle(3), genStop(5)}, //
        {genStart(0), genStart(5), genStop(10)},                  //
        {genStart(0), genStop(5), genSingle(7), genStop(10)}};

    "start->^stop (including)"_test         = [&expectedValues2, &expectedTags2] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues2, expectedTags2); };
    "start->^stop (including) n_max=0"_test = [&expectedValues2, &expectedTags2] { runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues2, expectedTags2, 0UZ); };

    "start->^stop (including. +pre/post)"_test = [] {
        const std::vector<std::vector<float>> expectedValues = {                                                    //
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},                                     //
            {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}, //
            {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}};
        const std::vector<std::vector<Tag>>   expectedTags   = {                                                                                                          //
            {Tag{0, {{"sample_rate", 1000.f}}}, genNoTrigger(2), genSingle(4), genNoTrigger(5), genStart(5), genSingle(8), genStop(10), genSingle(12), genStart(15)}, //
            {genSingle(0), genStop(2), genSingle(4), genStart(7), genStart(12), genStop(17), genSingle(19), genStop(22), genSingle(24)},                              //
            {genStart(2), genStart(7), genStop(12), genSingle(14), genStop(17), genSingle(19)}};
        runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 7, 7, expectedValues, expectedTags);
    };

    "single trigger (+pre/post)"_test = [] {
        const std::vector<std::vector<float>> expectedValues = {      //
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},                       //
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},          //
            {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},      //
            {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}, //
            {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}};
        const std::vector<std::vector<Tag>>   expectedTags   = {                                                                             //
            {Tag{0, {{"sample_rate", 1000.f}}}, genNoTrigger(2), genSingle(4), genNoTrigger(5), genStart(5), genSingle(8), genStop(10)}, //
            {genNoTrigger(1), genSingle(3), genNoTrigger(4), genStart(4), genSingle(7), genStop(9), genSingle(11)},                      //
            {genNoTrigger(0), genStart(0), genSingle(3), genStop(5), genSingle(7), genStart(10)},                                        //
            {genStart(0), genStop(5), genSingle(7), genStop(10), genSingle(12)},                                                         //
            {genStop(0), genSingle(2), genStop(5), genSingle(7)}};
        runTestDataSet(50U, "CMD_DIAG_TRIGGER1", 7, 7, expectedValues, expectedTags);
    };

    "start->stop (excluding, n_max)"_test = [] {
        const gr::Size_t                      nMaxSamples    = 6;
        const std::vector<std::vector<float>> expectedValues = { //
            {5, 6, 7, 8, 9},                                     //
            {15, 16, 17, 18, 19, 20},                            //
            {20, 21, 22, 23, 24, 25}};
        const std::vector<std::vector<Tag>>   expectedTags   = { //
            {genNoTrigger(0), genStart(0), genSingle(3)},    //
            {genStart(0), genStart(5)},                      //
            {genStart(0), genStop(5)}};
        runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, expectedTags, nMaxSamples);
    };

    "start->stop (excluding +pre/post, n_max)"_test = [] {
        const gr::Size_t                      nMaxSamples    = 14;
        const std::vector<std::vector<float>> expectedValues = {    //
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},         //
            {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}, //
            {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}};
        const std::vector<std::vector<Tag>>   expectedTags   = {                                                                                            //
            {Tag{0, {{"sample_rate", 1000.f}}}, genNoTrigger(2), genSingle(4), genNoTrigger(5), genStart(5), genSingle(8), genStop(10), genSingle(12)}, //
            {genSingle(0), genStop(2), genSingle(4), genStart(7), genStart(12)},                                                                        //
            {genStart(2), genStart(7), genStop(12)}};
        runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=2]", 7, 7, expectedValues, expectedTags, nMaxSamples);
    };

    "start->^stop (including, n_max)"_test = [] {
        const gr::Size_t                      nMaxSamples    = 6;
        const std::vector<std::vector<float>> expectedValues = { //
            {5, 6, 7, 8, 9, 10},                                 //
            {15, 16, 17, 18, 19, 20},                            //
            {20, 21, 22, 23, 24, 25}};
        const std::vector<std::vector<Tag>>   expectedTags   = {          //
            {genNoTrigger(0), genStart(0), genSingle(3), genStop(5)}, //
            {genStart(0), genStart(5)},                               //
            {genStart(0), genStop(5)}};
        runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 0, 0, expectedValues, expectedTags, nMaxSamples);
    };

    "start->^stop (including. +pre/post, n_max)"_test = [] {
        const gr::Size_t                      nMaxSamples    = 14;
        const std::vector<std::vector<float>> expectedValues = {    //
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},         //
            {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}, //
            {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}};
        const std::vector<std::vector<Tag>>   expectedTags   = {                                                                                            //
            {Tag{0, {{"sample_rate", 1000.f}}}, genNoTrigger(2), genSingle(4), genNoTrigger(5), genStart(5), genSingle(8), genStop(10), genSingle(12)}, //
            {genSingle(0), genStop(2), genSingle(4), genStart(7), genStart(12)},                                                                        //
            {genStart(2), genStart(7), genStop(12)}};
        runTestDataSet(50U, "[CMD_BP_START/FAIR.SELECTOR.C=1:S=1:P=1, CMD_BP_START/^FAIR.SELECTOR.C=1:S=1:P=2]", 7, 7, expectedValues, expectedTags, nMaxSamples);
    };

    "single trigger (+pre/post, n_max)"_test = [] {
        const gr::Size_t                      nMaxSamples    = 14;
        const std::vector<std::vector<float>> expectedValues = {      //
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},                       //
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},          //
            {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},      //
            {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}, //
            {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}};
        const std::vector<std::vector<Tag>>   expectedTags   = {                                                                             //
            {Tag{0, {{"sample_rate", 1000.f}}}, genNoTrigger(2), genSingle(4), genNoTrigger(5), genStart(5), genSingle(8), genStop(10)}, //
            {genNoTrigger(1), genSingle(3), genNoTrigger(4), genStart(4), genSingle(7), genStop(9), genSingle(11)},                      //
            {genNoTrigger(0), genStart(0), genSingle(3), genStop(5), genSingle(7), genStart(10)},                                        //
            {genStart(0), genStop(5), genSingle(7), genStop(10), genSingle(12)},                                                         //
            {genStop(0), genSingle(2), genStop(5), genSingle(7)}};
        runTestDataSet(50U, "CMD_DIAG_TRIGGER1", 7, 7, expectedValues, expectedTags, nMaxSamples);
    };
};

int main() { /* not needed for UT */ }
