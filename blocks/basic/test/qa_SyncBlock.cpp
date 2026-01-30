#include <boost/ut.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/SyncBlock.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <format>

struct TestParams {
    std::string   testName       = "";
    gr::Size_t    nSamples       = 0U;                                        // 0 -> take inValues[i].size()
    gr::Size_t    maxHistorySize = 0U;                                        // if 0 -> take default
    std::string   filter         = "";                                        // if "" -> take default
    std::uint64_t tolerance      = std::numeric_limits<std::uint64_t>::max(); // if max() -> take default

    std::vector<gr::Tensor<int>>      inValues;
    std::vector<std::vector<gr::Tag>> inTags;
    std::vector<gr::Tensor<int>>      expectedValues;
    std::vector<std::vector<gr::Tag>> expectedTags;
    std::size_t                       expectedNSamples = 0UZ;
};

void runTest(const TestParams& par) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    expect(eq(par.inValues.size(), par.inTags.size()));

    gr::Graph graph;

    std::size_t nPorts = par.inValues.size();

    property_map syncBlockParams = {{"n_ports", static_cast<gr::Size_t>(nPorts)}};
    if (par.maxHistorySize != 0) {
        syncBlockParams.insert_or_assign("max_history_size", par.maxHistorySize);
    }
    if (par.tolerance != std::numeric_limits<std::uint64_t>::max()) {
        syncBlockParams.insert_or_assign("tolerance", par.tolerance);
    }
    if (par.filter != "") {
        syncBlockParams.insert_or_assign("filter", par.filter);
    }
    auto& syncBlock = graph.emplaceBlock<SyncBlock<int>>(syncBlockParams);

    std::vector<TagSource<int, ProcessFunction::USE_PROCESS_BULK>*> sources;
    std::vector<TagSink<int, ProcessFunction::USE_PROCESS_BULK>*>   sinks;

    for (std::size_t i = 0; i < nPorts; i++) {
        property_map srcParams = {{"values", par.inValues[i]}, {"verbose_console", false}, {"disconnect_on_done", false}};
        if (par.nSamples != 0) {
            srcParams.insert_or_assign("n_samples_max", par.nSamples);
        } else {
            srcParams.insert_or_assign("n_samples_max", static_cast<gr::Size_t>(par.inValues[i].size()));
        }

        sources.push_back(std::addressof(graph.emplaceBlock<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>(srcParams)));
        sources[i]->_tags = par.inTags[i];
        expect(gr::ConnectionResult::SUCCESS == graph.connect(*sources[i], "out"s, syncBlock, "inputs#"s + std::to_string(i)));
    }

    for (std::size_t i = 0; i < nPorts; i++) {
        property_map sinkParams = {{"verbose_console", false}, {"disconnect_on_done", false}};
        if (par.expectedValues.empty()) {
            sinkParams.insert_or_assign("log_samples", false);
        }
        sinks.push_back(std::addressof(graph.emplaceBlock<TagSink<int, ProcessFunction::USE_PROCESS_BULK>>(sinkParams)));
        expect(gr::ConnectionResult::SUCCESS == graph.connect(syncBlock, "outputs#"s + std::to_string(i), *sinks[i], "in"s));
    }

    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    sched.runAndWait();

    for (std::size_t i = 0; i < sinks.size(); i++) {
        if (par.expectedValues.empty()) {
            expect(eq(par.expectedNSamples, sinks[i]->_nSamplesProduced));
        } else {
            expect(std::ranges::equal(sinks[i]->_samples, par.expectedValues[i])) << std::format("sinks[{}]->_samples does not match to expected values:\nSink:{}\nExpected:{}\n", i, sinks[i]->_samples, par.expectedValues[i]);
        }
    }

    for (std::size_t i = 0; i < sinks.size(); i++) {
        expect(equal_tag_lists(sinks[i]->_tags, par.expectedTags[i], {}));
    }
}

gr::Tag genSyncTag(std::size_t index, std::uint64_t triggerTime, std::string triggerName = "TriggerName") { //
    return {index, {{gr::tag::TRIGGER_NAME.shortKey(), triggerName}, {gr::tag::TRIGGER_TIME.shortKey(), triggerTime}}};
};

gr::Tag genDropTag(std::size_t index, std::size_t nSamplesDropped) { //
    return {index, {{gr::tag::N_DROPPED_SAMPLES.shortKey(), static_cast<gr::Size_t>(nSamplesDropped)}}};
};

gr::Tag genDropSyncTag(std::size_t index, std::size_t nSamplesDropped, std::uint64_t triggerTime, std::string triggerName = "TriggerName") { //
    return {index, {{gr::tag::N_DROPPED_SAMPLES.shortKey(), static_cast<gr::Size_t>(nSamplesDropped)}, {gr::tag::TRIGGER_NAME.shortKey(), triggerName}, {gr::tag::TRIGGER_TIME.shortKey(), triggerTime}}};
};

const boost::ut::suite SyncBlockTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    "SyncBlock basic test"_test = [] {
        runTest({                                                                                             //
            .tolerance = 2ULL,                                                                                //
            .inValues  =                                                                                      //
            {                                                                                                 //
                gr::Tensor<int>(data_from, {1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1}),                             //
                gr::Tensor<int>(data_from, {1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2}),                       //
                gr::Tensor<int>(data_from, {1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3})},                //
            .inTags =                                                                                         //
            {                                                                                                 //
                {genSyncTag(1, 99), genSyncTag(5, 201), genSyncTag(10, 301)},                                 //
                {genSyncTag(2, 100), genSyncTag(7, 199), genSyncTag(11, 299)},                                //
                {genSyncTag(3, 101), genSyncTag(9, 200), genSyncTag(12, 300)}},                               //
            .expectedValues =                                                                                 //
            {                                                                                                 //
                gr::Tensor<int>(data_from, {1, 0, 1, 2, 3, 0, 1, 2, 0, 1}),                                   //
                gr::Tensor<int>(data_from, {2, 0, 1, 2, 3, 0, 1, 2, 0, 1}),                                   //
                gr::Tensor<int>(data_from, {3, 0, 1, 2, 3, 0, 1, 2, 0, 1})},                                  //
            .expectedTags =                                                                                   //
            {                                                                                                 //
                {genSyncTag(1, 99), genSyncTag(5, 201), genDropSyncTag(8, 2, 301)},                           //
                {genDropTag(0, 1), genSyncTag(1, 100), genDropSyncTag(5, 1, 199), genDropSyncTag(8, 1, 299)}, //
                {genDropTag(0, 2), genSyncTag(1, 101), genDropSyncTag(5, 2, 200), genSyncTag(8, 300)}}});
    };

    "SyncBlock missing tag test"_test = [] {
        runTest({                                                                    //
            .tolerance = 2ULL,                                                       //
            .inValues  =                                                             //
            {                                                                        //
                gr::Tensor<int>(data_from, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),  //
                gr::Tensor<int>(data_from, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),  //
                gr::Tensor<int>(data_from, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})}, //
            .inTags =                                                                //
            {                                                                        //
                {genSyncTag(1, 100), genSyncTag(5, 200), genSyncTag(10, 300)},       //
                {genSyncTag(2, 100), genSyncTag(10, 300)},                           //
                {genSyncTag(4, 200), genSyncTag(10, 300)}},                          //
            .expectedValues =                                                        //
            {                                                                        //
                gr::Tensor<int>(data_from, {5, 6, 7, 8, 9, 10, 11}),                 //
                gr::Tensor<int>(data_from, {5, 6, 7, 8, 9, 10, 11}),                 //
                gr::Tensor<int>(data_from, {5, 6, 7, 8, 9, 10, 11})},                //
            .expectedTags =                                                          //
            {                                                                        //
                {genDropSyncTag(0, 5, 200), genSyncTag(5, 300)},                     // Sample 5 was copied to the output, including the sync tag, even though the tag was not used
                {genDropTag(0, 5), genSyncTag(5, 300)},                              //
                {genDropTag(0, 5), genSyncTag(5, 300)}}});
    };

    "SyncBlock isSync test"_test = [] {
        runTest({                                                                                                                           //
            .nSamples       = 300'000,                                                                                                      //
            .maxHistorySize = 32'000,                                                                                                       //
            .tolerance      = 2ULL,                                                                                                         //
            .inValues       = {{}, {}},                                                                                                     //
            .inTags         =                                                                                                               //
            {                                                                                                                               //
                {genSyncTag(10, 100), genSyncTag(100'100, 200), genSyncTag(201'000, 300)},                                                  //
                {genSyncTag(1, 100), genSyncTag(100'000, 200), genSyncTag(200'000, 300)}},                                                  //
            .expectedValues = {},                                                                                                           //                                                           //
            .expectedTags   =                                                                                                               //
            {                                                                                                                               //
                {genDropTag(0, 9), genSyncTag(1, 100), genDropTag(65537, 91), genSyncTag(100'000, 200), genDropSyncTag(200'000, 900, 300)}, // 65537 -> depends on buffer size
                {genSyncTag(1, 100), genSyncTag(100'000, 200), genSyncTag(200'000, 300)}},                                                  //
            .expectedNSamples = 299'000});
    };

    "SyncBlock back pressure test"_test = [] {
        runTest({                                                                          //
            .nSamples       = 300'000,                                                     //
            .maxHistorySize = 32'000,                                                      //
            .tolerance      = 2ULL,                                                        //
            .inValues       = {{}, {}},                                                    //
            .inTags         =                                                              //
            {                                                                              //
                {genSyncTag(1, 100), genSyncTag(1000, 200), genSyncTag(200'000, 300)},     //
                {genSyncTag(1, 100), genSyncTag(100'000, 200), genSyncTag(200'000, 300)}}, //
            .expectedValues = {},                                                          //
            .expectedTags   =                                                              //
            {                                                                              //
                {genSyncTag(1, 100), genDropTag(1000, 167000), genSyncTag(33'000, 300)},   //
                {genSyncTag(1, 100), genDropTag(1000, 167000), genSyncTag(33'000, 300)}},  //
            .expectedNSamples = 133'000});
    };

    "SyncBlock back pressure test 2"_test = [] {
        runTest({                                                                                                                   //
            .nSamples         = 300'000,                                                                                            //
            .maxHistorySize   = 32'000,                                                                                             //
            .tolerance        = 2ULL,                                                                                               //
            .inValues         = {{}, {}},                                                                                           //
            .inTags           = {{genSyncTag(100'000, 100)}, {genSyncTag(101'000, 100)}},                                           //
            .expectedValues   = {},                                                                                                 //
            .expectedTags     = {{genDropTag(0, 68000), genSyncTag(32'000, 100)}, {genDropTag(0, 69000), genSyncTag(32'000, 100)}}, //
            .expectedNSamples = 231'000});
    };
};

int main() {}
