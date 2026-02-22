#include <unordered_set>
#include <vector>

#include <boost/ut.hpp>

#include <gnuradio-4.0/meta/utils.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/basic/Selector.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using namespace std::string_literals;

struct TestParams {
    gr::Size_t                                     nSamples;
    std::vector<std::pair<gr::Size_t, gr::Size_t>> mapping;
    std::vector<gr::Tensor<double>>                inValues;
    std::vector<gr::Tensor<double>>                outValues;
    std::vector<std::vector<gr::Tag>>              inTags;
    std::vector<std::vector<gr::Tag>>              outTags;
    gr::Size_t                                     monitorSource;
    std::vector<double>                            monitorValues;
    bool                                           backPressure;
    std::vector<gr::Size_t>                        nSamplesSelectorInput; // check back pressure
    bool                                           syncCombinedPorts{true};
    bool                                           ignoreOrder{false};
};

std::vector<gr::Tensor<double>> values(std::initializer_list<std::initializer_list<double>> data) {
    std::vector<gr::Tensor<double>> result;
    for (const auto tensorData : data) {
        result.emplace_back(gr::data_from, tensorData);
    }
    return result;
}

void execute_selector_test(TestParams params, std::source_location location = std::source_location::current()) {
    using namespace boost::ut;
    using namespace gr::testing;

    const gr::Size_t nSources = static_cast<gr::Size_t>(params.inValues.size());
    const gr::Size_t nSinks   = static_cast<gr::Size_t>(params.outValues.size());

    gr::Graph                                                       graph;
    std::vector<TagSource<double>*>                                 sources;
    std::vector<TagSink<double, ProcessFunction::USE_PROCESS_ONE>*> sinks;
    gr::basic::Selector<double>*                                    selector;

    gr::Tensor<gr::Size_t> mapIn(gr::extents_from, {params.mapping.size()});
    gr::Tensor<gr::Size_t> mapOut(gr::extents_from, {params.mapping.size()});
    std::ranges::transform(params.mapping, mapIn.begin(), [](auto& p) { return p.first; });
    std::ranges::transform(params.mapping, mapOut.begin(), [](auto& p) { return p.second; });

    selector = std::addressof(graph.emplaceBlock<gr::basic::Selector<double>>({{"n_inputs", nSources}, {"n_outputs", nSinks}, {"map_in", mapIn}, {"map_out", mapOut}, {"back_pressure", params.backPressure}, {"sync_combined_ports", params.syncCombinedPorts}, {"disconnect_on_done", false}}));

    for (gr::Size_t i = 0; i < nSources; ++i) {
        sources.push_back(std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", params.nSamples}, {"values", params.inValues[i]}, {"disconnect_on_done", false}})));
        expect(sources[i]->settings().applyStagedParameters().forwardParameters.empty());
        sources[i]->_tags = params.inTags[i];
        expect(gr::ConnectionResult::SUCCESS == graph.connect(*sources[i], "out"s, *selector, "inputs#"s + std::to_string(i)));
    }

    for (gr::Size_t i = 0; i < nSinks; ++i) {
        sinks.push_back(std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>({{"disconnect_on_done", false}})));
        expect(sinks[i]->settings().applyStagedParameters().forwardParameters.empty());
        expect(gr::ConnectionResult::SUCCESS == graph.connect(*selector, "outputs#"s + std::to_string(i), *sinks[i], "in"s));
    }

    TagSink<double, ProcessFunction::USE_PROCESS_ONE>* monitorSink = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>({{"disconnect_on_done", false}}));
    expect(monitorSink->settings().applyStagedParameters().forwardParameters.empty());
    expect(gr::ConnectionResult::SUCCESS == graph.connect<"monitor">(*selector).to<"in">(*monitorSink));

    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    expect(sched.runAndWait().has_value());

    for (std::size_t i = 0; i < selector->inputs.size(); i++) {
        expect(eq(selector->inputs[i].streamReader().available(), params.nSamplesSelectorInput[i]));
    }

    for (std::size_t i = 0; i < sinks.size(); i++) {
        if (params.ignoreOrder) {
            std::ranges::sort(sinks[i]->_samples);
            std::ranges::sort(params.outValues[i]);
        }
        expect(std::ranges::equal(sinks[i]->_samples, params.outValues[i])) //
            << std::format("called from {}:{} -- test failed:\nparams.outValues[i] i={} samples={} outValues={}", location.file_name(), location.line(), i, sinks[i]->_samples, params.outValues[i]);
    }

    for (std::size_t i = 0; i < sinks.size(); i++) {
        expect(equal_tag_lists(sinks[i]->_tags, params.outTags[i], {}));
    }
}

const boost::ut::suite SelectorTest = [] {
    using namespace boost::ut;
    using namespace gr::basic;

    "Selector<T> constructor"_test = [] {
        Selector<double> block_nop({{"name", "block_nop"}});
        block_nop.init(block_nop.progress);
        expect(eq(block_nop.n_inputs, 0U));
        expect(eq(block_nop.n_outputs, 0U));
        expect(eq(block_nop.inputs.size(), 0U));
        expect(eq(block_nop.outputs.size(), 0U));
        expect(eq(block_nop._internalMappingInOut.size(), 0U));

        Selector<double> block({{"name", "block"}, {"n_inputs", 4U}, {"n_outputs", 3U}});
        block.init(block.progress);
        expect(eq(block.n_inputs, 4U));
        expect(eq(block.n_outputs, 3U));
        expect(eq(block.inputs.size(), 4U));
        expect(eq(block.outputs.size(), 3U));
        expect(eq(block._internalMappingInOut.size(), 0U));
    };

    "basic Selector<T>"_test = [] {
        using T = double;
        const Tensor<uint32_t> outputMap{data_from, {1U, 0U}};
        Selector<T>            block({{"n_inputs", 3U}, {"n_outputs", 2U}, {"map_in", std::vector<gr::Size_t>{0U, 1U}}, {"map_out", outputMap}}); // N.B. 3rd input is unconnected
        block.init(block.progress);
        expect(eq(block._internalMappingInOut.size(), 2U));

        using internal_mapping_t = decltype(block._internalMappingInOut);
        expect(block._internalMappingInOut == internal_mapping_t{{0U, {outputMap[0]}}, {1U, {outputMap[1]}}});
    };

    gr::Tag tag1{1, {{"key1", "value1"}}};
    gr::Tag tag2{2, {{"key2", "value2"}}};
    gr::Tag tag3{3, {{"key3", "value3"}}};

    // Tests without the back pressure

    "Selector<T> 1 to 1 mapping"_test = [tag1, tag2, tag3] {
        execute_selector_test({.nSamples = 5,                                                           //
            .mapping                     = {{0, 0}, {1, 1}, {2, 2}},                                    //
            .inValues                    = values({{1}, {2}, {3}}),                                     //
            .outValues                   = values({{1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {3, 3, 3, 3, 3}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},                                    //
            .outTags                     = {{tag1}, {tag2}, {tag3}},
            .monitorSource               = -1U, //
            .monitorValues               = {},  //
            .backPressure                = false,
            .nSamplesSelectorInput       = {0, 0, 0},
            .ignoreOrder                 = true});
    };

    "Selector<T> only one input used"_test = [tag1, tag2, tag3] {
        execute_selector_test({.nSamples = 5,                                 //
            .mapping                     = {{1, 1}},                          //
            .inValues                    = values({{1}, {2}, {3}}),           //
            .outValues                   = values({{}, {2, 2, 2, 2, 2}, {}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},          //
            .outTags                     = {{}, {tag2}, {}},
            .monitorSource               = -1U, //
            .monitorValues               = {},  //
            .backPressure                = false,
            .nSamplesSelectorInput       = {0, 0, 0},
            .ignoreOrder                 = true});
    };

    "Selector<T> all for one synch_combined_ports = false"_test = [tag1, tag2, tag3] {
        const Tag newTag1{6, tag1.map};
        const Tag newTag2{10, tag2.map};
        const Tag newTag3{13, tag3.map};
        execute_selector_test({.nSamples = 5,                                                               //
            .mapping                     = {{0, 1}, {1, 1}, {2, 1}},                                        //
            .inValues                    = values({{1}, {2}, {3}}),                                         //
            .outValues                   = values({{}, {1, 2, 2, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3}, {}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},                                        //
            .outTags                     = {{}, {newTag1, newTag2, newTag3}, {}},
            .monitorSource               = -1U, //
            .monitorValues               = {},  //
            .backPressure                = false,
            .nSamplesSelectorInput       = {0, 0, 0},
            .syncCombinedPorts           = false,
            .ignoreOrder                 = false});
    };

    "Selector<T> all for one synch_combined_ports = true"_test = [tag1, tag2, tag3] {
        const Tag newTag1{3, tag1.map};
        const Tag newTag2{7, tag2.map};
        const Tag newTag3{11, tag3.map};
        execute_selector_test({.nSamples = 5,                                                               //
            .mapping                     = {{0, 1}, {1, 1}, {2, 1}},                                        //
            .inValues                    = values({{1}, {2}, {3}}),                                         //
            .outValues                   = values({{}, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, {}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},                                        //
            .outTags                     = {{}, {newTag1, newTag2, newTag3}, {}},
            .monitorSource               = -1U, //
            .monitorValues               = {},  //
            .backPressure                = false,
            .nSamplesSelectorInput       = {0, 0, 0},
            .syncCombinedPorts           = true,
            .ignoreOrder                 = false});
    };

    "Selector<T> one for all"_test = [tag1, tag2, tag3] {
        execute_selector_test({.nSamples = 5,                                                           //
            .mapping                     = {{1, 0}, {1, 1}, {1, 2}},                                    //
            .inValues                    = values({{1}, {2}, {3}}),                                     //
            .outValues                   = values({{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},                                    //
            .outTags                     = {{tag2}, {tag2}, {tag2}},
            .monitorSource               = -1U, //
            .monitorValues               = {},  //
            .backPressure                = false,
            .nSamplesSelectorInput       = {0, 0, 0},
            .ignoreOrder                 = false});
    };

    // tests with the back pressure

    "Selector<T> 1 to 1 mapping, with back pressure"_test = [tag1, tag2, tag3] {
        execute_selector_test({.nSamples = 5,                                                           //
            .mapping                     = {{0, 0}, {1, 1}, {2, 2}},                                    //
            .inValues                    = values({{1}, {2}, {3}}),                                     //
            .outValues                   = values({{1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {3, 3, 3, 3, 3}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},                                    //
            .outTags                     = {{tag1}, {tag2}, {tag3}},
            .monitorSource               = -1U, //
            .monitorValues               = {},  //
            .backPressure                = true,
            .nSamplesSelectorInput       = {0, 0, 0},
            .ignoreOrder                 = false});
    };

    "Selector<T> only one input used, with back pressure"_test = [tag1, tag2, tag3] {
        execute_selector_test({.nSamples = 5,                                 //
            .mapping                     = {{1, 1}},                          //
            .inValues                    = values({{1}, {2}, {3}}),           //
            .outValues                   = values({{}, {2, 2, 2, 2, 2}, {}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},          //
            .outTags                     = {{}, {tag2}, {}},
            .monitorSource               = -1U, //
            .monitorValues               = {},  //
            .backPressure                = true,
            .nSamplesSelectorInput       = {5, 0, 5},
            .ignoreOrder                 = false});
    };

    "Selector<T> all for one, with back pressure"_test = [tag1, tag2, tag3] {
        const Tag newTag1{3, tag1.map};
        const Tag newTag2{7, tag2.map};
        const Tag newTag3{11, tag3.map};
        execute_selector_test({.nSamples = 5,                                                               //
            .mapping                     = {{0, 1}, {1, 1}, {2, 1}},                                        //
            .inValues                    = values({{1}, {2}, {3}}),                                         //
            .outValues                   = values({{}, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, {}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},                                        //
            .outTags                     = {{}, {newTag1, newTag2, newTag3}, {}},
            .monitorSource               = -1U, //
            .monitorValues               = {},  //
            .backPressure                = true,
            .nSamplesSelectorInput       = {0, 0, 0},
            .ignoreOrder                 = false});
    };

    "Selector<T> one for all, with back pressure"_test = [tag1, tag2, tag3] {
        execute_selector_test({.nSamples = 5,                                                           //
            .mapping                     = {{1, 0}, {1, 1}, {1, 2}},                                    //
            .inValues                    = values({{1}, {2}, {3}}),                                     //
            .outValues                   = values({{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},                                    //
            .outTags                     = {{tag2}, {tag2}, {tag2}},
            .monitorSource               = -1U, //
            .monitorValues               = {},  //
            .backPressure                = true,
            .nSamplesSelectorInput       = {5, 0, 5},
            .ignoreOrder                 = false});
    };

    // Tests with a monitor

    "Selector<T> 1 to 1 mapping, with monitor, monitor source already mapped"_test = [tag1, tag2, tag3] {
        execute_selector_test({.nSamples = 5,                                                           //
            .mapping                     = {{0, 0}, {1, 1}, {2, 2}},                                    //
            .inValues                    = values({{1}, {2}, {3}}),                                     //
            .outValues                   = values({{1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {3, 3, 3, 3, 3}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},                                    //
            .outTags                     = {{tag1}, {tag2}, {tag3}},
            .monitorSource               = 0U,              // set monitor index
            .monitorValues               = {1, 1, 1, 1, 1}, //
            .backPressure                = false,
            .nSamplesSelectorInput       = {0, 0, 0},
            .ignoreOrder                 = false});
    };

    "Selector<T> only one input used, with monitor, monitor source not mapped"_test = [tag1, tag2, tag3] {
        execute_selector_test({.nSamples = 5,                                 //
            .mapping                     = {{1, 1}},                          //
            .inValues                    = values({{1}, {2}, {3}}),           //
            .outValues                   = values({{}, {2, 2, 2, 2, 2}, {}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},          //
            .outTags                     = {{}, {tag2}, {}},
            .monitorSource               = 0U,              // set monitor index
            .monitorValues               = {1, 1, 1, 1, 1}, // monitor has values even if port is not mapped
            .backPressure                = false,
            .nSamplesSelectorInput       = {0, 0, 0},
            .ignoreOrder                 = false});
    };

    "Selector<T> all for one, with monitor, monitor source already mapped"_test = [tag1, tag2, tag3] {
        const Tag newTag1{3, tag1.map};
        const Tag newTag2{7, tag2.map};
        const Tag newTag3{11, tag3.map};
        execute_selector_test({.nSamples = 5,                                                               //
            .mapping                     = {{0, 1}, {1, 1}, {2, 1}},                                        //
            .inValues                    = values({{1}, {2}, {3}}),                                         //
            .outValues                   = values({{}, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, {}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},                                        //
            .outTags                     = {{}, {newTag1, newTag2, newTag3}, {}},
            .monitorSource               = 1U,              // set monitor index
            .monitorValues               = {2, 2, 2, 2, 2}, //
            .backPressure                = false,
            .nSamplesSelectorInput       = {0, 0, 0},
            .ignoreOrder                 = true});
    };

    "Selector<T> one for all, with monitor, monitor source already mapped"_test = [tag1, tag2, tag3] {
        execute_selector_test({.nSamples = 5,                                                           //
            .mapping                     = {{1, 0}, {1, 1}, {1, 2}},                                    //
            .inValues                    = values({{1}, {2}, {3}}),                                     //
            .outValues                   = values({{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}), //
            .inTags                      = {{tag1}, {tag2}, {tag3}},                                    //
            .outTags                     = {{tag2}, {tag2}, {tag2}},
            .monitorSource               = 1U,              //
            .monitorValues               = {2, 2, 2, 2, 2}, //
            .backPressure                = false,
            .nSamplesSelectorInput       = {0, 0, 0},
            .ignoreOrder                 = false});
    };
};

int main() { /* not needed for UT */ }
