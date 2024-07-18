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
    std::vector<std::vector<double>>               inValues;
    std::vector<std::vector<double>>               outValues;
    gr::Size_t                                     monitorSource;
    std::vector<double>                            monitorValues;
    bool                                           backPressure;
    bool                                           ignoreOrder{false};
};

void execute_selector_test(TestParams params) {
    using namespace boost::ut;
    using namespace gr::testing;

    const gr::Size_t nSources = static_cast<gr::Size_t>(params.inValues.size());
    const gr::Size_t nSinks   = static_cast<gr::Size_t>(params.outValues.size());

    gr::Graph                                                       graph;
    std::vector<TagSource<double>*>                                 sources;
    std::vector<TagSink<double, ProcessFunction::USE_PROCESS_ONE>*> sinks;
    gr::basic::Selector<double>*                                    selector;

    std::vector<gr::Size_t> mapIn(params.mapping.size());
    std::vector<gr::Size_t> mapOut(params.mapping.size());
    std::ranges::transform(params.mapping, mapIn.begin(), [](auto& p) { return p.first; });
    std::ranges::transform(params.mapping, mapOut.begin(), [](auto& p) { return p.second; });

    selector = std::addressof(graph.emplaceBlock<gr::basic::Selector<double>>({{"n_inputs", nSources}, //
        {"n_outputs", nSinks},                                                                         //
        {"map_in", mapIn},                                                                             //
        {"map_out", mapOut},                                                                           //
        {"back_pressure", params.backPressure}}));

    for (gr::Size_t i = 0; i < nSources; ++i) {
        sources.push_back(std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", params.nSamples}, {"values", params.inValues[i]}})));
        expect(sources[i]->settings().applyStagedParameters().forwardParameters.empty());
        expect(gr::ConnectionResult::SUCCESS == graph.connect(*sources[i], "out"s, *selector, "inputs#"s + std::to_string(i)));
    }

    for (gr::Size_t i = 0; i < nSinks; ++i) {
        sinks.push_back(std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>()));
        expect(sinks[i]->settings().applyStagedParameters().forwardParameters.empty());
        expect(gr::ConnectionResult::SUCCESS == graph.connect(*selector, "outputs#"s + std::to_string(i), *sinks[i], "in"s));
    }

    TagSink<double, ProcessFunction::USE_PROCESS_ONE>* monitorSink = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
    expect(monitorSink->settings().applyStagedParameters().forwardParameters.empty());
    expect(gr::ConnectionResult::SUCCESS == graph.connect<"monitor">(*selector).to<"in">(*monitorSink));

    gr::scheduler::Simple sched{std::move(graph)};
    expect(sched.runAndWait().has_value());

    if (!params.backPressure) {
        for (const auto& input : selector->inputs) {
            expect(eq(input.streamReader().available(), 0U));
        }
    }

    for (std::size_t i = 0; i < sinks.size(); i++) {
        if (params.ignoreOrder) {
            std::ranges::sort(sinks[i]->_samples);
            std::ranges::sort(params.outValues[i]);
        }
        expect(std::ranges::equal(sinks[i]->_samples, params.outValues[i])) << fmt::format("sinks[{}]->_samples does not match to expected values:\nSink:{}\nExpected:{}\n", i, sinks[i]->_samples, params.outValues[i]);
    }
}

const boost::ut::suite SelectorTest = [] {
    using namespace boost::ut;
    using namespace gr::basic;

    "Selector<T> constructor"_test = [] {
        Selector<double> block_nop({{"name", "block_nop"}});
        expect(block_nop.settings().applyStagedParameters().forwardParameters.empty());
        expect(eq(block_nop.n_inputs, 0U));
        expect(eq(block_nop.n_outputs, 0U));
        expect(eq(block_nop.inputs.size(), 0U));
        expect(eq(block_nop.outputs.size(), 0U));
        expect(eq(block_nop._internalMapping.size(), 0U));

        Selector<double> block({{"name", "block"}, {"n_inputs", 4U}, {"n_outputs", 3U}});
        expect(block.settings().applyStagedParameters().forwardParameters.empty());
        expect(eq(block.n_inputs, 4U));
        expect(eq(block.n_outputs, 3U));
        expect(eq(block.inputs.size(), 4U));
        expect(eq(block.outputs.size(), 3U));
        expect(eq(block._internalMapping.size(), 0U));
    };

    "basic Selector<T>"_test = [] {
        using T = double;
        const std::vector<uint32_t> outputMap{1U, 0U};
        Selector<T>                 block({{"n_inputs", 3U}, {"n_outputs", 2U}, {"map_in", std::vector<gr::Size_t>{0U, 1U}}, {"map_out", outputMap}}); // N.B. 3rd input is unconnected
        expect(block.settings().applyStagedParameters().forwardParameters.empty());
        expect(eq(block._internalMapping.size(), 2U));

        using internal_mapping_t = decltype(block._internalMapping);
        expect(block._internalMapping == internal_mapping_t{{0U, {outputMap[0]}}, {1U, {outputMap[1]}}});
    };

    // Tests without the back pressure

    "Selector<T> 1 to 1 mapping"_test = [] {
        execute_selector_test({.nSamples = 5,                                                   //
            .mapping                     = {{0, 0}, {1, 1}, {2, 2}},                            //
            .inValues                    = {{1}, {2}, {3}},                                     //
            .outValues                   = {{1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {3, 3, 3, 3, 3}}, //
            .monitorSource               = -1U,                                                 //
            .monitorValues               = {},                                                  //
            .backPressure                = false,
            .ignoreOrder                 = false});
    };

    "Selector<T> only one input used"_test = [] {
        execute_selector_test({.nSamples = 5,                         //
            .mapping                     = {{1, 1}},                  //
            .inValues                    = {{1}, {2}, {3}},           //
            .outValues                   = {{}, {2, 2, 2, 2, 2}, {}}, //
            .monitorSource               = -1U,                       //
            .monitorValues               = {},                        //
            .backPressure                = false,
            .ignoreOrder                 = false});
    };

    "Selector<T> all for one"_test = [] {
        execute_selector_test({.nSamples = 5,                                                       //
            .mapping                     = {{0, 1}, {1, 1}, {2, 1}},                                //
            .inValues                    = {{1}, {2}, {3}},                                         //
            .outValues                   = {{}, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, {}}, //
            .monitorSource               = -1U,                                                     //
            .monitorValues               = {},                                                      //
            .backPressure                = false,
            .ignoreOrder                 = true});
    };

    "Selector<T> one for all"_test = [] {
        execute_selector_test({.nSamples = 5,                                                   //
            .mapping                     = {{1, 0}, {1, 1}, {1, 2}},                            //
            .inValues                    = {{1}, {2}, {3}},                                     //
            .outValues                   = {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}, //
            .monitorSource               = -1U,                                                 //
            .monitorValues               = {},                                                  //
            .backPressure                = false,
            .ignoreOrder                 = false});
    };

    // tests with the back pressure

    "Selector<T> 1 to 1 mapping, with back pressure"_test = [] {
        execute_selector_test({.nSamples = 5,                                                   //
            .mapping                     = {{0, 0}, {1, 1}, {2, 2}},                            //
            .inValues                    = {{1}, {2}, {3}},                                     //
            .outValues                   = {{1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {3, 3, 3, 3, 3}}, //
            .monitorSource               = -1U,                                                 //
            .monitorValues               = {},                                                  //
            .backPressure                = true,
            .ignoreOrder                 = false});
    };

    "Selector<T> only one input used, with back pressure"_test = [] {
        execute_selector_test({.nSamples = 5,                         //
            .mapping                     = {{1, 1}},                  //
            .inValues                    = {{1}, {2}, {3}},           //
            .outValues                   = {{}, {2, 2, 2, 2, 2}, {}}, //
            .monitorSource               = -1U,                       //
            .monitorValues               = {},                        //
            .backPressure                = true,
            .ignoreOrder                 = false});
    };

    "Selector<T> all for one, with back pressure"_test = [] {
        execute_selector_test({.nSamples = 5,                                                       //
            .mapping                     = {{0, 1}, {1, 1}, {2, 1}},                                //
            .inValues                    = {{1}, {2}, {3}},                                         //
            .outValues                   = {{}, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, {}}, //
            .monitorSource               = -1U,                                                     //
            .monitorValues               = {},                                                      //
            .backPressure                = true,
            .ignoreOrder                 = true});
    };

    "Selector<T> one for all, with back pressure"_test = [] {
        execute_selector_test({.nSamples = 5,                                                   //
            .mapping                     = {{1, 0}, {1, 1}, {1, 2}},                            //
            .inValues                    = {{1}, {2}, {3}},                                     //
            .outValues                   = {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}, //
            .monitorSource               = -1U,                                                 //
            .monitorValues               = {},                                                  //
            .backPressure                = true,
            .ignoreOrder                 = false});
    };

    // Tests with a monitor

    "Selector<T> 1 to 1 mapping, with monitor, monitor source already mapped"_test = [] {
        execute_selector_test({.nSamples = 5,                                                   //
            .mapping                     = {{0, 0}, {1, 1}, {2, 2}},                            //
            .inValues                    = {{1}, {2}, {3}},                                     //
            .outValues                   = {{1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {3, 3, 3, 3, 3}}, //
            .monitorSource               = 0U,                                                  //
            .monitorValues               = {1, 1, 1, 1, 1},                                     //
            .backPressure                = false,
            .ignoreOrder                 = false});
    };

    "Selector<T> only one input used, with monitor, monitor source not mapped"_test = [] {
        execute_selector_test({.nSamples = 5,                         //
            .mapping                     = {{1, 1}},                  //
            .inValues                    = {{1}, {2}, {3}},           //
            .outValues                   = {{}, {2, 2, 2, 2, 2}, {}}, //
            .monitorSource               = 0U,                        //
            .monitorValues               = {1, 1, 1, 1, 1},           //
            .backPressure                = false,
            .ignoreOrder                 = false});
    };

    "Selector<T> all for one, with monitor, monitor source already mapped"_test = [] {
        execute_selector_test({.nSamples = 5,                                                       //
            .mapping                     = {{0, 1}, {1, 1}, {2, 1}},                                //
            .inValues                    = {{1}, {2}, {3}},                                         //
            .outValues                   = {{}, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, {}}, //
            .monitorSource               = 1U,                                                      //
            .monitorValues               = {2, 2, 2, 2, 2},                                         //
            .backPressure                = false,
            .ignoreOrder                 = true});
    };

    "Selector<T> one for all, with monitor, monitor source already mapped"_test = [] {
        execute_selector_test({.nSamples = 5,                                                   //
            .mapping                     = {{1, 0}, {1, 1}, {1, 2}},                            //
            .inValues                    = {{1}, {2}, {3}},                                     //
            .outValues                   = {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}, //
            .monitorSource               = 1U,                                                  //
            .monitorValues               = {2, 2, 2, 2, 2},                                     //
            .backPressure                = false,
            .ignoreOrder                 = false});
    };
};

int main() { /* not needed for UT */ }
