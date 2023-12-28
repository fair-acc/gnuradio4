#include <list>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/common_blocks.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

const boost::ut::suite DynamicBlocktests = [] {
    using namespace boost::ut;
    using namespace gr::testing;
    "Change number of ports dynamically"_test = [] {
        const gr::Size_t nInputs = 5;
        // const gr::Size_t nAdditionalInputs = 10; // total inputs = nInputs + nAdditionalInputs
        const gr::Size_t nSamples = 5;

        gr::Graph graph;

        auto& adder = graph.emplaceBlock<MultiAdder<double>>({{"n_inputs", nInputs}});
        expect(adder.settings().applyStagedParameters().forwardParameters.empty());
        auto& sink = graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>({});

        std::vector<TagSource<double>*> sources;
        for (std::size_t i = 0; i < nInputs; ++i) {
            sources.push_back(std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"mark_tag", false}})));
            expect(sources.back()->settings().applyStagedParameters().forwardParameters.empty());
            expect(gr::ConnectionResult::SUCCESS == graph.connect(*sources.back(), {"out", gr::meta::invalid_index}, adder, {"inputs", sources.size() - 1}));
        }
        expect(gr::ConnectionResult::SUCCESS == graph.connect<"out">(adder).to<"in">(sink));

        gr::scheduler::Simple sched(std::move(graph));

        expect(sched.runAndWait().has_value());

        expect(eq(sink._samples, std::vector<double>{0., 5., 10., 15., 20})) << "sinks samples does not match to expected values";

        // TODO: for the moment it s not allowed to change number of ports after they are connected
        // TODO: Emscripten does not like this test
        // expect(aborts([&adder] {
        //    std::ignore = adder.settings().set({ { "n_inputs", nInputs + nAdditionalInputs } });
        //    expect(adder.settings().applyStagedParameters().forwardParameters.empty());
        //}));
    };
};

int main() { /* tests are statically executed */ }
