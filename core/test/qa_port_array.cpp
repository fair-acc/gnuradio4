#include <boost/ut.hpp>

#include <gnuradio-4.0/basic/common_blocks.hpp>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/meta/utils.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <unordered_set>
#include <vector>

using namespace std::string_literals;

template<typename T>
struct ArrayPortsNode : gr::Block<ArrayPortsNode<T>> {
    static constexpr std::size_t nPorts = 4;

    std::array<gr::PortIn<T, gr::Async>, nPorts>  inputs;
    std::array<gr::PortOut<T, gr::Async>, nPorts> outputs;

    template<typename TInSpan, typename TOutSpan>
    gr::work::Status
    processBulk(const std::vector<TInSpan> &ins, const std::vector<TOutSpan> &outs) {
        for (std::size_t channelIndex = 0; channelIndex < ins.size(); ++channelIndex) {
            gr::ConsumableSpan auto  inputSpan  = ins[channelIndex];
            gr::PublishableSpan auto outputSpan = outs[channelIndex];
            auto                     available  = std::min(inputSpan.size(), outputSpan.size());

            if (available == 0) {
                return gr::work::Status::DONE;
            }

            for (std::size_t valueIndex = 0; valueIndex < available; ++valueIndex) {
                outputSpan[valueIndex] = inputSpan[valueIndex];
            }

            std::ignore = inputSpan.consume(available);
            outputSpan.publish(available);
        }
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(ArrayPortsNode, inputs, outputs);
static_assert(gr::HasProcessBulkFunction<ArrayPortsNode<int>>);

void
executeTest() {
    using namespace boost::ut;
    using namespace gr::testing;

    using TestNode = ArrayPortsNode<double>;

    const gr::Size_t nSamples = 5;

    gr::Graph                                                          graph;
    std::array<TagSource<double> *, 4>                                 sources;
    std::array<TagSink<double, ProcessFunction::USE_PROCESS_ONE> *, 4> sinks;

    auto *testNode = std::addressof(graph.emplaceBlock<TestNode>());

    sources[0] = std::addressof(graph.emplaceBlock<TagSource<double>>({ { "n_samples_max", nSamples }, { "values", std::vector{ 0. } } }));
    sources[1] = std::addressof(graph.emplaceBlock<TagSource<double>>({ { "n_samples_max", nSamples }, { "values", std::vector{ 1. } } }));
    sources[2] = std::addressof(graph.emplaceBlock<TagSource<double>>({ { "n_samples_max", nSamples }, { "values", std::vector{ 2. } } }));
    sources[3] = std::addressof(graph.emplaceBlock<TagSource<double>>({ { "n_samples_max", nSamples }, { "values", std::vector{ 3. } } }));

    sinks[0] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
    sinks[1] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
    sinks[2] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
    sinks[3] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());

    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[0]).to<"inputs", 0UZ>(*testNode)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[1]).to<"inputs", 1UZ>(*testNode)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[2]).to<"inputs", 2UZ>(*testNode)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[3]).to<"inputs", 3UZ>(*testNode)));

    // test also different connect API
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(*testNode, { "outputs", 0 }, *sinks[0], "in"s)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(*testNode, { "outputs", 1 }, *sinks[1], "in"s)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(*testNode, { "outputs", 2 }, *sinks[2], "in"s)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(*testNode, { "outputs", 3 }, *sinks[3], "in"s)));

    gr::scheduler::Simple sched{ std::move(graph) };
    sched.runAndWait();

    std::vector<std::vector<double>> expected_values{ { 0., 0., 0., 0., 0. }, { 1., 1., 1., 1., 1. }, { 2., 2., 2., 2., 2. }, { 3., 3., 3., 3., 3. } };
    for (std::size_t i = 0; i < sinks.size(); i++) {
        expect(sinks[i]->n_samples_produced == nSamples) << fmt::format("sinks[{}] mismatch in number of produced samples", i);
        expect(std::ranges::equal(sinks[i]->samples, expected_values[i])) << fmt::format("sinks[{}]->samples does not match to expected values", i);
    }
}

const boost::ut::suite SelectorTest = [] {
    using namespace boost::ut;

    "basic ports in arrays"_test = [] { executeTest(); };
};

int
main() { /* not needed for UT */
}
