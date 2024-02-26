#include <boost/ut.hpp>

#include <gnuradio-4.0/basic/common_blocks.hpp>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/meta/utils.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

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
execute_selector_test() {
    using namespace boost::ut;

    using TestNode = ArrayPortsNode<double>;

    const gr::Size_t nSamples = 5;

    gr::Graph                               graph;
    std::array<RepeatedSource<double> *, 4> sources;
    std::array<ValidatorSink<double> *, 4>  sinks;

    auto *testNode = std::addressof(graph.emplaceBlock<TestNode>());

    sources[0] = std::addressof(graph.emplaceBlock<RepeatedSource<double>>({ { "n_samples_max", nSamples }, { "id", 0U }, { "values", std::vector{ 0. } } }));
    sources[1] = std::addressof(graph.emplaceBlock<RepeatedSource<double>>({ { "n_samples_max", nSamples }, { "id", 1U }, { "values", std::vector{ 1. } } }));
    sources[2] = std::addressof(graph.emplaceBlock<RepeatedSource<double>>({ { "n_samples_max", nSamples }, { "id", 2U }, { "values", std::vector{ 2. } } }));
    sources[3] = std::addressof(graph.emplaceBlock<RepeatedSource<double>>({ { "n_samples_max", nSamples }, { "id", 3U }, { "values", std::vector{ 3. } } }));

    sinks[0] = std::addressof(graph.emplaceBlock<ValidatorSink<double>>({ { "id", 0U }, { "expected_values", std::vector{ 0., 0., 0., 0., 0. } } }));
    sinks[1] = std::addressof(graph.emplaceBlock<ValidatorSink<double>>({ { "id", 1U }, { "expected_values", std::vector{ 1., 1., 1., 1., 1. } } }));
    sinks[2] = std::addressof(graph.emplaceBlock<ValidatorSink<double>>({ { "id", 2U }, { "expected_values", std::vector{ 2., 2., 2., 2., 2. } } }));
    sinks[3] = std::addressof(graph.emplaceBlock<ValidatorSink<double>>({ { "id", 3U }, { "expected_values", std::vector{ 3., 3., 3., 3., 3. } } }));

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

    for (auto *sink : sinks) {
        assert(sink->verify());
    }
}

const boost::ut::suite SelectorTest = [] {
    using namespace boost::ut;

    "basic ports in arrays"_test = [] { execute_selector_test(); };
};

int
main() { /* not needed for UT */
}
