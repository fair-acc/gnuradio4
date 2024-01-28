#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/meta/utils.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <unordered_set>
#include <vector>

using namespace std::string_literals;

template<typename T>
struct RepeatedSource : public gr::Block<RepeatedSource<T>> {
    std::uint32_t  identifier = 0;
    std::uint32_t  remaining_events_count;
    std::vector<T> values;
    std::size_t    values_next = 0;

    gr::PortOut<T> out;

    void
    settingsChanged(const gr::property_map & /*old_settings*/, const gr::property_map &new_settings) noexcept {}

    gr::work::Result
    work(std::size_t requested_work) {
        if (values.empty()) {
            fmt::print("Values vector is empty\n");
            return { requested_work, 0UL, gr::work::Status::DONE };
        }
        if (values_next == values.size()) {
            values_next = 0;
        }

        if (remaining_events_count != 0) {
            auto &port   = gr::outputPort<0, gr::traits::port::kind::Stream>(this);
            auto &writer = port.streamWriter();
            auto  data   = writer.reserve_output_range(1UZ);

            auto value = values[values_next];
            data[0]    = value;
            data.publish(1UZ);

            remaining_events_count--;
            if (remaining_events_count == 0) {
                fmt::print("{}: Last value sent was {}\n", static_cast<void *>(this), values[values_next]);
            }

            values_next++;

            return { requested_work, 1UL, gr::work::Status::OK };
        } else {
            // TODO: Investigate what schedulers do when there is an event written, but we return DONE
            return { requested_work, 1UL, gr::work::Status::DONE };
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(RepeatedSource, identifier, remaining_events_count, values, out);

template<typename T>
struct ValidatorSink : public gr::Block<ValidatorSink<T>> {
    std::uint32_t identifier = 0;
    gr::PortIn<T> in;

    std::vector<T>                 expected_values;
    std::vector<T>::const_iterator expected_values_next;
    bool                           all_ok = true;

    bool
    verify() const {
        return all_ok && (expected_values_next == expected_values.cend());
    }

    void
    settingsChanged(const gr::property_map & /*old_settings*/, const gr::property_map &new_settings) noexcept {
        if (new_settings.contains("expected_values")) {
            expected_values_next = expected_values.cbegin();
        }
    }

    void
    processOne(T value) {
        if (expected_values_next == expected_values.cend()) {
            all_ok = false;
            fmt::print("Error: {}#{}: We got more values than expected\n", static_cast<void *>(this), identifier);

        } else {
            if (value != *expected_values_next) {
                all_ok = false;
                fmt::print("Error: {}#{}: Got a value {}, but wanted {} (position {})\n", static_cast<void *>(this), identifier, value, *expected_values_next,
                           std::distance(expected_values.cbegin(), expected_values_next));
            }

            expected_values_next++;
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(ValidatorSink, identifier, expected_values, in);

template<typename T>
struct ArrayPortsNode : gr::Block<ArrayPortsNode<T>> {
    static constexpr std::size_t                      port_count = 4;
    std::array<gr::PortIn<T, gr::Async>, port_count>  inputs;
    std::array<gr::PortOut<T, gr::Async>, port_count> outputs;

    gr::work::Status
    processBulk(auto &...args) {
        // When we don't write the correct signature for processBulk,
        // this will print out the types we should have used:
        gr::meta::print_types<decltype(args)...>{};
        return gr::work::Status::OK;
    }

    using input_reader_t  = typename gr::PortIn<T, gr::Async>::ReaderType;
    using output_writer_t = typename gr::PortOut<T, gr::Async>::WriterType;

    gr::work::Status
    processBulk(const std::vector<input_reader_t *> &ins, std::vector<output_writer_t *> &outs) {
        for (std::size_t channelIndex = 0; channelIndex < ins.size(); ++channelIndex) {
            auto *inputReader  = ins[channelIndex];
            auto *outputWriter = outs[channelIndex];

            auto available = std::min(inputReader->available(), outputWriter->available());
            if (available == 0) return gr::work::Status::DONE;

            auto inputSpan  = inputReader->get(available);
            auto outputSpan = outputWriter->reserve_output_range(available);

            for (std::size_t valueIndex = 0; valueIndex < available; ++valueIndex) {
                outputSpan[valueIndex] = inputSpan[valueIndex];
            }

            inputReader->consume(available);
            outputSpan.publish(available);
        }
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(ArrayPortsNode, inputs, outputs);

void
execute_selector_test() {
    using namespace boost::ut;

    using TestNode = ArrayPortsNode<double>;

    const std::uint32_t value_count = 5;

    gr::Graph                               graph;
    std::array<RepeatedSource<double> *, 4> sources;
    std::array<ValidatorSink<double> *, 4>  sinks;

    auto *testNode = std::addressof(graph.emplaceBlock<TestNode>());

    sources[0] = std::addressof(graph.emplaceBlock<RepeatedSource<double>>({ { "remaining_events_count", value_count }, //
                                                                             { "identifier", 0U },                      //
                                                                             { "values", std::vector{ 0.0 } } }));
    sources[1] = std::addressof(graph.emplaceBlock<RepeatedSource<double>>({ { "remaining_events_count", value_count }, //
                                                                             { "identifier", 1U },                      //
                                                                             { "values", std::vector{ 1.0 } } }));
    sources[2] = std::addressof(graph.emplaceBlock<RepeatedSource<double>>({ { "remaining_events_count", value_count }, //
                                                                             { "identifier", 2U },                      //
                                                                             { "values", std::vector{ 2.0 } } }));
    sources[3] = std::addressof(graph.emplaceBlock<RepeatedSource<double>>({ { "remaining_events_count", value_count }, //
                                                                             { "identifier", 3U },                      //
                                                                             { "values", std::vector{ 3.0 } } }));

    sinks[0] = std::addressof(graph.emplaceBlock<ValidatorSink<double>>({ { "identifier", 0U },
                                                                          { "expected_values", std::vector{
                                                                                                       0.0,
                                                                                                       0.0,
                                                                                                       0.0,
                                                                                                       0.0,
                                                                                                       0.0,
                                                                                               } } }));
    sinks[1] = std::addressof(graph.emplaceBlock<ValidatorSink<double>>({ { "identifier", 1U }, //
                                                                          { "expected_values", std::vector{
                                                                                                       1.0,
                                                                                                       1.0,
                                                                                                       1.0,
                                                                                                       1.0,
                                                                                                       1.0,
                                                                                               } } }));
    sinks[2] = std::addressof(graph.emplaceBlock<ValidatorSink<double>>({ { "identifier", 2U }, //
                                                                          { "expected_values", std::vector{
                                                                                                       2.0,
                                                                                                       2.0,
                                                                                                       2.0,
                                                                                                       2.0,
                                                                                                       2.0,
                                                                                               } } }));
    sinks[3] = std::addressof(graph.emplaceBlock<ValidatorSink<double>>({ { "identifier", 3U }, //
                                                                          { "expected_values", std::vector{
                                                                                                       3.0,
                                                                                                       3.0,
                                                                                                       3.0,
                                                                                                       3.0,
                                                                                                       3.0,
                                                                                               } } }));

    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[0]).to<"inputs", 0UZ>(*testNode)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[1]).to<0UZ, 1UZ>(*testNode)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[2]).to<0UZ, 2UZ>(*testNode)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(*sources[3], "out"s, *testNode, { "inputs", 3 })));

    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"outputs", 0UZ>(*testNode).to<"in">(*sinks[0])));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<0 /*"outputs"*/, 1UZ>(*testNode).to<"in">(*sinks[1])));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<0 /* outputs */, 2UZ>(*testNode).to<"in">(*sinks[2])));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(*testNode, { "outputs", 3 }, *sinks[3], "in"s)));

    expect(graph.performConnections());

    for (std::size_t iterration = 0; iterration < value_count * sources.size(); ++iterration) {
        const auto max = std::numeric_limits<std::size_t>::max();
        for (auto *source : sources) {
            source->work(max);
        }
        testNode->work(max);
        for (auto *sink : sinks) {
            sink->work(max);
        }
    }

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
