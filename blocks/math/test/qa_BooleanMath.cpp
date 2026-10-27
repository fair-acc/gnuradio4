#include <boost/ut.hpp>

#include <gnuradio-4.0/math/BooleanMath.hpp>

#include "gnuradio-4.0/meta/utils.hpp"
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

template<typename T>
struct TestParameters {
    std::vector<gr::Tensor<T>> inputs;
    gr::Tensor<T>              output;
};

template<typename T, typename BlockUnderTest>
void test_block_type(const TestParameters<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;
    const auto n_inputs = static_cast<Size_t>(p.inputs.size());

    // build test graph
    Graph graph;
    auto& block = graph.emplaceBlock<BlockUnderTest>({{"n_inputs", n_inputs}});
    for (Size_t i = 0; i < n_inputs; ++i) {
        auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.inputs[i]}, {"n_samples_max", static_cast<Size_t>(p.inputs[i].size())}});
        expect(graph.connect(src, "out"s, block, "in#"s + std::to_string(i)).has_value()) << std::format("Failed to connect output port of src {} to input port 'in#{}' of block", i, i);
    }
    auto& sink = graph.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>();
    expect(graph.connect(block, "out"s, sink, "in"s).has_value()) << "Failed to connect output port 'out' of block to input port of sink";

    // execute and confirm result
    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    expect(sched.runAndWait().has_value()) << "Failed to run graph: No value";
    expect(std::ranges::equal(sink._samples, p.output)) << std::format("Failed to validate block output: Expected {} but got {} for input {}", p.output, sink._samples, p.inputs);
};

const boost::ut::suite<"boolean math tests"> booleanMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto intTypes = std::tuple< uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t >();

    "AndConst"_test = []<typename T>(const T&) {
        expect(eq(AndConst<T>().processOne(T(1)), T(1))) << std::format("AndConst test for type {}\n", meta::type_name<T>());
        auto block = AndConst<T>();
        block.constant = T(0);
        block.init(block.progress);
        expect(eq(block.processOne(T(1)), T(0))) << std::format("AndConst test #2 for type {}\n", meta::type_name<T>());
    } | intTypes;

    "And"_test = []<typename T>(const T&) {
        test_block_type<T, And<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {T(1)}),
                       gr::Tensor<T>(gr::data_from, {T(1)})},
            .output = gr::Tensor<T>(gr::data_from, {T(1)})
        });
    } | intTypes;

    "Not"_test = []<typename T>(const T&) {
        expect(eq(Not<T>().processOne(T(1)), T(65534))) << std::format("Not test for type {}\n", meta::type_name<T>());
    } | std::tuple< uint16_t >();

    "Or"_test = []<typename T>(const T&) {
        test_block_type<T, Or<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {T(0)}),
                       gr::Tensor<T>(gr::data_from, {T(0)})},
            .output = gr::Tensor<T>(gr::data_from, {T(0)})
        });
    } | intTypes;

    "Xor"_test = []<typename T>(const T&) {
        test_block_type<T, Xor<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {T(1)}),
                       gr::Tensor<T>(gr::data_from, {T(1)})},
            .output = gr::Tensor<T>(gr::data_from, {T(0)})
        });
    } | intTypes;
};

int main() { /* not needed for UT */ }
