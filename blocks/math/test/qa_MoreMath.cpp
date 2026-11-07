#include <boost/ut.hpp>

#include <gnuradio-4.0/math/MoreMath.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

template<typename T>
struct TestParameters1 {
    std::vector<gr::Tensor<T>> inputs;
    gr::Tensor<T>              output;
};

template<typename T>
struct TestParameters2 {
    gr::Tensor<T> input;
    gr::Tensor<T> output;
    T             n;
    T             k;
};

template<typename T>
struct TestParameters3 {
    gr::Tensor<T> input;
    gr::Tensor<T> output;
};

template<typename T, typename BlockUnderTest>
void test_block_type1(const TestParameters1<T> p) {
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

template<typename T, typename BlockUnderTest>
void test_block_type2(const TestParameters2<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    // build test graph
    Graph graph;
    auto& block = graph.emplaceBlock<BlockUnderTest>({{"n", p.n}, {"k", p.k}});
    auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.input}, {"n_samples_max", static_cast<Size_t>(p.input.size())}});
    expect(graph.connect(src, "out"s, block, "in"s).has_value()) << "Failed to connect output port of src to input port 'in' of block";
    auto& sink = graph.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>();
    expect(graph.connect(block, "out"s, sink, "in"s).has_value()) << "Failed to connect output port 'out' of block to input port of sink";

    // execute and confirm result
    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    expect(sched.runAndWait().has_value()) << "Failed to run graph: No value";
    expect(std::ranges::equal(sink._samples, p.output)) << std::format("Failed to validate block output: Expected {} but got {} for input {}", p.output, sink._samples, p.input);
};

template<typename T, typename BlockUnderTest>
void test_block_type3(const TestParameters3<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    // build test graph
    Graph graph;
    auto& block = graph.emplaceBlock<BlockUnderTest>();
    auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.input}, {"n_samples_max", static_cast<Size_t>(p.input.size())}});
    expect(graph.connect(src, "out"s, block, "in"s).has_value()) << "Failed to connect output port of src to input port 'in' of block";
    auto& sink = graph.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>();
    expect(graph.connect(block, "out"s, sink, "in"s).has_value()) << "Failed to connect output port 'out' of block to input port of sink";

    // execute and confirm result
    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    expect(sched.runAndWait().has_value()) << "Failed to run graph: No value";
    expect(std::ranges::equal(sink._samples, p.output)) << std::format("Failed to validate block output: Expected {} but got {} for input {}", p.output, sink._samples, p.input);
};

const boost::ut::suite<"more math tests"> moreMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto kArithmeticTypes = std::tuple< uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double >();
    constexpr auto decimalTypes = std::tuple< float, double >();

    "Argmax"_test = []<typename T>(const T&) {
        test_block_type1<T, Argmax<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {T(4), T(7), T(17)})},
            .output = gr::Tensor<T>(gr::data_from, {T(2), T(0)})
        });
        test_block_type1<T, Argmax<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {T(4), T(7), T(17)}),
                       gr::Tensor<T>(gr::data_from, {T(18), T(18), T(18)})},
            .output = gr::Tensor<T>(gr::data_from, {T(0), T(1)})
        });
    } | kArithmeticTypes;

    "Max"_test = []<typename T>(const T&) {
        test_block_type1<T, Max<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {T(4)}),
                       gr::Tensor<T>(gr::data_from, {T(7)})},
            .output = gr::Tensor<T>(gr::data_from, {T(7)})
        });
    } | kArithmeticTypes;

    "Min"_test = []<typename T>(const T&) {
        test_block_type1<T, Min<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {T(4)}),
                       gr::Tensor<T>(gr::data_from, {T(7)})},
            .output = gr::Tensor<T>(gr::data_from, {T(4)})
        });
    } | kArithmeticTypes;

    "Log10"_test = []<typename T>(const T&) {
        test_block_type2<T, Log10<T>>({
            .input = gr::Tensor<T>(gr::data_from, {T(1), T(10)}),
            .output = gr::Tensor<T>(gr::data_from, {T(5), T(7)}),
            .n = T(2),
            .k = T(5)
        });
    } | decimalTypes;

    "RMS"_test = []<typename T>(const T&) {
        test_block_type3<T, RMS<T>>({
            .input = gr::Tensor<T>(gr::data_from, {T(2), T(2)}),
            .output = gr::Tensor<T>(gr::data_from, {T(2)})
        });
    } | decimalTypes;
};

int main() { /* not needed for UT */ }
