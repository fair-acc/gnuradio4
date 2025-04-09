#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <boost/ut.hpp>
#include <gnuradio-4.0/math/Math.hpp>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <volk/volk.h>

template<typename T>
struct TestParameters {
    std::vector<std::vector<T>> inputs;
    std::vector<T>              output;
};

template<typename T, typename BlockUnderTest>
void test_block(const TestParameters<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;
    const Size_t n_inputs = static_cast<Size_t>(p.inputs.size());

    // build test graph
    Graph graph;
    //auto& block = graph.emplaceBlock<BlockUnderTest>({{"n_inputs", n_inputs}});
    auto& block = std::is_same_v<BlockUnderTest,ConjugateImpl<T>>
        ? graph.emplaceBlock<BlockUnderTest>()
        : graph.emplaceBlock<BlockUnderTest>({{"n_inputs", n_inputs}});
    for (Size_t i = 0; i < n_inputs; ++i) {
        auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.inputs[i]}, {"n_samples_max", static_cast<Size_t>(p.inputs[i].size())}});
        std::string input_port=std::is_same_v<BlockUnderTest,ConjugateImpl<T>> ? "in"s : "in#" + std::to_string(i);
        expect(eq(graph.connect(src, "out"s, block, input_port), ConnectionResult::SUCCESS)) << fmt::format("Failed to connect output port of src {} to input port {}", i, input_port);
    }
    auto& sink = graph.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_ONE>>();
    expect(eq(graph.connect<"out">(block).template to<"in">(sink), ConnectionResult::SUCCESS)) << "Failed to connect output port 'out' of block to input port of sink";

    // execute and confirm result
    gr::scheduler::Simple scheduler{std::move(graph)};
    expect(scheduler.runAndWait().has_value()) << "Failed to run graph: No value";
    expect(std::ranges::equal(sink._samples, p.output)) << fmt::format("Failed to validate block output: Expected {} but got {} for input {}", p.output, sink._samples, p.inputs);
};


const boost::ut::suite<"basic math tests"> basicMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto kArithmeticTypes = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float,
                                                 double /*, gr::UncertainValue<float>, gr::UncertainValue<double>,
std::complex<float>, std::complex<double>*/>();

    // clang-format off

    "Add"_test = []<typename T>(const T&) {
        test_block<T, Add<T>>({
            .inputs = {{1, 2, 8, 17}},
            .output = { 1, 2, 8, 17}
        });
        test_block<T, Add<T>>({
            .inputs = {{1, 2,  3, T( 4.2)},
                       {5, 6,  7, T( 8.3)}},
            .output = { 6, 8, 10, T(12.5)}
        });
        test_block<T, Add<T>>({
            .inputs = {{12, 35, 18, 17},
                       {31, 15, 27, 36},
                       {83, 46, 37, 41}},
            .output = {126, 96, 82, 94}
        });
    } | kArithmeticTypes;

    "Subtract"_test = []<typename T>(const T&) {
        test_block<T, Subtract<T>>({
            .inputs = {{1, 2, 8, 17}},
            .output = { 1, 2, 8, 17}
        });
        test_block<T, Subtract<T>>({
            .inputs = {{9, 7, 5, T(3.5)},
                       {3, 2, 0, T(1.2)}},
            .output = { 6, 5, 5, T(2.3)}});
        test_block<T, Subtract<T>>({
            .inputs = {{15, 38, 88, 29},
                       { 3, 12, 26, 18},
                       { 0, 10, 50,  7}},
            .output = { 12, 16, 12,  4}});
    } | kArithmeticTypes;

    "Multiply"_test = []<typename T>(const T&) {
        test_block<T, Multiply<T>>({
            .inputs = {{1, 2, 8, 17}},
            .output = { 1, 2, 8, 17}
        });
        test_block<T, Multiply<T>>({
            .inputs = {{1,  2,  3, T( 4.0)},
                       {4,  5,  6, T( 7.1)}},
            .output = { 4, 10, 18, T(28.4)}});
        test_block<T, Multiply<T>>({
            .inputs = {{0,  1,   2,  3},
                       {4,  5,   6,  2},
                       {8,  9,  10, 11}},
            .output = { 0, 45, 120, 66}});
    } | kArithmeticTypes;

    "Divide"_test = []<typename T>(const T&) {
        test_block<T, Divide<T>>({
            .inputs = {{1, 2, 8, 17}},
            .output = { 1, 2, 8, 17}
        });
        test_block<T, Divide<T>>({
            .inputs = {{9, 4, 5, T(7.0)},
                       {3, 4, 1, T(2.0)}},
            .output = { 3, 1, 5, T(3.5)}});
        test_block<T, Divide<T>>({
            .inputs = {{0, 10, 40, 80},
                       {1,  2,  4, 20},
                       {1,  5,  5,  2}},
            .output = { 0,  1,  2,  2}});
    } | kArithmeticTypes;

    // clang-format on

    "AddConst"_test = []<typename T>(const T&) {
        expect(eq(AddConst<T>().processOne(T(4)), T(4) + T(1))) << fmt::format("AddConst test for type {}\n", meta::type_name<T>());
        auto block = AddConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress, block.ioThreadPool);
        expect(eq(block.processOne(T(4)), T(4) + T(2))) << fmt::format("AddConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "SubtractConst"_test = []<typename T>(const T&) {
        expect(eq(SubtractConst<T>().processOne(T(4)), T(4) - T(1))) << fmt::format("SubtractConst test for type {}\n", meta::type_name<T>());
        auto block = SubtractConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress, block.ioThreadPool);
        expect(eq(block.processOne(T(4)), T(4) - T(2))) << fmt::format("SubtractConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "MultiplyConst"_test = []<typename T>(const T&) {
        expect(eq(MultiplyConst<T>().processOne(T(4)), T(4) * T(1))) << fmt::format("MultiplyConst test for type {}\n", meta::type_name<T>());
        auto block = MultiplyConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress, block.ioThreadPool);
        expect(eq(block.processOne(T(4)), T(4) * T(2))) << fmt::format("MultiplyConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "DivideConst"_test = []<typename T>(const T&) {
        expect(eq(DivideConst<T>().processOne(T(4)), T(4) / T(1))) << fmt::format("SubtractConst test for type {}\n", meta::type_name<T>());
        auto block = DivideConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress, block.ioThreadPool);
        expect(eq(block.processOne(T(4)), T(4) / T(2))) << fmt::format("SubtractConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes; 
};
using namespace boost::ut;
using namespace gr;
using namespace gr::testing;
using namespace gr::blocks;
using namespace gr::blocks::math;

const boost::ut::suite<"complex conjugate math tests"> conjugateTests = [] {
    "Conjugate"_test = []<typename T>(const T&) {
        // Define test parameters:
        // Input: vector of complex numbers.
        // Expected output: each complex number with its imaginary part inverted.
        TestParameters<T> params{
            .inputs = {{
                T(1.0,  2.0),    // Expected conjugate: (1.0, -2.0)
                T(0.0, -3.0),    // Expected conjugate: (0.0,  3.0)
                T(2.0,  3.0)     // Expected conjugate: (2.0, -3.0)
            }},
            .output = {
                T(1.0, -2.0),
                T(0.0,  3.0),
                T(2.0, -3.0)
            }
        };

        // Execute the test graph: instantiate the Conjugate block and check the output.
        test_block<T, ConjugateImpl<T>>(params);
    } | std::tuple<std::complex<float>, std::complex<double>>();
};

int main() { /* not needed for UT */ }