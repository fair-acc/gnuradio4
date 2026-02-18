#include <boost/ut.hpp>

#include <gnuradio-4.0/math/Math.hpp>

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
void test_block(const TestParameters<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;
    const Size_t n_inputs = static_cast<Size_t>(p.inputs.size());

    // build test graph
    Graph graph;
    auto& block = graph.emplaceBlock<BlockUnderTest>({{"n_inputs", n_inputs}});
    for (Size_t i = 0; i < n_inputs; ++i) {
        auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.inputs[i]}, {"n_samples_max", static_cast<Size_t>(p.inputs[i].size())}});
        expect(eq(graph.connect2(src, "out"s, block, "in#"s + std::to_string(i)), ConnectionResult::SUCCESS)) << std::format("Failed to connect output port of src {} to input port 'in#{}' of block", i, i);
    }
    auto& sink = graph.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_ONE>>();
    expect(eq(graph.connect2(block, "out"s, sink, "in"s), ConnectionResult::SUCCESS)) << "Failed to connect output port 'out' of block to input port of sink";

    // execute and confirm result
    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    expect(sched.runAndWait().has_value()) << "Failed to run graph: No value";
    expect(std::ranges::equal(sink._samples, p.output)) << std::format("Failed to validate block output: Expected {} but got {} for input {}", p.output, sink._samples, p.inputs);
};

const boost::ut::suite<"basic math tests"> basicMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto kArithmeticTypes = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float,
                                                 double /*, gr::UncertainValue<float>, gr::UncertainValue<double>,
std::complex<float>, std::complex<double>*/>();

    "Add"_test = []<typename T>(const T&) { //
        test_block<T, Add<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})}, //
            .output = gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})    //
        });
        test_block<T, Add<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {T(1), T(2), T(3), T(4.2)}), //
                gr::Tensor<T>(gr::data_from, {T(5), T(6), T(7), T(8.3)})},       //
            .output = gr::Tensor<T>(gr::data_from, {T(6), T(8), T(10), T(12.5)}) //
        });
        test_block<T, Add<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {12, 35, 18, 17}), //
                gr::Tensor<T>(gr::data_from, {31, 15, 27, 36}),        //
                gr::Tensor<T>(gr::data_from, {83, 46, 37, 41})},       //
            .output = gr::Tensor<T>(gr::data_from, {126, 96, 82, 94})  //
        });
    } | kArithmeticTypes;

    "Subtract"_test = []<typename T>(const T&) {
        test_block<T, Subtract<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})}, //
            .output = gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})    //
        });
        test_block<T, Subtract<T>>({                                              //
            .inputs = {gr::Tensor<T>(gr::data_from, {T(9), T(7), T(5), T(3.5)}),  //
                gr::Tensor<T>(gr::data_from, {T(3), T(2), T(0), T(1.2)})},        //
            .output = gr::Tensor<T>(gr::data_from, {T(6), T(5), T(5), T(2.3)})}); //
        test_block<T, Subtract<T>>({                                              //
            .inputs = {gr::Tensor<T>(gr::data_from, {15, 38, 88, 29}),            //
                gr::Tensor<T>(gr::data_from, {3, 12, 26, 18}),                    //
                gr::Tensor<T>(gr::data_from, {0, 10, 50, 7})},                    //
            .output = gr::Tensor<T>(gr::data_from, {12, 16, 12, 4})});            //
    } | kArithmeticTypes;

    "Multiply"_test = []<typename T>(const T&) {
        test_block<T, Multiply<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})}, //
            .output = gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})    //
        });
        test_block<T, Multiply<T>>({                                             //
            .inputs = {gr::Tensor<T>(gr::data_from, {T(1), T(2), T(3), T(4.0)}), //
                gr::Tensor<T>(gr::data_from, {T(4), T(5), T(6), T(7.1)})},       //
            .output = gr::Tensor<T>(gr::data_from, {T(4), T(10), T(18), T(28.4)})});
        test_block<T, Multiply<T>>({                               //
            .inputs = {gr::Tensor<T>(gr::data_from, {0, 1, 2, 3}), //
                gr::Tensor<T>(gr::data_from, {4, 5, 6, 2}),        //
                gr::Tensor<T>(gr::data_from, {8, 9, 10, 11})},     //
            .output = gr::Tensor<T>(gr::data_from, {0, 45, 120, 66})});
    } | kArithmeticTypes;

    "Divide"_test = []<typename T>(const T&) {
        test_block<T, Divide<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})}, //
            .output = gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})    //
        });
        test_block<T, Divide<T>>({.inputs = {gr::Tensor<T>(gr::data_from, {T(9), T(4), T(5), T(7.0)}), //
                                      gr::Tensor<T>(gr::data_from, {T(3), T(4), T(1), T(2.0)})},       //
            .output                       = gr::Tensor<T>(gr::data_from, {T(3), T(1), T(5), T(3.5)})});
        test_block<T, Divide<T>>({.inputs = {gr::Tensor<T>(gr::data_from, {0, 10, 40, 80}), //
                                      gr::Tensor<T>(gr::data_from, {1, 2, 4, 20}),          //
                                      gr::Tensor<T>(gr::data_from, {1, 5, 5, 2})},          //
            .output                       = gr::Tensor<T>(gr::data_from, {0, 1, 2, 2})});
    } | kArithmeticTypes;

    "AddConst"_test = []<typename T>(const T&) {
        expect(eq(AddConst<T>().processOne(T(4)), T(4) + T(1))) << std::format("AddConst test for type {}\n", meta::type_name<T>());
        auto block = AddConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress);
        expect(eq(block.processOne(T(4)), T(4) + T(2))) << std::format("AddConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "SubtractConst"_test = []<typename T>(const T&) {
        expect(eq(SubtractConst<T>().processOne(T(4)), T(4) - T(1))) << std::format("SubtractConst test for type {}\n", meta::type_name<T>());
        auto block = SubtractConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress);
        expect(eq(block.processOne(T(4)), T(4) - T(2))) << std::format("SubtractConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "MultiplyConst"_test = []<typename T>(const T&) {
        expect(eq(MultiplyConst<T>().processOne(T(4)), T(4) * T(1))) << std::format("MultiplyConst test for type {}\n", meta::type_name<T>());
        auto block = MultiplyConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress);
        expect(eq(block.processOne(T(4)), T(4) * T(2))) << std::format("MultiplyConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "DivideConst"_test = []<typename T>(const T&) {
        expect(eq(DivideConst<T>().processOne(T(4)), T(4) / T(1))) << std::format("SubtractConst test for type {}\n", meta::type_name<T>());
        auto block = DivideConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress);
        expect(eq(block.processOne(T(4)), T(4) / T(2))) << std::format("SubtractConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;
};

int main() { /* not needed for UT */ }
