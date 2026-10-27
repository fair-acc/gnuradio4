#include <boost/ut.hpp>

#include <gnuradio-4.0/math/ComplexMath.hpp>

#include "gnuradio-4.0/meta/utils.hpp"
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

template<typename T>
struct TestParameters1 {
    gr::Tensor<T> input;
    gr::Tensor<T> output;
    std::string function_name;
};

template<typename T>
struct TestParameters2 {
    gr::Tensor<T> input;
    gr::Tensor<T> output;
    gr::Size_t decimation;
};

template<typename T>
struct TestParameters3 {
    std::vector<gr::Tensor<T>> inputs;
    std::vector<gr::Tensor<T>> outputs;
    gr::Tensor<T> matrix;
};

template<typename T, typename BlockUnderTest>
void test_block_type1(const TestParameters1<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    // build test graph
    Graph graph;
    auto& block = graph.emplaceBlock<BlockUnderTest>({{"function_name", p.function_name}});
    auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.input}, {"n_samples_max", p.input.size()}});
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
void test_block_type2(const TestParameters2<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    // build test graph
    Graph graph;
    auto& block = graph.emplaceBlock<BlockUnderTest>({{"decimation", p.decimation}});
    auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.input}, {"n_samples_max", p.input.size()}});
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
    const auto n_inputs = static_cast<Size_t>(p.inputs.size());
    const auto m_outputs = static_cast<Size_t>(p.outputs.size());

    // build test graph
    Graph graph;
    auto& block = graph.emplaceBlock<BlockUnderTest>({{"n_inputs", n_inputs}, {"m_outputs", m_outputs}, {"matrix", p.matrix}});
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
    for (Size_t i = 0; i < m_outputs; ++i) {
        expect(std::ranges::equal(sink._samples, p.outputs[i])) << std::format("Failed to validate block output: Expected {} but got {} for input {}", p.outputs[i], sink._samples, p.inputs);
    }
};

const boost::ut::suite<"complex math tests"> complexMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto complexTypes = std::tuple< std::complex<float>, std::complex<double> >();
    constexpr auto decimalTypes = std::tuple< float, double >();
    constexpr auto kArithmeticTypes = std::tuple< uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double >();

    "ComplexConjugate"_test = []<typename T>(const T&) {
        expect(eq(ComplexConjugate<T>().processOne(T(0, 1)), T(0, -1))) << std::format("ComplexConjugate test for type {}\n", meta::type_name<T>());
    } | complexTypes;

    "ExponentiateConstInt"_test = []<typename T>(const T&) {
        auto input = T(1, 2);
        auto result = T(-3, 4);
        auto real_result = static_cast<double>(result.real());
        auto imag_result = static_cast<double>(result.imag());
        auto block = ExponentiateConstInt<T>(property_map{{"exponent", 2}});
        block.init(block.progress);
        auto output = block.processOne(input);
        auto real_out = static_cast<double>(output.real());
        auto imag_out = static_cast<double>(output.imag());
        double epsilon = 0.000001;
        expect(approx(real_out, real_result, epsilon) && approx(imag_out, imag_result, epsilon)) << std::format("ExponentiateConstInt test for type {}\n", meta::type_name<T>());
    } | complexTypes;

    "MultiplyByTagValue"_test = []<typename T>(const T&) {
        expect(eq(MultiplyByTagValue<T>().processOne(T(1, 1)), T(1, 1))) << std::format("MultiplyByTagValue test for type {}\n", meta::type_name<T>());
        auto block = MultiplyByTagValue<T>(property_map{{"tag_value", T(2, 3)}});
        block.init(block.progress);
        expect(eq(block.processOne(T(1, 0)), T(2, 3))) << std::format("MultiplyByTagValue test #2 for type {}\n", meta::type_name<T>());
    } | complexTypes;

    "MultiplyConjugate"_test = []<typename T>(const T&) {
        expect(eq(MultiplyConjugate<T>().processOne(T(2, 1), T(3, 1)), T(7, 1))) << std::format("ComplexConjugate test for type {}\n", meta::type_name<T>());
    } | complexTypes;

    "Transcendental"_test = []<typename T>(const T&) {
        test_block_type1<T, Transcendental<T>>({
            .input = gr::Tensor<T>(gr::data_from, {T(0)}),
            .output = gr::Tensor<T>(gr::data_from, {T(0)}),
            .function_name = "sin"
        });
        test_block_type1<T, Transcendental<T>>({
           .input = gr::Tensor<T>(gr::data_from, {T(0)}),
           .output = gr::Tensor<T>(gr::data_from, {T(1)}),
           .function_name = "cos"
        });
    } | decimalTypes;

    "Integrate"_test = []<typename T>(const T&) {
        test_block_type2<T, Integrate<T>>({
            .input = gr::Tensor<T>(gr::data_from, {T(4), T(5), T(6)}),
            .output = gr::Tensor<T>(gr::data_from, {T(5)}),
            .decimation = 3
        });
    } | kArithmeticTypes;

    "MultiplyByMatrix"_test = []<typename T>(const T&) {
        /*
        test_block_type3<T, MultiplyByMatrix<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {T(1)}), //
                       gr::Tensor<T>(gr::data_from, {T(2)}), //
                       gr::Tensor<T>(gr::data_from, {T(3)})},
            .outputs = {gr::Tensor<T>(gr::data_from, {T(1)}), //
                        gr::Tensor<T>(gr::data_from, {T(2)}), //
                        gr::Tensor<T>(gr::data_from, {T(3)})},
            .matrix = gr::Tensor<T>(gr::data_from, {T(0), T(0), T(1), T(0), T(1), T(0), T(1), T(0), T(0)})
        });
        */
    } | std::tuple_cat(decimalTypes, complexTypes);
};

int main() { /* not needed for UT */ }
