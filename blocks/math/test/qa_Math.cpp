#include <boost/ut.hpp>

#include <gnuradio-4.0/math/Math.hpp>

#include <gnuradio-4.0/testing/StreamProbes.hpp>

template<typename T>
struct TestParameters {
    std::vector<gr::Tensor<T>> inputs;
    gr::Tensor<T>              output;
};

template<typename T>
std::vector<std::vector<T>> asVectors(const std::vector<gr::Tensor<T>>& tensors) {
    return tensors | std::views::transform([](const gr::Tensor<T>& tensor) { return std::vector<T>(tensor.begin(), tensor.end()); }) | std::ranges::to<std::vector>();
}

template<typename T, typename BlockUnderTest>
void expectOutput(const TestParameters<T>& p) {
    const auto                                  nInputs = p.inputs.size();
    gr::testing::ProbedPorts<BlockUnderTest, T> probed(asVectors(p.inputs), {{"n_inputs", static_cast<gr::Size_t>(nInputs)}}, gr::testing::indexedPorts("in", nInputs));
    boost::ut::expect(std::ranges::equal(probed.run()[0], p.output)) << std::format("expected {} for inputs {}", p.output, p.inputs);
}

template<typename T, typename BlockUnderTest>
void expectPairOutput(const TestParameters<T>& p) {
    gr::testing::ProbedPorts<BlockUnderTest, T> probed(asVectors(p.inputs), {}, {"in0", "in1"});
    boost::ut::expect(std::ranges::equal(probed.run()[0], p.output)) << std::format("pair-block output for type {}", gr::meta::type_name<T>());
}

template<typename T>
constexpr T val(double x) {
    if constexpr (gr::meta::complex_like<T>) {
        using V = typename T::value_type;
        return T{static_cast<V>(x), static_cast<V>(0)};
    } else {
        return static_cast<T>(x);
    }
}

const auto basicMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto kArithmeticTypes = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> /*, gr::UncertainValue<float>, gr::UncertainValue<double>*/>();
    constexpr auto kGraphTypes      = std::tuple<int32_t, float, double, std::complex<float>>();

    "Add"_test = []<typename T>(const T&) { //
        expectOutput<T, Add<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})}, //
            .output = gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})    //
        });
        expectOutput<T, Add<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {T(1), T(2), T(3), val<T>(4.2)}), //
                gr::Tensor<T>(gr::data_from, {T(5), T(6), T(7), val<T>(8.3)})},       //
            .output = gr::Tensor<T>(gr::data_from, {T(6), T(8), T(10), val<T>(12.5)}) //
        });
        expectOutput<T, Add<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {12, 35, 18, 17}), //
                gr::Tensor<T>(gr::data_from, {31, 15, 27, 36}),        //
                gr::Tensor<T>(gr::data_from, {83, 46, 37, 41})},       //
            .output = gr::Tensor<T>(gr::data_from, {126, 96, 82, 94})  //
        });
    } | kGraphTypes;

    "Subtract"_test = []<typename T>(const T&) {
        expectOutput<T, Subtract<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})}, //
            .output = gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})    //
        });
        expectOutput<T, Subtract<T>>({                                                 //
            .inputs = {gr::Tensor<T>(gr::data_from, {T(9), T(7), T(5), val<T>(3.5)}),  //
                gr::Tensor<T>(gr::data_from, {T(3), T(2), T(0), val<T>(1.2)})},        //
            .output = gr::Tensor<T>(gr::data_from, {T(6), T(5), T(5), val<T>(2.3)})}); //
        expectOutput<T, Subtract<T>>({                                                 //
            .inputs = {gr::Tensor<T>(gr::data_from, {15, 38, 88, 29}),                 //
                gr::Tensor<T>(gr::data_from, {3, 12, 26, 18}),                         //
                gr::Tensor<T>(gr::data_from, {0, 10, 50, 7})},                         //
            .output = gr::Tensor<T>(gr::data_from, {12, 16, 12, 4})});                 //
    } | kGraphTypes;

    "Multiply"_test = []<typename T>(const T&) {
        expectOutput<T, Multiply<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})}, //
            .output = gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})    //
        });
        expectOutput<T, Multiply<T>>({                                                //
            .inputs = {gr::Tensor<T>(gr::data_from, {T(1), T(2), T(3), val<T>(4.0)}), //
                gr::Tensor<T>(gr::data_from, {T(4), T(5), T(6), val<T>(7.1)})},       //
            .output = gr::Tensor<T>(gr::data_from, {T(4), T(10), T(18), val<T>(28.4)})});
        expectOutput<T, Multiply<T>>({                             //
            .inputs = {gr::Tensor<T>(gr::data_from, {0, 1, 2, 3}), //
                gr::Tensor<T>(gr::data_from, {4, 5, 6, 2}),        //
                gr::Tensor<T>(gr::data_from, {8, 9, 10, 11})},     //
            .output = gr::Tensor<T>(gr::data_from, {0, 45, 120, 66})});
    } | kGraphTypes;

    "Divide"_test = []<typename T>(const T&) {
        expectOutput<T, Divide<T>>({
            .inputs = {gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})}, //
            .output = gr::Tensor<T>(gr::data_from, {1, 2, 8, 17})    //
        });
        expectOutput<T, Divide<T>>({.inputs = {gr::Tensor<T>(gr::data_from, {T(9), T(4), T(5), val<T>(7.0)}), //
                                        gr::Tensor<T>(gr::data_from, {T(3), T(4), T(1), val<T>(2.0)})},       //
            .output                         = gr::Tensor<T>(gr::data_from, {T(3), T(1), T(5), val<T>(3.5)})});
        expectOutput<T, Divide<T>>({.inputs = {gr::Tensor<T>(gr::data_from, {0, 10, 40, 80}), //
                                        gr::Tensor<T>(gr::data_from, {1, 2, 4, 20}),          //
                                        gr::Tensor<T>(gr::data_from, {1, 5, 5, 2})},          //
            .output                         = gr::Tensor<T>(gr::data_from, {0, 1, 2, 2})});
    } | kGraphTypes;

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

const auto pairMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;
    constexpr auto kPairTypes = std::tuple<float, double, std::complex<float>, std::complex<double>, gr::complex<float>, gr::complex<double>>();

    "AddPair"_test = []<typename T>(const T&) {
        // eq() requires a stream-insertable type; gr::complex has none, so compare directly
        expect(AddPair<T>().processOne(T(4), T(3)) == T(4) + T(3)) << std::format("AddPair processOne test for type {}\n", meta::type_name<T>());
    } | kPairTypes;
    "SubtractPair"_test = []<typename T>(const T&) { expect(SubtractPair<T>().processOne(T(4), T(3)) == T(4) - T(3)) << std::format("SubtractPair processOne test for type {}\n", meta::type_name<T>()); } | kPairTypes;
    "MultiplyPair"_test = []<typename T>(const T&) { expect(MultiplyPair<T>().processOne(T(4), T(3)) == T(4) * T(3)) << std::format("MultiplyPair processOne test for type {}\n", meta::type_name<T>()); } | kPairTypes;
    "DividePair"_test   = []<typename T>(const T&) { expect(DividePair<T>().processOne(T(4), T(2)) == T(4) / T(2)) << std::format("DividePair processOne test for type {}\n", meta::type_name<T>()); } | kPairTypes;

    "AddPair port wiring"_test = [] {
        expectPairOutput<float, AddPair<float>>({.inputs = {gr::Tensor<float>(gr::data_from, {1, 2, 3, 4}), //
                                                     gr::Tensor<float>(gr::data_from, {5, 6, 7, 8})},       //
            .output                                      = gr::Tensor<float>(gr::data_from, {6, 8, 10, 12})});
    };
};

int main() {
    using namespace boost::ut;
    "basic math tests"_test    = basicMath;
    "two-port math tests"_test = pairMath;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
