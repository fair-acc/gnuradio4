#include <boost/ut.hpp>
#include "gnuradio-4.0/meta/utils.hpp"
#include <gnuradio-4.0/math/ComplexMath.hpp>
#include <gnuradio-4.0/Block.hpp>

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
        auto row = std::array{T(0), T(0.5) * std::numbers::pi_v<T>};
        auto inner_spans = std::span<const T, 2>{row};
        std::span<const T> input{inner_spans};
        std::vector<T> out_data = {T(0), T(0)};
        std::span<T> output(out_data);
        auto block = Transcendental<T>();
        block.function_name = "sin";
        block.init(block.progress);
        expect(block.processBulk(input, output) == gr::work::Status::OK &&
            eq(output[0], T(0)) && eq(output[1], T(1))) << std::format("Transcendental test for type {}\n", meta::type_name<T>());
        block.function_name = "cos";
        block.init(block.progress);
        double epsilon = 0.000001;
        expect(block.processBulk(input, output) == gr::work::Status::OK &&
            eq(output[0], T(1)) && approx(static_cast<double>(output[1]), static_cast<double>(T(0)), epsilon)) << std::format("Transcendental test #2 for type {}\n", meta::type_name<T>());
    } | decimalTypes;

    "Integrate"_test = []<typename T>(const T&) {
        auto row = std::array{T(4), T(5), T(6)};
        auto input = std::span<const T>{row};
        std::vector<T> out_data = {T(0)};
        std::span<T> output(out_data);
        auto block = Integrate<T>();
        block.decimation = 3;
        block.init(block.progress);
        expect(block.processBulk(input, output) == gr::work::Status::OK &&
            eq(output[0], T(5))) << std::format("Integrate test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "MultiplyByMatrix"_test = []<typename T>(const T&) {
        auto row = std::array{T(1), T(2), T(3)};
        auto inner_spans = std::span<const T, 3>{row};
        std::array<std::span<const T>, 1> inner_spans_storage{inner_spans};
        std::span<const std::span<const T>> inputs{inner_spans_storage};
        std::array<T, 3> arr = {T(0), T(0), T(0)};
        std::span<T> inner_span{arr};
        std::span<std::span<T>> outputs{&inner_span, 1};
        auto block = MultiplyByMatrix<T>();
        block.matrix = {T(0), T(0), T(1), T(0), T(1), T(0), T(1), T(0), T(0)};
        block.init(block.progress);
        expect(block.processBulk(inputs, outputs) == gr::work::Status::OK &&
            eq(outputs[0][0], T(3)) && eq(outputs[0][1], T(2)) && eq(outputs[0][2], T(1))) << std::format("MultiplyByMatrix test for type {}\n", meta::type_name<T>());
    } | std::tuple_cat(decimalTypes, complexTypes);
};

int main() { /* not needed for UT */ }
