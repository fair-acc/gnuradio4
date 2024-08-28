#include <boost/ut.hpp>

#include <gnuradio-4.0/math/Math.hpp>

const boost::ut::suite<"basic math tests"> basicMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto kArithmeticTypes = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float,
                                                 double /*, gr::UncertainValue<float>, gr::UncertainValue<double>,
std::complex<float>, std::complex<double>*/>();

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

int main() { /* not needed for UT */ }