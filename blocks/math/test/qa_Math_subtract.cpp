#include "qa_Math_common.hpp"

const boost::ut::suite<"math subtract tests"> mathSubtract = [] {
    using namespace boost::ut;
    using namespace gr::blocks::math;
    constexpr qa_math::arithmetic_types kArithmeticTypes{};

    "Subtract"_test = []<typename T>(const T&) {
        test_block<T, Subtract<T>>({.inputs = {{1, 2, 8, 17}}, .output = {1, 2, 8, 17}});
        test_block<T, Subtract<T>>({.inputs = {{9, 7, 5, T(3.5)}, {3, 2, 0, T(1.2)}}, .output = {6, 5, 5, T(2.3)}});
        test_block<T, Subtract<T>>({.inputs = {{15, 38, 88, 29}, {3, 12, 26, 18}, {0, 10, 50, 7}}, .output = {12, 16, 12, 4}});
    } | kArithmeticTypes;
};
