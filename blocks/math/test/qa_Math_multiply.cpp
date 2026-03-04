#include "qa_Math_common.hpp"

const boost::ut::suite<"math multiply tests"> mathMultiply = [] {
    using namespace boost::ut;
    using namespace gr::blocks::math;
    constexpr qa_math::arithmetic_types kArithmeticTypes{};

    "Multiply"_test = []<typename T>(const T&) {
        test_block<T, Multiply<T>>({.inputs = {{1, 2, 8, 17}}, .output = {1, 2, 8, 17}});
        test_block<T, Multiply<T>>({.inputs = {{1, 2, 3, T(4.0)}, {4, 5, 6, T(7.1)}}, .output = {4, 10, 18, T(28.4)}});
        test_block<T, Multiply<T>>({.inputs = {{0, 1, 2, 3}, {4, 5, 6, 2}, {8, 9, 10, 11}}, .output = {0, 45, 120, 66}});
    } | kArithmeticTypes;
};
