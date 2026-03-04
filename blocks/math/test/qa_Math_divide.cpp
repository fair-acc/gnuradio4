#include "qa_Math_common.hpp"

const boost::ut::suite<"math divide tests"> mathDivide = [] {
    using namespace boost::ut;
    using namespace gr::blocks::math;
    constexpr qa_math::arithmetic_types kArithmeticTypes{};

    "Divide"_test = []<typename T>(const T&) {
        test_block<T, Divide<T>>({.inputs = {{1, 2, 8, 17}}, .output = {1, 2, 8, 17}});
        test_block<T, Divide<T>>({.inputs = {{9, 4, 5, T(7.0)}, {3, 4, 1, T(2.0)}}, .output = {3, 1, 5, T(3.5)}});
        test_block<T, Divide<T>>({.inputs = {{0, 10, 40, 80}, {1, 2, 4, 20}, {1, 5, 5, 2}}, .output = {0, 1, 2, 2}});
    } | kArithmeticTypes;
};
