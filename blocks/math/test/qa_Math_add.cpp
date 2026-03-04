#include "qa_Math_common.hpp"

const boost::ut::suite<"math add tests"> mathAdd = [] {
    using namespace boost::ut;
    using namespace gr::blocks::math;
    constexpr qa_math::arithmetic_types kArithmeticTypes{};

    "Add"_test = []<typename T>(const T&) {
        test_block<T, Add<T>>({.inputs = {{1, 2, 8, 17}}, .output = {1, 2, 8, 17}});
        test_block<T, Add<T>>({.inputs = {{1, 2, 3, T(4.2)}, {5, 6, 7, T(8.3)}}, .output = {6, 8, 10, T(12.5)}});
        test_block<T, Add<T>>({.inputs = {{12, 35, 18, 17}, {31, 15, 27, 36}, {83, 46, 37, 41}}, .output = {126, 96, 82, 94}});
    } | kArithmeticTypes;
};
