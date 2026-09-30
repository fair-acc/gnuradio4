#include <boost/ut.hpp>
#include "gnuradio-4.0/meta/utils.hpp"
#include <gnuradio-4.0/math/MoreMath.hpp>
#include <gnuradio-4.0/Block.hpp>

const boost::ut::suite<"more math tests"> moreMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto decimalTypes = std::tuple< float, double >();
    constexpr auto kArithmeticTypes = std::tuple< uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double >();

    "Argmax"_test = []<typename T>(const T&) {
        auto row = std::array{T(4), T(7)};
        auto inner_spans = std::span<const T, 2>{row};
        std::array<std::span<const T>, 1> inner_spans_storage{inner_spans};
        std::span<const std::span<const T>> inputs{inner_spans_storage};
        std::vector<Size_t> out_data = {0, 0};
        std::span<Size_t> output(out_data);
        expect(Argmax<T>().processBulk(inputs, output) == gr::work::Status::OK &&
            eq(output[0], static_cast<Size_t>(0)) && eq(output[1], static_cast<Size_t>(1))) << std::format("Argmax test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "Max"_test = []<typename T>(const T&) {
        auto row = std::array{T(4), T(7)};
        auto inner_spans = std::span<const T, 2>{row};
        std::array<std::span<const T>, 1> inner_spans_storage{inner_spans};
        std::span<const std::span<const T>> inputs{inner_spans_storage};
        std::vector<T> out_data = {T(0)};
        std::span<T> output(out_data);
        expect(Max<T>().processBulk(inputs, output) == gr::work::Status::OK &&
            eq(output[0], T(7))) << std::format("Max test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "Min"_test = []<typename T>(const T&) {
        auto row = std::array{T(4), T(7)};
        auto inner_spans = std::span<const T, 2>{row};
        std::array<std::span<const T>, 1> inner_spans_storage{inner_spans};
        std::span<const std::span<const T>> inputs{inner_spans_storage};
        std::vector<T> out_data = {T(0)};
        std::span<T> output(out_data);
        expect(Min<T>().processBulk(inputs, output) == gr::work::Status::OK &&
            eq(output[0], T(4))) << std::format("Min test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "Log10"_test = []<typename T>(const T&) {
        auto row = std::array{T(1), T(10)};
        auto inner_spans = std::span<const T, 2>{row};
        std::span<const T> input{inner_spans};
        std::vector<T> out_data = {T(0), T(0)};
        std::span<T> output(out_data);
        auto block = Log10<T>();
        block.n = T(2);
        block.k = T(5);
        block.init(block.progress);
        expect(block.processBulk(input, output) == gr::work::Status::OK &&
            eq(output[0], T(5)) && eq(output[1], T(7))) << std::format("Log10 test for type {}\n", meta::type_name<T>());
    } | decimalTypes;

    "RMS"_test = []<typename T>(const T&) {
        auto row = std::array{T(2), T(2)};
        auto inner_spans = std::span<const T, 2>{row};
        std::span<const T> input{inner_spans};
        std::vector<T> out_data = {0};
        std::span<T> output(out_data);
        expect(RMS<T>().processBulk(input, output) == gr::work::Status::OK &&
            eq(output[0], static_cast<T>(2))) << std::format("RMS test for type {}\n", meta::type_name<T>());
    } | decimalTypes;
};

int main() { /* not needed for UT */ }
