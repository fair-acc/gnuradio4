#include <boost/ut.hpp>
#include "gnuradio-4.0/meta/utils.hpp"
#include <gnuradio-4.0/math/BooleanMath.hpp>
#include <gnuradio-4.0/Block.hpp>

const boost::ut::suite<"boolean math tests"> booleanMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto intTypes = std::tuple< uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t >();

    "AndConst"_test = []<typename T>(const T&) {
        expect(eq(AndConst<T>().processOne(T(1)), T(1))) << std::format("AndConst test for type {}\n", meta::type_name<T>());
        auto block = AndConst<T>();
        block.constant = T(0);
        block.init(block.progress);
        expect(eq(block.processOne(T(1)), T(0))) << std::format("AndConst test #2 for type {}\n", meta::type_name<T>());
    } | intTypes;

    "And"_test = []<typename T>(const T&) {
        auto row = std::array{T(1), T(1)};
        auto inner_spans = std::span<const T, 2>{row};
        std::array<std::span<const T>, 1> inner_spans_storage{inner_spans};
        std::span<const std::span<const T>> inputs{inner_spans_storage};
        std::vector<T> out_data = {T(0)};
        std::span<T> output(out_data);
        expect(And<T>().processBulk(inputs, output) == gr::work::Status::OK &&
            eq(output[0], T(1))) << std::format("And test for type {}\n", meta::type_name<T>());
    } | intTypes;

    "Not"_test = []<typename T>(const T&) {
        expect(eq(Not<T>().processOne(T(1)), T(65534))) << std::format("Not test for type {}\n", meta::type_name<T>());
    } | std::tuple< uint16_t >();

    "Or"_test = []<typename T>(const T&) {
        auto row = std::array{T(0), T(0)};
        auto inner_spans = std::span<const T, 2>{row};
        std::array<std::span<const T>, 1> inner_spans_storage{inner_spans};
        std::span<const std::span<const T>> inputs{inner_spans_storage};
        std::vector<T> out_data = {T(0)};
        std::span<T> output(out_data);
        expect(Or<T>().processBulk(inputs, output) == gr::work::Status::OK &&
            eq(output[0], T(0))) << std::format("Or test for type {}\n", meta::type_name<T>());
    } | intTypes;

    "Xor"_test = []<typename T>(const T&) {
        auto row = std::array{T(1), T(1)};
        auto inner_spans = std::span<const T, 2>{row};
        std::array<std::span<const T>, 1> inner_spans_storage{inner_spans};
        std::span<const std::span<const T>> inputs{inner_spans_storage};
        std::vector<T> out_data = {T(0)};
        std::span<T> output(out_data);
        expect(Xor<T>().processBulk(inputs, output) == gr::work::Status::OK &&
            eq(output[0], T(0))) << std::format("Xor test for type {}\n", meta::type_name<T>());
    } | intTypes;
};

int main() { /* not needed for UT */ }
