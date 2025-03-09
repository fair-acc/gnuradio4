#include <boost/ut.hpp>

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <fmt/format.h>

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace gr::testing {

enum class Color { Red, Green, Blue };

template<typename T>
struct NoBlock {};

template<typename T, int N = 0, bool Flag = true, Color = Color::Red>
struct Something {};

template<typename T>
using Alias = Something<T, 123, false, Color::Blue>;

const boost::ut::suite<"Portable Type names"> typeNameTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "Type name portability"_test = [] {
        // Ensure we get the same type name on all platforms
        expect(eq(gr::meta::type_name<uint8_t>(), "uint8"sv));
        expect(eq(gr::meta::type_name<uint16_t>(), "uint16"sv));
        expect(eq(gr::meta::type_name<uint32_t>(), "uint32"sv));
        expect(eq(gr::meta::type_name<uint64_t>(), "uint64"sv));
        expect(eq(gr::meta::type_name<int8_t>(), "int8"sv));
        expect(eq(gr::meta::type_name<int16_t>(), "int16"sv));
        expect(eq(gr::meta::type_name<int32_t>(), "int32"sv));
        expect(eq(gr::meta::type_name<int64_t>(), "int64"sv));
        expect(eq(gr::meta::type_name<float>(), "float32"sv));
        expect(eq(gr::meta::type_name<double>(), "float64"sv));
        expect(eq(gr::meta::type_name<std::complex<float>>(), "complex<float32>"sv));
        expect(eq(gr::meta::type_name<std::complex<double>>(), "complex<float64>"sv));
        expect(eq(gr::meta::type_name<std::string>(), "string"sv));
        expect(eq(gr::meta::type_name<std::chrono::system_clock>(), "std::chrono::system_clock"sv));
        expect(eq(gr::meta::type_name<NoBlock<uint8_t>>(), "gr::testing::NoBlock<uint8>"sv));
        expect(eq(gr::meta::type_name<NoBlock<uint16_t>>(), "gr::testing::NoBlock<uint16>"sv));
        expect(eq(gr::meta::type_name<NoBlock<uint32_t>>(), "gr::testing::NoBlock<uint32>"sv));
        expect(eq(gr::meta::type_name<NoBlock<uint64_t>>(), "gr::testing::NoBlock<uint64>"sv));
        expect(eq(gr::meta::type_name<NoBlock<int8_t>>(), "gr::testing::NoBlock<int8>"sv));
        expect(eq(gr::meta::type_name<NoBlock<int16_t>>(), "gr::testing::NoBlock<int16>"sv));
        expect(eq(gr::meta::type_name<NoBlock<int32_t>>(), "gr::testing::NoBlock<int32>"sv));
        expect(eq(gr::meta::type_name<NoBlock<int64_t>>(), "gr::testing::NoBlock<int64>"sv));
        expect(eq(gr::meta::type_name<NoBlock<float>>(), "gr::testing::NoBlock<float32>"sv));
        expect(eq(gr::meta::type_name<NoBlock<double>>(), "gr::testing::NoBlock<float64>"sv));
        expect(eq(gr::meta::type_name<NoBlock<std::complex<float>>>(), "gr::testing::NoBlock<complex<float32>>"sv));
        expect(eq(gr::meta::type_name<NoBlock<std::complex<double>>>(), "gr::testing::NoBlock<complex<float64>>"sv));
        expect(eq(gr::meta::type_name<NoBlock<std::string>>(), "gr::testing::NoBlock<string>"sv));
        expect(eq(gr::meta::type_name<NoBlock<std::chrono::system_clock>>(), "gr::testing::NoBlock<std::chrono::system_clock>"sv));
        expect(eq(gr::meta::type_name<NoBlock<Packet<float>>>(), "gr::testing::NoBlock<gr::Packet<float32>>"sv));
        expect(eq(gr::meta::type_name<NoBlock<Packet<double>>>(), "gr::testing::NoBlock<gr::Packet<float64>>"sv));
        expect(eq(gr::meta::type_name<NoBlock<Tensor<float>>>(), "gr::testing::NoBlock<gr::Tensor<float32>>"sv));
        expect(eq(gr::meta::type_name<NoBlock<Tensor<double>>>(), "gr::testing::NoBlock<gr::Tensor<float64>>"sv));
        expect(eq(gr::meta::type_name<NoBlock<DataSet<float>>>(), "gr::testing::NoBlock<gr::DataSet<float32>>"sv));
        expect(eq(gr::meta::type_name<NoBlock<DataSet<double>>>(), "gr::testing::NoBlock<gr::DataSet<float64>>"sv));
        expect(eq(gr::meta::type_name<Something<uint8_t>>(), "gr::testing::Something<uint8, 0, true, (gr::testing::Color)0>"sv));
        expect(eq(gr::meta::type_name<Something<uint8_t, 42>>(), "gr::testing::Something<uint8, 42, true, (gr::testing::Color)0>"sv));
        expect(eq(gr::meta::type_name<Something<uint8_t, 42, false>>(), "gr::testing::Something<uint8, 42, false, (gr::testing::Color)0>"sv));
        expect(eq(gr::meta::type_name<Something<uint8_t, 42, false, Color::Blue>>(), "gr::testing::Something<uint8, 42, false, (gr::testing::Color)2>"sv));
    };
};

} // namespace gr::testing

int main() { /* tests are statically executed */ }
