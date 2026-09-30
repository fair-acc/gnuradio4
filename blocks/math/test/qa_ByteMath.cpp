#include <boost/ut.hpp>
#include "gnuradio-4.0/meta/utils.hpp"
#include <gnuradio-4.0/math/ByteMath.hpp>
#include <gnuradio-4.0/Block.hpp>

const boost::ut::suite<"byte math tests"> byteMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto bitSizedTypes = std::tuple< uint8_t, uint16_t >();

    "PackKBits"_test = []<typename T>(const T&) {
        auto row = std::array{T(0), T(0), T(0), T(1), T(1), T(1), T(0), T(0),
        T(0), T(0), T(1), T(1), T(1), T(1), T(1), T(1)};
        auto inner_spans = std::span<const T, 16>{row};
        std::span<const T> input{inner_spans};
        std::vector<T> out_data = {T(0), T(0), T(0), T(0)};
        std::span<T> output(out_data);
        auto block = PackKBits<T>();
        block.k = static_cast<Size_t>(4);
        block.init(block.progress);
        expect(block.processBulk(input, output) == gr::work::Status::OK &&
         (eq(output[0], T(1)) && eq(output[1], T(12)) &&
          eq(output[2], T(3)) && eq(output[3], T(15))))
         << std::format("PackKBits test for type {}\n", meta::type_name<T>());
     } | bitSizedTypes;

    "PackedToUnpacked"_test = []<typename T>(const T&) {
        auto row = std::array{T(19)};
        auto inner_spans = std::span<const T, 1>{row};
        std::span<const T> input{inner_spans};
        std::vector<T> out_data = {T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0)};
        std::span<T> output(out_data);
        expect(PackedToUnpacked<T>().processBulk(input, output) == gr::work::Status::OK &&
               (std::ranges::equal(output, gr::Tensor<T>(gr::data_from,
         {T(0), T(0), T(0), T(1), T(0), T(0), T(1), T(1)}))))
            << std::format("PackedToUnpacked test for type {}\n", meta::type_name<T>());
    } | std::tuple< uint8_t >();

    "RepackBits"_test = []<typename T>(const T&) {
        auto row = std::array{T(25), T(26), T(27), T(28), T(29), T(30), T(31), T(32)};
        auto inner_spans = std::span<const T, 8>{row};
        std::span<const T> input{inner_spans};
        std::vector<T> out_data = {T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0),
            T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0)};
        std::span<T> output(out_data);
        auto block = RepackBits<T>();
        block.k = static_cast<Size_t>(8);
        block.l = static_cast<Size_t>(3);
        block.len_tag_key = "";
        block.align_output = false;
        block.endianness = std::endian::little;
        block.init(block.progress);
        expect(block.processBulk(input, output) == gr::work::Status::OK &&
               (std::ranges::equal(output, gr::Tensor<T>(gr::data_from,
         {T(1), T(3), T(0), T(5), T(1), T(6), T(6), T(0), T(4), T(3), T(4),
                T(6), T(1), T(4), T(7), T(0), T(7), T(3), T(0), T(0), T(2), T(0)}))))
            << std::format("RepackBits test for type {}\n", meta::type_name<T>());
    } | std::tuple< uint8_t >();

    "UnpackKBits"_test = []<typename T>(const T&) {
        auto row = std::array{T(6)};
        auto inner_spans = std::span<const T, 1>{row};
        std::span<const T> input{inner_spans};
        std::vector<T> out_data = {T(0), T(0), T(0), T(0)};
        std::span<T> output(out_data);
        auto block = UnpackKBits<T>();
        block.k = static_cast<Size_t>(4);
        block.init(block.progress);
        expect(block.processBulk(input, output) == gr::work::Status::OK &&
            (eq(output[0], T(0)) && eq(output[1], T(1)) && eq(output[2], T(1))
                && eq(output[3], T(0))))
            << std::format("UnpackKBits test for type {}\n", meta::type_name<T>());
    } | bitSizedTypes;

    "UnpackedToPacked"_test = []<typename T>(const T&) {
        auto row = std::array{T(0), T(0), T(0), T(1), T(0), T(0), T(1), T(1),
            T(0), T(0), T(1), T(1), T(1), T(1), T(1), T(1)};
        auto inner_spans = std::span<const T, 16>{row};
        std::span<const T> input{inner_spans};
        std::vector<T> out_data = {T(0), T(0)};
        std::span<T> output(out_data);
        expect(UnpackedToPacked<T>().processBulk(input, output) == gr::work::Status::OK &&
            ((eq(output[0], T(19)) && eq(output[1], T(63))) || eq(output[0], T(4927))))
            << std::format("UnpackedToPacked test for type {}\n", meta::type_name<T>());
    } | bitSizedTypes;
};

int main() { /* not needed for UT */ }
