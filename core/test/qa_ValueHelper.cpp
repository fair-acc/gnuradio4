#include <boost/ut.hpp>

#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/ValueHelper.hpp>

#include <gnuradio-4.0/formatter/ValueHelperFormatter.hpp>

#include <array>
#include <complex>
#include <cstdint>
#include <map>
#include <memory_resource>
#include <string>
#include <unordered_map>
#include <vector>

struct counting_resource : std::pmr::memory_resource {
    std::pmr::memory_resource* upstream{std::pmr::new_delete_resource()};
    std::size_t                alloc_count{0}, dealloc_count{0}, bytes_allocated{0};

    void reset() { alloc_count = dealloc_count = bytes_allocated = 0; }

    void* do_allocate(std::size_t n, std::size_t align) override {
        ++alloc_count;
        bytes_allocated += n;
        return upstream->allocate(n, align);
    }

    void do_deallocate(void* p, std::size_t n, std::size_t align) override {
        ++dealloc_count;
        bytes_allocated -= n;
        upstream->deallocate(p, n, align);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

const boost::ut::suite<"convertTo vector basic"> _convertTo_vector_basic = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "int32 tensor to vector"_test = [] {
        Value v{Tensor<std::int32_t>{data_from, {1, 2, 3, 4, 5}}};
        auto  result = convertTo<std::vector<std::int32_t>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 5UZ));
        expect(eq((*result)[0], 1));
        expect(eq((*result)[4], 5));
    };

    "float tensor to vector"_test = [] {
        Value v{Tensor<float>{data_from, {1.5f, 2.5f, 3.5f}}};
        auto  result = convertTo<std::vector<float>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
        expect(eq((*result)[1], 2.5f));
    };

    "double tensor to vector"_test = [] {
        Value v{Tensor<double>{data_from, {1.1, 2.2, 3.3}}};
        auto  result = convertTo<std::vector<double>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
        expect(eq((*result)[0], 1.1));
    };

    "complex<float> tensor to vector"_test = [] {
        using CF = std::complex<float>;
        Value v{Tensor<CF>{CF{1.0f, 2.0f}, CF{3.0f, 4.0f}}};
        auto  result = convertTo<std::vector<CF>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(eq((*result)[0], CF{1.0f, 2.0f}));
    };

    "int8 tensor to vector"_test = [] {
        Value v{Tensor<std::int8_t>{data_from, {std::int8_t{-1}, std::int8_t{0}, std::int8_t{127}}}};
        auto  result = convertTo<std::vector<std::int8_t>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
        expect(eq((*result)[0], std::int8_t{-1}));
    };

    "uint8 tensor to vector"_test = [] {
        Value v{Tensor<std::uint8_t>{data_from, {std::uint8_t{0}, std::uint8_t{128}, std::uint8_t{255}}}};
        auto  result = convertTo<std::vector<std::uint8_t>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
        expect(eq((*result)[2], std::uint8_t{255}));
    };

    "bool tensor to vector"_test = [] {
        Value v{Tensor<bool>{data_from, {true, false, true, true}}};
        auto  result = convertTo<std::vector<bool>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 4UZ));
        expect(eq((*result)[0], true));
        expect(eq((*result)[1], false));
    };

    "single element tensor"_test = [] {
        Value v{Tensor<int>{data_from, {42}}};
        auto  result = convertTo<std::vector<int>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 1UZ));
        expect(eq((*result)[0], 42));
    };

    "empty tensor to empty vector"_test = [] {
        Tensor<int> emptyTensor;
        Value       v{std::move(emptyTensor)};
        auto        result = convertTo<std::vector<int>, ConversionPolicy::Safe, RankPolicy::Flatten>(v);
        expect(result.has_value());
        expect(eq(result->size(), 0UZ));
    };
};

const boost::ut::suite<"convertTo vector type conversion"> _convertTo_vector_conversion = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "int32 to int64 widening"_test = [] {
        Value v{Tensor<std::int32_t>{data_from, {1, 2, 3}}};
        auto  result = convertTo<std::vector<std::int64_t>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq((*result)[0], std::int64_t{1}));
    };

    "float to double widening"_test = [] {
        Value v{Tensor<float>{data_from, {1.5f, 2.5f}}};
        auto  result = convertTo<std::vector<double>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq((*result)[0], 1.5));
    };

    "int8 to int16 widening"_test = [] {
        Value v{Tensor<std::int8_t>{data_from, {std::int8_t{-10}, std::int8_t{20}}}};
        auto  result = convertTo<std::vector<std::int16_t>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq((*result)[0], std::int16_t{-10}));
    };

    "uint8 to uint16 widening"_test = [] {
        Value v{Tensor<std::uint8_t>{data_from, {std::uint8_t{100}, std::uint8_t{200}}}};
        auto  result = convertTo<std::vector<std::uint16_t>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq((*result)[0], std::uint16_t{100}));
    };

    "complex<float> to complex<double> widening"_test = [] {
        using CF = std::complex<float>;
        using CD = std::complex<double>;
        Value v{Tensor<CF>{CF{1.0f, 2.0f}}};
        auto  result = convertTo<std::vector<CD>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq((*result)[0], CD{1.0, 2.0}));
    };

    "double to float narrowing"_test = [] {
        Value v{Tensor<double>{data_from, {1.5, 2.5}}};
        auto  result = convertTo<std::vector<float>, ConversionPolicy::Narrowing>(v);
        expect(result.has_value());
        expect(eq((*result)[0], 1.5f));
    };

    "int64 to int32 narrowing"_test = [] {
        Value v{Tensor<std::int64_t>{data_from, {std::int64_t{100}, std::int64_t{200}}}};
        auto  result = convertTo<std::vector<std::int32_t>, ConversionPolicy::Narrowing>(v);
        expect(result.has_value());
        expect(eq((*result)[0], 100));
    };

    "complex<double> to complex<float> narrowing"_test = [] {
        using CF = std::complex<float>;
        using CD = std::complex<double>;
        Value v{Tensor<CD>{CD{1.5, 2.5}}};
        auto  result = convertTo<std::vector<CF>, ConversionPolicy::Narrowing>(v);
        expect(result.has_value());
        expect(eq((*result)[0], CF{1.5f, 2.5f}));
    };

    "widening fails with Safe policy"_test = [] {
        Value v{Tensor<float>{data_from, {1.0f}}};
        auto  result = convertTo<std::vector<double>, ConversionPolicy::Safe>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::WideningNotAllowed));
    };

    "narrowing fails with Widening policy"_test = [] {
        Value v{Tensor<double>{data_from, {1.0}}};
        auto  result = convertTo<std::vector<float>, ConversionPolicy::Widening>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::NarrowingNotAllowed));
    };

    "bool to int conversion requires Unchecked"_test = [] {
        Value v{Tensor<bool>{data_from, {true, false, true}}};

        auto result_safe = convertTo<std::vector<int>>(v);
        expect(!result_safe.has_value());

        auto result_unchecked = convertTo<std::vector<int>, ConversionPolicy::Unchecked>(v);
        expect(result_unchecked.has_value());
        expect(eq((*result_unchecked)[0], 1));
        expect(eq((*result_unchecked)[1], 0));
    };
};

const boost::ut::suite<"convertTo vector rank policy"> _convertTo_vector_rank = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "rank-1 to vector Strict succeeds"_test = [] {
        Value v{Tensor<int>{data_from, {1, 2, 3}}};
        auto  result = convertTo<std::vector<int>, ConversionPolicy::Safe, RankPolicy::Strict>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
    };

    "rank-2 to vector Strict fails"_test = [] {
        Tensor<int> matrix({2, 3}, {1, 2, 3, 4, 5, 6});
        Value       v{std::move(matrix)};
        auto        result = convertTo<std::vector<int>, ConversionPolicy::Safe, RankPolicy::Strict>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::RankMismatch));
    };

    "rank-2 to vector Flatten succeeds"_test = [] {
        Tensor<int> matrix({2, 3}, {1, 2, 3, 4, 5, 6});
        Value       v{std::move(matrix)};
        auto        result = convertTo<std::vector<int>, ConversionPolicy::Safe, RankPolicy::Flatten>(v);
        expect(result.has_value());
        expect(eq(result->size(), 6UZ));
        expect(eq((*result)[0], 1));
        expect(eq((*result)[5], 6));
    };

    "rank-3 to vector Flatten succeeds"_test = [] {
        Tensor<int> cube({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
        Value       v{std::move(cube)};
        auto        result = convertTo<std::vector<int>, ConversionPolicy::Safe, RankPolicy::Flatten>(v);
        expect(result.has_value());
        expect(eq(result->size(), 8UZ));
    };
};

const boost::ut::suite<"convertTo array"> _convertTo_array = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "exact size succeeds"_test = [] {
        Value v{Tensor<int>{data_from, {1, 2, 3, 4, 5}}};
        auto  result = convertTo<std::array<int, 5>>(v);
        expect(result.has_value());
        expect(eq((*result)[0], 1));
        expect(eq((*result)[4], 5));
    };

    "single element to array"_test = [] {
        Value v{Tensor<float>{3.14f}};
        auto  result = convertTo<std::array<float, 1>>(v);
        expect(result.has_value());
        expect(eq((*result)[0], 3.14f));
    };

    "type conversion int to float"_test = [] {
        Value v{Tensor<int, 3UZ>{{1, 2, 3}}};
        auto  result = convertTo<std::array<float, 3UZ>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq((*result)[0], 1.0f));
    };

    "rank-2 with Flatten"_test = [] {
        Tensor<int> matrix({2, 3}, {1, 2, 3, 4, 5, 6});
        Value       v{std::move(matrix)};
        auto        result = convertTo<std::array<int, 6>, ConversionPolicy::Safe, RankPolicy::Flatten>(v);
        expect(result.has_value());
        expect(eq((*result)[5], 6));
    };

    "size mismatch fails"_test = [] {
        Value v{Tensor<int>{data_from, {1, 2, 3, 4, 5}}};
        auto  result = convertTo<std::array<int, 3>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::SizeMismatch));
    };
};

const boost::ut::suite<"convertTo Tensor dynamic"> _convertTo_tensor_dynamic = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "same type preserves shape"_test = [] {
        Tensor<int> matrix({3, 4}, std::vector<int>(12, 42));
        Value       v{std::move(matrix)};
        auto        result = convertTo<Tensor<int>>(v);
        expect(result.has_value());
        expect(eq(result->rank(), 2UZ));
        expect(eq(result->extent(0), 3UZ));
        expect(eq(result->extent(1), 4UZ));
    };

    "float to double widening"_test = [] {
        Value v{Tensor<float>{1.5f, 2.5f, 3.5f}};
        auto  result = convertTo<Tensor<double>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
        expect(eq((*result)[0], 1.5));
    };

    "move optimization no allocation"_test = [] {
        counting_resource cr;
        Value             v{Tensor<float>({100}, &cr)};
        cr.reset();
        auto result = convertTo<Tensor<float>>(std::move(v));
        expect(result.has_value());
        expect(eq(cr.alloc_count, 0UZ)) << "move should not allocate";
    };
};

const boost::ut::suite<"convertTo Tensor static"> _convertTo_tensor_static = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "exact size succeeds"_test = [] {
        Value v{Tensor<int>{data_from, {1, 2, 3, 4, 5, 6}}};
        auto  result = convertTo<Tensor<int, 6>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 6UZ));
    };

    "exact shape 2x3 succeeds"_test = [] {
        Tensor<int> matrix({2, 3}, {1, 2, 3, 4, 5, 6});
        Value       v{std::move(matrix)};
        auto        result = convertTo<Tensor<int, 2, 3>>(v);
        expect(result.has_value());
        expect(eq(result->extent(0), 2UZ));
        expect(eq(result->extent(1), 3UZ));
    };

    "shape 2x3 to 3x2 Reshape succeeds"_test = [] {
        Tensor<int> matrix({2, 3}, {1, 2, 3, 4, 5, 6});
        Value       v{std::move(matrix)};
        auto        result = convertTo<Tensor<int, 3, 2>, ConversionPolicy::Safe, RankPolicy::Reshape>(v);
        expect(result.has_value());
        expect(eq(result->size(), 6UZ));
    };

    "shape 2x3 to 3x2 Strict fails"_test = [] {
        Tensor<std::int32_t> matrix({2, 3}, {1, 2, 3, 4, 5, 6});
        Value                v{std::move(matrix)};
        auto                 result = convertTo<Tensor<std::int32_t, 3, 2>>(v);
        expect(!result.has_value());
    };

    "size mismatch fails"_test = [] {
        Value v{Tensor<int>{data_from, {1, 2, 3, 4, 5, 6}}};
        auto  result = convertTo<Tensor<int, 10>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::SizeMismatch));
    };
};

const boost::ut::suite<"convertTo Tensor semi-dynamic"> _convertTo_tensor_semi = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "rank-2 to dyn,dyn succeeds"_test = [] {
        Tensor<int> matrix({3, 4}, std::vector<int>(12, 7));
        Value       v{std::move(matrix)};
        auto        result = convertTo<Tensor<int, std::dynamic_extent, std::dynamic_extent>>(v);
        expect(result.has_value());
        expect(eq(result->rank(), 2UZ));
        expect(eq(result->extent(0), 3UZ));
        expect(eq(result->extent(1), 4UZ));
    };

    "rank-1 to dyn,dyn Strict fails"_test = [] {
        Value v{Tensor<int>{data_from, {1, 2, 3}}};
        auto  result = convertTo<Tensor<int, std::dynamic_extent, std::dynamic_extent>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::RankMismatch));
    };
};

const boost::ut::suite<"convertTo Tensor<Value>"> _convertTo_tensor_value = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "homogeneous int to vector<int>"_test = [] {
        Tensor<Value> tv{Value{1}, Value{2}, Value{3}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::vector<int>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
        expect(eq((*result)[0], 1));
        expect(eq((*result)[2], 3));
    };

    "float values to Tensor<float>"_test = [] {
        Tensor<Value> tv{Value{1.5f}, Value{2.5f}};
        Value         v{std::move(tv)};
        auto          result = convertTo<Tensor<float>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(eq((*result)[0], 1.5f));
        expect(eq((*result)[1], 2.5f));
    };

    "homogeneous int to array<int>"_test = [] {
        Tensor<Value> tv{Value{10}, Value{20}, Value{30}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::array<int, 3>>(v);
        expect(result.has_value());
        expect(eq((*result)[0], 10));
        expect(eq((*result)[2], 30));
    };

    // --- Widening conversions ---

    "int32 values to vector<double> with Widening"_test = [] {
        Tensor<Value> tv{Value{std::int32_t{1}}, Value{std::int32_t{2}}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::vector<double>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(eq((*result)[0], 1.0));
    };

    "int8 values to vector<int16> with Widening"_test = [] {
        Tensor<Value> tv{Value{std::int8_t{-10}}, Value{std::int8_t{20}}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::vector<std::int16_t>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq((*result)[0], std::int16_t{-10}));
    };

    "uint8 values to vector<uint16> with Widening"_test = [] {
        Tensor<Value> tv{Value{std::uint8_t{100}}, Value{std::uint8_t{200}}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::vector<std::uint16_t>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq((*result)[0], std::uint16_t{100}));
        expect(eq((*result)[1], std::uint16_t{200}));
    };

    "double values to vector<float> with Narrowing"_test = [] {
        Tensor<Value> tv{Value{1.5}, Value{2.5}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::vector<float>, ConversionPolicy::Narrowing>(v);
        expect(result.has_value());
        expect(eq((*result)[0], 1.5f));
    };

    "complex<double> to vector<complex<float>> with Narrowing"_test = [] {
        using CF = std::complex<float>;
        using CD = std::complex<double>;
        Tensor<Value> tv{Value{CD{1.0, 2.0}}, Value{CD{3.0, 4.0}}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::vector<CF>, ConversionPolicy::Narrowing>(v);
        expect(result.has_value());
        expect(eq((*result)[0], CF{1.0f, 2.0f}));
    };

    "float to vector<int> with Unchecked"_test = [] {
        Tensor<Value> tv{{Value{1.5f}, Value{2.5f}}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::vector<std::int32_t>, ConversionPolicy::Unchecked>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(eq((*result)[0], 1));
    };

    "bool to vector<int> with Unchecked"_test = [] {
        Tensor<Value> tv{Value{true}, Value{false}, Value{true}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::vector<int>, ConversionPolicy::Unchecked>(v);
        expect(result.has_value());
        expect(eq((*result)[0], 1));
        expect(eq((*result)[1], 0));
    };

    "Tensor<Value> to array rank-2 Flatten succeeds"_test = [] {
        Tensor<Value> tv({2, 2}, {Value{1}, Value{2}, Value{3}, Value{4}});
        Value         v{std::move(tv)};
        auto          result = convertTo<std::array<int, 4>, ConversionPolicy::Safe, RankPolicy::Flatten>(v);
        expect(result.has_value());
        expect(eq((*result)[0], 1));
        expect(eq((*result)[3], 4));
    };

    "Tensor<Value> to array rank-2 Strict fails"_test = [] {
        Tensor<Value> tv({2, 2}, {Value{1}, Value{2}, Value{3}, Value{4}});
        Value         v{std::move(tv)};
        auto          result = convertTo<std::array<int, 4>, ConversionPolicy::Safe, RankPolicy::Strict>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::RankMismatch));
    };

    "mixed types fails"_test = [] {
        Tensor<Value> tv{Value{1}, Value{std::string_view{"oops"}}, Value{3}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::vector<int>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::ElementTypeMismatch));
        expect(eq(result.error().index, 1UZ));
    };

    "Tensor<Value> to array size mismatch fails"_test = [] {
        Tensor<Value> tv{Value{1}, Value{2}, Value{3}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::array<int, 5>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::SizeMismatch));
    };
};

const boost::ut::suite<"convertTo map Value"> _convertTo_map_value = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "Map to unordered_map<string, Value>"_test = [] {
        Value::Map srcMap;
        srcMap["key1"] = Value{std::int32_t{42}};
        srcMap["key2"] = Value{std::string_view{"hello"}};
        Value v{std::move(srcMap)};
        auto  result = convertTo<std::unordered_map<std::string, Value>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(result->contains("key1"));
        expect(result->contains("key2"));
    };

    "Map to std::map<string, Value>"_test = [] {
        Value::Map srcMap;
        srcMap["alpha"] = Value{std::int32_t{1}};
        srcMap["beta"]  = Value{std::int32_t{2}};
        srcMap["gamma"] = Value{std::int32_t{3}};
        Value v{std::move(srcMap)};
        auto  result = convertTo<std::map<std::string, Value>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
        // std::map is ordered
        auto it = result->begin();
        expect(eq(it->first, std::string{"alpha"}));
    };

    "empty Map"_test = [] {
        Value::Map srcMap;
        Value      v{std::move(srcMap)};

        auto result_umap = convertTo<std::unordered_map<std::string, Value>>(v);
        expect(result_umap.has_value());
        expect(eq(result_umap->size(), 0UZ));

        Value v2{Value::Map{}};
        auto  result_map = convertTo<std::map<std::string, Value>>(v2);
        expect(result_map.has_value());
        expect(eq(result_map->size(), 0UZ));
    };

    "non-Map fails"_test = [] {
        Value v{Tensor<int>{data_from, {1, 2, 3}}};

        auto result_umap = convertTo<std::unordered_map<std::string, Value>>(v);
        expect(!result_umap.has_value());
        expect(eq(result_umap.error().kind, ConversionError::Kind::NotAMap));

        auto result_map = convertTo<std::map<std::string, Value>>(v);
        expect(!result_map.has_value());
        expect(eq(result_map.error().kind, ConversionError::Kind::NotAMap));
    };
};

const boost::ut::suite<"convertTo map typed"> _convertTo_map_typed = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "Map to map<string, int> same type"_test = [] {
        std::map<std::string, int> input{{"a", 1}, {"b", 2}, {"c", 3}};
        Value                      v{input};
        auto                       result = convertTo<std::map<std::string, int>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
        expect(eq(result->at("a"), 1));
        expect(eq(result->at("b"), 2));
        expect(eq(result->at("c"), 3));
    };

    "Map to unordered_map<string, int> same type"_test = [] {
        std::unordered_map<std::string, int> input{{"x", 10}, {"y", 20}};
        Value                                v{input};
        auto                                 result = convertTo<std::unordered_map<std::string, int>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(eq(result->at("x"), 10));
        expect(eq(result->at("y"), 20));
    };

    "Map to map<string, double> same type"_test = [] {
        std::map<std::string, double> input{{"pi", 3.14159}, {"e", 2.71828}};
        Value                         v{input};
        auto                          result = convertTo<std::map<std::string, double>>(v);
        expect(result.has_value());
        expect(eq(result->at("pi"), 3.14159));
        expect(eq(result->at("e"), 2.71828));
    };

    "Map to map<string, pmr::string> same type"_test = [] {
        std::map<std::string, std::string> input{{"key", "value"}, {"foo", "bar"}};
        Value                              v{input};
        auto                               result = convertTo<std::map<std::string, std::pmr::string>>(v);
        expect(result.has_value());
        expect(eq(result->at("key"), std::pmr::string{"value"}));
        expect(eq(result->at("foo"), std::pmr::string{"bar"}));
    };

    "Map to map<string, float> widening from int"_test = [] {
        std::map<std::string, int> input{{"a", 1}, {"b", 2}};
        Value                      v{input};

        auto result_safe = convertTo<std::map<std::string, float>>(v);
        expect(!result_safe.has_value());
        expect(eq(result_safe.error().kind, ConversionError::Kind::ElementTypeMismatch));

        auto result_widen = convertTo<std::map<std::string, float>, ConversionPolicy::Widening>(v);
        expect(result_widen.has_value());
        expect(eq(result_widen->at("a"), 1.0f));
        expect(eq(result_widen->at("b"), 2.0f));
    };

    "Map to map<string, double> widening from float"_test = [] {
        std::map<std::string, float> input{{"x", 1.5f}, {"y", 2.5f}};
        Value                        v{input};
        auto                         result = convertTo<std::map<std::string, double>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq(result->at("x"), 1.5));
        expect(eq(result->at("y"), 2.5));
    };

    "Map to map<string, int64> widening from int32"_test = [] {
        std::map<std::string, std::int32_t> input{{"small", 42}, {"big", 1000000}};
        Value                               v{input};
        auto                                result = convertTo<std::map<std::string, std::int64_t>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq(result->at("small"), std::int64_t{42}));
        expect(eq(result->at("big"), std::int64_t{1000000}));
    };

    "Map to map<string, int16> widening from int8"_test = [] {
        std::map<std::string, std::int8_t> input{{"a", std::int8_t{-10}}, {"b", std::int8_t{127}}};
        Value                              v{input};
        auto                               result = convertTo<std::map<std::string, std::int16_t>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq(result->at("a"), std::int16_t{-10}));
        expect(eq(result->at("b"), std::int16_t{127}));
    };

    "Map to map<string, uint32> widening from uint16"_test = [] {
        std::map<std::string, std::uint16_t> input{{"a", std::uint16_t{100}}, {"b", std::uint16_t{65535}}};
        Value                                v{input};
        auto                                 result = convertTo<std::map<std::string, std::uint32_t>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq(result->at("a"), std::uint32_t{100}));
        expect(eq(result->at("b"), std::uint32_t{65535}));
    };

    "Map to map<string, complex<double>> widening"_test = [] {
        using cf = std::complex<float>;
        using cd = std::complex<double>;
        std::map<std::string, cf> input{{"c1", cf{1.0f, 2.0f}}, {"c2", cf{3.0f, 4.0f}}};
        Value                     v{input};
        auto                      result = convertTo<std::map<std::string, cd>, ConversionPolicy::Widening>(v);
        expect(result.has_value());
        expect(eq(result->at("c1"), cd{1.0, 2.0}));
        expect(eq(result->at("c2"), cd{3.0, 4.0}));
    };

    "Map to map<string, int32> narrowing from int64"_test = [] {
        std::map<std::string, std::int64_t> input{{"a", 100L}, {"b", 200L}};
        Value                               v{input};

        auto result_widen = convertTo<std::map<std::string, std::int32_t>, ConversionPolicy::Widening>(v);
        expect(!result_widen.has_value());

        auto result_narrow = convertTo<std::map<std::string, std::int32_t>, ConversionPolicy::Narrowing>(v);
        expect(result_narrow.has_value());
        expect(eq(result_narrow->at("a"), 100));
        expect(eq(result_narrow->at("b"), 200));
    };

    "Map to map<string, complex<float>> narrowing"_test = [] {
        using cf = std::complex<float>;
        using cd = std::complex<double>;
        std::map<std::string, cd> input{{"c1", cd{1.5, 2.5}}, {"c2", cd{3.5, 4.5}}};
        Value                     v{input};
        auto                      result = convertTo<std::map<std::string, cf>, ConversionPolicy::Narrowing>(v);
        expect(result.has_value());
        expect(eq(result->at("c1"), cf{1.5f, 2.5f}));
        expect(eq(result->at("c2"), cf{3.5f, 4.5f}));
    };

    "empty map to typed map"_test = [] {
        std::map<std::string, int> input;
        Value                      v{input};
        auto                       result = convertTo<std::map<std::string, int>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 0UZ));
    };

    "large typed map conversion"_test = [] {
        std::map<std::string, int> input;
        for (int i = 0; i < 1000; ++i) {
            input["key" + std::to_string(i)] = i;
        }
        Value v{input};
        auto  result = convertTo<std::map<std::string, int>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 1000UZ));
        expect(eq(result->at("key0"), 0));
        expect(eq(result->at("key999"), 999));
    };

    "mixed types fails"_test = [] {
        std::map<std::string, Value> input{{"int_val", Value{42}}, {"dbl_val", Value{3.14}}};
        Value                        v{input};
        auto                         result = convertTo<std::map<std::string, int>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::ElementTypeMismatch));
    };

    "non-map Value to typed map fails"_test = [] {
        Value v{42};
        auto  result = convertTo<std::map<std::string, int>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::NotAMap));
    };

    "Tensor Value to typed map fails"_test = [] {
        Value v{Tensor<int>{data_from, {1, 2, 3}}};
        auto  result = convertTo<std::map<std::string, int>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::NotAMap));
    };
};

const boost::ut::suite<"Value generic map ctor round-trip"> _generic_map_roundtrip = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "std::map<string, int> round-trip"_test = [] {
        std::map<std::string, int> input{{"a", 1}, {"b", 2}, {"c", 3}};
        Value                      v{input};
        auto                       result = convertTo<std::map<std::string, Value>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
        expect(result->at("a").holds<std::int32_t>());
        expect(eq(*result->at("a").get_if<std::int32_t>(), 1));
        expect(eq(*result->at("b").get_if<std::int32_t>(), 2));
        expect(eq(*result->at("c").get_if<std::int32_t>(), 3));
    };

    "std::map<string, double> round-trip"_test = [] {
        std::map<std::string, double> input{{"x", 1.5}, {"y", 2.5}};
        Value                         v{input};
        auto                          result = convertTo<std::map<std::string, Value>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(result->at("x").holds<double>());
        expect(eq(*result->at("x").get_if<double>(), 1.5));
        expect(eq(*result->at("y").get_if<double>(), 2.5));
    };

    "std::map<string, string> round-trip"_test = [] {
        std::map<std::string, std::string> input{{"key", "value"}, {"foo", "bar"}};
        Value                              v{input};
        auto                               result = convertTo<std::map<std::string, Value>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(result->at("key").is_string());
        expect(eq(*result->at("key").get_if<std::pmr::string>(), std::pmr::string{"value"}));
    };

    "std::unordered_map<string, int> round-trip"_test = [] {
        std::unordered_map<std::string, int> input{{"a", 10}, {"b", 20}};
        Value                                v{input};
        auto                                 result = convertTo<std::unordered_map<std::string, Value>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(eq(*result->at("a").get_if<std::int32_t>(), 10));
        expect(eq(*result->at("b").get_if<std::int32_t>(), 20));
    };

    "std::map<string, complex<float>> round-trip"_test = [] {
        using cf = std::complex<float>;
        std::map<std::string, cf> input{{"c1", cf{1.0f, 2.0f}}, {"c2", cf{3.0f, 4.0f}}};
        Value                     v{input};
        auto                      result = convertTo<std::map<std::string, Value>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(result->at("c1").holds<cf>());
        expect(eq(*result->at("c1").get_if<cf>(), cf{1.0f, 2.0f}));
    };

    "mixed types via std::map<string, Value> round-trip"_test = [] {
        std::map<std::string, Value> input{{"int_val", Value{42}}, {"dbl_val", Value{3.14}}, {"str_val", Value{std::string{"hello"}}}};
        Value                        v{input};
        auto                         result = convertTo<std::map<std::string, Value>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 3UZ));
        expect(eq(*result->at("int_val").get_if<std::int32_t>(), 42));
        expect(eq(*result->at("dbl_val").get_if<double>(), 3.14));
        expect(result->at("str_val").is_string());
    };

    "nested map round-trip"_test = [] {
        std::map<std::string, Value> inner{{"nested_key", Value{42}}};
        std::map<std::string, Value> outer{{"inner_map", Value{inner}}, {"scalar", Value{100}}};
        Value                        v{outer};
        auto                         result = convertTo<std::map<std::string, Value>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 2UZ));
        expect(result->at("inner_map").is_map());
        expect(eq(*result->at("scalar").get_if<std::int32_t>(), 100));

        auto* innerMap = result->at("inner_map").get_if<Value::Map>();
        expect(innerMap != nullptr);
        if (innerMap) {
            expect(eq(innerMap->size(), 1UZ));
            expect(eq(*innerMap->at(std::pmr::string{"nested_key"}).get_if<std::int32_t>(), 42));
        }
    };

    "empty map round-trip"_test = [] {
        std::map<std::string, int> input;
        Value                      v{input};
        auto                       result = convertTo<std::map<std::string, Value>>(v);
        expect(result.has_value());
        expect(eq(result->size(), 0UZ));
    };
};

const boost::ut::suite<"convertTo_or"> _convertTo_or = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "success returns converted"_test = [] {
        Value            v{Tensor<int>{data_from, {1, 2, 3}}};
        std::vector<int> fallback{99};
        std::vector<int> result = convertTo_or<std::vector<int>>(v, fallback);
        expect(eq(result.size(), 3UZ));
        expect(eq(result[0], 1));
    };

    "failure returns fallback value"_test = [] {
        Value            v{std::string_view{"not a tensor"}};
        std::vector<int> fallback{99, 88};
        std::vector<int> result = convertTo_or<std::vector<int>>(v, fallback);
        expect(eq(result.size(), 2UZ));
        expect(eq(result[0], 99));
    };

    "success does not call factory"_test = [] {
        Value v{Tensor<int>{data_from, {1, 2, 3}}};
        bool  factory_called = false;
        auto  factory        = [&]() {
            factory_called = true;
            return std::vector<int>{99};
        };
        std::vector<int> result = convertTo_or<std::vector<int>>(v, factory);
        expect(eq(result.size(), 3UZ));
        expect(!factory_called);
    };

    "failure calls factory"_test = [] {
        Value v{std::string_view{"not a tensor"}};
        bool  factory_called = false;
        auto  factory        = [&]() {
            factory_called = true;
            return std::vector<int>{99, 88, 77};
        };
        std::vector<int> result = convertTo_or<std::vector<int>>(v, factory);
        expect(factory_called);
        expect(eq(result.size(), 3UZ));
        expect(eq(result[0], 99));
    };
};

const boost::ut::suite<"assignTo vector"> _assignTo_vector = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "basic assignment"_test = [] {
        std::vector<int> dst;
        Value            v{Tensor<int>{data_from, {1, 2, 3, 4, 5}}};
        auto             result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 5UZ));
        expect(eq(dst[0], 1));
    };

    "reuses capacity"_test = [] {
        std::vector<int> dst;
        dst.reserve(100);
        auto  original_capacity = dst.capacity();
        Value v{Tensor<int>{data_from, {1, 2, 3, 4, 5}}};
        auto  result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 5UZ));
        expect(dst.capacity() >= original_capacity);
        expect(eq(dst[0], 1));
    };

    "with type conversion"_test = [] {
        std::vector<double> dst;
        Value               v{Tensor<float>{1.5f, 2.5f, 3.5f}};
        auto                result = assignTo<ConversionPolicy::Widening>(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 3UZ));
        expect(eq(dst[0], 1.5));
    };

    "empty tensor clears vector"_test = [] {
        std::vector<int> dst{1, 2, 3};
        Tensor<int>      emptyTensor;
        Value            v{std::move(emptyTensor)};
        auto             result = assignTo<ConversionPolicy::Safe, RankPolicy::Flatten>(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 0UZ));
    };

    "type mismatch fails"_test = [] {
        std::vector<int> dst;
        Value            v{std::string_view{"not a tensor"}};
        auto             result = assignTo(dst, v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::NotATensor));
    };
};

const boost::ut::suite<"assignTo array"> _assignTo_array = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "exact size"_test = [] {
        std::array<int, 4> dst{};
        Value              v{Tensor<int>{data_from, {10, 20, 30, 40}}};
        auto               result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst[0], 10));
        expect(eq(dst[3], 40));
    };

    "with Flatten"_test = [] {
        std::array<int, 6UZ> dst{};
        Tensor<int>          matrix({2, 3}, {1, 2, 3, 4, 5, 6});
        Value                v{std::move(matrix)};
        auto                 result = assignTo<ConversionPolicy::Safe, RankPolicy::Flatten>(dst, v);
        expect(result.has_value());
        expect(eq(dst[0], 1));
        expect(eq(dst[5], 6));
    };

    "size mismatch fails"_test = [] {
        std::array<int, 3UZ> dst{};
        Value                v{Tensor<int>{data_from, {1, 2, 3, 4, 5}}};
        auto                 result = assignTo(dst, v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::SizeMismatch));
    };
};

const boost::ut::suite<"assignTo Tensor"> _assignTo_tensor = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "dynamic tensor reuses capacity"_test = [] {
        counting_resource cr;
        auto              dst = Tensor<int>(data_from, {10}, &cr);
        dst.reserve(100);
        auto original_capacity = dst.capacity();
        expect(eq(cr.alloc_count, 2UZ));

        Value v{Tensor<int>{data_from, {1, 2, 3}}};
        auto  result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(cr.alloc_count, 2UZ)) << "no new allocation";
        expect(eq(dst.size(), 3UZ));
        expect(ge(dst.capacity(), original_capacity));
    };

    "static tensor"_test = [] {
        Tensor<int, 4UZ> dst{};
        Value            v{Tensor<int>{data_from, {10, 20, 30, 40}}};
        auto             result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst[0UZ], 10));
        expect(eq(dst[3UZ], 40));
    };

    "move optimization"_test = [] {
        counting_resource cr;
        Tensor<float>     dst({1}, &cr);
        cr.reset();
        Value v{Tensor<float>{1.0f, 2.0f, 3.0f}};
        auto  result = assignTo(dst, std::move(v));
        expect(result.has_value());
        expect(eq(dst.size(), 3UZ));
    };
};

const boost::ut::suite<"assignTo map"> _assignTo_map = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "assignTo unordered_map<string, Value>"_test = [] {
        std::unordered_map<std::string, Value> dst;
        Value::Map                             srcMap;
        srcMap["a"] = Value{1};
        srcMap["b"] = Value{2};
        Value v{std::move(srcMap)};
        auto  result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 2UZ));
    };

    "assignTo std::map<string, Value>"_test = [] {
        std::map<std::string, Value> dst;
        Value::Map                   srcMap;
        srcMap["x"] = Value{10};
        srcMap["y"] = Value{20};
        Value v{std::move(srcMap)};
        auto  result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 2UZ));
        auto it = dst.begin();
        expect(eq(it->first, std::string{"x"}));
    };

    "assignTo empty map replaces contents"_test = [] {
        std::map<std::string, Value> dst;
        dst["existing"] = Value{42};
        Value::Map srcMap;
        Value      v{std::move(srcMap)};
        auto       result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 0UZ)); // replaced, not merged
    };

    "assignTo std::map from generic input"_test = [] {
        std::map<std::string, int> input{{"x", 100}, {"y", 200}};
        Value                      v{input};

        std::map<std::string, Value> dst;
        auto                         result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 2UZ));
        expect(eq(*dst.at("x").get_if<std::int32_t>(), 100));
    };

    "assignTo std::unordered_map from generic input"_test = [] {
        std::unordered_map<std::string, double> input{{"pi", 3.14159}, {"e", 2.71828}};
        Value                                   v{input};

        std::unordered_map<std::string, Value> dst;
        auto                                   result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 2UZ));
        expect(dst.at("pi").holds<double>());
    };

    "assignTo map<string, int>"_test = [] {
        std::map<std::string, int> input{{"x", 100}, {"y", 200}};
        Value                      v{input};

        std::map<std::string, int> dst;
        auto                       result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 2UZ));
        expect(eq(dst.at("x"), 100));
        expect(eq(dst.at("y"), 200));
    };

    "assignTo unordered_map<string, double>"_test = [] {
        std::unordered_map<std::string, double> input{{"pi", 3.14}, {"e", 2.71}};
        Value                                   v{input};

        std::unordered_map<std::string, double> dst;
        auto                                    result = assignTo(dst, v);
        expect(result.has_value());
        expect(eq(dst.size(), 2UZ));
        expect(eq(dst.at("pi"), 3.14));
    };

    "assignTo map<string, float> with widening"_test = [] {
        std::map<std::string, int> input{{"a", 1}, {"b", 2}};
        Value                      v{input};

        std::map<std::string, float> dst;
        auto                         result = assignTo<ConversionPolicy::Widening>(dst, v);
        expect(result.has_value());
        expect(eq(dst.at("a"), 1.0f));
        expect(eq(dst.at("b"), 2.0f));
    };

    "assignTo typed map with rvalue"_test = [] {
        std::map<std::string, int> input{{"x", 100}};
        Value                      v{input};

        std::map<std::string, int> dst;
        auto                       result = assignTo(dst, std::move(v));
        expect(result.has_value());
        expect(eq(dst.size(), 1UZ));
        expect(eq(dst.at("x"), 100));
    };

    "assignTo typed unordered_map with rvalue"_test = [] {
        std::unordered_map<std::string, double> input{{"val", 42.5}};
        Value                                   v{input};

        std::unordered_map<std::string, double> dst;
        auto                                    result = assignTo(dst, std::move(v));
        expect(result.has_value());
        expect(eq(dst.at("val"), 42.5));
    };
};

const boost::ut::suite<"ResourcePolicy"> _resource_policy = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "UseDefault uses provided resource"_test = [] {
        counting_resource cr;
        Value             v{Tensor<int>{data_from, {1, 2, 3}}};
        auto              result = convertTo<Tensor<int>, ConversionPolicy::Safe, RankPolicy::Strict, ResourcePolicy::UseDefault>(v, &cr);
        expect(result.has_value());
        expect(cr.alloc_count > 0UZ);
    };

    "InheritFromSource uses Value's resource"_test = [] {
        counting_resource cr_source;
        counting_resource cr_target;

        Value       v{Tensor<int>(data_from, {3}, &cr_source)};
        Tensor<int> fallback;
        v.or_else<Tensor<int>&>([&]() -> Tensor<int>& {
            expect(false) << "fallback should not be called";
            return fallback;
        }) = Tensor<int>(data_from, {1, 2, 3});

        cr_source.reset();
        cr_target.reset();

        auto result = convertTo<Tensor<int>, ConversionPolicy::Safe, RankPolicy::Strict, ResourcePolicy::InheritFromSource>(v, &cr_target);
        expect(result.has_value());
        expect(eq(cr_target.alloc_count, 0UZ)) << "should NOT use target resource";
    };
};

const boost::ut::suite<"memory_usage"> _memory_usage_suite = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "scalar"_test = [] {
        Value v{std::int32_t{42}};
        auto  usage = memory_usage(v);
        expect(usage >= sizeof(Value));
    };

    "string"_test = [] {
        Value v{std::string_view{"hello world"}};
        auto  usage = memory_usage(v);
        expect(usage > sizeof(Value));
    };

    "map"_test = [] {
        Value::Map m;
        m["key1"] = Value{1};
        m["key2"] = Value{2};
        Value v{std::move(m)};
        auto  usage = memory_usage(v);
        expect(usage > sizeof(Value));
    };
};

const boost::ut::suite<"error handling"> _error_handling = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "error has kind"_test = [] {
        Value v{std::string_view{"not a tensor"}};
        auto  result = convertTo<std::vector<int>>(v);
        expect(!result.has_value());
        expect(!result.error().hasIndex());
    };

    "element type mismatch has index"_test = [] {
        Tensor<Value> tv{Value{1}, Value{2.5}, Value{3}};
        Value         v{std::move(tv)};
        auto          result = convertTo<std::vector<int>>(v);
        expect(!result.has_value());
        expect(result.error().hasIndex());
        expect(eq(result.error().index, 1UZ));
    };

    "NotATensor for monostate"_test = [] {
        Value v;
        auto  result = convertTo<std::vector<int>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::NotATensor));
    };

    "NotATensor for scalar"_test = [] {
        Value v{std::int32_t{42}};
        auto  result = convertTo<std::vector<int>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::NotATensor));
    };

    "NotATensor for string"_test = [] {
        Value v{std::string_view{"hello"}};
        auto  result = convertTo<std::vector<std::uint8_t>>(v);
        expect(!result.has_value());
        expect(eq(result.error().kind, ConversionError::Kind::NotATensor));
    };
};

const boost::ut::suite<"edge cases"> _edge_cases = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::pmt;

    "large tensor"_test = [] {
        constexpr std::size_t N = 10000;
        std::vector<float>    data(N, 3.14f);
        Tensor<float>         largeTensor(data);
        Value                 v{std::move(largeTensor)};
        auto                  result = convertTo<std::vector<float>>(v);
        expect(result.has_value());
        expect(eq(result->size(), N));
    };

    "rank-3 tensor"_test = [] {
        Tensor<int> cube({2, 3, 4}, std::vector<int>(24, 1));
        Value       v{std::move(cube)};
        auto        result = convertTo<std::vector<int>, ConversionPolicy::Safe, RankPolicy::Flatten>(v);
        expect(result.has_value());
        expect(eq(result->size(), 24UZ));
    };

    "multiple conversions from same Value"_test = [] {
        Value v{Tensor<int>{data_from, {1, 2, 3}}};
        auto  result1 = convertTo<std::vector<int>>(v);
        auto  result2 = convertTo<std::vector<int>>(v);
        expect(result1.has_value());
        expect(result2.has_value());
        expect(eq(result1->size(), result2->size()));
    };

    "deeply nested map"_test = [] {
        std::map<std::string, Value> inner1{{"a", Value{1}}};
        std::map<std::string, Value> inner2{{"nested", Value{inner1}}};
        std::map<std::string, Value> outer{{"deep", Value{inner2}}};
        Value                        v{outer};
        auto                         result = convertTo<std::map<std::string, Value>>(v);
        expect(result.has_value());
        expect(result->at("deep").is_map());
    };
};

int main() { /* boost::ut auto-runs */ }
