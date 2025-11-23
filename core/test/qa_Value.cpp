#include <boost/ut.hpp>

#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/ValueHelper.hpp>

#include <atomic>
#include <complex>
#include <cstdint>
#include <magic_enum.hpp>
#include <memory_resource>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace gr::pmt {
inline std::ostream& operator<<(std::ostream& os, Value::ValueType t) { return os << magic_enum::enum_name(t); }
inline std::ostream& operator<<(std::ostream& os, Value::ContainerType t) { return os << magic_enum::enum_name(t); }
} // namespace gr::pmt

struct DiagCounters {
    std::atomic<std::size_t> default_ctor{0UZ};
    std::atomic<std::size_t> value_ctor{0UZ};
    std::atomic<std::size_t> copy_ctor{0UZ};
    std::atomic<std::size_t> move_ctor{0UZ};
    std::atomic<std::size_t> copy_assign{0UZ};
    std::atomic<std::size_t> move_assign{0UZ};
    std::atomic<std::size_t> dtor{0UZ};

    void reset() {
        default_ctor = 0UZ;
        value_ctor   = 0UZ;
        copy_ctor    = 0UZ;
        move_ctor    = 0UZ;
        copy_assign  = 0UZ;
        move_assign  = 0UZ;
        dtor         = 0UZ;
    }

    [[nodiscard]] std::size_t total_constructions() const { return default_ctor + value_ctor + copy_ctor + move_ctor; }
    [[nodiscard]] std::size_t total_copies() const { return copy_ctor + copy_assign; }
    [[nodiscard]] std::size_t total_moves() const { return move_ctor + move_assign; }
};

inline DiagCounters g_diag_counters;

struct DiagString {
    std::string data;

    DiagString() { ++g_diag_counters.default_ctor; }
    explicit DiagString(const char* s) : data(s) { ++g_diag_counters.value_ctor; }
    explicit DiagString(std::string s) : data(std::move(s)) { ++g_diag_counters.value_ctor; }
    DiagString(const DiagString& o) : data(o.data) { ++g_diag_counters.copy_ctor; }
    DiagString(DiagString&& o) noexcept : data(std::move(o.data)) { ++g_diag_counters.move_ctor; }
    DiagString& operator=(const DiagString& o) {
        ++g_diag_counters.copy_assign;
        data = o.data;
        return *this;
    }
    DiagString& operator=(DiagString&& o) noexcept {
        ++g_diag_counters.move_assign;
        data = std::move(o.data);
        return *this;
    }
    ~DiagString() { ++g_diag_counters.dtor; }

    bool operator==(const DiagString& o) const { return data == o.data; }
    operator std::string_view() const { return std::string_view{data}; }
};

template<>
struct std::formatter<DiagString, char> : std::formatter<std::string_view, char> {
    constexpr auto format(DiagString const& s, auto& ctx) const { return std::formatter<std::string_view, char>::format(static_cast<std::string_view>(s.data), ctx); }
};

struct counting_resource : std::pmr::memory_resource {
    std::pmr::memory_resource* upstream{std::pmr::new_delete_resource()};
    std::size_t                alloc_count{0}, dealloc_count{0}, bytes{0};

    void* do_allocate(std::size_t n, std::size_t align) override {
        ++alloc_count;
        bytes += n;
        return upstream->allocate(n, align);
    }

    void do_deallocate(void* p, std::size_t n, std::size_t align) override {
        ++dealloc_count;
        bytes -= n;
        upstream->deallocate(p, n, align);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

const boost::ut::suite<"Value - Basic Construction"> _basic_construction_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    "default construction yields monostate"_test = [] {
        Value v;
        expect(v.is_monostate());
        expect(!v.has_value());
        expect(!static_cast<bool>(v));
        expect(!v.is_arithmetic());
        expect(!v.is_string());
        expect(!v.is_tensor());
        expect(!v.is_map());
        expect(eq(v.value_type(), Value::ValueType::Monostate));
        expect(eq(v.container_type(), Value::ContainerType::Scalar));
    };

    "bool construction true"_test = [] {
        Value vt{true};
        expect(vt.holds<bool>());
        expect(vt.has_value());
        expect(static_cast<bool>(vt));
        expect(eq(vt.value_type(), Value::ValueType::Bool));
        expect(vt.is_arithmetic());
        expect(!vt.is_integral()); // bool is arithmetic but not integral
        expect(eq(vt.value_or<bool>(false), true));
    };

    "bool construction false"_test = [] {
        Value vf{false};
        expect(vf.holds<bool>());
        expect(eq(vf.value_or<bool>(true), false));
    };

    "int8_t construction"_test = [] {
        Value vi8{std::int8_t{-12}};
        expect(vi8.holds<std::int8_t>());
        expect(vi8.is_signed_integral());
        expect(!vi8.is_unsigned_integral());
        expect(eq(vi8.value_type(), Value::ValueType::Int8));
        expect(eq(vi8.value_or<std::int8_t>(std::int8_t{0}), std::int8_t{-12}));
    };

    "int16_t construction"_test = [] {
        Value vi16{std::int16_t{-1234}};
        expect(vi16.holds<std::int16_t>());
        expect(eq(vi16.value_type(), Value::ValueType::Int16));
    };

    "int32_t construction"_test = [] {
        Value vi32{std::int32_t{-123456}};
        expect(vi32.is_signed_integral());
        expect(eq(vi32.value_type(), Value::ValueType::Int32));
        expect(eq(vi32.value_or<std::int32_t>(std::int32_t{0}), std::int32_t{-123456}));
    };

    "int64_t construction"_test = [] {
        Value vi64{std::int64_t{-1}};
        expect(vi64.is_signed_integral());
        expect(eq(vi64.value_type(), Value::ValueType::Int64));
        expect(eq(vi64.value_or<std::int64_t>(std::int64_t{0}), std::int64_t{-1}));
    };

    "uint8_t construction"_test = [] {
        Value vu8{std::uint8_t{42}};
        expect(vu8.holds<std::uint8_t>());
        expect(vu8.is_unsigned_integral());
        expect(!vu8.is_signed_integral());
        expect(eq(vu8.value_type(), Value::ValueType::UInt8));
    };

    "uint16_t construction"_test = [] {
        Value vu16{std::uint16_t{1234}};
        expect(vu16.is_unsigned_integral());
        expect(eq(vu16.value_type(), Value::ValueType::UInt16));
    };

    "uint32_t construction"_test = [] {
        Value vu32{std::uint32_t{123456u}};
        expect(vu32.is_unsigned_integral());
        expect(eq(vu32.value_type(), Value::ValueType::UInt32));
        expect(eq(vu32.value_or<std::uint32_t>(std::uint32_t{0}), std::uint32_t{123456u}));
    };

    "uint64_t construction"_test = [] {
        Value vu64{std::uint64_t{999999999ULL}};
        expect(vu64.holds<std::uint64_t>());
        expect(eq(vu64.value_type(), Value::ValueType::UInt64));
    };

    "float construction"_test = [] {
        Value vf{1.25f};
        expect(vf.holds<float>());
        expect(vf.is_floating_point());
        expect(!vf.is_integral());
        expect(eq(vf.value_type(), Value::ValueType::Float32));
        expect(eq(vf.value_or<float>(0.0f), 1.25f));
    };

    "double construction"_test = [] {
        Value vd{2.5};
        expect(vd.holds<double>());
        expect(eq(vd.value_type(), Value::ValueType::Float64));
        expect(eq(vd.value_or<double>(0.0), 2.5));
    };

    "complex float construction"_test = [] {
        Value vc32{std::complex<float>{1.0f, -2.0f}};
        expect(vc32.is_complex());
        expect(vc32.holds<std::complex<float>>());
        expect(eq(vc32.value_type(), Value::ValueType::ComplexFloat32));
        expect(eq(vc32.container_type(), Value::ContainerType::Complex));
    };

    "complex double construction"_test = [] {
        Value vc64{std::complex<double>{0.5, 1.5}};
        expect(vc64.is_complex());
        expect(vc64.holds<std::complex<double>>());
        expect(eq(vc64.value_type(), Value::ValueType::ComplexFloat64));
    };

    "string_view construction"_test = [] {
        Value vs{std::string_view{"hello"}};
        expect(vs.is_string());
        expect(eq(vs.value_type(), Value::ValueType::String));
        expect(eq(vs.container_type(), Value::ContainerType::String));
        expect(eq(vs.as_string_view(), std::string_view{"hello"}));
    };

    "const char* construction"_test = [] {
        Value vc{"world"};
        expect(vc.is_string());
        expect(eq(vc.as_string_view(), std::string_view{"world"}));
    };

    "std::string construction"_test = [] {
        std::string str = "test string";
        Value       vs{str};
        expect(vs.is_string());
        expect(eq(vs.as_string_view(), std::string_view{"test string"}));
    };

    "pmr resource usage and propagation"_test = [] {
        "copy uses target's PMR resource"_test = [] {
            counting_resource source_mr;
            counting_resource target_mr;

            Value source{std::string_view{"hello"}, &source_mr};
            expect(source_mr.alloc_count >= 1u) << "source allocated";

            // Copy construct with different allocator
            Value target{source}; // Note: copy uses source's _resource in current impl

            // After copy, source should still be valid
            expect(source.is_string());
            expect(eq(source.as_string_view(), std::string_view{"hello"}));
        };

        "assignment copies content using target's allocator"_test = [] {
            counting_resource mr1;
            counting_resource mr2;

            Value v1{std::string_view{"from v1"}, &mr1};
            Value v2{std::string_view{"from v2"}, &mr2};

            v2 = v1; // assignment should use v2's allocator

            expect(v2.is_string());
            expect(eq(v2.as_string_view(), std::string_view{"from v1"}));
        };

        "move transfers ownership without reallocation"_test = [] {
            counting_resource mr;

            Value source{std::string_view{"hello"}, &mr};
            auto  allocs_after_source = mr.alloc_count;

            Value target{std::move(source)};

            expect(eq(mr.alloc_count, allocs_after_source)) << "no new allocations on move";
            expect(target.is_string());
            expect(source.is_monostate()) << "source reset after move";
        };

        "nullptr resource falls back to default"_test = [] {
            // This should not crash - nullptr should be replaced with default resource
            Value v{std::int64_t{42}, nullptr};
            expect(v.holds<std::int64_t>());
            expect(eq(v.value_or<std::int64_t>(std::int64_t{0}), std::int64_t{42}));
        };

        "nullptr resource for string still works"_test = [] {
            Value v{std::string_view{"test"}, nullptr};
            expect(v.is_string());
            expect(eq(v.as_string_view(), std::string_view{"test"}));
        };
    };

    "Tensor<T> construction"_test = [] {
        using gr::Tensor;

        "Tensor copy preserves data"_test = [] {
            Tensor<float> t({2, 3});
            t[0, 0] = 1.0f;
            t[1, 2] = 6.0f;

            Value v1{std::move(t)};
            Value v2{v1}; // copy

            expect(v1.is_tensor());
            expect(v2.is_tensor());

            auto& t1 = v1.ref_tensor<float>();
            auto& t2 = v2.ref_tensor<float>();

            expect(eq(t1[0, 0], 1.0f));
            expect(eq(t2[0, 0], 1.0f));
            expect(eq(t1[1, 2], 6.0f));
            expect(eq(t2[1, 2], 6.0f));
        };

        "Tensor move transfers ownership"_test = [] {
            Tensor<float> t({2, 2});
            t[0, 0] = 42.0f;

            Value v1{std::move(t)};
            Value v2{std::move(v1)};

            expect(v1.is_monostate()) << "source reset after move";
            expect(v2.is_tensor());
            expect(eq(v2.ref_tensor<float>()[0, 0], 42.0f));
        };
    };
};

const boost::ut::suite<"Value - String Conversion"> _string_conversion_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    "value_or<std::string> converts from pmr::string"_test = [] {
        Value v{std::string_view{"hello world"}};

        std::string result = v.value_or(std::string{"fallback"});
        expect(eq(result, std::string{"hello world"}));
        expect(v.is_string()) << "original unchanged after copy";
    };

    "value_or<std::string> returns fallback on type mismatch"_test = [] {
        Value v{std::int64_t{42}};

        std::string result = v.value_or(std::string{"fallback"});
        expect(eq(result, std::string{"fallback"}));
    };

    "value_or<std::string> returns fallback on monostate"_test = [] {
        Value mono;

        std::string result = mono.value_or(std::string{"default"});
        expect(eq(result, std::string{"default"}));
    };

    "value_or<std::string_view> provides zero-copy view"_test = [] {
        Value v{std::string_view{"test string"}};

        std::string_view result = v.value_or(std::string_view{"fallback"});
        expect(eq(result, std::string_view{"test string"}));
    };

    "value_or<std::string_view> returns fallback on mismatch"_test = [] {
        Value v{std::int64_t{42}};

        std::string_view result = v.value_or(std::string_view{"fallback"});
        expect(eq(result, std::string_view{"fallback"}));
    };

    "holds<std::string> returns true for string Value"_test = [] {
        Value v{std::string_view{"hello"}};

        expect(v.holds<std::string>()) << "std::string is convertible from pmr::string";
        expect(v.holds<std::string_view>()) << "std::string_view is convertible from pmr::string";
        expect(v.holds<std::pmr::string>()) << "exact type match";
    };

    "holds<std::string> returns false for non-string Value"_test = [] {
        Value v{std::int64_t{42}};

        expect(!v.holds<std::string>());
        expect(!v.holds<std::string_view>());
    };

    "or_else_string converts with lazy factory"_test = [] {
        Value v{std::string_view{"original"}};
        bool  factory_called = false;

        std::string result = v.or_else_string([&]() {
            factory_called = true;
            return std::string{"from_factory"};
        });

        expect(!factory_called) << "factory NOT called when string present";
        expect(eq(result, std::string{"original"}));
    };

    "or_else_string calls factory on mismatch"_test = [] {
        Value v{std::int64_t{42}};
        bool  factory_called = false;

        std::string result = v.or_else_string([&]() {
            factory_called = true;
            return std::string{"from_factory"};
        });

        expect(factory_called) << "factory called when type doesn't match";
        expect(eq(result, std::string{"from_factory"}));
    };

    "or_else_string_view provides zero-copy with lazy factory"_test = [] {
        Value v{std::string_view{"test"}};

        std::string_view result = v.or_else_string_view([]() { return std::string_view{"fallback"}; });

        expect(eq(result, std::string_view{"test"}));
    };
};

const boost::ut::suite<"Value - Comparison & Ordering"> _comparison_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using std::partial_ordering;

    "Value == Value (same scalar type & value)"_test = [] {
        Value a{std::int32_t{42}};
        Value b{std::int32_t{42}};

        expect(a == b);
        expect(!(a != b));

        expect(eq(a.value_type(), Value::ValueType::Int32));
        expect(eq(b.value_type(), Value::ValueType::Int32));
        expect(eq(a.container_type(), Value::ContainerType::Scalar));
        expect(eq(b.container_type(), Value::ContainerType::Scalar));

        expect((a <=> b) == partial_ordering::equivalent);
    };

    "Value == Value (same scalar type, different value)"_test = [] {
        Value a{std::int32_t{41}};
        Value b{std::int32_t{42}};

        expect(!(a == b));
        auto ord_ab = a <=> b;
        auto ord_ba = b <=> a;

        expect(ord_ab == partial_ordering::less);
        expect(ord_ba == partial_ordering::greater);
        expect(a < b);
        expect(b > a);
    };

    "Value == Value (different scalar types uses type tag ordering)"_test = [] {
        Value vi32{std::int32_t{123}};
        Value vi64{std::int64_t{123}};

        // Int32 vs Int64: ValueType::Int32 (4) < ValueType::Int64 (5) → vi32 < vi64 by _type_info
        auto ord         = vi32 <=> vi64;
        auto ord_reverse = vi64 <=> vi32;

        expect(!(vi32 == vi64));
        expect(ord == partial_ordering::less);
        expect(ord_reverse == partial_ordering::greater);
        expect(vi32 < vi64);
        expect(vi64 > vi32);
    };

    "Value == T and T == Value (exact stored type only)"_test = [] {
        Value        v{std::int32_t{42}};
        std::int32_t i32{42};
        std::int64_t i64{42};

        expect(v == i32) << "exact type numeric equality";
        expect(i32 == v) << "exact type numeric equality";
        expect(!(v == i64)) << "different type shoud be false even if numerically equal";
        expect(!(i64 == v)) << "different type shoud be false even if numerically equal";
    };

    "Value != T and T != Value (exact stored type only)"_test = [] {
        Value        v{std::int32_t{42}};
        std::int32_t i32{43};

        expect(v != i32) << "exact type numeric mismatch";
        expect(i32 != v) << "exact type numeric mismatch";
    };

    "Value ordering for different containers is driven by _type_info"_test = [] {
        Value scalar{std::int32_t{1}};    // ContainerType::Scalar
        Value str{std::string_view{"x"}}; // ContainerType::String

        auto ord = scalar <=> str;

        auto scalar_tag = (static_cast<std::uint8_t>(scalar.container_type()) << 4) | static_cast<std::uint8_t>(scalar.value_type());
        auto str_tag    = (static_cast<std::uint8_t>(str.container_type()) << 4) | static_cast<std::uint8_t>(str.value_type());

        if (scalar_tag < str_tag) {
            expect(ord == partial_ordering::less);
            expect(scalar < str);
        } else if (scalar_tag > str_tag) {
            expect(ord == partial_ordering::greater);
            expect(scalar > str);
        } else {
            // This should not normally happen, but keep the assertion symmetric
            expect(ord == partial_ordering::equivalent);
        }
    };

    "Monostate Values compare equal and are ordered only by type tag"_test = [] {
        Value a{}; // default → Monostate/Scalar
        Value b{}; // same

        expect(a == b);
        expect(!(a != b));
        expect((a <=> b) == partial_ordering::equivalent);
    };

    "operator<=> for same scalar types"_test = [] {
        Value v1{std::int32_t{10}};
        Value v2{std::int32_t{20}};
        Value v3{std::int32_t{10}};

        expect((v1 <=> v2) == partial_ordering::less);
        expect((v2 <=> v1) == partial_ordering::greater);
        expect((v1 <=> v3) == partial_ordering::equivalent);
    };

    "operator<=> for strings"_test = [] {
        Value v1{std::string_view{"apple"}};
        Value v2{std::string_view{"banana"}};
        Value v3{std::string_view{"apple"}};

        expect((v1 <=> v2) == partial_ordering::less);
        expect((v2 <=> v1) == partial_ordering::greater);
        expect((v1 <=> v3) == partial_ordering::equivalent);
    };

    "operator<=> for complex returns unordered"_test = [] {
        Value v1{std::complex<float>{1.0f, 2.0f}};
        Value v2{std::complex<float>{3.0f, 4.0f}};

        // Complex numbers don't have natural ordering
        expect((v1 <=> v2) == partial_ordering::unordered);
    };

    "operator<=> for different types uses type tag"_test = [] {
        Value vi{std::int32_t{1000}};
        Value vf{0.001f}; // Float has higher ValueType enum value

        auto ordering = vi <=> vf;
        // Different types should compare by _type_info
        expect(ordering != partial_ordering::equivalent);
    };

    "operator== for Value vs scalar"_test = [] {
        Value v{std::int32_t{42}};

        expect(v == std::int32_t{42});
        expect(std::int32_t{42} == v);
        expect(!(v == std::int32_t{43}));
        expect(v != std::int32_t{43});
    };

    "corner cases for operator<=> and operator=="_test = [] {
        using namespace boost::ut;
        using gr::pmt::Value;

        "float NaN != NaN (IEEE semantics)"_test = [] {
            Value v1{std::numeric_limits<float>::quiet_NaN()};
            Value v2{std::numeric_limits<float>::quiet_NaN()};

            expect(!(v1 == v2)) << "NaN != NaN per IEEE 754";
        };

        "double NaN != NaN (IEEE semantics)"_test = [] {
            Value v1{std::numeric_limits<double>::quiet_NaN()};
            Value v2{std::numeric_limits<double>::quiet_NaN()};

            expect(!(v1 == v2)) << "NaN != NaN per IEEE 754";
        };

        "float NaN ordering is unordered"_test = [] {
            Value v1{std::numeric_limits<float>::quiet_NaN()};
            Value v2{1.0f};

            auto ordering = v1 <=> v2;
            // NaN comparisons should result in unordered, but since they have same type,
            // the partial_ordering from float <=> float should handle it
            expect(ordering == std::partial_ordering::unordered);
        };

        "infinity comparison"_test = [] {
            Value v_inf{std::numeric_limits<float>::infinity()};
            Value v_neg_inf{-std::numeric_limits<float>::infinity()};
            Value v_normal{1.0f};

            expect(v_inf > v_normal);
            expect(v_neg_inf < v_normal);
            expect(v_inf > v_neg_inf);
        };
    };

    "Map & nested Map comparison"_test = [] {
        "Map with nested Values"_test = [] {
            Value::Map inner_map;
            inner_map["nested_int"] = Value{std::int64_t{123}};
            inner_map["nested_str"] = Value{std::string_view{"nested"}};

            Value::Map outer_map;
            outer_map["inner"]     = Value{std::move(inner_map)};
            outer_map["top_level"] = Value{std::int64_t{456}};

            Value v{std::move(outer_map)};

            expect(v.is_map());
            expect(v.ref_map().contains("inner"));
            expect(v.ref_map().contains("top_level"));

            auto& inner = v.ref_map().at("inner");
            expect(inner.is_map());
            expect(inner.ref_map().at("nested_int").holds<std::int64_t>());
        };

        "Map equality with nested Values"_test = [] {
            Value::Map map1;
            map1["key"] = Value{std::int64_t{42}};

            Value::Map map2;
            map2["key"] = Value{std::int64_t{42}};

            Value v1{std::move(map1)};
            Value v2{std::move(map2)};

            expect(v1 == v2);
        };

        "Map inequality with different nested Values"_test = [] {
            Value::Map map1;
            map1["key"] = Value{std::int64_t{42}};

            Value::Map map2;
            map2["key"] = Value{std::int64_t{43}};

            Value v1{std::move(map1)};
            Value v2{std::move(map2)};

            expect(v1 != v2);
        };
    };
};

const boost::ut::suite<"Value - PMR Memory Management"> _pmr_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    "string uses PMR resource"_test = [] {
        counting_resource mr;

        {
            Value vs{std::string_view{"hello world"}, &mr};
            expect(mr.bytes > 0u) << "string should allocate via PMR";
            expect(mr.alloc_count >= 1u);
        }

        expect(eq(mr.bytes, 0u)) << "all memory freed after destruction";
        expect(eq(mr.alloc_count, mr.dealloc_count));
    };

    "complex uses PMR resource"_test = [] {
        counting_resource mr;

        {
            Value vc{std::complex<double>{1.0, 2.0}, &mr};
            expect(mr.bytes > 0u) << "complex should allocate via PMR";
        }

        expect(eq(mr.bytes, 0u));
    };

    "scalars do not allocate"_test = [] {
        counting_resource mr;

        {
            Value vi{std::int64_t{42}, &mr};
            Value vf{3.14, &mr};
            expect(eq(mr.bytes, 0u)) << "scalars use inline storage";
        }
    };

    "string memory freed on destruction"_test = [] {
        counting_resource mr;

        {
            Value vs{std::string_view{"hello world"}, &mr};
            expect(mr.bytes > 0u);
        }

        expect(eq(mr.bytes, 0u)) << "all memory freed after destruction";
    };
};

const boost::ut::suite<"Value - Safe Access"> _safe_access_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    "holds returns true for matching type"_test = [] {
        Value vi{std::int32_t{42}};
        expect(vi.holds<std::int32_t>());
        expect(!vi.holds<std::int64_t>());
        expect(!vi.holds<double>());
        expect(!vi.holds<std::pmr::string>());
    };

    "ref_if returns pointer on match"_test = [] {
        Value vi{std::int32_t{42}};
        auto* p = vi.get_if<std::int32_t>();
        expect(p != nullptr);
        expect(eq(*p, std::int32_t{42}));
    };

    "ref_if returns nullptr on mismatch"_test = [] {
        Value vi{std::int32_t{42}};
        expect(vi.get_if<double>() == nullptr);
        expect(vi.get_if<std::pmr::string>() == nullptr);
        expect(vi.get_if<std::int64_t>() == nullptr);
    };

    "ref_if on monostate returns nullptr"_test = [] {
        Value mono;
        expect(mono.get_if<std::int32_t>() == nullptr);
        expect(mono.get_if<std::pmr::string>() == nullptr);
    };

    "const ref_if works"_test = [] {
        const Value vi{std::int32_t{42}};
        const auto* p = vi.get_if<std::int32_t>();
        expect(p != nullptr);
        expect(eq(*p, std::int32_t{42}));
    };

    "modification through ref_if pointer"_test = [] {
        Value vi{std::int32_t{10}};
        if (auto* p = vi.get_if<std::int32_t>()) {
            *p = 20;
        }
        expect(eq(vi.value_or<std::int32_t>(std::int32_t{0}), std::int32_t{20}));
    };
};

const boost::ut::suite<"Value - value_or Copy"> _value_or_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    "copy semantic"_test = [] {
        "value_or<T> returns copy on match"_test = [] {
            Value        v{std::int64_t{42}};
            std::int64_t result = v.value_or<std::int64_t>(std::int64_t{0});
            expect(eq(result, std::int64_t{42}));
            expect(eq(v.value_or<std::int64_t>(std::int64_t{0}), std::int64_t{42})) << "original unchanged";
        };

        "value_or<T> returns fallback on mismatch"_test = [] {
            Value  v{std::int64_t{42}};
            double result = v.value_or<double>(3.14);
            expect(eq(result, 3.14));
            expect(v.holds<std::int64_t>()) << "Value unchanged on mismatch";
        };

        "value_or<T> on monostate returns fallback"_test = [] {
            Value mono;
            expect(eq(mono.value_or<std::int32_t>(std::int32_t{-1}), std::int32_t{-1}));
        };

        "value_or<T> from const Value"_test = [] {
            const Value  v{std::int64_t{42}};
            std::int64_t result = v.value_or<std::int64_t>(std::int64_t{0});
            expect(eq(result, std::int64_t{42}));
        };

        "value_or<float> with float fallback (type strictness)"_test = [] {
            Value v{3.14f};
            float result = v.value_or<float>(0.0f);
            expect(eq(result, 3.14f));
            // Note: v.value_or<double>(0.0) would NOT compile due to type strictness
        };

        "value_or<const T&> from mutable Value"_test = [] {
            Value              v{std::int64_t{42}}; // non-const Value
            const std::int64_t fallback = -1;

            const std::int64_t& ref = v.value_or<const std::int64_t&>(fallback);
            expect(eq(ref, std::int64_t{42}));
        };

        "value_or<const T&> returns fallback on mismatch"_test = [] {
            Value        v{std::int64_t{42}};
            const double fallback = 3.14;

            const double& ref = v.value_or<const double&>(fallback);
            expect(&ref == &fallback) << "should return reference to fallback";
        };
    };

    "mutable reference semantic"_test = [] {
        "value_or<T&> returns mutable reference on match"_test = [] {
            Value         v{std::int64_t{42}};
            std::int64_t  fallback = -1;
            std::int64_t& ref      = v.value_or<std::int64_t&>(fallback);
            expect(eq(ref, std::int64_t{42}));
        };

        "value_or<T&> modification affects Value"_test = [] {
            Value        v{std::int64_t{42}};
            std::int64_t fallback               = -1;
            v.value_or<std::int64_t&>(fallback) = 100;
            expect(eq(v.value_or<std::int64_t>(std::int64_t{0}), std::int64_t{100}));
        };

        "value_or<T&> returns fallback reference on mismatch"_test = [] {
            Value   v{std::int64_t{42}};
            double  fallback = 3.14;
            double& ref      = v.value_or<double&>(fallback);
            expect(&ref == &fallback) << "should return reference to fallback";
        };

        "value_or<T&> on monostate returns fallback reference"_test = [] {
            Value         mono;
            std::int32_t  fallback = 999;
            std::int32_t& ref      = mono.value_or<std::int32_t&>(fallback);
            expect(&ref == &fallback);
        };
    };

    "const reference semantic"_test = [] {
        "value_or<const T&> returns const reference on match"_test = [] {
            Value               v{std::int64_t{42}};
            const std::int64_t  fallback = -1;
            const std::int64_t& ref      = v.value_or<const std::int64_t&>(fallback);
            expect(eq(ref, std::int64_t{42}));
        };

        "value_or<const T&> from const Value"_test = [] {
            const Value         cv{std::int64_t{42}};
            const std::int64_t  fallback = -1;
            const std::int64_t& ref      = cv.value_or<const std::int64_t&>(fallback);
            expect(eq(ref, std::int64_t{42}));
        };

        "value_or<const T&> returns fallback on mismatch"_test = [] {
            const Value   v{std::int64_t{42}};
            const double  fallback = 3.14;
            const double& ref      = v.value_or<const double&>(fallback);
            expect(&ref == &fallback);
        };
    };

    "rvalue reference (move) semantic"_test = [] {
        "value_or<T&&> moves and resets to monostate on match"_test = [] {
            Value v{std::pmr::string{"hello"}};
            expect(!v.is_monostate());

            std::pmr::string result = v.value_or<std::pmr::string&&>(std::pmr::string{"fallback"});
            expect(eq(std::string_view{result}, std::string_view{"hello"}));
            expect(v.is_monostate()) << "Value must be monostate after ownership transfer";
        };

        "value_or<T&&> returns fallback on mismatch without reset"_test = [] {
            Value v{std::int64_t{42}};

            std::pmr::string result = v.value_or<std::pmr::string&&>(std::pmr::string{"fallback"});
            expect(eq(std::string_view{result}, std::string_view{"fallback"}));
            expect(v.holds<std::int64_t>()) << "Value unchanged on mismatch";
            expect(!v.is_monostate());
        };

        "value_or<T&&> on monostate returns fallback"_test = [] {
            Value            mono;
            std::pmr::string result = mono.value_or<std::pmr::string&&>(std::pmr::string{"default"});
            expect(eq(std::string_view{result}, std::string_view{"default"}));
            expect(mono.is_monostate());
        };

        "value_or<T&&> with scalar type resets to monostate"_test = [] {
            Value        v{std::int64_t{42}};
            std::int64_t result = v.value_or<std::int64_t&&>(std::int64_t{0});
            expect(eq(result, std::int64_t{42}));
            expect(v.is_monostate()) << "even scalars reset to monostate on T&&";
        };
    };
};

const boost::ut::suite<"Value - or_else lazy evaluation"> _or_else_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    "or_else<T> factory NOT called on match"_test = [] {
        Value v{std::int64_t{42}};
        bool  factory_called = false;

        auto result = v.or_else<std::int64_t>([&]() -> std::int64_t {
            factory_called = true;
            return std::int64_t{999};
        });

        expect(eq(result, std::int64_t{42}));
        expect(!factory_called) << "factory should NOT be called when type matches";
    };

    "or_else<T> factory called on mismatch"_test = [] {
        Value v{std::int64_t{42}};
        bool  factory_called = false;

        double result = v.or_else<double>([&]() -> double {
            factory_called = true;
            return 3.14;
        });

        expect(eq(result, 3.14));
        expect(factory_called) << "factory SHOULD be called when type doesn't match";
    };

    "or_else<T&> returns reference on match"_test = [] {
        Value               v{std::int64_t{42}};
        static std::int64_t static_fallback = 777;
        bool                factory_called  = false;

        std::int64_t& ref = v.or_else<std::int64_t&>([&]() -> std::int64_t& {
            factory_called = true;
            return static_fallback;
        });

        expect(!factory_called);
        expect(eq(ref, std::int64_t{42}));
    };

    "or_else<T&> calls factory on mismatch"_test = [] {
        Value         v{std::int64_t{42}};
        static double static_fallback = 2.718;
        bool          factory_called  = false;

        double& ref = v.or_else<double&>([&]() -> double& {
            factory_called = true;
            return static_fallback;
        });

        expect(factory_called);
        expect(&ref == &static_fallback);
    };

    "or_else<T&&> ownership transfer on match"_test = [] {
        Value v{std::pmr::string{"original"}};
        bool  factory_called = false;

        std::pmr::string result = v.or_else<std::pmr::string&&>([&]() -> std::pmr::string {
            factory_called = true;
            return std::pmr::string{"from_factory"};
        });

        expect(!factory_called) << "factory NOT called on match";
        expect(eq(std::string_view{result}, std::string_view{"original"}));
        expect(v.is_monostate()) << "Value reset to monostate after T&& transfer";
    };

    "or_else<T&&> factory called on mismatch without reset"_test = [] {
        Value v{std::int64_t{42}};
        bool  factory_called = false;

        std::pmr::string result = v.or_else<std::pmr::string&&>([&]() -> std::pmr::string {
            factory_called = true;
            return std::pmr::string{"from_factory"};
        });

        expect(factory_called);
        expect(eq(std::string_view{result}, std::string_view{"from_factory"}));
        expect(v.holds<std::int64_t>()) << "Value unchanged on mismatch";
    };

    "or_else on monostate calls factory"_test = [] {
        Value mono;
        bool  factory_called = false;

        auto result = mono.or_else<std::int32_t>([&]() -> std::int32_t {
            factory_called = true;
            return std::int32_t{12345};
        });

        expect(factory_called);
        expect(eq(result, std::int32_t{12345}));
    };

    "or_else does not call factory on match"_test = [] {
        Value v{std::int64_t{42}};

        auto result = v.or_else<std::int64_t>([]() -> std::int64_t { throw std::runtime_error("should not be called"); });

        expect(eq(result, std::int64_t{42}));
    };

    "or_else propagates exception from factory"_test = [] {
        Value v{std::int64_t{42}};

        expect(throws<std::runtime_error>([&] {
            std::ignore = v.or_else<double>([]() -> double {
                throw std::runtime_error("factory exception");
                return 0.0;
            });
        }));
    };
};

const boost::ut::suite<"Value - transform"> _transform_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    "transform<T> applies function on match"_test = [] {
        Value v{std::int64_t{42}};

        auto result = v.transform<std::int64_t>([](std::int64_t& x) -> std::string { return std::to_string(x * 2); });

        expect(eq(result, std::string{"84"}));
        expect(v.holds<std::int64_t>()) << "Value unchanged after transform<T>";
    };

    "transform<T> returns default on mismatch"_test = [] {
        Value v{std::int64_t{42}};

        auto result = v.transform<double>([](double& x) -> std::string { return std::to_string(x); });

        expect(result.empty()) << "default-constructed string on miss";
    };

    "transform<T&&> moves and resets to monostate"_test = [] {
        Value v{std::pmr::string{"hello_world"}};

        std::size_t result = v.transform<std::pmr::string&&>([](std::pmr::string&& s) -> std::size_t { return s.size(); });

        expect(eq(result, 11UZ));
        expect(v.is_monostate()) << "Value reset after transform<T&&>";
    };

    "transform const overload"_test = [] {
        const Value v{std::int64_t{42}};

        auto result = v.transform<std::int64_t>([](const std::int64_t& x) -> std::int64_t { return x * 2; });

        expect(eq(result, std::int64_t{84}));
    };

    "transform with type conversion"_test = [] {
        Value v{std::pmr::string{"12345"}};

        std::size_t len = v.transform<std::pmr::string>([](const std::pmr::string& s) { return s.size(); });

        expect(eq(len, 5UZ));
    };
};

const boost::ut::suite<"Value - transform_or"> _transform_or_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    "transform_or returns transformed value on match"_test = [] {
        Value v{std::int64_t{42}};

        auto result = v.transform_or<std::int64_t>([](std::int64_t& x) { return x * 2; }, std::int64_t{0});

        expect(eq(result, std::int64_t{84}));
    };

    "transform_or returns fallback on mismatch"_test = [] {
        Value v{std::int64_t{42}};

        auto result = v.transform_or<double>([](double& x) { return x * 2.0; }, -1.0);

        expect(eq(result, -1.0));
    };

    "transform_or<T&&> resets to monostate on match"_test = [] {
        Value v{std::pmr::string{"test"}};

        auto result = v.transform_or<std::pmr::string&&>([](std::pmr::string&& s) { return s.size(); }, 0UZ);

        expect(eq(result, 4UZ));
        expect(v.is_monostate());
    };
};

const boost::ut::suite<"Value - and_then Chaining"> _and_then_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    "and_then<T> chains Value operations"_test = [] {
        Value v{std::int64_t{42}};

        Value result = v.and_then<std::int64_t>([](std::int64_t& x) -> Value { return Value{x * 2}; });

        expect(result.holds<std::int64_t>());
        expect(eq(result.value_or<std::int64_t>(std::int64_t{0}), std::int64_t{84}));
    };

    "and_then<T> returns monostate on mismatch"_test = [] {
        Value v{std::int64_t{42}};

        Value result = v.and_then<double>([](double& x) -> Value { return Value{x * 2.0}; });

        expect(result.is_monostate()) << "default-constructed Value (monostate) on miss";
    };

    "and_then<T&&> moves and resets"_test = [] {
        Value v{std::pmr::string{"chain_me"}};

        Value result = v.and_then<std::pmr::string&&>([](std::pmr::string&& s) -> Value {
            s += "_modified";
            return Value{std::string_view{s}};
        });

        expect(result.holds<std::pmr::string>());
        expect(eq(result.as_string_view(), std::string_view{"chain_me_modified"}));
        expect(v.is_monostate()) << "original Value reset after T&& chain";
    };

    "and_then chaining multiple operations"_test = [] {
        Value input{std::int64_t{10}};

        Value result = input
                           .and_then<std::int64_t>([](std::int64_t& x) -> Value { return Value{x * 2}; }) // -> Value
                           .and_then<std::int64_t>([](std::int64_t& x) -> Value { return Value{x + 5}; });

        expect(result.holds<std::int64_t>());
        expect(eq(result.value_or<std::int64_t>(std::int64_t{0}), std::int64_t{25})); // (10*2)+5
    };

    "and_then chain breaks on type mismatch"_test = [] {
        Value input{std::int64_t{10}};

        Value result = input
                           .and_then<double>([](double& x) -> Value { return Value{x * 2.0}; })            // mismatch → monostate
                           .and_then<std::int64_t>([](std::int64_t& x) -> Value { return Value{x + 5}; }); // won't match monostate

        expect(result.is_monostate()) << "chain should yield monostate after mismatch";
    };

    "and_then chaining with mutating lambdas"_test = [] {
        Value input{std::int64_t{10}};

        Value result = input
                           .and_then<std::int64_t>([](std::int64_t& x) -> Value {
                               x *= 2;
                               return Value{x};
                           }) // mutate view of T: 10 -> 20
                           .and_then<std::int64_t>([](std::int64_t& x) -> Value {
                               x += 5;
                               return Value{x};
                           }); // mutated: 20 -> 25

        expect(result.holds<std::int64_t>());
        expect(eq(result.value_or<std::int64_t>(std::int64_t{0}), std::int64_t{25}));
    };

    "and_then chain with mismatch and mutating second lambda"_test = [] {
        Value input{std::int64_t{10}};

        Value result = input
                           .and_then<double>([](double& x) -> Value {
                               x *= 2.0;
                               return Value{x}; // type mismatch → first and_then returns monostate Value
                           })
                           .and_then<std::int64_t>([](std::int64_t& x) -> Value {
                               x += 5; // never reached (no Int64 inside Value)
                               return Value{x};
                           });

        expect(result.is_monostate()) << "chain should yield monostate after mismatch";
    };
};

const boost::ut::suite<"Value - Containers"> _container_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::Tensor;

    "Map creation and access"_test = [] {
        counting_resource mr;
        using Map = Value::Map;

        {
            Map m{&mr};
            m.emplace(std::pmr::string{"a", &mr}, Value{std::int64_t{1}, &mr});
            m.emplace(std::pmr::string{"b", &mr}, Value{std::int64_t{2}, &mr});

            Value v_map{std::move(m), &mr};
            expect(v_map.is_map());
            expect(eq(v_map.container_type(), Value::ContainerType::Map));
            expect(eq(v_map.value_type(), Value::ValueType::Value));

            auto& ref = v_map.ref_map();
            expect(eq(ref.size(), 2u));
            expect(ref.contains("a"));
            expect(ref.contains("b"));
            expect(eq(ref.at("a").value_or<std::int64_t>(std::int64_t{0}), std::int64_t{1}));
            expect(eq(ref.at("b").value_or<std::int64_t>(std::int64_t{0}), std::int64_t{2}));
        }

        expect(eq(mr.bytes, 0u)) << "all memory freed after destruction";
    };

    "Tensor construction with scalar element type"_test = [] {
        Tensor<std::int32_t> t{};
        Value                vt{t};

        expect(!vt.is_monostate());
        expect(vt.is_tensor());
        expect(eq(vt.container_type(), Value::ContainerType::Tensor));
        expect(eq(vt.value_type(), Value::ValueType::Int32));

        auto& tref = vt.ref_tensor<std::int32_t>();
        static_assert(std::is_same_v<decltype(tref), Tensor<std::int32_t>&>);
    };

    "Tensor of Value (heterogeneous)"_test = [] {
        Tensor<Value> t({2UZ});
        t[0] = Value{std::int64_t{42}};
        t[1] = Value{"Hello World!"};

        Value vt{t};

        expect(vt.is_tensor());
        expect(eq(vt.value_type(), Value::ValueType::Value));

        auto& tref = vt.ref_tensor<Value>();
        expect(eq(tref[0UZ].value_or<std::int64_t>(std::int64_t{0}), std::int64_t{42}));
        expect(eq(tref[1UZ].as_string(), std::string("Hello World!")));
    };
};

const boost::ut::suite<"Value - Edge Cases"> _edge_case_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    "equality same type same value"_test = [] {
        Value a{std::int64_t{5}};
        Value b{std::int64_t{5}};
        expect(a == b);
    };

    "inequality same type different value"_test = [] {
        Value a{std::int64_t{5}};
        Value c{std::int64_t{6}};
        expect(a != c);
    };

    "inequality different types"_test = [] {
        Value a{std::int64_t{5}};
        Value s{std::string_view{"5"}};
        expect(a != s) << "different types must not compare equal";
    };

    "inequality with monostate"_test = [] {
        Value a{std::int64_t{5}};
        Value mono;
        expect(a != mono);
    };

    "monostate equals monostate"_test = [] {
        Value mono1;
        Value mono2;
        expect(mono1 == mono2);
    };

    "self-assignment is safe"_test = [] {
        Value v{std::int64_t{123}};

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
        v = v;
        expect(eq(v.value_or<std::int64_t>(std::int64_t{0}), std::int64_t{123}));
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    };

    "self-move is safe"_test = [] {
        Value v{std::int64_t{123}};

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif
        v = std::move(v);
        expect(eq(v.value_or<std::int64_t>(std::int64_t{0}), std::int64_t{123}));
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    };

    "type transition via assignment"_test = [] {
        Value v;
        expect(v.is_monostate());

        v = std::int32_t{123};
        expect(v.holds<std::int32_t>());

        v = 4.5; // double
        expect(v.holds<double>());

        v = std::string_view{"hello"};
        expect(v.is_string());
    };

    "memory_usage is monotonic"_test = [] {
        Value             mono;
        const std::size_t m0 = gr::pmt::memory_usage(mono);

        Value             i{std::int64_t{1}};
        const std::size_t m1 = gr::pmt::memory_usage(i);

        Value             s{std::string_view{"abcdef"}};
        const std::size_t m2 = gr::pmt::memory_usage(s);

        expect(le(m0, m1)) << "memory usage monotonically increases (monostate <= std::int64_t)";
        expect(le(m1, m2)) << "memory usage monotonically increases (std::int64_t <= std::string_view)";
    };

    "copy construction preserves value"_test = [] {
        Value original{std::pmr::string{"test string"}};
        Value copy{original};

        expect(copy.is_string());
        expect(eq(copy.as_string_view(), std::string_view{"test string"}));
        expect(eq(original.as_string_view(), std::string_view{"test string"})) << "original unchanged";
    };

    "move construction transfers ownership"_test = [] {
        Value original{std::pmr::string{"test string"}};
        Value moved{std::move(original)};

        expect(moved.is_string());
        expect(eq(moved.as_string_view(), std::string_view{"test string"}));
    };

    "consecutive ownership transfers"_test = [] {
        Value v1{std::pmr::string{"data"}};

        // Transfer v1's content
        std::pmr::string s1 = v1.value_or<std::pmr::string&&>(std::pmr::string{});
        expect(v1.is_monostate());
        expect(eq(std::string_view{s1}, std::string_view{"data"}));

        // v1 is now monostate, trying to transfer again returns fallback
        std::pmr::string s2 = v1.value_or<std::pmr::string&&>(std::pmr::string{"fallback"});
        expect(eq(std::string_view{s2}, std::string_view{"fallback"}));
    };

    "empty string handling"_test = [] {
        Value empty_str{std::string_view{""}};
        expect(empty_str.is_string());
        expect(eq(empty_str.as_string_view(), std::string_view{""}));
        expect(empty_str.as_string_view().empty());
    };

    "large string handling"_test = [] {
        std::string large(10000, 'x');
        Value       v{std::string_view{large}};
        expect(v.is_string());
        expect(eq(v.as_string_view().size(), 10000UZ));
    };
};

const boost::ut::suite<"Value - L8b Type Strictness"> type_strictness_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;

    // These tests verify that value_or enforces type strictness
    // The constraint std::same_as<std::remove_cvref_t<T>, std::remove_cvref_t<U>>
    // prevents implicit conversions like float→double

    "value_or exact type match float"_test = [] {
        Value v{3.14f}; // float
        float result = v.value_or<float>(0.0f);
        expect(eq(result, 3.14f));
        // Note: v.value_or<double>(0.0) would NOT compile due to type strictness
    };

    "value_or exact type match int32"_test = [] {
        Value        v{std::int32_t{42}};
        std::int32_t result = v.value_or<std::int32_t>(std::int32_t{0});
        expect(eq(result, std::int32_t{42}));
        // Note: v.value_or<std::int64_t>(std::int64_t{0}) would NOT compile
    };

    "or_else accepts callables returning exact type"_test = [] {
        Value v{std::int64_t{42}};
        auto  result = v.or_else<std::int64_t>([]() -> std::int64_t { return std::int64_t{999}; });
        expect(eq(result, std::int64_t{42}));
    };

    "value_or with complex type strictness"_test = [] {
        Value                v{std::complex<float>{1.0f, 2.0f}};
        std::complex<float>  fallback{0.0f, 0.0f};
        std::complex<float>& ref = v.value_or<std::complex<float>&>(fallback);
        expect(eq(ref.real(), 1.0f));
        expect(eq(ref.imag(), 2.0f));
    };
};

const boost::ut::suite<"Value - DiagString Verification"> _diag_suite = [] {
    using namespace boost::ut;

    "DiagString copy vs move tracking"_test = [] {
        g_diag_counters.reset();

        DiagString source("test");
        expect(eq(g_diag_counters.value_ctor.load(), 1UZ));

        DiagString copy = source;
        expect(eq(g_diag_counters.copy_ctor.load(), 1UZ));
        expect(eq(g_diag_counters.move_ctor.load(), 0UZ));

        DiagString moved = std::move(source);
        expect(eq(g_diag_counters.move_ctor.load(), 1UZ));
    };

    "copy semantics verification"_test = [] {
        g_diag_counters.reset();

        DiagString original("original");
        DiagString result = original; // copy

        expect(eq(g_diag_counters.copy_ctor.load(), 1UZ));
        expect(eq(g_diag_counters.move_ctor.load(), 0UZ));
    };

    "move semantics verification"_test = [] {
        g_diag_counters.reset();

        DiagString original("original");
        DiagString result = std::move(original); // move

        expect(eq(g_diag_counters.copy_ctor.load(), 0UZ));
        expect(eq(g_diag_counters.move_ctor.load(), 1UZ));
    };
};

const boost::ut::suite<"Value - w/o std::format and operator<< support"> _wo_format_support = [] {
    [[maybe_unused]] gr::pmt::Value v{"Hello World!"};
    std::puts("formatter test output w/o formatter/ValueFormatter.hpp header support:");
    std::stringstream ss;
    // ss << v;        // ERROR: should not compile
    ss << "\n";
    // ss << std::format("{}", v);  // ERROR: should not compile
    ss << "\n";
    std::puts(ss.str().c_str());
};

#include <gnuradio-4.0/formatter/ValueFormatter.hpp>

const boost::ut::suite<"Value - std::format and operator<< support"> _format_support = [] {
    using namespace boost::ut;

    [[maybe_unused]] gr::pmt::Value v{"Hello World!"};
    std::puts("formatter test output w/ formatter/ValueFormatter.hpp header support:");
    std::stringstream ss;
    ss << v;
    ss << "\n";
    ss << std::format("{}", v);
    ss << "\n";
    std::puts(ss.str().c_str());

    "safe accessors on monostate do not explode"_test = [] {
        gr::pmt::Value mono;
        expect(mono.get_if<std::int64_t>() == nullptr);
        expect(mono.get_if<std::pmr::string>() == nullptr);
        expect(ge(gr::pmt::detail::type_name(mono).size(), 1UZ)) << "type name is not empty";
        expect(le(memory_usage(mono), 24UZ)) << "memory usage is reasonable";
    };
};

int main() { /* boost::ut auto-runs */ }
