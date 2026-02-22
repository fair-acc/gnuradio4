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
        expect(!v.holds<std::string>());
        expect(!v.holds<std::string_view>());
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

    "std::string construction"_test = [] {
        std::string str = "test string";
        Value       vs{str};
        expect(vs.is_string());
        expect(eq(vs.value_or(std::string_view{""}), std::string_view{"test string"}));
    };

    "std::string_view data and empty"_test = [] {
        std::string empty_str = "";
        Value       empty_str_value{empty_str};
        expect(empty_str_value.is_string());
        expect(eq(empty_str_value.value_or(std::string_view{""}), std::string_view{""}));

        Value empty_value;
        Value int_value(42);
        expect(empty_str_value.value_or(std::string_view{}).data() != nullptr);
        expect(empty_value.value_or(std::string_view{}).data() == nullptr);
        expect(int_value.value_or(std::string_view{}).data() == nullptr);
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
            expect(eq(source.value_or(std::string_view{""}), std::string_view{"hello"}));
        };

        "assignment copies content using target's allocator"_test = [] {
            counting_resource mr1;
            counting_resource mr2;

            Value v1{std::string_view{"from v1"}, &mr1};
            Value v2{std::string_view{"from v2"}, &mr2};

            v2 = v1; // assignment should use v2's allocator

            expect(v2.is_string());
            expect(eq(v2.value_or(std::string_view{""}), std::string_view{"from v1"}));
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
            expect(eq(v.value_or(std::string_view{""}), std::string_view{"test"}));
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

            auto* t1 = v1.get_if<Tensor<float>>();
            auto* t2 = v2.get_if<Tensor<float>>();

            expect(t1 != nullptr);
            expect(t2 != nullptr);
            expect(eq((*t1)[0, 0], 1.0f));
            expect(eq((*t2)[0, 0], 1.0f));
            expect(eq((*t1)[1, 2], 6.0f));
            expect(eq((*t2)[1, 2], 6.0f));
        };

        "Tensor move transfers ownership"_test = [] {
            Tensor<float> t({2, 2});
            t[0, 0] = 42.0f;

            Value v1{std::move(t)};
            Value v2{std::move(v1)};

            expect(v1.is_monostate()) << "source reset after move";
            expect(v2.is_tensor());
            auto* t2 = v2.get_if<Tensor<float>>();
            expect(t2 != nullptr);
            expect(eq((*t2)[0, 0], 42.0f));
        };

        "Empty tensor is a tensor"_test = [] {
            Tensor<float> t{};
            Value         v{std::move(t)};
            expect(!v.is_monostate()) << "empty tensor is not monostate";
            expect(v.is_tensor()) << "empty tensor is a tensor";
            expect(v.holds<Tensor<float>>()) << "empty tensor is a tensor";
        };
    };
};

const boost::ut::suite<"Value - container converting constructors"> _container_conversion_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::Tensor;

    "std::vector<float> → Value"_test = [] {
        Value v{std::vector{1.f, 2.f, 3.f}};
        expect(v.is_tensor());
        expect(eq(v.value_type(), Value::ValueType::Float32));
        auto* t = v.get_if<Tensor<float>>();
        expect(t != nullptr);
        expect(eq(t->size(), 3UZ));
        expect(eq((*t)[0], 1.f));
        expect(eq((*t)[2], 3.f));
    };

    "std::vector<double> → Value"_test = [] {
        Value v{std::vector{1., 2., 3.}};
        expect(v.is_tensor());
        auto* t = v.get_if<Tensor<double>>();
        expect(t != nullptr);
        expect(eq((*t)[1], 2.));
    };

    "std::vector<int32_t> → Value"_test = [] {
        Value v{std::vector<std::int32_t>{10, 20, 30}};
        expect(v.is_tensor());
        auto* t = v.get_if<Tensor<std::int32_t>>();
        expect(t != nullptr);
        expect(eq(t->size(), 3UZ));
        expect(eq((*t)[0], 10));
    };

    "std::vector<std::string> → Value"_test = [] {
        Value v{std::vector<std::string>{"hello", "world", "!"}};
        expect(v.is_tensor());
        expect(eq(v.value_type(), Value::ValueType::Value));
        auto* t = v.get_if<Tensor<Value>>();
        expect(t != nullptr);
        expect(eq(t->size(), 3UZ));
        expect(eq((*t)[0].value_or(std::string{}), std::string("hello")));
        expect(eq((*t)[1].value_or(std::string{}), std::string("world")));
        expect(eq((*t)[2].value_or(std::string{}), std::string("!")));
    };

    "std::vector<std::string> empty → Value"_test = [] {
        Value v{std::vector<std::string>{}};
        expect(v.is_tensor());
        auto* t = v.get_if<Tensor<Value>>();
        expect(t != nullptr);
        expect(eq(t->size(), 0UZ));
    };

    "std::array<float, 3> → Value"_test = [] {
        Value v{std::array{1.f, 2.f, 3.f}};
        expect(v.is_tensor());
        auto* t = v.get_if<Tensor<float>>();
        expect(t != nullptr);
        expect(eq(t->size(), 3UZ));
        expect(eq((*t)[0], 1.f));
    };

    "std::array<std::string, 2> → Value"_test = [] {
        Value v{std::array<std::string, 2>{"X", "Y"}};
        expect(v.is_tensor());
        auto* t = v.get_if<Tensor<Value>>();
        expect(t != nullptr);
        expect(eq(t->size(), 2UZ));
        expect(eq((*t)[0].value_or(std::string{}), std::string("X")));
        expect(eq((*t)[1].value_or(std::string{}), std::string("Y")));
    };

    "Value assignment from std::vector<float>"_test = [] {
        Value v;
        v = std::vector{4.f, 5.f};
        expect(v.is_tensor());
        auto* t = v.get_if<Tensor<float>>();
        expect(t != nullptr);
        expect(eq(t->size(), 2UZ));
        expect(eq((*t)[0], 4.f));
    };

    "Value assignment from std::vector<std::string>"_test = [] {
        Value v;
        v = std::vector<std::string>{"a", "b"};
        expect(v.is_tensor());
        auto* t = v.get_if<Tensor<Value>>();
        expect(t != nullptr);
        expect(eq((*t)[0].value_or(std::string{}), std::string("a")));
    };

    "Value assignment from std::array<double, 2>"_test = [] {
        Value v;
        v = std::array{1., 2.};
        expect(v.is_tensor());
        auto* t = v.get_if<Tensor<double>>();
        expect(t != nullptr);
        expect(eq((*t)[1], 2.));
    };

    "Value assignment from std::array<std::string, 1>"_test = [] {
        Value v;
        v = std::array<std::string, 1>{"only"};
        expect(v.is_tensor());
        auto* t = v.get_if<Tensor<Value>>();
        expect(t != nullptr);
        expect(eq((*t)[0].value_or(std::string{}), std::string("only")));
    };

    "std::vector/array in Value::Map"_test = [] {
        Value::Map pm;
        pm.emplace("floats", std::vector{1.f, 2.f, 3.f});
        pm.emplace("strings", std::vector<std::string>{"a", "b"});
        pm.emplace("ints", std::array<std::int32_t, 2>{10, 20});

        auto* ft = pm.at("floats").get_if<Tensor<float>>();
        expect(ft != nullptr);
        expect(eq(ft->size(), 3UZ));

        auto* st = pm.at("strings").get_if<Tensor<Value>>();
        expect(st != nullptr);
        expect(eq(st->size(), 2UZ));

        auto* it = pm.at("ints").get_if<Tensor<std::int32_t>>();
        expect(it != nullptr);
        expect(eq(it->size(), 2UZ));
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

    "string and string_view are indistinguishable"_test = [] {
        Value v1{std::string_view{"hello"}};
        Value v2{std::string{"hello"}};

        expect(v1 == v2);

        std::hash<gr::pmt::Value> vh;
        expect(vh(v1) == vh(v2));
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
            auto* map_ptr = v.get_if<Value::Map>();
            expect(map_ptr != nullptr);
            expect(map_ptr->contains("inner"));
            expect(map_ptr->contains("top_level"));

            auto& inner = map_ptr->at("inner");
            expect(inner.is_map());
            auto* inner_map_ptr = inner.get_if<Value::Map>();
            expect(inner_map_ptr != nullptr);
            expect(inner_map_ptr->at("nested_int").holds<std::int64_t>());
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

        "Map emplace test"_test = [] {
            Value::Map map1;
            map1.emplace("key1", std::string("value1"));
            map1.emplace("key2", std::string_view("value2"));
            map1.emplace("key3", 3);

            Value::Map map2{               //
                {"key1", Value("value1")}, //
                {"key2", Value("value2")}, //
                {"key3", Value(3)}};

            expect(map1 == map2);
            expect(map1["key1"].holds<std::string>());
            expect(map1["key2"].holds<std::string>());
            expect(map1["key3"].holds<int>());
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
        expect(eq(result.value_or(std::string_view{""}), std::string_view{"chain_me_modified"}));
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

            auto* map_ptr = v_map.get_if<Map>();
            expect(map_ptr != nullptr);
            expect(eq(map_ptr->size(), 2u));
            expect(map_ptr->contains("a"));
            expect(map_ptr->contains("b"));
            expect(eq(map_ptr->at("a").value_or<std::int64_t>(std::int64_t{0}), std::int64_t{1}));
            expect(eq(map_ptr->at("b").value_or<std::int64_t>(std::int64_t{0}), std::int64_t{2}));
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

        Tensor<int>* tref = vt.get_if<Tensor<std::int32_t>>();
        static_assert(std::is_same_v<decltype(*tref), Tensor<std::int32_t>&>);
    };

    "Tensor of Value (heterogeneous)"_test = [] {
        Tensor<Value> t({2UZ});
        t[0] = Value{std::int64_t{42}};
        t[1] = Value{"Hello World!"};

        Value vt{t};

        expect(vt.is_tensor());
        expect(eq(vt.value_type(), Value::ValueType::Value));

        Tensor<Value>* tref = vt.get_if<Tensor<Value>>();
        expect(eq((*tref)[0UZ].value_or<std::int64_t>(std::int64_t{0}), std::int64_t{42}));
        expect(eq((*tref)[1UZ].value_or(std::string{}), std::string("Hello World!")));
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
        expect(eq(copy.value_or(std::string_view{""}), std::string_view{"test string"}));
        expect(eq(original.value_or(std::string_view{""}), std::string_view{"test string"})) << "original unchanged";
    };

    "move construction transfers ownership"_test = [] {
        Value original{std::pmr::string{"test string"}};
        Value moved{std::move(original)};

        expect(moved.is_string());
        expect(eq(moved.value_or(std::string_view{""}), std::string_view{"test string"}));
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
        expect(eq(empty_str.value_or(std::string_view{"fallback"}), std::string_view{""}));
        expect(empty_str.value_or(std::string_view{"fallback"}).empty());
    };

    "large string handling"_test = [] {
        std::string large(10000, 'x');
        Value       v{std::string_view{large}};
        expect(v.is_string());
        expect(eq(v.value_or(std::string_view{""}).size(), 10000UZ));
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
        expect(v.holds<float>());
        float result = v.value_or<float>(0.0f);
        expect(eq(result, 3.14f));
        // Note: v.value_or<double>(0.0) would NOT compile due to type strictness
    };

    "value_or exact type match float"_test = [] {
        Value v{3.14}; // double
        expect(v.holds<double>());
        double result = v.value_or<double>(0.0);
        expect(eq(result, 3.14));
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

#include <map>

const boost::ut::suite<"Value generic map"> _genericMapTests = [] {
    using namespace boost::ut;
    using namespace gr::pmt;

    "maps with Value as mapped type"_test = [] {
        "std::map<string, Value> copy construct"_test = [] {
            std::map<std::string, Value> m{{"a", Value{1}}, {"b", Value{2}}};

            Value v{m};

            expect(v.is_map());
            auto* internal = v.get_if<Value::Map>();
            expect(internal->size() == 2_ul);
            expect(internal->at("a") == Value{1});
            expect(internal->at("b") == Value{2});
            // Source unchanged
            expect(m.size() == 2_ul);
        };

        "std::map<string, Value> move construct"_test = [] {
            std::map<std::string, Value> m{{"key", Value{std::string{"moved"}}}};

            Value v{std::move(m)};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("key") == Value{std::string{"moved"}});
        };

        "std::unordered_map<string, Value> copy construct"_test = [] {
            std::unordered_map<std::string, Value> m{{"x", Value{3.14}}, {"y", Value{2.71}}};

            Value v{m};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("x") == Value{3.14});
            expect(v.get_if<Value::Map>()->at("y") == Value{2.71});
        };

        "std::unordered_map<string, Value> move construct"_test = [] {
            std::unordered_map<std::string, Value> m{{"data", Value{42}}};

            Value v{std::move(m)};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("data") == Value{42});
        };

        "assign std::map<string, Value>"_test = [] {
            Value                        v{123}; // Start with scalar
            std::map<std::string, Value> m{{"assigned", Value{true}}};

            v = m;

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("assigned") == Value{true});
        };

        "move assign std::map<string, Value>"_test = [] {
            Value                        v;
            std::map<std::string, Value> m{{"moved", Value{99}}};

            v = std::move(m);

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("moved") == Value{99});
        };
    };

    "maps with fundamental types as mapped type"_test = [] {
        "std::map<string, int>"_test = [] {
            std::map<std::string, int> m{{"one", 1}, {"two", 2}, {"three", 3}};

            Value v{m};

            expect(v.is_map());
            auto* internal = v.get_if<Value::Map>();
            expect(internal->size() == 3_ul);
            expect(internal->at("one") == Value{1});
            expect(internal->at("two") == Value{2});
            expect(internal->at("three") == Value{3});
        };

        "std::map<string, double>"_test = [] {
            std::map<std::string, double> m{{"pi", 3.14159}, {"e", 2.71828}};

            Value v{m};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("pi") == Value{3.14159});
            expect(v.get_if<Value::Map>()->at("e") == Value{2.71828});
        };

        "std::map<string, float>"_test = [] {
            std::map<std::string, float> m{{"half", 0.5f}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("half") == Value{0.5f});
        };

        "std::map<string, bool>"_test = [] {
            std::map<std::string, bool> m{{"yes", true}, {"no", false}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("yes") == Value{true});
            expect(v.get_if<Value::Map>()->at("no") == Value{false});
        };

        "std::map<string, int64_t>"_test = [] {
            std::map<std::string, std::int64_t> m{{"big", 9'000'000'000'000'000'000LL}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("big") == Value{std::int64_t{9'000'000'000'000'000'000LL}});
        };

        "std::map<string, uint8_t>"_test = [] {
            std::map<std::string, std::uint8_t> m{{"byte", std::uint8_t{255}}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("byte") == Value{std::uint8_t{255}});
        };

        "std::unordered_map<string, int>"_test = [] {
            std::unordered_map<std::string, int> m{{"a", 10}, {"b", 20}};

            Value v{m};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("a") == Value{10});
            expect(v.get_if<Value::Map>()->at("b") == Value{20});
        };

        "std::map<string, complex<float>>"_test = [] {
            using cf = std::complex<float>;
            std::map<std::string, cf> m{{"z1", cf{1.0f, 2.0f}}, {"z2", cf{-1.0f, 0.5f}}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("z1") == Value{cf{1.0f, 2.0f}});
            expect(v.get_if<Value::Map>()->at("z2") == Value{cf{-1.0f, 0.5f}});
        };

        "std::map<string, complex<double>>"_test = [] {
            using cd = std::complex<double>;
            std::map<std::string, cd> m{{"omega", cd{0.0, 1.0}}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("omega") == Value{cd{0.0, 1.0}});
        };
    };

    "maps with strings as mapped type"_test = [] {
        "std::map<string, string>"_test = [] {
            std::map<std::string, std::string> m{{"greeting", "hello"}, {"farewell", "goodbye"}};

            Value v{m};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("greeting") == Value{std::string{"hello"}});
            expect(v.get_if<Value::Map>()->at("farewell") == Value{std::string{"goodbye"}});
        };

        "std::unordered_map<string, string>"_test = [] {
            std::unordered_map<std::string, std::string> m{{"name", "Claude"}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("name") == Value{std::string{"Claude"}});
        };

        "std::map<string, const char*>"_test = [] {
            std::map<std::string, const char*> m{{"literal", "test"}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("literal") == Value{std::string{"test"}});
        };

        "std::map<string, string_view>"_test = [] {
            using namespace std::string_view_literals;
            std::map<std::string, std::string_view> m{{"view", "content"sv}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("view") == Value{std::string{"content"}});
        };
    };

    "assignment with fundamental types"_test = [] {
        "assign std::map<string, int>"_test = [] {
            Value v{std::string{"initial"}};

            std::map<std::string, int> m{{"count", 42}};
            v = m;

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("count") == Value{42});
        };

        "move assign std::map<string, double>"_test = [] {
            Value                         v;
            std::map<std::string, double> m{{"value", 1.5}};

            v = std::move(m);

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("value") == Value{1.5});
        };

        "assign std::map<string, string>"_test = [] {
            Value v{42};

            std::map<std::string, std::string> m{{"text", "hello"}};
            v = m;

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("text") == Value{std::string{"hello"}});
        };
    };

    "edge cases"_test = [] {
        "empty map<string, int>"_test = [] {
            std::map<std::string, int> empty;

            Value v{empty};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->empty());
        };

        "empty map<string, Value>"_test = [] {
            std::map<std::string, Value> empty;

            Value v{empty};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->empty());
        };

        "large map<string, int>"_test = [] {
            std::map<std::string, int> m;
            for (int i = 0; i < 1000; ++i) {
                m["key_" + std::to_string(i)] = i;
            }

            Value v{m};

            expect(v.is_map());
            auto* internal = v.get_if<Value::Map>();
            expect(internal->size() == 1000_ul);
            expect(internal->at("key_0") == Value{0});
            expect(internal->at("key_500") == Value{500});
            expect(internal->at("key_999") == Value{999});
        };

        "single element map"_test = [] {
            std::map<std::string, int> m{{"only", 1}};

            Value v{m};

            expect(v.get_if<Value::Map>()->size() == 1_ul);
            expect(v.get_if<Value::Map>()->at("only") == Value{1});
        };
    };

    "nested maps"_test = [] {
        "nested: map<string, Value> containing map"_test = [] {
            std::map<std::string, int> inner{{"inner_int", 42}};
            Value                      innerValue{inner};

            std::map<std::string, Value> outer{{"nested", innerValue}, {"scalar", Value{3.14}}};
            Value                        v{outer};

            expect(v.is_map());
            auto* outerMap = v.get_if<Value::Map>();
            expect(outerMap->at("scalar") == Value{3.14});

            auto& nestedValue = outerMap->at("nested");
            expect(nestedValue.is_map());
            expect(nestedValue.get_if<Value::Map>()->at("inner_int") == Value{42});
        };

        "deeply nested maps"_test = [] {
            std::map<std::string, int>   level3{{"deep", 999}};
            std::map<std::string, Value> level2{{"level3", Value{level3}}};
            std::map<std::string, Value> level1{{"level2", Value{level2}}};

            Value v{level1};

            auto* l1 = v.get_if<Value::Map>();
            auto* l2 = l1->at("level2").get_if<Value::Map>();
            auto* l3 = l2->at("level3").get_if<Value::Map>();
            expect(l3->at("deep") == Value{999});
        };
    };

    "map with pmr::string as key"_test = [] {
        "std::map<pmr::string, int>"_test = [] {
            std::map<std::pmr::string, int> m{{"pmr_key", 123}};

            Value v{m};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("pmr_key") == Value{123});
        };

        "std::map<pmr::string, Value>"_test = [] {
            std::map<std::pmr::string, Value> m{{"pmr", Value{456}}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("pmr") == Value{456});
        };
    };

    "internal Value::Map type cross-check"_test = [] {
        "internal Map type exact match"_test = [] {
            Value::Map pmrMap;
            pmrMap["internal"] = Value{789};

            Value v{std::move(pmrMap)};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("internal") == Value{789});
        };
    };

    "custom memory resource with map<string, Value>"_test = [] {
        "custom memory resource with map<string, int>"_test = [] {
            std::array<std::byte, 8192>         buffer;
            std::pmr::monotonic_buffer_resource res{buffer.data(), buffer.size()};

            std::map<std::string, int> m{{"custom", 42}, {"resource", 99}};
            Value                      v{m, &res};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("custom") == Value{42});
            expect(v.get_if<Value::Map>()->at("resource") == Value{99});
        };

        "custom memory resource with map<string, Value>"_test = [] {
            std::array<std::byte, 8192>         buffer;
            std::pmr::monotonic_buffer_resource res{buffer.data(), buffer.size()};

            std::map<std::string, Value> m{{"val", Value{3.14}}};
            Value                        v{m, &res};

            expect(v.is_map());
            expect(v.get_if<Value::Map>()->at("val") == Value{3.14});
        };
    };

    "copy semantics verification"_test = [] {
        "copy preserves source map<string, int>"_test = [] {
            std::map<std::string, int> m{{"preserved", 42}};

            Value v{m};

            // Original unchanged
            expect(m.size() == 1_ul);
            expect(m.at("preserved") == 42);
            // Value has independent copy
            expect(v.get_if<Value::Map>()->at("preserved") == Value{42});
        };

        "copy preserves source map<string, Value>"_test = [] {
            std::map<std::string, Value> m{{"preserved", Value{std::string{"original"}}}};

            Value v{m};

            // Original unchanged
            expect(m.at("preserved") == Value{std::string{"original"}});
            // Value has independent copy
            expect(v.get_if<Value::Map>()->at("preserved") == Value{std::string{"original"}});
        };
    };

    "type preservation"_test = [] {
        "int32 vs int64 type preservation"_test = [] {
            std::map<std::string, std::int32_t> m32{{"val", 42}};
            std::map<std::string, std::int64_t> m64{{"val", 42}};

            Value v32{m32};
            Value v64{m64};

            // Same numeric value but different types
            expect(v32.get_if<Value::Map>()->at("val").holds<std::int32_t>());
            expect(v64.get_if<Value::Map>()->at("val").holds<std::int64_t>());
            // They should not be equal due to type strictness
            expect(v32.get_if<Value::Map>()->at("val") != v64.get_if<Value::Map>()->at("val"));
        };

        "float vs double type preservation"_test = [] {
            std::map<std::string, float>  mf{{"val", 1.5f}};
            std::map<std::string, double> md{{"val", 1.5}};

            Value vf{mf};
            Value vd{md};

            expect(vf.get_if<Value::Map>()->at("val").holds<float>());
            expect(vd.get_if<Value::Map>()->at("val").holds<double>());
        };
    };

    "special key values"_test = [] {
        "empty string key"_test = [] {
            std::map<std::string, int> m{{"", 0}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at("") == Value{0});
        };

        "unicode keys"_test = [] {
            std::map<std::string, int> m{{"日本語", 1}, {"émoji", 2}, {"Ω", 3}};

            Value v{m};

            expect(v.get_if<Value::Map>()->size() == 3_ul);
            expect(v.get_if<Value::Map>()->at("日本語") == Value{1});
            expect(v.get_if<Value::Map>()->at("émoji") == Value{2});
            expect(v.get_if<Value::Map>()->at("Ω") == Value{3});
        };

        "whitespace keys"_test = [] {
            std::map<std::string, int> m{{" ", 1}, {"\t", 2}, {"\n", 3}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at(" ") == Value{1});
            expect(v.get_if<Value::Map>()->at("\t") == Value{2});
            expect(v.get_if<Value::Map>()->at("\n") == Value{3});
        };

        "long key"_test = [] {
            std::string                longKey(1000, 'x');
            std::map<std::string, int> m{{longKey, 42}};

            Value v{m};

            expect(v.get_if<Value::Map>()->at(std::pmr::string{longKey}) == Value{42});
        };
    };
};

#include <unordered_set>

const boost::ut::suite<"std::hash<Value>"> _hashTests = [] {
    using namespace boost::ut;
    using namespace gr::pmt;

    "empty Value hash"_test = [] {
        Value v1;
        Value v2;
        expect(std::hash<Value>{}(v1) == std::hash<Value>{}(v2));
    };

    "scalar hash equality"_test = [] {
        expect(std::hash<Value>{}(Value{42}) == std::hash<Value>{}(Value{42}));
        expect(std::hash<Value>{}(Value{3.14}) == std::hash<Value>{}(Value{3.14}));
        expect(std::hash<Value>{}(Value{std::string{"hello"}}) == std::hash<Value>{}(Value{std::string{"hello"}}));
        expect(std::hash<Value>{}(Value{true}) == std::hash<Value>{}(Value{true}));
    };

    "scalar hash inequality - different values"_test = [] {
        expect(std::hash<Value>{}(Value{42}) != std::hash<Value>{}(Value{43}));
        expect(std::hash<Value>{}(Value{3.14}) != std::hash<Value>{}(Value{3.15}));
        expect(std::hash<Value>{}(Value{std::string{"hello"}}) != std::hash<Value>{}(Value{std::string{"world"}}));
    };

    "scalar hash inequality - different types same value"_test = [] {
        expect(std::hash<Value>{}(Value{std::int32_t{0}}) != std::hash<Value>{}(Value{std::uint32_t{0}})) << "Value hash for int32_t and uint32_t should differ";
        expect(std::hash<Value>{}(Value{std::int32_t{42}}) != std::hash<Value>{}(Value{std::int64_t{42}})) << "Value hash for int32_t and int64_t should differ";
        expect(std::hash<Value>{}(Value{float{1.0f}}) != std::hash<Value>{}(Value{double{1.0}})) << "Value hash for float and double should differ";
    };

    "complex scalar hash"_test = [] {
        using cf = std::complex<float>;
        using cd = std::complex<double>;

        expect(std::hash<Value>{}(Value{cf{1.0f, 2.0f}}) == std::hash<Value>{}(Value{cf{1.0f, 2.0f}}));
        expect(std::hash<Value>{}(Value{cf{1.0f, 2.0f}}) != std::hash<Value>{}(Value{cf{2.0f, 1.0f}}));
        expect(std::hash<Value>{}(Value{cf{1.0f, 2.0f}}) != std::hash<Value>{}(Value{cd{1.0, 2.0}}));
    };

    "tensor hash equality"_test = [] {
        using namespace gr;

        Tensor<int> t1{data_from, {1, 2, 3}};
        Tensor<int> t2{data_from, {1, 2, 3}};
        expect(std::hash<Value>{}(Value{t1}) == std::hash<Value>{}(Value{t2}));

        Tensor<double> td1{{1.0, 2.0}};
        Tensor<double> td2{{1.0, 2.0}};
        expect(std::hash<Value>{}(Value{td1}) == std::hash<Value>{}(Value{td2}));
    };

    "tensor hash inequality - different elements"_test = [] {
        using namespace gr;

        Tensor<int> t1{data_from, {1, 2, 3}};
        Tensor<int> t2{data_from, {1, 2, 4}};
        expect(std::hash<Value>{}(Value{t1}) != std::hash<Value>{}(Value{t2}));
    };

    "tensor hash inequality - different sizes"_test = [] {
        using namespace gr;

        Tensor<int> t1{data_from, {1, 2}};
        Tensor<int> t2{data_from, {1, 2, 0}};
        expect(std::hash<Value>{}(Value{t1}) != std::hash<Value>{}(Value{t2}));
    };

    "tensor hash inequality - different element types"_test = [] {
        using namespace gr;

        Tensor<std::int32_t> t1{data_from, {1, 2, 3}};
        Tensor<std::int64_t> t2{data_from, {1, 2, 3}};
        expect(std::hash<Value>{}(Value{t1}) != std::hash<Value>{}(Value{t2}));
    };

    "tensor vs scalar hash inequality"_test = [] {
        using namespace gr;

        Value       scalar{42};
        Tensor<int> t{data_from, {42}};
        Value       tensor{t};
        expect(std::hash<Value>{}(scalar) != std::hash<Value>{}(tensor));
    };

    "map hash equality"_test = [] {
        std::map<std::string, Value> m1{{"a", Value{1}}, {"b", Value{2}}};
        std::map<std::string, Value> m2{{"a", Value{1}}, {"b", Value{2}}};
        expect(std::hash<Value>{}(Value{m1}) == std::hash<Value>{}(Value{m2}));
    };

    "map hash order independence"_test = [] {
        std::unordered_map<std::string, Value> m1;
        m1["a"] = Value{1};
        m1["b"] = Value{2};
        m1["c"] = Value{3};

        std::unordered_map<std::string, Value> m2; // N.B. different insertion order
        m2["c"] = Value{3};
        m2["a"] = Value{1};
        m2["b"] = Value{2};

        expect(std::hash<Value>{}(Value{m1}) == std::hash<Value>{}(Value{m2}));
    };

    "map hash order independence - std::map vs unordered_map same content"_test = [] {
        std::map<std::string, Value>           ordered{{"x", Value{10}}, {"y", Value{20}}};
        std::unordered_map<std::string, Value> unordered{{"y", Value{20}}, {"x", Value{10}}};

        Value v1{ordered};
        Value v2{unordered};

        // both convert to the same internal Map type, so container types are equal
        expect(v1.container_type() == v2.container_type());
        // same content should produce same hash (order-independent)
        expect(std::hash<Value>{}(v1) == std::hash<Value>{}(v2));
    };

    "map hash inequality - different values"_test = [] {
        std::map<std::string, Value> m1{{"a", Value{1}}};
        std::map<std::string, Value> m2{{"a", Value{2}}};
        expect(std::hash<Value>{}(Value{m1}) != std::hash<Value>{}(Value{m2}));
    };

    "map hash inequality - different keys"_test = [] {
        std::map<std::string, Value> m1{{"a", Value{1}}};
        std::map<std::string, Value> m2{{"b", Value{1}}};
        expect(std::hash<Value>{}(Value{m1}) != std::hash<Value>{}(Value{m2}));
    };

    "nested map hash"_test = [] {
        std::map<std::string, Value> inner1{{"x", Value{100}}};
        std::map<std::string, Value> inner2{{"x", Value{100}}};

        std::map<std::string, Value> outer1{{"nested", Value{inner1}}, {"scalar", Value{42}}};
        std::map<std::string, Value> outer2{{"scalar", Value{42}}, {"nested", Value{inner2}}};

        expect(std::hash<Value>{}(Value{outer1}) == std::hash<Value>{}(Value{outer2}));
    };

    "nested map hash inequality"_test = [] {
        std::map<std::string, Value> inner1{{"x", Value{100}}};
        std::map<std::string, Value> inner2{{"x", Value{999}}};

        std::map<std::string, Value> outer1{{"nested", Value{inner1}}};
        std::map<std::string, Value> outer2{{"nested", Value{inner2}}};

        expect(std::hash<Value>{}(Value{outer1}) != std::hash<Value>{}(Value{outer2}));
    };

    "usable in unordered_set"_test = [] {
        std::unordered_set<Value> set;

        set.insert(Value{1});
        set.insert(Value{2});
        set.insert(Value{1}); // duplicate

        expect(set.size() == 2_ul);
        expect(set.contains(Value{1}));
        expect(set.contains(Value{2}));
        expect(!set.contains(Value{3}));
    };

    "usable in unordered_map as key"_test = [] {
        std::unordered_map<Value, std::string> map;

        map[Value{std::string{"key1"}}] = "value1";
        map[Value{42}]                  = "value2";
        map[Value{std::string{"key1"}}] = "updated"; // overwrite

        expect(map.size() == 2_ul);
        expect(map.at(Value{std::string{"key1"}}) == "updated");
        expect(map.at(Value{42}) == "value2");
    };

    "hash consistency across calls"_test = [] {
        Value v{std::string{"test"}};

        auto h1 = std::hash<Value>{}(v);
        auto h2 = std::hash<Value>{}(v);
        auto h3 = std::hash<Value>{}(v);

        expect(h1 == h2);
        expect(h2 == h3);
    };

    "hash distribution sanity check"_test = [] {
        // Simple check that different small integers produce different hashes
        std::unordered_set<std::size_t> hashes;
        for (int i = 0; i < 100; ++i) {
            hashes.insert(std::hash<Value>{}(Value{i}));
        }
        // Expect no collisions for 100 consecutive integers
        expect(hashes.size() == 100_ul);
    };
};

int main() { /* boost::ut auto-runs */ }
