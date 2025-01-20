#include <boost/ut.hpp>
#include <complex>
#include <cstdint>
#include <fmt/format.h>
#include <string>
#include <variant>

#include <gnuradio-4.0/PmtTypeHelpers.hpp>

#include <gnuradio-4.0/meta/formatter.hpp>

namespace test {
enum class Colour { Red, Green, Blue };
}

using TestVariant = std::variant<bool, std::int8_t, std::int16_t, std::int32_t, std::int64_t, std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, //
    float, double, std::complex<float>, std::complex<double>, std::string, test::Colour>;

const boost::ut::suite<"pmt safe conversion tests"> _conversionTests = [] {
    using namespace boost::ut;
    using namespace std::string_literals;

    constexpr static auto unexpected = [](auto& exp) { return exp.has_value() ? fmt::format("unexpected value return: {}", exp.value()) : fmt::format("unexpected error return: {}", exp.error()); };

    "integral->integral (in-range)"_test = [] {
        TestVariant v   = 42;
        auto        res = pmtv::convert_safely<std::int64_t>(v);
        expect(res.has_value()) << unexpected(res);
        if (res) {
            expect(*res == 42_ll) << fmt::format("wrong value, got: {}", res.value());
        }
    };

    "integral->integral (out-of-range)"_test = [] {
        TestVariant v   = 1'000'000LL;
        auto        res = pmtv::convert_safely<std::int16_t>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "signed->unsigned (negative)"_test = [] {
        TestVariant v   = -1;
        auto        res = pmtv::convert_safely<std::uint32_t>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "integral->float (within mantissa)"_test = [] {
        // 2^24 = 16777216 is the approximate integer limit for exact float
        TestVariant v   = (1 << 24) - 1;
        auto        res = pmtv::convert_safely<float>(v);
        expect(res.has_value()) << unexpected(res);
        if (res) {
            expect(*res == static_cast<float>((1 << 24) - 1)) << fmt::format("wrong value, got: {}", *res);
        }
    };

    "integral->float (bit_width check, out-of-range for float mantissa)"_test = [] {
        // 2^24 = 16777216 is the approximate integer limit for exact float
        TestVariant v   = (1 << 24) + 1;
        auto        res = pmtv::convert_safely<float>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "float->integral (exact integer)"_test = [] {
        TestVariant v   = 42.0f;
        auto        res = pmtv::convert_safely<std::int32_t>(v);
        expect(res.has_value()) << unexpected(res);
        if (res) {
            expect(*res == 42);
        }
    };

    "float->integral (not exact integer)"_test = [] {
        TestVariant v   = 42.1f;
        auto        res = pmtv::convert_safely<std::int32_t>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "float->integral (out-of-range)"_test = [] {
        TestVariant v   = 3e9f; // 3,000,000,000 outside 32-bit int range
        auto        res = pmtv::convert_safely<std::int32_t>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "float->double (valid)"_test = [] {
        TestVariant v   = 1.2345f;
        auto        res = pmtv::convert_safely<double>(v);
        expect(res.has_value()) << unexpected(res);
    };

    "double->float (in-range)"_test = [] {
        TestVariant v   = 123.456;
        auto        res = pmtv::convert_safely<float>(v);
        expect(res.has_value()) << unexpected(res);
    };

    "double->float (out-of-range)"_test = [] {
        TestVariant v   = 1e40; // outside of float max ~3.4e38
        auto        res = pmtv::convert_safely<float>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "string->integral (valid)"_test = [] {
        TestVariant v   = "  123  # trailing comment"s;
        auto        res = pmtv::convert_safely<std::int32_t>(v);
        expect(res.has_value()) << unexpected(res);
        ;
        if (res) {
            expect(*res == 123) << fmt::format("wrong parsed value, got: {}", *res);
        }
    };

    "string->integral (invalid)"_test = [] {
        TestVariant v   = "12.3"s;
        auto        res = pmtv::convert_safely<std::int32_t>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "string->integral (out-of-range)"_test = [] {
        TestVariant v   = "999999999999"s; // too big
        auto        res = pmtv::convert_safely<std::int32_t>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "string->float (valid)"_test = [] {
        TestVariant v   = " 3.14159 #some comment"s;
        auto        res = pmtv::convert_safely<float>(v);
        expect(res.has_value()) << unexpected(res);
    };

    "string->float (invalid)"_test = [] {
        TestVariant v   = "not_a_float"s;
        auto        res = pmtv::convert_safely<double>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "string->float (out-of-range)"_test = [] {
        // Something bigger than double can hold
        TestVariant v   = "1e9999"s;
        auto        res = pmtv::convert_safely<double>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "string->bool (valid, true)"_test = [] {
        TestVariant v   = "TRUE"s;
        auto        res = pmtv::convert_safely<bool>(v);
        expect(res.has_value()) << unexpected(res);
        if (res) {
            expect(*res == true);
        }
    };

    "string->bool (valid, false)"_test = [] {
        TestVariant v   = "False"s;
        auto        res = pmtv::convert_safely<bool>(v);
        expect(res.has_value()) << unexpected(res);
        if (res) {
            expect(*res == false);
        }
    };

    "string->bool (invalid)"_test = [] {
        TestVariant v   = "notABool"s;
        auto        res = pmtv::convert_safely<bool>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "bool->string (valid, true)"_test = [] {
        TestVariant v   = true;
        auto        res = pmtv::convert_safely<std::string>(v);
        expect(res.has_value()) << unexpected(res);
        if (res) {
            expect(*res == "true"s);
        }
    };

    "bool->string (valid, false)"_test = [] {
        TestVariant v   = false;
        auto        res = pmtv::convert_safely<std::string>(v);
        expect(res.has_value()) << unexpected(res);
        if (res) {
            expect(*res == "false"s);
        }
    };

    "enum->string"_test = [] {
        TestVariant v   = test::Colour::Blue;
        auto        res = pmtv::convert_safely<std::string>(v);
        expect(res.has_value()) << unexpected(res);
        if (res) {
            expect(*res == "Blue"s);
        }
    };

    "enum->float (invalid)"_test = [] {
        TestVariant v   = test::Colour::Blue;
        auto        res = pmtv::convert_safely<float>(v);
        expect(!res.has_value()) << unexpected(res);
    };

    "string->enum (valid)"_test = [] {
        TestVariant v   = "Green"s;
        auto        res = pmtv::convert_safely<test::Colour>(v);
        expect(res.has_value()) << (res.has_value() ? "" : res.error());
        if (res) {
            expect(*res == test::Colour::Green);
        }
    };

    "string->enum (invalid)"_test = [] {
        TestVariant v   = "Magenta"s;
        auto        res = pmtv::convert_safely<test::Colour>(v);
        expect(!res.has_value()) << (res.has_value() ? "unexpected success, value is present" : fmt::format("Should fail invalid enum, got: {}", res.error()));
    };

    "complex<float> -> complex<double>"_test = [] {
        TestVariant v   = std::complex<float>(1.25f, -2.5f);
        auto        res = pmtv::convert_safely<std::complex<double>>(v);
        expect(res.has_value()) << (res.has_value() ? "" : "should succeed: complex<float> -> complex<double>");
        if (res) {
            expect(res->real() == 1.25) << "wrong real part";
            expect(res->imag() == -2.5) << "wrong imag part";
        }
    };

    "complex<double> -> complex<float>"_test = [] {
        TestVariant v   = std::complex<double>(3.32, -1.2345);
        auto        res = pmtv::convert_safely<std::complex<float>>(v);
        expect(res.has_value()) << (res.has_value() ? "" : "should succeed: complex<double> -> complex<float>");
        if (res) {
            expect(approx(res->real(), 3.32f, 1e-5f)) << "real part mismatch";
            expect(approx(res->imag(), -1.2345f, 1e-5f)) << "imag part mismatch";
        }
    };

    "integral -> complex<float>"_test = [] {
        TestVariant v   = std::int32_t{42};
        auto        res = pmtv::convert_safely<std::complex<float>>(v);
        expect(res.has_value()) << (res.has_value() ? "" : "should succeed: int -> complex<float>");
        if (res) {
            expect(res->real() == 42.0f) << "wrong real part";
            expect(res->imag() == 0.0f) << "imag should be 0";
        }
    };

    "double -> complex<double>"_test = [] {
        TestVariant v   = 123.456;
        auto        res = pmtv::convert_safely<std::complex<double>>(v);
        expect(res.has_value()) << (res.has_value() ? "" : "should succeed: double -> complex<double>");
        if (res) {
            expect(res->real() == 123.456) << "wrong real part";
            expect(res->imag() == 0.0) << "imag should be 0";
        }
    };

    "bool -> complex<float> (should fail)"_test = [] {
        TestVariant v   = true;
        auto        res = pmtv::convert_safely<std::complex<float>>(v);
        expect(!res.has_value()) << (res.has_value() ? "unexpected success, value is present" : fmt::format("should fail, got: {}", res.error()));
    };

    "complex<float> -> integral (not implemented)"_test = [] {
        TestVariant v   = std::complex<float>(1.0f, 2.0f);
        auto        res = pmtv::convert_safely<int>(v);
        expect(!res.has_value()) << (res.has_value() ? "unexpected success, value is present" : fmt::format("should fail, got: {}", res.error()));
    };

    "complex<double> (non-real-only) -> float"_test = [] {
        TestVariant v   = std::complex<double>(1.0, -3.32);
        auto        res = pmtv::convert_safely<float>(v);
        expect(!res.has_value()) << (res.has_value() ? "unexpected success, value is present" : fmt::format("should fail, got: {}", res.error()));
    };

    "complex<double> (real-only) -> floating-point"_test = []<typename T> {
        TestVariant v   = std::complex<T>(T(1), 0);
        auto        res = pmtv::convert_safely<float>(v);
        expect(res.has_value()) << unexpected(res);
        expect(eq(res.value(), 1.0f));
    } | std::tuple<double, float>{};

    "string -> complex<float> (not implemented)"_test = [] {
        TestVariant v   = "1.0, 2.0"s;
        auto        res = pmtv::convert_safely<std::complex<float>>(v);
        expect(!res.has_value()) << (res.has_value() ? "unexpected success, value is present" : fmt::format("should fail, got: {}", res.error()));
    };
};

const boost::ut::suite<"parse to minmal numeric type"> _parseToMinimalNumericType = [] {
    using namespace boost::ut;

    using MinimalNumericVariant = std::variant<std::int8_t, std::int16_t, std::int32_t, std::int64_t, std::uint8_t, float, double>;

    constexpr static auto variantToString = [](const MinimalNumericVariant& v) { return std::visit([](auto val) { return fmt::format("{} (type={})", val, typeid(val).name()); }, v); };

    "empty or comment only"_test = [] {
        "empty_string"_test = [] {
            auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("");
            expect(!res.has_value());
        };
        "comment only"_test = [] {
            auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("# comment");
            expect(!res.has_value());
        };
    };

    "integral fits int8_t"_test = [] {
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("127");
        expect(res.has_value()) << "should parse successfully";
        if (res) {
            // check the variant holds int8_t
            expect(std::holds_alternative<std::int8_t>(*res)) << fmt::format("wrong type => got {}", variantToString(*res));
            expect(std::get<std::int8_t>(*res) == 127) << "wrong value for int8_t";
        }
    };

    "integral fits uint8_t"_test = [] {
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("255");
        expect(res.has_value()) << "should parse successfully";
        if (res) {
            // because 255 is non-negative and fits in uint8_t
            expect(std::holds_alternative<std::uint8_t>(*res)) << fmt::format("wrong type => got {}", variantToString(*res));
            expect(std::get<std::uint8_t>(*res) == 255) << "wrong value for uint8_t";
        }
    };

    "negative integral no unsigned check"_test = [] {
        // -1 => should prefer int8_t if in range
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("-1");
        expect(res.has_value()) << "should parse successfully";
        if (res) {
            // with the smallest signed type that can hold -1 => int8_t
            expect(std::holds_alternative<std::int8_t>(*res)) << fmt::format("wrong type => got {}", variantToString(*res));
            expect(std::get<std::int8_t>(*res) == -1) << "wrong value for int8_t";
        }
    };

    "integral fits int16_t"_test = [] {
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("300");
        expect(res.has_value());
        if (res) {
            expect(std::holds_alternative<std::int16_t>(*res)) << fmt::format("wrong type => got {}", variantToString(*res));
            expect(std::get<std::int16_t>(*res) == 300);
        }
    };

    "integral fits int32_t"_test = [] {
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("40000");
        expect(res.has_value());
        if (res) {
            expect(std::holds_alternative<std::int32_t>(*res)) << fmt::format("wrong type => got {}", variantToString(*res));
            expect(std::get<std::int32_t>(*res) == 40000);
        }
    };

    "integral fits int64_t"_test = [] {
        // bigger than 2^31-1
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("9999999999");
        expect(res.has_value());
        if (res) {
            expect(std::holds_alternative<std::int64_t>(*res)) << fmt::format("wrong type => got {}", variantToString(*res));
            expect(std::get<std::int64_t>(*res) == 9999999999LL);
        }
    };

    "integral parse error"_test = [] {
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("42abc");
        expect(!res.has_value()) << "should fail parse => leftover 'abc'";
    };

    "integral out-of-range for 64-bit"_test = [] {
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("999999999999999999999999999");
        expect(!res.has_value()) << "should fail => out-of-range for 64-bit";
    };

    "float => float"_test = [] {
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("3.14159");
        expect(res.has_value());
        if (res) {
            expect(std::holds_alternative<float>(*res)) << fmt::format("wrong type => got {}", variantToString(*res));
        }
    };

    "float parse => double fallback"_test = [] {
        // ~3.4e38 is float max, let's try something bigger => fallback to double
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("9.999999e39");
        expect(res.has_value()) << "should parse as double fallback if out-of-range for float";
        if (res) {
            expect(std::holds_alternative<double>(*res)) << fmt::format("wrong type => got {}", variantToString(*res));
        }
    };

    "invalid float parse"_test = [] {
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("not_a_number");
        expect(!res.has_value()) << "should fail invalid float parse";
    };

    "double out-of-range"_test = [] {
        const auto res = pmtv::parseToMinimalNumeric<MinimalNumericVariant>("1e999999");
        expect(!res.has_value()) << "should fail => double parse out-of-range";
    };
};

int main() { /* tests are statically executed */ }
