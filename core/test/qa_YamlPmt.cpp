#include <boost/ut.hpp>

#include <cstdint>
#include <limits>
#include <variant>

#include <gnuradio-4.0/meta/formatter.hpp>

#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/YamlPmt.hpp>

using namespace gr;
namespace yaml = gr::pmt::yaml;

auto fuzzy_eq(std::string_view str1, std::string_view str2) {
    const auto len = std::min(str1.size(), str2.size());
    return std::equal(str1.data(), str1.data() + len, str2.data());
}

std::string variantTypeName(const pmt::Value& v) {
    std::string result;
    pmt::ValueVisitor([&result](const auto& arg) {
        // Get the type name of the current alternative
        using T = std::decay_t<decltype(arg)>;
        result  = gr::meta::type_name<T>();
    }).visit(v);
    return result;
}

bool diff(const gr::property_map& original, const gr::property_map& deserialized);

void printDiff(std::string_view key, const gr::pmt::Value& originalValue, const gr::pmt::Value& deserializedValue) {
    std::ostringstream originalOss;
    std::ostringstream deserializedOss;
    yaml::detail::serialize(originalOss, originalValue);
    yaml::detail::serialize<>(deserializedOss, deserializedValue);

    std::println("Difference found at key: {}\n    expected:     {}\n    deserialized: {}", key, originalOss.str(), deserializedOss.str());

    std::println("Values are equal? {}\n    original:     {}\n    deserialized: {}", (originalValue == deserializedValue), originalValue, deserializedValue);
    std::println("Stringified are equal? {}", (std::format("{}", originalValue) == std::format("{}", deserializedValue)));
}

// Work around NaN != NaN when comparing floats/doubles
template<typename T>
bool testEqual(const T& lhs, const T& rhs) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(lhs) && std::isnan(rhs)) {
            return true;
        } else {
            return lhs == rhs;
        }

    } else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view> || std::is_same_v<T, std::pmr::string>) {
        return lhs == rhs;

    } else if constexpr (std::is_same_v<T, gr::property_map>) {
        return !diff(lhs, rhs);

    } else if constexpr (std::ranges::random_access_range<T>) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (size_t i = 0; i < lhs.size(); ++i) {
            if (!testEqual(lhs[i], rhs[i])) {
                // Values of tensors have a broken operator==
                if (std::format("{}", lhs) != std::format("{}", rhs)) {
                    return false;
                }
            }
        }
        return true;

    } else {
        return lhs == rhs;
    }
}

std::ostream& printAnyString(std::ostream& out, auto s) {
    for (char c : s) {
        if (std::isprint(c)) {
            out << static_cast<char>(c);
        } else {
            out << '<' << static_cast<int>(c) << '>';
        }
    }
    return out;
}

bool diff(const gr::property_map& original, const gr::property_map& deserialized) {
    bool foundDiff = false;
    for (const auto& [key, originalValue] : original) {
        auto it = deserialized.find(key);
        if (it == deserialized.end()) {
            std::cout << "Missing key in deserialized map: '";
            printAnyString(std::cout, key) << "'\n";
            foundDiff = true;
            continue;
        }
        const auto& deserializedValue = it->second;
        if (originalValue.value_type() != deserializedValue.value_type() || originalValue.container_type() != deserializedValue.container_type()) {
            std::cout << "Found different types for: ";
            printAnyString(std::cout, key) << "'\n";
            std::cout << "  Expected: " << variantTypeName(originalValue) << "\n";
            std::cout << "  Deserialized: " << variantTypeName(deserializedValue) << "\n";
            foundDiff = true;
        } else {
            bool equal = false;
            pmt::ValueVisitor([&equal, &deserializedValue](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    equal = testEqual(arg, deserializedValue.value_or(T{}));
                } else {
                    equal = deserializedValue.is_monostate();
                }
            }).visit(originalValue);
            if (!equal) {
                printDiff(key, originalValue, deserializedValue);
                foundDiff = true;
            }
        }
    }
    for (const auto& [key, deserializedValue] : deserialized) {
        if (original.find(key) == original.end()) {
            std::cout << "Extra key in deserialized map: '";
            printAnyString(std::cout, key) << "'\n";
            foundDiff = true;
        }
    }
    return foundDiff;
}

template<typename T, typename Error>
std::string formatResult(const std::expected<T, Error>& result) {
    if (!result.has_value()) {
        const auto& error = result.error();
        return std::format("Error in {}:{}: {}", error.line, error.column, error.message);
    } else {
        return "<no error>";
    }
}

void testYAML(std::string_view src, const gr::property_map expected, std::source_location location = std::source_location::current()) {
    using namespace boost::ut;
    // First test that the deserialized map matches the expected map
    const auto deserializedMap = yaml::deserialize(src);
    if (deserializedMap) {
        expect(eq(diff(expected, *deserializedMap), false), location) << std::format("testYAML unexpected error at: {}:{}\n\n", location.file_name(), location.line());
    } else {
        expect(false, location) << std::format("testYAML unexpected error at: {}:{}:\n{}\n\n", location.file_name(), location.line(), yaml::formatAsLines(src, deserializedMap.error()));
        return;
    }

    // Then test that serializing and deserializing the map again results in the same map
    const auto serializedStr    = yaml::serialize(expected);
    const auto deserializedMap2 = yaml::deserialize(serializedStr);
    if (deserializedMap2) {
        expect(eq(diff(expected, *deserializedMap2), false), location) << std::format("testYAML unexpected error at: {}:{} - string:\n{}\n\n", location.file_name(), location.line(), serializedStr);
    } else {
        expect(false, location) << std::format("testYAML unexpected error at: {}:{}:\n {}\nYAML:\n{}", location.file_name(), location.line(), formatResult(deserializedMap2), serializedStr);
    }
}

const boost::ut::suite<"GrepTests"> _GrepTests = [] {
    using namespace boost::ut;
    static auto grepTest = [](const gr::property_map& map, std::initializer_list<std::string_view> patterns, std::source_location location = std::source_location::current()) {
        const auto& serialized = pmt::yaml::serialize(map);
        for (const auto& pattern : patterns) {
            expect(serialized.find(pattern) != std::string::npos) << std::format("Pattern [{}] not found in [{}], called from {}:{}", pattern, serialized, location.file_name(), location.line());
            if (serialized.find(pattern) == std::string::npos) {
                std::terminate();
            }
        }
    };

    "Basic serialization grep tests for int"_test = [] {
        {
            gr::property_map map;
            map["answer"] = 42;
            grepTest(map, {"42", "answer"});
        }
        {
            gr::property_map map;
            map["answer"] = Tensor<int>(data_from, {42, 43, 44});
            grepTest(map, {"42", "43", "44", "answer"});
        }
    };
    "Basic serialization grep tests for string"_test = [] {
        {
            gr::property_map map;
            map["answer"] = "Hello"s;
            grepTest(map, {"Hello", "answer"});
        }
        {
            gr::property_map map;
            map["answer"] = Tensor<pmt::Value>(data_from, {pmt::Value("42"s), pmt::Value("43"s), pmt::Value("44"s)});
            grepTest(map, {"42", "43", "44", "answer"});
        }
    };
    "Basic serialization grep tests for a mix"_test = [] {
        {
            gr::property_map map;
            map["answer"]   = 42;
            map["question"] = "universe"s;
            grepTest(map, {"42", "universe", "answer", "question"});
        }
        {
            gr::property_map map;
            map["answer"] = Tensor<pmt::Value>(data_from, {pmt::Value(42), pmt::Value("question"s), pmt::Value("universe"s)});
            grepTest(map, {"42", "universe", "answer", "question"});
        }
    };
    "Basic serialization grep tests for nested maps"_test = [] {
        {
            gr::property_map nested;
            nested["answer"]   = 42;
            nested["question"] = "universe"s;

            property_map map;
            map["nested"] = std::move(nested);
            grepTest(map, {"42", "universe", "answer", "question", "nested"});
        }
        {
            gr::property_map nested;
            nested["answer"]   = 42;
            nested["question"] = "universe"s;

            gr::property_map middle;
            middle["nested"] = std::move(nested);

            property_map map;
            map["middle"] = std::move(middle);
            grepTest(map, {"42", "universe", "answer", "question", "nested", "middle"});
        }
    };

    "Basic serialization grep tests for tensors"_test = [] {
        {
            property_map map;
            map["answers"] = Tensor<int>(data_from, {42, 43, 44});
            map["names"]   = Tensor<pmt::Value>(data_from, {pmt::Value("John"s), pmt::Value("Smith"s)});
            grepTest(map, {"42", "43", "44", "John", "Smith"});
        }
        {
            gr::property_map nested;
            nested["answer"]   = 42;
            nested["question"] = "universe"s;

            property_map map;
            map["nested"] = std::move(nested);
            grepTest(map, {"42", "universe", "answer", "question", "nested"});
        }
        {
            gr::property_map nested;
            nested["answer"]   = 42;
            nested["question"] = "universe"s;

            gr::property_map middle;
            middle["nested"] = std::move(nested);

            property_map map;
            map["middle"] = std::move(middle);
            grepTest(map, {"42", "universe", "answer", "question", "nested", "middle"});
        }
    };
};

const boost::ut::suite<"YamlPmtTests"> _yamlPmtTests = [] {
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast" // we want explicit casts for testing
#endif
    using namespace boost::ut;
    using namespace std::string_literals;
    using namespace std::string_view_literals;

    "Comments and Whitespace"_test = [] {
        constexpr std::string_view src1 = R"(# Comment
double: !!float64 42 # Comment
stringQ1: "#Hello1" # Comment
stringQ2: "   #Hello2   "       # Comment
string1: Hello1 # Comment
string2:        Hello2        # Comment
null:  # Comment
#Comment: 43
# string: | # Comment
# Hello
number:
  42
list:
  []
list2: [ 42
]
map:
  {}
)";

        gr::property_map expected;
        expected["double"]   = 42.0;
        expected["stringQ1"] = "#Hello1";
        expected["stringQ2"] = "   #Hello2   ";
        expected["string1"]  = "Hello1";
        expected["string2"]  = "Hello2";
        expected["null"]     = gr::pmt::Value();
        expected["number"]   = gr::pmt::Value(static_cast<int64_t>(42));
        expected["list"]     = gr::Tensor<gr::pmt::Value>{};
        expected["list2"]    = gr::Tensor<gr::pmt::Value>{pmt::Value(static_cast<int64_t>(42))};
        expected["map"]      = gr::property_map{};

        testYAML(src1, expected);
    };

    "Strings"_test = [] {
        constexpr std::string_view src = R"yaml(
empty: !!str ""
spaces_only: !!str "   "
value_with_colon: "value: with colon"
value_with_colon2: "value:\n  with colon"
value_with_colon3: std::ranges
multiline1: !!str |
  First line
  Second line
  Third line with trailing newline
  Null bytes (\x00\0)
  This is a quoted backslash "\"
multiline2: !!str >
  This is a long
  paragraph that will
  be folded into
  a single line
  with trailing newlines
  This is a quoted backslash "\"
  These are some invalid escapes \q\xZZ\x

multiline3: !!str |-
  First line
  Second line
  Third line without trailing newline

multiline4: !!str >-
  This is a long
  paragraph that will
  be folded into
  a single line without
  trailing newline


multiline_listlike: !!str |-
  - First line
  - Second line
multiline_maplike: !!str |-
  key: First line
  key2: Second line
multiline_with_empty: !!str |
  First line

  Third line
unicode: !!str "Hello 世界 🌍"
escapes: !!str  "Quote\"Backslash\\Not a comment#Tab\tNewline\nBackspace\b"
invalid_escapes: !!str "Invalid escapes \q\xZZ\x"
single_quoted: !!str '"quoted"'
special_chars: !!str "!@#$%^&*()"
unprintable_chars: !!str "\0\x01\x02\x03\x04\x05\x00\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F"
)yaml";

        gr::property_map expected;
        expected["empty"]                = ""s;
        expected["spaces_only"]          = "   "s;
        expected["value_with_colon"]     = "value: with colon"s;
        expected["value_with_colon2"]    = "value:\n  with colon"s;
        expected["value_with_colon3"]    = "std::ranges"s;
        expected["multiline1"]           = "First line\nSecond line\nThird line with trailing newline\nNull bytes (\x00\x00)\nThis is a quoted backslash \"\\\"\n"s;
        expected["multiline2"]           = "This is a long paragraph that will be folded into a single line with trailing newlines This is a quoted backslash \"\\\" These are some invalid escapes \\q\\xZZ\\x\n\n"s;
        expected["multiline3"]           = "First line\nSecond line\nThird line without trailing newline"s;
        expected["multiline4"]           = "This is a long paragraph that will be folded into a single line without trailing newline"s;
        expected["multiline_listlike"]   = "- First line\n- Second line"s;
        expected["multiline_maplike"]    = "key: First line\nkey2: Second line"s;
        expected["multiline_with_empty"] = "First line\n\nThird line\n"s;
        expected["unicode"]              = "Hello 世界 🌍"s;
        expected["invalid_escapes"]      = "Invalid escapes \\q\\xZZ\\x"s;
        expected["escapes"]              = "Quote\"Backslash\\Not a comment#Tab\tNewline\nBackspace\b"s;
        expected["single_quoted"]        = "\"quoted\""s;
        expected["special_chars"]        = "!@#$%^&*()"s;
        expected["unprintable_chars"]    = "\x00\x01\x02\x03\x04\x05\x00\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F"s;

        testYAML(src, expected);

        constexpr std::string_view extraCharacters = R"(
value2: !!str "string"#comment
value: !!str "a string" extra
)";
        expect(eq(formatResult(yaml::deserialize(extraCharacters)), "Error in 3:25: Unexpected characters after scalar value"sv));

        constexpr std::string_view extraMultiline = R"(
value: |- extra
  Hello
  World
)";
        expect(eq(formatResult(yaml::deserialize(extraMultiline)), "Error in 2:11: Unexpected characters after multi-line indicator"sv));
    };

    "Nulls"_test = [] {
        constexpr std::string_view src = R"yaml(
null_value: !!null null
null_value2: null
null_value3: !!null ~
null_value4: ~
null_value5: !!null anything
null_value6: Null
null_value7: NULL
null_value8:
not_null: NuLl
null_at_end:
# Comment, then empty line

)yaml";

        gr::property_map expected;
        expected["null_value"]  = gr::pmt::Value();
        expected["null_value2"] = gr::pmt::Value();
        expected["null_value3"] = gr::pmt::Value();
        expected["null_value4"] = gr::pmt::Value();
        expected["null_value5"] = gr::pmt::Value();
        expected["null_value6"] = gr::pmt::Value();
        expected["null_value7"] = gr::pmt::Value();
        expected["null_value8"] = gr::pmt::Value();
        expected["not_null"]    = "NuLl";
        expected["null_at_end"] = gr::pmt::Value();
        testYAML(src, expected);
    };

    "Bools"_test = [] {
        constexpr std::string_view src = R"yaml(
true: !!bool true
false: !!bool false
untagged_true: true
untagged_false: false
untagged_true2: True
untagged_false2: False
untagged_true3: TRUE
untagged_false3: FALSE
)yaml";

        gr::property_map expected;
        expected["true"]            = true;
        expected["false"]           = false;
        expected["untagged_true"]   = true;
        expected["untagged_false"]  = false;
        expected["untagged_true2"]  = true;
        expected["untagged_false2"] = false;
        expected["untagged_true3"]  = true;
        expected["untagged_false3"] = false;

        testYAML(src, expected);

        expect(eq(formatResult(yaml::deserialize("bool: !!bool 1")), "Error in 1:14: Invalid value for bool-type"sv));
        expect(eq(formatResult(yaml::deserialize("bool: !!bool TrUe")), "Error in 1:14: Invalid value for bool-type"sv));
        expect(eq(formatResult(yaml::deserialize("bool: !!bool 1")), "Error in 1:14: Invalid value for bool-type"sv));
        expect(eq(formatResult(yaml::deserialize("bool: !!bool FaLsE")), "Error in 1:14: Invalid value for bool-type"sv));
    };

    "Numbers"_test = [] {
        constexpr std::string_view src = R"yaml(
integers:
  hex: !!int64 0xFF
  oct: !!int64 0o77
  bin: !!int64 0b1010
  positive: !!int64 42
  negative: !!int64 -42
  zero: !!int64 0
  uint8: !!uint8 255
  uint16: !!uint16 65535
  uint32: !!uint32 4294967295
  uint64: !!uint64 18446744073709551615
  int8: !!int8 -128
  int16: !!int16 -32768
  int32: !!int32 -2147483648
  int64: !!int64 -9223372036854775808
  untagged: 42
  untagged_hex: 0xFF
  untagged_oct: 0o77
  untagged_bin: 0b1010
doubles:
  normal: !!float64 123.456
  scientific: !!float64 1.23e-4
  infinity: !!float64 .inf
  infinity2: !!float64 .Inf
  infinity3: !!float64 .INF
  neg_infinity: !!float64 -.inf
  neg_infinity2: !!float64 -.Inf
  neg_infinity3: !!float64 -.INF
  not_a_number: !!float64 .nan
  not_a_number2: !!float64 .NaN
  not_a_number3: !!float64 .NAN
  negative_zero: !!float64 -0.0
  untagged: 123.456
  untagged_scientific: 1.23e-4
  untagged_infinity: .inf
  untagged_infinity2: .Inf
  untagged_infinity3: .INF
  untagged_neg_infinity: -.inf
  untagged_neg_infinity2: -.Inf
  untagged_neg_infinity3: -.INF
  untagged_not_a_number: .nan
  untagged_not_a_number2: .NaN
  untagged_not_a_number3: .NAN
  untagged_negative_zero: -0.0
)yaml";

        gr::property_map expected;

        gr::property_map integers;
        integers["hex"]          = static_cast<int64_t>(255);
        integers["oct"]          = static_cast<int64_t>(63);
        integers["bin"]          = static_cast<int64_t>(10);
        integers["positive"]     = static_cast<int64_t>(42);
        integers["negative"]     = static_cast<int64_t>(-42);
        integers["zero"]         = static_cast<int64_t>(0);
        integers["uint8"]        = std::numeric_limits<uint8_t>::max();
        integers["uint16"]       = std::numeric_limits<uint16_t>::max();
        integers["uint32"]       = std::numeric_limits<uint32_t>::max();
        integers["uint64"]       = std::numeric_limits<uint64_t>::max();
        integers["int8"]         = std::numeric_limits<int8_t>::min();
        integers["int16"]        = std::numeric_limits<int16_t>::min();
        integers["int32"]        = std::numeric_limits<int32_t>::min();
        integers["int64"]        = std::numeric_limits<int64_t>::min();
        integers["untagged"]     = static_cast<int64_t>(42);
        integers["untagged_hex"] = static_cast<int64_t>(255);
        integers["untagged_oct"] = static_cast<int64_t>(63);
        integers["untagged_bin"] = static_cast<int64_t>(10);

        gr::property_map doubles;
        doubles["normal"]                 = 123.456;
        doubles["scientific"]             = 1.23e-4;
        doubles["infinity"]               = std::numeric_limits<double>::infinity();
        doubles["infinity2"]              = std::numeric_limits<double>::infinity();
        doubles["infinity3"]              = std::numeric_limits<double>::infinity();
        doubles["neg_infinity"]           = -std::numeric_limits<double>::infinity();
        doubles["neg_infinity2"]          = -std::numeric_limits<double>::infinity();
        doubles["neg_infinity3"]          = -std::numeric_limits<double>::infinity();
        doubles["not_a_number"]           = std::numeric_limits<double>::quiet_NaN();
        doubles["not_a_number2"]          = std::numeric_limits<double>::quiet_NaN();
        doubles["not_a_number3"]          = std::numeric_limits<double>::quiet_NaN();
        doubles["negative_zero"]          = -0.0;
        doubles["untagged"]               = 123.456;
        doubles["untagged_scientific"]    = 1.23e-4;
        doubles["untagged_infinity"]      = std::numeric_limits<double>::infinity();
        doubles["untagged_infinity2"]     = std::numeric_limits<double>::infinity();
        doubles["untagged_infinity3"]     = std::numeric_limits<double>::infinity();
        doubles["untagged_neg_infinity"]  = -std::numeric_limits<double>::infinity();
        doubles["untagged_neg_infinity2"] = -std::numeric_limits<double>::infinity();
        doubles["untagged_neg_infinity3"] = -std::numeric_limits<double>::infinity();
        doubles["untagged_not_a_number"]  = std::numeric_limits<double>::quiet_NaN();
        doubles["untagged_not_a_number2"] = std::numeric_limits<double>::quiet_NaN();
        doubles["untagged_not_a_number3"] = std::numeric_limits<double>::quiet_NaN();
        doubles["untagged_negative_zero"] = -0.0;

        expected["integers"] = integers;
        expected["doubles"]  = doubles;

        testYAML(src, expected);

        expect(fuzzy_eq(formatResult(yaml::deserialize("value: !!float64 string")), "Error in 1:18: std::invalid_argument exception for expected floating-point value of 'string' - error: stod"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("value: !!int64 0xGG")), "Error in 1:16: Invalid integral-type value 'GG' (error: Invalid argument)"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("value: !!int64 0o99")), "Error in 1:16: Invalid integral-type value '99' (error: Invalid argument)"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("value: !!int64 0b1234")), "Error in 1:16: Invalid integral-type value"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("value: !!int8 128")), "Error in 1:15: Invalid integral-type value"sv));
    };

    "Vectors"_test = [] {
        constexpr std::string_view src1 = R"(
stringVector: !!str
  - "Hello"
  - "World"
  - |-
    Multiple
    lines
boolVector: !!bool
  - true
  - false
  - true
pmtVectorWithBools:
# Comment

  - !!bool true
  - !!bool false
  - !!bool true
mixedPmtVector:
  - !!bool true
  - !!float64 42
  - !!str "Hello"
pmtVectorWithUntaggedBools:
  - true
  - false
  - true
floatVector: !!float32
  - 1.0
  - 2.0
  - 3.0
doubleVector: !!float64
  - 1.0
  - 2.0
  - 3.0
complexVector: !!complex64
  - (1.0, -1.0)
  - (2.0, -2.0)
  - (3.0, -3.0)
nullVector: !!null
  - null
  - null
  - null
emptyVector: !!str []
emptyPmtVector: []
emptyAfterNewline:
  []
flowDouble: !!float64 [1, 2, 3]
flowString1: !!str ["Hello1, ", "World1", "Multiple1\nlines1"]
flowString2: !!str [Hello2, World2, Single2]
flowString3: !!str [   Hello3   ,   World3  ,     Single3  ]
flowMultiline: !!str [ "Hello, "    , #]
  "][", # Comment ,
  "World"  ,
  "Multiple\nlines"
]
nestedVector:
  - !!str
    - 1
    - 2
  -
    - 3
    - 4
nestedVector2:
    - !!int64 42
    - !!str [1, 2]
    - !!str [3, 4]
    - { key: !!str [5, 6] }
nestedFlow: [ !!str [1, 2], [3, 4] ]
vectorWithBlockMap:
  - name: ArraySink
    id: gr::testing::ArraySink<double>
    parameters:
      name: Block
vectorWithColons:
  - "key: value"
  - "key2: value2"
)";

        gr::property_map expected;
        expected["boolVector"]                 = Tensor<bool>(gr::data_from, {true, false, true});
        expected["pmtVectorWithBools"]         = Tensor<pmt::Value>{pmt::Value(true), pmt::Value(false), pmt::Value(true)};
        expected["pmtVectorWithUntaggedBools"] = Tensor<pmt::Value>{pmt::Value(true), pmt::Value(false), pmt::Value(true)};
        expected["mixedPmtVector"]             = Tensor<pmt::Value>{gr::pmt::Value(true), gr::pmt::Value(42.0), gr::pmt::Value("Hello")};
        expected["floatVector"]                = Tensor<float>{1.0f, 2.0f, 3.0f};
        expected["doubleVector"]               = Tensor<double>{1.0, 2.0, 3.0};
        expected["stringVector"]               = Tensor<pmt::Value>{pmt::Value("Hello"), pmt::Value("World"), pmt::Value("Multiple\nlines")};
        expected["complexVector"]              = Tensor<std::complex<double>>{std::complex<double>{1.0, -1.0}, std::complex<double>{2.0, -2.0}, std::complex<double>{3.0, -3.0}};
        expected["nullVector"]                 = pmt::Value();
        expected["emptyVector"]                = Tensor<pmt::Value>{};
        expected["emptyPmtVector"]             = Tensor<pmt::Value>{};
        expected["emptyAfterNewline"]          = Tensor<pmt::Value>{};
        expected["flowDouble"]                 = Tensor<double>{1.0, 2.0, 3.0};
        expected["flowString1"]                = Tensor<pmt::Value>{pmt::Value("Hello1, "), pmt::Value("World1"), pmt::Value("Multiple1\nlines1")};
        expected["flowString2"]                = Tensor<pmt::Value>{pmt::Value("Hello2"), pmt::Value("World2"), pmt::Value("Single2")};
        expected["flowString3"]                = Tensor<pmt::Value>{pmt::Value("Hello3"), pmt::Value("World3"), pmt::Value("Single3")};
        expected["flowMultiline"]              = Tensor<pmt::Value>{pmt::Value("Hello, "), pmt::Value("]["), pmt::Value("World"), pmt::Value("Multiple\nlines")};
        expected["nestedVector"]               = Tensor<pmt::Value>{gr::pmt::Value(Tensor<pmt::Value>{pmt::Value("1"), pmt::Value("2")}), gr::pmt::Value(Tensor<pmt::Value>{pmt::Value(static_cast<int64_t>(3)), pmt::Value(static_cast<int64_t>(4))})};
        expected["nestedFlow"]                 = Tensor<pmt::Value>{gr::pmt::Value(Tensor<pmt::Value>{pmt::Value("1"), pmt::Value("2")}), gr::pmt::Value(Tensor<pmt::Value>{pmt::Value(static_cast<int64_t>(3)), pmt::Value(static_cast<int64_t>(4))})};
        expected["nestedVector2"]              = Tensor<pmt::Value>{                                                   //
            pmt::Value(static_cast<int64_t>(42)),                                                         //
            pmt::Value(Tensor<pmt::Value>{pmt::Value("1"), pmt::Value("2")}),                             //
            pmt::Value(Tensor<pmt::Value>{pmt::Value("3"), pmt::Value("4")}),                             //
            pmt::Value(gr::property_map{{"key", Tensor<pmt::Value>{pmt::Value("5"), pmt::Value("6")}}})}; //
        expected["vectorWithBlockMap"]         = Tensor<pmt::Value>{gr::property_map{{"name", gr::pmt::Value("ArraySink")}, {"id", gr::pmt::Value("gr::testing::ArraySink<double>")}, {"parameters", gr::pmt::Value(gr::property_map{{"name", gr::pmt::Value("Block")}})}}};
        expected["vectorWithColons"]           = Tensor<pmt::Value>{gr::pmt::Value("key: value"), gr::pmt::Value("key2: value2")};

        testYAML(src1, expected);

        expect(fuzzy_eq(formatResult(yaml::deserialize("key: !!int64 [foo, bar]")), "Error in 1:15: Invalid integral-type value 'foo' (error: Invalid argument)"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("key: !!str [foo, !!str bar]")), "Error in 1:24: Cannot have type tag for both list and list item"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("key: !!unknown [foo, bar]")), "Error in 1:6: Unsupported type"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("key: [foo\nbar]")), "Error in 2:1: Expected ',' or ']'"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("key:\n  - foo\n  bar")), "Error in 3:3: Expected list item"sv));

        constexpr std::string_view taggedListInList = R"(
value: !!str
    -
        - value
)";
        expect(fuzzy_eq(formatResult(yaml::deserialize(taggedListInList)), "Error in 3:6: Cannot have type tag for list containing lists"sv));

        constexpr std::string_view invalidListInList = R"(
value:
    - [ value
)";
        expect(fuzzy_eq(formatResult(yaml::deserialize(invalidListInList)), "Error in 4:1: Expected ',' or ']'"sv));

        constexpr std::string_view invalidScalarInList = R"(
value:
    - !!int64 value
)";
        expect(fuzzy_eq(formatResult(yaml::deserialize(invalidScalarInList)), "Error in 3:15: Invalid integral-type value 'value' (error: Invalid argument)"sv));

        constexpr std::string_view taggedListInTaggedList = R"(
value: !!str
    - !!str
      - value
    )";
        expect(fuzzy_eq(formatResult(yaml::deserialize(taggedListInTaggedList)), "Error in 3:12: Cannot have type tag for both list and list item"sv));

        constexpr std::string_view taggedListInTaggedList2 = R"(
value: !!str
    -
      - value
    )";
        expect(fuzzy_eq(formatResult(yaml::deserialize(taggedListInTaggedList2)), "Error in 3:6: Cannot have type tag for list containing lists"sv));

        constexpr std::string_view emptyTagInList = R"(
value:
    - !! "a string"
)";
        expect(fuzzy_eq(formatResult(yaml::deserialize(emptyTagInList)), "Error in 3:7: Unsupported type"sv));
    };

    "Maps"_test = [] {
        constexpr std::string_view src = R"yaml(
simple:
    key1: !!int8 42
    key2: !!int8 43
empty: {}
nested:
    key1:
        key2: !!int8 42 # comment to be ignored
        unknown_property: 42
        key3: !!int8 43
    key4:
        key5: !!int8 44
        key6: !!int8 45
flow: {key1: !!int8 42, key2: !!int8 43}
flow2: {key1: value1, key2: value2}
flow3: {key1: " value1  ", key2: "value2   "} # Add extra spaces inside quotes
flow4: {  key1  :  value1  ,  key2  : value2   } # Add extra spaces without quotes
flow_multiline: {key1: !!int8 42,
                 key2: !!int8 43}
flow_nested: {key1: {key2: !!int8 42, key3: !!int8 43}, key4: {key5: !!int8 44, key6: !!int8 45}}
flow_braces: {"}{": !!int8 42}
last: # End of document, null value
)yaml";

        gr::property_map expected;
        expected["simple"]         = gr::property_map{{"key1", gr::pmt::Value(static_cast<int8_t>(42))}, {"key2", gr::pmt::Value(static_cast<int8_t>(43))}};
        expected["empty"]          = gr::property_map{};
        expected["nested"]         = gr::property_map{{"key1", gr::property_map{{"key2", gr::pmt::Value(static_cast<int8_t>(42))}, {"unknown_property", gr::pmt::Value(static_cast<int64_t>(42))}, {"key3", gr::pmt::Value(static_cast<int8_t>(43))}}}, {"key4", gr::property_map{{"key5", gr::pmt::Value(static_cast<int8_t>(44))}, {"key6", gr::pmt::Value(static_cast<int8_t>(45))}}}};
        expected["flow"]           = gr::property_map{{"key1", gr::pmt::Value(static_cast<int8_t>(42))}, {"key2", gr::pmt::Value(static_cast<int8_t>(43))}};
        expected["flow2"]          = gr::property_map{{"key1", gr::pmt::Value("value1")}, {"key2", gr::pmt::Value("value2")}};
        expected["flow3"]          = gr::property_map{{"key1", gr::pmt::Value(" value1  ")}, {"key2", gr::pmt::Value("value2   ")}};
        expected["flow4"]          = gr::property_map{{"key1", gr::pmt::Value("value1")}, {"key2", gr::pmt::Value("value2")}};
        expected["flow_multiline"] = gr::property_map{{"key1", gr::pmt::Value(static_cast<int8_t>(42))}, {"key2", gr::pmt::Value(static_cast<int8_t>(43))}};
        expected["flow_nested"]    = gr::property_map{{"key1", gr::property_map{{"key2", gr::pmt::Value(static_cast<int8_t>(42))}, {"key3", gr::pmt::Value(static_cast<int8_t>(43))}}}, {"key4", gr::property_map{{"key5", gr::pmt::Value(static_cast<int8_t>(44))}, {"key6", gr::pmt::Value(static_cast<int8_t>(45))}}}};
        expected["flow_braces"]    = gr::property_map{{"}{", gr::pmt::Value(static_cast<int8_t>(42))}};
        expected["last"]           = pmt::Value();
        testYAML(src, expected);

        expect(fuzzy_eq(formatResult(yaml::deserialize("{")), "Error in 2:1: Unexpected end of document"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("key: {\nfoo: bar}")), "Error in 2:1: Flow sequence insufficiently indented"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("{ key: !!unknown foo }")), "Error in 1:8: Unsupported type"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("{ , }")), "Error in 1:3: Could not find key/value separator ':'"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("key: {key: foo\nbar}")), "Error in 2:1: Expected ',' or '}'"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("key: !!str { key: value }")), "Error in 1:6: Cannot have type tag for maps"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("key: { \"key\" }")), "Error in 1:14: Could not find key/value separator ':'"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("key: { \"\\u1234\": foo }")), "Error in 1:9: Parser limitation: Unicode escape sequences are not supported"sv));

        constexpr std::string_view mapListMix = R"(
value:
    key: value
    - value
)";
        expect(fuzzy_eq(formatResult(yaml::deserialize(mapListMix)), "Error in 4:5: Unexpected list item in map"sv));

        constexpr std::string_view taggedMapInList = R"(
value:
    - !!str
        key: value
)";

        expect(fuzzy_eq(formatResult(yaml::deserialize(taggedMapInList)), "Error in 3:7: Cannot have type tag for maps"sv));

        constexpr std::string_view invalidMapInList = R"(
value:
    - { key: value
)";

        expect(fuzzy_eq(formatResult(yaml::deserialize(invalidMapInList)), "Error in 4:1: Expected ',' or '}'"sv));

        constexpr std::string_view taggedMapBlock = R"(
value: !!str
    key: value
)";
        expect(fuzzy_eq(formatResult(yaml::deserialize(taggedMapBlock)), "Error in 2:8: Cannot have type tag for maps"sv));

        constexpr std::string_view invalidKeyComment = R"(
value: 42 # Comment
key#Comment: foo
)";
        expect(fuzzy_eq(formatResult(yaml::deserialize(invalidKeyComment)), "Error in 3:1: Could not find key/value separator ':'"sv));
    };

    "GRC"_test = [] {
        constexpr std::string_view src = R"(
blocks:
  - name: ArraySink<double>
    id: gr::testing::ArraySink<double>
    parameters:
      name: ArraySink<double>
  - name: ArraySource<double>
    id: gr::testing::ArraySource<double>
    parameters:
      name: ArraySource<double>
connections:
  - [ArraySource<double>, [0, 0], ArraySink<double>, [1, 1]]
  - [ArraySource<double>, [0, 1], ArraySink<double>, [1, 0]]
  - [ArraySource<double>, [1, 0], ArraySink<double>, [0, 0]]
  - [ArraySource<double>, [1, 1], ArraySink<double>, [0, 1]]
)";

        gr::property_map expected;
        gr::property_map block1;
        block1["name"]       = "ArraySink<double>";
        block1["id"]         = "gr::testing::ArraySink<double>";
        block1["parameters"] = gr::property_map{{"name", gr::pmt::Value("ArraySink<double>")}};
        gr::property_map block2;
        block2["name"]       = "ArraySource<double>";
        block2["id"]         = "gr::testing::ArraySource<double>";
        block2["parameters"] = gr::property_map{{"name", gr::pmt::Value("ArraySource<double>")}};
        expected["blocks"]   = Tensor<pmt::Value>{block1, block2};
        const auto zero      = pmt::Value{static_cast<int64_t>(0)};
        const auto one       = pmt::Value{static_cast<int64_t>(1)};
        using PmtVec         = Tensor<pmt::Value>;

        expected["connections"] = Tensor<pmt::Value>{                                                                                                  //
            PmtVec{pmt::Value("ArraySource<double>"), pmt::Value(Tensor{zero, zero}), pmt::Value("ArraySink<double>"), pmt::Value(Tensor{one, one})},  //
            PmtVec{pmt::Value("ArraySource<double>"), pmt::Value(Tensor{zero, one}), pmt::Value("ArraySink<double>"), pmt::Value(Tensor{one, zero})},  //
            PmtVec{pmt::Value("ArraySource<double>"), pmt::Value(Tensor{one, zero}), pmt::Value("ArraySink<double>"), pmt::Value(Tensor{zero, zero})}, //
            PmtVec{pmt::Value("ArraySource<double>"), pmt::Value(Tensor{one, one}), pmt::Value("ArraySink<double>"), pmt::Value(Tensor{zero, one})}};

        testYAML(src, expected);
    };

    "Complex"_test = [] {
        constexpr std::string_view src = R"(
complex: !!complex64 (1.0, -1.0)
complex2: !!complex32 (1.0, -1.0)
complex3: !!complex64 (1.0,-1.0)
complex4: !!complex32 (1.0,-1.0)
complex5: !!complex32 (  1.0  ,   -1.0)
)";

        gr::property_map expected;
        expected["complex"]  = std::complex<double>(1.0, -1.0);
        expected["complex2"] = std::complex<float>(1.0, -1.0);
        expected["complex3"] = std::complex<double>(1.0, -1.0);
        expected["complex4"] = std::complex<float>(1.0, -1.0);
        expected["complex5"] = std::complex<float>(1.0, -1.0);

        testYAML(src, expected);

        expect(fuzzy_eq(formatResult(yaml::deserialize("complex: !!complex64 (1.0, -1.0")), "Error in 1:22: Invalid value for complex<>-type"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("complex: !!complex64 (1.01.0)")), "Error in 1:22: Invalid value for complex<>-type"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("complex: !!complex64 Hello")), "Error in 1:22: Invalid value for complex<>-type"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("complex: !!complex64 (1.0, -1.0, 2.0)")), "Error in 1:22: Invalid value for complex<>-type"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("complex: !!complex64 (foo, bar)")), "Error in 1:22: std::invalid_argument exception for expected floating-point value of 'foo' - error: stod"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("complex: !!complex64 (1.0, bar)")), "Error in 1:22: std::invalid_argument exception for expected floating-point value of 'bar' - error: stod"sv));
    };

    "empty lines"_test = [] {
        "at the end of map"_test = [] {
            constexpr std::string_view src = R"(
key0: !!float32 0.5

)";

            gr::property_map expected;
            expected["key0"] = 0.5f;
            testYAML(src, expected);
        };

        "at the end of map - comment"_test = [] {
            constexpr std::string_view src = R"(
key0: !!float32 0.5
 # comment only
)";

            gr::property_map expected;
            expected["key0"] = 0.5f;
            testYAML(src, expected);
        };

        "in between map"_test = [] {
            constexpr std::string_view src = R"(
key0: !!float32 0.5

key1: !!float32 42.0
)";

            gr::property_map expected;
            expected["key0"] = 0.5f;
            expected["key1"] = 42.f;
            testYAML(src, expected);
        };

        "in between map - comment"_test = [] {
            constexpr std::string_view src = R"(
key0: !!float32 0.5
 # comment only
key1: !!float32 42.0
)";

            gr::property_map expected;
            expected["key0"] = 0.5f;
            expected["key1"] = 42.f;
            testYAML(src, expected);
        };

        "at the end list"_test = [] {
            constexpr std::string_view src = R"(
list1: !!float32
  - 0.5
  - 42

list2: !!float32
  - 43

)";

            gr::property_map expected;
            expected["list1"] = Tensor{0.5f, 42.f};
            expected["list2"] = Tensor{43.f};
            testYAML(src, expected);
        };

        "at the end list - comment"_test = [] {
            constexpr std::string_view src = R"(
list1: !!float32
  - 0.5
  - 42
 # just a comment
list2: !!float32
  - 43
 # another comment
)";

            gr::property_map expected;
            expected["list1"] = Tensor{0.5f, 42.f};
            expected["list2"] = Tensor{43.f};
            testYAML(src, expected);
        };

        "in between list"_test = [] {
            constexpr std::string_view src = R"(
list1: !!float32
  - 0.5

  - 42

list2: !!float32
  - 43

)";

            gr::property_map expected;
            expected["list1"] = Tensor{0.5f, 42.f};
            expected["list2"] = Tensor{43.f};
            testYAML(src, expected);
        };

        "in between list - comment"_test = [] {
            constexpr std::string_view src = R"(
list1: !!float32
  - 0.5
 # comment
  - 42
# comment 2
list2: !!float32
  - 43
  # com...ment
)";

            gr::property_map expected;
            expected["list1"] = Tensor{0.5f, 42.f};
            expected["list2"] = Tensor{43.f};
            testYAML(src, expected);
        };
    };

    "trim values"_test = [] {
        "trim float"_test = [] {
            constexpr std::string_view src = R"(key0: !!float32 0.5 )";

            gr::property_map expected;
            expected["key0"] = 0.5f;
            testYAML(src, expected);
        };
        "trim float - comment"_test = [] {
            constexpr std::string_view src = R"(key0: !!float32 0.5 # comment)";

            gr::property_map expected;
            expected["key0"] = 0.5f;
            testYAML(src, expected);
        };

        "trim float2"_test = [] {
            constexpr std::string_view src = R"(key0: 0.5 )";

            gr::property_map expected;
            expected["key0"] = 0.5;
            testYAML(src, expected);
        };
        "trim float - comment2"_test = [] {
            constexpr std::string_view src = R"(key0: 0.5 # comment)";

            gr::property_map expected;
            expected["key0"] = 0.5;
            testYAML(src, expected);
        };

        "trim String w/ space"_test = [] {
            constexpr std::string_view src = R"(key0: TestString )"; // note: space at the end

            gr::property_map expected;
            expected["key0"] = "TestString"s;
            testYAML(src, expected);
        };

        "trim String  w/ space in the middle"_test = [] {
            constexpr std::string_view src = R"(key0: Test String)";

            gr::property_map expected;
            expected["key0"] = "Test String"s;
            testYAML(src, expected);
        };

        "trim String  w/ space in the middle and comment"_test = [] {
            constexpr std::string_view src = R"(key0: Test String # comment)"; // note: space and comment at the end

            gr::property_map expected;
            expected["key0"] = "Test String"s;
            testYAML(src, expected);
        };

        "trim String  w/ space in the middle and comment"_test = [] {
            constexpr std::string_view src = R"(key0: "Test String" # comment)"; // note: space and comment at the end, quoted string

            gr::property_map expected;
            expected["key0"] = "Test String"s;
            testYAML(src, expected);
        };
    };

    "Odd Keys"_test = [] {
        constexpr std::string_view src = R"yaml(
: empty key
Key with spaces: !!int8 42
"quoted key with spaces": !!int8 43
"key with colon:": !!int8 44
"key with null byte \x00": !!int8 45
"key with newline \n": !!int8 46
"key with tab \t": !!int8 47
"key with CR \r": !!int8 48
"key with backslash \\": !!int8 49
"key with quote \"": !!int8 50
key::with::colons: !!int8 51
)yaml";

        gr::property_map expected;
        expected[""]                             = "empty key";
        expected["Key with spaces"]              = static_cast<int8_t>(42);
        expected["quoted key with spaces"]       = static_cast<int8_t>(43);
        expected["key with colon:"]              = static_cast<int8_t>(44);
        expected["key with null byte \x00"_spmr] = static_cast<int8_t>(45);
        expected["key with newline \n"]          = static_cast<int8_t>(46);
        expected["key with tab \t"]              = static_cast<int8_t>(47);
        expected["key with CR \r"]               = static_cast<int8_t>(48);
        expected["key with backslash \\"]        = static_cast<int8_t>(49);
        expected["key with quote \""]            = static_cast<int8_t>(50);
        expected["key::with::colons"]            = static_cast<int8_t>(51);

        testYAML(src, expected);
    };

    "Empty"_test = [] {
        testYAML({}, gr::property_map{});
        testYAML("  ", gr::property_map{});
        testYAML("---", gr::property_map{});
        testYAML("\n", gr::property_map{});
        testYAML("{}", gr::property_map{});
        testYAML("# Empty\n", gr::property_map{});
        testYAML("\n# Empty\n", gr::property_map{});
    };

    "Errors"_test = [] {
        expect(fuzzy_eq(formatResult(yaml::deserialize("value: !!")), "Error in 1:8: Unsupported type"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("value: !! a string")), "Error in 1:8: Unsupported type"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("value: !!unknown a string")), "Error in 1:8: Unsupported type"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("value: !!unknown a string")), "Error in 1:8: Unsupported type"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("value: \"Hello")), "Error in 1:8: Unterminated quote"sv));
        expect(fuzzy_eq(formatResult(yaml::deserialize("\"value: Hello")), "Error in 1:1: Unterminated quote"sv));
    };

    "Unsupported YAML"_test = [] {
        constexpr std::string_view unicode_escapes = R"(
unicode: !!str "\u1234"
)";
        expect(fuzzy_eq(formatResult(yaml::deserialize(unicode_escapes)), "Error in 2:18: Parser limitation: Unicode escape sequences are not supported"sv));

        constexpr std::string_view unicode_multiline = R"(
unicode: !!str |-
    \u1234
)";
        expect(fuzzy_eq(formatResult(yaml::deserialize(unicode_multiline)), "Error in 3:6: Parser limitation: Unicode escape sequences are not supported"sv));

        constexpr std::string_view multiple_documents = R"(
---
value: 42
---
value: 43
)";
        expect(fuzzy_eq(formatResult(yaml::deserialize(multiple_documents)), "Error in 4:1: Parser limitation: Multiple documents not supported"sv));
    };
};

const boost::ut::suite<"yaml error formatter"> _yamlFormatter = [] {
    using namespace boost::ut;
    using namespace std::string_literals;
    using namespace std::string_view_literals;

    constexpr auto testString = "Line one\nLine two with spaces at the end    \nLine three with extra text\nAnother line\nFinal line";

    "no error message"_test              = [] { expect(nothrow([] { std::println("no error message:\n{}", yaml::formatAsLines(testString, 2, 5)); })); };
    "short error message"_test           = [] { expect(nothrow([] { std::println("short error message:\n{}", yaml::formatAsLines(testString, 2, 5, "error")); })); };
    "long error message"_test            = [] { expect(nothrow([] { std::println("long error message:\n{}", yaml::formatAsLines(testString, 2, 5, "long error @column=5")); })); };
    "long error message @ column=0"_test = [] { expect(nothrow([] { std::println("long error message:\n{}", yaml::formatAsLines(testString, 2, 0, "long error @column=0")); })); };
};

int main() { /* tests are statically executed */ }
