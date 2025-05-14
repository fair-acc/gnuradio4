#include "pmtv/pmt.hpp"
#include <boost/ut.hpp>

#include <cstdint>
#include <gnuradio-4.0/meta/formatter.hpp>

#include <gnuradio-4.0/YamlPmt.hpp>
#include <limits>
#include <variant>

auto fuzzy_eq(std::string_view str1, std::string_view str2) {
    const auto len = std::min(str1.size(), str2.size());
    return std::equal(str1.data(), str1.data() + len, str2.data());
}

template<typename... Ts>
std::string variantTypeName(const std::variant<Ts...>& v) {
    return std::visit(
        [](auto&& arg) {
            // Get the type name of the current alternative
            using T = std::decay_t<decltype(arg)>;
            return gr::meta::type_name<T>();
        },
        v);
}

bool diff(const pmtv::map_t& original, const pmtv::map_t& deserialized);

void printDiff(const std::string& key, const pmtv::pmt& originalValue, const pmtv::pmt& deserializedValue) {
    std::ostringstream originalOss;
    std::ostringstream deserializedOss;
    pmtv::yaml::detail::serialize(originalOss, originalValue);
    pmtv::yaml::detail::serialize<>(deserializedOss, deserializedValue);
    std::cout << "Difference found at key: " << key << "\n";

    std::cout << "  Expected: " << originalOss.str() << "\n";
    std::cout << "  Deserialized: " << deserializedOss.str() << "\n";
}

// Work around NaN != NaN when comparing floats/doubles
template<typename T>
bool testEqual(const T& lhs, const T& rhs) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(lhs) && std::isnan(rhs)) {
            return true;
        }
    }

    if constexpr (std::ranges::random_access_range<T>) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (size_t i = 0; i < lhs.size(); ++i) {
            if (!testEqual(lhs[i], rhs[i])) {
                return false;
            }
        }
        return true;
    }

    if constexpr (std::is_same_v<T, pmtv::map_t>) {
        return !diff(lhs, rhs);
    }
    return lhs == rhs;
}

bool diff(const pmtv::map_t& original, const pmtv::map_t& deserialized) {
    bool foundDiff = false;
    for (const auto& [key, originalValue] : original) {
        auto it = deserialized.find(key);
        if (it == deserialized.end()) {
            std::cout << "Missing key in deserialized map: '" << key << "'\n";
            foundDiff = true;
            continue;
        }
        const auto& deserializedValue = it->second;
        if (originalValue.index() != deserializedValue.index()) {
            std::cout << "Found different types for: " << key << "\n";
            std::cout << "  Expected: " << variantTypeName(originalValue) << "\n";
            std::cout << "  Deserialized: " << variantTypeName(deserializedValue) << "\n";
            foundDiff = true;
        } else if (!std::visit(
                       [&](const auto& arg) {
                           using T = std::decay_t<decltype(arg)>;
                           return testEqual(arg, std::get<T>(deserializedValue));
                       },
                       originalValue)) {
            printDiff(key, originalValue, deserializedValue);
            foundDiff = true;
        }
    }
    for (const auto& [key, deserializedValue] : deserialized) {
        if (original.find(key) == original.end()) {
            std::cout << "Extra key in deserialized map: '" << key << "'\n";
            foundDiff = true;
        }
    }
    return foundDiff;
}

template<typename T>
std::string formatResult(const std::expected<T, pmtv::yaml::ParseError>& result) {
    if (!result.has_value()) {
        const auto& error = result.error();
        return std::format("Error in {}:{}: {}", error.line, error.column, error.message);
    } else {
        return "<no error>";
    }
}

void testYAML(std::string_view src, const pmtv::map_t expected, std::source_location location = std::source_location::current()) {
    using namespace boost::ut;
    // First test that the deserialized map matches the expected map
    const auto deserializedMap = pmtv::yaml::deserialize(src);
    if (deserializedMap) {
        expect(eq(diff(expected, *deserializedMap), false)) << std::format("testYAML unexpected error at: {}\n\n", location);
    } else {
        expect(false) << std::format("testYAML unexpected error at: {}:\n{}\n\n", location, pmtv::yaml::formatAsLines(src, deserializedMap.error()));
        return;
    }

    // Then test that serializing and deserializing the map again results in the same map
    const auto serializedStr    = pmtv::yaml::serialize(expected);
    const auto deserializedMap2 = pmtv::yaml::deserialize(serializedStr);
    if (deserializedMap2) {
        expect(eq(diff(expected, *deserializedMap2), false)) << std::format("testYAML unexpected error at: {} - string:\n{}\n\n", location, serializedStr);
    } else {
        expect(false) << std::format("testYAML unexpected error at: {}:\n {}\nYAML:\n{}", location, formatResult(deserializedMap2), serializedStr);
        ;
    }
}

const boost::ut::suite<"YamlPmtTests"> _yamlPmtTests = [] {
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast" // we want explicit casts for testing
#endif
    using namespace boost::ut;
    using namespace pmtv;
    using namespace std::string_literals;
    using namespace std::string_view_literals;

    "Comments and Whitespace"_test = [] {
        constexpr std::string_view src1 = R"(# Comment
double: !!float64 42 # Comment
string: "#Hello" # Comment
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

        pmtv::map_t expected;
        expected["double"] = 42.0;
        expected["string"] = "#Hello";
        expected["null"]   = std::monostate{};
        expected["number"] = static_cast<int64_t>(42);
        expected["list"]   = std::vector<pmtv::pmt>{};
        expected["list2"]  = std::vector<pmtv::pmt>{static_cast<int64_t>(42)};
        expected["map"]    = pmtv::map_t{};

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

        pmtv::map_t expected;
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

        pmtv::map_t expected;
        expected["null_value"]  = std::monostate{};
        expected["null_value2"] = std::monostate{};
        expected["null_value3"] = std::monostate{};
        expected["null_value4"] = std::monostate{};
        expected["null_value5"] = std::monostate{};
        expected["null_value6"] = std::monostate{};
        expected["null_value7"] = std::monostate{};
        expected["null_value8"] = std::monostate{};
        expected["not_null"]    = "NuLl";
        expected["null_at_end"] = std::monostate{};
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

        pmtv::map_t expected;
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

        pmtv::map_t expected;

        pmtv::map_t integers;
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

        pmtv::map_t doubles;
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
flowString: !!str ["Hello, ", "World", "Multiple\nlines"]
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

        pmtv::map_t expected;
        expected["boolVector"]                 = std::vector<bool>{true, false, true};
        expected["pmtVectorWithBools"]         = std::vector<pmtv::pmt>{true, false, true};
        expected["pmtVectorWithUntaggedBools"] = std::vector<pmtv::pmt>{true, false, true};
        expected["mixedPmtVector"]             = std::vector<pmtv::pmt>{true, 42.0, "Hello"};
        expected["floatVector"]                = std::vector<float>{1.0f, 2.0f, 3.0f};
        expected["doubleVector"]               = std::vector<double>{1.0, 2.0, 3.0};
        expected["stringVector"]               = std::vector<std::string>{"Hello", "World", "Multiple\nlines"};
        expected["complexVector"]              = std::vector<std::complex<double>>{{1.0, -1.0}, {2.0, -2.0}, {3.0, -3.0}};
        expected["nullVector"]                 = std::monostate{};
        expected["emptyVector"]                = std::vector<std::string>{};
        expected["emptyPmtVector"]             = std::vector<pmtv::pmt>{};
        expected["emptyAfterNewline"]          = std::vector<pmtv::pmt>{};
        expected["flowDouble"]                 = std::vector<double>{1.0, 2.0, 3.0};
        expected["flowString"]                 = std::vector<std::string>{"Hello, ", "World", "Multiple\nlines"};
        expected["flowMultiline"]              = std::vector<std::string>{"Hello, ", "][", "World", "Multiple\nlines"};
        expected["nestedVector"]               = std::vector<pmtv::pmt>{std::vector<std::string>{"1", "2"}, std::vector<pmtv::pmt>{static_cast<int64_t>(3), static_cast<int64_t>(4)}};
        expected["nestedFlow"]                 = std::vector<pmtv::pmt>{std::vector<std::string>{"1", "2"}, std::vector<pmtv::pmt>{static_cast<int64_t>(3), static_cast<int64_t>(4)}};
        expected["nestedVector2"]              = std::vector<pmtv::pmt>{static_cast<int64_t>(42), std::vector<std::string>{"1", "2"}, std::vector<std::string>{"3", "4"}, pmtv::map_t{{"key", std::vector<std::string>{"5", "6"}}}};
        expected["vectorWithBlockMap"]         = std::vector<pmtv::pmt>{pmtv::map_t{{"name", "ArraySink"}, {"id", "gr::testing::ArraySink<double>"}, {"parameters", pmtv::map_t{{"name", "Block"}}}}};
        expected["vectorWithColons"]           = std::vector<pmtv::pmt>{"key: value", "key2: value2"};

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
flow_multiline: {key1: !!int8 42,
                 key2: !!int8 43}
flow_nested: {key1: {key2: !!int8 42, key3: !!int8 43}, key4: {key5: !!int8 44, key6: !!int8 45}}
flow_braces: {"}{": !!int8 42}
last: # End of document, null value
)yaml";

        pmtv::map_t expected;
        expected["simple"]         = pmtv::map_t{{"key1", static_cast<int8_t>(42)}, {"key2", static_cast<int8_t>(43)}};
        expected["empty"]          = pmtv::map_t{};
        expected["nested"]         = pmtv::map_t{{"key1", pmtv::map_t{{"key2", static_cast<int8_t>(42)}, {"unknown_property", static_cast<int64_t>(42)}, {"key3", static_cast<int8_t>(43)}}}, {"key4", pmtv::map_t{{"key5", static_cast<int8_t>(44)}, {"key6", static_cast<int8_t>(45)}}}};
        expected["flow"]           = pmtv::map_t{{"key1", static_cast<int8_t>(42)}, {"key2", static_cast<int8_t>(43)}};
        expected["flow_multiline"] = pmtv::map_t{{"key1", static_cast<int8_t>(42)}, {"key2", static_cast<int8_t>(43)}};
        expected["flow_nested"]    = pmtv::map_t{{"key1", pmtv::map_t{{"key2", static_cast<int8_t>(42)}, {"key3", static_cast<int8_t>(43)}}}, {"key4", pmtv::map_t{{"key5", static_cast<int8_t>(44)}, {"key6", static_cast<int8_t>(45)}}}};
        expected["flow_braces"]    = pmtv::map_t{{"}{", static_cast<int8_t>(42)}};
        expected["last"]           = std::monostate{};
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

        pmtv::map_t expected;
        pmtv::map_t block1;
        block1["name"]       = "ArraySink<double>";
        block1["id"]         = "gr::testing::ArraySink<double>";
        block1["parameters"] = pmtv::map_t{{"name", "ArraySink<double>"}};
        pmtv::map_t block2;
        block2["name"]       = "ArraySource<double>";
        block2["id"]         = "gr::testing::ArraySource<double>";
        block2["parameters"] = pmtv::map_t{{"name", "ArraySource<double>"}};
        expected["blocks"]   = std::vector<pmtv::pmt>{block1, block2};
        const auto zero      = pmtv::pmt{static_cast<int64_t>(0)};
        const auto one       = pmtv::pmt{static_cast<int64_t>(1)};
        using PmtVec         = std::vector<pmtv::pmt>;

        expected["connections"] = std::vector<pmtv::pmt>{PmtVec{"ArraySource<double>", std::vector{zero, zero}, "ArraySink<double>", std::vector{one, one}}, PmtVec{"ArraySource<double>", std::vector{zero, one}, "ArraySink<double>", std::vector{one, zero}}, PmtVec{"ArraySource<double>", std::vector{one, zero}, "ArraySink<double>", std::vector{zero, zero}}, PmtVec{"ArraySource<double>", std::vector{one, one}, "ArraySink<double>", std::vector{zero, one}}};

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

        pmtv::map_t expected;
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

        pmtv::map_t expected;
        expected[""]                         = "empty key";
        expected["Key with spaces"]          = static_cast<int8_t>(42);
        expected["quoted key with spaces"]   = static_cast<int8_t>(43);
        expected["key with colon:"]          = static_cast<int8_t>(44);
        expected["key with null byte \x00"s] = static_cast<int8_t>(45);
        expected["key with newline \n"]      = static_cast<int8_t>(46);
        expected["key with tab \t"]          = static_cast<int8_t>(47);
        expected["key with CR \r"]           = static_cast<int8_t>(48);
        expected["key with backslash \\"]    = static_cast<int8_t>(49);
        expected["key with quote \""]        = static_cast<int8_t>(50);
        expected["key::with::colons"]        = static_cast<int8_t>(51);

        testYAML(src, expected);
    };

    "Empty"_test = [] {
        testYAML({}, pmtv::map_t{});
        testYAML("  ", pmtv::map_t{});
        testYAML("---", pmtv::map_t{});
        testYAML("\n", pmtv::map_t{});
        testYAML("{}", pmtv::map_t{});
        testYAML("# Empty\n", pmtv::map_t{});
        testYAML("\n# Empty\n", pmtv::map_t{});
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
    using namespace pmtv;
    using namespace std::string_literals;
    using namespace std::string_view_literals;

    constexpr auto testString = "Line one\nLine two with spaces at the end    \nLine three with extra text\nAnother line\nFinal line";

    "no error message"_test              = [] { expect(nothrow([] { std::println("no error message:\n{}", yaml::formatAsLines(testString, 2, 5)); })); };
    "short error message"_test           = [] { expect(nothrow([] { std::println("short error message:\n{}", yaml::formatAsLines(testString, 2, 5, "error")); })); };
    "long error message"_test            = [] { expect(nothrow([] { std::println("long error message:\n{}", yaml::formatAsLines(testString, 2, 5, "long error @column=5")); })); };
    "long error message @ column=0"_test = [] { expect(nothrow([] { std::println("long error message:\n{}", yaml::formatAsLines(testString, 2, 0, "long error @column=0")); })); };
};

int main() { /* tests are statically executed */ }
