#ifndef GNURADIO_YAML_PMT_HPP
#define GNURADIO_YAML_PMT_HPP

#include <algorithm>
#include <cassert>
#include <cctype>
#include <charconv>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <iomanip>
#include <list>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <pmtv/pmt.hpp>

#include <format>

namespace pmtv::yaml {

enum class TypeTagMode {
    None, /// do not set !!<type>
    Auto  /// set !!<type> type-tags explicitely
};

struct ParseError {
    std::size_t line   = std::numeric_limits<std::size_t>::max();
    std::size_t column = std::numeric_limits<std::size_t>::max();
    std::string message;
};

namespace detail {

template<typename T>
struct is_complex : std::false_type {};

template<typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template<typename T>
constexpr bool is_complex_v = is_complex<T>::value;

// serialization

inline std::string escapeString(std::string_view str, bool escapeForQuotedString) {
    std::string result;
    result.reserve(2 * str.size());
    for (char c : str) {
        if (c == '"' && escapeForQuotedString) {
            result.append("\\\"");
        } else if (c == '\n' && escapeForQuotedString) {
            result.append("\\n");
        } else if (c == '\\' && escapeForQuotedString) {
            result.append("\\\\");
        } else if (c == '\t' && escapeForQuotedString) {
            result.append("\\t");
        } else if (c == '\r' && escapeForQuotedString) {
            result.append("\\r");
        } else if (c == '\b') {
            result.append("\\b");
        } else if (std::iscntrl(static_cast<unsigned char>(c)) && c != '\n') {
            result.append(std::format("\\x{:02x}", static_cast<unsigned char>(c)));
        } else {
            result.push_back(c);
        }
    }
    return result;
}

inline void indent(std::ostream& os, int level) { os << std::setw(level * 2) << std::setfill(' ') << ""; }

template<TypeTagMode tagMode = TypeTagMode::Auto>
void serialize(std::ostream& os, const pmtv::pmt& value, int level = 0);

template<TypeTagMode tagMode>
inline void serializeString(std::ostream& os, std::string_view value, int level, bool useMultiline = false) noexcept {
    if (useMultiline) {
        const bool endsWithNewline = value.ends_with('\n');
        os << "|";
        if (!endsWithNewline) {
            os << "-";
        }
        os << "\n";
        std::istringstream stream(std::string(value.data(), value.size()));

        std::string line;
        while (std::getline(stream, line)) {
            indent(os, level + 1); // increase indentation for multi-line content
            os << escapeString(line, false) << "\n";
        }
    } else {
        if constexpr (tagMode == TypeTagMode::Auto) {
            os << std::format("\"{}\"\n", escapeString(value, true));
        } else {
            os << std::format("{}\n", escapeString(value, true));
        }
    }
}

template<typename T>
constexpr std::string_view tag_for_type() noexcept {
    if constexpr (std::is_same_v<T, std::monostate>) {
        return "!!null";
    } else if constexpr (std::is_same_v<T, bool>) {
        return "!!bool";
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        return "!!uint8";
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        return "!!uint16";
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return "!!uint32";
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        return "!!uint64";
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        return "!!int8";
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        return "!!int16";
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return "!!int32";
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return "!!int64";
    } else if constexpr (std::is_same_v<T, float>) {
        return "!!float32";
    } else if constexpr (std::is_same_v<T, double>) {
        return "!!float64";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return "!!complex32";
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return "!!complex64";
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "!!str";
    } else {
        return "";
    }
}

template<TypeTagMode tagMode>
void serialize(std::ostream& os, const pmtv::pmt& var, int level) {
    std::visit(
        [&os, level]<typename T>(const T& value) {
            if constexpr (tagMode == TypeTagMode::Auto && !std::is_same_v<T, pmtv::map_t>) {
                if constexpr (!std::is_same_v<T, std::string> && std::ranges::random_access_range<T>) {
                    os << tag_for_type<typename T::value_type>();
                } else {
                    os << tag_for_type<T>() << " ";
                }
            }
            if constexpr (std::same_as<T, std::monostate>) {
                os << "null\n";
            } else if constexpr (std::same_as<T, bool>) {
                os << (value ? "true" : "false") << "\n";
            } else if constexpr (std::is_integral_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    // write uint8_t and int8_t as integer, not char
                    os << static_cast<int>(value) << "\n";
                } else {
                    os << value << "\n";
                }
            } else if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(value)) {
                    os << ".nan\n";
                } else if (std::isinf(value)) {
                    os << (value < 0 ? "-.inf" : ".inf") << "\n";
                } else {
                    os << value << "\n";
                }
            } else if constexpr (is_complex_v<T>) {
                os << "(" << value.real() << "," << value.imag() << ")\n";
            } else if constexpr (std::is_same_v<T, std::string>) {
                // Use multiline for strings containing newlines and printable characters only
                bool multiline = value.contains('\n') && std::ranges::all_of(value, [](unsigned char c) { return std::isprint(c) || c == '\n'; });
                serializeString<tagMode>(os, value, level, multiline);
            } else if constexpr (std::same_as<T, pmtv::map_t>) {
                // flow-style formatting
                if (value.empty()) {
                    os << " {}\n";
                    return;
                }
                // block-style formatting
                os << "\n";
                for (const auto& [key, val] : value) {
                    indent(os, level + 1);
                    if (key.contains(':') || !std::ranges::all_of(key, ::isprint)) {
                        os << std::format("\"{}\": ", escapeString(key, true));
                    } else {
                        os << key << ": ";
                    }
                    serialize<tagMode>(os, val, level + 1);
                }
            } else if constexpr (std::ranges::random_access_range<T>) {
                // flow-style formatting
                if (value.empty()) {
                    os << " []\n";
                    return;
                }
                // block-style formatting
                os << "\n";
                for (const auto& item : value) {
                    indent(os, level + 1);
                    os << "- ";
                    constexpr TypeTagMode childTagMode = tagMode == TypeTagMode::Auto ? std::is_same_v<typename T::value_type, pmtv::pmt> ? TypeTagMode::Auto : TypeTagMode::None : tagMode;
                    serialize<childTagMode>(os, item, level + 1);
                }
            }
        },
        var);
}

// deserialization

struct ValueParseError {
    std::size_t offset;
    std::string message;
};

struct ParseContext {
    std::span<const std::string_view> lines;
    std::size_t                       lineIdx   = 0;
    std::size_t                       columnIdx = 0;

    std::string_view currentLine() { return lineIdx < lines.size() ? lines[lineIdx] : std::string_view{}; }

    bool documentStart() const { return lineIdx == 0 && columnIdx == 0; }

    bool startsWith(std::string_view sv) const {
        if (atEndOfLine()) {
            return false;
        }
        return lines[lineIdx].substr(columnIdx).starts_with(sv);
    }

    bool startsWithToken(std::string_view sv) const {
        if (atEndOfLine()) {
            return false;
        }
        if (columnIdx + sv.size() < lines[lineIdx].size() && !std::isspace(static_cast<unsigned char>(lines[lineIdx][columnIdx + sv.size()]))) {
            return false;
        }
        return lines[lineIdx].substr(columnIdx).starts_with(sv);
    }

    bool startsWith(char c) const {
        if (atEndOfLine()) {
            return false;
        }
        return lines[lineIdx][columnIdx] == c;
    }

    bool consumeIfStartsWith(std::string_view sv) {
        if (startsWith(sv)) {
            consume(sv.size());
            return true;
        }
        return false;
    }

    bool consumeIfStartsWith(char c) {
        if (startsWith(c)) {
            consume(1);
            return true;
        }
        return false;
    }

    bool consumeIfStartsWithToken(std::string_view sv) {
        if (startsWithToken(sv)) {
            consume(sv.size());
            return true;
        }
        return false;
    }

    char front() const {
        assert(lineIdx < lines.size());
        assert(columnIdx < lines[lineIdx].size());
        return lines[lineIdx][columnIdx];
    }

    void skipToNextLine() {
        if (lineIdx < lines.size()) {
            ++lineIdx;
            columnIdx = 0;
        }
    }

    void consume(std::size_t n) {
        columnIdx += n;
        assert(columnIdx <= lines[lineIdx].size());
    }

    void consumeSpaces() {
        if (atEndOfDocument() || atEndOfLine()) {
            return;
        }
        const std::size_t nextNonSpace = lines[lineIdx].find_first_not_of(' ', columnIdx);
        if (nextNonSpace != std::string_view::npos) {
            columnIdx = nextNonSpace;
        } else {
            columnIdx = lines[lineIdx].size();
        }
    }

    void consumeWhitespaceAndComments() {
        while (!atEndOfDocument()) {
            consumeSpaces();
            if (!atEndOfLine() && front() != '#') {
                break;
            }
            skipToNextLine();
        }
    }

    std::string_view remainingLine() const { return atEndOfLine() ? std::string_view{} : lines[lineIdx].substr(columnIdx); }

    bool atEndOfLine() const { return atEndOfDocument() || columnIdx == lines[lineIdx].size(); }

    bool atEndOfDocument() const { return lineIdx == lines.size(); }

    std::size_t currentIndent(std::string_view indentChars = " ") const { return lines[lineIdx].find_first_not_of(indentChars); }

    ParseError makeError(std::string message) const { return {.line = lineIdx + 1, .column = columnIdx + 1, .message = std::move(message)}; }

    ParseError makeErrorAtColumn(std::string message, std::size_t colIdx) const { return {.line = lineIdx + 1, .column = colIdx + 1, .message = std::move(message)}; }

    ParseError makeError(ValueParseError error) const { return {.line = lineIdx + 1, .column = columnIdx + 1 + error.offset, .message = std::move(error.message)}; }
};

inline std::vector<std::string_view> split(std::string_view str, std::string_view separator = "\n") {
    std::vector<std::string_view> lines;

    std::size_t start = 0;
    while (start < str.size()) {
        std::size_t end = str.find(separator, start);
        if (end == std::string_view::npos) {
            end = str.size();
        }
        std::string_view line = str.substr(start, end - start);
        lines.emplace_back(line);
        start = end + separator.size();
    }
    return lines;
}

template<typename R, template<typename> class Fnc, typename... Args>
R applyTag(std::string_view tag, Args&&... args) {
    if (tag == "!!bool") {
        return Fnc<bool>{}(std::forward<Args>(args)...);
    } else if (tag == "!!int8") {
        return Fnc<int8_t>{}(std::forward<Args>(args)...);
    } else if (tag == "!!int16") {
        return Fnc<int16_t>{}(std::forward<Args>(args)...);
    } else if (tag == "!!int32") {
        return Fnc<int32_t>{}(std::forward<Args>(args)...);
    } else if (tag == "!!int64") {
        return Fnc<int64_t>{}(std::forward<Args>(args)...);
    } else if (tag == "!!uint8") {
        return Fnc<uint8_t>{}(std::forward<Args>(args)...);
    } else if (tag == "!!uint16") {
        return Fnc<uint16_t>{}(std::forward<Args>(args)...);
    } else if (tag == "!!uint32") {
        return Fnc<uint32_t>{}(std::forward<Args>(args)...);
    } else if (tag == "!!uint64") {
        return Fnc<uint64_t>{}(std::forward<Args>(args)...);
    } else if (tag == "!!float32") {
        return Fnc<float>{}(std::forward<Args>(args)...);
    } else if (tag == "!!float64") {
        return Fnc<double>{}(std::forward<Args>(args)...);
    } else if (tag == "!!complex32") {
        return Fnc<std::complex<float>>{}(std::forward<Args>(args)...);
    } else if (tag == "!!complex64") {
        return Fnc<std::complex<double>>{}(std::forward<Args>(args)...);
    } else if (tag == "!!str") {
        return Fnc<std::string>{}(std::forward<Args>(args)...);
    } else {
        return Fnc<std::monostate>{}(std::forward<Args>(args)...);
    }
}

inline std::optional<std::string> parseBytesFromHex(std::string_view sv) {
    std::string result;
    if (sv.size() % 2 != 0) {
        return std::nullopt;
    }
    result.reserve(sv.size() / 2);
    for (std::size_t i = 0; i < sv.size(); i += 2) {
        std::string_view byte = sv.substr(i, 2);
        unsigned         u8;
        if (auto [_, ec] = std::from_chars(byte.data(), byte.data() + byte.size(), u8, 16); ec == std::errc{}) {
            result.push_back(static_cast<char>(u8));
        } else {
            return std::nullopt;
        }
    }
    return result;
}

inline std::expected<std::string, ValueParseError> resolveYamlEscapes_multiline(ParseContext& ctx) {
    std::string_view str = ctx.remainingLine();
    std::string      result;
    result.reserve(str.size());

    for (std::size_t i = 0UZ; i < str.size(); ++i) {
        if (str[i] == '\\' && i + 1 < str.size()) {
            ++i;
            switch (str[i]) {
            case '0': result.push_back('\0'); break;
            case 'x': {
                if (i + 2 >= str.size()) {
                    result.push_back('\\');
                    result.push_back(str[i]);
                    break;
                }
                const std::optional<std::string> byte = parseBytesFromHex(str.substr(i + 1, 2));
                if (!byte) {
                    result.push_back('\\');
                    result.push_back(str[i]);
                    break;
                }
                result.append(*byte);
                i += 2;
                break;
            }
            case 'u': {
                return std::unexpected(ValueParseError{i, "Parser limitation: Unicode escape sequences are not supported"});
            }
            default:
                result.push_back('\\');
                result.push_back(str[i]);
                break;
            }
        } else {
            result.push_back(str[i]);
        }
    }
    ctx.consume(str.size());
    return result;
};

inline std::expected<std::string, ValueParseError> resolveYamlEscapes_quoted(std::string_view str) {
    std::string result;
    result.reserve(str.size());
    for (std::size_t i = 0UZ; i < str.size(); ++i) {
        if (str[i] == '\\' && i + 1 < str.size()) {
            ++i;
            switch (str[i]) {
            case '\\': result.push_back('\\'); break;
            case 'b': result.push_back('\b'); break;
            case 'n': result.push_back('\n'); break;
            case 't': result.push_back('\t'); break;
            case 'r': result.push_back('\r'); break;
            case '"': result.push_back('"'); break;
            case '0': result.push_back('\0'); break;
            case 'x': {
                if (i + 2 >= str.size()) {
                    result.push_back('\\');
                    result.push_back(str[i]);
                    break;
                }
                const std::optional<std::string> byte = parseBytesFromHex(str.substr(i + 1, 2));
                if (!byte) {
                    result.push_back('\\');
                    result.push_back(str[i]);
                    break;
                }
                result.append(*byte);
                i += 2;
                break;
            }
            case 'u': {
                return std::unexpected(ValueParseError{i, "Parser limitation: Unicode escape sequences are not supported"});
            }
            default:
                result.push_back('\\');
                result.push_back(str[i]);
                break;
            }
        } else {
            result.push_back(str[i]);
        }
    }
    return result;
};

template<typename T>
std::expected<T, ValueParseError> parseAs(std::string_view sv) {
    auto trim = [](std::string_view s) {
        auto first = std::ranges::find_if_not(s, isspace);
        auto last  = std::ranges::find_if_not(s | std::views::reverse, isspace).base();
        return s.substr(static_cast<std::size_t>(first - s.begin()), static_cast<std::size_t>(last - first));
    };

    if constexpr (std::is_same_v<T, std::monostate>) {
        return std::monostate{};
    } else if constexpr (std::is_same_v<T, bool>) {
        sv = trim(sv); // trim leading and trailing whitespace
        if (sv == "true" || sv == "True" || sv == "TRUE") {
            return true;
        }
        if (sv == "false" || sv == "False" || sv == "FALSE") {
            return false;
        }
        return std::unexpected(ValueParseError{0UZ, "Invalid value for bool-type"});
    } else if constexpr (std::is_floating_point_v<T>) {
        sv = trim(sv); // trim leading and trailing whitespace
        if (sv == ".inf" || sv == ".Inf" || sv == ".INF") {
            return std::numeric_limits<T>::infinity();
        } else if (sv == "-.inf" || sv == "-.Inf" || sv == "-.INF") {
            return -std::numeric_limits<T>::infinity();
        } else if (sv == ".nan" || sv == ".NaN" || sv == ".NAN") {
            return std::numeric_limits<T>::quiet_NaN();
        }
        std::string tempParse(sv.data(), sv.size());
        try {
            if constexpr (std::is_same_v<T, float>) {
                return std::stof(tempParse);
            } else {
                return std::stod(tempParse);
            }
        } catch (std::invalid_argument& e) { // specifically: std::invalid_argument or std::out_of_range
            return std::unexpected(ValueParseError{0UZ, std::format("std::invalid_argument exception for expected floating-point value of '{}' - error: {}", tempParse, e.what())});
        } catch (std::out_of_range& e) { // specifically: std::invalid_argument or std::out_of_range
            return std::unexpected(ValueParseError{0UZ, std::format("std::out_of_range exception for expected floating-point value of '{}' - error: {}", tempParse, e.what())});
        } catch (std::exception& e) { // specifically: std::invalid_argument or std::out_of_range
            return std::unexpected(ValueParseError{0UZ, std::format("std::exception for expected floating-point value of '{}' - error: {}", tempParse, e.what())});
        }
    } else if constexpr (std::is_integral_v<T>) {
        sv                 = trim(sv); // trim leading and trailing whitespace
        auto parseWithBase = [](std::string_view s, int base) -> std::expected<T, ValueParseError> {
            T value;
            const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value, base);
            if (ec != std::errc{} || ptr != s.data() + s.size()) {
                return std::unexpected(ValueParseError{0UZ, std::format("Invalid integral-type value '{}' (error: {})", s, std::make_error_code(ec).message())});
            }
            return value;
        };
        if (sv.starts_with("0x")) {
            return parseWithBase(sv.substr(2), 16);
        } else if (sv.starts_with("0o")) {
            return parseWithBase(sv.substr(2), 8);
        } else if (sv.starts_with("0b")) {
            return parseWithBase(sv.substr(2), 2);
        }
        return parseWithBase(sv, 10);
    } else if constexpr (is_complex_v<T>) {
        sv = trim(sv);
        if (!sv.starts_with('(') || !sv.ends_with(')')) {
            return std::unexpected(ValueParseError{0UZ, "Invalid value for complex<>-type"});
        }
        sv.remove_prefix(1UZ);
        sv.remove_suffix(1UZ);
        const std::vector<std::string_view> segments = split(sv, ",");
        if (segments.size() != 2UZ) {
            return std::unexpected(ValueParseError{0UZ, "Invalid value for complex<>-type"});
        }
        using value_type = typename T::value_type;
        auto real        = parseAs<value_type>(trim(segments[0UZ]));
        if (!real) {
            return real;
        }
        auto imag = parseAs<value_type>(trim(segments[1]));
        if (!imag) {
            return imag;
        }
        return T{*real, *imag};
    } else if constexpr (std::is_same_v<T, std::string>) {
        // present default behaviour: no trimming for strings i.e. keep spaces if any
        return resolveYamlEscapes_quoted(sv);
    } else {
        static_assert(false, "Unsupported type");
        return std::monostate();
    }
}

template<typename T>
struct ParseAs {
    std::expected<T, ValueParseError> operator()(std::string_view sv) { return parseAs<T>(sv); }
};

inline bool isKnownTag(std::string_view tag) { return tag == "!!null" || tag == "!!bool" || tag == "!!uint8" || tag == "!!uint16" || tag == "!!uint32" || tag == "!!uint64" || tag == "!!int8" || tag == "!!int16" || tag == "!!int32" || tag == "!!int64" || tag == "!!float32" || tag == "!!float64" || tag == "!!complex32" || tag == "!!complex64" || tag == "!!str"; }

inline std::expected<std::string_view, ParseError> parseTag(ParseContext& ctx) {
    std::string_view tag;
    if (ctx.startsWith("!!")) {
        std::string_view line    = ctx.remainingLine();
        std::size_t      tag_end = line.find(' ');
        if (tag_end != std::string_view::npos) {
            tag = line.substr(0, tag_end);
            if (!isKnownTag(tag)) {
                return std::unexpected(ctx.makeError("Unsupported type"));
            }
            ctx.consume(tag_end + 1);
        } else {
            tag = line;
            if (!isKnownTag(tag)) {
                return std::unexpected(ctx.makeError("Unsupported type"));
            }
            ctx.consume(line.size());
        }
    }
    return tag;
}

std::expected<pmtv::map_t, ParseError> parseMap(ParseContext& ctx, int parent_indent_level);
std::expected<pmtv::pmt, ParseError>   parseList(ParseContext& ctx, std::string_view type_tag, int parent_indent_level);

inline size_t findClosingQuote(std::string_view sv, char quoteChar) {
    bool inEscape = false;
    for (size_t i = 1; i < sv.size(); ++i) {
        if (inEscape) {
            inEscape = false;
            continue;
        }
        if (sv[i] == '\\') {
            inEscape = true;
            continue;
        }
        if (sv[i] == quoteChar) {
            return i;
        }
    }
    return std::string_view::npos;
}

inline std::pair<std::size_t, std::size_t> findString(std::string_view sv, std::string_view extraDelimiters = {}) {
    if (sv.empty()) {
        return {0, 0};
    }
    const char firstChar = sv.front();
    const bool quoted    = firstChar == '"' || firstChar == '\'';
    if (!quoted) {
        // Check for extra delimiter first (',' for flow, ':' for keys)
        std::size_t delimPos = sv.find_first_of(extraDelimiters);
        if (delimPos != std::string_view::npos) {
            return {0, delimPos};
        }
        // Ignore trailing comments
        std::size_t commentPos = sv.find('#');
        if (commentPos != std::string_view::npos) {
            return {0, commentPos};
        }
        return {0, sv.size()};
    }

    std::size_t closePos = findClosingQuote(sv, firstChar);
    if (closePos != std::string_view::npos) {
        return {1, closePos - 1};
    }
    return {std::string_view::npos, std::string_view::npos}; // Unterminated quote
}

template<typename Fnc>
std::expected<pmtv::pmt, ParseError> parseNextString(ParseContext& ctx, std::string_view extraDelimiters, Fnc fnc) {
    auto [offset, length] = findString(ctx.remainingLine(), extraDelimiters);
    if (offset == std::string_view::npos) {
        return std::unexpected(ctx.makeError("Unterminated quote"));
    }
    ctx.consume(offset);
    const auto fncResult = fnc(offset, ctx.remainingLine().substr(0, length));
    if (!fncResult) {
        return std::unexpected(ctx.makeError(fncResult.error()));
    }
    // Only move cursor in the good case, to not lose the error location (column)
    ctx.consume(offset + length);
    return *fncResult;
}

inline std::expected<pmtv::pmt, ParseError> parsePlainScalar(ParseContext& ctx, std::string_view typeTag, std::string_view extraDelimiters = {}) {
    // if we have a type tag, enforce the type
    if (!typeTag.empty()) {
        return parseNextString(ctx, extraDelimiters, [typeTag](std::size_t, std::string_view sv) { return applyTag<std::expected<pmtv::pmt, ValueParseError>, ParseAs>(typeTag, sv); });
    }

    // fallback for parsing without a YAML tag
    return parseNextString(ctx, extraDelimiters, [&](std::size_t quoteOffset, std::string_view sv) -> std::expected<pmtv::pmt, ValueParseError> {
        // If it's quoted, treat as string
        if (quoteOffset > 0) {
            return resolveYamlEscapes_quoted(sv);
        }

        // null
        if (sv.empty() || sv == "null" || sv == "Null" || sv == "NULL" || sv == "~") {
            return std::monostate{};
        }

        // boolean
        if (sv == "true" || sv == "True" || sv == "TRUE") {
            return true;
        }
        if (sv == "false" || sv == "False" || sv == "FALSE") {
            return false;
        }

        // try numbers
        if (const std::expected<std::int64_t, ValueParseError> asInt = parseAs<std::int64_t>(sv)) {
            return *asInt;
        }
        if (const std::expected<double, ValueParseError> asDouble = parseAs<double>(sv)) {
            return *asDouble;
        }

        // Anything else: string
        return parseAs<std::string>(sv).transform_error([&](ValueParseError error) { return ValueParseError{quoteOffset + error.offset, error.message}; });
    });
}

inline std::expected<pmtv::pmt, ParseError> parseScalar(ParseContext& ctx, std::string_view typeTag, int currentIndentLevel) {
    const std::size_t initialLine = ctx.lineIdx;
    ctx.consumeWhitespaceAndComments();

    if (ctx.atEndOfDocument()) {
        return std::monostate{};
    }
    const bool skippedLines = ctx.lineIdx > initialLine;
    if (skippedLines && currentIndentLevel >= 0 && ctx.currentIndent() <= static_cast<std::size_t>(currentIndentLevel)) {
        return std::monostate{};
    }
    // handle multi-line indicators '|', '|-', '>', '>-'
    if ((typeTag == "!!str" || typeTag.empty()) && (!ctx.atEndOfLine() && (ctx.front() == '|' || ctx.front() == '>'))) {
        char indicator = ctx.front();
        ctx.consume(1);

        const bool trailingNewline = !ctx.consumeIfStartsWith('-');

        ctx.consumeSpaces();
        const auto [offset, length] = findString(ctx.remainingLine());
        if (length > 0) {
            return std::unexpected(ctx.makeError("Unexpected characters after multi-line indicator"));
        }
        std::ostringstream oss;
        const std::size_t  expectedIndent = static_cast<std::size_t>(currentIndentLevel + 2);

        bool firstLine = true;
        ctx.skipToNextLine();

        for (; !ctx.atEndOfDocument(); ctx.skipToNextLine()) {
            std::size_t lineIndent = ctx.currentIndent();
            if (lineIndent == std::string_view::npos) {
                // empty or whitespace-only line
                // folded style ('|'): empty line becomes newline
                // literal style ('>'): retain empty line
                oss << '\n';
                continue;
            }

            if (lineIndent < expectedIndent) {
                // indentation decreased; end of multi-line string
                break;
            }

            ctx.consume(expectedIndent);
            if (indicator == '>' && !firstLine) {
                oss << ' ';
            }
            std::expected<std::string, ValueParseError> resolved = resolveYamlEscapes_multiline(ctx);
            if (!resolved) {
                return std::unexpected(ctx.makeError(resolved.error()));
            }
            oss << *resolved;
            if (indicator == '|') {
                oss << '\n';
            }
            firstLine = false;
        }

        std::string result = oss.str();
        if (indicator == '|' && !trailingNewline) {
            // trim trailing newlines for literal block style '|-'
            while (!result.empty() && result.back() == '\n') {
                result.pop_back();
            }
        } else if (indicator == '>') {
            if (!trailingNewline) {
                // trim trailing spaces for folded block style '>-'
                while (!result.empty() && std::isspace(static_cast<unsigned char>(result.back()))) {
                    result.pop_back();
                }
            } else {
                // add trailing newline for folded block style '>'
                result.push_back('\n');
            }
        }
        ctx.consumeSpaces();
        return result;
    }

    std::expected<pmtv::pmt, ParseError> result = parsePlainScalar(ctx, typeTag);

    if (!result) {
        return std::unexpected(result.error());
    }

    ctx.consumeSpaces();
    if (!ctx.atEndOfLine()) {
        const auto [offset, length] = findString(ctx.remainingLine());
        if (offset > 0 || length > 0) {
            return std::unexpected(ctx.makeError("Unexpected characters after scalar value"));
        }
    }

    ctx.skipToNextLine();

    return result;
}

enum class ValueType { List, Map, Scalar };

inline std::expected<std::string, ParseError> parseKey(ParseContext& ctx, std::string_view extraDelimiters = {}) {
    ctx.consumeSpaces();

    if (ctx.startsWith("-")) {
        return std::unexpected(ctx.makeError("Unexpected list item in map"));
    }

    const auto& [quoteOffset, length] = findString(ctx.remainingLine(), extraDelimiters);
    if (quoteOffset == std::string_view::npos) {
        return std::unexpected(ctx.makeError("Unterminated quote"));
    }
    if (quoteOffset > 0) {
        // quoted
        auto maybeKey = resolveYamlEscapes_quoted(ctx.remainingLine().substr(quoteOffset, length));
        if (!maybeKey) {
            return std::unexpected(ctx.makeError(maybeKey.error()));
        }
        ctx.consume(2 * quoteOffset + length);
        ctx.consumeSpaces();
        if (!ctx.consumeIfStartsWithToken(":")) {
            return std::unexpected(ctx.makeError("Could not find key/value separator ':'"));
        }
        return *maybeKey;
    }

    // not quoted
    std::size_t colonPos = [](auto sv) {
        for (std::size_t pos = 0UZ; pos < sv.size(); ++pos) {
            pos = sv.find(':', pos);
            if (pos == std::string_view::npos) {
                return pos;
            }
            if (pos == sv.size() - 1 || std::isspace(static_cast<unsigned char>(sv[pos + 1]))) {
                return pos;
            }
        }
        return std::string_view::npos;
    }(ctx.remainingLine());

    std::size_t commentPos = ctx.remainingLine().find('#');
    if (colonPos == std::string_view::npos || (commentPos != std::string_view::npos && commentPos < colonPos)) {
        return std::unexpected(ctx.makeError("Could not find key/value separator ':'"));
    }
    std::string key(ctx.remainingLine().substr(0, colonPos));
    ctx.consume(colonPos + 1);

    return key;
}

inline ValueType peekToFindValueType(ParseContext ctx, int previousIndent) {
    const std::size_t initialLine = ctx.lineIdx;
    ctx.consumeWhitespaceAndComments();

    if (ctx.atEndOfDocument()) {
        return ValueType::Scalar;
    }

    const bool skippedLines = ctx.lineIdx > initialLine;
    if (skippedLines && previousIndent >= 0 && ctx.currentIndent() <= static_cast<std::size_t>(previousIndent)) {
        return ValueType::Scalar;
    }
    if (ctx.startsWith("[") || (ctx.startsWith("- ") || ctx.remainingLine() == "-")) {
        return ValueType::List;
    }

    if (ctx.startsWith("{")) {
        return ValueType::Map;
    }

    const std::expected<std::string, ParseError> key = parseKey(ctx);
    return key.has_value() ? ValueType::Map : ValueType::Scalar;
}

template<typename T>
struct ConvertList {
    pmtv::pmt operator()(const std::vector<pmtv::pmt>& list) {
        if constexpr (std::is_same_v<T, std::monostate>) {
            return std::monostate{};
        } else {
            auto resultView = list | std::views::transform([](const auto& item) { return std::get<T>(item); });
            return std::vector<T>(resultView.begin(), resultView.end());
        }
    }
};

enum class FlowType { List, Map };
template<FlowType Type>
std::expected<pmtv::pmt, ParseError> parseFlow(ParseContext& ctx, std::string_view typeTag, int parentIndentLevel) {
    using ResultType          = std::conditional_t<Type == FlowType::List, pmtv::pmt, pmtv::map_t>;
    using TemporaryResultType = std::conditional_t<Type == FlowType::List, std::vector<pmtv::pmt>, pmtv::map_t>;
    using ReturnType          = std::expected<ResultType, ParseError>;

    auto              makeError    = [&](std::string message) -> ReturnType { return std::unexpected(ctx.makeError(std::move(message))); };
    const std::size_t startLineIdx = ctx.lineIdx;

    constexpr char closingChar = Type == FlowType::List ? ']' : '}';

    TemporaryResultType result;

    while (!ctx.atEndOfDocument()) {
        ctx.consumeWhitespaceAndComments();

        if (ctx.atEndOfDocument()) {
            return makeError("Unexpected end of document");
        }
        if (ctx.consumeIfStartsWith(closingChar)) {
            // end of flow sequence
            break;
        }
        if (ctx.lineIdx > startLineIdx && parentIndentLevel >= 0 && ctx.currentIndent() <= static_cast<std::size_t>(parentIndentLevel)) {
            return makeError("Flow sequence insufficiently indented");
        }

        auto parseElementValue = [&] -> std::expected<pmtv::pmt, ParseError> {
            const std::expected<std::string_view, ParseError> maybeTag = parseTag(ctx);
            if (!maybeTag.has_value()) {
                return ReturnType{std::unexpected(maybeTag.error())};
            }
            std::string_view nestedTag = maybeTag.value();

            if (!typeTag.empty() && !nestedTag.empty()) {
                return makeError("Cannot have type tag for both list and list item");
            }

            const std::string_view localTag = !nestedTag.empty() ? nestedTag : typeTag;

            ctx.consumeWhitespaceAndComments();

            if (ctx.consumeIfStartsWith('[')) {
                return parseFlow<FlowType::List>(ctx, localTag, parentIndentLevel);
            }
            if (ctx.consumeIfStartsWith('{')) {
                return parseFlow<FlowType::Map>(ctx, localTag, parentIndentLevel);
            }

            constexpr std::string_view extraDelimiters = Type == FlowType::List ? ",]" : ",}";
            return parsePlainScalar(ctx, localTag, extraDelimiters);
        };

        if constexpr (Type == FlowType::List) {
            std::expected<pmtv::pmt, ParseError> value = parseElementValue();
            if (!value.has_value()) {
                return ReturnType{std::unexpected(value.error())};
            }
            result.emplace_back(value.value());
        } else {
            std::expected<std::string, ParseError> key = parseKey(ctx, ",");
            if (!key.has_value()) {
                return ReturnType{std::unexpected(key.error())};
            }
            ctx.consumeWhitespaceAndComments();
            std::expected<pmtv::pmt, ParseError> value = parseElementValue();
            if (!value.has_value()) {
                return ReturnType{std::unexpected(value.error())};
            }
            // result is a std::map<std::string, pmt, ...>
            result.insert_or_assign(key.value(), value.value());
        }
        ctx.consumeWhitespaceAndComments();
        if (ctx.consumeIfStartsWith(",")) {
            // continue to next value
        } else if (ctx.consumeIfStartsWith(closingChar)) {
            // end of flow sequence
            break;
        } else {
            if constexpr (Type == FlowType::List) {
                return makeError("Expected ',' or ']'");
            } else {
                return makeError("Expected ',' or '}'");
            }
        }
    }
    if constexpr (Type == FlowType::List) {
        if (typeTag.empty()) {
            return ReturnType{result};
        }
        return ReturnType{applyTag<pmtv::pmt, ConvertList>(typeTag, result)};
    } else {
        return ReturnType{result};
    }
}

inline std::expected<pmtv::map_t, ParseError> parseMap(ParseContext& ctx, int parentIndentLevel) {
    ctx.consumeWhitespaceAndComments();
    if (ctx.consumeIfStartsWith("{")) {
        auto result = parseFlow<FlowType::Map>(ctx, "", parentIndentLevel);
        ctx.skipToNextLine();

        if (!result) {
            return std::unexpected(result.error());
        }

        if (!std::holds_alternative<pmtv::map_t>(*result)) {
            return std::unexpected(ctx.makeError("Expected map in flow-style map"));
        }

        return std::get<pmtv::map_t>(*result);
    }

    pmtv::map_t map;
    bool        firstLine = true;

    while (!ctx.atEndOfDocument()) {
        if (ctx.startsWith("---")) {
            return std::unexpected(ctx.makeError("Parser limitation: Multiple documents not supported"));
        }
        ctx.consumeWhitespaceAndComments();

        const std::size_t line_indent = firstLine ? ctx.currentIndent(" -") : ctx.currentIndent(); // Ignore "-" if map is in a list

        if (parentIndentLevel >= 0 && line_indent <= static_cast<std::size_t>(parentIndentLevel)) {
            // indentation decreased; end of current map
            break;
        }

        firstLine = false;

        const std::expected<std::string, ParseError> maybeKey = parseKey(ctx);
        if (!maybeKey.has_value()) {
            return std::unexpected(maybeKey.error());
        }

        std::string key = maybeKey.value();

        ctx.consumeSpaces();
        std::size_t                                       tagPos   = ctx.columnIdx;
        const std::expected<std::string_view, ParseError> maybeTag = parseTag(ctx);
        if (!maybeTag.has_value()) {
            return std::unexpected(maybeTag.error());
        }
        std::string_view typeTag = maybeTag.value();
        ctx.consumeSpaces();

        const auto peekedType = peekToFindValueType(ctx, static_cast<int>(line_indent));

        switch (peekedType) {
        case ValueType::List: {
            std::expected<pmtv::pmt, ParseError> parsedValue = parseList(ctx, typeTag, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            map.insert_or_assign(key, parsedValue.value());
            break;
        }
        case ValueType::Map: {
            if (!typeTag.empty()) {
                return std::unexpected(ctx.makeErrorAtColumn("Cannot have type tag for maps", tagPos));
            }
            std::expected<pmtv::map_t, ParseError> parsedValue = parseMap(ctx, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            map.insert_or_assign(key, parsedValue.value());
            break;
        }
        case ValueType::Scalar: {
            std::expected<pmtv::pmt, ParseError> parsedValue = parseScalar(ctx, typeTag, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            map.insert_or_assign(key, parsedValue.value());
            break;
        }
        }
    }
    return map;
}

inline std::expected<pmtv::pmt, ParseError> parseList(ParseContext& ctx, std::string_view typeTag, int parentIndentLevel) {
    ctx.consumeWhitespaceAndComments();
    if (ctx.consumeIfStartsWith("[")) {
        auto l = parseFlow<FlowType::List>(ctx, typeTag, parentIndentLevel);
        ctx.skipToNextLine();
        return l;
    }

    // Use std::list instead of std::vector as a workaround to the ASAN problem with emscripten 4.0.8 due to vector reallocation.
    // A list never relocates its nodes, so references and iterators remain valid for the lifetime of the container.
    // Once parsing ends, we copy to std::vector.
    std::list<pmtv::pmt> list;

    while (!ctx.atEndOfDocument()) {
        ctx.consumeWhitespaceAndComments();

        const std::size_t line_indent = ctx.currentIndent();
        if (parentIndentLevel >= 0 && line_indent <= static_cast<size_t>(parentIndentLevel)) {
            // indentation decreased; end of current list
            break;
        }

        if (!ctx.consumeIfStartsWith("-")) {
            // not a list item
            return std::unexpected(ctx.makeError("Expected list item"));
        }

        ctx.consumeSpaces();

        const std::size_t itemIndent = ctx.columnIdx;

        const std::expected<std::string_view, ParseError> maybeLocalTag = parseTag(ctx);
        if (!maybeLocalTag.has_value()) {
            return std::unexpected(maybeLocalTag.error());
        }
        std::string_view localTag = maybeLocalTag.value();
        if (!typeTag.empty() && !localTag.empty()) {
            return std::unexpected(ctx.makeError("Cannot have type tag for both list and list item"));
        }

        const std::string_view tag = !typeTag.empty() ? typeTag : localTag;

        ctx.consumeSpaces();

        const ValueType peekedType = peekToFindValueType(ctx, static_cast<int>(line_indent));
        switch (peekedType) {
        case ValueType::List: {
            if (!typeTag.empty()) {
                return std::unexpected(ctx.makeErrorAtColumn("Cannot have type tag for list containing lists", itemIndent));
            }
            std::expected<pmtv::pmt, ParseError> parsedValue = parseList(ctx, tag, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            list.push_back(parsedValue.value());
            break;
        }
        case ValueType::Map: {
            if (!localTag.empty()) {
                return std::unexpected(ctx.makeErrorAtColumn("Cannot have type tag for maps", itemIndent));
            }
            std::expected<pmtv::map_t, ParseError> parsedValue = parseMap(ctx, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            list.emplace_back(parsedValue.value());
            break;
        }
        case ValueType::Scalar: {
            std::expected<pmtv::pmt, ParseError> parsedValue = parseScalar(ctx, tag, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            list.emplace_back(parsedValue.value());
            break;
        }
        }
    }
    std::vector<pmtv::pmt> vec(list.begin(), list.end());
    if (typeTag.empty()) {
        return vec;
    }
    return applyTag<pmtv::pmt, ConvertList>(typeTag, vec);
}

} // namespace detail

template<TypeTagMode tagMode = TypeTagMode::Auto>
std::string serialize(const pmtv::map_t& map) {
    std::ostringstream oss;
    if (!map.empty()) {
        detail::serialize<tagMode>(oss, map, -1); // Start at level -1 to avoid indenting top-level keys
    }
    return oss.str();
}

inline std::expected<pmtv::map_t, ParseError> deserialize(std::string_view yaml_str) {
    std::vector<std::string_view> lines = detail::split(yaml_str, "\n");
    detail::ParseContext          ctx{.lines = lines};
    ctx.consumeWhitespaceAndComments();
    ctx.consumeIfStartsWith("---");
    return detail::parseMap(ctx, -1);
}

namespace detail {
template<std::size_t N>
struct fixed_string {
    char data[N]{};
    constexpr fixed_string(const char (&arr)[N]) {
        for (std::size_t i = 0; i < N; ++i) {
            data[i] = arr[i];
        }
    }
};
} // namespace detail

template<detail::fixed_string lineEndMarker = "⏎", detail::fixed_string lineStartMarker = "│", detail::fixed_string chevron = "^">
std::string formatAsLines(std::string_view input, std::size_t line = std::numeric_limits<std::size_t>::max(), std::size_t column = std::numeric_limits<std::size_t>::max(), std::string_view errorMsg = {}) {
    std::size_t total = input.empty() ? 0UZ : 1UZ + static_cast<std::size_t>(std::ranges::count(input, '\n'));
    if (!total) {
        return std::string(input);
    }

    std::size_t width = std::to_string(total - 1UZ).size();
    std::string out;
    out.reserve(input.size() + total * 10UZ);

    std::size_t start = 0UZ;
    for (std::size_t i = 0UZ; i < total; ++i) {
        auto pos = input.find('\n', start);
        if (pos == std::string_view::npos) {
            pos = input.size();
        }
        auto lineView = input.substr(start, pos - start);

        std::format_to(std::back_inserter(out), "{:>{}}:{}{}{}\n", i, width, lineStartMarker.data, lineView, lineEndMarker.data);

        if (i == line) {
            std::size_t col = std::min(column, lineView.size());
            std::format_to(std::back_inserter(out), "{:>{}}", "", width + 2UZ);
            if (errorMsg.empty()) {
                std::format_to(std::back_inserter(out), "{:>{}}{}\n", ' ', col, chevron.data);
            } else if (errorMsg.size() < col + 1UZ) {
                if (col - errorMsg.length() == 0UZ) {
                    std::format_to(std::back_inserter(out), "{}{}\n", errorMsg, chevron.data);
                } else {
                    std::format_to(std::back_inserter(out), "{:>{}}{}{}\n", ' ', col - errorMsg.length(), errorMsg, chevron.data);
                }
            } else {
                if (col == 0UZ) {
                    std::format_to(std::back_inserter(out), "{}{}\n", chevron.data, errorMsg);
                } else {
                    std::format_to(std::back_inserter(out), "{:>{}}{}{}\n", ' ', col, chevron.data, errorMsg);
                }
            }
            std::format_to(std::back_inserter(out), "{:>{}}\n", "", width + 2UZ);
        }
        start = (pos == input.size()) ? pos : pos + 1;
    }
    return out;
}

template<detail::fixed_string lineEndMarker = "⏎", detail::fixed_string lineStartMarker = "│", detail::fixed_string chevron = "^">
constexpr std::string formatAsLines(std::string_view input, ParseError error = {}) {
    return formatAsLines<lineEndMarker, lineStartMarker, chevron>(input, error.line, error.column, error.message);
}

} // namespace pmtv::yaml

#endif // GNURADIO_YAML_PMT_HPP
