#include <algorithm>
#include <cassert>
#include <cctype>
#include <charconv>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <pmtv/pmt.hpp>

#include <fmt/format.h>

namespace pmtv::yaml {

struct ParseError {
    std::size_t line;
    std::size_t column;
    std::string message;
    std::string context;
};
namespace detail {

template<typename T>
struct is_complex : std::false_type {};

template<typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// serialization

inline std::string escapeString(std::string_view str, bool escapeForQuotedString) {
    std::string result;
    result.reserve(str.size());
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
        } else if (std::iscntrl(c) && c != '\n') {
            result.append(fmt::format("\\x{:02x}", static_cast<unsigned char>(c)));
        } else {
            result.push_back(c);
        }
    }
    return result;
}

inline void indent(std::ostream& os, int level) { os << std::setw(level * 2) << std::setfill(' ') << ""; }

enum class TypeTagMode { None, Auto };

template<TypeTagMode tagMode = TypeTagMode::Auto>
inline void serialize(std::ostream& os, const pmtv::pmt& value, int level = 0);

inline void serializeString(std::ostream& os, std::string_view value, int level, bool is_multiline = false, bool use_folded = false) noexcept {
    if (is_multiline) {
        const auto ends_with_newline = value.ends_with('\n');
        os << (use_folded ? ">" : "|");
        if (!ends_with_newline) {
            os << "-";
        }
        os << "\n";
        std::istringstream stream(std::string(value.data()));

        std::string line;
        while (std::getline(stream, line)) {
            indent(os, level + 1); // increase indentation for multi-line content
            os << escapeString(line, false) << "\n";
        }
    } else {
        os << '"' << escapeString(value, true) << '"' << "\n";
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
inline void serialize(std::ostream& os, const pmtv::pmt& var, int level) {
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
                bool multiline  = value.contains('\n') && std::ranges::all_of(value, [](char c) { return std::isprint(c) || c == '\n'; });
                bool use_folded = value.contains("  "); // Use folded if indented lines are detected
                serializeString(os, value, level, multiline, use_folded);
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
                        os << '"' << escapeString(key, true) << "\": ";
                    } else {
                        os << key << ": ";
                    }
                    serialize<TypeTagMode::Auto>(os, val, level + 1);
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
                    constexpr auto childTagMode = std::is_same_v<typename T::value_type, pmtv::pmt> ? TypeTagMode::Auto : TypeTagMode::None;
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

    bool documentStart() const { return lineIdx == 0 && columnIdx == 0; }

    bool startsWith(std::string_view sv) const {
        if (atEndOfLine()) {
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

    char front() const { return lines[lineIdx][columnIdx]; }

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
        const auto nextNonSpace = lines[lineIdx].find_first_not_of(' ', columnIdx);
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

    std::size_t currentIndent() const { return lines[lineIdx].find_first_not_of(' '); }

    ParseError makeError(std::string message) const { return {.line = lineIdx + 1, .column = columnIdx + 1, .message = std::move(message)}; }

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
        lines.emplace_back(str.data() + start, end - start);
        start = end + 1;
    }
    return lines;
}

template<typename R, template<typename> class Fnc, typename... Args>
inline R applyTag(std::string_view tag, Args&&... args) {
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
    result.reserve(sv.size() / 2);
    for (std::size_t i = 0; i < sv.size(); i += 2) {
        std::string_view byte = sv.substr(i, 2);
        char             c;
        if (auto [_, ec] = std::from_chars(byte.begin(), byte.end(), c, 16); ec == std::errc{}) {
            result.push_back(c);
        } else {
            return std::nullopt;
        }
    }
    return result;
}

inline std::expected<std::string, ValueParseError> resolveYamlEscapes_multiline(ParseContext& ctx) {
    auto        str = ctx.remainingLine();
    std::string result;
    result.reserve(str.size());

    for (auto i = 0UZ; i < str.size(); ++i) {
        if (str[i] == '\\' && i + 1 < str.size()) {
            ++i;
            switch (str[i]) {
            case '0': result.push_back('\0'); break;
            case 'x': {
                if (i + 2 >= str.size()) {
                    return std::unexpected(ValueParseError{i, "Invalid escape sequence"});
                }
                const auto byte = parseBytesFromHex(str.substr(i + 1, 2));
                if (!byte) {
                    return std::unexpected(ValueParseError{i, "Invalid escape sequence"});
                }
                result.append(*byte);
                i += 2;
                break;
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
    for (auto i = 0UZ; i < str.size(); ++i) {
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
                    return std::unexpected(ValueParseError{i, "Invalid escape sequence"});
                }
                const auto byte = parseBytesFromHex(str.substr(i + 1, 2));
                if (!byte) {
                    return std::unexpected(ValueParseError{i, "Invalid escape sequence"});
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
    if constexpr (std::is_same_v<T, std::monostate>) {
        return std::monostate{};
    } else if constexpr (std::is_same_v<T, bool>) {
        if (sv == "true" || sv == "True" || sv == "TRUE") {
            return true;
        }
        if (sv == "false" || sv == "False" || sv == "FALSE") {
            return false;
        }
        return std::unexpected(ValueParseError{0UZ, "Invalid value for type"});
    } else if constexpr (std::is_arithmetic_v<T>) {
        if constexpr (std::is_floating_point_v<T>) {
            if (sv == ".inf" || sv == ".Inf" || sv == ".INF") {
                return std::numeric_limits<T>::infinity();
            } else if (sv == "-.inf" || sv == "-.Inf" || sv == "-.INF") {
                return -std::numeric_limits<T>::infinity();
            } else if (sv == ".nan" || sv == ".NaN" || sv == ".NAN") {
                return std::numeric_limits<T>::quiet_NaN();
            }
        }

        if constexpr (std::is_integral_v<T>) {
            if (sv.contains(".")) {
                // from_chars() accepts "123.456", but we reject it
                return std::unexpected(ValueParseError{0UZ, "Invalid value for type"});
            }

            auto parseWithBase = [](std::string_view s, int base) -> std::expected<T, ValueParseError> {
                T value;
                const auto [_, ec] = std::from_chars(s.begin(), s.end(), value, base);
                if (ec != std::errc{}) {
                    return std::unexpected(ValueParseError{0UZ, "Invalid value for type"});
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
        }
        T value;
        const auto [_, ec] = std::from_chars(sv.begin(), sv.end(), value);
        if (ec != std::errc{}) {
            return std::unexpected(ValueParseError{0UZ, "Invalid value for type"});
        }
        return value;
    } else if constexpr (is_complex_v<T>) {
        auto trim = [](std::string_view s) {
            while (!s.empty() && std::isspace(s.front())) {
                s.remove_prefix(1);
            }
            while (!s.empty() && std::isspace(s.back())) {
                s.remove_suffix(1);
            }
            return s;
        };
        sv = trim(sv);
        if (!sv.starts_with('(') || !sv.ends_with(')')) {
            return std::unexpected(ValueParseError{0UZ, "Invalid value for type"});
        }
        auto trimmed = sv;
        trimmed.remove_prefix(1);
        trimmed.remove_suffix(1);
        const auto segments = split(trimmed, ",");
        if (segments.size() != 2) {
            return std::unexpected(ValueParseError{0UZ, "Invalid value for type"});
        }
        using value_type = typename T::value_type;
        auto real        = parseAs<value_type>(trim(segments[0]));
        if (!real) {
            return real;
        }
        auto imag = parseAs<value_type>(trim(segments[1]));
        if (!imag) {
            return imag;
        }
        return T{*real, *imag};
    } else if constexpr (std::is_same_v<T, std::string>) {
        return resolveYamlEscapes_quoted(sv);
    } else {
        static_assert(false, "Unsupported type");
        return std::monostate();
    }
}

template<typename T>
struct ParseAs {
    auto operator()(std::string_view sv) { return parseAs<T>(sv); }
};

inline bool isKnownTag(std::string_view tag) { return tag == "!!null" || tag == "!!bool" || tag == "!!uint8" || tag == "!!uint16" || tag == "!!uint32" || tag == "!!uint64" || tag == "!!int8" || tag == "!!int16" || tag == "!!int32" || tag == "!!int64" || tag == "!!float32" || tag == "!!float64" || tag == "!!complex32" || tag == "!!complex64" || tag == "!!str"; }

inline std::expected<std::string_view, ParseError> parseTag(ParseContext& ctx) {
    std::string_view tag;
    if (ctx.startsWith("!!")) {
        auto line    = ctx.remainingLine();
        auto tag_end = line.find(' ');
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
    if (sv.size() < 2) {
        return {0, sv.size()};
    }
    const char firstChar = sv.front();
    const auto quoted    = firstChar == '"' || firstChar == '\'';
    if (!quoted) {
        // Check for extra delimiter first (',' for flow, ':' for keys)
        auto delimPos = sv.find_first_of(extraDelimiters);
        if (delimPos != std::string_view::npos) {
            return {0, delimPos};
        }
        // Ignore trailing comments
        auto commentPos = sv.find('#');
        if (commentPos != std::string_view::npos) {
            return {0, commentPos};
        }
        return {0, sv.size()};
    }

    auto closePos = findClosingQuote(sv, firstChar);
    if (closePos != std::string_view::npos) {
        return {1, closePos - 1};
    }
    return {1, sv.size() - 1}; // Unterminated quote
}

template<typename Fnc>
inline std::expected<pmtv::pmt, ParseError> parseNextString(ParseContext& ctx, std::string_view extraDelimiters, Fnc fnc) {
    auto [offset, length] = findString(ctx.remainingLine(), extraDelimiters);
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
        if (const auto asInt = parseAs<int64_t>(sv)) {
            return *asInt;
        }
        if (const auto asDouble = parseAs<double>(sv)) {
            return *asDouble;
        }

        // Anything else: string
        return parseAs<std::string>(sv).transform_error([&](ValueParseError error) { return ValueParseError{quoteOffset + error.offset, error.message}; });
    });
}

inline std::expected<pmtv::pmt, ParseError> parseScalar(ParseContext& ctx, std::string_view typeTag, int currentIndentLevel) {
    // remove leading spaces
    ctx.consumeSpaces();

    // handle multi-line indicators '|', '|-', '>', '>-'
    if ((typeTag == "!!str" || typeTag.empty()) && (!ctx.atEndOfLine() && (ctx.front() == '|' || ctx.front() == '>'))) {
        char indicator = ctx.front();
        ctx.consume(1);

        const auto trailingNewline = !ctx.consumeIfStartsWith('-');

        ctx.consumeSpaces();
        const auto& [_, length] = findString(ctx.remainingLine());
        if (length > 0) {
            return std::unexpected(ctx.makeError("Unexpected characters after multi-line indicator"));
        }
        std::ostringstream oss;
        const auto         expectedIndent = static_cast<std::size_t>(currentIndentLevel + 2);

        bool firstLine = true;
        ctx.skipToNextLine();

        for (; !ctx.atEndOfDocument(); ctx.skipToNextLine()) {
            auto lineIndent = ctx.currentIndent();
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
            auto resolved = resolveYamlEscapes_multiline(ctx);
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

    const auto result = parsePlainScalar(ctx, typeTag);

    if (!result) {
        return std::unexpected(result.error());
    }

    ctx.consumeSpaces();
    if (!ctx.atEndOfLine()) {
        const auto& [offset, length] = findString(ctx.remainingLine());
        if (offset > 0 || length > 0) {
            return std::unexpected(ctx.makeError("Unexpected characters after scalar value"));
        }
    }

    ctx.skipToNextLine();

    return result;
}

enum class ValueType { List, Map, Scalar };

inline ValueType peekToFindValueType(ParseContext ctx, int previousIndent) {
    ctx.consumeSpaces();
    if (ctx.startsWith("[")) {
        return ValueType::List;
    }
    if (ctx.startsWith("{")) {
        return ValueType::Map;
    }
    if (!ctx.atEndOfLine()) {
        return ValueType::Scalar;
    }
    ctx.skipToNextLine();
    while (!ctx.atEndOfDocument()) {
        if (ctx.startsWith("#")) {
            ctx.skipToNextLine();
            continue;
        }
        const auto indent = ctx.currentIndent();
        if (indent == std::string_view::npos) {
            ctx.skipToNextLine();
            continue;
        }
        if (previousIndent >= 0 && indent <= static_cast<std::size_t>(previousIndent)) {
            return ValueType::Scalar;
        }
        ctx.consumeSpaces();
        if (ctx.startsWith("-")) {
            return ValueType::List;
        }
        if (ctx.remainingLine().find(':') != std::string_view::npos) {
            return ValueType::Map;
        }
        return ValueType::Scalar;
    }
    return ValueType::Scalar;
}

inline std::expected<std::string, ParseError> parseKey(ParseContext& ctx, std::string_view extraDelimiters = {}) {
    ctx.consumeSpaces();

    if (ctx.startsWith("-")) {
        return std::unexpected(ctx.makeError("Unexpected list item in map."));
    }

    const auto& [quoteOffset, length] = findString(ctx.remainingLine(), extraDelimiters);
    if (quoteOffset > 0) {
        // quoted
        auto maybeKey = resolveYamlEscapes_quoted(ctx.remainingLine().substr(quoteOffset, length));
        if (!maybeKey) {
            return std::unexpected(ctx.makeError(maybeKey.error()));
        }
        ctx.consume(2 * quoteOffset + length);
        ctx.consumeSpaces();
        if (!ctx.atEndOfLine() && ctx.front() != ':') {
            return std::unexpected(ctx.makeError("Could not find key/value separator ':'"));
        }
        ctx.consume(1);
        return *maybeKey;
    }

    // not quoted
    auto colonPos   = ctx.remainingLine().find(':');
    auto commentPos = ctx.remainingLine().find('#');
    if (colonPos == std::string_view::npos || (commentPos != std::string_view::npos && commentPos < colonPos)) {
        return std::unexpected(ctx.makeError("Could not find key/value separator ':'"));
    }
    auto key = std::string(ctx.remainingLine().substr(0, colonPos));
    ctx.consume(colonPos + 1);

    return key;
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
inline auto parseFlow(ParseContext& ctx, std::string_view typeTag, int parentIndentLevel) {
    using ResultType          = std::conditional_t<Type == FlowType::List, pmtv::pmt, pmtv::map_t>;
    using TemporaryResultType = std::conditional_t<Type == FlowType::List, std::vector<pmtv::pmt>, pmtv::map_t>;
    using ReturnType          = std::expected<ResultType, ParseError>;
    auto       makeError      = [&](std::string message) -> ReturnType { return std::unexpected(ctx.makeError(std::move(message))); };
    const auto startLineIdx   = ctx.lineIdx;

    constexpr auto closingChar = Type == FlowType::List ? ']' : '}';

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
            const auto maybeTag = parseTag(ctx);
            if (!maybeTag.has_value()) {
                return ReturnType{std::unexpected(maybeTag.error())};
            }
            auto nestedTag = maybeTag.value();

            if (!typeTag.empty() && !nestedTag.empty()) {
                return makeError("Cannot have type tag for both list and list item");
            }

            const auto localTag = !nestedTag.empty() ? nestedTag : typeTag;

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
            auto value = parseElementValue();
            if (!value.has_value()) {
                return ReturnType{std::unexpected(value.error())};
            }
            result.push_back(std::move(value.value()));
        } else {
            auto key = parseKey(ctx, ",");
            if (!key.has_value()) {
                return ReturnType{std::unexpected(key.error())};
            }
            ctx.consumeWhitespaceAndComments();
            auto value = parseElementValue();
            if (!value.has_value()) {
                return ReturnType{std::unexpected(value.error())};
            }
            result[std::move(key.value())] = std::move(value.value());
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
    if (ctx.consumeIfStartsWith("{")) {
        auto map = parseFlow<FlowType::Map>(ctx, "", parentIndentLevel);
        ctx.skipToNextLine();
        return map;
    }

    pmtv::map_t map;

    while (!ctx.atEndOfDocument()) {
        ctx.consumeSpaces();
        if (ctx.atEndOfLine() || ctx.startsWith("#")) {
            // skip empty lines and comments
            ctx.skipToNextLine();
            continue;
        }

        const auto line_indent = ctx.currentIndent();

        if (parentIndentLevel >= 0 && line_indent <= static_cast<std::size_t>(parentIndentLevel)) {
            // indentation decreased; end of current map
            break;
        }

        const auto maybeKey = parseKey(ctx);
        if (!maybeKey.has_value()) {
            return std::unexpected(maybeKey.error());
        }

        auto key = maybeKey.value();

        ctx.consumeSpaces();
        const auto maybeTag = parseTag(ctx);
        if (!maybeTag.has_value()) {
            return std::unexpected(maybeTag.error());
        }
        auto typeTag = maybeTag.value();
        ctx.consumeSpaces();

        const auto peekedType = peekToFindValueType(ctx, static_cast<int>(line_indent));

        switch (peekedType) {
        case ValueType::List: {
            auto parsedValue = parseList(ctx, typeTag, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            map[key] = parsedValue.value();
            break;
        }
        case ValueType::Map: {
            if (!typeTag.empty()) {
                return std::unexpected(ctx.makeError("Cannot have type tag for map entry"));
            }
            auto parsedValue = parseMap(ctx, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            map[key] = parsedValue.value();
            break;
        }
        case ValueType::Scalar: {
            auto parsedValue = parseScalar(ctx, typeTag, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            map[key] = parsedValue.value();
            break;
        }
        }
    }
    return map;
}

inline std::expected<pmtv::pmt, ParseError> parseList(ParseContext& ctx, std::string_view typeTag, int parentIndentLevel) {
    if (ctx.consumeIfStartsWith("[")) {
        auto l = parseFlow<FlowType::List>(ctx, typeTag, parentIndentLevel);
        ctx.skipToNextLine();
        return l;
    }

    std::vector<pmtv::pmt> list;

    ctx.skipToNextLine();

    while (!ctx.atEndOfDocument()) {
        ctx.consumeSpaces();
        if (ctx.atEndOfLine() || ctx.startsWith("#")) {
            // skip empty lines and comments
            ctx.skipToNextLine();
            continue;
        }

        const std::size_t line_indent = ctx.currentIndent();
        if (parentIndentLevel >= 0 && line_indent <= static_cast<size_t>(parentIndentLevel)) {
            // indentation decreased; end of current list
            break;
        }

        ctx.consumeSpaces();

        if (!ctx.consumeIfStartsWith('-')) {
            // not a list item
            return std::unexpected(ctx.makeError("Expected list item"));
        }

        ctx.consumeSpaces();

        const auto maybeLocalTag = parseTag(ctx);
        if (!maybeLocalTag.has_value()) {
            return std::unexpected(maybeLocalTag.error());
        }
        auto localTag = maybeLocalTag.value();
        if (!typeTag.empty() && !localTag.empty()) {
            return std::unexpected(ctx.makeError("Cannot have type tag for both list and list item"));
        }

        const auto tag = !typeTag.empty() ? typeTag : localTag;

        ctx.consumeSpaces();

        const auto peekedType = peekToFindValueType(ctx, static_cast<int>(line_indent));
        switch (peekedType) {
        case ValueType::List: {
            if (!typeTag.empty()) {
                return std::unexpected(ctx.makeError("Cannot have type tag for list containing lists"));
            }
            auto parsedValue = parseList(ctx, tag, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            list.push_back(parsedValue.value());
            break;
        }
        case ValueType::Map: {
            if (!typeTag.empty()) {
                return std::unexpected(ctx.makeError("Cannot have type tag for maps"));
            }
            auto parsedValue = parseMap(ctx, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            list.push_back(parsedValue.value());
            break;
        }
        case ValueType::Scalar: {
            auto parsedValue = parseScalar(ctx, tag, static_cast<int>(line_indent));
            if (!parsedValue.has_value()) {
                return std::unexpected(parsedValue.error());
            }
            list.push_back(parsedValue.value());
            break;
        }
        }
    }

    if (typeTag.empty()) {
        return list;
    }
    // TODO maybe avoid the conversion from pmtv::pmt back type_tag's T and make this whole function a template,
    // but check for code size increase
    return applyTag<pmtv::pmt, ConvertList>(typeTag, list);
}

} // namespace detail

inline std::string serialize(const pmtv::map_t& map) {
    std::ostringstream oss;
    if (!map.empty()) {
        detail::serialize<detail::TypeTagMode::Auto>(oss, map, -1); // Start at level -1 to avoid indenting top-level keys
    }
    return oss.str();
}

inline std::expected<pmtv::map_t, ParseError> deserialize(std::string_view yaml_str) {
    auto                 lines = detail::split(yaml_str, "\n");
    detail::ParseContext ctx{.lines = lines};
    ctx.consumeIfStartsWith("---");
    return detail::parseMap(ctx, -1);
}

} // namespace pmtv::yaml
