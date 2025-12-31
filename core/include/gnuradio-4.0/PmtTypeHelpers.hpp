#ifndef PMTTYPEHELPERS_HPP
#define PMTTYPEHELPERS_HPP

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/ValueHelper.hpp>

#ifdef __GNUC__
// ignore warning from external libraries we don't control
#pragma GCC diagnostic push
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace gr::pmt {

namespace detail {

template<typename T>
struct is_complex : std::false_type {};

template<typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template<typename T>
concept VariantLike = requires(T v) {
    { std::variant_size_v<T> } -> std::convertible_to<std::size_t>;
    {
        std::visit([](auto&&) {}, v)
    };
};

template<class, class>
struct variant_contains : std::false_type {};

template<class T, class... Ts>
struct variant_contains<std::variant<Ts...>, T> : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

template<class Variant, class T>
inline constexpr bool variant_contains_v = variant_contains<Variant, T>::value;

template<typename T>
[[nodiscard]] constexpr bool inRange(std::int64_t val);

template<std::signed_integral T>
[[nodiscard]] constexpr bool inRange(std::int64_t val) {
    return val >= static_cast<std::int64_t>(std::numeric_limits<T>::min()) && val <= static_cast<std::int64_t>(std::numeric_limits<T>::max());
}
template<std::unsigned_integral T>
[[nodiscard]] constexpr bool inRange(std::int64_t val) {
    return val < 0 ? false : static_cast<unsigned long long>(val) <= static_cast<std::uint64_t>(std::numeric_limits<T>::max());
}

constexpr std::string_view trimAndCutComment(std::string_view input) {
    constexpr auto is_space = [](unsigned char ch) { return std::isspace(ch); };

    // remove leading whitespace
    input.remove_prefix(static_cast<std::size_t>(std::ranges::find_if_not(input, is_space) - input.begin()));

    // cut off at `#` if present
    if (auto pos = input.find('#'); pos != std::string_view::npos) {
        input = input.substr(0, pos);
    }

    // remove trailing whitespace
    input.remove_suffix(static_cast<std::size_t>(input.end() - std::ranges::find_if_not(input | std::views::reverse, is_space).base()));

    return input;
}

template<typename T>
requires(std::is_floating_point_v<T>)
static std::expected<T, std::string> parseStringToFloat(std::string_view trimmed) {
    using namespace std::string_literals;
#if defined(__clang__)
    // Fallback to std::strtof / strtod for Clang versions prior to 20
    if constexpr (std::is_same_v<T, float>) {
        char* endPtr = nullptr;
        float valF   = std::strtof(trimmed.data(), &endPtr);
        if (endPtr == trimmed.data() + trimmed.size() && !std::isinf(valF)) {
            return valF;
        }
        if (std::isinf(valF)) {
            return std::unexpected("float parse out-of-range"s);
        }
        return std::unexpected("invalid float parse"s);
    } else {
        // double
        char*  endPtr = nullptr;
        double valD   = std::strtod(trimmed.data(), &endPtr);
        if (endPtr == trimmed.data() + trimmed.size() && !std::isinf(valD)) {
            return static_cast<T>(valD);
        }
        if (std::isinf(valD)) {
            return std::unexpected("double parse out-of-range"s);
        }
        return std::unexpected("invalid double parse"s);
    }
#else
    // Use std::from_chars for supported compilers
    T parsedVal{};
    auto [p, errorCode] = std::from_chars(trimmed.data(), trimmed.data() + trimmed.size(), parsedVal, std::chars_format::general);
    if (errorCode == std::errc() && p == trimmed.data() + trimmed.size()) {
        return parsedVal;
    }
    if (errorCode == std::errc::result_out_of_range) {
        return std::unexpected("floating-point out-of-range");
    }
    return std::unexpected("invalid floating-point parse");
#endif
}

} // namespace detail

template<class T, bool strictCheck = false, typename From>
[[nodiscard]] constexpr std::expected<T, std::string> convert_safely(const From& srcValue) {
    using namespace std::string_literals;
    using S = std::decay_t<decltype(srcValue)>;

    // 1) same type => trivial
    if constexpr (std::is_same_v<S, T>) {
        return srcValue;
    }

    if constexpr (strictCheck) {
        return std::unexpected(std::format("strict-check enabled: source {} and target {} type do not match", typeid(S).name(), typeid(T).name()));
    }

    // 2) source is bool (special case, needed to be treated first because of overlapping with integral)
    else if constexpr (std::is_same_v<S, bool>) {
        if constexpr (std::is_same_v<T, std::string>) {
            return srcValue ? "true" : "false";
        } else if constexpr (std::is_integral_v<T>) {
            return std::unexpected("unsupported bool to integer conversion"s);
        } else if constexpr (std::is_floating_point_v<T>) {
            return std::unexpected("unsupported bool to floating-point conversion"s);
        }
    }

    // 3) source is integral
    else if constexpr (std::is_integral_v<S>) {
        // 3a) target is integral
        if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_unsigned_v<T> && std::is_signed_v<S>) {
                if (srcValue < 0) {
                    return std::unexpected(std::format("negative integer {} cannot be converted to unsigned type", srcValue));
                }
            }
            constexpr auto digitsS = std::numeric_limits<S>::digits;
            constexpr auto digitsT = std::numeric_limits<T>::digits;
            if constexpr (digitsS <= digitsT) {
                return static_cast<T>(srcValue); // always safe
            } else {                             // range check
                if (static_cast<std::int64_t>(srcValue) >= std::numeric_limits<T>::lowest() && static_cast<std::int64_t>(srcValue) <= std::numeric_limits<T>::max()) {
                    return static_cast<T>(srcValue);
                }
                return std::unexpected(std::format("out-of-range integer conversion: {} not in [{}..{}]", srcValue, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()));
            }
        }
        // 3b) target is floating‐point
        else if constexpr (std::is_floating_point_v<T>) {
            // integer->float
            constexpr auto digitsS = std::numeric_limits<S>::digits;
            constexpr auto digitsT = std::numeric_limits<T>::digits;
            if constexpr (digitsS <= digitsT) {
                return static_cast<T>(srcValue);
            } else {
                using WideType     = std::conditional_t<(sizeof(S) < sizeof(long long)), long long, S>;
                const auto wideVal = static_cast<WideType>(srcValue);
                if (wideVal == std::numeric_limits<WideType>::min()) {
                    return std::unexpected(std::format("cannot handle integer min()={} when checking bit width", wideVal));
                }
                const WideType magnitude = (wideVal < 0) ? -wideVal : wideVal;
                const auto     bitWidth  = std::bit_width(static_cast<std::make_unsigned_t<WideType>>(magnitude));
                if (bitWidth <= digitsT) {
                    return static_cast<T>(srcValue);
                }
                return std::unexpected(std::format("integer bit_width({})={} > floating-point mantissa={}", srcValue, bitWidth, digitsT));
            }
        }
        // 3c) target is std::complex<T>
        else if constexpr (detail::is_complex_v<T>) { // value becomes real-part
            auto conv = convert_safely<typename T::value_type>(srcValue);
            if (conv.has_value()) {
                return T{conv.value()};
            }
            return std::unexpected(std::format("unsupported complex conversion {}", conv.error()));
        }
        // 3d) no match
        else {
            return std::unexpected(std::format("no valid safe conversion for <integral {}> -> <{}>", typeid(S).name(), typeid(T).name()));
        }
    }

    // 4) source is floating-point
    else if constexpr (std::is_floating_point_v<S>) {
        // 4a) target is integral
        if constexpr (std::is_integral_v<T>) {
            if (!std::isfinite(srcValue)) {
                return std::unexpected(std::format("floating-point value {} is not finite (NaN/inf)", srcValue));
            }
            if (srcValue < static_cast<S>(std::numeric_limits<T>::lowest()) || srcValue > static_cast<S>(std::numeric_limits<T>::max())) {
                return std::unexpected(std::format("floating-point value {} is outside [{}, {}]", srcValue, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()));
            }
            if (srcValue != std::nearbyint(srcValue)) {
                return std::unexpected(std::format("floating-point value {} is not an integer", srcValue));
            }
            return static_cast<T>(srcValue);
        }
        // 4b) target is floating‐point
        else if constexpr (std::is_floating_point_v<T>) {
            if (static_cast<long double>(srcValue) >= static_cast<long double>(std::numeric_limits<T>::lowest()) && static_cast<long double>(srcValue) <= static_cast<long double>(std::numeric_limits<T>::max())) {
                return static_cast<T>(srcValue);
            }
            return std::unexpected(std::format("floating-point conversion from {} is outside [{}, {}]", srcValue, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()));
        }
        // 4c) target is std::complex<T>
        else if constexpr (detail::is_complex_v<T>) { // value becomes real-part
            auto conv = convert_safely<typename T::value_type>(srcValue);
            if (conv.has_value()) {
                return T{conv.value()};
            }
            return std::unexpected(std::format("unsupported complex conversion {}", conv.error()));
        }
        // 4d) no match
        else {
            return std::unexpected(std::format("no valid safe conversion for <float {}> src = {} -> <{}>", typeid(S).name(), srcValue, typeid(T).name()));
        }
    }

    // 5) source is complex
    else if constexpr (detail::is_complex_v<S>) {
        // 5a) complex->complex
        if constexpr (detail::is_complex_v<T>) {
            using RealS = typename S::value_type;
            using RealT = typename T::value_type;
            if constexpr (std::is_same_v<S, T>) {
                return srcValue; // trivial
            } else {
                // Must convert real/imag individually
                auto realPart = convert_safely<RealT>(std::variant<RealS>{srcValue.real()});
                if (!realPart) {
                    return std::unexpected(realPart.error());
                }
                auto imagPart = convert_safely<RealT>(std::variant<RealS>{srcValue.imag()});
                if (!imagPart) {
                    return std::unexpected(imagPart.error());
                }
                return T{*realPart, *imagPart};
            }
        }
        // 5b) complex->integral or float (only for real-valued complex)
        else if constexpr (std::is_arithmetic_v<T>) {
            if (std::imag(srcValue) != 0) {
                return std::unexpected(std::format("cannot convert non-real-valued std:complex<{}> src= {} +{}i -> <{}>", //
                    typeid(T).name(), std::real(srcValue), std::imag(srcValue), typeid(T).name()));
            }
            auto conv = convert_safely<T>(std::variant<typename S::value_type>{srcValue.real()});
            if (conv.has_value()) {
                return conv.value();
            }
            return std::unexpected(std::format("cannot convert non-real-valued std:complex<{}> src= {} +{}i -> <{}> reason: {}", //
                typeid(T).name(), std::real(srcValue), std::imag(srcValue), typeid(T).name(), conv.error()));
        }

        return std::unexpected(std::format("no valid safe conversion for std:complex<{}> src= {} +{}i -> <{}>", //
            typeid(T).name(), std::real(srcValue), std::imag(srcValue), typeid(T).name()));
    }

    // 6) source is string
    else if constexpr (std::is_same_v<S, std::string>) {
        // 6a) string->enum
        if constexpr (std::is_enum_v<T>) {
            if (auto maybeEnum = magic_enum::enum_cast<T>(srcValue)) {
                return *maybeEnum;
            }
            auto possibleEnumValues = []<typename U>(const U&) -> std::string {
                constexpr auto vals  = magic_enum::enum_values<U>();
                auto           names = vals | std::views::transform(magic_enum::enum_name<U>);
                return std::format("{}", gr::join(names, ", "));
            };
            return std::unexpected(std::format("'{}' is not a valid enum '{}' value: [{}]", srcValue, magic_enum::enum_type_name<T>(), possibleEnumValues(T{})));
        }
        // 6b) string->integral (excluding bool)
        else if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            const auto trimmed = detail::trimAndCutComment(srcValue);
            T          parsedVal{};
            auto [p, errorCode] = std::from_chars(trimmed.data(), trimmed.data() + trimmed.size(), parsedVal, 10);
            if (errorCode == std::errc::invalid_argument || p != trimmed.data() + trimmed.size()) {
                return std::unexpected(std::format("cannot parse '{}' as an integer", srcValue));
            }
            if (errorCode == std::errc::result_out_of_range) {
                return std::unexpected(std::format("integer out-of-range: '{}'", srcValue));
            }
            return parsedVal;
        }
        // 6c) string->float
        else if constexpr (std::is_floating_point_v<T>) {
            const auto trimmed = detail::trimAndCutComment(srcValue);
            auto       result  = detail::parseStringToFloat<T>(trimmed);
            if (!result) {
                return std::unexpected(std::format("cannot parse '{}' as floating-point: {}", srcValue, result.error()));
            }
            return *result;
        }
        // 6d) string->bool
        else if constexpr (std::is_same_v<T, bool>) {
            std::string s = srcValue;
            std::ranges::transform(s, s.begin(), [](unsigned char c) { return std::tolower(c); });
            if (s == "true" || s == "1") {
                return true;
            } else if (s == "false" || s == "0") {
                return false;
            }
            return std::unexpected(std::format("cannot parse '{}' as bool", srcValue));
        }
        // fallback
        else {
            return std::unexpected(std::format("no safe conversion for std::string src = {} -> <{}>", srcValue, typeid(T).name()));
        }
    }

    // 7) source is enum
    else if constexpr (std::is_enum_v<S>) {
        if constexpr (std::is_same_v<T, std::string>) {
            return std::string(magic_enum::enum_name(srcValue));
        }
        return std::unexpected(std::format("no safe conversion for {} src = {} -> <{}>", magic_enum::enum_type_name<S>(), magic_enum::enum_name(srcValue), typeid(T).name()));
    }

    // fallback
    return std::unexpected(std::format("no safe conversion for <{}> -> <{}>", typeid(S).name(), typeid(T).name()));
}

template<class T, bool strictCheck = false>
[[nodiscard]] constexpr std::expected<T, std::string> convert_safely(const pmt::Value& v) {
    std::expected<T, std::string> result;
    ValueVisitor([&result](const auto& from) { result = convert_safely<T, strictCheck>(from); }).visit(v);
    return result;
}

template<typename TMinimalNumericVariant, typename R = std::expected<TMinimalNumericVariant, std::string>>
[[nodiscard]] constexpr R parseToMinimalNumeric(std::string_view numericString) {
    const std::string_view str = detail::trimAndCutComment(numericString);
    if (str.empty()) {
        return std::unexpected(std::format("empty or comment-only string: '{}'", numericString));
    }

    const bool mightBeFloat = (str.find('.') != std::string_view::npos) || (str.find('e') != std::string_view::npos) || (str.find('E') != std::string_view::npos);

    if (!mightBeFloat) { // attempt to parse as signed or unsigned integral
        const bool isNegative = (!str.empty() && str.front() == '-');

        if (isNegative) { // needs to be 64-bit signed integer
            std::int64_t val{};

            const auto [p, errorCode] = std::from_chars(str.data(), str.data() + str.size(), val, 10);
            if (errorCode == std::errc::invalid_argument || p != str.data() + str.size()) {
                return std::unexpected(std::format("invalid integer - cannot parse '{}'", str));
            } else if (errorCode == std::errc::result_out_of_range) {
                return std::unexpected(std::format("out-of-range for signed 64-bit - cannot parse '{}'", str));
            }

            // check smallest signed-integer type that can hold `val`
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int8_t>) {
                if (detail::inRange<std::int8_t>(val)) {
                    return TMinimalNumericVariant(static_cast<std::int8_t>(val));
                }
            }
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int16_t>) {
                if (detail::inRange<std::int16_t>(val)) {
                    return TMinimalNumericVariant(static_cast<std::int16_t>(val));
                }
            }
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int32_t>) {
                if (detail::inRange<std::int32_t>(val)) {
                    return TMinimalNumericVariant(static_cast<std::int32_t>(val));
                }
            }
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int64_t>) {
                return TMinimalNumericVariant(val);
            }

            return std::unexpected(std::format("no suitable integral type available (negative) for: {}", val));
        } else { // needs to be 64-bit unsigned integer
            std::uint64_t valU{};
            auto [p, errorCode] = std::from_chars(str.data(), str.data() + str.size(), valU, 10);
            if (errorCode == std::errc::invalid_argument || p != str.data() + str.size()) {
                return std::unexpected(std::format("invalid integer - cannot parse '{}'", str));
            } else if (errorCode == std::errc::result_out_of_range) {
                return std::unexpected(std::format("out-of-range for unsigned 64-bit - cannot parse '{}'", str));
            }

            // check smallest integer type that can hold `val` with preference to signed-types
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int8_t>) {
                if (valU <= static_cast<unsigned long long>(std::numeric_limits<std::int8_t>::max())) {
                    return TMinimalNumericVariant(static_cast<std::int8_t>(valU));
                }
            }
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::uint8_t>) {
                if (valU <= std::numeric_limits<std::uint8_t>::max()) {
                    return TMinimalNumericVariant(static_cast<std::uint8_t>(valU));
                }
            }
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int16_t>) {
                if (valU <= static_cast<unsigned long long>(std::numeric_limits<std::int16_t>::max())) {
                    return TMinimalNumericVariant(static_cast<std::int16_t>(valU));
                }
            }
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::uint16_t>) {
                if (valU <= std::numeric_limits<std::uint16_t>::max()) {
                    return TMinimalNumericVariant(static_cast<std::uint16_t>(valU));
                }
            }
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int32_t>) {
                if (valU <= static_cast<unsigned long long>(std::numeric_limits<std::int32_t>::max())) {
                    return TMinimalNumericVariant(static_cast<std::int32_t>(valU));
                }
            }
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::uint32_t>) {
                if (valU <= std::numeric_limits<std::uint32_t>::max()) {
                    return TMinimalNumericVariant(static_cast<std::uint32_t>(valU));
                }
            }
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int64_t>) {
                if (valU <= static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max())) {
                    return TMinimalNumericVariant(static_cast<std::int64_t>(valU));
                }
            }
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::uint64_t>) {
                if (valU <= std::numeric_limits<std::uint64_t>::max()) {
                    return TMinimalNumericVariant(static_cast<std::uint64_t>(valU));
                }
            }

            return std::unexpected(std::format("no suitable integral type available (non-negative) for: {}", valU));
        }
    } else { // mightBeFloat - prefer float if in the variant, fallback to double
        if constexpr (detail::variant_contains_v<TMinimalNumericVariant, float>) {
            if (std::expected<float, std::string> tryFloat = convert_safely<float>(std::variant<std::string>{std::string(str)}); tryFloat) { // successfully parsed as float
                return TMinimalNumericVariant(*tryFloat);
            } else { // attempt conversion to double
                if constexpr (detail::variant_contains_v<TMinimalNumericVariant, double>) {
                    if (std::expected<double, std::string> tryDouble = convert_safely<double>(std::variant<std::string>{std::string(str)}); tryDouble) {
                        return TMinimalNumericVariant(*tryDouble);
                    } else {
                        return std::unexpected(tryDouble.error());
                    }
                }
                // no double in variant => fail with float's error
                return std::unexpected(tryFloat.error());
            }
        } else if constexpr (detail::variant_contains_v<TMinimalNumericVariant, double>) {
            // we don't have float, but we do have double in the variant
            std::expected<double, std::string> tryDouble = convert_safely<double>(std::variant<std::string>{std::string(str)});
            if (tryDouble) {
                return TMinimalNumericVariant(*tryDouble);
            }
            return std::unexpected(tryDouble.error());
        } else {
            // no float or double in the variant => cannot parse a float number
            return std::unexpected("no float/double in the variant => cannot parse");
        }
    }
}

// forward-declare helper functions -> implemented in the corresponding .cpp file.

// ---- fundamental types ----
extern template std::expected<bool, std::string>                 convert_safely<bool, false, pmt>(const pmt&);
extern template std::expected<std::int8_t, std::string>          convert_safely<std::int8_t, false, pmt>(const pmt&);
extern template std::expected<std::uint8_t, std::string>         convert_safely<std::uint8_t, false, pmt>(const pmt&);
extern template std::expected<std::int16_t, std::string>         convert_safely<std::int16_t, false, pmt>(const pmt&);
extern template std::expected<std::uint16_t, std::string>        convert_safely<std::uint16_t, false, pmt>(const pmt&);
extern template std::expected<std::int32_t, std::string>         convert_safely<std::int32_t, false, pmt>(const pmt&);
extern template std::expected<std::uint32_t, std::string>        convert_safely<std::uint32_t, false, pmt>(const pmt&);
extern template std::expected<std::int64_t, std::string>         convert_safely<std::int64_t, false, pmt>(const pmt&);
extern template std::expected<std::uint64_t, std::string>        convert_safely<std::uint64_t, false, pmt>(const pmt&);
extern template std::expected<float, std::string>                convert_safely<float, false, pmt>(const pmt&);
extern template std::expected<double, std::string>               convert_safely<double, false, pmt>(const pmt&);
extern template std::expected<std::complex<float>, std::string>  convert_safely<std::complex<float>, false, pmt>(const pmt&);
extern template std::expected<std::complex<double>, std::string> convert_safely<std::complex<double>, false, pmt>(const pmt&);
extern template std::expected<std::string, std::string>          convert_safely<std::string, false, pmt>(const pmt&);

// ---- vector-of-fundamentals ----
extern template std::expected<std::vector<bool>, std::string>                 convert_safely<std::vector<bool>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::int8_t>, std::string>          convert_safely<std::vector<std::int8_t>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::uint8_t>, std::string>         convert_safely<std::vector<std::uint8_t>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::int16_t>, std::string>         convert_safely<std::vector<std::int16_t>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::uint16_t>, std::string>        convert_safely<std::vector<std::uint16_t>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::int32_t>, std::string>         convert_safely<std::vector<std::int32_t>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::uint32_t>, std::string>        convert_safely<std::vector<std::uint32_t>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::int64_t>, std::string>         convert_safely<std::vector<std::int64_t>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::uint64_t>, std::string>        convert_safely<std::vector<std::uint64_t>, false, pmt>(const pmt&);
extern template std::expected<std::vector<float>, std::string>                convert_safely<std::vector<float>, false, pmt>(const pmt&);
extern template std::expected<std::vector<double>, std::string>               convert_safely<std::vector<double>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::complex<float>>, std::string>  convert_safely<std::vector<std::complex<float>>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::complex<double>>, std::string> convert_safely<std::vector<std::complex<double>>, false, pmt>(const pmt&);
extern template std::expected<std::vector<std::string>, std::string>          convert_safely<std::vector<std::string>, false, pmt>(const pmt&);
extern template std::expected<map_t, std::string>                             convert_safely<map_t, false, pmt>(const pmt&);

} // namespace pmtv

#endif // PMTTYPEHELPERS_HPP
