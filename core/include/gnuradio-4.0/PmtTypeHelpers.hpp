#ifndef PMTTYPEHELPERS_HPP
#define PMTTYPEHELPERS_HPP

#include <bit>
#include <charconv>
#include <cmath>
#include <expected>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <variant>

#include <fmt/format.h>

#ifdef __GNUC__
#pragma GCC diagnostic push // ignore warning of external libraries that from this lib-context we do not have any control over
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

namespace pmtv {

namespace detail {

template<typename T>
struct is_complex : std::false_type {};

template<typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template<class Variant, class T>
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
    constexpr auto is_space = [](const unsigned char ch) { return std::isspace(ch); };

    input.remove_prefix(static_cast<std::size_t>(std::ranges::find_if_not(input, is_space) - input.begin())); // remove leading whitespace
    if (auto pos = input.find('#'); pos != std::string_view::npos) {                                          // cut off at `#` if present
        input = input.substr(0, pos);
    }
    input.remove_suffix(static_cast<std::size_t>(input.end() - std::ranges::find_if_not(input | std::views::reverse, is_space).base())); // remove trailing whitespace

    return input;
}
} // namespace detail

template<class T, class... Ts>
std::expected<T, std::string> safely_convert(const std::variant<Ts...>& v) {
    return std::visit(
        [&](auto&& srcValue) -> std::expected<T, std::string> {
            using S = std::decay_t<decltype(srcValue)>;

            if constexpr (std::is_same_v<S, T>) { // same type => trivially OK
                return srcValue;
            } else if constexpr (std::is_integral_v<S> && std::is_integral_v<T>) { // integral -> integral
                if constexpr (std::is_unsigned_v<T> && std::is_signed_v<S>) {
                    if (srcValue < 0) {
                        return std::unexpected(fmt::format("negative integer {} cannot be converted to unsigned type", srcValue));
                    }
                }
                constexpr auto digitsS = std::numeric_limits<S>::digits;
                constexpr auto digitsT = std::numeric_limits<T>::digits;

                if constexpr (digitsS <= digitsT) { // fewer or equal bits => always safe
                    return static_cast<T>(srcValue);
                } else { // same or more bits => range check
                    if (static_cast<std::int64_t>(srcValue) >= std::numeric_limits<T>::lowest() && static_cast<std::int64_t>(srcValue) <= std::numeric_limits<T>::max()) {
                        return static_cast<T>(srcValue);
                    }
                    return std::unexpected(fmt::format("out-of-range integer conversion: {} not in [{}..{}]", //
                        srcValue, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()));
                }
            } else if constexpr (std::is_floating_point_v<S> && std::is_floating_point_v<T>) { // floating-point -> floating-point (mixed precision)
                if (static_cast<double>(srcValue) >= static_cast<double>(std::numeric_limits<T>::lowest()) && static_cast<double>(srcValue) <= static_cast<double>(std::numeric_limits<T>::max())) {
                    return static_cast<T>(srcValue);
                }
                return std::unexpected(fmt::format("floating-point conversion from {} is outside [{}, {}]", //
                    srcValue, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()));

            } else if constexpr (std::is_integral_v<S> && std::is_floating_point_v<T>) { // integer -> floating-point
                constexpr auto digitsS = std::numeric_limits<S>::digits;
                constexpr auto digitsT = std::numeric_limits<T>::digits;

                if constexpr (digitsS <= digitsT) { // integer has <= bits w.r.t. target mantissa
                    return static_cast<T>(srcValue);
                } else { // check bit width vs mantissa
                    using WideType = std::conditional_t<(sizeof(S) < sizeof(long long)), long long, S>;

                    const WideType wideVal = static_cast<WideType>(srcValue);
                    // Handle the corner case of min() if negative
                    if (wideVal == std::numeric_limits<WideType>::min()) {
                        return std::unexpected(fmt::format("cannot handle integer min()={} when checking bit width", wideVal));
                    }
                    const auto        magnitude = (wideVal < 0) ? -wideVal : wideVal;
                    const std::size_t bitWidth  = static_cast<std::size_t>(std::bit_width(static_cast<std::make_unsigned_t<WideType>>(magnitude)));
                    if (bitWidth <= digitsT) {
                        return static_cast<T>(srcValue);
                    }
                    return std::unexpected(fmt::format("integer bit_width({})={} > floating-point mantissa={}", srcValue, bitWidth, digitsT));
                }
            } else if constexpr (std::is_floating_point_v<S> && std::is_integral_v<T>) { // floating-point -> integer
                if (!std::isfinite(srcValue)) {
                    return std::unexpected(fmt::format("floating-point value {} is not finite (NaN/inf) -> no integer representation", srcValue));
                }
                if (srcValue < static_cast<S>(std::numeric_limits<T>::lowest()) || srcValue > static_cast<S>(std::numeric_limits<T>::max())) {
                    return std::unexpected(fmt::format("[floating-point value {} is outside [{}, {}]", srcValue, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()));
                }
                if (srcValue != std::nearbyint(srcValue)) { // check whether conversion is exact
                    return std::unexpected(fmt::format("floating-point value {} is not an integer", srcValue));
                }
                return static_cast<T>(srcValue);
            } else if constexpr (std::is_enum_v<S> && std::is_same_v<T, std::string>) { // enum -> string
                return std::string(magic_enum::enum_name(srcValue));
            } else if constexpr (std::is_same_v<S, std::string> && std::is_enum_v<T>) { // string -> enum
                const auto maybeEnum = magic_enum::enum_cast<T>(srcValue);
                if (maybeEnum.has_value()) {
                    return maybeEnum.value();
                }
                auto enumValuesToString = []<typename U>(const U&) -> std::string {
                    static_assert(std::is_enum_v<U>, "T must be an enumeration type");
                    constexpr auto values = magic_enum::enum_values<U>();

                    auto names = values | std::views::transform(magic_enum::enum_name<U>);
                    return fmt::format("{}", fmt::join(names, ", "));
                };

                return std::unexpected(fmt::format("'{}' is not a valid enum '{}' value: [{}]", srcValue, magic_enum::enum_type_name<T>(), enumValuesToString(T{})));
            }

            // N.B. the following conversions are a bit opinionated and may have secondary pitfalls in their runtime usage
            else if constexpr (std::is_same_v<S, std::string> && std::is_integral_v<T> && not std::is_same_v<T, bool>) { // string -> integral
                const std::string_view trimmed = detail::trimAndCutComment(srcValue);

                T parsedVal{};
                auto [p, errorCode] = std::from_chars(trimmed.data(), trimmed.data() + trimmed.size(), parsedVal, 10 /* base=10 by default */);
                if (errorCode == std::errc::invalid_argument || p != trimmed.data() + trimmed.size()) {
                    return std::unexpected(fmt::format("cannot parse '{}' as an integer", srcValue));
                }
                if (errorCode == std::errc::result_out_of_range) {
                    return std::unexpected(fmt::format("integer out-of-range: '{}'", srcValue));
                }
                return parsedVal;
            }

            else if constexpr (std::is_same_v<S, std::string> && std::is_floating_point_v<T>) { // string -> floating-point
                const std::string_view trimmed = detail::trimAndCutComment(srcValue);

                T parsedVal{};
                auto [p, errorCode] = std::from_chars(trimmed.data(), trimmed.data() + trimmed.size(), parsedVal, std::chars_format::general);
                if (errorCode == std::errc::invalid_argument || p != trimmed.data() + trimmed.size()) {
                    return std::unexpected(fmt::format("cannot parse '{}' as a floating-point", srcValue));
                } else if (errorCode == std::errc::result_out_of_range) {
                    return std::unexpected(fmt::format("floating-point out-of-range: '{}'", srcValue));
                }
                return parsedVal;
            }

            else if constexpr (std::is_same_v<S, std::string> && std::is_same_v<T, bool>) { // string -> bool
                std::string s = srcValue;
                // transform to lowercase to accept "True", "TRUE", etc.
                std::ranges::transform(s, s.begin(), [](unsigned char c) { return std::tolower(c); });

                if (s == "true" || s == "1") {
                    return true;
                } else if (s == "false" || s == "0") {
                    return false;
                }

                return std::unexpected(fmt::format("cannot parse '{}' as bool", srcValue));
            } else if constexpr (std::is_same_v<S, bool> && std::is_same_v<T, std::string>) { // bool -> string
                return srcValue ? "true" : "false";
            }

            // std::complex<..> conversions
            else if constexpr (detail::is_complex_v<S> && detail::is_complex_v<T>) {
                // complex<float> <-> complex<double>
                // (no other complex combos)
                using RealS = typename S::value_type;
                using RealT = typename T::value_type;
                if constexpr (std::is_same_v<S, T>) {
                    // same type => trivial
                    return srcValue;
                } else {
                    // must convert real/imag individually
                    auto realPart = safely_convert<RealT>(std::variant<RealS>{srcValue.real()});
                    if (!realPart.has_value()) {
                        return std::unexpected(realPart.error());
                    }
                    auto imagPart = safely_convert<RealT>(std::variant<RealS>{srcValue.imag()});
                    if (!imagPart.has_value()) {
                        return std::unexpected(imagPart.error());
                    }
                    return T{*realPart, *imagPart};
                }
            } else if constexpr (!std::is_same_v<S, bool> && std::is_arithmetic_v<S> && detail::is_complex_v<T>) {
                // scalar -> complex<float/double>, but exclude bool
                // set imag = 0
                using RealT  = typename T::value_type;
                auto realVal = safely_convert<RealT>(std::variant<S>{srcValue});
                if (!realVal.has_value()) {
                    return std::unexpected(realVal.error());
                }
                return T(*realVal, RealT{0});
            }

            // fallback
            else {
                return std::unexpected(fmt::format("No valid rule for <{}> -> <{}>", typeid(S).name(), typeid(T).name()));
            }
        },
        v);
}

template<typename TMinimalNumericVariant, typename R = std::expected<TMinimalNumericVariant, std::string>>
constexpr R parseToMinimalNumeric(std::string_view numericString) {
    // 1) trim/comment-check
    auto str = detail::trimAndCutComment(numericString);
    if (str.empty()) {
        return std::unexpected(fmt::format("empty or comment-only string: '{}'", numericString));
    }

    // 2) float vs integer heuristic
    const bool mightBeFloat = (str.find('.') != std::string_view::npos) || (str.find('e') != std::string_view::npos) || (str.find('E') != std::string_view::npos);

    if (!mightBeFloat) {
        const bool isNegative = (!str.empty() && str.front() == '-');

        if (isNegative) {
            std::int64_t val{}; // parse as signed std::int64_t
            {
                auto [p, errorCode] = std::from_chars(str.data(), str.data() + str.size(), val, 10);
                if (errorCode == std::errc::invalid_argument || p != str.data() + str.size()) {
                    return std::unexpected("invalid integer parse");
                } else if (errorCode == std::errc::result_out_of_range) {
                    return std::unexpected("integer parse out-of-range for signed 64-bit");
                }
            }

            // We only check signed types if negative
            // check int8, int16, int32, int64 in ascending order
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
                // definitely fits if we're here
                return TMinimalNumericVariant(static_cast<std::int64_t>(val));
            }
            return std::unexpected("no suitable integral type available (negative)");
        } else {
            // parse as unsigned long long
            unsigned long long valU{};
            {
                auto [p, errorCode] = std::from_chars(str.data(), str.data() + str.size(), valU, 10);
                if (errorCode == std::errc::invalid_argument || p != str.data() + str.size()) {
                    return std::unexpected("invalid integer parse");
                } else if (errorCode == std::errc::result_out_of_range) {
                    return std::unexpected("integer parse out-of-range for unsigned 64-bit");
                }
            }

            // Now we have valU. We'll check in ascending order of size: int8_t, uint8_t, int16_t, etc.
            // But only if those types are in the variant.

            // If valU > INT64_MAX, none of the signed types can fit it. So we skip them if it definitely can't fit.
            // We'll still check smaller unsigned types first (like uint8_t, uint16_t, etc.).

            // check int8_t
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int8_t>) {
                if (valU <= static_cast<unsigned long long>(std::numeric_limits<std::int8_t>::max())) {
                    return TMinimalNumericVariant(static_cast<std::int8_t>(valU));
                }
            }
            // check uint8_t
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::uint8_t>) {
                if (valU <= std::numeric_limits<std::uint8_t>::max()) {
                    return TMinimalNumericVariant(static_cast<std::uint8_t>(valU));
                }
            }
            // check int16_t
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int16_t>) {
                if (valU <= static_cast<unsigned long long>(std::numeric_limits<std::int16_t>::max())) {
                    return TMinimalNumericVariant(static_cast<std::int16_t>(valU));
                }
            }
            // check uint16_t
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::uint16_t>) {
                if (valU <= std::numeric_limits<std::uint16_t>::max()) {
                    return TMinimalNumericVariant(static_cast<std::uint16_t>(valU));
                }
            }
            // check int32_t
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int32_t>) {
                if (valU <= static_cast<unsigned long long>(std::numeric_limits<std::int32_t>::max())) {
                    return TMinimalNumericVariant(static_cast<std::int32_t>(valU));
                }
            }
            // check uint32_t
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::uint32_t>) {
                if (valU <= std::numeric_limits<std::uint32_t>::max()) {
                    return TMinimalNumericVariant(static_cast<std::uint32_t>(valU));
                }
            }
            // check int64_t
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::int64_t>) {
                if (valU <= static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max())) {
                    return TMinimalNumericVariant(static_cast<std::int64_t>(valU));
                }
            }
            // check uint64_t
            if constexpr (detail::variant_contains_v<TMinimalNumericVariant, std::uint64_t>) {
                if (valU <= std::numeric_limits<std::uint64_t>::max()) {
                    return TMinimalNumericVariant(static_cast<std::uint64_t>(valU));
                }
            }

            return std::unexpected("no suitable integral type available (non-negative)");
        }
    } else {
        // mightBeFloat => parse float or double
        if constexpr (detail::variant_contains_v<TMinimalNumericVariant, float>) {
            float fv{};
            auto [p, errorCode] = std::from_chars(str.data(), str.data() + str.size(), fv, std::chars_format::general);
            if (errorCode == std::errc() && p == str.data() + str.size()) {
                return TMinimalNumericVariant(fv);
            } else if (errorCode == std::errc::result_out_of_range) {
                // fallback to double if present
            } else if (errorCode == std::errc::invalid_argument) {
                // try double
            }
        }
        if constexpr (detail::variant_contains_v<TMinimalNumericVariant, double>) {
            double dv{};
            auto [p, errorCode] = std::from_chars(str.data(), str.data() + str.size(), dv, std::chars_format::general);
            if (errorCode == std::errc() && p == str.data() + str.size()) {
                return TMinimalNumericVariant(dv);
            }
            if (errorCode == std::errc::result_out_of_range) {
                return std::unexpected("double parse out-of-range");
            }
            return std::unexpected("invalid float parse");
        }
        return std::unexpected("no suitable floating type available");
    }
}

} // namespace pmtv

#endif // PMTTYPEHELPERS_HPP
