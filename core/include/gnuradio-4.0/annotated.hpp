#ifndef GNURADIO_ANNOTATED_HPP
#define GNURADIO_ANNOTATED_HPP

#include <string_view>
#include <type_traits>
#include <utility>

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr {

/**
 * @brief a template wrapping structure, which holds a static documentation (e.g. mark down) string as its value.
 * It's used as a trait class to annotate other template classes (e.g. blocks or fields).
 */
template<gr::meta::fixed_string doc_string>
struct Doc {
    static constexpr gr::meta::fixed_string value = doc_string;
};

using EmptyDoc = Doc<"">; // nomen-est-omen

template<typename T>
struct is_doc : std::false_type {};

template<gr::meta::fixed_string N>
struct is_doc<Doc<N>> : std::true_type {};

template<typename T>
concept Documentation = is_doc<T>::value;

/**
 * @brief Unit is a template structure, which holds a static physical-unit (i.e. SI unit) string as its value.
 * It's used as a trait class to annotate other template classes (e.g. blocks or fields).
 */
template<gr::meta::fixed_string doc_string>
struct Unit {
    static constexpr gr::meta::fixed_string value = doc_string;
};

using EmptyUnit = Unit<"">; // nomen-est-omen

template<typename T>
struct is_unit : std::false_type {};

template<gr::meta::fixed_string N>
struct is_unit<Unit<N>> : std::true_type {};

template<typename T>
concept UnitType = is_unit<T>::value;

static_assert(Documentation<EmptyDoc>);
static_assert(UnitType<EmptyUnit>);
static_assert(!UnitType<EmptyDoc>);
static_assert(!Documentation<EmptyUnit>);

/**
 * @brief Annotates field etc. that the entity is visible from a UI perspective.
 */
struct Visible {};

/**
 * @brief Annotates block, indicating to calling schedulers that it may block due IO.
 */
template<bool UseIoThread = true>
struct BlockingIO {
    [[maybe_unused]] constexpr static bool useIoThread = UseIoThread;
};

/**
 * @brief Annotates block, indicating to perform resampling based on the provided ratio.
 *
 * The ratio between numerator and denominator defines the number of samples to be interpolated or decimated.
 * - If the ratio is greater than 1, interpolation occurs.
 * - If the ratio is less than 1, decimation occurs.
 * - If the ratio is 1, no effect on the sampling rate.
 *
 * @tparam numerator Top number in the input-to-output sample ratio.
 * @tparam denominator Bottom number in the input-to-output sample ratio.
 * @tparam isConst Specifies if the resampling ratio is constant or can be modified during run-time.
 */
template<std::size_t numerator = 1LU, std::size_t denominator = 1LU, bool isConst = false>
struct ResamplingRatio {
    static_assert(numerator > 0, "Numerator in ResamplingRatio must be >= 0");
    static constexpr std::size_t kNumerator   = numerator;
    static constexpr std::size_t kDenominator = denominator;
    static constexpr bool        kIsConst     = isConst;
    static constexpr bool        kEnabled     = !isConst || (kNumerator != 1LU) || (kDenominator != 1LU);
};

template<typename T>
concept IsResamplingRatio = requires {
    T::kNumerator;
    T::kDenominator;
    T::kIsConst;
    T::kEnabled;
} && std::is_base_of_v<ResamplingRatio<T::kNumerator, T::kDenominator, T::kIsConst>, T>;

template<typename T>
using is_resampling_ratio = std::bool_constant<IsResamplingRatio<T>>;

static_assert(is_resampling_ratio<ResamplingRatio<1, 1024>>::value);
static_assert(!is_resampling_ratio<int>::value);

/**
 * @brief Annotates block, indicating the stride control for data processing.
 *
 * Stride determines the number of samples between consecutive data processing events:
 * - If stride is less than N, it indicates overlap.
 * - If stride is greater than N, it indicates skipped samples.
 * - If stride is equal to 0, it indicates back-to-back processing without skipping.
 *
 * @tparam stride The number of samples between data processing events.
 * @tparam isConst Specifies if the stride is constant or can be modified during run-time.
 */
template<std::size_t stride = 0LU, bool isConst = false>
struct Stride {
    static_assert(stride >= 0, "Stride must be >= 0");

    static constexpr std::size_t kStride  = stride;
    static constexpr bool        kIsConst = isConst;
    static constexpr bool        kEnabled = !isConst || (stride > 0);
};

template<typename T>
concept IsStride = requires {
    T::kStride;
    T::kIsConst;
    T::kEnabled;
} && std::is_base_of_v<Stride<T::kStride, T::kIsConst>, T>;

template<typename T>
using is_stride = std::bool_constant<IsStride<T>>;

static_assert(is_stride<Stride<10, true>>::value);
static_assert(!is_stride<int>::value);

/**
 * @brief Annotates templated block, indicating which port data types are supported.
 */
template<typename... Ts>
struct SupportedTypes {};

template<typename T>
struct is_supported_types : std::false_type {};

template<typename... Ts>
struct is_supported_types<SupportedTypes<Ts...>> : std::true_type {};

using DefaultSupportedTypes = SupportedTypes<>;

static_assert(gr::meta::is_instantiation_of<DefaultSupportedTypes, SupportedTypes>);
static_assert(gr::meta::is_instantiation_of<SupportedTypes<float, double>, SupportedTypes>);

/**
 * @brief Represents limits and optional validation for an Annotated<..> type.
 *
 * The `Limits` structure defines lower and upper bounds for a value of type `T`.
 * Additionally, it allows for an optional custom validation function to be provided.
 * This function should take a value of type `T` and return a `bool`, indicating
 * whether the value passes the custom validation or not.
 *
 * Example:
 * ```
 * Annotated<float, "example float", Visible, Limits<0.f, 1024.f>>             exampleVar1;
 * // or:
 * constexpr auto isPowerOfTwo = [](const int &val) { return val > 0 && (val & (val - 1)) == 0; };
 * Annotated<float, "example float", Visible, Limits<0.f, 1024.f, isPowerOfTwo>> exampleVar2;
 * // or:
 * Annotated<float, "example float", Visible, Limits<0.f, 1024.f, [](const int &val) { return val > 0 && (val & (val - 1)) == 0; }>> exampleVar2;
 * ```
 */
template<auto LowerLimit, decltype(LowerLimit) UpperLimit, auto Validator = nullptr>
    requires(requires(decltype(Validator) f, decltype(LowerLimit) v) {
        { f(v) } -> std::same_as<bool>;
    } || Validator == nullptr)
struct Limits {
    using ValueType                                    = decltype(LowerLimit);
    static constexpr ValueType           MinRange      = LowerLimit;
    static constexpr ValueType           MaxRange      = UpperLimit;
    static constexpr decltype(Validator) ValidatorFunc = Validator;

    static constexpr bool
    validate(const ValueType &value) noexcept {
        if constexpr (LowerLimit == UpperLimit) { // ignore range checks
            if constexpr (Validator != nullptr) {
                try {
                    return Validator(value);
                } catch (...) {
                    return false;
                }
            } else {
                return true; // if no validator and limits are same, return true by default
            }
        }
        if constexpr (Validator != nullptr) {
            try {
                return value >= LowerLimit && value <= UpperLimit && Validator(value);
            } catch (...) {
                return false;
            }
        } else {
            return value >= LowerLimit && value <= UpperLimit;
        }
        return true;
    }
};

template<typename T>
struct is_limits : std::false_type {};

template<auto LowerLimit, decltype(LowerLimit) UpperLimit, auto Validator>
struct is_limits<Limits<LowerLimit, UpperLimit, Validator>> : std::true_type {};

template<typename T>
concept Limit    = is_limits<T>::value;

using EmptyLimit = Limits<0, 0>; // nomen-est-omen

static_assert(Limit<EmptyLimit>);

/**
 * @brief Annotated is a template class that acts as a transparent wrapper around another type.
 * It allows adding additional meta-information to a type, such as documentation, unit, and visibility.
 * The meta-information is supplied as template parameters.
 */
template<typename T, gr::meta::fixed_string description_ = "", typename... Arguments>
struct Annotated {
    using value_type = T;
    using LimitType  = typename gr::meta::typelist<Arguments...>::template find_or_default<is_limits, EmptyLimit>;
    T value;

    Annotated() = default;

    template<typename U>
        requires std::constructible_from<T, U> && (!std::same_as<std::remove_cvref_t<U>, Annotated>)
    explicit(false)
    Annotated(U &&input) noexcept(std::is_nothrow_constructible_v<T, U>) : value(std::forward<U>(input)) {}

    template<typename U>
        requires std::assignable_from<T &, U>
    Annotated &
    operator=(U &&input) noexcept(std::is_nothrow_assignable_v<T, U>) {
        value = std::forward<U>(input);
        return *this;
    }

    inline explicit(false) constexpr
    operator T &() noexcept {
        return value;
    }

    inline explicit(false) constexpr
    operator const T &() const noexcept {
        return value;
    }

    constexpr bool
    operator==(const Annotated &other) const noexcept {
        return value == other.value;
    }

    template<typename U>
    constexpr bool
    operator==(const U &other) const noexcept {
        if constexpr (requires { other.value; }) {
            return value == other.value;
        } else {
            return value == other;
        }
    }

    template<typename U>
        requires std::is_same_v<std::remove_cvref_t<U>, T>
    [[nodiscard]] constexpr bool
    validate_and_set(U &&value_) {
        if constexpr (std::is_same_v<LimitType, EmptyLimit>) {
            value = std::forward<U>(value_);
            return true;
        } else {
            if (LimitType::validate(static_cast<typename LimitType::ValueType>(value_))) { // N.B. implicit casting needed until clang supports floats as NTTPs
                value = std::forward<U>(value_);
                return true;
            } else {
                return false;
            }
        }
    }

    operator std::string_view() const noexcept
        requires std::is_same_v<T, std::string>
    {
        return std::string_view(value); // Convert from std::string to std::string_view
    }

    // meta-information
    inline static constexpr std::string_view
    description() noexcept {
        return std::string_view{ description_ };
    }

    inline static constexpr std::string_view
    documentation() noexcept {
        using Documentation = typename gr::meta::typelist<Arguments...>::template find_or_default<is_doc, EmptyDoc>;
        return std::string_view{ Documentation::value };
    }

    inline static constexpr std::string_view
    unit() noexcept {
        using PhysicalUnit = typename gr::meta::typelist<Arguments...>::template find_or_default<is_unit, EmptyUnit>;
        return std::string_view{ PhysicalUnit::value };
    }

    inline static constexpr bool
    visible() noexcept {
        return gr::meta::typelist<Arguments...>::template contains<Visible>;
    }
};

template<typename T>
struct is_annotated : std::false_type {};

template<typename T, gr::meta::fixed_string str, typename... Args>
struct is_annotated<gr::Annotated<T, str, Args...>> : std::true_type {};

template<typename T>
concept AnnotatedType = is_annotated<T>::value;

template<typename T>
struct unwrap_if_wrapped {
    using type = T;
};

template<typename U, gr::meta::fixed_string str, typename... Args>
struct unwrap_if_wrapped<gr::Annotated<U, str, Args...>> {
    using type = U;
};

/**
 * @brief A type trait class that extracts the underlying type `T` from an `Annotated` instance.
 * If the given type is not an `Annotated`, it returns the type itself.
 */
template<typename T>
using unwrap_if_wrapped_t = typename unwrap_if_wrapped<T>::type;

} // namespace gr

template<typename... Ts>
struct gr::meta::typelist<gr::SupportedTypes<Ts...>> : gr::meta::typelist<Ts...> {};

#ifdef FMT_FORMAT_H_

#include <fmt/core.h>
#include <fmt/ostream.h>

template<typename T, gr::meta::fixed_string description, typename... Arguments>
struct fmt::formatter<gr::Annotated<T, description, Arguments...>> {
    using Type = std::remove_const_t<T>;
    fmt::formatter<Type> value_formatter;

    template<typename FormatContext>
    constexpr auto
    parse(FormatContext &ctx) {
        return value_formatter.parse(ctx);
    }

    template<typename FormatContext>
    constexpr auto
    format(const gr::Annotated<T, description, Arguments...> &annotated, FormatContext &ctx) {
        // TODO: add switch for printing only brief and/or meta-information
        return value_formatter.format(annotated.value, ctx);
    }
};

namespace gr {
template<typename T, gr::meta::fixed_string description, typename... Arguments>
inline std::ostream &
operator<<(std::ostream &os, const gr::Annotated<T, description, Arguments...> &v) {
    // TODO: add switch for printing only brief and/or meta-information
    return os << fmt::format("{}", v.value);
}
} // namespace gr

#endif // FMT_FORMAT_H_

#endif // GNURADIO_ANNOTATED_HPP
