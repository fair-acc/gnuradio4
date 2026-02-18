#ifndef GNURADIO_ANNOTATED_HPP
#define GNURADIO_ANNOTATED_HPP

#include <format>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>

#include <gnuradio-4.0/meta/formatter.hpp>
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
 * @brief Disable default tag forwarding.
 *
 * There are two types of tag forwarding: (1) default All-To-All, and (2) user-implemented.
 *
 * By default, tag forwarding operates as All-To-All. Before tags on input ports are forwarded, they are merged.
 * If a block has multiple ports and tags on these ports contain maps with identical keys, only one value for each key
 * will be retained in the merged tag. This may lead to potential information loss, as it’s not guaranteed which
 * value will be kept.
 *
 * This default behavior is generally sufficient. However, if it’s not suitable for your use case, you can disable it
 * by adding the `NoDefaultTagForwarding` attribute to the template parameters. In such cases, the block should implement
 * custom tag forwarding in the `processBulk` function. The `InputSpanLike` and `OutputSpanLike` APIs are available to simplify
 * with custom tag forwarding.
 */
struct NoDefaultTagForwarding {};

/**
 * @brief Specifies the Tag Forwarding Policy for blocks with chunked data constraints, namely `input_chunk_size != 1`.
 *
 * If this attribute is omitted, the default `Forward` policy applies. The available policies are:
 * - Forward (default, no attribute required): Processes the tag as if it belongs to the first sample of the **next** `processBulk(..)` call.
 * - Backward (`BackwardTagForwarding` is set): Processes the tag as if it belongs to the first sample of the **current** `processBulk(..)` call.
 */
struct BackwardTagForwarding {};

/**
 * @brief Annotates block, indicating to perform resampling based on the provided `inputChunkSize` and `outputChunkSize`.
 * For each `inputChunkSize` input samples, `outputChunkSize` output samples are published.
 * Thus the total number of input/output samples can be calculated as `nInput = k * inputChunkSize` and `nOutput = k * outputChunkSize`.
 * They also act as constraints for the minimum number of input samples (`inputChunkSize`)  and output samples (`outputChunkSize`).
 *
 * │inputChunkSize│...│inputChunkSize│ ───► │outputChunkSize│...│outputChunkSize│
 * └──────────────┘   └──────────────┘      └───────────────┘   └───────────────┘
 *    nInputs = k * inputChunkSize              nOutputs = k * outputChunkSize
 *
 * The comparison between `inputChunkSize` and `outputChunkSize` determines whether to perform interpolation or decimation.
 * - If `inputChunkSize` > `outputChunkSize`, decimation occurs.
 * - If `inputChunkSize` < `outputChunkSize`, interpolation occurs.
 * - If `inputChunkSize` == `outputChunkSize`, there is no effect on the sampling rate.
 *
 * @tparam inputChunkSize input chunk size.
 * @tparam outputChunkSize output chunk size.
 * @tparam isConst Specifies if the resampling is constant or can be modified during run-time.
 */
template<gr::Size_t inputChunkSize = 1U, gr::Size_t outputChunkSize = 1U, bool isConst = false>
struct Resampling {
    static_assert(outputChunkSize > 0, "outputChunkSize in ResamplingRatio must be >= 0");
    static constexpr gr::Size_t kInputChunkSize  = inputChunkSize;
    static constexpr gr::Size_t kOutputChunkSize = outputChunkSize;
    static constexpr bool       kIsConst         = isConst;
    static constexpr bool       kEnabled         = !isConst || (kOutputChunkSize != 1LU) || (kInputChunkSize != 1LU);
};

template<typename T>
concept IsResampling = requires {
    T::kInputChunkSize;
    T::kOutputChunkSize;
    T::kIsConst;
    T::kEnabled;
} && std::is_base_of_v<Resampling<T::kInputChunkSize, T::kOutputChunkSize, T::kIsConst>, T>;

template<typename T>
using is_resampling = std::bool_constant<IsResampling<T>>;

static_assert(is_resampling<Resampling<1024, 1>>::value);
static_assert(!is_resampling<int>::value);

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
template<std::uint64_t stride = 0U, bool isConst = false>
struct Stride {
    static_assert(stride >= 0U, "Stride must be >= 0");

    static constexpr gr::Size_t kStride  = stride;
    static constexpr bool       kIsConst = isConst;
    static constexpr bool       kEnabled = !isConst || (stride > 0U);
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

enum class IncompleteFinalUpdateEnum { DROP, PULL_FORWARD, PUSH_BACKWARD };

template<IncompleteFinalUpdateEnum updatePolicy>
struct IncompleteFinalUpdatePolicy {
    static constexpr IncompleteFinalUpdateEnum kIncompleteFinalUpdatePolicy = updatePolicy;
};

template<typename T>
concept IsIncompleteFinalUpdatePolicy = requires { T::kIncompleteFinalUpdatePolicy; } && std::is_base_of_v<IncompleteFinalUpdatePolicy<T::kIncompleteFinalUpdatePolicy>, T>;

template<typename T>
using is_incompleteFinalUpdatePolicy = std::bool_constant<IsIncompleteFinalUpdatePolicy<T>>;

static_assert(is_incompleteFinalUpdatePolicy<IncompleteFinalUpdatePolicy<IncompleteFinalUpdateEnum::DROP>>::value);

enum class UICategory {
    None,        /// No UI contribution (default).
    MenuBar,     /// Global app menu bar items (File/Edit/View…).
    Toolbar,     /// Compact, frequently used actions/toggles.
    StatusBar,   /// Always-visible, low-interaction status readouts.
    Content,     /// Primary viewport output (plots/canvases/dashboards).
    Panel,       /// Secondary panels (inspectors, settings, lists, logs).
    Overlay,     /// Layered HUD over Content (cursors/markers/annotations).
    ContextMenu, /// Right-click / long-press contextual popup menus.
    Dialog,      /// Modal/semi-modal short workflows (export/config/confirm).
    Notification /// Transient non-modal feedback (toast/banner).
};

/**
 * @brief Annotates block, indicating that it is drawable and provides a  mandatory `void draw()` method.
 *
 * @tparam category_ ui category where it
 * @tparam toolkit_ specifies the applicable UI toolkit (e.g. 'console', 'ImGui', 'Qt', etc.)
 */
template<UICategory category_, gr::meta::fixed_string toolkit_ = "">
struct Drawable {
    static constexpr UICategory             kCategory = category_;
    static constexpr gr::meta::fixed_string kToolkit  = toolkit_;
};

template<typename T>
concept IsDrawable = requires {
    T::kCategory;
    T::kToolkit;
} && std::is_base_of_v<Drawable<T::kCategory, T::kToolkit>, T>;

template<typename T>
using is_drawable = std::bool_constant<IsDrawable<T>>;

using NotDrawable = Drawable<UICategory::None, "">; // nomen-est-omen
static_assert(is_drawable<NotDrawable>::value);
static_assert(is_drawable<Drawable<UICategory::Content, "console">>::value);
static_assert(!is_drawable<int>::value);

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

    static constexpr bool validate(const ValueType& value) noexcept {
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
concept Limit = is_limits<T>::value;

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
    explicit(false) Annotated(U&& input) noexcept(std::is_nothrow_constructible_v<T, U>) : value(static_cast<T>(std::forward<U>(input))) {}

    template<typename U>
    requires std::assignable_from<T&, U>
    Annotated& operator=(U&& input) noexcept(std::is_nothrow_assignable_v<T, U>) {
        value = static_cast<T>(std::forward<U>(input));
        return *this;
    }

    inline explicit(false) constexpr operator T&() noexcept { return value; }

    inline explicit(false) constexpr operator const T&() const noexcept { return value; }

    constexpr bool operator==(const Annotated& other) const noexcept { return value == other.value; }

    template<typename U>
    constexpr bool operator==(const U& other) const noexcept {
        if constexpr (requires { other.value; }) {
            return value == other.value;
        } else {
            return value == other;
        }
    }

    template<typename U>
    requires std::is_same_v<std::remove_cvref_t<U>, T>
    [[nodiscard]] constexpr bool validate_and_set(U&& value_) {
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
    inline static constexpr std::string_view description() noexcept { return std::string_view{description_}; }

    inline static constexpr std::string_view documentation() noexcept {
        using Documentation = typename gr::meta::typelist<Arguments...>::template find_or_default<is_doc, EmptyDoc>;
        return std::string_view{Documentation::value};
    }

    inline static constexpr std::string_view unit() noexcept {
        using PhysicalUnit = typename gr::meta::typelist<Arguments...>::template find_or_default<is_unit, EmptyUnit>;
        return std::string_view{PhysicalUnit::value};
    }

    inline static constexpr bool visible() noexcept { return gr::meta::typelist<Arguments...>::template contains<Visible>; }

    // forwarding member functions
    template<typename... Args>
    constexpr auto operator()(Args&&... args) -> decltype(auto) {
        return value(std::forward<Args>(args)...);
    }

    template<typename... Args>
    constexpr auto operator()(Args&&... args) const -> decltype(auto) {
        return value(std::forward<Args>(args)...);
    }

    template<typename Arg>
    constexpr auto operator[](Arg&& arg) -> decltype(auto) {
        return value[std::forward<Arg>(arg)];
    }

    template<typename Arg>
    constexpr auto operator[](Arg&& arg) const -> decltype(auto) {
        return value[std::forward<Arg>(arg)];
    }

    template<typename Arg>
    constexpr auto operator->*(Arg&& arg) -> decltype(auto) {
        return value.*std::forward<Arg>(arg);
    }

    template<typename Arg>
    constexpr auto operator->*(Arg&& arg) const -> decltype(auto) {
        return value.*std::forward<Arg>(arg);
    }

    constexpr T*       operator->() noexcept { return &value; }
    constexpr const T* operator->() const noexcept { return &value; }
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

template<typename T, gr::meta::fixed_string description, typename... Arguments>
struct std::formatter<gr::Annotated<T, description, Arguments...>> {
    using Type = std::remove_const_t<T>;
    std::formatter<Type> value_formatter;

    template<typename FormatContext>
    constexpr auto parse(FormatContext& ctx) {
        return value_formatter.parse(ctx);
    }

    template<typename FormatContext>
    constexpr auto format(const gr::Annotated<T, description, Arguments...>& annotated, FormatContext& ctx) const {
        return value_formatter.format(annotated.value, ctx);
    }
};

namespace gr {
template<typename T, gr::meta::fixed_string description, typename... Arguments>
inline std::ostream& operator<<(std::ostream& os, const gr::Annotated<T, description, Arguments...>& v) {
    // TODO: add switch for printing only brief and/or meta-information
    return os << std::format("{}", v.value);
}
} // namespace gr

#endif // GNURADIO_ANNOTATED_HPP
