#ifndef GRAPH_PROTOTYPE_ANNOTATED_HPP
#define GRAPH_PROTOTYPE_ANNOTATED_HPP

#include <string_view>
#include <type_traits>
#include <utils.hpp>

namespace fair::graph {

/**
 * @brief a template wrapping structure, which holds a static documentation (e.g. mark down) string as its value.
 * It's used as a trait class to annotate other template classes (e.g. blocks or fields).
 */
template<fair::meta::fixed_string doc_string>
struct Doc {
    static constexpr fair::meta::fixed_string value = doc_string;
};

using EmptyDoc = Doc<"">; // nomen-est-omen

template<typename T>
struct is_doc : std::false_type {};

template<fair::meta::fixed_string N>
struct is_doc<Doc<N>> : std::true_type {};

template<typename T>
concept Documentation = is_doc<T>::value;

/**
 * @brief Unit is a template structure, which holds a static physical-unit (i.e. SI unit) string as its value.
 * It's used as a trait class to annotate other template classes (e.g. blocks or fields).
 */
template<fair::meta::fixed_string doc_string>
struct Unit {
    static constexpr fair::meta::fixed_string value = doc_string;
};

using EmptyUnit = Unit<"">; // nomen-est-omen

template<typename T>
struct is_unit : std::false_type {};

template<fair::meta::fixed_string N>
struct is_unit<Unit<N>> : std::true_type {};

template<typename T>
concept UnitType = is_unit<T>::value;

static_assert(Documentation<EmptyDoc>);
static_assert(UnitType<EmptyUnit>);
static_assert(!UnitType<EmptyDoc>);
static_assert(!Documentation<EmptyUnit>);

/**
 * @brief Annotates field etc. that the entity is visible.
 */
struct Visible {};

/**
 * @brief Annotates node, indicating to calling schedulers that it may block due IO.
 */
struct BlockingIO {};

/**
 * @brief Annotates templated node, indicating which port data types are supported.
 */
template<typename... Ts>
struct SupportedTypes {};

template<typename T>
struct is_supported_types : std::false_type {};

template<typename... Ts>
struct is_supported_types<SupportedTypes<Ts...>> : std::true_type {};

using DefaultSupportedTypes = SupportedTypes<>;

static_assert(fair::meta::is_instantiation_of<DefaultSupportedTypes, SupportedTypes>);
static_assert(fair::meta::is_instantiation_of<SupportedTypes<float, double>, SupportedTypes>);

/**
 * @brief Annotated is a template class that acts as a transparent wrapper around another type.
 * It allows adding additional meta-information to a type, such as documentation, unit, and visibility.
 * The meta-information is supplied as template parameters.
 */
template<typename T, fair::meta::fixed_string description_ = "", typename... Arguments>
struct Annotated {
    using value_type = T;
    T value;

    Annotated() = default;

    constexpr Annotated(const T &value_) noexcept(std::is_nothrow_copy_constructible_v<T>) : value(value_) {}

    constexpr Annotated(T &&value_) noexcept(std::is_nothrow_move_constructible_v<T>) : value(std::move(value_)) {}

    // N.B. intentional implicit assignment and conversion operators to have a transparent wrapper
    // this does not affect the conversion of the wrapped value type 'T' itself
    constexpr Annotated &
    operator=(const T &value_) noexcept {
        value = value_;
        return *this;
    }

    constexpr Annotated &
    operator=(T &&value_) noexcept {
        value = std::move(value_);
        return *this;
    }

    inline explicit(false) constexpr
    operator T &() noexcept {
        return value;
    }

    inline explicit(false) constexpr operator const T &() const noexcept { return value; }

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

    // meta-information
    static constexpr std::string_view
    description() noexcept {
        return std::string_view{ description_ };
    }

    static constexpr std::string_view
    documentation() noexcept {
        using Documentation = typename fair::meta::typelist<Arguments...>::template find_or_default<is_doc, EmptyDoc>;
        return std::string_view{ Documentation::value };
    }

    static constexpr std::string_view
    unit() noexcept {
        using PhysicalUnit = typename fair::meta::typelist<Arguments...>::template find_or_default<is_unit, EmptyUnit>;
        return std::string_view{ PhysicalUnit::value };
    }

    static constexpr bool
    visible() noexcept {
        return fair::meta::typelist<Arguments...>::template contains<Visible>;
    }
};

template<typename T>
struct is_annotated : std::false_type {};

template<typename T, fair::meta::fixed_string str, typename... Args>
struct is_annotated<fair::graph::Annotated<T, str, Args...>> : std::true_type {};

template<typename T>
concept AnnotatedType = is_annotated<T>::value;

template<typename T>
struct inner_type {
    using type = T;
};

template<typename U, fair::meta::fixed_string str, typename... Args>
struct inner_type<fair::graph::Annotated<U, str, Args...>> {
    using type = U;
};

/**
 * @brief A type trait class that extracts the underlying type `T` from an `Annotated` instance.
 * If the given type is not an `Annotated`, it returns the type itself.
 */
template<typename T>
using inner_type_t = typename inner_type<T>::type;

} // namespace fair::graph

template<typename... Ts>
struct fair::meta::typelist<fair::graph::SupportedTypes<Ts...>> : fair::meta::typelist<Ts...> {};

#endif // GRAPH_PROTOTYPE_ANNOTATED_HPP
