#ifndef GNURADIO_REFLECTION_HPP
#define GNURADIO_REFLECTION_HPP

#include <gnuradio-4.0/meta/typelist.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <array>
// #include <string> // for type_name specialization
#include <tuple>
#ifdef _MSC_VER
#include <vector> // for type_name specialization
#endif

// recursive macro implementation inspired by https://www.scs.stanford.edu/~dm/blog/va-opt.html

#define GR_REFLECT_LIGHT_PARENS ()

#define GR_REFLECT_LIGHT_EXPAND(...)  GR_REFLECT_LIGHT_EXPAND3(GR_REFLECT_LIGHT_EXPAND3(GR_REFLECT_LIGHT_EXPAND3(GR_REFLECT_LIGHT_EXPAND3(__VA_ARGS__))))
#define GR_REFLECT_LIGHT_EXPAND3(...) GR_REFLECT_LIGHT_EXPAND2(GR_REFLECT_LIGHT_EXPAND2(GR_REFLECT_LIGHT_EXPAND2(GR_REFLECT_LIGHT_EXPAND2(__VA_ARGS__))))
#define GR_REFLECT_LIGHT_EXPAND2(...) GR_REFLECT_LIGHT_EXPAND1(GR_REFLECT_LIGHT_EXPAND1(GR_REFLECT_LIGHT_EXPAND1(GR_REFLECT_LIGHT_EXPAND1(__VA_ARGS__))))
#define GR_REFLECT_LIGHT_EXPAND1(...) __VA_ARGS__

#define GR_REFLECT_LIGHT_TO_STRINGS(...)         __VA_OPT__(GR_REFLECT_LIGHT_EXPAND(GR_REFLECT_LIGHT_TO_STRINGS_IMPL(__VA_ARGS__)))
#define GR_REFLECT_LIGHT_TO_STRINGS_IMPL(x, ...) ::gr::meta::constexpr_string<#x>() __VA_OPT__(, GR_REFLECT_LIGHT_TO_STRINGS_AGAIN GR_REFLECT_LIGHT_PARENS(__VA_ARGS__))
#define GR_REFLECT_LIGHT_TO_STRINGS_AGAIN()      GR_REFLECT_LIGHT_TO_STRINGS_IMPL

#define GR_REFLECT_LIGHT_COUNT_ARGS(...)         0 __VA_OPT__(+GR_REFLECT_LIGHT_EXPAND(GR_REFLECT_LIGHT_COUNT_ARGS_IMPL(__VA_ARGS__)))
#define GR_REFLECT_LIGHT_COUNT_ARGS_IMPL(x, ...) 1 __VA_OPT__(+GR_REFLECT_LIGHT_COUNT_ARGS_AGAIN GR_REFLECT_LIGHT_PARENS(__VA_ARGS__))
#define GR_REFLECT_LIGHT_COUNT_ARGS_AGAIN()      GR_REFLECT_LIGHT_COUNT_ARGS_IMPL

#define GR_REFLECT_LIGHT_DECLTYPES(...)         __VA_OPT__(GR_REFLECT_LIGHT_EXPAND(GR_REFLECT_LIGHT_DECLTYPES_IMPL(__VA_ARGS__)))
#define GR_REFLECT_LIGHT_DECLTYPES_IMPL(x, ...) decltype(x) __VA_OPT__(, GR_REFLECT_LIGHT_DECLTYPES_AGAIN GR_REFLECT_LIGHT_PARENS(__VA_ARGS__))
#define GR_REFLECT_LIGHT_DECLTYPES_AGAIN()      GR_REFLECT_LIGHT_DECLTYPES_IMPL

#define GR_MAKE_REFLECTABLE(T, ...)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \
    friend void* gr_refl_determine_base_type(T const&, ...) { return nullptr; }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    template<std::derived_from<T> GrRefl_U>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    \
    requires(not std::is_same_v<GrRefl_U, T>) and std::is_void_v<std::remove_pointer_t<decltype(gr_refl_determine_base_type(std::declval<::gr::refl::detail::make_dependent_t<GrRefl_U, T>>(), 0))>>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           \
    friend T* gr_refl_determine_base_type(GrRefl_U const&, int) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              \
        return nullptr;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    template<std::derived_from<T> GrRefl_U, typename GrRefl_Not>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    requires(not std::is_same_v<GrRefl_U, T>) and (not std::derived_from<GrRefl_Not, T>) and std::is_void_v<std::remove_pointer_t<decltype(gr_refl_determine_base_type(std::declval<::gr::refl::detail::make_dependent_t<GrRefl_U, T>>(), std::declval<GrRefl_Not>()))>>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
    friend T* gr_refl_determine_base_type(GrRefl_U const&, GrRefl_Not const&) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        return nullptr;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    using gr_refl_class_name = ::gr::meta::constexpr_string<#T>;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    constexpr auto gr_refl_members_as_tuple()& { return std::tie(__VA_ARGS__); }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    constexpr auto gr_refl_members_as_tuple() const& { return std::tie(__VA_ARGS__); }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    using gr_refl_data_member_types = ::gr::meta::typelist<GR_REFLECT_LIGHT_DECLTYPES(__VA_ARGS__)>;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    static constexpr std::integral_constant<std::size_t, GR_REFLECT_LIGHT_COUNT_ARGS(__VA_ARGS__)> gr_refl_data_member_count{};                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    static constexpr auto gr_refl_data_member_names = std::tuple { GR_REFLECT_LIGHT_TO_STRINGS(__VA_ARGS__) }

namespace gr::refl {

using std::size_t;

namespace detail {

template<typename T, typename U>
struct make_dependent {
    using type = U;
};

template<typename T, typename U>

using make_dependent_t = typename make_dependent<T, U>::type;

template<typename T>
concept class_type = std::is_class_v<T>;

struct None {};

template<typename T, typename Excluding>
using find_base = std::remove_pointer_t<decltype(gr_refl_determine_base_type(std::declval<T>(), std::declval<Excluding>()))>;

template<typename T, typename Last = None>
struct base_type_impl {
    using type = void;
};

// if Last is None we're starting the search
template<class_type T>
struct base_type_impl<T, None> {
    using type = typename base_type_impl<T, std::remove_pointer_t<decltype(gr_refl_determine_base_type(std::declval<T>(), 0))>>::type;
};

// if Last is void => there's no base type (void)
template<class_type T>
struct base_type_impl<T, void> {
    using type = void;
};

// otherwise, if find_base<T, Last> is void, Last is the base type
template<class_type T, class_type Last>
requires std::derived_from<T, Last> and std::is_void_v<find_base<T, Last>>
struct base_type_impl<T, Last> {
    using type = Last;
};

// otherwise, find_base<T, Last> is the next Last => recurse
template<class_type T, class_type Last>
requires std::derived_from<T, Last> and (not std::is_void_v<find_base<T, Last>>)
struct base_type_impl<T, Last> {
    using type = typename base_type_impl<T, find_base<T, Last>>::type;
};

template<typename T>
constexpr typename base_type_impl<T>::type const& to_base_type(T const& obj) {
    return obj;
}

template<typename T>
constexpr typename base_type_impl<T>::type& to_base_type(T& obj) {
    return obj;
}

template<auto X>
inline constexpr std::integral_constant<std::remove_const_t<decltype(X)>, X> ic = {};

template<typename T>
consteval auto type_to_string(T*) {
#ifdef __GNUC__
    constexpr auto   fun         = __PRETTY_FUNCTION__;
    constexpr size_t fun_size    = sizeof(__PRETTY_FUNCTION__) - 1;
    constexpr auto   offset_size = [&]() -> std::pair<size_t, size_t> {
        size_t offset = 0;
        for (; offset < fun_size and fun[offset] != '='; ++offset)
            ;
        if (offset + 2 >= fun_size or offset < 20 or fun[offset + 1] != ' ' or fun[offset - 2] != 'T') {
            return {0, fun_size};
        }
        offset += 2; // skip over '= '
        size_t size = 0;
        for (; offset + size < fun_size and fun[offset + size] != ']'; ++size)
            ;
        return {offset, size};
    }();
#elif defined _MSC_VER
    constexpr auto   fun         = __FUNCSIG__;
    constexpr size_t fun_size    = sizeof(__FUNCSIG__) - 1;
    constexpr auto   offset_size = [&]() -> std::pair<size_t, size_t> {
        size_t offset = 0;
        for (; offset < fun_size and fun[offset] != '<'; ++offset)
            ;
        if (offset + 2 >= fun_size or offset < 20 or fun[offset - 1] != 'g') {
            return {0, fun_size};
        }
        offset += 1; // skip over '<'
        // remove 'struct ', 'union ', 'class ', or 'enum ' prefix.
        if (std::string_view(fun + offset, 7) == "struct ") {
            offset += 7;
        } else if (std::string_view(fun + offset, 6) == "class ") {
            offset += 6;
        } else if (std::string_view(fun + offset, 6) == "union ") {
            offset += 6;
        } else if (std::string_view(fun + offset, 5) == "enum ") {
            offset += 5;
        }
        size_t size = 0;
        for (; offset + size < fun_size and fun[offset + size] != '('; ++size)
            ;
        return {offset, size - 1};
    }();
#else
#error "Compiler not supported."
#endif
    constexpr size_t offset = offset_size.first;
    constexpr size_t size   = offset_size.second;
    static_assert(offset < fun_size);
    static_assert(size <= fun_size);
    static_assert(offset + size <= fun_size);
    constexpr size_t comma_nospace_count = [&] {
        size_t count = 0;
        for (size_t i = offset; i < offset + size - 1; ++i) {
            if (fun[i] == ',' and fun[i + 1] != ' ') {
                ++count;
            }
        }
        return count;
    }();
    if constexpr (comma_nospace_count == 0) {
        return ::gr::meta::fixed_string<size>(fun + offset, fun + offset + size);
    } else {
        ::gr::meta::fixed_string<size + comma_nospace_count> buf = {};
        size_t                                               r   = offset;
        size_t                                               w   = 0;
        for (; r < offset + size; ++w, ++r) {
            buf[w] = fun[r];
            if (fun[r] == ',' and fun[r + 1] != ' ') {
                buf[++w] = ' ';
            }
        }
        return buf;
    }
}

template<auto X>
consteval auto nttp_to_string() {
#ifdef __GNUC__
    constexpr auto   fun         = __PRETTY_FUNCTION__;
    constexpr size_t fun_size    = sizeof(__PRETTY_FUNCTION__) - 1;
    constexpr auto   offset_size = [&]() -> std::pair<size_t, size_t> {
        size_t offset = 0;
        for (; offset < fun_size and fun[offset] != '='; ++offset)
            ;
        if (offset + 2 >= fun_size or offset < 20 or fun[offset + 1] != ' ' or fun[offset - 2] != 'X') {
            return {0, fun_size};
        }
        offset += 2; // skip over '= '
        size_t size = 0;
        for (; offset + size < fun_size and fun[offset + size] != ']'; ++size)
            ;
        return {offset, size};
    }();
#elif defined _MSC_VER
    constexpr auto   fun         = __FUNCSIG__;
    constexpr size_t fun_size    = sizeof(__FUNCSIG__) - 1;
    constexpr auto   offset_size = [&]() -> std::pair<size_t, size_t> {
        size_t offset = 0;
        for (; offset < fun_size and fun[offset] != '<'; ++offset)
            ;
        if (offset + 2 >= fun_size or offset < 20 or fun[offset - 1] != 'g') {
            return {0, fun_size};
        }
        offset += 1; // skip over '<'
        size_t size = 0;
        for (; offset + size < fun_size and fun[offset + size] != '('; ++size)
            ;
        return {offset, size - 1};
    }();
#else
#error "Compiler not supported."
#endif
    constexpr size_t offset = offset_size.first;
    constexpr size_t size   = offset_size.second;
    static_assert(offset < fun_size);
    static_assert(size <= fun_size);
    static_assert(offset + size <= fun_size);
    return ::gr::meta::fixed_string<size>(fun + offset, fun + offset + size);
}
} // namespace detail

template<typename T>
concept reflectable = std::is_class_v<std::remove_cvref_t<T>> and requires {
    { std::remove_cvref_t<T>::gr_refl_data_member_count } -> std::convertible_to<size_t>;
};

template<typename T>
inline constexpr auto type_name = ::gr::meta::constexpr_string<detail::type_to_string(static_cast<T*>(nullptr))>();

template<auto T>
requires std::is_enum_v<decltype(T)>
inline constexpr auto enum_name = ::gr::meta::constexpr_string<detail::nttp_to_string<T>()>();

template<auto T>
inline constexpr auto nttp_name = ::gr::meta::constexpr_string<detail::nttp_to_string<T>()>();

#define GR_SPECIALIZE_TYPE_NAME(T)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \
    template<>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
    inline constexpr auto type_name<T> = ::gr::meta::constexpr_string<#T> {}

GR_SPECIALIZE_TYPE_NAME(bool);
GR_SPECIALIZE_TYPE_NAME(char);
GR_SPECIALIZE_TYPE_NAME(wchar_t);
GR_SPECIALIZE_TYPE_NAME(char8_t);
GR_SPECIALIZE_TYPE_NAME(char16_t);
GR_SPECIALIZE_TYPE_NAME(char32_t);
GR_SPECIALIZE_TYPE_NAME(signed char);
GR_SPECIALIZE_TYPE_NAME(unsigned char);
GR_SPECIALIZE_TYPE_NAME(short);
GR_SPECIALIZE_TYPE_NAME(unsigned short);
GR_SPECIALIZE_TYPE_NAME(int);
GR_SPECIALIZE_TYPE_NAME(unsigned int);
GR_SPECIALIZE_TYPE_NAME(long);
GR_SPECIALIZE_TYPE_NAME(unsigned long);
GR_SPECIALIZE_TYPE_NAME(long long);
GR_SPECIALIZE_TYPE_NAME(unsigned long long);
GR_SPECIALIZE_TYPE_NAME(float);
GR_SPECIALIZE_TYPE_NAME(double);
GR_SPECIALIZE_TYPE_NAME(long double);
GR_SPECIALIZE_TYPE_NAME(std::string);
GR_SPECIALIZE_TYPE_NAME(std::string_view);

#undef GR_SPECIALIZE_TYPE_NAME

#ifdef _MSC_VER
template<typename T>
inline constexpr auto type_name<std::vector<T>> = ::gr::meta::constexpr_string<"std::vector<" + type_name<T> + '>'>{};
#endif

template<typename T>
inline constexpr auto class_name
#ifdef _MSC_VER
    = resize(type_name<T>.value, detail::ic<type_name<T>.value.find_char('<')>);
#else
    = type_name<T>.resize(type_name<T>.find_char(detail::ic<'<'>));
#endif

template<typename T>
using base_type = typename detail::base_type_impl<T>::type;

template<typename T>
constexpr size_t data_member_count = 0;

template<reflectable T>
requires std::is_void_v<base_type<T>>
constexpr size_t data_member_count<T> = T::gr_refl_data_member_count;

template<reflectable T>
requires(not std::is_void_v<base_type<T>>)
constexpr size_t data_member_count<T> = T::gr_refl_data_member_count + data_member_count<base_type<T>>;

template<typename T, size_t Idx>
constexpr auto data_member_name = [] {
    static_assert(Idx < data_member_count<T>);
    return ::gr::meta::constexpr_string<"Error">();
}();

template<reflectable T, size_t Idx>
requires(Idx < data_member_count<base_type<T>>)
constexpr auto data_member_name<T, Idx> = data_member_name<base_type<T>, Idx>;

template<reflectable T, size_t Idx>
requires(Idx >= data_member_count<base_type<T>>) and (Idx < data_member_count<T>)
constexpr auto data_member_name<T, Idx> = std::get<Idx - data_member_count<base_type<T>>>(T::gr_refl_data_member_names);

template<reflectable T, ::gr::meta::fixed_string Name>
constexpr auto data_member_index = detail::ic<[]<size_t... Is>(std::index_sequence<Is...>) { //
    return ((Name == data_member_name<T, Is>.value ? Is : 0) + ...);
}(std::make_index_sequence<data_member_count<T>>())>;

template<size_t Idx>
constexpr decltype(auto) data_member(reflectable auto&& obj) {
    using Class    = std::remove_cvref_t<decltype(obj)>;
    using BaseType = base_type<Class>;

    constexpr size_t base_size = data_member_count<BaseType>;

    if constexpr (Idx < base_size) {
        return data_member<Idx>(detail::to_base_type(obj));
    } else {
        return std::get<Idx - base_size>(obj.gr_refl_members_as_tuple());
    }
}

template<::gr::meta::fixed_string Name>
constexpr decltype(auto) data_member(reflectable auto&& obj) {
    return data_member<data_member_index<std::remove_cvref_t<decltype(obj)>, Name>>(obj);
}

constexpr decltype(auto) all_data_members(reflectable auto&& obj) {
    using B = base_type<std::remove_cvref_t<decltype(obj)>>;
    if constexpr (std::is_void_v<B>) {
        return obj.gr_refl_members_as_tuple();
    } else {
        return std::tuple_cat(all_data_members(static_cast<B&>(obj)), obj.gr_refl_members_as_tuple());
    }
}

namespace detail {
template<size_t N>
struct data_member_id : ::gr::meta::fixed_string<N> {
    static constexpr bool is_name = N != 0;

    const size_t index;

    consteval data_member_id(const char (&txt)[N + 1])
    requires(N != 0)
        : ::gr::meta::fixed_string<N>(txt), index(size_t(-1)) {}

    consteval data_member_id(std::convertible_to<size_t> auto idx)
    requires(N == 0)
        : ::gr::meta::fixed_string<0>(), index(size_t(idx)) {}

    consteval ::gr::meta::fixed_string<N> const& string() const { return *this; }
};

template<size_t N>
data_member_id(const char (&str)[N]) -> data_member_id<N - 1>;

template<std::convertible_to<size_t> T>
data_member_id(T) -> data_member_id<0>;

template<typename T, data_member_id Idx>
struct data_member_type_impl : data_member_type_impl<T, data_member_index<T, Idx.string()>> {};

template<typename T, data_member_id Idx>
requires(not Idx.is_name) and (Idx.index >= data_member_count<base_type<T>>)
struct data_member_type_impl<T, Idx> {
    using type = typename T::gr_refl_data_member_types::template at<Idx.index - data_member_count<base_type<T>>>;
};

template<typename T, data_member_id Idx>
requires(not Idx.is_name) and (Idx.index < data_member_count<base_type<T>>)
struct data_member_type_impl<T, Idx> {
    using type = typename data_member_type_impl<base_type<T>, Idx.index>::type;
};
} // namespace detail

template<reflectable T, detail::data_member_id Id>
using data_member_type = typename detail::data_member_type_impl<T, Id>::type;

template<reflectable T, template<typename, size_t> class Pred>
constexpr std::array find_data_members = []<size_t... Is>(std::index_sequence<Is...>) {
    constexpr size_t matches = (Pred<T, Is>::value + ...);

    constexpr std::array results = {(Pred<T, Is>::value ? Is : size_t(-1))...};

    std::array<size_t, matches> r = {};

    size_t i = 0;

    for (size_t idx : results) {
        if (idx != size_t(-1)) {
            r[i++] = idx;
        }
    }
    return r;
}(std::make_index_sequence<data_member_count<T>>());

template<reflectable T, template<typename> class Pred>
constexpr std::array find_data_members_by_type = []<size_t... Is>(std::index_sequence<Is...>) {
    constexpr size_t matches = (Pred<data_member_type<T, Is>>::value + ...);

    constexpr std::array results = {(Pred<data_member_type<T, Is>>::value ? Is : size_t(-1))...};

    std::array<size_t, matches> r = {};

    size_t i = 0;

    for (size_t idx : results) {
        if (idx != size_t(-1)) {
            r[i++] = idx;
        }
    }
    return r;
}(std::make_index_sequence<data_member_count<T>>());

namespace detail {
template<size_t N, typename = decltype(std::make_index_sequence<N>())>
constexpr std::array<std::size_t, N> iota_array;

template<size_t N, size_t... Values>
constexpr std::array<std::size_t, N> iota_array<N, std::index_sequence<Values...>> = {Values...};
} // namespace detail

template<reflectable T, std::array Idxs = detail::iota_array<data_member_count<T>>>
using data_member_types = decltype([]<size_t... Is>(std::index_sequence<Is...>) -> ::gr::meta::typelist<data_member_type<T, Idxs[Is]>...> { //
    return {};
}(std::make_index_sequence<Idxs.size()>()));

template<reflectable T>
constexpr void for_each_data_member_index(auto&& fun) {
    [&]<size_t... Is>(std::index_sequence<Is...>) { //
        (fun(detail::ic<Is>), ...);
    }(std::make_index_sequence<data_member_count<T>>());
}

namespace detail {
template<typename IdxSeq, auto Fun>
struct make_typelist_from_index_sequence_impl;

template<size_t... Is, auto Fun>
struct make_typelist_from_index_sequence_impl<std::index_sequence<Is...>, Fun> {
    using type = ::gr::meta::concat<decltype(Fun(ic<Is>))...>;
};
} // namespace detail

/**
 * Constructs a ::gr::meta::typelist via concatenation of all type lists returned from applying \p Fun to each index in the
 * given std::index_sequence \p IdxSeq.
 *
 * \tparam IdxSeq  The sequence of indexes to pass to \p Fun.
 * \tparam Fun     A function object (e.g. Lambda) that is called for every integer in \p IdxSeq. It is passed an
 *                 std::integral_constant<std::size_t, Idx> and needs to return a ::gr::meta::typelist object. The return
 *                 types of all \p Fun invocations are then concatenated (::gr::meta::concat) to the resulting typelist.
 */
template<typename IdxSeq, auto Fun>
using make_typelist_from_index_sequence = typename detail::make_typelist_from_index_sequence_impl<IdxSeq, Fun>::type;

} // namespace gr::refl

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

#endif // GNURADIO_REFLECTION_HPP
