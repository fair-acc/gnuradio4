#ifndef GNURADIO_TYPELIST_HPP
#define GNURADIO_TYPELIST_HPP

#include <bit>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <string_view>
#include <string>

namespace fair::meta {

template<typename... Ts>
struct typelist;

// concat ///////////////
namespace detail {
template<typename...>
struct concat_impl;

template<>
struct concat_impl<> {
    using type = typelist<>;
};

template<typename A>
struct concat_impl<A> {
    using type = typelist<A>;
};

template<typename... As>
struct concat_impl<typelist<As...>> {
    using type = typelist<As...>;
};

template<typename A, typename B>
struct concat_impl<A, B> {
    using type = typelist<A, B>;
};

template<typename... As, typename B>
struct concat_impl<typelist<As...>, B> {
    using type = typelist<As..., B>;
};

template<typename A, typename... Bs>
struct concat_impl<A, typelist<Bs...>> {
    using type = typelist<A, Bs...>;
};

template<typename... As, typename... Bs>
struct concat_impl<typelist<As...>, typelist<Bs...>> {
    using type = typelist<As..., Bs...>;
};

template<typename A, typename B, typename C>
struct concat_impl<A, B, C> {
    using type = typename concat_impl<typename concat_impl<A, B>::type, C>::type;
};

template<typename A, typename B, typename C, typename D, typename... More>
struct concat_impl<A, B, C, D, More...> {
    using type =
            typename concat_impl<typename concat_impl<A, B>::type, typename concat_impl<C, D>::type,
                                 typename concat_impl<More...>::type>::type;
};
} // namespace detail

template<typename... Ts>
using concat = typename detail::concat_impl<Ts...>::type;

// split_at, left_of, right_of ////////////////
namespace detail {
template<unsigned N>
struct splitter;

template<>
struct splitter<0> {
    template<typename...>
    using first = typelist<>;
    template<typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<1> {
    template<typename T0, typename...>
    using first = typelist<T0>;
    template<typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<2> {
    template<typename T0, typename T1, typename...>
    using first = typelist<T0, T1>;
    template<typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<4> {
    template<typename T0, typename T1, typename T2, typename T3, typename...>
    using first = typelist<T0, T1, T2, T3>;
    template<typename, typename, typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<8> {
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7, typename...>
    using first = typelist<T0, T1, T2, T3, T4, T5, T6, T7>;

    template<typename, typename, typename, typename, typename, typename, typename, typename,
             typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<16> {
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14, typename T15, typename...>
    using first = typelist<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>;

    template<typename, typename, typename, typename, typename, typename, typename, typename, typename,
             typename, typename, typename, typename, typename, typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<unsigned N>
struct splitter {
    static constexpr unsigned FirstSplit = std::has_single_bit(N) ? N / 2 : std::bit_floor(N);
    using A                              = splitter<FirstSplit>;
    using B                              = splitter<N - FirstSplit>;

    template<typename... Ts>
    using first = concat<typename A::template first<Ts...>,
                         typename B::template first<typename A::template second<Ts...>>>;

    template<typename... Ts>
    using second = typename B::template second<typename A::template second<Ts...>>;
};
} // namespace detail

template<unsigned N, typename List>
struct split_at;

template<unsigned N, typename... Ts>
struct split_at<N, typelist<Ts...>> {
    using first  = typename detail::splitter<N>::template first<Ts...>;
    using second = typename detail::splitter<N>::template second<Ts...>;
};

template<std::size_t N, typename List>
using left_of = typename split_at<N, List>::first;

template<std::size_t N, typename List>
using right_of = typename split_at<N + 1, List>::second;

// remove_at /////////////
template<std::size_t Idx, typename List>
using remove_at = concat<left_of<Idx, List>, right_of<Idx, List>>;

// first_type ////////////
namespace detail {
template<typename List>
struct first_type_impl {};

template<typename T0, typename... Ts>
struct first_type_impl<typelist<T0, Ts...>> {
    using type = T0;
};
} // namespace detail

template<typename List>
using first_type = typename detail::first_type_impl<List>::type;

// transform_types ////////////
namespace detail {
template<template<typename> class Template, typename List>
struct transform_types_impl;

template<template<typename> class Template, typename... Ts>
struct transform_types_impl<Template, typelist<Ts...>> {
    using type = typelist<Template<Ts>...>;
};
} // namespace detail

template<template<typename> class Template, typename List>
using transform_types = typename detail::transform_types_impl<Template, List>::type;

// transform_value_type
template<typename T>
using transform_value_type = typename T::value_type;

// reduce ////////////////
namespace detail {
template<template<typename, typename> class Method, typename List>
struct reduce_impl;

template<template<typename, typename> class Method, typename T0>
struct reduce_impl<Method, typelist<T0>> {
    using type = T0;
};

template<template<typename, typename> class Method, typename T0, typename T1, typename... Ts>
struct reduce_impl<Method, typelist<T0, T1, Ts...>>
    : public reduce_impl<Method, typelist<typename Method<T0, T1>::type, Ts...>> {};

template<template<typename, typename> class Method, typename T0, typename T1, typename T2,
         typename T3, typename... Ts>
struct reduce_impl<Method, typelist<T0, T1, T2, T3, Ts...>>
    : public reduce_impl<
              Method, typelist<typename Method<T0, T1>::type, typename Method<T2, T3>::type, Ts...>> {
};
} // namespace detail

template<template<typename, typename> class Method, typename List>
using reduce = typename detail::reduce_impl<Method, List>::type;

// typelist /////////////////
template<typename T>
concept is_typelist_v = requires { typename T::typelist_tag; };

template<typename... Ts>
struct typelist {
    using this_t = typelist<Ts...>;
    using typelist_tag = std::true_type;

    static inline constexpr std::integral_constant<std::size_t, sizeof...(Ts)> size = {};

    template<template<typename...> class Other>
    using apply = Other<Ts...>;

    template<std::size_t I>
    using at = first_type<typename detail::splitter<I>::template second<Ts...>>;

    template <typename Head>
    using prepend = typelist<Head, Ts...>;

    template<typename... Other>
    static constexpr inline bool are_equal = std::same_as<typelist, meta::typelist<Other...>>;

    template<typename... Other>
    static constexpr inline bool are_convertible_to = (std::convertible_to<Ts, Other> && ...);

    template<typename... Other>
    static constexpr inline bool are_convertible_from = (std::convertible_to<Other, Ts> && ...);

    template<typename F, typename Tup>
        requires(sizeof...(Ts) == std::tuple_size_v<std::remove_cvref_t<Tup>>)
    static constexpr auto
    construct(Tup &&args_tuple) {
        return std::apply(
                []<typename... Args>(Args &&...args) {
                    return std::make_tuple(F::template apply<Ts>(std::forward<Args>(args))...);
                },
                std::forward<Tup>(args_tuple));
    }

    template<template<typename> typename Trafo>
    using transform = meta::transform_types<Trafo, this_t>;

    template<template<typename...> typename Pred>
    constexpr static bool all_of = (Pred<Ts>::value && ...);

    template<template<typename...> typename Pred>
    constexpr static bool none_of = (!Pred<Ts>::value && ...);

    using safe_head = std::remove_pointer_t<decltype([] {
        if constexpr (sizeof...(Ts) > 0) {
            return static_cast<this_t::at<0>*>(nullptr);
        } else {
            return static_cast<void*>(nullptr);
        }
    }())>;

    template<typename Matcher = typename this_t::safe_head>
    constexpr static bool all_same =
        ((std::is_same_v<Matcher, Ts> && ...));

    template<template<typename...> typename Predicate>
    using filter = concat<std::conditional_t<Predicate<Ts>::value, typelist<Ts>, typelist<>>...>;

    using tuple_type    = std::tuple<Ts...>;
    using tuple_or_type = std::remove_pointer_t<decltype(
            [] {
                if constexpr (sizeof...(Ts) == 0) {
                    return static_cast<void*>(nullptr);
                } else if constexpr (sizeof...(Ts) == 1) {
                    return static_cast<at<0>*>(nullptr);
                } else {
                    return static_cast<tuple_type*>(nullptr);
                }
            }())>;

};


namespace detail {
    template <template <typename...> typename OtherTypelist, typename... Args>
    meta::typelist<Args...> to_typelist_helper(OtherTypelist<Args...>*);
} // namespace detail

template <typename OtherTypelist>
using to_typelist = decltype(detail::to_typelist_helper(static_cast<OtherTypelist*>(nullptr)));

} // namespace fair::meta

#endif // include guard
