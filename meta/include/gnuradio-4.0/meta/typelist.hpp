#ifndef GNURADIO_TYPELIST_HPP
#define GNURADIO_TYPELIST_HPP

#include <bit>
#include <concepts>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace gr::meta {

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
    using type = typename concat_impl<typename concat_impl<A, B>::type, typename concat_impl<C, D>::type, typename concat_impl<More...>::type>::type;
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
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename...>
    using first = typelist<T0, T1, T2, T3, T4, T5, T6, T7>;

    template<typename, typename, typename, typename, typename, typename, typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<16> {
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename...>
    using first = typelist<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>;

    template<typename, typename, typename, typename, typename, typename, typename, typename, typename, typename, typename, typename, typename, typename, typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<unsigned N>
struct splitter {
    static constexpr unsigned FirstSplit = std::has_single_bit(N) ? N / 2 : std::bit_floor(N);
    using A                              = splitter<FirstSplit>;
    using B                              = splitter<N - FirstSplit>;

    template<typename... Ts>
    using first = concat<typename A::template first<Ts...>, typename B::template first<typename A::template second<Ts...>>>;

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

template<template<typename, size_t> class Template, typename List, typename IdxSeq = decltype(std::make_index_sequence<List::size()>())>
struct transform_types_indexed_impl;

template<template<typename, size_t> class Template, typename... Ts, size_t... Is>
struct transform_types_indexed_impl<Template, typelist<Ts...>, std::index_sequence<Is...>> {
    using type = typelist<Template<Ts, Is>...>;
};
} // namespace detail

template<template<typename> class Template, typename List>
using transform_types = typename detail::transform_types_impl<Template, List>::type;

template<template<typename, size_t> class Template, typename List>
using transform_types_indexed = typename detail::transform_types_indexed_impl<Template, List>::type;

// transform_value_type
template<typename T>
using transform_value_type = typename T::value_type;

namespace detail {
template<bool Cond, template<typename> class Tpl1, template<typename> class Tpl2, typename T>
struct conditional_specialization;

template<template<typename> class Tpl1, template<typename> class Tpl2, typename T>
struct conditional_specialization<true, Tpl1, Tpl2, T> {
    using type = Tpl1<T>;
};

template<template<typename> class Tpl1, template<typename> class Tpl2, typename T>
struct conditional_specialization<false, Tpl1, Tpl2, T> {
    using type = Tpl2<T>;
};

template<typename CondFun, template<typename> class Tpl1, template<typename> class Tpl2, typename List>
struct transform_conditional_impl;

template<typename CondFun, template<typename> class Tpl1, template<typename> class Tpl2, typename... Ts>
struct transform_conditional_impl<CondFun, Tpl1, Tpl2, typelist<Ts...>> {
    using type = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) -> typelist<typename conditional_specialization<CondFun()(Is), Tpl1, Tpl2, Ts>::type...> { return {}; }(std::make_index_sequence<sizeof...(Ts)>()));
};
} // namespace detail

// Transform all types in List:
// For all types T with index I in List:
// If CondFun()(I) is true use Tpl1<T>, otherwise use Tpl2<T>
template<class CondFun, template<typename> class Tpl1, template<typename> class Tpl2, typename List>
using transform_conditional = typename detail::transform_conditional_impl<CondFun, Tpl1, Tpl2, List>::type;

// reduce ////////////////
namespace detail {
template<template<typename, typename> class Method, typename List>
struct reduce_impl;

template<template<typename, typename> class Method, typename T0>
struct reduce_impl<Method, typelist<T0>> {
    using type = T0;
};

template<template<typename, typename> class Method, typename T0, typename T1, typename... Ts>
struct reduce_impl<Method, typelist<T0, T1, Ts...>> : public reduce_impl<Method, typelist<typename Method<T0, T1>::type, Ts...>> {};

template<template<typename, typename> class Method, typename T0, typename T1, typename T2, typename T3, typename... Ts>
struct reduce_impl<Method, typelist<T0, T1, T2, T3, Ts...>> : public reduce_impl<Method, typelist<typename Method<T0, T1>::type, typename Method<T2, T3>::type, Ts...>> {};
} // namespace detail

template<template<typename, typename> class Method, typename List>
using reduce = typename detail::reduce_impl<Method, List>::type;

namespace detail {

template<template<typename> typename Pred, typename... Items>
struct find_type;

template<template<typename> typename Pred>
struct find_type<Pred> {
    using type = typelist<>;
};

template<template<typename> typename Pred, typename First, typename... Rest>
struct find_type<Pred, First, Rest...> {
    using type = typename std::conditional_t<Pred<First>::value, typelist<First, typename find_type<Pred, Rest...>::type>, typename find_type<Pred, Rest...>::type>;
};

template<template<typename> typename Predicate, typename DefaultType, typename... Ts>
struct find_type_or_default_impl {
    using type = DefaultType;
};

template<template<typename> typename Predicate, typename DefaultType, typename Head, typename... Ts>
struct find_type_or_default_impl<Predicate, DefaultType, Head, Ts...> : std::conditional_t<Predicate<Head>::value, find_type_or_default_impl<Predicate, Head, Ts...>, find_type_or_default_impl<Predicate, DefaultType, Ts...>> {};

template<std::size_t Index, typename... Ts>
struct at_impl;

template<typename T0, typename... Ts>
struct at_impl<0, T0, Ts...> {
    using type = T0;
};

template<typename T0, typename T1, typename... Ts>
struct at_impl<1, T0, T1, Ts...> {
    using type = T1;
};

template<typename T0, typename T1, typename T2, typename... Ts>
struct at_impl<2, T0, T1, T2, Ts...> {
    using type = T2;
};

template<typename T0, typename T1, typename T2, typename T3, typename... Ts>
struct at_impl<3, T0, T1, T2, T3, Ts...> {
    using type = T3;
};

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename... Ts>
struct at_impl<4, T0, T1, T2, T3, T4, Ts...> {
    using type = T4;
};

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename... Ts>
struct at_impl<5, T0, T1, T2, T3, T4, T5, Ts...> {
    using type = T5;
};

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename... Ts>
struct at_impl<6, T0, T1, T2, T3, T4, T5, T6, Ts...> {
    using type = T6;
};

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename... Ts>
struct at_impl<7, T0, T1, T2, T3, T4, T5, T6, T7, Ts...> {
    using type = T7;
};

template<std::size_t Index, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename... Ts>
requires(Index >= 8)
struct at_impl<Index, T0, T1, T2, T3, T4, T5, T6, T7, Ts...> : at_impl<Index - 8, Ts...> {};

} // namespace detail

// typelist /////////////////
template<typename T>
concept is_typelist_v = requires { typename T::typelist_tag; };

template<typename... Ts>
struct typelist {
    using this_t       = typelist<Ts...>;
    using typelist_tag = std::true_type;

    static inline constexpr std::integral_constant<std::size_t, sizeof...(Ts)> size = {};

    static inline constexpr auto index_sequence = std::make_index_sequence<sizeof...(Ts)>();

    template<template<typename...> class Other>
    using apply = Other<Ts...>;

    template<class F, std::size_t... Is, typename... LeadingArguments>
    static constexpr void for_each_impl(F&& f, std::index_sequence<Is...>, LeadingArguments&&... args) {
        (f(std::forward<LeadingArguments>(args)..., std::integral_constant<std::size_t, Is>{}, static_cast<Ts*>(nullptr)), ...);
    }

    template<class F, typename... LeadingArguments>
    static constexpr void for_each(F&& f, LeadingArguments&&... args) {
        for_each_impl(std::forward<F>(f), index_sequence, std::forward<LeadingArguments>(args)...);
    }

    template<std::size_t I>
    using at = detail::at_impl<I, Ts...>::type;

    template<typename... Heads>
    using prepend = typelist<Heads..., Ts...>;

    template<typename... Other>
    static constexpr inline bool are_equal = std::same_as<typelist, meta::typelist<Other...>>;

    template<typename... Other>
    static constexpr inline bool are_convertible_to = (std::convertible_to<Ts, Other> && ...);

    template<typename... Other>
    static constexpr inline bool are_convertible_from = (std::convertible_to<Other, Ts> && ...);

    template<template<typename> typename Trafo>
    using transform = meta::transform_types<Trafo, this_t>;

    template<template<typename, size_t> typename Trafo>
    using transform_with_index = meta::transform_types_indexed<Trafo, this_t>;

    template<template<typename...> typename Pred>
    constexpr static bool all_of = (Pred<Ts>::value && ...);

    template<template<typename...> typename Pred>
    constexpr static bool any_of = (Pred<Ts>::value || ...);

    template<template<typename...> typename Pred>
    constexpr static bool none_of = (!Pred<Ts>::value && ...);

    template<typename DefaultType>
    using safe_head_default = std::remove_pointer_t<decltype([] {
        if constexpr (sizeof...(Ts) > 0) {
            return static_cast<this_t::at<0>*>(nullptr);
        } else {
            return static_cast<DefaultType*>(nullptr);
        }
    }())>;

    using safe_head = std::remove_pointer_t<decltype([] {
        if constexpr (sizeof...(Ts) > 0) {
            return static_cast<this_t::at<0>*>(nullptr);
        } else {
            return static_cast<void*>(nullptr);
        }
    }())>;

    template<typename Matcher = typename this_t::safe_head>
    constexpr static bool all_same = ((std::is_same_v<Matcher, Ts> && ...));

    template<typename T, template<typename...> typename... Predicates>
    static constexpr bool eval_pred_all_of = (Predicates<T>::value and ...);

    template<template<typename...> typename... Predicates>
    using filter = concat<std::conditional_t<eval_pred_all_of<Ts, Predicates...>, typelist<Ts>, typelist<>>...>;

    template<typename ToErase>
    using erase = concat<std::conditional_t<std::is_same_v<ToErase, Ts>, typelist<>, typelist<Ts>>...>;

    template<template<typename> typename Pred>
    using find = typename detail::find_type<Pred, Ts...>::type;

    template<template<typename> typename Pred, typename DefaultType>
    using find_or_default = typename detail::find_type_or_default_impl<Pred, DefaultType, Ts...>::type;

    template<typename Needle>
    static constexpr std::size_t index_of() {
        std::size_t result = static_cast<std::size_t>(-1);
        gr::meta::typelist<Ts...>::for_each([&](auto index, auto* t) {
            if constexpr (std::is_same_v<Needle, std::remove_pointer_t<decltype(t)>>) {
                result = index;
            }
        });
        return result;
    }

    template<typename T>
    inline static constexpr bool contains = std::disjunction_v<std::is_same<T, Ts>...>;

    using tuple_type    = std::tuple<Ts...>;
    using tuple_or_type = std::remove_pointer_t<decltype([] {
        if constexpr (sizeof...(Ts) == 0) {
            return static_cast<void*>(nullptr);
        } else if constexpr (sizeof...(Ts) == 1) {
            return static_cast<at<0>*>(nullptr);
        } else {
            return static_cast<tuple_type*>(nullptr);
        }
    }())>;
};

template<typename T, typename... Ts>
constexpr bool is_any_of_v = std::disjunction_v<std::is_same<T, Ts>...>;

namespace detail {
template<template<typename...> typename OtherTypelist, typename... Args>
meta::typelist<Args...> to_typelist_helper(OtherTypelist<Args...>*);

template<typename T, size_t... Is>
meta::typelist<std::remove_pointer_t<decltype([](size_t) -> std::add_pointer_t<T> { return nullptr; }(Is))>...> array_to_typelist_helper(std::index_sequence<Is...>);

template<template<typename, size_t> typename ArrayLike, typename T, size_t N>
decltype(array_to_typelist_helper<T>(std::make_index_sequence<N>())) to_typelist_helper(ArrayLike<T, N>*);
} // namespace detail

template<typename OtherTypelist>
using to_typelist = decltype(detail::to_typelist_helper(static_cast<OtherTypelist*>(nullptr)));

static_assert(std::same_as<to_typelist<std::array<int, 3>>, meta::typelist<int, int, int>>);

namespace detail {
template<typename T>
struct flatten_impl;

template<typename... Ts>
struct flatten_impl<typelist<Ts...>> {
    using type = concat<Ts...>;
};
} // namespace detail

// Flatten a typelist of typelists into a single typelist.
template<typename T>
using flatten = typename detail::flatten_impl<T>::type;

namespace detail {
template<auto Collection, template<auto> class UnaryOp, typename = decltype(std::make_index_sequence<Collection.size()>())>
struct transform_to_typelist_impl;

template<auto Collection, template<auto> class UnaryOp, size_t... Is>
struct transform_to_typelist_impl<Collection, UnaryOp, std::index_sequence<Is...>> {
    using type = typelist<UnaryOp<Collection[Is]>...>;
};
} // namespace detail

// Build a typelist from applying UnaryOp to all elements in the given Collection.
template<auto Collection, template<auto> class UnaryOp>
using transform_to_typelist = typename detail::transform_to_typelist_impl<Collection, UnaryOp>::type;

} // namespace gr::meta

#endif // GNURADIO_TYPELIST_HPP
