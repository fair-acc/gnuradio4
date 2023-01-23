#ifndef GNURADIO_GRAPH_UTILS_HPP
#define GNURADIO_GRAPH_UTILS_HPP

#include <functional>
#include <iostream>
#include <string>
#include <string_view>

#include "typelist.hpp"
#include "vir/simd.h"

#ifndef __EMSCRIPTEN__
#include <cxxabi.h>
#include <iostream>
#include <typeinfo>
#endif

namespace fair::meta {

template<typename... Ts>
struct print_types;

template<typename CharT, std::size_t SIZE>
struct fixed_string {
    constexpr static std::size_t N            = SIZE;
    CharT                        _data[N + 1] = {};

    constexpr explicit(false) fixed_string(const CharT (&str)[N + 1]) noexcept {
        if constexpr (N != 0)
            for (std::size_t i = 0; i < N; ++i) _data[i] = str[i];
    }

    [[nodiscard]] constexpr std::size_t
    size() const noexcept {
        return N;
    }

    [[nodiscard]] constexpr bool
    empty() const noexcept {
        return N == 0;
    }

    [[nodiscard]] constexpr explicit operator std::string_view() const noexcept { return { _data, N }; }

    [[nodiscard]] explicit           operator std::string() const noexcept { return { _data, N }; }

    [[nodiscard]]                    operator const char *() const noexcept { return _data; }

    [[nodiscard]] constexpr bool
    operator==(const fixed_string &other) const noexcept {
        return std::string_view{ _data, N } == std::string_view(other);
    }

    template<std::size_t N2>
    [[nodiscard]] friend constexpr bool
    operator==(const fixed_string &, const fixed_string<CharT, N2> &) {
        return false;
    }
};

template<typename CharT, std::size_t N>
fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;

template<typename T>
[[nodiscard]] constexpr std::string
type_name() noexcept {
#ifndef __EMSCRIPTEN__
    std::string type_name = typeid(T).name();
    int         status;
    char       *demangled_name = abi::__cxa_demangle(type_name.c_str(), nullptr, nullptr, &status);
    if (status == 0) {
        return demangled_name;
    } else {
        return typeid(T).name();
    }
#else
    return typeid(T).name(); // TODO: to be replaced by refl-cpp
#endif
}

template<fixed_string val>
struct message_type {};

template<class... T>
constexpr bool always_false = false;

struct dummy_t {};

template<typename F, typename... Args>
auto
invoke_void_wrapped(F &&f, Args &&...args) {
    if constexpr (std::is_same_v<void, std::invoke_result_t<F, Args...>>) {
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
        return dummy_t{};
    } else {
        return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    }
}

static_assert(std::is_same_v<decltype(invoke_void_wrapped([] {})), dummy_t>);
static_assert(std::is_same_v<decltype(invoke_void_wrapped([] { return 42; })), int>);
static_assert(std::is_same_v<decltype(invoke_void_wrapped([](int) {}, 42)), dummy_t>);
static_assert(std::is_same_v<decltype(invoke_void_wrapped([](int i) { return i; }, 42)), int>);

#if HAVE_SOURCE_LOCATION
[[gnu::always_inline]] inline void
precondition(bool cond, const std::source_location loc = std::source_location::current()) {
    struct handle {
        [[noreturn]] static void
        failure(std::source_location const &loc) {
            std::clog << "failed precondition in " << loc.file_name() << ':' << loc.line() << ':' << loc.column() << ": `" << loc.function_name() << "`\n";
            __builtin_trap();
        }
    };

    if (not cond) [[unlikely]]
        handle::failure(loc);
}
#else
[[gnu::always_inline]] inline void
precondition(bool cond) {
    struct handle {
        [[noreturn]] static void
        failure() {
            std::clog << "failed precondition\n";
            __builtin_trap();
        }
    };

    if (not cond) [[unlikely]]
        handle::failure();
}
#endif

namespace stdx = vir::stdx;

template<typename V, typename T = void>
concept any_simd = stdx::is_simd_v<V> && (std::same_as<T, void> || std::same_as<T, typename V::value_type>);

template<typename V, typename T>
concept t_or_simd = std::same_as<V, T> || any_simd<V, T>;

template<typename T>
concept vectorizable = std::constructible_from<stdx::simd<T>>;

template<typename A, typename B>
struct wider_native_simd_size : std::conditional<(stdx::native_simd<A>::size() > stdx::native_simd<B>::size()), A, B> {};

template<typename A>
struct wider_native_simd_size<A, A> {
    using type = A;
};

template<typename V>
struct rebind_simd_helper {
    template<typename T>
    using rebind = stdx::rebind_simd_t<T, V>;
};

struct simd_load_ctor {
    template<any_simd W>
    static constexpr W
    apply(typename W::value_type const *ptr) {
        return W(ptr, stdx::element_aligned);
    }
};

template<typename List>
using reduce_to_widest_simd = stdx::native_simd<meta::reduce<wider_native_simd_size, List>>;

template<typename V, typename List>
using transform_by_rebind_simd = meta::transform_types<rebind_simd_helper<V>::template rebind, List>;

template<typename List>
using transform_to_widest_simd = transform_by_rebind_simd<reduce_to_widest_simd<List>, List>;

template<typename Node>
concept source_node = requires(Node &node, typename Node::input_port_types::tuple_type const &inputs) {
                          {
                              [](Node &n, auto &inputs) {
                                  if constexpr (Node::input_port_types::size > 0) {
                                      return []<std::size_t... Is>(Node & n_inside, auto const &tup, std::index_sequence<Is...>)->decltype(n_inside.process_one(std::get<Is>(tup)...)) { return {}; }
                                      (n, inputs, std::make_index_sequence<Node::input_port_types::size>());
                                  } else {
                                      return n.process_one();
                                  }
                              }(node, inputs)
                              } -> std::same_as<typename Node::return_type>;
                      };

template<typename Node>
concept sink_node = requires(Node &node, typename Node::input_port_types::tuple_type const &inputs) {
                        {
                            [](Node &n, auto &inputs) {
                                []<std::size_t... Is>(Node & n_inside, auto const &tup, std::index_sequence<Is...>) {
                                    if constexpr (Node::output_port_types::size > 0) {
                                        auto a [[maybe_unused]] = n_inside.process_one(std::get<Is>(tup)...);
                                    } else {
                                        n_inside.process_one(std::get<Is>(tup)...);
                                    }
                                }
                                (n, inputs, std::make_index_sequence<Node::input_port_types::size>());
                            }(node, inputs)
                        };
                    };

template<typename Node>
concept any_node = source_node<Node> || sink_node<Node>;

template<typename Node>
concept node_can_process_simd = any_node<Node> && requires(Node &n, typename transform_to_widest_simd<typename Node::input_port_types>::template apply<std::tuple> const &inputs) {
                                                      {
                                                          []<std::size_t... Is>(Node & n, auto const &tup, std::index_sequence<Is...>)->decltype(n.process_one(std::get<Is>(tup)...)) { return {}; }
                                                          (n, inputs, std::make_index_sequence<Node::input_port_types::size>())
                                                          } -> any_simd<typename Node::return_type>;
                                                  };

template<fixed_string Name, typename PortList>
consteval int
indexForName() {
    auto helper = []<std::size_t... Ids>(std::index_sequence<Ids...>) {
        int result = -1;
        ((PortList::template at<Ids>::static_name() == Name ? (result = Ids) : 0), ...);
        return result;
    };
    return helper(std::make_index_sequence<PortList::size>());
}

template<typename... Lambdas>
struct overloaded : Lambdas... {
    using Lambdas::operator()...;
};

template<typename... Lambdas>
overloaded(Lambdas...) -> overloaded<Lambdas...>;

namespace detail {
template<template<typename...> typename Mapper, template<typename...> typename Wrapper, typename... Args>
Wrapper<Mapper<Args>...> *
type_transform_impl(Wrapper<Args...> *);

template<template<typename...> typename Mapper, typename T>
Mapper<T> *
type_transform_impl(T *);

template<template<typename...> typename Mapper>
void *
type_transform_impl(void *);

template<template<typename...> typename Mapper>
fair::meta::dummy_t *
type_transform_impl(fair::meta::dummy_t *);
} // namespace detail

template<template<typename...> typename Mapper, typename T>
using type_transform = std::remove_pointer_t<decltype(detail::type_transform_impl<Mapper>(static_cast<T *>(nullptr)))>;

static_assert(std::is_same_v<std::vector<int>, type_transform<std::vector, int>>);
static_assert(std::is_same_v<std::tuple<std::vector<int>, std::vector<float>>, type_transform<std::vector, std::tuple<int, float>>>);
static_assert(std::is_same_v<void, type_transform<std::vector, void>>);
static_assert(std::is_same_v<dummy_t, type_transform<std::vector, dummy_t>>);

} // namespace fair::meta

#endif // include guard
