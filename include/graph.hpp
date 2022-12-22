#ifndef GRAPH_PROTOTYPE_GRAPH_HPP
#define GRAPH_PROTOTYPE_GRAPH_HPP

#ifndef GRAPH_PROTOTYPE_TYPELIST_HPP
#include "typelist.hpp"
#endif

#include "vir/simd.h"

#include <iostream>
#include <ranges>
#include <tuple>

#if !__has_include(<source_location>)
#define HAVE_SOURCE_LOCATION 0
#else
#include <source_location>
#if defined __cpp_lib_source_location && __cpp_lib_source_location >= 201907L
#define HAVE_SOURCE_LOCATION 1
#else
#define HAVE_SOURCE_LOCATION 0
#endif
#endif

namespace fair::graph {
using std::size_t;

namespace stdx = vir::stdx;

namespace detail {
#if HAVE_SOURCE_LOCATION
[[gnu::always_inline]] inline void
precondition(bool cond, const std::source_location loc = std::source_location::current()) {
    struct handle {
        [[noreturn]] static void
        failure(std::source_location const &loc) {
            std::clog << "failed precondition in " << loc.file_name() << ':' << loc.line() << ':'
                      << loc.column() << ": `" << loc.function_name() << "`\n";
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

template<typename V, typename T = void>
concept any_simd = stdx::is_simd_v<V>
                && (std::same_as<T, void> || std::same_as<T, typename V::value_type>);

template<typename V, typename T>
concept t_or_simd = std::same_as<V, T> || any_simd<V, T>;

template<typename T>
concept vectorizable = std::constructible_from<stdx::simd<T>>;

template<typename A, typename B>
struct wider_native_simd_size
    : std::conditional<(stdx::native_simd<A>::size() > stdx::native_simd<B>::size()), A, B> {};

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
    template<detail::any_simd W>
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

// Constrain to the static description of a port.
template<typename P>
concept port_desc = true;

// Constrain to any type usable as template argument to make_node
template<typename Impl>
concept node_impl = requires {
    typename Impl::input_ports;
    typename Impl::output_ports;
} && requires(Impl &n, typename Impl::input_ports::tuple_type const &inputs) {
    {
        []<size_t... Is>(Impl &n, auto const &tup, std::index_sequence<Is...>)
                -> decltype(n.process_one(std::get<Is>(tup)...)) {
            return {};
        }(n, inputs, std::make_index_sequence<Impl::input_ports::size>())
    } -> std::same_as<typename Impl::output_ports::tuple_or_type>;
};

template<typename Node>
concept any_node
        = requires(Node                                                              &n,
                   typename std::remove_cvref_t<Node>::input_ports::tuple_type const &inputs) {
              { n.in } -> std::same_as<typename std::remove_cvref_t<Node>::input_ports const &>;
              { n.out } -> std::same_as<typename std::remove_cvref_t<Node>::output_ports const &>;
              {
                  []<size_t... Is>(Node &n, auto const &tup, std::index_sequence<Is...>)
                          -> decltype(n.process_one(std::get<Is>(tup)...)) {
                      return {};
                  }(n, inputs,
                          std::make_index_sequence<std::remove_cvref_t<Node>::input_ports::size>())
              } -> std::same_as<typename std::remove_cvref_t<Node>::return_type>;
          };

template<typename Impl>
concept impl_can_process_simd
        = requires(Impl &n,
                   typename transform_to_widest_simd<typename Impl::input_ports::typelist>::
                           template apply<std::tuple> const &inputs) {
              {
                  []<size_t... Is>(Impl &n, auto const &tup, std::index_sequence<Is...>)
                          -> decltype(n.process_one(std::get<Is>(tup)...)) {
                      return {};
                  }(n, inputs, std::make_index_sequence<Impl::input_ports::size>())
              } -> detail::any_simd<typename Impl::output_ports::tuple_or_type>;
          };

// Workaround bug in Clang:
// std::is_constructible should be a valid template argument for the template template parameter
// `template <typename> class T`, but because std::is_constructible<T, Args...> has a parameter
// pack, Clang considers the program ill-formed. As so often, another indirection solves the
// problem. *Sigh*
template <typename T>
using is_constructible = std::is_constructible<T>;
} // namespace detail

enum class port_direction { in, out };

template<port_direction D, int I, typename T>
struct port_id {
    using type                                       = T;
    static inline constexpr port_direction direction = D;
    static inline constexpr int            id        = I;
};

template<port_direction D, typename... Types>
struct portlist {
    static inline constexpr port_direction                                direction = D;
    static inline constexpr std::integral_constant<int, sizeof...(Types)> size      = {};
    //
    using typelist      = meta::typelist<Types...>;
    using tuple_type    = std::tuple<Types...>;
    using tuple_or_type = std::conditional_t<sizeof...(Types) == 1,
                                             typename typelist::template at<0>, tuple_type>;

    template<int I>
    using at = port_id<D, I, typename typelist::template at<I>>;

    template<typename I>
    constexpr at<I::value>
    operator[](I) const {
        return {};
    }

    template<typename... Other>
    static constexpr inline bool are_equal = std::same_as<typelist, meta::typelist<Other...>>;

    template<typename... Other>
    static constexpr inline bool are_convertible_to = (std::convertible_to<Types, Other> && ...);

    template<typename... Other>
    static constexpr inline bool are_convertible_from = (std::convertible_to<Other, Types> && ...);
};

template<port_direction D, typename... Types>
struct portlist<D, meta::typelist<Types...>> : portlist<D, Types...> {};

template<typename... Types>
using make_input_ports = portlist<port_direction::in, Types...>;

template<typename... Types>
using make_output_ports = portlist<port_direction::out, Types...>;

// simple non-reentrant circular buffer
template<typename T, size_t Size>
class port_data {
    static_assert(std::has_single_bit(Size), "Size must be a power-of-2 value");
    alignas(64) std::array<T, Size> m_buffer      = {};
    size_t                         m_read_offset  = 0;
    size_t                         m_write_offset = 0;

    static inline constexpr size_t s_bitmask      = Size - 1;

public:
    static inline constexpr std::integral_constant<size_t, Size> size = {};

    size_t
    can_read() const {
        return m_write_offset >= m_read_offset ? m_write_offset - m_read_offset
                                               : size - m_read_offset;
    }

    size_t
    can_write() const {
        return m_write_offset >= m_read_offset ? size - m_write_offset
                                               : m_read_offset - m_write_offset;
    }

    std::span<const T>
    request_read() {
        return request_read(can_read());
    }

    std::span<const T>
    request_read(size_t n) {
        detail::precondition(can_read() >= n);
        const auto begin = m_buffer.begin() + m_read_offset;
        m_read_offset += n;
        m_read_offset &= s_bitmask;
        return std::span<const T>{ begin, n };
    }

    std::span<T>
    request_write() {
        return request_write(can_write());
    }

    std::span<T>
    request_write(size_t n) {
        detail::precondition(can_write() >= n);
        const auto begin = m_buffer.begin() + m_write_offset;
        m_write_offset += n;
        m_write_offset &= s_bitmask;
        return std::span<T>{ begin, n };
    }
};

namespace detail {
template<node_impl Impl>
class node : public Impl {
public:
    using input_ports                        = typename Impl::input_ports;
    using output_ports                       = typename Impl::output_ports;
    using return_type                        = typename output_ports::tuple_or_type;

    static inline constexpr input_ports  in  = {};
    static inline constexpr output_ports out = {};

    using Impl::Impl;

    template<std::size_t N>
    [[gnu::always_inline]] constexpr bool
    process_batch_simd_epilogue(size_t n, auto out_ptr, auto... in_ptr) {
        if constexpr (N == 0) return true;
        else if (N <= n) {
            using In0 = meta::first_type<typename input_ports::typelist>;
            using V   = stdx::resize_simd_t<N, stdx::native_simd<In0>>;
            using Vs  = meta::transform_types<detail::rebind_simd_helper<V>::template rebind,
                                             typename input_ports::typelist>;
            const std::tuple input_simds = Vs::template construct<detail::simd_load_ctor>(
                    std::tuple{ in_ptr... });
            const stdx::simd result = std::apply(
                    [this](auto... args) {
                        return this->Impl::process_one(args...);
                    },
                    input_simds);
            result.copy_to(out_ptr, stdx::element_aligned);
            return process_batch_simd_epilogue<N / 2>(n - N, out_ptr + N, (in_ptr + N)...);
        } else
            return process_batch_simd_epilogue<N / 2>(n, out_ptr, in_ptr...);
    }

    // If Impl::process_one is const-qualified, then this process_batch should also be
    // const-qualified. Can't easily do that. This overload works around the issue.
    // The two overloads can be simplified using 'deducing this' (C++23) together with a suitable
    // requires clause.
    template<std::ranges::forward_range... Ins>
        requires node_impl<const Impl> && (std::ranges::sized_range<Ins> && ...)
              && input_ports::template
    are_equal<std::ranges::range_value_t<Ins>...> constexpr bool process_batch(
            port_data<return_type, 1024> &out, Ins &&...inputs) const {
        return const_cast<node *>(this)->process_batch(out, std::forward<Ins>(inputs)...);
    }

    template<std::ranges::forward_range... Ins>
        requires(std::ranges::sized_range<Ins> && ...) && input_ports::template
    are_equal<std::ranges::range_value_t<Ins>...> constexpr bool
    process_batch(port_data<return_type, 1024> &out, Ins &&...inputs) {
        const auto  &in0 = std::get<0>(std::tie(inputs...));
        const size_t n   = std::ranges::size(in0);
        detail::precondition(((n == std::ranges::size(inputs)) && ...));
        auto &&out_range = out.request_write(n);
        // if SIMD makes sense (i.e. input and output ranges are contiguous and all types are
        // vectorizable)
        if constexpr ((std::ranges::contiguous_range<decltype(out_range)> && ...
                       && std::ranges::contiguous_range<Ins>) &&detail::vectorizable<return_type>
                      && detail::impl_can_process_simd<Impl>
                      && meta::transform_types<detail::is_constructible,
                                               meta::transform_types<stdx::native_simd,
                                                                     typename input_ports::typelist>>::
                              template apply<std::conjunction>::value) {
            using V  = detail::reduce_to_widest_simd<typename input_ports::typelist>;
            using Vs = detail::transform_by_rebind_simd<V, typename input_ports::typelist>;
            size_t i = 0;
            for (i = 0; i + V::size() <= n; i += V::size()) {
                const std::tuple input_simds = Vs::template construct<detail::simd_load_ctor>(
                        std::tuple{ (std::ranges::data(inputs) + i)... });
                const stdx::simd result = std::apply(
                        [this](auto... args) {
                            return this->Impl::process_one(args...);
                        },
                        input_simds);
                result.copy_to(std::ranges::data(out_range) + i, stdx::element_aligned);
            }

            return process_batch_simd_epilogue<std::bit_ceil(V::size())
                                               / 2>(n - i, std::ranges::data(out_range) + i,
                                                    (std::ranges::data(inputs) + i)...);
        } else { // no explicit SIMD
            auto             out_it    = out_range.begin();
            std::tuple       it_tuple  = { std::ranges::begin(inputs)... };
            const std::tuple end_tuple = { std::ranges::end(inputs)... };
            while (std::get<0>(it_tuple) != std::get<0>(end_tuple)) {
                *out_it = std::apply(
                        [this](auto &...its) {
                            return this->Impl::process_one((*its++)...);
                        },
                        it_tuple);
                ++out_it;
            }
            return true;
        }
    }
};

template<detail::node_impl Left, int OutId, detail::node_impl Right, int InId>
class merged_node_impl {
public:
    using input_ports = make_input_ports<
            meta::concat<typename Left::input_ports::typelist,
                         meta::remove_at<InId, typename Right::input_ports::typelist>>>;
    using output_ports = make_output_ports<
            meta::concat<meta::remove_at<OutId, typename Left::output_ports::typelist>,
                         typename Right::output_ports::typelist>>;
    using return_type = typename output_ports::tuple_or_type;

    Left  left;
    Right right;

    constexpr merged_node_impl(Left l, Right r) : left(std::move(l)), right(std::move(r)) {}

    constexpr merged_node_impl(Left l)
        requires std::is_default_constructible_v<Right>
        : left(std::move(l))
        , right() {}

    constexpr merged_node_impl(Right r)
        requires std::is_default_constructible_v<Left>
        : left()
        , right(std::move(r)) {}

    constexpr merged_node_impl()
        requires std::is_default_constructible_v<Left> && std::is_default_constructible_v<Right>
    = default;

    template<detail::any_simd... Ts>
        requires detail::vectorizable<return_type> && input_ports::template
    are_equal<typename std::remove_cvref_t<Ts>::value_type...> &&detail::impl_can_process_simd<Left>
            &&detail::impl_can_process_simd<Right> constexpr stdx::rebind_simd_t<
                    return_type, meta::first_type<meta::typelist<std::remove_cvref_t<Ts>...>>>
            process_one(Ts... inputs) {
        return apply_right(std::tie(inputs...),
                           apply_left(std::tie(inputs...),
                                      std::make_index_sequence<Left::input_ports::size()>()),
                           std::make_index_sequence<InId>(),
                           std::make_index_sequence<Right::input_ports::size() - InId - 1>());
    }

    template<typename... Ts>
        requires node_impl<const Left> && node_impl<const Right> && input_ports::template
    are_equal<std::remove_cvref_t<Ts>...> constexpr typename output_ports::tuple_or_type
    process_one(Ts &&...inputs) const {
        return const_cast<merged_node_impl *>(this)->process_one(std::forward<Ts>(inputs)...);
    }

    template<typename... Ts>
        requires input_ports::template
    are_equal<std::remove_cvref_t<Ts>...> constexpr typename output_ports::tuple_or_type
    process_one(Ts &&...inputs) {
        if constexpr (Left::output_ports::size
                      == 1) { // only the result from the right node needs to be returned
            return apply_right(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                               apply_left(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                          std::make_index_sequence<Left::input_ports::size()>()),
                               std::make_index_sequence<InId>(),
                               std::make_index_sequence<Right::input_ports::size() - InId - 1>());
        } else {
            // left produces a tuple
            auto left_out = apply_left(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                       std::make_index_sequence<Left::input_ports::size()>());
            auto right_out
                    = apply_right(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                  std::move(std::get<OutId>(left_out)),
                                  std::make_index_sequence<InId>(),
                                  std::make_index_sequence<Right::input_ports::size() - InId - 1>());

            if constexpr (Left::output_ports::size == 2 && Right::output_ports::size == 1)
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)),
                                       std::move(right_out));
            else if constexpr (Left::output_ports::size == 2)
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))),
                                      std::move(right_out));
            else if constexpr (Right::output_ports::size == 1)
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>,
                                                                 std::index_sequence<Js...>) {
                    return std::make_tuple(std::move(std::get<Is...>(left_out)),
                                           std::move(std::get<OutId + 1 + Js...>(left_out)),
                                           std::move(right_out));
                }(std::make_index_sequence<OutId>(),
                       std::make_index_sequence<Left::output_ports::size - OutId - 1>());
            else
                return [&]<std::size_t... Is, std::size_t... Js,
                           std::size_t... Ks>(std::index_sequence<Is...>,
                                              std::index_sequence<Js...>,
                                              std::index_sequence<Ks...>) {
                    return std::make_tuple(std::move(std::get<Is...>(left_out)),
                                           std::move(std::get<OutId + 1 + Js...>(left_out)),
                                           std::move(std::get<Ks...>(right_out)));
                }(std::make_index_sequence<OutId>(),
                       std::make_index_sequence<Left::output_ports::size - OutId - 1>(),
                       std::make_index_sequence<Right::output_ports::size>());
        }
    }

private:
    template<std::size_t... Is>
    [[gnu::always_inline]] constexpr auto
    apply_left(auto &&input_tuple, std::index_sequence<Is...>) {
        return left.process_one(std::get<Is>(input_tuple)...);
    }

    template<std::size_t... Is, std::size_t... Js>
    [[gnu::always_inline]] constexpr auto
    apply_right(auto &&input_tuple, auto &&tmp, std::index_sequence<Is...>,
                std::index_sequence<Js...>) {
        constexpr int first_offset  = Left::input_ports::size;
        constexpr int second_offset = Left::input_ports::size + sizeof...(Is);
        static_assert(second_offset + sizeof...(Js)
                      == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
        return right.process_one(std::get<first_offset + Is>(input_tuple)..., std::move(tmp),
                                 std::get<second_offset + Js>(input_tuple)...);
    }
};

template<any_node T>
struct node_impl_of {};

template<node_impl Impl>
struct node_impl_of<node<Impl>> {
    using type = Impl;
};

} // namespace detail

template<detail::node_impl Impl>
using make_node = detail::node<Impl>;

template<detail::any_node Left, int OutId, detail::any_node Right, int InId>
    requires std::same_as<typename Left::output_ports::template at<OutId>::type,
                          typename Right::input_ports::template at<InId>::type>
using merged_node
        = make_node<detail::merged_node_impl<typename detail::node_impl_of<Left>::type, OutId,
                                             typename detail::node_impl_of<Right>::type, InId>>;

} // namespace fair::graph

#endif // GRAPH_PROTOTYPE_GRAPH_HPP
