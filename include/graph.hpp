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

template<typename Node>
concept node_can_process_simd
        = any_node<Node>
       && requires(Node &n,
                   typename transform_to_widest_simd<typename std::remove_cvref_t<
                           Node>::input_ports::typelist>::template apply<std::tuple> const &inputs) {
              {
                  []<size_t... Is>(Node &n, auto const &tup, std::index_sequence<Is...>)
                          -> decltype(n.process_one(std::get<Is>(tup)...)) {
                      return {};
                  }(n, inputs,
                          std::make_index_sequence<std::remove_cvref_t<Node>::input_ports::size>())
              } -> detail::any_simd<typename std::remove_cvref_t<Node>::return_type>;
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

template<typename Derived, typename InputPorts, typename OutputPorts>
class node {
public:
    using input_ports                        = InputPorts;
    using output_ports                       = OutputPorts;
    using return_type                        = typename output_ports::tuple_or_type;

    static inline constexpr input_ports  in  = {};
    static inline constexpr output_ports out = {};

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
                        return static_cast<Derived *>(this)->process_one(args...);
                    },
                    input_simds);
            result.copy_to(out_ptr, stdx::element_aligned);
            return process_batch_simd_epilogue<N / 2>(n - N, out_ptr + N, (in_ptr + N)...);
        } else
            return process_batch_simd_epilogue<N / 2>(n, out_ptr, in_ptr...);
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
                      && detail::node_can_process_simd<Derived>
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
                            return static_cast<Derived *>(this)->process_one(args...);
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
                            return static_cast<Derived *>(this)->process_one((*its++)...);
                        },
                        it_tuple);
                ++out_it;
            }
            return true;
        }
    }

protected:
    constexpr node() noexcept = default;
};

template<detail::any_node Left, detail::any_node Right, int OutId, int InId>
class merged_node
    : public node<merged_node<Left, Right, OutId, InId>,
                  make_input_ports<
                          meta::concat<typename Left::input_ports::typelist,
                                       meta::remove_at<InId, typename Right::input_ports::typelist>>>,
                  make_output_ports<
                          meta::concat<meta::remove_at<OutId, typename Left::output_ports::typelist>,
                                       typename Right::output_ports::typelist>>> {
private:
    using base = node<
            merged_node,
            make_input_ports<
                    meta::concat<typename Left::input_ports::typelist,
                                 meta::remove_at<InId, typename Right::input_ports::typelist>>>,
            make_output_ports<meta::concat<meta::remove_at<OutId, typename Left::output_ports::typelist>,
                                           typename Right::output_ports::typelist>>>;

    Left  left;
    Right right;

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

public:
    using input_ports  = typename base::input_ports;
    using output_ports = typename base::output_ports;
    using return_type  = typename output_ports::tuple_or_type;

    [[gnu::always_inline]] constexpr merged_node(Left l, Right r)
        : left(std::move(l)), right(std::move(r)) {}

    template<detail::any_simd... Ts>
        requires detail::vectorizable<return_type> && input_ports::template
    are_equal<typename std::remove_cvref_t<Ts>::value_type...> &&detail::node_can_process_simd<Left>
            &&detail::node_can_process_simd<Right> constexpr stdx::rebind_simd_t<
                    return_type, meta::first_type<meta::typelist<std::remove_cvref_t<Ts>...>>>
            process_one(Ts... inputs) {
        return apply_right(std::tie(inputs...),
                           apply_left(std::tie(inputs...),
                                      std::make_index_sequence<Left::input_ports::size()>()),
                           std::make_index_sequence<InId>(),
                           std::make_index_sequence<Right::input_ports::size() - InId - 1>());
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
};

template<int OutId, int InId, detail::any_node A, detail::any_node B>
    requires std::same_as<typename std::remove_cvref_t<A>::output_ports::template at<OutId>::type,
                          typename std::remove_cvref_t<B>::input_ports::template at<InId>::type>
[[gnu::always_inline]] constexpr auto
merge(A &&a, B &&b) -> merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    return { std::forward<A>(a), std::forward<B>(b) };
}
} // namespace fair::graph

#endif // GRAPH_PROTOTYPE_GRAPH_HPP
