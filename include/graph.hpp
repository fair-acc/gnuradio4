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
    template<typename CharT, std::size_t SIZE>
    struct fixed_string {
        constexpr static std::size_t N = SIZE;
        CharT _data[N + 1] = {};

        constexpr explicit(false) fixed_string(const CharT (&str)[N + 1]) noexcept {
            if constexpr (N != 0) for (std::size_t i = 0; i < N; ++i) _data[i] = str[i];
        }

        [[nodiscard]] constexpr std::size_t size() const noexcept { return N; }
        [[nodiscard]] constexpr bool empty() const noexcept { return N == 0; }
        [[nodiscard]] constexpr explicit operator std::string_view() const noexcept { return {_data, N}; }
        [[nodiscard]] constexpr explicit operator std::string() const noexcept { return {_data, N}; }
        [[nodiscard]] operator const char *() const noexcept { return _data; }
        [[nodiscard]] constexpr bool operator==(const fixed_string &other) const noexcept {
            return std::string_view{_data, N} == std::string_view(other);
        }

        template<std::size_t N2>
        [[nodiscard]] friend constexpr bool operator==(const fixed_string &, const fixed_string<CharT, N2> &) { return false; }
    };
    template<typename CharT, std::size_t N>
    fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;
} // namespace fair::graph


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
concept any_node = requires(Node &n, typename Node::input_port_types::tuple_type const &inputs) {
    { n.in } -> std::same_as<typename Node::input_port_types const &>;
    { n.out } -> std::same_as<typename Node::output_port_types const &>;
    {
        []<std::size_t... Is>(Node &n, auto const &tup, std::index_sequence<Is...>)
                -> decltype(n.process_one(std::get<Is>(tup)...)) {
            return {};
        }(n, inputs, std::make_index_sequence<Node::input_port_types::size>())
    } -> std::same_as<typename Node::return_type>;
};

template<typename Node>
concept node_can_process_simd
        = any_node<Node>
       && requires(Node &n,
                   typename transform_to_widest_simd<typename Node::input_port_types>::
                           template apply<std::tuple> const &inputs) {
              {
                  []<std::size_t... Is>(Node &n, auto const &tup, std::index_sequence<Is...>)
                          -> decltype(n.process_one(std::get<Is>(tup)...)) {
                      return {};
                  }(n, inputs, std::make_index_sequence<Node::input_port_types::size>())
              } -> detail::any_simd<typename Node::return_type>;
          };

} // namespace detail

enum class port_direction { in, out };

template<port_direction D, fixed_string Name, typename T>
struct port_id {
    using port_id_tag = std::true_type;
    using type                                       = T;
    static inline constexpr port_direction direction = D;
    static inline constexpr fixed_string   name      = Name;
};

template<typename T>
concept is_port_id_v = requires { typename T::port_id_tag; };

template<typename T>
using is_port_id = std::conditional_t<is_port_id_v<T>, std::true_type, std::false_type>;

template <typename T>
struct is_port_id_or_list : std::false_type {};

template <typename T>
    requires is_port_id_v<T>
struct is_port_id_or_list<T> : std::true_type {};

template <typename T>
    requires (meta::is_typelist_v<T> and T::template all_of<is_port_id>)
struct is_port_id_or_list<T> : std::true_type {};



template<typename Port>
using port_id_type = typename Port::type;

template<typename Port>
    requires is_port_id_v<Port>
struct port_id_name {
    static inline constexpr fixed_string name = Port::name;
};

template<fixed_string Name, typename T>
using in = port_id<port_direction::in, Name, T>;

template<typename Port>
using is_in_port = std::integral_constant<bool, Port::direction == port_direction::in>;

template<fixed_string Name, typename T>
using out = port_id<port_direction::out, Name, T>;

template<typename Port>
using is_out_port = std::integral_constant<bool, Port::direction == port_direction::out>;

namespace detail {
    template<typename T, auto>
    using just_t = T;

    template<typename T, std::size_t... Is>
    consteval fair::meta::typelist<just_t<T, Is>...>
    repeated_ports_impl(std::index_sequence<Is...>) {
        return {};
    }

} // namespace detail

// TODO: Add port index to BaseName
template<std::size_t Count, port_direction direction, fixed_string BaseName, typename T>
using repeated_ports =
    decltype(detail::repeated_ports_impl<port_id<direction, BaseName, T>>(std::make_index_sequence<Count>()));

// simple non-reentrant circular buffer
template<typename T, std::size_t Size>
class port_data {
    static_assert(std::has_single_bit(Size), "Size must be a power-of-2 value");
    alignas(64) std::array<T, Size> m_buffer           = {};
    std::size_t                         m_read_offset  = 0;
    std::size_t                         m_write_offset = 0;

    static inline constexpr std::size_t s_bitmask      = Size - 1;

public:
    static inline constexpr std::integral_constant<std::size_t, Size> size = {};

    std::size_t
    can_read() const {
        return m_write_offset >= m_read_offset ? m_write_offset - m_read_offset
                                               : size - m_read_offset;
    }

    std::size_t
    can_write() const {
        return m_write_offset >= m_read_offset ? size - m_write_offset
                                               : m_read_offset - m_write_offset;
    }

    std::span<const T>
    request_read() {
        return request_read(can_read());
    }

    std::span<const T>
    request_read(std::size_t n) {
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
    request_write(std::size_t n) {
        detail::precondition(can_write() >= n);
        const auto begin = m_buffer.begin() + m_write_offset;
        m_write_offset += n;
        m_write_offset &= s_bitmask;
        return std::span<T>{ begin, n };
    }
};

template<typename...>
struct node_ports_data;

template<meta::is_typelist_v InputPorts, meta::is_typelist_v OutputPorts>
    requires InputPorts::template all_of<is_port_id>
          && OutputPorts::template all_of<is_port_id>
struct node_ports_data<InputPorts, OutputPorts> {
    using input_ports = InputPorts;
    using output_ports = OutputPorts;

    using input_port_types =
        typename input_ports
        ::template transform<port_id_type>;
    using output_port_types =
        typename output_ports
        ::template transform<port_id_type>;

    using all_ports = meta::concat<input_ports, output_ports>;

    static_assert(all_ports::size > 0);
};

template<is_port_id_v... Ports>
struct node_ports_data<Ports...> {
    using all_ports = meta::concat<
        std::conditional_t<fair::meta::is_typelist_v<Ports>,
            Ports,
            meta::typelist<Ports>
        >...>;

    using input_ports =
        typename all_ports
        ::template filter<is_in_port>;
    using output_ports =
        typename all_ports
        ::template filter<is_out_port>;

    using input_port_types =
        typename input_ports
        ::template transform<port_id_type>;
    using output_port_types =
        typename output_ports
        ::template transform<port_id_type>;

    static_assert(all_ports::size > 0);
};

// Ports can either be a list of port_id instances,
// or two typelists containing port_id instances -- one for input
// ports and one for output ports
template<typename Derived, typename... Arguments>
class node: public meta::typelist<Arguments...>
                       ::template filter<is_port_id_or_list>
                       ::template apply<node_ports_data> {
public:
    using base = typename meta::typelist<Arguments...>
                     ::template filter<is_port_id_or_list>
                     ::template apply<node_ports_data>;

    using all_ports = typename base::all_ports;
    using input_ports = typename base::input_ports;
    using output_ports = typename base::output_ports;
    using input_port_types = typename base::input_port_types;
    using output_port_types = typename base::output_port_types;

    using return_type = typename output_port_types::tuple_or_type;

    static inline constexpr input_port_types  in  = {};
    static inline constexpr output_port_types out = {};

    Derived* self() { return this; }
    const Derived* self() const { return this; }

    template<std::size_t N>
    [[gnu::always_inline]] constexpr bool
    process_batch_simd_epilogue(std::size_t n, auto out_ptr, auto... in_ptr) {
        if constexpr (N == 0) return true;
        else if (N <= n) {
            using In0 = meta::first_type<input_port_types>;
            using V   = stdx::resize_simd_t<N, stdx::native_simd<In0>>;
            using Vs  = meta::transform_types<detail::rebind_simd_helper<V>::template rebind,
                                              input_port_types>;
            const std::tuple input_simds = Vs::template construct<detail::simd_load_ctor>(
                    std::tuple{ in_ptr... });
            const stdx::simd result = std::apply(
                    [this](auto... args) {
                        return self()->process_one(args...);
                    },
                    input_simds);
            result.copy_to(out_ptr, stdx::element_aligned);
            return process_batch_simd_epilogue<N / 2>(n, out_ptr + N, (in_ptr + N)...);
        } else
            return process_batch_simd_epilogue<N / 2>(n, out_ptr, in_ptr...);
    }

    template<std::ranges::forward_range... Ins>
        requires(std::ranges::sized_range<Ins> && ...)
        && input_port_types::template are_equal<std::ranges::range_value_t<Ins>...>
    constexpr bool
    process_batch(port_data<return_type, 1024> &out, Ins &&...inputs) {
        const auto  &in0    = std::get<0>(std::tie(inputs...));
        const std::size_t n = std::ranges::size(in0);
        detail::precondition(((n == std::ranges::size(inputs)) && ...));
        auto &&out_range = out.request_write(n);
        // if SIMD makes sense (i.e. input and output ranges are contiguous and all types are
        // vectorizable)
        if constexpr ((std::ranges::contiguous_range<decltype(out_range)> && ... && std::ranges::contiguous_range<Ins>)
                      && detail::vectorizable<return_type>
                      && detail::node_can_process_simd<Derived>
                      && input_port_types
                         ::template transform<stdx::native_simd>
                         ::template all_of<std::is_constructible>) {
            using V  = detail::reduce_to_widest_simd<input_port_types>;
            using Vs = detail::transform_by_rebind_simd<V, input_port_types>;
            std::size_t i = 0;
            for (i = 0; i + V::size() <= n; i += V::size()) {
                const std::tuple input_simds = Vs::template construct<detail::simd_load_ctor>(
                        std::tuple{ (std::ranges::data(inputs) + i)... });
                const stdx::simd result = std::apply(
                        [this](auto... args) {
                            return self()->process_one(args...);
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
                            return self()->process_one((*its++)...);
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

template<
    detail::any_node Left,
    detail::any_node Right,
    int OutId,
    int InId
    >
class merged_node
    : public node<
        merged_node<Left, Right, OutId, InId>,
        meta::concat<
            typename Left::input_ports,
            meta::remove_at<InId, typename Right::input_ports>
        >,
        meta::concat<
            meta::remove_at<OutId, typename Left::output_ports>,
            typename Right::output_ports
        >
    >{
private:
    // copy-paste from above, keep in sync
    using base = node<
        merged_node<Left, Right, OutId, InId>,
        meta::concat<
            typename Left::input_ports,
            meta::remove_at<InId, typename Right::input_ports>
        >,
        meta::concat<
            meta::remove_at<OutId, typename Left::output_ports>,
            typename Right::output_ports
        >
    >;

    Left  left;
    Right right;

    template<std::size_t I>
    [[gnu::always_inline]] constexpr auto
    apply_left(auto &&input_tuple) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return left.process_one(std::get<Is>(input_tuple)...);
        } (std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    [[gnu::always_inline]] constexpr auto
    apply_right(auto &&input_tuple, auto &&tmp) {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr int first_offset  = Left::input_port_types::size;
            constexpr int second_offset = Left::input_port_types::size + sizeof...(Is);
            static_assert(second_offset + sizeof...(Js)
                          == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return right.process_one(std::get<first_offset + Is>(input_tuple)..., std::move(tmp),
                                     std::get<second_offset + Js>(input_tuple)...);
        } (std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    using input_port_types  = typename base::input_port_types;
    using output_port_types = typename base::output_port_types;
    using return_type  = typename base::return_type;

    [[gnu::always_inline]] constexpr merged_node(Left l, Right r)
        : left(std::move(l)), right(std::move(r)) {}

    template<detail::any_simd... Ts>
        requires detail::vectorizable<return_type>
        && input_port_types::template are_equal<typename std::remove_cvref_t<Ts>::value_type...>
        && detail::node_can_process_simd<Left>
        && detail::node_can_process_simd<Right>
    constexpr stdx::rebind_simd_t<return_type, meta::first_type<meta::typelist<std::remove_cvref_t<Ts>...>>>
    process_one(Ts... inputs) {
        return apply_right<InId, Right::input_port_types::size() - InId - 1>
            (std::tie(inputs...), apply_left<Left::input_port_types::size()>(std::tie(inputs...)));
    }

    template<typename... Ts>
        // In order to have nicer error messages, this is checked in the function body
        // requires input_port_types::template are_equal<std::remove_cvref_t<Ts>...>
    constexpr return_type
    process_one(Ts &&...inputs) {
        if constexpr (!input_port_types::template are_equal<std::remove_cvref_t<Ts>...>) {
            meta::print_types<
                decltype(this),
                input_port_types,
                std::remove_cvref_t<Ts>...> error{};
        }

        if constexpr (Left::output_port_types::size
                      == 1) { // only the result from the right node needs to be returned
            return apply_right<InId, Right::input_port_types::size() - InId - 1>
                (std::forward_as_tuple(std::forward<Ts>(inputs)...),
                               apply_left<Left::input_port_types::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out = apply_left<Left::input_port_types::size()>
                            (std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, Right::input_port_types::size() - InId - 1>
                             (std::forward_as_tuple(std::forward<Ts>(inputs)...), std::move(std::get<OutId>(left_out)));

            if constexpr (Left::output_port_types::size == 2 && Right::output_port_types::size == 1) {
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)),
                                       std::move(right_out));

            } else if constexpr (Left::output_port_types::size == 2) {
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))),
                                      std::move(right_out));

            } else if constexpr (Right::output_port_types::size == 1) {
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>,
                                                                 std::index_sequence<Js...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))...,
                                           std::move(std::get<OutId + 1 + Js>(left_out))...,
                                           std::move(right_out));
                }(std::make_index_sequence<OutId>(),
                  std::make_index_sequence<Left::output_port_types::size - OutId - 1>());

            } else {
                return [&]<std::size_t... Is, std::size_t... Js,
                           std::size_t... Ks>(std::index_sequence<Is...>,
                                              std::index_sequence<Js...>,
                                              std::index_sequence<Ks...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))...,
                                           std::move(std::get<OutId + 1 + Js>(left_out))...,
                                           std::move(std::get<Ks>(right_out)...));
                }(std::make_index_sequence<OutId>(),
                  std::make_index_sequence<Left::output_port_types::size - OutId - 1>(),
                  std::make_index_sequence<Right::output_port_types::size>());
            }
        }
    }
};

template<int OutId, int InId, detail::any_node A, detail::any_node B>
    requires std::same_as<typename std::remove_cvref_t<A>::output_port_types::template at<OutId>,
                          typename std::remove_cvref_t<B>::input_port_types::template at<InId>>
[[gnu::always_inline]] constexpr auto
merge_by_index(A &&a, B &&b) -> merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    return { std::forward<A>(a), std::forward<B>(b) };
}

namespace detail {
    template<fixed_string Name, typename PortList>
    consteval int indexForName() {
        auto helper = [] <std::size_t... Ids> (std::index_sequence<Ids...>) {
            int result = -1;
            ((PortList::template at<Ids>::name == Name ? (result = Ids) : 0), ...);
            return result;
        };
        return helper(std::make_index_sequence<PortList::size>());
    }
} // namespace detail

template<fixed_string OutName, fixed_string InName, detail::any_node A, detail::any_node B>
[[gnu::always_inline]] constexpr auto
merge(A &&a, B &&b) {
    constexpr int OutId = detail::indexForName<OutName, typename A::output_ports>();
    constexpr int InId = detail::indexForName<InName, typename B::input_ports>();
    static_assert(OutId != -1);
    static_assert(InId != -1);
    static_assert(std::same_as<typename std::remove_cvref_t<A>::output_port_types::template at<OutId>,
                               typename std::remove_cvref_t<B>::input_port_types::template at<InId>>,
                  "Port types do not match");
    return merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId>{ std::forward<A>(a), std::forward<B>(b) };
}
} // namespace fair::graph

#endif // GRAPH_PROTOTYPE_GRAPH_HPP
