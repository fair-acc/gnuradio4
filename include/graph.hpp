#ifndef GRAPH_PROTOTYPE_GRAPH_HPP
#define GRAPH_PROTOTYPE_GRAPH_HPP

#ifndef GRAPH_PROTOTYPE_TYPELIST_HPP
#include "typelist.hpp"
#endif

#include <tuple>

namespace fair::graph {
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

template<typename Derived, typename InputPorts, typename OutputPorts>
class node {
public:
    using input_ports                        = InputPorts;
    using output_ports                       = OutputPorts;

    static inline constexpr input_ports  in  = {};
    static inline constexpr output_ports out = {};

protected:
    constexpr node() noexcept = default;
};

namespace detail {
template<typename Node>
concept any_node = requires(Node &n, typename Node::input_ports::tuple_type const &inputs) {
    { n.in } -> std::same_as<typename Node::input_ports const &>;
    { n.out } -> std::same_as<typename Node::output_ports const &>;
    // doesn't compile:
    // std::apply([&n](auto&&... ins) { n.process_one(ins...); }, inputs);
};
} // namespace detail

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
    apply_right(auto &&input_tuple, const typename Left::output_ports::tuple_or_type &tmp,
                std::index_sequence<Is...>, std::index_sequence<Js...>) {
        constexpr int first_offset  = Left::input_ports::size;
        constexpr int second_offset = Left::input_ports::size + sizeof...(Is);
        static_assert(second_offset + sizeof...(Js)
                      == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
        return right.process_one(std::get<first_offset + Is>(input_tuple)..., tmp,
                                 std::get<second_offset + Js>(input_tuple)...);
    }

public:
    using input_ports  = typename base::input_ports;
    using output_ports = typename base::output_ports;
    using return_type  = typename output_ports::tuple_or_type;
  
    [[gnu::always_inline]] constexpr merged_node(Left l, Right r)
        : left(std::move(l)), right(std::move(r)) {}

    template<typename... Ts>
    requires input_ports::template are_equal<std::remove_cvref_t<Ts>...> constexpr
            typename output_ports::tuple_or_type
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
                }
            (std::make_index_sequence<OutId>(),
             std::make_index_sequence<Left::output_ports::size - OutId - 1>());
            else return [&]<std::size_t... Is, std::size_t... Js,
                            std::size_t... Ks>(std::index_sequence<Is...>,
                                               std::index_sequence<Js...>,
                                               std::index_sequence<Ks...>) {
                return std::make_tuple(std::move(std::get<Is...>(left_out)),
                                       std::move(std::get<OutId + 1 + Js...>(left_out)),
                                       std::move(std::get<Ks...>(right_out)));
            }
            (std::make_index_sequence<OutId>(),
             std::make_index_sequence<Left::output_ports::size - OutId - 1>(),
             std::make_index_sequence<Right::output_ports::size>());
        }
    }
};

template<int OutId, int InId, detail::any_node A, detail::any_node B>
requires std::same_as<typename std::remove_cvref_t<A>::output_ports::template at<OutId>::type,
                      typename std::remove_cvref_t<B>::input_ports::template at<
                              InId>::type> [[gnu::always_inline]] constexpr auto
merge(A &&a, B &&b) -> merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    return { std::forward<A>(a), std::forward<B>(b) };
}
} // namespace fair::graph

#endif // GRAPH_PROTOTYPE_GRAPH_HPP
