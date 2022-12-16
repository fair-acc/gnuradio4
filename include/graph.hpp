#ifndef GRAPH_PROTOTYPE_GRAPH_HPP
#define GRAPH_PROTOTYPE_GRAPH_HPP

#ifndef GRAPH_PROTOTYPE_TYPELIST_HPP
#include "typelist.hpp"
#endif

#include <tuple>

namespace fair::graph {
template<typename T>
concept any_node = true;

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
};

template<any_node Left, any_node Right, int OutId, int InId>
class merged_node
    : public node<merged_node<Left, Right, OutId, InId>,
                  make_input_ports<
                          meta::concat<typename Left::input_ports::typelist,
                                       meta::remove_at<InId, typename Right::input_ports::typelist>>>,
                  make_output_ports<meta::concat<typename Left::output_ports::typelist,
                                                 typename Right::output_ports::typelist>>> {
private:
    using base = node<merged_node,
                      make_input_ports<meta::concat<
                              typename Left::input_ports::typelist,
                              meta::remove_at<InId, typename Right::input_ports::typelist>>>,
                      make_output_ports<meta::concat<typename Left::output_ports::typelist,
                                                     typename Right::output_ports::typelist>>>;

    Left  left;
    Right right;

    template<std::size_t... Is>
    constexpr auto
    apply_left(auto &&input_tuple, std::index_sequence<Is...>) {
        return left.process_one(std::get<Is>(input_tuple)...);
    }

    template<std::size_t... Is, std::size_t... Js>
    constexpr auto
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

    constexpr merged_node(Left l, Right r) : left(std::move(l)), right(std::move(r)) {}

    template<typename... Ts>

    requires input_ports::template are_equal<std::remove_cvref_t<Ts>...> constexpr
            typename output_ports::tuple_or_type
            process_one(Ts &&...inputs) {
        if constexpr (Left::output_ports::size == 1 && Right::output_ports::size == 1) {
            static_assert(std::tuple_size_v<return_type> == 2);
            return_type ret;
            auto       &left_out  = std::get<0>(ret);
            auto       &right_out = std::get<1>(ret);
            left_out              = apply_left(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                               std::make_index_sequence<Left::input_ports::size()>());
            right_out
                    = apply_right(std::forward_as_tuple(std::forward<Ts>(inputs)...), left_out,
                                  std::make_index_sequence<InId>(),
                                  std::make_index_sequence<Right::input_ports::size() - InId - 1>());

            return ret;
        } else {
            auto left_out = [&]() {
                auto tmp = apply_left(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                      std::make_index_sequence<Left::input_ports::size()>());
                if constexpr (Left::output_ports::size == 1) return std::make_tuple(std::move(tmp));
                else
                    return tmp;
            }();
            auto right_out = [&]() {
                auto tmp = apply_right(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                       std::get<OutId>(left_out), std::make_index_sequence<InId>(),
                                       std::make_index_sequence<Right::input_ports::size() - InId
                                                                - 1>());
                if constexpr (Right::output_ports::size == 1)
                    return std::make_tuple(std::move(tmp));
                else
                    return tmp;
            }();
            return std::tuple_cat(std::move(left_out), std::move(right_out));
        }
    }
};

template<int OutId, int InId, any_node A, any_node B>

requires std::same_as<typename std::remove_cvref_t<A>::output_ports::template at<OutId>::type,
                      typename std::remove_cvref_t<B>::input_ports::template at<
                              InId>::type> [[gnu::always_inline]] constexpr auto
merge(A &&a, B &&b) -> merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    return { std::forward<A>(a), std::forward<B>(b) };
}
} // namespace fair::graph

#endif // GRAPH_PROTOTYPE_GRAPH_HPP
