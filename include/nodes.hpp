#ifndef GRAPH_PROTOTYPE_NODES_HPP
#define GRAPH_PROTOTYPE_NODES_HPP

#include "graph.hpp"

template<typename T, int Depth>
    requires(Depth > 0)
class delay_impl {
    std::array<T, Depth> buffer = {};
    int                  pos    = 0;

public:
    using input_ports           = fair::graph::make_input_ports<T>;
    using output_ports          = fair::graph::make_output_ports<T>;

    [[nodiscard]] constexpr T
    process_one(T in) noexcept {
        T ret       = buffer[pos];
        buffer[pos] = in;
        if (pos == Depth - 1) {
            pos = 0;
        } else {
            ++pos;
        }
        return ret;
    }
};

template<typename T, int Depth>
using delay = fair::graph::make_node<delay_impl<T, Depth>>;

namespace detail {
template<typename T, auto>
using just_t = T;

template<typename T, std::size_t... Is>
consteval fair::graph::make_output_ports<just_t<T, Is>...>
make_multiple_output_ports(std::index_sequence<Is...>) {
    return {};
}
} // namespace detail

template<typename T, int Count>
class duplicate_impl
{
public:
    using input_ports  = fair::graph::make_input_ports<T>;
    using output_ports = decltype(detail::make_multiple_output_ports<T>(
            std::make_index_sequence<Count>()));
    using return_type = typename output_ports::tuple_or_type;

    [[nodiscard]] constexpr return_type
    process_one(T a) const noexcept {
        if constexpr (Count == 1) return a;
        else
            return [&a]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::make_tuple(((void) Is, a)...);
            }(std::make_index_sequence<Count>());
    }
};

template<typename T, int Count = 2>
using duplicate = fair::graph::make_node<duplicate_impl<T, Count>>;

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
class adder_impl {
public:
    using input_ports  = fair::graph::make_input_ports<T, T>;
    using output_ports = fair::graph::make_output_ports<R>;

    template<fair::graph::detail::t_or_simd<T> V>
    [[gnu::always_inline, nodiscard]] constexpr auto
    process_one(V a, V b) const noexcept {
        return a + b;
    }
};

template<typename T>
using adder = fair::graph::make_node<adder_impl<T>>;

template<typename T, typename R = decltype(std::declval<T>() * std::declval<T>())>
class scale_impl {
    T _factor = {};

public:
    using input_ports  = fair::graph::make_input_ports<T>;
    using output_ports = fair::graph::make_output_ports<R>;

    constexpr scale_impl(T f) noexcept : _factor(f) {}

    template<fair::graph::detail::t_or_simd<T> V>
    [[gnu::always_inline, nodiscard]] constexpr auto
    process_one(V a) const noexcept {
        return a * _factor;
    }
};

template<typename T>
using scale = fair::graph::make_node<scale_impl<T>>;

template<typename T>
class saturate_impl {
    T _min, _max;

public:
    using input_ports  = fair::graph::make_input_ports<T>;
    using output_ports = fair::graph::make_output_ports<T>;

    constexpr saturate_impl(T min, T max) : _min(min), _max(max) {}

    template<fair::graph::detail::t_or_simd<T> V>
    [[gnu::always_inline]] constexpr auto
    process_one(V x) const noexcept {
        using std::clamp;
        return clamp(x, V(_min), V(_max));
    }
};

template<typename T>
using saturate = fair::graph::make_node<saturate_impl<T>>;

#endif // GRAPH_PROTOTYPE_NODES_HPP
