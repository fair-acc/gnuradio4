#include <plugin.hpp>

#include <charconv>
#include <vector>

GP_PLUGIN("Good Base Plugin", "Unknown", "LGPL3", "v1")

namespace good {
namespace fg = fair::graph;

template<typename T>
auto
factor(fair::graph::node_construction_params params) {
    T    factor = 1;
    auto value  = params.value("factor"sv);
    std::ignore = std::from_chars(value.begin(), value.end(), factor);
    return factor;
}

template<typename T>
class math_base {
protected:
    T _factor = static_cast<T>(1.0f);

public:
    fg::IN<T, 0, 1024>  in;
    fg::OUT<T, 0, 1024> out;

    math_base() = delete;

    explicit math_base(fair::graph::node_construction_params params) : _factor(factor<T>(params)) {}
};

template<typename T>
class multiply : public fg::node<multiply<T>>, public math_base<T> {
public:
    using math_base<T>::math_base;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(const V &a) const noexcept {
        return a * math_base<T>::_factor;
    }
};

template<typename T>
class divide : public fg::node<divide<T>>, public math_base<T> {
public:
    using math_base<T>::math_base;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(const V &a) const noexcept {
        return a / math_base<T>::_factor;
    }
};

} // namespace good

ENABLE_REFLECTION_FOR_TEMPLATE(good::multiply, in, out);
GP_PLUGIN_REGISTER_NODE(good::multiply, float, double);

ENABLE_REFLECTION_FOR_TEMPLATE(good::divide, in, out);
GP_PLUGIN_REGISTER_NODE(good::divide, float, double);
