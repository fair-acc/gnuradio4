#include <plugin.hpp>

#include <charconv>
#include <vector>

GP_PLUGIN("Good Base Plugin", "Unknown", "LGPL3", "v1")

namespace good {
namespace fg = fair::graph;

template<typename T>
auto
factor(const fair::graph::property_map &params) {
    T factor = 1;
    if (auto it = params.find("factor"s); it != params.end()) {
        auto &variant = it->second;
        auto *ptr     = std::get_if<T>(&variant);
        if (ptr) {
            factor = *ptr;
        }
    }
    return factor;
}

template<typename T>
class math_base {
protected:
    T _factor = static_cast<T>(1.0f);

public:
    fg::PortIn<T, fg::RequiredSamples<1, 1024>>  in;
    fg::PortOut<T, fg::RequiredSamples<1, 1024>> out;

    math_base() = delete;

    explicit math_base(const fair::graph::property_map &params) : _factor(factor<T>(params)) {}
};

template<typename T>
class multiply : public fg::block<multiply<T>>, public math_base<T> {
public:
    using math_base<T>::math_base;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(const V &a) const noexcept {
        return a * math_base<T>::_factor;
    }
};

template<typename T>
class divide : public fg::block<divide<T>>, public math_base<T> {
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
GP_PLUGIN_REGISTER_BLOCK(good::multiply, float, double);

ENABLE_REFLECTION_FOR_TEMPLATE(good::divide, in, out);
GP_PLUGIN_REGISTER_BLOCK(good::divide, float, double);
