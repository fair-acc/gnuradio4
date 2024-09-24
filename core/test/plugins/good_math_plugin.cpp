#include <charconv>
#include <vector>

#include <gnuradio-4.0/plugin.hpp>

GR_PLUGIN("Good Math Plugin", "Unknown", "LGPL3", "v1")

namespace good {

template<typename T>
auto
factor(const gr::property_map &params) {
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
    gr::PortIn<T, gr::RequiredSamples<1, 1024>>  in;
    gr::PortOut<T, gr::RequiredSamples<1, 1024>> out;

    math_base() = delete;

    explicit math_base(const gr::property_map &params) : _factor(factor<T>(params)) {}
};

template<typename T>
class multiply : public gr::Block<multiply<T>>, public math_base<T> {
public:
    using math_base<T>::math_base;
    using math_base<T>::in;
    using math_base<T>::out;

    GR_MAKE_REFLECTABLE(multiply, in, out);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(const V &a) const noexcept {
        return a * math_base<T>::_factor;
    }
};

template<typename T>
class divide : public gr::Block<divide<T>>, public math_base<T> {
public:
    using math_base<T>::math_base;
    using math_base<T>::in;
    using math_base<T>::out;

    GR_MAKE_REFLECTABLE(divide, in, out);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(const V &a) const noexcept {
        return a / math_base<T>::_factor;
    }
};

} // namespace good

namespace bts = gr::traits::block;

auto registerMultiply = gr::registerBlock<good::multiply, float, double>(grPluginInstance());

static_assert(bts::all_input_ports<good::multiply<float>>::size == 1);
static_assert(std::is_same_v<bts::all_input_port_types<good::multiply<float>>, gr::meta::typelist<float>>);
static_assert(bts::stream_input_ports<good::multiply<float>>::size == 1);
static_assert(std::is_same_v<bts::stream_input_port_types<good::multiply<float>>, gr::meta::typelist<float>>);
static_assert(bts::all_output_ports<good::multiply<float>>::size == 1);
static_assert(std::is_same_v<bts::all_output_port_types<good::multiply<float>>, gr::meta::typelist<float>>);
static_assert(bts::stream_output_ports<good::multiply<float>>::size == 1);
static_assert(std::is_same_v<bts::stream_output_port_types<good::multiply<float>>, gr::meta::typelist<float>>);

auto registerDivide = gr::registerBlock<good::divide, float, double>(grPluginInstance());
