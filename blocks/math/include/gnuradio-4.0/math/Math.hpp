#ifndef GNURADIO_MATH_HPP
#define GNURADIO_MATH_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::math {

namespace detail {
template<typename T>
T
defaultValue() noexcept {
    if constexpr (gr::arithmetic_or_complex_like<T> || gr::UncertainValueLike<T>) {
        return static_cast<T>(1);
    } else if constexpr (std::is_arithmetic_v<T>) {
        return static_cast<T>(1);
    } else {
        return T{};
    }
}
} // namespace detail

template<typename T, char op>
struct MathOpImpl : public gr::Block<MathOpImpl<T, op>> {
    PortIn<T>  in;
    PortOut<T> out;
    T          value = detail::defaultValue<T>();

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V
    processOne(const V &a) const noexcept {
        if constexpr (op == '*') {
            return a * value;
        } else if constexpr (op == '/') {
            return a / value;
        } else if constexpr (op == '+') {
            return a + value;
        } else if constexpr (op == '-') {
            return a - value;
        } else {
            static_assert(gr::meta::always_false<T>, "unknown op");
            return V{};
        }
    }
};

template<typename T>
using AddConst = MathOpImpl<T, '+'>;
template<typename T>
using SubtractConst = MathOpImpl<T, '-'>;
template<typename T>
using MultiplyConst = MathOpImpl<T, '*'>;
template<typename T>
using DivideConst = MathOpImpl<T, '/'>;

} // namespace gr::blocks::math

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, char op), (gr::blocks::math::MathOpImpl<T, op>), in, out, value);

// clang-format off
const inline auto registerConstMath = gr::registerBlock<gr::blocks::math::AddConst,      uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry())
                                    | gr::registerBlock<gr::blocks::math::SubtractConst, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry())
                                    | gr::registerBlock<gr::blocks::math::MultiplyConst, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry())
                                    | gr::registerBlock<gr::blocks::math::DivideConst,   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry());
// clang-format on

#endif // GNURADIO_MATH_HPP
