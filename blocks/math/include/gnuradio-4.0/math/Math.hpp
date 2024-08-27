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

    GR_MAKE_REFLECTABLE(MathOpImpl, in, out, value);

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


template<typename T, char op>
struct MathOpMultiPortImpl : public gr::Block<MathOpMultiPortImpl<T, op>,  Doc<R""(
@brief Math block combining multiple inputs into a single output with a given operation

Depending on the operator op this block computes:
- Multiply (op='*'): out = in_1 * in_2 * in_3 * ...
- Divide (op='/'): out = in_1 / in_2 / in_3 / ...
- Add (op='+'): out = in_1 + in_2 + in_3 + ...
- Subtract (op='-'): out = in_1 - in_2 - in_3 - ...
)"">> {
    std::vector<PortIn<T>>    in;
    std::array<PortOut<T>, 1> out;

    GR_MAKE_REFLECTABLE(MathOpMultiPortImpl, in, out, value);

    template<gr::ConsumableSpan TInSpan, gr::PublishableSpan TOutSpan>
    gr::work::Status processBulk(const std::span<TInSpan> &ins, std::span<TOutSpan> &outs) {
        for (std::size_t n=0; n < ins.size(); n++) {
            for (std::size_t i=0; i < ins[n].size(); i++) {
                if (n == 0) {
                    outs[0][i] = ins[0][i];
                } else if constexpr (op == '*') {
                    outs[0][i] *= ins[n][i];
                } else if constexpr (op == '/') {
                    outs[0][i] /= ins[n][i];
                } else if constexpr (op == '+') {
                    outs[0][i] += ins[n][i];
                } else if constexpr (op == '-') {
                    outs[0][i] -= ins[n][i];
                } else {
                    static_assert(gr::meta::always_false<T>, "unknown op");
                }
            }
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
using Add = MathOpMultiPortImpl<T, '+'>;
template<typename T>
using Subtract = MathOpMultiPortImpl<T, '-'>;
template<typename T>
using Multiply = MathOpMultiPortImpl<T, '*'>;
template<typename T>
using Divide = MathOpMultiPortImpl<T, '/'>;



} // namespace gr::blocks::math

// clang-format off
const inline auto registerConstMath = gr::registerBlock<gr::blocks::math::AddConst,      uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry())
                                    | gr::registerBlock<gr::blocks::math::SubtractConst, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry())
                                    | gr::registerBlock<gr::blocks::math::MultiplyConst, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry())
                                    | gr::registerBlock<gr::blocks::math::DivideConst,   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry());
const inline auto registerMultiMath = gr::registerBlock<gr::blocks::math::Add,      uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry())
                                    | gr::registerBlock<gr::blocks::math::Subtract, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry())
                                    | gr::registerBlock<gr::blocks::math::Multiply, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry())
                                    | gr::registerBlock<gr::blocks::math::Divide,   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry());
// clang-format on

#endif // GNURADIO_MATH_HPP
