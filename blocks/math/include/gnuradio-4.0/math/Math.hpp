#ifndef GNURADIO_MATH_HPP
#define GNURADIO_MATH_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::math {

namespace detail {
template<typename T>
T defaultValue() noexcept {
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
    [[nodiscard]] constexpr V processOne(const V& a) const noexcept {
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

template<typename T, typename op>
requires(std::is_arithmetic_v<T>)
struct MathOpMultiPortImpl : public gr::Block<MathOpMultiPortImpl<T, op>> {
    using Description = Doc<R""(
    @brief Math block combining multiple inputs into a single output with a given operation

    Depending on the operator op this block computes:
    - Multiply: out = in_1 * in_2 * in_3 * ...
    - Divide: out = in_1 / in_2 / in_3 / ...
    - Add: out = in_1 + in_2 + in_3 + ...
    - Subtract: out = in_1 - in_2 - in_3 - ...
    )"">;

    // ports
    std::vector<PortIn<T>> in;
    PortOut<T>             out;

    // settings
    Annotated<gr::Size_t, "n_inputs", Visible, Doc<"Number of inputs">, Limits<1U, 32U>> n_inputs = 0U;

    GR_MAKE_REFLECTABLE(MathOpMultiPortImpl, in, out, n_inputs);

    void settingsChanged(const gr::property_map& old_settings, const gr::property_map& new_settings) {
        if (new_settings.contains("n_inputs") && old_settings.at("n_inputs") != new_settings.at("n_inputs")) {
            in.resize(n_inputs);
        }
    }

    template<gr::InputSpanLike TInSpan>
    gr::work::Status processBulk(const std::span<TInSpan>& ins, gr::OutputSpanLike auto& sout) const {
        std::copy(ins[0].begin(), ins[0].end(), sout.begin());
        for (std::size_t n = 1; n < ins.size(); n++) {
            std::transform(sout.begin(), sout.end(), ins[n].begin(), sout.begin(), op{});
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
using Add = MathOpMultiPortImpl<T, std::plus<T>>;
template<typename T>
using Subtract = MathOpMultiPortImpl<T, std::minus<T>>;
template<typename T>
using Multiply = MathOpMultiPortImpl<T, std::multiplies<T>>;
template<typename T>
using Divide = MathOpMultiPortImpl<T, std::divides<T>>;

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
