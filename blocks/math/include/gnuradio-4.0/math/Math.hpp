#ifndef GNURADIO_MATH_HPP
#define GNURADIO_MATH_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <vector>
#include <algorithm>
#include <complex>
#include <functional>
#include <span>
#include <volk/volk.h>
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


template<typename T, typename op>
struct MathOpImpl : Block<MathOpImpl<T, op>> {
    PortIn<T>  in{};
    PortOut<T> out{};
    T          value = detail::defaultValue<T>();

    GR_MAKE_REFLECTABLE(MathOpImpl, in, out, value);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V& a) const noexcept {
        if constexpr (gr::meta::any_simd<V, T>) {
            if constexpr (std::same_as<op, std::multiplies<T>>) {
                return a * value;
            } else if constexpr (std::same_as<op, std::divides<T>>) {
                return a / value;
            } else if constexpr (std::same_as<op, std::plus<T>>) {
                return a + value;
            } else if constexpr (std::same_as<op, std::minus<T>>) {
                return a - value;
            } else {
                static_assert(gr::meta::always_false<T>, "unknown op");
                return V{};
            }
        } else { // non-simd branch
            return op()(a, value);
        }
    }
};

template<typename T>
using AddConst = MathOpImpl<T, std::plus<T>>;

template<typename T>
using SubtractConst = MathOpImpl<T, std::minus<T>>;

template<typename T>
using MultiplyConst = MathOpImpl<T, std::multiplies<T>>;

template<typename T>
using DivideConst = MathOpImpl<T, std::divides<T>>;

// Macros for constant blocks registration
GR_REGISTER_BLOCK("gr::blocks::math::AddConst", gr::blocks::math::MathOpImpl, ([T], '+'), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::SubtractConst", gr::blocks::math::MathOpImpl, ([T], '-'), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::MultiplyConst", gr::blocks::math::MathOpImpl, ([T], '*'), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::DivideConst", gr::blocks::math::MathOpImpl, ([T], '/'), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ]);

template<typename T, typename op>
requires(std::is_arithmetic_v<T>)
struct MathOpMultiPortImpl : Block<MathOpMultiPortImpl<T, op>> {
    using Description = Doc<R""(@brief Math block combining multiple inputs into a single output with a given operation

    Depending on the operator op this block computes:
    - Multiply: out = in_1 * in_2 * in_3 * ...
    - Divide: out = in_1 / in_2 / in_3 / ...
    - Add: out = in_1 + in_2 + in_3 + ...
    - Subtract: out = in_1 - in_2 - in_3 - ...
    )"">;

    // Ports
    std::vector<PortIn<T>> in;
    PortOut<T>             out;

    // Settings
    Annotated<gr::Size_t, "n_inputs", Visible, Doc<"Number of inputs">, Limits<1U, 32U>> n_inputs = 0U;

    GR_MAKE_REFLECTABLE(MathOpMultiPortImpl, in, out, n_inputs);

    void settingsChanged(const gr::property_map& old_settings, const gr::property_map& new_settings) {
        if (new_settings.contains("n_inputs") && old_settings.at("n_inputs") != new_settings.at("n_inputs")) {
            in.resize(n_inputs);
        }
    }

    template<gr::InputSpanLike TInSpan>
    gr::work::Status processBulk(const std::span<TInSpan>& ins, gr::OutputSpanLike auto& sout) const {
        // Initialize output with the first input stream.
        std::copy(ins[0].begin(), ins[0].end(), sout.begin());
        // Apply the operator for every subsequent input stream.
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

// Registration macros for multiport math operations
GR_REGISTER_BLOCK("gr::blocks::math::Add", gr::blocks::math::MathOpImpl, ([T], std::plus<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::Subtract", gr::blocks::math::MathOpImpl, ([T], std::minus<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::Multiply", gr::blocks::math::MathOpImpl, ([T], std::multiplies<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::Divide", gr::blocks::math::MathOpImpl, ([T], std::divides<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ]);


// Conjugate Block

template<typename T>    // complex<float> complex<double>
struct ConjugateImpl : Block<ConjugateImpl<T>> {
    using Description = Doc<R"("@brief This block returns the conjugate by inverting the imaginary part of the input signal")">;

    // Ports
    PortIn<T> in{};
    PortOut<T> out{};

    GR_MAKE_REFLECTABLE(ConjugateImpl, in, out);   // macros used for the reflection class 
    
    static std::shared_ptr<ConjugateImpl<T>> make() {     // returns a smart pointer to ConjugateImpl
        return std::make_shared<ConjugateImpl<T>>();      // new instance of ConjugateImpl dynamically and wraps it ina shrared_ptr.
    }
    
    gr::work::Status processBulk(gr::InputSpanLike auto& input_span, gr::OutputSpanLike auto& output_span) const noexcept {
        if (input_span.size() != output_span.size()) {
            return gr::work::Status::ERROR;
        }
        
        if constexpr (std::is_same_v<T, std::complex<float>>) {
            volk_32fc_conjugate_32fc(
                reinterpret_cast<std::complex<float>*>(output_span.data()),
                reinterpret_cast<const std::complex<float>*>(input_span.data()),
                static_cast<unsigned int>(input_span.size())
            );            
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            std::transform(input_span.begin(), input_span.end(), output_span.begin(),
                [](const std::complex<double>& z) {
                    return std::conj(z);
                });
        }
        return gr::work::Status::OK;
    }
};

// Register the Conjugate block for the float and double complex types
GR_REGISTER_BLOCK("gr::blocks::math::Conjugate", gr::blocks::math::ConjugateImpl, std::tuple<std::complex<float>, std::complex<double>>);

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_HPP
