#ifndef GNURADIO_COMPLEXMATH_HPP
#define GNURADIO_COMPLEXMATH_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/meta/utils.hpp>
#include <complex>
#include <cmath>
#include <functional>

namespace gr::blocks::math {

namespace detail {
inline Size_t default_exponent() noexcept {
    return 1;
}

inline std::complex<float> default_tag_value() noexcept {
    return {1.f, 0.f};
}
} // namespace detail

GR_REGISTER_BLOCK("gr::blocks::math::ComplexConjugate", gr::blocks::math::ComplexConjugate, [T], [ std::complex<float>, std::complex<double> ])

template<typename T>
struct ComplexConjugate : Block<ComplexConjugate<T>> {
    using Description = Doc<"(@brief convert complex numbers into their complex conjugates.">;
    PortIn<T> in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(ComplexConjugate, in, out);

    template<meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V &a) const noexcept {
        return std::conj(a);
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::ExponentiateConstInt", gr::blocks::math::ExponentiateConstInt, [T], [ std::complex<float>, std::complex<double> ])

template<typename T>
struct ExponentiateConstInt : Block<ExponentiateConstInt<T>> {
    using Description = Doc<"(@brief exponentiate complex numbers with an integer exponent.">;
    PortIn<T> in;
    PortOut<T> out;
    Size_t exponent = detail::default_exponent();

    GR_MAKE_REFLECTABLE(ExponentiateConstInt, in, out, exponent);

    template<meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V &a) const noexcept {
        return std::pow(a, static_cast<float>(exponent));
   }
};

GR_REGISTER_BLOCK("gr::blocks::math::MultiplyByTagValue", gr::blocks::math::MultiplyByTagValue, [T], [ std::complex<float>, std::complex<double> ])

template<typename T>
struct MultiplyByTagValue : Block<MultiplyByTagValue<T>> {
    using Description = Doc<"(@brief multiply by a (constant complex) tag value.">;
    PortIn<T> in;
    PortOut<T> out;
    T tag_value = detail::default_tag_value();

    GR_MAKE_REFLECTABLE(MultiplyByTagValue, in, out, tag_value);

    template<meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V &a) const noexcept {
        return a * tag_value;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::MultiplyConjugate", gr::blocks::math::MultiplyConjugate, [T], [ std::complex<float>, std::complex<double> ])

template<typename T>
struct MultiplyConjugate : Block<MultiplyConjugate<T>> {
    using Description = Doc<"(@brief multiply stream 0 by complex conjugate of stream 1.">;
    PortIn<T> in1;
    PortIn<T> in2;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(MultiplyConjugate, in1, in2, out);

    template<meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V &a, const V &b) const noexcept {
        return a * std::conj(b);
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::Transcendental", gr::blocks::math::Transcendental, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct Transcendental : Block<Transcendental<T>> {
    using Description = Doc<R""(@brief perform a transcendental math operation on a decimal and/or complex input.)"">;
    PortIn<T> in;
    PortOut<T> out;
    std::string function_name;
    using MathFunc = T (*)(T);

    std::optional<T> evaluate_transcendental(std::string name, T input) {
        static const std::unordered_map<std::string, MathFunc> funcs = {
            {"sin", [](T x) { return std::sin(x); }},
            {"cos", [](T x) { return std::cos(x); }},
            {"tan", [](T x) { return std::tan(x); }},
            {"sinh", [](T x) { return std::sinh(x); }},
            {"cosh", [](T x) { return std::cosh(x); }},
            {"tanh", [](T x) { return std::tanh(x); }},
            {"exp", [](T x) { return std::exp(x); }},
            {"log", [](T x) { return std::log(x); }},
            {"log10", [](T x) { return std::log10(x); }},
            {"sqrt", [](T x) { return std::sqrt(x); }},
            {"asin", [](T x) { return std::asin(x); }},
            {"acos", [](T x) { return std::acos(x); }},
            {"atan", [](T x) { return std::atan(x); }}
        };
        if (auto it = funcs.find(name); it != funcs.end()) {
            return it->second(input);
        }
        return std::nullopt;
    };

    GR_MAKE_REFLECTABLE(Transcendental, in, out, function_name);

    template<gr::InputSpanLike TInSpan>
    gr::work::Status processBulk(const TInSpan& in_span, gr::OutputSpanLike auto& out_span) {
        if (in_span.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        size_t n_items = in_span.size();
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        for (size_t i = 0; i < n_items; ++i) {
            auto result = evaluate_transcendental(function_name, in_span[i]);
            // Store result in output stream
            if (!result.has_value()) {
                return gr::work::Status::OK;
            }
            out_span[i] = result.value();
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::Integrate", gr::blocks::math::Integrate, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct Integrate : Block<Integrate<T>> {
    using Description = Doc<"(@brief perform a successive sampling and decimation of the input data stream.">;
    PortIn<T> in;
    PortOut<T> out;
    Size_t decimation = 1;

    GR_MAKE_REFLECTABLE(Integrate, in, out, decimation);

    template<gr::InputSpanLike TInSpan>
    gr::work::Status processBulk(const TInSpan& in_span, gr::OutputSpanLike auto& out_span) {
        if (in_span.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        auto n_items = static_cast<Size_t>(in_span.size());
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        Size_t rest = n_items % decimation;
        T sum = static_cast<T>(0);
        size_t count = 0;
        for (Size_t i = 0; i < n_items; ++i) {
            sum += in_span[i];
            if (i % decimation == decimation - 1) {
                out_span[count] = sum / decltype(sum)(decimation);
                ++count;
                sum = static_cast<T>(0);
            }
        }
        if (rest > static_cast<Size_t>(0)) {
            out_span[count] = sum / decltype(sum)(rest);
        }
        out_span.publish(count);

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::MultiplyByMatrix", gr::blocks::math::MultiplyByMatrix, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct MultiplyByMatrix : Block<MultiplyByMatrix<T>> {
    using Description = Doc<"(@brief perform a matrix multiplication to the elements of a fixed number input data streams.">;
    std::vector<PortIn<T>> in;
    std::vector<PortOut<T>> out;
    Tensor<T> matrix = {};

    // settings
    Annotated<gr::Size_t, "n_inputs", Visible, Doc<"Number of inputs">, Limits<1U, 32U>> n_inputs = 0U;
    Annotated<gr::Size_t, "m_outputs", Visible, Doc<"Number of outputs">, Limits<1U, 32U>> m_outputs = 0U;

    GR_MAKE_REFLECTABLE(MultiplyByMatrix, in, out, n_inputs, m_outputs, matrix);

    void settingsChanged(const gr::property_map& old_settings, const gr::property_map& new_settings) {
        if (new_settings.contains("n_inputs") && old_settings.find_value("n_inputs") != new_settings.find_value("n_inputs")) {
            in.resize(n_inputs);
        }
        if (new_settings.contains("m_outputs") && old_settings.find_value("m_outputs") != new_settings.find_value("m_outputs")) {
            out.resize(m_outputs);
        }
    }

    template<gr::InputSpanLike TInSpan, gr::OutputSpanLike TOutSpan>
        gr::work::Status processBulk(std::span<TInSpan>& in_span, std::span<TOutSpan>& out_span) {
        if (in_span.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        auto n_items = static_cast<Size_t>(in_span[0].size());
        auto m_items = static_cast<Size_t>(out_span[0].size());
        if (n_items == 0 || m_items == 0 || n_items != m_items) {
            return gr::work::Status::OK;
        }
        auto cols = static_cast<Size_t>(in_span.size());
        auto rows = static_cast<Size_t>(out_span.size());
        if (matrix.size() != rows * cols) {
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        for (Size_t i = 0; i < n_items; ++i) {
            for (Size_t j = 0; j < rows; ++j) {
                T sum = static_cast<T>(0);
                for (Size_t k = 0; k < cols; ++k) {
                    sum += matrix[j * cols + k] * in_span[k][i];
                }
                out_span[j][i] = sum;
            }
        }

        return gr::work::Status::OK;
    }
};

}// namespace gr::blocks::math

#endif // GNURADIO_COMPLEXMATH_HPP
