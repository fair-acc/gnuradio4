#ifndef GNURADIO_MATH_HPP
#define GNURADIO_MATH_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <span>
#include <utility>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetMath.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

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

template<typename TOp>
struct Transparent;

// std::plus<T> and its kin take only T; their transparent std::plus<> form also takes a SIMD vector and a scalar
template<template<typename> typename TOp, typename T>
struct Transparent<TOp<T>> {
    using type = TOp<void>;
};
} // namespace detail

template<typename T>
struct Maximum {
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept { return std::max(a, b); }
};

template<typename T>
struct Minimum {
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept { return std::min(a, b); }
};

template<typename T>
struct MultipliesConjugate {
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept {
        using std::conj;
        return a * conj(b);
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::AddConst", gr::blocks::math::ConstOp, ([T], std::plus<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::SubtractConst", gr::blocks::math::ConstOp, ([T], std::minus<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::AddPair", gr::blocks::math::PairOp, ([T], std::plus<[T]>), [ float, double, std::complex<float>, std::complex<double>, gr::complex<float>, gr::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::SubtractPair", gr::blocks::math::PairOp, ([T], std::minus<[T]>), [ float, double, std::complex<float>, std::complex<double>, gr::complex<float>, gr::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::MultiplyPair", gr::blocks::math::PairOp, ([T], std::multiplies<[T]>), [ float, double, std::complex<float>, std::complex<double>, gr::complex<float>, gr::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::DividePair", gr::blocks::math::PairOp, ([T], std::divides<[T]>), [ float, double, std::complex<float>, std::complex<double>, gr::complex<float>, gr::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::MultiplyConjugate", gr::blocks::math::PairOp, ([T], gr::blocks::math::MultipliesConjugate<[T]>), [ std::complex<float>, std::complex<double>, gr::complex<float>, gr::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::MultiplyConst", gr::blocks::math::ConstOp, ([T], std::multiplies<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::DivideConst", gr::blocks::math::ConstOp, ([T], std::divides<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::AndConst", gr::blocks::math::ConstOp, ([T], std::bit_and<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])

template<typename T, typename op>
struct ConstOp : Block<ConstOp<T, op>> {
    using Description = Doc<"@brief applies `op` to each sample and `value`: AddConst, SubtractConst, MultiplyConst, DivideConst and AndConst (GR3: And Const) are its registered forms">;

    PortIn<T>  in{};
    PortOut<T> out{};
    T          value = detail::defaultValue<T>();

    GR_MAKE_REFLECTABLE(ConstOp, in, out, value);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V& a) const noexcept {
        if constexpr (gr::meta::any_simd<V, T>) {
            return typename detail::Transparent<op>::type{}(a, value);
        } else {
            return op{}(a, value);
        }
    }
};

template<typename T>
using AddConst = ConstOp<T, std::plus<T>>;
template<typename T>
using SubtractConst = ConstOp<T, std::minus<T>>;
template<typename T>
using MultiplyConst = ConstOp<T, std::multiplies<T>>;
template<typename T>
using DivideConst = ConstOp<T, std::divides<T>>;

GR_REGISTER_BLOCK(gr::blocks::math::MagnitudeNormalise, [T], [ std::complex<float>, std::complex<double>, gr::complex<float>, gr::complex<double> ])

template<meta::complex_like T>
struct MagnitudeNormalise : Block<MagnitudeNormalise<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief each sample divided by its magnitude plus `eps`, z / (|z| + eps): its phase alone, the PHAT weighting of a cross-spectrum; 0 stays 0">;
    PortIn<T>  in;
    PortOut<T> out;

    Annotated<R, "eps", Doc<"added to each magnitude, absolute">, Limits<R{0}, std::numeric_limits<R>::max()>> eps = static_cast<R>(1e-12);

    GR_MAKE_REFLECTABLE(MagnitudeNormalise, in, out, eps);

    [[nodiscard]] constexpr T processOne(T input) const noexcept { // by parts: one body for std::complex and the device-safe gr::complex
        const R magnitude = std::sqrt(input.real() * input.real() + input.imag() * input.imag()) + eps.value;
        return magnitude > R{0} ? T{input.real() / magnitude, input.imag() / magnitude} : T{};
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::Add", gr::blocks::math::ReduceOp, ([T], std::plus<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::Subtract", gr::blocks::math::ReduceOp, ([T], std::minus<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::Multiply", gr::blocks::math::ReduceOp, ([T], std::multiplies<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::Divide", gr::blocks::math::ReduceOp, ([T], std::divides<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::And", gr::blocks::math::ReduceOp, ([T], std::bit_and<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])
GR_REGISTER_BLOCK("gr::blocks::math::Or", gr::blocks::math::ReduceOp, ([T], std::bit_or<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])
GR_REGISTER_BLOCK("gr::blocks::math::Xor", gr::blocks::math::ReduceOp, ([T], std::bit_xor<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])
GR_REGISTER_BLOCK("gr::blocks::math::Max", gr::blocks::math::ReduceOp, ([T], gr::blocks::math::Maximum<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::math::Min", gr::blocks::math::ReduceOp, ([T], gr::blocks::math::Minimum<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T, typename op>
requires gr::arithmetic_or_complex_like<gr::meta::fundamental_base_value_type_t<T>>
struct ReduceOp : Block<ReduceOp<T, op>> {
    using Description = Doc<R""(@brief folds the `n_inputs` inputs sample by sample with `op`: out = op(op(in#0, in#1), in#2) …

Add, Subtract, Multiply, Divide, And, Or, Xor, Max (GR3: Max) and Min (GR3: Min) are its registered forms.)"">;

    std::vector<PortIn<T>> in;
    PortOut<T>             out;

    Annotated<gr::Size_t, "n_inputs", Visible, Doc<"Number of inputs">, Limits<1U, 32U>> n_inputs = 0U;

    GR_MAKE_REFLECTABLE(ReduceOp, in, out, n_inputs);

    void settingsChanged(const gr::property_map& old_settings, const gr::property_map& new_settings) {
        if (new_settings.contains("n_inputs") && old_settings.find_value("n_inputs") != new_settings.find_value("n_inputs")) {
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
using Add = ReduceOp<T, std::plus<T>>;
template<typename T>
using Subtract = ReduceOp<T, std::minus<T>>;
template<typename T>
using Multiply = ReduceOp<T, std::multiplies<T>>;
template<typename T>
using Divide = ReduceOp<T, std::divides<T>>;

template<typename T, typename op>
requires(gr::arithmetic_or_complex_like<T> || gr::UncertainValueLike<T>)
struct PairOp : Block<PairOp<T, op>> {
    using Description = Doc<R""(@brief Combines two streams sample by sample.

The two-port form of Add/Subtract/Multiply/Divide, plus MultiplyConjugate: in0 · conj(in1) (GR3: Multiply Conjugate). It fixes the
number of inputs at two, which is what lets it run per-sample on an accelerator: a port count that is only known
at run time cannot be handed to a kernel.
)"">;

    PortIn<T>  in0;
    PortIn<T>  in1;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(PairOp, in0, in1, out);

    [[nodiscard]] constexpr T processOne(T a, T b) const noexcept { return op{}(a, b); }
};

template<typename T>
using AddPair = PairOp<T, std::plus<T>>;
template<typename T>
using SubtractPair = PairOp<T, std::minus<T>>;
template<typename T>
using MultiplyPair = PairOp<T, std::multiplies<T>>;
template<typename T>
using DividePair = PairOp<T, std::divides<T>>;

GR_REGISTER_BLOCK(gr::blocks::math::DataSetBinaryOp, [T], [ float, double ])

template<std::floating_point T>
struct DataSetBinaryOp : Block<DataSetBinaryOp<T>> {
    using Description = Doc<R""(@brief combines two DataSets element by element, signal by signal: in0 `op` in1 as gr::dataset::mathFunction computes it, for every signal.

Axes, signal names, quantities, units, meta information and timing events come from in0, so the result keeps in0's units; each signal's range
is the result's own minimum and maximum. DataSets of other extents or another number of signals stop the stream with an error.)"">;

    PortIn<DataSet<T>>  in0;
    PortIn<DataSet<T>>  in1;
    PortOut<DataSet<T>> out;

    Annotated<gr::dataset::MathOp, "op", Doc<"ADD, SUBTRACT, MULTIPLY, DIVIDE (x/0: NaN); SQR, SQRT, LOG10, DB of in0 + in1; INV_DB, IDENTITY of in0">, Visible> op = gr::dataset::MathOp::ADD;

    GR_MAKE_REFLECTABLE(DataSetBinaryOp, in0, in1, out, op);

    [[nodiscard]] DataSet<T> processOne(const DataSet<T>& a, const DataSet<T>& b) {
        if (!std::ranges::equal(a.extents, b.extents) || a.signal_values.size() != b.signal_values.size()) [[unlikely]] {
            this->emitErrorMessage("processOne()", std::format("in0 has extents {} and {} values, in1 {} and {}; both need the same", a.extents, a.signal_values.size(), b.extents, b.signal_values.size()));
            this->requestStop();
            return a;
        }
        DataSet<T> result = a;
        std::ranges::transform(a.signal_values, b.signal_values, result.signal_values.begin(), [mode = op.value](T x, T y) { return gr::dataset::detail::applyMathOperation<T>(mode, x, y); });
        const std::size_t nSignals  = result.signal_names.size();
        const std::size_t perSignal = nSignals == 0UZ ? 0UZ : result.signal_values.size() / nSignals;
        result.signal_ranges.resize(nSignals);
        for (std::size_t s = 0UZ; s < nSignals && perSignal > 0UZ; ++s) {
            const auto [low, high]  = std::ranges::minmax(std::span(result.signal_values).subspan(s * perSignal, perSignal));
            result.signal_ranges[s] = Range<T>{.min = low, .max = high};
        }
        return result;
    }
};

GR_REGISTER_BLOCK(gr::blocks::math::Clamp, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct Clamp : Block<Clamp<T>> {
    using Description = Doc<R""(@brief limits each sample to [`low`, `high`] (GR3: Rail); a reversed band is swapped.

A complex sample has its magnitude limited and keeps its phase; zero, which has no phase, becomes `low` on the real axis.)"">;
    using R           = meta::fundamental_base_value_type_t<T>;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<R, "low", Doc<"lower bound; of the magnitude for complex samples">>  low  = std::numeric_limits<R>::lowest();
    Annotated<R, "high", Doc<"upper bound; of the magnitude for complex samples">> high = std::numeric_limits<R>::max();

    GR_MAKE_REFLECTABLE(Clamp, in, out, low, high);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V& sample) const noexcept {
        const R lowest  = std::min(low.value, high.value);
        const R highest = std::max(low.value, high.value);
        if constexpr (meta::complex_like<T>) {
            const R power     = sample.real() * sample.real() + sample.imag() * sample.imag();
            const R magnitude = [&] {
                if (power < std::numeric_limits<R>::infinity()) [[likely]] {
                    return std::sqrt(power);
                }
                const R scale = std::max(std::abs(sample.real()), std::abs(sample.imag()));
                return scale * std::sqrt((sample.real() / scale) * (sample.real() / scale) + (sample.imag() / scale) * (sample.imag() / scale));
            }();
            if (magnitude == R{0}) {
                return lowest > R{0} ? T(lowest, R{0}) : sample;
            }
            return sample * (std::clamp(magnitude, lowest, highest) / magnitude);
        } else {
            V limited                                   = sample;
            vir::stdx::where(sample < lowest, limited)  = V(lowest);
            vir::stdx::where(sample > highest, limited) = V(highest);
            return limited;
        }
    }
};

GR_REGISTER_BLOCK(gr::blocks::math::Not, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])

template<std::integral T>
struct Not : Block<Not<T>> {
    using Description = Doc<"@brief inverts every bit of each sample (GR3: Not)">;

    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(Not, in, out);

    template<meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V& sample) const noexcept {
        return static_cast<V>(~sample);
    }
};

GR_REGISTER_BLOCK(gr::blocks::math::Power, [T], [ std::complex<float>, std::complex<double>, gr::complex<float>, gr::complex<double> ])

template<meta::complex_like T>
struct Power : Block<Power<T>> {
    using Description = Doc<"@brief raises each sample to the integer `exponent` by repeated squaring (GR3: Exponentiate Const Int)">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<std::int32_t, "exponent", Doc<"integer power; a negative one gives the reciprocal">> exponent = 1;

    GR_MAKE_REFLECTABLE(Power, in, out, exponent);

    [[nodiscard]] constexpr T processOne(T base) const noexcept {
        const std::int64_t signedPower = exponent.value;
        T                  product{1};
        for (auto remaining = static_cast<std::uint64_t>(signedPower < 0 ? -signedPower : signedPower); remaining > 0U; remaining >>= 1U) {
            if ((remaining & 1U) != 0U) {
                product *= base;
            }
            base *= base;
        }
        return signedPower < 0 ? T{1} / product : product;
    }
};

GR_REGISTER_BLOCK(gr::blocks::math::Log10, [T], [ float, double ])

template<std::floating_point T>
struct Log10 : Block<Log10<T>> {
    using Description = Doc<"@brief scale · log10(x) + offset of each sample; scale = 10 gives a power in dB; x is floored at the smallest normal number, as GR3 does (GR3: Log10)">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<T, "scale", Doc<"factor on log10(x)">>   scale  = T{1};
    Annotated<T, "offset", Doc<"added after scaling">> offset = T{0};

    GR_MAKE_REFLECTABLE(Log10, in, out, scale, offset);

    [[nodiscard]] T processOne(T sample) const noexcept { return scale.value * std::log10(std::max(sample, std::numeric_limits<T>::min())) + offset.value; }
};

enum class TranscendentalFunction : std::uint8_t { sin, cos, tan, sinh, cosh, tanh, exp, log, log10, sqrt, asin, acos, atan };

GR_REGISTER_BLOCK(gr::blocks::math::Transcendental, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::floating_point<T> || meta::complex_like<T>
struct Transcendental : Block<Transcendental<T>> {
    using Description = Doc<"@brief applies `function` to each sample (GR3: Transcendental)">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<TranscendentalFunction, "function", Doc<"by name, e.g. cos">> function = TranscendentalFunction::sin;

    GR_MAKE_REFLECTABLE(Transcendental, in, out, function);

    [[nodiscard]] T processOne(T sample) const noexcept {
        using enum TranscendentalFunction;
        switch (function.value) {
        case sin: return std::sin(sample);
        case cos: return std::cos(sample);
        case tan: return std::tan(sample);
        case sinh: return std::sinh(sample);
        case cosh: return std::cosh(sample);
        case tanh: return std::tanh(sample);
        case exp: return std::exp(sample);
        case log: return std::log(sample);
        case log10: return std::log10(sample);
        case sqrt: return std::sqrt(sample);
        case asin: return std::asin(sample);
        case acos: return std::acos(sample);
        case atan: return std::atan(sample);
        }
        std::unreachable();
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_HPP
