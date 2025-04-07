#ifndef CONVERTERBLOCKS_HPP
#define CONVERTERBLOCKS_HPP

#include <tuple>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::type::converter {

GR_REGISTER_BLOCK(gr::blocks::type::converter::Convert, ([T], [U]), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T, typename R>
requires std::is_arithmetic_v<T> && std::is_arithmetic_v<R>
struct Convert : Block<Convert<T, R>> {
    using Description = Doc<"(@brief basic block to perform a input to output data type conversion (N.B. w/o scaling))">;
    PortIn<T>  in;
    PortOut<R> out;

    GR_MAKE_REFLECTABLE(Convert, in, out);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(const V& input) const noexcept {
        if constexpr (gr::meta::any_simd<V>) { // simd case
            using RetType = vir::stdx::rebind_simd_t<R, V>;
            return vir::stdx::static_simd_cast<RetType>(input);
        } else { // non-simd case
            return static_cast<R>(input);
        }
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::ScalingConvert, ([T], [U]), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T, typename R>
requires std::is_arithmetic_v<T> && std::is_arithmetic_v<R>
struct ScalingConvert : Block<ScalingConvert<T, R>> {
    using Description = Doc<R""(@brief basic block to perform a input to output data type conversion

Performs scaling, i.e. 'R output = R(input * scale)'
)"">;
    PortIn<T>  in;
    PortOut<R> out;
    T          scale = static_cast<T>(1);

    GR_MAKE_REFLECTABLE(ScalingConvert, in, out, scale);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(const V& input) const noexcept {
        if constexpr (gr::meta::any_simd<V>) { // simd case
            using RetType = vir::stdx::rebind_simd_t<R, V>;
            return vir::stdx::static_simd_cast<RetType>(input * scale);
        } else { // non-simd case
            return static_cast<R>(input * scale);
        }
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::Abs, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct Abs : Block<Abs<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief basic block to calculate the absolute (magnitude) value of a complex or arithmetic input stream">;
    PortIn<T>  in;
    PortOut<R> abs;

    GR_MAKE_REFLECTABLE(Abs, in, abs);

    [[nodiscard]] constexpr R processOne(T input) const noexcept {
        if constexpr (std::is_unsigned_v<T>) {
            using TSigned = std::make_signed_t<T>;
            return static_cast<R>(std::abs(static_cast<TSigned>(input)));
        } else {
            return static_cast<R>(std::abs(input));
        }
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::Imag, [T], [ std::complex<float>, std::complex<double> ])

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct Imag : Block<Imag<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief basic block to the imaginary component of a complex input stream">;
    PortIn<T>  in;
    PortOut<R> imag;

    GR_MAKE_REFLECTABLE(Imag, in, imag);

    [[nodiscard]] constexpr R processOne(T input) const noexcept { return std::imag(input); }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::Real, [T], [ std::complex<float>, std::complex<double> ])

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct Real : Block<Real<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief basic block to the real component of a complex input stream">;
    PortIn<T>  in;
    PortOut<R> real;

    GR_MAKE_REFLECTABLE(Real, in, real);

    [[nodiscard]] constexpr R processOne(T input) const noexcept { return std::real(input); }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::Arg, [T], [ std::complex<float>, std::complex<double> ])

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct Arg : Block<Arg<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief basic block to calculate the argument (phase angle, [radians]) of a complex input stream">;
    PortIn<T>  in;
    PortOut<R> arg;

    GR_MAKE_REFLECTABLE(Arg, in, arg);

    [[nodiscard]] constexpr R processOne(T input) const noexcept { return std::arg(input); }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::RadiansToDegree, [T], [ float, double ])

template<std::floating_point T>
struct RadiansToDegree : Block<RadiansToDegree<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief convert radians to degree">;
    PortIn<T>  rad;
    PortOut<R> deg;

    GR_MAKE_REFLECTABLE(RadiansToDegree, rad, deg);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(const V& radians) const noexcept {
        return (radians / std::numbers::pi_v<T>)*static_cast<T>(180);
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::DegreeToRadians, [T], [ float, double ])

template<std::floating_point T>
struct DegreeToRadians : Block<DegreeToRadians<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief convert radians to degree">;
    PortIn<T>  deg;
    PortOut<R> rad;

    GR_MAKE_REFLECTABLE(DegreeToRadians, deg, rad);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(const V& degree) const noexcept {
        return (degree / static_cast<T>(180)) * std::numbers::pi_v<T>;
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::ToRealImag, [T], [ std::complex<float>, std::complex<double> ])

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct ToRealImag : Block<ToRealImag<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"(@brief decompose complex (or arithmetic) numbers their real and imaginary component">;
    PortIn<T>  in;
    PortOut<R> real;
    PortOut<R> imag;

    GR_MAKE_REFLECTABLE(ToRealImag, in, real, imag);

    [[nodiscard]] constexpr std::tuple<R, R> processOne(T complexIn) const noexcept { // some SIMD-fication potential here
        return {static_cast<R>(std::real(complexIn)), static_cast<R>(std::imag(complexIn))};
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::ToRealImag, [T], [ float, double ])

template<std::floating_point T>
struct RealImagToComplex : Block<RealImagToComplex<T>> {
    using R           = std::complex<T>;
    using Description = Doc<"(@brief compose complex (or arithmetic) numbers from their real and imaginary component">;
    PortIn<T>  real;
    PortIn<T>  imag;
    PortOut<R> out;

    GR_MAKE_REFLECTABLE(RealImagToComplex, real, imag, out);

    [[nodiscard]] constexpr R processOne(T re, T im) const noexcept { // some SIMD-fication potential here
        return {re, im};
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::ToRealImag, [T], [ std::complex<float>, std::complex<double> ])

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct ToMagPhase : Block<ToMagPhase<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"(@brief decompose complex (or arithmetic) numbers their magnitude (abs) and phase (arg, [rad]) component">;
    PortIn<T>  in;
    PortOut<R> mag;
    PortOut<R> phase;

    GR_MAKE_REFLECTABLE(ToMagPhase, in, mag, phase);

    [[nodiscard]] constexpr std::tuple<R, R> processOne(T complexIn) const noexcept { // some SIMD-fication potential here
        return {static_cast<R>(std::abs(complexIn)), static_cast<R>(std::arg(complexIn))};
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::ToRealImag, [T], [ float, double ])

template<typename T>
requires std::is_arithmetic_v<T>
struct MagPhaseToComplex : public gr::Block<MagPhaseToComplex<T>> {
    using R           = std::complex<T>;
    using Description = Doc<"(@brief compose complex (or arithmetic) numbers from their abs and phase ([rad]) components">;
    PortIn<T>  mag;
    PortIn<T>  phase;
    PortOut<R> out;

    GR_MAKE_REFLECTABLE(MagPhaseToComplex, mag, phase, out);

    [[nodiscard]] constexpr std::complex<T> processOne(T r, T theta) const noexcept { // some SIMD-fication potential here
        return std::polar(r, theta);
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::ComplexToInterleaved, ([T], [U]), [ std::complex<float>, std::complex<double> ], [ float, double ])

template<typename T, typename R>
requires meta::complex_like<T> && (std::floating_point<R> || std::is_same_v<R, std::int8_t> || std::is_same_v<R, std::int16_t>)
struct ComplexToInterleaved : Block<ComplexToInterleaved<T, R>, Resampling<1UZ, 2UZ, true>> {
    using Description = Doc<R""(@brief convert stream of complex to a stream of interleaved specified type.

The output stream contains twice as many output items as input items.
For every complex input item, we produce two output items that alternate between the real and imaginary component of the complex value.)"">;
    PortIn<T>  in;
    PortOut<R> interleaved;

    GR_MAKE_REFLECTABLE(ComplexToInterleaved, in, interleaved);

    [[nodiscard]] constexpr work::Status processBulk(std::span<const T> complexInput, std::span<R> interleavedOut) const noexcept { // some SIMD-fication potential here (needs permute)
        for (std::size_t i = 0; i < complexInput.size(); ++i) {
            interleavedOut[2 * i]     = static_cast<R>(complexInput[i].real());
            interleavedOut[2 * i + 1] = static_cast<R>(complexInput[i].imag());
        }
        return work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::InterleavedToComplex, ([T], [U]), [ float, double ], [ std::complex<float>, std::complex<double> ])

template<typename T, typename R>
requires(std::floating_point<T> || std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::int16_t>) && meta::complex_like<R>
struct InterleavedToComplex : public gr::Block<InterleavedToComplex<T, R>, gr::Resampling<2UZ, 1UZ, true>> {
    using Description = Doc<R""(@brief convert stream of interleaved values to a stream of complex numbers.

The input stream contains twice as many input items as output items.
For every pair of interleaved input items (real, imag), we produce one complex output item.
)"">;
    gr::PortIn<T>  interleaved;
    gr::PortOut<R> out;

    GR_MAKE_REFLECTABLE(InterleavedToComplex, interleaved, out);

    [[nodiscard]] constexpr work::Status processBulk(std::span<const T> interleavedInput, std::span<R> complexOut) const noexcept { // some SIMD-fication potential here (needs permute)
        for (std::size_t i = 0; i < complexOut.size(); ++i) {
            complexOut[i] = R{static_cast<typename R::value_type>(interleavedInput[2 * i]), static_cast<typename R::value_type>(interleavedInput[2 * i + 1])};
        }
        return work::Status::OK;
    }
};

} // namespace gr::blocks::type::converter

#endif // CONVERTERBLOCKS_HPP
