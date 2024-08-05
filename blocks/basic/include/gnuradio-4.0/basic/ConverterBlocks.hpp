#ifndef CONVERTERBLOCKS_HPP
#define CONVERTERBLOCKS_HPP

#include <tuple>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::type::converter {

template<typename T, typename R>
requires std::is_arithmetic_v<T> && std::is_arithmetic_v<R>
struct Convert : public gr::Block<Convert<T, R>> {
    using Description = Doc<"(@brief basic block to perform a input to output data type conversion (N.B. w/o scaling))">;
    PortIn<T>  in;
    PortOut<R> out;

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

template<typename T, typename R>
requires std::is_arithmetic_v<T> && std::is_arithmetic_v<R>
struct ScalingConvert : public gr::Block<ScalingConvert<T, R>> {
    using Description = Doc<R""(@brief basic block to perform a input to output data type conversion

Performs scaling, i.e. 'R output = R(input * scale)'
)"">;
    PortIn<T>  in;
    PortOut<R> out;
    T          scale = static_cast<T>(1);

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

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct Abs : public gr::Block<Abs<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief basic block to calculate the absolute (magnitude) value of a complex or arithmetic input stream">;
    PortIn<T>  in;
    PortOut<R> abs;

    [[nodiscard]] constexpr R processOne(T input) const noexcept {
        if constexpr (std::is_unsigned_v<T>) {
            using TSigned = std::make_signed_t<T>;
            return static_cast<R>(std::abs(static_cast<TSigned>(input)));
        } else {
            return static_cast<R>(std::abs(input));
        }
    }
};

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct Imag : public gr::Block<Imag<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief basic block to the imaginary component of a complex input stream">;
    PortIn<T>  in;
    PortOut<R> imag;

    [[nodiscard]] constexpr R processOne(T input) const noexcept { return std::imag(input); }
};

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct Real : public gr::Block<Real<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief basic block to the real component of a complex input stream">;
    PortIn<T>  in;
    PortOut<R> real;

    [[nodiscard]] constexpr R processOne(T input) const noexcept { return std::real(input); }
};

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct Arg : public gr::Block<Arg<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief basic block to calculate the argument (phase angle, [radians]) of a complex input stream">;
    PortIn<T>  in;
    PortOut<R> arg;

    [[nodiscard]] constexpr R processOne(T input) const noexcept { return std::arg(input); }
};

template<std::floating_point T>
struct RadiansToDegree : public gr::Block<RadiansToDegree<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief convert radians to degree">;
    PortIn<T>  rad;
    PortOut<R> deg;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(const V& radians) const noexcept {
        return (radians / std::numbers::pi_v<T>)*static_cast<T>(180);
    }
};

template<std::floating_point T>
struct DegreeToRadians : public gr::Block<DegreeToRadians<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"@brief convert radians to degree">;
    PortIn<T>  deg;
    PortOut<R> rad;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(const V& degree) const noexcept {
        return (degree / static_cast<T>(180)) * std::numbers::pi_v<T>;
    }
};

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct ToRealImag : public gr::Block<ToRealImag<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"(@brief decompose complex (or arithmetic) numbers their real and imaginary component">;
    PortIn<T>  in;
    PortOut<R> real;
    PortOut<R> imag;

    [[nodiscard]] constexpr std::tuple<R, R> processOne(T complexIn) const noexcept { // some SIMD-fication potential here
        return {static_cast<R>(std::real(complexIn)), static_cast<R>(std::imag(complexIn))};
    }
};

template<std::floating_point T>
struct RealImagToComplex : public gr::Block<RealImagToComplex<T>> {
    using R           = std::complex<T>;
    using Description = Doc<"(@brief compose complex (or arithmetic) numbers from their real and imaginary component">;
    PortIn<T>  real;
    PortIn<T>  imag;
    PortOut<R> out;

    [[nodiscard]] constexpr R processOne(T re, T im) const noexcept { // some SIMD-fication potential here
        return {re, im};
    }
};

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct ToMagPhase : public gr::Block<ToRealImag<T>> {
    using R           = meta::fundamental_base_value_type_t<T>;
    using Description = Doc<"(@brief decompose complex (or arithmetic) numbers their magnitude (abs) and phase (arg, [rad]) component">;
    PortIn<T>  in;
    PortOut<R> mag;
    PortOut<R> phase;

    [[nodiscard]] constexpr std::tuple<R, R> processOne(T complexIn) const noexcept { // some SIMD-fication potential here
        return {static_cast<R>(std::abs(complexIn)), static_cast<R>(std::arg(complexIn))};
    }
};

template<typename T>
requires std::is_arithmetic_v<T>
struct MagPhaseToComplex : public gr::Block<MagPhaseToComplex<T>> {
    using R           = std::complex<T>;
    using Description = Doc<"(@brief compose complex (or arithmetic) numbers from their abs and phase ([rad]) components">;
    PortIn<T>  mag;
    PortIn<T>  phase;
    PortOut<R> out;

    [[nodiscard]] constexpr std::complex<T> processOne(T r, T theta) const noexcept { // some SIMD-fication potential here
        return std::polar(r, theta);
    }
};

template<typename T, typename R>
requires meta::complex_like<T> && (std::floating_point<R> || std::is_same_v<R, std::int8_t> || std::is_same_v<R, std::int16_t>)
struct ComplexToInterleaved : public gr::Block<ComplexToInterleaved<T, R>, Resampling<1UZ, 2UZ, true>> {
    using Description = Doc<R""(@brief convert stream of complex to a stream of interleaved specified type.

The output stream contains twice as many output items as input items.
For every complex input item, we produce two output items that alternate between the real and imaginary component of the complex value.)"">;
    PortIn<T>  in;
    PortOut<R> interleaved;

    [[nodiscard]] constexpr work::Status processBulk(std::span<const T> complexInput, std::span<R> interleavedOut) const noexcept { // some SIMD-fication potential here (needs permute)
        for (std::size_t i = 0; i < complexInput.size(); ++i) {
            interleavedOut[2 * i]     = static_cast<R>(complexInput[i].real());
            interleavedOut[2 * i + 1] = static_cast<R>(complexInput[i].imag());
        }
        return work::Status::OK;
    }
};

template<typename T, typename R>
requires(std::floating_point<T> || std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::int16_t>) && meta::complex_like<R>
struct InterleavedToComplex : public gr::Block<InterleavedToComplex<T, R>, gr::Resampling<2UZ, 1UZ, true>> {
    using Description = Doc<R""(@brief convert stream of interleaved values to a stream of complex numbers.

The input stream contains twice as many input items as output items.
For every pair of interleaved input items (real, imag), we produce one complex output item.
)"">;
    gr::PortIn<T>  interleaved;
    gr::PortOut<R> out;

    [[nodiscard]] constexpr work::Status processBulk(std::span<const T> interleavedInput, std::span<R> complexOut) const noexcept { // some SIMD-fication potential here (needs permute)
        for (std::size_t i = 0; i < complexOut.size(); ++i) {
            complexOut[i] = R{static_cast<typename R::value_type>(interleavedInput[2 * i]), static_cast<typename R::value_type>(interleavedInput[2 * i + 1])};
        }
        return work::Status::OK;
    }
};

} // namespace gr::blocks::type::converter
ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::Convert, in, out)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::ScalingConvert, in, out, scale)

ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::Abs, in, abs)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::Imag, in, imag)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::Real, in, real)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::Arg, in, arg)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::RadiansToDegree, rad, deg)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::DegreeToRadians, deg, rad)

ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::ToRealImag, in, real, imag)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::RealImagToComplex, real, imag, out)

ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::ToMagPhase, in, mag, phase)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::MagPhaseToComplex, mag, phase, out)

ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::ComplexToInterleaved, in, interleaved)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::blocks::type::converter::InterleavedToComplex, interleaved, out)

/*
 TODO: temporarily disabled due to excessive compile-times on CI
namespace gr::blocks::type::converter {
using TSupportedTypes    = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double>; // N.B. 10 base types
using TComplexTypes      = std::tuple<std::complex<float>, std::complex<double>>;                                               // N.B. 2 (valid) complex types
using TCommonRawSDRTypes = std::tuple<int8_t, int16_t, float, double>;

// clang-format off
const inline auto registerConverterBlocks =
    gr::registerBlockTT<Convert, TSupportedTypes, TSupportedTypes>(gr::globalBlockRegistry()) // N.B. source of long compile-times: 10 x 10 type instantiations
  | gr::registerBlockTT<ScalingConvert, TSupportedTypes, TSupportedTypes>(gr::globalBlockRegistry()) // N.B. source of long compile-times: 10 x 10 type instantiations
  | gr::registerBlock<Abs, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>>(gr::globalBlockRegistry())
  | gr::registerBlock<Imag, std::complex<float>, std::complex<double>>(gr::globalBlockRegistry())
  | gr::registerBlock<Real, std::complex<float>, std::complex<double>>(gr::globalBlockRegistry())
  | gr::registerBlock<Arg, std::complex<float>, std::complex<double>>(gr::globalBlockRegistry())
  | gr::registerBlock<RadiansToDegree, float, double>(gr::globalBlockRegistry())
  | gr::registerBlock<DegreeToRadians, float, double>(gr::globalBlockRegistry())
  | gr::registerBlock<ToRealImag, std::complex<float>, std::complex<double>>(gr::globalBlockRegistry())
  | gr::registerBlock<RealImagToComplex, float, double>(gr::globalBlockRegistry())
  | gr::registerBlock<ToMagPhase, std::complex<float>, std::complex<double>>(gr::globalBlockRegistry())
  | gr::registerBlock<MagPhaseToComplex, float, double>(gr::globalBlockRegistry())
  | gr::registerBlockTT<ComplexToInterleaved, TComplexTypes, TCommonRawSDRTypes>(gr::globalBlockRegistry())
  | gr::registerBlockTT<InterleavedToComplex, TCommonRawSDRTypes, TComplexTypes>(gr::globalBlockRegistry());
// clang-format on
} // namespace gr::blocks::type::converter
*/

#endif // CONVERTERBLOCKS_HPP
