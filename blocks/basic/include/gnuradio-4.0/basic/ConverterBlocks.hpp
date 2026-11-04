#ifndef CONVERTERBLOCKS_HPP
#define CONVERTERBLOCKS_HPP

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::type::converter {

// floating to integer: round to nearest (ties to even), saturate, NaN to 0; integer to integer: saturate; to floating: plain cast
template<typename R, typename T>
[[nodiscard]] constexpr R convertSample(T value) noexcept {
    if constexpr (std::floating_point<T> && std::integral<R>) {
        if (std::isnan(value)) {
            return R{};
        }
        const T rounded = std::nearbyint(value);
        if (rounded >= static_cast<T>(std::numeric_limits<R>::max())) {
            return std::numeric_limits<R>::max();
        }
        if (rounded <= static_cast<T>(std::numeric_limits<R>::lowest())) {
            return std::numeric_limits<R>::lowest();
        }
        return static_cast<R>(rounded);
    } else if constexpr (std::integral<T> && std::integral<R>) {
        if (std::in_range<R>(value)) {
            return static_cast<R>(value);
        }
        return std::cmp_less(value, 0) ? std::numeric_limits<R>::lowest() : std::numeric_limits<R>::max();
    } else {
        return static_cast<R>(value);
    }
}

/// true when `convertSample<R>` is a plain cast, so a SIMD cast gives the same result
template<typename T, typename R>
inline constexpr bool kConvertsByCast = [] {
    if constexpr (std::floating_point<R>) {
        return true;
    } else if constexpr (std::integral<T> && std::integral<R>) {
        return std::in_range<R>(std::numeric_limits<T>::lowest()) && std::in_range<R>(std::numeric_limits<T>::max());
    } else {
        return false;
    }
}();

GR_REGISTER_BLOCK(gr::blocks::type::converter::Convert, ([T], [U]), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T, typename R>
requires std::is_arithmetic_v<T> && std::is_arithmetic_v<R>
struct Convert : Block<Convert<T, R>> {
    using Description = Doc<"Converts each sample to R without scaling: floating to integer rounds to nearest even, saturates and maps NaN to 0; integer narrowing saturates.">;
    PortIn<T>  in;
    PortOut<R> out;

    GR_MAKE_REFLECTABLE(Convert, in, out);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(const V& input) const noexcept {
        if constexpr (gr::meta::any_simd<V>) {
            using RetType = vir::stdx::rebind_simd_t<R, V>;
            if constexpr (kConvertsByCast<T, R>) {
                return vir::stdx::static_simd_cast<RetType>(input);
            } else {
                return RetType([&](auto lane) { return convertSample<R>(static_cast<T>(input[lane])); });
            }
        } else {
            return convertSample<R>(input);
        }
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::ScalingConvert, ([T], [U]), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T, typename R>
requires std::is_arithmetic_v<T> && std::is_arithmetic_v<R>
struct ScalingConvert : Block<ScalingConvert<T, R>> {
    using Description = Doc<"Multiplies by scale and converts like Convert; integer-to-integer scaling floors the product, matching fixed-point high-byte extraction.">;
    using Scale       = std::conditional_t<std::integral<T> && std::integral<R>, double, std::common_type_t<T, R, float>>;

    PortIn<T>  in;
    PortOut<R> out;

    Annotated<Scale, "scale", Doc<"factor applied before conversion, e.g. 1/32768 for int16 to float">> scale = Scale(1);

    GR_MAKE_REFLECTABLE(ScalingConvert, in, out, scale);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(const V& input) const noexcept {
        const auto scaled = [factor = scale.value](T sample) {
            if (factor == Scale{1}) {
                return convertSample<R>(sample);
            }
            const Scale product = static_cast<Scale>(sample) * factor;
            if constexpr (std::integral<T> && std::integral<R>) {
                return convertSample<R>(std::floor(product));
            } else {
                return convertSample<R>(product);
            }
        };
        if constexpr (gr::meta::any_simd<V>) {
            using RetType = vir::stdx::rebind_simd_t<R, V>;
            if constexpr (std::floating_point<R>) {
                return vir::stdx::static_simd_cast<RetType>(vir::stdx::static_simd_cast<vir::stdx::rebind_simd_t<Scale, V>>(input) * scale.value);
            } else {
                return RetType([&](auto lane) { return scaled(static_cast<T>(input[lane])); });
            }
        } else {
            return scaled(input);
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

GR_REGISTER_BLOCK(gr::blocks::type::converter::Conjugate, [T], [ std::complex<float>, std::complex<double>, gr::complex<float>, gr::complex<double> ])

template<meta::complex_like T>
struct Conjugate : Block<Conjugate<T>> {
    using Description = Doc<"@brief complex conjugate of each sample; `x * conj(y)` is the cross-spectrum of two spectra">;
    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(Conjugate, in, out);

    [[nodiscard]] constexpr T processOne(T input) const noexcept {
        using std::conj;
        return conj(input);
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::SwapEndianness, [T], [ uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires(std::is_arithmetic_v<T> || meta::complex_like<T>) && (sizeof(meta::fundamental_base_value_type_t<T>) == 2UZ || sizeof(meta::fundamental_base_value_type_t<T>) == 4UZ || sizeof(meta::fundamental_base_value_type_t<T>) == 8UZ)
struct SwapEndianness : Block<SwapEndianness<T>> {
    using Description = Doc<"@brief reverses the byte order of each sample, a complex sample's two parts each on their own (GR3: Endian Swap)">;

    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(SwapEndianness, in, out);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V& sample) const noexcept {
        using R            = meta::fundamental_base_value_type_t<T>;
        using Bits         = std::conditional_t<sizeof(R) == 2UZ, std::uint16_t, std::conditional_t<sizeof(R) == 4UZ, std::uint32_t, std::uint64_t>>;
        const auto swapped = [](auto value) {
            const auto swappedOne = [](R one) { return std::bit_cast<R>(std::byteswap(std::bit_cast<Bits>(one))); };
            if constexpr (meta::any_simd<decltype(value)>) {
                return decltype(value)([&](auto lane) { return swappedOne(static_cast<R>(value[lane])); });
            } else {
                return swappedOne(value);
            }
        };
        if constexpr (meta::complex_like<T>) {
            return T(swapped(sample.real()), swapped(sample.imag()));
        } else {
            return swapped(sample);
        }
    }
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

GR_REGISTER_BLOCK(gr::blocks::type::converter::RealImagToComplex, [T], [ float, double ])

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

GR_REGISTER_BLOCK(gr::blocks::type::converter::ToMagPhase, [T], [ std::complex<float>, std::complex<double> ])

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

GR_REGISTER_BLOCK(gr::blocks::type::converter::MagPhaseToComplex, [T], [ float, double ])

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

GR_REGISTER_BLOCK(gr::blocks::type::converter::ComplexToInterleaved, ([T], [U]), [ std::complex<float>, std::complex<double> ], [ float, double, int8_t, int16_t ])

template<typename T, typename R>
requires meta::complex_like<T> && (std::floating_point<R> || std::is_same_v<R, std::int8_t> || std::is_same_v<R, std::int16_t>)
struct ComplexToInterleaved : Block<ComplexToInterleaved<T, R>, Resampling<1UZ, 2UZ, true>> {
    using Description = Doc<"Interleaves each complex sample as (I, Q), or (Q, I) with swap_iq, times scale; to integer it rounds to nearest even and saturates.">;
    using Scale       = typename T::value_type;

    PortIn<T>  in;
    PortOut<R> interleaved;

    Annotated<Scale, "scale", Doc<"factor applied before conversion">>            scale   = Scale(1);
    Annotated<bool, "swap_iq", Doc<"emit the imaginary part first (Q, I order)">> swap_iq = false;

    GR_MAKE_REFLECTABLE(ComplexToInterleaved, in, interleaved, scale, swap_iq);

    [[nodiscard]] constexpr work::Status processBulk(std::span<const T> complexInput, std::span<R> interleavedOut) const noexcept { // some SIMD-fication potential here (needs permute)
        for (std::size_t i = 0; i < complexInput.size(); ++i) {
            interleavedOut[2 * i]     = convertSample<R>((swap_iq ? complexInput[i].imag() : complexInput[i].real()) * scale.value);
            interleavedOut[2 * i + 1] = convertSample<R>((swap_iq ? complexInput[i].real() : complexInput[i].imag()) * scale.value);
        }
        return work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::InterleavedToComplex, ([T], [U]), [ float, double, int8_t, int16_t ], [ std::complex<float>, std::complex<double> ])

template<typename T, typename R>
requires(std::floating_point<T> || std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::int16_t>) && meta::complex_like<R>
struct InterleavedToComplex : public gr::Block<InterleavedToComplex<T, R>, gr::Resampling<2UZ, 1UZ, true>> {
    using Description = Doc<"Joins each (I, Q) pair, or (Q, I) with swap_iq, into one complex sample, times scale.">;
    using Scale       = typename R::value_type;

    gr::PortIn<T>  interleaved;
    gr::PortOut<R> out;

    Annotated<Scale, "scale", Doc<"factor applied after conversion">>                    scale   = Scale(1);
    Annotated<bool, "swap_iq", Doc<"the input carries the imaginary part first (Q, I)">> swap_iq = false;

    GR_MAKE_REFLECTABLE(InterleavedToComplex, interleaved, out, scale, swap_iq);

    [[nodiscard]] constexpr work::Status processBulk(std::span<const T> interleavedInput, std::span<R> complexOut) const noexcept { // some SIMD-fication potential here (needs permute)
        for (std::size_t i = 0; i < complexOut.size(); ++i) {
            complexOut[i] = R{static_cast<Scale>(interleavedInput[2 * i + (swap_iq ? 1UZ : 0UZ)]) * scale.value, static_cast<Scale>(interleavedInput[2 * i + (swap_iq ? 0UZ : 1UZ)]) * scale.value};
        }
        return work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::blocks::type::converter::MagnitudeSquared, [T], [ std::complex<float>, std::complex<double> ])

template<meta::complex_like T>
struct MagnitudeSquared : Block<MagnitudeSquared<T>> {
    using Description = Doc<"Produces the squared magnitude of each complex sample.">;
    using R           = meta::fundamental_base_value_type_t<T>;
    PortIn<T>  in;
    PortOut<R> out;
    GR_MAKE_REFLECTABLE(MagnitudeSquared, in, out);

    [[nodiscard]] constexpr R processOne(T sample) const noexcept { return std::norm(sample); }
};

} // namespace gr::blocks::type::converter

#endif // CONVERTERBLOCKS_HPP
