#include <gnuradio-4.0/basic/ConverterBlocks.hpp>

#include <iostream>
#include <tuple>

namespace unittest {
template<class T>
[[nodiscard]] constexpr auto abs_diff(const std::complex<T>& t, const std::complex<T>& u) {
    return std::abs(t - u);
}
} // namespace unittest

template<typename... Ts>
inline constexpr std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& tuple) noexcept {
    std::apply(
        [&os](const Ts&... args) {
            os << '(';
            std::size_t n = 0;
            ((os << args << (++n != sizeof...(Ts) ? ", " : "")), ...);
            os << ')';
        },
        tuple);
    return os;
}

#include <boost/ut.hpp>

const boost::ut::suite<"basic Conversion tests"> basicConversion = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::type::converter;
    namespace stdx = vir::stdx;

    using TArithmeticTypes = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double>;
    struct ConvertBlock {};
    struct ScalingConvertBlock {};

    "ConversionTest"_test = []<typename TSwitch>(TSwitch&& /*noop*/) {
        constexpr static bool kIsScalingBlock = std::is_same_v<TSwitch, ScalingConvertBlock>;

        "static block properties"_test = [] {
            using TBlock = std::conditional_t<kIsScalingBlock, ScalingConvert<int32_t, float>, Convert<int32_t, float>>;
            static_assert(gr::traits::block::can_processOne_scalar<TBlock>);
            static_assert(gr::traits::block::can_processOne_simd<TBlock>);
            static_assert(HasProcessOneFunction<TBlock>);
            static_assert(HasConstProcessOneFunction<TBlock>);
            static_assert(HasRequiredProcessFunction<TBlock>);

            // To force the runtime code-coverage
            expect(HasProcessOneFunction<TBlock>);
            expect(HasConstProcessOneFunction<TBlock>);
        };

        "up-convert std::uint8_t to ..."_test = []<typename R>(R /*noop*/) {
            using T        = uint8_t;
            using TConvert = std::conditional_t<kIsScalingBlock, ScalingConvert<T, R>, Convert<T, R>>;
            TConvert converter;
            R        result = converter.processOne(/*input*/ static_cast<T>(42));

            expect(eq(result, static_cast<R>(42))) << fmt::format("convert from {} to {} failed\n", gr::meta::type_name<T>(), gr::meta::type_name<R>());
        } | TArithmeticTypes();

        "up-convert std::simd<std::uint8_t> to ..."_test = []<typename R>(R /*noop*/) {
            using T       = uint8_t;
            // careful here: max_fixed_size is 32 *except with AVX-512 and sizeof(T) == 1* where it is 64.
            using V       = std::conditional_t<stdx::native_simd<T>::size() <= stdx::simd_abi::max_fixed_size<R>, stdx::native_simd<T>, stdx::simd<T, stdx::simd_abi::deduce_t<T, stdx::simd_abi::max_fixed_size<R>>>>;
            using RetType = stdx::rebind_simd_t<R, V>;

            using TConvert = std::conditional_t<kIsScalingBlock, ScalingConvert<T, R>, Convert<T, R>>;
            TConvert converter;
            RetType  result = converter.processOne(/*input*/ static_cast<V>(42));

            for (size_t i = 0; i < RetType::size(); ++i) {
                expect(eq(static_cast<R>(result[i]), static_cast<R>(42))) << fmt::format("up-convert from {} to {} failed at index {}\n", gr::meta::type_name<V>(), gr::meta::type_name<RetType>(), i);
            }
        } | TArithmeticTypes();

        "down-convert std::simd<float> to ..."_test = []<typename R>(R /*noop*/) {
            using T       = float;
            using V       = stdx::native_simd<T>;
            using RetType = stdx::rebind_simd_t<R, V>;

            using TConvert = std::conditional_t<kIsScalingBlock, ScalingConvert<T, R>, Convert<T, R>>;
            TConvert converter;
            RetType  result = converter.processOne(/*input*/ static_cast<V>(42));
            if constexpr (kIsScalingBlock) {
                converter.scale = static_cast<T>(0.5);
                result          = converter.processOne(/*input*/ static_cast<V>(42));
            } else {
                result = converter.processOne(/*input*/ static_cast<V>(21));
            }

            for (size_t i = 0; i < RetType::size(); ++i) {
                expect(eq(static_cast<R>(result[i]), static_cast<R>(21))) << fmt::format("down-convert from {} to {} failed at index {}", gr::meta::type_name<V>(), gr::meta::type_name<RetType>(), i);
            }
        } | TArithmeticTypes();

        "down-convert std::simd<int64_t> to ..."_test = []<typename R>(R /*noop*/) {
            using T       = int64_t;
            using V       = stdx::native_simd<T>;
            using RetType = stdx::rebind_simd_t<R, V>;

            using TConvert = std::conditional_t<kIsScalingBlock, ScalingConvert<T, R>, Convert<T, R>>;
            TConvert converter;
            if constexpr (kIsScalingBlock) {
                converter.scale = static_cast<T>(1.0);
            }
            RetType result = converter.processOne(/*input*/ static_cast<V>(42));

            for (size_t i = 0; i < RetType::size(); ++i) {
                expect(eq(static_cast<R>(result[i]), static_cast<R>(42))) << fmt::format("down-convert from {} to {} failed at index {}", gr::meta::type_name<V>(), gr::meta::type_name<RetType>(), i);
            }
        } | TArithmeticTypes();
    } | std::tuple<ConvertBlock, ScalingConvertBlock>();
};

const boost::ut::suite<"complex To/From conversion tests"> complexConversion = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::type::converter;
    namespace stdx = vir::stdx;

    constexpr auto kArithmeticTypes = std::tuple<float, double>();

    "std::complex<T> to ..."_test = []<typename T>(T /*noop*/) {
        "std::abs([complex<T>, T])"_test = [] {
            static_assert(HasConstProcessOneFunction<Abs<std::complex<T>>>);
            static_assert(HasConstProcessOneFunction<Abs<T>>);
            expect(HasConstProcessOneFunction<Abs<std::complex<T>>>);
            expect(HasConstProcessOneFunction<Abs<T>>);

            Abs<std::complex<T>> complexConverter;
            expect(eq(complexConverter.processOne(std::complex<T>{1, 0}), T(1)));
            expect(eq(complexConverter.processOne(std::complex<T>{0, 1}), T(1)));
            expect(eq(complexConverter.processOne(std::complex<T>{1, 1}), T(std::sqrt(2))));

            Abs<T> converter;
            expect(eq(converter.processOne(T(1)), T(1)));
        };

        "std::Real([complex<T>, T])"_test = [] {
            static_assert(HasConstProcessOneFunction<Real<std::complex<T>>>);
            static_assert(HasConstProcessOneFunction<Real<T>>);
            expect(HasConstProcessOneFunction<Real<std::complex<T>>>);
            expect(HasConstProcessOneFunction<Real<T>>);

            Real<std::complex<T>> complexConverter;
            expect(eq(complexConverter.processOne(std::complex<T>{1, 0}), T(1)));
            expect(eq(complexConverter.processOne(std::complex<T>{0, 1}), T(0)));
            expect(eq(complexConverter.processOne(std::complex<T>{1, 1}), T(1)));

            Real<T> converter;
            expect(eq(converter.processOne(T(1)), T(1)));
        };

        "std::Imag([complex<T>, T])"_test = [] {
            static_assert(HasConstProcessOneFunction<Imag<std::complex<T>>>);
            static_assert(HasConstProcessOneFunction<Imag<T>>);
            expect(HasConstProcessOneFunction<Imag<std::complex<T>>>);
            expect(HasConstProcessOneFunction<Imag<T>>);

            Imag<std::complex<T>> complexConverter;
            expect(eq(complexConverter.processOne(std::complex<T>{1, 0}), T(0)));
            expect(eq(complexConverter.processOne(std::complex<T>{0, 1}), T(1)));
            expect(eq(complexConverter.processOne(std::complex<T>{1, 1}), T(1)));

            Imag<T> converter;
            expect(eq(converter.processOne(T(1)), T(0)));
        };

        "std::arg([complex<T>, T])"_test = [] {
            static_assert(HasConstProcessOneFunction<Arg<std::complex<T>>>);
            static_assert(HasConstProcessOneFunction<Arg<T>>);
            expect(HasConstProcessOneFunction<Arg<std::complex<T>>>);
            expect(HasConstProcessOneFunction<Arg<T>>);

            Arg<std::complex<T>> complexConverter;
            expect(eq(complexConverter.processOne(std::complex<T>{1, 0}), T(0.0) * std::numbers::pi_v<T>));
            expect(eq(complexConverter.processOne(std::complex<T>{0, 1}), T(0.5) * std::numbers::pi_v<T>));
            expect(eq(complexConverter.processOne(std::complex<T>{1, 1}), T(0.25) * std::numbers::pi_v<T>));

            Arg<T> converter;
            expect(eq(converter.processOne(T(1)), T(0)));
        };

        "rad <-> deg"_test = [] {
            static_assert(HasConstProcessOneFunction<RadiansToDegree<T>>);
            static_assert(HasConstProcessOneFunction<DegreeToRadians<T>>);
            expect(HasConstProcessOneFunction<RadiansToDegree<T>>);
            expect(HasConstProcessOneFunction<DegreeToRadians<T>>);

            RadiansToDegree<T> rad2deg;
            expect(eq(rad2deg.processOne(T(0)), T(0))) << "0 [rad] = 0°";
            expect(eq(rad2deg.processOne(T(0.5) * std::numbers::pi_v<T>), T(90))) << "pi/2 [rad] = 90°";
            expect(eq(rad2deg.processOne(std::numbers::pi_v<T>), T(180))) << "pi [rad] = 180°";

            DegreeToRadians<T> deg2rad;
            expect(eq(deg2rad.processOne(T(0)), T(0))) << "0° = 0 [rad]";
            expect(eq(deg2rad.processOne(T(90)), T(0.5) * std::numbers::pi_v<T>)) << "90° = pi/2 [rad]";
            expect(eq(deg2rad.processOne(T(180)), T(std::numbers::pi_v<T>))) << "180° = pi [rad]";

            for (T angle = T(0); angle < T(360); angle += T(10)) {
                T val = rad2deg.processOne(deg2rad.processOne(angle));
                expect(approx(val, angle, T(1e-3))) << fmt::format("identity {}° - diff {}°", angle, (val - angle));
            }
        };

        "complex <-> {real, imag}"_test = [] {
            static_assert(HasConstProcessOneFunction<ToRealImag<std::complex<T>>>);
            static_assert(HasConstProcessOneFunction<RealImagToComplex<T>>);
            expect(HasConstProcessOneFunction<ToRealImag<std::complex<T>>>);
            expect(HasConstProcessOneFunction<RealImagToComplex<T>>);

            ToRealImag<std::complex<T>> complexConverter;
            expect(eq(complexConverter.processOne(std::complex<T>{1, 0}), std::tuple<T, T>{1, 0}));
            expect(eq(complexConverter.processOne(std::complex<T>{0, 1}), std::tuple<T, T>{0, 1}));
            expect(eq(complexConverter.processOne(std::complex<T>{1, 1}), std::tuple<T, T>{1, 1}));

            RealImagToComplex<T> realImagConverter;
            expect(eq(realImagConverter.processOne(T(1), T(0)), std::complex<T>{1, 0}));
            expect(eq(realImagConverter.processOne(T(0), T(1)), std::complex<T>{0, 1}));
            expect(eq(realImagConverter.processOne(T(1), T(1)), std::complex<T>{1, 1}));
        };

        "complex <-> {magnitude, phase}"_test = [] {
            static_assert(HasConstProcessOneFunction<ToMagPhase<std::complex<T>>>);
            static_assert(HasConstProcessOneFunction<MagPhaseToComplex<T>>);
            expect(HasConstProcessOneFunction<ToMagPhase<std::complex<T>>>);
            expect(HasConstProcessOneFunction<MagPhaseToComplex<T>>);

            ToMagPhase<std::complex<T>> magPhaseConverter;
            expect(eq(magPhaseConverter.processOne(std::complex<T>{1, 0}), std::tuple<T, T>{1, 0}));
            expect(eq(magPhaseConverter.processOne(std::complex<T>{0, 1}), std::tuple<T, T>{1, std::numbers::pi_v<T> / 2}));
            expect(eq(magPhaseConverter.processOne(std::complex<T>{1, 1}), std::tuple<T, T>{std::sqrt(2), std::numbers::pi_v<T> / 4}));

            MagPhaseToComplex<T> magPhaseToComplexConverter;
            expect(unittest::abs_diff(magPhaseToComplexConverter.processOne(T(1), T(0)), std::complex<T>{1, 0}) < T(1e-3));
            expect(unittest::abs_diff(magPhaseToComplexConverter.processOne(T(1), T(std::numbers::pi_v<T> / 2)), std::complex<T>{0, 1}) < T(1e-3));
            expect(unittest::abs_diff(magPhaseToComplexConverter.processOne(T(std::sqrt(2)), T(std::numbers::pi_v<T> / 4)), std::complex<T>{1, 1}) < T(1e-3));
        };

        "complex <-> interleaved"_test = []<typename R> {
            static_assert(HasNoexceptProcessBulkFunction<ComplexToInterleaved<std::complex<T>, R>>);
            static_assert(HasNoexceptProcessBulkFunction<InterleavedToComplex<R, std::complex<T>>>);
            expect(HasNoexceptProcessBulkFunction<ComplexToInterleaved<std::complex<T>, R>>);
            expect(HasNoexceptProcessBulkFunction<InterleavedToComplex<R, std::complex<T>>>);

            std::vector<std::complex<T>> complexData = {{1, 2}, {3, 4}, {5, 6}};
            std::vector<T>               expected{1, 2, 3, 4, 5, 6};
            std::vector<R>               interleavedData(complexData.size() * 2);
            std::vector<std::complex<T>> outputComplexData(complexData.size());

            ComplexToInterleaved<std::complex<T>, R> complexToInterleaved;
            InterleavedToComplex<R, std::complex<T>> interleavedToComplex;

            expect(complexToInterleaved.processBulk(complexData, interleavedData) == gr::work::Status::OK);
            expect(eq(interleavedData[0], R(1.0f)));
            expect(eq(interleavedData[1], R(2.0f)));
            expect(eq(interleavedData[2], R(3.0f)));
            expect(eq(interleavedData[3], R(4.0f)));
            expect(eq(interleavedData[4], R(5.0f)));
            expect(eq(interleavedData[5], R(6.0f)));

            expect(interleavedToComplex.processBulk(interleavedData, outputComplexData) == gr::work::Status::OK);
            expect(eq(outputComplexData[0], std::complex<T>{T(1.0f), T(2.0f)}));
            expect(eq(outputComplexData[1], std::complex<T>{T(3.0f), T(4.0f)}));
            expect(eq(outputComplexData[2], std::complex<T>{T(5.0f), T(6.0f)}));
        } | std::tuple<float, double, std::int8_t, std::int16_t>();
    } | kArithmeticTypes;
};

int main() { /* not needed for UT */ }
