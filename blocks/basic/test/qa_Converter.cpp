#include <gnuradio-4.0/basic/ConverterBlocks.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>

#include <algorithm>
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

            expect(eq(result, static_cast<R>(42))) << std::format("convert from {} to {} failed\n", gr::meta::type_name<T>(), gr::meta::type_name<R>());
        } | TArithmeticTypes();

        "up-convert std::simd<std::uint8_t> to ..."_test = []<typename R>(R /*noop*/) {
            using T = uint8_t;
            // careful here: max_fixed_size is 32 *except with AVX-512 and sizeof(T) == 1* where it is 64.
            using V       = std::conditional_t<stdx::native_simd<T>::size() <= stdx::simd_abi::max_fixed_size<R>, stdx::native_simd<T>, stdx::simd<T, stdx::simd_abi::deduce_t<T, stdx::simd_abi::max_fixed_size<R>>>>;
            using RetType = stdx::rebind_simd_t<R, V>;

            using TConvert = std::conditional_t<kIsScalingBlock, ScalingConvert<T, R>, Convert<T, R>>;
            TConvert converter;
            RetType  result = converter.processOne(/*input*/ static_cast<V>(42));

            for (size_t i = 0; i < RetType::size(); ++i) {
                expect(eq(static_cast<R>(result[i]), static_cast<R>(42))) << std::format("up-convert from {} to {} failed at index {}\n", gr::meta::type_name<V>(), gr::meta::type_name<RetType>(), i);
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
                expect(eq(static_cast<R>(result[i]), static_cast<R>(21))) << std::format("down-convert from {} to {} failed at index {}", gr::meta::type_name<V>(), gr::meta::type_name<RetType>(), i);
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
                expect(eq(static_cast<R>(result[i]), static_cast<R>(42))) << std::format("down-convert from {} to {} failed at index {}", gr::meta::type_name<V>(), gr::meta::type_name<RetType>(), i);
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

        "the conjugate negates the imaginary part only"_test = [] {
            static_assert(HasConstProcessOneFunction<Conjugate<std::complex<T>>>);

            Conjugate<std::complex<T>> conjugate;
            expect(eq(conjugate.processOne(std::complex<T>{1, 2}), std::complex<T>{1, -2}));
            expect(eq(conjugate.processOne(std::complex<T>{-3, -4}), std::complex<T>{-3, 4}));
            expect(eq(conjugate.processOne(std::complex<T>{5, 0}), std::complex<T>{5, 0}));
            const std::complex<T> z{3, 4};
            expect(eq(z * conjugate.processOne(z), std::complex<T>{25, 0})) << "z conj(z) is the power |z|^2";
        };

        "the conjugate of the device-safe gr::complex negates the imaginary part only"_test = [] {
            static_assert(HasConstProcessOneFunction<Conjugate<gr::complex<T>>>);

            Conjugate<gr::complex<T>> conjugate;
            expect(conjugate.processOne(gr::complex<T>{1, 2}) == gr::complex<T>{1, -2});
            expect(conjugate.processOne(gr::complex<T>{-3, -4}) == gr::complex<T>{-3, 4});
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
                expect(approx(val, angle, T(1e-3))) << std::format("identity {}° - diff {}°", angle, (val - angle));
            }
        };

        "complex <-> {real, imag}"_test = [] {
            static_assert(HasConstProcessOneFunction<ToRealImag<std::complex<T>>>);
            static_assert(HasConstProcessOneFunction<RealImagToComplex<T>>);
            expect(HasConstProcessOneFunction<ToRealImag<std::complex<T>>>);
            expect(HasConstProcessOneFunction<RealImagToComplex<T>>);

            ToRealImag<std::complex<T>> complexConverter;
            expect(complexConverter.processOne(std::complex<T>{1, 0}) == std::tuple<T, T>{1, 0});
            expect(complexConverter.processOne(std::complex<T>{0, 1}) == std::tuple<T, T>{0, 1});
            expect(complexConverter.processOne(std::complex<T>{1, 1}) == std::tuple<T, T>{1, 1});

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
            expect(magPhaseConverter.processOne(std::complex<T>{1, 0}) == std::tuple<T, T>{1, 0});
            expect(magPhaseConverter.processOne(std::complex<T>{0, 1}) == std::tuple<T, T>{1, std::numbers::pi_v<T> / 2});
            expect(magPhaseConverter.processOne(std::complex<T>{1, 1}) == std::tuple<T, T>{std::sqrt(2), std::numbers::pi_v<T> / 4});

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

const boost::ut::suite<"ConversionBoundaryTests"> conversionBoundaryTests = [] {
    using namespace boost::ut;
    using namespace gr::blocks::type::converter;
    using gr::testing::operator""_domain_test;
    "floating conversion rounds ties to even and saturates with NaN mapped to zero"_domain_test = [](gr::testing::DomainContext& domain) {
        domain.launchRange(9UZ, [](const gr::testing::DeviceTestHandle& handle, std::size_t i) {
            constexpr std::array<float, 9UZ>        kInput{-std::numeric_limits<float>::infinity(), -32768.6f, -0.5f, 0.5f, 1.5f, 2.5f, 32767.6f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
            constexpr std::array<std::int16_t, 9UZ> kExpected{-32768, -32768, 0, 0, 2, 2, 32767, 32767, 0};
            gr::testing::expect(handle, convertSample<std::int16_t>(kInput[i]) == kExpected[i]);
        });
    } | gr::testing::kAllDomains;
    "integer to float accepts a fractional scale and unit scale preserves uint64"_test = [] {
        ScalingConvert<std::int16_t, float> convert;
        convert.scale = 1.f / 32768.f;
        expect(eq(convert.processOne(std::int16_t{-32768}), -1.f));
        ScalingConvert<std::uint64_t, std::uint64_t> exact;
        expect(eq(exact.processOne(std::numeric_limits<std::uint64_t>::max()), std::numeric_limits<std::uint64_t>::max()));
    };
    "interleaved IQ scaling and swapping are explicit in each direction"_test = [] {
        ComplexToInterleaved<std::complex<float>, std::int8_t> split;
        split.scale   = 2.f;
        split.swap_iq = true;
        const std::array<std::complex<float>, 1UZ> source{std::complex<float>{1.5f, -2.5f}};
        std::array<std::int8_t, 2UZ>               interleaved{};
        expect(split.processBulk(source, interleaved) == gr::work::Status::OK);
        expect(eq(interleaved[0], std::int8_t{-5}) && eq(interleaved[1], std::int8_t{3}));
        InterleavedToComplex<std::int8_t, std::complex<float>> join;
        join.scale   = 0.5f;
        join.swap_iq = true;
        std::array<std::complex<float>, 1UZ> restored{};
        expect(join.processBulk(interleaved, restored) == gr::work::Status::OK);
        expect(eq(restored[0], source[0]));
    };
    "char and short scaling follows signed eight bit shifts"_test = [] {
        ScalingConvert<std::int8_t, std::int16_t> widen;
        widen.scale = 256.0;
        expect(eq(widen.processOne(std::int8_t{-128}), std::int16_t{-32768}));
        expect(eq(widen.processOne(std::int8_t{127}), std::int16_t{32512}));
        ScalingConvert<std::int16_t, std::int8_t> narrow;
        narrow.scale = 1.0 / 256.0;
        constexpr std::array<std::int16_t, 8UZ> kInput{-32768, -257, -256, -1, 0, 255, 256, 32767};
        constexpr std::array<std::int8_t, 8UZ>  kExpected{-128, -2, -1, -1, 0, 0, 1, 127};
        for (std::size_t i = 0UZ; i < kInput.size(); ++i) {
            expect(eq(narrow.processOne(kInput[i]), kExpected[i]));
        }
    };

    "integer narrowing saturates instead of wrapping, with or without scale"_test = [] {
        expect(eq(Convert<std::int8_t, std::uint8_t>().processOne(std::int8_t{-1}), std::uint8_t{0}));
        expect(eq(Convert<std::int32_t, std::int16_t>().processOne(40000), std::int16_t{32767}));
        expect(eq(Convert<std::int32_t, std::int16_t>().processOne(-40000), std::int16_t{-32768}));
        expect(eq(Convert<std::uint64_t, std::int64_t>().processOne(std::numeric_limits<std::uint64_t>::max()), std::numeric_limits<std::int64_t>::max()));
        expect(eq(Convert<std::int64_t, std::uint64_t>().processOne(std::int64_t{-5}), std::uint64_t{0}));
        expect(eq(Convert<std::int64_t, std::uint64_t>().processOne(std::numeric_limits<std::int64_t>::max()), std::uint64_t{9223372036854775807ULL}));
        expect(eq(ScalingConvert<std::int32_t, std::int16_t>().processOne(40000), std::int16_t{32767})) << "scale 1 saturates as any other scale";
    };

    "double to int32 rounds half to even and saturates at both ends"_test = [] {
        Convert<double, std::int32_t> convert;
        expect(eq(convert.processOne(2.5), 2));
        expect(eq(convert.processOne(3.5), 4));
        expect(eq(convert.processOne(-2.5), -2));
        expect(eq(convert.processOne(1.e10), std::numeric_limits<std::int32_t>::max()));
        expect(eq(convert.processOne(-1.e10), std::numeric_limits<std::int32_t>::lowest()));
        expect(eq(Convert<float, std::int64_t>().processOne(1.e19f), std::numeric_limits<std::int64_t>::max()));
    };

    "a SIMD narrowing saturates each lane, a SIMD widening is exact"_test = [] {
        using V = vir::stdx::native_simd<std::int32_t>;
        const V    input([](auto lane) { return static_cast<std::int32_t>(lane) * 20000 - 40000; });
        const auto narrowed = Convert<std::int32_t, std::int16_t>().processOne(input);
        const auto widened  = Convert<std::int32_t, double>().processOne(input);
        for (std::size_t lane = 0UZ; lane < V::size(); ++lane) {
            const std::int32_t value = static_cast<std::int32_t>(lane) * 20000 - 40000; // -40000, -20000, 0, 20000, 40000, ...
            expect(eq(narrowed[lane], static_cast<std::int16_t>(std::clamp(value, -32768, 32767))));
            expect(eq(widened[lane], static_cast<double>(input[lane])));
        }
    };

    "a three plus four i sample has power twenty five"_test = [] {
        gr::blocks::type::converter::MagnitudeSquared<std::complex<double>> magnitude;
        expect(eq(magnitude.processOne({3.0, 4.0}), 25.0));
        expect(eq(magnitude.processOne({0.0, 0.0}), 0.0));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
