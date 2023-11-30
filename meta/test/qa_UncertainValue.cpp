#include <boost/ut.hpp>

#include <complex>

#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include <gnuradio-4.0/meta/formatter.hpp>

#include <fmt/format.h>

namespace test::detail {

template<class TLhs, class TRhs, class TEpsilon>
[[nodiscard]] constexpr auto
approx(const TLhs &lhs, const TRhs &rhs, const TEpsilon &epsilon) {
    if constexpr (gr::meta::complex_like<TLhs>) {
        return boost::ut::detail::and_{ boost::ut::detail::approx_{ lhs.real(), rhs.real(), epsilon }, boost::ut::detail::approx_{ lhs.imag(), rhs.imag(), epsilon } };
    } else {
        return boost::ut::detail::approx_{ lhs, rhs, epsilon };
    }
}
} // namespace test::detail

const boost::ut::suite uncertainValue = [] {
    using namespace boost::ut;
    using namespace std::literals::complex_literals;
    using namespace gr;
    using test::detail::approx;

    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        // conditionally enable visual tests outside the CI
        boost::ext::ut::cfg<override> = { .tag = { "visual", "benchmarks" } };
    }

    "addition UncertainValue + UncertainValue"_test = [] {
        constexpr UncertainValue<double> a{ 10.0, 3.0 };
        constexpr UncertainValue<double> b{ 20.0, 4.0 };

        expect(30.0_d == (a + b).value) << "value: 10. + 20.";
        expect(5.0_d == (a + b).uncertainty) << "uncertainty: sqrt(3.0^2 + 4.0^2) = sqrt(9 + 16) = sqrt(25) = 5"; //
    };

    "addition UncertainValue + arithmetic value"_test = [] {
        UncertainValue<double> a{ 10.0, 3.0 };
        const double           b{ 20.0 };

        expect(30.0_d == (a + b).value) << "value: 10. + 20.";
        expect(3.0_d == (a + b).uncertainty) << "uncertainty: 3.0 (no change)";
    };

    "addition arithmetic value + UncertainValue"_test = [] {
        double                 a{ 10.0 };
        UncertainValue<double> b{ 20.0, 4.0 };

        expect(30.0_d == (a + b).value) << "value: 10. + 20.";
        expect(4.0_d == (a + b).uncertainty) << "uncertainty: 3.0 (no change)";
    };

    "addition arithmetic value + arithmetic value"_test = [] {
        constexpr const double a{ 10.0 };
        constexpr const double b{ 20.0 };

        constexpr UncertainValue<double> result1 = a + b;       // lvalue assignment
        constexpr UncertainValue<double> result2 = 10.0 + 20.0; // rvalue assignment
        expect(30.0_d == result1.value) << "value: 10. + 20.";
        expect(0.0_d == result1.uncertainty) << "uncertainty: 0.0 (no change)";
        expect(30.0_d == result2.value) << "value: 10. + 20.";
        expect(0.0_d == result2.uncertainty) << "uncertainty: 0.0 (no change)";
    };

    "operator += UncertainValue"_test = [] {
        UncertainValue<double> a{ 10.0, 2.0 };
        UncertainValue<double> b{ 5.0, 1.5 };
        a += b;
        expect(eq(15.0, a.value));
        expect(eq(2.5, a.uncertainty)); // Assuming appropriate uncertainty calculation
    };

    "subtraction UncertainValue - UncertainValue"_test = [] {
        constexpr UncertainValue<double> a{ 20.0, 3.0 };
        constexpr UncertainValue<double> b{ 10.0, 4.0 };

        expect(10.0_d == (a - b).value) << "value: 20. - 10.";
        expect(5.0_d == (a - b).uncertainty) << "uncertainty: sqrt(3.0^2 + 4.0^2) = sqrt(9 + 16) = sqrt(25) = 5";
    };

    "subtraction UncertainValue - arithmetic value"_test = [] {
        constexpr UncertainValue<double> a{ 20.0, 3.0 };
        constexpr const double           b{ 10.0 };

        expect(10.0_d == (a - b).value) << "value: 20. - 10.";
        expect(3.0_d == (a - b).uncertainty) << "uncertainty: 3.0 (no change)";
    };

    "subtraction arithmetic value - UncertainValue"_test = [] {
        constexpr double                 a{ 20.0 };
        constexpr UncertainValue<double> b{ 10.0, 4.0 };

        expect(10.0_d == (a - b).value) << "value: 20. - 10.";
        expect(4.0_d == (a - b).uncertainty) << "uncertainty: sqrt(3.0^2 + 4.0^2) = sqrt(9 + 16) = sqrt(25) = 5";
    };

    "subtraction arithmetic value - arithmetic value"_test = [] {
        constexpr double a{ 20.0 };
        constexpr double b{ 10.0 };

        constexpr UncertainValue<double> result = a - b;
        expect(constant<10.0_d == result.value>) << "value: 20. - 10.";
        expect(constant<0.0_d == result.uncertainty>) << "uncertainty: 0.0 (no change)";
    };

    "operator -= UncertainValue"_test = [] {
        constexpr UncertainValue<double> a0{ 4.0, 0.5 };
        UncertainValue<double>           a{ 4.0, 0.5 };
        UncertainValue<double>           b{ 3.0, 0.2 };
        a -= b;
        expect(eq((a0 - b).value, a.value));
        expect(eq((a0 - b).uncertainty, a.uncertainty)); // Assuming appropriate uncertainty calculation
    };

    "multiplication UncertainValue * UncertainValue"_test = [] {
        constexpr UncertainValue<double> a{ 4.0, 0.5 };
        constexpr UncertainValue<double> b{ 3.0, 0.5 };

        expect(12.0_d == (a * b).value) << "value: 4. * 3.";
        expect(2.5_d == (a * b).uncertainty) << "uncertainty: sqrt((4*0.5)^2 + (3*0.5)^2) = sqrt(4 + 2.25) = sqrt(6.25) = 2.5";
    };

    "multiplication UncertainValue * arithmetic value"_test = [] {
        constexpr UncertainValue<double> a{ 4.0, 0.5 };
        constexpr double                 b{ 3.0 };

        expect(12.0_d == (a * b).value) << "value: 4. * 3.";
        expect(1.5_d == (a * b).uncertainty) << "uncertainty: 0.5 * 3 = 1.5";
    };

    "multiplication arithmetic value * UncertainValue"_test = [] {
        constexpr double                 a{ 4.0 };
        constexpr UncertainValue<double> b{ 3.0, 0.5 };

        expect(constant<12.0_d == (a * b).value>) << "value: 4. * 3.";
        expect(constant<2.0_d == (a * b).uncertainty>) << "uncertainty: 0.5 * 4 = 1.5";
    };

    "multiplication arithmetic value * arithmetic value"_test = [] {
        constexpr double a{ 4.0 };
        constexpr double b{ 3.0 };

        constexpr UncertainValue<double> result = a * b;
        expect(constant<12.0_d == result.value>) << "value: 4. * 3.";
        expect(constant<0.0_d == result.uncertainty>) << "uncertainty: 0.0 (no change)";
    };

    "operator *= UncertainValue"_test = [] {
        constexpr UncertainValue<double> a0{ 4.0, 0.5 };
        UncertainValue<double>           a{ 4.0, 0.5 };
        UncertainValue<double>           b{ 3.0, 0.2 };
        a *= b;
        expect(eq((a0 * b).value, a.value));
        expect(eq((a0 * b).uncertainty, a.uncertainty)); // Assuming appropriate uncertainty calculation
    };

    "division UncertainValue / UncertainValue"_test = [] {
        constexpr UncertainValue<double> a{ 16.0, 2.0 };
        constexpr UncertainValue<double> b{ 4.0, 0.5 };

        expect(4.0_d == (a / b).value) << "value: 16. / 4.";
        expect(0.707_d == (a / b).uncertainty) << "uncertainty: sqrt((2/4)^2 + (16*0.5/4^2)^2) = sqrt(0.25 + 0.25) = sqrt(0.5) ≈ 0.707";
    };

    "division UncertainValue / arithmetic value"_test = [] {
        constexpr UncertainValue<double> a{ 16.0, 2.0 };
        constexpr const double           b{ 4.0 };

        expect(4.0_d == (a / b).value) << "value: 16. / 4.";
        expect(0.5_d == (a / b).uncertainty) << "uncertainty: 2.0 / 4 = 0.5";
    };

    "division arithmetic value / UncertainValue"_test = [] {
        constexpr double                 a{ 16.0 };
        constexpr UncertainValue<double> b{ 4.0, 0.5 };

        expect(4.0_d == (a / b).value) << "value: 16. / 4.";
        expect(0.5_d == (a / b).uncertainty);
    };

    "division arithmetic value / arithmetic value"_test = [] {
        constexpr double a{ 16.0 };
        constexpr double b{ 4.0 };

        constexpr UncertainValue<double> result = a / b;
        expect(constant<4.0_d == result.value>) << "value: 16. / 4.";
        expect(constant<0.0_d == result.uncertainty>) << "uncertainty: 0.0 (no change)";
    };

    "operator /= UncertainValue"_test = [] {
        constexpr UncertainValue<double> a0{ 4.0, 0.5 };
        UncertainValue<double>           a{ 4.0, 0.5 };
        UncertainValue<double>           b{ 3.0, 0.2 };
        a /= b;
        expect(eq((a0 / b).value, a.value));
        expect(eq((a0 / b).uncertainty, a.uncertainty)); // Assuming appropriate uncertainty calculation
    };

    // Complex Addition Tests
    using Complex                                                     = std::complex<double>;
    "addition UncertainValue<Complex> + UncertainValue<Complex>"_test = [] {
        UncertainValue<Complex> a{ 10.0 + 5.0i, 3.0 + 4.0i };
        UncertainValue<Complex> b{ 20.0 + 10.0i, 4.0 + 3.0i };

        expect(approx(30.0 + 15.0i, (a + b).value, 1e-9)) << "value: (10 + 5i) + (20 + 10i)";
        expect(approx(5.0 + 5.0i, (a + b).uncertainty, 1e-9)) << "uncertainty: sqrt((3^2 + 4^2), (1^2 + 2^2))";
    };

    "addition UncertainValue<Complex> + Complex"_test = [] {
        UncertainValue<Complex> a{ 10.0 + 5.0i, 3.0 + 4.0i };
        Complex                 b{ 20.0 + 10.0i };

        expect(approx(30.0 + 15.0i, (a + b).value, 1e-9)) << "value: (10 + 5i) + (20 + 10i)";
        expect(approx(3.0 + 4.0i, (a + b).uncertainty, 1e-9)) << "uncertainty: 3 + 4i (no change)";
    };

    "addition Complex + UncertainValue<Complex>"_test = [] {
        Complex                 a{ 10.0 + 5.0i };
        UncertainValue<Complex> b{ 20.0 + 10.0i, 4.0 + 2.0i };

        expect(approx(30.0 + 15.0i, (a + b).value, 1e-9)) << "value: (10 + 5i) + (20 + 10i)";
        expect(approx(4.0 + 2.0i, (a + b).uncertainty, 1e-9)) << "uncertainty: 4 + 2i (no change)";
    };

    "addition UncertainValue<Complex> + arithmetic value"_test = [] {
        UncertainValue<Complex> a{ 10.0 + 5.0i, 3.0 + 4.0i };
        UncertainValue<double>  b{ 20.0, 4.0 };

        expect(approx(30.0 + 5.0i, (a + b).value, 1e-9)) << "value: (10 + 5i) + (20 + 10i)";
        expect(approx(5.0 + 4.0i, (a + b).uncertainty, 1e-9)) << "uncertainty: 3 + 4i (no change)";
    };

    "addition UncertainValue<Complex> + UncertainValue<arithmetic>"_test = [] {
        UncertainValue<Complex> a{ 10.0 + 5.0i, 3.0 + 4.0i };
        double                  b{ 20.0 };

        expect(approx(30.0 + 5.0i, (a + b).value, 1e-9)) << "value: (10 + 5i) + (20 + 10i)";
        expect(approx(3.0 + 4.0i, (a + b).uncertainty, 1e-9)) << "uncertainty: 3 + 4i (no change)";
    };

    "addition UncertainValue<arithmetic> + UncertainValue<Complex>"_test = [] {
        UncertainValue<double>  a{ 10.0, 3.0 };
        UncertainValue<Complex> b{ 20.0 + 10.0i, 4.0 + 2.0i };

        expect(approx(30.0 + 10.0i, (a + b).value, 1e-9)) << "value: (10 + 5i) + (20 + 10i)";
        expect(approx(5.0 + 2.0i, (a + b).uncertainty, 1e-9)) << "uncertainty: sqrt((3^2 + 4^2), (1^2 + 2^2))";
    };

    "addition arithmetic value + UncertainValue<Complex>"_test = [] {
        double                  a{ 10.0 };
        UncertainValue<Complex> b{ 20.0 + 10.0i, 4.0 + 2.0i };

        expect(approx(30.0 + 10.0i, (a + b).value, 1e-9)) << "value: (10 + 5i) + (20 + 10i)";
        expect(approx(4.0 + 2.0i, (a + b).uncertainty, 1e-9)) << "uncertainty: 4 + 2i (no change)";
    };

    // Complex Subtraction Tests
    "subtraction UncertainValue<Complex> - UncertainValue<Complex>"_test = [] {
        UncertainValue<Complex> a{ 20.0 + 10.0i, 3.0 + 4.0i };
        UncertainValue<Complex> b{ 10.0 + 5.0i, 4.0 + 3.0i };

        expect(approx(10.0 + 5.0i, (a - b).value, 1e-9)) << "value: (20 + 10i) - (10 + 5i)";
        expect(approx(5.0 + 5.0i, (a - b).uncertainty, 1e-9)) << "uncertainty: sqrt((3^2 + 4^2), (1^2 + 2^2))";
    };

    "subtraction UncertainValue<Complex> - Complex"_test = [] {
        UncertainValue<Complex> a{ 20.0 + 10.0i, 3.0 + 1.0i };
        Complex                 b{ 10.0 + 5.0i };

        expect(approx(10.0 + 5.0i, (a - b).value, 1e-9)) << "value: (20 + 10i) - (10 + 5i)";
        expect(approx(3.0 + 1.0i, (a - b).uncertainty, 1e-9)) << "uncertainty: 3 + 1i (no change)";
    };

    "subtraction Complex - UncertainValue<Complex>"_test = [] {
        Complex                 a{ 20.0 + 10.0i };
        UncertainValue<Complex> b{ 10.0 + 5.0i, 4.0 + 2.0i };

        expect(approx(10.0 + 5.0i, (a - b).value, 1e-9)) << "value: (20 + 10i) - (10 + 5i)";
        expect(approx(4.0 + 2.0i, (a - b).uncertainty, 1e-9)) << "uncertainty: 4 + 2i (no change)";
    };

    "subtraction UncertainValue<Complex> - arithmetic value"_test = [] {
        UncertainValue<Complex> a{ 20.0 + 10.0i, 3.0 + 4.0i };
        double                  b{ 10.0 };

        expect(approx(10.0 + 10.0i, (a - b).value, 1e-9)) << "value: (20 + 10i) - 10";
        expect(approx(3.0 + 4.0i, (a - b).uncertainty, 1e-9)) << "uncertainty: 3 + 4i (no change)";
    };

    "subtraction UncertainValue<Complex> - UncertainValue<arithmetic>"_test = [] {
        UncertainValue<Complex> a{ 20.0 + 10.0i, 3.0 + 4.0i };
        UncertainValue<double>  b{ 10.0, 4.0 };

        expect(approx(10.0 + 10.0i, (a - b).value, 1e-9)) << "value: (20 + 10i) - 10";
        expect(approx(5.0 + 4.0i, (a - b).uncertainty, 1e-9)) << "uncertainty: sqrt((3^2 + 4^2), (4^2))";
    };

    "subtraction arithmetic value - UncertainValue<Complex>"_test = [] {
        double                  a{ 20.0 };
        UncertainValue<Complex> b{ 10.0 + 5.0i, 4.0 + 3.0i };

        expect(approx(10.0 - 5.0i, (a - b).value, 1e-9)) << "value: 20 - (10 + 5i)";
        expect(approx(4.0 + 3.0i, (a - b).uncertainty, 1e-9)) << "uncertainty: 4 + 3i (no change)";
    };

    "subtraction UncertainValue<arithmetic> - UncertainValue<Complex>"_test = [] {
        UncertainValue<double>  a{ 20.0, 3.0 };
        UncertainValue<Complex> b{ 10.0 + 5.0i, 4.0 + 2.0i };

        expect(approx(10.0 - 5.0i, (a - b).value, 1e-9)) << "value: 20 - (10 + 5i)";
        expect(approx(5.0 + 2.0i, (a - b).uncertainty, 1e-9)) << "uncertainty: sqrt((3^2), (4^2 + 2^2))";
    };

    // Complex Multiplication Tests
    "multiplication UncertainValue<Complex> * UncertainValue<Complex>"_test = [] {
        UncertainValue<Complex> a{ 2.0 + 3.0i, .75 + 4.0i };
        UncertainValue<Complex> b{ 4.0 + 5.0i, 2.0 + 3.0i };

        expect(approx(-7.0 + 22.0i, (a * b).value, 1e-9));
        expect(approx(5.0 + 21.9317i, (a * b).uncertainty, 1e-3));
    };

    "multiplication UncertainValue<Complex> * Complex"_test = [] {
        UncertainValue<Complex> a{ 2.0 + 3.0i, 0.2 + 0.3i };
        Complex                 b{ 4.0 + 5.0i };

        expect(approx(-7.0 + 22.0i, (a * b).value, 1e-9)) << "value: (2 + 3i) * (4 + 5i)";
        expect(approx(-0.7 + 2.2i, (a * b).uncertainty, 1e-9));
    };

    "multiplication Complex * UncertainValue<Complex>"_test = [] {
        Complex                 a{ 2.0 + 3.0i };
        UncertainValue<Complex> b{ 4.0 + 5.0i, 0.4 + 0.5i };

        expect(approx(-7.0 + 22.0i, (a * b).value, 1e-9)) << "value: (2 + 3i) * (4 + 5i)";
        expect(approx(-0.7 + 2.2i, (a * b).uncertainty, 1e-9));
    };

    "multiplication UncertainValue<Complex> * arithmetic value"_test = [] {
        UncertainValue<Complex> a{ 2.0 + 3.0i, 0.2 + 0.3i };
        double                  b{ 4.0 };

        expect(approx(8.0 + 12.0i, (a * b).value, 1e-9)) << "value: (2 + 3i) * 4";
        expect(approx(0.8 + 1.2i, (a * b).uncertainty, 1e-9));
    };

    "multiplication UncertainValue<Complex> * UncertainValue<arithmetic>"_test = [] {
        UncertainValue<Complex> a{ 2.0 + 3.0i, 0.2 + 0.3i };
        UncertainValue<double>  b{ 4.0, 0.4 };

        expect(approx(8.0 + 12.0i, (a * b).value, 1e-9)) << "value: (2 + 3i) * 4";
        expect(approx(1.13137 + 0.0i, (a * b).uncertainty, 1e-3));
    };

    "multiplication arithmetic value * UncertainValue<Complex>"_test = [] {
        double                  a{ 2.0 };
        UncertainValue<Complex> b{ 4.0 + 5.0i, 0.4 + 0.5i };

        expect(approx(8.0 + 10.0i, (a * b).value, 1e-9)) << "value: 2 * (4 + 5i)";
        expect(approx(0.8 + 1.0i, (a * b).uncertainty, 1e-9)) << "uncertainty: 2 * (0.4 + 0.5i)";
    };

    "multiplication UncertainValue<arithmetic> * UncertainValue<Complex>"_test = [] {
        UncertainValue<double>  a{ 2.0, 0.2 };
        UncertainValue<Complex> b{ 4.0 + 5.0i, 0.4 + 0.5i };

        expect(approx(8.0 + 10.0i, (a * b).value, 1e-9)) << "value: 2 * (4 + 5i)";
        expect(approx(1.13137 + 0i, (a * b).uncertainty, 1e-3));
    };

    // Complex Division Tests
    "division UncertainValue<Complex> / UncertainValue<Complex>"_test = [] {
        UncertainValue<Complex> a{ 8.0 + 6.0i, 0.8 + 0.6i };
        UncertainValue<Complex> b{ 3.0 + 4.0i, 0.3 + 0.4i };

        expect(approx((8.0 + 6.0i) / (3.0 + 4.0i), (a / b).value, 1e-9)) << "value: (8 + 6i) / (3 + 4i)";
        expect(approx(0.28342 + 0.17809i, (a / b).uncertainty, 1e-3));
    };

    "division UncertainValue<Complex> / Complex"_test = [] {
        UncertainValue<Complex> a{ 2.0 + 3.0i, 0.2 + 0.3i };
        Complex                 b{ 4.0 + 5.0i };

        expect(approx((2.0 + 3.0i) / (4.0 + 5.0i), (a / b).value, 1e-9)) << "value: (2 + 3i) / (4 + 5i)";
        expect(approx(0.0312348 + 0.0468521i, (a / b).uncertainty, 1e-3));
    };

    "division Complex / UncertainValue<Complex>"_test = [] {
        Complex                 a{ 2.0 + 3.0i };
        UncertainValue<Complex> b{ 4.0 + 5.0i, 0.4 + 0.5i };

        expect(approx((2.0 + 3.0i) / (4.0 + 5.0i), (a / b).value, 1e-9)) << "value: (2 + 3i) / (4 + 5i)";
        expect(approx(0.0351761 + 0.0439701i, (a / b).uncertainty, 1e-3));
    };

    "division UncertainValue<Complex> / arithmetic value"_test = [] {
        UncertainValue<Complex> a{ 2.0 + 3.0i, 0.2 + 0.3i };
        double                  b{ 4.0 };

        expect(approx((2.0 + 3.0i) / 4.0, (a / b).value, 1e-9)) << "value: (2 + 3i) / 4";
        expect(approx((0.2 + 0.3i) / 4.0, (a / b).uncertainty, 1e-9)) << "uncertainty: (0.2 + 0.3i) / 4";
    };

    "division UncertainValue<Complex> / UncertainValue<arithmetic>"_test = [] {
        UncertainValue<Complex> a{ 2.0 + 3.0i, 0.2 + 0.3i };
        UncertainValue<double>  b{ 4.0, 0.4 };

        expect(approx((2.0 + 3.0i) / 4.0, (a / b).value, 1e-9)) << "value: (2 + 3i) / 4";
        expect(approx(0.0707107 + 0.075i, (a / b).uncertainty, 1e-3));
    };

    "division arithmetic value / UncertainValue<Complex>"_test = [] {
        double                  a{ 2.0 };
        UncertainValue<Complex> b{ 4.0 + 5.0i, 0.4 + 0.5i };

        expect(approx(2.0 / (4.0 + 5.0i), (a / b).value, 1e-9)) << "value: 2 / (4 + 5i)";
        expect(approx(0.0195122 + 0.0243902i, (a / b).uncertainty, 1e-3));
    };

    "division UncertainValue<arithmetic> / UncertainValue<Complex>"_test = [] {
        UncertainValue<double>  a{ 2.0, 0.2 };
        UncertainValue<Complex> b{ 4.0 + 5.0i, 0.4 + 0.5i };

        expect(approx(2.0 / (4.0 + 5.0i), (a / b).value, 1e-9)) << "value: 2 / (4 + 5i)";
        expect(approx(0.05 + 0.0243902i, (a / b).uncertainty, 1e-3));
    };

    // std::pow(…, …) overloads

    "std::pow(UncertainValue<arithmetic>, arithmetic)"_test = [] {
        using Un = UncertainValue<double>;
        expect(eq(4.0, (std::pow(Un{ 2.0, 1.0 }, 2.0)).value));
        expect(eq(4.0, (std::pow(Un{ 2.0, 1.0 }, 2.0)).uncertainty));

        expect(eq(0.0, (std::pow(Un{ 0.0, 1.0 }, 2.0)).value));
        expect(eq(0.0, (std::pow(Un{ 0.0, 1.0 }, 2.0)).uncertainty));

        expect(eq(1.0, (std::pow(Un{ 3.0, 1.0 }, 0.0)).value));
        expect(eq(0.0, (std::pow(Un{ 3.0, 1.0 }, 0.0)).uncertainty));

        expect(eq(1.0, (std::pow(Un{ 0.0, 1.0 }, 0.0)).value));
        expect(eq(0.0, (std::pow(Un{ 0.0, 1.0 }, 0.0)).uncertainty));
    };

    "std::pow(UncertainValue<arithmetic>, UncertainValue<arithmetic>)"_test = [] {
        using Un = UncertainValue<double>;
        using std::numbers::e;

        expect(eq(e * e, (std::pow(Un{ e, e * 2.0 }, Un{ 2.0, 3.0 })).value));
        expect(approx(e * e * 5.0, (std::pow(Un{ e, e * 2.0 }, Un{ 2.0, 3.0 })).uncertainty, 1e-3));

        // reproduce previous special case of uncertain value + arithmetic, here via setting uncertainty of exponent to 0
        expect(eq(4.0, (std::pow(Un{ 2.0, 1.0 }, Un{ 2.0, 0.0 })).value));
        expect(eq(4.0, (std::pow(Un{ 2.0, 1.0 }, Un{ 2.0, 0.0 })).uncertainty));

        expect(eq(0.0, (std::pow(Un{ 0.0, 1.0 }, Un{ 2.0, 0.0 })).value));
        expect(eq(0.0, (std::pow(Un{ 0.0, 1.0 }, Un{ 2.0, 0.0 })).uncertainty));

        expect(eq(1.0, (std::pow(Un{ 3.0, 1.0 }, Un{ 0.0, 0.0 })).value));
        expect(eq(0.0, (std::pow(Un{ 3.0, 1.0 }, Un{ 0.0, 0.0 })).uncertainty));

        expect(eq(1.0, (std::pow(Un{ 0.0, 1.0 }, Un{ 0.0, 0.0 })).value));
        expect(eq(0.0, (std::pow(Un{ 0.0, 1.0 }, Un{ 0.0, 0.0 })).uncertainty));

        expect(eq(0.0, (std::pow(Un{ 0.0, 0.1 }, Un{ 2.0, 0.1 })).value));
        expect(eq(0.0, (std::pow(Un{ 0.0, 0.1 }, Un{ 2.0, 0.1 })).uncertainty));
    };

    "std::pow(UncertainValue<std::complex<double>, arithmetic)"_test = [] {
        using ComplexUncertain = UncertainValue<std::complex<double>>;

        {
            ComplexUncertain base{ { std::sqrt(2.0), std::sqrt(2.0) }, 0.1 + 0.1i };
            double           exponent = 2.0;

            auto result = std::pow(base, exponent);
            expect(approx(std::pow(base.value, exponent), result.value, 1e-9));
            expect(approx(std::complex<double>{ 0.4, 0.4 }, result.uncertainty, 1e-6));
        }

        {
            ComplexUncertain base{ 0.0 + 0.0i, 0.1 + 0.1i };
            double           exponent = 2.0;

            auto result = std::pow(base, exponent);
            expect(approx(0.0 + 0.0i, result.value, 1e-9));
            expect(approx(0.0 + 0.0i, result.uncertainty, 1e-9));
        }

        {
            ComplexUncertain base{ 1.0 + 2.0i, 0.1 + 0.1i };
            double           exponent = 0.0;

            auto result = std::pow(base, exponent);
            expect(approx(1.0 + 0.0i, result.value, 1e-9));
            expect(approx(0.0 + 0.0i, result.uncertainty, 1e-9));
        }

        {
            ComplexUncertain base{ 0.0, 0.1 + 0.1i };
            double           exponent = 0.0;

            auto result = std::pow(base, exponent);
            expect(approx(1.0 + 0.0i, result.value, 1e-9));
            expect(approx(0.0 + 0.0i, result.uncertainty, 1e-9));
        }
    };

    "std::pow(UncertainValue<std::complex<double>, UncertainValue<arithmetic>)"_test = [] {
        using ComplexUncertain = UncertainValue<std::complex<double>>;
        using Un               = UncertainValue<double>;

        {
            ComplexUncertain base{ { std::sqrt(2.0), std::sqrt(2.0) }, 0.1 + 0.1i };
            Un               exponent = { 2.0, 0.0 };

            auto result = std::pow(base, exponent);
            expect(approx(std::pow(base.value, exponent.value), result.value, 1e-9));
            expect(approx(std::complex<double>{ 0.565685, 0 }, result.uncertainty, 1e-6)); // calculated with Wolfram
        }

        {
            ComplexUncertain base{ { std::sqrt(2.0), std::sqrt(2.0) }, 0.1 + 0.1i };
            ComplexUncertain exponent = { { 2.0 }, { 0.0 } };

            auto result = std::pow(base, exponent);
            expect(approx(std::pow(base.value, exponent.value), result.value, 1e-9));
            expect(approx(std::complex<double>{ 0.565685, 0.0 }, result.uncertainty, 1e-6)); // calculated with Wolfram
        }

        {
            ComplexUncertain base{ { std::sqrt(2.0), std::sqrt(2.0) }, 0.1 + 0.1i };
            Un               exponent = { 2.0, 0.1 };

            auto result = std::pow(base, exponent);
            expect(approx(std::pow(base.value, exponent.value), result.value, 1e-9));
            expect(approx(std::complex<double>{ 0.703966, 0.0 }, result.uncertainty, 1e-3)); // calculated with Wolfram
        }

        {
            ComplexUncertain base{ { std::sqrt(2.0), std::sqrt(2.0) }, 0.1 + 0.1i };
            ComplexUncertain exponent{ { 2.0 }, 0.1 + 0.1i };

            auto result = std::pow(base, exponent);
            expect(approx(std::pow(base.value, exponent.value), result.value, 1e-9));
            expect(approx(std::complex<double>{ 0.81923, 0.0 }, result.uncertainty, 1e-3)); // calculated with Wolfram
        }

        {
            ComplexUncertain base{ { 0.0, 0.0 }, { 0.1, 0.1 } };
            Un               exponent = { 2.0, 0.0 };

            auto result = std::pow(base, exponent);
            expect(approx(std::complex<double>{ 0.0, 0.0 }, result.value, 1e-9));
            expect(approx(std::complex<double>{ 0.0, 0.0 }, result.uncertainty, 1e-9));
        }

        {
            ComplexUncertain base{ { 1.0, 2.0 }, { 0.1, 0.1 } };
            Un               exponent = { 0.0, 0.0 };

            auto result = std::pow(base, exponent);
            expect(approx(std::complex<double>{ 1.0, 0.0 }, result.value, 1e-9));
            expect(approx(std::complex<double>{ 0.0, 0.0 }, result.uncertainty, 1e-9));
        }

        {
            ComplexUncertain base{ { 0.0 }, { 0.1, 0.1 } };
            Un               exponent = { 0.0, 0.0 };

            auto result = std::pow(base, exponent);
            expect(approx(std::complex<double>{ 1.0, 0.0 }, result.value, 1e-9));
            expect(approx(std::complex<double>{ 0.0, 0.0 }, result.uncertainty, 1e-9));
        }
    };

    "std::sqrt(UncertainValue<double>)"_test = [] {
        using UncertainDouble = UncertainValue<double>;

        {
            UncertainDouble value{ 4.0, 0.2 }; // sqrt(4) = 2, uncertainty should be 0.2 / (2*2) = 0.05
            auto            result = std::sqrt(value);
            expect(approx(2.0, result.value, 1e-9));
            expect(approx(0.05, result.uncertainty, 1e-9));
        }

        {
            UncertainDouble value{ 0.0, 0.1 }; // sqrt(0) = 0, uncertainty should be 0
            auto            result = std::sqrt(value);
            expect(approx(0.0, result.value, 1e-9));
            expect(approx(0.0, result.uncertainty, 1e-9));
        }
    };

    "std::sqrt(UncertainValue<std::complex<double>>)"_test = [] {
        using ComplexUncertain = UncertainValue<std::complex<double>>;

        {
            ComplexUncertain value{ { 4.0, 4.0 }, { 0.2, 0.2 } }; // sqrt(4+4i)
            auto             result = std::sqrt(value);
            expect(approx(std::sqrt(value.value), result.value, 1e-9));
            expect(approx(std::complex<double>{ 0.0420448, 0.0420448 }, result.uncertainty, 1e-3));
        }

        {
            ComplexUncertain value{ { 0.0, 0.0 }, { 0.1, 0.1 } }; // sqrt(0+0i)
            auto             result = std::sqrt(value);
            expect(approx(std::complex<double>{ 0.0, 0.0 }, result.value, 1e-9));
            expect(approx(std::complex<double>{ 0.0, 0.0 }, result.uncertainty, 1e-9));
        }
    };

    "basic box-car low-pass filter examples"_test = [] {
        UncertainValue value{ 4.0, 1.0 };

        UncertainValue sum1{ 0.0, 0.0 };
        UncertainValue sum2{ 0.0, 0.0 };
        for (std::size_t i = 0UZ; i < 16UZ; i++) {
            sum1 += value / 16.0;
            sum2 = sum2 + value / 16.0;
        }
        expect(eq(4.0, sum1.value));
        expect(eq(4.0, sum2.value));
        expect(approx(value.uncertainty / sqrt(16), sum1.uncertainty, 1e-9));
        expect(approx(value.uncertainty / sqrt(16), sum2.uncertainty, 1e-9));
    };

    tag("visual") / "visual examples"_test = [] {
        // uncorrelated values
        UncertainValue uValueA{ 4.0, 2.0 };
        UncertainValue uValueB{ 2.0, 1.0 };
        static_assert(UncertainValueLike<decltype(uValueA)>);
        static_assert(!UncertainValueLike<double>);

        // uncorrelated operations
        fmt::print("uncorrelated values:\n");
        fmt::print("{} + {} = {}\n", uValueA, uValueB, uValueA + uValueB);
        fmt::print("{} - {} = {}\n", uValueA, uValueB, uValueA - uValueB);
        fmt::print("{} * {} = {}\n", uValueA, uValueB, uValueA * uValueB);
        fmt::print("{} / {} = {}\n", uValueA, uValueB, uValueA / uValueB);

        fmt::print("mixed-regular values:\n");
        fmt::print("{} + {} = {}\n", uValueA, uValueB.value, uValueA + uValueB.value);
        fmt::print("{} - {} = {}\n", uValueA, uValueB.value, uValueA - uValueB.value);
        fmt::print("{} * {} = {}\n", uValueA, uValueB.value, uValueA * uValueB.value);
        fmt::print("{} / {} = {}\n", uValueA, uValueB.value, uValueA / uValueB.value);

        // complex values
        using namespace std::complex_literals;
        UncertainValue uValueAC{ 4. + 1i, +1. + 1i };
        UncertainValue uValueBC{ 2. - 1i, +2. + 2i };

        fmt::print("uncorrelated values - complex:\n");
        fmt::print("{} + {} = {}\n", uValueAC, uValueBC, uValueAC + uValueBC);
        fmt::print("{} - {} = {}\n", uValueAC, uValueBC, uValueAC - uValueBC);
        fmt::print("{} * {} = {}\n", uValueAC, uValueBC, uValueAC * uValueBC);
        fmt::print("{} / {} = {}\n", uValueAC, uValueBC, uValueAC / uValueBC);
    };
};

const boost::ut::suite uncertainValueTrigonometric = [] {
    using namespace boost::ut;
    using namespace std::literals::complex_literals;
    using namespace gr;
    using test::detail::approx;
    using std::numbers::pi;

    "sin(UncertainValue<…>)"_test = [] {
        {
            UncertainValue<double> x{ pi / 6, 0.01 }; // 30 degrees
            auto                   result = std::sin(x);
            expect(approx(0.5, result.value, 1e-6)) << "sin(π/6)";
            expect(approx(0.01 * std::cos(pi / 6), result.uncertainty, 1e-6)) << "Uncertainty in sin(π/6)";
        }

        {
            UncertainValue<double> x{ pi / 3, 0.1 }; // 60 degrees
            auto                   result = std::sin(x);
            expect(approx(std::sqrt(3) / 2, result.value, 1e-6)) << "sin(π/3)";
            expect(approx(0.1 * std::cos(pi / 3), result.uncertainty, 1e-6)) << "Uncertainty in sin(π/3)";
        }

        {
            UncertainValue<std::complex<double>> x{ 1.0 + 2.0i, 0.1 + 0.2i };
            auto                                 result = std::sin(x);

            expect(approx(std::sin(x.value), result.value, 1e-6));
            expect(approx(std::complex<double>{ 0.81994, 0 }, result.uncertainty, 1e-6)); // TODO check
        }
    };

    "cos(UncertainValue<…>)"_test = [] {
        {
            UncertainValue<double> x{ pi / 3, 0.01 }; // 60 degrees
            auto                   result = std::cos(x);
            expect(approx(0.5, result.value, 1e-6)) << "cos(π/3)";
            expect(approx(0.01 * std::sin(pi / 3), result.uncertainty, 1e-6)) << "Uncertainty in cos(π/3)";
        }

        {
            UncertainValue<double> x{ pi / 6, 0.1 }; // 30 degrees
            auto                   result = std::cos(x);
            expect(approx(std::sqrt(3) / 2, result.value, 1e-6)) << "cos(π/6)";
            expect(approx(0.1 * std::sin(pi / 6), result.uncertainty, 1e-6)) << "Uncertainty in cos(π/6)";
        }

        {
            UncertainValue<std::complex<double>> x{ 1.0 + 1.0i, 0.1 + 0.1i };
            auto                                 result = std::cos(x);

            expect(approx(std::cos(x.value), result.value, 1e-6));
            expect(approx(std::complex<double>{ 0.20441, 0 }, result.uncertainty, 1e-6)); // TODO check
        }
    };
};

const boost::ut::suite uncertainValueExpTests = [] {
    using namespace boost::ut;
    using namespace std::literals::complex_literals;
    using namespace gr;
    using test::detail::approx;
    using std::numbers::pi;

    "exp(UncertainValue<arithmetic>)"_test = [] {
        UncertainValue<double> x{ 2.0, 0.1 }; // x = 2.0 ± 0.1
        auto                   result              = std::exp(x);
        auto                   expectedValue       = std::exp(x.value);
        auto                   expectedUncertainty = std::exp(x.value) * x.uncertainty; // d(exp(x))/dx = exp(x)

        expect(approx(expectedValue, result.value, 1e-6)) << "Value of exp(2.0)";
        expect(approx(expectedUncertainty, result.uncertainty, 1e-6)) << "Uncertainty in exp(2.0)";
    };

    "exp(UncertainValue<complex>)"_test = [] {
        {
            UncertainValue<std::complex<double>> x{ { 1.0, 2.0 }, { 0.1, 0.1 } }; // x = (1.0 + 2.0i) ± (0.1 + 0.1i)
            auto                                 result = std::exp(x);
            //            auto                                 expectedUncertainty = std::complex<double>{
            //                std::hypot(std::exp(x.value.real()) * std::cos(x.value.imag()) * x.uncertainty.real(), std::exp(x.value.real()) * std::sin(x.value.imag()) * x.uncertainty.imag()),
            //                std::hypot(std::exp(x.value.real()) * std::sin(x.value.imag()) * x.uncertainty.real(), std::exp(x.value.real()) * std::cos(x.value.imag()) * x.uncertainty.imag())
            //            };

            expect(approx(std::exp(x.value), result.value, 1e-6));
            // expect(approx(expectedUncertainty, result.uncertainty, 1e-6)); // TODO: check if uncertainties are propagated as absolute (real-valued only) or retain their complex nature)
        }
        {
            using std::numbers::pi;
            UncertainValue<std::complex<double>> x{ { 0.0, 0 * pi / 2 }, { 0.1, 0.1 } }; // x = i * pi / 2 ± (0.1 + 0.1i)

            expect(approx(std::exp(x.value), std::exp(x).value, 1e-6));
            expect(approx(std::sin(UncertainValue<std::complex<double>>{ { pi / 2 }, { 0.1, 0.1 } }).value, std::exp(x).value, 1e-6));
            // expect(approx(std::sin(UncertainValue<std::complex<double>>{ { pi / 2 }, { 0.1, 0.1 } }).uncertainty, std::exp(x).uncertainty, 1e-6)); // TODO: check if uncertainties are propagated as
            // absolute (real-valued only) or retain their complex nature)
        }
    };
};

int
main() { /* tests are statically executed */
}