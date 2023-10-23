#include <boost/ut.hpp>

#include <complex>

#include <fmt/format.h>

#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::meta::test {

const boost::ut::suite complexFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::complex_literals;
    using C                = std::complex<double>;
    "fmt::formatter<std::complex<T>>"_test = [] {
        fmt::print("{}\n", C(1.234, 1123456789012));
        expect("(1+1i)" == fmt::format("{}", C(1., +1.)));
        expect("(1-1i)" == fmt::format("{}", C(1., -1.)));
        expect("1" == fmt::format("{}", C(1., 0.)));
        expect("(1.234+1.12346e+12i)" == fmt::format("{}", C(1.234, 1123456789012)));
        expect("(1+1i)" == fmt::format("{:g}", C(1., +1.)));
        expect("(1-1i)" == fmt::format("{:g}", C(1., -1.)));
        expect("1" == fmt::format("{:g}", C(1., 0.)));
        expect("(1.12346e+12+1.234i)" == fmt::format("{:g}", C(1123456789012, 1.234)));
        expect("1.12346e+12" == fmt::format("{:g}", C(1123456789012, 0)));
        expect("(1.234+1.12346e+12i)" == fmt::format("{:g}", C(1.234, 1123456789012)));
        expect("(1.12346E+12+1.234i)" == fmt::format("{:G}", C(1123456789012, 1.234)));
        expect("1.12346E+12" == fmt::format("{:G}", C(1123456789012, 0)));
        expect("(1.234+1.12346E+12i)" == fmt::format("{:G}", C(1.234, 1123456789012)));

        expect("(1.000000+1.000000i)" == fmt::format("{:f}", C(1., +1.)));
        expect("(1.000000-1.000000i)" == fmt::format("{:f}", C(1., -1.)));
        expect("1.000000" == fmt::format("{:f}", C(1., 0.)));
        expect("(1.000000+1.000000i)" == fmt::format("{:F}", C(1., +1.)));
        expect("(1.000000-1.000000i)" == fmt::format("{:F}", C(1., -1.)));
        expect("1.000000" == fmt::format("{:F}", C(1., 0.)));

        expect("(1.000000e+00+1.000000e+00i)" == fmt::format("{:e}", C(1., +1.)));
        expect("(1.000000e+00-1.000000e+00i)" == fmt::format("{:e}", C(1., -1.)));
        expect("1.000000e+00" == fmt::format("{:e}", C(1., 0.)));
        expect("(1.000000E+00+1.000000E+00i)" == fmt::format("{:E}", C(1., +1.)));
        expect("(1.000000E+00-1.000000E+00i)" == fmt::format("{:E}", C(1., -1.)));
        expect("1.000000E+00" == fmt::format("{:E}", C(1., 0.)));
    };
};

}

int
main() { /* tests are statically executed */
}