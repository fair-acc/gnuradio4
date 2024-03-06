#include <boost/ut.hpp>

#include <complex>
#include <expected>

#include <fmt/format.h>

#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::meta::test {

const boost::ut::suite complexFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::complex_literals;
    using C                                = std::complex<double>;
    "fmt::formatter<std::complex<T>>"_test = [] {
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

const boost::ut::suite uncertainValueFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::complex_literals;
    using UncertainDouble  = gr::UncertainValue<double>;
    using UncertainComplex = gr::UncertainValue<std::complex<double>>;

    "fmt::formatter<gr::meta::UncertainValue<T>>"_test = [] {
        // Test with UncertainValue<double>
        expect("(1.23 ± 0.45)" == fmt::format("{}", UncertainDouble{ 1.23, 0.45 }));
        expect("(3.14 ± 0.01)" == fmt::format("{}", UncertainDouble{ 3.14, 0.01 }));
        expect("(0 ± 0)" == fmt::format("{}", UncertainDouble{ 0, 0 }));

        // Test with UncertainValue<std::complex<double>>
        expect("((1+2i) ± (0.1+0.2i))" == fmt::format("{}", UncertainComplex{ { 1, 2 }, { 0.1, 0.2 } }));
        expect("((3.14+1.59i) ± (0.01+0.02i))" == fmt::format("{}", UncertainComplex{ { 3.14, 1.59 }, { 0.01, 0.02 } }));
        expect("(0 ± 0)" == fmt::format("{}", UncertainComplex{ { 0, 0 }, { 0, 0 } }));
    };
};

const boost::ut::suite propertyMapFormatter = [] {
    using namespace boost::ut;

    "fmt::formatter<gr::property_map>"_test = [] {
        gr::property_map pmInt{ { "key0", 0 }, { "key1", 1 }, { "key2", 2 } };
        expect("{ key0: 0, key1: 1, key2: 2 }" == fmt::format("{}", pmInt));

        gr::property_map pmFloat{ { "key0", 0.01f }, { "key1", 1.01f }, { "key2", 2.01f } };
        expect("{ key0: 0.01, key1: 1.01, key2: 2.01 }" == fmt::format("{}", pmFloat));
    };
};

const boost::ut::suite vectorBoolFormatter = [] {
    using namespace boost::ut;

    "fmt::formatter<vector<bool>>"_test = [] {
        std::vector<bool> boolVector{ true, false, true };
        expect("[true, false, true]" == fmt::format("{}", boolVector));
        expect("[true, false, true]" == fmt::format("{:c}", boolVector));
        expect("[true false true]" == fmt::format("{:s}", boolVector));
    };
};

const boost::ut::suite sourceLocationFormatter = [] {
    using namespace boost::ut;

    "fmt::formatter<std::source_location>"_test = [] {
        auto loc = fmt::format("{}", std::source_location::current());
        fmt::println("location formatter test: {}", loc);
        expect(ge(loc.size(), 0UZ));
    };
};

const boost::ut::suite expectedFormatter = [] {
    using namespace boost::ut;
    using namespace std::string_literals;
    using Expected = std::expected<int, std::string>;

    auto value = fmt::format("{}", Expected(5));
    fmt::println("expected formatter test: {}", value);
    expect(eq(value, "<std::expected-value: 5>"s));

    auto error = fmt::format("{}", Expected(std::unexpected("Error")));
    fmt::println("expected formatter test: {}", error);
    expect(eq(error, "<std::unexpected: Error>"s));
};

} // namespace gr::meta::test

int
main() { /* tests are statically executed */
}