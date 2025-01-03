#include <boost/ut.hpp>

#include <complex>
#include <expected>

#include <fmt/format.h>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::meta::test {

const boost::ut::suite complexFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::complex_literals;
    using namespace std::literals::string_literals;

    using C                                = std::complex<double>;
    "fmt::formatter<std::complex<T>>"_test = [] {
        expect(eq("(1+1i)"s, fmt::format("{}", C(1., +1.))));
        expect(eq("(1-1i)"s, fmt::format("{}", C(1., -1.))));
        expect(eq("1"s, fmt::format("{}", C(1., 0.))));
        expect(eq("(1.234+1.12346e+12i)"s, fmt::format("{}", C(1.234, 1123456789012))));
        expect(eq("(1+1i)"s, fmt::format("{:g}", C(1., +1.))));
        expect(eq("(1-1i)"s, fmt::format("{:g}", C(1., -1.))));
        expect(eq("1"s, fmt::format("{:g}", C(1., 0.))));
        expect(eq("(1.12346e+12+1.234i)"s, fmt::format("{:g}", C(1123456789012, 1.234))));
        expect(eq("1.12346e+12"s, fmt::format("{:g}", C(1123456789012, 0))));
        expect(eq("(1.234+1.12346e+12i)"s, fmt::format("{:g}", C(1.234, 1123456789012))));
        expect(eq("(1.12346E+12+1.234i)"s, fmt::format("{:G}", C(1123456789012, 1.234))));
        expect(eq("1.12346E+12"s, fmt::format("{:G}", C(1123456789012, 0))));
        expect(eq("(1.234+1.12346E+12i)"s, fmt::format("{:G}", C(1.234, 1123456789012))));

        expect(eq("(1.000000+1.000000i)"s, fmt::format("{:f}", C(1., +1.))));
        expect(eq("(1.000000-1.000000i)"s, fmt::format("{:f}", C(1., -1.))));
        expect(eq("1.000000"s, fmt::format("{:f}", C(1., 0.))));
        expect(eq("(1.000000+1.000000i)"s, fmt::format("{:F}", C(1., +1.))));
        expect(eq("(1.000000-1.000000i)"s, fmt::format("{:F}", C(1., -1.))));
        expect(eq("1.000000"s, fmt::format("{:F}", C(1., 0.))));

        expect(eq("(1.000000e+00+1.000000e+00i)"s, fmt::format("{:e}", C(1., +1.))));
        expect(eq("(1.000000e+00-1.000000e+00i)"s, fmt::format("{:e}", C(1., -1.))));
        expect(eq("1.000000e+00"s, fmt::format("{:e}", C(1., 0.))));
        expect(eq("(1.000000E+00+1.000000E+00i)"s, fmt::format("{:E}", C(1., +1.))));
        expect(eq("(1.000000E+00-1.000000E+00i)"s, fmt::format("{:E}", C(1., -1.))));
        expect(eq("1.000000E+00"s, fmt::format("{:E}", C(1., 0.))));
    };
};

const boost::ut::suite uncertainValueFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::complex_literals;
    using namespace std::literals::string_literals;
    using UncertainDouble  = gr::UncertainValue<double>;
    using UncertainComplex = gr::UncertainValue<std::complex<double>>;

    "fmt::formatter<gr::UncertainValue<T>>"_test = [] {
        // Test with UncertainValue<double>
        expect(eq("(1.23 ± 0.45)"s, fmt::format("{}", UncertainDouble{1.23, 0.45})));
        expect(eq("(3.14 ± 0.01)"s, fmt::format("{}", UncertainDouble{3.14, 0.01})));
        expect(eq("(0 ± 0)"s, fmt::format("{}", UncertainDouble{0, 0})));

        // Test with UncertainValue<std::complex<double>>
        expect(eq("((1+2i) ± (0.1+0.2i))"s, fmt::format("{}", UncertainComplex{{1, 2}, {0.1, 0.2}})));
        expect(eq("((3.14+1.59i) ± (0.01+0.02i))"s, fmt::format("{}", UncertainComplex{{3.14, 1.59}, {0.01, 0.02}})));
        expect(eq("(0 ± 0)"s, fmt::format("{}", UncertainComplex{{0, 0}, {0, 0}})));

        // Test with UncertainValue<double> and float number formatting
        expect(eq("(1.230 ± 0.450)"s, fmt::format("{:1.3f}", UncertainDouble{1.23, 0.45})));
        expect(eq("(3.140 ± 0.010)"s, fmt::format("{:1.3f}", UncertainDouble{3.14, 0.01})));
        expect(eq("(0.000 ± 0.000)"s, fmt::format("{:1.3f}", UncertainDouble{0, 0})));

        std::stringstream ss;
        ss << UncertainDouble{1.23, 0.45};
        expect(eq("(1.23 ± 0.45)"s, ss.str()));
    };
};

const boost::ut::suite propertyMapFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "fmt::formatter<gr::property_map>"_test = [] {
        gr::property_map pmInt{{"key0", 0}, {"key1", 1}, {"key2", 2}};
        expect(eq("{ key0: 0, key1: 1, key2: 2 }"s, fmt::format("{}", pmInt)));

        gr::property_map pmFloat{{"key0", 0.01f}, {"key1", 1.01f}, {"key2", 2.01f}};
        expect(eq("{ key0: 0.01, key1: 1.01, key2: 2.01 }"s, fmt::format("{}", pmFloat)));
    };
};

const boost::ut::suite vectorBoolFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "fmt::formatter<vector<bool>>"_test = [] {
        std::vector<bool> boolVector{true, false, true};
        expect(eq("[true, false, true]"s, fmt::format("{}", boolVector)));
        expect(eq("[true, false, true]"s, fmt::format("{:c}", boolVector)));
        expect(eq("[true false true]"s, fmt::format("{:s}", boolVector)));
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

const boost::ut::suite<"Range<T> formatter"> _rangeFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "fmt::formatter<Range<std::int32t>>"_test = [] {
        gr::Range<std::int32_t> range{-2, 2};
        expect(eq("[min: -2, max: 2]"s, fmt::format("{}", range)));
    };
    "fmt::formatter<Range<float>>"_test = [] {
        gr::Range<float> range{-2.5f, 2.5f};
        expect(eq("[min: -2.5, max: 2.5]"s, fmt::format("{}", range)));
    };
};

} // namespace gr::meta::test

int main() { /* tests are statically executed */ }
