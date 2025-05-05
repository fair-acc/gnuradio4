#include <boost/ut.hpp>

#include <complex>
#include <expected>

#include <format>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::meta::test {

const boost::ut::suite complexFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::complex_literals;
    using namespace std::literals::string_literals;

    using C                                = std::complex<double>;
    "std::formatter<std::complex<T>>"_test = [] {
        expect(eq("(1+1i)"s, std::format("{}", C(1., +1.))));
        expect(eq("(1-1i)"s, std::format("{}", C(1., -1.))));
        expect(eq("1"s, std::format("{}", C(1., 0.))));
        expect(eq("(1.234+1.12346e+12i)"s, std::format("{}", C(1.234, 1123456789012))));
        expect(eq("(1+1i)"s, std::format("{:g}", C(1., +1.))));
        expect(eq("(1-1i)"s, std::format("{:g}", C(1., -1.))));
        expect(eq("1"s, std::format("{:g}", C(1., 0.))));
        expect(eq("(1.12346e+12+1.234i)"s, std::format("{:g}", C(1123456789012, 1.234))));
        expect(eq("1.12346e+12"s, std::format("{:g}", C(1123456789012, 0))));
        expect(eq("(1.234+1.12346e+12i)"s, std::format("{:g}", C(1.234, 1123456789012))));
        expect(eq("(1.12346E+12+1.234i)"s, std::format("{:G}", C(1123456789012, 1.234))));
        expect(eq("1.12346E+12"s, std::format("{:G}", C(1123456789012, 0))));
        expect(eq("(1.234+1.12346E+12i)"s, std::format("{:G}", C(1.234, 1123456789012))));

        expect(eq("(1.000000+1.000000i)"s, std::format("{:f}", C(1., +1.))));
        expect(eq("(1.000000-1.000000i)"s, std::format("{:f}", C(1., -1.))));
        expect(eq("1.000000"s, std::format("{:f}", C(1., 0.))));
        expect(eq("(1.000000+1.000000i)"s, std::format("{:F}", C(1., +1.))));
        expect(eq("(1.000000-1.000000i)"s, std::format("{:F}", C(1., -1.))));
        expect(eq("1.000000"s, std::format("{:F}", C(1., 0.))));

        expect(eq("(1.000000e+00+1.000000e+00i)"s, std::format("{:e}", C(1., +1.))));
        expect(eq("(1.000000e+00-1.000000e+00i)"s, std::format("{:e}", C(1., -1.))));
        expect(eq("1.000000e+00"s, std::format("{:e}", C(1., 0.))));
        expect(eq("(1.000000E+00+1.000000E+00i)"s, std::format("{:E}", C(1., +1.))));
        expect(eq("(1.000000E+00-1.000000E+00i)"s, std::format("{:E}", C(1., -1.))));
        expect(eq("1.000000E+00"s, std::format("{:E}", C(1., 0.))));
    };
};

const boost::ut::suite uncertainValueFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::complex_literals;
    using namespace std::literals::string_literals;
    using UncertainDouble  = gr::UncertainValue<double>;
    using UncertainComplex = gr::UncertainValue<std::complex<double>>;

    "std::formatter<gr::UncertainValue<T>>"_test = [] {
        // Test with UncertainValue<double>
        expect(eq("(1.23 ± 0.45)"s, std::format("{}", UncertainDouble{1.23, 0.45})));
        expect(eq("(3.14 ± 0.01)"s, std::format("{}", UncertainDouble{3.14, 0.01})));
        expect(eq("(0 ± 0)"s, std::format("{}", UncertainDouble{0, 0})));

        // Test with UncertainValue<std::complex<double>>
        expect(eq("((1+2i) ± (0.1+0.2i))"s, std::format("{}", UncertainComplex{{1, 2}, {0.1, 0.2}})));
        expect(eq("((3.14+1.59i) ± (0.01+0.02i))"s, std::format("{}", UncertainComplex{{3.14, 1.59}, {0.01, 0.02}})));
        expect(eq("(0 ± 0)"s, std::format("{}", UncertainComplex{{0, 0}, {0, 0}})));

        // Test with UncertainValue<double> and float number formatting
        expect(eq("(1.230 ± 0.450)"s, std::format("{:1.3f}", UncertainDouble{1.23, 0.45})));
        expect(eq("(3.140 ± 0.010)"s, std::format("{:1.3f}", UncertainDouble{3.14, 0.01})));
        expect(eq("(0.000 ± 0.000)"s, std::format("{:1.3f}", UncertainDouble{0, 0})));

        std::stringstream ss;
        ss << UncertainDouble{1.23, 0.45};
        expect(eq("(1.23 ± 0.45)"s, ss.str()));
    };
};

const boost::ut::suite propertyMapFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "std::formatter<gr::property_map>"_test = [] {
        gr::property_map pmInt{{"key0", 0}, {"key1", 1}, {"key2", 2}};
        expect(eq("{ key0: 0, key1: 1, key2: 2 }"s, std::format("{}", pmInt)));

        gr::property_map pmFloat{{"key0", 0.01f}, {"key1", 1.01f}, {"key2", 2.01f}};
        expect(eq("{ key0: 0.01, key1: 1.01, key2: 2.01 }"s, std::format("{}", pmFloat)));
    };
};

const boost::ut::suite vectorBoolFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "std::formatter<vector<bool>>"_test = [] {
        std::vector<bool> boolVector{true, false, true};
        expect(eq("[true, false, true]"s, std::format("{}", boolVector)));
        expect(eq("[true, false, true]"s, std::format("{:c}", boolVector)));
        expect(eq("[true false true]"s, std::format("{:s}", boolVector)));
    };
};

const boost::ut::suite expectedFormatter = [] {
    using namespace boost::ut;
    using namespace std::string_literals;
    using Expected = std::expected<int, std::string>;

    auto value = std::format("{}", Expected(5));
    std::println("expected formatter test: {}", value);

    auto error = std::format("{}", Expected(std::unexpected("Error")));
    std::println("expected formatter test: {}", error);
#if FMT_VERSION < 110000
    expect(eq(value, "<std::expected-value: 5>"s));
    expect(eq(error, "<std::unexpected: Error>"s));
#else
    expect(eq(value, "expected(5)"s));
    expect(eq(error, "unexpected(\"Error\")"s));
#endif
};

const boost::ut::suite<"Range<T> formatter"> _rangeFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "std::formatter<Range<std::int32t>>"_test = [] {
        gr::Range<std::int32_t> range{-2, 2};
        expect(eq("[min: -2, max: 2]"s, std::format("{}", range)));
    };
    "std::formatter<Range<float>>"_test = [] {
        gr::Range<float> range{-2.5f, 2.5f};
        expect(eq("[min: -2.5, max: 2.5]"s, std::format("{}", range)));
    };
};

} // namespace gr::meta::test

int main() { /* tests are statically executed */ }
