#include <boost/ut.hpp>

#include <array>
#include <complex>
#include <expected>
#include <list>
#include <map>
#include <vector>

#include <format>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::meta::test {

const boost::ut::suite<"Source Location formatter"> sourceLocationFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "std::formatter<std::source_location>"_test = [] {
        const auto loc = std::source_location::current();
        expect(!std::format("{:s}", loc).empty());
        expect(!std::format("{:t}", loc).empty());
        expect(!std::format("{:f}", loc).empty());
    };
};

const boost::ut::suite<"Pointer helper"> pointerHelper = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "gr::ptr<T*>"_test = [] {
        int  val = 42;
        auto str = gr::ptr(&val);
        expect(str.starts_with("0x")); // not deeply checked, just format
    };
};

const boost::ut::suite<"gr::join helper"> joinHelpers = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "gr::join<vector<int>>"_test = [] {
        std::vector<int> v{1, 2, 3};
        expect(eq("1, 2, 3"s, gr::join(v)));
    };

    "gr::format_join<vector<int>>"_test = [] {
        std::vector<int>   v{1, 2, 3};
        std::ostringstream oss;
        gr::format_join(std::ostream_iterator<char>(oss), v, "; ");
        expect(eq("1; 2; 3"s, oss.str()));
    };
};

const boost::ut::suite<"std::complex formatter"> complexFormatter = [] {
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

const boost::ut::suite<"expected formatter"> expectedFormatter = [] {
    using namespace boost::ut;
    using namespace std::string_literals;
    using Expected = std::expected<int, std::string>;

    auto value = std::format("{}", Expected(5));
    std::println("expected formatter test: {}", value);

    auto error = std::format("{}", Expected(std::unexpected("Error")));
    std::println("expected formatter test: {}", error);
    expect(eq(value, "<std::expected-value: 5>"s));
    expect(eq(error, "<std::unexpected: Error>"s));
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

const boost::ut::suite<"std::pair<,> formatter"> pairFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "std::formatter<std::pair<T1, T2>>"_test = [] {
        expect(eq("(42, test)"s, std::format("{}", std::pair{42, "test"s})));
        expect(eq("(3.14, 1.59)"s, std::format("{}", std::pair{3.14, 1.59})));
    };
};

const boost::ut::suite<"Formatable range/Collection formatter"> rangeFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "std::formatter<std::vector<int>>"_test = [] {
        std::vector<int> v{1, 2, 3};
        expect(eq("[1, 2, 3]"s, std::format("{}", v)));
    };

    "std::formatter<std::array<float, N>>"_test = [] {
        std::array<float, 3> arr{1.1f, 2.2f, 3.3f};
        expect(eq("[1.1, 2.2, 3.3]"s, std::format("{}", arr)));
    };

    "std::formatter<std::list<std::string>>"_test = [] {
        std::list<std::string> l{"foo", "bar", "baz"};
        expect(eq("[foo, bar, baz]"s, std::format("{}", l)));
    };

    "std::formatter<std::span<int>>"_test = [] {
        int            raw[] = {10, 20, 30};
        std::span<int> s{raw};
        expect(eq("[10, 20, 30]"s, std::format("{}", s)));
    };

    "std::formatter<std::ranges::iota_view<int, int>>"_test = [] {
        auto range = std::views::iota(0, 4);
        expect(eq("[0, 1, 2, 3]"s, std::format("{}", range)));
    };

    "std::formatter<std::map<int, std::string>> (pair formatter)"_test = [] {
        std::map<int, std::string> m{{1, "one"}, {2, "two"}};
        std::string                joined;
        for (const auto& p : m) {
            if (!joined.empty()) {
                joined += ", ";
            }
            joined += std::format("{}", p);
        }
        expect(eq("(1, one), (2, two)"s, joined));
    };
};

const boost::ut::suite<"std::exception formatter"> exceptionFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "std::formatter<std::exception>"_test = [] {
        std::runtime_error err("failure");
        expect(eq("failure"s, std::format("{}", err)));
    };
};

enum class Mode { A, B, C };

const boost::ut::suite enumFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "std::formatter<enum class>"_test = [] {
        expect(eq("A"s, std::format("{}", Mode::A)));
        expect(eq("     B"s, std::format("{:>6}", Mode::B)));
        expect(eq("42"s, std::format("{}", static_cast<Mode>(42)))); // fallback
    };
};

} // namespace gr::meta::test

int main() { /* tests are statically executed */ }
