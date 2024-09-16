#include <boost/ut.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::meta::test {

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

} // namespace gr::meta::test

int main() { /* tests are statically executed */ }