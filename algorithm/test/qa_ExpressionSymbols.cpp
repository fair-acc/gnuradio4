#include <boost/ut.hpp>

#include <cmath>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/algorithm/ExpressionSymbols.hpp>

namespace {
struct Thresholds : gr::Block<Thresholds> {
    gr::PortIn<double>  in;
    gr::PortOut<double> out;

    gr::Annotated<double, "limit">       threshold = 2.0;
    gr::Annotated<float, "single scale"> scale     = 3.f;

    GR_MAKE_REFLECTABLE(Thresholds, in, out, threshold, scale);

    [[nodiscard]] constexpr double processOne(double x) const noexcept { return x; }
};
} // namespace

const boost::ut::suite<"ExpressionSymbols"> expressionSymbolsTests = [] {
    using namespace boost::ut;
    using gr::algorithm::ExpressionSymbols;
    using gr::algorithm::UnknownIdentifiers;

    "a setting is bound by its name and by its description, by reference"_test = [] {
        Thresholds                 block;
        ExpressionSymbols<double>  symbols;
        exprtk::expression<double> byName;
        exprtk::expression<double> byDescription;
        gr::algorithm::bindSettings(symbols.table, block);
        expect(symbols.compile("threshold * 2", byName).has_value());
        expect(symbols.compile("limit * 2", byDescription).has_value());
        block.threshold = 5.0;
        expect(eq(byName.value(), 10.0)) << "a later change of the setting is seen without recompiling";
        expect(eq(byDescription.value(), 10.0));
    };

    "a setting of another type is not bound"_test = [] {
        Thresholds                 block;
        ExpressionSymbols<double>  symbols;
        exprtk::expression<double> expression;
        gr::algorithm::bindSettings(symbols.table, block);
        expect(!symbols.compile("scale", expression).has_value()) << "a float setting does not alias a double variable";
    };

    "an unknown identifier is refused unless it may come from a tag"_test = [] {
        ExpressionSymbols<double>  symbols;
        exprtk::expression<double> expression;
        expect(!symbols.compile("sampel_rate > 1", expression).has_value()) << "by default a typo stays a compile error";
    };

    "a tag field is NaN until a tag carries it, then keeps the last value"_test = [] {
        ExpressionSymbols<double>  symbols;
        exprtk::expression<double> expression;
        double                     x = 1.0;
        symbols.table.add_variable("x", x);
        expect(symbols.compile("x + sample_rate", expression, UnknownIdentifiers::readFromTags).has_value()) << fatal;
        expect(eq(symbols.tagFields.size(), 1UZ));
        expect(std::isnan(expression.value()));

        symbols.assign(gr::property_map_view{gr::property_map{{std::string(gr::tag::SAMPLE_RATE.key()), 1000.f}}});
        expect(eq(expression.value(), 1001.0)) << "the gr:-prefixed key and a float value are both accepted";
        symbols.assign(gr::property_map_view{gr::property_map{{"unrelated", 1}}});
        expect(eq(expression.value(), 1001.0)) << "a tag without the field leaves it unchanged";
        symbols.assign(gr::property_map_view{gr::property_map{{"sample_rate", 2000.0}}});
        expect(eq(expression.value(), 2001.0)) << "an unprefixed key is read too";
    };

    "a tag field and a setting with the same name resolve to the setting"_test = [] {
        Thresholds                 block;
        ExpressionSymbols<double>  symbols;
        exprtk::expression<double> expression;
        gr::algorithm::bindSettings(symbols.table, block);
        expect(symbols.compile("threshold", expression, UnknownIdentifiers::readFromTags).has_value());
        expect(symbols.tagFields.empty()) << "a bound setting is never a tag field";
        symbols.assign(gr::property_map_view{gr::property_map{{"threshold", 9.0}}});
        expect(eq(expression.value(), 2.0));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
