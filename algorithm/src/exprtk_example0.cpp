#include <gnuradio-4.0/meta/formatter.hpp>

#include <exprtk.hpp>

int main() {
    exprtk::symbol_table<float> symbol_table;
    float                       x = 1.f;
    symbol_table.add_variable("x", x);
    symbol_table.add_constants();

    exprtk::expression<float> expression;
    expression.register_symbol_table(symbol_table);

    const std::string exprString = "clamp(-1.0, sin(2 * pi * x) + cos(x / 2 * pi), +1.0)";
    if (exprtk::parser<float> parser; !parser.compile(exprString, expression)) {
        fmt::println(stderr, "Expression parsing failed.");
        return -1;
    }

    x            = 0.5;
    float result = expression.value(); // evaluate expression
    fmt::println("Result: {}", result);

    return 0;
}
