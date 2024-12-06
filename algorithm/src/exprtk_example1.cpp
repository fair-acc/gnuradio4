#include <fmt/core.h>
#include <ranges>
#include <string>

#include <exprtk.hpp>

int main() {
    using T = float;

    T x = T(0);

    exprtk::function_compositor<T> compositor;
    using function_t = typename exprtk::function_compositor<T>::function;
    compositor.add(function_t("fibonacci").var("x").expression(R"(
           switch {
               case x == 0 : 0;
               case x == 1 : 1;
               default     : {
                   var prev := 0;
                   var curr := 1;
                   while ((x -= 1) > 0) {
                       var temp := prev;
                       prev := curr;
                       curr += temp;
                   };
               };
           }
       )"));

    exprtk::symbol_table<T>& symbol_table = compositor.symbol_table();
    symbol_table.add_constants();
    symbol_table.add_variable("x", x);

    std::string expression_str = "fibonacci(x)";

    exprtk::expression<T> expression;
    expression.register_symbol_table(symbol_table);

    exprtk::parser<T> parser;
    parser.compile(expression_str, expression);

    for (std::size_t i = 0UZ; i < 10UZ; ++i) {
        x              = static_cast<T>(i);
        const T result = expression.value(); // evaluate expression

        fmt::println("fibonacci({:3}) = {:10.0f}", i, result);
    }

    return 0;
}
