#include <fmt/core.h>
#include <fmt/ranges.h>
#include <numbers>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#include <exprtk.hpp>

int main() {
    using T = float;

    const std::string sgfilter_program = R"(
       var weight[9] := {
           -21, 14, 39,
           54, 59, 54,
           39, 14, -21
       };

       if (v_in[] >= weight[]) {
           const var lower_bound := trunc(weight[] / 2);
           const var upper_bound := v_in[] - lower_bound;

           v_out := 0;

           for (var i := lower_bound; i < upper_bound; i += 1) {
               for (var j := -lower_bound; j <= lower_bound; j += 1) {
                   v_out[i] += weight[j + lower_bound] * v_in[i + j];
               };
           };

           v_out /= sum(weight);
       }
   )";

    constexpr std::size_t n = 1024UZ;

    std::vector<T> v_in;
    std::vector<T> v_out(n);

    std::mt19937                      rng(std::random_device{}());
    std::uniform_real_distribution<T> noise_dist(-0.25, 0.25);

    // generate signal with noise
    for (T t = T(-5); t <= T(5); t += T(10.0 / n)) {
        const T noise = noise_dist(rng);
        v_in.push_back(std::sin(2.f * std::numbers::pi_v<T> * t) + noise);
    }

    exprtk::symbol_table<T> symbol_table;
    symbol_table.add_vector("v_in", v_in);
    symbol_table.add_vector("v_out", v_out);

    exprtk::expression<T> expression;
    expression.register_symbol_table(symbol_table);

    exprtk::parser<T> parser;
    parser.compile(sgfilter_program, expression);

    expression.value(); // evaluate expression

    for (std::size_t i = 0; i < v_out.size(); ++i) {
        fmt::print("{:10.6f}\t{:10.6f}\n", v_in[i], v_out[i]);
    }

    return 0;
}
