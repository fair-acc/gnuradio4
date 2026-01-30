#include <numbers>
#include <print>
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
    constexpr T t_min = -5;
    constexpr T t_max = 5;
    constexpr T dt    = (t_max - t_min) / T(n);

    auto time_range = std::views::iota(0UZ, n + 1UZ) | std::views::transform([&rng, &noise_dist](std::size_t i) {
        T t = t_min + dt * T(i);
        return std::sin(2.f * std::numbers::pi_v<T> * t) + noise_dist(rng);
    });
    std::ranges::copy(time_range, std::back_inserter(v_in));

    exprtk::symbol_table<T> symbol_table;
    symbol_table.add_vector("v_in", v_in);
    symbol_table.add_vector("v_out", v_out);

    exprtk::expression<T> expression;
    expression.register_symbol_table(symbol_table);

    exprtk::parser<T> parser;
    parser.compile(sgfilter_program, expression);

    expression.value(); // evaluate expression

    for (std::size_t i = 0; i < v_out.size(); ++i) {
        std::print("{:10.6f}\t{:10.6f}\n", v_in[i], v_out[i]);
    }

    return 0;
}
