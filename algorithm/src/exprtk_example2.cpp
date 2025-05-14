#include <format>
#include <print>
#include <ranges>
#include <string>

#include <exprtk.hpp>

#include <gnuradio-4.0/meta/formatter.hpp>

struct VectorInfo {
    std::size_t size;
    ssize_t     index;
};

VectorInfo computeVectorInfo(void* base_ptr, void* end_ptr, std::size_t elementSize, void* access_ptr) {
    if (!base_ptr || !end_ptr || !access_ptr) {
        throw std::invalid_argument("null pointer(s) provided.");
    }

    auto base   = static_cast<std::byte*>(base_ptr);
    auto end    = static_cast<std::byte*>(end_ptr);
    auto access = static_cast<std::byte*>(access_ptr);

    if (end < base) {
        throw std::out_of_range(std::format("invalid vector boundaries [{}, {}]", base_ptr, end_ptr));
    }

    return {static_cast<std::size_t>(end - base) / elementSize, (access - base) / static_cast<ssize_t>(elementSize)};
}

struct vector_access_rtc : public exprtk::vector_access_runtime_check {
    std::unordered_map<void*, std::string> vector_map;

    bool handle_runtime_violation(violation_context& context) override {
        auto               itr         = vector_map.find(static_cast<void*>(context.base_ptr));
        const std::string& vector_name = (itr != vector_map.end()) ? itr->second : "Unknown";

        const auto typeSize      = static_cast<std::size_t>(context.type_size);
        auto [vecSize, vecIndex] = computeVectorInfo(context.base_ptr, context.end_ptr, typeSize, context.access_ptr);
        throw std::runtime_error(std::format("vector access '{}[{}]' outside of [0, {}[ (typesize: {})", vector_name, vecIndex, vecSize, typeSize));
        return false; // should never reach here
    }
};

int main() {
    using T = float;

    const std::string bubblesort_program = R"(
       var upper_bound := v[];
       repeat
           var new_upper_bound := 0;
           for (var i := 1; i < upper_bound; i += 1) {
               if (v[i - 1] > v[i]) {
                   v[i - 1] <=> v[i];
                   new_upper_bound := i;
               };
           };
           upper_bound := new_upper_bound;
       until (upper_bound <= 1);

      for (var i := 8; i < 10; i += 1) {
        v[i] := 0; // causes deliberate out-of-bound exception
      }
   )";

    std::array<T, 7> arr{T(9.1), T(2.2), T(1.3), T(5.4), T(7.5), T(4.6), T(3.7)};
    std::println("Input array:  {}", gr::join(arr));

    exprtk::symbol_table<T> symbol_table;
    symbol_table.add_vector("v", arr.data(), arr.size());

    exprtk::expression<T> expression;
    expression.register_symbol_table(symbol_table);

    vector_access_rtc vec_rtc;

    vec_rtc.vector_map[arr.data()] = "v";

    exprtk::parser<T> parser;
    parser.register_vector_access_runtime_check(vec_rtc);

    if (!parser.compile(bubblesort_program, expression)) { // more proper parser error handling
        std::println("Error: {}\tExpression:{}", parser.error(), bubblesort_program);

        for (std::size_t i = 0; i < parser.error_count(); ++i) {
            const auto error = parser.get_error(i);

            std::println("Error: {:2}  Position: {:2} Type: [{:14}] Msg: {}\tExpression: {}", //
                static_cast<unsigned int>(i),                                                 //
                static_cast<unsigned int>(error.token.position),                              //
                exprtk::parser_error::to_str(error.mode), error.diagnostic, bubblesort_program);
        }

        return -1;
    }

    try {
        expression.value();                   // evaluate expression
    } catch (std::runtime_error& exception) { // handle runtime errors
        std::println("Caught Exception: {}", exception.what());
    }

    std::println("Sorted array: {}", gr::join(arr));

    return 0;
}
