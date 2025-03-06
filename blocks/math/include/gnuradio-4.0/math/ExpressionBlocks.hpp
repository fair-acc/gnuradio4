#ifndef EXPRESSIONBLOCKS_HPP
#define EXPRESSIONBLOCKS_HPP

#include <algorithm>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <exprtk.hpp>

namespace gr::blocks::math {

namespace detail {

inline std::string formatParserError(const auto& parser, std::string_view expression) {
    std::stringstream ss;
    for (std::size_t i = 0; i < parser.error_count(); ++i) {
        const auto error = parser.get_error(i);

        ss << fmt::format("ExprTk Parser Error({:2}):  Position: {:2}\nType: [{:14}] Msg: {}; expression:\n{}\n", //
            static_cast<unsigned int>(i),                                                                         //
            static_cast<unsigned int>(error.token.position),                                                      //
            exprtk::parser_error::to_str(error.mode), error.diagnostic, expression);
    }
    return ss.str();
}

struct VectorInfo {
    std::size_t size;
    ssize_t     index;
};

inline VectorInfo computeVectorInfo(void* base_ptr, void* end_ptr, std::size_t elementSize, void* access_ptr) {
    if (!base_ptr || !end_ptr || !access_ptr) {
        throw std::invalid_argument("null pointer(s) provided.");
    }

    auto base   = static_cast<std::byte*>(base_ptr);
    auto end    = static_cast<std::byte*>(end_ptr);
    auto access = static_cast<std::byte*>(access_ptr);

    if (end < base) {
        throw std::out_of_range(fmt::format("invalid vector boundaries [{}, {}]", base_ptr, end_ptr));
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
        throw gr::exception(fmt::format("vector access '{name}[{index}]' outside of [0, {size}[ (typesize: {typesize})", //
            fmt::arg("name", vector_name), fmt::arg("size", vecSize), fmt::arg("index", vecIndex), fmt::arg("typesize", typeSize)));
        return false; // should never reach here
    }
};

} // namespace detail

GR_REGISTER_BLOCK(gr::blocks::math::ExpressionSISO, [ float, double ]);

template<typename T>
requires std::floating_point<T>
struct ExpressionSISO : Block<ExpressionSISO<T>> {
    using Description = Doc<R""(@brief Single-Input-Single-Output (SISO) expression evaluator.

This block uses ExprTK to compute a user-defined expression for each input sample.
The input sample is referenced by the variable `x`, and the output is produced as the evaluated expression.

Examples:
- `y := a * x + b`           // (simple linear scaling)
- `a * x + b`                // (as above, the 'y:=' is optional)
- `y := sin(pi * x)`         // (non-linear transformation)
- `y := y + 0.1*x`           // (recursive IIR-like update using `y` as state)
- `y := clamp(-1.0, x, 1.0)` // (clamping the input range)

For full syntax, conditionals, loops, and advanced features:
@see https://www.partow.net/programming/exprtk/index.html
@see https://github.com/ArashPartow/exprtk
)"">;
    template<typename U, fixed_string description = "", typename... Arguments>
    using A = Annotated<U, description, Arguments...>;

    PortIn<T>  in;
    PortOut<T> out;

    A<std::string, "expr string", Doc<"for syntax see: https://github.com/ArashPartow/exprtk">> expr_string = "clamp(-1.0, sin(2 * pi * x) + cos(x / 2 * pi), +1.0)";
    A<T, "a", Doc<"free parameter 'a' for use in expressions">, Visible>                        param_a     = T(1.0);
    A<T, "b", Doc<"free parameter 'b' for use in expressions">, Visible>                        param_b     = T(0.0);
    A<T, "c", Doc<"free parameter 'c' for use in expressions">, Visible>                        param_c     = T(0.0);

    GR_MAKE_REFLECTABLE(ExpressionSISO, in, out, expr_string, param_a, param_b, param_c);

    exprtk::symbol_table<T> _symbol_table{};
    exprtk::expression<T>   _expression{};
    T                       _in;
    T                       _out;

    void initExpression(std::source_location location = std::source_location::current()) {
        reset();
        _symbol_table.clear();
        _symbol_table.add_variable("x", _in);
        _symbol_table.add_variable("y", _out);

        _symbol_table.add_variable(std::string(param_a.description()), param_a.value);
        _symbol_table.add_variable(std::string(param_b.description()), param_b.value);
        _symbol_table.add_variable(std::string(param_c.description()), param_c.value);

        _symbol_table.add_constants();
        _expression.register_symbol_table(_symbol_table);

        if (exprtk::parser<T> parser; !parser.compile(expr_string, _expression)) {
            throw gr::exception(detail::formatParserError(parser, expr_string), location);
        }
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("expr_string")) {
            initExpression();
        }
    }

    void start() { initExpression(); }

    void reset() {
        _in  = T(0);
        _out = T(0);
    }

    [[nodiscard]] constexpr T processOne(T input) {
        _in  = input;
        _out = _expression.value(); // evaluate expression, _out == 'y' defined to allow for recursion
        return _out;
    }
};

GR_REGISTER_BLOCK(gr::blocks::math::ExpressionDISO, [ float, double ]);

template<typename T>
requires std::floating_point<T>
struct ExpressionDISO : Block<ExpressionDISO<T>> {
    using Description = Doc<R""(@brief Dual-Input-Single-Output (DISO) expression evaluator.

This block uses ExprTK to compute a user-defined expression from two input samples.
The two input samples are referenced by variables `x` and `y`, and the output is `z` (the evaluated expression).

Examples:
- `z := a * (x + y)`                     // (combining two inputs linearly)
- `a * (x + y)`                          // (as above, the 'y:=' is optional)
- `z := sin(x) * cos(y)`                 // (more complex trigonometric transformations)
- `z := z + (x - y)`                     // (recursive usage: `z` can store state)
- `z := inrange(-1, x+y, 1) ? (x+y) : 0` //  (conditional logic)

For full syntax, conditionals, loops, and advanced features:
@see https://www.partow.net/programming/exprtk/index.html
@see https://github.com/ArashPartow/exprtk
)"">;
    template<typename U, fixed_string description = "", typename... Arguments>
    using A = Annotated<U, description, Arguments...>;

    PortIn<T>  in0;
    PortIn<T>  in1;
    PortOut<T> out;

    A<std::string, "expr string", Doc<"for syntax see: https://github.com/ArashPartow/exprtk">> expr_string = "a*(x+y)";
    A<T, "a", Doc<"free parameter 'a' for use in expressions">, Visible>                        param_a     = T(1.0);
    A<T, "b", Doc<"free parameter 'b' for use in expressions">, Visible>                        param_b     = T(0.0);
    A<T, "c", Doc<"free parameter 'c' for use in expressions">, Visible>                        param_c     = T(0.0);

    GR_MAKE_REFLECTABLE(ExpressionDISO, in0, in1, out, expr_string, param_a, param_b, param_c);

    exprtk::symbol_table<T> _symbol_table{};
    exprtk::expression<T>   _expression{};
    T                       _in0;
    T                       _in1;
    T                       _out;

    void initExpression(std::source_location location = std::source_location::current()) {
        reset();
        _symbol_table.clear();
        _symbol_table.add_variable("x", _in0);
        _symbol_table.add_variable("y", _in1);
        _symbol_table.add_variable("z", _out);

        _symbol_table.add_variable(std::string(param_a.description()), param_a.value);
        _symbol_table.add_variable(std::string(param_b.description()), param_b.value);
        _symbol_table.add_variable(std::string(param_c.description()), param_c.value);

        _symbol_table.add_constants();
        _expression.register_symbol_table(_symbol_table);

        if (exprtk::parser<T> parser; !parser.compile(expr_string, _expression)) {
            throw gr::exception(detail::formatParserError(parser, expr_string), location);
        }
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("expr_string")) {
            initExpression();
        }
    }

    void start() { initExpression(); };

    void reset() {
        _in0 = T(0);
        _in1 = T(0);
        _out = T(0);
    }

    [[nodiscard]] constexpr T processOne(T input0, T input1) {
        _in0 = input0;
        _in1 = input1;
        _out = _expression.value(); // evaluate expression, _out == 'y' defined to allow for recursion
        return _out;
    }
};

GR_REGISTER_BLOCK(gr::blocks::math::ExpressionBulk, [ float, double ]);

template<typename T>
requires std::floating_point<T>
struct ExpressionBulk : Block<ExpressionBulk<T>> {
    using Description = Doc<R""(@@brief Bulk array expression evaluator.

This block uses ExprTK to process arrays of input samples (`vecIn`) and produce arrays of output samples (`vecOut`) per work call.
The user-defined expression can manipulate entire arrays at once.

For example:
- `vecOut := a * vecIn;`                  (simple scaling of all input samples)
- `for (i,0,vecIn.size()) vecOut[i] := vecIn[i] + c;` (element-wise operations)
- `vecOut := vecOut + a * vecIn;`         (recursive updates across consecutive calls)

Complex operations (e.g., loops, conditions, indexing) are supported by ExprTK.
For full syntax, conditionals, loops, and advanced features:
@see https://www.partow.net/programming/exprtk/index.html
@see https://github.com/ArashPartow/exprtk
)"">;
    template<typename U, fixed_string description = "", typename... Arguments>
    using A = Annotated<U, description, Arguments...>;

    PortIn<T>  in;
    PortOut<T> out;

    A<std::string, "expr string", Doc<"for syntax see: https://github.com/ArashPartow/exprtk">> expr_string    = "vecOut := a * vecIn;";
    A<T, "a", Doc<"free parameter 'a' for use in expressions">, Visible>                        param_a        = T(1.0);
    A<T, "b", Doc<"free parameter 'b' for use in expressions">, Visible>                        param_b        = T(0.0);
    A<T, "c", Doc<"free parameter 'c' for use in expressions">, Visible>                        param_c        = T(0.0);
    A<bool, "runtime_checks", Doc<"e.g. vector index range checks etc.">, Visible>              runtime_checks = true;

    GR_MAKE_REFLECTABLE(ExpressionBulk, in, out, expr_string, param_a, param_b, param_c, runtime_checks);

    // vector_views that reference _vecInData and _vecOutData
    // will be registered once and then just rebased as needed.
    // N.B. _maxBaseSize limits the maximum chunk size and needs
    // to be defined in-advance due to ExprTk constraints
    std::array<T, 1UZ>           _arrOutDummy{T(0)}; // only needed for initialising
    static constexpr std::size_t _maxBaseSize = 1UZ << 16;
    exprtk::vector_view<T>       _vecIn       = exprtk::make_vector_view<T>(_arrOutDummy.data(), _maxBaseSize);
    exprtk::vector_view<T>       _vecOut      = exprtk::make_vector_view<T>(_arrOutDummy.data(), _maxBaseSize);

    std::vector<T>            _vecInData{};
    std::vector<T>            _vecOutData{};
    detail::vector_access_rtc _vec_rtc{};
    exprtk::symbol_table<T>   _symbol_table{};
    exprtk::expression<T>     _expression{};

    void initExpression(std::source_location location = std::source_location::current()) {
        _expression = exprtk::expression<T>();
        _symbol_table.clear();

        if (_vecInData.empty() || _vecIn.data() == _arrOutDummy.data()) {
            _vecInData.resize(1UZ);
        }
        if (_vecOutData.empty() || _vecOut.data() == _arrOutDummy.data()) {
            _vecOutData.resize(1UZ);
        }

        // rebase vector views to current data
        _vecIn.rebase(_vecInData.data());
        _vecIn.set_size(_vecInData.size());
        _vecOut.rebase(_vecOutData.data());
        _vecOut.set_size(_vecOutData.size());

        _symbol_table.add_vector("vecIn", _vecIn);
        _symbol_table.add_vector("vecOut", _vecOut);

        _symbol_table.add_variable(std::string(param_a.description()), param_a.value);
        _symbol_table.add_variable(std::string(param_b.description()), param_b.value);
        _symbol_table.add_variable(std::string(param_c.description()), param_c.value);

        _symbol_table.add_constants();
        _expression.register_symbol_table(_symbol_table);

        exprtk::parser<T> parser;
        if (runtime_checks) {
            _vec_rtc.vector_map[_vecIn.data()]  = "vecIn";
            _vec_rtc.vector_map[_vecOut.data()] = "vecOut";
            parser.register_vector_access_runtime_check(_vec_rtc);
        }

        if (!parser.compile(expr_string, _expression)) {
            throw gr::exception(detail::formatParserError(parser, expr_string), location);
        }
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("expr_string")) {
            initExpression();
        }
    }

    void start() { initExpression(); }

    work::Status processBulk(InputSpanLike auto& inputSpan, OutputSpanLike auto& outputSpan) {
        if (inputSpan.size() != _vecInData.size() || outputSpan.size() != _vecOutData.size()) {
            _vecInData.resize(std::min(inputSpan.size(), _maxBaseSize));
            _vecOutData.resize(std::min(outputSpan.size(), _maxBaseSize));

            // rebase vector views to new internal memory storage of backing buffer vectors
            _vecIn.rebase(_vecInData.data());
            _vecIn.set_size(_vecInData.size());
            _vecOut.rebase(_vecOutData.data());
            _vecOut.set_size(_vecOutData.size());

            if (runtime_checks) {
                _vec_rtc.vector_map.clear();
                _vec_rtc.vector_map[_vecIn.data()]  = "vecIn";
                _vec_rtc.vector_map[_vecOut.data()] = "vecOut";
            }
        }

        using PtrDiff_t = std::iter_difference_t<decltype(inputSpan.begin())>;
        std::ranges::copy_n(inputSpan.begin(), static_cast<PtrDiff_t>(_vecInData.size()), _vecInData.begin());
        _expression.value(); // evaluate expression, exception handled by caller
        std::ranges::copy_n(_vecOutData.begin(), static_cast<PtrDiff_t>(_vecOutData.size()), outputSpan.begin());

        std::ignore = inputSpan.consume(_vecInData.size());
        outputSpan.publish(_vecOutData.size());
        return work::Status::OK;
    }
};

} // namespace gr::blocks::math

#endif // EXPRESSIONBLOCKS_HPP
