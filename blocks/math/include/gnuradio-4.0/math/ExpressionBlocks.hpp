#ifndef EXPRESSIONBLOCKS_HPP
#define EXPRESSIONBLOCKS_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <exprtk.hpp>

namespace gr::blocks::math {

template<typename T>
requires std::floating_point<T>
struct ExpressionSISO : Block<ExpressionSISO<T>> {
    using Description = Doc<R""(@brief ExpressionSISO

<provide summary, link to https://github.com/ArashPartow/exprtk for more examples on the syntax, ... expand more>
)"">;
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = Annotated<U, description, Arguments...>;

    PortIn<T>  in;
    PortOut<T> out;

    A<std::string, "expr string", Doc<"for syntax see: https://github.com/ArashPartow/exprtk">> expr_string = "clamp(-1.0, sin(2 * pi * x) + cos(x / 2 * pi), +1.0)";
    A<T, "a", Doc<"free parameter 'a' for use in expressions">, Visible>                        param_a     = T(1.0);
    A<T, "b", Doc<"free parameter 'b' for use in expressions">, Visible>                        param_b     = T(0.0);
    A<T, "c", Doc<"free parameter 'c' for use in expressions">, Visible>                        param_c     = T(0.0);

    GR_MAKE_REFLECTABLE(ExpressionSISO, in, out, expr_string, param_a, param_b, param_c);

    exprtk::symbol_table<T> _symbol_table;
    exprtk::expression<T>   _expression;
    T                       _in;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("expr_string")) {
            if (exprtk::parser<T> parser; !parser.compile(expr_string, _expression)) {
                throw gr::exception(fmt::format("Expression parsing failed: expression:\n{}\n", expr_string));
            }
        }
    }

    void start() {
        _symbol_table.add_variable("x", _in);

        _symbol_table.add_variable(std::string(param_a.description()), param_a);
        _symbol_table.add_variable(std::string(param_b.description()), param_b);
        _symbol_table.add_variable(std::string(param_c.description()), param_c);

        _symbol_table.add_constants();
        _expression.register_symbol_table(_symbol_table);

        if (exprtk::parser<T> parser; !parser.compile(expr_string, _expression)) {
            throw gr::exception(fmt::format("Expression parsing failed: expression:\n{}\n", expr_string));
        }
    }

    void stop() { _symbol_table.clear(); }

    [[nodiscard]] constexpr T processOne(T input) {
        _in = input;
        return _expression.value(); // evaluate expression;
    }
};

template<typename T>
requires std::floating_point<T>
struct ExpressionDISO : Block<ExpressionDISO<T>> {
    using Description = Doc<R""(@brief ExpressionDISO

<provide summary, DISO : Dual-Input-Single-Output, link to https://github.com/ArashPartow/exprtk for more examples on the syntax, ... expand more>
)"">;
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = Annotated<U, description, Arguments...>;

    PortIn<T>  in0;
    PortIn<T>  in1;
    PortOut<T> out;

    A<std::string, "expr string", Doc<"for syntax see: https://github.com/ArashPartow/exprtk">> expr_string = "a*(x0+x1)";
    A<T, "a", Doc<"free parameter 'a' for use in expressions">, Visible>                        param_a     = T(1.0);
    A<T, "b", Doc<"free parameter 'b' for use in expressions">, Visible>                        param_b     = T(0.0);
    A<T, "c", Doc<"free parameter 'c' for use in expressions">, Visible>                        param_c     = T(0.0);

    GR_MAKE_REFLECTABLE(ExpressionDISO, in0, in1, out, expr_string, param_a, param_b, param_c);

    exprtk::symbol_table<T> _symbol_table;
    exprtk::expression<T>   _expression;
    T                       _in0;
    T                       _in1;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("expr_string")) {
            if (exprtk::parser<T> parser; !parser.compile(expr_string, _expression)) {
                throw gr::exception(fmt::format("Expression parsing failed: expression:\n{}\n", expr_string));
            }
        }
    }

    void start() {
        _symbol_table.add_variable("x0", _in0);
        _symbol_table.add_variable("x1", _in1);

        _symbol_table.add_variable(std::string(param_a.description()), param_a);
        _symbol_table.add_variable(std::string(param_b.description()), param_b);
        _symbol_table.add_variable(std::string(param_c.description()), param_c);

        _symbol_table.add_constants();
        _expression.register_symbol_table(_symbol_table);

        if (exprtk::parser<T> parser; !parser.compile(expr_string, _expression)) {
            throw gr::exception(fmt::format("Expression parsing failed: expression:\n{}\n", expr_string));
        }
    }

    void stop() { _symbol_table.clear(); }

    [[nodiscard]] constexpr T processOne(T input0, T input1) {
        _in0 = input0;
        _in1 = input1;
        return _expression.value(); // evaluate expression;
    }
};

template<typename T>
requires std::floating_point<T>
struct ExpressionBulk : public gr::Block<ExpressionBulk<T>> {
    using Description = Doc<R""(@brief ExpressionBulk

This block uses ExprTK to evaluate user-defined expressions on arrays of input samples to produce arrays of output samples.

The `expr_string` can reference arrays `v_in` and `v_out`. The user must ensure their expression sets `v_out` appropriately.
For example, a simple pass-through could be:
v_out := v_in;
For more complex operations (e.g. smoothing filters), you can use loops and conditions as shown in the ExprTK documentation.

@see https://github.com/ArashPartow/exprtk
)"">;
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    PortIn<T>  in;
    PortOut<T> out;

    A<std::string, "expr string", Doc<"for syntax see: https://github.com/ArashPartow/exprtk">> expr_string = "v_out := a * v_in;";
    A<T, "a", Doc<"free parameter 'a' for use in expressions">, Visible>                        param_a     = T(1.0);
    A<T, "b", Doc<"free parameter 'b' for use in expressions">, Visible>                        param_b     = T(0.0);
    A<T, "c", Doc<"free parameter 'c' for use in expressions">, Visible>                        param_c     = T(0.0);

    GR_MAKE_REFLECTABLE(ExpressionBulk, in, out, expr_string, param_a, param_b, param_c);

    exprtk::symbol_table<T> _symbol_table;
    exprtk::expression<T>   _expression;

    // Arrays for ExprTK processing
    std::vector<T> v_in;
    std::vector<T> v_out;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("expr_string")) {
            if (exprtk::parser<T> parser; !parser.compile(expr_string, _expression)) {
                throw gr::exception(fmt::format("Expression parsing failed:\n{}\n", expr_string.value));
            }
        }
    }

    void start() {
        _symbol_table.add_vector("v_in", v_in);
        _symbol_table.add_vector("v_out", v_out);

        _symbol_table.add_variable(std::string(param_a.description()), param_a);
        _symbol_table.add_variable(std::string(param_b.description()), param_b);
        _symbol_table.add_variable(std::string(param_c.description()), param_c);

        _symbol_table.add_constants();
        _expression.register_symbol_table(_symbol_table);

        if (exprtk::parser<T> parser; !parser.compile(expr_string, _expression)) {
            throw gr::exception(fmt::format("Expression parsing failed:\n{}\n", expr_string.value));
        }
    }

    void stop() { _symbol_table.clear(); }

    work::Status processBulk(InputSpanLike auto& inputSpan, OutputSpanLike auto& outputSpan) {
        const std::size_t n = inputSpan.size();

        v_in.resize(n);
        v_out.resize(n);

        std::copy_n(inputSpan.begin(), n, v_in.begin());
        _expression.value();

        std::copy_n(v_out.begin(), n, outputSpan.begin());

        return work::Status::OK;
    }
};

} // namespace gr::blocks::math

const inline static auto registerConstMath = gr::registerBlock<gr::blocks::math::ExpressionSISO, float, double>(gr::globalBlockRegistry())   //
                                             + gr::registerBlock<gr::blocks::math::ExpressionDISO, float, double>(gr::globalBlockRegistry()) //
                                             + gr::registerBlock<gr::blocks::math::ExpressionBulk, float, double>(gr::globalBlockRegistry());

#endif // EXPRESSIONBLOCKS_HPP
