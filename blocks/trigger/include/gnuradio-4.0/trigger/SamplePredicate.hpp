#ifndef GNURADIO_TRIGGER_SAMPLEPREDICATE_HPP
#define GNURADIO_TRIGGER_SAMPLEPREDICATE_HPP

#include <optional>
#include <string>
#include <type_traits>

#include <exprtk.hpp> // only the headers that include this one pay for it; the enum path needs nothing

#include <gnuradio-4.0/trigger/SampleTest.hpp>

namespace gr::blocks::trigger::detail {

template<typename T>
struct SamplePredicate {
    Comparison comparison = Comparison::greater;
    T          threshold  = T{};
    bool       compiled   = false;

    T                       _sample = T{};
    exprtk::symbol_table<T> _symbols{};
    exprtk::expression<T>   _expression{};

    [[nodiscard]] std::optional<std::string> configure(std::string_view predicate, std::string_view expression, T threshold_) {
        comparison = parseComparison(predicate);
        threshold  = threshold_;
        compiled   = false;
        if (expression.empty()) {
            return std::nullopt;
        }
        if constexpr (!std::is_floating_point_v<T>) {
            return std::string("an expression needs a floating-point sample type; the comparison is used instead");
        } else {
            _symbols.clear();
            _symbols.add_variable("x", _sample);
            _symbols.add_variable("threshold", threshold);
            _symbols.add_constants();
            _expression.register_symbol_table(_symbols);
            exprtk::parser<T> parser;
            if (!parser.compile(std::string(expression), _expression)) {
                return std::string(parser.error());
            }
            compiled = true;
            return std::nullopt;
        }
    }

    [[nodiscard]] bool operator()(const T& sample) {
        if (!compiled) {
            return holds(comparison, sample, threshold);
        }
        _sample = sample;
        return _expression.value() != T{};
    }
};

} // namespace gr::blocks::trigger::detail

#endif // GNURADIO_TRIGGER_SAMPLEPREDICATE_HPP
