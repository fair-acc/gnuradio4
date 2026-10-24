#ifndef GNURADIO_TRIGGER_SAMPLEPREDICATE_HPP
#define GNURADIO_TRIGGER_SAMPLEPREDICATE_HPP

#include <concepts>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

#include <exprtk.hpp>

#include <gnuradio-4.0/algorithm/ExpressionSymbols.hpp>
#include <gnuradio-4.0/trigger/SampleTest.hpp>

namespace gr::blocks::trigger::detail {

/** the expression's symbols for a floating-point sample type, nothing for the others: an integer predicate compares only */
template<typename T>
struct SymbolsFor {
    using type = std::monostate;
};

template<std::floating_point T>
struct SymbolsFor<T> {
    using type = gr::algorithm::ExpressionSymbols<T>;
};

template<typename T>
struct SamplePredicate {
    Comparison comparison = Comparison::greater;
    T          threshold  = T{};
    bool       compiled   = false;

    T                            _sample = T{};
    typename SymbolsFor<T>::type _symbols;
    exprtk::expression<T>        _expression{};

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
            _symbols.table.clear();
            _symbols.table.add_variable("x", _sample);
            _symbols.table.add_variable("threshold", threshold);
            if (const auto built = _symbols.compile(std::string(expression), _expression); !built) {
                return built.error();
            }
            compiled = true;
            return std::nullopt;
        }
    }

    [[nodiscard]] std::string_view hostOnlyReason() const noexcept { return compiled ? "an ExprTk 'expression' runs on the host only" : ""; }

    [[nodiscard]] bool operator()(const T& sample) {
        if (!compiled) {
            return holds(comparison, sample, threshold);
        }
        _sample = sample;
        return _expression.value() != T{};
    }
};

/**
 * a body's test of one sample: the whole predicate on the host; in a kernel, which `hostOnlyReason` keeps from an
 * expression, the comparison the settings give
 */
template<bool kOnHost, typename T>
[[nodiscard]] bool passes(SamplePredicate<T>& test, Comparison comparison, const T& sample, const T& threshold) {
    if constexpr (kOnHost) {
        return test(sample);
    } else {
        return holds(comparison, sample, threshold);
    }
}

} // namespace gr::blocks::trigger::detail

#endif // GNURADIO_TRIGGER_SAMPLEPREDICATE_HPP
