#ifndef GR_ONNX_PREPROCESS_HPP
#define GR_ONNX_PREPROCESS_HPP

#include <gnuradio-4.0/Message.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <expected>
#include <format>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <exprtk.hpp>

namespace gr::blocks::onnx {

enum class ErrorPolicy { Stop, Passthrough };

enum class ResampleMode { None, Linear };

enum class NormaliseMode { None, LogMAD, MinMax, ZScore, Expression };

[[nodiscard]] inline std::optional<NormaliseMode> normaliseModeFromString(std::string_view name) {
    if (name == "LogMAD") {
        return NormaliseMode::LogMAD;
    }
    if (name == "MinMax") {
        return NormaliseMode::MinMax;
    }
    if (name == "ZScore") {
        return NormaliseMode::ZScore;
    }
    if (name == "Expression") {
        return NormaliseMode::Expression;
    }
    if (name == "None") {
        return NormaliseMode::None;
    }
    if (name == "InGraph") {
        return NormaliseMode::None; // the graph normalises internally, so the host must not do it again
    }
    return std::nullopt;
}

// A model may declare how its input must be normalised. An explicit setting always wins, so this
// only fills in what the caller left at the default: otherwise loading a re-exported model would
// silently retarget a caller who had chosen deliberately. Metadata is taken generically because it
// is defined alongside the session, which this header does not depend on.
void adoptDeclaredNormalisation(const auto& metadata, auto& mode, auto& expression) {
    const auto declared = normaliseModeFromString(metadata.normaliseMode);
    if (!declared || mode != NormaliseMode::None) {
        return;
    }
    mode = *declared;
    if (*declared == NormaliseMode::Expression && !metadata.normaliseExpr.empty() && expression.value.empty()) {
        expression = metadata.normaliseExpr;
    }
}

template<typename T>
requires std::floating_point<T>
struct OnnxPreprocess {
    static constexpr std::size_t kMaxBaseSize = 1UZ << 16; // ExprTk vector_view constraint: maximum chunk size

    NormaliseMode _normaliseMode = NormaliseMode::None;
    T             _clipMin       = T(-5);
    T             _clipMax       = T(10);
    bool          _compiled      = false;
    std::string   _exprString;

    std::array<T, 1UZ>     _viewPlaceholder{T(0)};
    exprtk::vector_view<T> _vecIn  = exprtk::make_vector_view<T>(_viewPlaceholder.data(), kMaxBaseSize);
    exprtk::vector_view<T> _vecOut = exprtk::make_vector_view<T>(_viewPlaceholder.data(), kMaxBaseSize);
    std::vector<T>         _vecInData{};
    std::vector<T>         _vecOutData{};

    T _scalarN      = T(0);
    T _scalarMedian = T(0);
    T _scalarMAD    = T(0);
    T _scalarMin    = T(0);
    T _scalarMax    = T(0);
    T _scalarMean   = T(0);
    T _scalarStd    = T(0);

    exprtk::symbol_table<T>                      _symbolTable{};
    exprtk::expression<T>                        _expression{};
    [[nodiscard]] std::expected<void, gr::Error> configure(NormaliseMode normaliseMode, std::string_view expr = "", T clipMinVal = T(-5), T clipMaxVal = T(10)) {
        _normaliseMode = normaliseMode;
        _clipMin       = clipMinVal;
        _clipMax       = clipMaxVal;

        if (_normaliseMode == NormaliseMode::Expression) {
            return compileExpression(expr);
        }
        return {};
    }

    [[nodiscard]] std::expected<void, gr::Error> compileExpression(std::string_view expr) {
        _exprString = std::string(expr);
        _expression = exprtk::expression<T>();
        _symbolTable.clear();
        _compiled = false;

        if (_vecInData.empty()) {
            _vecInData.resize(1UZ);
        }
        if (_vecOutData.empty()) {
            _vecOutData.resize(1UZ);
        }

        _vecIn.rebase(_vecInData.data());
        _vecIn.set_size(_vecInData.size());
        _vecOut.rebase(_vecOutData.data());
        _vecOut.set_size(_vecOutData.size());

        _symbolTable.add_vector("vecIn", _vecIn);
        _symbolTable.add_vector("vecOut", _vecOut);
        _symbolTable.add_variable("n", _scalarN);
        _symbolTable.add_variable("median", _scalarMedian);
        _symbolTable.add_variable("mad", _scalarMAD);
        _symbolTable.add_variable("min_val", _scalarMin);
        _symbolTable.add_variable("max_val", _scalarMax);
        _symbolTable.add_variable("mean_val", _scalarMean);
        _symbolTable.add_variable("std_val", _scalarStd);
        _symbolTable.add_constants();
        _expression.register_symbol_table(_symbolTable);

        if (exprtk::parser<T> parser; !parser.compile(_exprString, _expression)) {
            std::string errMsg;
            for (std::size_t i = 0; i < parser.error_count(); ++i) {
                const auto error = parser.get_error(i);
                errMsg += std::format("ExprTk Parser Error({:2}):  Position: {:2}\nType: [{:14}] Msg: {}; expression:\n{}\n", static_cast<unsigned int>(i), static_cast<unsigned int>(error.token.position), exprtk::parser_error::to_str(error.mode), error.diagnostic, _exprString);
            }
            return std::unexpected(gr::Error{errMsg});
        }
        _compiled = true;
        return {};
    }

    static void resample(std::span<const T> input, std::span<T> output) {
        const std::size_t inSize  = input.size();
        const std::size_t outSize = output.size();
        if (inSize == 0 || outSize == 0) {
            return;
        }
        if (inSize == outSize) {
            std::copy(input.begin(), input.end(), output.begin());
            return;
        }
        if (outSize == 1UZ) {
            output[0] = input[inSize / 2UZ]; // a single output bin has no interval to interpolate across
            return;
        }
        for (std::size_t i = 0; i < outSize; ++i) {
            T    srcIdx = static_cast<T>(i) * static_cast<T>(inSize - 1) / static_cast<T>(outSize - 1);
            auto lo     = static_cast<std::size_t>(srcIdx);
            auto hi     = std::min(lo + 1, inSize - 1);
            T    frac   = srcIdx - static_cast<T>(lo);
            output[i]   = input[lo] * (T(1) - frac) + input[hi] * frac;
        }
    }

    void normalise(std::span<const T> raw, std::span<T> out) {
        switch (_normaliseMode) {
        case NormaliseMode::None:
            if (out.size() >= raw.size()) {
                std::copy(raw.begin(), raw.end(), out.begin());
            }
            break;
        case NormaliseMode::LogMAD: normaliseLogMAD(raw, out, _clipMin, _clipMax); break;
        case NormaliseMode::MinMax: normaliseMinMax(raw, out); break;
        case NormaliseMode::ZScore: normaliseZScore(raw, out, _clipMin, _clipMax); break;
        case NormaliseMode::Expression: normaliseExpression(raw, out); break;
        }
    }

    static void normaliseLogMAD(std::span<const T> raw, std::span<T> out, T clipMin = T(-5), T clipMax = T(10)) {
        const std::size_t n = raw.size();
        if (n == 0 || out.size() < n) {
            return;
        }

        const T minVal = std::ranges::min(raw);
        std::ranges::transform(raw, out.begin(), [minVal](T v) {
            const T logValue = std::log10(v - minVal + T(1));
            return std::isfinite(logValue) ? logValue : T(0);
        });

        std::vector<T> sorted(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n));
        auto           midIt = sorted.begin() + static_cast<std::ptrdiff_t>(n / 2);
        std::nth_element(sorted.begin(), midIt, sorted.end());
        T median = *midIt;
        if (n % 2 == 0) {
            T lower = *std::max_element(sorted.begin(), midIt);
            median  = (lower + median) * T(0.5);
        }

        std::vector<T> absdev(n);
        std::ranges::transform(out.first(n), absdev.begin(), [median](T v) { return std::abs(v - median); });
        auto madIt = absdev.begin() + static_cast<std::ptrdiff_t>(n / 2);
        std::nth_element(absdev.begin(), madIt, absdev.end());
        T mad = *madIt;
        if (n % 2 == 0) {
            T lower = *std::max_element(absdev.begin(), madIt);
            mad     = (lower + mad) * T(0.5);
        }

        if (mad < T(1e-10)) {
            const T mean = std::reduce(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n), T(0)) / static_cast<T>(n);
            const T var  = std::transform_reduce(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n), T(0), std::plus<>{}, [mean](T v) {
                const T d = v - mean;
                return d * d;
            });
            mad          = std::sqrt(var / static_cast<T>(n)) + T(1e-10);
        }

        const T scale = T(1) / (T(1.4826) * mad + T(1e-10));
        std::ranges::transform(out.first(n), out.begin(), [median, scale, clipMin, clipMax](T v) { return std::clamp((v - median) * scale, clipMin, clipMax); });
    }

    static void normaliseMinMax(std::span<const T> raw, std::span<T> out) {
        const std::size_t n = raw.size();
        if (n == 0 || out.size() < n) {
            return;
        }

        const auto [minVal, maxVal] = std::ranges::minmax(raw);

        const T range = maxVal - minVal;
        if (range < T(1e-10)) {
            std::ranges::fill(out.first(n), T(0));
            return;
        }

        const T invRange = T(1) / range;
        std::ranges::transform(raw, out.begin(), [minVal, invRange](T v) { return (v - minVal) * invRange; });
    }

    static void normaliseZScore(std::span<const T> raw, std::span<T> out, T clipMin = T(-5), T clipMax = T(10)) {
        const std::size_t n = raw.size();
        if (n == 0 || out.size() < n) {
            return;
        }

        const T mean = std::reduce(raw.begin(), raw.end(), T(0)) / static_cast<T>(n);
        const T var  = std::transform_reduce(raw.begin(), raw.end(), T(0), std::plus<>{}, [mean](T v) {
            const T d = v - mean;
            return d * d;
        });

        const T invStd = T(1) / std::max(std::sqrt(var / static_cast<T>(n)), T(1e-10));
        std::ranges::transform(raw, out.begin(), [mean, invStd, clipMin, clipMax](T v) { return std::clamp((v - mean) * invStd, clipMin, clipMax); });
    }

private:
    void normaliseExpression(std::span<const T> raw, std::span<T> out) {
        const std::size_t n = raw.size();
        if (n == 0 || out.size() < n) {
            return;
        }
        assert(_compiled && "OnnxPreprocess: ExprTk expression not compiled; call compileExpression() first");

        const std::size_t effectiveN = std::min(n, kMaxBaseSize);

        // ExprTk vector views must be rebased whenever the backing buffers move
        if (_vecInData.size() != effectiveN || _vecOutData.size() != effectiveN) {
            _vecInData.resize(effectiveN);
            _vecOutData.resize(effectiveN);

            _vecIn.rebase(_vecInData.data());
            _vecIn.set_size(effectiveN);
            _vecOut.rebase(_vecOutData.data());
            _vecOut.set_size(effectiveN);
        }

        std::copy_n(raw.begin(), static_cast<std::ptrdiff_t>(effectiveN), _vecInData.begin());
        std::fill(_vecOutData.begin(), _vecOutData.end(), T(0));

        _scalarN = static_cast<T>(effectiveN);

        const auto [minV, maxV] = std::ranges::minmax(_vecInData);
        _scalarMin              = minV;
        _scalarMax              = maxV;
        _scalarMean             = std::reduce(_vecInData.begin(), _vecInData.end(), T(0)) / static_cast<T>(effectiveN);

        const T var = std::transform_reduce(_vecInData.begin(), _vecInData.end(), T(0), std::plus<>{}, [mean = _scalarMean](T v) {
            const T d = v - mean;
            return d * d;
        });
        _scalarStd  = std::sqrt(var / static_cast<T>(effectiveN));

        std::vector<T> sorted(_vecInData.begin(), _vecInData.begin() + static_cast<std::ptrdiff_t>(effectiveN));
        auto           midIt = sorted.begin() + static_cast<std::ptrdiff_t>(effectiveN / 2);
        std::nth_element(sorted.begin(), midIt, sorted.end());
        _scalarMedian = *midIt;
        if (effectiveN % 2 == 0 && effectiveN > 1) {
            T lower       = *std::max_element(sorted.begin(), midIt);
            _scalarMedian = (lower + _scalarMedian) * T(0.5);
        }

        std::vector<T> absdev(effectiveN);
        std::ranges::transform(_vecInData, absdev.begin(), [median = _scalarMedian](T v) { return std::abs(v - median); });
        auto madIt = absdev.begin() + static_cast<std::ptrdiff_t>(effectiveN / 2);
        std::nth_element(absdev.begin(), madIt, absdev.end());
        _scalarMAD = *madIt;
        if (effectiveN % 2 == 0 && effectiveN > 1) {
            T lower    = *std::max_element(absdev.begin(), madIt);
            _scalarMAD = (lower + _scalarMAD) * T(0.5);
        }

        _expression.value();

        std::copy_n(_vecOutData.begin(), static_cast<std::ptrdiff_t>(effectiveN), out.begin());
    }
};

} // namespace gr::blocks::onnx

#endif // GR_ONNX_PREPROCESS_HPP
