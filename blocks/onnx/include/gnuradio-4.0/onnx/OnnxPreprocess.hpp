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
    if (name == "None") {
        return NormaliseMode::None;
    }
    return std::nullopt;
}

template<typename T>
requires std::floating_point<T>
struct OnnxPreprocess {
    [[nodiscard]] std::expected<void, gr::Error> configure(ResampleMode resampleMode, NormaliseMode normaliseMode, std::string_view expr = "", T clipMinVal = T(-5), T clipMaxVal = T(10)) {
        _resampleMode  = resampleMode;
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

    // linear resampling from input size to output size
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
        for (std::size_t i = 0; i < outSize; ++i) {
            T    srcIdx = static_cast<T>(i) * static_cast<T>(inSize - 1) / static_cast<T>(outSize - 1);
            auto lo     = static_cast<std::size_t>(srcIdx);
            auto hi     = std::min(lo + 1, inSize - 1);
            T    frac   = srcIdx - static_cast<T>(lo);
            output[i]   = input[lo] * (T(1) - frac) + input[hi] * frac;
        }
    }

    // dispatch normalisation by mode
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

    // log-MAD normalisation matching Python spectrum_to_normalized()
    //
    // 1. shift so min value -> 1.0
    // 2. log10
    // 3. robust z-score using median and MAD (scaled by 1.4826)
    // 4. clip to [clipMin, clipMax]
    static void normaliseLogMAD(std::span<const T> raw, std::span<T> out, T clipMin = T(-5), T clipMax = T(10)) {
        const std::size_t n = raw.size();
        if (n == 0 || out.size() < n) {
            return;
        }

        T minVal = raw[0];
        for (std::size_t i = 1; i < n; ++i) {
            minVal = std::min(minVal, raw[i]);
        }

        // shift + log10
        for (std::size_t i = 0; i < n; ++i) {
            T shifted = raw[i] - minVal + T(1);
            out[i]    = std::log10(shifted);
        }

        // sanitise non-finite values
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(out[i])) {
                out[i] = T(0);
            }
        }

        // compute median via nth_element on a copy
        std::vector<T> sorted(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n));
        auto           midIt = sorted.begin() + static_cast<std::ptrdiff_t>(n / 2);
        std::nth_element(sorted.begin(), midIt, sorted.end());
        T median = *midIt;
        if (n % 2 == 0) {
            T lower = *std::max_element(sorted.begin(), midIt);
            median  = (lower + median) * T(0.5);
        }

        // compute MAD
        std::vector<T> absdev(n);
        for (std::size_t i = 0; i < n; ++i) {
            absdev[i] = std::abs(out[i] - median);
        }
        auto madIt = absdev.begin() + static_cast<std::ptrdiff_t>(n / 2);
        std::nth_element(absdev.begin(), madIt, absdev.end());
        T mad = *madIt;
        if (n % 2 == 0) {
            T lower = *std::max_element(absdev.begin(), madIt);
            mad     = (lower + mad) * T(0.5);
        }

        // fallback if MAD is near zero
        if (mad < T(1e-10)) {
            T sum = T(0);
            for (std::size_t i = 0; i < n; ++i) {
                sum += out[i];
            }
            T mean = sum / static_cast<T>(n);
            T var  = T(0);
            for (std::size_t i = 0; i < n; ++i) {
                T d = out[i] - mean;
                var += d * d;
            }
            mad = std::sqrt(var / static_cast<T>(n)) + T(1e-10);
        }

        T scale = T(1) / (T(1.4826) * mad + T(1e-10));
        for (std::size_t i = 0; i < n; ++i) {
            T v    = (out[i] - median) * scale;
            out[i] = std::clamp(v, clipMin, clipMax);
        }
    }

    // min-max normalisation to [0, 1]
    static void normaliseMinMax(std::span<const T> raw, std::span<T> out) {
        const std::size_t n = raw.size();
        if (n == 0 || out.size() < n) {
            return;
        }

        T minVal = raw[0];
        T maxVal = raw[0];
        for (std::size_t i = 1; i < n; ++i) {
            minVal = std::min(minVal, raw[i]);
            maxVal = std::max(maxVal, raw[i]);
        }

        T range = maxVal - minVal;
        if (range < T(1e-10)) {
            std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n), T(0));
            return;
        }

        T invRange = T(1) / range;
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = (raw[i] - minVal) * invRange;
        }
    }

    // standard z-score normalisation with clipping
    static void normaliseZScore(std::span<const T> raw, std::span<T> out, T clipMin = T(-5), T clipMax = T(10)) {
        const std::size_t n = raw.size();
        if (n == 0 || out.size() < n) {
            return;
        }

        T sum = T(0);
        for (std::size_t i = 0; i < n; ++i) {
            sum += raw[i];
        }
        T mean = sum / static_cast<T>(n);

        T var = T(0);
        for (std::size_t i = 0; i < n; ++i) {
            T d = raw[i] - mean;
            var += d * d;
        }
        T stddev = std::sqrt(var / static_cast<T>(n));
        if (stddev < T(1e-10)) {
            stddev = T(1e-10);
        }

        T invStd = T(1) / stddev;
        for (std::size_t i = 0; i < n; ++i) {
            T v    = (raw[i] - mean) * invStd;
            out[i] = std::clamp(v, clipMin, clipMax);
        }
    }

private:
    ResampleMode  _resampleMode  = ResampleMode::None;
    NormaliseMode _normaliseMode = NormaliseMode::None;
    T             _clipMin       = T(-5);
    T             _clipMax       = T(10);
    bool          _compiled      = false;
    std::string   _exprString;

    // ExprTk vector_views following the ExpressionBulk rebase pattern
    // N.B. _maxBaseSize limits the maximum chunk size due to ExprTk constraints
    static constexpr std::size_t _maxBaseSize = 1UZ << 16;
    std::array<T, 1UZ>           _arrDummy{T(0)};
    exprtk::vector_view<T>       _vecIn  = exprtk::make_vector_view<T>(_arrDummy.data(), _maxBaseSize);
    exprtk::vector_view<T>       _vecOut = exprtk::make_vector_view<T>(_arrDummy.data(), _maxBaseSize);
    std::vector<T>               _vecInData{};
    std::vector<T>               _vecOutData{};

    // scalar variables exposed to ExprTk
    T _scalarN      = T(0);
    T _scalarMedian = T(0);
    T _scalarMAD    = T(0);
    T _scalarMin    = T(0);
    T _scalarMax    = T(0);
    T _scalarMean   = T(0);
    T _scalarStd    = T(0);

    exprtk::symbol_table<T> _symbolTable{};
    exprtk::expression<T>   _expression{};

    // compute statistics and evaluate the ExprTk expression
    void normaliseExpression(std::span<const T> raw, std::span<T> out) {
        const std::size_t n = raw.size();
        if (n == 0 || out.size() < n) {
            return;
        }
        assert(_compiled && "OnnxPreprocess: ExprTk expression not compiled; call compileExpression() first");

        const std::size_t effectiveN = std::min(n, _maxBaseSize);

        // resize backing buffers if needed and rebase vector views
        if (_vecInData.size() != effectiveN || _vecOutData.size() != effectiveN) {
            _vecInData.resize(effectiveN);
            _vecOutData.resize(effectiveN);

            _vecIn.rebase(_vecInData.data());
            _vecIn.set_size(effectiveN);
            _vecOut.rebase(_vecOutData.data());
            _vecOut.set_size(effectiveN);
        }

        // copy input data
        std::copy_n(raw.begin(), static_cast<std::ptrdiff_t>(effectiveN), _vecInData.begin());
        std::fill(_vecOutData.begin(), _vecOutData.end(), T(0));

        // pre-compute statistics
        _scalarN = static_cast<T>(effectiveN);

        // min, max, sum
        T minV = _vecInData[0];
        T maxV = _vecInData[0];
        T sum  = T(0);
        for (std::size_t i = 0; i < effectiveN; ++i) {
            minV = std::min(minV, _vecInData[i]);
            maxV = std::max(maxV, _vecInData[i]);
            sum += _vecInData[i];
        }
        _scalarMin  = minV;
        _scalarMax  = maxV;
        _scalarMean = sum / static_cast<T>(effectiveN);

        // stddev
        T var = T(0);
        for (std::size_t i = 0; i < effectiveN; ++i) {
            T d = _vecInData[i] - _scalarMean;
            var += d * d;
        }
        _scalarStd = std::sqrt(var / static_cast<T>(effectiveN));

        // median via nth_element on a copy
        std::vector<T> sorted(_vecInData.begin(), _vecInData.begin() + static_cast<std::ptrdiff_t>(effectiveN));
        auto           midIt = sorted.begin() + static_cast<std::ptrdiff_t>(effectiveN / 2);
        std::nth_element(sorted.begin(), midIt, sorted.end());
        _scalarMedian = *midIt;
        if (effectiveN % 2 == 0 && effectiveN > 1) {
            T lower       = *std::max_element(sorted.begin(), midIt);
            _scalarMedian = (lower + _scalarMedian) * T(0.5);
        }

        // MAD (median absolute deviation)
        std::vector<T> absdev(effectiveN);
        for (std::size_t i = 0; i < effectiveN; ++i) {
            absdev[i] = std::abs(_vecInData[i] - _scalarMedian);
        }
        auto madIt = absdev.begin() + static_cast<std::ptrdiff_t>(effectiveN / 2);
        std::nth_element(absdev.begin(), madIt, absdev.end());
        _scalarMAD = *madIt;
        if (effectiveN % 2 == 0 && effectiveN > 1) {
            T lower    = *std::max_element(absdev.begin(), madIt);
            _scalarMAD = (lower + _scalarMAD) * T(0.5);
        }

        // evaluate the expression
        _expression.value();

        // copy results to output
        std::copy_n(_vecOutData.begin(), static_cast<std::ptrdiff_t>(effectiveN), out.begin());
    }
};

} // namespace gr::blocks::onnx

#endif // GR_ONNX_PREPROCESS_HPP
