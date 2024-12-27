#ifndef DATASETMATH_HPP
#define DATASETMATH_HPP

#include <fmt/format.h>

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include "DataSetHelper.hpp"

namespace gr::dataset {

enum class MathOp { ADD = 0, SUBTRACT, MULTIPLY, DIVIDE, SQR, SQRT, LOG10, DB, INV_DB, IDENTITY };

template<typename T>
[[nodiscard]] constexpr bool sameHorizontalBase(const DataSet<T>& ds1, const DataSet<T>& ds2) {
    if (ds1.axisCount() == 0 || ds2.axisCount() == 0) {
        return false;
    }
    const auto& x1 = ds1.axisValues(0);
    const auto& x2 = ds2.axisValues(0);
    if (x1.size() != x2.size()) {
        return false;
    }
    for (std::size_t i = 0; i < x1.size(); i++) {
        if (x1[i] != x2[i]) {
            return false;
        }
    }
    return true;
}

namespace detail {
template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T applyMathOperation(MathOp op, T y1, T y2) {
    switch (op) {
    case MathOp::ADD: return y1 + y2;
    case MathOp::SUBTRACT: return y1 - y2;
    case MathOp::MULTIPLY: return y1 * y2;
    case MathOp::DIVIDE: return (y2 == TValue(0)) ? std::numeric_limits<TValue>::quiet_NaN() : (y1 / y2);
    case MathOp::SQR: return (y1 + y2) * (y1 + y2);
    case MathOp::SQRT: return (y1 + y2) > TValue(0) ? gr::math::sqrt(y1 + y2) : std::numeric_limits<TValue>::quiet_NaN();
    case MathOp::LOG10: return tenLog10((y1 + y2));
    case MathOp::DB: return decibel((y1 + y2));
    case MathOp::INV_DB: return inverseDecibel(y1);
    case MathOp::IDENTITY:
    default: return (y1 + y2);
    }
}
} // namespace detail

/*!
 * \brief mathFunction(DataSet, DataSet, MathOp) => merges or interpolates ds2 onto ds1’s base
 */
template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr DataSet<T> mathFunction(const DataSet<T>& ds1, const DataSet<T>& ds2, MathOp op) {

    // Create new DataSet
    DataSet<T> ret;
    ret.axis_names        = ds1.axis_names;
    ret.axis_units        = ds1.axis_units;
    ret.axis_values       = ds1.axis_values; // or we do an interpolation base
    ret.layout            = ds1.layout;
    ret.extents           = ds1.extents;
    ret.signal_names      = {"mathOp"};
    ret.signal_quantities = {ds1.signal_quantities.empty() ? "quantity" : ds1.signal_quantities[0]};
    ret.signal_units      = {ds1.signal_units.empty() ? "" : ds1.signal_units[0]};
    ret.signal_ranges     = {{T(0.0), T(1.0)}};   // placeholder
    ret.meta_information  = ds1.meta_information; // or merge
    ret.timing_events     = ds1.timing_events;

    std::size_t dataCount = 0U;
    if (!ret.axis_values.empty()) {
        dataCount = ret.axis_values[0].size();
    }
    ret.signal_values.resize(dataCount);

    bool needsInterpolation = !sameHorizontalBase(ds1, ds2);

    for (std::size_t i = 0; i < dataCount; i++) {
        TValue x             = gr::value(getIndexValue(ds1, dim::X, i));
        T      Y1            = getIndexValue(ds1, dim::Y, i);
        T      Y2            = needsInterpolation ? getValue(ds2, dim::Y, x) : getIndexValue(ds2, dim::Y, i);
        T      result        = detail::applyMathOperation<T>(op, Y1, Y2);
        ret.signal_values[i] = static_cast<T>(result);
    }
    return ret;
}

template<typename T, std::convertible_to<T> U, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr DataSet<T> mathFunction(const DataSet<T>& ds, U value, MathOp op) {

    DataSet<T> ret       = ds; // copy
    auto       dataCount = ds.axis_values.empty() ? 0UZ : ds.axis_values[0].size();
    for (std::size_t i = 0; i < dataCount; i++) {
        T Y1     = getIndexValue(ds, dim::Y, i);
        T result = 0;
        switch (op) {
        case MathOp::ADD: result = Y1 + value; break;
        case MathOp::SUBTRACT: result = Y1 - value; break;
        case MathOp::MULTIPLY: result = Y1 * value; break;
        case MathOp::DIVIDE: result = (value == TValue(0)) ? std::numeric_limits<TValue>::quiet_NaN() : (Y1 / value); break;
        case MathOp::SQR: result = (Y1 + value) * (Y1 + value); break;
        case MathOp::SQRT: result = (Y1 + value) > TValue(0) ? gr::math::sqrt(Y1 + value) : std::numeric_limits<TValue>::quiet_NaN(); break;
        case MathOp::LOG10: result = tenLog10(Y1 + value); break;
        case MathOp::DB: result = decibel(Y1 + value); break;
        case MathOp::INV_DB: result = inverseDecibel(Y1); break;
        case MathOp::IDENTITY:
        default: result = Y1; break;
        }
        ret.signal_values[i] = static_cast<T>(result);
    }
    return ret;
}

template<typename T, std::convertible_to<T> U>
[[nodiscard]] constexpr DataSet<T> addFunction(const DataSet<T>& ds, U value) {
    return mathFunction(ds, value, MathOp::ADD);
}
template<typename T>
[[nodiscard]] constexpr DataSet<T> addFunction(const DataSet<T>& ds1, const DataSet<T>& ds2) {
    return mathFunction(ds1, ds2, MathOp::ADD);
}

// WIP complete other convenience and missing math functions

} // namespace gr::dataset

#endif // DATASETMATH_HPP
