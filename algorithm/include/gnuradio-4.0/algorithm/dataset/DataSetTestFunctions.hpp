#ifndef DATASETTESTFUNCTIONS_HPP
#define DATASETTESTFUNCTIONS_HPP

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include "DataSetHelper.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gr::dataset::generate {

namespace detail {
template<typename T>
constexpr auto initialize = [](typename gr::meta::fundamental_base_value_type_t<T> value, typename gr::meta::fundamental_base_value_type_t<T> uncertainty = {}) -> T {
    if constexpr (gr::UncertainValueLike<T>) {
        return T{value, uncertainty};
    } else {
        return value;
    }
};
} // namespace detail

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr gr::DataSet<T> triangular(std::string name, std::size_t count, TValue offset = 0, TValue amplitude = 1) {
    assert(count > 2UZ);
    gr::DataSet<T> ds;
    ds.signal_names      = {name};
    ds.signal_quantities = {"Amplitude"};
    ds.signal_units      = {""};
    ds.axis_names        = {"Time"};
    ds.axis_units        = {"s"};
    ds.axis_values.resize(1UZ); // one X-axis
    ds.axis_values[gr::dataset::dim::X].resize(count);
    ds.signal_values.resize(count);
    ds.meta_information.resize(1UZ);
    ds.timing_events.resize(1UZ);
    ds.extents = {1, static_cast<std::int32_t>(count)};

    for (std::size_t i = 0UZ; i < count; ++i) {
        ds.axis_values[0][i] = static_cast<TValue>(i);
    }

    std::size_t midpointLeft = count / 2;
    for (std::size_t i = 0UZ; i < midpointLeft; ++i) {
        TValue factor                   = static_cast<TValue>(i) / static_cast<TValue>(midpointLeft - (count & 1 ? 0UZ : 1UZ));
        ds.signal_values[i]             = detail::initialize<T>(offset + amplitude * factor, amplitude / TValue(10));
        ds.signal_values[count - i - 1] = ds.signal_values[i]; // ensures symmetry
    }
    if (count & 1) { // centre point
        ds.signal_values[midpointLeft] = detail::initialize<T>(offset + amplitude, amplitude / TValue(10));
    }

    ds.signal_ranges.push_back({T(offset), T(amplitude)});
    return ds;
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr gr::DataSet<T> ramp(std::string name, std::size_t count, TValue offset = TValue(0), TValue amplitude = TValue(1)) {
    gr::DataSet<T> ds;
    ds.signal_names      = {name};
    ds.signal_quantities = {"Amplitude"};
    ds.signal_units      = {""};
    ds.axis_names        = {"Time"};
    ds.axis_units        = {"s"};
    ds.axis_values.resize(1);
    ds.axis_values[0].resize(count);
    ds.signal_values.resize(count);
    ds.meta_information.resize(1UZ);
    ds.timing_events.resize(1UZ);
    ds.extents = {1, static_cast<std::int32_t>(count)};

    for (std::size_t i = 0; i < count; i++) {
        ds.axis_values[0][i] = static_cast<T>(i);
        ds.signal_values[i]  = detail::initialize<T>(TValue(offset) + TValue(amplitude) * TValue(i) / TValue(count), TValue(amplitude) / TValue(10));
    }
    ds.signal_ranges.push_back({T(offset), T(amplitude)});
    return ds;
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>, std::convertible_to<TValue> U>
[[nodiscard]] constexpr gr::DataSet<T> gaussFunction(std::string name, std::size_t count, U mean = TValue(0), U sigma = U(3), U offset = U(0), U amplitude = U(1)) {
    if (count <= 2) {
        throw std::invalid_argument("Count must be greater than 2");
    }
    if (sigma <= 0) {
        throw std::invalid_argument("Sigma must be positive");
    }

    gr::DataSet<T> ds;
    ds.signal_names      = {name};
    ds.signal_quantities = {"Amplitude"};
    ds.signal_units      = {""};
    ds.axis_names        = {"Time"};
    ds.axis_units        = {"s"};
    ds.axis_values.resize(1);
    ds.axis_values[0].resize(count);
    ds.signal_values.resize(count);
    ds.meta_information.resize(1UZ);
    ds.timing_events.resize(1UZ);
    ds.extents = {1, static_cast<std::int32_t>(count)};

    auto gauss = [](U x, U mu, U sigma) -> TValue { return std::exp(-std::pow((TValue(x) - TValue(mu)) / TValue(sigma), 2) / U(2)) / (TValue(sigma) * std::sqrt(TValue(2) * std::numbers::pi_v<TValue>)); };

    for (std::size_t i = 0; i < count; ++i) {
        ds.axis_values[0][i] = static_cast<TValue>(i);
        const TValue val     = gauss(static_cast<U>(i), U(mean), U(sigma)) * TValue(amplitude) + TValue(offset);
        ds.signal_values[i]  = detail::initialize<T>(val, TValue(amplitude) / TValue(10));
    }

    auto minmax = std::minmax_element(ds.signal_values.begin(), ds.signal_values.end());
    ds.signal_ranges.push_back(Range<T>{*minmax.first, *minmax.second});

    return ds;
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] gr::DataSet<T> randomStepFunction(std::string name, std::size_t count, std::uint64_t seed = 0U) {
    if (count == 0) {
        throw std::invalid_argument("Count must be greater than 0");
    }

    gr::DataSet<T> ds;
    ds.signal_names      = {name};
    ds.signal_quantities = {"Amplitude"};
    ds.signal_units      = {""};
    ds.axis_names        = {"Time"};
    ds.axis_units        = {"s"};
    ds.axis_values.resize(1);
    ds.axis_values[0].resize(count);
    ds.signal_values.resize(count);
    ds.meta_information.resize(1);
    ds.timing_events.resize(1);
    ds.extents = {1, static_cast<std::int32_t>(count)};

    std::random_device                         rd;
    std::mt19937_64                            rng(seed == 0 ? rd() : seed);
    std::uniform_int_distribution<std::size_t> dist(0, count - 1);
    std::size_t                                step = dist(rng);

    for (std::size_t i = 0; i < count; ++i) {
        ds.axis_values[0][i] = static_cast<TValue>(i);
        TValue val           = static_cast<TValue>(i < step ? 0.0 : 1.0);
        ds.signal_values[i]  = detail::initialize<T>(static_cast<TValue>(val), static_cast<TValue>(1) / static_cast<TValue>(10));
    }

    ds.signal_ranges.push_back(Range<T>{static_cast<T>(0), static_cast<T>(1)});

    return ds;
}

} // namespace gr::dataset::generate

#endif // DATASETTESTFUNCTIONS_HPP
