#ifndef GNURADIO_TRIGGER_SAMPLETEST_HPP
#define GNURADIO_TRIGGER_SAMPLETEST_HPP

#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>

#include <gnuradio-4.0/meta/reflection.hpp>

namespace gr::blocks::trigger {

/** the numeric conditions, no hidden tolerance; inside and outside are the closed range [lower, upper] */
enum class Comparison : std::uint8_t { greater, greaterEqual, less, lessEqual, equal, notEqual, inside, outside };

/** an unknown name is nullopt: the caller refuses it and keeps its previous comparison */
[[nodiscard]] constexpr std::optional<Comparison> parseComparison(std::string_view text) noexcept { return gr::meta::parseEnum<Comparison>(text); }

/** greater* read `lower`, less* read `upper`, equal and notEqual read `lower` */
template<typename T>
[[nodiscard]] constexpr bool holds(Comparison comparison, const T& sample, const T& lower, const T& upper) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        if (sample != sample) { // NaN holds nothing, not even notEqual: invalid data is no pass
            return false;
        }
    }
    switch (comparison) {
    case Comparison::greater: return sample > lower;
    case Comparison::greaterEqual: return sample >= lower;
    case Comparison::less: return sample < upper;
    case Comparison::lessEqual: return sample <= upper;
    case Comparison::equal: return sample == lower;
    case Comparison::notEqual: return sample != lower;
    case Comparison::inside: return lower <= sample && sample <= upper;
    case Comparison::outside: return sample < lower || upper < sample;
    }
    return false;
}

template<typename T>
[[nodiscard]] constexpr bool holds(Comparison comparison, const T& sample, const T& threshold) noexcept {
    return holds(comparison, sample, threshold, threshold);
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_SAMPLETEST_HPP
