#ifndef GNURADIO_TRIGGER_SAMPLETEST_HPP
#define GNURADIO_TRIGGER_SAMPLETEST_HPP

#include <cstdint>
#include <string_view>

#include <gnuradio-4.0/meta/reflection.hpp>

namespace gr::blocks::trigger {

enum class Comparison : std::uint8_t { greater, greaterEqual, less, lessEqual, equal, notEqual };

[[nodiscard]] constexpr Comparison parseComparison(std::string_view text) noexcept { return gr::meta::parseEnum<Comparison>(text).value_or(Comparison::greater); }

template<typename T>
[[nodiscard]] constexpr bool holds(Comparison comparison, const T& sample, const T& threshold) noexcept {
    switch (comparison) {
    case Comparison::greater: return sample > threshold;
    case Comparison::greaterEqual: return sample >= threshold;
    case Comparison::less: return sample < threshold;
    case Comparison::lessEqual: return sample <= threshold;
    case Comparison::equal: return sample == threshold;
    case Comparison::notEqual: return sample != threshold;
    }
    return false;
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_SAMPLETEST_HPP
