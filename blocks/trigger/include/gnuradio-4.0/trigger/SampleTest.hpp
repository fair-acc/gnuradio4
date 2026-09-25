#ifndef GNURADIO_TRIGGER_SAMPLETEST_HPP
#define GNURADIO_TRIGGER_SAMPLETEST_HPP

#include <cstdint>
#include <string_view>

namespace gr::blocks::trigger {

enum class Comparison : std::uint8_t { greater, greater_equal, less, less_equal, equal, not_equal };

[[nodiscard]] constexpr Comparison parseComparison(std::string_view text) noexcept { return gr::meta::parseEnum<Comparison>(text).value_or(Comparison::greater); }

template<typename T>
[[nodiscard]] constexpr bool holds(Comparison comparison, const T& sample, const T& threshold) noexcept {
    switch (comparison) {
    case Comparison::greater: return sample > threshold;
    case Comparison::greater_equal: return sample >= threshold;
    case Comparison::less: return sample < threshold;
    case Comparison::less_equal: return sample <= threshold;
    case Comparison::equal: return sample == threshold;
    case Comparison::not_equal: return sample != threshold;
    }
    return false;
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_SAMPLETEST_HPP
