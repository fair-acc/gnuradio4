#ifndef GNURADIO_ALGORITHM_STEERED_POWER_HPP
#define GNURADIO_ALGORITHM_STEERED_POWER_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>

#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::algorithm {

/** curves contains pairs * length samples, length >= 2, zero lag at length / 2; lags contains pairs sample offsets, in the same pair order. Out-of-window lags contribute zero. */
template<std::floating_point T>
[[nodiscard]] GR_DEVICE_FN constexpr T computeSteeredPower(const T* curves, const T* lags, std::size_t pairs, std::size_t length) noexcept {
    const T centre = static_cast<T>(length / 2UZ);
    T       power  = T{0};
    for (std::size_t p = 0UZ; p < pairs; ++p) {
        const T where = centre + lags[p];
        if (where >= T{0} && where <= static_cast<T>(length - 1UZ)) {
            const std::size_t below    = std::min(static_cast<std::size_t>(where), length - 2UZ);
            const T           fraction = where - static_cast<T>(below);
            power += curves[p * length + below] * (T{1} - fraction) + curves[p * length + below + 1UZ] * fraction;
        }
    }
    return power;
}

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_STEERED_POWER_HPP
