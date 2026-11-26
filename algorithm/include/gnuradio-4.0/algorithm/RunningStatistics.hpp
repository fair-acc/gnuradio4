#ifndef GNURADIO_ALGORITHM_RUNNINGSTATISTICS_HPP
#define GNURADIO_ALGORITHM_RUNNINGSTATISTICS_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace gr::algorithm {

/**
 * Count, mean, deviations and extremes of finite values that arrive one at a time, in one pass by Welford's method [1],
 * stable where the values sit far from zero: an interval of 1 s known to 1 ps loses its deviation in `sum(x*x)/n - mean^2`.
 * [1] B. P. Welford, "Note on a method for calculating corrected sums of squares and products", Technometrics 4(3), pp. 419-420, 1962.
 *
 *   RunningStatistics statistics; statistics.add(x); const double spread = statistics.sampleStdDev();
 */
struct RunningStatistics {
    std::uint64_t count    = 0U;
    double        mean     = 0.0;
    double        m2       = 0.0; /// sum of the squared deviations from the running mean
    double        smallest = 0.0; /// meaningless while `count` is 0, as is `largest`
    double        largest  = 0.0;

    constexpr void add(double value) noexcept {
        smallest = count == 0U || value < smallest ? value : smallest;
        largest  = count == 0U || value > largest ? value : largest;
        ++count;
        const double deviation = value - mean;
        mean += deviation / static_cast<double>(count);
        m2 += deviation * (value - mean);
    }

    /** 0 without a value */
    [[nodiscard]] constexpr double populationStdDev() const noexcept { return count > 0U ? std::sqrt(m2 / static_cast<double>(count)) : 0.0; }

    /** 0 below two values */
    [[nodiscard]] constexpr double sampleStdDev() const noexcept { return count > 1U ? std::sqrt(m2 / static_cast<double>(count - 1U)) : 0.0; }
};

static_assert(std::is_trivially_copyable_v<RunningStatistics>, "a kernel accumulates by value");

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_RUNNINGSTATISTICS_HPP
