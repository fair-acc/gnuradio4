#ifndef GNURADIO_ALGORITHM_HISTOGRAM_HPP
#define GNURADIO_ALGORITHM_HISTOGRAM_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace gr::algorithm {

/**
 * Bins values and keeps the figures that describe them, in one pass over the data.
 *
 * The bins are a view, not a container: whoever uses this owns the storage, which keeps the accumulator trivially
 * copyable so a kernel can hold it by value and bin straight into device memory.
 *
 * The mean and deviation come from the values themselves rather than from the bins, by Welford's method -- one pass,
 * and stable where the values sit far from zero, which a naive sum of squares is not: an interval of 1 s known to 1 ps
 * loses its deviation entirely in `sum(x*x)`.
 *
 * @code
 * std::array<std::uint64_t, 32> storage{};
 * HistogramAccumulator<double> histogram{.binMin = 0., .binMax = 1., .bins = storage};
 * histogram.add(0.25);
 * const double spread = histogram.stddev();
 * @endcode
 */
template<typename T = double>
requires std::is_floating_point_v<T>
struct HistogramAccumulator {
    T                        binMin = T(0);
    T                        binMax = T(1);
    std::span<std::uint64_t> bins{};

    std::uint64_t entries   = 0U; // values inside the range, which is what the figures describe
    std::uint64_t underflow = 0U;
    std::uint64_t overflow  = 0U;
    T             mean      = T(0);
    T             m2        = T(0); // Welford's running sum of squared deviations
    T             smallest  = T(0);
    T             largest   = T(0);

    constexpr void clear() noexcept {
        std::ranges::fill(bins, 0U);
        entries   = 0U;
        underflow = 0U;
        overflow  = 0U;
        mean      = T(0);
        m2        = T(0);
        smallest  = T(0);
        largest   = T(0);
    }

    constexpr void add(T sample) noexcept {
        if (bins.empty() || !(binMax > binMin)) {
            return;
        }
        if (sample < binMin) {
            ++underflow;
            return;
        }
        if (sample >= binMax) {
            ++overflow;
            return;
        }
        const T           position = (sample - binMin) / (binMax - binMin);
        const std::size_t bin      = static_cast<std::size_t>(position * static_cast<T>(bins.size()));
        bins[bin < bins.size() ? bin : bins.size() - 1UZ] += 1U;

        smallest = entries == 0U || sample < smallest ? sample : smallest;
        largest  = entries == 0U || sample > largest ? sample : largest;
        ++entries;

        const T deviation = sample - mean;
        mean += deviation / static_cast<T>(entries);
        m2 += deviation * (sample - mean);
    }

    [[nodiscard]] constexpr T stddev() const noexcept { return entries > 1U ? std::sqrt(m2 / static_cast<T>(entries - 1U)) : T(0); }

    [[nodiscard]] constexpr T binWidth() const noexcept { return bins.empty() ? T(0) : (binMax - binMin) / static_cast<T>(bins.size()); }

    [[nodiscard]] constexpr T binCentre(std::size_t bin) const noexcept { return binMin + (static_cast<T>(bin) + T(0.5)) * binWidth(); }
};

static_assert(std::is_trivially_copyable_v<HistogramAccumulator<double>>, "a kernel holds the accumulator by value");
static_assert(std::is_trivially_copyable_v<HistogramAccumulator<float>>);

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_HISTOGRAM_HPP
