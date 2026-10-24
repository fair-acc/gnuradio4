#ifndef GNURADIO_ALGORITHM_HISTOGRAM_HPP
#define GNURADIO_ALGORITHM_HISTOGRAM_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace gr::algorithm {

/** the one of `nBins` equal bins over [binMin, binMax) that holds `sample`, which must lie inside; the last bin absorbs rounding at the upper edge */
[[nodiscard]] constexpr std::size_t binContaining(double sample, double binMin, double binMax, std::size_t nBins) noexcept {
    const std::size_t bin = static_cast<std::size_t>((sample - binMin) / (binMax - binMin) * static_cast<double>(nBins));
    return bin < nBins ? bin : nBins - 1UZ;
}

/**
 * Bins values and keeps the figures that describe them, in one pass; the bins are a view the caller owns, so the accumulator stays trivially copyable and a kernel can bin into device memory.
 * A NaN fails every comparison and counts as overflow, as in ROOT's TAxis::FindBin.
 * Mean and deviation come from the values by Welford's method, stable where the values sit far from zero (an interval of 1 s known to 1 ps loses its deviation in `sum(x*x)`).
 *
 *   HistogramAccumulator histogram{.binMin = 0., .binMax = 1., .bins = storage};
 *   histogram.add(0.25);
 */
struct HistogramAccumulator {
    double                   binMin = 0.0;
    double                   binMax = 1.0;
    std::span<std::uint64_t> bins{};

    std::uint64_t entriesInRange = 0U;
    std::uint64_t underflow      = 0U;
    std::uint64_t overflow       = 0U;
    double        mean           = 0.0;
    double        m2             = 0.0;
    double        smallest       = 0.0;
    double        largest        = 0.0;

    constexpr void clear() noexcept {
        std::ranges::fill(bins, 0U);
        *this = HistogramAccumulator{.binMin = binMin, .binMax = binMax, .bins = bins};
    }

    constexpr void add(double sample) noexcept {
        if (bins.empty() || !(binMax > binMin)) {
            return;
        }
        if (sample < binMin) {
            ++underflow;
            return;
        }
        if (!(sample < binMax)) {
            ++overflow;
            return;
        }
        bins[binContaining(sample, binMin, binMax, bins.size())] += 1U;

        smallest = entriesInRange == 0U || sample < smallest ? sample : smallest;
        largest  = entriesInRange == 0U || sample > largest ? sample : largest;
        ++entriesInRange;

        const double deviation = sample - mean;
        mean += deviation / static_cast<double>(entriesInRange);
        m2 += deviation * (sample - mean);
    }

    [[nodiscard]] constexpr double stddev() const noexcept { return entriesInRange > 1U ? std::sqrt(m2 / static_cast<double>(entriesInRange - 1U)) : 0.0; }

    [[nodiscard]] constexpr double binWidth() const noexcept { return bins.empty() ? 0.0 : (binMax - binMin) / static_cast<double>(bins.size()); }

    [[nodiscard]] constexpr double binCentre(std::size_t bin) const noexcept { return binMin + (static_cast<double>(bin) + 0.5) * binWidth(); }
};

static_assert(std::is_trivially_copyable_v<HistogramAccumulator>, "a kernel holds the accumulator by value");

/**
 * Bins (x, y) pairs, each with a weight, into a row-major [nBinsY, nBinsX] grid, `x` fastest; the bins are a view the caller owns, as in `HistogramAccumulator`, so a kernel can bin into device memory.
 * Contents are weights rather than counts, so `decay` can age them; a pair outside `[xMin, xMax) x [yMin, yMax)`, or with a NaN coordinate, is added to `outsideRange` instead of an edge bin.
 *
 *   HistogramAccumulator2D histogram{.xMin = 0., .xMax = 1., .yMin = 0., .yMax = 1., .nBinsX = 4UZ, .bins = storage};
 *   histogram.add(0.25, 0.75);
 */
struct HistogramAccumulator2D {
    double            xMin   = 0.0;
    double            xMax   = 1.0;
    double            yMin   = 0.0;
    double            yMax   = 1.0;
    std::size_t       nBinsX = 0UZ;
    std::span<double> bins{};

    /** sums of the weights added, aged as the bins */
    double entriesInRange = 0.0;
    double outsideRange   = 0.0;

    constexpr void clear() noexcept {
        std::ranges::fill(bins, 0.0);
        entriesInRange = 0.0;
        outsideRange   = 0.0;
    }

    constexpr void add(double x, double y, double weight = 1.0) noexcept {
        if (nBinsX == 0UZ || bins.size() < nBinsX || !(xMax > xMin) || !(yMax > yMin)) {
            return;
        }
        if (!(x >= xMin && x < xMax && y >= yMin && y < yMax)) {
            outsideRange += weight;
            return;
        }
        const std::size_t column = binContaining(x, xMin, xMax, nBinsX);
        const std::size_t row    = binContaining(y, yMin, yMax, nBinsY());
        bins[row * nBinsX + column] += weight;
        entriesInRange += weight;
    }

    /** `factor`, e.g. exp(-dt / tau) for exponential forgetting over dt */
    constexpr void decay(double factor) noexcept {
        for (double& bin : bins) {
            bin *= factor;
        }
        entriesInRange *= factor;
        outsideRange *= factor;
    }

    [[nodiscard]] constexpr std::size_t nBinsY() const noexcept { return nBinsX == 0UZ ? 0UZ : bins.size() / nBinsX; }

    [[nodiscard]] constexpr double binWidthX() const noexcept { return nBinsX == 0UZ ? 0.0 : (xMax - xMin) / static_cast<double>(nBinsX); }

    [[nodiscard]] constexpr double binWidthY() const noexcept { return nBinsY() == 0UZ ? 0.0 : (yMax - yMin) / static_cast<double>(nBinsY()); }

    [[nodiscard]] constexpr double binCentreX(std::size_t column) const noexcept { return xMin + (static_cast<double>(column) + 0.5) * binWidthX(); }

    [[nodiscard]] constexpr double binCentreY(std::size_t row) const noexcept { return yMin + (static_cast<double>(row) + 0.5) * binWidthY(); }
};

static_assert(std::is_trivially_copyable_v<HistogramAccumulator2D>, "a kernel holds the accumulator by value");

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_HISTOGRAM_HPP
