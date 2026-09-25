#ifndef GNURADIO_ALGORITHM_HISTOGRAM_2D_HPP
#define GNURADIO_ALGORITHM_HISTOGRAM_2D_HPP

#include <algorithm>
#include <cstddef>
#include <span>
#include <type_traits>

#include <gnuradio-4.0/algorithm/Histogram.hpp>

namespace gr::algorithm {

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

#endif // GNURADIO_ALGORITHM_HISTOGRAM_2D_HPP
