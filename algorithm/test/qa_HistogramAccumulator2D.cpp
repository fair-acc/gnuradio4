#include <gnuradio-4.0/algorithm/Histogram2D.hpp>

#include <boost/ut.hpp>

#include <array>
#include <cstddef>
#include <format>
#include <limits>
#include <span>

#include <gnuradio-4.0/test/DeviceTestHelper.hpp>

using namespace gr::testing;
using Accumulator = gr::algorithm::HistogramAccumulator2D;

namespace {
constexpr std::size_t kColumns = 4UZ;
constexpr std::size_t kRows    = 2UZ;
constexpr std::size_t kBins    = kColumns * kRows;
constexpr std::size_t kPairs   = 7UZ;

struct Pair {
    double x;
    double y;
    double weight;
};

constexpr std::array<Pair, kPairs> kPairsIn{{
    {0.1, 0.1, 1.0},
    {0.3, 0.8, 1.0},
    {0.9, 0.6, 1.0},
    {0.6, 0.2, 2.5},
    {-0.1, 0.5, 1.0},
    {0.5, 1.0, 1.0},
    {std::numeric_limits<double>::quiet_NaN(), 0.5, 1.0},
}};

constexpr std::array<double, kBins> kExpectedBins{1.0, 0.0, 2.5, 0.0, 0.0, 1.0, 0.0, 1.0};

constexpr Accumulator kGeometry{.xMin = -1., .xMax = 1., .yMin = 10., .yMax = 20., .nBinsX = 4UZ, .bins = {}};
static_assert(
    [] {
        std::array<double, 20UZ> storage{};
        Accumulator              grid = kGeometry;
        grid.bins                     = std::span<double>{storage};
        return grid.nBinsY() == 5UZ && grid.binWidthX() == 0.5 && grid.binWidthY() == 2.0 && grid.binCentreX(0UZ) == -0.75 && grid.binCentreY(4UZ) == 19.0;
    }(),
    "4 columns over [-1, 1) are 0.5 wide, the first centred at -0.75; 5 rows over [10, 20) are 2 high, the last centred at 19");
} // namespace

int main() {
    static_assert(std::is_trivially_copyable_v<Accumulator>, "a kernel holds the accumulator by value");

    "each pair lands in the bin that holds it, weighted, and a pair outside or NaN is counted apart"_domain_test = [](auto& ctx) {
        Pair*        pairs     = ctx.template alloc<Pair>(kPairs);
        double*      bins      = ctx.template alloc<double>(kBins);
        Accumulator* histogram = ctx.template alloc<Accumulator>(1UZ);
        for (std::size_t i = 0UZ; i < kPairs; ++i) {
            pairs[i] = kPairsIn[i];
        }
        for (std::size_t bin = 0UZ; bin < kBins; ++bin) {
            bins[bin] = 0.0;
        }
        *histogram = Accumulator{.xMin = 0., .xMax = 1., .yMin = 0., .yMax = 1., .nBinsX = kColumns, .bins = std::span<double>{bins, kBins}};

        ctx.launch([pairs, histogram](const DeviceTestHandle& device) {
            for (std::size_t i = 0UZ; i < kPairs; ++i) {
                histogram->add(pairs[i].x, pairs[i].y, pairs[i].weight);
            }
            expect(device, histogram->outsideRange == 3.0, "the kernel counted {} pairs outside", histogram->outsideRange);
        });

        for (std::size_t bin = 0UZ; bin < kBins; ++bin) {
            boost::ut::expect(boost::ut::eq(bins[bin], kExpectedBins[bin])) << std::format("bin {} (row {}, column {})", bin, bin / kColumns, bin % kColumns);
        }
        boost::ut::expect(boost::ut::eq(histogram->entriesInRange, 5.5)) << "three pairs of weight 1 and one of 2.5";
        boost::ut::expect(boost::ut::eq(histogram->outsideRange, 3.0)) << "x below the range, y on its excluded upper edge, and a NaN";
    } | kAllDomains;

    "decay scales every bin and both sums by the same factor"_domain_test = [](auto& ctx) {
        double*      bins      = ctx.template alloc<double>(kBins);
        Accumulator* histogram = ctx.template alloc<Accumulator>(1UZ);
        for (std::size_t bin = 0UZ; bin < kBins; ++bin) {
            bins[bin] = 0.0;
        }
        *histogram = Accumulator{.xMin = 0., .xMax = 1., .yMin = 0., .yMax = 1., .nBinsX = kColumns, .bins = std::span<double>{bins, kBins}};

        ctx.launch([histogram](const DeviceTestHandle&) {
            histogram->add(0.1, 0.1, 4.0);
            histogram->add(0.9, 0.9, 2.0);
            histogram->add(2.0, 0.5, 8.0);
            histogram->decay(0.25);
        });

        boost::ut::expect(boost::ut::eq(bins[0UZ], 1.0)) << "4 x 0.25";
        boost::ut::expect(boost::ut::eq(bins[kBins - 1UZ], 0.5)) << "2 x 0.25";
        boost::ut::expect(boost::ut::eq(histogram->entriesInRange, 1.5)) << "the in-range sum ages with its bins";
        boost::ut::expect(boost::ut::eq(histogram->outsideRange, 2.0)) << "and so does the outside sum";
    } | kAllDomains;

    "an accumulator with no bins or an inverted range counts nothing"_domain_test = [](auto& ctx) {
        double*      bins     = ctx.template alloc<double>(kBins);
        Accumulator* empty    = ctx.template alloc<Accumulator>(1UZ);
        Accumulator* inverted = ctx.template alloc<Accumulator>(1UZ);
        for (std::size_t bin = 0UZ; bin < kBins; ++bin) {
            bins[bin] = 0.0;
        }
        *empty    = Accumulator{.xMin = 0., .xMax = 1., .yMin = 0., .yMax = 1., .nBinsX = kColumns, .bins = {}};
        *inverted = Accumulator{.xMin = 0., .xMax = 1., .yMin = 1., .yMax = 0., .nBinsX = kColumns, .bins = std::span<double>{bins, kBins}};

        ctx.launch([empty, inverted](const DeviceTestHandle& device) {
            empty->add(0.5, 0.5);
            inverted->add(0.5, 0.5);
            expect(device, empty->entriesInRange == 0.0 && inverted->entriesInRange == 0.0, "a map without bins or with an inverted range admitted a pair");
        });

        boost::ut::expect(boost::ut::eq(empty->entriesInRange + empty->outsideRange, 0.0)) << "without storage there is nothing to count against";
        boost::ut::expect(boost::ut::eq(inverted->entriesInRange + inverted->outsideRange, 0.0));
        boost::ut::expect(boost::ut::eq(bins[0UZ] + bins[kBins - 1UZ], 0.0));
    } | kAllDomains;

    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
