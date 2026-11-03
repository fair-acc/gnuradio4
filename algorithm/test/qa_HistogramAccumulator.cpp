#include <gnuradio-4.0/algorithm/Histogram.hpp>

#include <boost/ut.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <span>

#include <gnuradio-4.0/test/DeviceTestHelper.hpp>

using namespace gr::testing;
using Accumulator = gr::algorithm::HistogramAccumulator;

namespace {
constexpr std::size_t kSamples = 16UZ;
constexpr std::size_t kBins    = 4UZ;

constexpr std::array<double, kSamples> kValues{0.1, 0.2, 0.3, 0.45, 0.5, 0.55, 0.7, 0.8, 0.9, 0.95, -0.2, 1.4, 0.05, 0.65, 0.35, 0.99};

[[nodiscard]] Accumulator hostReference() {
    static std::array<std::uint64_t, kBins> storage{};
    storage.fill(0U);
    Accumulator histogram{.binMin = 0., .binMax = 1., .bins = std::span<std::uint64_t>{storage}};
    for (const double sample : kValues) {
        histogram.add(sample);
    }
    return histogram;
}
static_assert(gr::algorithm::binContaining(0.0, 0., 1., 4UZ) == 0UZ, "a quarter of [0, 1) per bin, the lower edge inclusive: first bin");
static_assert(gr::algorithm::binContaining(0.25, 0., 1., 4UZ) == 1UZ, "a quarter of [0, 1) per bin, the lower edge inclusive: second bin starts at 0.25");
static_assert(gr::algorithm::binContaining(0.99, 0., 1., 4UZ) == 3UZ, "a quarter of [0, 1) per bin, the lower edge inclusive: last bin");
static_assert(gr::algorithm::binContaining(0.0, -1., 1., 4UZ) == 2UZ, "0 is the lower edge of the third of four bins over [-1, 1)");
static_assert(gr::algorithm::binContaining(-1.0, -1., 1., 4UZ) == 0UZ, "-1 is the lower edge of the first of four bins over [-1, 1)");

namespace histogram2d {
using Accumulator = gr::algorithm::HistogramAccumulator2D;

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

void run() {
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
}
} // namespace histogram2d
} // namespace

int main() {
    static_assert(std::is_trivially_copyable_v<Accumulator>, "a kernel holds the accumulator by value");

    const Accumulator expected = hostReference();

    "a kernel bins into device memory and arrives at the same figures"_domain_test = [&](auto& ctx) {
        boost::ut::expect(boost::ut::eq(expected.entriesInRange, std::uint64_t{14})) << "the host reference must itself be right, or the test proves nothing";
        boost::ut::expect(boost::ut::eq(expected.underflow, std::uint64_t{1}));
        boost::ut::expect(boost::ut::eq(expected.overflow, std::uint64_t{1}));

        double*        samples   = ctx.template alloc<double>(kSamples);
        std::uint64_t* bins      = ctx.template alloc<std::uint64_t>(kBins);
        Accumulator*   histogram = ctx.template alloc<Accumulator>(1UZ);

        for (std::size_t i = 0UZ; i < kSamples; ++i) {
            samples[i] = kValues[i];
        }
        for (std::size_t bin = 0UZ; bin < kBins; ++bin) {
            bins[bin] = 0U;
        }
        *histogram = Accumulator{.binMin = 0., .binMax = 1., .bins = std::span<std::uint64_t>{bins, kBins}};

        ctx.launch([samples, histogram](const DeviceTestHandle& device) {
            for (std::size_t i = 0UZ; i < kSamples; ++i) {
                histogram->add(samples[i]);
            }
            expect(device, histogram->entriesInRange > 0U, "the kernel counted nothing at all");
            expect(device, histogram->underflow == 1U, "the kernel counted {} underflows", histogram->underflow);
        });

        boost::ut::expect(boost::ut::eq(histogram->entriesInRange, expected.entriesInRange)) << "a kernel counts the same entries as the host";
        boost::ut::expect(boost::ut::eq(histogram->underflow, expected.underflow));
        boost::ut::expect(boost::ut::eq(histogram->overflow, expected.overflow));
        for (std::size_t bin = 0UZ; bin < kBins; ++bin) {
            boost::ut::expect(boost::ut::eq(bins[bin], expected.bins[bin])) << std::format("bin {} differs between host and kernel", bin);
        }
        boost::ut::expect(boost::ut::approx(histogram->mean, expected.mean, 1e-12)) << "and the same mean, by the same one-pass method";
        boost::ut::expect(boost::ut::approx(histogram->stddev(), expected.stddev(), 1e-12));
        boost::ut::expect(boost::ut::approx(histogram->smallest, 0.05, 1e-12)) << "the smallest value inside the range, not the underflow";
        boost::ut::expect(boost::ut::approx(histogram->largest, 0.99, 1e-12)) << "and the largest, not the overflow";
    } | kAllDomains;

    "an accumulator with no bins counts nothing rather than writing where it has no storage"_domain_test = [](auto& ctx) {
        double*      samples   = ctx.template alloc<double>(kSamples);
        Accumulator* histogram = ctx.template alloc<Accumulator>(1UZ);

        for (std::size_t i = 0UZ; i < kSamples; ++i) {
            samples[i] = kValues[i];
        }
        *histogram = Accumulator{.binMin = 0., .binMax = 1., .bins = {}};

        ctx.launch([samples, histogram](const DeviceTestHandle& device) {
            for (std::size_t i = 0UZ; i < kSamples; ++i) {
                histogram->add(samples[i]);
            }
            expect(device, histogram->entriesInRange == 0U, "an accumulator without bins counted {} entries", histogram->entriesInRange);
        });

        boost::ut::expect(boost::ut::eq(histogram->entriesInRange, std::uint64_t{0})) << "the negative control must stay silent";
        boost::ut::expect(boost::ut::eq(histogram->underflow, std::uint64_t{0}));
    } | kAllDomains;

    "an inverted range is refused, in a kernel as on the host"_domain_test = [](auto& ctx) {
        std::uint64_t* bins      = ctx.template alloc<std::uint64_t>(kBins);
        Accumulator*   histogram = ctx.template alloc<Accumulator>(1UZ);
        for (std::size_t bin = 0UZ; bin < kBins; ++bin) {
            bins[bin] = 0U;
        }
        *histogram = Accumulator{.binMin = 1., .binMax = 0., .bins = std::span<std::uint64_t>{bins, kBins}};

        ctx.launch([histogram](const DeviceTestHandle& device) {
            histogram->add(0.5);
            expect(device, histogram->entriesInRange == 0U, "an inverted range admitted a value");
        });

        boost::ut::expect(boost::ut::eq(histogram->entriesInRange, std::uint64_t{0}));
    } | kAllDomains;

    "a NaN is counted as overflow, not binned, and leaves the figures alone"_domain_test = [](auto& ctx) {
        std::uint64_t* bins      = ctx.template alloc<std::uint64_t>(kBins);
        Accumulator*   histogram = ctx.template alloc<Accumulator>(1UZ);
        for (std::size_t bin = 0UZ; bin < kBins; ++bin) {
            bins[bin] = 0U;
        }
        *histogram = Accumulator{.binMin = 0., .binMax = 1., .bins = std::span<std::uint64_t>{bins, kBins}};

        ctx.launch([histogram](const DeviceTestHandle& device) {
            histogram->add(std::numeric_limits<double>::quiet_NaN());
            expect(device, histogram->overflow == 1U, "a NaN was counted {} times as overflow", histogram->overflow);
        });

        boost::ut::expect(boost::ut::eq(histogram->overflow, std::uint64_t{1})) << "a NaN fails both edge comparisons, so the upper-edge test catches it";
        boost::ut::expect(boost::ut::eq(histogram->underflow, std::uint64_t{0}));
        boost::ut::expect(boost::ut::eq(histogram->entriesInRange, std::uint64_t{0}));
        for (std::size_t bin = 0UZ; bin < kBins; ++bin) {
            boost::ut::expect(boost::ut::eq(bins[bin], std::uint64_t{0})) << std::format("a NaN must not land in bin {}", bin);
        }
        boost::ut::expect(boost::ut::eq(histogram->mean, 0.)) << "nor reach the mean";
    } | kAllDomains;

    histogram2d::run();

    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
