#include <gnuradio-4.0/algorithm/Histogram.hpp>

#include <boost/ut.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>

#include <gnuradio-4.0/test/DeviceTestHelper.hpp>

using namespace gr::testing;
using Accumulator = gr::algorithm::HistogramAccumulator<double>;

namespace {
constexpr std::size_t kSamples = 16UZ;
constexpr std::size_t kBins    = 4UZ;

/// values in every bin, plus one below the range and one above it, so a kernel that ignored the edges would be caught
constexpr std::array<double, kSamples> kValues{0.1, 0.2, 0.3, 0.45, 0.5, 0.55, 0.7, 0.8, 0.9, 0.95, -0.2, 1.4, 0.05, 0.65, 0.35, 0.99};

/// the same accumulator on host storage, which is what the kernel is held to
[[nodiscard]] Accumulator onHost() {
    static std::array<std::uint64_t, kBins> storage{};
    storage.fill(0U);
    Accumulator histogram{.binMin = 0., .binMax = 1., .bins = std::span<std::uint64_t>{storage}};
    for (const double sample : kValues) {
        histogram.add(sample);
    }
    return histogram;
}
} // namespace

int main() {
    static_assert(std::is_trivially_copyable_v<Accumulator>, "a kernel holds the accumulator by value");

    const Accumulator expected = onHost();

    "a kernel bins into device memory and arrives at the same figures"_domain_test = [&](auto& ctx) {
        boost::ut::expect(boost::ut::eq(expected.entries, std::uint64_t{14})) << "the host reference must itself be right, or the test proves nothing";
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
            expect(device, histogram->entries > 0U, "the kernel counted nothing at all");
            expect(device, histogram->underflow == 1U, "the kernel counted {} underflows", histogram->underflow);
        });

        boost::ut::expect(boost::ut::eq(histogram->entries, expected.entries)) << "a kernel counts the same entries as the host";
        boost::ut::expect(boost::ut::eq(histogram->underflow, expected.underflow));
        boost::ut::expect(boost::ut::eq(histogram->overflow, expected.overflow));
        for (std::size_t bin = 0UZ; bin < kBins; ++bin) {
            boost::ut::expect(boost::ut::eq(bins[bin], expected.bins[bin])) << std::format("bin {} differs between host and kernel", bin);
        }
        // the square root stays on the host: nothing in the accumulation needs it, and the figures follow from m2
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
            expect(device, histogram->entries == 0U, "an accumulator without bins counted {} entries", histogram->entries);
        });

        boost::ut::expect(boost::ut::eq(histogram->entries, std::uint64_t{0})) << "the negative control must stay silent";
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
            expect(device, histogram->entries == 0U, "an inverted range admitted a value");
        });

        boost::ut::expect(boost::ut::eq(histogram->entries, std::uint64_t{0}));
    } | kAllDomains;

    return 0;
}
