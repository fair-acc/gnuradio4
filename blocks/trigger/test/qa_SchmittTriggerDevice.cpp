#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>

#include <boost/ut.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gnuradio-4.0/test/DeviceTestHelper.hpp>

using namespace gr::testing;

namespace {
/// no look-ahead, so an edge is decided on the sample it happened on -- the form a trigger block can use
using Comparator = gr::trigger::SchmittTrigger<float, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION, 32UZ>;

constexpr std::size_t kSamples = 16UZ;

/// two pulses and a runt, so a kernel that answered a constant would be caught
constexpr std::array<float, kSamples> kWave{0.f, 0.f, 3.f, 3.f, 0.f, 0.f, 0.f, 3.f, 3.f, 3.f, 0.f, 0.f, 0.6f, 0.6f, 0.f, 0.f};

[[nodiscard]] std::vector<std::size_t> onHost() {
    Comparator               comparator{0.5f, 1.f}; // hysteresis 0.5 about a threshold of 1
    std::vector<std::size_t> edges;
    for (std::size_t i = 0UZ; i < kSamples; ++i) {
        if (comparator.processOne(kWave[i]) != gr::trigger::EdgeDetection::NONE) {
            edges.push_back(i);
        }
    }
    return edges;
}
} // namespace

int main() {
    static_assert(std::is_trivially_copyable_v<Comparator>, "a kernel holds the comparator by value");

    const std::vector<std::size_t> expected = onHost();

    "a hysteretic comparator finds the same edges inside a kernel"_domain_test = [&](auto& ctx) {
        boost::ut::expect(!expected.empty()) << "the scenario must produce edges on the host, or it proves nothing";

        float*         samples    = ctx.template alloc<float>(kSamples);
        std::uint32_t* foundAt    = ctx.template alloc<std::uint32_t>(kSamples);
        std::uint32_t* foundCount = ctx.template alloc<std::uint32_t>(1UZ);
        Comparator*    comparator = ctx.template alloc<Comparator>(1UZ);

        for (std::size_t i = 0UZ; i < kSamples; ++i) {
            samples[i] = kWave[i];
            foundAt[i] = 0U;
        }
        *foundCount = 0U;
        *comparator = Comparator{0.5f, 1.f};

        ctx.launch([samples, foundAt, foundCount, comparator](const DeviceTestHandle& device) {
            for (std::size_t i = 0UZ; i < kSamples; ++i) {
                if (comparator->processOne(samples[i]) != gr::trigger::EdgeDetection::NONE) {
                    foundAt[*foundCount] = static_cast<std::uint32_t>(i);
                    ++(*foundCount);
                }
            }
            expect(device, *foundCount > 0U, "the kernel found no edge at all");
        });

        boost::ut::expect(boost::ut::eq(static_cast<std::size_t>(*foundCount), expected.size())) << "the kernel finds as many edges as the host";
        for (std::size_t i = 0UZ; i < std::min(static_cast<std::size_t>(*foundCount), expected.size()); ++i) {
            boost::ut::expect(boost::ut::eq(static_cast<std::size_t>(foundAt[i]), expected[i])) << "and at the same samples";
        }
    } | kAllDomains;

    "a comparator that never crosses its threshold reports nothing"_domain_test = [](auto& ctx) {
        float*         samples    = ctx.template alloc<float>(kSamples);
        std::uint32_t* foundCount = ctx.template alloc<std::uint32_t>(1UZ);
        Comparator*    comparator = ctx.template alloc<Comparator>(1UZ);

        for (std::size_t i = 0UZ; i < kSamples; ++i) {
            samples[i] = 0.1f; // well below the threshold
        }
        *foundCount = 0U;
        *comparator = Comparator{0.5f, 1.f};

        ctx.launch([samples, foundCount, comparator](const DeviceTestHandle& device) {
            for (std::size_t i = 0UZ; i < kSamples; ++i) {
                if (comparator->processOne(samples[i]) != gr::trigger::EdgeDetection::NONE) {
                    ++(*foundCount);
                }
            }
            expect(device, *foundCount == 0U, "a flat signal produced {} edge(s)", *foundCount);
        });

        boost::ut::expect(boost::ut::eq(static_cast<std::size_t>(*foundCount), 0UZ)) << "the negative control must stay silent";
    } | kAllDomains;

    return 0;
}
