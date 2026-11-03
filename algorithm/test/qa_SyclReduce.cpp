#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gnuradio-4.0/algorithm/Reduce.hpp>
#include <gnuradio-4.0/algorithm/SyclReduce.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

#if GR_DEVICE_HAS_SYCL
const boost::ut::suite<"SyclReduce argMax"> _syclReduce = [] {
    using namespace boost::ut;
    using gr::algorithm::Reduce;
    using gr::algorithm::SyclReduce;

    "the device kernel finds the same maximum as the host"_test = [] {
        // more than one work group, and the peak deliberately in the last group so a per-group reduction that
        // forgets to combine its partials cannot pass
        constexpr std::size_t                 kCount = 4UZ * SyclReduce::kWorkGroupSize + 17UZ;
        std::vector<float>                    values(kCount);
        std::mt19937                          rng{42U};
        std::uniform_real_distribution<float> dist{-1.f, 1.f};
        for (float& v : values) {
            v = dist(rng);
        }
        values[kCount - 3UZ] = 7.5f;

        sycl::queue queue{};
        float*      shared = sycl::malloc_shared<float>(values.size(), queue); // the kernel dereferences this
        expect(shared != nullptr) << "no shared USM available";
        std::copy(values.begin(), values.end(), shared);
        const auto device = SyclReduce::argMax<float>(queue, shared, values.size());
        const auto host   = Reduce::argMaxHost<float>(values);
        expect(eq(device.index, host.index)) << "the device and the host must agree on where the peak is";
        expect(std::abs(device.value - host.value) < 1e-6f) << "and on its value";
        expect(eq(device.index, kCount - 3UZ));
        sycl::free(shared, queue);
    };

    "an empty span reduces to nothing rather than reading memory"_test = [] {
        sycl::queue queue{};
        const auto  best = SyclReduce::argMax<float>(queue, nullptr, 0UZ);
        expect(eq(best.index, 0UZ));
    };

    "a NaN never wins"_test = [] {
        // every comparison against NaN is false, so a reduction that seeds with a NaN or propagates one would
        // report it as the maximum; the real peak must survive
        const std::vector<float> values{1.f, std::numeric_limits<float>::quiet_NaN(), 3.f, std::numeric_limits<float>::quiet_NaN(), 2.f};
        sycl::queue              queue{};
        float*                   shared = sycl::malloc_shared<float>(values.size(), queue);
        expect(shared != nullptr) << "no shared USM available";
        std::copy(values.begin(), values.end(), shared);
        const auto device = SyclReduce::argMax<float>(queue, shared, values.size());
        expect(eq(device.index, 2UZ)) << "the largest real value wins over any NaN";
        expect(std::abs(device.value - 3.f) < 1e-6f);
        sycl::free(shared, queue);
    };
};
#endif

int main() { return boost::ut::cfg<boost::ut::override>.run({.report_errors = true}); }
