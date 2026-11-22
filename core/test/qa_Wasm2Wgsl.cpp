// The translator's contract: a kernel on a WebGPU domain computes what the same C++ computes on the host. Each case
// runs one `parallelFor` lambda on every served domain and compares it with the identical expression evaluated here in
// plain C++, over inputs that reach the corners of the lowering (i64 as two words, the atomic heap, br_table, the stack).

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <tuple>

#include <boost/ut.hpp>

#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>

namespace gr::wasm2wgsl_test {
/// weak, so the compiler cannot prove it never throws: a call from a `noexcept` function then goes through
/// Emscripten's `invoke_*` and its `__THREW__` check, the shape the trigger blocks have
[[gnu::weak, gnu::noinline]] std::int32_t mayThrow(std::int32_t x) { return x * 3 + 1; }
[[nodiscard]] std::int32_t                callsMayThrow(std::int32_t x) noexcept { return mayThrow(x) - mayThrow(x / 2); }
} // namespace gr::wasm2wgsl_test

namespace {

constexpr std::array<std::int64_t, 12> kWide{0, 1, -1, 2, -7, 0x7FFFFFFF, 0x80000000, 0x100000001, -0x100000001, std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::min(), 0x123456789ABCDEF};

/// every operation the lowering spells with helpers, folded into one word per pair so a mismatch names its pair
[[nodiscard]] constexpr std::uint64_t wideOps(std::int64_t a, std::int64_t b) noexcept {
    const auto     ua     = static_cast<std::uint64_t>(a);
    const auto     ub     = static_cast<std::uint64_t>(b);
    const unsigned shift  = static_cast<unsigned>(ub & 63U);
    std::uint64_t  folded = ua + ub;
    folded ^= (ua - ub) * 3U;
    folded ^= ua * ub;
    folded ^= ub == 0U ? 0U : ua / ub;
    folded ^= ub == 0U ? 0U : ua % ub;
    folded ^= (b == 0 || (a == std::numeric_limits<std::int64_t>::min() && b == -1)) ? 0U : static_cast<std::uint64_t>(a / b);
    folded ^= ua << shift;
    folded ^= ua >> shift;
    folded ^= static_cast<std::uint64_t>(a >> shift);
    folded ^= std::rotl(ua, static_cast<int>(shift));
    folded ^= static_cast<std::uint64_t>(std::countl_zero(ua)) << 1U;
    folded ^= static_cast<std::uint64_t>(std::popcount(ub)) << 9U;
    folded ^= (a < b ? 1U : 0U) | (ua < ub ? 2U : 0U) | (a == b ? 4U : 0U);
    return folded;
}

/// a switch dense enough to become a br_table, a loop with an early exit, and a local array indexed at run time
[[nodiscard]] constexpr std::int32_t controlFlow(std::int32_t x) noexcept {
    std::array<std::int32_t, 8> local{};
    for (std::int32_t k = 0; k < 8; ++k) {
        local[static_cast<std::size_t>(k)] = x * k - k;
    }
    std::int32_t result = local[static_cast<std::size_t>(x & 7)];
    switch (x % 6) {
    case 0: result += 11; break;
    case 1: result -= 3; break;
    case 2: result *= 5; break;
    case 3: result ^= 0x55; break;
    case 4: result = -result; break;
    default: result += x; break;
    }
    for (std::int32_t k = 0; k < 32; ++k) {
        if ((result >> k) == 0) {
            return result + k;
        }
    }
    return result;
}

[[nodiscard]] std::uint32_t ulpDistance(float a, float b) {
    const auto ordered = [](float f) {
        const auto bits = std::bit_cast<std::int32_t>(f);
        return bits < 0 ? std::numeric_limits<std::int32_t>::min() - bits : bits;
    };
    return static_cast<std::uint32_t>(std::abs(static_cast<std::int64_t>(ordered(a)) - static_cast<std::int64_t>(ordered(b))));
}

} // namespace

int main() {
    using namespace boost::ut;
    using gr::testing::operator""_domain_test;

    "64-bit integers: arithmetic, division, shifts and comparisons match the host"_domain_test = [](auto& ctx) {
        constexpr std::size_t kPairs  = kWide.size() * kWide.size();
        std::uint64_t* const  results = ctx.template alloc<std::uint64_t>(kPairs);
        std::int64_t* const   inputs  = ctx.template alloc<std::int64_t>(kWide.size());
        std::ranges::copy(kWide, inputs);
        const std::uint64_t dispatchedBefore = gr::testing::gpuDispatchesOn(ctx.domain());
        ctx.launchRange(kPairs, [results, inputs](const gr::testing::DeviceTestHandle&, std::size_t i) { results[i] = wideOps(inputs[i / kWide.size()], inputs[i % kWide.size()]); });
        if (std::ranges::contains(gr::testing::kWebGpuDomainNames, ctx.domain())) {
            expect(gt(gr::testing::gpuDispatchesOn(ctx.domain()), dispatchedBefore)) << std::format("{}: the kernel ran on the adapter", ctx.domain());
        }
        for (std::size_t i = 0UZ; i < kPairs; ++i) {
            expect(eq(results[i], wideOps(kWide[i / kWide.size()], kWide[i % kWide.size()]))) << std::format("{}: pair ({}, {})", ctx.domain(), kWide[i / kWide.size()], kWide[i % kWide.size()]);
        }
    } | gr::testing::kAllDomains;

    "byte outputs written by neighbouring work items keep every byte"_domain_test = [](auto& ctx) {
        constexpr std::size_t kN               = 1027UZ; // not a multiple of four: the last word is shared with nothing
        std::uint8_t* const   bytes            = ctx.template alloc<std::uint8_t>(kN);
        const std::uint64_t   dispatchedBefore = gr::testing::gpuDispatchesOn(ctx.domain());
        ctx.launchRange(kN, [bytes](const gr::testing::DeviceTestHandle&, std::size_t i) { bytes[i] = static_cast<std::uint8_t>(~(i * 37U)); });
        if (std::ranges::contains(gr::testing::kWebGpuDomainNames, ctx.domain())) {
            expect(gt(gr::testing::gpuDispatchesOn(ctx.domain()), dispatchedBefore)) << std::format("{}: the kernel ran on the adapter", ctx.domain());
        }
        for (std::size_t i = 0UZ; i < kN; ++i) {
            expect(eq(bytes[i], static_cast<std::uint8_t>(~(i * 37U)))) << std::format("{}: byte {}", ctx.domain(), i);
        }
    } | gr::testing::kAllDomains;

    "32- and 64-bit atomic adds from every work item sum exactly"_domain_test = [](auto& ctx) {
        constexpr std::size_t kN             = 4096UZ;
        std::uint32_t* const  narrow         = ctx.template alloc<std::uint32_t>(1UZ);
        std::uint64_t* const  wide           = ctx.template alloc<std::uint64_t>(1UZ);
        *narrow                              = 0U;
        *wide                                = 0xFFFFFFF0ULL; // the low word carries into the high one
        const std::uint64_t dispatchedBefore = gr::testing::gpuDispatchesOn(ctx.domain());
        ctx.launchRange(kN, [narrow, wide](const gr::testing::DeviceTestHandle&, std::size_t i) {
            std::atomic_ref<std::uint32_t>(*narrow).fetch_add(static_cast<std::uint32_t>(i), std::memory_order_relaxed);
            std::atomic_ref<std::uint64_t>(*wide).fetch_add(1U, std::memory_order_relaxed);
        });
        if (std::ranges::contains(gr::testing::kWebGpuDomainNames, ctx.domain())) {
            expect(gt(gr::testing::gpuDispatchesOn(ctx.domain()), dispatchedBefore)) << std::format("{}: the kernel ran on the adapter", ctx.domain());
        }
        expect(eq(*narrow, static_cast<std::uint32_t>(kN * (kN - 1UZ) / 2UZ))) << ctx.domain();
        expect(eq(*wide, 0xFFFFFFF0ULL + kN)) << ctx.domain();
    } | gr::testing::kAllDomains;

    "a switch, an early loop exit and a local array indexed at run time match the host"_domain_test = [](auto& ctx) {
        constexpr std::size_t kN               = 512UZ;
        std::int32_t* const   results          = ctx.template alloc<std::int32_t>(kN);
        const std::uint64_t   dispatchedBefore = gr::testing::gpuDispatchesOn(ctx.domain());
        ctx.launchRange(kN, [results](const gr::testing::DeviceTestHandle&, std::size_t i) { results[i] = controlFlow(static_cast<std::int32_t>(i) - 256); });
        if (std::ranges::contains(gr::testing::kWebGpuDomainNames, ctx.domain())) {
            expect(gt(gr::testing::gpuDispatchesOn(ctx.domain()), dispatchedBefore)) << std::format("{}: the kernel ran on the adapter", ctx.domain());
        }
        for (std::size_t i = 0UZ; i < kN; ++i) {
            expect(eq(results[i], controlFlow(static_cast<std::int32_t>(i) - 256))) << std::format("{}: x = {}", ctx.domain(), static_cast<std::int32_t>(i) - 256);
        }
    } | gr::testing::kAllDomains;

    "sin and cos stay within 2 ulp of the host up to |x| = 8192"_domain_test = [](auto& ctx) {
        constexpr std::size_t kN               = 4096UZ;
        float* const          values           = ctx.template alloc<float>(3UZ * kN);
        const std::uint64_t   dispatchedBefore = gr::testing::gpuDispatchesOn(ctx.domain());
        ctx.launchRange(kN, [values](const gr::testing::DeviceTestHandle&, std::size_t i) {
            const float x         = -8192.f + 4.f * static_cast<float>(i) + 0.37f;
            values[3UZ * i]       = std::sin(x);
            values[3UZ * i + 1UZ] = std::cos(x);
            values[3UZ * i + 2UZ] = x;
        });
        if (std::ranges::contains(gr::testing::kWebGpuDomainNames, ctx.domain())) {
            expect(gt(gr::testing::gpuDispatchesOn(ctx.domain()), dispatchedBefore)) << std::format("{}: the kernel ran on the adapter", ctx.domain());
        }
        // accuracy is judged on the kernel's own argument: a GPU's float addition may round its x differently from wasm's
        std::uint32_t worst         = 0U;
        std::uint32_t worstArgument = 0U;
        std::string   where;
        for (std::size_t i = 0UZ; i < kN; ++i) {
            const float x = values[3UZ * i + 2UZ];
            worstArgument = std::max(worstArgument, ulpDistance(x, -8192.f + 4.f * static_cast<float>(i) + 0.37f));
            for (const auto& [got, want, name] : {std::tuple{values[3UZ * i], std::sin(x), "sin"}, std::tuple{values[3UZ * i + 1UZ], std::cos(x), "cos"}}) {
                if (const std::uint32_t distance = ulpDistance(got, want); distance > worst) {
                    worst = distance;
                    where = std::format("{}({}) = {} against {}", name, x, got, want);
                }
            }
        }
        expect(le(worstArgument, 1U)) << std::format("{}: the kernel's argument is within 1 ulp of the host's", ctx.domain());
        expect(le(worst, 2U)) << std::format("{}: worst distance {} ulp at {}", ctx.domain(), worst, where);
    } | gr::testing::kAllDomains;

    "a call the compiler must assume may throw runs on the adapter as a direct call"_domain_test = [](auto& ctx) {
        constexpr std::size_t kN               = 512UZ;
        std::int32_t* const   results          = ctx.template alloc<std::int32_t>(kN);
        const std::uint64_t   dispatchedBefore = gr::testing::gpuDispatchesOn(ctx.domain());
        ctx.launchRange(kN, [results](const gr::testing::DeviceTestHandle&, std::size_t i) { results[i] = gr::wasm2wgsl_test::callsMayThrow(static_cast<std::int32_t>(i) - 256); });
        if (std::ranges::contains(gr::testing::kWebGpuDomainNames, ctx.domain())) {
            expect(gt(gr::testing::gpuDispatchesOn(ctx.domain()), dispatchedBefore)) << std::format("{}: the kernel ran on the adapter", ctx.domain());
        }
        for (std::size_t i = 0UZ; i < kN; ++i) {
            const auto x = static_cast<std::int32_t>(i) - 256;
            expect(eq(results[i], (x * 3 + 1) - (x / 2 * 3 + 1))) << std::format("{}: x = {}", ctx.domain(), x);
        }
    } | gr::testing::kAllDomains;

    "hypot stays within 2 ulp of the correctly rounded result, without overflow or underflow"_domain_test = [](auto& ctx) {
        constexpr std::size_t kN               = 4096UZ;
        float* const          values           = ctx.template alloc<float>(3UZ * kN);
        const std::uint64_t   dispatchedBefore = gr::testing::gpuDispatchesOn(ctx.domain());
        ctx.launchRange(kN, [values](const gr::testing::DeviceTestHandle&, std::size_t i) {
            const float scale     = std::ldexp(1.f, static_cast<int>(i % 241UZ) - 120); // 2^-120 .. 2^120
            const float x         = scale * (1.f + static_cast<float>(i % 7UZ) * 0.37f);
            const float y         = i % 5UZ == 0UZ ? x : scale * static_cast<float>(i % 11UZ) * 0.29f; // equal and zero pairs included
            values[3UZ * i]       = std::hypot(x, y);
            values[3UZ * i + 1UZ] = x;
            values[3UZ * i + 2UZ] = y;
        });
        if (std::ranges::contains(gr::testing::kWebGpuDomainNames, ctx.domain())) {
            expect(gt(gr::testing::gpuDispatchesOn(ctx.domain()), dispatchedBefore)) << std::format("{}: the kernel ran on the adapter", ctx.domain());
        }
        std::uint32_t worst = 0U;
        std::string   where;
        for (std::size_t i = 0UZ; i < kN; ++i) {
            const float x    = values[3UZ * i + 1UZ];
            const float y    = values[3UZ * i + 2UZ];
            const float want = static_cast<float>(std::hypot(static_cast<double>(x), static_cast<double>(y)));
            if (const std::uint32_t distance = ulpDistance(values[3UZ * i], want); distance > worst) {
                worst = distance;
                where = std::format("hypot({}, {}) = {} against {}", x, y, values[3UZ * i], want);
            }
        }
        expect(le(worst, 2U)) << std::format("{}: worst distance {} ulp at {}", ctx.domain(), worst, where);
    } | gr::testing::kAllDomains;

    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
