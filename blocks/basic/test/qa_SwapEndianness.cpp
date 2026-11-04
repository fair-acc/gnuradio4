#include <boost/ut.hpp>

#include <algorithm>
#include <bit>
#include <complex>
#include <cstdint>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/basic/ConverterBlocks.hpp>
#include <gnuradio-4.0/device/DeviceRelocatable.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::blocks::type::converter::SwapEndianness;

static_assert(gr::traits::block::can_processOne_simd<SwapEndianness<std::uint32_t>>);
static_assert(gr::traits::block::can_processOne_simd<SwapEndianness<double>>);
static_assert(!gr::traits::block::can_processOne_simd<SwapEndianness<std::complex<float>>>, "a complex sample has no SIMD lane type");
static_assert(gr::device::DeviceRelocatable<SwapEndianness<float>>);

const auto definitions = [] {
    "the bytes of each sample come out in reverse order"_test = [] {
        expect(eq(SwapEndianness<std::uint16_t>().processOne(std::uint16_t{0x1234}), std::uint16_t{0x3412}));
        expect(eq(SwapEndianness<std::int16_t>().processOne(std::int16_t{0x1234}), std::int16_t{0x3412}));
        expect(eq(SwapEndianness<std::uint32_t>().processOne(0x12345678U), 0x78563412U));
        expect(eq(SwapEndianness<std::int64_t>().processOne(std::int64_t{0x0123456789ABCDEF}), std::bit_cast<std::int64_t>(0xEFCDAB8967452301ULL)));
        expect(eq(std::bit_cast<std::uint32_t>(SwapEndianness<float>().processOne(1.f)), 0x0000803FU)) << "1.0f is 0x3F800000";
        expect(eq(std::bit_cast<std::uint64_t>(SwapEndianness<double>().processOne(1.)), 0x000000000000F03FULL)) << "1.0 is 0x3FF0000000000000";
    };

    "a complex sample has its two parts swapped each on their own"_test = [] {
        const std::complex<float> swapped = SwapEndianness<std::complex<float>>().processOne(std::complex<float>(1.f, -2.f));
        expect(eq(std::bit_cast<std::uint32_t>(swapped.real()), 0x0000803FU));
        expect(eq(std::bit_cast<std::uint32_t>(swapped.imag()), 0x000000C0U)) << "-2.0f is 0xC0000000";
    };

    "a complex<double> sample has its two parts swapped each on their own"_test = [] {
        const std::complex<double> swapped = SwapEndianness<std::complex<double>>().processOne(std::complex<double>(1., -2.));
        expect(eq(std::bit_cast<std::uint64_t>(swapped.real()), 0x000000000000F03FULL)) << "1.0 is 0x3FF0000000000000";
        expect(eq(std::bit_cast<std::uint64_t>(swapped.imag()), 0x00000000000000C0ULL)) << "-2.0 is 0xC000000000000000";
    };

    "float, double and int16 vectors swap lane by lane"_test = [] {
        const auto expectLanes = []<typename R>(R base) {
            using V            = vir::stdx::native_simd<R>;
            const V    lanes   = V([&](auto lane) { return static_cast<R>(base * static_cast<R>(lane + 1)); });
            const auto swapped = SwapEndianness<R>().processOne(lanes);
            for (std::size_t lane = 0UZ; lane < V::size(); ++lane) {
                using Bits  = std::conditional_t<sizeof(R) == 2UZ, std::uint16_t, std::conditional_t<sizeof(R) == 4UZ, std::uint32_t, std::uint64_t>>;
                const R out = swapped[lane];
                const R in  = lanes[lane];
                expect(eq(std::bit_cast<Bits>(out), std::byteswap(std::bit_cast<Bits>(in)))) << "lane " << lane;
            }
        };
        expectLanes(1.5f);
        expectLanes(2.25);
        expectLanes(std::int16_t{0x0102});
    };

    "a vector swaps lane by lane"_test = [] {
        using V            = vir::stdx::native_simd<std::uint64_t>;
        const V    lanes   = V([](auto lane) { return 0x0102030405060708ULL * (lane + 1ULL); });
        const auto swapped = SwapEndianness<std::uint64_t>().processOne(lanes);
        for (std::size_t lane = 0UZ; lane < V::size(); ++lane) {
            expect(eq(swapped[lane], std::byteswap(0x0102030405060708ULL * (lane + 1ULL)))) << "lane " << lane;
        }
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "swapping twice is the identity, and once is std::byteswap, on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        std::vector<std::uint32_t> words(1024UZ);
        for (std::size_t i = 0UZ; i < words.size(); ++i) {
            words[i] = static_cast<std::uint32_t>(i) * 0x01000193U + 0x12345678U;
        }
        const std::string placed   = std::string(domain) + "!";
        auto&             source   = fixture.template emplace<gr::testing::ProbeSource<std::uint32_t>>({{"values", words}, {"n_samples_max", static_cast<gr::Size_t>(words.size())}, {"mark_tag", false}});
        auto&             once     = fixture.template emplace<SwapEndianness<std::uint32_t>>({{"gr:compute_domain", placed}});
        auto&             twice    = fixture.template emplace<SwapEndianness<std::uint32_t>>({{"gr:compute_domain", placed}});
        auto&             onceSink = fixture.template emplace<gr::testing::ProbeSink<std::uint32_t>>({{"log_samples", true}});
        auto&             backSink = fixture.template emplace<gr::testing::ProbeSink<std::uint32_t>>({{"log_samples", true}});
        expect(fixture.template connect<"out", "in">(source, once).has_value());
        expect(fixture.template connect<"out", "in">(once, twice).has_value());
        expect(fixture.template connect<"out", "in">(once, onceSink).has_value());
        expect(fixture.template connect<"out", "in">(twice, backSink).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);

        expect(fatal(eq(onceSink._samples.size(), words.size()))) << domain;
        expect(fatal(eq(backSink._samples.size(), words.size()))) << domain;
        const auto byteswapped = words | std::views::transform([](std::uint32_t word) { return std::byteswap(word); });
        expect(std::ranges::equal(onceSink._samples, byteswapped)) << domain;
        expect(std::ranges::equal(backSink._samples, words)) << domain;
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

} // namespace

int main() {
    "SwapEndianness"_test                 = definitions;
    "SwapEndianness on every domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
