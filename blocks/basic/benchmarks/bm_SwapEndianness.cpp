#include <CopyGraph.hpp>
#include <benchmark.hpp>

#include <bit>
#include <cstdint>
#include <span>
#include <vector>

#include <gnuradio-4.0/basic/ConverterBlocks.hpp>

namespace {
constexpr gr::Size_t kSamples = 1U << 22U;

template<typename T>
struct ScalarByteswap : gr::Block<ScalarByteswap<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;
    GR_MAKE_REFLECTABLE(ScalarByteswap, in, out);

    [[nodiscard]] constexpr T processOne(T sample) const noexcept { return std::byteswap(sample); }
};
} // namespace

inline const boost::ut::suite<"SwapEndianness"> _swap = [] {
    using namespace benchmark;
    using gr::blocks::type::converter::SwapEndianness;
    "control: Copy<uint32_t>, the same graph"_benchmark.repeat<5>(kSamples)            = [] { gr::testing::runCopyGraph<gr::testing::Copy<std::uint32_t>, std::uint32_t>(kSamples); };
    "SwapEndianness<uint32_t>, SIMD processOne"_benchmark.repeat<5>(kSamples)          = [] { gr::testing::runCopyGraph<SwapEndianness<std::uint32_t>, std::uint32_t>(kSamples); };
    "SwapEndianness<uint16_t>, SIMD processOne"_benchmark.repeat<5>(kSamples)          = [] { gr::testing::runCopyGraph<SwapEndianness<std::uint16_t>, std::uint16_t>(kSamples); };
    "SwapEndianness<uint64_t>, SIMD processOne"_benchmark.repeat<5>(kSamples)          = [] { gr::testing::runCopyGraph<SwapEndianness<std::uint64_t>, std::uint64_t>(kSamples); };
    "control: scalar std::byteswap processOne<uint32_t>"_benchmark.repeat<5>(kSamples) = [] { gr::testing::runCopyGraph<ScalarByteswap<std::uint32_t>, std::uint32_t>(kSamples); };
    "control: scalar std::byteswap processOne<uint64_t>"_benchmark.repeat<5>(kSamples) = [] { gr::testing::runCopyGraph<ScalarByteswap<std::uint64_t>, std::uint64_t>(kSamples); };

    static std::vector<std::uint32_t> words(kSamples, 0x12345678U);
    "control: std::byteswap over a span<uint32_t>, no graph"_benchmark.repeat<5>(kSamples) = [] {
        for (auto& word : words) {
            word = std::byteswap(word);
        }
        force_to_memory(words);
    };
};

int main() {}
