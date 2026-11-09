#include <boost/ut.hpp>

#include <gnuradio-4.0/math/RepackBits.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>

namespace {
template<typename T>
std::vector<T> repacked(const std::vector<T>& words, gr::Size_t bitsIn, gr::Size_t bitsOut, std::string_view bitOrder) {
    gr::testing::ProbedChain<gr::blocks::math::RepackBits<T>, T> chain(words, {{"bits_in", bitsIn}, {"bits_out", bitsOut}, {"bit_order", std::string(bitOrder)}});
    return chain.run();
}
} // namespace

const auto repackBitsTests = [] {
    using namespace boost::ut;
    using Byte = std::uint8_t;

    "a byte unpacks to its bits, least significant first"_test                                = [] { expect(repacked<Byte>({0x13}, 8U, 1U, "lsbFirst") == std::vector<Byte>{1, 1, 0, 0, 1, 0, 0, 0}); };
    "a byte unpacks to its bits, most significant first (GR3: Packed to Unpacked)"_test       = [] { expect(repacked<Byte>({0x13}, 8U, 1U, "msbFirst") == std::vector<Byte>{0, 0, 0, 1, 0, 0, 1, 1}); };
    "sixteen bits pack into two bytes, most significant first (GR3: Unpacked to Packed)"_test = [] { expect(repacked<Byte>({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1}, 1U, 8U, "msbFirst") == std::vector<Byte>{0x13, 0x3F}); };
    "sixteen bits pack into one 16-bit word"_test                                             = [] { expect(repacked<std::uint16_t>({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1}, 1U, 16U, "msbFirst") == std::vector<std::uint16_t>{4927}); };
    "four bits per word pack and unpack (GR3: Pack K Bits, Unpack K Bits)"_test               = [] {
        expect(repacked<Byte>({0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}, 1U, 4U, "msbFirst") == std::vector<Byte>{1, 12, 3, 15});
        expect(repacked<Byte>({6}, 4U, 1U, "msbFirst") == std::vector<Byte>{0, 1, 1, 0});
    };
    "an incomplete output word is held back"_test = [] {
        expect(repacked<Byte>({0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0}, 1U, 8U, "msbFirst") == std::vector<Byte>{0x13});
        expect(repacked<Byte>({1}, 1U, 8U, "lsbFirst").empty());
    };
    "three bits regroup into two in chunks of lcm(3, 2) = 6 bits"_test = [] {
        expect(repacked<Byte>({0b101, 0b011}, 3U, 2U, "lsbFirst") == std::vector<Byte>{1, 3, 1});
        expect(repacked<Byte>({0b101, 0b011}, 3U, 2U, "msbFirst") == std::vector<Byte>{2, 2, 3});
    };
    "bytes regroup into 3-bit words, least significant first (GR3: Repack Bits)"_test = [] { expect(repacked<Byte>({25, 26, 27, 28, 29, 30, 31, 32}, 8U, 3U, "lsbFirst") == std::vector<Byte>{1, 3, 0, 5, 1, 6, 6, 0, 4, 3, 4, 6, 1, 4, 7, 0}); };
    "bits above bits_in are ignored"_test                                             = [] { expect(repacked<Byte>({0xF1, 0xE2}, 4U, 4U, "lsbFirst") == std::vector<Byte>{0x1, 0x2}); };
    "a 64-bit word unpacks at the maximum width"_test                                 = [] {
        const std::vector<std::uint64_t> bits = repacked<std::uint64_t>({0x8000'0000'0000'0001ULL}, 64U, 1U, "lsbFirst");
        expect(eq(bits.size(), 64UZ));
        expect(eq(std::ranges::count(bits, 1ULL), 2));
        expect(eq(bits.front(), 1ULL) and eq(bits.back(), 1ULL));
    };
    "a width beyond the word is refused and the previous width kept"_test = [] {
        gr::blocks::math::RepackBits<Byte> block;
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();
        std::ignore = block.settings().set({{"bits_out", gr::Size_t{9U}}});
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(block.bits_out.value, gr::Size_t{8U}));
    };
    "the output rate is the input rate times bits_in / bits_out, at default settings too"_test = [] {
        gr::testing::ProbedChain<gr::blocks::math::RepackBits<Byte>, Byte> unpacked({0x13}, {{"bits_in", gr::Size_t{8U}}, {"bits_out", gr::Size_t{1U}}});
        std::ignore = unpacked.run();
        expect(eq(unpacked.sink->sample_rate, 8.f * gr::testing::kProbeRate));
        gr::testing::ProbedChain<gr::blocks::math::RepackBits<Byte>, Byte> packed({1, 0, 0, 0, 0, 0, 0, 1}, {});
        std::ignore = packed.run();
        expect(eq(packed.sink->sample_rate, gr::testing::kProbeRate / 8.f));
    };
    "a bits_out set by a tag regroups from the tagged sample on"_test = [] {
        gr::testing::ProbedChain<gr::blocks::math::RepackBits<Byte>, Byte> chain({1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1}, {});
        chain.source->_tags = {{8UZ, {{"bits_out", gr::Size_t{4U}}}}};
        expect(chain.run() == std::vector<Byte>{0x81, 0x3, 0xC});
    };
    "default settings pack eight bits, least significant first"_test = [] {
        gr::testing::ProbedChain<gr::blocks::math::RepackBits<Byte>, Byte> chain({1, 0, 0, 0, 0, 0, 0, 1}, {});
        expect(chain.run() == std::vector<Byte>{0x81});
    };
};

int main() {
    using namespace boost::ut;
    "RepackBits"_test = repackBitsTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
