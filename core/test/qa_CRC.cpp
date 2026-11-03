#include <boost/ut.hpp>

#include <gnuradio-4.0/CRC.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace {

constexpr std::span<const std::byte> asBytes(std::string_view s) noexcept { return std::span<const std::byte>(reinterpret_cast<const std::byte*>(s.data()), s.size()); }

std::vector<std::byte> makeBuffer(std::size_t n) {
    std::vector<std::byte> v(n);
    std::uint32_t          s = 0x9E3779B9U;
    for (std::size_t i = 0; i < n; ++i) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v[i] = static_cast<std::byte>(s & 0xFFU);
    }
    return v;
}

template<gr::crc::Flavour F>
void exhaustiveScalarVsSimd() {
    using namespace boost::ut;
    for (std::size_t n = 0; n <= 256; ++n) {
        const auto buf = makeBuffer(n);
        const auto sp  = std::span<const std::byte>(buf.data(), buf.size());
        const auto s   = gr::crc::computeScalar<F>(sp);
        const auto v   = gr::crc::compute<F>(sp);
        expect(eq(s, v)) << "flavour " << static_cast<int>(F) << " n = " << n;
    }
}

template<gr::crc::Flavour F>
void largeSizesScalarVsSimd() {
    using namespace boost::ut;
    for (std::size_t n : {std::size_t{1023}, std::size_t{1024}, std::size_t{4095}, std::size_t{4096}, std::size_t{65'535}, std::size_t{65'536}, std::size_t{1'048'576}}) {
        const auto buf = makeBuffer(n);
        const auto sp  = std::span<const std::byte>(buf.data(), buf.size());
        const auto s   = gr::crc::computeScalar<F>(sp);
        const auto v   = gr::crc::compute<F>(sp);
        expect(eq(s, v)) << "flavour " << static_cast<int>(F) << " n = " << n;
    }
}

template<gr::crc::Flavour F>
void streamingAssociative() {
    using namespace boost::ut;
    const auto buf  = makeBuffer(257);
    const auto sp   = std::span<const std::byte>(buf.data(), buf.size());
    const auto full = gr::crc::compute<F>(sp);
    for (std::size_t split : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{9}, std::size_t{31}, std::size_t{32}, std::size_t{128}, std::size_t{256}, std::size_t{257}}) {
        auto crc = gr::crc::Traits<F>::kInit;
        crc      = gr::crc::updateSimd<F>(crc, sp.subspan(0, split));
        crc      = gr::crc::updateSimd<F>(crc, sp.subspan(split));
        crc ^= gr::crc::Traits<F>::kXorOut; // final XOR-out (stays in Reg<F>; no promotion)
        expect(eq(full, crc)) << "flavour " << static_cast<int>(F) << " split = " << split;
    }
}

} // namespace

const boost::ut::suite<"CRC"> _crc_tests = [] {
    using namespace boost::ut;
    using gr::crc::Flavour;

    "canonical check-vectors per flavour — '123456789'"_test = [] {
        const auto in = asBytes("123456789");
        expect(eq(gr::crc::Traits<Flavour::CRC32_IEEE>::kCheck_123456789, gr::crc::compute<Flavour::CRC32_IEEE>(in)));
        expect(eq(gr::crc::Traits<Flavour::CRC32C_CASTAGNOLI>::kCheck_123456789, gr::crc::compute<Flavour::CRC32C_CASTAGNOLI>(in)));
        expect(eq(gr::crc::Traits<Flavour::CRC16_MODBUS>::kCheck_123456789, gr::crc::compute<Flavour::CRC16_MODBUS>(in)));
    };

    "empty input — known result per flavour"_test = [] {
        // for each flavour: empty input → init ^ xorOut
        expect(eq(0x00000000U, gr::crc::compute<Flavour::CRC32_IEEE>({})));
        expect(eq(0x00000000U, gr::crc::compute<Flavour::CRC32C_CASTAGNOLI>({})));
        expect(eq(std::uint16_t{0xFFFFU}, gr::crc::compute<Flavour::CRC16_MODBUS>({})));
    };

    "scalar matches SIMD — exhaustive 0..256 bytes, all flavours"_test = [] {
        exhaustiveScalarVsSimd<Flavour::CRC32_IEEE>();
        exhaustiveScalarVsSimd<Flavour::CRC32C_CASTAGNOLI>();
        exhaustiveScalarVsSimd<Flavour::CRC16_MODBUS>();
    };

    "scalar matches SIMD — large sizes, all flavours"_test = [] {
        largeSizesScalarVsSimd<Flavour::CRC32_IEEE>();
        largeSizesScalarVsSimd<Flavour::CRC32C_CASTAGNOLI>();
        largeSizesScalarVsSimd<Flavour::CRC16_MODBUS>();
    };

    "streaming update is associative, all flavours"_test = [] {
        streamingAssociative<Flavour::CRC32_IEEE>();
        streamingAssociative<Flavour::CRC32C_CASTAGNOLI>();
        streamingAssociative<Flavour::CRC16_MODBUS>();
    };

    "constexpr evaluation at compile time — CRC-32C of '123456789'"_test = [] {
        constexpr std::array<std::byte, 9> bytes{std::byte{'1'}, std::byte{'2'}, std::byte{'3'}, //
            std::byte{'4'}, std::byte{'5'}, std::byte{'6'},                                      //
            std::byte{'7'}, std::byte{'8'}, std::byte{'9'}};
        constexpr auto                     crc = gr::crc::computeScalar<gr::crc::Flavour::CRC32C_CASTAGNOLI>(std::span<const std::byte>(bytes));
        static_assert(crc == 0xE3069283U, "CRC-32C of \"123456789\" must be 0xE3069283");
    };
};

int main() { return 0; }
