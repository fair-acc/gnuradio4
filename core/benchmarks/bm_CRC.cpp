#include <benchmark.hpp>

#include <gnuradio-4.0/CRC.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <span>
#include <vector>

namespace {

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

// volatile sinks — prevent the compiler from eliding the work as dead code.
volatile std::uint64_t g_crcSink = 0U;
volatile std::byte     g_copySink{};

template<std::size_t N>
void registerGroup(std::string_view sizeLabel) {
    constexpr std::size_t nRepeat = 100UZ;
    static const auto     src     = makeBuffer(N);
    static auto           dst     = std::vector<std::byte>(N);
    static const auto     sp      = std::span<const std::byte>(src.data(), src.size());
    using F                       = gr::crc::Flavour;

    // scaling = N (bytes per iteration) so ops/s normalises to bytes/s.
    ::benchmark::benchmark<nRepeat>(std::format("{} copy_only             ", sizeLabel), N) = [] {
        std::memcpy(dst.data(), src.data(), N);
        g_copySink = dst[N - 1UZ];
    };

    ::benchmark::benchmark<nRepeat>(std::format("{} CRC-32  IEEE       simd  ", sizeLabel), N) = [] { g_crcSink = gr::crc::compute<F::CRC32_IEEE>(sp); };
    ::benchmark::benchmark<nRepeat>(std::format("{} CRC-32  IEEE       scalar", sizeLabel), N) = [] { g_crcSink = gr::crc::computeScalar<F::CRC32_IEEE>(sp); };
    ::benchmark::benchmark<nRepeat>(std::format("{} CRC-32C Castagnoli simd  ", sizeLabel), N) = [] { g_crcSink = gr::crc::compute<F::CRC32C_CASTAGNOLI>(sp); };
    ::benchmark::benchmark<nRepeat>(std::format("{} CRC-32C Castagnoli scalar", sizeLabel), N) = [] { g_crcSink = gr::crc::computeScalar<F::CRC32C_CASTAGNOLI>(sp); };
    ::benchmark::benchmark<nRepeat>(std::format("{} CRC-16  MODBUS     simd  ", sizeLabel), N) = [] { g_crcSink = gr::crc::compute<F::CRC16_MODBUS>(sp); };
    ::benchmark::benchmark<nRepeat>(std::format("{} CRC-16  MODBUS     scalar", sizeLabel), N) = [] { g_crcSink = gr::crc::computeScalar<F::CRC16_MODBUS>(sp); };

    ::benchmark::results::add_separator();
}

} // namespace

inline const boost::ut::suite _bm_crc_64B  = [] { registerGroup<64UZ>("    64 B "); };
inline const boost::ut::suite _bm_crc_1KB  = [] { registerGroup<1024UZ>("   1 KiB"); };
inline const boost::ut::suite _bm_crc_64KB = [] { registerGroup<65536UZ>("  64 KiB"); };
inline const boost::ut::suite _bm_crc_1MiB = [] { registerGroup<1048576UZ>("   1 MiB"); };

int main() { return 0; }
