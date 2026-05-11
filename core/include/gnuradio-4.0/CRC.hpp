#ifndef GNURADIO_CRC_HPP
#define GNURADIO_CRC_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include <vir/simd.h>

namespace gr::crc {

/// CRC engine; `Flavour` picks polynomial / init / xor-out / reflected:
///   CRC32_IEEE        — IEEE 802.3, gzip, PNG, PKZIP (poly 0xEDB88320, refl)
///   CRC32C_CASTAGNOLI — iSCSI, SCTP, Btrfs           (poly 0x82F63B78, refl)
///   CRC16_MODBUS      — Modbus RTU                   (poly 0xA001,     refl)
/// Engine also supports non-reflected (byte-at-a-time); add a Traits<> specialisation
/// to extend. Without PCLMULQDQ folding (not in std::simd) CRC is dependent-chain
/// limited; SIMD here gains on loads only — see bm_CRC.

namespace stdx = vir::stdx;

enum class Flavour : std::uint8_t { CRC32_IEEE, CRC32C_CASTAGNOLI, CRC16_MODBUS };

template<Flavour F>
struct Traits;

template<>
struct Traits<Flavour::CRC32_IEEE> {
    using Reg                              = std::uint32_t;
    static constexpr Reg  kPoly            = 0xEDB88320U;
    static constexpr Reg  kInit            = 0xFFFFFFFFU;
    static constexpr Reg  kXorOut          = 0xFFFFFFFFU;
    static constexpr bool kReflected       = true;
    static constexpr Reg  kCheck_123456789 = 0xCBF43926U;
};

template<>
struct Traits<Flavour::CRC32C_CASTAGNOLI> {
    using Reg                              = std::uint32_t;
    static constexpr Reg  kPoly            = 0x82F63B78U;
    static constexpr Reg  kInit            = 0xFFFFFFFFU;
    static constexpr Reg  kXorOut          = 0xFFFFFFFFU;
    static constexpr bool kReflected       = true;
    static constexpr Reg  kCheck_123456789 = 0xE3069283U;
};

template<>
struct Traits<Flavour::CRC16_MODBUS> {
    using Reg                              = std::uint16_t;
    static constexpr Reg  kPoly            = 0xA001U;
    static constexpr Reg  kInit            = 0xFFFFU;
    static constexpr Reg  kXorOut          = 0x0000U;
    static constexpr bool kReflected       = true;
    static constexpr Reg  kCheck_123456789 = 0x4B37U;
};

template<Flavour F>
using Reg = typename Traits<F>::Reg;

namespace detail {

// 8 × 256 × sizeof(T) tables per flavour in .rodata (~4–16 KiB).
template<Flavour F>
consteval std::array<std::array<Reg<F>, 256>, 8> computeTables() noexcept {
    using T               = Reg<F>;
    constexpr T    poly   = Traits<F>::kPoly;
    constexpr bool refl   = Traits<F>::kReflected;
    constexpr int  width  = static_cast<int>(sizeof(T) * 8);
    constexpr T    topBit = static_cast<T>(T{1} << (width - 1));

    std::array<std::array<T, 256>, 8> t{};
    if constexpr (refl) {
        for (std::uint32_t b = 0; b < 256; ++b) {
            T c = static_cast<T>(b);
            for (int i = 0; i < 8; ++i) {
                c = static_cast<T>((c >> 1) ^ ((c & 1U) ? poly : T{0}));
            }
            t[0][b] = c;
        }
        for (std::uint32_t b = 0; b < 256; ++b) {
            T c = t[0][b];
            for (std::size_t i = 1; i < 8; ++i) {
                c       = static_cast<T>((c >> 8) ^ t[0][c & 0xFFU]);
                t[i][b] = c;
            }
        }
    } else {
        for (std::uint32_t b = 0; b < 256; ++b) {
            T c = static_cast<T>(static_cast<T>(b) << (width - 8));
            for (int i = 0; i < 8; ++i) {
                c = static_cast<T>((c & topBit) ? static_cast<T>((c << 1) ^ poly) : static_cast<T>(c << 1));
            }
            t[0][b] = c;
        }
        for (std::uint32_t b = 0; b < 256; ++b) {
            T c = t[0][b];
            for (std::size_t i = 1; i < 8; ++i) {
                c       = static_cast<T>(static_cast<T>(c << 8) ^ t[0][(c >> (width - 8)) & 0xFFU]);
                t[i][b] = c;
            }
        }
    }
    return t;
}

template<Flavour F>
inline constexpr auto kTables = computeTables<F>();

template<Flavour F>
[[nodiscard]] constexpr Reg<F> step(Reg<F> crc, std::uint8_t byte) noexcept {
    using T = Reg<F>;
    if constexpr (Traits<F>::kReflected) {
        return static_cast<T>((crc >> 8) ^ kTables<F>[0][(crc ^ byte) & 0xFFU]);
    } else {
        constexpr int width = static_cast<int>(sizeof(T) * 8);
        return static_cast<T>(static_cast<T>(crc << 8) ^ kTables<F>[0][((crc >> (width - 8)) ^ byte) & 0xFFU]);
    }
}

template<Flavour F>
[[nodiscard]] constexpr Reg<F> advance8(Reg<F> crc, std::uint64_t block) noexcept {
    using T = Reg<F>;
    static_assert(Traits<F>::kReflected, "slice-by-8 only implemented for reflected CRCs");

    if constexpr (sizeof(T) == 2) {
        crc = static_cast<T>(crc ^ static_cast<T>(block));
        return static_cast<T>(                     //
            kTables<F>[7][crc & 0xFFU]             //
            ^ kTables<F>[6][(crc >> 8) & 0xFFU]    //
            ^ kTables<F>[5][(block >> 16) & 0xFFU] //
            ^ kTables<F>[4][(block >> 24) & 0xFFU] //
            ^ kTables<F>[3][(block >> 32) & 0xFFU] //
            ^ kTables<F>[2][(block >> 40) & 0xFFU] //
            ^ kTables<F>[1][(block >> 48) & 0xFFU] //
            ^ kTables<F>[0][(block >> 56) & 0xFFU]);
    } else if constexpr (sizeof(T) == 4) {
        crc ^= static_cast<T>(block);
        const std::uint32_t hi = static_cast<std::uint32_t>(block >> 32);
        return kTables<F>[0][(hi >> 24) & 0xFFU]    //
               ^ kTables<F>[1][(hi >> 16) & 0xFFU]  //
               ^ kTables<F>[2][(hi >> 8) & 0xFFU]   //
               ^ kTables<F>[3][hi & 0xFFU]          //
               ^ kTables<F>[4][(crc >> 24) & 0xFFU] //
               ^ kTables<F>[5][(crc >> 16) & 0xFFU] //
               ^ kTables<F>[6][(crc >> 8) & 0xFFU]  //
               ^ kTables<F>[7][crc & 0xFFU];
    } else {
        crc ^= block;
        return kTables<F>[0][(crc >> 56) & 0xFFU]   //
               ^ kTables<F>[1][(crc >> 48) & 0xFFU] //
               ^ kTables<F>[2][(crc >> 40) & 0xFFU] //
               ^ kTables<F>[3][(crc >> 32) & 0xFFU] //
               ^ kTables<F>[4][(crc >> 24) & 0xFFU] //
               ^ kTables<F>[5][(crc >> 16) & 0xFFU] //
               ^ kTables<F>[6][(crc >> 8) & 0xFFU]  //
               ^ kTables<F>[7][crc & 0xFFU];
    }
}

} // namespace detail

template<Flavour F>
[[nodiscard]] constexpr Reg<F> updateScalar(Reg<F> crc, std::span<const std::byte> input) noexcept {
    std::size_t       i = 0;
    const std::size_t n = input.size();

    if constexpr (Traits<F>::kReflected) {
        while (n - i >= 8) {
            std::uint64_t block = 0;
            for (std::size_t k = 0; k < 8; ++k) {
                block |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(input[i + k])) << (8U * k);
            }
            crc = detail::advance8<F>(crc, block);
            i += 8;
        }
    }
    while (i < n) {
        crc = detail::step<F>(crc, std::to_integer<std::uint8_t>(input[i++]));
    }
    return crc;
}

template<Flavour F>
[[nodiscard]] inline Reg<F> updateSimd(Reg<F> crc, std::span<const std::byte> input) noexcept {
    using V                 = stdx::native_simd<std::uint8_t>;
    constexpr std::size_t W = V::size();

    // reinterpret_cast is conformant here: std::uint8_t is a character type and may alias std::byte.
    const auto* p   = reinterpret_cast<const std::uint8_t*>(input.data());
    const auto* end = p + input.size();

    if constexpr (W >= 8 && (W % 8) == 0) {
        alignas(V) std::array<std::uint8_t, W> staging;
        while (end - p >= static_cast<std::ptrdiff_t>(W)) {
            V chunk;
            chunk.copy_from(p, stdx::element_aligned);
            chunk.copy_to(staging.data(), stdx::element_aligned);
            if constexpr (Traits<F>::kReflected) {
                for (std::size_t i = 0; i < W; i += 8) {
                    std::uint64_t block;
                    std::memcpy(&block, staging.data() + i, 8);
                    crc = detail::advance8<F>(crc, block);
                }
            } else {
                for (std::size_t i = 0; i < W; ++i) {
                    crc = detail::step<F>(crc, staging[i]);
                }
            }
            p += W;
        }
    }

    return updateScalar<F>(crc, std::span<const std::byte>(reinterpret_cast<const std::byte*>(p), static_cast<std::size_t>(end - p)));
}

template<Flavour F>
[[nodiscard]] inline Reg<F> compute(std::span<const std::byte> input) noexcept {
    return static_cast<Reg<F>>(updateSimd<F>(Traits<F>::kInit, input) ^ Traits<F>::kXorOut);
}

template<Flavour F>
[[nodiscard]] constexpr Reg<F> computeScalar(std::span<const std::byte> input) noexcept {
    return static_cast<Reg<F>>(updateScalar<F>(Traits<F>::kInit, input) ^ Traits<F>::kXorOut);
}

} // namespace gr::crc

#endif // GNURADIO_CRC_HPP
