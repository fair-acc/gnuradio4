#ifndef GNURADIO_WIREFORMAT_HPP
#define GNURADIO_WIREFORMAT_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>

namespace gr::wire {

// V1 stores every multi-byte integer in native order and is defined as little-endian canonical
// (see the format doc below). Reject big-endian hosts at compile time rather than silently
// emitting a non-conforming, unreadable byte image.
static_assert(std::endian::native == std::endian::little, "gr::wire V1 requires little-endian hosts");

/**
 * @brief Canonical element wire-format (V1) — compact 8-byte self-describing prefix.
 *
 * Shared by gr::pmt::Value/ValueMap/ValueView (each value-record) and
 * gr::ByteRingBuffer (each ring record): a walker skips and type-dispatches any
 * element from the first 8 bytes alone. Fresh design, hard cutover; full spec
 * and rationale in docs/CORE_WireFormat.md.
 *
 *   [0] u32 size           element → next element; == capacity (skip field)
 *   [4] u8  valueType      kValueTypeEscape (0xFF) ⇒ precise type in typeName
 *   [5] u8  containerType  Scalar|Complex|String|Tensor|Map|…
 *   [6] u8  flags          kFlagReadOnly | kFlagChecksumPresent | kFlagAnnotationsPresent | …
 *   [7] u8  payloadOffset  payload start; == kPrefixBytes when no optional region
 *   [8] …   optional region (typeName iff escape; annotations iff flagged; checksum iff flagged)
 *   [po]…   payload        length type-derived, not stored
 *
 * Little-endian canonical; size rounding/alignment is the container's policy
 * (Value/ValueMap 16 B, ByteRingBuffer kCacheLine), not a prefix property.
 *
 * TL;DR:  for (p = first; p < end; p = wire::nextElement(p))
 *             use(wire::containerType(p), wire::payload(p));
 */

inline constexpr std::uint8_t kValueTypeEscape        = 0xFFU;
inline constexpr std::uint8_t kFlagReadOnly           = 0x01U;
inline constexpr std::uint8_t kFlagChecksumPresent    = 0x02U;
inline constexpr std::uint8_t kFlagAnnotationsPresent = 0x04U; // optional: SI quantity / unit / description (per-Value)

inline constexpr std::size_t kOffSize          = 0UZ;
inline constexpr std::size_t kOffValueType     = 4UZ;
inline constexpr std::size_t kOffContainerType = 5UZ;
inline constexpr std::size_t kOffFlags         = 6UZ;
inline constexpr std::size_t kOffPayloadOffset = 7UZ;
inline constexpr std::size_t kPrefixBytes      = 8UZ; // also the default payloadOffset
static_assert(kPrefixBytes <= 255UZ, "payloadOffset is stored in a u8");

[[nodiscard]] inline std::uint32_t elementSize(const std::byte* h) noexcept {
    std::uint32_t s;
    std::memcpy(&s, h + kOffSize, sizeof(s));
    return s;
}
inline void setElementSize(std::byte* h, std::uint32_t size) noexcept { std::memcpy(h + kOffSize, &size, sizeof(size)); }

[[nodiscard]] constexpr std::uint8_t valueType(const std::byte* h) noexcept { return static_cast<std::uint8_t>(h[kOffValueType]); }
[[nodiscard]] constexpr std::uint8_t containerType(const std::byte* h) noexcept { return static_cast<std::uint8_t>(h[kOffContainerType]); }
[[nodiscard]] constexpr std::uint8_t flags(const std::byte* h) noexcept { return static_cast<std::uint8_t>(h[kOffFlags]); }
[[nodiscard]] constexpr std::uint8_t payloadOffset(const std::byte* h) noexcept { return static_cast<std::uint8_t>(h[kOffPayloadOffset]); }

[[nodiscard]] constexpr const std::byte* payload(const std::byte* h) noexcept { return h + payloadOffset(h); }
[[nodiscard]] constexpr std::byte*       payload(std::byte* h) noexcept { return h + payloadOffset(h); }
[[nodiscard]] inline const std::byte*    nextElement(const std::byte* h) noexcept { return h + elementSize(h); }

// Writes only the prefix (never the payload) into a caller-sized region.
inline void writeHeaderSized(std::byte* h, std::uint32_t size, std::uint8_t vType, std::uint8_t cType, std::uint8_t flagBits = 0U, std::uint8_t payloadOff = static_cast<std::uint8_t>(kPrefixBytes)) noexcept {
    setElementSize(h, size);
    h[kOffValueType]     = static_cast<std::byte>(vType);
    h[kOffContainerType] = static_cast<std::byte>(cType);
    h[kOffFlags]         = static_cast<std::byte>(flagBits);
    h[kOffPayloadOffset] = static_cast<std::byte>(payloadOff);
}

// Annotation sub-block (per-Value optional region). UTF-8 throughout; each field absent via len == 0.
// Encoded as: [u8 quantity_len][q...][u8 unit_len][u...][u16 description_len LE][d...].
// Non-owning: `string_view`s alias the source buffer passed to readAnnotation; copy into an
// owning struct if outliving that buffer.
struct Annotation {
    std::string_view quantity{};
    std::string_view unit{};
    std::string_view description{};
};

[[nodiscard]] constexpr std::size_t annotationByteSize(const Annotation& a) noexcept { //
    return 1UZ + a.quantity.size() + 1UZ + a.unit.size() + 2UZ + a.description.size();
}

/// Serialise `a` into `out`; returns the number of bytes written, or nullopt if `out` is
/// too small or any field exceeds its u8 / u16 length cap (silent length-prefix truncation
/// would otherwise produce a syntactically valid but semantically corrupt encoding).
[[nodiscard]] inline std::optional<std::size_t> writeAnnotation(std::span<std::byte> out, const Annotation& a) noexcept {
    if (a.quantity.size() > 0xFFU || a.unit.size() > 0xFFU || a.description.size() > 0xFFFFU) {
        return std::nullopt;
    }
    if (out.size() < annotationByteSize(a)) {
        return std::nullopt;
    }
    std::byte* p = out.data();
    *p++         = static_cast<std::byte>(static_cast<std::uint8_t>(a.quantity.size()));
    std::memcpy(p, a.quantity.data(), a.quantity.size());
    p += a.quantity.size();
    *p++ = static_cast<std::byte>(static_cast<std::uint8_t>(a.unit.size()));
    std::memcpy(p, a.unit.data(), a.unit.size());
    p += a.unit.size();
    const std::uint16_t dLen = static_cast<std::uint16_t>(a.description.size());
    std::memcpy(p, &dLen, sizeof(dLen));
    p += sizeof(dLen);
    std::memcpy(p, a.description.data(), a.description.size());
    p += a.description.size();
    return static_cast<std::size_t>(p - out.data());
}

struct AnnotationRead {
    Annotation  a{};
    std::size_t bytesRead = 0UZ;
};

[[nodiscard]] inline std::optional<AnnotationRead> readAnnotation(std::span<const std::byte> in) noexcept {
    std::size_t pos = 0UZ;
    if (in.size() < pos + 1UZ) {
        return std::nullopt;
    }
    const std::uint8_t qLen = static_cast<std::uint8_t>(in[pos]);
    pos += 1UZ;
    if (in.size() < pos + qLen) {
        return std::nullopt;
    }
    const std::string_view quantity(reinterpret_cast<const char*>(in.data() + pos), qLen);
    pos += qLen;
    if (in.size() < pos + 1UZ) {
        return std::nullopt;
    }
    const std::uint8_t uLen = static_cast<std::uint8_t>(in[pos]);
    pos += 1UZ;
    if (in.size() < pos + uLen) {
        return std::nullopt;
    }
    const std::string_view unit(reinterpret_cast<const char*>(in.data() + pos), uLen);
    pos += uLen;
    if (in.size() < pos + sizeof(std::uint16_t)) {
        return std::nullopt;
    }
    std::uint16_t dLen;
    std::memcpy(&dLen, in.data() + pos, sizeof(dLen));
    pos += sizeof(dLen);
    if (in.size() < pos + dLen) {
        return std::nullopt;
    }
    const std::string_view description(reinterpret_cast<const char*>(in.data() + pos), dLen);
    pos += dLen;
    return AnnotationRead{.a = Annotation{.quantity = quantity, .unit = unit, .description = description}, .bytesRead = pos};
}

} // namespace gr::wire

#endif // GNURADIO_WIREFORMAT_HPP
