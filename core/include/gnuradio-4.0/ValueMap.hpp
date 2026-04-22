#ifndef GNURADIO_VALUEMAP_HPP
#define GNURADIO_VALUEMAP_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <iterator>
#include <map>
#include <memory_resource>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

/**
 * @brief Contiguous packed PMR-backed map of keys to Value payloads.
 *
 * Replaces gr::pmt::Value::Map (a std::pmr::unordered_map) with a single
 * contiguous allocation holding a fixed-size header, a packed array of
 * PackedEntry records, and a variable-size payload pool for strings (and,
 * later, tensors and nested ValueMaps). Keys are either a canonical uint16
 * ID (registered in namespace gr::pmt::keys below) or an inline 28-byte
 * string (27 chars + length byte) stored directly in the entry. Lookup is
 * linear — cache-friendly at N <= 30, which covers all observed Tag payloads.
 *
 * The in-memory layout IS the wire / USM format: blob() hands out a
 * pointer-free, offset-addressed std::span<const std::byte> that can be
 * memcpy'd to disk or to SYCL USM-shared memory. Device kernels may read any
 * field and atomically append entries if slackBytes > 0 was reserved at
 * construction. Little-endian hosts only.
 *
 * Iterator, pointer, and view invalidation follows std::vector, NOT
 * std::map: any mutating call (emplace, insert_or_assign, erase, clear,
 * reserve, shrink_to_fit) may invalidate all outstanding iterators. Call
 * reserve(entries, payload_bytes) up front to keep iterators stable during
 * a known-size batch of inserts.
 *
 * TL;DR:
 *   ValueMap map(&my_arena);
 *   map.insert_or_assign("sample_rate", 1'000'000.0f); // string key; canonical names get a 16-bit id internally
 *   map.insert_or_assign("my_extension", 42);          // any string key works
 *   if (map.contains("sample_rate")) {
 *       float hz = map.at("sample_rate").value_or<float>(0.f); // by-value Value, then typed unwrap
 *   }
 *   Value anyVal = map.at("missing_key");              // Monostate iff missing; never throws
 *   auto  blob   = map.blob();                         // 0-copy wire / USM view
 *
 * Canonical-name registry (`gr::pmt::keys`) is internal — used by `dispatchValueType<F>`
 * and the wire-format / serialisation paths to compress the 20 well-known Tag keys to
 * 16-bit IDs in the on-the-wire representation. Not exposed in the user-facing API.
 */

/**
 * @brief Canonical key registry for gr::pmt::ValueMap.
 *
 * Maps compile-time string names (taken verbatim from Tag.hpp's DefaultTag
 * declarations) to stable 16-bit IDs and the Value::ValueType they bind to.
 * Lookup is constexpr in both directions; an unknown key returns the sentinel
 * ID 0 (bound type: Monostate). The registry is append-only — IDs never
 * change, entries are only added.
 *
 * ID allocation ranges:
 *   0x0000          reserved (sentinel: "no entry")
 *   0x0001..0x007F  GR4 core keys (this header)
 *   0x0080..0x00FF  SigMF core extensions (reserved for future import)
 *   0x0100..0x7FFE  user extensions
 *   0x7FFF          reserved
 *   0x8000          inline-key flag (not an ID; used in PackedEntry.keyId)
 *   0x8001..0xFFFE  reserved
 *   0xFFFF          end-marker sentinel
 */
namespace gr::pmt::keys {

struct CanonicalKey {
    std::uint16_t    id;
    std::string_view name;
    Value::ValueType boundType;
    std::string_view unit;
};

// Sentinels — see range table above.
inline constexpr std::uint16_t kIdUnknown    = 0x0000;
inline constexpr std::uint16_t kInlineKeyId  = 0x8000;
inline constexpr std::uint16_t kSpilledKeyId = 0x8001; // key bytes spilled to payload pool (>27 chars)
inline constexpr std::uint16_t kEndMarkerId  = 0xFFFF;

// Authoritative canonical key list. Names + C++ types pinned from
// Tag.hpp:196-216 (DefaultTag<Name, Type, Unit, Description> declarations).
// Value::ValueType is the YaS-inspired tag used in PackedEntry.valueType.
inline constexpr std::array<CanonicalKey, 20> kCanonical = {{
    {0x0001, "sample_rate", Value::ValueType::Float32, "Hz"},
    {0x0002, "signal_name", Value::ValueType::String, ""},
    {0x0003, "num_channels", Value::ValueType::UInt32, ""},
    {0x0004, "signal_quantity", Value::ValueType::String, ""},
    {0x0005, "signal_unit", Value::ValueType::String, ""},
    {0x0006, "signal_min", Value::ValueType::Float32, "a.u."},
    {0x0007, "signal_max", Value::ValueType::Float32, "a.u."},
    {0x0008, "n_dropped_samples", Value::ValueType::UInt32, ""},
    {0x0009, "frequency", Value::ValueType::Float64, "Hz"},
    {0x000A, "rx_overflow", Value::ValueType::Bool, ""},
    {0x000B, "trigger_name", Value::ValueType::String, ""},
    {0x000C, "trigger_time", Value::ValueType::UInt64, "ns"},
    {0x000D, "trigger_offset", Value::ValueType::Float32, "s"},
    {0x000E, "trigger_meta_info", Value::ValueType::Value, ""},
    {0x000F, "local_time", Value::ValueType::UInt64, "ns"},
    {0x0010, "context", Value::ValueType::String, ""},
    {0x0011, "ctx_time", Value::ValueType::UInt64, ""},
    {0x0012, "reset_default", Value::ValueType::Bool, ""},
    {0x0013, "store_default", Value::ValueType::Bool, ""},
    {0x0014, "end_of_stream", Value::ValueType::Bool, ""},
}};

// Name → ID (compile-time). Returns kIdUnknown when the name is not in kCanonical.
template<gr::meta::fixed_string Name>
inline constexpr std::uint16_t idOf = []() consteval {
    const auto it = std::ranges::find(kCanonical, std::string_view{static_cast<const char*>(Name._data)}, &CanonicalKey::name);
    return it == kCanonical.end() ? kIdUnknown : it->id;
}();

// Name → ID (runtime mirror of idOf<>).
[[nodiscard]] inline constexpr std::uint16_t lookupId(std::string_view name) noexcept {
    const auto it = std::ranges::find(kCanonical, name, &CanonicalKey::name);
    return it == kCanonical.end() ? kIdUnknown : it->id;
}

// ID → bound Value::ValueType. Returns Monostate when the id is not registered.
template<std::uint16_t Id>
inline constexpr Value::ValueType boundTypeOf = []() consteval {
    const auto it = std::ranges::find(kCanonical, Id, &CanonicalKey::id);
    return it == kCanonical.end() ? Value::ValueType::Monostate : it->boundType;
}();

// ID → unit string_view. Empty when the id is not registered.
template<std::uint16_t Id>
inline constexpr std::string_view unitOf = []() consteval {
    const auto it = std::ranges::find(kCanonical, Id, &CanonicalKey::id);
    return it == kCanonical.end() ? std::string_view{} : it->unit;
}();

namespace detail {
// ValueType → C++ binding (single source of truth for the wire-format ↔ C++ map).
// Returns std::type_identity<void> for Monostate / Value (nested) / Tensor — those
// are not directly storable as inline scalars and go through the untyped Value API.
template<Value::ValueType VT>
consteval auto valueTypeToCpp() {
    if constexpr (VT == Value::ValueType::Bool) {
        return std::type_identity<bool>{};
    } else if constexpr (VT == Value::ValueType::Int8) {
        return std::type_identity<std::int8_t>{};
    } else if constexpr (VT == Value::ValueType::Int16) {
        return std::type_identity<std::int16_t>{};
    } else if constexpr (VT == Value::ValueType::Int32) {
        return std::type_identity<std::int32_t>{};
    } else if constexpr (VT == Value::ValueType::Int64) {
        return std::type_identity<std::int64_t>{};
    } else if constexpr (VT == Value::ValueType::UInt8) {
        return std::type_identity<std::uint8_t>{};
    } else if constexpr (VT == Value::ValueType::UInt16) {
        return std::type_identity<std::uint16_t>{};
    } else if constexpr (VT == Value::ValueType::UInt32) {
        return std::type_identity<std::uint32_t>{};
    } else if constexpr (VT == Value::ValueType::UInt64) {
        return std::type_identity<std::uint64_t>{};
    } else if constexpr (VT == Value::ValueType::Float32) {
        return std::type_identity<float>{};
    } else if constexpr (VT == Value::ValueType::Float64) {
        return std::type_identity<double>{};
    } else if constexpr (VT == Value::ValueType::ComplexFloat32) {
        return std::type_identity<std::complex<float>>{};
    } else if constexpr (VT == Value::ValueType::ComplexFloat64) {
        return std::type_identity<std::complex<double>>{};
    } else if constexpr (VT == Value::ValueType::String) {
        return std::type_identity<std::string_view>{};
    } else {
        return std::type_identity<void>{};
    }
}
} // namespace detail

// ID → mapped C++ scalar / string-view type. void for nested
// (boundTypeOf<Id> == Value::ValueType::Value / Monostate) — those go
// through the untyped Value API.
template<std::uint16_t Id>
using CanonicalCppType = typename decltype(detail::valueTypeToCpp<boundTypeOf<Id>>())::type;

// `Name` is a registered canonical key in kCanonical. Used to constrain typed-name
// access (`at<"name">`, etc.) so a typo at a call site is a compile error.
template<gr::meta::fixed_string Name>
concept CanonicalName = (idOf<Name> != kIdUnknown);

// Compile-time integrity checks: IDs unique, names unique, all IDs in the
// reserved core range. Evaluated once at translation-unit load.
namespace detail {
consteval bool canonicalIdsUnique() {
    for (std::size_t i = 0; i < kCanonical.size(); ++i) {
        for (std::size_t j = i + 1; j < kCanonical.size(); ++j) {
            if (kCanonical[i].id == kCanonical[j].id) {
                return false;
            }
        }
    }
    return true;
}
consteval bool canonicalNamesUnique() {
    for (std::size_t i = 0; i < kCanonical.size(); ++i) {
        for (std::size_t j = i + 1; j < kCanonical.size(); ++j) {
            if (kCanonical[i].name == kCanonical[j].name) {
                return false;
            }
        }
    }
    return true;
}
consteval bool canonicalIdsInCoreRange() {
    for (const auto& k : kCanonical) {
        if (k.id == kIdUnknown || k.id > 0x007F) {
            return false;
        }
    }
    return true;
}
} // namespace detail

static_assert(detail::canonicalIdsUnique(), "kCanonical has duplicate IDs");
static_assert(detail::canonicalNamesUnique(), "kCanonical has duplicate names");
static_assert(detail::canonicalIdsInCoreRange(), "kCanonical IDs must lie in 0x0001..0x007F");

} // namespace gr::pmt::keys

namespace gr::pmt {

inline constexpr std::array<char, 4> kBlobMagic          = {'G', 'R', '4', 'M'};
inline constexpr std::uint8_t        kBlobVersion        = 1U;
inline constexpr std::size_t         kBlobAlignment      = 16UZ; // USM / SYCL requirement
inline constexpr std::size_t         kMaxInlineKeyLength = 27UZ; // 27 chars + length byte

inline constexpr std::uint8_t kEntryFlagInlineScalar = 0x01; // scalar in inlineValue
inline constexpr std::uint8_t kEntryFlagOffsetLength = 0x02; // payloadOffset/length valid
inline constexpr std::uint8_t kEntryFlagNestedMap    = 0x08; // payload is a nested ValueMap blob (recursive); only meaningful when valueType == Value::ValueType::Value
inline constexpr std::uint8_t kEntryFlagTensor       = 0x10; // payload is a tensor sub-blob (see Tensor wire-format below); only meaningful when valueType == Value::ValueType::Value

inline constexpr std::uint8_t kHeaderFlagOverflow = 0x01; // device append hit slackBytes limit
inline constexpr std::uint8_t kHeaderFlagFrozen   = 0x02; // advisory: further mutation disallowed
// kMaxTensorRank / kMaxTensorElements / kTensorBlobHeaderSize / kTensorEncodingVariableSize moved
// to Value.hpp post Q1 (Value owns its own tensor byte-blob format).

struct alignas(16) Header {
    char          magic[4];      // {'G','R','4','M'}
    std::uint8_t  version;       // kBlobVersion
    std::uint8_t  flags;         // Header.flags bits
    std::uint16_t entryCount;    // atomic_ref target on device-append path
    std::uint32_t totalSize;     // bytes, incl. this header + payload
    std::uint32_t payloadOffset; // byte offset of payload pool start
    std::uint32_t payloadUsed;   // atomic_ref target on device-append path
    std::uint32_t slackBytes;    // device-side append budget (0 = disabled)
    std::uint64_t reserved;      // round to 32 bytes; future per-blob flags/version extension
};
static_assert(sizeof(Header) == 32UZ);
static_assert(alignof(Header) == 16UZ);

// 48-byte budget: 64-bit inlineValue (large enough for f64 / u64 and a
// pointer/offset pair) + 28-byte inlineKey (27 chars + length byte). The
// 27-char ceiling fits all 20 Tag.hpp canonical names and the longest known
// SigMF extension key ("ntia-core:classification", 24 chars). Longer keys
// overflow into the payload pool (same path as tensor / nested ValueMap).
//
// inlineKey encoding when keyId == kInlineKeyId:
//   inlineKey[0]      = length byte (0..27)
//   inlineKey[1..len] = character data
//   inlineKey[len+1..27] = zero-padded
struct alignas(16) PackedEntry {
    std::uint64_t inlineValue;   // scalar inline; complex<double> spills to pool
    std::uint32_t payloadOffset; // into payload pool (0 if inline scalar)
    std::uint32_t payloadLength; // bytes (0 if inline scalar)
    std::uint16_t keyId;         // canonical id, kInlineKeyId = inline key, kEndMarkerId = end marker
    std::uint8_t  valueType;     // see Value::ValueType (single-byte tag)
    std::uint8_t  flags;         // kEntryFlag* bits
    char          inlineKey[28]; // zero-padded when keyId == kInlineKeyId; 27 chars + length
};
static_assert(sizeof(PackedEntry) == 48UZ);
static_assert(alignof(PackedEntry) == 16UZ);

// Per-element header inside a Tensor<Value> sub-blob. Mirrors the value-bearing fields of
// PackedEntry minus the 30 bytes of key state (keyId + inlineKey) and the standalone
// payloadOffset (offset is implicit by sequential packing within the tensor sub-blob).
// Fixed-size element tensors (Tensor<float>, Tensor<int64_t>, …) skip this struct entirely
// and store contiguous element bytes after the tensor header + extents.
struct alignas(8) PackedTensorElement {
    std::uint64_t inlineValue;   // inline scalar payload (≤8B); 0 for variable-size elements
    std::uint32_t payloadLength; // bytes following the header (0 for inline scalars)
    std::uint8_t  valueType;     // Value::ValueType (single-byte tag)
    std::uint8_t  flags;         // kEntryFlag* bits — same semantics as PackedEntry.flags
    std::uint16_t reserved;
};
static_assert(sizeof(PackedTensorElement) == 16UZ);
static_assert(alignof(PackedTensorElement) == 8UZ);

enum class DeserialiseError : std::uint8_t {
    None = 0U,
    TooSmall, // span shorter than sizeof(Header)
    MagicMismatch,
    VersionUnsupported,
    CorruptOffset,      // an entry's payloadOffset/length exceeds totalSize
    AlignmentViolation, // span.data() not 16-byte aligned
};

namespace detail {

// C++ type → Value::ValueType. Inverse of valueTypeToCpp; Monostate for unsupported T.
template<typename T>
consteval Value::ValueType cppToValueType() {
    using U = std::remove_cvref_t<T>;
    if constexpr (std::same_as<U, bool>) {
        return Value::ValueType::Bool;
    } else if constexpr (std::same_as<U, std::int8_t>) {
        return Value::ValueType::Int8;
    } else if constexpr (std::same_as<U, std::int16_t>) {
        return Value::ValueType::Int16;
    } else if constexpr (std::same_as<U, std::int32_t>) {
        return Value::ValueType::Int32;
    } else if constexpr (std::same_as<U, std::int64_t>) {
        return Value::ValueType::Int64;
    } else if constexpr (std::same_as<U, std::uint8_t>) {
        return Value::ValueType::UInt8;
    } else if constexpr (std::same_as<U, std::uint16_t>) {
        return Value::ValueType::UInt16;
    } else if constexpr (std::same_as<U, std::uint32_t>) {
        return Value::ValueType::UInt32;
    } else if constexpr (std::same_as<U, std::uint64_t>) {
        return Value::ValueType::UInt64;
    } else if constexpr (std::same_as<U, float>) {
        return Value::ValueType::Float32;
    } else if constexpr (std::same_as<U, double>) {
        return Value::ValueType::Float64;
    } else if constexpr (std::same_as<U, std::complex<float>>) {
        return Value::ValueType::ComplexFloat32;
    } else if constexpr (std::same_as<U, std::complex<double>>) {
        return Value::ValueType::ComplexFloat64;
    } else if constexpr (std::same_as<U, Value>) {
        return Value::ValueType::Value;
    } else {
        return Value::ValueType::Monostate;
    }
}

// Runtime ValueType → callback with std::type_identity<T>. Centralises the runtime
// → compile-time bridge so `valueTypeToCpp` remains the single source of truth.
// F must be invocable with std::type_identity<T> for every supported binding plus
// std::type_identity<void> (for Monostate / nested / unsupported tags).
template<typename F>
constexpr decltype(auto) dispatchValueType(Value::ValueType vt, F&& f) {
    switch (vt) {
    case Value::ValueType::Bool: return f(std::type_identity<bool>{});
    case Value::ValueType::Int8: return f(std::type_identity<std::int8_t>{});
    case Value::ValueType::Int16: return f(std::type_identity<std::int16_t>{});
    case Value::ValueType::Int32: return f(std::type_identity<std::int32_t>{});
    case Value::ValueType::Int64: return f(std::type_identity<std::int64_t>{});
    case Value::ValueType::UInt8: return f(std::type_identity<std::uint8_t>{});
    case Value::ValueType::UInt16: return f(std::type_identity<std::uint16_t>{});
    case Value::ValueType::UInt32: return f(std::type_identity<std::uint32_t>{});
    case Value::ValueType::UInt64: return f(std::type_identity<std::uint64_t>{});
    case Value::ValueType::Float32: return f(std::type_identity<float>{});
    case Value::ValueType::Float64: return f(std::type_identity<double>{});
    case Value::ValueType::ComplexFloat32: return f(std::type_identity<std::complex<float>>{});
    case Value::ValueType::ComplexFloat64: return f(std::type_identity<std::complex<double>>{});
    case Value::ValueType::String: return f(std::type_identity<std::string_view>{});
    case Value::ValueType::Value: return f(std::type_identity<Value>{});
    default: return f(std::type_identity<void>{});
    }
}

// Trait: T fits in the 8-byte inlineValue slot (numeric scalars + std::complex<float>).
template<typename T>
concept InlineScalar = std::same_as<std::remove_cvref_t<T>, bool>             //
                       || std::same_as<std::remove_cvref_t<T>, std::int8_t>   //
                       || std::same_as<std::remove_cvref_t<T>, std::int16_t>  //
                       || std::same_as<std::remove_cvref_t<T>, std::int32_t>  //
                       || std::same_as<std::remove_cvref_t<T>, std::int64_t>  //
                       || std::same_as<std::remove_cvref_t<T>, std::uint8_t>  //
                       || std::same_as<std::remove_cvref_t<T>, std::uint16_t> //
                       || std::same_as<std::remove_cvref_t<T>, std::uint32_t> //
                       || std::same_as<std::remove_cvref_t<T>, std::uint64_t> //
                       || std::same_as<std::remove_cvref_t<T>, float>         //
                       || std::same_as<std::remove_cvref_t<T>, double>        //
                       || std::same_as<std::remove_cvref_t<T>, std::complex<float>>;

// Trait: T is too large for inline storage — must spill to the payload pool. Today only
// std::complex<double> (16 B); tensor / nested-map paths have their own concepts.
template<typename T>
concept PayloadScalar = std::same_as<std::remove_cvref_t<T>, std::complex<double>>;

template<typename T>
concept StringLike = std::convertible_to<T, std::string_view> && !InlineScalar<T> && !PayloadScalar<T>;

// Wire-format-supported tensor element types. Inline scalars + complex<double> use the
// fixed-size contiguous layout; Value uses the variable-size per-element header layout
// (PackedTensorElement + payload). std::pmr::string is NOT supported as a Tensor element
// type — Tensor.hpp:163 already rejects std::string; string-array-like data goes through
// Tensor<Value> where each element is a String-typed Value.
template<typename T>
concept TensorElementType = InlineScalar<T> || PayloadScalar<T> || std::same_as<T, Value>;

// Pack a scalar into the uint64 inlineValue slot at offset 0; upper bytes zero.
// All InlineScalar types — including bool — share the same byte layout: bool occupies
// the single low byte (0/1), enabling uniform `T&` aliasing in the typed fluent.
// std::complex<float> goes through the same path (8 bytes total, fits exactly).
template<InlineScalar T>
inline void writeInlineScalar(PackedEntry& e, T v) noexcept {
    e.inlineValue = 0U;
    if constexpr (sizeof(T) == sizeof(std::uint64_t)) {
        e.inlineValue = std::bit_cast<std::uint64_t>(v);
    } else {
        const auto buf = std::bit_cast<std::array<std::byte, sizeof(T)>>(v);
        std::memcpy(&e.inlineValue, buf.data(), sizeof(T));
    }
}

template<InlineScalar T>
[[nodiscard]] inline T readInlineScalar(const PackedEntry& e) noexcept {
    if constexpr (sizeof(T) == sizeof(std::uint64_t)) {
        return std::bit_cast<T>(e.inlineValue);
    } else {
        std::array<std::byte, sizeof(T)> buf;
        std::memcpy(buf.data(), &e.inlineValue, sizeof(T));
        return std::bit_cast<T>(buf);
    }
}

// Pack a short string into inlineKey[28]. name.size() must be <= 27.
inline void setInlineKey(PackedEntry& e, std::string_view name) noexcept {
    e.keyId        = keys::kInlineKeyId;
    e.inlineKey[0] = static_cast<char>(name.size());
    std::memset(e.inlineKey + 1, 0, 27);
    std::memcpy(e.inlineKey + 1, name.data(), name.size());
}

[[nodiscard]] inline std::string_view readInlineKey(const PackedEntry& e) noexcept {
    const auto len = static_cast<std::size_t>(static_cast<std::uint8_t>(e.inlineKey[0]));
    return {e.inlineKey + 1, len};
}

// Spilled-key encoding (keyId == kSpilledKeyId): inlineKey[0..3] holds the u32 payload
// offset, inlineKey[4..7] holds the u32 length; remaining bytes are zero-padded. Used for
// keys longer than kMaxInlineKeyLength (27 chars).
inline void setSpilledKey(PackedEntry& e, std::uint32_t offset, std::uint32_t length) noexcept {
    e.keyId = keys::kSpilledKeyId;
    std::memset(e.inlineKey, 0, sizeof(e.inlineKey));
    std::memcpy(e.inlineKey, &offset, sizeof(offset));
    std::memcpy(e.inlineKey + sizeof(offset), &length, sizeof(length));
}

[[nodiscard]] inline std::pair<std::uint32_t, std::uint32_t> readSpilledKeyOffsetLength(const PackedEntry& e) noexcept {
    std::uint32_t offset = 0;
    std::uint32_t length = 0;
    std::memcpy(&offset, e.inlineKey, sizeof(offset));
    std::memcpy(&length, e.inlineKey + sizeof(offset), sizeof(length));
    return {offset, length};
}

[[nodiscard]] inline std::string_view readSpilledKey(const std::byte* blobBase, const PackedEntry& e) noexcept {
    const auto [offset, length] = readSpilledKeyOffsetLength(e);
    return {reinterpret_cast<const char*>(blobBase + offset), length};
}

// Convert any supported key type to a string_view (non-owning; must outlive the call).
template<typename K>
[[nodiscard]] inline std::string_view keyToStringView(const K& key) noexcept {
    if constexpr (std::same_as<std::remove_cvref_t<K>, const char*> || std::is_array_v<std::remove_cvref_t<K>>) {
        return std::string_view{key};
    } else {
        return std::string_view{key};
    }
}

// Hard cap on nested-map recursion depth. Bounds stack usage and protects against
// pathological / malformed sub-blobs (matters once Phase 1b·b's from_blob accepts
// untrusted byte spans). Real-world tag-meta nesting is shallow (≤ 2-3 levels).
inline constexpr std::uint32_t kMaxDecodeDepth = 32U;

// Forward declarations for the tensor encode/decode helpers — defined after `decodeEntry`
// so that the Value/Tensor recursion (Tensor<Value> elements that are themselves nested
// ValueMaps or Tensors) can call back into `decodeEntry` for sub-blob walking.
[[nodiscard]] inline Value decodeEntry(const PackedEntry& e, const std::byte* blobBase, std::pmr::memory_resource* resource, std::uint32_t depth);
[[nodiscard]] inline Value decodeTensorBlob(const std::byte* tensorBase, std::uint32_t tensorBytes, std::pmr::memory_resource* resource, std::uint32_t depth);
[[nodiscard]] inline Value decodeTensorElement(const PackedTensorElement& elem, const std::byte* payloadData, std::pmr::memory_resource* resource, std::uint32_t depth);

// Phase 1e Step C: builds a Value containing a nested ValueMap from a packed-entries array.
// Defined after ValueMap class is fully defined (Value::Map is a forward-declared ValueMap
// here — sizeof / new requires full def).
[[nodiscard]] inline Value buildNestedMapValue(const std::byte* base, const PackedEntry* nestedEntries, std::uint16_t declaredCount, std::pmr::memory_resource* resource, std::uint32_t depth);
inline void                encodeTensorElement(std::pmr::vector<std::byte>& out, const Value& val, std::pmr::memory_resource* resource, std::uint32_t depth);
template<typename Tens>
[[nodiscard]] std::pmr::vector<std::byte> encodeTensorBlob(const Tens& tensor, std::pmr::memory_resource* resource, std::uint32_t depth);

// Decode a single PackedEntry into a Value, with payload offsets resolved against `blobBase`.
// `blobBase` is the start of the blob the entry lives in — `_blob` for the top level,
// `_blob + parent.payloadOffset` for entries inside a nested-map sub-blob. Recurses into
// further nested maps with `depth` capped at kMaxDecodeDepth. Free function so it can be
// called from inside a sub-blob view without needing a parent ValueMap object.
[[nodiscard]] inline Value decodeEntry(const PackedEntry& e, const std::byte* blobBase, std::pmr::memory_resource* resource, std::uint32_t depth = 0U) {
    const auto vt = static_cast<Value::ValueType>(e.valueType);

    // Special-case Tensor sub-blob (Tensor<T>, including Tensor<Value> with mixed elements).
    if (vt == Value::ValueType::Value && (e.flags & kEntryFlagTensor) != 0U && (e.flags & kEntryFlagOffsetLength) != 0U) {
        if (depth >= kMaxDecodeDepth) {
            return Value{resource};
        }
        return decodeTensorBlob(blobBase + e.payloadOffset, e.payloadLength, resource, depth + 1U);
    }

    // Special-case nested ValueMap before the type-driven dispatch — recursive read.
    if (vt == Value::ValueType::Value && (e.flags & kEntryFlagNestedMap) != 0U && (e.flags & kEntryFlagOffsetLength) != 0U) {
        if (depth >= kMaxDecodeDepth) {
            return Value{resource}; // refuse to recurse further — protects stack vs pathological input
        }
        const auto* base = blobBase + e.payloadOffset;
        if (e.payloadLength < sizeof(Header)) {
            return Value{resource}; // sub-blob too small to even hold a Header
        }
        const auto* nestedH         = std::launder(reinterpret_cast<const Header*>(base));
        const auto  declaredCount   = nestedH->entryCount;
        const auto  entryArrayBytes = static_cast<std::size_t>(declaredCount) * sizeof(PackedEntry);
        // Bounds check: declared entry array must fit in the sub-blob.
        if (sizeof(Header) + entryArrayBytes > e.payloadLength) {
            return Value{resource}; // corrupt / truncated sub-blob — refuse to read past the end
        }
        const auto* nestedE = std::launder(reinterpret_cast<const PackedEntry*>(base + sizeof(Header)));
        return buildNestedMapValue(base, nestedE, declaredCount, resource, depth);
    }

    return dispatchValueType(vt, [&]<typename T>(std::type_identity<T>) -> Value {
        if constexpr (std::same_as<T, std::string_view>) {
            if ((e.flags & kEntryFlagOffsetLength) == 0U) {
                return Value{resource};
            }
            return Value{std::string_view{reinterpret_cast<const char*>(blobBase + e.payloadOffset), e.payloadLength}, resource};
        } else if constexpr (InlineScalar<T>) {
            return Value{readInlineScalar<T>(e), resource};
        } else if constexpr (PayloadScalar<T>) {
            if ((e.flags & kEntryFlagOffsetLength) == 0U) {
                return Value{resource};
            }
            T out;
            std::memcpy(&out, blobBase + e.payloadOffset, sizeof(T));
            return Value{out, resource};
        } else {
            return Value{resource}; // Monostate / unsupported
        }
    });
}

// Read a tensor sub-blob at `tensorBase` (length `tensorBytes`) and rehydrate a Value
// holding gr::Tensor<ElemT> for the dispatched element type. Element-type re-dispatch via
// dispatchValueType lets the same routine cover Tensor<float> through Tensor<Value>.
[[nodiscard]] inline Value decodeTensorBlob(const std::byte* tensorBase, std::uint32_t tensorBytes, std::pmr::memory_resource* resource, std::uint32_t depth) {
    if (tensorBytes < kTensorBlobHeaderSize) {
        return Value{resource};
    }
    const auto elementVT     = static_cast<Value::ValueType>(static_cast<std::uint8_t>(tensorBase[0]));
    const auto rank          = static_cast<std::size_t>(static_cast<std::uint8_t>(tensorBase[1]));
    const auto encodingFlags = static_cast<std::uint8_t>(tensorBase[2]);
    const bool variableSize  = (encodingFlags & kTensorEncodingVariableSize) != 0U;
    if (rank > kMaxTensorRank) {
        return Value{resource};
    }
    std::uint32_t elementCount;
    std::memcpy(&elementCount, tensorBase + 4, sizeof(elementCount));
    if (elementCount > kMaxTensorElements) {
        return Value{resource};
    }
    const std::size_t extentsBytes = 4UZ * rank;
    if (kTensorBlobHeaderSize + extentsBytes > tensorBytes) {
        return Value{resource};
    }
    std::array<std::size_t, kMaxTensorRank> extentsBuf{};
    for (std::size_t i = 0UZ; i < rank; ++i) {
        std::uint32_t ext;
        std::memcpy(&ext, tensorBase + kTensorBlobHeaderSize + 4UZ * i, sizeof(ext));
        extentsBuf[i] = static_cast<std::size_t>(ext);
    }
    const std::span<const std::size_t> extents{extentsBuf.data(), rank};
    const std::byte*                   elementData  = tensorBase + kTensorBlobHeaderSize + extentsBytes;
    const std::uint32_t                elementBytes = tensorBytes - static_cast<std::uint32_t>(kTensorBlobHeaderSize + extentsBytes);

    return dispatchValueType(elementVT, [&]<typename T>(std::type_identity<T>) -> Value {
        if constexpr (std::same_as<T, std::string_view>) {
            return Value{resource}; // Tensor<std::string> not a valid C++ type — string arrays go through Tensor<Value>
        } else if constexpr (TensorElementType<T>) {
            if (variableSize != std::same_as<T, Value>) {
                return Value{resource}; // wire/element-type mismatch
            }
            if constexpr (std::same_as<T, Value>) {
                gr::pmr::vector<Value, true> elems(elementCount, Value{resource}, resource);
                std::uint32_t                offset = 0U;
                for (std::uint32_t i = 0U; i < elementCount; ++i) {
                    if (offset + sizeof(PackedTensorElement) > elementBytes) {
                        return Value{resource};
                    }
                    PackedTensorElement headerCopy;
                    std::memcpy(&headerCopy, elementData + offset, sizeof(headerCopy));
                    offset += static_cast<std::uint32_t>(sizeof(headerCopy));
                    if (headerCopy.payloadLength > elementBytes - offset) {
                        return Value{resource};
                    }
                    elems[i] = decodeTensorElement(headerCopy, elementData + offset, resource, depth);
                    offset += headerCopy.payloadLength;
                }
                gr::Tensor<Value> tensor(gr::extents_from, extents, resource);
                tensor._data = std::move(elems);
                return Value{std::move(tensor), resource};
            } else {
                if (elementCount != 0U && static_cast<std::size_t>(elementCount) > elementBytes / sizeof(T)) {
                    return Value{resource};
                }
                gr::Tensor<T> tensor(gr::extents_from, extents, resource);
                if (elementCount > 0U) {
                    std::memcpy(std::to_address(tensor._data.begin()), elementData, sizeof(T) * elementCount);
                }
                return Value{std::move(tensor), resource};
            }
        } else {
            return Value{resource};
        }
    });
}

// Decode a single PackedTensorElement (+ trailing payload bytes) into a Value. Mirrors
// decodeEntry's dispatch but uses the per-element header (no key, offset implicit).
[[nodiscard]] inline Value decodeTensorElement(const PackedTensorElement& elem, const std::byte* payloadData, std::pmr::memory_resource* resource, std::uint32_t depth) {
    const auto vt = static_cast<Value::ValueType>(elem.valueType);

    if (vt == Value::ValueType::Value && (elem.flags & kEntryFlagTensor) != 0U) {
        if (depth >= kMaxDecodeDepth) {
            return Value{resource};
        }
        return decodeTensorBlob(payloadData, elem.payloadLength, resource, depth + 1U);
    }

    if (vt == Value::ValueType::Value && (elem.flags & kEntryFlagNestedMap) != 0U) {
        if (depth >= kMaxDecodeDepth) {
            return Value{resource};
        }
        if (elem.payloadLength < sizeof(Header)) {
            return Value{resource};
        }
        const auto* nestedH         = std::launder(reinterpret_cast<const Header*>(payloadData));
        const auto  declaredCount   = nestedH->entryCount;
        const auto  entryArrayBytes = static_cast<std::size_t>(declaredCount) * sizeof(PackedEntry);
        if (sizeof(Header) + entryArrayBytes > elem.payloadLength) {
            return Value{resource};
        }
        const auto* nestedE = std::launder(reinterpret_cast<const PackedEntry*>(payloadData + sizeof(Header)));
        return buildNestedMapValue(payloadData, nestedE, declaredCount, resource, depth);
    }

    return dispatchValueType(vt, [&]<typename T>(std::type_identity<T>) -> Value {
        if constexpr (std::same_as<T, std::string_view>) {
            return Value{std::string_view{reinterpret_cast<const char*>(payloadData), elem.payloadLength}, resource};
        } else if constexpr (InlineScalar<T>) {
            PackedEntry tmp;
            tmp.inlineValue = elem.inlineValue;
            return Value{readInlineScalar<T>(tmp), resource};
        } else if constexpr (PayloadScalar<T>) {
            if (elem.payloadLength < sizeof(T)) {
                return Value{resource};
            }
            T out;
            std::memcpy(&out, payloadData, sizeof(T));
            return Value{out, resource};
        } else {
            return Value{resource};
        }
    });
}

// Body defined after the ValueMap class — encodeTensorElement constructs a fresh ValueMap
// for the nested-map element case.

// Build the full tensor sub-blob bytes from a Tensor<T>. Layout: [elementValueType:1]
// [rank:1][encodingFlags:1][reserved:1][elementCount:4][extents:4*rank], then either
// contiguous element bytes (fixed-size T) or a sequence of PackedTensorElement+payload
// (T == Value).
template<typename Tens>
[[nodiscard]] std::pmr::vector<std::byte> encodeTensorBlob(const Tens& tensor, std::pmr::memory_resource* resource, std::uint32_t depth) {
    using ElemT = typename gr::tensor_traits<std::remove_cvref_t<Tens>>::value_type;
    static_assert(TensorElementType<ElemT>, "ValueMap: Tensor element type must be an inline scalar, std::complex<double>, or gr::pmt::Value");

    const std::size_t rank = tensor.rank();
    if (rank > kMaxTensorRank) {
        throw std::length_error{"gr::pmt::ValueMap: tensor rank exceeds kMaxTensorRank"};
    }
    const auto        extents      = tensor.extents();
    const std::size_t elementCount = tensor.size();
    if (elementCount > kMaxTensorElements) {
        throw std::length_error{"gr::pmt::ValueMap: tensor element count exceeds kMaxTensorElements"};
    }

    constexpr Value::ValueType elemVT       = cppToValueType<ElemT>();
    constexpr bool             variableSize = std::same_as<ElemT, Value>;

    const std::size_t           extentsBytes = 4UZ * rank;
    std::pmr::vector<std::byte> out(resource);
    out.resize(kTensorBlobHeaderSize + extentsBytes);

    out[0]                 = static_cast<std::byte>(elemVT);
    out[1]                 = static_cast<std::byte>(rank);
    out[2]                 = static_cast<std::byte>(variableSize ? kTensorEncodingVariableSize : 0U);
    out[3]                 = std::byte{0U};
    const std::uint32_t ec = static_cast<std::uint32_t>(elementCount);
    std::memcpy(out.data() + 4, &ec, sizeof(ec));
    for (std::size_t i = 0UZ; i < rank; ++i) {
        const std::uint32_t ext = static_cast<std::uint32_t>(extents[i]);
        std::memcpy(out.data() + kTensorBlobHeaderSize + 4UZ * i, &ext, sizeof(ext));
    }

    if constexpr (variableSize) {
        out.reserve(out.size() + elementCount * sizeof(PackedTensorElement));
        for (std::size_t i = 0UZ; i < elementCount; ++i) {
            encodeTensorElement(out, *(tensor._data.data() + i), resource, depth + 1U);
        }
    } else {
        if (elementCount > 0UZ) {
            const auto byteCount = elementCount * sizeof(ElemT);
            out.resize(out.size() + byteCount);
            std::memcpy(out.data() + kTensorBlobHeaderSize + extentsBytes, std::to_address(tensor._data.begin()), byteCount);
        }
    }

    return out;
}

} // namespace detail

class ValueMap {
public:
    class const_iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using iterator_concept  = std::random_access_iterator_tag;
        using value_type        = std::pair<std::string_view, Value>;
        using reference         = value_type; // by value; see class doc on invalidation
        using pointer           = void;
        using difference_type   = std::ptrdiff_t;

        constexpr const_iterator() = default;
        constexpr const_iterator(const ValueMap* m, std::uint16_t i) noexcept : _map(m), _index(i) {}

        [[nodiscard]] value_type           operator*() const; // defined after ValueMap is complete (accesses private state)
        [[nodiscard]] constexpr value_type operator[](difference_type n) const { return *(*this + n); }

        constexpr const_iterator& operator++() noexcept {
            ++_index;
            return *this;
        }
        constexpr const_iterator operator++(int) noexcept {
            auto tmp = *this;
            ++_index;
            return tmp;
        }
        constexpr const_iterator& operator--() noexcept {
            --_index;
            return *this;
        }
        constexpr const_iterator operator--(int) noexcept {
            auto tmp = *this;
            --_index;
            return tmp;
        }

        constexpr const_iterator& operator+=(difference_type n) noexcept {
            _index = static_cast<std::uint16_t>(static_cast<difference_type>(_index) + n);
            return *this;
        }
        constexpr const_iterator& operator-=(difference_type n) noexcept { return *this += -n; }

        [[nodiscard]] friend constexpr const_iterator  operator+(const_iterator a, difference_type n) noexcept { return a += n; }
        [[nodiscard]] friend constexpr const_iterator  operator+(difference_type n, const_iterator a) noexcept { return a += n; }
        [[nodiscard]] friend constexpr const_iterator  operator-(const_iterator a, difference_type n) noexcept { return a -= n; }
        [[nodiscard]] friend constexpr difference_type operator-(const_iterator a, const_iterator b) noexcept { return static_cast<difference_type>(a._index) - static_cast<difference_type>(b._index); }

        [[nodiscard]] constexpr auto operator<=>(const const_iterator& other) const noexcept { return _index <=> other._index; }
        [[nodiscard]] constexpr bool operator==(const const_iterator& other) const noexcept = default;

    private:
        friend class ValueMap;
        const ValueMap* _map   = nullptr;
        std::uint16_t   _index = 0U;
    };

    using iterator               = const_iterator; // mutation goes through member functions, not iterator deref-assign
    using reverse_iterator       = std::reverse_iterator<const_iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // STL-parity nested types — let template code dispatch on these the same way it does for std::map.
    using key_type    = std::string_view;
    using mapped_type = Value;
    using value_type  = std::pair<std::string_view, Value>;
    using size_type   = std::size_t;

    // Default ctor — non-explicit so legacy `property_map x = {};` and `func(... = {})`
    // copy-list-init patterns continue to work after the alias swap (Value::Map = ValueMap).
    ValueMap(std::pmr::memory_resource* resource = std::pmr::get_default_resource(), std::uint32_t initialCapacityEntries = 8U, std::uint32_t slackBytes = 0U) : _resource(resource ? resource : std::pmr::get_default_resource()) { _allocateBlob(initialCapacityEntries, /*initialPayloadReserve=*/0U, slackBytes); }

    ValueMap(const ValueMap& other, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(resource ? resource : std::pmr::get_default_resource()) { _copyBlobFrom(other); }

    /// View-mode factory — aliases external bytes (typically another ValueMap blob's payload pool).
    /// `_resource == nullptr` is the discriminant; mutators assert in debug, no-op in release.
    /// Lifetime is bound to the source bytes; caller must keep them alive while the view exists.
    [[nodiscard]] static ValueMap makeView(std::span<const std::byte> bytes) noexcept { return ValueMap{ViewModeTag{}, bytes}; }

    /// Initializer-list constructor for `property_map{ {"k1", v1}, {"k2", v2}, ... }` brace-init
    /// patterns. Uses the public `value_type` (= `std::pair<std::string_view, Value>`) so brace-
    /// initializers convert via Value's implicit ctors from scalars / strings / containers.
    ValueMap(std::initializer_list<value_type> init, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(resource ? resource : std::pmr::get_default_resource()) {
        _allocateBlob(static_cast<std::uint32_t>(std::max<std::size_t>(init.size(), std::size_t{8})), 0U, 0U);
        for (const auto& p : init) {
            (void)emplace(p.first, p.second);
        }
    }

    /// Iterator-pair (range) constructor — for `property_map{first, last}` patterns from
    /// std::initializer_list iterators or any pair-yielding range.
    template<typename InputIt>
    requires requires(InputIt it) {
        { (*it).first };
        { (*it).second };
    }
    ValueMap(InputIt first, InputIt last, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(resource ? resource : std::pmr::get_default_resource()) {
        _allocateBlob(8U, 0U, 0U);
        for (; first != last; ++first) {
            const auto& p = *first;
            (void)emplace(p.first, p.second);
        }
    }

    [[nodiscard]] constexpr bool is_view() const noexcept { return _resource == nullptr; }

    /// Materialise a view-mode (or owning) blob into a fresh owning ValueMap allocated against
    /// the given PMR resource (defaults to the global default resource). Bytes are deep-copied.
    [[nodiscard]] ValueMap owned(std::pmr::memory_resource* resource = std::pmr::get_default_resource()) const { return ValueMap(*this, resource ? resource : std::pmr::get_default_resource()); }

    ValueMap(ValueMap&& other) noexcept : _blob(other._blob), _resource(other._resource), _capacity(other._capacity), _header(other._header), _entries(other._entries) {
        other._blob     = nullptr;
        other._capacity = 0U;
        other._header   = nullptr;
        other._entries  = nullptr;
        // other._resource left valid so other may be reassigned or destroyed.
    }

    // Conversion construction from any associative container whose key is convertible to
    // std::string_view and whose mapped_type is gr::pmt::Value (e.g. std::map<std::string, Value>,
    // Value::Map = std::pmr::unordered_map<std::pmr::string, Value, …>). Per-entry value is
    // re-typed via dispatchValueType so the on-blob inline-scalar / payload-pool layout is
    // populated correctly. Monostate / nested-Value / complex / Tensor entries are skipped
    // for now (re-enabled in Phase 1b·d when extended value-type coverage lands).
    template<typename Map>
    requires requires {
        typename Map::key_type;
        typename Map::mapped_type;
        requires std::convertible_to<typename Map::key_type, std::string_view>;
        requires std::same_as<std::remove_cvref_t<typename Map::mapped_type>, Value>;
    }
    explicit ValueMap(const Map& src, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(resource ? resource : std::pmr::get_default_resource()) {
        _allocateBlob(static_cast<std::uint32_t>(std::max<std::size_t>(src.size(), std::size_t{8})), 0U, 0U);
        for (const auto& [key, value] : src) {
            const std::string_view sv{key};
            if (value.is_map()) {
                if (auto srcMap = value.template get_if<Value::Map>()) {
                    emplace(sv, srcMap->owned(_resource));
                }
                continue;
            }
            if (value.is_tensor()) {
                detail::dispatchValueType(value.value_type(), [&]<typename T>(std::type_identity<T>) {
                    if constexpr (detail::TensorElementType<T>) {
                        if (auto tensor = value.template get_if<gr::Tensor<T>>()) {
                            emplace(sv, *tensor);
                        }
                    }
                });
                continue;
            }
            detail::dispatchValueType(value.value_type(), [&]<typename T>(std::type_identity<T>) {
                if constexpr (detail::InlineScalar<T>) {
                    emplace(sv, value.template value_or<T>(T{}));
                } else if constexpr (detail::PayloadScalar<T>) {
                    emplace(sv, value.template value_or<T>(T{}));
                } else if constexpr (std::same_as<T, std::string_view>) {
                    emplace(sv, value.value_or(std::string_view{}));
                }
            });
        }
    }

    ValueMap& operator=(const ValueMap& other) {
        if (this == &other) {
            return *this;
        }
        ValueMap tmp(other, _resource);
        swap(*this, tmp);
        return *this;
    }

    ValueMap& operator=(ValueMap&& other) {
        if (this == &other) {
            return *this;
        }
        if (_resource == other._resource) {
            _deallocateBlob();
            _blob           = other._blob;
            _capacity       = other._capacity;
            _header         = other._header;
            _entries        = other._entries;
            other._blob     = nullptr;
            other._capacity = 0U;
            other._header   = nullptr;
            other._entries  = nullptr;
        } else {
            ValueMap tmp(other, _resource); // may allocate via _resource
            swap(*this, tmp);
            // invalidate source per move-semantics
            other.clear();
        }
        return *this;
    }

    ~ValueMap() { _deallocateBlob(); }

    friend void swap(ValueMap& a, ValueMap& b) noexcept {
        std::swap(a._blob, b._blob);
        std::swap(a._resource, b._resource);
        std::swap(a._capacity, b._capacity);
        std::swap(a._header, b._header);
        std::swap(a._entries, b._entries);
    }

    template<typename K, typename V>
    std::pair<const_iterator, bool> emplace(K&& key, V&& value) {
        if (_isViewAndAssertNoMutation()) [[unlikely]] {
            return {end(), false};
        }
        const auto sv = detail::keyToStringView(key);
        if (auto it = find(sv); it != end()) {
            return {it, false};
        }
        const_iterator new_it = _insertNew(sv, std::forward<V>(value));
        return {new_it, new_it != end()};
    }

    template<typename K, typename V>
    std::pair<const_iterator, bool> insert_or_assign(K&& key, V&& value) {
        if (_isViewAndAssertNoMutation()) [[unlikely]] {
            return {end(), false};
        }
        const auto sv = detail::keyToStringView(key);
        if (auto it = find(sv); it != end()) {
            _assignValueAt(it._index, std::forward<V>(value));
            return {it, false};
        }
        const_iterator new_it = _insertNew(sv, std::forward<V>(value));
        return {new_it, new_it != end()};
    }

    /// STL-parity try_emplace — inserts only if key is absent. Mirrors std::map::try_emplace.
    /// Returns `{iter, true}` on insert, `{iter_to_existing, false}` if key already present.
    template<typename K, typename... Args>
    std::pair<const_iterator, bool> try_emplace(K&& key, Args&&... args) {
        if (_isViewAndAssertNoMutation()) [[unlikely]] {
            return {end(), false};
        }
        const auto sv = detail::keyToStringView(key);
        if (auto it = find(sv); it != end()) {
            return {it, false};
        }
        const_iterator new_it = _insertNew(sv, std::forward<Args>(args)...);
        return {new_it, new_it != end()};
    }

    /// STL-parity insert by value_type — accepts brace-init `{"k", val}`. Returns
    /// `{iter, true}` on success; `{iter_to_existing, false}` if key already present
    /// (does NOT overwrite — matches `std::map::insert(value_type)`).
    std::pair<const_iterator, bool> insert(const value_type& p) { return emplace(p.first, p.second); }

    /// Generic pair-shaped insert (typename PairT with `.first` / `.second`).
    template<typename PairT>
    requires requires(const PairT& p) {
        { p.first };
        { p.second };
    } && (!std::same_as<std::remove_cvref_t<PairT>, value_type>)
    std::pair<const_iterator, bool> insert(const PairT& p) {
        return emplace(p.first, p.second);
    }

    /// STL-parity range insert. Iterator value_type must be a pair-like with `.first` / `.second`.
    template<typename InputIt>
    void insert(InputIt first, InputIt last) {
        for (; first != last; ++first) {
            const auto& p = *first;
            (void)emplace(p.first, p.second);
        }
    }

    /// std::initializer_list of {key, value} pairs — for `property_map{ {"k", Value{...}}, ... }`
    /// brace-initialisation patterns common in tag-construction call sites.
    template<typename PairT>
    requires requires(const PairT& p) {
        { p.first };
        { p.second };
    }
    void insert(std::initializer_list<PairT> init) {
        for (const auto& p : init) {
            (void)emplace(p.first, p.second);
        }
    }

    void erase(const_iterator it) {
        if (_isViewAndAssertNoMutation()) [[unlikely]] {
            return;
        }
        if (it._map != this || it._index >= size()) {
            return;
        }
        // O(N) compact; payload orphaned until next shrink_to_fit.
        const auto n = _header->entryCount;
        for (std::uint16_t i = it._index; i + 1U < n; ++i) {
            _entries[i] = _entries[i + 1U];
        }
        --_header->entryCount;
    }

    // STL-style erase by key. Returns the number of entries removed (0 or 1).
    template<typename K>
    requires(!std::same_as<std::remove_cvref_t<K>, const_iterator>)
    size_type erase(const K& key) {
        if (_isViewAndAssertNoMutation()) [[unlikely]] {
            return 0UZ;
        }
        const auto it = find(key);
        if (it == end()) {
            return 0UZ;
        }
        erase(it);
        return 1UZ;
    }

    void clear() noexcept {
        if (_isViewAndAssertNoMutation()) [[unlikely]] {
            return;
        }
        if (_header) {
            _header->entryCount  = 0U;
            _header->payloadUsed = 0U;
        }
    }

    void reserve(std::uint32_t entries, std::uint32_t payload_bytes = 0U) {
        if (_isViewAndAssertNoMutation()) [[unlikely]] {
            return;
        }
        const auto cur_cap = _entryCapacity();
        const auto needed  = std::max<std::uint32_t>(entries, _header ? _header->entryCount : 0U);
        if (needed > cur_cap || payload_bytes > _payloadCapacity()) {
            _grow(needed, payload_bytes);
        }
    }

    void shrink_to_fit() {
        if (_isViewAndAssertNoMutation()) [[unlikely]] {
            return;
        }
        const auto entries = _header ? _header->entryCount : 0U;
        const auto payload = _header ? _header->payloadUsed : 0U;
        _grow(entries, payload, /*shrink=*/true);
    }

    template<typename K>
    [[nodiscard]] constexpr const_iterator find(const K& key) const noexcept {
        const auto sv          = detail::keyToStringView(key);
        const auto canonicalId = keys::lookupId(sv);
        const auto n           = _header ? _header->entryCount : 0U;
        for (std::uint16_t i = 0U; i < n; ++i) {
            if (_entryMatches(_entries[i], sv, canonicalId)) {
                return const_iterator{this, i};
            }
        }
        return end();
    }

    template<typename K>
    [[nodiscard]] constexpr bool contains(const K& key) const noexcept {
        return find(key) != end();
    }

    // STL-style count — 0 or 1, since keys are unique.
    template<typename K>
    [[nodiscard]] constexpr size_type count(const K& key) const noexcept {
        return contains(key) ? 1UZ : 0UZ;
    }

    /**
     * @brief Return the Value bound to `key`, or a Monostate Value if the key is absent. No throws.
     *
     * Returned Value is OWNING (routed through `_entryToValue` → `decodeEntry`), so it survives
     * the source ValueMap's death — callers can safely return it from a function whose source
     * is a stack-local. This is the canonical way to escape iter scope:
     *
     * @code
     * std::optional<pmt::Value> Settings::get(const std::string& key) const noexcept {
     *     auto res = get(std::array{key});  // builds a temp ValueMap
     *     if (!res.contains(key)) {
     *         return std::nullopt;
     *     }
     *     return res.at(key);  // owning Value; res can die after the return
     * }
     * @endcode
     *
     * Iterator deref `(*it).second` would yield a view-mode Value here, which would dangle
     * when `res` dies — `at()` materialises into owning storage instead.
     */
    template<typename K>
    [[nodiscard]] Value at(const K& key) const noexcept {
        const auto it = find(key);
        return it == end() ? Value{_resource} : _entryToValue(_entries[it._index]);
    }

    // EXCEPTION (by design): const operator[] throws on missing key. Library / framework /
    // SYCL code that must stay exception-free MUST use at() / find() / contains() instead.
    // Returns Value by value (the on-blob storage has no Value object to alias).
    template<typename K>
    [[nodiscard]] Value operator[](const K& key) const {
        const auto it = find(key);
        if (it == end()) {
            throw std::out_of_range{"gr::pmt::ValueMap::operator[]: key not present"};
        }
        return _entryToValue(_entries[it._index]);
    }

    /// Writable subscript proxy — returned by non-const `operator[]`. Mirrors std::map semantics:
    /// reading a missing key auto-vivifies a default Value entry; assigning forwards to
    /// `insert_or_assign`. The proxy eagerly caches the current Value for the key on
    /// construction so that pointer-returning accessors (`get_if<T>()`) yield pointers
    /// stable for the proxy's full lifetime — supporting `auto* p = m[k].get_if<T>()` patterns.
    class SubscriptProxy {
        ValueMap*        _map;
        Value            _cache;
        bool             _wasMissing;
        std::string      _ownedKey; // owned copy so the proxy survives caller scope ends
        std::string_view _key;

    public:
        SubscriptProxy(ValueMap* m, std::string_view k) noexcept(false) : _map(m), _cache(m->_resource), _wasMissing(false), _ownedKey(k.data(), k.size()), _key(_ownedKey) {
            if (auto it = _map->find(_key); it != _map->end()) {
                _cache = _map->_entryToValue(_map->_entries[it._index]);
            } else {
                _wasMissing = true;
                // Auto-vivify a default monostate entry (matches std::map::operator[] semantics)
                // so subsequent reads / writes through the same key see a consistent state.
                _map->insert_or_assign(_key, Value{_map->_resource});
            }
        }

        SubscriptProxy(const SubscriptProxy&)            = delete;
        SubscriptProxy& operator=(const SubscriptProxy&) = delete;
        SubscriptProxy(SubscriptProxy&&)                 = delete;
        SubscriptProxy& operator=(SubscriptProxy&&)      = delete;

        // Implicit conversion to Value — copy of the cached value. Lifetime independent.
        operator Value() const& { return _cache; }
        operator Value() && { return std::move(_cache); }

        // Write: forwards to insert_or_assign. Updates the cache so chained access reads
        // the just-written value (`m[k] = 7; auto v = m[k];` yields 7 in both reads).
        template<typename V>
        requires(!std::is_same_v<std::remove_cvref_t<V>, SubscriptProxy>)
        SubscriptProxy& operator=(V&& v) {
            _map->insert_or_assign(_key, v);
            _cache = Value{std::forward<V>(v), _map->_resource};
            return *this;
        }

        // Read forwarders to the cached Value — pointers and references are stable for the
        // proxy's lifetime.
        template<typename T>
        [[nodiscard]] bool holds() const {
            return _cache.template holds<T>();
        }

        [[nodiscard]] bool is_monostate() const { return _cache.is_monostate(); }
        [[nodiscard]] bool is_string() const { return _cache.is_string(); }
        [[nodiscard]] bool is_tensor() const { return _cache.is_tensor(); }
        [[nodiscard]] bool is_map() const { return _cache.is_map(); }

        template<typename T>
        [[nodiscard]] auto value_or(T&& def) const {
            return _cache.value_or(std::forward<T>(def));
        }

        template<typename T>
        [[nodiscard]] auto get_if() const {
            return _cache.template get_if<T>();
        }

        [[nodiscard]] Value::ValueType     value_type() const { return _cache.value_type(); }
        [[nodiscard]] Value::ContainerType container_type() const { return _cache.container_type(); }
    };

    template<typename K>
    [[nodiscard]] SubscriptProxy operator[](const K& key) {
        return SubscriptProxy{this, detail::keyToStringView(key)};
    }

    // EXCEPTION (by design, user-requested STL parity): typed-name `at<>` throws on missing
    // key, and (in debug builds, gated on gr::meta::kDebugBuild) on stored-vs-canonical
    // type mismatch. For inline scalars: returns T& or const T& aliasing the on-blob byte
    // slot — const-ness is propagated from the implicit object via C++23 deducing-this.
    // For String-typed canonical keys: returns std::string_view by value. Library /
    // framework / SYCL code MUST use find() / contains() first to avoid the throw.
    template<gr::meta::fixed_string Name, typename Self>
    requires keys::CanonicalName<Name>
    [[nodiscard]] decltype(auto) at(this Self&& self) {
        using T            = keys::CanonicalCppType<keys::idOf<Name>>;
        constexpr auto kId = keys::idOf<Name>;
        const auto     idx = self.template _findCanonicalIndex<kId>();
        if (idx >= self.size()) {
            throw std::out_of_range{"gr::pmt::ValueMap::at<Name>: canonical key not present"};
        }
        ValueMap::template _debugAssertCanonicalType<kId>(self._entries[idx]);
        if constexpr (std::same_as<T, std::string_view>) {
            return self._readStringView(self._entries[idx]);
        } else {
            using PtrT = std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, const T*, T*>;
            return *std::launder(reinterpret_cast<PtrT>(&self._entries[idx].inlineValue));
        }
    }

    // STL-style equal_range — returns the half-open `[first, second)` range over entries
    // matching `key`. Since keys are unique, the range is at most 1 element wide.
    template<typename K>
    [[nodiscard]] constexpr std::pair<const_iterator, const_iterator> equal_range(const K& key) const noexcept {
        const auto it = find(key);
        if (it == end()) {
            return {end(), end()};
        }
        const auto next = const_iterator{this, static_cast<std::uint16_t>(it._index + 1U)};
        return {it, next};
    }

    // Move-merge entries from `other` into `*this`. Entries whose keys already exist in
    // `*this` are left in `other`; non-conflicting entries are moved out. Mirrors std::map::merge
    // on the success / no-op axis.
    //
    // Fast-path: copies the source PackedEntry + its payload bytes (string / complex<double> /
    // nested-map sub-blob / tensor sub-blob) directly between blobs. No round-trip through Value
    // — same byte format on both sides. Source is batch-compacted at the end (vs O(N²) per-entry
    // shift of the old erase-while-iterating loop).
    /// Rvalue-overload for merge — accepts `merge(makeMap())` patterns.
    void merge(ValueMap&& other) { merge(other); }

    void merge(ValueMap& other) {
        if (_isViewAndAssertNoMutation()) [[unlikely]] {
            return;
        }
        if (this == &other || !other._header) {
            return;
        }
        const std::uint16_t nSrc = other._header->entryCount;
        if (nSrc == 0U) {
            return;
        }
        // Pre-size destination's entry array + payload pool to absorb the worst case (all source
        // entries non-conflicting). Single grow up front instead of per-entry _grow churn.
        std::uint32_t totalPayloadGrowth = 0U;
        for (std::uint16_t i = 0U; i < nSrc; ++i) {
            const auto& src = other._entries[i];
            if ((src.flags & kEntryFlagOffsetLength) != 0U) {
                totalPayloadGrowth += src.payloadLength;
            }
            if (src.keyId == keys::kSpilledKeyId) {
                const auto [_, length] = detail::readSpilledKeyOffsetLength(src);
                totalPayloadGrowth += length;
            }
        }
        const std::uint32_t curEntries = _header ? _header->entryCount : 0U;
        const std::uint32_t curPayload = _header ? _header->payloadUsed : 0U;
        reserve(curEntries + nSrc, curPayload + totalPayloadGrowth);

        std::pmr::vector<bool> moved(nSrc, false, _resource);
        for (std::uint16_t i = 0U; i < nSrc; ++i) {
            const PackedEntry& sourceEntry = other._entries[i];
            const auto         key         = other._readKey(sourceEntry);
            if (contains(key)) {
                continue;
            }
            _ensureEntrySlot();
            const std::uint16_t destIndex = _header->entryCount;
            std::memcpy(&_entries[destIndex], &sourceEntry, sizeof(PackedEntry));
            ++_header->entryCount;
            bool       committed = false;
            const auto rollback  = gr::on_scope_exit{[&]() noexcept {
                if (!committed) {
                    --_header->entryCount;
                }
            }};
            if (sourceEntry.keyId == keys::kSpilledKeyId) {
                // Copy spilled key bytes too; offset gets updated to ours.
                const std::span<const std::byte> srcKey{reinterpret_cast<const std::byte*>(key.data()), key.size()};
                const auto                       newKeyOffset = _appendPayloadSafe(srcKey);
                detail::setSpilledKey(_entries[destIndex], newKeyOffset, static_cast<std::uint32_t>(key.size()));
            }
            if ((sourceEntry.flags & kEntryFlagOffsetLength) != 0U) {
                // Copy variable-size payload bytes verbatim (string / complex<double> / nested
                // map sub-blob / tensor sub-blob — all have offsets internal to the source blob
                // OR no internal offsets at all, so byte-copy preserves correctness).
                const std::span<const std::byte> src{other._blob + sourceEntry.payloadOffset, sourceEntry.payloadLength};
                const auto                       newOffset = _appendPayloadSafe(src);
                _entries[destIndex].payloadOffset          = newOffset; // re-index after possible grow
            }
            committed = true;
            moved[i]  = true;
        }
        // Batch-compact source: write surviving entries to the front, drop trailing slack.
        std::uint16_t writeIdx = 0U;
        for (std::uint16_t i = 0U; i < nSrc; ++i) {
            if (!moved[i]) {
                if (writeIdx != i) {
                    other._entries[writeIdx] = other._entries[i];
                }
                ++writeIdx;
            }
        }
        other._header->entryCount = writeIdx;
    }

    [[nodiscard]] constexpr const_iterator         begin() const noexcept { return const_iterator{this, 0U}; }
    [[nodiscard]] constexpr const_iterator         end() const noexcept { return const_iterator{this, static_cast<std::uint16_t>(_header ? _header->entryCount : 0U)}; }
    [[nodiscard]] constexpr const_iterator         cbegin() const noexcept { return begin(); }
    [[nodiscard]] constexpr const_iterator         cend() const noexcept { return end(); }
    [[nodiscard]] constexpr const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator{end()}; }
    [[nodiscard]] constexpr const_reverse_iterator rend() const noexcept { return const_reverse_iterator{begin()}; }
    [[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }
    [[nodiscard]] constexpr const_reverse_iterator crend() const noexcept { return rend(); }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return _header ? _header->entryCount : 0U; }
    [[nodiscard]] constexpr bool        empty() const noexcept { return size() == 0U; }

    [[nodiscard]] constexpr std::span<const std::byte> blob() const noexcept {
        if (!_header) {
            return {};
        }
        return {_blob, _header->totalSize};
    }

    [[nodiscard]] constexpr std::pmr::memory_resource* resource() const noexcept { return _resource; }

    // Deserialise a ValueMap from a byte span (typically a blob received over the wire or
    // mapped from disk). Validates magic / version / alignment / bounds / payload offsets
    // before copying the bytes into a fresh owned blob via `resource`. LE-only.
    [[nodiscard]] static std::expected<ValueMap, DeserialiseError> from_blob(std::span<const std::byte> bytes, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) {
        if (bytes.size() < sizeof(Header)) {
            return std::unexpected{DeserialiseError::TooSmall};
        }
        if ((reinterpret_cast<std::uintptr_t>(bytes.data()) & (kBlobAlignment - 1UZ)) != 0UZ) {
            return std::unexpected{DeserialiseError::AlignmentViolation};
        }
        const auto* hdr = std::launder(reinterpret_cast<const Header*>(bytes.data()));
        if (std::memcmp(hdr->magic, kBlobMagic.data(), kBlobMagic.size()) != 0) {
            return std::unexpected{DeserialiseError::MagicMismatch};
        }
        if (hdr->version != kBlobVersion) {
            return std::unexpected{DeserialiseError::VersionUnsupported};
        }
        if (hdr->totalSize > bytes.size() || hdr->payloadOffset > hdr->totalSize) {
            return std::unexpected{DeserialiseError::CorruptOffset};
        }
        const auto entryArrayBytes = static_cast<std::size_t>(hdr->entryCount) * sizeof(PackedEntry);
        if (sizeof(Header) + entryArrayBytes > hdr->payloadOffset) {
            return std::unexpected{DeserialiseError::CorruptOffset};
        }
        const auto* srcEntries = std::launder(reinterpret_cast<const PackedEntry*>(bytes.data() + sizeof(Header)));
        for (std::uint16_t i = 0U; i < hdr->entryCount; ++i) {
            const auto& e = srcEntries[i];
            if ((e.flags & kEntryFlagOffsetLength) != 0U) {
                if (e.payloadOffset < hdr->payloadOffset || e.payloadOffset > hdr->totalSize || e.payloadLength > hdr->totalSize - e.payloadOffset) {
                    return std::unexpected{DeserialiseError::CorruptOffset};
                }
            }
            if (e.keyId == keys::kSpilledKeyId) {
                const auto [keyOffset, keyLength] = detail::readSpilledKeyOffsetLength(e);
                if (keyOffset < hdr->payloadOffset || keyOffset > hdr->totalSize || keyLength > hdr->totalSize - keyOffset) {
                    return std::unexpected{DeserialiseError::CorruptOffset};
                }
            }
        }
        // All bounds OK — copy bytes into our own owned blob.
        ValueMap out{resource ? resource : std::pmr::get_default_resource()};
        out._deallocateBlob();
        out._blob     = static_cast<std::byte*>(out._resource->allocate(hdr->totalSize, kBlobAlignment));
        out._capacity = hdr->totalSize;
        std::memcpy(out._blob, bytes.data(), hdr->totalSize);
        out._header  = std::launder(reinterpret_cast<Header*>(out._blob));
        out._entries = std::launder(reinterpret_cast<PackedEntry*>(out._blob + sizeof(Header)));
        return out;
    }

    void freeze() noexcept {
        if (_header) {
            _header->flags |= kHeaderFlagFrozen;
        }
    }
    [[nodiscard]] constexpr bool is_frozen() const noexcept { return _header && (_header->flags & kHeaderFlagFrozen); }

    // Order-independent equality — matches std::pmr::unordered_map's operator==. Two maps
    // are equal iff they have the same set of (key, value) pairs. Used by tag-equivalence
    // checks (e.g. blocks/testing/TagMonitors.hpp `tag1.map == tag2.map`). Not `noexcept`
    // because Value::operator== isn't (nested / tensor values may allocate during compare).
    // Early returns: same instance / aliased blob / size mismatch — to make the O(N²) full
    // content comparison rare in practice.
    [[nodiscard]] bool operator==(const ValueMap& other) const {
        if (this == &other) {
            return true;
        }
        if (_blob == other._blob && _blob != nullptr) {
            return true;
        }
        if (size() != other.size()) {
            return false;
        }
        for (const auto& [key, value] : *this) {
            const auto otherIt = other.find(key);
            if (otherIt == other.end() || !(value == (*otherIt).second)) {
                return false;
            }
        }
        return true;
    }

    // Snapshots into ordinary STL maps. Useful for migration / interop with code that
    // expects std::map or std::unordered_map. Each entry's Value is copy-constructed.
    [[nodiscard]] std::map<std::string, Value, std::less<>> to_std_map() const {
        std::map<std::string, Value, std::less<>> out;
        for (const auto& [key, value] : *this) {
            out.emplace(std::string{key}, value);
        }
        return out;
    }

    [[nodiscard]] std::unordered_map<std::string, Value> to_std_unordered_map() const {
        std::unordered_map<std::string, Value> out;
        out.reserve(size());
        for (const auto& [key, value] : *this) {
            out.emplace(std::string{key}, value);
        }
        return out;
    }

private:
    // tag-dispatched view-mode constructor. Public-facing factory: makeView().
    struct ViewModeTag {};
    ValueMap(ViewModeTag, std::span<const std::byte> bytes) noexcept : _blob(const_cast<std::byte*>(bytes.data())), _resource(nullptr), _capacity(static_cast<std::uint32_t>(bytes.size())), _header(nullptr), _entries(nullptr) {
        if (bytes.size() >= sizeof(Header)) {
            _header  = std::launder(reinterpret_cast<Header*>(_blob));
            _entries = std::launder(reinterpret_cast<PackedEntry*>(_blob + sizeof(Header)));
        }
    }

    // Mutator guard for view-mode (resource == nullptr). Per Q5 hybrid policy: assert in
    // debug builds (catches misuse early), no-op in release builds (caller's mutator returns
    // an empty result). Returns true iff the call should bail out.
    [[nodiscard]] bool _isViewAndAssertNoMutation() const noexcept {
        assert(!is_view() && "ValueMap: cannot mutate a view-mode (resource == nullptr) ValueMap");
        return is_view();
    }

    // layout invariants: see comment in the skeleton block
    std::byte*                 _blob     = nullptr;
    std::pmr::memory_resource* _resource = std::pmr::get_default_resource();
    std::uint32_t              _capacity = 0U;
    Header*                    _header   = nullptr;
    PackedEntry*               _entries  = nullptr;

    // Returns the index of the entry with the given canonical id, or size() if absent.
    template<std::uint16_t Id>
    [[nodiscard]] constexpr std::uint16_t _findCanonicalIndex() const noexcept {
        const std::span entries{_entries, _header ? _header->entryCount : std::uint16_t{0}};
        const auto      it = std::ranges::find(entries, Id, &PackedEntry::keyId);
        return static_cast<std::uint16_t>(it - entries.begin());
    }

    // Debug-only sanity check: stored value type matches the canonical binding for `Id`.
    // Catches `m.emplace("sample_rate", std::uint64_t{42})` followed by `m.at<"sample_rate">()`
    // type-mismatch UB before it silently aliases bytes through the wrong type.
    template<std::uint16_t Id>
    static void _debugAssertCanonicalType([[maybe_unused]] const PackedEntry& e) noexcept {
        if constexpr (gr::meta::kDebugBuild) {
            constexpr auto expectedVt = keys::boundTypeOf<Id>;
            assert(static_cast<Value::ValueType>(e.valueType) == expectedVt //
                   && "ValueMap::at<Name>: stored type does not match canonical binding for Name");
        }
    }

    // erase + return the iterator to the next surviving entry. Used by merge().
    [[nodiscard]] const_iterator erase_one(const_iterator it) {
        if (it._map != this || it._index >= size()) {
            return end();
        }
        const auto removedIndex = it._index;
        erase(it);
        return const_iterator{this, removedIndex};
    }

    void _allocateBlob(std::uint32_t entries, std::uint32_t payloadReserve, std::uint32_t slackBytes) {
        const auto entryBytes   = static_cast<std::uint32_t>(entries) * static_cast<std::uint32_t>(sizeof(PackedEntry));
        const auto payloadStart = static_cast<std::uint32_t>(sizeof(Header)) + entryBytes;
        const auto total        = payloadStart + payloadReserve + slackBytes;
        _blob                   = static_cast<std::byte*>(_resource->allocate(total, kBlobAlignment));
        _capacity               = total;
        _header                 = std::launder(reinterpret_cast<Header*>(_blob));
        _entries                = std::launder(reinterpret_cast<PackedEntry*>(_blob + sizeof(Header)));
        std::memcpy(_header->magic, kBlobMagic.data(), kBlobMagic.size());
        _header->version       = kBlobVersion;
        _header->flags         = 0U;
        _header->entryCount    = 0U;
        _header->totalSize     = total;
        _header->payloadOffset = payloadStart;
        _header->payloadUsed   = 0U;
        _header->slackBytes    = slackBytes;
        _header->reserved      = 0U;
    }

    void _deallocateBlob() noexcept {
        if (_blob && _resource) {
            _resource->deallocate(_blob, _capacity, kBlobAlignment);
        }
        _blob     = nullptr;
        _header   = nullptr;
        _entries  = nullptr;
        _capacity = 0U;
    }

    void _copyBlobFrom(const ValueMap& other) {
        if (!other._header || other._capacity == 0U) {
            _allocateBlob(8U, 0U, 0U);
            return;
        }
        const auto total = other._capacity;
        _blob            = static_cast<std::byte*>(_resource->allocate(total, kBlobAlignment));
        _capacity        = total;
        std::memcpy(_blob, other._blob, total);
        _header  = std::launder(reinterpret_cast<Header*>(_blob));
        _entries = std::launder(reinterpret_cast<PackedEntry*>(_blob + sizeof(Header)));
    }

    [[nodiscard]] std::uint32_t _entryCapacity() const noexcept {
        if (!_header) {
            return 0U;
        }
        return (_header->payloadOffset - static_cast<std::uint32_t>(sizeof(Header))) / static_cast<std::uint32_t>(sizeof(PackedEntry));
    }

    [[nodiscard]] std::uint32_t _payloadCapacity() const noexcept {
        if (!_header) {
            return 0U;
        }
        return _capacity - _header->payloadOffset - _header->slackBytes;
    }

    // Grow (or, if shrink=true, right-size) the blob so that it has room for
    // `newEntryArray` entries and at least `newPayloadBytes` payload.
    void _grow(std::uint32_t requestedEntries, std::uint32_t newPayloadBytes, bool shrink = false) {
        const std::uint16_t curEntries = _header ? _header->entryCount : std::uint16_t{0};
        const std::uint32_t curPayload = _header ? _header->payloadUsed : 0U;
        const std::uint32_t slack      = _header ? _header->slackBytes : 0U;

        std::uint32_t targetEntryCap = std::max<std::uint32_t>(requestedEntries, curEntries);
        std::uint32_t targetPayload  = std::max<std::uint32_t>(newPayloadBytes, curPayload);

        if (!shrink) {
            const auto curEntryCap   = _entryCapacity();
            const auto curPayloadCap = _payloadCapacity();
            targetEntryCap           = std::max<std::uint32_t>(targetEntryCap, std::max<std::uint32_t>(curEntryCap + curEntryCap / 2U, 8U));
            targetPayload            = std::max<std::uint32_t>(targetPayload, std::max<std::uint32_t>(curPayloadCap + curPayloadCap / 2U, 0U));
        }

        const auto entryBytes   = targetEntryCap * static_cast<std::uint32_t>(sizeof(PackedEntry));
        const auto payloadStart = static_cast<std::uint32_t>(sizeof(Header)) + entryBytes;
        const auto total        = payloadStart + targetPayload + slack;

        auto* newBlob       = static_cast<std::byte*>(_resource->allocate(total, kBlobAlignment));
        auto* newHeader     = std::launder(reinterpret_cast<Header*>(newBlob));
        auto* newEntryArray = std::launder(reinterpret_cast<PackedEntry*>(newBlob + sizeof(Header)));
        auto* newPayload    = newBlob + payloadStart;

        // header
        std::memcpy(newHeader->magic, kBlobMagic.data(), kBlobMagic.size());
        newHeader->version       = kBlobVersion;
        newHeader->flags         = _header ? _header->flags : 0U;
        newHeader->entryCount    = curEntries;
        newHeader->totalSize     = total;
        newHeader->payloadOffset = payloadStart;
        newHeader->payloadUsed   = curPayload;
        newHeader->slackBytes    = slack;
        newHeader->reserved      = 0U;

        // entries — relocate any payload-bearing offsets (value payloads + spilled keys) into the
        // new blob's payload region.
        if (_entries && curEntries > 0U) {
            std::memcpy(newEntryArray, _entries, curEntries * sizeof(PackedEntry));
            const auto oldPayloadOffset = _header->payloadOffset;
            std::ranges::for_each(std::span<PackedEntry>{newEntryArray, curEntries}, [oldPayloadOffset, payloadStart](PackedEntry& entry) {
                if ((entry.flags & kEntryFlagOffsetLength) != 0U) {
                    entry.payloadOffset = payloadStart + (entry.payloadOffset - oldPayloadOffset);
                }
                if (entry.keyId == keys::kSpilledKeyId) {
                    auto [keyOffset, keyLength] = detail::readSpilledKeyOffsetLength(entry);
                    keyOffset                   = payloadStart + (keyOffset - oldPayloadOffset);
                    detail::setSpilledKey(entry, keyOffset, keyLength);
                }
            });
        }

        // payload
        if (_header && curPayload > 0U) {
            std::memcpy(newPayload, _blob + _header->payloadOffset, curPayload);
        }

        _deallocateBlob();
        _blob     = newBlob;
        _capacity = total;
        _header   = newHeader;
        _entries  = newEntryArray;
    }

    // Reserve `bytes` in the payload pool and return the absolute offset into
    // _blob. May grow the blob (invalidates all prior pointers into it).
    //
    // EXCEPTION (by design): throws std::length_error if the request would overflow the
    // 4 GiB blob limit (uint32 payloadUsed / payloadOffset / totalSize fields). Library /
    // SYCL code that must stay exception-free should pre-validate against the per-source
    // size cap before calling. Bubbles up through every public mutator (emplace,
    // insert_or_assign, copy/move-assign, conversion ctors, merge).
    [[nodiscard]] std::uint32_t _appendPayload(std::span<const std::byte> src) {
        if (!_header) {
            _allocateBlob(8U, static_cast<std::uint32_t>(std::min<std::size_t>(src.size(), std::numeric_limits<std::uint32_t>::max())), 0U);
        }
        const std::size_t curUsed = _header->payloadUsed;
        if (src.size() > std::numeric_limits<std::uint32_t>::max() - curUsed) {
            throw std::length_error{"gr::pmt::ValueMap::_appendPayload: payload would exceed 4 GiB blob limit"};
        }
        const auto requested = static_cast<std::uint32_t>(src.size());
        if (curUsed + requested > _payloadCapacity()) {
            _grow(_header->entryCount, static_cast<std::uint32_t>(curUsed) + requested);
        }
        const auto offset = _header->payloadOffset + _header->payloadUsed;
        std::memcpy(_blob + offset, src.data(), src.size());
        _header->payloadUsed += requested;
        return offset;
    }

    // Self-alias-safe append: if `src` references our own _blob (e.g. `m.emplace("k", m)`,
    // or re-emplacing an iterator-yielded inline-key string_view), snapshot the bytes into
    // a resource-owned temp BEFORE calling _appendPayload — _appendPayload may grow / free
    // _blob, which would otherwise dangle src.data() between snapshot and memcpy.
    // Pointer comparison uses std::less<> for a standard-blessed total order across
    // unrelated allocations (built-in </>= are implementation-defined for cross-alloc).
    [[nodiscard]] std::uint32_t _appendPayloadSafe(std::span<const std::byte> src) {
        const bool aliasesSelf = (_blob != nullptr) //
                                 && !std::less<>{}(src.data(), _blob) && std::less<>{}(src.data(), _blob + _capacity);
        if (!aliasesSelf) {
            return _appendPayload(src);
        }
        std::pmr::vector<std::byte> snapshot(src.begin(), src.end(), _resource);
        return _appendPayload(std::span<const std::byte>{snapshot});
    }

    void _ensureEntrySlot() {
        if (!_header) {
            _allocateBlob(8U, 0U, 0U);
        }
        if (_header->entryCount >= _entryCapacity()) {
            _grow(_header->entryCount + 1U, _header->payloadUsed);
        }
    }

    // Resolve an entry's key string view (handles canonical / inline / spilled).
    [[nodiscard]] std::string_view _readKey(const PackedEntry& e) const noexcept {
        if (e.keyId == keys::kInlineKeyId) {
            return detail::readInlineKey(e);
        }
        if (e.keyId == keys::kSpilledKeyId) {
            return detail::readSpilledKey(_blob, e);
        }
        if (e.keyId != 0U && e.keyId <= keys::kCanonical.size()) {
            return keys::kCanonical[e.keyId - 1U].name; // dense range fast path
        }
        for (const auto& k : keys::kCanonical) { // sparse fallback
            if (k.id == e.keyId) {
                return k.name;
            }
        }
        return {};
    }

    // Match an entry against a key. `canonicalId` is the resolved canonical id
    // (kIdUnknown if the caller did not resolve; we still match inline / spilled keys).
    [[nodiscard]] bool _entryMatches(const PackedEntry& e, std::string_view key, std::uint16_t canonicalId) const noexcept {
        if (canonicalId != keys::kIdUnknown && e.keyId == canonicalId) {
            return true;
        }
        if (e.keyId == keys::kInlineKeyId) {
            return detail::readInlineKey(e) == key;
        }
        if (e.keyId == keys::kSpilledKeyId) {
            return detail::readSpilledKey(_blob, e) == key;
        }
        return false;
    }

    [[nodiscard]] Value _entryToValue(const PackedEntry& e) const {
        // view-mode (_resource == nullptr) decodes against the global default resource so that any
        // sub-allocations (Tensor<Value> elements, nested-Map decoder vectors) have a real resource
        // to allocate from.
        return detail::decodeEntry(e, _blob, _resource ? _resource : std::pmr::get_default_resource());
    }

    template<typename Cpp>
    [[nodiscard]] std::optional<Cpp> _readTyped(const PackedEntry& e) const {
        if constexpr (detail::InlineScalar<Cpp>) {
            if (static_cast<Value::ValueType>(e.valueType) != detail::cppToValueType<Cpp>()) {
                return std::nullopt;
            }
            return detail::readInlineScalar<Cpp>(e);
        } else if constexpr (std::same_as<Cpp, std::string_view>) {
            if (static_cast<Value::ValueType>(e.valueType) != Value::ValueType::String) {
                return std::nullopt;
            }
            return _readStringView(e);
        } else {
            return std::nullopt;
        }
    }

    [[nodiscard]] std::string_view _readStringView(const PackedEntry& e) const noexcept {
        if ((e.flags & kEntryFlagOffsetLength) == 0U) {
            return {};
        }
        return {reinterpret_cast<const char*>(_blob + e.payloadOffset), e.payloadLength};
    }

    // Set the entry-at-`index`'s keyId + inlineKey based on `key`. Long keys spill to the payload
    // pool; the spill's _appendPayloadSafe can grow the blob, so we re-fetch the entry reference
    // via `index` after the append (the previous PackedEntry& would dangle). Returns true
    // unconditionally; allocation failures bubble up through _appendPayloadSafe's std::length_error.
    [[nodiscard]] bool _writeKey(std::uint16_t index, std::string_view key) {
        const auto canonicalId = keys::lookupId(key);
        if (canonicalId != keys::kIdUnknown) {
            PackedEntry& e = _entries[index];
            e.keyId        = canonicalId;
            std::memset(e.inlineKey, 0, sizeof(e.inlineKey));
            return true;
        }
        if (key.size() <= kMaxInlineKeyLength) {
            detail::setInlineKey(_entries[index], key);
            return true;
        }
        const auto bytes  = std::span<const std::byte>{reinterpret_cast<const std::byte*>(key.data()), key.size()};
        const auto offset = _appendPayloadSafe(bytes);
        detail::setSpilledKey(_entries[index], offset, static_cast<std::uint32_t>(key.size()));
        return true;
    }

    template<typename V>
    [[nodiscard]] const_iterator _insertNew(std::string_view key, V&& value) {
        _ensureEntrySlot();
        const auto index = _header->entryCount;
        std::memset(&_entries[index], 0, sizeof(PackedEntry));

        if (!_writeKey(index, key)) {
            return end();
        }

        // Commit the slot BEFORE writing the value so that a payload-triggered _grow()
        // copies the new entry's key (otherwise grow sees entryCount=N and the
        // newly-written-but-uncommitted slot is dropped). Roll back the bump if the
        // value-write throws (bad_alloc / length_error from _appendPayload), so callers
        // see strong exception safety: either the entry is fully inserted or the map is
        // unchanged.
        ++_header->entryCount;
        bool       committed = false;
        const auto rollback  = gr::on_scope_exit{[&]() noexcept {
            if (!committed) {
                --_header->entryCount;
            }
        }};
        _assignValueAt(index, std::forward<V>(value));
        committed = true;
        return const_iterator{this, index};
    }

    // Write value into entry at `index`, re-fetching the PackedEntry reference after
    // any grow-triggered reallocation invalidates outstanding pointers.
    template<typename V>
    void _assignValueAt(std::uint16_t index, V&& value) {
        using U = std::remove_cvref_t<V>;
        if constexpr (detail::InlineScalar<U>) {
            PackedEntry& e  = _entries[index];
            e.valueType     = static_cast<std::uint8_t>(detail::cppToValueType<U>());
            e.flags         = kEntryFlagInlineScalar;
            e.payloadOffset = 0U;
            e.payloadLength = 0U;
            detail::writeInlineScalar<U>(e, value);
        } else if constexpr (detail::PayloadScalar<U>) {
            // 16-byte scalar (currently only std::complex<double>) — must spill to payload pool.
            const auto   bytes  = std::span<const std::byte>{reinterpret_cast<const std::byte*>(&value), sizeof(U)};
            const auto   offset = _appendPayload(bytes);
            PackedEntry& e      = _entries[index];
            e.valueType         = static_cast<std::uint8_t>(detail::cppToValueType<U>());
            e.flags             = kEntryFlagOffsetLength;
            e.payloadOffset     = offset;
            e.payloadLength     = static_cast<std::uint32_t>(sizeof(U));
            e.inlineValue       = 0U;
        } else if constexpr (std::same_as<U, ValueMap>) {
            // Nested ValueMap — copy the source's blob() bytes into our payload pool. On read,
            // _entryToValue rehydrates a Value::Map by walking the sub-blob. _appendPayloadSafe
            // protects against self-emplace (`m.emplace("k", m)`) where _grow would otherwise
            // free the bytes we're about to memcpy from.
            const auto   srcBlob = value.blob();
            const auto   srcSize = srcBlob.size();
            const auto   offset  = _appendPayloadSafe(srcBlob);
            PackedEntry& e       = _entries[index];
            e.valueType          = static_cast<std::uint8_t>(Value::ValueType::Value);
            e.flags              = kEntryFlagOffsetLength | kEntryFlagNestedMap;
            e.payloadOffset      = offset;
            e.payloadLength      = static_cast<std::uint32_t>(srcSize);
            e.inlineValue        = 0U;
        } else if constexpr (detail::StringLike<V>) {
            const std::string_view sv{std::forward<V>(value)};
            const auto             srcSize = sv.size();
            const auto             span    = std::span<const std::byte>{reinterpret_cast<const std::byte*>(sv.data()), srcSize};
            const auto             offset  = _appendPayloadSafe(span); // may reallocate; snapshot if sv aliases our payload pool
            // Re-index after possible grow.
            PackedEntry& e  = _entries[index];
            e.valueType     = static_cast<std::uint8_t>(Value::ValueType::String);
            e.flags         = kEntryFlagOffsetLength;
            e.payloadOffset = offset;
            e.payloadLength = static_cast<std::uint32_t>(srcSize);
            e.inlineValue   = 0U;
        } else if constexpr (gr::TensorLike<U>) {
            // Tensor<T> — encode the sub-blob (header + extents + element data, recursive
            // for Tensor<Value> elements that are themselves nested maps / tensors) and
            // append it to our payload pool. _appendPayloadSafe protects against the
            // tensor-source bytes living inside our own pool.
            auto         tensorBlob = detail::encodeTensorBlob(value, _resource, /*depth=*/0U);
            const auto   srcSize    = tensorBlob.size();
            const auto   offset     = _appendPayloadSafe(std::span<const std::byte>{tensorBlob});
            PackedEntry& e          = _entries[index];
            e.valueType             = static_cast<std::uint8_t>(Value::ValueType::Value);
            e.flags                 = kEntryFlagOffsetLength | kEntryFlagTensor;
            e.payloadOffset         = offset;
            e.payloadLength         = static_cast<std::uint32_t>(srcSize);
            e.inlineValue           = 0U;
        } else if constexpr (std::same_as<U, Value>) {
            // Polymorphic gr::pmt::Value — runtime-dispatch on its value/container type and
            // delegate back to the typed _assignValueAt overload. Mirrors the converter-from-Map
            // logic in ValueMap(const Map&, resource) so emplace/insert_or_assign(K, Value) works.
            if (value.is_map()) {
                if (auto srcMap = value.template get_if<Value::Map>()) {
                    _assignValueAt(index, srcMap->owned(_resource));
                }
            } else if (value.is_tensor()) {
                detail::dispatchValueType(value.value_type(), [&]<typename T>(std::type_identity<T>) {
                    if constexpr (detail::TensorElementType<T>) {
                        if (auto tensor = value.template get_if<gr::Tensor<T>>()) {
                            _assignValueAt(index, *tensor);
                        }
                    }
                });
            } else {
                detail::dispatchValueType(value.value_type(), [&]<typename T>(std::type_identity<T>) {
                    if constexpr (detail::InlineScalar<T>) {
                        _assignValueAt(index, value.template value_or<T>(T{}));
                    } else if constexpr (detail::PayloadScalar<T>) {
                        _assignValueAt(index, value.template value_or<T>(T{}));
                    } else if constexpr (std::same_as<T, std::string_view>) {
                        _assignValueAt(index, value.value_or(std::string_view{}));
                    }
                });
            }
        } else {
            static_assert(detail::InlineScalar<U> || detail::PayloadScalar<U> || detail::StringLike<V> || std::same_as<U, ValueMap> || gr::TensorLike<U> || std::same_as<U, Value>, "ValueMap: insert value must be an inline scalar, std::complex<double>, string-like, a nested ValueMap, a gr::Tensor<T>, or gr::pmt::Value.");
        }
    }

    template<std::uint16_t Id, typename V>
    void _upsertCanonical(V&& value) {
        const std::span entries{_entries, _header ? _header->entryCount : std::uint16_t{0}};
        if (const auto it = std::ranges::find(entries, Id, &PackedEntry::keyId); it != entries.end()) {
            _assignValueAt(static_cast<std::uint16_t>(it - entries.begin()), std::forward<V>(value));
            return;
        }
        _ensureEntrySlot();
        const auto   index = _header->entryCount;
        PackedEntry& e     = _entries[index];
        std::memset(&e, 0, sizeof(PackedEntry));
        e.keyId = Id;
        // Commit the slot BEFORE writing the value (see _insertNew comment).
        ++_header->entryCount;
        _assignValueAt(index, std::forward<V>(value));
    }
};

inline ValueMap::const_iterator::value_type ValueMap::const_iterator::operator*() const {
    const auto&            e   = _map->_entries[_index];
    const std::string_view key = _map->_readKey(e); // handles canonical / inline / spilled
    // Inline switch on valueType — bypasses dispatchValueType's lambda chain and Value's
    // ensure_resource/storage-zeroing. For inline scalars the Value ctor is the cheapest one
    // (single bytewise read); for Strings it's view-mode (alloc-free, aliases blob payload).
    auto*      res = _map->_resource;
    const auto vt  = static_cast<Value::ValueType>(e.valueType);
    switch (vt) {
    case Value::ValueType::Bool: return {key, Value{detail::readInlineScalar<bool>(e), res}};
    case Value::ValueType::Int8: return {key, Value{detail::readInlineScalar<std::int8_t>(e), res}};
    case Value::ValueType::Int16: return {key, Value{detail::readInlineScalar<std::int16_t>(e), res}};
    case Value::ValueType::Int32: return {key, Value{detail::readInlineScalar<std::int32_t>(e), res}};
    case Value::ValueType::Int64: return {key, Value{detail::readInlineScalar<std::int64_t>(e), res}};
    case Value::ValueType::UInt8: return {key, Value{detail::readInlineScalar<std::uint8_t>(e), res}};
    case Value::ValueType::UInt16: return {key, Value{detail::readInlineScalar<std::uint16_t>(e), res}};
    case Value::ValueType::UInt32: return {key, Value{detail::readInlineScalar<std::uint32_t>(e), res}};
    case Value::ValueType::UInt64: return {key, Value{detail::readInlineScalar<std::uint64_t>(e), res}};
    case Value::ValueType::Float32: return {key, Value{detail::readInlineScalar<float>(e), res}};
    case Value::ValueType::Float64: return {key, Value{detail::readInlineScalar<double>(e), res}};
    case Value::ValueType::ComplexFloat32: return {key, Value{detail::readInlineScalar<std::complex<float>>(e), res}};
    case Value::ValueType::String:
        if ((e.flags & kEntryFlagOffsetLength) != 0U) {
            // View-mode (Phase 1d step 3): alloc-free; aliases the blob payload bytes.
            return {key, Value::makeView(Value::ValueType::String, Value::ContainerType::String, _map->_blob + e.payloadOffset, e.payloadLength, res)};
        }
        return {key, Value{res}};
    default:
        // ComplexFloat64 / nested ValueMap / Tensor — still allocates via decodeEntry.
        // (View-mode iter for these types deferred — copy/value_or callers don't handle view-mode
        // owning materialisation without API changes, and the iter perf gain doesn't yet justify
        // the migration cost.)
        return {key, _map->_entryToValue(e)};
    }
}

namespace detail {

// Defined here (not inside the earlier detail:: block) because the nested-map case
// needs the full ValueMap definition above to construct a sub-blob from a Value::Map.
inline void encodeTensorElement(std::pmr::vector<std::byte>& out, const Value& val, std::pmr::memory_resource* resource, std::uint32_t depth) {
    PackedTensorElement elem;
    std::memset(&elem, 0, sizeof(elem));
    elem.valueType = static_cast<std::uint8_t>(val.value_type());

    std::pmr::vector<std::byte> payload(resource);

    if (val.is_map()) {
        if (auto srcMap = val.template get_if<Value::Map>()) {
            const auto srcBlob = srcMap->blob();
            payload.assign(srcBlob.begin(), srcBlob.end());
        }
        elem.flags = kEntryFlagOffsetLength | kEntryFlagNestedMap;
    } else if (val.is_tensor() && depth + 1U < kMaxDecodeDepth) {
        dispatchValueType(val.value_type(), [&]<typename T>(std::type_identity<T>) {
            if constexpr (TensorElementType<T>) {
                if (auto nested = val.template get_if<gr::Tensor<T>>()) {
                    auto nestedBlob = encodeTensorBlob(*nested, resource, depth + 1U);
                    payload         = std::move(nestedBlob);
                }
            }
        });
        elem.flags = kEntryFlagOffsetLength | kEntryFlagTensor;
    } else if (val.is_string()) {
        const auto sv = val.value_or(std::string_view{});
        payload.assign(reinterpret_cast<const std::byte*>(sv.data()), reinterpret_cast<const std::byte*>(sv.data()) + sv.size());
        elem.flags = kEntryFlagOffsetLength;
    } else if (val.value_type() == Value::ValueType::ComplexFloat64) {
        if (const auto* p = val.template get_if<std::complex<double>>()) {
            payload.resize(sizeof(*p));
            std::memcpy(payload.data(), p, sizeof(*p));
        }
        elem.flags = kEntryFlagOffsetLength;
    } else {
        dispatchValueType(val.value_type(), [&]<typename T>(std::type_identity<T>) {
            if constexpr (InlineScalar<T>) {
                if (const auto* p = val.template get_if<T>()) {
                    PackedEntry tmp;
                    writeInlineScalar<T>(tmp, *p);
                    elem.inlineValue = tmp.inlineValue;
                }
            }
        });
        elem.flags = kEntryFlagInlineScalar;
    }

    elem.payloadLength = static_cast<std::uint32_t>(payload.size());

    const auto headerBegin = reinterpret_cast<const std::byte*>(&elem);
    out.insert(out.end(), headerBegin, headerBegin + sizeof(elem));
    out.insert(out.end(), payload.begin(), payload.end());
}

} // namespace detail

// Phase 1e Step C: nested-map decoder helper. Forward-declared at top of file (in
// `gr::pmt::detail` namespace); defined here after ValueMap is complete since `Value::Map`
// is a forward-decl alias to ValueMap until this point.
namespace detail {
inline Value buildNestedMapValue(const std::byte* base, const PackedEntry* nestedEntries, std::uint16_t declaredCount, std::pmr::memory_resource* resource, std::uint32_t depth) {
    Value::Map map{resource};
    for (std::uint16_t i = 0U; i < declaredCount; ++i) {
        const auto&      ne = nestedEntries[i];
        std::string_view key;
        if (ne.keyId == keys::kInlineKeyId) {
            key = readInlineKey(ne);
        } else if (ne.keyId == keys::kSpilledKeyId) {
            key = readSpilledKey(base, ne); // sub-blob–relative
        } else if (const auto it = std::ranges::find(keys::kCanonical, ne.keyId, &keys::CanonicalKey::id); it != keys::kCanonical.end()) {
            key = it->name;
        }
        map.insert_or_assign(std::string_view{key}, decodeEntry(ne, base, resource, depth + 1U));
    }
    return Value{map, resource};
}
} // namespace detail

// Phase 1e Step C: out-of-class definition of Value::init_from_map. Lives here (not in
// Value.hpp) because the body needs `Map = ValueMap` to be fully defined for sizeof / new /
// member access — and ValueMap.hpp is the canonical full-definition site (Value.hpp can only
// forward-declare ValueMap to avoid the circular include).
template<detail::ExternalValueMap T>
inline void Value::init_from_map(T&& map) {
    using DecayedMap           = std::remove_cvref_t<T>;
    using MappedType           = typename DecayedMap::mapped_type;
    constexpr bool isValueType = std::same_as<MappedType, Value>;

    set_types(ValueType::Value, ContainerType::Map);

    // Q1.B: build a temporary ValueMap with the entries, then copy its blob bytes into Value's
    // own byte-blob storage (no heap ValueMap object kept).
    Map tmp(_resource, static_cast<std::uint32_t>(std::max<std::size_t>(map.size(), std::size_t{8})));
    for (const auto& [key, val] : map) {
        if constexpr (isValueType) {
            tmp.insert_or_assign(std::string_view{key}, val);
        } else {
            tmp.insert_or_assign(std::string_view{key}, Value{val, _resource});
        }
    }

    const auto srcBlob = tmp.blob();
    if (srcBlob.empty()) {
        _storage.ptr   = nullptr;
        _payloadLength = 0U;
        return;
    }
    auto* bytes = static_cast<std::byte*>(_resource->allocate(srcBlob.size(), kBlobAlignment));
    std::memcpy(bytes, srcBlob.data(), srcBlob.size());
    _storage.ptr   = bytes;
    _payloadLength = static_cast<std::uint32_t>(srcBlob.size());
}

// Q1.B: out-of-class definition of Value::get_if<Tensor<T>>() — decode the byte-blob into a fresh
// owning Tensor<T> via decodeTensorBlob and return it inside std::optional. Allocates per call.
template<typename T>
requires gr::TensorLike<T>
inline std::optional<T> Value::get_if() const noexcept {
    using ElemT = std::remove_const_t<typename T::value_type>;
    if (!is_tensor() || value_type() != get_value_type<ElemT>() || _storage.ptr == nullptr || _payloadLength == 0U) {
        return std::nullopt;
    }
    auto*       res     = _resource ? _resource : std::pmr::get_default_resource();
    const auto* base    = static_cast<const std::byte*>(_storage.ptr);
    Value       decoded = detail::decodeTensorBlob(base, _payloadLength, res, /*depth=*/0U);
    if (!decoded.is_tensor()) {
        return std::nullopt;
    }
    // decoded is a Value with byte-blob Tensor storage; extract Tensor<ElemT> via the byte-blob
    // header. For fixed-size T, alias-and-copy the data section. For T=Value, walk per-element
    // PackedTensorElement headers and decode each into a Value.
    const auto rank          = static_cast<std::uint8_t>(base[1]);
    const auto encodingFlags = static_cast<std::uint8_t>(base[2]);
    if (rank > kMaxTensorRank) {
        return std::nullopt;
    }
    std::array<std::size_t, kMaxTensorRank> extentsBuf{};
    for (std::size_t i = 0UZ; i < rank; ++i) {
        std::uint32_t ext;
        std::memcpy(&ext, base + kTensorBlobHeaderSize + 4UZ * i, sizeof(ext));
        extentsBuf[i] = static_cast<std::size_t>(ext);
    }
    const std::span<const std::size_t> extents{extentsBuf.data(), rank};
    std::uint32_t                      elementCount;
    std::memcpy(&elementCount, base + 4, sizeof(elementCount));

    (void)encodingFlags;
    if constexpr (std::same_as<ElemT, Value>) {
        // Variable-size element encoding: walk per-element headers.
        gr::pmr::vector<Value, true> elems(elementCount, Value{res}, res);
        const auto*                  elementData  = base + kTensorBlobHeaderSize + 4UZ * rank;
        const std::uint32_t          elementBytes = _payloadLength - static_cast<std::uint32_t>(kTensorBlobHeaderSize + 4UZ * rank);
        std::uint32_t                offset       = 0U;
        for (std::uint32_t i = 0U; i < elementCount; ++i) {
            if (offset + sizeof(PackedTensorElement) > elementBytes) {
                return std::nullopt;
            }
            PackedTensorElement headerCopy;
            std::memcpy(&headerCopy, elementData + offset, sizeof(headerCopy));
            offset += static_cast<std::uint32_t>(sizeof(headerCopy));
            if (headerCopy.payloadLength > elementBytes - offset) {
                return std::nullopt;
            }
            elems[i] = detail::decodeTensorElement(headerCopy, elementData + offset, res, /*depth=*/0U);
            offset += headerCopy.payloadLength;
        }
        gr::Tensor<Value> tensor(gr::extents_from, extents, res);
        tensor._data = std::move(elems);
        return T(std::move(tensor)); // parens to bypass init_list ctor (Tensor<Value>→Value is implicit, brace-init would build a 1-elem tensor)
    } else if constexpr (std::same_as<ElemT, bool>) {
        // bool path: read 1 byte per element from the blob; Tensor<bool> can't take .data() but
        // can be filled element-by-element through its iterator.
        gr::Tensor<bool> tensor(gr::extents_from, extents, res);
        const auto*      dataPtr = base + kTensorBlobHeaderSize + 4UZ * rank;
        for (std::uint32_t i = 0U; i < elementCount; ++i) {
            tensor._data[i] = (static_cast<std::uint8_t>(dataPtr[i]) != 0U);
        }
        return T(std::move(tensor));
    } else {
        // Fixed-size element types: TensorView aliases the data section, owned() materialises into
        // a fresh Tensor<ElemT> at the requested resource (decoded copy).
        if (auto view = this->template get_if<gr::TensorView<ElemT>>()) {
            return T(view->owned(res));
        }
        return std::nullopt;
    }
}

// Q1.B: out-of-class definition of get_if<TensorView<Value>>() — decode the blob into a snapshot
// Tensor<Value> and wrap it in the TensorView<Value> partial specialisation (which owns the
// snapshot internally so iteration / element access stays valid for the optional's lifetime).
template<typename T>
requires(gr::TensorViewLike<T> && std::same_as<std::remove_const_t<typename T::value_type>, gr::pmt::Value>)
inline std::optional<T> Value::get_if() const noexcept {
    if (auto tensor = this->template get_if<gr::Tensor<gr::pmt::Value>>()) {
        return T(std::move(*tensor));
    }
    return std::nullopt;
}

// Q1.B: out-of-class definition of Value::init_from_tensor. Body needs `detail::encodeTensorBlob`
// which is defined in this header (and depends on ValueMap for nested Tensor<Value> elements
// containing nested-map values). Tensor<std::pmr::string> / Tensor<std::string> are converted
// to Tensor<Value> on the fly so the API surface stays uniform.
template<TensorLike Tens>
inline void Value::init_from_tensor(Tens&& tensor) {
    using T = typename std::remove_cvref_t<Tens>::value_type;
    if constexpr (std::same_as<T, std::pmr::string> || std::same_as<T, std::string>) {
        gr::Tensor<Value> wrapped(gr::extents_from, tensor.extents(), _resource);
        for (std::size_t i = 0UZ; i < tensor.size(); ++i) {
            wrapped._data[i] = Value{std::string_view{tensor._data[i]}, _resource};
        }
        set_types(ValueType::Value, ContainerType::Tensor);
        auto blob = detail::encodeTensorBlob(wrapped, _resource, /*depth=*/0U);
        if (blob.empty()) {
            _storage.ptr   = nullptr;
            _payloadLength = 0U;
            return;
        }
        auto* bytes = static_cast<std::byte*>(_resource->allocate(blob.size(), kBlobAlignment));
        std::memcpy(bytes, blob.data(), blob.size());
        _storage.ptr   = bytes;
        _payloadLength = static_cast<std::uint32_t>(blob.size());
        return;
    } else {
        set_types(get_value_type<T>(), ContainerType::Tensor);
        auto blob = detail::encodeTensorBlob(tensor, _resource, /*depth=*/0U);
        if (blob.empty()) {
            _storage.ptr   = nullptr;
            _payloadLength = 0U;
            return;
        }
        auto* bytes = static_cast<std::byte*>(_resource->allocate(blob.size(), kBlobAlignment));
        std::memcpy(bytes, blob.data(), blob.size());
        _storage.ptr   = bytes;
        _payloadLength = static_cast<std::uint32_t>(blob.size());
    }
}

} // namespace gr::pmt

namespace std {
template<>
struct hash<gr::pmt::ValueMap> {
    /// Order-independent hash (matches std::pmr::unordered_map semantics): per-entry hash
    /// (key XOR value) accumulated by addition. Empty map → 0.
    [[nodiscard]] std::size_t operator()(const gr::pmt::ValueMap& m) const noexcept {
        std::size_t accumulator = 0UZ;
        for (auto [k, v] : m) {
            const std::size_t entryHash = std::hash<std::string_view>{}(k) ^ std::hash<gr::pmt::Value>{}(v);
            accumulator += entryHash;
        }
        return accumulator;
    }
};
} // namespace std

#endif // GNURADIO_VALUEMAP_HPP
