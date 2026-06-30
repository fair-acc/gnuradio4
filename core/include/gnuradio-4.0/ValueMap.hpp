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
 * Replaces the prior std::pmr::unordered_map<pmr::string, Value> storage with a single
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
 * field; publish ordering for shared/device use is provided by the enclosing
 * ring buffer, not by blob-level atomics. Little-endian hosts only.
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
 *   float hz   = map.value_or<float>("sample_rate", 0.f); // typed unwrap, alloc-free, default on miss
 *   auto  found = map.find_value("missing_key");        // std::optional<Value>; nullopt iff missing
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

inline constexpr std::array<char, 4> kBlobMagic     = {'G', 'R', '4', 'M'};
inline constexpr std::uint8_t        kBlobVersion   = 1U;
inline constexpr std::size_t         kBlobAlignment = 16UZ; // USM / SYCL requirement
static_assert(std::has_single_bit(kBlobAlignment) && kBlobAlignment <= 255UZ, "kBlobAlignment: power-of-two, fits the u8 alignment-recovery byte");
inline constexpr std::size_t kMaxInlineKeyLength = 34UZ; // 1 B length + 34 chars + 1 B '\0' guard = 36 B (= inlineKey[]). Cap is 34, NOT 35: at L=35 the chars fill [1..35] leaving no trailing NUL byte for C-string consumers.

inline constexpr std::uint8_t kEntryFlagInlineScalar = 0x01; // PackedTensorElement-only: tensor element packs an inline scalar (≤8 B) directly in the element header. PackedEntry never sets this — value-records always live in the payload pool.
inline constexpr std::uint8_t kEntryFlagOffsetLength = 0x02; // payloadOffset/length valid
inline constexpr std::uint8_t kEntryFlagNestedMap    = 0x08; // payload is a nested ValueMap blob (recursive); only meaningful when valueType == Value::ValueType::Value
inline constexpr std::uint8_t kEntryFlagTensor       = 0x10; // payload is a tensor sub-blob (see Tensor wire-format below); only meaningful when valueType == Value::ValueType::Value

inline constexpr std::uint8_t kHeaderFlagOverflow    = 0x01; // device append hit payloadCapacity limit
inline constexpr std::uint8_t kHeaderFlagFrozen      = 0x02; // advisory: further mutation disallowed
inline constexpr std::uint8_t kHeaderFlagDebugGuards = 0x04; // blob written with debug-mode hex-dump guards: "!EOD\0" payload-end marker, 0xAA slack fill, 0xCA tensor sub-blob signature, vacated entry slots zero-filled. Hex-dump aid only — `from_blob` does NOT validate the guards. Set on debug-build blobs; reserved bit pattern, future readers may opt in.
// kMaxTensorRank / kMaxTensorElements / kTensorBlobHeaderSize / kTensorEncodingVariableSize live
// in Value.hpp — Value owns its own tensor byte-blob format.

// "!EOD\0" — written at `_blob[payloadOffset + payloadUsed]` after every mutation in debug builds.
// Marks the first byte past live payload data; subsequent appends overwrite it. Helps `xxd` /
// debugger memory views show where live data ends without bleeding into slack / free regions.
inline constexpr std::array<std::byte, 5> kPayloadEndMarker{std::byte{'!'}, std::byte{'E'}, std::byte{'O'}, std::byte{'D'}, std::byte{0}};
inline constexpr std::byte                kSlackFillPattern{0xAA};       // alternating bit pattern, obvious in hex dumps
inline constexpr std::byte                kTensorSubBlobSignature{0xCA}; // written into tensor sub-blob's reserved byte (header[3])

struct alignas(8) Header {
    char          magic[4];        // {'G','R','4','M'}
    std::uint8_t  version;         // kBlobVersion
    std::uint8_t  flags;           // Header.flags bits (Overflow / Frozen / DebugGuards / ...)
    std::uint16_t entryCount;      // number of committed entries
    std::uint16_t entryCapacity;   // explicit (was derived from payloadOffset). cap on entryCount; resizes only via _grow
    std::uint16_t _reserved;       // padding to align next u32
    std::uint32_t totalSize;       // bytes, incl. this header + payload
    std::uint32_t payloadOffset;   // byte offset of payload pool start
    std::uint32_t payloadUsed;     // bytes used in the payload pool
    std::uint32_t payloadCapacity; // cap on payloadUsed; explicit field (not derived from totalSize - payloadOffset)
    std::uint32_t payloadFreeHead; // absolute blob offset of first free chunk in payload pool, 0 = none
};
static_assert(sizeof(Header) == 32UZ);
static_assert(alignof(Header) == 8UZ);

// Free-chunk header inside the payload pool. erase() reclaims payload regions ≥ kFreeChunkHeaderSize
// bytes by writing this header at the freed region's start and prepending it to the singly-linked
// free list rooted at Header::payloadFreeHead. _appendPayload first-fit-searches the list before
// extending the pool. Smaller freed regions (< kFreeChunkHeaderSize bytes) are orphaned until
// the next shrink_to_fit / blob realloc.
inline constexpr std::size_t kFreeChunkHeaderSize = 8UZ;
struct alignas(4) FreeChunk {
    std::uint32_t length;     // total chunk length in bytes (including this 8-byte header)
    std::uint32_t nextOffset; // absolute blob offset of next free chunk; 0 = end of list
};
static_assert(sizeof(FreeChunk) == kFreeChunkHeaderSize);

// 48-byte budget — 12-byte fixed header (payloadOffset/payloadLength + keyId + valueType +
// flags) + 36-byte inlineKey (35 chars + length byte). The 35-char ceiling fits all 20
// Tag.hpp canonical names and the longest known SigMF extension keys ("ntia-emitter:antenna-
// polarization" et al.). Longer keys spill to the payload pool via kSpilledKeyId.
//
// All values (including inline scalars) live in the payload pool as value-records — there is
// no inline-value field on PackedEntry. payloadOffset always points at the entry's value
// record; iter / decode aliases the bytes via `Value::makeView`.
//
// inlineKey encoding when keyId == kInlineKeyId:
//   inlineKey[0]      = length byte (0..kMaxInlineKeyLength = 34)
//   inlineKey[1..len] = character data
//   inlineKey[len+1..35] = zero-padded (provides a NUL terminator after the chars; len <= 34 so byte 35 is always zero)
//
// Spilled-key encoding when keyId == kSpilledKeyId (key length > kMaxInlineKeyLength):
//   inlineKey[0..3]   = u32 payload-pool offset
//   inlineKey[4..7]   = u32 length
//   inlineKey[8..35]  = zero-padded
// alignas(8) — same reasoning as Header. PackedEntry arrays within nested blobs start at
// parent_record + 8 + sizeof(Header) = parent + 40, which is 8-aligned but not 16-aligned;
// alignas(8) is the conservative lower bound. Top-level entry arrays are 16-aligned by
// virtue of the blob's 16-aligned allocator and the fixed Header offset (32).
struct alignas(8) PackedEntry {
    std::uint32_t payloadOffset; // into payload pool — points at the entry's value-record
    std::uint32_t payloadLength; // bytes occupied by the value-record (incl. its 8 B header)
    std::uint16_t keyId;         // canonical id, kInlineKeyId = inline key, kSpilledKeyId = spilled, kEndMarkerId = sentinel
    std::uint8_t  valueType;     // see Value::ValueType (single-byte tag)
    std::uint8_t  flags;         // kEntryFlag* bits (kEntryFlagOffsetLength always set; kEntryFlagInlineScalar is PackedTensorElement-only)
    char          inlineKey[36]; // zero-padded when keyId == kInlineKeyId; up to 34 chars + length byte at [0] + trailing NUL guard at [len+1..35]
};
static_assert(sizeof(PackedEntry) == 48UZ);
static_assert(alignof(PackedEntry) == 8UZ);

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

// True for value_types that map to a fixed-size scalar/complex element (i.e. a Tensor<T> with
// well-defined `sizeof(T)`-stride wire format). Excludes String / Value / Map (variable-size or
// no usable T at all). Used by encodeTensorBlob to detect when a homogeneous Tensor<Value> can
// be re-routed through the typed Tensor<T> encoder.
[[nodiscard]] inline bool isFixedSizeTensorElementType(Value::ValueType vt) noexcept;

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
// fixed-size contiguous layout. Value uses the variable-size per-element header layout
// (PackedTensorElement + payload) only when cells are heterogeneous or hold variable-size
// element types (String / Map / nested Tensor); homogeneous fixed-size Tensor<Value> content
// is re-routed through the typed encoder by encodeTensorBlob (see F2). std::pmr::string is
// NOT supported as a Tensor element type — Tensor.hpp:163 already rejects std::string;
// string-array-like data goes through Tensor<Value> where each element is a String-typed Value.
template<typename T>
concept TensorElementType = InlineScalar<T> || PayloadScalar<T> || std::same_as<T, Value>;

inline bool isFixedSizeTensorElementType(Value::ValueType vt) noexcept {
    return dispatchValueType(vt, []<typename T>(std::type_identity<T>) -> bool {
        if constexpr (TensorElementType<T> && !std::same_as<T, Value> && !std::same_as<T, std::string_view>) {
            return true;
        } else {
            return false;
        }
    });
}

// Pack a scalar into a 64-bit slot at offset 0; upper bytes zero. Used by PackedTensorElement
// for tensor-element inline scalars (PackedEntry no longer carries an inline-value slot —
// every PackedEntry value lives as a value-record in the payload pool).
// All InlineScalar types — including bool — share the same byte layout: bool occupies the
// single low byte (0/1). std::complex<float> goes through the same path (8 bytes total).
template<InlineScalar T>
inline void writeInlineScalar(PackedTensorElement& e, T v) noexcept {
    e.inlineValue = 0U;
    if constexpr (sizeof(T) == sizeof(std::uint64_t)) {
        e.inlineValue = std::bit_cast<std::uint64_t>(v);
    } else {
        const auto buf = std::bit_cast<std::array<std::byte, sizeof(T)>>(v);
        std::memcpy(&e.inlineValue, buf.data(), sizeof(T));
    }
}

template<InlineScalar T>
[[nodiscard]] inline T readInlineScalar(const PackedTensorElement& e) noexcept {
    if constexpr (sizeof(T) == sizeof(std::uint64_t)) {
        return std::bit_cast<T>(e.inlineValue);
    } else {
        std::array<std::byte, sizeof(T)> buf;
        std::memcpy(buf.data(), &e.inlineValue, sizeof(T));
        return std::bit_cast<T>(buf);
    }
}

// Pack a short string into inlineKey[36]. name.size() must be <= kMaxInlineKeyLength (34).
// The byte at index 1 + name.size() is guaranteed zero, providing a NUL terminator after
// the chars so the iterator can return the key as a `const char*` pointing at &inlineKey[1].
inline void setInlineKey(PackedEntry& e, std::string_view name) noexcept {
    e.keyId        = keys::kInlineKeyId;
    e.inlineKey[0] = static_cast<char>(name.size());
    std::memset(e.inlineKey + 1, 0, sizeof(e.inlineKey) - 1U); // zero-fill chars + guard tail
    std::memcpy(e.inlineKey + 1, name.data(), name.size());
}

[[nodiscard]] constexpr std::string_view readInlineKey(const PackedEntry& e) noexcept {
    const auto len = static_cast<std::size_t>(static_cast<std::uint8_t>(e.inlineKey[0]));
    return {e.inlineKey + 1, len};
}

// Spilled-key encoding (keyId == kSpilledKeyId): inlineKey[0..3] holds the u32 payload
// offset, inlineKey[4..7] holds the u32 length; remaining bytes are zero-padded. Used for
// keys longer than kMaxInlineKeyLength (34 chars).
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
// Specialised for `const char[N]` string literals — uses the compile-time extent instead of
// the strlen-doing string_view(const char*) ctor. The generic branch covers everything that
// implicitly converts to string_view (const char*, std::string, std::pmr::string,
// std::string_view, gr::meta::fixed_string, …).
template<typename K>
[[nodiscard]] inline constexpr std::string_view keyToStringView(const K& key) noexcept {
    if constexpr (std::is_bounded_array_v<std::remove_cvref_t<K>>) {
        return std::string_view{key, std::extent_v<std::remove_cvref_t<K>> - 1U};
    } else {
        return std::string_view{key};
    }
}

// Hard cap on nested-map recursion depth. Bounds stack usage and protects against
// pathological / malformed sub-blobs (relevant whenever from_blob accepts untrusted byte
// spans). Real-world tag-meta nesting is shallow (≤ 2-3 levels).
inline constexpr std::uint32_t kMaxDecodeDepth = 32U;

// Forward declarations for the tensor encode/decode helpers — defined after `decodeEntry`
// so that the Value/Tensor recursion (Tensor<Value> elements that are themselves nested
// ValueMaps or Tensors) can call back into `decodeEntry` for sub-blob walking.
[[nodiscard]] inline Value decodeEntry(const PackedEntry& e, const std::byte* blobBase, std::pmr::memory_resource* resource, std::uint32_t depth);
[[nodiscard]] inline Value decodeTensorBlob(const std::byte* tensorBase, std::uint32_t tensorBytes, std::pmr::memory_resource* resource, std::uint32_t depth);
[[nodiscard]] inline Value decodeTensorElement(const PackedTensorElement& elem, const std::byte* payloadData, std::pmr::memory_resource* resource, std::uint32_t depth);

// Build a Value containing a nested ValueMap from a packed-entries array. Defined after
// ValueMap is fully defined (Value forward-declares ValueMap; sizeof / new for nested ValueMaps
// need the full definition here).
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

    // C3: every entry has a value-record at e.payloadOffset =
    // [size:4][vt:1][ct:1][flags:1][payloadOffset:1][content]. Content starts at +8; len = payloadLength - 8.
    // Inline scalars share the same shape (8 B inline-payload zero-padded to 8 B).
    if ((e.flags & kEntryFlagOffsetLength) == 0U || e.payloadOffset == 0U) {
        return Value{resource}; // Monostate / unset entry
    }
    const std::byte*    recStart    = blobBase + e.payloadOffset;
    const std::byte*    contentBase = recStart + kRecHeaderBytes;
    const std::uint32_t contentLen  = e.payloadLength > kRecHeaderBytes ? e.payloadLength - kRecHeaderBytes : 0U;

    // Special-case Tensor sub-blob.
    if ((e.flags & kEntryFlagTensor) != 0U) {
        if (depth >= kMaxDecodeDepth) {
            return Value{resource};
        }
        return decodeTensorBlob(contentBase, contentLen, resource, depth + 1U);
    }

    // Special-case nested ValueMap.
    if ((e.flags & kEntryFlagNestedMap) != 0U) {
        if (depth >= kMaxDecodeDepth) {
            return Value{resource};
        }
        if (contentLen < sizeof(Header)) {
            return Value{resource};
        }
        const auto* nestedH         = std::launder(reinterpret_cast<const Header*>(contentBase));
        const auto  declaredCount   = nestedH->entryCount;
        const auto  entryArrayBytes = static_cast<std::size_t>(declaredCount) * sizeof(PackedEntry);
        if (sizeof(Header) + entryArrayBytes > contentLen) {
            return Value{resource};
        }
        const auto* nestedE = std::launder(reinterpret_cast<const PackedEntry*>(contentBase + sizeof(Header)));
        return buildNestedMapValue(contentBase, nestedE, declaredCount, resource, depth);
    }

    return dispatchValueType(vt, [&]<typename T>(std::type_identity<T>) -> Value {
        if constexpr (std::same_as<T, std::string_view>) {
            // String content = chars + '\0' guard. Length = strlen-style scan (records are padded).
            const auto* p   = reinterpret_cast<const char*>(contentBase);
            const auto* nul = static_cast<const char*>(std::memchr(p, '\0', contentLen));
            const auto  len = nul != nullptr ? static_cast<std::size_t>(nul - p) : contentLen;
            return Value{std::string_view{p, len}, resource};
        } else if constexpr (InlineScalar<T>) {
            T out;
            std::memcpy(&out, contentBase, sizeof(T));
            return Value{out, resource};
        } else if constexpr (PayloadScalar<T>) {
            if (contentLen < sizeof(T)) {
                return Value{resource};
            }
            T out;
            std::memcpy(&out, contentBase, sizeof(T));
            return Value{out, resource};
        } else {
            return Value{resource};
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
    // Extents area is padded so subsequent element data lands at an 8-aligned offset; see
    // paddedTensorExtentsBytes(rank) for the contract.
    const std::size_t extentsBytes = paddedTensorExtentsBytes(rank);
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
                    elems[i]            = decodeTensorElement(headerCopy, elementData + offset, resource, depth);
                    const auto paddedPL = paddedElementPayloadBytes(headerCopy.payloadLength);
                    offset += std::min<std::uint32_t>(paddedPL, elementBytes - offset);
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
            return Value{readInlineScalar<T>(elem), resource};
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
    assert(rank <= kMaxTensorRank);
    const auto        extents      = tensor.extents();
    const std::size_t elementCount = tensor.size();
    assert(elementCount <= kMaxTensorElements);

    constexpr Value::ValueType elemVT       = cppToValueType<ElemT>();
    constexpr bool             variableSize = std::same_as<ElemT, Value>;

    // Padded so element data lands at an 8-aligned offset within the sub-blob.
    const std::size_t           extentsBytes = paddedTensorExtentsBytes(rank);
    std::pmr::vector<std::byte> out(resource);
    out.resize(kTensorBlobHeaderSize + extentsBytes);
    std::byte* const outBase = out.data();
    [[assume(outBase != nullptr)]]; // post-resize(N>0) is non-null; silence GCC -Wnull-dereference

    outBase[0] = static_cast<std::byte>(elemVT);
    outBase[1] = static_cast<std::byte>(rank);
    outBase[2] = static_cast<std::byte>(variableSize ? kTensorEncodingVariableSize : 0U);
    // Item F (debug-only): 0xCA recognition byte so a hex-dump consumer can spot tensor sub-blobs
    // in the payload pool. Reader doesn't validate this byte, so the wire format stays compatible.
    outBase[3]             = gr::meta::kDebugBuild ? kTensorSubBlobSignature : std::byte{0U};
    const std::uint32_t ec = static_cast<std::uint32_t>(elementCount);
    std::memcpy(outBase + 4, &ec, sizeof(ec));
    for (std::size_t i = 0UZ; i < rank; ++i) {
        const std::uint32_t ext = static_cast<std::uint32_t>(extents[i]);
        std::memcpy(out.data() + kTensorBlobHeaderSize + 4UZ * i, &ext, sizeof(ext));
    }

    if constexpr (variableSize) {
        // F2: re-route homogeneous fixed-size content through the typed Tensor<T> encoder so
        // the wire format collapses to contiguous bytes (no per-element PackedTensorElement
        // headers). The variable-size path below is reached only for genuinely heterogeneous
        // content or for variable-size element types (String / Map / nested Tensor).
        Value::ValueType commonVT               = Value::ValueType::Monostate;
        bool             isHomogeneousFixedSize = elementCount > 0UZ;
        if (isHomogeneousFixedSize) {
            commonVT = (tensor._data.data() + 0)->value_type();
            for (std::size_t i = 1UZ; i < elementCount; ++i) {
                if ((tensor._data.data() + i)->value_type() != commonVT) {
                    isHomogeneousFixedSize = false;
                    break;
                }
            }
            if (isHomogeneousFixedSize) {
                isHomogeneousFixedSize = isFixedSizeTensorElementType(commonVT);
            }
        }
        if (isHomogeneousFixedSize) {
            return dispatchValueType(commonVT, [&]<typename T>(std::type_identity<T>) -> std::pmr::vector<std::byte> {
                if constexpr (TensorElementType<T> && !std::same_as<T, Value> && !std::same_as<T, std::string_view>) {
                    gr::Tensor<T> typedTensor(gr::extents_from, std::span<const std::size_t>{extents.begin(), extents.end()}, resource);
                    for (std::size_t i = 0UZ; i < elementCount; ++i) {
                        if (const auto* scalar = (tensor._data.data() + i)->template get_if<T>()) {
                            typedTensor._data[i] = *scalar;
                        }
                    }
                    return encodeTensorBlob(typedTensor, resource, depth);
                } else {
                    return std::pmr::vector<std::byte>(resource); // unreachable — gated by isFixedSizeTensorElementType
                }
            });
        }

        out.reserve(out.size() + elementCount * sizeof(PackedTensorElement));
        for (std::size_t i = 0UZ; i < elementCount; ++i) {
            encodeTensorElement(out, *(tensor._data.data() + i), resource, depth + 1U);
        }
    } else if constexpr (std::same_as<ElemT, bool>) {
        // bool: 1 byte per element on the wire. Iterate via the tensor's iter (works for both
        // owning Tensor<bool> and TensorView<bool> proxy iter — sizeof(bool) may vary across
        // platforms, so element-wise write is portable).
        if (elementCount > 0UZ) {
            const auto baseOffset = out.size();
            out.resize(out.size() + elementCount);
            std::size_t i = 0UZ;
            for (bool b : tensor) {
                out[baseOffset + i++] = static_cast<std::byte>(b ? 1U : 0U);
            }
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

/**
 * @brief Contiguous packed-blob ordered map of `string_view → Value`.
 *
 * Drop-in alternative to `std::pmr::unordered_map<std::pmr::string, Value>` for the GR4
 * tag / settings / message hot paths: insertion-order iteration, transparent string-view
 * lookup, and a single contiguous PMR allocation that round-trips byte-for-byte through
 * `from_blob` / `as_blob` (USM/SYCL/IPC-portable — no host pointers on the wire).
 *
 * @par Basic usage
 * @code
 *   gr::pmt::ValueMap m;
 *   m.emplace("rate", 1000.0);                           // f64 inline scalar
 *   m.insert_or_assign("name", std::string{"TIQ_FFT"});  // String → payload pool
 *   m.emplace("samples", Tensor<int32_t>({4}));          // Tensor<int32> sub-blob
 *
 *   const auto rate = m.value_or<double>("rate", 0.0);                // typed, alloc-free
 *   const auto sv   = m.value_or<std::string_view>("name", "");       // alloc-free view
 *
 *   for (const auto& [k, v] : m) {                                     // VIEW-mode Values
 *       fmt::print("{} = {}\n", k, v.value_or<std::string>("?"));
 *   }
 *
 *   m.erase("name");
 *   m.merge(std::move(other));                                          // moves non-dup entries (std::map parity, skip on conflict)
 *   for (const auto& [k, v] : other) m.insert_or_assign(k, v);          // STL idiom for last-source-wins copy
 * @endcode
 *
 * @par Iterator semantics — important
 * `const_iterator::value_type` is `std::pair<std::string_view, Value>` returned **by value**
 * (the on-blob layout has no stored pair to point at — same situation as `std::vector<bool>`'s
 * proxy iterator). Consequences:
 *   - `(*it).second` and `it->second` both work. `operator->` returns a small `ArrowProxy`
 *     holder that owns the materialised pair for the duration of the surrounding
 *     full-expression — same pattern `std::regex_iterator` and `std::istream_iterator` use.
 *     Do not take the address of the pair (`auto* p = &it->second` dangles at end of
 *     statement); copy it out instead (`auto v = it->second;`).
 *   - Each `(*it)` materialises a fresh pair. Bind once if you need the value twice:
 *     `auto entry = (*it); entry.second.is_string(); entry.second.value_or(...);`
 *   - Iter-yielded Values are **view-mode** and alias the parent blob — they dangle if the
 *     source map is destroyed, grown (insert / merge / reserve), or moves resource. For an
 *     owning copy that survives the parent, use `at(key)` or `value_or<T>(default)`.
 *
 * @par Owning vs view-mode access
 * Variable-size accessors come in two forms:
 *   - **Owning** (alloc per call): `at(key)`, `operator[](key)`, `value_or<T>(default)` —
 *     deep-copy bytes from the pool into a fresh Value owned by the caller's resource.
 *     Safe to outlive the source map.
 *   - **View-mode** (alloc-free): iterator deref, `get_if<TensorView<T>>()`,
 *     `get_if<ValueMap>()`, `get_if<std::string_view>()` — alias the blob bytes;
 *     lifetime-bound to the source map.
 *
 * @par Lookup / mutation API at a glance
 *   - `at(K) → Value`             owning copy; returns Monostate Value if missing (no throw)
 *   - `operator[](K) const → V`   throws `std::out_of_range` if missing — exception path
 *   - `find(K) → const_iterator`  alloc-free; `end()` if missing
 *   - `contains(K) → bool`
 *   - `emplace(K, V)`             insert; no-op if key exists
 *   - `insert_or_assign(K, V)`    insert or replace
 *   - `erase(K) → bool`           reclaims payload via in-pool free list
 *   - `merge(ValueMap&&)`         moves non-conflicting entries from source; dups stay on source
 *   - `as_blob() / from_blob()`   serialise / deserialise the contiguous byte image
 *
 * @warning `operator[]` is the only throwing accessor. Library / framework / SYCL code
 *          that must stay exception-free MUST use `at()` / `find()` / `contains()`.
 *
 * @note Reflected-field key style: per the GR4 style guide reflected settings keys are
 *       `snake_case` (e.g. `sample_rate`); ValueMap is the wire backing of those settings.
 *
 * @par Wire Format / Binary Layout (alignof 16; one contiguous PMR allocation)
 *
 * Single contiguous blob: 32-byte `Header`, then a fixed-size `PackedEntry` array (each
 * row 48 B, interleaving key + value), then a shared variable-size payload pool. Splitting
 * fixed rows from a shared pool gives O(1) indexed random access (uniform stride suits SIMD
 * + USM device iteration), one cache line per entry, and in-place mutation without shifting
 * pair blobs. Publish ordering for shared/device use is provided by the enclosing ring
 * buffer, not by blob-level atomics.
 *
 * @code
 *  ┌──────────────────────────────────────────────────────────────────────────────┐
 *  │                       gr::pmt::ValueMap blob                                 │
 *  │                                                                              │
 *  │ offset  ┌────────────────────────────────────────┐ ◄── _blob, _header        │
 *  │   0     │ magic[4]    = "GR4M"                   │ ┐                         │
 *  │         │ version (u8) │ flags (u8) │ entryCount │ │                         │
 *  │         │                              (u16)     │ │                         │
 *  │         │ entryCapacity (u16)│ _reserved (u16)   │ │                         │
 *  │         │ totalSize         (u32)                │ │ Header                  │
 *  │         │ payloadOffset     (u32) ─────────┐     │ │   (32 B, alignof 16)    │
 *  │         │ payloadUsed       (u32)          │     │ │   ───────────────────   │
 *  │         │ payloadCapacity   (u32)          │     │ │   tells the reader      │
 *  │         │ payloadFreeHead   (u32) ──┐      │     │ │   where the entry       │
 *  │         ├───────────────────────────┼──────┼─────┤ ┘   payload pool begins   │
 *  │  32     │ PackedEntry[0]  (48 B)    │      │     │ ┐ ◄── _entries            │
 *  │         │   payloadOffset (u32)─────┼──────┼─┐   │ │                         │
 *  │         │   payloadLength (u32)     │      │ │   │ │  PackedEntry array      │
 *  │         │   keyId (u16)             │      │ │   │ │   (entryCount × 48 B,   │
 *  │         │   valueType (ValueType,u8)│      │ │   │ │    alignof 16)          │
 *  │         │   flags (u8)              │      │ │   │ │   ─────────────────     │
 *  │         │   inlineKey [36 B]        │      │ │   │ │  one row per key, value │
 *  │         │                           │      │ │   │ │  always lives in pool   │
 *  │         ├───────────────────────────┼──────┼─┤   │ │     → payloadOffset/    │
 *  │  80     │ PackedEntry[1]   ...      │      │ │   │ │       payloadLength     │
 *  │         ├───────────────────────────┼──────┼─┤   │ │       point at the      │
 *  │   …     │ PackedEntry[entryCount-1] │      │ │   │ │       value-record bytes│
 *  │         ├───────────────────────────┼──────┼─┤   │ │   inline key (≤35 ch)   │
 *  │         │ PackedEntry[entryCount]   │      │ │   │ │     → inlineKey[]       │
 *  │         │   keyId = 0xFFFF          │      │ │   │ │   spilled key (>35 ch)  │
 *  │         │   (kEndMarkerId sentinel) │      │ │   │ │     → keyId=Spilled,    │
 *  │         │                           │      │ │   │ │       offset/len in     │
 *  │         │                           │      │ │   │ │       inlineKey[0..7]   │
 *  │         ├───────────────────────────┘      │ │   │ │   sentinel slot         │
 *  │   …     │ payload pool — heterogeneous  ◄──┘ │   │ ┘   (always-on guard)     │
 *  │         │ blob of:                           │   │ ┐ ◄── _blob+payloadOffset │
 *  │         │   • spilled key bytes              │◄──┘   │                       │
 *  │         │   • String bytes + '\0' guard      │       │ Payload pool          │
 *  │         │   • complex<double> raw bytes      │       │   (totalSize          │
 *  │         │   • Tensor<T> sub-blobs            │       │    − payloadOffset    │
 *  │         │   • nested ValueMap blobs (recur.) │       │    bytes)             │
 *  │         │   • FreeChunk{length,nextOffset}─┐ │       │   ─────────────       │
 *  │         │                                  ▼ │       │  written front-to-    │
 *  │         │   FreeChunk{...} ──► FreeChunk{0,0}│       │  back via _appendPay- │
 *  │         ├────────────────────────────────────┤       │  load; freed regions  │
 *  │         │ slack / unused (kSlackFillPattern  │       │  ≥ 8 B chained into a │
 *  │         │ 0xAA in debug)                     │       │  free list rooted at  │
 *  │ totalSz └────────────────────────────────────┘       ┘  Header.payloadFreeHead│
 *  └──────────────────────────────────────────────────────────────────────────────┘
 * @endcode
 *
 * Decode rules (`PackedEntry` row → in-memory Value):
 * | flags / valueType                  | meaning                                  | Value materialisation                                |
 * |------------------------------------|------------------------------------------|------------------------------------------------------|
 * | `OffsetLength`, any inline scalar  | 16 B value-record in pool ([8 B header]  | view-mode aliases the 8 B inline payload; alloc-free |
 * |                                    | + 8 B inline-payload region)             |                                                      |
 * | `OffsetLength`, ValueType=String   | bytes in pool ([header + chars + '\0'])  | view-mode (alloc-free) OR owning copy on `at()`      |
 * | `OffsetLength`, vt=ComplexFloat64  | 16 raw bytes in pool                     | owning copy: 16 B heap                               |
 * | `OffsetLength` + `Tensor`, vt=Value| tensor sub-blob in pool                  | view-mode `TensorView<T>` aliases bytes;             |
 * |                                    |                                          | `value_or<Tensor<T>>` deep-copies                    |
 * | `OffsetLength` + `NestedMap`, vt=V | full ValueMap sub-blob in pool           | view-mode `ValueMap{view=true}` aliases;             |
 * |                                    |                                          | `owned()` deep-copies                                |
 * | keyId = `kInlineKeyId  (0x8000)`   | inlineKey[0]=len, [1..len]=chars         | string_view into `inlineKey`                         |
 * | keyId = `kSpilledKeyId (0x8001)`   | inlineKey[0..3]=offset, [4..7]=len       | string_view into payload pool                        |
 * | keyId = `kEndMarkerId  (0xFFFF)`   | sentinel slot past the last valid entry  | iterators stop here                                  |
 *
 * Nested maps and `Tensor<Value>` reuse the same byte format recursively in the parent's
 * payload pool, so an N-level nested tree is one heap allocation, one contiguous blob.
 *
 * @see gr::pmt::Value — the 24-byte transient handle that decoded entries materialise into.
 */

/**
 * @brief Trivially-copyable POD handle over a ValueMap's blob bytes.
 *
 * Carries the four pointers/size that any read/in-place-mutation needs (`_blob`, `_capacity`,
 * cached `_header` / `_entries` re-derived from `_blob`). No `_resource` — that lives on the
 * RAII owner (`ValueMap`). Suitable for SYCL/CUDA kernel by-value capture once the underlying
 * blob is in USM/shared memory.
 *
 * Read API + in-place mutation API will move here in a follow-up step; for now this is a bare
 * base for `ValueMap` to inherit.
 */
struct ValueMapView {
    std::byte*    _blob{nullptr};
    std::uint32_t _capacity{0U};
    Header*       _header{nullptr};
    PackedEntry*  _entries{nullptr};

    class const_iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using iterator_concept  = std::random_access_iterator_tag;
        // Pair built by value — on-blob layout has no stored pair. `it->second` resolves via
        // ArrowProxy's chained operator->; taking `&it->second` dangles after end of full-expression.
        using value_type = std::pair<std::string_view, ValueView>;
        using reference  = value_type;

        struct ArrowProxy {
            value_type                                _pair;
            [[nodiscard]] constexpr const value_type* operator->() const noexcept { return &_pair; }
        };
        using pointer         = ArrowProxy;
        using difference_type = std::ptrdiff_t;

        constexpr const_iterator() = default;
        constexpr const_iterator(const ValueMapView* m, std::uint16_t i) noexcept : _map(m), _index(i) {}

        [[nodiscard]] value_type           operator*() const;
        [[nodiscard]] constexpr ArrowProxy operator->() const { return ArrowProxy{operator*()}; }
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
        friend struct ValueMapView;
        friend class ValueMap;
        const ValueMapView* _map   = nullptr;
        std::uint16_t       _index = 0U;
    };

    using iterator               = const_iterator;
    using reverse_iterator       = std::reverse_iterator<const_iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    struct const_key_iterator {
        using iterator_concept  = std::forward_iterator_tag;
        using iterator_category = std::forward_iterator_tag;
        using value_type        = std::string_view;
        using reference         = value_type;
        using pointer           = void;
        using difference_type   = std::ptrdiff_t;

        const ValueMapView* _map{nullptr};
        std::uint16_t       _index{0U};

        constexpr const_key_iterator() = default;
        constexpr const_key_iterator(const ValueMapView* m, std::uint16_t i) noexcept : _map(m), _index(i) {}

        [[nodiscard]] value_type      operator*() const noexcept { return _map->_readKey(_map->_entries[_index]); }
        constexpr const_key_iterator& operator++() noexcept {
            ++_index;
            return *this;
        }
        constexpr const_key_iterator operator++(int) noexcept {
            auto tmp = *this;
            ++_index;
            return tmp;
        }
        constexpr bool operator==(const const_key_iterator& o) const noexcept { return _map == o._map && _index == o._index; }
    };

    struct keys_view {
        const ValueMapView*                        _map;
        [[nodiscard]] constexpr const_key_iterator begin() const noexcept { return {_map, 0U}; }
        [[nodiscard]] constexpr const_key_iterator end() const noexcept { return {_map, static_cast<std::uint16_t>(_map->_header ? _map->_header->entryCount : 0U)}; }
        [[nodiscard]] constexpr std::size_t        size() const noexcept { return _map->size(); }
        [[nodiscard]] constexpr bool               empty() const noexcept { return _map->empty(); }
    };

    [[nodiscard]] constexpr std::size_t size() const noexcept { return _header ? _header->entryCount : 0U; }
    [[nodiscard]] constexpr bool        empty() const noexcept { return size() == 0U; }
    [[nodiscard]] constexpr std::size_t max_size() const noexcept { return std::numeric_limits<std::uint16_t>::max() - 3UZ; }

    [[nodiscard]] constexpr std::span<const std::byte> blob() const noexcept {
        if (!_header) {
            return {};
        }
        return {_blob, _header->totalSize};
    }

    // No mutationCount field: the ring model treats this as dead weight. If needed, reintroduce
    // as a per-ring atomic at the ring layer rather than on the blob.

    [[nodiscard]] constexpr const_iterator         begin() const noexcept { return const_iterator{this, 0U}; }
    [[nodiscard]] constexpr const_iterator         end() const noexcept { return const_iterator{this, static_cast<std::uint16_t>(_header ? _header->entryCount : 0U)}; }
    [[nodiscard]] constexpr const_iterator         cbegin() const noexcept { return begin(); }
    [[nodiscard]] constexpr const_iterator         cend() const noexcept { return end(); }
    [[nodiscard]] constexpr const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator{end()}; }
    [[nodiscard]] constexpr const_reverse_iterator rend() const noexcept { return const_reverse_iterator{begin()}; }
    [[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }
    [[nodiscard]] constexpr const_reverse_iterator crend() const noexcept { return rend(); }

    [[nodiscard]] constexpr keys_view keys() const noexcept { return keys_view{this}; }

    template<typename K>
    [[nodiscard]] constexpr const_iterator find(const K& key) const noexcept {
        const auto sv          = detail::keyToStringView(key);
        const auto canonicalId = keys::lookupId(sv);
        const auto n           = _header ? _header->entryCount : 0U;
        for (std::uint16_t i = 0U; i < n; ++i) {
            const PackedEntry& e = _entries[i];
            // canonical match (cheap, by id) → inline-key match → spilled-key match
            if (canonicalId != keys::kIdUnknown && e.keyId == canonicalId) {
                return const_iterator{this, i};
            }
            if (e.keyId == keys::kInlineKeyId && detail::readInlineKey(e) == sv) {
                return const_iterator{this, i};
            }
            if (e.keyId == keys::kSpilledKeyId && detail::readSpilledKey(_blob, e) == sv) {
                return const_iterator{this, i};
            }
        }
        return end();
    }

    template<typename K>
    [[nodiscard]] constexpr bool contains(const K& key) const noexcept {
        return find(key) != end();
    }

    template<typename K>
    [[nodiscard]] constexpr std::size_t count(const K& key) const noexcept {
        return contains(key) ? 1UZ : 0UZ;
    }

    template<typename K>
    [[nodiscard]] constexpr std::pair<const_iterator, const_iterator> equal_range(const K& key) const noexcept {
        const auto it = find(key);
        if (it == end()) {
            return {end(), end()};
        }
        const auto next = const_iterator{this, static_cast<std::uint16_t>(it._index + 1U)};
        return {it, next};
    }

protected:
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

    [[nodiscard]] std::uint32_t _tryAppendValueRecord(Value::ValueType vt, Value::ContainerType ct, std::span<const std::byte> contentBytes, std::uint8_t recFlags = 0U) noexcept {
        if (!_header) {
            return _capacity + 1U;
        }
        const std::uint32_t contentLen = static_cast<std::uint32_t>(contentBytes.size());
        const std::uint32_t recSize    = alignToRecord(kRecHeaderBytes + contentLen);
        const std::uint32_t curUsed    = _header->payloadUsed;
        // Align payloadUsed up to kRecAlignment so the record starts 16-aligned (see comment
        // in _appendValueRecord for the spilled-key precedent).
        const std::uint32_t alignedUsed  = (curUsed + kRecAlignment - 1U) & ~(kRecAlignment - 1U);
        const std::uint32_t padBeforeRec = alignedUsed - curUsed;
        if (recSize > _header->payloadCapacity || alignedUsed + recSize > _header->payloadCapacity) {
            return _capacity + 1U;
        }
        const std::uint32_t offset   = _header->payloadOffset + alignedUsed;
        const std::uint32_t padBytes = recSize - kRecHeaderBytes - contentLen;
        std::byte*          dst      = _blob + offset;
        if (padBeforeRec > 0U) {
            std::memset(_blob + _header->payloadOffset + curUsed, 0, padBeforeRec);
        }
        gr::wire::writeHeaderSized(dst, recSize, static_cast<std::uint8_t>(vt), static_cast<std::uint8_t>(ct), recFlags);
        if (contentLen > 0U) {
            std::memcpy(dst + kRecOffsetPayload, contentBytes.data(), contentLen);
        }
        if (padBytes > 0U) {
            std::memset(dst + kRecOffsetPayload + contentLen, 0, padBytes);
        }
        _header->payloadUsed = alignedUsed + recSize;
        return offset;
    }

    [[nodiscard]] std::uint16_t _reserveEntrySlot() noexcept {
        if (!_header) {
            return std::numeric_limits<std::uint16_t>::max();
        }
        const std::uint16_t cur = _header->entryCount;
        if (static_cast<std::uint32_t>(cur) + 1U >= _header->entryCapacity) {
            return std::numeric_limits<std::uint16_t>::max(); // need room for cur + sentinel
        }
        return cur;
    }

    void _publishEntrySlot(std::uint16_t index) noexcept {
        _entries[index + 1U].keyId = keys::kEndMarkerId; // sentinel after the new entry
        _header->entryCount        = static_cast<std::uint16_t>(index + 1U);
    }

    // Byte count of a fixed-element-type tensor's on-blob sub-blob (without the value-record
    // header). Mirrors the layout produced by detail::encodeTensorBlob for the non-Value branch.
    // Extents area is padded so subsequent element data lands at an 8-aligned offset.
    template<typename ElemT, typename TensorT>
    [[nodiscard]] static constexpr std::size_t _tensorBlobBytes(const TensorT& tensor) noexcept {
        constexpr std::size_t elemBytes = std::same_as<ElemT, bool> ? 1UZ : sizeof(ElemT);
        return kTensorBlobHeaderSize + paddedTensorExtentsBytes(tensor.rank()) + tensor.size() * elemBytes;
    }

    // Emit a fixed-element-type tensor sub-blob (no Value elements) directly into `dst`.
    // Caller guarantees `dst` has _tensorBlobBytes<ElemT>(tensor) bytes available.
    template<typename ElemT, typename TensorT>
    static void _writeTensorBlob(std::byte* dst, const TensorT& tensor) noexcept {
        constexpr Value::ValueType elemVT       = detail::cppToValueType<ElemT>();
        const std::size_t          rank         = tensor.rank();
        const std::size_t          elementCount = tensor.size();
        const auto                 extents      = tensor.extents();
        dst[0]                                  = std::byte{static_cast<std::uint8_t>(elemVT)};
        dst[1]                                  = std::byte{static_cast<std::uint8_t>(rank)};
        dst[2]                                  = std::byte{0U}; // fixed-element types: no kTensorEncodingVariableSize
        dst[3]                                  = gr::meta::kDebugBuild ? kTensorSubBlobSignature : std::byte{0U};
        const std::uint32_t ec                  = static_cast<std::uint32_t>(elementCount);
        std::memcpy(dst + 4, &ec, sizeof(ec));
        for (std::size_t i = 0UZ; i < rank; ++i) {
            const std::uint32_t ext = static_cast<std::uint32_t>(extents[i]);
            std::memcpy(dst + kTensorBlobHeaderSize + 4UZ * i, &ext, sizeof(ext));
        }
        const std::size_t dataOffset = kTensorBlobHeaderSize + paddedTensorExtentsBytes(rank);
        if (elementCount > 0UZ) {
            if constexpr (std::same_as<ElemT, bool>) {
                std::size_t i = 0UZ;
                for (bool b : tensor) {
                    dst[dataOffset + i++] = static_cast<std::byte>(b ? 1U : 0U);
                }
            } else {
                std::memcpy(dst + dataOffset, std::to_address(tensor._data.begin()), elementCount * sizeof(ElemT));
            }
        }
    }

    // Push the dying entry's value-record and spilled-key region (if any) onto the payload
    // free list. Regions smaller than kFreeChunkHeaderSize (8 B) are orphaned until the next
    // shrink_to_fit (only reachable via the owning ValueMap path).
    void _reclaimEntryPayload(const PackedEntry& dyingEntry) noexcept {
        if ((dyingEntry.flags & kEntryFlagOffsetLength) != 0U && dyingEntry.payloadLength > 0U) {
            const std::uint32_t recSize = alignToRecord(dyingEntry.payloadLength);
            if (recSize >= kFreeChunkHeaderSize) {
                FreeChunk fc{.length = recSize, .nextOffset = _header->payloadFreeHead};
                std::memcpy(_blob + dyingEntry.payloadOffset, &fc, sizeof(fc));
                _header->payloadFreeHead = dyingEntry.payloadOffset;
            }
        }
        if (dyingEntry.keyId == keys::kSpilledKeyId) {
            const auto [keyOffset, keyLength] = detail::readSpilledKeyOffsetLength(dyingEntry);
            const std::uint32_t keySpan       = keyLength + 1U; // spilled keys carry a trailing NUL outside keyLength
            if (keySpan >= kFreeChunkHeaderSize) {
                FreeChunk fc{.length = keySpan, .nextOffset = _header->payloadFreeHead};
                std::memcpy(_blob + keyOffset, &fc, sizeof(fc));
                _header->payloadFreeHead = keyOffset;
            }
        }
    }

public:
    template<typename T, typename K>
    requires(detail::InlineScalar<std::remove_cvref_t<T>>)
    [[nodiscard]] bool try_emplace(const K& key, T value) noexcept {
        const auto sv = detail::keyToStringView(key);
        if (sv.size() > kMaxInlineKeyLength) {
            return false; // spilled keys consume payload — use the owning ValueMap path
        }
        if (find(sv) != end()) {
            return false;
        }
        // Reserve payload first (16-B inline-scalar value-record).
        std::array<std::byte, 8U> bytes{};
        if constexpr (std::same_as<std::remove_cvref_t<T>, bool>) {
            bytes[0] = std::byte{static_cast<std::uint8_t>(value ? 1U : 0U)};
        } else {
            const auto buf = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
            std::memcpy(bytes.data(), buf.data(), sizeof(T));
        }

        const std::uint16_t index = _reserveEntrySlot();
        if (index == std::numeric_limits<std::uint16_t>::max()) {
            return false;
        }
        const std::uint32_t offset = _tryAppendValueRecord(detail::cppToValueType<std::remove_cvref_t<T>>(), Value::ContainerType::Scalar, std::span<const std::byte>{bytes});
        if (offset > _capacity) {
            return false;
        }
        PackedEntry& e = _entries[index];
        std::memset(&e, 0, sizeof(PackedEntry));
        const auto canonicalId = keys::lookupId(sv);
        if (canonicalId != keys::kIdUnknown) {
            e.keyId = canonicalId;
        } else {
            e.keyId = keys::kInlineKeyId;
            detail::setInlineKey(e, sv);
        }
        e.valueType     = static_cast<std::uint8_t>(detail::cppToValueType<std::remove_cvref_t<T>>());
        e.flags         = kEntryFlagOffsetLength;
        e.payloadOffset = offset;
        e.payloadLength = std::max<std::uint32_t>(kRecHeaderBytes + static_cast<std::uint32_t>(bytes.size()), kRecMinSize);
        _publishEntrySlot(index); // entryCount bump is the final store
        return true;
    }

    // bounded (ValueMapView) mutators return bool (false = no room / type mismatch); the owning ValueMap overload returns std::pair<iterator, bool>
    template<typename T, typename K>
    requires(detail::InlineScalar<std::remove_cvref_t<T>>)
    [[nodiscard]] bool insert_or_assign(const K& key, T value) noexcept {
        const auto it = find(key);
        if (it == end()) {
            return try_emplace(key, value);
        }
        PackedEntry& e = _entries[it._index];
        if ((e.flags & kEntryFlagOffsetLength) == 0U || e.payloadOffset == 0U) {
            return false;
        }
        if (static_cast<Value::ValueType>(e.valueType) == detail::cppToValueType<std::remove_cvref_t<T>>()) {
            // same type: in-place overwrite of the 8 B inline payload tail
            std::byte* payload = _blob + e.payloadOffset + kRecOffsetPayload;
            if constexpr (std::same_as<std::remove_cvref_t<T>, bool>) {
                payload[0] = std::byte{static_cast<std::uint8_t>(value ? 1U : 0U)};
            } else {
                const auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
                std::memcpy(payload, bytes.data(), sizeof(T));
            }
            return true;
        }
        // type mismatch: append a new value-record (the old payload region is leaked until
        // shrink_to_fit; the entry slot itself is recycled).
        std::array<std::byte, 8U> bytes{};
        if constexpr (std::same_as<std::remove_cvref_t<T>, bool>) {
            bytes[0] = std::byte{static_cast<std::uint8_t>(value ? 1U : 0U)};
        } else {
            const auto buf = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
            std::memcpy(bytes.data(), buf.data(), sizeof(T));
        }
        const std::uint32_t offset = _tryAppendValueRecord(detail::cppToValueType<std::remove_cvref_t<T>>(), Value::ContainerType::Scalar, std::span<const std::byte>{bytes});
        if (offset > _capacity) {
            return false;
        }
        e.valueType     = static_cast<std::uint8_t>(detail::cppToValueType<std::remove_cvref_t<T>>());
        e.flags         = kEntryFlagOffsetLength;
        e.payloadOffset = offset;
        e.payloadLength = std::max<std::uint32_t>(kRecHeaderBytes + static_cast<std::uint32_t>(bytes.size()), kRecMinSize);
        return true;
    }

    template<typename K>
    [[nodiscard]] bool try_emplace(const K& key, std::string_view value) noexcept {
        const auto sv = detail::keyToStringView(key);
        if (sv.size() > kMaxInlineKeyLength) {
            return false;
        }
        if (find(sv) != end()) {
            return false;
        }
        if (!_header) {
            return false;
        }
        const std::uint32_t contentLen   = static_cast<std::uint32_t>(value.size()) + 1U; // chars + NUL
        const std::uint32_t recSize      = alignToRecord(kRecHeaderBytes + contentLen);
        const std::uint32_t curUsed      = _header->payloadUsed;
        const std::uint32_t alignedUsed  = (curUsed + kRecAlignment - 1U) & ~(kRecAlignment - 1U);
        const std::uint32_t padBeforeRec = alignedUsed - curUsed;
        if (recSize > _header->payloadCapacity || alignedUsed + recSize > _header->payloadCapacity) {
            return false;
        }
        const std::uint16_t index = _reserveEntrySlot();
        if (index == std::numeric_limits<std::uint16_t>::max()) {
            return false;
        }
        // Write the value-record bytes directly — _tryAppendValueRecord can't append the +NUL
        // byte since it accepts a single content span. Mirror its curUsed→kRecAlignment rounding.
        const std::uint32_t offset = _header->payloadOffset + alignedUsed;
        std::byte*          dst    = _blob + offset;
        if (padBeforeRec > 0U) {
            std::memset(_blob + _header->payloadOffset + curUsed, 0, padBeforeRec);
        }
        gr::wire::writeHeaderSized(dst, recSize, static_cast<std::uint8_t>(Value::ValueType::String), static_cast<std::uint8_t>(Value::ContainerType::String), 0U);
        if (!value.empty()) {
            std::memcpy(dst + kRecOffsetPayload, value.data(), value.size());
        }
        dst[kRecOffsetPayload + value.size()] = std::byte{0}; // NUL guard
        const std::uint32_t padBytes          = recSize - kRecHeaderBytes - contentLen;
        if (padBytes > 0U) {
            std::memset(dst + kRecOffsetPayload + value.size() + 1U, 0, padBytes);
        }
        _header->payloadUsed = alignedUsed + recSize;

        PackedEntry& e = _entries[index];
        std::memset(&e, 0, sizeof(PackedEntry));
        const auto canonicalId = keys::lookupId(sv);
        if (canonicalId != keys::kIdUnknown) {
            e.keyId = canonicalId;
        } else {
            e.keyId = keys::kInlineKeyId;
            detail::setInlineKey(e, sv);
        }
        e.valueType     = static_cast<std::uint8_t>(Value::ValueType::String);
        e.flags         = kEntryFlagOffsetLength;
        e.payloadOffset = offset;
        e.payloadLength = kRecHeaderBytes + static_cast<std::uint32_t>(value.size()) + 1U;
        _publishEntrySlot(index);
        return true;
    }

    template<typename K>
    [[nodiscard]] bool insert_or_assign(const K& key, std::string_view value) noexcept {
        const auto it = find(key);
        if (it == end()) {
            return try_emplace(key, value);
        }
        PackedEntry& e = _entries[it._index];
        if (static_cast<Value::ValueType>(e.valueType) != Value::ValueType::String) {
            return false; // type-mismatch overwrite would change recSize — defer to growing path
        }
        if ((e.flags & kEntryFlagOffsetLength) == 0U || e.payloadOffset == 0U) {
            return false;
        }
        const std::uint32_t newPayloadLength = kRecHeaderBytes + static_cast<std::uint32_t>(value.size()) + 1U;
        const std::uint32_t newRecSize       = alignToRecord(newPayloadLength);
        const std::uint32_t oldRecSize       = alignToRecord(e.payloadLength);
        if (newRecSize != oldRecSize) {
            return false; // size-change requires reallocation; not bounded
        }
        std::byte* dst = _blob + e.payloadOffset + kRecOffsetPayload;
        if (!value.empty()) {
            std::memcpy(dst, value.data(), value.size());
        }
        dst[value.size()]           = std::byte{0};
        const std::uint32_t padding = newRecSize - kRecHeaderBytes - static_cast<std::uint32_t>(value.size()) - 1U;
        if (padding > 0U) {
            std::memset(dst + value.size() + 1U, 0, padding);
        }
        e.payloadLength = newPayloadLength;
        return true;
    }

    template<typename T, typename K>
    requires(detail::PayloadScalar<std::remove_cvref_t<T>>)
    [[nodiscard]] bool try_emplace(const K& key, T value) noexcept {
        const auto sv = detail::keyToStringView(key);
        if (sv.size() > kMaxInlineKeyLength) {
            return false;
        }
        if (find(sv) != end()) {
            return false;
        }
        const auto          bytes = std::span<const std::byte>{reinterpret_cast<const std::byte*>(&value), sizeof(T)};
        const std::uint16_t index = _reserveEntrySlot();
        if (index == std::numeric_limits<std::uint16_t>::max()) {
            return false;
        }
        const std::uint32_t offset = _tryAppendValueRecord(detail::cppToValueType<std::remove_cvref_t<T>>(), Value::ContainerType::Complex, bytes);
        if (offset > _capacity) {
            return false;
        }
        PackedEntry& e = _entries[index];
        std::memset(&e, 0, sizeof(PackedEntry));
        const auto canonicalId = keys::lookupId(sv);
        if (canonicalId != keys::kIdUnknown) {
            e.keyId = canonicalId;
        } else {
            e.keyId = keys::kInlineKeyId;
            detail::setInlineKey(e, sv);
        }
        e.valueType     = static_cast<std::uint8_t>(detail::cppToValueType<std::remove_cvref_t<T>>());
        e.flags         = kEntryFlagOffsetLength;
        e.payloadOffset = offset;
        e.payloadLength = static_cast<std::uint32_t>(kRecHeaderBytes + sizeof(T));
        _publishEntrySlot(index);
        return true;
    }

    template<typename T, typename K>
    requires(detail::PayloadScalar<std::remove_cvref_t<T>>)
    [[nodiscard]] bool insert_or_assign(const K& key, T value) noexcept {
        const auto it = find(key);
        if (it == end()) {
            return try_emplace(key, value);
        }
        PackedEntry& e = _entries[it._index];
        if (static_cast<Value::ValueType>(e.valueType) != detail::cppToValueType<std::remove_cvref_t<T>>()) {
            return false;
        }
        if ((e.flags & kEntryFlagOffsetLength) == 0U || e.payloadOffset == 0U) {
            return false;
        }
        // PayloadScalar (currently only std::complex<double>): fixed 24 B record, payload is 16 B.
        std::byte* payload = _blob + e.payloadOffset + kRecOffsetPayload;
        std::memcpy(payload, &value, sizeof(T));
        return true;
    }

    template<typename TensorT, typename K>
    requires(gr::TensorLike<TensorT>                                                                            //
             && detail::TensorElementType<typename gr::tensor_traits<std::remove_cvref_t<TensorT>>::value_type> //
             && !std::same_as<typename gr::tensor_traits<std::remove_cvref_t<TensorT>>::value_type, Value>)
    [[nodiscard]] bool try_emplace(const K& key, const TensorT& tensor) noexcept {
        using ElemT   = typename gr::tensor_traits<std::remove_cvref_t<TensorT>>::value_type;
        const auto sv = detail::keyToStringView(key);
        if (sv.size() > kMaxInlineKeyLength) {
            return false;
        }
        if (find(sv) != end()) {
            return false;
        }
        if (!_header) {
            return false;
        }
        if (tensor.rank() > kMaxTensorRank || tensor.size() > kMaxTensorElements) {
            return false;
        }
        const std::uint32_t contentLen = static_cast<std::uint32_t>(_tensorBlobBytes<ElemT>(tensor));
        const std::uint32_t recSize    = alignToRecord(kRecHeaderBytes + contentLen);
        const std::uint32_t curUsed    = _header->payloadUsed;
        if (recSize > _header->payloadCapacity || curUsed + recSize > _header->payloadCapacity) {
            return false;
        }
        const std::uint16_t index = _reserveEntrySlot();
        if (index == std::numeric_limits<std::uint16_t>::max()) {
            return false;
        }
        const std::uint32_t offset = _header->payloadOffset + curUsed;
        std::byte*          dst    = _blob + offset;
        gr::wire::writeHeaderSized(dst, recSize, static_cast<std::uint8_t>(detail::cppToValueType<ElemT>()), static_cast<std::uint8_t>(Value::ContainerType::Tensor), kEntryFlagTensor);
        _writeTensorBlob<ElemT>(dst + kRecOffsetPayload, tensor);
        const std::uint32_t padBytes = recSize - kRecHeaderBytes - contentLen;
        if (padBytes > 0U) {
            std::memset(dst + kRecOffsetPayload + contentLen, 0, padBytes);
        }
        _header->payloadUsed += recSize;

        PackedEntry& e = _entries[index];
        std::memset(&e, 0, sizeof(PackedEntry));
        const auto canonicalId = keys::lookupId(sv);
        if (canonicalId != keys::kIdUnknown) {
            e.keyId = canonicalId;
        } else {
            e.keyId = keys::kInlineKeyId;
            detail::setInlineKey(e, sv);
        }
        e.valueType     = static_cast<std::uint8_t>(detail::cppToValueType<ElemT>());
        e.flags         = kEntryFlagOffsetLength | kEntryFlagTensor;
        e.payloadOffset = offset;
        e.payloadLength = kRecHeaderBytes + contentLen;
        _publishEntrySlot(index);
        return true;
    }

    template<typename TensorT, typename K>
    requires(gr::TensorLike<TensorT>                                                                            //
             && detail::TensorElementType<typename gr::tensor_traits<std::remove_cvref_t<TensorT>>::value_type> //
             && !std::same_as<typename gr::tensor_traits<std::remove_cvref_t<TensorT>>::value_type, Value>)
    [[nodiscard]] bool insert_or_assign(const K& key, const TensorT& tensor) noexcept {
        using ElemT   = typename gr::tensor_traits<std::remove_cvref_t<TensorT>>::value_type;
        const auto it = find(key);
        if (it == end()) {
            return try_emplace(key, tensor);
        }
        PackedEntry& e = _entries[it._index];
        if ((e.flags & kEntryFlagTensor) == 0U || static_cast<Value::ValueType>(e.valueType) != detail::cppToValueType<ElemT>()) {
            return false;
        }
        if ((e.flags & kEntryFlagOffsetLength) == 0U || e.payloadOffset == 0U) {
            return false;
        }
        if (tensor.rank() > kMaxTensorRank || tensor.size() > kMaxTensorElements) {
            return false;
        }
        const std::uint32_t newPayloadLength = kRecHeaderBytes + static_cast<std::uint32_t>(_tensorBlobBytes<ElemT>(tensor));
        const std::uint32_t newRecSize       = alignToRecord(newPayloadLength);
        const std::uint32_t oldRecSize       = alignToRecord(e.payloadLength);
        if (newRecSize != oldRecSize) {
            return false; // byte-count must match (rank/elementCount can reshape within that bound)
        }
        std::byte* tensorDst = _blob + e.payloadOffset + kRecOffsetPayload;
        _writeTensorBlob<ElemT>(tensorDst, tensor);
        const std::uint32_t contentLen = newPayloadLength - kRecHeaderBytes;
        const std::uint32_t padBytes   = newRecSize - kRecHeaderBytes - contentLen;
        if (padBytes > 0U) {
            std::memset(tensorDst + contentLen, 0, padBytes);
        }
        e.payloadLength = newPayloadLength;
        return true;
    }

    const_iterator erase(const_iterator it) noexcept {
        if (!_header || it._map != this || it._index >= _header->entryCount) {
            return end();
        }
        const PackedEntry& dyingEntry   = _entries[it._index];
        const auto         removedIndex = it._index;
        _reclaimEntryPayload(dyingEntry);
        const auto n = _header->entryCount;
        for (std::uint16_t i = removedIndex; i + 1U < n; ++i) {
            _entries[i] = _entries[i + 1U];
        }
        --_header->entryCount;
        if (_header->entryCount < _header->entryCapacity) {
            _entries[_header->entryCount].keyId = keys::kEndMarkerId;
        }
        return const_iterator{this, removedIndex};
    }

    const_iterator erase(const_iterator first, const_iterator last) noexcept {
        if (!_header) {
            return end();
        }
        const std::uint16_t n = _header->entryCount;
        if (first._map != this || last._map != this || first._index > last._index || first._index > n) {
            return end();
        }
        const std::uint16_t lo = first._index;
        const std::uint16_t hi = std::min<std::uint16_t>(last._index, n);
        for (std::uint16_t i = lo; i < hi; ++i) {
            _reclaimEntryPayload(_entries[i]);
        }
        const std::uint16_t tail = static_cast<std::uint16_t>(n - hi);
        if (tail > 0U) {
            std::memmove(&_entries[lo], &_entries[hi], static_cast<std::size_t>(tail) * sizeof(PackedEntry));
        }
        _header->entryCount = static_cast<std::uint16_t>(lo + tail);
        if (_header->entryCount < _header->entryCapacity) {
            _entries[_header->entryCount].keyId = keys::kEndMarkerId;
        }
        return const_iterator{this, lo};
    }

    template<typename K>
    requires(!std::same_as<std::remove_cvref_t<K>, const_iterator>)
    std::size_t erase(const K& key) noexcept {
        const auto it = find(key);
        if (it == end()) {
            return 0UZ;
        }
        std::ignore = erase(it);
        return 1UZ;
    }

    void clear() noexcept {
        if (!_header) {
            return;
        }
        _header->entryCount      = 0U;
        _header->payloadUsed     = 0U;
        _header->payloadFreeHead = 0U;
        if (_entries) {
            _entries[0].keyId = keys::kEndMarkerId;
        }
    }

    [[nodiscard]] ValueMap owned(std::pmr::memory_resource* resource = std::pmr::get_default_resource()) const;

    [[nodiscard]] bool operator==(const ValueMapView& other) const {
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

    // Read-only value extractors — base-member-only (find/_entries/_blob), so they live on the non-owning view: a
    // `Tag` (map = ValueMapView) reads through these. No-resource find_value returns a blob-view Value (alloc-free,
    // valid while the blob lives); pass a resource for an owning copy.
    template<typename K>
    [[nodiscard]] std::optional<Value> find_value(const K& key, std::pmr::memory_resource* resource = nullptr) const {
        const auto it = find(key);
        if (it == end()) {
            return std::nullopt;
        }
        const auto& e    = _entries[it._index];
        Value       view = ((e.flags & kEntryFlagOffsetLength) != 0U && e.payloadOffset != 0U) //
                               ? Value::makeView(Value::ValueType::Monostate, Value::ContainerType::Scalar, _blob + e.payloadOffset, e.payloadLength, nullptr)
                               : Value{};
        return resource ? Value{view, resource} : view;
    }

    template<typename T, typename K>
    [[nodiscard]] auto get_if(const K& key) const noexcept {
        // no resource by design — returns a pointer into the blob; an owning temporary here would dangle
        using Result = decltype(std::declval<const Value&>().template get_if<T>());
        if (auto opt = find_value(key)) {
            return std::as_const(*opt).template get_if<T>();
        }
        return Result{};
    }

    template<typename T, typename K, typename U = T>
    [[nodiscard]] T value_or(const K& key, U&& def, std::pmr::memory_resource* resource = nullptr) const {
        // no explicit template arg — lets Value's non-template string / string_view overloads win.
        if (auto opt = find_value(key, resource)) {
            return std::as_const(*opt).value_or(T(std::forward<U>(def)));
        }
        return T(std::forward<U>(def));
    }

    template<typename T, typename K, typename F>
    [[nodiscard]] T or_else(const K& key, F&& factory, std::pmr::memory_resource* resource = nullptr) const {
        if (auto opt = find_value(key, resource)) {
            return std::as_const(*opt).template or_else<T>(std::forward<F>(factory));
        }
        return std::forward<F>(factory)();
    }

    template<typename T, typename K>
    [[nodiscard]] bool holds(const K& key) const noexcept {
        auto opt = find_value(key);
        return opt && opt->template holds<T>();
    }

    template<typename K, typename F>
    [[nodiscard]] auto transform(const K& key, F&& func, std::pmr::memory_resource* resource = nullptr) const -> std::optional<std::invoke_result_t<F, const Value&>> {
        if (auto v = find_value(key, resource)) {
            return std::invoke(std::forward<F>(func), std::as_const(*v));
        }
        return std::nullopt;
    }
};
static_assert(std::is_trivially_copyable_v<ValueMapView>);

class ValueMap : public ValueMapView {
public:
    using key_type        = std::string_view;
    using mapped_type     = Value;
    using value_type      = std::pair<std::string_view, Value>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type; // by value (matches const_iterator::reference; on-blob layout has no stored pair)
    using const_reference = value_type;
    using pointer         = void; // matches const_iterator::pointer (no addressable element to point at)
    using const_pointer   = void;
    using allocator_type  = std::pmr::polymorphic_allocator<value_type>;

    ValueMap(std::pmr::memory_resource* resource = std::pmr::get_default_resource(), std::uint32_t initialCapacityEntries = 8U, std::uint32_t initialPayloadCapacity = 0U) : _resource(resource ? resource : std::pmr::get_default_resource()) { _allocateBlob(initialCapacityEntries, initialPayloadCapacity); }
    ValueMap(const ValueMap& other, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(resource ? resource : std::pmr::get_default_resource()) { _copyBlobFrom(other); }
    explicit ValueMap(const allocator_type& alloc) : ValueMap(alloc.resource()) {}
    ValueMap(const ValueMap& other, const allocator_type& alloc) : ValueMap(other, alloc.resource()) {}

    [[nodiscard]] static ValueMap makeView(std::span<const std::byte> bytes) noexcept { return ValueMap{ViewModeTag{}, bytes}; }

    [[nodiscard]] static ValueMap makeAt(std::span<std::byte> slot, std::uint32_t payloadCapacity, std::uint32_t entryCapacity = 8U) noexcept {
        const auto entryBytes   = entryCapacity * static_cast<std::uint32_t>(sizeof(PackedEntry));
        const auto payloadStart = static_cast<std::uint32_t>(sizeof(Header)) + entryBytes;
        const auto required     = payloadStart + payloadCapacity;
        if (slot.size() < required) {
            return ValueMap{ViewModeTag{}, std::span<const std::byte>{slot.data(), 0UZ}};
        }
        if ((reinterpret_cast<std::uintptr_t>(slot.data()) & (kBlobAlignment - 1UZ)) != 0UZ) {
            return ValueMap{ViewModeTag{}, std::span<const std::byte>{slot.data(), 0UZ}};
        }
        // Initialise the wire-format header in place.
        auto* hdr = std::launder(reinterpret_cast<Header*>(slot.data()));
        std::memcpy(hdr->magic, kBlobMagic.data(), kBlobMagic.size());
        hdr->version         = kBlobVersion;
        hdr->flags           = gr::meta::kDebugBuild ? kHeaderFlagDebugGuards : std::uint8_t{0};
        hdr->entryCount      = 0U;
        hdr->entryCapacity   = static_cast<std::uint16_t>(entryCapacity);
        hdr->_reserved       = 0U;
        hdr->totalSize       = required;
        hdr->payloadOffset   = payloadStart;
        hdr->payloadUsed     = 0U;
        hdr->payloadCapacity = payloadCapacity;
        hdr->payloadFreeHead = 0U;
        // Zero the entry array (sentinel emission needs a defined kEndMarkerId at row[0]).
        std::memset(slot.data() + sizeof(Header), 0, entryBytes);
        // Sentinel: row[entryCount] = row[0] gets keyId = kEndMarkerId.
        auto* entries    = std::launder(reinterpret_cast<PackedEntry*>(slot.data() + sizeof(Header)));
        entries[0].keyId = keys::kEndMarkerId;
        // Fixed-buffer-mutable: not a view (mutators run) but _resource is the no-realloc sentinel.
        ValueMap m{ViewModeTag{}, std::span<const std::byte>{slot.data(), required}};
        m._resource = _fixedBufferResource();
        return m;
    }

    ValueMap(std::initializer_list<value_type> init, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(resource ? resource : std::pmr::get_default_resource()) {
        _allocateBlob(static_cast<std::uint32_t>(std::max<std::size_t>(init.size(), 8UZ)), 0U);
        for (const auto& p : init) {
            std::ignore = emplace(p.first, p.second);
        }
    }

    template<typename InputIt>
    requires requires(InputIt it) {
        { (*it).first };
        { (*it).second };
    }
    ValueMap(InputIt first, InputIt last, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(resource ? resource : std::pmr::get_default_resource()) {
        _allocateBlob(8U, 0U);
        for (; first != last; ++first) {
            const auto& p = *first;
            std::ignore   = emplace(p.first, p.second);
        }
    }

    [[nodiscard]] constexpr bool is_view() const noexcept { return _resource == nullptr; }

    ValueMap(const ValueMapView& other, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(resource ? resource : std::pmr::get_default_resource()) { _copyBlobFrom(other); }
    ValueMap(ValueMap&& other) noexcept          //
        : ValueMapView{._blob = other._blob,     //
              ._capacity      = other._capacity, //
              ._header        = other._header,   //
              ._entries       = other._entries},       //
          _resource(other._resource) {
        other._blob     = nullptr;
        other._capacity = 0U;
        other._header   = nullptr;
        other._entries  = nullptr;
        // other._resource left valid so other may be reassigned or destroyed.
    }
    ValueMap(ValueMap&& other, const allocator_type& alloc) noexcept : ValueMap(std::move(other), alloc.resource()) {}
    ValueMap(ValueMap&& other, std::pmr::memory_resource* resource) : _resource(resource ? resource : std::pmr::get_default_resource()) {
        if (other._resource == _resource) {
            _blob           = other._blob;
            _capacity       = other._capacity;
            _header         = other._header;
            _entries        = other._entries;
            other._blob     = nullptr;
            other._capacity = 0U;
            other._header   = nullptr;
            other._entries  = nullptr;
        } else {
            _copyBlobFrom(other);
        }
    }

    template<typename Map>
    requires requires {
        typename Map::key_type;
        typename Map::mapped_type;
        requires std::convertible_to<typename Map::key_type, std::string_view>;
        requires std::same_as<std::remove_cvref_t<typename Map::mapped_type>, Value>;
    }
    explicit ValueMap(const Map& src, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(resource ? resource : std::pmr::get_default_resource()) {
        _allocateBlob(static_cast<std::uint32_t>(std::max<std::size_t>(src.size(), 8UZ)), 0U);
        for (const auto& [key, value] : src) {
            std::ignore = emplace(std::string_view{key}, value);
        }
    }

    ValueMap& operator=(const ValueMap& other) {
        if (this == &other) {
            return *this;
        }
        if (_resource == other._resource && _blob != nullptr && other._capacity > 0U && _capacity >= other._capacity) {
            std::memcpy(_blob, other._blob, other._capacity);
            _header  = std::launder(reinterpret_cast<Header*>(_blob));
            _entries = std::launder(reinterpret_cast<PackedEntry*>(_blob + sizeof(Header)));
            return *this;
        }
        ValueMap tmp(other, _resource);
        swap(*this, tmp);
        return *this;
    }

    ValueMap& operator=(std::initializer_list<value_type> init) {
        clear();
        insert(init);
        return *this;
    }

    ValueMap& operator=(ValueMap&& other) noexcept {
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
            ValueMap tmp(other, _resource); // may allocate via _resource → terminate on OOM
            swap(*this, tmp);
            // invalidate source per move-semantics
            other.clear();
        }
        return *this;
    }

    // Re-home from a packed-blob view, reusing the blob in place when the resource matches and capacity suffices
    // (no allocation), else reconstructing on `resource`. Single-writer-exclusive: a full-blob overwrite is not
    // concurrent-reader-safe.
    void assignFrom(const ValueMapView& src, std::pmr::memory_resource* resource) noexcept {
        if (_blob != nullptr && src._blob == _blob) { // self-assignment via an aliasing view: dest already holds src's bytes (avoids an overlapping memcpy)
            return;
        }
        std::pmr::memory_resource* res = resource ? resource : std::pmr::get_default_resource();
        if (_resource == res && _blob != nullptr && src._header != nullptr && src._capacity > 0U && _capacity >= src._capacity) {
            std::memcpy(_blob, src._blob, src._capacity);
            _header  = std::launder(reinterpret_cast<Header*>(_blob));
            _entries = std::launder(reinterpret_cast<PackedEntry*>(_blob + sizeof(Header)));
            return;
        }
        ValueMap tmp(src, res);
        swap(*this, tmp);
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
        assert(!is_view() && "ValueMap: cannot mutate a view-mode (resource == nullptr) ValueMap");
        if (is_view()) [[unlikely]] {
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
        assert(!is_view() && "ValueMap: cannot mutate a view-mode (resource == nullptr) ValueMap");
        if (is_view()) [[unlikely]] {
            return {end(), false};
        }
        const auto sv = detail::keyToStringView(key);
        if (auto it = find(sv); it != end()) {
            if (!_assignValueAt(it._index, std::forward<V>(value))) {
                return {end(), false};
            }
            return {it, false};
        }
        const_iterator new_it = _insertNew(sv, std::forward<V>(value));
        return {new_it, new_it != end()};
    }

    template<typename K, typename V>
    const_iterator emplace_hint(const_iterator /*hint*/, K&& key, V&& value) {
        return emplace(std::forward<K>(key), std::forward<V>(value)).first;
    }

    template<typename K, typename... Args>
    std::pair<const_iterator, bool> try_emplace(K&& key, Args&&... args) {
        assert(!is_view() && "ValueMap: cannot mutate a view-mode (resource == nullptr) ValueMap");
        if (is_view()) [[unlikely]] {
            return {end(), false};
        }
        const auto sv = detail::keyToStringView(key);
        if (auto it = find(sv); it != end()) {
            return {it, false};
        }
        const_iterator new_it = _insertNew(sv, std::forward<Args>(args)...);
        return {new_it, new_it != end()};
    }

    std::pair<const_iterator, bool> insert(const value_type& p) { return emplace(p.first, p.second); }

    template<typename PairT>
    requires requires(const PairT& p) {
        { p.first };
        { p.second };
    } && (!std::same_as<std::remove_cvref_t<PairT>, value_type>)
    std::pair<const_iterator, bool> insert(const PairT& p) {
        return emplace(p.first, p.second);
    }

    template<typename PairT>
    requires requires(const PairT& p) {
        { p.first };
        { p.second };
    }
    std::pair<const_iterator, bool> insert_or_assign(const PairT& p) {
        return insert_or_assign(p.first, p.second);
    }

    template<typename InputIt>
    void insert(InputIt first, InputIt last) {
        for (; first != last; ++first) {
            const auto& p = *first;
            std::ignore   = emplace(p.first, p.second);
        }
    }

    template<typename PairT>
    requires requires(const PairT& p) {
        { p.first };
        { p.second };
    }
    void insert(std::initializer_list<PairT> init) {
        for (const auto& p : init) {
            std::ignore = emplace(p.first, p.second);
        }
    }

    void clear() noexcept {
        assert(!is_view() && "ValueMap: cannot mutate a view-mode (resource == nullptr) ValueMap");
        if (is_view()) [[unlikely]] {
            return;
        }
        if (_header) {
            _header->entryCount      = 0U;
            _header->payloadUsed     = 0U;
            _header->payloadFreeHead = 0U; // pool reset → no orphaned regions to chain
            if constexpr (gr::meta::kDebugBuild) {
                // Zero-fill the (now-empty) entry array + slack-fill the (now-empty) payload pool
                // so a hex dump of a freshly-cleared blob shows obviously-empty bytes.
                std::memset(_entries, 0, _entryCapacity() * sizeof(PackedEntry));
                if (_payloadCapacity() > 0U) {
                    std::memset(_blob + _header->payloadOffset, std::to_integer<int>(kSlackFillPattern), _payloadCapacity());
                }
                _writePayloadEndMarker();
            }
            _writeEntrySentinel();
        }
    }

    void reserve(std::uint32_t entries, std::uint32_t payload_bytes = 0U) {
        assert(!is_view() && "ValueMap: cannot mutate a view-mode (resource == nullptr) ValueMap");
        if (is_view()) [[unlikely]] {
            return;
        }
        const auto cur_cap = _entryCapacity();
        // +1 for the entry sentinel (kEndMarkerId at row[entryCount]) so the always-on guarantee
        // survives the case where the user reserves exactly N and inserts exactly N entries.
        const auto needed = std::max<std::uint32_t>(entries + 1U, (_header ? _header->entryCount : 0U) + 1U);
        if (needed > cur_cap || payload_bytes > _payloadCapacity()) {
            _grow(needed, payload_bytes);
        }
    }

    void shrink_to_fit() {
        assert(!is_view() && "ValueMap: cannot mutate a view-mode (resource == nullptr) ValueMap");
        if (is_view()) [[unlikely]] {
            return;
        }
        const auto entries = _header ? _header->entryCount : 0U;
        const auto payload = _header ? _header->payloadUsed : 0U;
        _grow(entries, payload, /*shrink=*/true);
    }

    template<std::ranges::input_range Keys>
    requires std::convertible_to<std::ranges::range_reference_t<Keys>, std::string_view>
    [[nodiscard]] ValueMap project(const Keys& keys, std::pmr::memory_resource* resource = nullptr) const {
        ValueMap result(resource ? resource : (_resource ? _resource : std::pmr::get_default_resource()));
        for (const auto& key : keys) {
            if (const auto v = find_value(std::string_view{key})) {
                result.insert_or_assign(std::string_view{key}, *v);
            }
        }
        return result;
    }

    template<std::predicate<std::string_view> Pred>
    [[nodiscard]] ValueMap filter_keys(Pred&& pred, std::pmr::memory_resource* resource = nullptr) const {
        ValueMap result(resource ? resource : (_resource ? _resource : std::pmr::get_default_resource()));
        for (auto it = begin(); it != end(); ++it) {
            const std::string_view k{it->first};
            if (std::invoke(pred, k)) {
                result.insert_or_assign(k, it->second);
            }
        }
        return result;
    }

    class SubscriptProxy {
        ValueMap*        _map;
        Value            _cache;
        bool             _wasMissing;
        std::string_view _key; // aliases the caller's key; the non-copyable/non-movable proxy must not outlive it

    public:
        SubscriptProxy(ValueMap* m, std::string_view k) noexcept(false) : _map(m), _cache(m->_resource), _wasMissing(false), _key(k) {
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

        operator Value() const& { return _cache; }
        operator Value() && { return std::move(_cache); }

        template<typename V>
        requires(!std::is_same_v<std::remove_cvref_t<V>, SubscriptProxy>)
        SubscriptProxy& operator=(V&& v) {
            _map->insert_or_assign(_key, v);
            _cache = Value{std::forward<V>(v), _map->_resource};
            return *this;
        }

        template<typename T>
        [[nodiscard]] bool holds() const {
            return _cache.template holds<T>();
        }

        [[nodiscard]] constexpr bool is_monostate() const noexcept { return _cache.is_monostate(); }
        [[nodiscard]] constexpr bool is_string() const noexcept { return _cache.is_string(); }
        [[nodiscard]] constexpr bool is_tensor() const noexcept { return _cache.is_tensor(); }
        [[nodiscard]] constexpr bool is_map() const noexcept { return _cache.is_map(); }

        template<typename T>
        [[nodiscard]] auto value_or(T&& def) const {
            return _cache.value_or(std::forward<T>(def));
        }

        [[nodiscard]] constexpr Value::ValueType     value_type() const noexcept { return _cache.value_type(); }
        [[nodiscard]] constexpr Value::ContainerType container_type() const noexcept { return _cache.container_type(); }
    };

    template<typename K>
    [[nodiscard]] SubscriptProxy operator[](const K& key) {
        return SubscriptProxy{this, detail::keyToStringView(key)};
    }

    template<gr::meta::fixed_string Name, typename Self>
    requires keys::CanonicalName<Name>
    [[nodiscard]] decltype(auto) at(this Self&& self) {
        using T            = keys::CanonicalCppType<keys::idOf<Name>>;
        constexpr auto kId = keys::idOf<Name>;
        // Linear scan for the canonical key id; entries hold the id directly (no string compare).
        const std::span entries{self._entries, self._header ? self._header->entryCount : std::uint16_t{0}};
        const auto      it  = std::ranges::find(entries, kId, &PackedEntry::keyId);
        const auto      idx = static_cast<std::uint16_t>(it - entries.begin());
        if (idx >= self.size()) [[unlikely]] {
            assert(false && "ValueMap::at<Name>: canonical key not present"); // precondition: caller ensured presence (e.g. via contains/get_if)
            std::unreachable();
        }
        if constexpr (gr::meta::kDebugBuild) {
            constexpr auto expectedVt = keys::boundTypeOf<kId>;
            assert(static_cast<Value::ValueType>(self._entries[idx].valueType) == expectedVt //
                   && "ValueMap::at<Name>: stored type does not match canonical binding for Name");
        }
        if constexpr (std::same_as<T, std::string_view>) {
            return self._readStringView(self._entries[idx]);
        } else {
            // C3: inline scalars live as a 16-B value-record in the payload pool — return a
            // reference into the record's 8-B inline payload region (record header is 8 B).
            // Lifetime: bound to the map's blob.
            using PtrT       = std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, const T*, T*>;
            using BytePtrT   = std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, const std::byte*, std::byte*>;
            BytePtrT recBase = self._blob + self._entries[idx].payloadOffset;
            return *std::launder(reinterpret_cast<PtrT>(recBase + kRecHeaderBytes));
        }
    }

    void merge(ValueMap&& other) { merge(other); }

    void merge(ValueMap& other) {
        assert(!is_view() && "ValueMap: cannot mutate a view-mode (resource == nullptr) ValueMap");
        if (is_view()) [[unlikely]] {
            return;
        }
        if (this == &other || !other._header) {
            return;
        }
        const std::uint16_t nSrc = other._header->entryCount;
        if (nSrc == 0U) {
            return;
        }

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

        constexpr std::size_t                         kInlineBitmapWords = 4UZ;
        std::array<std::uint64_t, kInlineBitmapWords> inlineBits{};
        std::pmr::vector<std::uint64_t>               spillBits(_resource ? _resource : std::pmr::get_default_resource());
        const std::size_t                             wordCount  = (static_cast<std::size_t>(nSrc) + 63UZ) / 64UZ;
        std::uint64_t* const                          movedWords = (wordCount <= kInlineBitmapWords) ? inlineBits.data() : (spillBits.assign(wordCount, 0ULL), spillBits.data());
        const auto                                    markMoved  = [&](std::uint16_t i) noexcept { movedWords[i >> 6] |= (1ULL << (i & 63U)); };
        const auto                                    wasMoved   = [&](std::uint16_t i) noexcept { return (movedWords[i >> 6] & (1ULL << (i & 63U))) != 0ULL; };
        for (std::uint16_t i = 0U; i < nSrc; ++i) {
            const PackedEntry& sourceEntry = other._entries[i];
            const auto         key         = other._readKey(sourceEntry);
            if (contains(key)) {
                continue;
            }
            if (!_ensureEntrySlot()) {
                break; // fixed-buffer capacity exhausted: partial merge, kHeaderFlagOverflow set
            }
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
                if (newKeyOffset == kInvalidOffset) {
                    break;
                }
                detail::setSpilledKey(_entries[destIndex], newKeyOffset, static_cast<std::uint32_t>(key.size()));
            }
            if ((sourceEntry.flags & kEntryFlagOffsetLength) != 0U) {
                // Copy variable-size payload bytes verbatim (string / complex<double> / nested
                // map sub-blob / tensor sub-blob — all have offsets internal to the source blob
                // OR no internal offsets at all, so byte-copy preserves correctness).
                const std::span<const std::byte> src{other._blob + sourceEntry.payloadOffset, sourceEntry.payloadLength};
                const auto                       newOffset = _appendPayloadSafe(src);
                if (newOffset == kInvalidOffset) {
                    break;
                }
                _entries[destIndex].payloadOffset = newOffset; // re-index after possible grow
            }
            committed = true;
            markMoved(i);
        }
        // The loop bypasses _assignValueAt; re-emit the trailing sentinel + run the no-op
        // mutation hook here so future readers / consistency checks see a finalised state.
        _writeEntrySentinel();
        // Batch-compact source: write surviving entries to the front, drop trailing slack.
        std::uint16_t writeIdx = 0U;
        for (std::uint16_t i = 0U; i < nSrc; ++i) {
            if (!wasMoved(i)) {
                if (writeIdx != i) {
                    other._entries[writeIdx] = other._entries[i];
                }
                ++writeIdx;
            }
        }
        if (writeIdx != nSrc) {
            other._header->entryCount = writeIdx;
            other._writeEntrySentinel();
        }
    }

    [[nodiscard]] constexpr std::pmr::memory_resource* resource() const noexcept { return _resource; }
    [[nodiscard]] allocator_type                       get_allocator() const noexcept { return allocator_type{_resource}; }

    [[nodiscard]] static std::expected<void, DeserialiseError> _validateTensorSubBlob(std::span<const std::byte> tensorBytes, std::uint32_t depth) {
        if (depth >= detail::kMaxDecodeDepth) {
            return std::unexpected{DeserialiseError::CorruptOffset};
        }
        if (tensorBytes.size() < kTensorBlobHeaderSize) {
            return std::unexpected{DeserialiseError::CorruptOffset};
        }
        const auto elementVT     = static_cast<Value::ValueType>(static_cast<std::uint8_t>(tensorBytes[0]));
        const auto rank          = static_cast<std::size_t>(static_cast<std::uint8_t>(tensorBytes[1]));
        const auto encodingFlags = static_cast<std::uint8_t>(tensorBytes[2]);
        const bool variableSize  = (encodingFlags & kTensorEncodingVariableSize) != 0U;
        if (rank > kMaxTensorRank) {
            return std::unexpected{DeserialiseError::CorruptOffset};
        }
        std::uint32_t elementCount;
        std::memcpy(&elementCount, tensorBytes.data() + 4, sizeof(elementCount));
        if (elementCount > kMaxTensorElements) {
            return std::unexpected{DeserialiseError::CorruptOffset};
        }
        const std::size_t extentsBytes = paddedTensorExtentsBytes(rank);
        if (kTensorBlobHeaderSize + extentsBytes > tensorBytes.size()) {
            return std::unexpected{DeserialiseError::CorruptOffset};
        }
        const std::size_t elementDataOff = kTensorBlobHeaderSize + extentsBytes;
        const std::size_t elementBytes   = tensorBytes.size() - elementDataOff;

        // Cross-check: variable-size encoding bit must agree with element-VT shape class.
        const bool elementIsVarShape = elementVT == Value::ValueType::Value || elementVT == Value::ValueType::String;
        if (variableSize != elementIsVarShape) {
            return std::unexpected{DeserialiseError::CorruptOffset};
        }

        if (!variableSize) {
            const std::size_t elemSize = detail::dispatchValueType(elementVT, []<typename T>(std::type_identity<T>) -> std::size_t {
                if constexpr (detail::TensorElementType<T> && !std::same_as<T, Value>) {
                    return sizeof(T);
                } else {
                    return 0UZ;
                }
            });
            if (elemSize == 0UZ) {
                return std::unexpected{DeserialiseError::CorruptOffset};
            }
            if (static_cast<std::size_t>(elementCount) > elementBytes / elemSize) {
                return std::unexpected{DeserialiseError::CorruptOffset};
            }
            return {};
        }

        // Variable-size: walk PackedTensorElement[]. Element payloads are raw sub-blob bytes
        // — no `[size:4][vt:1][ct:1][flags:1][payloadOffset:1]` value-record wrapper (cf. encodeTensorElement).
        const std::byte* elementData = tensorBytes.data() + elementDataOff;
        std::size_t      offset      = 0UZ;
        for (std::uint32_t i = 0U; i < elementCount; ++i) {
            if (sizeof(PackedTensorElement) > elementBytes - offset) {
                return std::unexpected{DeserialiseError::CorruptOffset};
            }
            PackedTensorElement headerCopy;
            std::memcpy(&headerCopy, elementData + offset, sizeof(headerCopy));
            offset += sizeof(headerCopy);
            if (static_cast<std::size_t>(headerCopy.payloadLength) > elementBytes - offset) {
                return std::unexpected{DeserialiseError::CorruptOffset};
            }
            const auto                       subVT = static_cast<Value::ValueType>(headerCopy.valueType);
            const std::span<const std::byte> subBytes{elementData + offset, headerCopy.payloadLength};
            if (subVT == Value::ValueType::Value && (headerCopy.flags & kEntryFlagTensor) != 0U) {
                if (auto inner = _validateTensorSubBlob(subBytes, depth + 1U); !inner.has_value()) {
                    return std::unexpected{inner.error()};
                }
            } else if (subVT == Value::ValueType::Value && (headerCopy.flags & kEntryFlagNestedMap) != 0U) {
                if (auto inner = _validateBlob(subBytes, depth + 1U); !inner.has_value()) {
                    return std::unexpected{inner.error()};
                }
            }
            const std::uint32_t paddedPL = paddedElementPayloadBytes(headerCopy.payloadLength);
            offset += std::min<std::uint32_t>(paddedPL, static_cast<std::uint32_t>(elementBytes - offset));
        }
        return {};
    }

    [[nodiscard]] static std::expected<void, DeserialiseError> _validateBlob(std::span<const std::byte> bytes, std::uint32_t depth = 0U) {
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

                if (e.payloadLength < kRecHeaderBytes) {
                    return std::unexpected{DeserialiseError::CorruptOffset};
                }
                const std::uint32_t recordSize         = gr::wire::elementSize(bytes.data() + e.payloadOffset);
                const std::uint32_t expectedRecordSize = alignToRecord(e.payloadLength);
                if (recordSize != expectedRecordSize) {
                    return std::unexpected{DeserialiseError::CorruptOffset};
                }
                // Record-prefix sanity at the trust boundary: the payloadOffset byte must be the
                // canonical kPrefixBytes (the decoder reads content at a fixed +kRecHeaderBytes),
                // and valueType must be a known enum (decode dispatches on it). Reject malformed
                // prefixes here rather than relying on the decoder's graceful-default handling.
                if (gr::wire::payloadOffset(bytes.data() + e.payloadOffset) != gr::wire::kPrefixBytes) {
                    return std::unexpected{DeserialiseError::CorruptOffset};
                }
                if (e.valueType > static_cast<std::uint8_t>(Value::ValueType::Value)) {
                    return std::unexpected{DeserialiseError::CorruptOffset};
                }
                // Nested-map / tensor entry — recurse into the value-record's inner sub-blob.
                // The value-record is `[size:4][vt:1][ct:1][flags:1][payloadOffset:1][sub-blob...]`, so the
                // sub-blob starts at +8 and runs for (payloadLength - 8) bytes.
                const bool isNestedMap = (e.flags & kEntryFlagNestedMap) != 0U;
                const bool isTensor    = (e.flags & kEntryFlagTensor) != 0U;
                if (isNestedMap || isTensor) {
                    if (depth + 1U >= detail::kMaxDecodeDepth) {
                        return std::unexpected{DeserialiseError::CorruptOffset};
                    }
                    const std::size_t minSubBlobSize = isNestedMap ? sizeof(Header) : kTensorBlobHeaderSize;
                    if (e.payloadLength < kRecHeaderBytes + minSubBlobSize) {
                        return std::unexpected{DeserialiseError::CorruptOffset};
                    }
                    const std::span<const std::byte> innerBlob{bytes.data() + e.payloadOffset + kRecHeaderBytes, e.payloadLength - kRecHeaderBytes};
                    auto                             inner = isNestedMap ? _validateBlob(innerBlob, depth + 1U) : _validateTensorSubBlob(innerBlob, depth + 1U);
                    if (!inner.has_value()) {
                        return std::unexpected{inner.error()};
                    }
                }
            }
            if (e.keyId == keys::kInlineKeyId) {
                if (static_cast<std::uint8_t>(e.inlineKey[0]) > kMaxInlineKeyLength) {
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

        const std::size_t walkCap = static_cast<std::size_t>(hdr->totalSize) / kFreeChunkHeaderSize + 1UZ;
        std::uint32_t     cur     = hdr->payloadFreeHead;
        std::size_t       walked  = 0UZ;
        while (cur != 0U) {
            if (++walked > walkCap) {
                return std::unexpected{DeserialiseError::CorruptOffset};
            }
            if (cur < hdr->payloadOffset || cur > hdr->totalSize - kFreeChunkHeaderSize) {
                return std::unexpected{DeserialiseError::CorruptOffset};
            }
            FreeChunk fc;
            std::memcpy(&fc, bytes.data() + cur, sizeof(fc));
            if (fc.length < kFreeChunkHeaderSize || fc.length > hdr->totalSize - cur) {
                return std::unexpected{DeserialiseError::CorruptOffset};
            }
            cur = fc.nextOffset;
        }
        return {};
    }

    [[nodiscard]] static std::expected<ValueMap, DeserialiseError> from_blob(std::span<const std::byte> bytes, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) {
        if (auto v = _validateBlob(bytes); !v.has_value()) {
            return std::unexpected{v.error()};
        }
        const auto* hdr = std::launder(reinterpret_cast<const Header*>(bytes.data()));
        // All bounds OK — copy bytes into our own owned blob.
        ValueMap out{resource ? resource : std::pmr::get_default_resource()};
        out._deallocateBlob();
        out._blob     = out._alignedAllocate(hdr->totalSize);
        out._capacity = hdr->totalSize;
        std::memcpy(out._blob, bytes.data(), hdr->totalSize);
        out._header  = std::launder(reinterpret_cast<Header*>(out._blob));
        out._entries = std::launder(reinterpret_cast<PackedEntry*>(out._blob + sizeof(Header)));
        assert(reinterpret_cast<std::uintptr_t>(out._blob) % kBlobAlignment == 0U && "from_blob: out._blob must be kBlobAlignment-aligned");
        return out;
    }

    void freeze() noexcept {
        assert(!is_view() && "ValueMap: cannot freeze a view-mode (resource == nullptr) ValueMap");
        if (is_view()) [[unlikely]] { // never write a viewed (possibly read-only) blob
            return;
        }
        if (_header) {
            _header->flags |= kHeaderFlagFrozen;
        }
    }
    [[nodiscard]] constexpr bool is_frozen() const noexcept { return _header && (_header->flags & kHeaderFlagFrozen); }
    // set when a fixed-buffer-mutable map (makeAt) rejected a mutation for lack of capacity
    [[nodiscard]] constexpr bool overflowed() const noexcept { return _header && (_header->flags & kHeaderFlagOverflow); }

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
    std::pmr::memory_resource* _resource = std::pmr::get_default_resource();

    // tag-dispatched view-mode constructor. Public-facing factory: makeView().
    struct ViewModeTag {};
    ValueMap(ViewModeTag, std::span<const std::byte> bytes) noexcept      //
        : ValueMapView{._blob = const_cast<std::byte*>(bytes.data()),     //
              ._capacity      = static_cast<std::uint32_t>(bytes.size()), //
              ._header        = nullptr,                                  //
              ._entries       = nullptr},                                       //
          _resource(nullptr) {
        if (bytes.size() >= sizeof(Header)) {
            _header  = std::launder(reinterpret_cast<Header*>(_blob));
            _entries = std::launder(reinterpret_cast<PackedEntry*>(_blob + sizeof(Header)));
        }
    }

    // Fixed-buffer-mutable mode: makeAt() sets _resource to this sentinel so the map is NOT a
    // view (is_view() stays false → mutators run unchanged) yet the external blob is never
    // reallocated/freed. null_memory_resource's deallocate is a no-op; its allocate throws —
    // but every grow-decision site bails (kHeaderFlagOverflow) before any allocate, so it
    // never runs. Owning maps (_resource = real allocator) and views (nullptr) are unaffected.
    [[nodiscard]] static std::pmr::memory_resource* _fixedBufferResource() noexcept { return std::pmr::null_memory_resource(); }
    [[nodiscard]] bool                              _isFixedBuffer() const noexcept { return _resource == _fixedBufferResource(); }
    [[nodiscard]] std::pmr::memory_resource*        _scratchResource() const noexcept { return (_resource != nullptr && _resource != _fixedBufferResource()) ? _resource : std::pmr::get_default_resource(); }
    void                                            _markOverflow() noexcept {
        if (_header) {
            _header->flags |= kHeaderFlagOverflow;
        }
    }
    static constexpr std::uint32_t kInvalidOffset = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] const_iterator erase_one(const_iterator it) {
        if (it._map != this || it._index >= size()) {
            return end();
        }
        const auto removedIndex = it._index;
        erase(it);
        return const_iterator{this, removedIndex};
    }

    // Over-allocate by kBlobAlignment bytes so we can manually align the user-visible blob
    // pointer regardless of whether the upstream resource honours the alignment argument
    // (sanitizer redzone instrumentation can shift the returned pointer by 8). The offset
    // between the raw allocation and the aligned blob (in [1, kBlobAlignment]) is stored in
    // the single byte just before the aligned pointer, so deallocate can recover it.
    [[nodiscard]] std::byte* _alignedAllocate(std::uint32_t usefulSize) {
        assert(_resource != nullptr && _resource != _fixedBufferResource() && "_alignedAllocate requires an owning resource");
        const std::size_t    rawSize   = static_cast<std::size_t>(usefulSize) + kBlobAlignment;
        auto*                raw       = static_cast<std::byte*>(_resource->allocate(rawSize, alignof(std::max_align_t)));
        const std::uintptr_t rawAddr   = reinterpret_cast<std::uintptr_t>(raw);
        const std::uintptr_t alignedTo = (rawAddr + kBlobAlignment) & ~(kBlobAlignment - 1U);
        auto*                aligned   = reinterpret_cast<std::byte*>(alignedTo);
        const std::ptrdiff_t offset    = aligned - raw;
        assert(offset >= 1 && offset <= static_cast<std::ptrdiff_t>(kBlobAlignment));
        *(aligned - 1) = static_cast<std::byte>(offset);
        return aligned;
    }

    void _alignedDeallocate(std::byte* aligned, std::uint32_t usefulSize) noexcept {
        if (!aligned || !_resource || _resource == _fixedBufferResource()) {
            return;
        }
        const std::ptrdiff_t offset  = std::to_integer<std::ptrdiff_t>(*(aligned - 1));
        auto*                raw     = aligned - offset;
        const std::size_t    rawSize = static_cast<std::size_t>(usefulSize) + kBlobAlignment;
        _resource->deallocate(raw, rawSize, alignof(std::max_align_t));
    }

    void _allocateBlob(std::uint32_t entries, std::uint32_t payloadCapacity) {
        const auto entryBytes   = static_cast<std::uint32_t>(entries) * static_cast<std::uint32_t>(sizeof(PackedEntry));
        const auto payloadStart = static_cast<std::uint32_t>(sizeof(Header)) + entryBytes;
        const auto total        = payloadStart + payloadCapacity;
        _blob                   = _alignedAllocate(total);
        _capacity               = total;
        assert(reinterpret_cast<std::uintptr_t>(_blob) % kBlobAlignment == 0U && "_blob must be kBlobAlignment-aligned");
        _header  = std::launder(reinterpret_cast<Header*>(_blob));
        _entries = std::launder(reinterpret_cast<PackedEntry*>(_blob + sizeof(Header)));
        std::memcpy(_header->magic, kBlobMagic.data(), kBlobMagic.size());
        _header->version         = kBlobVersion;
        _header->flags           = gr::meta::kDebugBuild ? kHeaderFlagDebugGuards : std::uint8_t{0};
        _header->entryCount      = 0U;
        _header->entryCapacity   = static_cast<std::uint16_t>(entries);
        _header->_reserved       = 0U;
        _header->totalSize       = total;
        _header->payloadOffset   = payloadStart;
        _header->payloadUsed     = 0U;
        _header->payloadCapacity = payloadCapacity;
        _header->payloadFreeHead = 0U;
        // Debug builds: zero-init the entry array + slack-fill the payload pool so hex dumps show
        // "obviously not data" patterns rather than uninitialised heap garbage. The marker write
        // is also debug-only — it lives outside `payloadUsed`, readers ignore it either way.
        if constexpr (gr::meta::kDebugBuild) {
            std::memset(_entries, 0, entryBytes);
            if (payloadCapacity > 0U) {
                std::memset(_blob + payloadStart, std::to_integer<int>(kSlackFillPattern), payloadCapacity);
            }
            _writePayloadEndMarker();
        }
        _writeEntrySentinel(); // always-on sentinel emission (the slot is reserved unconditionally).
    }

    void _deallocateBlob() noexcept {
        _alignedDeallocate(_blob, _capacity);
        _blob     = nullptr;
        _header   = nullptr;
        _entries  = nullptr;
        _capacity = 0U;
    }

    void _copyBlobFrom(const ValueMapView& other) {
        if (!other._header || other._capacity == 0U) {
            _allocateBlob(8U, 0U);
            return;
        }
        const auto total = other._capacity;
        _blob            = _alignedAllocate(total);
        _capacity        = total;
        std::memcpy(_blob, other._blob, total);
        _header  = std::launder(reinterpret_cast<Header*>(_blob));
        _entries = std::launder(reinterpret_cast<PackedEntry*>(_blob + sizeof(Header)));
        assert(reinterpret_cast<std::uintptr_t>(_blob) % kBlobAlignment == 0U && "_blob must be kBlobAlignment-aligned");
    }

    [[nodiscard]] std::uint32_t _entryCapacity() const noexcept { return _header ? _header->entryCapacity : 0U; }

    [[nodiscard]] std::uint32_t _payloadCapacity() const noexcept { return _header ? _header->payloadCapacity : 0U; }

    void _grow(std::uint32_t requestedEntries, std::uint32_t newPayloadBytes, bool shrink = false) {
        if (_isFixedBuffer()) { // fixed external buffer: never reallocate (reserve/shrink_to_fit become no-ops)
            _markOverflow();
            return;
        }
        const std::uint16_t curEntries = _header ? _header->entryCount : std::uint16_t{0};
        const std::uint32_t curPayload = _header ? _header->payloadUsed : 0U;

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
        const auto total        = payloadStart + targetPayload;

        auto* newBlob = _alignedAllocate(total);
        assert(reinterpret_cast<std::uintptr_t>(newBlob) % kBlobAlignment == 0U && "_grow: newBlob must be kBlobAlignment-aligned");
        auto* newHeader     = std::launder(reinterpret_cast<Header*>(newBlob));
        auto* newEntryArray = std::launder(reinterpret_cast<PackedEntry*>(newBlob + sizeof(Header)));
        auto* newPayload    = newBlob + payloadStart;

        std::memcpy(newHeader->magic, kBlobMagic.data(), kBlobMagic.size());
        newHeader->version         = kBlobVersion;
        newHeader->flags           = _header ? _header->flags : (gr::meta::kDebugBuild ? kHeaderFlagDebugGuards : std::uint8_t{0});
        newHeader->entryCount      = curEntries;
        newHeader->entryCapacity   = static_cast<std::uint16_t>(targetEntryCap);
        newHeader->_reserved       = 0U;
        newHeader->totalSize       = total;
        newHeader->payloadOffset   = payloadStart;
        newHeader->payloadUsed     = curPayload;
        newHeader->payloadCapacity = targetPayload;
        newHeader->payloadFreeHead = 0U; // rebased below alongside payloads

        const auto liveEntryBytes = curEntries * sizeof(PackedEntry);
        const auto tailEntryBytes = entryBytes - liveEntryBytes;
        if (tailEntryBytes > 0U) {
            std::memset(reinterpret_cast<std::byte*>(newEntryArray) + liveEntryBytes, 0, tailEntryBytes);
        }

        if constexpr (gr::meta::kDebugBuild) {
            const auto unusedPayload = total - payloadStart - curPayload;
            if (unusedPayload > 0U) {
                std::memset(newPayload + curPayload, std::to_integer<int>(kSlackFillPattern), unusedPayload);
            }
        }

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

        if (_header && curPayload > 0U) {
            std::memcpy(newPayload, _blob + _header->payloadOffset, curPayload);
        }

        if (_header && _header->payloadFreeHead != 0U) {
            const auto oldPayloadOffset = _header->payloadOffset;
            const auto rebase           = [oldPayloadOffset, payloadStart](std::uint32_t off) -> std::uint32_t { return off == 0U ? 0U : payloadStart + (off - oldPayloadOffset); };
            newHeader->payloadFreeHead  = rebase(_header->payloadFreeHead);
            std::uint32_t cur           = newHeader->payloadFreeHead;
            while (cur != 0U) {
                FreeChunk fc;
                std::memcpy(&fc, newBlob + cur, sizeof(fc));
                fc.nextOffset = rebase(fc.nextOffset);
                std::memcpy(newBlob + cur, &fc, sizeof(fc));
                cur = fc.nextOffset;
            }
        }

        _deallocateBlob();
        _blob     = newBlob;
        _capacity = total;
        _header   = newHeader;
        _entries  = newEntryArray;
        _writeEntrySentinel(); // re-emit at the new tail (capacity changed)
        if constexpr (gr::meta::kDebugBuild) {
            _writePayloadEndMarker();
        }
    }

    void _freeListPush(std::uint32_t offset, std::uint32_t length) noexcept {
        if (!_header || length < kFreeChunkHeaderSize) {
            return;
        }
        FreeChunk fc{.length = length, .nextOffset = _header->payloadFreeHead};
        std::memcpy(_blob + offset, &fc, sizeof(fc));
        _header->payloadFreeHead = offset;
    }

    void _writeEntrySentinel() noexcept {
        if (!_header || _header->entryCount >= _entryCapacity()) {
            return; // capacity exhausted (no room for the sentinel slot — _grow next insert handles it)
        }
        PackedEntry& sentinel = _entries[_header->entryCount];
        if constexpr (gr::meta::kDebugBuild) {
            std::memset(&sentinel, 0, sizeof(PackedEntry));
        }
        sentinel.keyId = keys::kEndMarkerId;
    }

    void _writePayloadEndMarker() noexcept {
        if (!_header) {
            return;
        }
        const auto end   = _header->payloadOffset + _header->payloadUsed;
        const auto avail = _capacity > end ? _capacity - end : 0UZ;
        const auto n     = std::min<std::size_t>(avail, kPayloadEndMarker.size());
        if (n > 0UZ) {
            std::memcpy(_blob + end, kPayloadEndMarker.data(), n);
        }
    }
    // Wire-format: write a value-record (8 B header + content bytes; padded to ≥ 16 B) into
    // the payload pool and return its blob offset. Record format is Value's _data layout —
    // [size:u32][vt:u8][ct:u8][flags:u16][payload bytes]. Iter aliases this directly via Value's
    // _data pointer, achieving the alloc-free view-mode iter perf path.
    //
    // Allocates blob bytes in-place via _appendPayload (snapshot path for self-aliased content),
    // then writes the header bytes after the copy lands. No temp pmr::vector — keeps insert hot
    // path alloc-free aside from the underlying pool growth.
    [[nodiscard]] std::uint32_t _appendValueRecord(Value::ValueType vt, Value::ContainerType ct, std::span<const std::byte> contentBytes, std::uint8_t recFlags = 0U) {
        const std::uint32_t contentLen = static_cast<std::uint32_t>(contentBytes.size());
        const std::uint32_t recSize    = alignToRecord(kRecHeaderBytes + contentLen);
        const std::uint32_t padBytes   = recSize - kRecHeaderBytes - contentLen;

        // 1) Snapshot content if it aliases our blob (self-emplace protection); otherwise pass-through.
        const bool                  aliasesSelf = (_blob != nullptr) && !std::less<>{}(contentBytes.data(), _blob) && std::less<>{}(contentBytes.data(), _blob + _capacity);
        std::pmr::vector<std::byte> snap{_scratchResource()};
        if (aliasesSelf) {
            snap.assign(contentBytes.begin(), contentBytes.end());
            contentBytes = std::span<const std::byte>{snap};
        }

        // 2) Reserve recSize bytes in the pool. Try free-list first (matched-fit); else bump payloadUsed (and grow if needed).
        std::uint32_t offset;
        if (auto reused = _freeListPopFit(recSize); reused.has_value()) {
            offset = *reused;
        } else {
            if (!_header) {
                _allocateBlob(8U, recSize);
            }
            // Align payloadUsed up to kRecAlignment so the new record starts at a 16-aligned
            // offset within the payload pool (preserves Header/PackedEntry alignment contracts
            // when the previous payload write was a non-aligned spilled key).
            const std::uint32_t curUsed      = _header->payloadUsed;
            const std::uint32_t alignedUsed  = alignToRecord(curUsed) == curUsed ? curUsed : ((curUsed + kRecAlignment - 1U) & ~(kRecAlignment - 1U));
            const std::uint32_t padBeforeRec = alignedUsed - curUsed;
            if (recSize > std::numeric_limits<std::uint32_t>::max() - alignedUsed) {
                _markOverflow(); // u32 wraparound — same failure channel as fixed-buffer overflow
                return kInvalidOffset;
            }
            if (alignedUsed + recSize > _payloadCapacity()) {
                if (_isFixedBuffer()) {
                    _markOverflow();
                    return kInvalidOffset;
                }
                _grow(_header->entryCount, alignedUsed + recSize);
            }
            if (padBeforeRec > 0U) {
                std::memset(_blob + _header->payloadOffset + curUsed, 0, padBeforeRec);
            }
            offset               = _header->payloadOffset + alignedUsed;
            _header->payloadUsed = alignedUsed + recSize;
        }

        // 3) Write header bytes directly into the pool at offset.
        std::byte* dst = _blob + offset;
        gr::wire::writeHeaderSized(dst, recSize, static_cast<std::uint8_t>(vt), static_cast<std::uint8_t>(ct), recFlags);

        // 4) Write content bytes (and zero the padding tail).
        if (contentLen > 0U) {
            std::memcpy(dst + 8, contentBytes.data(), contentLen);
        }
        if (padBytes > 0U) {
            std::memset(dst + 8 + contentLen, 0, padBytes);
        }

        if constexpr (gr::meta::kDebugBuild) {
            _writePayloadEndMarker();
        }
        return offset;
    }

    [[nodiscard]] std::optional<std::uint32_t> _freeListPopFit(std::uint32_t bytes) noexcept {
        if (!_header || _header->payloadFreeHead == 0U || bytes == 0U) {
            return std::nullopt;
        }

        const std::size_t walkCap = static_cast<std::size_t>(_capacity) / kFreeChunkHeaderSize + 1UZ;
        std::size_t       walked  = 0UZ;
        std::uint32_t     prev    = 0U;
        std::uint32_t     cur     = _header->payloadFreeHead;
        while (cur != 0U) {
            if (++walked > walkCap) {
                assert(false && "ValueMap free-list walk exceeded cap — likely in-process corruption");
                return std::nullopt;
            }
            FreeChunk fc;
            std::memcpy(&fc, _blob + cur, sizeof(fc));
            if (fc.length >= bytes) {
                // unlink: head or middle
                if (prev == 0U) {
                    _header->payloadFreeHead = fc.nextOffset;
                } else {
                    FreeChunk pfc;
                    std::memcpy(&pfc, _blob + prev, sizeof(pfc));
                    pfc.nextOffset = fc.nextOffset;
                    std::memcpy(_blob + prev, &pfc, sizeof(pfc));
                }
                // Split if the trailing tail is large enough to host its own header. The wire
                // format requires the entry's payloadLength to equal the value-record's `size`
                // field (see _validateBlob), so we can't extend the entry to swallow a small
                // remainder; 1..7-byte tails are accepted as a known minor fragmentation source
                // until a future PR adds a back-fill / coalesce pass.
                const std::uint32_t remainder = fc.length - bytes;
                if (remainder >= kFreeChunkHeaderSize) {
                    _freeListPush(cur + bytes, remainder);
                }
                return cur;
            }
            prev = cur;
            cur  = fc.nextOffset;
        }
        return std::nullopt;
    }

    [[nodiscard]] std::uint32_t _appendPayload(std::span<const std::byte> src, std::uint32_t extraGuardBytes = 0U) {
        if (!_header) {
            _allocateBlob(8U, static_cast<std::uint32_t>(std::min<std::size_t>(src.size() + extraGuardBytes, std::numeric_limits<std::uint32_t>::max())));
        }
        const auto requested = static_cast<std::uint32_t>(src.size()) + extraGuardBytes;
        // Reuse a freed payload region first (eager in-place reclamation; no _grow / no
        // payloadUsed bump). Any caller that erased something larger than `requested` recovers
        // those bytes here without a shrink_to_fit pass.
        if (auto reused = _freeListPopFit(requested); reused.has_value()) {
            std::memcpy(_blob + *reused, src.data(), src.size());
            if (extraGuardBytes > 0U) {
                std::memset(_blob + *reused + src.size(), 0, extraGuardBytes);
            }
            return *reused;
        }
        const std::size_t curUsed = _header->payloadUsed;
        if (requested > std::numeric_limits<std::uint32_t>::max() - curUsed) {
            _markOverflow(); // 4 GiB u32 limit — same failure channel as fixed-buffer overflow
            return kInvalidOffset;
        }
        if (curUsed + requested > _payloadCapacity()) {
            if (_isFixedBuffer()) {
                _markOverflow();
                return kInvalidOffset;
            }
            _grow(_header->entryCount, static_cast<std::uint32_t>(curUsed) + requested);
        }
        const auto offset = _header->payloadOffset + _header->payloadUsed;
        std::memcpy(_blob + offset, src.data(), src.size());
        if (extraGuardBytes > 0U) {
            std::memset(_blob + offset + src.size(), 0, extraGuardBytes);
        }
        _header->payloadUsed += requested;
        if constexpr (gr::meta::kDebugBuild) {
            _writePayloadEndMarker();
        }
        return offset;
    }

    [[nodiscard]] std::uint32_t _appendPayloadSafe(std::span<const std::byte> src, std::uint32_t extraGuardBytes = 0U) {
        const bool aliasesSelf = (_blob != nullptr) //
                                 && !std::less<>{}(src.data(), _blob) && std::less<>{}(src.data(), _blob + _capacity);
        if (!aliasesSelf) {
            return _appendPayload(src, extraGuardBytes);
        }
        std::pmr::vector<std::byte> snapshot(src.begin(), src.end(), _scratchResource());
        return _appendPayload(std::span<const std::byte>{snapshot}, extraGuardBytes);
    }

    [[nodiscard]] bool _ensureEntrySlot() {
        if (!_header) {
            _allocateBlob(8U, 0U);
        }

        if (_header->entryCount + 1U >= _entryCapacity()) {
            if (_isFixedBuffer()) {
                _markOverflow();
                return false;
            }
            _grow(_header->entryCount + 2U, _header->payloadUsed);
        }
        return true;
    }

    [[nodiscard]] Value _entryToValue(const PackedEntry& e) const { return detail::decodeEntry(e, _blob, _scratchResource()); }

    template<typename Cpp>
    [[nodiscard]] std::optional<Cpp> _readTyped(const PackedEntry& e) const {
        if constexpr (detail::InlineScalar<Cpp>) {
            if (static_cast<Value::ValueType>(e.valueType) != detail::cppToValueType<Cpp>()) {
                return std::nullopt;
            }
            // C3: inline scalars live in a 16-B value-record at e.payloadOffset (8-B header + 8-B inline payload).
            if ((e.flags & kEntryFlagOffsetLength) == 0U || e.payloadOffset == 0U) {
                return std::nullopt;
            }
            const std::byte* recPayload = _blob + e.payloadOffset + kRecHeaderBytes;
            if constexpr (std::same_as<Cpp, bool>) {
                return std::byte{0} != recPayload[0];
            } else if constexpr (sizeof(Cpp) == sizeof(std::uint64_t)) {
                std::uint64_t bits;
                std::memcpy(&bits, recPayload, sizeof(bits));
                return std::bit_cast<Cpp>(bits);
            } else {
                std::array<std::byte, sizeof(Cpp)> buf;
                std::memcpy(buf.data(), recPayload, sizeof(Cpp));
                return std::bit_cast<Cpp>(buf);
            }
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
        // Record at e.payloadOffset is [size:4][vt:1][ct:1][flags:1][payloadOffset:1][chars+'\0' guard].
        // String chars start at byte 8 of the record. Length = (recSize - 8 - 1) excluding NUL guard.
        if ((e.flags & kEntryFlagOffsetLength) == 0U || e.payloadOffset == 0U) {
            return {};
        }
        const std::byte*    recStart = _blob + e.payloadOffset;
        const std::uint32_t recSize  = gr::wire::elementSize(recStart);
        if (recSize < 16U) { // header (8) + at least 8 padding for short strings
            return {};
        }
        const auto* chars = reinterpret_cast<const char*>(recStart + 8);
        // Use strlen to find the actual NUL guard (records are padded to ≥ 16 B for short strings).
        const auto  cap = static_cast<std::size_t>(recSize - kRecHeaderBytes);
        const auto* nul = static_cast<const char*>(std::memchr(chars, '\0', cap));
        const auto  len = nul != nullptr ? static_cast<std::size_t>(nul - chars) : cap;
        return {chars, len};
    }

    template<typename V>
    [[nodiscard]] const_iterator _insertNew(std::string_view key, V&& value) {
        if (!_ensureEntrySlot()) {
            return end();
        }
        const auto index = _header->entryCount;
        std::memset(&_entries[index], 0, sizeof(PackedEntry));

        // Write key: canonical id (cheap, no payload), inline (<=34 chars), or spilled (payload-allocated).
        if (const auto canonicalId = keys::lookupId(key); canonicalId != keys::kIdUnknown) {
            _entries[index].keyId = canonicalId;
        } else if (key.size() <= kMaxInlineKeyLength) {
            detail::setInlineKey(_entries[index], key);
        } else {
            const auto bytes  = std::span<const std::byte>{reinterpret_cast<const std::byte*>(key.data()), key.size()};
            const auto offset = _appendPayloadSafe(bytes, /*extraGuardBytes=*/1U); // \0 guard for hex-dump readability + stale-pointer safety
            if (offset == kInvalidOffset) {
                return end();
            }
            detail::setSpilledKey(_entries[index], offset, static_cast<std::uint32_t>(key.size()));
        }

        ++_header->entryCount;
        bool       committed = false;
        const auto rollback  = gr::on_scope_exit{[&]() noexcept {
            if (!committed) {
                --_header->entryCount;
            }
        }};
        if (!_assignValueAt(index, std::forward<V>(value))) {
            return const_iterator{this, index}; // scope-exit rolls entryCount back to `index`, so this == end() to the caller
        }
        committed = true;
        _writeEntrySentinel(); // re-emit sentinel at the new tail (index = entryCount)
        return const_iterator{this, index};
    }

    template<typename V>
    [[nodiscard]] bool _assignValueAt(std::uint16_t index, V&& value) {
        using U = std::remove_cvref_t<V>;
        if constexpr (std::same_as<U, Value> || std::same_as<U, ValueView>) {
            if (!value.is_monostate()) {
                const std::span<const std::byte> srcRec = value.recordSpan();
                if (srcRec.size() >= kRecHeaderBytes) {
                    const auto         vt       = static_cast<Value::ValueType>(gr::wire::valueType(srcRec.data()));
                    const auto         ct       = static_cast<Value::ContainerType>(gr::wire::containerType(srcRec.data()));
                    const std::uint8_t recFlags = gr::wire::flags(srcRec.data());
                    const auto         content  = srcRec.subspan(kRecHeaderBytes);
                    // Capture old payload region for post-write reclamation (mirrors the
                    // typed-branch logic below — kept inline here since the typed branches don't
                    // run for the Value source).
                    const std::uint16_t oldEntryFlags         = _entries[index].flags;
                    const std::uint32_t oldPayloadOffsetV     = _entries[index].payloadOffset;
                    const std::uint32_t oldPayloadLengthV     = _entries[index].payloadLength;
                    const std::byte*    capturedBlobV         = _blob;
                    const std::uint32_t capturedPayloadStartV = _header ? _header->payloadOffset : 0U;
                    const auto          offset                = _appendValueRecord(vt, ct, content, recFlags);
                    if (offset == kInvalidOffset) {
                        return false;
                    }
                    PackedEntry& e = _entries[index];
                    e.valueType    = static_cast<std::uint8_t>(vt);
                    e.flags        = kEntryFlagOffsetLength;
                    if (vt == Value::ValueType::Value && ct == Value::ContainerType::Map) {
                        e.flags |= kEntryFlagNestedMap;
                    }
                    if (ct == Value::ContainerType::Tensor) {
                        e.flags |= kEntryFlagTensor;
                    }
                    e.payloadOffset = offset;
                    e.payloadLength = static_cast<std::uint32_t>(srcRec.size());
                    if ((oldEntryFlags & kEntryFlagOffsetLength) != 0U && oldPayloadLengthV > 0U) {
                        std::uint32_t freeOffset = oldPayloadOffsetV;
                        if (_blob != capturedBlobV && _header) {
                            freeOffset = _header->payloadOffset + (oldPayloadOffsetV - capturedPayloadStartV);
                        }
                        const std::uint32_t recSizeOld = alignToRecord(oldPayloadLengthV);
                        _freeListPush(freeOffset, recSizeOld);
                    }
                }
            }
            return true;
        }
        const std::uint16_t oldFlags             = _entries[index].flags;
        const std::uint32_t oldPayloadOffset     = _entries[index].payloadOffset;
        const std::uint32_t oldPayloadLength     = _entries[index].payloadLength;
        const std::byte*    capturedBlob         = _blob;
        const std::uint32_t capturedPayloadStart = _header ? _header->payloadOffset : 0U;
        if constexpr (detail::InlineScalar<U>) {
            std::array<std::byte, 8U> bytes{};
            if constexpr (std::same_as<U, bool>) {
                bytes[0] = std::byte{static_cast<std::uint8_t>(value ? 1U : 0U)};
            } else if constexpr (sizeof(U) == sizeof(std::uint64_t)) {
                const auto u = std::bit_cast<std::uint64_t>(value);
                std::memcpy(bytes.data(), &u, sizeof(U));
            } else {
                const auto buf = std::bit_cast<std::array<std::byte, sizeof(U)>>(value);
                std::memcpy(bytes.data(), buf.data(), sizeof(U));
            }
            constexpr Value::ContainerType ct     = std::same_as<U, std::complex<float>> ? Value::ContainerType::Complex : Value::ContainerType::Scalar;
            constexpr Value::ValueType     vt     = detail::cppToValueType<U>();
            const auto                     offset = _appendValueRecord(vt, ct, std::span<const std::byte>{bytes});
            if (offset == kInvalidOffset) {
                return false;
            }
            PackedEntry& e  = _entries[index];
            e.valueType     = static_cast<std::uint8_t>(vt);
            e.flags         = kEntryFlagOffsetLength;
            e.payloadOffset = offset;
            e.payloadLength = 16U;
        } else if constexpr (detail::PayloadScalar<U>) {
            // 16-byte scalar (currently only std::complex<double>) — record = 8 header + 16 bytes = 24 B.
            const auto bytes  = std::span<const std::byte>{reinterpret_cast<const std::byte*>(&value), sizeof(U)};
            const auto offset = _appendValueRecord(detail::cppToValueType<U>(), Value::ContainerType::Complex, bytes);
            if (offset == kInvalidOffset) {
                return false;
            }
            PackedEntry& e  = _entries[index];
            e.valueType     = static_cast<std::uint8_t>(detail::cppToValueType<U>());
            e.flags         = kEntryFlagOffsetLength;
            e.payloadOffset = offset;
            e.payloadLength = static_cast<std::uint32_t>(kRecHeaderBytes + sizeof(U));
        } else if constexpr (std::same_as<U, ValueMap>) {
            const auto srcBlob = value.blob();
            const auto offset  = _appendValueRecord(Value::ValueType::Value, Value::ContainerType::Map, srcBlob);
            if (offset == kInvalidOffset) {
                return false;
            }
            PackedEntry& e  = _entries[index];
            e.valueType     = static_cast<std::uint8_t>(Value::ValueType::Value);
            e.flags         = kEntryFlagOffsetLength | kEntryFlagNestedMap;
            e.payloadOffset = offset;
            e.payloadLength = static_cast<std::uint32_t>(kRecHeaderBytes + srcBlob.size());
        } else if constexpr (detail::StringLike<V>) {
            // String — record = 8 header + chars + '\0' guard.
            const std::string_view      sv{std::forward<V>(value)};
            const auto                  srcSize = sv.size();
            std::pmr::vector<std::byte> content(srcSize + 1U, std::byte{0}, _scratchResource());
            if (srcSize > 0U) {
                std::memcpy(content.data(), sv.data(), srcSize);
            }
            // content[srcSize] = '\0' from zero-init
            const auto offset = _appendValueRecord(Value::ValueType::String, Value::ContainerType::String, std::span<const std::byte>{content});
            if (offset == kInvalidOffset) {
                return false;
            }
            PackedEntry& e  = _entries[index];
            e.valueType     = static_cast<std::uint8_t>(Value::ValueType::String);
            e.flags         = kEntryFlagOffsetLength;
            e.payloadOffset = offset;
            e.payloadLength = static_cast<std::uint32_t>(kRecHeaderBytes + srcSize + 1U);
        } else if constexpr (gr::TensorLike<U>) {
            auto       tensorBlob = detail::encodeTensorBlob(value, _scratchResource(), /*depth=*/0U);
            const auto elemVT     = !tensorBlob.empty() ? static_cast<Value::ValueType>(static_cast<std::uint8_t>(tensorBlob[0])) : Value::ValueType::Monostate;
            const auto offset     = _appendValueRecord(elemVT, Value::ContainerType::Tensor, std::span<const std::byte>{tensorBlob}, kEntryFlagTensor);
            if (offset == kInvalidOffset) {
                return false;
            }
            PackedEntry& e  = _entries[index];
            e.valueType     = static_cast<std::uint8_t>(elemVT);
            e.flags         = kEntryFlagOffsetLength | kEntryFlagTensor;
            e.payloadOffset = offset;
            e.payloadLength = static_cast<std::uint32_t>(kRecHeaderBytes + tensorBlob.size());
        } else {
            static_assert(detail::InlineScalar<U> || detail::PayloadScalar<U> || detail::StringLike<V> || std::same_as<U, ValueMap> || gr::TensorLike<U> || std::same_as<U, Value> || std::same_as<U, ValueView>, "ValueMap: insert value must be an inline scalar, std::complex<double>, string-like, a nested ValueMap, a gr::Tensor<T>, gr::pmt::Value, or gr::pmt::ValueView.");
        }

        if ((oldFlags & kEntryFlagOffsetLength) != 0U && oldPayloadLength > 0U) {
            std::uint32_t freeOffset = oldPayloadOffset;
            if (_blob != capturedBlob && _header) {
                freeOffset = _header->payloadOffset + (oldPayloadOffset - capturedPayloadStart);
            }
            const std::uint32_t recSize = alignToRecord(oldPayloadLength);
            _freeListPush(freeOffset, recSize);
        }
        return true;
    }
};

inline ValueMap ValueMapView::owned(std::pmr::memory_resource* resource) const { return ValueMap{*this, resource ? resource : std::pmr::get_default_resource()}; }

template<typename T>
requires std::same_as<T, ValueMap>
inline std::optional<ValueMap> ValueView::get_if() const noexcept {
    if (!is_map() || payloadByteCount() == 0U) {
        return std::nullopt;
    }
    return ValueMap::makeView(std::span<const std::byte>{payloadAs<std::byte>(), payloadByteCount()});
}

inline ValueMapView::const_iterator::value_type ValueMapView::const_iterator::operator*() const {
    const auto&            e   = _map->_entries[_index];
    const std::string_view key = _map->_readKey(e);
    if ((e.flags & kEntryFlagOffsetLength) != 0U && e.payloadOffset != 0U) {
        return {key, ValueView{._data = _map->_blob + e.payloadOffset}};
    }
    return {key, ValueView{}}; // Monostate / unset entry
}

namespace detail {

inline void encodeTensorElement(std::pmr::vector<std::byte>& out, const Value& val, std::pmr::memory_resource* resource, std::uint32_t depth) {
    PackedTensorElement elem;
    std::memset(&elem, 0, sizeof(elem));
    elem.valueType = static_cast<std::uint8_t>(val.value_type());

    std::pmr::vector<std::byte> payload(resource);

    if (val.is_map()) {
        if (auto srcMap = val.template get_if<ValueMap>()) {
            const auto srcBlob = srcMap->blob();
            payload.assign(srcBlob.begin(), srcBlob.end());
        }
        elem.flags = kEntryFlagOffsetLength | kEntryFlagNestedMap;
    } else if (val.is_tensor() && depth + 1U < kMaxDecodeDepth) {
        dispatchValueType(val.value_type(), [&]<typename T>(std::type_identity<T>) {
            if constexpr (TensorElementType<T>) {
                gr::Tensor<T> nested     = val.template value_or<gr::Tensor<T>>(gr::Tensor<T>{});
                auto          nestedBlob = encodeTensorBlob(nested, resource, depth + 1U);
                payload                  = std::move(nestedBlob);
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
                    writeInlineScalar<T>(elem, *p);
                }
            }
        });
        elem.flags = kEntryFlagInlineScalar;
    }

    elem.payloadLength = static_cast<std::uint32_t>(payload.size());

    const auto headerBegin = reinterpret_cast<const std::byte*>(&elem);
    out.insert(out.end(), headerBegin, headerBegin + sizeof(elem));
    out.insert(out.end(), payload.begin(), payload.end());
    const auto paddedLen = paddedElementPayloadBytes(elem.payloadLength);
    if (paddedLen > elem.payloadLength) {
        out.resize(out.size() + (paddedLen - elem.payloadLength), std::byte{0});
    }
}

} // namespace detail

namespace detail {
inline Value buildNestedMapValue(const std::byte* base, const PackedEntry* nestedEntries, std::uint16_t declaredCount, std::pmr::memory_resource* resource, std::uint32_t depth) {
    ValueMap map{resource};
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
        map.insert_or_assign(key, decodeEntry(ne, base, resource, depth + 1U));
    }
    return Value{map, resource};
}
} // namespace detail

template<detail::ExternalValueMap T>
void Value::init_from_map(T&& map) {
    using DecayedMap           = std::remove_cvref_t<T>;
    using MappedType           = typename DecayedMap::mapped_type;
    constexpr bool isValueType = std::same_as<MappedType, Value>;

    // build temp ValueMap, then copy its blob bytes into our value-record (8 B header + blob).
    ValueMap tmp(_resource, static_cast<std::uint32_t>(std::max<std::size_t>(map.size(), 8UZ)));
    for (const auto& [key, val] : map) {
        if constexpr (isValueType) {
            tmp.insert_or_assign(key, val);
        } else {
            tmp.insert_or_assign(key, Value{val, _resource});
        }
    }
    const auto          srcBlob = tmp.blob();
    const std::uint32_t recSize = static_cast<std::uint32_t>(kRecOffsetPayload + srcBlob.size());
    allocAndWriteHeader(std::max<std::uint32_t>(recSize, kRecMinSize), ValueType::Value, ContainerType::Map);
    if (!srcBlob.empty()) {
        std::memcpy(_data + kRecOffsetPayload, srcBlob.data(), srcBlob.size());
    } else {
        std::memset(_data + kRecOffsetPayload, 0, kRecMinSize - static_cast<std::uint32_t>(kRecOffsetPayload));
    }
}

template<typename T>
requires(gr::TensorViewLike<T> && std::same_as<std::remove_const_t<typename T::value_type>, gr::pmt::Value>)
inline std::optional<T> Value::get_if() const {
    if (!is_tensor() || value_type() != get_value_type<gr::pmt::Value>() || payloadByteCount() < kTensorBlobHeaderSize) {
        return std::nullopt;
    }
    const auto* base = payloadAs<std::byte>();
    const auto  rank = static_cast<std::uint8_t>(base[1]);
    if (rank > kMaxTensorRank) {
        return std::nullopt;
    }
    if constexpr (gr::meta::kDebugBuild) {
        if (payloadByteCount() < kTensorBlobHeaderSize + paddedTensorExtentsBytes(rank)) {
            return std::nullopt;
        }
    }
    T view{};
    view._bytes    = std::span<const std::byte>{base, payloadByteCount()};
    view._rank     = static_cast<std::size_t>(rank);
    view._resource = _resource ? _resource : std::pmr::get_default_resource();
    for (std::size_t i = 0UZ; i < rank; ++i) {
        std::uint32_t ext;
        std::memcpy(&ext, base + kTensorBlobHeaderSize + 4UZ * i, sizeof(ext));
        view._extents[i] = static_cast<std::size_t>(ext);
    }
    std::uint32_t elementCount;
    std::memcpy(&elementCount, base + 4, sizeof(elementCount));
    view._elementCount = static_cast<std::size_t>(elementCount);
    return view;
}

} // namespace gr::pmt

// Bodies for TensorView<gr::pmt::Value> live here because they need the full ValueMap type.
namespace gr {

template<std::size_t... Ex>
inline gr::pmt::Value TensorView<gr::pmt::Value, Ex...>::operator[](std::size_t i) const {
    using gr::pmt::kTensorBlobHeaderSize;
    using gr::pmt::PackedTensorElement;
    using gr::pmt::paddedTensorExtentsBytes;
    using gr::pmt::Value;
    using gr::pmt::detail::decodeTensorElement;
    const std::size_t extentsBytes = paddedTensorExtentsBytes(_rank);
    if (i >= _elementCount || _bytes.size() < kTensorBlobHeaderSize + extentsBytes) {
        return Value{_resource};
    }
    const std::size_t headerSize = kTensorBlobHeaderSize + extentsBytes;
    const std::byte*  elemBase   = _bytes.data() + headerSize;
    const std::size_t totalElem  = _bytes.size() - headerSize;
    std::uint32_t     cursor     = 0U;
    for (std::size_t k = 0UZ; k <= i; ++k) {
        if (cursor + sizeof(PackedTensorElement) > totalElem) {
            return Value{_resource};
        }
        PackedTensorElement hdr;
        std::memcpy(&hdr, elemBase + cursor, sizeof(hdr));
        if (hdr.payloadLength > totalElem - cursor - sizeof(hdr)) {
            return Value{_resource};
        }
        if (k == i) {
            return decodeTensorElement(hdr, elemBase + cursor + sizeof(hdr), _resource ? _resource : std::pmr::get_default_resource(), 0U);
        }
        cursor += static_cast<std::uint32_t>(sizeof(hdr)) + gr::pmt::paddedElementPayloadBytes(hdr.payloadLength);
    }
    return Value{_resource};
}

template<std::size_t... Ex>
inline gr::pmt::Value TensorView<gr::pmt::Value, Ex...>::const_iterator::operator*() const {
    using gr::pmt::kTensorBlobHeaderSize;
    using gr::pmt::PackedTensorElement;
    using gr::pmt::Value;
    using gr::pmt::detail::decodeTensorElement;
    if (!_view || _index >= _view->_elementCount) {
        return Value{_view ? _view->_resource : nullptr};
    }
    const std::size_t headerSize = kTensorBlobHeaderSize + gr::pmt::paddedTensorExtentsBytes(_view->_rank);
    if (_view->_bytes.size() < headerSize) {
        return Value{_view->_resource};
    }
    const std::byte*  elemBase  = _view->_bytes.data() + headerSize;
    const std::size_t totalElem = _view->_bytes.size() - headerSize;
    if (_byteCursor + sizeof(PackedTensorElement) > totalElem) {
        return Value{_view->_resource};
    }
    PackedTensorElement hdr;
    std::memcpy(&hdr, elemBase + _byteCursor, sizeof(hdr));
    if (hdr.payloadLength > totalElem - _byteCursor - sizeof(hdr)) {
        return Value{_view->_resource};
    }
    return decodeTensorElement(hdr, elemBase + _byteCursor + sizeof(hdr), _view->_resource ? _view->_resource : std::pmr::get_default_resource(), 0U);
}

template<std::size_t... Ex>
inline auto TensorView<gr::pmt::Value, Ex...>::const_iterator::operator++() -> const_iterator& {
    using gr::pmt::kTensorBlobHeaderSize;
    using gr::pmt::PackedTensorElement;
    if (!_view || _index >= _view->_elementCount) {
        return *this;
    }
    const std::size_t headerSize = kTensorBlobHeaderSize + gr::pmt::paddedTensorExtentsBytes(_view->_rank);
    if (_view->_bytes.size() < headerSize) {
        ++_index;
        return *this;
    }
    const std::byte*  elemBase  = _view->_bytes.data() + headerSize;
    const std::size_t totalElem = _view->_bytes.size() - headerSize;
    if (_byteCursor + sizeof(PackedTensorElement) > totalElem) {
        ++_index;
        return *this;
    }
    PackedTensorElement hdr;
    std::memcpy(&hdr, elemBase + _byteCursor, sizeof(hdr));
    _byteCursor += static_cast<std::uint32_t>(sizeof(hdr)) + gr::pmt::paddedElementPayloadBytes(hdr.payloadLength);
    ++_index;
    return *this;
}

template<std::size_t... Ex>
inline auto TensorView<gr::pmt::Value, Ex...>::owned(std::pmr::memory_resource* mr) const -> container_t {
    using gr::pmt::kTensorBlobHeaderSize;
    using gr::pmt::PackedTensorElement;
    using gr::pmt::Value;
    using gr::pmt::detail::decodeTensorElement;
    if (mr == nullptr) {
        mr = std::pmr::get_default_resource();
    }
    const std::span<const std::size_t> extentsView{_extents.data(), _rank};
    container_t                        result(gr::extents_from, extentsView, mr);
    // Override the ctor's product(extents) sizing — empty extents = 1 (scalar tensor convention),
    // but a rank-0/elementCount-0 view must materialise as empty, not as 1 monostate.
    result._data.resize(_elementCount);
    if (_elementCount == 0UZ) {
        return result;
    }
    gr::pmr::vector<Value, true> elems(_elementCount, Value{mr}, mr);
    const std::size_t            headerSize = kTensorBlobHeaderSize + gr::pmt::paddedTensorExtentsBytes(_rank);
    if (_bytes.size() < headerSize) {
        result._data = std::move(elems);
        return result;
    }
    const std::byte*  elemBase  = _bytes.data() + headerSize;
    const std::size_t totalElem = _bytes.size() - headerSize;
    std::uint32_t     cursor    = 0U;
    for (std::size_t k = 0UZ; k < _elementCount; ++k) {
        if (cursor + sizeof(PackedTensorElement) > totalElem) {
            break;
        }
        PackedTensorElement hdr;
        std::memcpy(&hdr, elemBase + cursor, sizeof(hdr));
        if (hdr.payloadLength > totalElem - cursor - sizeof(hdr)) {
            break;
        }
        elems[k] = decodeTensorElement(hdr, elemBase + cursor + sizeof(hdr), mr, 0U);
        cursor += static_cast<std::uint32_t>(sizeof(hdr)) + gr::pmt::paddedElementPayloadBytes(hdr.payloadLength);
    }
    result._data = std::move(elems);
    return result;
}

} // namespace gr

namespace gr::pmt {

template<TensorLike Tens>
void Value::init_from_tensor(Tens&& tensor) {
    using T = typename std::remove_cvref_t<Tens>::value_type;
    if (_resource == nullptr) {
        _resource = std::pmr::get_default_resource();
    }
    auto storeBlob = [this](const auto& blob, ValueType vt) {
        const std::uint32_t recSize = static_cast<std::uint32_t>(kRecOffsetPayload + blob.size());
        allocAndWriteHeader(std::max<std::uint32_t>(recSize, kRecMinSize), vt, ContainerType::Tensor);
        if (!blob.empty()) {
            std::memcpy(_data + kRecOffsetPayload, blob.data(), blob.size());
        } else {
            std::memset(_data + kRecOffsetPayload, 0, kRecMinSize - static_cast<std::uint32_t>(kRecOffsetPayload));
        }
    };
    if constexpr (std::same_as<T, std::pmr::string> || std::same_as<T, std::string>) {
        gr::Tensor<Value> wrapped(gr::extents_from, tensor.extents(), _resource);
        for (std::size_t i = 0UZ; i < tensor.size(); ++i) {
            wrapped._data[i] = Value{std::string_view{tensor._data[i]}, _resource};
        }
        const auto blob   = detail::encodeTensorBlob(wrapped, _resource, /*depth=*/0U);
        const auto wireVT = blob.empty() ? ValueType::Value : static_cast<ValueType>(static_cast<std::uint8_t>(blob[0]));
        storeBlob(blob, wireVT);
    } else {
        const auto blob   = detail::encodeTensorBlob(tensor, _resource, /*depth=*/0U);
        const auto wireVT = blob.empty() ? get_value_type<T>() : static_cast<ValueType>(static_cast<std::uint8_t>(blob[0]));
        storeBlob(blob, wireVT);
    }
}

template<typename Pred>
inline std::size_t erase_if(ValueMap& map, Pred pred) {
    std::size_t                                removed = 0UZ;
    std::pmr::vector<ValueMap::const_iterator> dropList(map.resource() ? map.resource() : std::pmr::get_default_resource());
    dropList.reserve(map.size());
    for (auto it = map.begin(); it != map.end(); ++it) {
        if (pred(*it)) {
            dropList.push_back(it);
        }
    }
    // erase back-to-front so earlier indices stay valid. O(n²) worst case (per-erase entry-array shift) — acceptable for the small maps here; single-pass compaction deferred.
    for (auto it = dropList.rbegin(); it != dropList.rend(); ++it) {
        map.erase(*it);
        ++removed;
    }
    return removed;
}

} // namespace gr::pmt

namespace std {
template<>
struct hash<gr::pmt::ValueMap> {
    [[nodiscard]] std::size_t operator()(const gr::pmt::ValueMap& m) const noexcept {
        std::size_t accumulator = 0UZ;
        for (auto [k, v] : m) {
            const std::size_t entryHash = std::hash<std::string_view>{}(k) ^ std::hash<gr::pmt::ValueView>{}(v);
            accumulator += entryHash;
        }
        return accumulator;
    }
};
} // namespace std

#endif // GNURADIO_VALUEMAP_HPP
