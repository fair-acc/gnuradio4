#ifndef GNURADIO_VALUE_HPP
#define GNURADIO_VALUE_HPP

#include <array>
#include <bit>
#include <cassert>
#include <complex>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/WireFormat.hpp>

namespace gr::pmt {

struct ValueView;
class Value;
class ValueMap; // forward declaration — full definition in ValueMap.hpp; Map operations live in Value.cpp

// Hoisted here so ValueView can use them without depending on the full Value definition.
// `Value::ValueType` / `Value::ContainerType` re-export these inside the class.
// clang-format off
enum class ValueType : std::uint8_t {
    Monostate = 0U,
    Bool = 1U,
    Int8 = 2U, Int16 = 3U, Int32 = 4U, Int64 = 5U,
    UInt8 = 6U, UInt16 = 7U, UInt32 = 8U, UInt64 = 9U,
    Float32 = 10U, Float64 = 11U,
    ComplexFloat32 = 12U, ComplexFloat64 = 13U,
    String = 14U,
    Value = 15U // for Tensor<Value> and Map<string,Value>
};
// clang-format on
enum class ContainerType : std::uint8_t { Scalar = 0U, Complex = 1U, String = 2U, Tensor = 3U, Map = 4U };

// Value-record prefix — see WireFormat.hpp / docs/CORE_WireFormat.md. Alignment
// is a Value/ValueMap policy (16 B), not a wire-prefix property.
inline constexpr std::size_t   kRecOffsetSize          = gr::wire::kOffSize;
inline constexpr std::size_t   kRecOffsetValueType     = gr::wire::kOffValueType;
inline constexpr std::size_t   kRecOffsetContainerType = gr::wire::kOffContainerType;
inline constexpr std::size_t   kRecOffsetFlags         = gr::wire::kOffFlags;
inline constexpr std::size_t   kRecOffsetPayloadOffset = gr::wire::kOffPayloadOffset;
inline constexpr std::size_t   kRecOffsetPayload       = gr::wire::kPrefixBytes;
inline constexpr std::uint32_t kRecHeaderBytes         = static_cast<std::uint32_t>(kRecOffsetPayload);
inline constexpr std::uint32_t kRecMinSize             = 16U; // covers complex<double> + alignas(8) Header/PackedEntry
inline constexpr std::uint32_t kRecAlignment           = 16U;
static_assert(std::has_single_bit(kRecAlignment), "kRecAlignment must be a power of two");
inline constexpr std::uint32_t kTensorExtentsAlignment = 8U; // tensor-internal padding within the payload (unchanged; independent of the record header)

[[nodiscard]] constexpr std::uint32_t alignToRecord(std::uint32_t payloadLength) noexcept {
    const std::uint32_t rounded = (payloadLength + kRecAlignment - 1U) & ~(kRecAlignment - 1U);
    return rounded < kRecMinSize ? kRecMinSize : rounded;
}

[[nodiscard]] constexpr std::size_t paddedTensorExtentsBytes(std::size_t rank) noexcept {
    const std::size_t raw = 4UZ * rank;
    return (raw + kTensorExtentsAlignment - 1UZ) & ~(kTensorExtentsAlignment - 1UZ);
}

[[nodiscard]] constexpr std::uint32_t paddedElementPayloadBytes(std::uint32_t payloadLength) noexcept { return (payloadLength + kTensorExtentsAlignment - 1U) & ~(kTensorExtentsAlignment - 1U); }

[[nodiscard]] const std::byte* monostateRecord() noexcept;

} // namespace gr::pmt

namespace gr::pmt {

// Tensor wire-format (also used by ValueMap's tensor sub-blob):
//
//   offset  size  field
//   ------  ----  ----------------------------------------------------------------------------
//      0      1   elementValueType  (Value::ValueType byte: Float32 / Int64 / String / Value …)
//      1      1   rank              (0 .. kMaxTensorRank)
//      2      1   encodingFlags     bit 0 = variableSizeElements (set iff elementVT ∈ {String, Value})
//      3      1   reserved = 0
//      4      4   elementCount      (product of extents; 0 if any extent is 0; 1 for rank-0)
//      8     4*r  extents[0..r-1]   (one u32 per extent)
//
// Then EITHER (variableSizeElements == 0 → fixed-size scalar / complex elements):
//      8+4r  elementCount × sizeof(elementCpp)  contiguous element data (alignment matches Tensor's)
//
// OR    (variableSizeElements == 1 → string / Value elements): per-element [PackedTensorElement]
//      headers + packed payload bytes — defined in detail::encodeTensorElement / decodeTensorElement
//      in ValueMap.hpp (kept there because nested-ValueMap encoding belongs alongside ValueMap).
inline constexpr std::uint8_t  kMaxTensorRank              = 8U;       // mirrors gr::detail::kMaxRank in Tensor.hpp
inline constexpr std::uint32_t kMaxTensorElements          = 1U << 24; // sanity cap (~16M elements per tensor)
inline constexpr std::size_t   kTensorBlobHeaderSize       = 8UZ;      // elementValueType[1] + rank[1] + encodingFlags[1] + reserved[1] + elementCount[4]
inline constexpr std::uint8_t  kTensorEncodingVariableSize = 0x01;

namespace detail {
[[noreturn, gnu::cold]] inline void value_type_mismatch() noexcept {
    assert(false && "gr::pmt::Value type mismatch");
    std::unreachable();
    std::abort();
}

// `Tensor` / `ValueMap` return by value because byte-blob storage has no stable `T*`.
template<typename T>
using return_t = std::conditional_t<std::is_rvalue_reference_v<T>, std::remove_cvref_t<T>, std::conditional_t<gr::TensorLike<std::remove_cvref_t<T>>, std::remove_cvref_t<T>, T>>;

template<typename T>
inline constexpr bool is_const_ref_v = std::is_lvalue_reference_v<T> && std::is_const_v<std::remove_reference_t<T>>;

template<typename T>
inline constexpr bool is_mutable_ref_v = std::is_lvalue_reference_v<T> && !std::is_const_v<std::remove_reference_t<T>>;

template<typename T>
inline constexpr bool is_std_string_v = std::same_as<std::remove_cvref_t<T>, std::string>;

template<typename T>
inline constexpr bool is_string_view_v = std::same_as<std::remove_cvref_t<T>, std::string_view>;

template<typename T>
inline constexpr bool is_string_convertible_v = is_std_string_v<T> || is_string_view_v<T>;

template<typename T>
concept ValueScalarType = std::same_as<std::remove_cvref_t<T>, bool>                                                                                                                                                                                 //
                          || std::same_as<std::remove_cvref_t<T>, std::int8_t> || std::same_as<std::remove_cvref_t<T>, std::int16_t> || std::same_as<std::remove_cvref_t<T>, std::int32_t> || std::same_as<std::remove_cvref_t<T>, std::int64_t>     //
                          || std::same_as<std::remove_cvref_t<T>, std::uint8_t> || std::same_as<std::remove_cvref_t<T>, std::uint16_t> || std::same_as<std::remove_cvref_t<T>, std::uint32_t> || std::same_as<std::remove_cvref_t<T>, std::uint64_t> //
                          || std::same_as<std::remove_cvref_t<T>, float> || std::same_as<std::remove_cvref_t<T>, double>                                                                                                                             //
                          || std::same_as<std::remove_cvref_t<T>, std::complex<float>> || std::same_as<std::remove_cvref_t<T>, std::complex<double>>;                                                                                                //

template<typename T>
concept ValueConvertible = std::same_as<std::remove_cvref_t<T>, Value> || ValueScalarType<T> || std::convertible_to<T, std::string_view>;

template<typename T>
concept ValueComparable = ValueScalarType<T>;

template<typename M>
concept ValueMapLike = requires(const std::remove_cvref_t<M>& m) {
    typename std::remove_cvref_t<M>::key_type;
    typename std::remove_cvref_t<M>::mapped_type;
    { m.begin() } -> std::input_iterator;
    { m.end() } -> std::sentinel_for<decltype(m.begin())>;
} && std::convertible_to<typename std::remove_cvref_t<M>::key_type, std::string_view> && ValueConvertible<typename std::remove_cvref_t<M>::mapped_type>;

template<typename M>
concept ExternalValueMap = ValueMapLike<M> && !std::same_as<std::remove_cvref_t<M>, ValueMap>;

constexpr std::string value_to_string(const ValueView&); // forward declaration
} // namespace detail

/**
 * @brief Value — a compact polymorphic type-erased container.
 *
 * Value is a 16-byte handle (`std::byte* _data` + `std::pmr::memory_resource* _resource`) into a
 * single PMR allocation that holds the value-record bytes (header + payload). The storage layout
 * IS the wire format — `recordSpan()` returns a `std::span<const std::byte>` that can be
 * memcpy'd to disk, sent over an IPC channel, or placed in SYCL/CUDA Unified Shared Memory and
 * read by a device kernel via the trivially-copyable `ValueView` base.
 *
 * The implementation is type-strict and suppresses implicit-conversion errors. PMR allocators
 * permit transitions between host and device in heterogeneous computing environments
 * (SYCL/CUDA/…). All complex types — including inline scalars — go through a single PMR
 * allocation per Value; no STL-side heap fragmentation, no exception path, no RTTI dependency.
 *
 * @par Supported types
 *   - Scalars: bool, int8/16/32/64, uint8/16/32/64, float, double
 *   - Complex: std::complex<float / double>
 *   - Strings: raw byte-blob ([size:4][chars][\\0])
 *   - Containers: Tensor<T>, ValueMap (nested polymorphic key-value map)
 *
 * @par Trivially-copyable view base
 *   `Value` publicly inherits from `ValueView` (8 B, single `std::byte*` pointer into the value
 *   record). The view carries the entire read API and the bounded in-place mutators; the owning
 *   `Value` adds RAII semantics + the PMR resource. SYCL/CUDA kernels slice to `ValueView` by
 *   value — no allocator state crosses the host-device boundary.
 *
 * @par Ownership Semantics for value_or<T>()
 * The template parameter encodes the **lifetime relationship**:
 * | Category      | Meaning                  | Value's Role After Call    |
 * |---------------|--------------------------|----------------------------|
 * | `T`           | "Give me a copy"         | Retains ownership          |
 * | `T&`          | "Let me modify yours"    | Retains ownership          |
 * | `const T&`    | "Let me borrow for read" | Retains ownership          |
 * | `T&&`         | "Give it to me"          | Relinquishes → monostate   |
 *
 * @par Basic Usage
 * @code
 *  Value v{std::string_view{"hello"}};
 *  std::string      s  = v.value_or<std::string>("");       // auto-convert, allocates
 *  std::string_view sv = v.value_or<std::string_view>(""); // zero-copy view
 *  bool has_str = v.holds<std::string>();                   // true (convertible)
 *
 *  v = std::int64_t{42};
 *  int64_t          x = v.value_or<std::int64_t>(0);       // copy, v unchanged
 *  int64_t&         r = v.value_or<std::int64_t&>(fb);     // modify in-place
 * @endcode
 *

* @par Tensor / TensorView usage rule
 * Prefer typed access at the call site: `value.get_if<TensorView<T>>()` for known T. Fixed-size
 * element types alias the wire bytes directly with no per-element decode. `Tensor<Value>` /
 * `TensorView<Value>` is the transit type when T isn't known at compile time, and the only path
 * for two element types the typed `Tensor<T>` rejects: T = `std::string` (use `Tensor<Value>`
 * with String cells) and T = `ValueMap` (use `Tensor<Value>` with Map cells). For genuinely
 * heterogeneous indexed data (cell 0 int, cell 1 float, cell 2 string …) use a Map keyed by
 * index — `Tensor<Value>` permits heterogeneous content as a fallback, not the recommended
 * pattern.
 *
 * @note `get_if<std::pmr::string>` is dropped — strings are stored as a raw byte-blob,
 *       not a `std::pmr::string` object. Use `get_if<std::string_view>()` for alloc-free
 *       view access (works for owning + view modes) or `value_or<std::string>(default)`
 *       for an owning copy.
 *
 * @warning Unchecked accessors have undefined behavior on type mismatch (assert in debug).
 *          Use holds<T>(), get_if<T>(), or value_or() for safe access.
 *
 * @par Wire Format
 *
 * A Value has no standalone wire format — it is a host-side handle. It acquires a wire form
 * only when **stored inside a container**: a `PackedEntry` row of a `gr::pmt::ValueMap`, or a
 * `PackedTensorElement` row of a `Tensor<Value>`. A `PackedEntry` carries the type tag bytes
 * (`valueType`, `flags`) plus a `payloadOffset` / `payloadLength` pair pointing at a value-record
 * in the container's shared payload pool — it has **no** `inlineValue` slot; every value (scalars
 * included) lives in the pool. A `PackedTensorElement` instead carries an 8-byte `inlineValue`
 * slot that holds inline scalars (<= 8 B) directly, spilling only larger payloads to the pool. The
 * byte sequence is host/device/IPC-portable (no host pointers) and round-trips verbatim through
 * `ValueMap::from_blob` / `as_blob`.
 *
 * Per-`ValueType` encoding (scalar byte layout — the `inlineValue` cells map to the
 * `PackedTensorElement` inline slot; a `PackedEntry` stores the identical bytes as the content of
 * its pooled value-record):
 *
 * @code
 *  ┌────────────────────────┬──────────────────────────────────────────────────────────┐
 *  │ ValueType              │ Wire encoding inside the row + (optional) payload pool   │
 *  ├────────────────────────┼──────────────────────────────────────────────────────────┤
 *  │ Monostate              │ tag only — no inlineValue, no payload                    │
 *  │ Bool                   │ inlineValue (low byte; 1 / 0)                            │
 *  │ Int8/16/32/64          │ inlineValue (sign-extended into 8 B)                     │
 *  │ UInt8/16/32/64         │ inlineValue (zero-extended into 8 B)                     │
 *  │ Float32                │ inlineValue (low 4 bytes)                                │
 *  │ Float64                │ inlineValue (full 8 bytes)                               │
 *  │ ComplexFloat32         │ inlineValue ([float re | float im], 8 B total)           │
 *  │ ComplexFloat64         │ payload pool — 16 raw bytes [double re | double im]      │
 *  │ String                 │ payload pool — UTF-8 chars + trailing '\0' guard         │
 *  │                        │   (the '\0' is +1 byte outside payloadLength)            │
 *  │ Value (ContainerType   │ payload pool — Tensor sub-blob:                          │
 *  │       = Tensor)        │   [ TensorHeader 8 B ][ extents 4*rank ][ element bytes ]│
 *  │ Value (ContainerType   │ payload pool — full nested ValueMap blob (recursive;     │
 *  │       = Map)           │   see ValueMap.hpp class doc for blob layout)            │
 *  └────────────────────────┴──────────────────────────────────────────────────────────┘
 *
 *  Tensor sub-blob (variable size; written by detail::encodeTensorBlob):
 *  ──────────────────────────────────────────────────────────────────────────────────────
 *  byte  0   : elementValueType  (Value::ValueType byte: Float32 / Int64 / Value / …)
 *  byte  1   : rank
 *  byte  2   : encodingFlags  (kTensorEncodingVariableSize set iff cells are variable-size:
 *              elementValueType ∈ {String, Value} with heterogeneous or non-fixed-size content)
 *  byte  3   : reserved        (debug build = 0xCA — kTensorSubBlobSignature)
 *  bytes 4-7 : elementCount    (u32)
 *  bytes 8 + 4*i : extent[i] (u32)                 ← for i in 0..rank
 *  then element bytes:
 *     • fixed-size T (intN, uintN, float, double, complex<float/double>): contiguous T values
 *     • Tensor<bool>: 1 byte per element (0 / 1)
 *     • variable-size cells (Tensor<Value> with heterogeneous content, or any String/Map/
 *       nested-Tensor cell): PackedTensorElement records (16 B each) + per-element pool tail.
 *       Tensor<Value> with homogeneous fixed-size cells is re-routed by the encoder through
 *       the typed-T path (elementValueType becomes that T, encoding becomes contiguous).
 * @endcode
 *
 * Decode (wire row → in-memory Value):
 *   - **inline scalars** copy bits from `inlineValue` into the constructed Value's `_storage`
 *     (no allocation; ComplexFloat32 deep-copies its 8 wire bytes to a heap `complex<float>`
 *     in the in-memory Value, even though the wire form is inline);
 *   - **pooled** ValueTypes either deep-copy the pool bytes into a fresh owning Value
 *     (`at()` / `operator[]`) or hand back a view-mode Value that aliases the pool
 *     (iterator deref, `get_if<TensorView<T>>`, `get_if<ValueMap>`).
 *
 * In-memory handle (host-side, transient — implementation detail): `sizeof(Value) = 24 B`,
 * `alignof = 8`. Holds the type tag, view-mode flag, view-mode payload byte count, an 8-byte
 * storage union (inline scalar bits **or** pointer to off-handle bytes for variable-size
 * containers), and a PMR resource pointer (used to allocate / grow / free off-handle bytes;
 * never serialised). The handle never appears on the wire — it is built by `at()` / iterator-
 * deref from the row + pool and discarded by `~Value`. See the field-level comments on the
 * class definition for the exact bit layout.
 *
 * @see gr::pmt::ValueMap — the contiguous packed blob that owns rows + payload pool and
 *      decodes into transient Value handles on access.
 */

/**
 * @brief Trivially-copyable POD handle over a value-record's bytes.
 *
 * Single member: `_data` aliasing the value-record's bytes (header + content). No resource,
 * no RAII — the bytes are owned by something else (a `Value`, a `ValueMap`'s payload pool, or
 * a USM/IPC buffer). Suitable for SYCL/CUDA kernel by-value capture.
 */
struct ValueView {
    std::byte* _data{const_cast<std::byte*>(monostateRecord())};

    [[nodiscard]] std::uint32_t              recSize() const noexcept { return gr::wire::elementSize(_data); }
    [[nodiscard]] constexpr std::uint8_t     recValueTypeByte() const noexcept { return gr::wire::valueType(_data); }
    [[nodiscard]] constexpr std::uint8_t     recContainerTypeByte() const noexcept { return gr::wire::containerType(_data); }
    [[nodiscard]] std::uint8_t               recFlags() const noexcept { return gr::wire::flags(_data); }
    [[nodiscard]] std::uint8_t               recPayloadOffset() const noexcept { return gr::wire::payloadOffset(_data); }
    [[nodiscard]] constexpr const std::byte* recPayload() const noexcept { return gr::wire::payload(_data); }
    [[nodiscard]] constexpr std::byte*       recPayloadMutable() noexcept {
        if (!std::is_constant_evaluated()) {
            assert(_data != monostateRecord() && "recPayloadMutable() on read-only monostate sentinel");
        }
        return gr::wire::payload(_data);
    }

    [[nodiscard]] std::uint32_t payloadByteCount() const noexcept {
        const std::uint32_t sz  = recSize();
        const std::uint32_t off = recPayloadOffset();
        return sz > off ? sz - off : 0U;
    }

    [[nodiscard]] std::uint32_t strLen() const noexcept {
        const auto* p   = payloadAs<char>();
        const auto  cap = payloadByteCount();
        if (p == nullptr || cap == 0U) {
            return 0U;
        }
        const auto* nul = static_cast<const char*>(std::memchr(p, '\0', cap));
        return nul != nullptr ? static_cast<std::uint32_t>(nul - p) : cap;
    }

    template<typename T>
    [[nodiscard]] T inlineAs() const noexcept {
        static_assert(sizeof(T) <= 8, "inlineAs<T> only for ≤8-byte scalar types");
        T v{};
        std::memcpy(&v, recPayload(), sizeof(T));
        return v;
    }

    template<typename T = std::byte>
    [[nodiscard]] const T* payloadAs() const noexcept {
        static_assert(std::is_trivially_copyable_v<T>, "payloadAs<T>: T must be trivially copyable");
        assert(reinterpret_cast<std::uintptr_t>(recPayload()) % alignof(T) == 0 && "payloadAs<T>: payload not T-aligned");
        return reinterpret_cast<const T*>(recPayload());
    }

    [[nodiscard]] std::span<const std::byte> recordSpan() const noexcept { return std::span<const std::byte>{_data, recSize()}; }

    [[nodiscard]] constexpr ValueType     value_type() const noexcept { return static_cast<ValueType>(recValueTypeByte()); }
    [[nodiscard]] constexpr ContainerType container_type() const noexcept { return static_cast<ContainerType>(recContainerTypeByte()); }

    [[nodiscard]] constexpr bool is_monostate() const noexcept { return value_type() == ValueType::Monostate; }
    [[nodiscard]] constexpr bool is_arithmetic() const noexcept { return (container_type() == ContainerType::Scalar || container_type() == ContainerType::Complex) && value_type() != ValueType::Monostate; }
    [[nodiscard]] constexpr bool is_integral() const noexcept { return container_type() == ContainerType::Scalar && (value_type() >= ValueType::Int8 && value_type() <= ValueType::UInt64); }
    [[nodiscard]] constexpr bool is_signed_integral() const noexcept { return container_type() == ContainerType::Scalar && (value_type() >= ValueType::Int8 && value_type() <= ValueType::Int64); }
    [[nodiscard]] constexpr bool is_unsigned_integral() const noexcept { return container_type() == ContainerType::Scalar && (value_type() >= ValueType::UInt8 && value_type() <= ValueType::UInt64); }
    [[nodiscard]] constexpr bool is_floating_point() const noexcept { return container_type() == ContainerType::Scalar && (value_type() == ValueType::Float32 || value_type() == ValueType::Float64); }
    [[nodiscard]] constexpr bool is_complex() const noexcept { return container_type() == ContainerType::Complex; }
    [[nodiscard]] constexpr bool is_string() const noexcept { return container_type() == ContainerType::String; }
    [[nodiscard]] constexpr bool is_tensor() const noexcept { return container_type() == ContainerType::Tensor; }
    [[nodiscard]] constexpr bool is_map() const noexcept { return container_type() == ContainerType::Map; }

    [[nodiscard]] constexpr bool     has_value() const noexcept { return !is_monostate(); }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return has_value(); }

    template<typename T>
    static constexpr ValueType get_value_type() {
        if constexpr (std::same_as<T, bool>) {
            return ValueType::Bool;
        } else if constexpr (std::same_as<T, std::int8_t>) {
            return ValueType::Int8;
        } else if constexpr (std::same_as<T, std::int16_t>) {
            return ValueType::Int16;
        } else if constexpr (std::same_as<T, std::int32_t>) {
            return ValueType::Int32;
        } else if constexpr (std::same_as<T, std::int64_t>) {
            return ValueType::Int64;
        } else if constexpr (std::same_as<T, std::uint8_t>) {
            return ValueType::UInt8;
        } else if constexpr (std::same_as<T, std::uint16_t>) {
            return ValueType::UInt16;
        } else if constexpr (std::same_as<T, std::uint32_t>) {
            return ValueType::UInt32;
        } else if constexpr (std::same_as<T, std::uint64_t> || std::same_as<T, std::size_t>) {
            return ValueType::UInt64;
        } else if constexpr (std::same_as<T, float>) {
            return ValueType::Float32;
        } else if constexpr (std::same_as<T, double>) {
            return ValueType::Float64;
        } else if constexpr (std::same_as<T, std::complex<float>>) {
            return ValueType::ComplexFloat32;
        } else if constexpr (std::same_as<T, std::complex<double>>) {
            return ValueType::ComplexFloat64;
        } else if constexpr (std::same_as<T, std::pmr::string>) {
            return ValueType::String;
        } else if constexpr (std::same_as<T, gr::pmt::Value>) {
            return ValueType::Value;
        } else {
            return ValueType::Monostate;
        }
    }

    template<typename T>
    static constexpr ContainerType get_container_type() {
        if constexpr (std::same_as<T, std::complex<float>> || std::same_as<T, std::complex<double>>) {
            return ContainerType::Complex;
        } else if constexpr (std::same_as<T, std::pmr::string>) {
            return ContainerType::String;
        } else {
            return ContainerType::Scalar;
        }
    }

    template<typename T>
    requires(!meta::is_instantiation_of<T, std::vector>)
    [[nodiscard]] bool holds() const noexcept;

    template<typename T>
    requires(!std::is_array_v<T> && !meta::is_instantiation_of<T, std::vector> && !std::is_same_v<T, std::string> && !std::is_same_v<T, std::string_view> && !std::is_same_v<T, std::pmr::string> && !std::is_same_v<T, Tensor<std::string>> && !std::is_same_v<T, ValueMap> && !gr::TensorViewLike<T> && !gr::TensorLike<T>)
    [[nodiscard]] T* get_if() noexcept;

    template<typename T>
    requires(!std::is_array_v<T> && !meta::is_instantiation_of<T, std::vector> && !std::is_same_v<T, std::string> && !std::is_same_v<T, std::string_view> && !std::is_same_v<T, std::pmr::string> && !std::is_same_v<T, Tensor<std::string>> && !std::is_same_v<T, std::monostate> && !std::is_same_v<T, ValueMap> && !gr::TensorViewLike<T> && !gr::TensorLike<T>)
    [[nodiscard]] const T* get_if() const noexcept {
#ifdef __EMSCRIPTEN__
        static_assert(!std::is_same_v<std::size_t, T>);
#endif
        return const_cast<ValueView*>(this)->get_if<T>();
    }

    template<typename T>
    requires std::same_as<T, std::string_view>
    [[nodiscard]] std::optional<std::string_view> get_if() const noexcept {
        if (!is_string()) {
            return std::nullopt;
        }
        return std::string_view{payloadAs<char>(), strLen()};
    }

    template<typename T>
    requires(gr::TensorViewLike<T> && !std::same_as<std::remove_const_t<typename T::value_type>, gr::pmt::Value>)
    [[nodiscard]] std::optional<T> get_if() const noexcept {
        using ElemT = std::remove_const_t<typename T::value_type>;
        if (!is_tensor() || value_type() != get_value_type<ElemT>()) {
            return std::nullopt;
        }
        const auto* base = payloadAs<std::byte>();
        if (base == nullptr || payloadByteCount() < kTensorBlobHeaderSize) {
            return std::nullopt;
        }
        const auto rank          = static_cast<std::uint8_t>(base[1]);
        const auto encodingFlags = static_cast<std::uint8_t>(base[2]);
        if (rank > kMaxTensorRank || (encodingFlags & kTensorEncodingVariableSize) != 0U) {
            return std::nullopt; // variable-size element types can't alias contiguous bytes
        }
        // precondition: makeView-injected bytes are well-formed (untrusted input goes through from_blob -> _validateTensorSubBlob)
        const std::size_t extentsBytes = paddedTensorExtentsBytes(rank);
        assert(payloadByteCount() >= kTensorBlobHeaderSize + extentsBytes);
        std::array<std::size_t, kMaxTensorRank> extentsBuf{};
        for (std::size_t i = 0UZ; i < rank; ++i) {
            std::uint32_t ext;
            std::memcpy(&ext, base + kTensorBlobHeaderSize + 4UZ * i, sizeof(ext));
            extentsBuf[i] = static_cast<std::size_t>(ext);
        }
        const std::span<const std::size_t> extents{extentsBuf.data(), rank};
        const auto*                        elementData = base + kTensorBlobHeaderSize + extentsBytes;
        std::uint32_t                      elementCount;
        std::memcpy(&elementCount, base + 4, sizeof(elementCount));
        assert(static_cast<std::uint64_t>(elementCount) * sizeof(ElemT) <= payloadByteCount() - (kTensorBlobHeaderSize + extentsBytes));
        if constexpr (std::same_as<ElemT, bool>) {
            return T{elementData, extents, static_cast<std::size_t>(elementCount)};
        } else {
            // Build via non-const ElemT TensorView and implicit-convert to T (handles TensorView<const T>).
            // Override _size to elementCount because TensorView's ctor computes size = product(extents)
            // = 1 for empty extents (rank-0 scalar convention) — wrong for our elementCount=0 case.
            gr::TensorView<ElemT> nonConstView{const_cast<ElemT*>(reinterpret_cast<const ElemT*>(elementData)), extents};
            nonConstView._data._size     = static_cast<std::size_t>(elementCount);
            nonConstView._data._capacity = static_cast<std::size_t>(elementCount);
            return T{nonConstView};
        }
    }

    [[nodiscard]] std::string value_or(const std::string& default_val) const {
        if (is_string()) {
            return std::string(payloadAs<char>(), strLen());
        }
        return default_val;
    }

    [[nodiscard, gnu::always_inline]] std::string_view value_or(std::string_view default_val) const noexcept {
        if (is_string()) {
            return std::string_view{payloadAs<char>(), strLen()};
        }
        return default_val;
    }

    template<typename T>
    requires(!std::is_same_v<T, std::monostate> && !detail::is_string_convertible_v<std::remove_cvref_t<T>>) && (!std::is_reference_v<T> || detail::is_const_ref_v<T>)
    [[nodiscard]] auto value_or(T&& default_val) const -> detail::return_t<T> {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (gr::TensorLike<Raw>) {
            using ElemT = std::remove_const_t<typename Raw::value_type>;
            static_assert(!std::same_as<ElemT, Value>, "ValueView::value_or<Tensor<Value>>: sub-Value decode needs Value::_resource; bind to `const pmt::Value v = (*it).second;` and call v.value_or<Tensor<Value>>(default).");
            if (auto view = get_if<gr::TensorView<ElemT>>()) {
                return view->owned(std::pmr::get_default_resource());
            }
            return static_cast<Raw>(std::forward<T>(default_val));
        } else if constexpr (std::is_lvalue_reference_v<T>) {
            if (auto* p = get_if<Raw>()) {
                return *p;
            }
            return std::forward<T>(default_val);
        } else {
            if (auto* p = get_if<T>()) {
                return *p;
            }
            return std::forward<T>(default_val);
        }
    }

    template<typename F>
    [[nodiscard]] std::string or_else_string(F&& factory) const {
        if (is_string()) {
            return std::string(payloadAs<char>(), strLen());
        }
        return std::forward<F>(factory)();
    }
    template<typename F>
    [[nodiscard]] std::string_view or_else_string_view(F&& factory) const noexcept(noexcept(std::forward<F>(factory)())) {
        if (is_string()) {
            return std::string_view{payloadAs<char>(), strLen()};
        }
        return std::forward<F>(factory)();
    }

    template<typename T, typename F>
    requires(!std::is_reference_v<T>)
    [[nodiscard]] T or_else(F&& factory) const {
        if (auto* p = get_if<T>()) {
            return *p;
        }
        return std::forward<F>(factory)();
    }

    template<typename T, typename F>
    requires(!std::is_reference_v<T>)
    [[nodiscard]] auto transform(F&& func) const -> std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<const T&>()))> {
        using R = std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<const T&>()))>;
        if (auto* p = get_if<T>()) {
            return std::forward<F>(func)(*p);
        }
        return R{};
    }

    template<typename T, typename F, typename D>
    requires(!std::is_reference_v<T>)
    [[nodiscard]] auto transform_or(F&& func, D&& default_val) const -> std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<const T&>()))> {
        using R = std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<const T&>()))>;
        if (auto* p = get_if<T>()) {
            return std::forward<F>(func)(*p);
        }
        return static_cast<R>(std::forward<D>(default_val));
    }

    template<typename T>
    requires std::same_as<T, ValueMap>
    [[nodiscard("view aliases source bytes; call .owned() before escaping *this scope")]] std::optional<ValueMap> get_if() const noexcept;
    [[nodiscard]] Value                                                                                           owned(std::pmr::memory_resource* resource = std::pmr::get_default_resource()) const;
    [[nodiscard]] bool                                                                                            compare_scalar_eq(const ValueView& other) const noexcept;
    [[nodiscard]] std::partial_ordering                                                                           compare_scalar_order(const ValueView& other) const noexcept;
};
static_assert(std::is_trivially_copyable_v<ValueView>);
static_assert(sizeof(ValueView) == sizeof(std::byte*));

class Value : public ValueView {
public:
    using ValueType                                               = gr::pmt::ValueType;
    using ContainerType                                           = gr::pmt::ContainerType;
    static constexpr std::size_t          kRecOffsetSize          = gr::pmt::kRecOffsetSize;
    static constexpr std::size_t          kRecOffsetValueType     = gr::pmt::kRecOffsetValueType;
    static constexpr std::size_t          kRecOffsetContainerType = gr::pmt::kRecOffsetContainerType;
    static constexpr std::size_t          kRecOffsetFlags         = gr::pmt::kRecOffsetFlags;
    static constexpr std::size_t          kRecOffsetPayload       = gr::pmt::kRecOffsetPayload;
    static constexpr std::uint32_t        kRecMinSize             = gr::pmt::kRecMinSize;
    [[nodiscard]] static const std::byte* monostateRecord() noexcept { return gr::pmt::monostateRecord(); }

    struct MapHash {
        using is_transparent = void;

        std::size_t operator()(const char* txt) const { return std::hash<std::string_view>{}(txt); }
        std::size_t operator()(std::string_view txt) const { return std::hash<std::string_view>{}(txt); }
        template<typename CharT, typename Traits, typename Alloc>
        std::size_t operator()(const std::basic_string<CharT, Traits, Alloc>& txt) const {
            return std::hash<std::basic_string<CharT, Traits, Alloc>>{}(txt);
        }
    };

    struct MapEqual {
        using is_transparent = void;

        bool operator()(const auto& left, const auto& right) const { return std::string_view(left) == std::string_view(right); }
    };

private:
    std::pmr::memory_resource* _resource{nullptr}; // nullptr ⇒ view-mode

    [[nodiscard]] static std::pmr::memory_resource* ensure_resource(std::pmr::memory_resource* r) noexcept { return r != nullptr ? r : std::pmr::get_default_resource(); }

    void allocAndWriteHeader(std::uint32_t size, ValueType vt, ContainerType ct, std::uint8_t flags = 0U) {
        if (_resource == nullptr) {
            _resource = std::pmr::get_default_resource();
        }
        assert(size >= kRecMinSize);
        _data = static_cast<std::byte*>(_resource->allocate(size, alignof(std::max_align_t)));
        gr::wire::writeHeaderSized(_data, size, static_cast<std::uint8_t>(vt), static_cast<std::uint8_t>(ct), flags);
        // payload bytes left uninitialised; caller writes them
    }

    void copy_from(const ValueView& other);
    void destroy() noexcept;

    void initStringTensor(std::ranges::sized_range auto& strings) {
        Tensor<Value> tensor({std::ranges::size(strings)});
        std::ranges::transform(strings, tensor.begin(), [this](auto& str) { return Value(std::move(str), _resource); });
        init_from_tensor(std::move(tensor));
    }

    Value& assignStringTensor(std::ranges::sized_range auto& strings) {
        destroy();
        initStringTensor(strings);
        return *this;
    }

    template<detail::ExternalValueMap T>
    void init_from_map(T&& map);

public:
    Value() noexcept = default; // _data → monostateRecord, _resource = nullptr (view of sentinel)

    explicit Value(std::pmr::memory_resource* resource) noexcept {
        if (resource != nullptr) {
            _resource = resource;
            allocAndWriteHeader(kRecMinSize, ValueType::Monostate, ContainerType::Scalar);
            std::memset(recPayloadMutable(), 0, 8);
        }
    }

    Value(std::monostate, std::pmr::memory_resource* resource = nullptr) noexcept : Value(resource) {}

    template<typename T>
    void _initInlineScalar(T v, std::pmr::memory_resource* resource, ValueType vt, ContainerType ct) {
        _resource = ensure_resource(resource);
        allocAndWriteHeader(kRecMinSize, vt, ct);
        std::memset(recPayloadMutable(), 0, 8);          // zero-pad sub-8-byte types
        std::memcpy(recPayloadMutable(), &v, sizeof(T)); // write low bytes
    }

    Value(bool v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar<std::uint8_t>(v ? 1U : 0U, resource, ValueType::Bool, ContainerType::Scalar); }
    Value(int8_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::Int8, ContainerType::Scalar); }
    Value(int16_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::Int16, ContainerType::Scalar); }
    Value(int32_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::Int32, ContainerType::Scalar); }
    Value(int64_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::Int64, ContainerType::Scalar); }
    Value(uint8_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::UInt8, ContainerType::Scalar); }
    Value(uint16_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::UInt16, ContainerType::Scalar); }
    Value(uint32_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::UInt32, ContainerType::Scalar); }
    Value(uint64_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::UInt64, ContainerType::Scalar); }
    Value(float v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::Float32, ContainerType::Scalar); }
    Value(double v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::Float64, ContainerType::Scalar); }
    Value(std::complex<float> v, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) { _initInlineScalar(v, resource, ValueType::ComplexFloat32, ContainerType::Complex); } // 8 B fits inline
    Value(std::complex<double> v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());                                                                                      // 16 B record + 16 B payload appended; out-of-line in Value.cpp
    Value(std::string_view v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(const std::string& v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(const std::pmr::string& v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(const char* v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());

    Value(const Value& other) : Value(other, other._resource) {}
    Value(const Value& other, std::pmr::memory_resource* resource);
    Value(const ValueView& other, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(Value&& other) noexcept;
    Value& operator=(const Value& other);
    Value& operator=(Value&& other) noexcept; // cross-resource path may allocate; OOM → std::terminate (PMR convention)
    ~Value();

    [[nodiscard]] constexpr std::pmr::memory_resource* resource() const noexcept { return _resource; }

    // INTERNAL — UNSAFE: `base` MUST point at a value-record from a verified source (`from_blob`,
    // an owning Value's allocation, …). Use `from_blob` for untrusted inputs.
    [[nodiscard]] static Value makeView(ValueType /*vt*/, ContainerType /*ct*/, const std::byte* base, std::uint32_t /*length*/, std::pmr::memory_resource* /*resource*/) noexcept {
        Value v;
        v._data     = const_cast<std::byte*>(base);
        v._resource = nullptr; // view-mode discriminator
        return v;
    }

    Value& operator=(bool v);
    Value& operator=(int8_t v);
    Value& operator=(int16_t v);
    Value& operator=(int32_t v);
    Value& operator=(int64_t v);
    Value& operator=(uint8_t v);
    Value& operator=(uint16_t v);
    Value& operator=(uint32_t v);
    Value& operator=(uint64_t v);

    Value& operator=(float v);
    Value& operator=(double v);
    Value& operator=(std::complex<float> v);
    Value& operator=(std::complex<double> v);
    Value& operator=(std::string_view v);
    Value& operator=(const std::string& v);
    Value& operator=(const std::pmr::string& v);
    Value& operator=(const char* v);

    // Callers constructing Value from a Tensor MUST include <gnuradio-4.0/ValueMap.hpp>.
    template<TensorLike Tens>
    void init_from_tensor(Tens&& tensor);

    template<TensorLike TensorCollection>
    Value(TensorCollection tensor, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(ensure_resource(resource)) {
        init_from_tensor(std::move(tensor));
    }

    template<TensorLike TensorCollection>
    Value& operator=(TensorCollection tensor) {
        destroy();
        if (_resource == nullptr) {
            _resource = std::pmr::get_default_resource();
        }
        init_from_tensor(std::move(tensor));
        return *this;
    }

    template<detail::ValueScalarType T, typename Alloc>
    Value(std::vector<T, Alloc> vec, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : Value(Tensor<T>(std::pmr::vector<T>(vec.begin(), vec.end(), resource ? resource : std::pmr::get_default_resource())), resource) {}

    template<detail::ValueScalarType T, typename Alloc>
    Value& operator=(std::vector<T, Alloc> vec) {
        auto* res    = ensure_resource(_resource);
        return *this = Tensor<T>(std::pmr::vector<T>(vec.begin(), vec.end(), res));
    }

    template<detail::ValueScalarType T, std::size_t N>
    Value(std::array<T, N> arr, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : Value(Tensor<T>(std::pmr::vector<T>(arr.begin(), arr.end(), resource ? resource : std::pmr::get_default_resource())), resource) {}

    template<detail::ValueScalarType T, std::size_t N>
    Value& operator=(std::array<T, N> arr) {
        auto* res    = ensure_resource(_resource);
        return *this = Tensor<T>(std::pmr::vector<T>(arr.begin(), arr.end(), res));
    }

    template<typename Alloc>
    Value(std::vector<std::string, Alloc> strings, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(ensure_resource(resource)) {
        initStringTensor(strings);
    }

    template<typename Alloc>
    Value& operator=(std::vector<std::string, Alloc> strings) {
        return assignStringTensor(strings);
    }

    template<std::size_t N>
    Value(std::array<std::string, N> strings, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(ensure_resource(resource)) {
        initStringTensor(strings);
    }

    template<std::size_t N>
    Value& operator=(std::array<std::string, N> strings) {
        return assignStringTensor(strings);
    }

    Value(ValueMap map, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value& operator=(ValueMap map);

    template<detail::ExternalValueMap T>
    Value(T&& map, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(ensure_resource(resource)) {
        init_from_map(std::forward<T>(map));
    }

    template<detail::ExternalValueMap T>
    Value& operator=(T&& map) {
        destroy();
        init_from_map(std::forward<T>(map));
        return *this;
    }

    [[nodiscard]] constexpr bool is_view() const noexcept { return _resource == nullptr; }

    using ValueView::get_if; // re-expose base overloads so the typed overloads below don't hide them

    // May allocate via `_resource` for sub-Value decode; not noexcept.
    template<typename T>
    requires(gr::TensorViewLike<T> && std::same_as<std::remove_const_t<typename T::value_type>, gr::pmt::Value>)
    [[nodiscard]] std::optional<T> get_if() const;

    // get_if<ValueMap> is inherited from ValueView (same view-mode body; lifetime bound to *this).

    [[nodiscard]] std::string value_or(const std::string& default_val) const& {
        if (is_string()) {
            return std::string(payloadAs<char>(), strLen());
        }
        return default_val;
    }

    [[nodiscard, gnu::always_inline]] std::string_view value_or(std::string_view default_val) const& noexcept {
        if (is_string()) {
            return std::string_view{payloadAs<char>(), strLen()};
        }
        return default_val;
    }

    template<typename T> // mutable
    requires(!detail::is_string_convertible_v<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, ValueMap>)
    [[nodiscard]] auto value_or(T&& default_val) & -> detail::return_t<T> {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (gr::TensorLike<Raw>) {
            // Owning-materialisation path for Tensor<T>. Routes exclusively through the view-based
            // get_if + view->owned() — no `get_if<Tensor<T>>` decoded-copy form exists in the API.
            using ElemT = std::remove_const_t<typename Raw::value_type>;
            if (auto view = get_if<gr::TensorView<ElemT>>()) {
                return view->owned(_resource ? _resource : std::pmr::get_default_resource());
            }
            return static_cast<Raw>(std::forward<T>(default_val));
        } else if constexpr (std::is_rvalue_reference_v<T>) {
            // T&& → ownership transfer: move out, reset to monostate
            if (auto* p = get_if<Raw>()) {
                Raw tmp = std::move(*p);
                clear();
                return tmp;
            }
            return static_cast<Raw>(std::forward<T>(default_val));
        } else if constexpr (std::is_lvalue_reference_v<T>) {
            // T& or const T& → borrow: return reference
            if (auto* p = get_if<Raw>()) {
                return *p;
            }
            return std::forward<T>(default_val);
        } else {
            // T → copy: return by value
            if (auto* p = get_if<T>()) {
                return *p;
            }
            return std::forward<T>(default_val);
        }
    }

    template<typename T> // const value_or — only T and const T& (not T& or T&&)
    requires(!std::is_same_v<T, std::monostate> && !detail::is_string_convertible_v<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, ValueMap>) && (!std::is_reference_v<T> || detail::is_const_ref_v<T>)
    [[nodiscard]] auto value_or(T&& default_val) const& -> detail::return_t<T> {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (gr::TensorLike<Raw>) {
            using ElemT = std::remove_const_t<typename Raw::value_type>;
            if (auto view = get_if<gr::TensorView<ElemT>>()) {
                return view->owned(_resource ? _resource : std::pmr::get_default_resource());
            }
            return static_cast<Raw>(std::forward<T>(default_val));
        } else if constexpr (std::is_lvalue_reference_v<T>) {
            // const T& requested
            if (auto* p = get_if<Raw>()) {
                return *p;
            }
            return std::forward<T>(default_val);
        } else {
            // T requested: return by value (copy)
            if (auto* p = get_if<T>()) {
                return *p;
            }
            return std::forward<T>(default_val);
        }
    }

    template<typename F>
    [[nodiscard]] std::string or_else_string(F&& factory) const& {
        if (is_string()) {
            return std::string(payloadAs<char>(), strLen());
        }
        return std::forward<F>(factory)();
    }

    template<typename F>
    [[nodiscard]] std::string_view or_else_string_view(F&& factory) const& noexcept(noexcept(std::forward<F>(factory)())) {
        if (is_string()) {
            return std::string_view{payloadAs<char>(), strLen()};
        }
        return std::forward<F>(factory)();
    }

    template<typename T, typename F>
    requires(!std::is_lvalue_reference_v<T>)
    [[nodiscard]] auto or_else(F&& factory) & -> std::remove_reference_t<T> {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (std::is_rvalue_reference_v<T>) {
            if (auto* p = get_if<Raw>()) {
                Raw tmp = std::move(*p);
                clear();
                return tmp;
            }
        } else {
            if (auto* p = get_if<T>()) {
                return *p;
            }
        }
        return std::forward<F>(factory)();
    }

    template<typename T, typename F>
    requires(!std::is_reference_v<T>)
    [[nodiscard]] T or_else(F&& factory) const& {
        if (auto* p = get_if<T>()) {
            return *p;
        }
        return std::forward<F>(factory)();
    }

    template<typename T, typename F>
    requires(!std::is_lvalue_reference_v<T>)
    [[nodiscard]] auto transform(F&& func) & {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (std::is_rvalue_reference_v<T>) {
            using R = std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<Raw&&>()))>;
            if (auto* p = get_if<Raw>()) {
                R result = std::forward<F>(func)(std::move(*p));
                clear();
                return result;
            }
            return R{};
        } else {
            using R = std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<T&>()))>;
            if (auto* p = get_if<T>()) {
                R result = std::forward<F>(func)(*p);
                return result;
            }
            return R{};
        }
    }

    template<typename T, typename F>
    requires(!std::is_reference_v<T>)
    [[nodiscard]] auto transform(F&& func) const& -> std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<const T&>()))> {
        using R = std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<const T&>()))>;
        if (auto* p = get_if<T>()) {
            return std::forward<F>(func)(*p);
        }
        return R{};
    }

    template<typename T, typename F>
    requires(!std::is_reference_v<T>)
    [[nodiscard]] auto transform(F&& func) && -> std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<T&>()))> {
        using Raw = std::remove_cvref_t<T>;
        using R   = std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<T&>()))>;
        if (auto* p = get_if<Raw>()) {
            auto tmp = std::move(*p);
            clear();
            return std::forward<F>(func)(tmp);
        }
        return R{}; // mismatch → default-constructed R (monostate Value)
    }

    template<typename T, typename F, typename D>
    requires(!std::is_lvalue_reference_v<T>)
    [[nodiscard]] auto transform_or(F&& func, D&& default_val) & {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (std::is_rvalue_reference_v<T>) {
            using R = std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<Raw&&>()))>;
            if (auto* p = get_if<Raw>()) {
                R result = std::forward<F>(func)(std::move(*p));
                clear();
                return result;
            }
            return static_cast<R>(std::forward<D>(default_val));
        } else {
            using R = std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<T&>()))>;
            if (auto* p = get_if<T>()) {
                R result = std::forward<F>(func)(*p);
                return result;
            }
            return static_cast<R>(std::forward<D>(default_val));
        }
    }

    template<typename T, typename F, typename D>
    requires(!std::is_reference_v<T>)
    [[nodiscard]] auto transform_or(F&& func, D&& default_val) const& -> std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<const T&>()))> {
        using R = std::remove_cvref_t<decltype(std::forward<F>(func)(std::declval<const T&>()))>;
        if (auto* p = get_if<T>()) {
            return std::forward<F>(func)(*p);
        }
        return static_cast<R>(std::forward<D>(default_val));
    }

    [[nodiscard]] bool                  operator==(const Value& other) const;
    [[nodiscard]] std::partial_ordering operator<=>(const Value& other) const;

    friend bool operator==(const ValueView&, const ValueView&);

    template<detail::ValueComparable T>
    friend bool operator==(const ValueView&, const T&);
    template<detail::ValueComparable T>
    friend bool operator==(const T&, const ValueView&);

    void clear() noexcept {
        destroy();
        // Restore the Monostate sentinel — view-mode (no owning bytes to write).
        _data     = const_cast<std::byte*>(monostateRecord());
        _resource = nullptr;
    }

    friend constexpr std::string detail::value_to_string(const ValueView&);
    friend void                  swap(Value& a, Value& b) noexcept;
};

static_assert(sizeof(gr::pmt::Value) == sizeof(std::byte*) + sizeof(std::pmr::memory_resource*), "Value handle is std::byte* + std::pmr::memory_resource* (== 2 pointers)");

[[nodiscard]] bool operator==(const ValueView&, const ValueView&);

inline Value ValueView::owned(std::pmr::memory_resource* resource) const { return Value{*this, resource ? resource : std::pmr::get_default_resource()}; }

template<detail::ValueComparable T>
bool operator==(const ValueView&, const T&);
template<detail::ValueComparable T>
bool operator==(const T&, const ValueView&);

// Value-side delegates break the overload ambiguity between the member operator==(Value)
// (Value(T) implicit ctor) and the ValueView template (Value -> ValueView slice).
template<detail::ValueComparable T>
inline bool operator==(const Value& lhs, const T& rhs) {
    return static_cast<const ValueView&>(lhs) == rhs;
}
template<detail::ValueComparable T>
inline bool operator==(const T& lhs, const Value& rhs) {
    return lhs == static_cast<const ValueView&>(rhs);
}
inline bool operator==(const Value& lhs, const ValueView& rhs) { return static_cast<const ValueView&>(lhs) == rhs; }
inline bool operator==(const ValueView& lhs, const Value& rhs) { return lhs == static_cast<const ValueView&>(rhs); }

} // namespace gr::pmt

namespace gr {
/**
 * @brief View over the variable-size wire-format bytes of a Tensor<Value>.
 *
 * Aliases the source Value's blob bytes — valid for the source Value's lifetime. Element access
 * decodes lazily and yields owning Values allocated at the view's resource. `owned(mr)` snapshots
 * the whole tensor.
 *
 * Random access is O(i) (sequential walk over packed records); the variable-size encoding only
 * fires for heterogeneous content — homogeneous fixed-size cells take the typed `TensorView<T>`
 * path via the F2 wire-format collapse.
 */
template<std::size_t... Ex>
struct TensorView<gr::pmt::Value, Ex...> {
    using value_type   = gr::pmt::Value;
    using element_type = gr::pmt::Value;
    using container_t  = Tensor<gr::pmt::Value, Ex...>;

    std::span<const std::byte>                    _bytes; // tensor sub-blob: header + extents + per-element records
    std::array<std::size_t, gr::detail::kMaxRank> _extents{};
    std::size_t                                   _rank{};
    std::size_t                                   _elementCount{};
    std::pmr::memory_resource*                    _resource{}; // arena for decoded Values (host-only)

    constexpr TensorView() noexcept                             = default;
    constexpr TensorView(const TensorView&) noexcept            = default;
    constexpr TensorView(TensorView&&) noexcept                 = default;
    constexpr TensorView& operator=(const TensorView&) noexcept = default;
    constexpr TensorView& operator=(TensorView&&) noexcept      = default;

    [[nodiscard]] constexpr std::size_t                  size() const noexcept { return _elementCount; }
    [[nodiscard]] constexpr std::size_t                  rank() const noexcept { return _rank; }
    [[nodiscard]] constexpr bool                         empty() const noexcept { return _elementCount == 0UZ; }
    [[nodiscard]] constexpr std::span<const std::size_t> extents() const noexcept { return {_extents.data(), _rank}; }
    [[nodiscard]] constexpr std::span<const std::byte>   bytes() const noexcept { return _bytes; }

    [[nodiscard]] gr::pmt::Value operator[](std::size_t i) const;

    struct const_iterator {
        using iterator_category = std::input_iterator_tag;
        using iterator_concept  = std::input_iterator_tag;
        using value_type        = gr::pmt::Value;
        using difference_type   = std::ptrdiff_t;
        using reference         = gr::pmt::Value;

        struct ArrowProxy {
            gr::pmt::Value                                _val;
            [[nodiscard]] constexpr const gr::pmt::Value* operator->() const noexcept { return &_val; }
        };

        const TensorView* _view       = nullptr;
        std::size_t       _index      = 0UZ;
        std::uint32_t     _byteCursor = 0U;

        [[nodiscard]] gr::pmt::Value operator*() const;
        [[nodiscard]] ArrowProxy     operator->() const { return ArrowProxy{**this}; }
        const_iterator&              operator++();
        [[nodiscard]] const_iterator operator++(int) {
            const_iterator tmp = *this;
            ++*this;
            return tmp;
        }
        [[nodiscard]] friend constexpr bool operator==(const const_iterator& a, const const_iterator& b) noexcept { return a._view == b._view && a._index == b._index; }
    };

    using iterator = const_iterator;

    [[nodiscard]] constexpr const_iterator begin() const noexcept { return const_iterator{this, 0UZ, 0U}; }
    [[nodiscard]] constexpr const_iterator end() const noexcept { return const_iterator{this, _elementCount, 0U}; }

    [[nodiscard]] container_t owned(std::pmr::memory_resource* mr = std::pmr::get_default_resource()) const;
};

} // namespace gr

// clang-format off
// Scalar set used in extern-template instantiations. `std::pmr::string` excluded — string storage
// is a raw byte-blob; use `get_if<std::string_view>` or `value_or<std::string>(default)` instead.
#define GR_PMT_VALUE_SCALAR_TYPES \
    X(bool)                       \
    X(std::int8_t)                \
    X(std::int16_t)               \
    X(std::int32_t)               \
    X(std::int64_t)               \
    X(std::uint8_t)               \
    X(std::uint16_t)              \
    X(std::uint32_t)              \
    X(std::uint64_t)              \
    X(float)                      \
    X(double)                     \
    X(std::complex<float>)        \
    X(std::complex<double>)

#define GR_PMT_VALUE_TENSOR_ELEMENT_TYPES \
    GR_PMT_VALUE_SCALAR_TYPES             \
    X(std::pmr::string)                   \
    X(gr::pmt::Value)

namespace gr::pmt {

#define X(T)                                                                    \
    extern template bool     ValueView::holds<T>() const noexcept;              \
    extern template T*       ValueView::get_if<T>() noexcept;                   \
    extern template const T* ValueView::get_if<T>() const noexcept;
GR_PMT_VALUE_SCALAR_TYPES
#undef X

#define X(T) extern template bool ValueView::holds<Tensor<T>>() const noexcept;
GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

extern template bool ValueView::holds<std::pmr::string>() const noexcept;

#define X(T)                                                         \
    extern template bool operator== <T>(const ValueView&, const T&); \
    extern template bool operator== <T>(const T&, const ValueView&);

GR_PMT_VALUE_SCALAR_TYPES
#undef X

extern template bool ValueView::holds<std::string>() const noexcept;
extern template bool ValueView::holds<std::string_view>() const noexcept;

#ifdef __EMSCRIPTEN__
template<>
bool               ValueView::holds<std::size_t>() const noexcept = delete;
#endif

// clang-format on

} // namespace gr::pmt

namespace std {

template<>
struct hash<gr::pmt::ValueView> {
    [[nodiscard]] std::size_t operator()(const gr::pmt::ValueView& v) const noexcept;

private:
    static constexpr std::size_t hashCombine(std::size_t seed, std::size_t h) noexcept { return seed ^ (h + 0x9e3779b9UZ + (seed << 6) + (seed >> 2)); }

    template<typename T>
    static std::size_t hashValue(const T& value) noexcept {
        if constexpr (gr::meta::complex_like<T>) {
            using VT = typename T::value_type;
            return hashCombine(std::hash<VT>{}(value.real()), std::hash<VT>{}(value.imag()));
        } else {
            return std::hash<T>{}(value);
        }
    }

    static std::size_t hashTensorElements(gr::TensorLike auto const& tensor) noexcept {
        std::size_t seed = std::hash<std::size_t>{}(tensor.size());
        for (const auto& elem : tensor) {
            seed = hashCombine(seed, hashValue(elem));
        }
        return seed;
    }

    static std::size_t hashScalar(const gr::pmt::ValueView& v) noexcept;
    static std::size_t hashTensor(const gr::pmt::ValueView& v) noexcept;
    static std::size_t hashMap(const gr::pmt::ValueView& v) noexcept;
};

template<>
struct hash<gr::pmt::Value> {
    [[nodiscard]] std::size_t operator()(const gr::pmt::Value& v) const noexcept { return std::hash<gr::pmt::ValueView>{}(static_cast<const gr::pmt::ValueView&>(v)); }
};

} // namespace std

#endif // GNURADIO_VALUE_HPP
