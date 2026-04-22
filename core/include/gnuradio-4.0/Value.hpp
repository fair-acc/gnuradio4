#ifndef GNURADIO_VALUE_HPP
#define GNURADIO_VALUE_HPP

#include <cassert>
#include <complex>
#include <concepts>
#include <cstdint>
#include <memory_resource>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <unordered_map> // TODO: replace with std::flat_map once widely available (libc++ >=20, stdlibc++ >=15)

#include <gnuradio-4.0/Tensor.hpp>

namespace gr::pmt {

class Value;
class ValueMap; // forward declaration — full definition in ValueMap.hpp; Map operations live in Value.cpp

} // namespace gr::pmt

namespace gr {
// Q1.B: partial specialisation of TensorView for T=gr::pmt::Value.
//
// Tensor<Value>'s on-blob encoding is variable-size (per-element [PackedTensorElement] header +
// variable-length payload), so a TensorView<Value> can NOT alias a contiguous Value array out of
// the byte-blob — the elements are not contiguous Value objects on disk. To keep the API symmetric
// with TensorView<T> for fixed-size T (used by std::vector<Value> settings fields and the same
// generic templated dispatch), this specialisation owns a Tensor<Value> internally that is decoded
// once on construction. Iteration / size / extents / element access forward to the decoded snapshot.
//
// "View" is a slight misnomer for this specialisation (it owns a copy), but the user-facing API
// (begin/end/size/operator[]/data/owned) is the same shape as the non-specialised case, so generic
// caller code reads the same.
template<std::size_t... Ex>
struct TensorView<gr::pmt::Value, Ex...> {
    using value_type     = gr::pmt::Value;
    using element_type   = gr::pmt::Value;
    using container_t    = Tensor<gr::pmt::Value, Ex...>;
    using iterator       = typename container_t::iterator;
    using const_iterator = typename container_t::const_iterator;

    container_t _data{}; // owning decoded snapshot

    TensorView()                                 = default;
    TensorView(const TensorView&)                = default;
    TensorView(TensorView&&) noexcept            = default;
    TensorView& operator=(const TensorView&)     = default;
    TensorView& operator=(TensorView&&) noexcept = default;

    // Construct from an owning Tensor<Value>. Used by Value::get_if<TensorView<Value>>() after
    // decoding the byte-blob into a stable Tensor<Value>.
    explicit TensorView(container_t snapshot) : _data(std::move(snapshot)) {}

    template<std::size_t... TEx>
    requires(sizeof...(TEx) > 0 || sizeof...(Ex) == 0)
    explicit TensorView(const Tensor<gr::pmt::Value, TEx...>& source) : _data(source) {}

    template<std::size_t... TEx>
    requires(sizeof...(TEx) > 0 || sizeof...(Ex) == 0)
    explicit TensorView(Tensor<gr::pmt::Value, TEx...>& source) : _data(source) {}

    // Forward-range API
    [[nodiscard]] auto        begin() noexcept { return _data.begin(); }
    [[nodiscard]] auto        end() noexcept { return _data.end(); }
    [[nodiscard]] auto        begin() const noexcept { return _data.begin(); }
    [[nodiscard]] auto        end() const noexcept { return _data.end(); }
    [[nodiscard]] std::size_t size() const noexcept { return _data.size(); }
    [[nodiscard]] std::size_t rank() const noexcept { return _data.rank(); }
    [[nodiscard]] auto        extents() const noexcept { return _data.extents(); }
    [[nodiscard]] auto        strides() const noexcept { return _data.strides(); }
    [[nodiscard]] auto*       data() noexcept { return _data.data(); }
    [[nodiscard]] const auto* data() const noexcept { return _data.data(); }

    [[nodiscard]] gr::pmt::Value&       operator[](std::size_t i) noexcept { return _data[i]; }
    [[nodiscard]] const gr::pmt::Value& operator[](std::size_t i) const noexcept { return _data[i]; }

    [[nodiscard]] container_t owned(std::pmr::memory_resource* res = std::pmr::get_default_resource()) const { return container_t{_data, res}; }
};
} // namespace gr

namespace gr::pmt {

// ─── Tensor byte-blob format (Q1 inversion: Value owns its own tensor bytes) ────────────────
// Format (mirrors what ValueMap stored in tensor sub-blobs pre-Q1; now Value's intrinsic shape):
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

/// Helper to determine return type based on requested type category.
/// For T&& requests, returns T (by value after move); otherwise returns T as-is.
/// Tensor / ValueMap exceptions: byte-blob storage has no stable T*, so reference-to-storage
/// returns are unsupported — value_or always returns by value.
template<typename T>
using return_t = std::conditional_t<std::is_rvalue_reference_v<T>, std::remove_reference_t<T>, std::conditional_t<gr::TensorLike<std::remove_cvref_t<T>>, std::remove_cvref_t<T>, T>>;

/// Check if T is a const lvalue reference.
template<typename T>
inline constexpr bool is_const_ref_v = std::is_lvalue_reference_v<T> && std::is_const_v<std::remove_reference_t<T>>;

/// Check if T is a non-const lvalue reference.
template<typename T>
inline constexpr bool is_mutable_ref_v = std::is_lvalue_reference_v<T> && !std::is_const_v<std::remove_reference_t<T>>;

template<typename T>
inline constexpr bool is_std_string_v = std::same_as<std::remove_cvref_t<T>, std::string>;

template<typename T>
inline constexpr bool is_string_view_v = std::same_as<std::remove_cvref_t<T>, std::string_view>;

template<typename T>
inline constexpr bool is_string_convertible_v = is_std_string_v<T> || is_string_view_v<T>;

template<typename T, typename... Ts>
concept IsAnyOf = (std::same_as<T, Ts> || ...);

// clang-format off
template<typename T>
concept ValueScalarType = IsAnyOf<std::remove_cvref_t<T>,
    bool,
    std::int8_t, std::int16_t, std::int32_t, std::int64_t,
    std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t,
    signed long, signed long long,       // platform-dependent aliasing: may differ from intN_t
    unsigned long, unsigned long long,   // platform-dependent aliasing: may differ from uintN_t
    float, double, std::complex<float>, std::complex<double>>;
// clang-format on

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

constexpr std::string value_to_string(const Value&); // forward declaration
} // namespace detail

/**
 * @brief Value a compact polymorphic type container
 *
 * Value provides type-erased storage for scalars, strings, tensors, and maps with a 24-byte
 * footprint. Scalars are stored inline (zero allocation); complex types use PMR allocators.
 * The implementation is type-strict and suppresses the common source of errors due to
 * implicit conversions. The PMR allocators permit transitions between 'host' and 'device'
 * in heterogeneous computing environments (e.g. SYCL/CUDA/...)
 *
 * @par Supported Types
 *   - Scalars: bool, int8–64, uint8–64, float, double (inline storage)
 *   - Complex: std::complex<float/double> (PMR heap)
 *   - Strings: raw byte-blob (PMR heap, [size:4][chars][\0])
 *   - Containers: Tensor<T>, Map (PMR heap)
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
 * @note `get_if<std::pmr::string>` is dropped — strings are stored as a raw byte-blob,
 *       not a `std::pmr::string` object. Use `get_if<std::string_view>()` for alloc-free
 *       view access (works for owning + view modes) or `value_or<std::string>(default)`
 *       for an owning copy.
 *
 * @warning Unchecked accessors have undefined behavior on type mismatch (assert in debug).
 *          Use holds<T>(), get_if<T>(), or value_or() for safe access.
 */
class Value {
public:
    // clang-format off
    enum class ValueType : uint8_t {
        Monostate = 0U,
        Bool = 1U,
        Int8 = 2U, Int16 = 3U, Int32 = 4U, Int64 = 5U,
        UInt8 = 6U, UInt16 = 7U, UInt32 = 8U, UInt64 = 9U,
        Float32 = 10U, Float64 = 11U,
        ComplexFloat32 = 12U, ComplexFloat64 = 13U,
        String = 14U,
        Value = 15U  // for Tensor<Value> and Map<string,Value>
    };
    // clang-format on
    enum class ContainerType : uint8_t { Scalar = 0U, Complex = 1U, String = 2U, Tensor = 3U, Map = 4U };

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

    // Phase 1e Step C: Map alias is the contiguous packed-blob ValueMap (was a
    // std::pmr::unordered_map<std::pmr::string, Value, MapHash, MapEqual>). Storage stays as
    // a heap pointer (`_storage.ptr → ValueMap*`) for now; the byte-blob storage redesign
    // happens in a follow-up step. Map operations live in Value.cpp where ValueMap is fully
    // defined.
    using Map = ValueMap;

    // View-mode flag (bit 0 of _flags). When set, _storage.ptr aliases external bytes
    // (typically a ValueMap blob's payload pool) and _payloadLength carries the byte count
    // — the destructor does NOT free _storage.ptr. View-mode is only meaningful for variable-
    // size containers (String, Tensor, Map). Inline scalars never use view-mode.
    static constexpr std::uint8_t kFlagViewMode = 0x01;

    uint8_t       _value_type : 4 {0U}; // bits 0-3 of byte 0
    uint8_t       _container_type : 4 {0U};
    uint8_t       _flags{0U};         // byte 1: kFlagViewMode | (reserved bits 1..7)
    uint16_t      _reserved{0U};      // bytes 2-3
    std::uint32_t _payloadLength{0U}; // bytes 4-7: byte length of view-mode payload (0 for owning)
    union Storage {
        bool          b;
        std::int8_t   i8;
        std::int16_t  i16;
        std::int32_t  i32;
        std::int64_t  i64;
        std::uint8_t  u8;
        std::uint16_t u16;
        std::uint32_t u32;
        std::uint64_t u64;
        float         f32;
        double        f64;
        void*         ptr;

        Storage() : u64(0) {}

    } _storage{};
    std::pmr::memory_resource* _resource;

private:
    [[nodiscard]] static std::pmr::memory_resource* ensure_resource(std::pmr::memory_resource* r) noexcept { return r != nullptr ? r : std::pmr::get_default_resource(); }

    // Inlined here so hot-path callers (e.g. ValueMap iterator's Value construction) collapse
    // through the bit-field writes instead of paying a function-call layer per Value ctor.
    void set_types(ValueType vt, ContainerType ct) noexcept {
        _value_type     = static_cast<unsigned char>(static_cast<std::uint8_t>(vt) & 0x0F);
        _container_type = static_cast<unsigned char>(static_cast<std::uint8_t>(ct) & 0x0F);
    }
    void copy_from(const Value& other);
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
        } else if constexpr (std::same_as<T, std::int64_t> || std::same_as<T, signed long> || std::same_as<T, signed long long>) {
            return ValueType::Int64;
        } else if constexpr (std::same_as<T, std::uint8_t>) {
            return ValueType::UInt8;
        } else if constexpr (std::same_as<T, std::uint16_t>) {
            return ValueType::UInt16;
        } else if constexpr (std::same_as<T, std::uint32_t>) {
            return ValueType::UInt32;
        } else if constexpr (std::same_as<T, std::uint64_t> || std::same_as<T, std::size_t> || std::same_as<T, unsigned long> || std::same_as<T, unsigned long long>) {
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
        } else if constexpr (std::same_as<T, Value>) {
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

    [[nodiscard]] bool                  compare_scalar_eq(const Value& other) const noexcept;
    [[nodiscard]] std::partial_ordering compare_scalar_order(const Value& other) const noexcept;

    // Body moved to ValueMap.hpp (Phase 1e Step C: forward-decl Map = ValueMap means sizeof(Map)
    // and `new Map(...)` need ValueMap's full definition — only available after ValueMap.hpp
    // is included). Callers constructing Value from a non-ValueMap source map MUST include
    // <gnuradio-4.0/ValueMap.hpp>.
    template<detail::ExternalValueMap T>
    void init_from_map(T&& map);

public:
    // ───────────────────────────────────────────────────────────────────────────────────────────────
    // CONSTRUCTION
    // ───────────────────────────────────────────────────────────────────────────────────────────────

    Value(std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(bool v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(int8_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(int16_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(int32_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(int64_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(uint8_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(uint16_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(uint32_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(uint64_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
#if defined(__APPLE__) && defined(__aarch64__) // unsigned long != uint64_t on Apple Silicon
    Value(unsigned long v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
#endif
    Value(float v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(double v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(std::complex<float> v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(std::complex<double> v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(std::string_view v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(const std::string& v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(const std::pmr::string& v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(const char* v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value(std::monostate, std::pmr::memory_resource* resource = std::pmr::get_default_resource());

    // copy/move/destructor
    Value(const Value& other) : Value(other, other._resource) {}
    Value(const Value& other, std::pmr::memory_resource* resource) : _value_type(other._value_type), _container_type(other._container_type), _storage{}, _resource(ensure_resource(resource ? resource : other._resource)) { copy_from(other); }
    Value(Value&& other) noexcept;
    Value& operator=(const Value& other);
    Value& operator=(Value&& other); // cross-resource path may allocate
    ~Value();

    /**
     * @brief Construct a view-mode Value that aliases external bytes (does not own them).
     *
     * View-mode is a within-iteration optimisation: `ValueMap::const_iterator::operator*()` yields
     * view-mode Values for variable-size entries (currently String only; Tensor / Map land in
     * Phase 1e Q1) so the per-deref Value carries a pointer + length into the source ValueMap's
     * blob instead of allocating its own bytes.
     *
     * @par Lifetime contract
     *
     * View-mode bytes live as long as the SOURCE container is alive AND not mutated. Any
     * `emplace`/`insert_or_assign`/`erase`/`clear`/`reserve`/`shrink_to_fit`/`merge` on the source
     * ValueMap invalidates outstanding view-mode Values (same rule as `std::vector` iterator
     * invalidation). Caller discipline only — no `freeze()`-and-iterate enforcement.
     *
     * @par Escaping iter scope
     *
     * The two ways view-mode Values leave their iter scope have OPPOSING lifetime needs:
     *
     * 1. Storing or returning a Value (`auto v = (*it).second; return v;`) — by default,
     *    move-ctor preserves view-mode (cheap pointer transfer), so the destination dangles
     *    when the source ValueMap dies. Caller MUST either bind to lvalue and copy explicitly
     *    (`pmt::Value owned{view}; return owned;` — copy-ctor materialises) OR route through
     *    `at()` instead of iterator deref (`at()` uses `decodeEntry` → owning Value).
     *    See `Settings::CtxSettingsBase::get(string)` for the canonical pattern.
     *
     * 2. Extracting a string_view / pointer from a temp Value (`auto sv = (*it).second.value_or(sv{});`)
     *    — `value_or<string_view>()` and `get_if<T>()` return references/pointers into the
     *    Value's storage, which die at end of full expression. Caller MUST bind the Value to
     *    a const lvalue first (`const pmt::Value e = (*it).second; auto sv = e.value_or(sv{});`)
     *    OR copy out to `std::string` (`auto s = (*it).second.value_or(std::string{});`).
     *
     * Copy-ctor (one-arg) materialises against source's resource; two-arg version materialises
     * against an explicit target resource. Move-ctor preserves view-mode — losing this would
     * defeat the iter perf optimisation by forcing materialisation on every implicit move.
     */
    [[nodiscard]] static Value makeView(ValueType vt, ContainerType ct, const std::byte* base, std::uint32_t length, std::pmr::memory_resource* resource) noexcept {
        Value v(resource);
        v.set_types(vt, ct);
        v._flags         = kFlagViewMode;
        v._payloadLength = length;
        v._storage.ptr   = const_cast<std::byte*>(base); // view storage; never written through
        return v;
    }

    // type-specific assignment
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

    // Q1.B: Tensor byte-blob storage. Body in ValueMap.hpp where encodeTensorBlob is defined.
    // Users constructing Value from a Tensor MUST include <gnuradio-4.0/ValueMap.hpp>.
    template<TensorLike Tens>
    void init_from_tensor(Tens&& tensor);

    template<TensorLike TensorCollection>
    Value(TensorCollection tensor, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(ensure_resource(resource)) {
        init_from_tensor(std::move(tensor));
    }

    template<TensorLike TensorCollection>
    Value& operator=(TensorCollection tensor) {
        destroy();
        init_from_tensor(std::move(tensor));
        return *this;
    }

    // converting constructors for std::vector<T> / std::array<T, N> → Value (numeric → Tensor<T>, string → Tensor<Value>)
    template<detail::ValueScalarType T, typename Alloc>
    Value(std::vector<T, Alloc> vec, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : Value(Tensor<T>(std::pmr::vector<T>(vec.begin(), vec.end(), resource)), resource) {}

    template<detail::ValueScalarType T, typename Alloc>
    Value& operator=(std::vector<T, Alloc> vec) {
        return *this = Tensor<T>(std::pmr::vector<T>(vec.begin(), vec.end(), _resource));
    }

    template<detail::ValueScalarType T, std::size_t N>
    Value(std::array<T, N> arr, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : Value(Tensor<T>(std::pmr::vector<T>(arr.begin(), arr.end(), resource)), resource) {}

    template<detail::ValueScalarType T, std::size_t N>
    Value& operator=(std::array<T, N> arr) {
        return *this = Tensor<T>(std::pmr::vector<T>(arr.begin(), arr.end(), _resource));
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

    Value(Map map, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    Value& operator=(Map map);

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

    // ───────────────────────────────────────────────────────────────────────────────────────────────
    // TYPE QUERIES
    // ───────────────────────────────────────────────────────────────────────────────────────────────

    [[nodiscard]] constexpr ValueType     value_type() const noexcept { return static_cast<ValueType>(_value_type); }
    [[nodiscard]] constexpr ContainerType container_type() const noexcept { return static_cast<ContainerType>(_container_type); }

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
    [[nodiscard]] constexpr bool is_view() const noexcept { return (_flags & kFlagViewMode) != 0U; }

    [[nodiscard]] constexpr bool     has_value() const noexcept { return !is_monostate(); }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return has_value(); }

    /// Returns true if Value holds type T or a type convertible to T.
    /// Special cases:
    ///   - holds<std::string>() returns true if is_string() (auto-convertible)
    ///   - holds<std::string_view>() returns true if is_string() (zero-copy view)
    template<typename T>
    requires(!meta::is_instantiation_of<T, std::vector>)
    [[nodiscard]] bool holds() const noexcept;

    /// Safe pointer access — returns `T*` to the owning storage on hit, nullptr on type mismatch
    /// (including when this is a view-mode Value — view-mode aliases external bytes, no `T*` to
    /// hand out; use the view-type overload instead, e.g. `get_if<std::string_view>()`).
    /// @note For std::string / std::pmr::string, use `value_or<std::string>(default)` (auto-converts,
    ///       allocates) or `get_if<std::string_view>()` for alloc-free view (Phase 1e: strings are
    ///       stored as raw byte-blobs, not as `std::pmr::string` objects).
    template<typename T>
    requires(!std::is_array_v<T> && !meta::is_instantiation_of<T, std::vector> && !std::is_same_v<T, std::string> && !std::is_same_v<T, std::string_view> && !std::is_same_v<T, std::pmr::string> && !std::is_same_v<T, Tensor<std::string>> && !std::is_same_v<T, ValueMap> && !gr::TensorViewLike<T> && !gr::TensorLike<T>)
    [[nodiscard]] T* get_if() noexcept;

    template<typename T>
    requires(!std::is_array_v<T> && !meta::is_instantiation_of<T, std::vector> && !std::is_same_v<T, std::string> && !std::is_same_v<T, std::string_view> && !std::is_same_v<T, std::pmr::string> && !std::is_same_v<T, Tensor<std::string>> && !std::is_same_v<T, std::monostate> && !std::is_same_v<T, ValueMap> && !gr::TensorViewLike<T> && !gr::TensorLike<T>)
    [[nodiscard]] const T* get_if() const noexcept {
#ifdef __EMSCRIPTEN__
        static_assert(!std::is_same_v<std::size_t, T>);
#endif
        return const_cast<Value*>(this)->get_if<T>();
    }

    /// Q1.B: get_if<Tensor<T>>() — Tensor storage is byte-blob; no stable Tensor<T>* exists.
    /// Returns std::optional<Tensor<T>> (decoded copy on hit, allocates per call), nullopt on
    /// type mismatch / empty. For zero-copy access prefer get_if<TensorView<T>>(). Body in
    /// ValueMap.hpp where decodeTensorBlob is defined.
    template<typename T>
    requires gr::TensorLike<T>
    [[nodiscard]] std::optional<T> get_if() const noexcept;

    /// View-type overload — returns std::optional<std::string_view> on hit, std::nullopt on type
    /// mismatch. Works uniformly across owning + view modes (alloc-free). The string_view aliases
    /// the owning byte-blob's bytes (Phase 1e) or the view-mode external bytes — lifetime-bound to
    /// the source Value.
    template<typename T>
    requires std::same_as<T, std::string_view>
    [[nodiscard]] std::optional<std::string_view> get_if() const noexcept {
        if (!is_string()) {
            return std::nullopt;
        }
        return std::string_view{static_cast<const char*>(_storage.ptr), _payloadLength};
    }

    /// View-type overload — returns std::optional<TensorView<U,...>> on hit, std::nullopt on type
    /// mismatch. Non-owning view aliasing the owning Tensor's element storage; lifetime-bound to
    /// the source Value (Phase 1e step B preparation: read-only callers should prefer this over
    /// `get_if<Tensor<U,...>>()` so the eventual storage redesign — heap Tensor → byte-blob — does
    /// not break them). Same span-style const-laxness as std::span: a mutable TensorView can be
    /// obtained from a const Value; pick `TensorView<const U,...>` for an enforced read-only view.
    /// Q1.B: get_if<TensorView<T>>() — works uniformly across owning + view-mode byte-blob storage.
    ///
    /// For fixed-size T (bool, intN/uintN, float, double, complex<float>, complex<double>): aliases
    /// the data section of the tensor blob (header + extents + data). Zero allocation. Lifetime
    /// bound to *this. bool is supported because the byte-blob stores 1 byte per bool (contiguous);
    /// the TensorView<bool> ctor (T* data, extents) accepts this directly.
    ///
    /// For T=Value (variable-size element encoding): partial specialisation in this header decodes
    /// the blob into a Tensor<Value> and wraps it in a TensorView<Value> snapshot.
    template<typename T>
    requires(gr::TensorViewLike<T> && !std::same_as<std::remove_const_t<typename T::value_type>, gr::pmt::Value>)
    [[nodiscard]] std::optional<T> get_if() const noexcept {
        using ElemT = std::remove_const_t<typename T::value_type>;
        if (!is_tensor() || value_type() != get_value_type<ElemT>()) {
            return std::nullopt;
        }
        const auto* base = static_cast<const std::byte*>(_storage.ptr);
        if (base == nullptr || _payloadLength < kTensorBlobHeaderSize) {
            return std::nullopt;
        }
        const auto rank          = static_cast<std::uint8_t>(base[1]);
        const auto encodingFlags = static_cast<std::uint8_t>(base[2]);
        if (rank > kMaxTensorRank || (encodingFlags & kTensorEncodingVariableSize) != 0U) {
            return std::nullopt; // variable-size element types can't alias contiguous bytes
        }
        std::array<std::size_t, kMaxTensorRank> extentsBuf{};
        for (std::size_t i = 0UZ; i < rank; ++i) {
            std::uint32_t ext;
            std::memcpy(&ext, base + kTensorBlobHeaderSize + 4UZ * i, sizeof(ext));
            extentsBuf[i] = static_cast<std::size_t>(ext);
        }
        const std::span<const std::size_t> extents{extentsBuf.data(), rank};
        const auto*                        elementData = base + kTensorBlobHeaderSize + 4UZ * rank;
        // Build via non-const ElemT TensorView and implicit-convert to T (handles TensorView<const T>).
        gr::TensorView<ElemT> nonConstView{const_cast<ElemT*>(reinterpret_cast<const ElemT*>(elementData)), extents};
        return T{nonConstView};
    }

    /// Specialisation for TensorView<Value>: decode the byte-blob into a Tensor<Value> snapshot
    /// and return it inside the TensorView<Value> partial specialisation (which carries a
    /// container_t Tensor<Value> internally). Body in ValueMap.hpp where decodeTensorBlob lives.
    template<typename T>
    requires(gr::TensorViewLike<T> && std::same_as<std::remove_const_t<typename T::value_type>, gr::pmt::Value>)
    [[nodiscard]] std::optional<T> get_if() const noexcept;

    /// View-mode-aware ValueMap accessor (post Q1). Returns std::optional<ValueMap> on hit
    /// where the inner ValueMap is in view-mode (alloc-free; aliases the source Value's bytes
    /// when in view-mode, or aliases the owning ValueMap's blob when in owning mode).
    /// Lifetime-bound to the source Value. For an owning copy, call `result->owned(resource)`.
    /// Defined out-of-line in Value.cpp because ValueMap is forward-declared here.
    template<typename T>
    requires std::same_as<T, ValueMap>
    [[nodiscard]] std::optional<ValueMap> get_if() const noexcept;

    // ───────────────────────────────────────────────────────────────────────────────────────────────
    // UNIFIED VALUE ACCESS — value_or<T|T&|const T&|T&&>(fallback)
    // ───────────────────────────────────────────────────────────────────────────────────────────────
    //
    // The template parameter T encodes ownership/lifetime semantics:
    //
    //   value_or<T>(fallback)         → Copy: returns T by value, Value retains ownership
    //   value_or<T&>(fallback)        → Borrow mutable: returns T&, modify in-place
    //   value_or<const T&>(fallback)  → Borrow const: returns const T&, read without copy
    //   value_or<T&&>(fallback)       → Transfer: moves out and resets Value to monostate
    //
    // STRING CONVERSION:
    //   value_or<std::string>(fallback)      → auto-convert from byte-blob (allocates)
    //   value_or<std::string_view>(fallback) → zero-copy view of byte-blob
    //   These always return by value; reference variants are not supported for conversions.
    //
    // TYPE STRICTNESS: For non-string types, the fallback type must match exactly.
    // On type mismatch, returns fallback; Value is unchanged (except T&& resets on match).

    // ─── String conversion overloads (always by-value) ───────────────────────────────────────────

    /// value_or for std::string - auto-converts from byte-blob (allocates)
    [[nodiscard]] std::string value_or(std::string default_val) const& {
        if (is_string()) {
            return std::string(static_cast<const char*>(_storage.ptr), _payloadLength);
        }
        return default_val;
    }

    /// value_or for std::string_view - zero-copy view of byte-blob
    /// @warning Returned view is invalidated if Value is modified or destroyed
    [[nodiscard]] std::string_view value_or(std::string_view default_val) const& noexcept {
        if (is_string()) {
            return std::string_view{static_cast<const char*>(_storage.ptr), _payloadLength};
        }
        return default_val;
    }

    // ─── Generic value_or (excludes string conversions handled above) ────────────────────────────

    /// monadic value_or — supports T, T&, const T&, T&&
    /// @return    copy/reference/moved value on match, fallback on mismatch
    template<typename T> // mutable
    requires(!detail::is_string_convertible_v<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, ValueMap>)
    [[nodiscard]] auto value_or(T&& default_val) & -> detail::return_t<T> {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (gr::TensorLike<Raw>) {
            // Tensor storage is byte-blob: get_if returns std::optional<Raw> (decoded snapshot).
            // No stable Raw* exists, so reference returns are unsupported — always returns by value.
            if (auto opt = get_if<Raw>()) {
                return std::move(*opt);
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
            return static_cast<T>(std::forward<T>(default_val));
        }
    }

    template<typename T> // const value_or — only T and const T& (not T& or T&&)
    requires(!std::is_same_v<T, std::monostate> && !detail::is_string_convertible_v<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, ValueMap>) && (!std::is_reference_v<T> || detail::is_const_ref_v<T>)
    [[nodiscard]] auto value_or(T&& default_val) const& -> detail::return_t<T> {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (gr::TensorLike<Raw>) {
            if (auto opt = get_if<Raw>()) {
                return std::move(*opt);
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
            return static_cast<T>(std::forward<T>(default_val));
        }
    }

    // ───────────────────────────────────────────────────────────────────────────────────────────────
    // LAZY VALUE ACCESS — or_else<T|T&|const T&|T&&>(factory)
    // ───────────────────────────────────────────────────────────────────────────────────────────────
    //
    // Like value_or but with lazy fallback evaluation. Factory is only invoked on type mismatch.
    // Use when computing the fallback is expensive (I/O, allocation, complex computation).
    //
    // Same ownership semantics as value_or: T&& transfers ownership and resets to monostate.

    // ─── String conversion or_else variants ──────────────────────────────────────────────────────

    /// or_else for std::string - auto-converts from byte-blob
    template<typename F>
    [[nodiscard]] std::string or_else_string(F&& factory) const& {
        if (is_string()) {
            return std::string(static_cast<const char*>(_storage.ptr), _payloadLength);
        }
        return std::forward<F>(factory)();
    }

    /// or_else for std::string_view - zero-copy view of byte-blob
    template<typename F>
    [[nodiscard]] std::string_view or_else_string_view(F&& factory) const& noexcept {
        if (is_string()) {
            return std::string_view{static_cast<const char*>(_storage.ptr), _payloadLength};
        }
        return std::forward<F>(factory)();
    }

    // ─── Generic or_else ─────────────────────────────────────────────────────────────────────────

    /// monadic or_else with lazy factory
    template<typename T, typename F> // mutable
    [[nodiscard]] auto or_else(F&& factory) & -> detail::return_t<T> {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (std::is_rvalue_reference_v<T>) {
            // T&& -> ownership transfer
            if (auto* p = get_if<Raw>()) {
                Raw tmp = std::move(*p);
                clear();
                return tmp;
            }
        } else if constexpr (std::is_lvalue_reference_v<T>) {
            if (auto* p = get_if<Raw>()) {
                return *p;
            }
        } else {
            if (auto* p = get_if<T>()) {
                return *p;
            }
        }
        return std::forward<F>(factory)();
    }

    template<typename T, typename F> // const
    requires(!std::is_reference_v<T> || detail::is_const_ref_v<T>)
    [[nodiscard]] auto or_else(F&& factory) const& -> detail::return_t<T> {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (std::is_lvalue_reference_v<T>) {
            if (auto* p = get_if<Raw>()) {
                return *p;
            }
        } else {
            if (auto* p = get_if<T>()) {
                return *p;
            }
        }
        return std::forward<F>(factory)();
    }

    // ───────────────────────────────────────────────────────────────────────────────────────────────
    // TRANSFORM — apply function if type matches
    // ───────────────────────────────────────────────────────────────────────────────────────────────
    //
    // Applies func to the stored value if type matches, returns default-constructed result otherwise.
    // Use for "extract and convert" patterns: v.transform<string>([](auto& s) { return s.size(); });
    //
    // For T&&, moves from storage AND resets to monostate (ownership transfer).

    /// monadic transform - supports T and T&&
    template<typename T, typename F> // mutable
    requires(!std::is_lvalue_reference_v<T>)
    [[nodiscard]] auto transform(F&& func) & {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (std::is_rvalue_reference_v<T>) {
            using R = decltype(std::forward<F>(func)(std::declval<Raw&&>()));
            if (auto* p = get_if<Raw>()) {
                auto result = std::forward<F>(func)(std::move(*p));
                clear();
                return result;
            }
            return R{};
        } else {
            using R = decltype(std::forward<F>(func)(std::declval<T&>()));
            if (auto* p = get_if<T>()) {
                return std::forward<F>(func)(*p);
            }
            return R{};
        }
    }

    template<typename T, typename F> // const
    requires(!std::is_reference_v<T>)
    [[nodiscard]] auto transform(F&& func) const& {
        using R = decltype(std::forward<F>(func)(std::declval<const T&>()));
        if (auto* p = get_if<T>()) {
            return std::forward<F>(func)(*p);
        }
        return R{};
    }

    // ───────────────────────────────────────────────────────────────────────────────────────────────
    // TRANSFORM_OR — apply function if type matches, return fallback otherwise
    // ───────────────────────────────────────────────────────────────────────────────────────────────
    //
    // Like transform but with explicit fallback instead of default-constructed value.

    /// monadic transform_or — supports T and T&&
    template<typename T, typename F, typename D> // mutable
    requires(!std::is_lvalue_reference_v<T>)
    [[nodiscard]] auto transform_or(F&& func, D&& default_val) & {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (std::is_rvalue_reference_v<T>) {
            using R = decltype(std::forward<F>(func)(std::declval<Raw&&>()));
            if (auto* p = get_if<Raw>()) {
                R result = std::forward<F>(func)(std::move(*p));
                clear();
                return result;
            }
            return static_cast<R>(std::forward<D>(default_val));
        } else {
            using R = decltype(std::forward<F>(func)(std::declval<T&>()));
            if (auto* p = get_if<T>()) {
                return std::forward<F>(func)(*p);
            }
            return static_cast<R>(std::forward<D>(default_val));
        }
    }

    template<typename T, typename F, typename D> // const
    requires(!std::is_reference_v<T>)
    [[nodiscard]] auto transform_or(F&& func, D&& default_val) const& {
        using R = decltype(std::forward<F>(func)(std::declval<const T&>()));
        if (auto* p = get_if<T>()) {
            return std::forward<F>(func)(*p);
        }
        return static_cast<R>(std::forward<D>(default_val));
    }

    // ───────────────────────────────────────────────────────────────────────────────────────────────
    // AND_THEN — monadic bind (func returns Value or similar wrapper)
    // ───────────────────────────────────────────────────────────────────────────────────────────────
    //
    // Like transform but func must return a Value. Enables fluent chaining:
    //   v.and_then<int>([](auto& x) { return Value{x * 2}; })
    //    .and_then<int>([](auto& x) { return Value{to_string(x)}; });
    //
    // Returns default-constructed Value (monostate) on type mismatch.
    // For T&&, moves from storage and resets to monostate.

    /// monadic and_then — supports T and T&&
    template<typename T, typename F> // mutable
    requires(!std::is_lvalue_reference_v<T>)
    [[nodiscard]] auto and_then(F&& func) & {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (std::is_rvalue_reference_v<T>) {
            using R = decltype(std::forward<F>(func)(std::declval<Raw&&>()));
            if (auto* p = get_if<Raw>()) {
                R result = std::forward<F>(func)(std::move(*p));
                clear();
                return result;
            }
            return R{};
        } else {
            using R = decltype(std::forward<F>(func)(std::declval<T&>()));
            if (auto* p = get_if<T>()) {
                return std::forward<F>(func)(*p);
            }
            return R{};
        }
    }

    template<typename T, typename F> // const
    requires(!std::is_reference_v<T>)
    [[nodiscard]] auto and_then(F&& func) const& {
        using R = decltype(std::forward<F>(func)(std::declval<const T&>()));
        if (auto* p = get_if<T>()) {
            return std::forward<F>(func)(*p);
        }
        return R{};
    }

    // Rvalue and_then — move out of storage and call func(T&) on a local
    template<typename T, typename F>
    requires(!std::is_reference_v<T>)
    [[nodiscard]] auto and_then(F&& func) && {
        using Raw = std::remove_cvref_t<T>;
        using R   = decltype(std::forward<F>(func)(std::declval<T&>()));

        if (auto* p = get_if<Raw>()) {
            auto tmp = std::move(*p);
            clear();
            return std::forward<F>(func)(tmp);
        }
        return R{}; // mismatch → default-constructed R (monostate Value)
    }

    // ───────────────────────────────────────────────────────────────────────────────────────────────
    // COMPARISON
    // ───────────────────────────────────────────────────────────────────────────────────────────────

    [[nodiscard]] bool                  operator==(const Value& other) const;
    [[nodiscard]] std::partial_ordering operator<=>(const Value& other) const;

    template<detail::ValueComparable T>
    friend bool operator==(const Value&, const T&);
    template<detail::ValueComparable T>
    friend bool operator==(const T&, const Value&);

    void clear() noexcept {
        destroy();
        set_types(ValueType::Monostate, ContainerType::Scalar);
    }

    friend constexpr std::string detail::value_to_string(const Value&);
    friend void                  swap(Value& a, Value& b) noexcept;
};

static_assert(sizeof(gr::pmt::Value) <= 24UZ, "minimise Value struct size");

template<detail::ValueComparable T>
bool operator==(const Value&, const T&);
template<detail::ValueComparable T>
bool operator==(const T&, const Value&);

} // namespace gr::pmt

// ═══════════════════════════════════════════════════════════════════════════════════════════════════
// EXPLICIT TEMPLATE INSTANTIATION DECLARATIONS
// ═══════════════════════════════════════════════════════════════════════════════════════════════════
// clang-format off
// Direct-accessor scalar types (used for get_if<T>, holds<T>, value_or<T>, Value == T comparisons).
// std::pmr::string is intentionally excluded — Phase 1e dropped get_if<std::pmr::string>; the
// String storage is a raw byte-blob, not a std::pmr::string object. Use get_if<std::string_view>()
// for view access or value_or<std::string>(default) for an owning copy.
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

// Tensor element types (wider than SCALAR — Tensor<std::pmr::string> + Tensor<Value> both work).
#define GR_PMT_VALUE_TENSOR_ELEMENT_TYPES \
    GR_PMT_VALUE_SCALAR_TYPES             \
    X(std::pmr::string)                   \
    X(gr::pmt::Value)

namespace gr::pmt {

// Direct scalar accessors (excluding std::pmr::string and gr::pmt::Value — handled below).
#define X(T)                                                                    \
    extern template bool     Value::holds<T>() const noexcept;                  \
    extern template T*       Value::get_if<T>() noexcept;                       \
    extern template const T* Value::get_if<T>() const noexcept;
GR_PMT_VALUE_SCALAR_TYPES
#undef X

extern template bool   Value::holds<gr::pmt::Value>() const noexcept;
extern template Value* Value::get_if<gr::pmt::Value>() noexcept;
extern template const Value* Value::get_if<gr::pmt::Value>() const noexcept;

// Tensor accessors + Tensor ctors/operator= (works for ALL element types including pmr::string and Value).
// Q1.B: get_if<Tensor<T>> returns std::optional<Tensor<T>> (decoded from byte-blob); body in
// ValueMap.hpp (needs decodeTensorBlob) and is implicitly instantiated.
#define X(T)                                                                  \
    extern template Value& Value::operator=(Tensor<T>&& tensor);              \
    extern template Value& Value::operator=(const Tensor<T>& tensor);         \
    extern template bool   Value::holds<Tensor<T>>() const noexcept;
GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

// holds<std::pmr::string>() is still meaningful (a type predicate equivalent to is_string()),
// even though get_if<std::pmr::string> is dropped post Phase 1e.
extern template bool Value::holds<std::pmr::string>() const noexcept;

#define X(T)                                                     \
    extern template bool operator== <T>(const Value&, const T&); \
    extern template bool operator== <T>(const T&, const Value&);

GR_PMT_VALUE_SCALAR_TYPES
#undef X

// string type specializations (still meaningful as predicates)
extern template bool Value::holds<std::string>() const noexcept;
extern template bool Value::holds<std::string_view>() const noexcept;

#ifdef __EMSCRIPTEN__
template<>
bool               Value::holds<std::size_t>() const noexcept = delete;
#endif

// platform-dependent aliasing: unsigned long may differ from uint64_t (e.g. Apple ARM64)
extern template bool                 Value::holds<unsigned long>() const noexcept;
extern template unsigned long*       Value::get_if<unsigned long>() noexcept;
extern template const unsigned long* Value::get_if<unsigned long>() const noexcept;

// clang-format on

} // namespace gr::pmt

namespace std {

template<>
struct hash<gr::pmt::Value> {
    [[nodiscard]] std::size_t operator()(const gr::pmt::Value& v) const noexcept;

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

    template<typename T>
    static std::size_t hashTensorElements(const gr::Tensor<T>& tensor) noexcept {
        std::size_t seed = std::hash<std::size_t>{}(tensor.size());
        for (const auto& elem : tensor) {
            seed = hashCombine(seed, hashValue(elem));
        }
        return seed;
    }

    static std::size_t hashScalar(const gr::pmt::Value& v) noexcept;
    static std::size_t hashTensor(const gr::pmt::Value& v) noexcept;
    static std::size_t hashMap(const gr::pmt::Value& v) noexcept;
};

} // namespace std

#endif // GNURADIO_VALUE_HPP
