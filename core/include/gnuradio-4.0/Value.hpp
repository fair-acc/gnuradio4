#ifndef GNURADIO_VALUE_HPP
#define GNURADIO_VALUE_HPP

#include <cassert>
#include <complex>
#include <concepts>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <unordered_map> // TODO: replace with std::flat_map once widely available (libc++ >=20, stdlibc++ >=15)

#include <gnuradio-4.0/Tensor.hpp>

namespace gr::pmt {

class Value;

namespace detail {
[[noreturn, gnu::cold]] inline void value_type_mismatch() noexcept {
    assert(false && "gr::pmt::Value type mismatch");
    std::unreachable();
    std::abort();
}

/// Helper to determine return type based on requested type category.
/// For T&& requests, returns T (by value after move); otherwise returns T as-is.
template<typename T>
using return_t = std::conditional_t<std::is_rvalue_reference_v<T>, std::remove_reference_t<T>, T>;

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

template<typename T>
concept ValueScalarType = std::same_as<std::remove_cvref_t<T>, bool>                                                                                                                                                                                 //
                          || std::same_as<std::remove_cvref_t<T>, std::int8_t> || std::same_as<std::remove_cvref_t<T>, std::int16_t> || std::same_as<std::remove_cvref_t<T>, std::int32_t> || std::same_as<std::remove_cvref_t<T>, std::int64_t>     //
                          || std::same_as<std::remove_cvref_t<T>, std::uint8_t> || std::same_as<std::remove_cvref_t<T>, std::uint16_t> || std::same_as<std::remove_cvref_t<T>, std::uint32_t> || std::same_as<std::remove_cvref_t<T>, std::uint64_t> //
                          || std::same_as<std::remove_cvref_t<T>, float> || std::same_as<std::remove_cvref_t<T>, double>                                                                                                                             //
                          || std::same_as<std::remove_cvref_t<T>, std::complex<float>> || std::same_as<std::remove_cvref_t<T>, std::complex<double>>;                                                                                                //

template<typename T>
concept ValueConvertible = std::same_as<std::remove_cvref_t<T>, Value> || ValueScalarType<T> || std::convertible_to<T, std::string_view>;

template<typename T>
concept ValueComparable = ValueScalarType<T> || std::same_as<std::remove_cvref_t<T>, std::pmr::string>;

template<typename M>
concept ValueMapLike = requires(const std::remove_cvref_t<M>& m) {
    typename std::remove_cvref_t<M>::key_type;
    typename std::remove_cvref_t<M>::mapped_type;
    { m.begin() } -> std::input_iterator;
    { m.end() } -> std::sentinel_for<decltype(m.begin())>;
} && std::convertible_to<typename std::remove_cvref_t<M>::key_type, std::string_view> && ValueConvertible<typename std::remove_cvref_t<M>::mapped_type>;

template<typename M>
concept ExternalValueMap = ValueMapLike<M> && !std::same_as<std::remove_cvref_t<M>, std::pmr::unordered_map<std::pmr::string, Value>>;

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
 *   - Strings: std::pmr::string (PMR heap)
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
 *  std::pmr::string s = v.value_or<std::pmr::string&&>(...);// ownership transfer
 * @endcode
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

    using Map = std::pmr::unordered_map<std::pmr::string, Value, MapHash, MapEqual>; // TODO: replace with std::flat_map or other more simpler key-value map once available (libc++ >=20, stdlibc++ >=15)

    uint8_t _value_type : 4 {0U};
    uint8_t _container_type : 4 {0U};
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

    void set_types(ValueType vt, ContainerType ct) noexcept;
    void copy_from(const Value& other);
    void destroy() noexcept;

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

    template<detail::ExternalValueMap T>
    void init_from_map(T&& map) {
        using DecayedMap           = std::remove_cvref_t<T>;
        using MappedType           = typename DecayedMap::mapped_type;
        constexpr bool isValueType = std::same_as<MappedType, Value>;
        constexpr bool canMove     = std::is_rvalue_reference_v<T&&>;

        set_types(ValueType::Value, ContainerType::Map);
        void* mem    = _resource->allocate(sizeof(Map), alignof(Map));
        auto* newMap = new (mem) Map(_resource);
        newMap->reserve(map.size());

        for (auto& [key, val] : map) {
            if constexpr (isValueType && canMove) {
                newMap->emplace(std::pmr::string(key, _resource), std::move(val));
            } else if constexpr (isValueType) {
                newMap->emplace(std::pmr::string(key, _resource), val);
            } else {
                newMap->emplace(std::pmr::string(key, _resource), Value{val, _resource});
            }
        }
        _storage.ptr = newMap;
    }

public:
    // ───────────────────────────────────────────────────────────────────────────────────────────────
    // CONSTRUCTION
    // ───────────────────────────────────────────────────────────────────────────────────────────────

    explicit Value(std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(bool v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(int8_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(int16_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(int32_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(int64_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(uint8_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(uint16_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(uint32_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(uint64_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(float v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(double v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(std::complex<float> v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(std::complex<double> v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(std::string_view v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(const std::string& v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(const char* v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    explicit Value(std::monostate, std::pmr::memory_resource* resource = std::pmr::get_default_resource());

#ifdef __EMSCRIPTEN__
    explicit Value(std::size_t v, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
#endif

    // copy/move/destructor
    Value(const Value& other) : Value(other, std::pmr::get_default_resource()) {}
    Value(const Value& other, std::pmr::memory_resource* resource) : _value_type(other._value_type), _container_type(other._container_type), _storage{}, _resource(ensure_resource(resource ? resource : other._resource)) { copy_from(other); }
    Value(Value&& other) noexcept;
    Value& operator=(const Value& other);
    Value& operator=(Value&& other) noexcept;
    ~Value();

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

#ifdef __EMSCRIPTEN__
    Value& operator=(std::size_t v);
#endif

    Value& operator=(float v);
    Value& operator=(double v);
    Value& operator=(std::complex<float> v);
    Value& operator=(std::complex<double> v);
    Value& operator=(std::string_view v);
    Value& operator=(const std::string& v);
    Value& operator=(const char* v);

    template<TensorLike TensorCollection>
    Value(TensorCollection tensor, std::pmr::memory_resource* resource = std::pmr::get_default_resource()) : _resource(ensure_resource(resource)) {
        using T = TensorCollection::value_type;
        set_types(get_value_type<T>(), ContainerType::Tensor);
        void* mem    = _resource->allocate(sizeof(Tensor<T>), alignof(Tensor<T>));
        _storage.ptr = new (mem) Tensor<T>(std::move(tensor));
    }

    template<TensorLike TensorCollection>
    Value& operator=(TensorCollection tensor) {
        using T = TensorCollection::value_type;
        destroy();
        set_types(get_value_type<T>(), ContainerType::Tensor);
        void* mem    = _resource->allocate(sizeof(Tensor<T>), alignof(Tensor<T>));
        _storage.ptr = new (mem) Tensor<T>(std::move(tensor));
        return *this;
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

    [[nodiscard]] constexpr bool     has_value() const noexcept { return !is_monostate(); }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return has_value(); }

    /// Returns true if Value holds type T or a type convertible to T.
    /// Special cases:
    ///   - holds<std::string>() returns true if is_string() (auto-convertible)
    ///   - holds<std::string_view>() returns true if is_string() (zero-copy view)
    template<typename T>
    requires(!meta::is_instantiation_of<T, std::vector>)
    [[nodiscard]] bool holds() const noexcept;

    /// Safe pointer access - returns nullptr on type mismatch
    /// @note For std::string/std::string_view, use value_or() instead (requires conversion)
    template<typename T>
    requires(!std::is_array_v<T> && !meta::is_instantiation_of<T, std::vector> && !std::is_same_v<T, std::string> && !std::is_same_v<T, Tensor<std::string>>
#ifdef __EMSCRIPTEN__
             && !std::is_same_v<T, std::size_t>
#endif
        )
    [[nodiscard]] T* get_if() noexcept;

    template<typename T>
    requires(!std::is_array_v<T> && !meta::is_instantiation_of<T, std::vector> && !std::is_same_v<T, std::string> && !std::is_same_v<T, Tensor<std::string>> && !std::is_same_v<T, std::monostate>)
    [[nodiscard]] const T* get_if() const noexcept {
        return const_cast<Value*>(this)->get_if<T>();
    }

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
    //   value_or<std::string>(fallback)      → auto-convert from pmr::string (allocates)
    //   value_or<std::string_view>(fallback) → zero-copy view of pmr::string
    //   These always return by value; reference variants are not supported for conversions.
    //
    // TYPE STRICTNESS: For non-string types, the fallback type must match exactly.
    // On type mismatch, returns fallback; Value is unchanged (except T&& resets on match).

    // ─── String conversion overloads (always by-value) ───────────────────────────────────────────

    /// value_or for std::string - auto-converts from pmr::string (allocates)
    [[nodiscard]] std::string value_or(std::string default_val) const& {
        if (is_string()) {
            return std::string(*static_cast<const std::pmr::string*>(_storage.ptr));
        }
        return default_val;
    }

    /// value_or for std::string_view - zero-copy view of pmr::string
    /// @warning Returned view is invalidated if Value is modified or destroyed
    [[nodiscard]] std::string_view value_or(std::string_view default_val) const& noexcept {
        if (is_string()) {
            const auto* str = static_cast<const std::pmr::string*>(_storage.ptr);
            return std::string_view{str->data(), str->size()};
        }
        return default_val;
    }

    // ─── Generic value_or (excludes string conversions handled above) ────────────────────────────

    /// monadic value_or — supports T, T&, const T&, T&&
    /// @return    copy/reference/moved value on match, fallback on mismatch
    template<typename T> // mutable
    requires(!detail::is_string_convertible_v<std::remove_cvref_t<T>>)
    [[nodiscard]] auto value_or(T&& default_val) & -> detail::return_t<T> {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (std::is_rvalue_reference_v<T>) {
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
    requires(!std::is_same_v<T, std::monostate> && !detail::is_string_convertible_v<std::remove_cvref_t<T>>) && (!std::is_reference_v<T> || detail::is_const_ref_v<T>)
    [[nodiscard]] auto value_or(T&& default_val) const& -> detail::return_t<T> {
        using Raw = std::remove_cvref_t<T>;
        if constexpr (std::is_lvalue_reference_v<T>) {
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

    /// or_else for std::string - auto-converts from pmr::string
    template<typename F>
    [[nodiscard]] std::string or_else_string(F&& factory) const& {
        if (is_string()) {
            return std::string(*static_cast<const std::pmr::string*>(_storage.ptr));
        }
        return std::forward<F>(factory)();
    }

    /// or_else for std::string_view - zero-copy view
    template<typename F>
    [[nodiscard]] std::string_view or_else_string_view(F&& factory) const& noexcept {
        if (is_string()) {
            const auto* str = static_cast<const std::pmr::string*>(_storage.ptr);
            return std::string_view{str->data(), str->size()};
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
    X(std::complex<double>)       \
    X(std::pmr::string)

#define GR_PMT_VALUE_TENSOR_ELEMENT_TYPES \
    GR_PMT_VALUE_SCALAR_TYPES             \
    X(gr::pmt::Value)

namespace gr::pmt {

#define X(T)                                                                    \
    extern template Value&   Value::operator=(Tensor<T>&& tensor);              \
    extern template Value&   Value::operator=(const Tensor<T>& tensor);         \
    extern template bool     Value::holds<T>() const noexcept;                  \
    extern template T*       Value::get_if<T>() noexcept;                       \
    extern template const T* Value::get_if<T>() const noexcept;

GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

#define X(T)                                                     \
    extern template bool operator== <T>(const Value&, const T&); \
    extern template bool operator== <T>(const T&, const Value&);

GR_PMT_VALUE_SCALAR_TYPES
#undef X

// string type specializations (convertible from pmr::string)
extern template bool Value::holds<std::string>() const noexcept;
extern template bool Value::holds<std::string_view>() const noexcept;

#ifdef __EMSCRIPTEN__
extern template bool               Value::holds<std::size_t>() const noexcept;
#endif

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
