#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/ValueMap.hpp> // Phase 1e Step C: Value::Map = ValueMap; full def needed for Map operations

#include <compare>
#include <concepts>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>

// template definitions
namespace gr::pmt {

// ═══════════════════════════════════════════════════════════════════════════════
// TYPE CHECKING
// ═══════════════════════════════════════════════════════════════════════════════

template<typename T>
requires(!meta::is_instantiation_of<T, std::vector>)
bool Value::holds() const noexcept {
    if constexpr (std::same_as<T, Map>) {
        return is_map();
    } else if constexpr (std::same_as<T, std::pmr::string>) {
        return is_string();
    } else if constexpr (std::same_as<T, std::string>) {
        return is_string();
    } else if constexpr (std::same_as<T, std::string_view>) {
        return is_string();
    } else if constexpr (gr::TensorLike<T>) {
        return is_tensor() && value_type() == get_value_type<typename T::value_type>();
    } else {
        return value_type() == get_value_type<T>() && container_type() == get_container_type<T>();
    }
}

template<typename T>
requires(!std::is_array_v<T> && !meta::is_instantiation_of<T, std::vector> && !std::is_same_v<T, std::string> && !std::is_same_v<T, std::string_view> && !std::is_same_v<T, std::pmr::string> && !std::is_same_v<T, Tensor<std::string>> && !std::is_same_v<T, ValueMap> && !gr::TensorViewLike<T> && !gr::TensorLike<T>)
T* Value::get_if() noexcept {
    if (!holds<T>()) [[unlikely]] {
        return nullptr;
    }
    if constexpr (std::same_as<T, bool>) {
        return &_storage.b;
    } else if constexpr (std::same_as<T, std::int8_t>) {
        return &_storage.i8;
    } else if constexpr (std::same_as<T, std::int16_t>) {
        return &_storage.i16;
    } else if constexpr (std::same_as<T, std::int32_t>) {
        return &_storage.i32;
    } else if constexpr (std::same_as<T, std::int64_t>) {
        return &_storage.i64;
    } else if constexpr (std::same_as<T, std::uint8_t>) {
        return &_storage.u8;
    } else if constexpr (std::same_as<T, std::uint16_t>) {
        return &_storage.u16;
    } else if constexpr (std::same_as<T, std::uint32_t>) {
        return &_storage.u32;
    } else if constexpr (std::same_as<T, std::uint64_t>) {
        return &_storage.u64;
    } else if constexpr (std::same_as<T, unsigned long> || std::same_as<T, unsigned long long>) {
        static_assert(sizeof(T) == sizeof(std::uint64_t) && alignof(T) == alignof(std::uint64_t));
        return reinterpret_cast<T*>(&_storage.u64);
    } else if constexpr (std::same_as<T, signed long> || std::same_as<T, signed long long>) {
        static_assert(sizeof(T) == sizeof(std::int64_t) && alignof(T) == alignof(std::int64_t));
        return reinterpret_cast<T*>(&_storage.i64);
    } else if constexpr (std::same_as<T, float>) {
        return &_storage.f32;
    } else if constexpr (std::same_as<T, double>) {
        return &_storage.f64;
    } else if constexpr (std::same_as<T, std::complex<float>> || std::same_as<T, std::complex<double>>) {
        return static_cast<T*>(_storage.ptr);
    } else if constexpr (std::same_as<T, std::string>) {
        static_assert(gr::meta::always_false<T>, "Use value_or<std::string>() for owning copy, or get_if<std::string_view>() for alloc-free view");
    } else if constexpr (std::same_as<T, Value>) {
        return this;
    } else {
        static_assert(gr::meta::always_false<T>, "Unsupported type for get_if<T>()");
    }
}

} // namespace gr::pmt

namespace gr::pmt {

// set_types now inline in Value.hpp (hot-path inlining for Value ctors called from ValueMap iter).

void Value::copy_from(const Value& other) {
    assert(_resource != nullptr);
#if defined(__cpp_assume) && __cpp_assume >= 202207L
    [[assume(_resource != nullptr)]];
#endif
    // Note: Copy uses THIS object's _resource (target's allocator), not other's.
    // This allows copying between different memory domains (e.g., host ↔ device).
    switch (other.container_type()) {
    case ContainerType::Scalar: _storage = other._storage; break;

    case ContainerType::Complex:
        if (other.value_type() == ValueType::ComplexFloat32) {
            auto* src    = static_cast<const std::complex<float>*>(other._storage.ptr);
            auto* mem    = static_cast<std::complex<float>*>(_resource->allocate(sizeof(std::complex<float>), alignof(std::complex<float>)));
            auto* dst    = std::construct_at(mem, *src);
            _storage.ptr = dst;
        } else {
            auto* src    = static_cast<const std::complex<double>*>(other._storage.ptr);
            auto* mem    = static_cast<std::complex<double>*>(_resource->allocate(sizeof(std::complex<double>), alignof(std::complex<double>)));
            auto* dst    = std::construct_at(mem, *src);
            _storage.ptr = dst;
        }
        break;

    case ContainerType::String: {
        // Phase 1e: raw byte-blob storage (no std::pmr::string object). Source layout is uniform
        // across owning + view modes: _storage.ptr → bytes, _payloadLength = byte count.
        const auto srcView = std::string_view{static_cast<const char*>(other._storage.ptr), other._payloadLength};
        auto*      bytes   = static_cast<char*>(_resource->allocate(srcView.size() + 1U, 1U));
        std::memcpy(bytes, srcView.data(), srcView.size());
        bytes[srcView.size()] = '\0'; // c_str-safe terminator
        _storage.ptr          = bytes;
        _payloadLength        = static_cast<std::uint32_t>(srcView.size());
        _flags                = static_cast<std::uint8_t>(_flags & ~kFlagViewMode); // target is owning regardless of source
        break;
    }

    case ContainerType::Tensor: {
        // Q1.B: Tensor storage is byte-blob (header + extents + data); copy bytes (memcpy).
        // Source might be view-mode (aliasing external bytes) or owning byte-blob — same layout.
        const auto srcLen   = static_cast<std::size_t>(other._payloadLength);
        const auto srcBytes = static_cast<const std::byte*>(other._storage.ptr);
        if (srcLen == 0U || srcBytes == nullptr) {
            _storage.ptr   = nullptr;
            _payloadLength = 0U;
            break;
        }
        auto* bytes = static_cast<std::byte*>(_resource->allocate(srcLen, kBlobAlignment));
        std::memcpy(bytes, srcBytes, srcLen);
        _storage.ptr   = bytes;
        _payloadLength = static_cast<std::uint32_t>(srcLen);
        _flags         = static_cast<std::uint8_t>(_flags & ~kFlagViewMode);
        break;
    }

    case ContainerType::Map: {
        // Q1.B: Map storage is the ValueMap blob bytes. Source bytes might come from a view-mode
        // alias (other.is_view() set) or from owning storage (post-Q1.B both shapes are the same).
        const auto srcLen   = static_cast<std::size_t>(other._payloadLength);
        const auto srcBytes = static_cast<const std::byte*>(other._storage.ptr);
        if (srcLen == 0U || srcBytes == nullptr) {
            _storage.ptr   = nullptr;
            _payloadLength = 0U;
            break;
        }
        auto* bytes = static_cast<std::byte*>(_resource->allocate(srcLen, kBlobAlignment));
        std::memcpy(bytes, srcBytes, srcLen);
        _storage.ptr   = bytes;
        _payloadLength = static_cast<std::uint32_t>(srcLen);
        _flags         = static_cast<std::uint8_t>(_flags & ~kFlagViewMode);
        break;
    }
    }
}

void Value::destroy() noexcept {
    if (!_resource) {
        // moved from value
        return;
    }
#if defined(__cpp_assume) && __cpp_assume >= 202207L
    [[assume(_resource != nullptr)]];
#endif
    if (is_view()) {
        // View-mode: _storage.ptr aliases external bytes (we don't own them). Just disassociate.
        _storage.ptr   = nullptr;
        _payloadLength = 0U;
        _flags         = static_cast<std::uint8_t>(_flags & ~kFlagViewMode);
        return;
    }
    if (is_monostate() || container_type() == ContainerType::Scalar || _storage.ptr == nullptr) {
        return;
    }

    switch (container_type()) {
    case ContainerType::Complex:
        if (value_type() == ValueType::ComplexFloat32) {
            auto* ptr = static_cast<std::complex<float>*>(_storage.ptr);
            std::destroy_at(ptr);
            _resource->deallocate(ptr, sizeof(std::complex<float>), alignof(std::complex<float>));
        } else if (value_type() == ValueType::ComplexFloat64) {
            auto* ptr = static_cast<std::complex<double>*>(_storage.ptr);
            std::destroy_at(ptr);
            _resource->deallocate(ptr, sizeof(std::complex<double>), alignof(std::complex<double>));
        }
        break;

    case ContainerType::String: {
        // Phase 1e: raw byte-blob (no destructor; deallocate raw bytes including '\0').
        _resource->deallocate(_storage.ptr, static_cast<std::size_t>(_payloadLength) + 1U, 1U);
        _payloadLength = 0U;
        break;
    }

    case ContainerType::Tensor: {
        // Q1.B: Tensor storage is byte-blob (no destructor; deallocate raw bytes).
        if (_storage.ptr != nullptr && _payloadLength > 0U) {
            _resource->deallocate(_storage.ptr, _payloadLength, kBlobAlignment);
        }
        _payloadLength = 0U;
        break;
    }

    case ContainerType::Map: {
        // Q1.B: Map storage is byte-blob (ValueMap blob bytes). Free the bytes; no destructor.
        if (_storage.ptr != nullptr && _payloadLength > 0U) {
            _resource->deallocate(_storage.ptr, _payloadLength, kBlobAlignment);
        }
        _payloadLength = 0U;
        break;
    }

    default: break;
    }

    _storage.ptr = nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTRUCTORS
// ═══════════════════════════════════════════════════════════════════════════════

void swap(Value& a, Value& b) noexcept {
    {
        auto tmp      = a._value_type;
        a._value_type = b._value_type;
        b._value_type = tmp & 0x0F;
    }
    {
        auto tmp          = a._container_type;
        a._container_type = b._container_type;
        b._container_type = tmp & 0x0F;
    }
    std::swap(a._flags, b._flags);
    std::swap(a._payloadLength, b._payloadLength);
    std::swap(a._storage, b._storage);
    std::swap(a._resource, b._resource);
}

Value::Value(std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) { set_types(ValueType::Monostate, ContainerType::Scalar); }

Value::Value(std::monostate, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) { set_types(ValueType::Monostate, ContainerType::Scalar); }

Value::Value(bool v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::Bool, ContainerType::Scalar);
    _storage.b = v;
}

Value::Value(int8_t v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::Int8, ContainerType::Scalar);
    _storage.i8 = v;
}

Value::Value(int16_t v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::Int16, ContainerType::Scalar);
    _storage.i16 = v;
}

Value::Value(int32_t v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::Int32, ContainerType::Scalar);
    _storage.i32 = v;
}

Value::Value(int64_t v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::Int64, ContainerType::Scalar);
    _storage.i64 = v;
}

Value::Value(uint8_t v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::UInt8, ContainerType::Scalar);
    _storage.u8 = v;
}

Value::Value(uint16_t v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::UInt16, ContainerType::Scalar);
    _storage.u16 = v;
}

Value::Value(uint32_t v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::UInt32, ContainerType::Scalar);
    _storage.u32 = v;
}

Value::Value(uint64_t v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::UInt64, ContainerType::Scalar);
    _storage.u64 = v;
}

#if defined(__APPLE__) && defined(__aarch64__)
Value::Value(unsigned long v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::UInt64, ContainerType::Scalar);
    _storage.u64 = v;
}
#endif

Value::Value(float v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::Float32, ContainerType::Scalar);
    _storage.f32 = v;
}

Value::Value(double v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::Float64, ContainerType::Scalar);
    _storage.f64 = v;
}

Value::Value(std::complex<float> v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::ComplexFloat32, ContainerType::Complex);
    void* mem    = _resource->allocate(sizeof(std::complex<float>), alignof(std::complex<float>));
    _storage.ptr = new (mem) std::complex<float>(v);
}

Value::Value(std::complex<double> v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::ComplexFloat64, ContainerType::Complex);
    void* mem    = _resource->allocate(sizeof(std::complex<double>), alignof(std::complex<double>));
    _storage.ptr = new (mem) std::complex<double>(v);
}

Value::Value(std::string_view v, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::String, ContainerType::String);
    auto* bytes = static_cast<char*>(_resource->allocate(v.size() + 1U, 1U));
    std::memcpy(bytes, v.data(), v.size());
    bytes[v.size()] = '\0';
    _storage.ptr    = bytes;
    _payloadLength  = static_cast<std::uint32_t>(v.size());
}

Value::Value(const std::string& v, std::pmr::memory_resource* resource) : Value(std::string_view(v), resource) {}

Value::Value(const std::pmr::string& v, std::pmr::memory_resource* resource) : Value(std::string_view(v), resource) {}

Value::Value(const char* v, std::pmr::memory_resource* resource) : Value(std::string_view(v), resource) {}

Value::Value(Map map, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::Value, ContainerType::Map);
    // Q1.B: store the source ValueMap's blob bytes directly. If source is empty (no header), leave
    // _storage at default (nullptr / 0).
    const auto srcBlob = map.blob();
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

// ═══════════════════════════════════════════════════════════════════════════════
// COPY/MOVE CONSTRUCTORS AND ASSIGNMENT
// ═══════════════════════════════════════════════════════════════════════════════

Value::Value(Value&& other) noexcept : _value_type(other._value_type), _container_type(other._container_type), _flags(other._flags), _payloadLength(other._payloadLength), _storage(other._storage), _resource(other._resource) {
    other.set_types(ValueType::Monostate, ContainerType::Scalar);
    other._flags         = 0U;
    other._payloadLength = 0U;
    other._storage.u64   = 0UZ;
    other._resource      = nullptr;
}

Value::~Value() { destroy(); }

Value& Value::operator=(const Value& other) {
    if (this == &other) {
        return *this;
    }

    Value tmp(other, _resource);
    swap(*this, tmp);

    return *this;
}

Value& Value::operator=(Value&& other) {
    if (this == &other) {
        return *this;
    }
    if (_resource == other._resource) {
        destroy();
        _value_type     = other._value_type;
        _container_type = other._container_type;
        _flags          = other._flags;
        _payloadLength  = other._payloadLength;
        _storage        = other._storage;
        other.set_types(ValueType::Monostate, ContainerType::Scalar);
        other._flags         = 0U;
        other._payloadLength = 0U;
        other._storage.u64   = 0UZ;
        other._resource      = nullptr;
    } else {
        // Cross-resource: preserve target's resource per std::pmr::vector semantics
        // (propagate_on_container_move_assignment = false_type). Materialises view-mode source.
        Value tmp(other, _resource);
        swap(*this, tmp);
        other.destroy();
        other.set_types(ValueType::Monostate, ContainerType::Scalar);
        other._flags         = 0U;
        other._payloadLength = 0U;
        other._storage.u64   = 0UZ;
        other._resource      = nullptr;
    }
    return *this;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCALAR ASSIGNMENT OPERATORS
// ═══════════════════════════════════════════════════════════════════════════════

Value& Value::operator=(bool v) {
    destroy();
    set_types(ValueType::Bool, ContainerType::Scalar);
    _storage.b = v;
    return *this;
}

Value& Value::operator=(int8_t v) {
    destroy();
    set_types(ValueType::Int8, ContainerType::Scalar);
    _storage.i8 = v;
    return *this;
}

Value& Value::operator=(int16_t v) {
    destroy();
    set_types(ValueType::Int16, ContainerType::Scalar);
    _storage.i16 = v;
    return *this;
}

Value& Value::operator=(int32_t v) {
    destroy();
    set_types(ValueType::Int32, ContainerType::Scalar);
    _storage.i32 = v;
    return *this;
}

Value& Value::operator=(int64_t v) {
    destroy();
    set_types(ValueType::Int64, ContainerType::Scalar);
    _storage.i64 = v;
    return *this;
}

Value& Value::operator=(uint8_t v) {
    destroy();
    set_types(ValueType::UInt8, ContainerType::Scalar);
    _storage.u8 = v;
    return *this;
}

Value& Value::operator=(uint16_t v) {
    destroy();
    set_types(ValueType::UInt16, ContainerType::Scalar);
    _storage.u16 = v;
    return *this;
}

Value& Value::operator=(uint32_t v) {
    destroy();
    set_types(ValueType::UInt32, ContainerType::Scalar);
    _storage.u32 = v;
    return *this;
}

Value& Value::operator=(uint64_t v) {
    destroy();
    set_types(ValueType::UInt64, ContainerType::Scalar);
    _storage.u64 = v;
    return *this;
}

Value& Value::operator=(float v) {
    destroy();
    set_types(ValueType::Float32, ContainerType::Scalar);
    _storage.f32 = v;
    return *this;
}

Value& Value::operator=(double v) {
    destroy();
    set_types(ValueType::Float64, ContainerType::Scalar);
    _storage.f64 = v;
    return *this;
}

Value& Value::operator=(std::complex<float> v) {
    destroy();
    set_types(ValueType::ComplexFloat32, ContainerType::Complex);
    void* mem    = _resource->allocate(sizeof(std::complex<float>), alignof(std::complex<float>));
    _storage.ptr = new (mem) std::complex<float>(v);
    return *this;
}

Value& Value::operator=(std::complex<double> v) {
    destroy();
    set_types(ValueType::ComplexFloat64, ContainerType::Complex);
    void* mem    = _resource->allocate(sizeof(std::complex<double>), alignof(std::complex<double>));
    _storage.ptr = new (mem) std::complex<double>(v);
    return *this;
}

Value& Value::operator=(std::string_view v) {
    destroy();
    set_types(ValueType::String, ContainerType::String);
    auto* bytes = static_cast<char*>(_resource->allocate(v.size() + 1U, 1U));
    std::memcpy(bytes, v.data(), v.size());
    bytes[v.size()] = '\0';
    _storage.ptr    = bytes;
    _payloadLength  = static_cast<std::uint32_t>(v.size());
    return *this;
}

Value& Value::operator=(const std::string& v) { return operator=(std::string_view(v)); }
Value& Value::operator=(const std::pmr::string& v) { return operator=(std::string_view(v)); }

Value& Value::operator=(const char* v) { return operator=(std::string_view(v)); }

Value& Value::operator=(Map map) {
    destroy();
    set_types(ValueType::Value, ContainerType::Map);
    void* mem    = _resource->allocate(sizeof(Map), alignof(Map));
    _storage.ptr = new (mem) Map(map, _resource);
    return *this;
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPARISON HELPER METHODS (reduces switch statement duplication)
// ═══════════════════════════════════════════════════════════════════════════════

bool Value::compare_scalar_eq(const Value& other) const noexcept {
#if defined(__cpp_assume) && __cpp_assume >= 202207L
    [[assume(_value_type == other._value_type && container_type() == ContainerType::Scalar)]];
#endif
    assert(_value_type == other._value_type && container_type() == ContainerType::Scalar);

    switch (value_type()) {
    case ValueType::Monostate: return true;
    case ValueType::Bool: return _storage.b == other._storage.b;
    case ValueType::Int8: return _storage.i8 == other._storage.i8;
    case ValueType::Int16: return _storage.i16 == other._storage.i16;
    case ValueType::Int32: return _storage.i32 == other._storage.i32;
    case ValueType::Int64: return _storage.i64 == other._storage.i64;
    case ValueType::UInt8: return _storage.u8 == other._storage.u8;
    case ValueType::UInt16: return _storage.u16 == other._storage.u16;
    case ValueType::UInt32: return _storage.u32 == other._storage.u32;
    case ValueType::UInt64: return _storage.u64 == other._storage.u64;
    case ValueType::Float32: return _storage.f32 == other._storage.f32;
    case ValueType::Float64: return _storage.f64 == other._storage.f64;
    default: return false;
    }
}

std::partial_ordering Value::compare_scalar_order(const Value& other) const noexcept {
#if defined(__cpp_assume) && __cpp_assume >= 202207L
    [[assume(_value_type == other._value_type && container_type() == ContainerType::Scalar)]];
#endif
    assert(_value_type == other._value_type && container_type() == ContainerType::Scalar);

    switch (value_type()) {
    case ValueType::Monostate: return std::partial_ordering::equivalent;
    case ValueType::Bool: return _storage.b <=> other._storage.b;
    case ValueType::Int8: return _storage.i8 <=> other._storage.i8;
    case ValueType::Int16: return _storage.i16 <=> other._storage.i16;
    case ValueType::Int32: return _storage.i32 <=> other._storage.i32;
    case ValueType::Int64: return _storage.i64 <=> other._storage.i64;
    case ValueType::UInt8: return _storage.u8 <=> other._storage.u8;
    case ValueType::UInt16: return _storage.u16 <=> other._storage.u16;
    case ValueType::UInt32: return _storage.u32 <=> other._storage.u32;
    case ValueType::UInt64: return _storage.u64 <=> other._storage.u64;
    case ValueType::Float32: return _storage.f32 <=> other._storage.f32;
    case ValueType::Float64: return _storage.f64 <=> other._storage.f64;
    default: return std::partial_ordering::unordered;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EQUALITY OPERATORS
// ═══════════════════════════════════════════════════════════════════════════════

bool Value::operator==(const Value& other) const {
    if (value_type() != other.value_type() || container_type() != other.container_type()) {
        return false;
    }

    switch (container_type()) {
    case ContainerType::Scalar: return compare_scalar_eq(other);

    case ContainerType::Complex:
        if (value_type() == ValueType::ComplexFloat32) {
            return *static_cast<const std::complex<float>*>(_storage.ptr) == *static_cast<const std::complex<float>*>(other._storage.ptr);
        } else {
            return *static_cast<const std::complex<double>*>(_storage.ptr) == *static_cast<const std::complex<double>*>(other._storage.ptr);
        }

    case ContainerType::String: return std::string_view{static_cast<const char*>(_storage.ptr), _payloadLength} == std::string_view{static_cast<const char*>(other._storage.ptr), other._payloadLength};

    case ContainerType::Map: {
        // Q1.B: Map storage is byte-blob; construct view-mode ValueMaps over both blobs and use
        // the order-independent equality operator.
        const auto lhsView = ValueMap::makeView(std::span<const std::byte>{static_cast<const std::byte*>(_storage.ptr), _payloadLength});
        const auto rhsView = ValueMap::makeView(std::span<const std::byte>{static_cast<const std::byte*>(other._storage.ptr), other._payloadLength});
        return lhsView == rhsView;
    }

    case ContainerType::Tensor:
        // Q1.B: byte-blob with deterministic encoding — bytewise compare.
        if (_payloadLength != other._payloadLength) {
            return false;
        }
        return _storage.ptr == other._storage.ptr || std::memcmp(_storage.ptr, other._storage.ptr, _payloadLength) == 0;

    default: return false;
    }
}

std::partial_ordering Value::operator<=>(const Value& other) const {
    using std::partial_ordering;

    // primary ordering: type tag (container+value) via _type_info
    if (value_type() != other.value_type() || container_type() != other.container_type()) {
        return static_cast<std::uint8_t>(_value_type) <=> static_cast<std::uint8_t>(other._value_type);
    }

    switch (container_type()) {
    case ContainerType::Scalar: return compare_scalar_order(other);

    case ContainerType::String: {
        const auto ls = std::string_view{static_cast<const char*>(_storage.ptr), _payloadLength};
        const auto rs = std::string_view{static_cast<const char*>(other._storage.ptr), other._payloadLength};
        return ls <=> rs;
    }

    case ContainerType::Complex:
    case ContainerType::Map:
    case ContainerType::Tensor: // no natural value ordering for these types
        return partial_ordering::unordered;
    }

    return partial_ordering::unordered; // should not reach
}

template<detail::ValueComparable T>
bool operator==(const Value& lhs, const T& rhs) {
    if (auto* lhsRef = lhs.get_if<std::remove_cvref_t<T>>()) {
        return *lhsRef == rhs;
    }
    return false;
}

template<detail::ValueComparable T>
bool operator==(const T& lhs, const Value& rhs) {
    if (auto* rhsRef = rhs.get_if<std::remove_cvref_t<T>>()) {
        return lhs == *rhsRef;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPLICIT TEMPLATE INSTANTIATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// clang-format off

// Tensor constructors and operator= (works for all element types including pmr::string and Value)
#define X(T)                                                                      \
    template Value::Value(Tensor<T> tensor, std::pmr::memory_resource* resource); \
    template Value&           Value::operator=(Tensor<T> tensor);
GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

// Direct scalar accessors (excludes std::pmr::string — get_if dropped Phase 1e)
#define X(T)                                                            \
    template bool     Value::holds<T>() const noexcept;                 \
    template T*       Value::get_if<T>() noexcept;                      \
    template const T* Value::get_if<T>() const noexcept;
GR_PMT_VALUE_SCALAR_TYPES
#undef X

// Tensor accessors (works for all element types including pmr::string and Value)
#define X(T)                                                                  \
    template bool             Value::holds<Tensor<T>>() const noexcept;       \
    /* get_if<Tensor<T>> moved to header (returns std::optional<Tensor<T>>; needs decodeTensorBlob) */
GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

// Comparison operators
#define X(T)                                              \
    template bool operator== <T>(const Value&, const T&); \
    template bool operator== <T>(const T&, const Value&);
GR_PMT_VALUE_SCALAR_TYPES
#undef X
// clang-format on

// platform-dependent aliasing: unsigned long may differ from uint64_t (e.g. Apple ARM64)
#if defined(__APPLE__) && defined(__aarch64__)
template bool                 Value::holds<unsigned long>() const noexcept;
template unsigned long*       Value::get_if<unsigned long>() noexcept;
template const unsigned long* Value::get_if<unsigned long>() const noexcept;
#endif

// type predicates (still meaningful even after get_if<…> dropped for these types)
template bool Value::holds<std::pmr::string>() const noexcept;
template bool Value::holds<std::string>() const noexcept;
template bool Value::holds<std::string_view>() const noexcept;

// Map accessors — get_if<ValueMap>() returns std::optional<ValueMap> (view-mode aliasing the
// source's bytes). Owning ValueMap pointer access is no longer exposed; callers must use the
// optional path (which works uniformly across owning + view modes; the inner ValueMap is in
// view-mode and lifetime-bound to *this).
template bool Value::holds<Value::Map>() const noexcept;

template<typename T>
requires std::same_as<T, ValueMap>
std::optional<ValueMap> Value::get_if() const noexcept {
    if (!is_map() || _storage.ptr == nullptr || _payloadLength == 0U) {
        return std::nullopt;
    }
    // Q1.B: Map storage is byte-blob (ValueMap blob bytes), uniform across owning + view modes.
    // Return view-mode ValueMap aliasing the same bytes; lifetime bound to *this.
    return ValueMap::makeView(std::span<const std::byte>{static_cast<const std::byte*>(_storage.ptr), _payloadLength});
}

template std::optional<ValueMap> Value::get_if<ValueMap>() const noexcept;

} // namespace gr::pmt

namespace std {

std::size_t hash<gr::pmt::Value>::hashScalar(const gr::pmt::Value& v) noexcept {
    if (auto* p = v.get_if<bool>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<std::int8_t>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<std::int16_t>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<std::int32_t>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<std::int64_t>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<std::uint8_t>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<std::uint16_t>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<std::uint32_t>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<std::uint64_t>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<float>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<double>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<std::complex<float>>()) {
        return hashValue(*p);
    }
    if (auto* p = v.get_if<std::complex<double>>()) {
        return hashValue(*p);
    }
    if (auto sv = v.get_if<std::string_view>()) { // view-aware: works for both owning + view-mode strings
        return std::hash<std::string_view>{}(*sv);
    }
    return 0;
}

std::size_t hash<gr::pmt::Value>::hashTensor(const gr::pmt::Value& v) noexcept {
    if (auto p = v.get_if<gr::Tensor<std::int8_t>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<std::int16_t>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<std::int32_t>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<std::int64_t>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<std::uint8_t>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<std::uint16_t>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<std::uint32_t>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<std::uint64_t>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<float>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<double>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<std::complex<float>>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<std::complex<double>>>()) {
        return hashTensorElements(*p);
    }
    if (auto p = v.get_if<gr::Tensor<gr::pmt::Value>>()) {
        return hashTensorElements(*p);
    }
    return 0;
}

std::size_t hash<gr::pmt::Value>::hashMap(const gr::pmt::Value& v) noexcept {
    std::size_t accumulator = 0UZ; // order-independent: accumulate with addition

    auto hashMapImpl = [&accumulator](const auto& map) {
        for (const auto& [key, val] : map) {
            std::size_t entryHash = hashCombine(std::hash<std::string_view>{}(std::string_view{key}), std::hash<gr::pmt::Value>{}(val));
            accumulator += entryHash;
        }
    };

    if (auto p = v.get_if<gr::pmt::Value::Map>()) {
        hashMapImpl(*p);
        return accumulator;
    }
    return accumulator;
}

std::size_t hash<gr::pmt::Value>::operator()(const gr::pmt::Value& v) const noexcept {
    using namespace gr::pmt;

    std::size_t seed = std::hash<int>{}(static_cast<int>(v.container_type()));
    seed             = hashCombine(seed, std::hash<int>{}(static_cast<int>(v.value_type())));

    if (v.is_monostate()) {
        return seed;
    }

    using enum Value::ContainerType;
    switch (v.container_type()) {
    case Scalar: return hashCombine(seed, hashScalar(v));
    case Complex: return hashCombine(seed, hashScalar(v)); // hashScalar handles complex types
    case String: return hashCombine(seed, hashScalar(v));  // hashScalar handles pmr::string
    case Tensor: return hashCombine(seed, hashTensor(v));
    case Map: return hashCombine(seed, hashMap(v));
    default: return seed;
    }
}

} // namespace std
