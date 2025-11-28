#include <gnuradio-4.0/Value.hpp>

#include <compare>
#include <concepts>
#include <sstream>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>

// template definitions
namespace gr::pmt {

// ═══════════════════════════════════════════════════════════════════════════════
// TYPE TRAIT HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

template<typename T>
constexpr Value::ValueType Value::get_value_type() {
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
    } else if constexpr (std::same_as<T, std::uint64_t>) {
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
constexpr Value::ContainerType Value::get_container_type() {
    if constexpr (std::same_as<T, std::complex<float>> || std::same_as<T, std::complex<double>>) {
        return ContainerType::Complex;
    } else if constexpr (std::same_as<T, std::pmr::string>) {
        return ContainerType::String;
    } else {
        return ContainerType::Scalar;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TYPE CHECKING
// ═══════════════════════════════════════════════════════════════════════════════

template<typename T>
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
    } else if constexpr (std::same_as<T, float>) {
        return &_storage.f32;
    } else if constexpr (std::same_as<T, double>) {
        return &_storage.f64;
    } else if constexpr (std::same_as<T, std::complex<float>> || std::same_as<T, std::complex<double>>) {
        return static_cast<T*>(_storage.ptr);
    } else if constexpr (std::same_as<T, std::pmr::string>) {
        return static_cast<std::pmr::string*>(_storage.ptr);
    } else if constexpr (std::same_as<T, std::string_view> || std::same_as<T, std::string>) {
        static_assert(gr::meta::always_false<T>, "Use value_or<std::string>() or value_or<std::string_view>() for string conversions");
    } else if constexpr (gr::TensorLike<T>) {
        return static_cast<T*>(_storage.ptr);
    } else if constexpr (std::same_as<T, Map>) {
        return static_cast<Map*>(_storage.ptr);
    } else if constexpr (std::same_as<T, Value>) {
        return this;
    } else {
        static_assert(gr::meta::always_false<T>, "Unsupported type for get_if<T>()");
    }
}

} // namespace gr::pmt

namespace gr::pmt {

void Value::set_types(ValueType vt, ContainerType ct) noexcept {
    _value_type     = static_cast<unsigned char>(static_cast<std::uint8_t>(vt) & 0x0F);
    _container_type = static_cast<unsigned char>(static_cast<std::uint8_t>(ct) & 0x0F);
}

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
        auto* src    = static_cast<const std::pmr::string*>(other._storage.ptr);
        auto* mem    = static_cast<std::pmr::string*>(_resource->allocate(sizeof(std::pmr::string), alignof(std::pmr::string)));
        auto* dst    = std::construct_at(mem, *src, _resource); // copy + allocator
        _storage.ptr = dst;
        break;
    }

    case ContainerType::Tensor: {
        // Dispatch based on value_type
        switch (other.value_type()) {
#define X(T)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   \
    case get_value_type<T>(): {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        auto* src    = static_cast<const Tensor<T>*>(other._storage.ptr);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
        auto* mem    = static_cast<Tensor<T>*>(_resource->allocate(sizeof(Tensor<T>), alignof(Tensor<T>)));                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    \
        auto* dst    = std::construct_at(mem, *src);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           \
        _storage.ptr = dst;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    \
        break;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
    }
            GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

        default: break;
        }
        break;
    }

    case ContainerType::Map: {
        auto* src    = static_cast<const Map*>(other._storage.ptr);
        auto* mem    = static_cast<Map*>(_resource->allocate(sizeof(Map), alignof(Map)));
        auto* dst    = std::construct_at(mem, *src, _resource); // copy + allocator
        _storage.ptr = dst;
        break;
    }
    }
}

void Value::destroy() noexcept {
    assert(_resource != nullptr);
#if defined(__cpp_assume) && __cpp_assume >= 202207L
    [[assume(_resource != nullptr)]];
#endif
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
        auto* str = static_cast<std::pmr::string*>(_storage.ptr);
        std::destroy_at(str);
        _resource->deallocate(str, sizeof(std::pmr::string), alignof(std::pmr::string));
        break;
    }

    case ContainerType::Tensor: {
        // dispatch based on value_type to call correct destructor
        switch (value_type()) {
#define X(T)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   \
    case get_value_type<T>(): {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        auto* tensor = static_cast<Tensor<T>*>(_storage.ptr);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  \
        std::destroy_at(tensor);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
        _resource->deallocate(tensor, sizeof(Tensor<T>), alignof(Tensor<T>));                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  \
        break;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
    }
            GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

        default: break;
        }
        break;
    }

    case ContainerType::Map: {
        auto* map = static_cast<Map*>(_storage.ptr);
        std::destroy_at(map);
        _resource->deallocate(map, sizeof(Map), alignof(Map));
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
    std::swap(a._storage, b._storage);
    std::swap(a._resource, b._resource);
}

Value::Value(std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) { set_types(ValueType::Monostate, ContainerType::Scalar); }

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
    void* mem    = _resource->allocate(sizeof(std::pmr::string), alignof(std::pmr::string));
    _storage.ptr = new (mem) std::pmr::string(v, _resource);
}

Value::Value(const std::string& v, std::pmr::memory_resource* resource) : Value(std::string_view(v), resource) {}

Value::Value(const char* v, std::pmr::memory_resource* resource) : Value(std::string_view(v), resource) {}

Value::Value(Map map, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(ValueType::Value, ContainerType::Map);
    void* mem    = _resource->allocate(sizeof(Map), alignof(Map));
    _storage.ptr = new (mem) Map(std::move(map), _resource);
}

// ═══════════════════════════════════════════════════════════════════════════════
// COPY/MOVE CONSTRUCTORS AND ASSIGNMENT
// ═══════════════════════════════════════════════════════════════════════════════

Value::Value(Value&& other) noexcept : _value_type(other._value_type), _container_type(other._container_type), _storage(other._storage), _resource(other._resource) {
    other.set_types(ValueType::Monostate, ContainerType::Scalar);
    other._storage.u64 = 0UZ;
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

Value& Value::operator=(Value&& other) noexcept {
    if (this != &other) {
        destroy();
        _value_type     = other._value_type;
        _container_type = other._container_type;
        _storage        = other._storage;
        _resource       = other._resource;
        other.set_types(ValueType::Monostate, ContainerType::Scalar);
        other._storage.u64 = 0UZ;
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
    void* mem    = _resource->allocate(sizeof(std::pmr::string), alignof(std::pmr::string));
    _storage.ptr = new (mem) std::pmr::string(v, _resource);
    return *this;
}

Value& Value::operator=(const std::string& v) { return operator=(std::string_view(v)); }

Value& Value::operator=(const char* v) { return operator=(std::string_view(v)); }

Value& Value::operator=(Map map) {
    destroy();
    set_types(ValueType::Value, ContainerType::Map);
    void* mem    = _resource->allocate(sizeof(Map), alignof(Map));
    _storage.ptr = new (mem) Map(std::move(map), _resource);
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

    case ContainerType::String: return *static_cast<const std::pmr::string*>(_storage.ptr) == *static_cast<const std::pmr::string*>(other._storage.ptr);

    case ContainerType::Map: return *static_cast<const Map*>(_storage.ptr) == *static_cast<const Map*>(other._storage.ptr);

    case ContainerType::Tensor:
        // Tensor equality would require type dispatch and element-wise comparison
        // For now, just compare pointers (identity check)
        // TODO: Implement proper element-wise comparison via Tensor::operator==
        return _storage.ptr == other._storage.ptr;

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
        const auto& ls = *static_cast<const std::pmr::string*>(_storage.ptr);
        const auto& rs = *static_cast<const std::pmr::string*>(other._storage.ptr);
        return std::string_view{ls} <=> std::string_view{rs};
    }

    case ContainerType::Complex:
    case ContainerType::Map:
    case ContainerType::Tensor: // no natural value ordering for these types
        return partial_ordering::unordered;
    }

    return partial_ordering::unordered; // should not reach
}

template<typename T>
bool operator==(const Value& lhs, const T& rhs) {
    if (auto* lhsRef = lhs.get_if<std::remove_cvref_t<T>>()) {
        return *lhsRef == rhs;
    }
    return false;
}

template<typename T>
bool operator==(const T& lhs, const Value& rhs) {
    if (auto* rhsRef = rhs.get_if<std::remove_cvref_t<T>>()) {
        return lhs == *rhsRef;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TENSOR TEMPLATE IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

template<typename T>
Value::Value(Tensor<T> tensor, std::pmr::memory_resource* resource) : _resource(ensure_resource(resource)) {
    set_types(get_value_type<T>(), ContainerType::Tensor);
    void* mem    = _resource->allocate(sizeof(Tensor<T>), alignof(Tensor<T>));
    _storage.ptr = new (mem) Tensor<T>(std::move(tensor));
}

template<typename T>
Value& Value::operator=(Tensor<T> tensor) {
    destroy();
    set_types(get_value_type<T>(), ContainerType::Tensor);
    void* mem    = _resource->allocate(sizeof(Tensor<T>), alignof(Tensor<T>));
    _storage.ptr = new (mem) Tensor<T>(std::move(tensor));
    return *this;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPLICIT TEMPLATE INSTANTIATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// clang-format off

// Tensor constructors and accessors
#define X(T)                                                                      \
    template Value::Value(Tensor<T> tensor, std::pmr::memory_resource* resource); \
    template Value&           Value::operator=(Tensor<T> tensor);

GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

// scalar accessors for all types
#define X(T)                                                            \
    template bool     Value::holds<T>() const noexcept;                 \
    template T*       Value::get_if<T>() noexcept;                      \
    template const T* Value::get_if<T>() const noexcept;                \
    template bool             Value::holds<Tensor<T>>() const noexcept; \
    template Tensor<T>*       Value::get_if<Tensor<T>>() noexcept;      \
    template const Tensor<T>* Value::get_if<Tensor<T>>() const noexcept;

GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

// Comparison operators
#define X(T)                                              \
    template bool operator== <T>(const Value&, const T&); \
    template bool operator== <T>(const T&, const Value&);
GR_PMT_VALUE_SCALAR_TYPES
#undef X
// clang-format on

// string type specializations (convertible from pmr::string)
template bool              Value::holds<Value::Map>() const noexcept;
template bool              Value::holds<std::string>() const noexcept;
template bool              Value::holds<std::string_view>() const noexcept;
template Value::Map*       Value::get_if<Value::Map>() noexcept;
template const Value::Map* Value::get_if<Value::Map>() const noexcept;

} // namespace gr::pmt
