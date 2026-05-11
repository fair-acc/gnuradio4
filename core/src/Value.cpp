#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/ValueMap.hpp> // ValueMap is forward-declared in Value.hpp; full def needed here for Map operations

#include <compare>
#include <concepts>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>

namespace gr::pmt {

template<typename T>
requires(!meta::is_instantiation_of<T, std::vector>)
bool ValueView::holds() const noexcept {
    if constexpr (std::same_as<T, gr::pmt::ValueMap>) {
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
T* ValueView::get_if() noexcept {
    if (!holds<T>()) [[unlikely]] {
        return nullptr;
    }
    // T is fixed-size POD (variable-length is gated by the static_asserts below); view-mode
    // writes match std::span<T> semantics — they intentionally modify the aliased source.
    if constexpr (std::same_as<T, bool>) {
        return reinterpret_cast<bool*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, std::int8_t>) {
        return reinterpret_cast<std::int8_t*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, std::int16_t>) {
        return reinterpret_cast<std::int16_t*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, std::int32_t>) {
        return reinterpret_cast<std::int32_t*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, std::int64_t>) {
        return reinterpret_cast<std::int64_t*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, std::uint8_t>) {
        return reinterpret_cast<std::uint8_t*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, std::uint16_t>) {
        return reinterpret_cast<std::uint16_t*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, std::uint32_t>) {
        return reinterpret_cast<std::uint32_t*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, std::uint64_t>) {
        return reinterpret_cast<std::uint64_t*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, float>) {
        return reinterpret_cast<float*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, double>) {
        return reinterpret_cast<double*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, std::complex<float>> || std::same_as<T, std::complex<double>>) {
        return reinterpret_cast<T*>(recPayloadMutable());
    } else if constexpr (std::same_as<T, std::string>) {
        static_assert(gr::meta::always_false<T>, "Use value_or<std::string>() for owning copy, or get_if<std::string_view>() for alloc-free view");
    } else {
        static_assert(gr::meta::always_false<T>, "Unsupported type for get_if<T>()");
    }
}

} // namespace gr::pmt

namespace gr::pmt {

// Shared read-only Monostate sentinel record (stable address; lazily initialised).
const std::byte* monostateRecord() noexcept {
    alignas(16) static const std::array<std::byte, gr::pmt::kRecMinSize> record = [] {
        std::array<std::byte, gr::pmt::kRecMinSize> a{};
        gr::wire::writeHeaderSized(a.data(), gr::pmt::kRecMinSize, static_cast<std::uint8_t>(ValueType::Monostate), static_cast<std::uint8_t>(ContainerType::Scalar));
        return a;
    }();
    return record.data();
}

void Value::copy_from(const ValueView& other) {
    // Allocate-and-copy the source record into a fresh allocation owned by *our* resource.
    // Caller (operator= / two-arg ctor) is responsible for invoking destroy() first if needed.
    if (_resource == nullptr) {
        // View target: alias the same bytes as source.
        _data = other._data;
        return;
    }
    const auto sz = other.recSize();
    if (sz < kRecMinSize) {
        // Source is the Monostate sentinel or invalid; reset to our own monostate sentinel view.
        _data = const_cast<std::byte*>(monostateRecord());
        return;
    }
    _data = static_cast<std::byte*>(_resource->allocate(sz, alignof(std::max_align_t)));
    std::memcpy(_data, other._data, sz);
}

void Value::destroy() noexcept {
    // keeps _resource so the emptied Value can re-allocate from the same pool; move-assign nulls it to abandon the source
    if (_resource == nullptr) {
        // View-mode: aliased bytes; not ours to free.
        _data = const_cast<std::byte*>(monostateRecord());
        return;
    }
    if (_data == monostateRecord()) {
        // Pointing at the read-only sentinel; nothing to free.
        return;
    }
    const auto sz = recSize();
    if (sz >= kRecMinSize) {
        _resource->deallocate(_data, sz, alignof(std::max_align_t));
    }
    _data = const_cast<std::byte*>(monostateRecord());
}

void swap(Value& a, Value& b) noexcept {
    std::swap(a._data, b._data);
    std::swap(a._resource, b._resource);
}

Value::Value(const Value& other, std::pmr::memory_resource* resource) {
    // No ensure_resource(): a null resource here means "preserve view-mode" (the discriminator).
    _resource = resource ? resource : other._resource;
    copy_from(other);
}

Value::Value(const ValueView& other, std::pmr::memory_resource* resource) {
    // ValueView has no _resource of its own; null target resource is normalised to default-PMR
    // (constructing an owning Value in view-mode is intentionally not exposed here — callers
    // wanting view-mode use Value::makeView).
    _resource = ensure_resource(resource);
    copy_from(other);
}

Value::Value(std::complex<double> v, std::pmr::memory_resource* resource) {
    _resource                       = ensure_resource(resource);
    const std::uint32_t naturalSize = static_cast<std::uint32_t>(kRecOffsetPayload) + static_cast<std::uint32_t>(sizeof(v));
    allocAndWriteHeader(std::max<std::uint32_t>(naturalSize, kRecMinSize), ValueType::ComplexFloat64, ContainerType::Complex);
    std::memcpy(_data + kRecOffsetPayload, &v, sizeof(v));
}

Value::Value(std::string_view v, std::pmr::memory_resource* resource) {
    _resource                       = ensure_resource(resource);
    const std::uint32_t contentLen  = static_cast<std::uint32_t>(v.size()) + 1U; // +1 for guard
    const std::uint32_t naturalSize = static_cast<std::uint32_t>(kRecOffsetPayload) + contentLen;
    const std::uint32_t recSize     = std::max<std::uint32_t>(naturalSize, kRecMinSize);
    allocAndWriteHeader(recSize, ValueType::String, ContainerType::String);
    std::memset(_data + kRecOffsetPayload, 0, recSize - kRecOffsetPayload);
    std::memcpy(_data + kRecOffsetPayload, v.data(), v.size());
    // _data[kRecOffsetPayload + v.size()] is implicitly '\0' from the memset.
}

Value::Value(const std::string& v, std::pmr::memory_resource* resource) : Value(std::string_view(v), resource) {}
Value::Value(const std::pmr::string& v, std::pmr::memory_resource* resource) : Value(std::string_view(v), resource) {}
Value::Value(const char* v, std::pmr::memory_resource* resource) : Value(std::string_view(v), resource) {}

Value::Value(ValueMap map, std::pmr::memory_resource* resource) {
    _resource          = ensure_resource(resource);
    const auto srcBlob = map.blob();
    if (srcBlob.empty()) {
        // Empty map: just emit a header pointing at no payload.
        allocAndWriteHeader(kRecMinSize, ValueType::Value, ContainerType::Map);
        std::memset(_data + kRecOffsetPayload, 0, kRecMinSize - static_cast<std::uint32_t>(kRecOffsetPayload));
        return;
    }
    const std::uint32_t recSize = static_cast<std::uint32_t>(kRecOffsetPayload + srcBlob.size());
    allocAndWriteHeader(recSize, ValueType::Value, ContainerType::Map);
    std::memcpy(_data + kRecOffsetPayload, srcBlob.data(), srcBlob.size());
}

Value::Value(Value&& other) noexcept : ValueView{std::exchange(other._data, const_cast<std::byte*>(monostateRecord()))}, _resource(std::exchange(other._resource, nullptr)) {}

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
    if (this == &other) {
        return *this;
    }
    if (_resource == other._resource) {
        // Same allocator (or both nullptr / view-mode): straight pointer transfer.
        destroy();
        _data           = other._data;
        _resource       = other._resource;
        other._data     = const_cast<std::byte*>(monostateRecord());
        other._resource = nullptr;
    } else {
        // Cross-resource: preserve target's resource (PMR convention).
        Value tmp(other, _resource);
        swap(*this, tmp);
        other.destroy();
        other._resource = nullptr;
    }
    return *this;
}

Value& Value::operator=(bool v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(int8_t v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(int16_t v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(int32_t v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(int64_t v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(uint8_t v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(uint16_t v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(uint32_t v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(uint64_t v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(float v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(double v) {
    *this = Value(v, _resource);
    return *this;
}

Value& Value::operator=(std::complex<float> v) {
    *this = Value(v, _resource);
    return *this;
}
Value& Value::operator=(std::complex<double> v) {
    *this = Value(v, _resource);
    return *this;
}
Value& Value::operator=(std::string_view v) {
    *this = Value(v, _resource);
    return *this;
}
Value& Value::operator=(const std::string& v) { return operator=(std::string_view(v)); }
Value& Value::operator=(const std::pmr::string& v) { return operator=(std::string_view(v)); }
Value& Value::operator=(const char* v) { return operator=(std::string_view(v)); }
Value& Value::operator=(ValueMap map) {
    *this = Value(std::move(map), _resource);
    return *this;
}

bool ValueView::compare_scalar_eq(const ValueView& other) const noexcept {
#if defined(__cpp_assume) && __cpp_assume >= 202207L
    [[assume(recValueTypeByte() == other.recValueTypeByte() && container_type() == ContainerType::Scalar)]];
#endif
    assert(recValueTypeByte() == other.recValueTypeByte() && container_type() == ContainerType::Scalar);

    switch (value_type()) {
    case ValueType::Monostate: return true;
    case ValueType::Bool: return inlineAs<bool>() == other.inlineAs<bool>();
    case ValueType::Int8: return inlineAs<std::int8_t>() == other.inlineAs<std::int8_t>();
    case ValueType::Int16: return inlineAs<std::int16_t>() == other.inlineAs<std::int16_t>();
    case ValueType::Int32: return inlineAs<std::int32_t>() == other.inlineAs<std::int32_t>();
    case ValueType::Int64: return inlineAs<std::int64_t>() == other.inlineAs<std::int64_t>();
    case ValueType::UInt8: return inlineAs<std::uint8_t>() == other.inlineAs<std::uint8_t>();
    case ValueType::UInt16: return inlineAs<std::uint16_t>() == other.inlineAs<std::uint16_t>();
    case ValueType::UInt32: return inlineAs<std::uint32_t>() == other.inlineAs<std::uint32_t>();
    case ValueType::UInt64: return inlineAs<std::uint64_t>() == other.inlineAs<std::uint64_t>();
    case ValueType::Float32: return inlineAs<float>() == other.inlineAs<float>();
    case ValueType::Float64: return inlineAs<double>() == other.inlineAs<double>();
    default: return false;
    }
}

std::partial_ordering ValueView::compare_scalar_order(const ValueView& other) const noexcept {
#if defined(__cpp_assume) && __cpp_assume >= 202207L
    [[assume(recValueTypeByte() == other.recValueTypeByte() && container_type() == ContainerType::Scalar)]];
#endif
    assert(recValueTypeByte() == other.recValueTypeByte() && container_type() == ContainerType::Scalar);

    switch (value_type()) {
    case ValueType::Monostate: return std::partial_ordering::equivalent;
    case ValueType::Bool: return inlineAs<bool>() <=> other.inlineAs<bool>();
    case ValueType::Int8: return inlineAs<std::int8_t>() <=> other.inlineAs<std::int8_t>();
    case ValueType::Int16: return inlineAs<std::int16_t>() <=> other.inlineAs<std::int16_t>();
    case ValueType::Int32: return inlineAs<std::int32_t>() <=> other.inlineAs<std::int32_t>();
    case ValueType::Int64: return inlineAs<std::int64_t>() <=> other.inlineAs<std::int64_t>();
    case ValueType::UInt8: return inlineAs<std::uint8_t>() <=> other.inlineAs<std::uint8_t>();
    case ValueType::UInt16: return inlineAs<std::uint16_t>() <=> other.inlineAs<std::uint16_t>();
    case ValueType::UInt32: return inlineAs<std::uint32_t>() <=> other.inlineAs<std::uint32_t>();
    case ValueType::UInt64: return inlineAs<std::uint64_t>() <=> other.inlineAs<std::uint64_t>();
    case ValueType::Float32: return inlineAs<float>() <=> other.inlineAs<float>();
    case ValueType::Float64: return inlineAs<double>() <=> other.inlineAs<double>();
    default: return std::partial_ordering::unordered;
    }
}

bool operator==(const ValueView& lhs, const ValueView& rhs) {
    if (lhs.value_type() != rhs.value_type() || lhs.container_type() != rhs.container_type()) {
        return false;
    }

    switch (lhs.container_type()) {
    case Value::ContainerType::Scalar: return lhs.compare_scalar_eq(rhs);
    case Value::ContainerType::Complex:
        if (lhs.value_type() == Value::ValueType::ComplexFloat32) {
            return *lhs.payloadAs<std::complex<float>>() == *rhs.payloadAs<std::complex<float>>();
        } else {
            return *lhs.payloadAs<std::complex<double>>() == *rhs.payloadAs<std::complex<double>>();
        }
    case Value::ContainerType::String: return std::string_view{lhs.payloadAs<char>(), lhs.strLen()} == std::string_view{rhs.payloadAs<char>(), rhs.strLen()};
    case Value::ContainerType::Map: {
        const auto lhsView = ValueMap::makeView(std::span<const std::byte>{lhs.payloadAs<std::byte>(), lhs.payloadByteCount()});
        const auto rhsView = ValueMap::makeView(std::span<const std::byte>{rhs.payloadAs<std::byte>(), rhs.payloadByteCount()});
        return lhsView == rhsView;
    }
    case Value::ContainerType::Tensor:
        if (lhs.payloadByteCount() != rhs.payloadByteCount()) {
            return false;
        }
        return lhs.payloadAs<std::byte>() == rhs.payloadAs<std::byte>() || std::memcmp(lhs.payloadAs<std::byte>(), rhs.payloadAs<std::byte>(), lhs.payloadByteCount()) == 0;
    default: return false;
    }
}

bool Value::operator==(const Value& other) const { return static_cast<const ValueView&>(*this) == static_cast<const ValueView&>(other); }

std::partial_ordering Value::operator<=>(const Value& other) const {
    using std::partial_ordering;

    // primary ordering: type tag (container+value) via _type_info
    if (value_type() != other.value_type() || container_type() != other.container_type()) {
        return recValueTypeByte() <=> other.recValueTypeByte();
    }

    switch (container_type()) {
    case ContainerType::Scalar: return compare_scalar_order(other);

    case ContainerType::String: {
        const auto ls = std::string_view{payloadAs<char>(), strLen()};
        const auto rs = std::string_view{other.payloadAs<char>(), other.strLen()};
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
bool operator==(const ValueView& lhs, const T& rhs) {
    if (auto* lhsRef = lhs.get_if<std::remove_cvref_t<T>>()) {
        return *lhsRef == rhs;
    }
    return false;
}

template<detail::ValueComparable T>
bool operator==(const T& lhs, const ValueView& rhs) {
    if (auto* rhsRef = rhs.get_if<std::remove_cvref_t<T>>()) {
        return lhs == *rhsRef;
    }
    return false;
}

// clang-format off

// Tensor constructors and operator= (works for all element types including pmr::string and Value).
// init_from_tensor<Tensor<T>> is the inline body these forward to (defined in ValueMap.hpp).
// GCC instantiates it transitively from operator='s explicit instantiation; Clang does not — the
// standard does not require explicit instantiation to drag inline callees in. List it directly.
#define X(T)                                                                                  \
    template Value::Value(Tensor<T> tensor, std::pmr::memory_resource* resource);             \
    template Value& Value::operator=(Tensor<T> tensor);                                       \
    template void   Value::init_from_tensor<Tensor<T>>(Tensor<T>&& tensor);
GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

// Direct scalar accessors (excludes std::pmr::string — String storage is a byte-blob, see Value.hpp)
#define X(T)                                                                \
    template bool     ValueView::holds<T>() const noexcept;                 \
    template T*       ValueView::get_if<T>() noexcept;                      \
    template const T* ValueView::get_if<T>() const noexcept;
GR_PMT_VALUE_SCALAR_TYPES
#undef X

// Tensor accessors (works for all element types including pmr::string and Value)
#define X(T)                                                                  \
    template bool             ValueView::holds<Tensor<T>>() const noexcept;
GR_PMT_VALUE_TENSOR_ELEMENT_TYPES
#undef X

// Comparison operators
#define X(T)                                              \
    template bool operator== <T>(const ValueView&, const T&); \
    template bool operator== <T>(const T&, const ValueView&);
GR_PMT_VALUE_SCALAR_TYPES
#undef X
// clang-format on

// type predicates (still meaningful even after get_if<…> dropped for these types)
template bool ValueView::holds<std::pmr::string>() const noexcept;
template bool ValueView::holds<std::string>() const noexcept;
template bool ValueView::holds<std::string_view>() const noexcept;

// get_if<ValueMap>() returns optional<ValueMap> (view-mode); owning Map pointer access not exposed.
template bool ValueView::holds<gr::pmt::ValueMap>() const noexcept;

} // namespace gr::pmt

namespace std {

std::size_t hash<gr::pmt::ValueView>::hashScalar(const gr::pmt::ValueView& v) noexcept {
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

std::size_t hash<gr::pmt::ValueView>::hashTensor(const gr::pmt::ValueView& v) noexcept {
    // Hash via TensorView (zero-copy alias for fixed-size T) — hashTensorElements iterates
    // element-by-element. Tensor<Value> needs sub-Value decode, which requires Value::_resource;
    // materialise into a transient owning Value before dispatching.
    if (std::optional<gr::TensorView<bool>> view = v.get_if<gr::TensorView<bool>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<std::int8_t>> view = v.get_if<gr::TensorView<std::int8_t>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<std::int16_t>> view = v.get_if<gr::TensorView<std::int16_t>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<std::int32_t>> view = v.get_if<gr::TensorView<std::int32_t>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<std::int64_t>> view = v.get_if<gr::TensorView<std::int64_t>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<std::uint8_t>> view = v.get_if<gr::TensorView<std::uint8_t>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<std::uint16_t>> view = v.get_if<gr::TensorView<std::uint16_t>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<std::uint32_t>> view = v.get_if<gr::TensorView<std::uint32_t>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<std::uint64_t>> view = v.get_if<gr::TensorView<std::uint64_t>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<float>> view = v.get_if<gr::TensorView<float>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<double>> view = v.get_if<gr::TensorView<double>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<std::complex<float>>> view = v.get_if<gr::TensorView<std::complex<float>>>()) {
        return hashTensorElements(*view);
    }
    if (std::optional<gr::TensorView<std::complex<double>>> view = v.get_if<gr::TensorView<std::complex<double>>>()) {
        return hashTensorElements(*view);
    }
    if (v.holds<gr::Tensor<gr::pmt::Value>>()) {
        const gr::pmt::Value materialised = v.owned();
        if (auto view = materialised.get_if<gr::TensorView<gr::pmt::Value>>()) {
            return hashTensorElements(*view);
        }
    }
    return 0;
}

std::size_t hash<gr::pmt::ValueView>::hashMap(const gr::pmt::ValueView& v) noexcept {
    std::size_t accumulator = 0UZ; // order-independent: accumulate with addition

    auto hashMapImpl = [&accumulator](const auto& map) {
        for (const auto& [key, val] : map) {
            std::size_t entryHash = hashCombine(std::hash<std::string_view>{}(std::string_view{key}), std::hash<gr::pmt::ValueView>{}(val));
            accumulator += entryHash;
        }
    };

    if (auto p = v.get_if<gr::pmt::ValueMap>()) {
        hashMapImpl(*p);
        return accumulator;
    }
    return accumulator;
}

std::size_t hash<gr::pmt::ValueView>::operator()(const gr::pmt::ValueView& v) const noexcept {
    using namespace gr::pmt;

    std::size_t seed = std::hash<int>{}(static_cast<int>(v.container_type()));
    seed             = hashCombine(seed, std::hash<int>{}(static_cast<int>(v.value_type())));

    if (v.is_monostate()) {
        return seed;
    }

    using enum ContainerType;
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
