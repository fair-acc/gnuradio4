#ifndef GNURADIO_VALUEHELPER_HPP
#define GNURADIO_VALUEHELPER_HPP

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <array>
#include <concepts>
#include <expected>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace gr::pmt {

enum class ConversionPolicy {
    Safe,      // T→T only
    Widening,  // Safe + int→larger_int, float→double, etc.
    Narrowing, // Widening + double→float, int64→int32, etc.
    Unchecked  // all static_cast conversions
};

enum class RankPolicy {
    Strict,  // rank and extents must match exactly
    Flatten, // any rank → rank-1 (linearize)
    Reshape  // size-preserving reshape (rank/extents may differ)
};

enum class ResourcePolicy {
    UseDefault,       // use provided/default PMR resource
    InheritFromSource // use source Value's PMR resource
};

struct ConversionError {
    enum class Kind : std::uint8_t { None = 0U, TypeMismatch, ElementTypeMismatch, RankMismatch, ExtentsMismatch, SizeMismatch, NotATensor, NotAMap, EmptySourceNotAllowed, NarrowingNotAllowed, WideningNotAllowed };

    Kind             kind = Kind::None;
    std::string_view message{};
    std::size_t      index = std::dynamic_extent;

    [[nodiscard]] constexpr bool hasIndex() const noexcept { return index != std::dynamic_extent; }
    bool                         operator==(const ConversionError&) const noexcept = default;
};

inline bool operator==(ConversionError::Kind lhs, ConversionError::Kind rhs) noexcept { return static_cast<std::uint8_t>(lhs) == static_cast<std::uint8_t>(rhs); }

namespace detail { // type traits

template<typename T>
concept ValidTarget = gr::meta::vector_type<T> || gr::meta::array_type<T> || TensorLike<T> || gr::meta::map_type<T>;

template<typename T>
inline constexpr bool supports_pmr_v = false;
template<typename E>
inline constexpr bool supports_pmr_v<gr::Tensor<E>> = true;
template<typename E, std::size_t... Ex>
requires((Ex == std::dynamic_extent) && ...)
inline constexpr bool supports_pmr_v<gr::Tensor<E, Ex...>> = true;

template<typename T>
inline constexpr bool is_fixed_size_v = false;
template<typename E, std::size_t N>
inline constexpr bool is_fixed_size_v<std::array<E, N>> = true;
template<typename E, std::size_t... Ex>
requires(gr::detail::all_static_v<Ex...>)
inline constexpr bool is_fixed_size_v<gr::Tensor<E, Ex...>> = true;

template<typename T>
inline constexpr std::size_t static_size_v = 0;
template<typename E, std::size_t N>
inline constexpr std::size_t static_size_v<std::array<E, N>> = N;
template<typename E, std::size_t... Ex>
requires(gr::detail::all_static_v<Ex...>)
inline constexpr std::size_t static_size_v<gr::Tensor<E, Ex...>> = (Ex * ... * 1UZ);

template<typename From, typename To, typename = void>
inline constexpr bool is_static_castable_v = false;
template<typename From, typename To>
inline constexpr bool is_static_castable_v<From, To, std::void_t<decltype(static_cast<To>(std::declval<From>()))>> = true;

template<typename From, typename To>
inline constexpr bool is_same_type_v = std::same_as<std::remove_cv_t<From>, std::remove_cv_t<To>>;

template<typename From, typename To>
inline constexpr bool is_widening_v = (std::integral<From> && std::integral<To> && !std::same_as<From, bool> && !std::same_as<To, bool> && sizeof(From) < sizeof(To) && std::is_signed_v<From> == std::is_signed_v<To>) || (std::same_as<From, float> && std::same_as<To, double>) || (std::same_as<From, std::complex<float>> && std::same_as<To, std::complex<double>>) || (std::integral<From> && !std::same_as<From, bool> && std::floating_point<To> && sizeof(From) <= sizeof(To));

template<typename From, typename To>
inline constexpr bool is_narrowing_v = !is_same_type_v<From, To> && !is_widening_v<From, To> && is_static_castable_v<From, To>;

template<typename From, typename To, ConversionPolicy P>
inline constexpr bool conversion_allowed_v = is_same_type_v<From, To> || (P >= ConversionPolicy::Widening && is_widening_v<From, To>) || (P >= ConversionPolicy::Narrowing && is_narrowing_v<From, To>) || (P == ConversionPolicy::Unchecked && is_static_castable_v<From, To>);

template<typename F, typename T>
concept FallbackFactory = std::invocable<F> && std::convertible_to<std::invoke_result_t<F>, T>;

template<typename F, typename T>
concept FallbackValue = std::convertible_to<F, T> && !FallbackFactory<F, T>;

// value type mapping
template<typename T>
constexpr Value::ValueType valueTypeFor() noexcept {
    if constexpr (std::same_as<T, bool>) {
        return Value::ValueType::Bool;
    } else if constexpr (std::same_as<T, std::int8_t>) {
        return Value::ValueType::Int8;
    } else if constexpr (std::same_as<T, std::int16_t>) {
        return Value::ValueType::Int16;
    } else if constexpr (std::same_as<T, std::int32_t>) {
        return Value::ValueType::Int32;
    } else if constexpr (std::same_as<T, std::int64_t>) {
        return Value::ValueType::Int64;
    } else if constexpr (std::same_as<T, std::uint8_t>) {
        return Value::ValueType::UInt8;
    } else if constexpr (std::same_as<T, std::uint16_t>) {
        return Value::ValueType::UInt16;
    } else if constexpr (std::same_as<T, std::uint32_t>) {
        return Value::ValueType::UInt32;
    } else if constexpr (std::same_as<T, std::uint64_t>) {
        return Value::ValueType::UInt64;
    } else if constexpr (std::same_as<T, float>) {
        return Value::ValueType::Float32;
    } else if constexpr (std::same_as<T, double>) {
        return Value::ValueType::Float64;
    } else if constexpr (std::same_as<T, std::complex<float>>) {
        return Value::ValueType::ComplexFloat32;
    } else if constexpr (std::same_as<T, std::complex<double>>) {
        return Value::ValueType::ComplexFloat64;
    } else if constexpr (std::same_as<T, Value>) {
        return Value::ValueType::Value;
    } else {
        return Value::ValueType::Monostate;
    }
}

// Tensor<T> → vector

template<typename DstT, typename SrcT, ConversionPolicy CP, RankPolicy RP>
std::expected<std::vector<DstT>, ConversionError> tensorToVector(const Tensor<SrcT>& src) {
    if constexpr (!conversion_allowed_v<SrcT, DstT, CP>) {
        if constexpr (is_widening_v<SrcT, DstT>) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::WideningNotAllowed});
        } else if constexpr (is_static_castable_v<SrcT, DstT>) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::NarrowingNotAllowed});
        } else {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});
        }
    } else {
        if constexpr (RP == RankPolicy::Strict) {
            if (src.rank() != 1) {
                return std::unexpected(ConversionError{.kind = ConversionError::Kind::RankMismatch, .message = "strict requires rank=1 for vector"});
            }
        }
        std::vector<DstT> result;
        result.reserve(src.size());
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif
        for (const SrcT& elem : src) {
            result.emplace_back(static_cast<DstT>(elem));
        }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
        return result;
    }
}

// Tensor<T> → array

template<typename DstT, std::size_t N, typename SrcT, ConversionPolicy CP, RankPolicy RP>
std::expected<std::array<DstT, N>, ConversionError> tensorToArray(TensorOf<SrcT> auto const& src) {
    if constexpr (!conversion_allowed_v<SrcT, DstT, CP>) {
        if constexpr (is_widening_v<SrcT, DstT>) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::WideningNotAllowed});
        } else if constexpr (is_static_castable_v<SrcT, DstT>) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::NarrowingNotAllowed});
        } else {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});
        }
    } else {
        if constexpr (RP == RankPolicy::Strict) {
            if (src.rank() != 1) {
                return std::unexpected(ConversionError{.kind = ConversionError::Kind::RankMismatch, .message = "strict requires rank=1 for array"});
            }
        }
        if (src.size() != N) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::SizeMismatch});
        }
        std::array<DstT, N> result;
        auto                it = src.begin();
        for (std::size_t i = 0; i < N; ++i, ++it) {
            result[i] = static_cast<DstT>(*it); // NOLINT(bugprone-signed-char-misuse)
        }
        return result;
    }
}

// Tensor<T> → Tensor

template<TensorLike DstTensor, typename SrcT, ConversionPolicy CP, RankPolicy RP>
std::expected<DstTensor, ConversionError> tensorToTensor(TensorOf<SrcT> auto const& src, std::pmr::memory_resource* mr) {
    using DstT = typename DstTensor::value_type;

    if constexpr (!conversion_allowed_v<SrcT, DstT, CP>) {
        if constexpr (is_widening_v<SrcT, DstT>) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::WideningNotAllowed});
        } else if constexpr (is_static_castable_v<SrcT, DstT>) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::NarrowingNotAllowed});
        } else {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});
        }
    } else {
        if constexpr (gr::tensor_traits<DstTensor>::static_rank && RP == RankPolicy::Strict) {
            constexpr std::size_t dstRank = gr::tensor_traits<DstTensor>::rank;
            if (src.rank() != dstRank) {
                return std::unexpected(ConversionError{.kind = ConversionError::Kind::RankMismatch});
            }
            if constexpr (gr::tensor_traits<DstTensor>::all_static) {
                DstTensor tmp{};
                if (src.extents().size() != dstRank) {
                    return std::unexpected(ConversionError{.kind = ConversionError::Kind::ExtentsMismatch});
                }
                for (std::size_t i = 0UZ; i < dstRank; ++i) {
                    if (src.extent(i) != tmp.extent(i)) {
                        return std::unexpected(ConversionError{.kind = ConversionError::Kind::SizeMismatch});
                    }
                }
            }
        }

        if constexpr (gr::tensor_traits<DstTensor>::all_static) {
            if (src.size() != static_size_v<DstTensor>) {
                return std::unexpected(ConversionError{.kind = ConversionError::Kind::SizeMismatch});
            }
            DstTensor result;
            auto      srcIt = src.begin();
            for (auto& dstElem : result) {
                dstElem = static_cast<DstT>(*srcIt++); // NOLINT(bugprone-signed-char-misuse)
            }
            return result;
        } else if constexpr (gr::tensor_traits<DstTensor>::static_rank) {
            DstTensor result(src.extents(), mr);
            auto      srcIt = src.begin();
            for (auto& dstElem : result) {
                dstElem = static_cast<DstT>(*srcIt++); // NOLINT(bugprone-signed-char-misuse)
            }
            return result;
        } else {
            if constexpr (is_same_type_v<SrcT, DstT>) {
                return DstTensor(src, mr);
            } else {
                DstTensor result(mr);
                result.resize(src.extents());
                auto srcIt = src.begin();
                for (auto& dstElem : result) {
                    dstElem = static_cast<DstT>(*srcIt++); // NOLINT(bugprone-signed-char-misuse)
                }
                return result;
            }
        }
    }
}

// Tensor<Value> element conversion

template<typename DstT, ConversionPolicy CP>
std::expected<DstT, ConversionError> tryConvertElement(const Value& elem, std::size_t idx) {
    if (auto* ptr = elem.get_if<DstT>()) {
        return *ptr; // exact match
    }

    // Handle string types: std::pmr::string in Value, may want std::string or std::pmr::string
    if constexpr (std::same_as<DstT, std::string> || std::same_as<DstT, std::pmr::string>) {
        if (auto* p = elem.get_if<std::pmr::string>()) {
            return DstT{*p};
        }
    }

    if constexpr (CP >= ConversionPolicy::Widening) {
        if constexpr (std::same_as<DstT, double>) {
            if (auto* p = elem.get_if<float>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int32_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int64_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, float>) {
            if (auto* p = elem.get_if<std::int32_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int16_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::int64_t>) {
            if (auto* p = elem.get_if<std::int32_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int16_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int8_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::int32_t>) {
            if (auto* p = elem.get_if<std::int16_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int8_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::int16_t>) {
            if (auto* p = elem.get_if<std::int8_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::uint64_t>) {
            if (auto* p = elem.get_if<std::uint32_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::uint16_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::uint8_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::uint32_t>) {
            if (auto* p = elem.get_if<std::uint16_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::uint8_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::uint16_t>) {
            if (auto* p = elem.get_if<std::uint8_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::complex<double>>) {
            if (auto* p = elem.get_if<std::complex<float>>()) {
                return static_cast<DstT>(*p);
            }
        }
    }

    if constexpr (CP >= ConversionPolicy::Narrowing) {
        if constexpr (std::same_as<DstT, float>) {
            if (auto* p = elem.get_if<double>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::int32_t>) {
            if (auto* p = elem.get_if<std::int64_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<double>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<float>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::int16_t>) {
            if (auto* p = elem.get_if<std::int32_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int64_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::int8_t>) {
            if (auto* p = elem.get_if<std::int16_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int32_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::uint32_t>) {
            if (auto* p = elem.get_if<std::uint64_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::uint16_t>) {
            if (auto* p = elem.get_if<std::uint32_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::uint8_t>) {
            if (auto* p = elem.get_if<std::uint16_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::uint32_t>()) {
                return static_cast<DstT>(*p);
            }
        } else if constexpr (std::same_as<DstT, std::complex<float>>) {
            if (auto* p = elem.get_if<std::complex<double>>()) {
                return static_cast<DstT>(*p);
            }
        }
    }

    if constexpr (CP == ConversionPolicy::Unchecked) {
        if constexpr (std::is_arithmetic_v<DstT>) {
            if (auto* p = elem.get_if<bool>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int8_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int16_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int32_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::int64_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::uint8_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::uint16_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::uint32_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::uint64_t>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<float>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<double>()) {
                return static_cast<DstT>(*p);
            }
        }
        if constexpr (std::same_as<DstT, std::complex<float>> || std::same_as<DstT, std::complex<double>>) {
            if (auto* p = elem.get_if<std::complex<float>>()) {
                return static_cast<DstT>(*p);
            }
            if (auto* p = elem.get_if<std::complex<double>>()) {
                return static_cast<DstT>(*p);
            }
        }
    }

    return std::unexpected(ConversionError{.kind = ConversionError::Kind::ElementTypeMismatch, .index = idx});
}

// Tensor<Value> → vector

template<typename DstT, ConversionPolicy CP, RankPolicy RP>
std::expected<std::vector<DstT>, ConversionError> tensorOfValueToVector(const Tensor<Value>& src) {
    if constexpr (RP == RankPolicy::Strict) {
        if (src.rank() != 1) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::RankMismatch, .message = "strict requires rank=1 for vector"});
        }
    }
    std::vector<DstT> result;
    result.reserve(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        auto converted = tryConvertElement<DstT, CP>(src.data()[i], i);
        if (!converted) {
            return std::unexpected(converted.error());
        }
        result.emplace_back(std::move(*converted));
    }
    return result;
}

// Tensor<Value> → array

template<typename DstT, std::size_t N, ConversionPolicy CP, RankPolicy RP>
std::expected<std::array<DstT, N>, ConversionError> tensorOfValueToArray(const Tensor<Value>& src) {
    if constexpr (RP == RankPolicy::Strict) {
        if (src.rank() != 1) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::RankMismatch, .message = "strict requires rank=1 for array"});
        }
    }
    if (src.size() != N) {
        return std::unexpected(ConversionError{.kind = ConversionError::Kind::SizeMismatch});
    }
    std::array<DstT, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        auto converted = tryConvertElement<DstT, CP>(src.data()[i], i);
        if (!converted) {
            return std::unexpected(converted.error());
        }
        result[i] = std::move(*converted);
    }
    return result;
}

// Tensor<Value> → Tensor

template<TensorLike DstTensor, ConversionPolicy CP, RankPolicy RP>
std::expected<DstTensor, ConversionError> tensorOfValueToTensor(const Tensor<Value>& src, std::pmr::memory_resource* mr) {
    using DstT = typename DstTensor::value_type;

    if constexpr (gr::tensor_traits<DstTensor>::static_rank && RP == RankPolicy::Strict) {
        constexpr std::size_t dstRank = gr::tensor_traits<DstTensor>::rank;
        if (src.rank() != dstRank) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::RankMismatch});
        }
    }

    if constexpr (gr::tensor_traits<DstTensor>::all_static) {
        if (src.size() != static_size_v<DstTensor>) {
            return std::unexpected(ConversionError{.kind = ConversionError::Kind::SizeMismatch});
        }
        DstTensor result;
        auto      dstIt = result.begin();
        for (std::size_t i = 0; i < src.size(); ++i) {
            auto converted = tryConvertElement<DstT, CP>(src.data()[i], i);
            if (!converted) {
                return std::unexpected(converted.error());
            }
            *dstIt++ = std::move(*converted);
        }
        return result;
    } else if constexpr (gr::tensor_traits<DstTensor>::static_rank) {
        DstTensor result(src.extents(), mr);
        auto      dstIt = result.begin();
        for (std::size_t i = 0; i < src.size(); ++i) {
            auto converted = tryConvertElement<DstT, CP>(src.data()[i], i);
            if (!converted) {
                return std::unexpected(converted.error());
            }
            *dstIt++ = std::move(*converted);
        }
        return result;
    } else {
        DstTensor result(mr);
        result.resize(src.extents());
        auto dstIt = result.begin();
        for (std::size_t i = 0; i < src.size(); ++i) {
            auto converted = tryConvertElement<DstT, CP>(src.data()[i], i);
            if (!converted) {
                return std::unexpected(converted.error());
            }
            *dstIt++ = std::move(*converted);
        }
        return result;
    }
}

// map conversions (Map<string, Value> -> map<string, Value>)

template<typename DstMap>
requires gr::meta::is_instantiation_of<DstMap, std::unordered_map> && std::same_as<typename DstMap::mapped_type, Value>
inline std::expected<DstMap, ConversionError> mapToStdMap(const Value::Map& src) {
    DstMap result;
    result.reserve(src.size());
    for (const auto& [key, val] : src) {
        result.emplace(std::string(key), val);
    }
    return result;
}

template<typename DstMap>
requires gr::meta::is_instantiation_of<DstMap, std::map> && std::same_as<typename DstMap::mapped_type, Value>
inline std::expected<DstMap, ConversionError> mapToStdMap(const Value::Map& src) {
    DstMap result;
    for (const auto& [key, val] : src) {
        result.emplace(std::string(key), val);
    }
    return result;
}

// map conversions with element type conversion (Map<string, Value> -> map<string, T>)

template<typename DstMap, ConversionPolicy CP>
requires gr::meta::is_instantiation_of<DstMap, std::unordered_map> && (!std::same_as<typename DstMap::mapped_type, Value>)
inline std::expected<DstMap, ConversionError> mapToTypedStdMap(const Value::Map& src) {
    using DstValueT = typename DstMap::mapped_type;
    DstMap result;
    result.reserve(src.size());
    std::size_t idx = 0;
    for (const auto& [key, val] : src) {
        auto converted = tryConvertElement<DstValueT, CP>(val, idx);
        if (!converted) {
            return std::unexpected(converted.error());
        }
        result.emplace(std::string(key), std::move(*converted));
        ++idx;
    }
    return result;
}

template<typename DstMap, ConversionPolicy CP>
requires gr::meta::is_instantiation_of<DstMap, std::map> && (!std::same_as<typename DstMap::mapped_type, Value>)
inline std::expected<DstMap, ConversionError> mapToTypedStdMap(const Value::Map& src) {
    using DstValueT = typename DstMap::mapped_type;
    DstMap      result;
    std::size_t idx = 0;
    for (const auto& [key, val] : src) {
        auto converted = tryConvertElement<DstValueT, CP>(val, idx);
        if (!converted) {
            return std::unexpected(converted.error());
        }
        result.emplace(std::string(key), std::move(*converted));
        ++idx;
    }
    return result;
}

// dispatch: Value → vector

template<typename DstT, ConversionPolicy CP, RankPolicy RP>
std::expected<std::vector<DstT>, ConversionError> valueToVector(const Value& v) {
    if (!v.is_tensor()) {
        return std::unexpected(ConversionError{.kind = ConversionError::Kind::NotATensor});
    }

    // clang-format off
    switch (v.value_type()) {
#define DISPATCH_CASE(SrcType)                                              \
    case valueTypeFor<SrcType>():                                           \
        if (auto* t = (const_cast<Value&>(v)).get_if<Tensor<SrcType>>())    \
            return tensorToVector<DstT, SrcType, CP, RP>(*t);               \
        return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});

        DISPATCH_CASE(bool)
        DISPATCH_CASE(std::int8_t)
        DISPATCH_CASE(std::int16_t)
        DISPATCH_CASE(std::int32_t)
        DISPATCH_CASE(std::int64_t)
        DISPATCH_CASE(std::uint8_t)
        DISPATCH_CASE(std::uint16_t)
        DISPATCH_CASE(std::uint32_t)
        DISPATCH_CASE(std::uint64_t)
        DISPATCH_CASE(float)
        DISPATCH_CASE(double)
        DISPATCH_CASE(std::complex<float>)
        DISPATCH_CASE(std::complex<double>)
#undef DISPATCH_CASE

    case Value::ValueType::Value:
        if (auto* t = const_cast<Value&>(v).get_if<Tensor<Value>>()) return tensorOfValueToVector<DstT, CP, RP>(*t);
        return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});

    default: return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});
    }
    // clang-format on
}

// dispatch: Value → array

template<typename DstT, std::size_t N, ConversionPolicy CP, RankPolicy RP>
std::expected<std::array<DstT, N>, ConversionError> valueToArray(const Value& v) {
    if (!v.is_tensor()) {
        return std::unexpected(ConversionError{.kind = ConversionError::Kind::NotATensor});
    }

    // clang-format off
    switch (v.value_type()) {
#define DISPATCH_CASE(SrcType)                                              \
    case valueTypeFor<SrcType>():                                           \
        if (auto* t = const_cast<Value&>(v).get_if<Tensor<SrcType>>())      \
            return tensorToArray<DstT, N, SrcType, CP, RP>(*t);             \
        return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});

        DISPATCH_CASE(bool)
        DISPATCH_CASE(std::int8_t)
        DISPATCH_CASE(std::int16_t)
        DISPATCH_CASE(std::int32_t)
        DISPATCH_CASE(std::int64_t)
        DISPATCH_CASE(std::uint8_t)
        DISPATCH_CASE(std::uint16_t)
        DISPATCH_CASE(std::uint32_t)
        DISPATCH_CASE(std::uint64_t)
        DISPATCH_CASE(float)
        DISPATCH_CASE(double)
        DISPATCH_CASE(std::complex<float>)
        DISPATCH_CASE(std::complex<double>)
#undef DISPATCH_CASE

    case Value::ValueType::Value:
        if (auto* t = const_cast<Value&>(v).get_if<Tensor<Value>>()) return tensorOfValueToArray<DstT, N, CP, RP>(*t);
        return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});

    default: return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});
    }
    // clang-format on
}

// dispatch: Value → Tensor

template<TensorLike DstTensor, ConversionPolicy CP, RankPolicy RP>
std::expected<DstTensor, ConversionError> valueToTensor(const Value& v, std::pmr::memory_resource* mr) {
    if (!v.is_tensor()) {
        return std::unexpected(ConversionError{.kind = ConversionError::Kind::NotATensor});
    }

    // clang-format off
    switch (v.value_type()) {
#define DISPATCH_CASE(SrcType)                                              \
    case valueTypeFor<SrcType>():                                           \
        if (auto* t = const_cast<Value&>(v).get_if<Tensor<SrcType>>())      \
            return tensorToTensor<DstTensor, SrcType, CP, RP>(*t, mr);      \
        return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});

        DISPATCH_CASE(bool)
        DISPATCH_CASE(std::int8_t)
        DISPATCH_CASE(std::int16_t)
        DISPATCH_CASE(std::int32_t)
        DISPATCH_CASE(std::int64_t)
        DISPATCH_CASE(std::uint8_t)
        DISPATCH_CASE(std::uint16_t)
        DISPATCH_CASE(std::uint32_t)
        DISPATCH_CASE(std::uint64_t)
        DISPATCH_CASE(float)
        DISPATCH_CASE(double)
        DISPATCH_CASE(std::complex<float>)
        DISPATCH_CASE(std::complex<double>)
#undef DISPATCH_CASE

    case Value::ValueType::Value:
        if (auto* t = const_cast<Value&>(v).get_if<Tensor<Value>>()) return tensorOfValueToTensor<DstTensor, CP, RP>(*t, mr);
        return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});

    default: return std::unexpected(ConversionError{.kind = ConversionError::Kind::TypeMismatch});
    }
    // clang-format on
}

// dispatch: Value → map

template<gr::meta::map_type DstMap, ConversionPolicy CP = ConversionPolicy::Safe>
std::expected<DstMap, ConversionError> valueToMap(const Value& v) {
    if (auto* m = v.get_if<Value::Map>()) {
        if constexpr (std::same_as<typename DstMap::mapped_type, Value>) {
            return mapToStdMap<DstMap>(*m);
        } else {
            return mapToTypedStdMap<DstMap, CP>(*m);
        }
    }
    return std::unexpected(ConversionError{.kind = ConversionError::Kind::NotAMap});
}

} // namespace detail

// public API: convertTo

template<detail::ValidTarget Target, ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, ResourcePolicy ResP = ResourcePolicy::UseDefault>
std::expected<Target, ConversionError> convertTo(const Value& value, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
    if constexpr (ResP == ResourcePolicy::InheritFromSource) {
        static_assert(detail::supports_pmr_v<Target>, "InheritFromSource requires PMR-capable target");
    }

    [[maybe_unused]] std::pmr::memory_resource* actualMr = mr;
    if constexpr (ResP == ResourcePolicy::InheritFromSource && detail::supports_pmr_v<Target>) {
        actualMr = value._resource;
    }

    if constexpr (gr::meta::vector_type<Target>) {
        return detail::valueToVector<typename Target::value_type, CP, RP>(value);
    } else if constexpr (gr::meta::array_type<Target>) {
        return detail::valueToArray<typename Target::value_type, std::tuple_size_v<Target>, CP, RP>(value);
    } else if constexpr (gr::TensorLike<Target>) {
        return detail::valueToTensor<Target, CP, RP>(value, actualMr);
    } else if constexpr (gr::meta::map_type<Target>) {
        return detail::valueToMap<Target, CP>(value);
    }
}

template<detail::ValidTarget Target, ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, ResourcePolicy ResP = ResourcePolicy::UseDefault>
std::expected<Target, ConversionError> convertTo(Value&& value, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
    if constexpr (ResP == ResourcePolicy::InheritFromSource) {
        static_assert(detail::supports_pmr_v<Target>, "InheritFromSource requires PMR-capable target");
    }

    [[maybe_unused]] std::pmr::memory_resource* actualMr = mr;
    if constexpr (ResP == ResourcePolicy::InheritFromSource && detail::supports_pmr_v<Target>) {
        actualMr = value._resource;
    }

    // move optimization for same-type dynamic Tensor
    if constexpr (TensorLike<Target> && !detail::is_fixed_size_v<Target>) {
        using DstT = typename Target::value_type;
        if (value.value_type() == detail::valueTypeFor<DstT>()) {
            if (auto* srcTensor = value.get_if<Tensor<DstT>>()) {
                if constexpr (gr::tensor_traits<Target>::static_rank && RP == RankPolicy::Strict) {
                    if (srcTensor->rank() != gr::tensor_traits<Target>::rank) {
                        return std::unexpected(ConversionError{.kind = ConversionError::Kind::RankMismatch});
                    }
                }
                return Target(std::move(*srcTensor), actualMr);
            }
        }
    }

    return convertTo<Target, CP, RP, ResP>(static_cast<const Value&>(value), mr);
}

// public API: convertTo_or

template<detail::ValidTarget Target, ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, ResourcePolicy ResP = ResourcePolicy::UseDefault, detail::FallbackValue<Target> F>
Target convertTo_or(const Value& value, F&& fallback, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
    auto result = convertTo<Target, CP, RP, ResP>(value, mr);
    return result ? std::move(*result) : static_cast<Target>(std::forward<F>(fallback));
}

template<detail::ValidTarget Target, ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, ResourcePolicy ResP = ResourcePolicy::UseDefault, detail::FallbackFactory<Target> F>
Target convertTo_or(const Value& value, F&& factory, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
    auto result = convertTo<Target, CP, RP, ResP>(value, mr);
    return result ? std::move(*result) : std::forward<F>(factory)();
}

// public API: assignTo vector

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, typename T, typename A>
std::expected<void, ConversionError> assignTo(std::vector<T, A>& dst, const Value& value) {
    auto result = convertTo<std::vector<T>, CP, RP>(value);
    if (!result) {
        return std::unexpected(result.error());
    }
    if (dst.capacity() >= result->size()) {
        dst.resize(result->size());
        std::copy(result->begin(), result->end(), dst.begin());
    } else {
        dst = std::move(*result);
    }
    return {};
}

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, typename T, typename A>
std::expected<void, ConversionError> assignTo(std::vector<T, A>& dst, Value&& value) {
    auto result = convertTo<std::vector<T>, CP, RP>(std::move(value));
    if (!result) {
        return std::unexpected(result.error());
    }
    if (dst.capacity() >= result->size()) {
        dst.resize(result->size());
        std::copy(result->begin(), result->end(), dst.begin());
    } else {
        dst = std::move(*result);
    }
    return {};
}

// public API: assignTo array

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, typename T, std::size_t N>
std::expected<void, ConversionError> assignTo(std::array<T, N>& dst, const Value& value) {
    auto result = convertTo<std::array<T, N>, CP, RP>(value);
    if (!result) {
        return std::unexpected(result.error());
    }
    dst = std::move(*result);
    return {};
}

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, typename T, std::size_t N>
std::expected<void, ConversionError> assignTo(std::array<T, N>& dst, Value&& value) {
    return assignTo<CP, RP>(dst, static_cast<const Value&>(value));
}

// public API: assignTo Tensor

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, TensorLike TensorT>
std::expected<void, ConversionError> assignTo(TensorT& dst, const Value& value) {
    // same-type optimization: copy directly to preserve dst's capacity
    using T = typename gr::tensor_traits<TensorT>::value_type;
    if constexpr (!gr::tensor_traits<TensorT>::all_static) {
        if (value.value_type() == detail::valueTypeFor<T>() && value.is_tensor()) {
            if (auto* srcTensor = const_cast<Value&>(value).get_if<Tensor<T>>()) {
                if constexpr (RP == RankPolicy::Strict && gr::tensor_traits<TensorT>::static_rank) {
                    if (srcTensor->rank() != dst.rank()) {
                        return std::unexpected(ConversionError{.kind = ConversionError::Kind::RankMismatch});
                    }
                }
                if (dst.capacity() >= srcTensor->size()) {
                    dst.resize(srcTensor->extents());
                    std::copy(srcTensor->begin(), srcTensor->end(), dst.begin());
                    return {};
                }
            }
        }
    }

    // fall back to convertTo using dst's resource
    std::pmr::memory_resource* mr = [&]() {
        if constexpr (gr::tensor_traits<TensorT>::all_static) {
            return std::pmr::get_default_resource();
        } else {
            return dst.resource();
        }
    }();

    auto result = convertTo<TensorT, CP, RP>(value, mr);
    if (!result) {
        return std::unexpected(result.error());
    }

    if constexpr (gr::tensor_traits<TensorT>::all_static) {
        std::copy(result->begin(), result->end(), dst.begin());
    } else {
        if (dst.capacity() >= result->size()) {
            dst.resize(result->extents());
            std::copy(result->begin(), result->end(), dst.begin());
        } else {
            dst = std::move(*result);
        }
    }
    return {};
}

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, TensorLike TensorT>
std::expected<void, ConversionError> assignTo(TensorT& dst, Value&& value) {
    // move optimization for same-type
    using T = typename gr::tensor_traits<TensorT>::value_type;
    if constexpr (!gr::tensor_traits<TensorT>::all_static) {
        if (value.value_type() == detail::valueTypeFor<T>()) {
            if (auto* srcTensor = value.get_if<Tensor<T>>()) {
                if constexpr (gr::tensor_traits<TensorT>::static_rank && RP == RankPolicy::Strict) {
                    if (srcTensor->rank() != dst.rank()) {
                        return std::unexpected(ConversionError{.kind = ConversionError::Kind::RankMismatch});
                    }
                }
                dst = std::move(*srcTensor);
                return {};
            }
        }
    }
    return assignTo<CP, RP>(dst, static_cast<const Value&>(value));
}

// public API: assignTo unordered_map<string, Value>

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict>
std::expected<void, ConversionError> assignTo(std::unordered_map<std::string, Value>& dst, const Value& value) {
    auto result = convertTo<std::unordered_map<std::string, Value>, CP, RP>(value);
    if (!result) {
        return std::unexpected(result.error());
    }
    dst = std::move(*result);
    return {};
}

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict>
std::expected<void, ConversionError> assignTo(std::unordered_map<std::string, Value>& dst, Value&& value) {
    return assignTo<CP, RP>(dst, static_cast<const Value&>(value));
}

// public API: assignTo std::map<string, Value>

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict>
std::expected<void, ConversionError> assignTo(std::map<std::string, Value>& dst, const Value& value) {
    auto result = convertTo<std::map<std::string, Value>, CP, RP>(value);
    if (!result) {
        return std::unexpected(result.error());
    }
    dst = std::move(*result);
    return {};
}

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict>
std::expected<void, ConversionError> assignTo(std::map<std::string, Value>& dst, Value&& value) {
    return assignTo<CP, RP>(dst, static_cast<const Value&>(value));
}

// public API: assignTo typed unordered_map<string, V>

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, typename V>
requires(!std::same_as<V, Value>)
std::expected<void, ConversionError> assignTo(std::unordered_map<std::string, V>& dst, const Value& value) {
    auto result = convertTo<std::unordered_map<std::string, V>, CP, RP>(value);
    if (!result) {
        return std::unexpected(result.error());
    }
    dst = std::move(*result);
    return {};
}

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, typename V>
requires(!std::same_as<V, Value>)
std::expected<void, ConversionError> assignTo(std::unordered_map<std::string, V>& dst, Value&& value) {
    return assignTo<CP, RP>(dst, static_cast<const Value&>(value));
}

// public API: assignTo typed std::map<string, V>

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, typename V>
requires(!std::same_as<V, Value>)
std::expected<void, ConversionError> assignTo(std::map<std::string, V>& dst, const Value& value) {
    auto result = convertTo<std::map<std::string, V>, CP, RP>(value);
    if (!result) {
        return std::unexpected(result.error());
    }
    dst = std::move(*result);
    return {};
}

template<ConversionPolicy CP = ConversionPolicy::Safe, RankPolicy RP = RankPolicy::Strict, typename V>
requires(!std::same_as<V, Value>)
std::expected<void, ConversionError> assignTo(std::map<std::string, V>& dst, Value&& value) {
    return assignTo<CP, RP>(dst, static_cast<const Value&>(value));
}

// utility: memory_usage

inline constexpr std::size_t memory_usage(const Value& value) noexcept {
    std::size_t size = sizeof(Value);
    switch (value.container_type()) {
    case Value::ContainerType::Complex: size += (value.value_type() == Value::ValueType::ComplexFloat32) ? sizeof(std::complex<float>) : sizeof(std::complex<double>); break;
    case Value::ContainerType::String: size += sizeof(std::pmr::string) + static_cast<const std::pmr::string*>(value._storage.ptr)->capacity(); break;
    case Value::ContainerType::Map:
        if (const auto* map = value.get_if<Value::Map>()) {
            size += sizeof(Value::Map) + map->bucket_count() * sizeof(void*);
            for (const auto& [k, v] : *map) {
                size += k.capacity() + memory_usage(v);
            }
        }
        break;
    case Value::ContainerType::Tensor: size += sizeof(void*); break;
    default: break;
    }
    return size;
}

} // namespace gr::pmt

// explicit template instantiation declarations (reduce compile-time in multiple TUs)

namespace gr::pmt {

// clang-format off
#define GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(T)                                                       \
    extern template std::expected<std::vector<T>, ConversionError> convertTo<std::vector<T>>(         \
        const Value&, std::pmr::memory_resource*);                                                    \
    extern template std::expected<std::vector<T>, ConversionError> convertTo<std::vector<T>>(         \
        Value&&, std::pmr::memory_resource*);                                                         \
    extern template std::expected<Tensor<T>, ConversionError> convertTo<Tensor<T>>(                   \
        const Value&, std::pmr::memory_resource*);                                                    \
    extern template std::expected<Tensor<T>, ConversionError> convertTo<Tensor<T>>(                   \
        Value&&, std::pmr::memory_resource*);

GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(std::int8_t)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(std::int16_t)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(std::int32_t)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(std::int64_t)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(std::uint8_t)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(std::uint16_t)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(std::uint32_t)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(std::uint64_t)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(float)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(double)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(std::complex<float>)
GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(std::complex<double>)

#undef GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE

extern template std::expected<std::unordered_map<std::string, Value>, ConversionError>
    convertTo<std::unordered_map<std::string, Value>>(const Value&, std::pmr::memory_resource*);
extern template std::expected<std::map<std::string, Value>, ConversionError>
    convertTo<std::map<std::string, Value>>(const Value&, std::pmr::memory_resource*);
// clang-format on

struct ValueVisitor {
private:
    // std::function_ref is C++26, we can not use it
    const void* handler = nullptr;

#define MAKE_HANDLER_MEMBER(Type, Name)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
    using Name##_handler_t           = void(const void*, Type);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
    Name##_handler_t* Name##_handler = nullptr

    MAKE_HANDLER_MEMBER(bool, bool);
    MAKE_HANDLER_MEMBER(std::int8_t, int8_t);
    MAKE_HANDLER_MEMBER(std::int16_t, int16_t);
    MAKE_HANDLER_MEMBER(std::int32_t, int32_t);
    MAKE_HANDLER_MEMBER(std::int64_t, int64_t);
    MAKE_HANDLER_MEMBER(std::uint8_t, uint8_t);
    MAKE_HANDLER_MEMBER(std::uint16_t, uint16_t);
    MAKE_HANDLER_MEMBER(std::uint32_t, uint32_t);
    MAKE_HANDLER_MEMBER(std::uint64_t, uint64_t);
    MAKE_HANDLER_MEMBER(float, float);
    MAKE_HANDLER_MEMBER(double, double);
    MAKE_HANDLER_MEMBER(std::complex<float>, complex_float);
    MAKE_HANDLER_MEMBER(std::complex<double>, complex_double);

    MAKE_HANDLER_MEMBER(std::string_view, string_view);
    MAKE_HANDLER_MEMBER(const Value::Map&, property_map);

    MAKE_HANDLER_MEMBER(const Tensor<bool>&, tensor_bool);
    MAKE_HANDLER_MEMBER(const Tensor<std::int8_t>&, tensor_int8_t);
    MAKE_HANDLER_MEMBER(const Tensor<std::int16_t>&, tensor_int16_t);
    MAKE_HANDLER_MEMBER(const Tensor<std::int32_t>&, tensor_int32_t);
    MAKE_HANDLER_MEMBER(const Tensor<std::int64_t>&, tensor_int64_t);
    MAKE_HANDLER_MEMBER(const Tensor<std::uint8_t>&, tensor_uint8_t);
    MAKE_HANDLER_MEMBER(const Tensor<std::uint16_t>&, tensor_uint16_t);
    MAKE_HANDLER_MEMBER(const Tensor<std::uint32_t>&, tensor_uint32_t);
    MAKE_HANDLER_MEMBER(const Tensor<std::uint64_t>&, tensor_uint64_t);
    MAKE_HANDLER_MEMBER(const Tensor<float>&, tensor_float);
    MAKE_HANDLER_MEMBER(const Tensor<double>&, tensor_double);
    MAKE_HANDLER_MEMBER(const Tensor<std::complex<float>>&, tensor_complex_float);
    MAKE_HANDLER_MEMBER(const Tensor<std::complex<double>>&, tensor_complex_double);

    MAKE_HANDLER_MEMBER(const Tensor<Value>&, tensor_value);

    MAKE_HANDLER_MEMBER(std::monostate, monostate);

#undef MAKE_HANDLER_MEMBER

public:
#define MAKE_FIELD_INIT(Type, Name) Name##_handler(+[](const void* handlerPtr, Type value) { (*static_cast<const Handler*>(handlerPtr))(value); })

    template<typename Handler>
    ValueVisitor(const Handler& _handler)                        //
        : handler(std::addressof(_handler)),                     //
          MAKE_FIELD_INIT(bool, bool),                           //
          MAKE_FIELD_INIT(std::int8_t, int8_t),                  //
          MAKE_FIELD_INIT(std::int16_t, int16_t),                //
          MAKE_FIELD_INIT(std::int32_t, int32_t),                //
          MAKE_FIELD_INIT(std::int64_t, int64_t),                //
          MAKE_FIELD_INIT(std::uint8_t, uint8_t),                //
          MAKE_FIELD_INIT(std::uint16_t, uint16_t),              //
          MAKE_FIELD_INIT(std::uint32_t, uint32_t),              //
          MAKE_FIELD_INIT(std::uint64_t, uint64_t),              //
          MAKE_FIELD_INIT(float, float),                         //
          MAKE_FIELD_INIT(double, double),                       //
          MAKE_FIELD_INIT(std::complex<float>, complex_float),   //
          MAKE_FIELD_INIT(std::complex<double>, complex_double), //

          MAKE_FIELD_INIT(std::string_view, string_view),   //
          MAKE_FIELD_INIT(const Value::Map&, property_map), //

          MAKE_FIELD_INIT(const Tensor<bool>&, tensor_bool),                           //
          MAKE_FIELD_INIT(const Tensor<std::int8_t>&, tensor_int8_t),                  //
          MAKE_FIELD_INIT(const Tensor<std::int16_t>&, tensor_int16_t),                //
          MAKE_FIELD_INIT(const Tensor<std::int32_t>&, tensor_int32_t),                //
          MAKE_FIELD_INIT(const Tensor<std::int64_t>&, tensor_int64_t),                //
          MAKE_FIELD_INIT(const Tensor<std::uint8_t>&, tensor_uint8_t),                //
          MAKE_FIELD_INIT(const Tensor<std::uint16_t>&, tensor_uint16_t),              //
          MAKE_FIELD_INIT(const Tensor<std::uint32_t>&, tensor_uint32_t),              //
          MAKE_FIELD_INIT(const Tensor<std::uint64_t>&, tensor_uint64_t),              //
          MAKE_FIELD_INIT(const Tensor<float>&, tensor_float),                         //
          MAKE_FIELD_INIT(const Tensor<double>&, tensor_double),                       //
          MAKE_FIELD_INIT(const Tensor<std::complex<float>>&, tensor_complex_float),   //
          MAKE_FIELD_INIT(const Tensor<std::complex<double>>&, tensor_complex_double), //

          MAKE_FIELD_INIT(const Tensor<Value>&, tensor_value), //
          MAKE_FIELD_INIT(std::monostate, monostate)           //
    //
    {}
#undef MAKE_FIELD_INIT

    bool visit(const Value& value);
};

} // namespace gr::pmt

#endif // GNURADIO_VALUEHELPER_HPP
