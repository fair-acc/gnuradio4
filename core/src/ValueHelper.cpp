#include <exception>

#include <gnuradio-4.0/ValueHelper.hpp>

namespace gr::pmt {

// clang-format off
#define GR_PMT_CONVERT_INSTANTIATE_SAME_TYPE(T)                                                                            \
template std::expected<std::vector<T>, ConversionError> convertTo<std::vector<T>>(const Value&,                        \
std::pmr::memory_resource*);        \
template std::expected<std::vector<T>, ConversionError> convertTo<std::vector<T>>(Value&&, std::pmr::memory_resource*); \
template std::expected<Tensor<T>, ConversionError>      convertTo<Tensor<T>>(const Value&, std::pmr::memory_resource*); \
template std::expected<Tensor<T>, ConversionError>      convertTo<Tensor<T>>(Value&&, std::pmr::memory_resource*);

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
// clang-format on

template std::expected<std::unordered_map<std::string, Value>, ConversionError> convertTo<std::unordered_map<std::string, Value>>(const Value&, std::pmr::memory_resource*);
template std::expected<std::map<std::string, Value>, ConversionError>           convertTo<std::map<std::string, Value>>(const Value&, std::pmr::memory_resource*);

bool ValueVisitor::visit(const Value& value) {
#define MAKE_VISITOR_CHECK(Type, Name)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \
    if (value.holds<Type>()) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
        Name##_handler(handler, value.value_or(Type{}));                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
        return true;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           \
    }

    MAKE_VISITOR_CHECK(bool, bool)
    MAKE_VISITOR_CHECK(std::int8_t, int8_t)
    MAKE_VISITOR_CHECK(std::int16_t, int16_t)
    MAKE_VISITOR_CHECK(std::int32_t, int32_t)
    MAKE_VISITOR_CHECK(std::int64_t, int64_t)
    MAKE_VISITOR_CHECK(std::uint8_t, uint8_t)
    MAKE_VISITOR_CHECK(std::uint16_t, uint16_t)
    MAKE_VISITOR_CHECK(std::uint32_t, uint32_t)
    MAKE_VISITOR_CHECK(std::uint64_t, uint64_t)
    MAKE_VISITOR_CHECK(float, float)
    MAKE_VISITOR_CHECK(double, double)
    MAKE_VISITOR_CHECK(std::complex<float>, complex_float)
    MAKE_VISITOR_CHECK(std::complex<double>, complex_double)

    MAKE_VISITOR_CHECK(std::string_view, string_view)
    MAKE_VISITOR_CHECK(Value::Map, property_map)

    MAKE_VISITOR_CHECK(Tensor<bool>, tensor_bool)
    MAKE_VISITOR_CHECK(Tensor<std::int8_t>, tensor_int8_t)
    MAKE_VISITOR_CHECK(Tensor<std::int16_t>, tensor_int16_t)
    MAKE_VISITOR_CHECK(Tensor<std::int32_t>, tensor_int32_t)
    MAKE_VISITOR_CHECK(Tensor<std::int64_t>, tensor_int64_t)
    MAKE_VISITOR_CHECK(Tensor<std::uint8_t>, tensor_uint8_t)
    MAKE_VISITOR_CHECK(Tensor<std::uint16_t>, tensor_uint16_t)
    MAKE_VISITOR_CHECK(Tensor<std::uint32_t>, tensor_uint32_t)
    MAKE_VISITOR_CHECK(Tensor<std::uint64_t>, tensor_uint64_t)
    MAKE_VISITOR_CHECK(Tensor<float>, tensor_float)
    MAKE_VISITOR_CHECK(Tensor<double>, tensor_double)
    MAKE_VISITOR_CHECK(Tensor<std::complex<float>>, tensor_complex_float)
    MAKE_VISITOR_CHECK(Tensor<std::complex<double>>, tensor_complex_double)

    MAKE_VISITOR_CHECK(Tensor<std::pmr::string>, tensor_pmr_string)

    MAKE_VISITOR_CHECK(Tensor<Value>, tensor_value)

    if (value.is_monostate()) {
        monostate_handler(handler, std::monostate());
        return true;
    }

    // nothing matched
    std::println("ERROR nothing matched _value_type {} _container_type {} {}", //
        value.value_type(), value.container_type(), value.holds<std::uint32_t>());
    std::terminate();
    return false;
}

} // namespace gr::pmt
