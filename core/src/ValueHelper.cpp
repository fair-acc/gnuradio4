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

} // namespace gr::pmt
