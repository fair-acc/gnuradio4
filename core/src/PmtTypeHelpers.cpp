#include <gnuradio-4.0/PmtTypeHelpers.hpp>

#include <expected>

namespace gr::pmt {

// implement/instantiate helper functions for the given specific types

// ---- fundamental types ----
template std::expected<bool, std::string>                 convert_safely<bool, false>(const pmt::Value&);
template std::expected<std::int8_t, std::string>          convert_safely<std::int8_t, false>(const pmt::Value&);
template std::expected<std::uint8_t, std::string>         convert_safely<std::uint8_t, false>(const pmt::Value&);
template std::expected<std::int16_t, std::string>         convert_safely<std::int16_t, false>(const pmt::Value&);
template std::expected<std::uint16_t, std::string>        convert_safely<std::uint16_t, false>(const pmt::Value&);
template std::expected<std::int32_t, std::string>         convert_safely<std::int32_t, false>(const pmt::Value&);
template std::expected<std::uint32_t, std::string>        convert_safely<std::uint32_t, false>(const pmt::Value&);
template std::expected<std::int64_t, std::string>         convert_safely<std::int64_t, false>(const pmt::Value&);
template std::expected<std::uint64_t, std::string>        convert_safely<std::uint64_t, false>(const pmt::Value&);
template std::expected<float, std::string>                convert_safely<float, false>(const pmt::Value&);
template std::expected<double, std::string>               convert_safely<double, false>(const pmt::Value&);
template std::expected<std::complex<float>, std::string>  convert_safely<std::complex<float>, false>(const pmt::Value&);
template std::expected<std::complex<double>, std::string> convert_safely<std::complex<double>, false>(const pmt::Value&);
template std::expected<std::string, std::string>          convert_safely<std::string, false>(const pmt::Value&);

// ---- vector-of-fundamentals ----
template std::expected<Tensor<bool>, std::string>                 convert_safely<Tensor<bool>, false>(const pmt::Value&);
template std::expected<Tensor<std::int8_t>, std::string>          convert_safely<Tensor<std::int8_t>, false>(const pmt::Value&);
template std::expected<Tensor<std::uint8_t>, std::string>         convert_safely<Tensor<std::uint8_t>, false>(const pmt::Value&);
template std::expected<Tensor<std::int16_t>, std::string>         convert_safely<Tensor<std::int16_t>, false>(const pmt::Value&);
template std::expected<Tensor<std::uint16_t>, std::string>        convert_safely<Tensor<std::uint16_t>, false>(const pmt::Value&);
template std::expected<Tensor<std::int32_t>, std::string>         convert_safely<Tensor<std::int32_t>, false>(const pmt::Value&);
template std::expected<Tensor<std::uint32_t>, std::string>        convert_safely<Tensor<std::uint32_t>, false>(const pmt::Value&);
template std::expected<Tensor<std::int64_t>, std::string>         convert_safely<Tensor<std::int64_t>, false>(const pmt::Value&);
template std::expected<Tensor<std::uint64_t>, std::string>        convert_safely<Tensor<std::uint64_t>, false>(const pmt::Value&);
template std::expected<Tensor<float>, std::string>                convert_safely<Tensor<float>, false>(const pmt::Value&);
template std::expected<Tensor<double>, std::string>               convert_safely<Tensor<double>, false>(const pmt::Value&);
template std::expected<Tensor<std::complex<float>>, std::string>  convert_safely<Tensor<std::complex<float>>, false>(const pmt::Value&);
template std::expected<Tensor<std::complex<double>>, std::string> convert_safely<Tensor<std::complex<double>>, false>(const pmt::Value&);
template std::expected<pmt::Value::Map, std::string>              convert_safely<pmt::Value::Map, false>(const pmt::Value&);

} // namespace gr::pmt
