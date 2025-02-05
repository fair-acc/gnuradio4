#include <gnuradio-4.0/PmtTypeHelpers.hpp>

namespace pmtv {

// implement/instantiate helper functions for the given specific types

// ---- fundamental types ----
template std::expected<bool, std::string>                 convert_safely<bool, false, pmt>(const pmt&);
template std::expected<std::int8_t, std::string>          convert_safely<std::int8_t, false, pmt>(const pmt&);
template std::expected<std::uint8_t, std::string>         convert_safely<std::uint8_t, false, pmt>(const pmt&);
template std::expected<std::int16_t, std::string>         convert_safely<std::int16_t, false, pmt>(const pmt&);
template std::expected<std::uint16_t, std::string>        convert_safely<std::uint16_t, false, pmt>(const pmt&);
template std::expected<std::int32_t, std::string>         convert_safely<std::int32_t, false, pmt>(const pmt&);
template std::expected<std::uint32_t, std::string>        convert_safely<std::uint32_t, false, pmt>(const pmt&);
template std::expected<std::int64_t, std::string>         convert_safely<std::int64_t, false, pmt>(const pmt&);
template std::expected<std::uint64_t, std::string>        convert_safely<std::uint64_t, false, pmt>(const pmt&);
template std::expected<float, std::string>                convert_safely<float, false, pmt>(const pmt&);
template std::expected<double, std::string>               convert_safely<double, false, pmt>(const pmt&);
template std::expected<std::complex<float>, std::string>  convert_safely<std::complex<float>, false, pmt>(const pmt&);
template std::expected<std::complex<double>, std::string> convert_safely<std::complex<double>, false, pmt>(const pmt&);
template std::expected<std::string, std::string>          convert_safely<std::string, false, pmt>(const pmt&);

// ---- vector-of-fundamentals ----
template std::expected<std::vector<bool>, std::string>                 convert_safely<std::vector<bool>, false, pmt>(const pmt&);
template std::expected<std::vector<std::int8_t>, std::string>          convert_safely<std::vector<std::int8_t>, false, pmt>(const pmt&);
template std::expected<std::vector<std::uint8_t>, std::string>         convert_safely<std::vector<std::uint8_t>, false, pmt>(const pmt&);
template std::expected<std::vector<std::int16_t>, std::string>         convert_safely<std::vector<std::int16_t>, false, pmt>(const pmt&);
template std::expected<std::vector<std::uint16_t>, std::string>        convert_safely<std::vector<std::uint16_t>, false, pmt>(const pmt&);
template std::expected<std::vector<std::int32_t>, std::string>         convert_safely<std::vector<std::int32_t>, false, pmt>(const pmt&);
template std::expected<std::vector<std::uint32_t>, std::string>        convert_safely<std::vector<std::uint32_t>, false, pmt>(const pmt&);
template std::expected<std::vector<std::int64_t>, std::string>         convert_safely<std::vector<std::int64_t>, false, pmt>(const pmt&);
template std::expected<std::vector<std::uint64_t>, std::string>        convert_safely<std::vector<std::uint64_t>, false, pmt>(const pmt&);
template std::expected<std::vector<float>, std::string>                convert_safely<std::vector<float>, false, pmt>(const pmt&);
template std::expected<std::vector<double>, std::string>               convert_safely<std::vector<double>, false, pmt>(const pmt&);
template std::expected<std::vector<std::complex<float>>, std::string>  convert_safely<std::vector<std::complex<float>>, false, pmt>(const pmt&);
template std::expected<std::vector<std::complex<double>>, std::string> convert_safely<std::vector<std::complex<double>>, false, pmt>(const pmt&);
template std::expected<std::vector<std::string>, std::string>          convert_safely<std::vector<std::string>, false, pmt>(const pmt&);
template std::expected<map_t, std::string>                             convert_safely<map_t, false, pmt>(const pmt&);

} // namespace pmtv
