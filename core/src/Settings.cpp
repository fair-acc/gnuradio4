#include <gnuradio-4.0/Settings.hpp>

namespace gr {

namespace detail {
template std::size_t hash_combine<std::size_t>(std::size_t seed, std::size_t const& v) noexcept;
}

namespace settings {

template std::expected<bool, std::string>                 convertParameter<bool>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::int8_t, std::string>          convertParameter<std::int8_t>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::uint8_t, std::string>         convertParameter<std::uint8_t>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::int16_t, std::string>         convertParameter<std::int16_t>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::uint16_t, std::string>        convertParameter<std::uint16_t>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::int32_t, std::string>         convertParameter<std::int32_t>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::uint32_t, std::string>        convertParameter<std::uint32_t>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::int64_t, std::string>         convertParameter<std::int64_t>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::uint64_t, std::string>        convertParameter<std::uint64_t>(std::string_view key, const pmtv::pmt& value);
template std::expected<float, std::string>                convertParameter<float>(std::string_view key, const pmtv::pmt& value);
template std::expected<double, std::string>               convertParameter<double>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::complex<float>, std::string>  convertParameter<std::complex<float>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::complex<double>, std::string> convertParameter<std::complex<double>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::string, std::string>          convertParameter<std::string>(std::string_view key, const pmtv::pmt& value);

// Specialisation declarations for std::string and vectors
template std::expected<std::vector<bool>, std::string>                 convertParameter<std::vector<bool>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::int8_t>, std::string>          convertParameter<std::vector<std::int8_t>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::uint8_t>, std::string>         convertParameter<std::vector<std::uint8_t>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::int16_t>, std::string>         convertParameter<std::vector<std::int16_t>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::uint16_t>, std::string>        convertParameter<std::vector<std::uint16_t>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::int32_t>, std::string>         convertParameter<std::vector<std::int32_t>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::uint32_t>, std::string>        convertParameter<std::vector<std::uint32_t>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::int64_t>, std::string>         convertParameter<std::vector<std::int64_t>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::uint64_t>, std::string>        convertParameter<std::vector<std::uint64_t>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<float>, std::string>                convertParameter<std::vector<float>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<double>, std::string>               convertParameter<std::vector<double>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::complex<float>>, std::string>  convertParameter<std::vector<std::complex<float>>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::complex<double>>, std::string> convertParameter<std::vector<std::complex<double>>>(std::string_view key, const pmtv::pmt& value);
template std::expected<std::vector<std::string>, std::string>          convertParameter<std::vector<std::string>>(std::string_view key, const pmtv::pmt& value);

template std::expected<property_map, std::string> convertParameter<property_map>(std::string_view key, const pmtv::pmt& value);

} // namespace settings

} // namespace gr
