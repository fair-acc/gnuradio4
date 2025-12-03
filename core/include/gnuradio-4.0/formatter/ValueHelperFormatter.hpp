#ifndef GNURADIO_VALUEHELPERFORMAT_HPP
#define GNURADIO_VALUEHELPERFORMAT_HPP

#include <gnuradio-4.0/formatter/ValueFormatter.hpp>

#include <magic_enum.hpp>
#include <ostream>

namespace gr::pmt {
inline std::ostream& operator<<(std::ostream& os, ConversionError::Kind k) { return os << magic_enum::enum_name(k); }
} // namespace gr::pmt

#endif // GNURADIO_VALUEHELPERFORMAT_HPP
