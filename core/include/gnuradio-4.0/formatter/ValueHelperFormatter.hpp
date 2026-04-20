#ifndef GNURADIO_VALUEHELPERFORMAT_HPP
#define GNURADIO_VALUEHELPERFORMAT_HPP

#include <gnuradio-4.0/formatter/ValueFormatter.hpp>

#include <gnuradio-4.0/meta/reflection.hpp>
#include <ostream>

namespace gr::pmt {
inline std::ostream& operator<<(std::ostream& os, ConversionError::Kind k) { return os << gr::meta::enumName(k).value_or(""); }
} // namespace gr::pmt

#endif // GNURADIO_VALUEHELPERFORMAT_HPP
