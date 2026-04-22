#ifndef GNURADIO_VALUEMAPFORMATTER_HPP
#define GNURADIO_VALUEMAPFORMATTER_HPP

#include <gnuradio-4.0/ValueMap.hpp>
#include <gnuradio-4.0/formatter/ValueFormatter.hpp>

#include <ostream>

namespace gr::pmt {

// ostream printer for diagnostics / logging / boost::ut failure messages. Kept in a
// separate formatter header so ValueMap.hpp itself stays iostream-free per the project's
// embedded-friendly policy.
inline std::ostream& operator<<(std::ostream& os, const ValueMap& map) {
    os << '{';
    bool first = true;
    for (const auto& [key, value] : map) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << '"' << key << "\":" << value;
    }
    return os << '}';
}

} // namespace gr::pmt

#endif // GNURADIO_VALUEMAPFORMATTER_HPP
