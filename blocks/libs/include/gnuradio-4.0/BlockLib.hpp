#ifndef GR_BLOCKLIB_INIT_HPP

#include <cstddef>
#include <gnuradio-4.0/Export.hpp>

namespace gr {
class BlockRegistry;
}

GNURADIO_EXPORT std::size_t grBlockLibInit(gr::BlockRegistry& registry);

#endif
