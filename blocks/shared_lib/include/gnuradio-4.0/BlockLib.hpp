#ifndef GR_BLOCKLIB_INIT_HPP

#include <cstddef>
#include <gnuradio-blocklib-core_export.h>

namespace gr {
class BlockRegistry;
}

GNURADIO_BLOCKLIB_CORE_EXPORT std::size_t grBlockLibInit(gr::BlockRegistry& registry);

#endif
