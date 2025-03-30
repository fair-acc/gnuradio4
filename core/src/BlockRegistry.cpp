#include <gnuradio-4.0/BlockRegistry.hpp>

extern "C" {
GNURADIO_EXPORT
gr::BlockRegistry* grGlobalBlockRegistry([[maybe_unused]] std::source_location location) {
    static gr::BlockRegistry s_instance;
#ifndef NDEBUG
    fmt::print("This is the registry {}, called from {} {}\n", //
        static_cast<void*>(std::addressof(s_instance)),        //
        location.file_name(), location.line());
#endif
    return std::addressof(s_instance);
}
}

namespace gr {
BlockRegistry& globalBlockRegistry(std::source_location location) { return *grGlobalBlockRegistry(location); }
} // namespace gr
