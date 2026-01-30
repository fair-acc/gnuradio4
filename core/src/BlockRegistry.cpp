#include <gnuradio-4.0/BlockRegistry.hpp>

extern "C" {
GNURADIO_EXPORT
gr::BlockRegistry* grGlobalBlockRegistry([[maybe_unused]] std::source_location location) {
    static gr::BlockRegistry s_instance;
    return std::addressof(s_instance);
}

GNURADIO_EXPORT
gr::SchedulerRegistry* grGlobalSchedulerRegistry([[maybe_unused]] std::source_location location) {
    static gr::SchedulerRegistry s_instance;
    return std::addressof(s_instance);
}
}

namespace gr {
BlockRegistry&     globalBlockRegistry(std::source_location location) { return *grGlobalBlockRegistry(location); }
SchedulerRegistry& globalSchedulerRegistry(std::source_location location) { return *grGlobalSchedulerRegistry(location); }
} // namespace gr
