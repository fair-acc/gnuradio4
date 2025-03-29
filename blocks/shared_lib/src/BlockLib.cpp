#include <gnuradio-4.0/BlockLib.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

extern "C" {
bool gr_blocklib_init_module_GrFilterBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrElectricalBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrHttpBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrFileIoBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrTestingBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrBasicBlocks(gr::BlockRegistry& registry);
}

std::size_t grBlockLibInit(gr::BlockRegistry& registry) {
    std::size_t result = 0UZ;
    result += gr_blocklib_init_module_GrFilterBlocks(registry);
    result += gr_blocklib_init_module_GrElectricalBlocks(registry);
    result += gr_blocklib_init_module_GrHttpBlocks(registry);
    result += gr_blocklib_init_module_GrFileIoBlocks(registry);
    result += gr_blocklib_init_module_GrTestingBlocks(registry);
    result += gr_blocklib_init_module_GrBasicBlocks(registry);
    return result;
}

auto grBlockLibInitInvoke = [] { return grBlockLibInit(*grGlobalBlockRegistry()); }();
