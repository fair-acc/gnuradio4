#include <gnuradio-4.0/BlockLib.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

gr::BlockRegistry& grBlockLibRegistry() { return gr::globalBlockRegistry(); }
