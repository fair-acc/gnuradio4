#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <cassert>
#include <iostream>

using namespace std::string_literals;

// This tests automatic loading of .so files that are not
// plugins, but ordinary dynamic block libraries -- they
// just need to be in a path that the PluginLoader searches in.
//
// This is intentionally not a ut test as it tests
// how a normal application would use the block registry
// in the lifetime of main

extern "C" {
std::size_t gr_blocklib_init_module_GrBasicBlocks(gr::BlockRegistry&);
}

int main() {
    gr_blocklib_init_module_GrBasicBlocks(gr::globalBlockRegistry());

    auto known = gr::globalBlockRegistry().knownBlocks();
    std::ranges::sort(known);
    std::vector<std::string> desired{
        "gr::basic::DataSink<float32>"s,          //
        "gr::basic::DataSink<float64>"s,          //
        "gr::basic::DataSetSink<float32>"s,       //
        "gr::basic::DataSetSink<float64>"s,       //
        "gr::basic::FunctionGenerator<float32>"s, //
        "gr::basic::FunctionGenerator<float64>"s, //
        "gr::basic::Selector<float32>"s,          //
        "gr::basic::Selector<float64>"s,          //
        "gr::basic::SignalGenerator<float32>"s,   //
        "gr::basic::SignalGenerator<float64>"s    //
    };
    std::ranges::sort(desired);

    assert((std::ranges::includes(known, desired)));
    std::cout << "All ok\n";
}
