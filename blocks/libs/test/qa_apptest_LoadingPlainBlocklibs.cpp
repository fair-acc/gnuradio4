#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

#include <cassert>
#include <print>

// This tests automatic loading of .so files that are not
// plugins, but ordinary dynamic block libraries -- they
// just need to be in a path that the PluginLoader searches in.
//
// This is intentionally not a ut test as it tests
// how a normal application would use the global block
// registry in the lifetime of main

int main() {
    gr::globalPluginLoader();

    auto known = gr::globalBlockRegistry().keys();
    std::ranges::sort(known);
    std::vector<std::string> desired{
        //
        "builtin_counter<float32>"s,              //
        "builtin_counter<float64>"s,              //
        "builtin_multiply<float32>"s,             //
        "builtin_multiply<float64>"s,             //
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
    std::println("All ok");
}
