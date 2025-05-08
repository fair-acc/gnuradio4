#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <cassert>
#include <iostream>

#include <GrBasicBlocks.hpp>

using namespace std::string_literals;

int main() {
    gr_blocklib_init_module_GrBasicBlocks(gr::globalBlockRegistry());

    auto known = gr::globalBlockRegistry().keys();
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
