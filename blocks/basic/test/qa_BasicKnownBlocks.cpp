#include <boost/ut.hpp>

#include <gnuradio-4.0/basic/clock_source.hpp>
#include <gnuradio-4.0/basic/common_blocks.hpp>
#include <gnuradio-4.0/basic/DataSink.hpp>
#include <gnuradio-4.0/basic/FunctionGenerator.hpp>
#include <gnuradio-4.0/basic/Selector.hpp>
#include <gnuradio-4.0/basic/SignalGenerator.hpp>

const boost::ut::suite KnownBlockTests = [] {
    using namespace boost::ut;
    using namespace std::string_literals;

    "Registered"_test = [] {
        auto known = gr::globalBlockRegistry().knownBlocks();
        std::ranges::sort(known);

        std::vector<std::string> desired{
            //
            "builtin_counter"s,              //
            "builtin_multiply"s,             //
            "gr::basic::DataSink"s,          //
            "gr::basic::FunctionGenerator"s, //
            "gr::basic::Selector"s,          //
            "gr::basic::SignalGenerator"s    //
        };
        std::ranges::sort(desired);

        expect((std::ranges::includes(known, desired)));
    };
};

int
main() {
    return boost::ut::cfg<boost::ut::override>.run();
}
