#include <boost/ut.hpp>

#include <gnuradio-4.0/basic/CommonBlocks.hpp>
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

        expect((std::ranges::includes(known, desired)));
    };
};

int main() { return boost::ut::cfg<boost::ut::override>.run(); }
