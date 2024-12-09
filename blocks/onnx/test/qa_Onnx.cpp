#include <boost/ut.hpp>

#include <gnuradio-4.0/onnx/Onnx.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

template<typename T>
struct TestParameters {
    std::vector<std::vector<T>> inputs;
    std::vector<T>              output;
};

const boost::ut::suite<"basic onnx tests"> basicMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::onnx;
};

int main() { /* not needed for UT */ }