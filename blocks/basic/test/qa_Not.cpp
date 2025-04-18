#include <boost/ut.hpp>
#include <gnuradio-4.0/basic/Not.hpp>

using namespace gr::basic;
using namespace boost::ut;

const suite NotTests = [] {
    "uint8_t support"_test = [] {
        Not<uint8_t> notBlock;

        expect(eq(notBlock.processOne(0xFF), 0x00));
        expect(eq(notBlock.processOne(0x00), 0xFF));
        expect(eq(notBlock.processOne(0xAA), 0x55)); // ~(10101010) = 01010101
    };

    "int16_t support"_test = [] {
        Not<int16_t> notBlock;

        expect(eq(notBlock.processOne(static_cast<int16_t>(0x00FF)), static_cast<int16_t>(0xFF00)));
        expect(eq(notBlock.processOne(static_cast<int16_t>(0x0000)), static_cast<int16_t>(0xFFFF)));
        expect(eq(notBlock.processOne(static_cast<int16_t>(0xAAAA)), static_cast<int16_t>(0x5555)));
    };


    "int32_t support"_test = [] {
        Not<int32_t> notBlock;

        expect(eq(notBlock.processOne(0x000000FF), static_cast<int32_t>(0xFFFFFF00)));
        expect(eq(notBlock.processOne(0x00000000), static_cast<int32_t>(0xFFFFFFFF)));
        expect(eq(notBlock.processOne(0x0000AAAA), static_cast<int32_t>(0xFFFF5555)));
    };
};


int main() { return boost::ut::cfg<boost::ut::override>.run(); }