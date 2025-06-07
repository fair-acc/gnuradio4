#include <boost/ut.hpp>
#include <gnuradio-4.0/basic/Xor.hpp>

using namespace gr::basic;
using namespace boost::ut;

const suite XorTests = [] {
    "uint8_t support"_test = [] {
        Xor<uint8_t> xorBlock;

        expect(eq(xorBlock.processOne(0xAA, 0x55), 0xFF));
        expect(eq(xorBlock.processOne(0xFF, 0xFF), 0x00));
        expect(eq(xorBlock.processOne(0x01, 0x03), 0x02)); // 1 ^ 2 = 3
    };

    "int16_t support"_test = [] {
        Xor<int16_t> xorBlock;

        expect(eq(xorBlock.processOne(static_cast<int16_t>(0x00AA), static_cast<int16_t>(0x0055)), static_cast<int16_t>(0x00FF)));
        expect(eq(xorBlock.processOne(static_cast<int16_t>(0xAAAA), static_cast<int16_t>(0xAAAA)), static_cast<int16_t>(0x0000)));
        expect(eq(xorBlock.processOne(static_cast<int16_t>(0x0F0F), static_cast<int16_t>(0xF0F0)), static_cast<int16_t>(0xFFFF)));
    };

    "int32_t support"_test = [] {
        Xor<int32_t> xorBlock;

        expect(eq(xorBlock.processOne(0x000000AA, 0x00000055), 0x000000FF));
        expect(eq(xorBlock.processOne(0x0000AAAA, 0x0000AAAA), 0x00000000));
        expect(eq(xorBlock.processOne(0x00000F0F, 0x0000F0F0), 0x0000FFFF));
    };
};

int main() { return boost::ut::cfg<boost::ut::override>.run(); }