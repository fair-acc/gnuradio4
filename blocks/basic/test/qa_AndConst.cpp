#include <boost/ut.hpp>
#include <gnuradio-4.0/basic/AndConst.hpp>

using namespace gr::basic;
using namespace boost::ut;

const suite AndConstTests = [] {
    "uint8_t support"_test = [] {
        AndConst<uint8_t> andConstBlock;

        // Test default constant = 1
        andConstBlock.constant = 1;
        expect(eq(andConstBlock.processOne(0xFF), 0x01));
        expect(eq(andConstBlock.processOne(0x00), 0x00));
        expect(eq(andConstBlock.processOne(0xAA), 0x00));

        // Test constant = 0
        andConstBlock.constant = 0;
        expect(eq(andConstBlock.processOne(0xFF), 0x00));
        expect(eq(andConstBlock.processOne(0x00), 0x00));
        expect(eq(andConstBlock.processOne(0xAA), 0x00));
    };

    "int16_t support"_test = [] {
        AndConst<int16_t> andConstBlock;

        // Test default constant = 1
        andConstBlock.constant = 1;
        expect(eq(andConstBlock.processOne(static_cast<int16_t>(0xFF)), 0x0001));
        expect(eq(andConstBlock.processOne(static_cast<int16_t>(0x00)), 0x0000));
        expect(eq(andConstBlock.processOne(static_cast<int16_t>(0xAA)), 0x0000));

        // Test constant = 0
        andConstBlock.constant = 0;
        expect(eq(andConstBlock.processOne(static_cast<int16_t>(0xFF)), 0x0000));
        expect(eq(andConstBlock.processOne(static_cast<int16_t>(0x00)), 0x0000));
        expect(eq(andConstBlock.processOne(static_cast<int16_t>(0xAA)), 0x0000));
    };

    "int32_t support"_test = [] {
        AndConst<int32_t> andConstBlock;

        // Test default constant = 1
        andConstBlock.constant = 1;
        expect(eq(andConstBlock.processOne((0xFF)), 0x00000001));
        expect(eq(andConstBlock.processOne((0x00)), 0x00000000));
        expect(eq(andConstBlock.processOne((0xAA)), 0x00000000));

        // Test constant = 0
        andConstBlock.constant = 0;
        expect(eq(andConstBlock.processOne((0xFF)), 0x00000000));
        expect(eq(andConstBlock.processOne((0x00)), 0x00000000));
        expect(eq(andConstBlock.processOne((0xAA)), 0x00000000));
    };
};

int main() { return boost::ut::cfg<boost::ut::override>.run(); }
