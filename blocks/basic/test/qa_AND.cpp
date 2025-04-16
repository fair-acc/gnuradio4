#include <boost/ut.hpp>
#include <gnuradio-4.0/basic/And.hpp>
// #include <limits>

using namespace gr::basic;
using namespace boost::ut;

const suite AndTests = [] {
    "Bitwise AND operations"_test = [] {
        And<uint8_t> andBlock;
        // Basic bitwise AND operation
        expect(eq(andBlock.processOne(0xFF, 0x0F), 0x0F));  // 0xFF & 0x0F = 0x0F
        expect(eq(andBlock.processOne(0x00, 0xFF), 0x00));
        expect(eq(andBlock.processOne(0xAB, 0x22), 0x22));
    };

    "Edge cases"_test = [] {
        And<uint8_t> andBlock;
        // Edge cases for boundary values (max and min values)
        expect(eq(andBlock.processOne(0xFF, 0xFF), 0xFF));  // 0xFF & 0xFF = 0xFF
        expect(eq(andBlock.processOne(0x00, 0x00), 0x00));  // 0x00 & 0x00 = 0x00
    };

    "int16_t support"_test = [] {
        And<int16_t> andBlock;
        // Test with int16_t values
        expect(eq(andBlock.processOne(0x7FFF, 0x00FF), 0x00FF));  // 0x7FFF & 0x00FF = 0x0101
        expect(eq(andBlock.processOne(static_cast<int16_t>(-1), static_cast<int16_t>(0xAABB)), static_cast<int16_t>(0xAABB))); // -1 & 0xAABB = 0xAABB
    };

     "int32_t support"_test = [] {
        And<int32_t> andBlock;
        // Test bitwise AND on int32_t values (max and min values)
        expect(eq(andBlock.processOne(0xFFFF, 0x0F0F), 0x0F0F));  // 0xFFFF (1111111111111111) & 0x0F0F (0000111100001111) = 0x0F0F
        expect(eq(andBlock.processOne(-1, 0x0000), 0x0000));  // -1 & 0x0000 = 0x0000
    };

    "Boundary cases"_test = [] {
        And<int32_t> andBlock;
        // Boundary test cases for int32_t limits
        expect(eq(andBlock.processOne(std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min()), 0));  // MAX & MIN = 0
        expect(eq(andBlock.processOne(std::numeric_limits<int32_t>::max(), -1), std::numeric_limits<int32_t>::max()));  // MAX & -1 = MAX
    };

    "Negative values"_test = [] {
        And<int16_t> andBlock;
        // Test with negative values for int16_t
        expect(eq(andBlock.processOne(static_cast<int16_t>(-1), static_cast<int16_t>(-1)), static_cast<int16_t>(-1)));  // -1 & -1 = -1

        /*
              1111 1111 1111 1011   (-5 in two's complement)
            & 0000 0000 0000 0011   (3 in binary)
            --------------------
            0000 0000 0000 0011   (Result = 3)
         */
        expect(eq(andBlock.processOne(static_cast<int16_t>(-5), static_cast<int16_t>(3)), static_cast<int16_t>(3)));  // -5 & 3 = 3
    };
};

int main() { return boost::ut::cfg<boost::ut::override>.run(); }