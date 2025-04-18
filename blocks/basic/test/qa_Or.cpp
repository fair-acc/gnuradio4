#include <boost/ut.hpp>
#include <gnuradio-4.0/basic/Or.hpp>

using namespace gr::basic;
using namespace boost::ut;

const suite OrTests = [] {
    "uint8_t support"_test = [] {
        Or<uint8_t> orBlock;

        expect(eq(orBlock.processOne(0xAA, 0x55), 0xFF));
        expect(eq(orBlock.processOne(0x00, 0xFF), 0xFF));
        expect(eq(orBlock.processOne(0x01, 0x02), 0x03)); // 1 | 2 = 3
    };

    "int16_t support"_test = [] {
        Or<int16_t> orBlock;

        expect(eq(orBlock.processOne(static_cast<int16_t>(0x00AA), static_cast<int16_t>(0x0055)), static_cast<int16_t>(0x00FF)));
        expect(eq(orBlock.processOne(static_cast<int16_t>(0x0000), static_cast<int16_t>(0xFFFF)), static_cast<int16_t>(0xFFFF)));
        expect(eq(orBlock.processOne(static_cast<int16_t>(0xAAAA), static_cast<int16_t>(0x5555)), static_cast<int16_t>(0xFFFF)));
    };

    "int32_t support"_test = [] {
        Or<int32_t> orBlock;

        expect(eq(orBlock.processOne(0x000000AA, 0x00000055), 0x000000FF));
        expect(eq(orBlock.processOne(0x00000000, 0x0000FFFF), 0x0000FFFF));
        expect(eq(orBlock.processOne(0x0000AAAA, 0x00005555), 0x0000FFFF));
    };
};

int main() { return boost::ut::cfg<boost::ut::override>.run(); }