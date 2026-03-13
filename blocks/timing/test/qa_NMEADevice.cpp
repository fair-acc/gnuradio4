#include <boost/ut.hpp>

#include <gnuradio-4.0/NMEADevice.hpp>

using namespace boost::ut;
using namespace gr::timing;

const boost::ut::suite<"NMEADevice"> nmeaDeviceTests = [] {
    "parseHex16 converts hex strings"_test = [] {
        expect(eq(gr::timing::detail::parseHex16("0000"), std::uint16_t{0}));
        expect(eq(gr::timing::detail::parseHex16("FFFF"), std::uint16_t{0xFFFF}));
        expect(eq(gr::timing::detail::parseHex16("1546"), std::uint16_t{0x1546}));
        expect(eq(gr::timing::detail::parseHex16("01a8"), std::uint16_t{0x01a8}));
        expect(eq(gr::timing::detail::parseHex16("abCD"), std::uint16_t{0xabCD}));
    };

    "parseHex16 handles trailing whitespace"_test = [] {
        expect(eq(gr::timing::detail::parseHex16("1546\n"), std::uint16_t{0x1546}));
        expect(eq(gr::timing::detail::parseHex16("01a8\r\n"), std::uint16_t{0x01a8}));
    };

    "isNMEALine recognises valid NMEA prefixes"_test = [] {
        expect(gr::timing::detail::isNMEALine("$GPRMC,120000.00,A,,,,,,,,,*00"));
        expect(gr::timing::detail::isNMEALine("$GNRMC,test"));
        expect(gr::timing::detail::isNMEALine("$GPGGA,test"));
        expect(!gr::timing::detail::isNMEALine("")) << "empty line";
        expect(!gr::timing::detail::isNMEALine("GPRMC")) << "missing $";
        expect(!gr::timing::detail::isNMEALine("$1234")) << "digits after $";
        expect(!gr::timing::detail::isNMEALine("$GP")) << "too short";
    };

    "isKnownGpsDevice matches known VID/PID pairs"_test = [] {
        expect(gr::timing::detail::isKnownGpsDevice(0x1546, 0x01a8)) << "u-blox 8";
        expect(gr::timing::detail::isKnownGpsDevice(0x067b, 0x2303)) << "Prolific PL2303";
        expect(gr::timing::detail::isKnownGpsDevice(0x1a86, 0x7523)) << "CH340";
        expect(!gr::timing::detail::isKnownGpsDevice(0x0000, 0x0000)) << "unknown device";
        expect(!gr::timing::detail::isKnownGpsDevice(0x1546, 0xFFFF)) << "known vendor, wrong product";
    };

    "lookupDescription returns description for known devices"_test = [] {
        auto desc = gr::timing::detail::lookupDescription(0x1546, 0x01a8);
        expect(!desc.empty()) << "u-blox 8 has description";
        expect(desc.find("u-blox") != std::string::npos);

        expect(gr::timing::detail::lookupDescription(0x0000, 0x0000).empty()) << "unknown device returns empty";
    };

    "kKnownGpsReceiverIds is non-empty and well-formed"_test = [] {
        expect(gt(kKnownGpsReceiverIds.size(), 5UZ)) << "at least 5 known devices";
        for (const auto& entry : kKnownGpsReceiverIds) {
            expect(entry.vendorId != 0) << "VID is non-zero";
            expect(entry.productId != 0) << "PID is non-zero";
            expect(!entry.description.empty()) << "description is non-empty";
        }
    };

    "BaudRate enum values match expected rates"_test = [] {
        expect(eq(static_cast<std::uint32_t>(BaudRate::Baud4800), 4800U));
        expect(eq(static_cast<std::uint32_t>(BaudRate::Baud9600), 9600U));
        expect(eq(static_cast<std::uint32_t>(BaudRate::Baud115200), 115200U));
    };

    "NMEADeviceInfo default construction"_test = [] {
        NMEADeviceInfo info;
        expect(info.devicePath.empty());
        expect(info.model.empty());
        expect(info.vendor.empty());
        expect(eq(info.vendorId, std::uint16_t{0}));
        expect(eq(info.productId, std::uint16_t{0}));
        expect(!info.knownGpsDevice);
    };

    "SerialPort default state and move semantics"_test = [] {
        SerialPort port;
        expect(!port.isOpen()) << "default-constructed port is closed";

        SerialPort moved = std::move(port);
        expect(!moved.isOpen()) << "moved-from closed port is still closed";
    };

#if !defined(__EMSCRIPTEN__)
    "SerialPort::open rejects nonexistent device"_test = [] {
        auto result = SerialPort::open("/dev/nonexistent_gps_device_12345", BaudRate::Baud9600);
        expect(!result.has_value()) << "opening nonexistent device fails";
        expect(!result.error().empty()) << "error message is non-empty";
    };

    "selectNMEADevice with nonexistent path returns error"_test = [] {
        auto result = selectNMEADevice("/dev/nonexistent_gps_device_12345", false);
        expect(!result.has_value()) << "nonexistent path fails";
        expect(!result.error().empty()) << "error message is non-empty";
    };

    "discoverNMEADevices returns a vector"_test = [] {
        auto devices = discoverNMEADevices(false);
        // may be empty if no GPS hardware — just verify it doesn't crash
        expect(true) << std::format("discovered {} device(s)", devices.size());
    };
#endif

#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)
    "SerialPort close on moved-from port is safe"_test = [] {
        SerialPort a;
        SerialPort b = std::move(a);
        a.close(); // should be no-op on moved-from port
        expect(!a.isOpen());
        expect(!b.isOpen());
    };
#endif
};

int main() { return 0; }
